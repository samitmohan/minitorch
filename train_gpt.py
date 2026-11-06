"""Train TransformerLM on uint16 token files. Defaults are the CS336 assignment 1 TinyStories run.

    uv run python train_gpt.py --run tinystories                       # 40k steps x 32 x 256 = 327.68M tokens
    uv run python train_gpt.py --run smoke --steps 300 --d-model 128 --layers 2 --heads 4 --d-ff 384
    CUPY_TF32=1 uv run --no-dev python train_gpt.py --run tinystories --device cuda   # bqa3, RTX 3090

Writes runs/<run>/config.json, metrics.jsonl (one JSON line per step, plus eval lines),
samples.jsonl, and ckpt.npz.
"""
import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from minitorch import (
    AdamW,
    LossScaler,
    TransformerLM,
    autocast,
    clip_grad_norm,
    cosine_lr,
    cross_entropy_loss,
    no_grad,
)
from minitorch.tokenizer import Tokenizer

EOT = "<|endoftext|>"


def get_batch(data, batch_size, context):
    """Random windows of `context` tokens; targets are the inputs shifted by one."""
    starts = np.random.randint(0, len(data) - context - 1, size=batch_size)
    x = np.stack([data[s:s + context] for s in starts]).astype(np.int64)
    y = np.stack([data[s + 1:s + context + 1] for s in starts]).astype(np.int64)
    return x, y


def evaluate(model, data, args):
    with no_grad():
        return float(np.mean([cross_entropy_loss(model(x), y).item()
                              for x, y in (get_batch(data, args.batch, args.context) for _ in range(args.eval_batches))]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "metal", "cuda"])
    p.add_argument("--train", default="data/tinystories_train.bin")
    p.add_argument("--valid", default="data/tinystories_valid.bin")
    p.add_argument("--tokenizer", default="data/tinystories_bpe")
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=1344)
    # CS336 ablations; the defaults are the modern model. SiLU matches SwiGLU's parameters at --d-ff 2048 (4 x d_model)
    p.add_argument("--norm", default="pre", choices=["pre", "post", "none"])
    p.add_argument("--no-rope", action="store_true")
    p.add_argument("--ffn", default="swiglu", choices=["swiglu", "silu"])
    # mixed precision: the matrix multiplies in 16 bits, the weights in float32. float16 (CuPy has no bfloat16) scales the loss
    p.add_argument("--amp", default=None, choices=["bfloat16", "float16"])
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=40_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--sample-every", type=int, default=2000)
    p.add_argument("--peak-tflops", type=float, default=None, help="hardware peak, to log MFU")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)
    out = Path("runs") / args.run
    out.mkdir(parents=True, exist_ok=False)  # an existing run keeps its metrics and checkpoint
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    train = np.memmap(args.train, dtype=np.uint16, mode="r")
    valid = np.memmap(args.valid, dtype=np.uint16, mode="r")
    tok = Tokenizer.from_files(f"{args.tokenizer}/vocab.json", f"{args.tokenizer}/merges.txt", [EOT])

    model = TransformerLM(args.vocab_size, args.context, args.d_model, args.layers, args.heads, args.d_ff,
                          norm=args.norm, rope=not args.no_rope, ffn=args.ffn).to(args.device)
    params = model.parameters()
    # 6 FLOPs per parameter per token (2 forward, 4 backward); the embedding is a lookup, not a matmul
    n_matmul = sum(q.data.size for q in params) - model.embed.weight.data.size
    opt = AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scaler = LossScaler() if args.amp == "float16" else None
    print(f"{sum(q.data.size for q in params):,} params ({n_matmul:,} in matmuls), "
          f"{len(train):,} train tokens -> {out}")

    with open(out / "metrics.jsonl", "w") as metrics, open(out / "samples.jsonl", "w") as samples:
        start = time.time()
        for step in range(1, args.steps + 1):
            t0 = time.perf_counter()
            opt.lr = cosine_lr(step, args.lr, args.min_lr, args.warmup, args.steps)
            x, y = get_batch(train, args.batch, args.context)
            with autocast(args.amp) if args.amp else nullcontext():
                loss = cross_entropy_loss(model(x), y)
            opt.zero_grad()
            (loss * scaler.scale if scaler else loss).backward()
            overflow = scaler is not None and not scaler.unscale(params)
            if not overflow:  # an overflowed float16 gradient skips the update, not the step's eval and checkpoint
                grad_norm = clip_grad_norm(params, args.clip)
                opt.step()
            dt = time.perf_counter() - t0

            tokens_per_s = args.batch * args.context / dt
            row = {"step": step, "loss": round(loss.item(), 4), "lr": opt.lr, "grad_norm": None if overflow else round(float(grad_norm), 4),
                   "tokens_per_s": round(tokens_per_s), "tflops": round(6 * n_matmul * tokens_per_s / 1e12, 4),
                   "time": round(time.time() - start, 1)}
            if args.peak_tflops:
                row["mfu"] = round(row["tflops"] / args.peak_tflops, 4)
            if step % args.eval_every == 0 or step == args.steps:
                row["val_loss"] = round(evaluate(model, valid, args), 4)
                model.save(out / "ckpt.npz")
                print(f"step {step:6d} | loss {row['loss']:.4f} | val {row['val_loss']:.4f} | {row['tokens_per_s']:,} tok/s")
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()

            if step % args.sample_every == 0 or step == args.steps:
                prompt = np.array([tok.encode("Once upon a time")])
                ids = model.generate(prompt, min(100, args.context - prompt.shape[1]), temperature=0.8, top_p=0.95, eos_id=tok.ids[EOT.encode()])
                samples.write(json.dumps({"step": step, "text": tok.decode(ids[0].tolist())}) + "\n")
                samples.flush()

if __name__ == "__main__":
    main()
