"""LoRA fine-tune of Qwen3-0.6B on GSM8K's training split (SPEC.md M3).

    CUPY_TF32=1 uv run --no-dev python train_lora.py --run lora_gsm8k --device cuda    # bqa3
    uv run --no-dev python train_lora.py --run lora_smoke --device cuda --steps 4       # a quick check

Each example is the evaluation prompt (bench/gsm8k.py) followed by GSM8K's reference
solution, its <<calculator>> annotations removed, ending "#### <answer>". The loss covers
only the reply. Writes runs/<run>/config.json, metrics.jsonl, and adapter.npz.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from bench.gsm8k import PROMPT
from minitorch import AdamW, clip_grad_norm, cosine_lr, cross_entropy_loss
from minitorch.lora import apply_lora, save_adapter
from minitorch.qwen3 import chat_prompt, load_qwen3, load_tokenizer


def examples(tok, path):
    """(token ids, number of prompt tokens) for every question in a GSM8K .jsonl file."""
    out = []
    with open(path) as f:
        for line in f:
            q = json.loads(line)
            prompt = tok.encode(chat_prompt([{"role": "user", "content": PROMPT.format(question=q["question"])}]))
            reply = tok.encode(re.sub(r"<<[^>]*>>", "", q["answer"]) + "<|im_end|>")
            out.append((prompt + reply, len(prompt)))
    return out


def make_batch(rows, pad):
    """Right-padded inputs and targets, and a mask that is True where the target is a reply token."""
    T = max(len(ids) for ids, _ in rows)
    x = np.full((len(rows), T), pad, dtype=np.int64)
    mask = np.zeros((len(rows), T - 1), dtype=bool)
    for i, (ids, n_prompt) in enumerate(rows):
        x[i, :len(ids)] = ids
        mask[i, n_prompt - 1:len(ids) - 1] = True  # position t predicts token t + 1
    return x[:, :-1], x[:, 1:], mask


def reply_logits(model, inputs, mask):
    """Logits at the reply positions only, and those positions' flat indices into inputs."""
    h = model.hidden(inputs)
    rows = np.flatnonzero(mask.reshape(-1))
    # the 151,936-way head only where the loss looks: on every position it would be B * T * 151,936 floats
    return model.project(h.reshape(-1, h.shape[-1])[rows]), rows


def reply_loss(model, inputs, targets, mask):
    logits, rows = reply_logits(model, inputs, mask)
    return cross_entropy_loss(logits, targets.reshape(-1)[rows])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--device", default="cuda", choices=["cpu", "metal", "cuda"])
    p.add_argument("--model", default="data/Qwen3-0.6B")
    p.add_argument("--data", default="data/gsm8k/train.jsonl")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32)
    p.add_argument("--batch", type=int, default=8)
    # shortcut: every intermediate stays alive for backward. Measured peak on the 3090, including 3.4 GB of other processes:
    # 10.9 GB at 1 example per pass, 17.9 GB at 2, 22.8 GB at 4, out of memory at 8. Recomputing each block in backward
    # (activation checkpointing) would allow whole-batch passes
    p.add_argument("--micro", type=int, default=2, help="examples per forward pass; their gradients add up to the batch's")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps", type=int, default=None, help="stop early, for a smoke test")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)
    out = Path("runs") / args.run
    out.mkdir(parents=True, exist_ok=False)
    tok = load_tokenizer(args.model)
    data = examples(tok, args.data)
    total = args.steps or args.epochs * -(-len(data) // args.batch)
    (out / "config.json").write_text(json.dumps(vars(args) | {"examples": len(data), "steps": total}, indent=2))

    model = load_qwen3(args.model).to(args.device)
    apply_lora(model, args.rank, args.alpha)
    params = model.parameters()
    opt = AdamW(params, lr=args.lr, weight_decay=0.0)
    pad = tok.ids[b"<|endoftext|>"]
    print(f"{len(data):,} examples, {sum(q.data.size for q in params):,} LoRA parameters, {total} steps -> {out}")

    order = []
    with open(out / "metrics.jsonl", "w") as metrics:
        start = time.time()
        for step in range(1, total + 1):
            if not order:
                order = list(np.random.permutation(len(data)))
            # sorted by length, so each micro-batch pads to a similar length
            rows = sorted((data[order.pop()] for _ in range(min(args.batch, len(order)))), key=lambda r: len(r[0]))
            reply_tokens = sum(len(ids) - n_prompt for ids, n_prompt in rows)
            t0 = time.perf_counter()
            opt.lr = cosine_lr(step, args.lr, 0.0, args.warmup, total)
            opt.zero_grad()
            batch_loss, tokens = 0.0, 0
            for i in range(0, len(rows), args.micro):
                inputs, targets, mask = make_batch(rows[i:i + args.micro], pad)
                # weighted by its share of reply tokens, so the summed gradient is the whole batch's mean
                loss = reply_loss(model, inputs, targets, mask) * (mask.sum() / reply_tokens)
                loss.backward()
                batch_loss, tokens = batch_loss + loss.item(), tokens + inputs.size
                del loss  # drop this graph's saved activations before the next forward builds its own
            grad_norm = clip_grad_norm(params, args.clip)
            opt.step()
            dt = time.perf_counter() - t0
            row = {"step": step, "loss": round(batch_loss, 4), "lr": opt.lr, "grad_norm": round(float(grad_norm), 4),
                   "tokens_per_s": round(tokens / dt), "reply_tokens": reply_tokens, "time": round(time.time() - start, 1)}
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            if step % 50 == 0 or step == total:
                print(f"step {step:5d}/{total} | loss {row['loss']:.4f} | {row['tokens_per_s']:,} tok/s | {row['time']:.0f} s", flush=True)
                save_adapter(model, out / "adapter.npz", args.rank, args.alpha)


if __name__ == "__main__":
    main()
