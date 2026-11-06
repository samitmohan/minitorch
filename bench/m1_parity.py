"""Loss-curve parity: the same small LM trained by minitorch and by PyTorch, step for step.

Both start from the same weights, see the same batches, and use the same AdamW,
cosine schedule, and clipping. Writes bench/results/m1_parity.json and a plot.

    uv run python bench/m1_parity.py --steps 200
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from minitorch import (
    AdamW,
    TransformerLM,
    clip_grad_norm,
    cosine_lr,
    cross_entropy_loss,
)
from train_gpt import get_batch

RESULTS = Path(__file__).resolve().parent / "results"


def torch_forward(P, idx, n_layers, n_heads):
    """TransformerLM in plain torch over the same state_dict names; Linear weights are (in, out) like minitorch."""
    def rms(x, w):
        return TF.rms_norm(x, (x.shape[-1],), w, eps=1e-5)

    def rope(t):
        D = t.shape[-1]
        inv = 1.0 / 10000.0 ** (torch.arange(0, D, 2, dtype=torch.float64) / D)
        ang = torch.outer(torch.arange(t.shape[2], dtype=torch.float64), inv)
        cos, sin = torch.cat([ang.cos()] * 2, -1).float(), torch.cat([ang.sin()] * 2, -1).float()
        return t * cos + torch.cat([-t[..., D // 2:], t[..., :D // 2]], -1) * sin

    x = P["embed.weight"][idx]
    B, T, C = x.shape
    for i in range(n_layers):
        b = f"blocks.{i}."
        h = rms(x, P[b + "attn_norm.weight"])
        q, k, v = (rope((h @ P[b + f"attn.{n}_proj.weight"]).reshape(B, T, n_heads, -1).transpose(1, 2)) if n != "v"
                   else (h @ P[b + "attn.v_proj.weight"]).reshape(B, T, n_heads, -1).transpose(1, 2)
                   for n in "qkv")
        a = TF.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
        x = x + a @ P[b + "attn.o_proj.weight"]
        h = rms(x, P[b + "ffn_norm.weight"])
        x = x + (TF.silu(h @ P[b + "ffn.w1.weight"]) * (h @ P[b + "ffn.w3.weight"])) @ P[b + "ffn.w2.weight"]
    return rms(x, P["norm.weight"]) @ P["head.weight"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/tinystories_train.bin")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--context", type=int, default=128)
    p.add_argument("--float64", action="store_true", help="separates rounding drift from real differences")
    args = p.parse_args()
    cfg = {"vocab_size": 10_000, "context_length": args.context, "d_model": 256, "n_layers": 4, "n_heads": 8,
           "d_ff": 704}
    lr, min_lr, warmup = 3e-3, 3e-4, 20

    np.random.seed(0)
    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    model = TransformerLM(**cfg)
    if args.float64:
        for q in model.parameters():
            q.data = q.data.astype(np.float64)
    names = list(model.state_dict())
    P = {n: torch.tensor(v, requires_grad=True) for n, v in model.state_dict().items()}
    mt_params = model.parameters()
    assert len(names) == len(mt_params), "state_dict and parameters() must line up"
    opt = AdamW(mt_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    topt = torch.optim.AdamW(P.values(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    n_params = sum(v.size for v in model.state_dict().values())

    rows = []
    for step in range(1, args.steps + 1):
        x, y = get_batch(data, args.batch, args.context)
        opt.lr = cosine_lr(step, lr, min_lr, warmup, args.steps)
        for g in topt.param_groups:
            g["lr"] = opt.lr

        t0 = time.perf_counter()
        loss = cross_entropy_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        clip_grad_norm(mt_params, 1.0)
        opt.step()
        t_mt = time.perf_counter() - t0

        t0 = time.perf_counter()
        tloss = TF.cross_entropy(torch_forward(P, torch.tensor(x), cfg["n_layers"], cfg["n_heads"]).reshape(-1, 10_000),
                                 torch.tensor(y).reshape(-1))
        topt.zero_grad()
        tloss.backward()
        torch.nn.utils.clip_grad_norm_(P.values(), 1.0)
        topt.step()
        t_th = time.perf_counter() - t0

        rows.append({"step": step, "minitorch": loss.item(), "torch": tloss.item(),
                     "abs_diff": abs(loss.item() - tloss.item()), "ms_minitorch": t_mt * 1e3, "ms_torch": t_th * 1e3})
        if step % 20 == 0 or step == 1:
            r = rows[-1]
            print(f"step {step:4d} | minitorch {r['minitorch']:.5f} | torch {r['torch']:.5f} | diff {r['abs_diff']:.2e}")

    diffs = [r["abs_diff"] for r in rows]
    summary = {"config": cfg, "params": n_params, "steps": args.steps, "batch": args.batch,
               "max_abs_loss_diff": max(diffs), "final_abs_loss_diff": diffs[-1],
               "median_ms_per_step": {"minitorch": float(np.median([r["ms_minitorch"] for r in rows])),
                                      "torch_cpu": float(np.median([r["ms_torch"] for r in rows]))},
               "rows": rows}
    summary["dtype"] = "float64" if args.float64 else "float32"
    stem = "m1_parity_f64" if args.float64 else "m1_parity"
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{stem}.json").write_text(json.dumps(summary, indent=1) + "\n")

    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    steps = [r["step"] for r in rows]
    a1.plot(steps, [r["torch"] for r in rows], label="PyTorch", lw=3, alpha=0.5)
    a1.plot(steps, [r["minitorch"] for r in rows], label="minitorch", lw=1)
    a1.set(xlabel="step", ylabel="train loss", title=f"{n_params / 1e6:.1f}M-param LM, same weights and batches")
    a1.legend()
    a2.semilogy(steps, diffs)
    a2.set(xlabel="step", ylabel="|loss difference|", title="per-step gap")
    fig.tight_layout()
    fig.savefig(RESULTS / f"{stem}.png", dpi=120)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
