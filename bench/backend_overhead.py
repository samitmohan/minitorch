"""What each backend costs per op, and where a GPU backend loses to NumPy.

    uv run python bench/backend_overhead.py                  # cpu + metal, on the Mac
    uv run --no-dev python bench/backend_overhead.py --devices cuda   # bqa3

1. elementwise: 64 chained multiplies on n floats, timed per op three ways:
   the raw backend array, the backend with a sync after every op, and through
   minitorch Tensors (no_grad), which adds the Python cost of the framework.
2. matmul: square float32 (and bfloat16 on metal) matmuls, in TFLOP/s.
3. train step: one CS336 TinyStories step (fwd + bwd + AdamW) per device, and the
   same model written in plain MLX (mlx.nn, its fused kernels) as the target.

Writes bench/results/backend_overhead_<devices>.json.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from minitorch import (
    AdamW,
    Tensor,
    TransformerLM,
    backend,
    clip_grad_norm,
    cross_entropy_loss,
    no_grad,
)

RESULTS = Path(__file__).resolve().parent / "results"
CS336 = {"vocab_size": 10_000, "context_length": 256, "d_model": 512, "n_layers": 4, "n_heads": 16, "d_ff": 1344}
MATMUL_PARAMS = 17_576_448  # CS336 config without the embedding table: 6 FLOPs per parameter per token


def sync(a):
    # .item() waits for the device on every backend (MLX evaluates, CuPy synchronizes)
    a.reshape(-1)[0].item()


def median_time(fn, min_seconds=0.3, min_reps=3):
    fn()  # warm up: kernel compilation, allocator
    times = []
    while sum(times) < min_seconds or len(times) < min_reps:
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def elementwise(device, K=64):
    xp = backend.get(device)
    rows = []
    for n in [10**3, 10**4, 10**5, 10**6, 10**7]:
        host = np.random.rand(n).astype(np.float32)
        a, t = xp.asarray(host), Tensor(host, device=device)

        def raw(a=a):
            x = a
            for _ in range(K):
                x = x * 1.0001
            sync(x)

        def raw_sync_each(a=a):
            x = a
            for _ in range(K):
                x = x * 1.0001
                sync(x)

        def framework(t=t):
            x = t
            with no_grad():
                for _ in range(K):
                    x = x * 1.0001
            sync(x.data)

        for path, fn in [("raw", raw), ("sync every op", raw_sync_each), ("minitorch", framework)]:
            rows.append({"device": device, "path": path, "n": n, "us_per_op": round(median_time(fn) / K * 1e6, 2)})
            print(f"  elementwise {device:5s} {path:14s} n={n:>9,}: {rows[-1]['us_per_op']:9.2f} us/op")
    return rows


def matmul(device):
    xp = backend.get(device)
    rows = []
    for dtype in ["float32", "bfloat16"] if device == "metal" else ["float32"]:
        for n in [256, 512, 1024, 2048, 4096]:
            a = xp.asarray(np.random.randn(n, n).astype(np.float32)).astype(xp.dtype(dtype))
            b = xp.asarray(np.random.randn(n, n).astype(np.float32)).astype(xp.dtype(dtype))
            s = median_time(lambda a=a, b=b: sync(a @ b))
            rows.append({"device": device, "dtype": precision(device, dtype), "n": n, "tflops": round(2 * n**3 / s / 1e12, 3)})
            print(f"  matmul {device:5s} {dtype:8s} n={n:5d}: {rows[-1]['tflops']:.3f} TFLOP/s")
    return rows


def precision(device, dtype="float32"):
    # CUPY_TF32=1 runs float32 matmuls in TF32 on the tensor cores
    return "tf32" if device == "cuda" and dtype == "float32" and os.environ.get("CUPY_TF32") == "1" else dtype


def step_row(impl, device, batch, seconds):
    tok_s = batch * CS336["context_length"] / seconds
    row = {"impl": impl, "device": device, "precision": precision(device), "batch": batch, "ms": round(seconds * 1000, 1),
           "tokens_per_s": round(tok_s), "tflops": round(6 * MATMUL_PARAMS * tok_s / 1e12, 3)}
    print(f"  train step {impl:30s} {device:5s} B={batch}: {row['ms']:8.1f} ms, {row['tokens_per_s']:7,} tok/s, {row['tflops']:.3f} TFLOP/s")
    return row


def train_step(device, batch):
    np.random.seed(0)
    model = TransformerLM(**CS336).to(device)
    params = model.parameters()
    assert sum(p.data.size for p in params) - model.embed.weight.data.size == MATMUL_PARAMS
    opt = AdamW(params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)

    def step():
        x = np.random.randint(0, CS336["vocab_size"], (batch, CS336["context_length"] + 1))
        loss = cross_entropy_loss(model(x[:, :-1]), x[:, 1:])
        opt.zero_grad()
        loss.backward()
        clip_grad_norm(params, 1.0)
        opt.step()
        loss.item()

    return step_row("minitorch", device, batch, median_time(step, min_seconds=3, min_reps=4))


def mlx_train_step(batch):
    """The same model in plain MLX: the fused-kernel target for the metal backend."""
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx import nn

    class Block(nn.Module):
        def __init__(self, d, heads, d_ff):
            super().__init__()
            self.heads = heads
            self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
            self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))
            self.w1, self.w2, self.w3 = nn.Linear(d, d_ff, bias=False), nn.Linear(d_ff, d, bias=False), nn.Linear(d, d_ff, bias=False)
            self.rope = nn.RoPE(d // heads, base=10000.0)

        def __call__(self, x):
            B, T, D = x.shape
            h = self.n1(x)
            q, k, v = (p(h).reshape(B, T, self.heads, -1).transpose(0, 2, 1, 3) for p in (self.q, self.k, self.v))
            a = mx.fast.scaled_dot_product_attention(self.rope(q), self.rope(k), v, scale=(D // self.heads) ** -0.5, mask="causal")
            x = x + self.o(a.transpose(0, 2, 1, 3).reshape(B, T, D))
            h = self.n2(x)
            return x + self.w2(nn.silu(self.w1(h)) * self.w3(h))

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(CS336["vocab_size"], CS336["d_model"])
            self.blocks = [Block(CS336["d_model"], CS336["n_heads"], CS336["d_ff"]) for _ in range(CS336["n_layers"])]
            self.norm = nn.RMSNorm(CS336["d_model"])
            self.head = nn.Linear(CS336["d_model"], CS336["vocab_size"], bias=False)

        def __call__(self, idx):
            x = self.embed(idx)
            for b in self.blocks:
                x = b(x)
            return self.head(self.norm(x))

    model = LM()
    opt = optim.AdamW(learning_rate=1e-3, betas=[0.9, 0.95], weight_decay=0.1)
    loss_and_grad = nn.value_and_grad(model, lambda m, x, y: nn.losses.cross_entropy(m(x), y, reduction="mean"))

    def step():
        x = mx.array(np.random.randint(0, CS336["vocab_size"], (batch, CS336["context_length"] + 1)))
        loss, grads = loss_and_grad(model, x[:, :-1], x[:, 1:])
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)

    return step_row("plain MLX (mlx.nn, fused kernels)", "metal", batch, median_time(step, min_seconds=3, min_reps=4))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--devices", nargs="+", default=["cpu", "metal"], choices=backend.DEVICES)
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()

    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], check=False, capture_output=True, text=True).stdout.strip()
    out = {"machine": chip or platform.processor(), "platform": platform.platform(), "numpy": np.__version__,
           "elementwise": [], "matmul": [], "train_step": []}
    for device in args.devices:
        print(device)
        out["elementwise"] += elementwise(device)
        out["matmul"] += matmul(device)
        out["train_step"].append(train_step(device, args.batch))
    if "cuda" in args.devices:
        import cupy as cp
        out["gpu"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    if "metal" in args.devices:
        import mlx.core as mx
        out["mlx"] = mx.__version__
        out["train_step"].append(mlx_train_step(args.batch))

    path = RESULTS / f"backend_overhead_{'_'.join(args.devices)}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
