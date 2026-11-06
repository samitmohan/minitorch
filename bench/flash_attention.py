"""Causal attention forward on the M4 GPU: my three Metal kernels against the
materialized-scores version and MLX's own fused kernel (the oracle).

    uv run python bench/flash_attention.py

Shapes: the CS336 training step, a Qwen3-0.6B-shaped prefill (GQA 16/8 heads,
D=128), and a batch of 8 decode steps against a 2048-token cache. TFLOP/s counts
only the causal half; GB/s counts reading Q, K, V and writing O once, the
minimum any kernel must move. Writes bench/results/flash_attention.json.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from minitorch.backend.metal_attention import attention

RESULTS = Path(__file__).resolve().parent / "results"
SHAPES = [  # name, B, H, Hkv, T, S, D
    ("CS336 training", 32, 16, 16, 256, 256, 32),
    ("prefill 1k", 4, 16, 8, 1024, 1024, 64),
    ("Qwen3 prefill 2k", 1, 16, 8, 2048, 2048, 128),
    ("decode, 2k cache", 8, 16, 8, 1, 2048, 128),
]


def materialized(q, k, v):
    """The textbook version: the whole (T, S) score matrix goes through memory."""
    _, H, T, D = q.shape
    S, rep = k.shape[2], H // k.shape[1]
    k, v = mx.repeat(k, rep, axis=1), mx.repeat(v, rep, axis=1)
    s = (q * D ** -0.5) @ k.swapaxes(-1, -2) + mx.triu(mx.full((T, S), -mx.inf), k=S - T + 1)
    return mx.softmax(s, axis=-1) @ v


IMPLS = {
    "materialized scores": materialized,
    "v1 thread per row": lambda q, k, v: attention(q, k, v, "v1")[0],
    "v2 threadgroup tiles": lambda q, k, v: attention(q, k, v, "v2")[0],
    "v3 simdgroup matmul": lambda q, k, v: attention(q, k, v, "v3")[0],
    "mx.fast.sdpa (oracle)": lambda q, k, v: mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask="causal"),
}


def per_call(fn, q, k, v, calls=10):
    """Seconds per call, with `calls` launches queued per sync so the ~140 us sync is amortized."""
    mx.eval(fn(q, k, v))
    times = []
    while sum(times) < 0.5 or len(times) < 3:
        t0 = time.perf_counter()
        mx.eval([fn(q, k, v) for _ in range(calls)])
        times.append((time.perf_counter() - t0) / calls)
    return float(np.median(times))


def main():
    rows = []
    for name, B, H, Hkv, T, S, D in SHAPES:
        visible = T * (S - T) + T * (T + 1) // 2  # (query, key) pairs the causal mask lets through
        for dtype in [mx.float32, mx.bfloat16]:
            rng = np.random.default_rng(0)
            q, k, v = (mx.array(rng.standard_normal(s).astype(np.float32)).astype(dtype)
                       for s in [(B, H, T, D), (B, Hkv, S, D), (B, Hkv, S, D)])
            oracle = IMPLS["mx.fast.sdpa (oracle)"](q, k, v).astype(mx.float32)
            nbytes = (2 * B * H * T * D + 2 * B * Hkv * S * D) * q.itemsize
            for impl, fn in IMPLS.items():
                s = per_call(fn, q, k, v)
                err = mx.abs(fn(q, k, v).astype(mx.float32) - oracle).max().item()
                rows.append({"shape": name, "B": B, "H": H, "Hkv": Hkv, "T": T, "S": S, "D": D,
                             "dtype": str(dtype).split(".")[-1], "impl": impl, "ms": round(s * 1e3, 4),
                             "tflops": round(4 * B * H * D * visible / s / 1e12, 3),
                             "gb_s": round(nbytes / s / 1e9, 1), "max_err_vs_oracle": float(f"{err:.2e}")})
                r = rows[-1]
                print(f"{name:17s} {r['dtype']:8s} {impl:22s} {r['ms']:9.3f} ms {r['tflops']:7.3f} TFLOP/s "
                      f"{r['gb_s']:7.1f} GB/s  err {err:.1e}")
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], check=False, capture_output=True, text=True).stdout.strip()
    out = {"device": f"{chip} GPU", "mlx": mx.__version__, "rows": rows}
    (RESULTS / "flash_attention.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
