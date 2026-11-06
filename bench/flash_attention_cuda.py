"""Causal attention on the RTX 3090: the Triton FlashAttention-2 kernels against the composed math.

    CUPY_TF32=1 uv run --no-dev python bench/flash_attention_cuda.py    # bqa3

"composed" is mlops.Attention's own math (the score matrix written, softmaxed, and read back)
run on a CUDA namespace without the fused kernels; "Triton" is backend/cuda_attention.py.
Each is timed forward alone and forward plus backward, at the shapes bench/flash_attention.py
uses on the Mac. Writes bench/results/flash_attention_cuda.json.
"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from minitorch import Tensor, mlops
from minitorch.backend import cuda

RESULTS = Path(__file__).resolve().parent / "results"
SHAPES = [  # name, B, H, Hkv, T, S, D
    ("CS336 training", 32, 16, 16, 256, 256, 32),
    ("prefill 1k", 4, 16, 8, 1024, 1024, 64),
    ("Qwen3 prefill 2k", 1, 16, 8, 2048, 2048, 128),
]
COMPOSED = SimpleNamespace(**{name: getattr(cuda, name) for name in cuda.__all__ if not name.startswith("fused")})


def forward(xp, parents, dout):
    ctx = mlops.Attention(xp, *parents)
    return ctx, ctx.forward(*(t.data for t in parents))


def forward_backward(xp, parents, dout):
    ctx, _ = forward(xp, parents, dout)
    return ctx.backward(dout)


def per_call(fn, *args):
    """Milliseconds per call, the device finished each time."""
    fn(*args)
    cp.cuda.Device().synchronize()
    times = []
    while sum(times) < 0.5 or len(times) < 3:
        start = time.perf_counter()
        fn(*args)
        cp.cuda.Device().synchronize()
        times.append(time.perf_counter() - start)
    return float(np.median(times)) * 1000


def main():
    rows = []
    for name, B, H, Hkv, T, S, D in SHAPES:
        rng = np.random.default_rng(0)
        q, k, v, dout = (cp.asarray(rng.standard_normal(s), cp.float32) for s in [(B, H, T, D), (B, Hkv, S, D), (B, Hkv, S, D), (B, H, T, D)])
        parents = [Tensor(a, requires_grad=True, device="cuda") for a in (q, k, v)]
        for impl, xp in (("composed", COMPOSED), ("Triton", cuda)):
            rows.append({"shape": name, "B": B, "H": H, "Hkv": Hkv, "T": T, "S": S, "D": D, "impl": impl,
                         "forward_ms": round(per_call(forward, xp, parents, dout), 3),
                         "forward_backward_ms": round(per_call(forward_backward, xp, parents, dout), 3)})
            print(rows[-1], flush=True)
    result = {"device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), "dtype": "float32 (TF32 matmuls)", "rows": rows}
    (RESULTS / "flash_attention_cuda.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
