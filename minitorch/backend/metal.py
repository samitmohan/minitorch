"""MLX backend: the Apple GPU through Metal.

MLX copies the NumPy API, so most of this module is re-exports. Two differences
matter. MLX arrays are immutable, which is why no gradient in mlops.py is ever
updated in place. And MLX is lazy: each op only records itself, and the work runs
when a value is needed (`.item()`, a print, or `realize`). The optimizer realizes
the weights after every step, or each step's graph would hang off the last one.

MLX has no float64 on the GPU; float64 input comes in as float32.
"""
import mlx.core as mx
import numpy as np
from mlx.core import (
    broadcast_to,
    concatenate,
    cumsum,
    exp,
    expand_dims,
    float32,
    full,
    inf,
    log,
    ones_like,
    pad,
    repeat,
    sigmoid,
    sign,
    stack,
    triu,
    where,
    zeros_like,
)

from . import cpu, metal_attention, metal_quant

__all__ = [
    "asarray",
    "broadcast_to",
    "clear_cache",
    "col2im",
    "concatenate",
    "cumsum",
    "dtype",
    "exp",
    "expand_dims",
    "float32",
    "full",
    "fused_attention",
    "im2col",
    "inf",
    "is_float",
    "log",
    "ones_like",
    "pad",
    "quant_matmul",
    "realize",
    "repeat",
    "scatter_add",
    "sigmoid",
    "sign",
    "stack",
    "to_numpy",
    "triu",
    "where",
    "zeros_like",
]


def asarray(x, dtype=None):
    if isinstance(x, mx.array):
        return x if dtype is None or x.dtype == dtype else x.astype(dtype)
    return mx.array(x) if dtype is None else mx.array(x, dtype=dtype)


def to_numpy(x):
    # NumPy has no bfloat16; widen it first (exact, bf16 is a truncated float32)
    return np.array(x.astype(mx.float32) if x.dtype == mx.bfloat16 else x)


def dtype(name):
    return getattr(mx, name)


def is_float(dt):
    return mx.issubdtype(dt, mx.floating)


realize = mx.eval
clear_cache = mx.clear_cache  # freed buffers stay in MLX's cache for reuse until this


def scatter_add(shape, key, values):
    return mx.zeros(shape, dtype=values.dtype).at[key].add(values)


# shortcut: im2col runs on the device, but col2im (conv's backward) round-trips through the CPU; write it in MLX if a CNN trains on metal
def im2col(x, *args):
    return cpu.im2col(x, *args, xp=mx)  # on the device; col2im, used only in training, still goes through the CPU


def col2im(cols, *args):
    return mx.array(cpu.col2im(to_numpy(cols), *args))


def fused_attention(q, k, v):
    """mlops.Attention's forward as one Metal kernel; None for shapes the kernels don't cover (the reference math runs)."""
    D = q.shape[-1]
    if D % 8 or D > 128:
        return None
    # one query row (decode) leaves v3's 32-row threadgroups mostly idle; v1 measured faster there
    return metal_attention.attention(q, k, v, "v1" if q.shape[2] == 1 else "v3")


def quant_matmul(x, packed, scales, biases, bits, group):
    """x (M, K) @ the dequantized (N, K) weight, transposed, for small M (decode): see metal_quant.py."""
    return metal_quant.quant_matmul(x, packed, scales, biases, bits, group)
