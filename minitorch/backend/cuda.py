"""CuPy backend: an NVIDIA GPU through CUDA.

Verified on bqa3 (RTX 3090, CuPy 14.2, CUDA 13): tests/backends.py --device cuda
passes, and it trained the CS336 TinyStories model (val loss 1.399, runs/tinystories).

CuPy copies the NumPy API, arrays included, so this module is cpu.py with cupy
in place of numpy. CuPy launches each op as it is called and returns before
the GPU finishes; `.item()` and `to_numpy` wait for it.
Set CUPY_TF32=1 to run float32 matmuls on the tensor cores in TF32.
"""
import cupy as cp
import cupyx
from cupy import (
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
    sign,
    stack,
    triu,
    where,
    zeros_like,
)

from . import cpu, cuda_attention
from .cpu import dtype, is_float, realize

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
    "fused_attention_backward",
    "im2col",
    "inf",
    "is_float",
    "log",
    "ones_like",
    "pad",
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



def clear_cache():
    """Give the blocks CuPy's memory pool keeps for reuse back to the GPU."""
    cp.get_default_memory_pool().free_all_blocks()

def asarray(x, dtype=None):
    return cp.asarray(x, dtype=dtype)


def to_numpy(x):
    return cp.asnumpy(x)


def fused_attention(q, k, v):
    """mlops.Attention's forward as Triton kernels; None for shapes they don't cover (the reference math runs)."""
    return cuda_attention.attention(q, k, v) if cuda_attention.supported(q) else None


def fused_attention_backward(q, k, v, out, lse, grad):
    return cuda_attention.attention_backward(q, k, v, out, lse, grad)


def sigmoid(x):
    return 1.0 / (1.0 + cp.exp(-x))


def scatter_add(shape, key, values):
    out = cp.zeros(shape, dtype=values.dtype)
    cupyx.scatter_add(out, key, values)
    return out


# shortcut: im2col runs on the device, but col2im (conv's backward) round-trips through the CPU; write it in CuPy if a CNN trains on cuda
def im2col(x, *args):
    return cpu.im2col(x, *args, xp=cp)  # on the device; col2im, used only in training, still goes through the CPU


def col2im(cols, *args):
    return cp.asarray(cpu.col2im(cp.asnumpy(cols), *args))

