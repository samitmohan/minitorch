"""NumPy backend: the readable reference."""
import numpy as np
from numpy import (
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


def asarray(x, dtype=None):
    """Any array, list, or number -> an array of this backend."""
    return np.asarray(x, dtype=dtype)


def to_numpy(x):
    return np.array(x)  # a copy, as every other backend's device-to-host transfer is


def dtype(name):
    """"float32", "float16", ... -> this backend's dtype object."""
    return np.dtype(name)


def is_float(dt):
    return np.issubdtype(dt, np.floating)


def realize(*arrays):
    """Finish computing `arrays`. NumPy computes each op as it is called, so there is nothing to do."""


def clear_cache():
    """Give the memory a backend keeps for reuse back to the system. NumPy keeps none."""


def sigmoid(x):
    # exp(-x) overflows to inf for very negative x, and 1 / inf = 0 is the right answer
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


def scatter_add(shape, key, values):
    """zeros(shape) with `values` added at `key`, summing where an index repeats."""
    out = np.zeros(shape, dtype=values.dtype)
    np.add.at(out, key, values)
    return out


def im2col(x, kh, kw, stride, padding, xp=np):
    """(N, C, H, W) -> (N*OH*OW, C*kh*kw): one row per output pixel, holding its input patch.

    Built from kh * kw strided slices, one per kernel offset, stacked: no writes into an array,
    so the same code runs on the GPU backends (MLX arrays cannot be written in place).
    """
    N, C = x.shape[:2]
    if padding > 0:
        x = xp.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    OH = (x.shape[2] - kh) // stride + 1
    OW = (x.shape[3] - kw) // stride + 1
    # channels last first, so the stack lands in row order and the reshape copies nothing: one 256-channel
    # 3x3 im2col at 512x512 peaked at 2.95 GB this way on the M4, 5.37 GB stacking NCHW and transposing after
    x = x.transpose(0, 2, 3, 1)
    patches = [x[:, y:y + stride * OH:stride, xx:xx + stride * OW:stride, :] for y in range(kh) for xx in range(kw)]
    return xp.stack(patches, axis=-1).reshape(N * OH * OW, C * kh * kw)  # (N, OH, OW, C, kh*kw) -> rows


def col2im(cols, shape, kh, kw, stride, padding):
    """Inverse of im2col: add every patch row back onto the pixels it came from."""
    N, C, H, W = shape
    H_pad, W_pad = H + 2 * padding, W + 2 * padding
    OH = (H_pad - kh) // stride + 1
    OW = (W_pad - kw) // stride + 1

    cols = cols.reshape(N, OH, OW, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H_pad, W_pad), dtype=cols.dtype)
    for y in range(kh):
        for xx in range(kw):
            img[:, :, y:y + stride * OH:stride, xx:xx + stride * OW:stride] += cols[:, :, y, xx]
    return img[:, :, padding:H_pad - padding, padding:W_pad - padding]
