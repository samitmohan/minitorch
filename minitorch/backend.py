import numpy as np

try:
    import cupy as cp
    _GPU_AVAILABLE = True
except ImportError:
    cp = None
    _GPU_AVAILABLE = False


def gpu_available():
    return _GPU_AVAILABLE


def get_array_module(data):
    if _GPU_AVAILABLE and isinstance(data, cp.ndarray):
        return cp
    return np


def to_device(data, device):
    if device == "cuda":
        if not _GPU_AVAILABLE:
            raise RuntimeError("CuPy is not installed - cannot move to CUDA")
        if isinstance(data, np.ndarray):
            return cp.asarray(data)
        return data
    elif device == "cpu":
        if _GPU_AVAILABLE and isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        return data
    else:
        raise ValueError(f"Unknown device: {device}")
