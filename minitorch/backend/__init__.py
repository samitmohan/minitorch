"""Array backends.

A backend is a module exposing the same few functions over its own array type:
`exp`, `log`, `where`, `broadcast_to`, `scatter_add`, `im2col`, and so on (see
cpu.py for the full list). mlops.py calls these through `xp`, so every gradient
is written once and runs on any backend.

    cpu    NumPy, the readable reference every other backend is checked against
    metal  MLX on the Apple GPU
    cuda   CuPy on an NVIDIA GPU

A backend is imported on first use: MLX only installs on macOS, CuPy only with CUDA.
"""
import functools
import importlib

DEVICES = ("cpu", "metal", "cuda")


@functools.cache
def get(device):
    if device not in DEVICES:
        raise ValueError(f"unknown device {device!r}, expected one of {DEVICES}")
    return importlib.import_module(f"{__name__}.{device}")
