"""Read .safetensors files: the format Hugging Face checkpoints ship in.

    [8 bytes: header length N, little-endian] [N bytes: JSON header] [raw tensor bytes]

The header maps each tensor name to its dtype, shape, and [start, end) byte offsets
into the raw bytes that follow it. No pickle, nothing to execute: reading a tensor is
slicing bytes and reinterpreting them.
"""
import json

import numpy as np

_DTYPES = {"F32": np.float32, "F16": np.float16, "F64": np.float64, "I64": np.int64, "I32": np.int32, "U8": np.uint8}


def load_safetensors(path):
    """Every tensor in the file as a float NumPy array, by name. bfloat16 is widened to float32."""
    return dict(iter_safetensors(path))


def iter_safetensors(path):
    """(name, array) for each tensor in the file, one at a time: the file is memory-mapped, so only
    the tensor in hand takes memory, and a loader can place it before reading the next."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        b = raw[start:end]
        if info["dtype"] == "BF16":
            # bfloat16 is the top half of a float32, so shifting its 16 bits up rebuilds the float32 exactly
            arr = (b.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            arr = np.array(b.view(_DTYPES[info["dtype"]]))
        yield name, arr.reshape(info["shape"])
