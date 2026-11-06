"""Group-wise weight quantization: int8 or int4 weights, dequantized inside the matmul.

A weight is stored (out, in), as MLX stores it. Each output row is cut into groups of
`group` inputs, and a group keeps a float scale and bias beside its integers:

    w = scale * q + bias,    q in [0, 2^bits)

Integers pack 32 // bits to a uint32 word, low bits first: MLX's layout, so
mx.quantized_matmul can check this module on the very same bytes. Decoding one token
reads every weight once, so its speed is set by bytes per weight: 2 for bfloat16,
about 1.06 for int8 and 0.56 for int4 once the scales and biases are counted.
"""
import numpy as np

from . import backend
from .layers import Linear
from .module import Module
from .tensor import Tensor


def quantize(w, bits=4, group=64):
    """w: (out, in) float array -> packed uint32 (out, in * bits / 32), scales and biases (out, in / group)."""
    n_out, n_in = w.shape
    assert 32 % bits == 0 and n_in % group == 0 and group % (32 // bits) == 0, (w.shape, bits, group)
    g = w.astype(np.float32).reshape(n_out, n_in // group, group)
    lo, hi = g.min(axis=-1, keepdims=True), g.max(axis=-1, keepdims=True)
    scale = np.where(hi > lo, (hi - lo) / (2**bits - 1), 1.0)  # a constant group: any scale works, bias carries it
    q = np.clip(np.round((g - lo) / scale), 0, 2**bits - 1).astype(np.uint32).reshape(n_out, -1, 32 // bits)
    shifts = np.arange(32 // bits, dtype=np.uint32) * bits
    packed = np.bitwise_or.reduce(q << shifts, axis=-1)
    return packed, scale[..., 0], lo[..., 0]


def dequantize(packed, scales, biases, bits, group, xp):
    """The inverse of quantize, in the backend `xp` and in the scales' dtype."""
    shifts = xp.asarray(np.arange(32 // bits, dtype=np.uint32) * bits)
    q = (packed[..., None] >> shifts) & (2**bits - 1)
    q = q.reshape(packed.shape[0], -1, group).astype(scales.dtype)
    return (q * scales[..., None] + biases[..., None]).reshape(packed.shape[0], -1)


class QuantLinear(Module):
    """An inference-only Linear (no bias) with group-quantized weights.

    Quantize after moving the model (quantize_model): the packed integers are raw backend
    arrays, outside the Tensor machinery that would cast them to float.
    """
    def __init__(self, weight, bits, group):
        super().__init__()
        xp = backend.get(weight.device)
        packed, scales, biases = quantize(weight.numpy().T, bits, group)  # minitorch keeps (in, out)
        self.device, self.bits, self.group = weight.device, bits, group
        self.packed = xp.asarray(packed)
        self.scales, self.biases = xp.asarray(scales, dtype=weight.dtype), xp.asarray(biases, dtype=weight.dtype)

    def forward(self, x):
        assert not x.requires_grad, "QuantLinear is inference only: run it under no_grad()"
        assert x.device == self.device, f"input on {x.device}, weights on {self.device}"
        xp = backend.get(self.device)
        flat = x.data.reshape(-1, x.shape[-1])
        fused = getattr(xp, "quant_matmul", None)
        if fused is not None and flat.shape[0] <= 8:  # decode: read the packed weights directly
            y = fused(flat, self.packed, self.scales, self.biases, self.bits, self.group)
        else:  # prefill: dequantize once, then one large matmul
            y = flat @ dequantize(self.packed, self.scales, self.biases, self.bits, self.group, xp).T
        return Tensor(y.reshape(*x.shape[:-1], y.shape[-1]), device=self.device)

    def nbytes(self):
        return sum(a.nbytes for a in (self.packed, self.scales, self.biases))


def quantize_model(model, bits=4, group=64):
    """Replace every matmul of a TransformerLM by a QuantLinear, the output head included.

    A tied head gets its own quantized copy of the embedding table; the lookup keeps reading
    the full-precision table, one row per token.
    """
    for block in model.blocks:
        for parent in (block.attn, block.ffn):
            for name, val in list(parent.__dict__.items()):
                if isinstance(val, Linear):
                    assert val.bias is None, f"{name} has a bias"
                    setattr(parent, name, QuantLinear(val.weight, bits, group))
    head = model.head.weight if model.head is not None else model.embed.weight.T
    model.head = QuantLinear(head, bits, group)
    return model
