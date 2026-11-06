"""LoRA: train a low-rank update beside each frozen weight instead of the weight itself.

    y = x W + (alpha / r) * (x A) B        W (in, out) frozen; A (in, r) and B (r, out) trained

B starts at zero, so training starts from exactly the base model. Only A and B,
r * (in + out) numbers per layer, get gradients and optimizer state. After training,
merge_adapter folds the update into W, so the fine-tuned model runs at the base
model's speed and in any dtype.
"""
import numpy as np

from .layers import Linear
from .module import Module
from .tensor import Tensor

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3")  # every matmul in a TransformerLM block


class LoRALinear(Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        n_in, n_out = base.weight.shape
        self.base = base
        # cast after dividing: since NumPy 2, a float32 array divided by np.sqrt's float64 scalar comes out float64
        init = (np.random.randn(n_in, r) / np.sqrt(n_in)).astype(np.float32)
        self.A = Tensor(init, requires_grad=True, device=base.weight.device)
        self.B = Tensor(np.zeros((r, n_out), np.float32), requires_grad=True, device=base.weight.device)
        self.scale = alpha / r

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        delta = (flat @ self.A * self.scale) @ self.B  # scaling the rank-r middle, not the full-width output, saves a copy
        return self.base(x) + delta.reshape(*x.shape[:-1], delta.shape[-1])


def apply_lora(model, r=16, alpha=32):
    """Freeze every weight of a TransformerLM, then wrap each target Linear in a LoRALinear.

    Afterwards model.parameters() is exactly the LoRA weights, ready for an optimizer.
    """
    for _, p in model.named_parameters():
        p.requires_grad = False
    for block in model.blocks:
        for parent in (block.attn, block.ffn):
            for name in TARGETS:
                if isinstance(getattr(parent, name, None), Linear):
                    setattr(parent, name, LoRALinear(getattr(parent, name), r, alpha))
    return model


def save_adapter(model, path, r, alpha):
    np.savez(path, r=r, alpha=alpha, **{name: p.numpy() for name, p in model.named_parameters()})


def merge_adapter(model, path):
    """Add a saved adapter's (alpha / r) A B into the matching weights of a plain (un-wrapped) model."""
    adapter = np.load(path)
    scale = float(adapter["alpha"]) / float(adapter["r"])
    params = dict(model.named_parameters())
    layers = [k[:-len(".A")] for k in adapter.files if k.endswith(".A")]
    for layer in layers:
        w = params[f"{layer}.weight"]  # a KeyError here means the adapter was trained on another model
        assert w.device == "cpu", "merge the adapter before moving the model to a device"
        w.data = w.data + (scale * (adapter[f"{layer}.A"] @ adapter[f"{layer}.B"])).astype(w.dtype)
    return len(layers)
