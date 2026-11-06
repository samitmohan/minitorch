"""Activation checkpointing: keep a block's input, not its activations, and recompute them in backward.

Training keeps every intermediate result of the forward pass alive until backward has used
it, so memory grows with layers x tokens. Wrapped in checkpoint(), a block's forward runs
without recording a graph, and only its input stays alive. When backward reaches the block,
it runs the block again, this time recording, and backpropagates through that copy. Memory
for activations drops to one block's worth plus every block's input; the price is a second
forward pass of each block.

This defines no new derivative, only when the existing ones (mlops.py) run.
"""
from . import backend
from . import tensor as T
from .mlops import Function


class _Checkpoint(Function):
    def forward(self, *arrays, fn):
        self.fn, self.arrays = fn, arrays
        device = self.parents[0].device
        with T.no_grad():
            out = fn(*[T.Tensor(a, device=device) for a in arrays]).data
        # a lazy backend (MLX) would otherwise run this forward later, inside one big evaluation that holds everything
        self.xp.realize(out)
        return out

    def backward(self, grad):
        device = self.parents[0].device
        leaves = [T.Tensor(a, requires_grad=needs, device=device) for a, needs in zip(self.arrays, self.needs_grad)]
        out = self.fn(*leaves)
        # d/d(anything) of sum(out * grad) is grad pushed back through fn: weights inside fn collect their .grad here
        (out * T.Tensor(grad, device=device)).sum().backward()
        # finish this block's backward now, so its recomputed activations are freed before the next block's arrive
        weights = self.fn.parameters() if hasattr(self.fn, "parameters") else []
        self.xp.realize(*[t.grad for t in leaves + list(weights) if t.grad is not None])
        return tuple(leaf.grad for leaf in leaves)


def checkpoint(fn, *inputs):
    """fn(*inputs), recomputed during backward instead of storing its activations. fn must be deterministic."""
    if not T.grad_enabled():
        return fn(*inputs)
    device = inputs[0].device
    ctx = _Checkpoint(backend.get(device), *inputs)
    out = T.Tensor(ctx.forward(*[x.data for x in inputs], fn=fn), device=device)
    # recorded even when no input needs a gradient: weights inside fn may (LoRA trains adapters inside frozen layers)
    out.requires_grad, out._ctx = True, ctx
    return out
