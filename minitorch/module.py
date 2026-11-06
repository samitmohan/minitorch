import os
from pathlib import Path

import numpy as np

from . import backend
from .tensor import Tensor


def _dedup(params):
    # drop tensors that appear more than once (e.g. tied weights) so optimizers
    # don't build duplicate state or step them twice, keeping first-seen order
    seen = set()
    out = []
    for p in params:
        if id(p) not in seen:
            seen.add(id(p))
            out.append(p)
    return out


class Module:
    """Base class for layers and models.

    Subclass it and implement `forward`. Any `Tensor` or `Module` assigned as an
    attribute is discovered automatically by `parameters()`, `state_dict()`, and
    the `train`/`eval` switches.
    """
    def __init__(self):
        self._training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        """Collect every trainable `Tensor` in this module and its children."""
        return _dedup(p for _, p in self.named_parameters())

    def named_parameters(self, prefix=""):
        """(dotted name, tensor) for every trainable tensor, named as in state_dict(), without copying."""
        for name, val in self.__dict__.items():
            if isinstance(val, Tensor) and val.requires_grad:
                yield prefix + name, val
            elif isinstance(val, Module):
                yield from val.named_parameters(f"{prefix}{name}.")
            elif isinstance(val, (list, tuple)):
                for i, item in enumerate(val):
                    if isinstance(item, Module):
                        yield from item.named_parameters(f"{prefix}{name}.{i}.")

    def train(self):
        self._training = True
        for val in self.__dict__.values():
            if isinstance(val, Module):
                val.train()
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, Module):
                        item.train()
        return self

    def eval(self):
        self._training = False
        for val in self.__dict__.values():
            if isinstance(val, Module):
                val.eval()
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, Module):
                        item.eval()
        return self

    def to(self, device=None, dtype=None):
        """Move every parameter to `device` and/or cast it to `dtype` ("bfloat16", ...), in place.

        In place, so an optimizer built before the move still holds the same tensors.
        Every tensor moves, frozen ones (requires_grad=False) included. Gradients are
        dropped: they would sit on the old device.
        """
        moved = []
        for val in self.__dict__.values():
            items = val if isinstance(val, (list, tuple)) else [val]
            for item in items:
                if isinstance(item, Tensor):
                    copy = item.to(device, dtype)
                    item.data, item.device, item.grad = copy.data, copy.device, None
                    moved.append(item)
                elif isinstance(item, Module):
                    item.to(device, dtype)
        # compute the moved arrays now: MLX would otherwise leave a lazy cast tied to this thread's
        # stream, and a model handed to another thread (a serving engine) could not evaluate it
        for t in moved:
            backend.get(t.device).realize(t.data)
        return self

    def save(self, path):
        """Write all parameters to a .npz file. Written beside it first, then renamed over it, so a crash mid-write
        leaves the previous checkpoint whole instead of a truncated file in its place."""
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp.npz")  # savez appends .npz to a name without it
        np.savez(tmp, **self.state_dict())
        os.replace(tmp, path)

    def load(self, path):
        """Load parameters from a .npz file written by save(). Refuses a file whose names or shapes differ from the
        model's: a checkpoint from another architecture would otherwise load in part and run with random weights."""
        with np.load(path) as data:
            state = {k: data[k] for k in data.files}
        expected = {k: v.shape for k, v in self.state_dict().items()}
        got = {k: v.shape for k, v in state.items()}
        assert got == expected, f"{path} does not match this model: {sorted(set(got) ^ set(expected))[:4] or 'shapes differ'}"
        self.load_state_dict(state)
        return self

    def state_dict(self):
        state = {}
        for name, val in self.__dict__.items():
            if isinstance(val, Tensor) and val.requires_grad:
                state[name] = val.numpy()
            elif isinstance(val, np.ndarray):
                # BatchNorm running_mean/running_var are plain arrays, not
                # parameters. Omitting them makes a reloaded model normalize
                # with (0, 1) in eval mode.
                state[name] = val.copy()
            elif isinstance(val, Module):
                for k, v in val.state_dict().items():
                    state[f"{name}.{k}"] = v
            elif isinstance(val, (list, tuple)):
                for i, item in enumerate(val):
                    if isinstance(item, Module):
                        for k, v in item.state_dict().items():
                            state[f"{name}.{i}.{k}"] = v
        return state

    def load_state_dict(self, state):
        for name, val in self.__dict__.items():
            if isinstance(val, Tensor) and val.requires_grad:
                if name in state:
                    assert tuple(state[name].shape) == tuple(val.shape), (name, state[name].shape, val.shape)
                    val.data = backend.get(val.device).asarray(state[name]).astype(val.dtype)
            elif isinstance(val, np.ndarray):
                if name in state:
                    setattr(self, name, state[name].copy())
            elif isinstance(val, Module):
                child_state = {}
                prefix = f"{name}."
                for k, v in state.items():
                    if k.startswith(prefix):
                        child_state[k[len(prefix):]] = v
                if child_state:
                    val.load_state_dict(child_state)
            elif isinstance(val, (list, tuple)):
                for i, item in enumerate(val):
                    if isinstance(item, Module):
                        child_state = {}
                        prefix = f"{name}.{i}."
                        for k, v in state.items():
                            if k.startswith(prefix):
                                child_state[k[len(prefix):]] = v
                        if child_state:
                            item.load_state_dict(child_state)


class Sequential(Module):
    """Chain modules so the output of each feeds the next.

    Parameter collection, train/eval, and state_dict all come from Module, which
    already walks the `layers` list.
    """
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
