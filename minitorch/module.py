import numpy as np
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
        params = []
        for val in self.__dict__.values():
            if isinstance(val, Tensor) and val.requires_grad:
                params.append(val)
            elif isinstance(val, Module):
                params.extend(val.parameters())
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, Tensor) and item.requires_grad:
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return _dedup(params)

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

    def save(self, path):
        """Write all parameters to a .npz file."""
        np.savez(path, **self.state_dict())

    def load(self, path):
        """Load parameters from a .npz file written by save()."""
        with np.load(path) as data:
            self.load_state_dict({k: data[k] for k in data.files})
        return self

    def state_dict(self):
        state = {}
        for name, val in self.__dict__.items():
            if isinstance(val, Tensor) and val.requires_grad:
                state[name] = val.data.copy()
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
                    val.data = state[name].copy()
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
    """Chain modules so the output of each feeds the next."""
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return _dedup(params)

    def train(self):
        self._training = True
        for layer in self.layers:
            if isinstance(layer, Module):
                layer.train()
        return self

    def eval(self):
        self._training = False
        for layer in self.layers:
            if isinstance(layer, Module):
                layer.eval()
        return self

    def state_dict(self):
        state = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'state_dict'):
                for k, v in layer.state_dict().items():
                    state[f"layers.{i}.{k}"] = v
        return state

    def load_state_dict(self, state):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'load_state_dict'):
                child_state = {}
                prefix = f"layers.{i}."
                for k, v in state.items():
                    if k.startswith(prefix):
                        child_state[k[len(prefix):]] = v
                if child_state:
                    layer.load_state_dict(child_state)
