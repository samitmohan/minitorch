import numpy as np
from .tensor import Tensor


class Module:
    def __init__(self):
        self._training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
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
        return params

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
        return params

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
