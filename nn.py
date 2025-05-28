import random
from typing import Any
from tinygrad.engine import Value


class Base:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0  # reset

    def parameters(self):
        return []  # returns weight and bias for each neuron


class Neuron(Base):
    def __init__(self, inp, non_lin=True) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(inp)]
        self.b = Value(0)
        self.non_lin = non_lin

    def __call__(self, x):
        activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return (
            activation.tanh() if self.non_lin else activation
        )  # if linear -> no need to apply activation function (can use relu also)

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self) -> str:
        return f"{'tanh' if self.nonlin else 'Linear'} Neuron ({len(self.w)})"


class Layer(Base):
    def __init__(self, inp, output) -> None:
        self.neurons = [Neuron(inp) for _ in range(output)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return (
            out[0] if len(out) == 1 else out
        )  # return list only if there are multiple values

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self) -> str:
        return f"Layer [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Base):
    # passes inp and out for multiple layers
    def __init__(self, inp, output_list):
        size = [inp] + output_list
        self.layers = [Layer(size[i], size[i + 1]) for i in range(len(output_list))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
