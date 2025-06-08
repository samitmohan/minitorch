import numpy as np
from .tensor import Tensor

class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.weight = Tensor(
            np.random.randn(in_features, out_features).astype(np.float32),
            requires_grad=True
        )
        self.bias = Tensor(
            np.zeros(out_features, dtype=np.float32),
            requires_grad=True
        ) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        assert x.data.ndim == 2, "Linear expects a 2D input (batch, features)"
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y

    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params


class ReLU:
    def __call__(self, x: Tensor) -> Tensor:
        data = np.maximum(0, x.data)
        out = Tensor(data, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad:
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                x.grad += (x.data > 0) * out.grad

        out._backward = _backward
        out._prev = {x}
        out._op = 'relu'
        return out


class BatchNorm1d:
    def __init__(self, num_features, eps=1e-5):
        self.gamma = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)
        self.eps = eps

    def __call__(self, x: Tensor) -> Tensor:
        # Expect input shape (batch_size, num_features)
        assert x.data.ndim == 2 and x.data.shape[1] == self.gamma.data.size, \
            "BatchNorm1d expects input shape (batch, num_features)"
        # Compute batch statistics (training mode)
        mean = Tensor(x.data.mean(axis=0), requires_grad=False)
        var = Tensor(x.data.var(axis=0), requires_grad=False)
        # Normalize
        x_hat = (x + (mean * -1.0)) * Tensor(1.0 / np.sqrt(var.data + self.eps), requires_grad=False)
        out = x_hat * self.gamma + self.beta

        def _backward():
            raise NotImplementedError("BatchNorm1d backward is not implemented.")

        out._backward = _backward
        out._prev = {x, self.gamma, self.beta}
        out._op = 'batchnorm'
        return out

    def parameters(self):
        return [self.gamma, self.beta]
