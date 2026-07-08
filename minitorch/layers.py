import numpy as np
from .tensor import Tensor, _accum_grad
from .module import Module
from . import functional as F


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, init='kaiming'):
        super().__init__()
        assert in_features > 0 and out_features > 0, "features must be positive"
        if init == 'xavier':
            # glorot uniform - good for sigmoid/tanh
            scale = np.sqrt(2.0 / (in_features + out_features))
        else:
            # kaiming - good for relu
            scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            (np.random.randn(in_features, out_features) * scale).astype(np.float32),
            requires_grad=True
        )
        self.bias = Tensor(
            np.zeros(out_features, dtype=np.float32),
            requires_grad=True
        ) if bias else None

    def forward(self, x):
        assert x.data.shape[-1] == self.weight.data.shape[0], \
            f"Linear expects last dim {self.weight.data.shape[0]}, got {x.data.shape[-1]}"
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


class ReLU(Module):
    def forward(self, x):
        return F.relu(x)


class Sigmoid(Module):
    def forward(self, x):
        return F.sigmoid(x)


class Tanh(Module):
    def forward(self, x):
        return F.tanh(x)


class Softmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return F.softmax(x, axis=self.axis)


class GELU(Module):
    def forward(self, x):
        return F.gelu(x)


class LayerNorm(Module):
    """Normalize over the last dimension. Works on any (..., num_features) input."""
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        assert num_features > 0, "num_features must be positive"
        self.gamma = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        assert x.data.shape[-1] == self.gamma.data.size, \
            f"LayerNorm expects last dim {self.gamma.data.size}, got {x.data.shape[-1]}"
        mean = x.mean(axis=-1, keepdims=True)
        diff = x - mean
        var = (diff ** 2).mean(axis=-1, keepdims=True)
        x_hat = diff / (var + self.eps) ** 0.5
        return x_hat * self.gamma + self.beta


class Embedding(Module):
    """Lookup table mapping integer ids to dense vectors."""
    def __init__(self, num_embeddings, embed_dim):
        super().__init__()
        assert num_embeddings > 0 and embed_dim > 0, "sizes must be positive"
        self.weight = Tensor(
            (np.random.randn(num_embeddings, embed_dim) * 0.02).astype(np.float32),
            requires_grad=True
        )

    def forward(self, idx):
        # idx is a numpy int array of any shape; getitem scatters grad back on backward
        return self.weight[np.asarray(idx)]


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        assert 0.0 <= p < 1.0, "dropout probability must be in [0, 1)"
        self.p = p

    def forward(self, x):
        if not self._training or self.p == 0.0:
            return x
        mask = (np.random.rand(*x.data.shape) > self.p).astype(np.float32)
        scale = 1.0 / (1.0 - self.p)
        out = Tensor(x.data * mask * scale, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad:
                _accum_grad(x, out.grad * mask * scale)

        out._backward = _backward
        out._prev = {x}
        out._op = 'dropout'
        return out


class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        assert num_features > 0, "num_features must be positive"
        self.gamma = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

    def forward(self, x):
        assert x.data.ndim == 2 and x.data.shape[1] == self.gamma.data.size, \
            f"Expected input shape (batch, {self.gamma.data.size}), got {x.data.shape}"

        if self._training:
            mean = x.mean(axis=0, keepdims=True)
            diff = x - mean
            var = (diff ** 2).mean(axis=0, keepdims=True)
            x_hat = diff / (var + self.eps) ** 0.5

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data.flatten()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data.flatten()
        else:
            rm = Tensor(self.running_mean.reshape(1, -1))
            rv = Tensor(self.running_var.reshape(1, -1))
            x_hat = (x - rm) / (rv + self.eps) ** 0.5

        return x_hat * self.gamma + self.beta
