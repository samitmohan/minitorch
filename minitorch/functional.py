import numpy as np
from .tensor import Tensor, _accum_grad

# Functional versions of activations - no Module needed


def relu(x):
    data = np.maximum(0, x.data)
    out = Tensor(data, requires_grad=x.requires_grad)

    def _backward():
        if x.requires_grad:
            _accum_grad(x, (x.data > 0) * out.grad)

    out._backward = _backward
    out._prev = {x}
    out._op = 'relu'
    return out


def sigmoid(x):
    s = 1.0 / (1.0 + np.exp(-x.data))
    out = Tensor(s, requires_grad=x.requires_grad)

    def _backward():
        if x.requires_grad:
            _accum_grad(x, out.data * (1.0 - out.data) * out.grad)

    out._backward = _backward
    out._prev = {x}
    out._op = 'sigmoid'
    return out


def tanh(x):
    t = np.tanh(x.data)
    out = Tensor(t, requires_grad=x.requires_grad)

    def _backward():
        if x.requires_grad:
            _accum_grad(x, (1.0 - out.data ** 2) * out.grad)

    out._backward = _backward
    out._prev = {x}
    out._op = 'tanh'
    return out


def softmax(x, axis=-1):
    shifted = x.data - x.data.max(axis=axis, keepdims=True)
    exps = np.exp(shifted)
    s = exps / exps.sum(axis=axis, keepdims=True)
    out = Tensor(s, requires_grad=x.requires_grad)

    def _backward():
        if x.requires_grad:
            g = out.grad * out.data
            g = g - out.data * g.sum(axis=axis, keepdims=True)
            _accum_grad(x, g)

    out._backward = _backward
    out._prev = {x}
    out._op = 'softmax'
    return out


def log_softmax(x, axis=-1):
    shifted = x.data - x.data.max(axis=axis, keepdims=True)
    log_sum_exp = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    data = shifted - log_sum_exp
    out = Tensor(data, requires_grad=x.requires_grad)

    def _backward():
        if x.requires_grad:
            s = np.exp(out.data)
            grad = out.grad - s * out.grad.sum(axis=axis, keepdims=True)
            _accum_grad(x, grad)

    out._backward = _backward
    out._prev = {x}
    out._op = 'log_softmax'
    return out
