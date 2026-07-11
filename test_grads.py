#!/usr/bin/env python3
"""Gradient correctness check: analytic backward vs numerical finite differences.

No pytest, no torch, no CI matrix. Just run it:

    python test_grads.py

Each check perturbs every input and compares the numerical gradient against the
one from .backward(). Exits nonzero if any gradient is wrong.
"""
import numpy as np

from minitorch import (
    Tensor, Linear, Conv2d, MaxPool2d, BatchNorm1d, LayerNorm, Embedding,
    mse_loss, cross_entropy_loss, bce_loss,
)
from minitorch import functional as F
from minitorch.grad_check import check_gradient


def randt(*shape, positive=False):
    data = np.random.rand(*shape) + 0.5 if positive else np.random.randn(*shape)
    return Tensor(data, requires_grad=True)


CHECKS = []


def check(name):
    def register(fn):
        CHECKS.append((name, fn))
        return fn
    return register


# arithmetic

@check("add")
def _():
    a, b = randt(3, 4), randt(3, 4)
    check_gradient(lambda: (a + b).sum(), [a, b])


@check("sub")
def _():
    a, b = randt(3, 4), randt(3, 4)
    check_gradient(lambda: (a - b).sum(), [a, b])


@check("mul")
def _():
    a, b = randt(3, 4), randt(3, 4)
    check_gradient(lambda: (a * b).sum(), [a, b])


@check("div")
def _():
    a, b = randt(3, 4), randt(3, 4, positive=True)
    check_gradient(lambda: (a / b).sum(), [a, b])


@check("pow")
def _():
    a = randt(3, 4)
    check_gradient(lambda: (a ** 2).sum(), [a])


@check("neg")
def _():
    a = randt(3, 4)
    check_gradient(lambda: (-a).sum(), [a])


# broadcasting

@check("add-broadcast")
def _():
    a, b = randt(3, 4), randt(4)
    check_gradient(lambda: (a + b).sum(), [a, b])


@check("mul-broadcast")
def _():
    a, b = randt(3, 4), randt(1, 4)
    check_gradient(lambda: (a * b).sum(), [a, b])


# matmul

@check("matmul-2d")
def _():
    a, b = randt(3, 5), randt(5, 2)
    check_gradient(lambda: (a @ b).sum(), [a, b])


@check("matmul-batched")
def _():
    a, b = randt(4, 3, 5), randt(4, 5, 2)
    check_gradient(lambda: (a @ b).sum(), [a, b])


@check("matmul-broadcast-batch")
def _():
    a, b = randt(4, 3, 5), randt(5, 2)
    check_gradient(lambda: (a @ b).sum(), [a, b])


# reductions

@check("sum-axis")
def _():
    a = randt(3, 4)
    check_gradient(lambda: (a.sum(axis=0) ** 2).sum(), [a])


@check("mean-axis")
def _():
    a = randt(3, 4)
    check_gradient(lambda: (a.mean(axis=1) ** 2).sum(), [a])


@check("max")
def _():
    a = randt(3, 4)
    check_gradient(lambda: a.max(axis=1).sum(), [a])


@check("min")
def _():
    a = randt(3, 4)
    check_gradient(lambda: a.min().sum(), [a])


# elementwise

@check("exp")
def _():
    a = randt(3, 4)
    check_gradient(lambda: a.exp().sum(), [a])


@check("log")
def _():
    a = randt(3, 4, positive=True)
    check_gradient(lambda: a.log().sum(), [a])


# shape

@check("reshape")
def _():
    a = randt(3, 4)
    check_gradient(lambda: (a.reshape(2, 6) ** 2).sum(), [a])


@check("transpose")
def _():
    a = randt(3, 4)
    check_gradient(lambda: (a.transpose() ** 2).sum(), [a])


# activations

@check("relu")
def _():
    a = randt(3, 4)
    check_gradient(lambda: F.relu(a).sum(), [a])


@check("sigmoid")
def _():
    a = randt(3, 4)
    check_gradient(lambda: F.sigmoid(a).sum(), [a])


@check("tanh")
def _():
    a = randt(3, 4)
    check_gradient(lambda: F.tanh(a).sum(), [a])


@check("gelu")
def _():
    a = randt(3, 4)
    check_gradient(lambda: F.gelu(a).sum(), [a])


@check("softmax")
def _():
    a = randt(3, 4)
    check_gradient(lambda: (F.softmax(a, axis=-1) ** 2).sum(), [a])


@check("log_softmax")
def _():
    a = randt(3, 4)
    check_gradient(lambda: F.log_softmax(a, axis=-1).sum(), [a])


# layers

@check("linear")
def _():
    layer = Linear(5, 3)
    x = randt(4, 5)
    check_gradient(lambda: layer(x).sum(), [x, layer.weight, layer.bias])


@check("layernorm")
def _():
    ln = LayerNorm(6)
    x = randt(3, 6)
    check_gradient(lambda: ln(x).sum(), [x, ln.gamma, ln.beta])


@check("batchnorm")
def _():
    bn = BatchNorm1d(4)
    x = randt(8, 4)
    check_gradient(lambda: bn(x).sum(), [x, bn.gamma, bn.beta])


@check("conv2d")
def _():
    conv = Conv2d(2, 3, 3, padding=1)
    x = randt(2, 2, 5, 5)
    check_gradient(lambda: conv(x).sum(), [x, conv.weight, conv.bias])


@check("maxpool2d")
def _():
    pool = MaxPool2d(2)
    x = randt(1, 2, 4, 4)
    check_gradient(lambda: pool(x).sum(), [x])


@check("embedding")
def _():
    emb = Embedding(10, 4)
    idx = np.array([[1, 3, 5], [2, 2, 9]])
    check_gradient(lambda: emb(idx).sum(), [emb.weight])


# losses

@check("mse_loss")
def _():
    pred, target = randt(4, 3), randt(4, 3)
    check_gradient(lambda: mse_loss(pred, target), [pred])


@check("cross_entropy_loss")
def _():
    logits = randt(4, 3)
    target = Tensor(np.array([0, 1, 2, 1]))
    check_gradient(lambda: cross_entropy_loss(logits, target), [logits])


@check("bce_loss")
def _():
    p = Tensor(np.random.rand(3, 2) * 0.8 + 0.1, requires_grad=True)
    target = Tensor((np.random.rand(3, 2) > 0.5).astype(np.float32))
    check_gradient(lambda: bce_loss(p, target), [p])


def main():
    np.random.seed(0)
    failed = []
    for name, fn in CHECKS:
        try:
            fn()
            print(f"[ok]   {name}")
        except Exception as e:
            print(f"[FAIL] {name}: {str(e).splitlines()[0]}")
            failed.append(name)
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} gradient checks passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
