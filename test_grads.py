#!/usr/bin/env python3
"""Gradient correctness check: analytic backward vs numerical finite differences.

No pytest, no torch, no CI matrix. Just run it:

    python test_grads.py

Each check perturbs every input and compares the numerical gradient against the
one from .backward(). Exits nonzero if any gradient is wrong.
"""
import numpy as np

from minitorch import (
    Attention,
    BatchNorm1d,
    Conv2d,
    Embedding,
    LayerNorm,
    Linear,
    MaxPool2d,
    RMSNorm,
    SwiGLU,
    Tensor,
    TransformerLM,
    bce_loss,
    cat,
    cross_entropy_loss,
    mse_loss,
    stack,
    where,
)
from minitorch import functional as F
from minitorch.grad_check import check_gradient
from minitorch.layers import DepthAttention


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


@check("getitem-slice")
def _():
    a = randt(4, 5)
    check_gradient(lambda: (a[1:3, ::2] ** 2).sum(), [a])


@check("getitem-gather-repeat")
def _():
    a = randt(4, 3)
    idx = np.array([0, 2, 2, 3])  # row 2 picked twice: its grads must add
    check_gradient(lambda: (a[idx] ** 2).sum(), [a])


@check("cat")
def _():
    a, b = randt(2, 3), randt(4, 3)
    check_gradient(lambda: (cat([a, b], axis=0) ** 2).sum(), [a, b])


@check("stack")
def _():
    a, b = randt(2, 3), randt(2, 3)
    check_gradient(lambda: (stack([a, b], axis=1) ** 2).sum(), [a, b])


@check("where")
def _():
    a, b = randt(3, 4), randt(1, 4)
    mask = np.random.rand(3, 4) > 0.5
    check_gradient(lambda: (where(mask, a, b) ** 2).sum(), [a, b])


@check("clamp")
def _():
    a = randt(3, 4)
    check_gradient(lambda: (a.clamp(-0.5, 0.5) ** 2).sum(), [a])


@check("abs")
def _():
    a = randt(3, 4)
    check_gradient(lambda: (a.abs() ** 2).sum(), [a])


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
    w = Tensor(np.random.randn(3, 6))  # with a plain .sum() the true input gradient is exactly 0, which hides a wrong one
    check_gradient(lambda: (ln(x) * w).sum(), [x, ln.gamma, ln.beta])


@check("batchnorm")
def _():
    bn = BatchNorm1d(4)
    x = randt(8, 4)
    w = Tensor(np.random.randn(8, 4))  # as for layernorm: a plain .sum() makes the input gradient 0
    check_gradient(lambda: (bn(x) * w).sum(), [x, bn.gamma, bn.beta])


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


@check("silu")
def _():
    a = randt(3, 4)
    check_gradient(lambda: F.silu(a).sum(), [a])


@check("rmsnorm")
def _():
    norm = RMSNorm(6)
    x = randt(3, 6)
    w = Tensor(np.random.randn(3, 6))  # non-uniform upstream grad, so the check sees cross-terms
    check_gradient(lambda: (norm(x) * w).sum(), [x, norm.weight])


@check("depth-attention")
def _():
    depth = DepthAttention(6)
    depth.query.data = np.random.randn(6)  # off zero, where every source's weight would be the same
    sources = [randt(2, 3, 6) for _ in range(3)]
    w = Tensor(np.random.randn(2, 3, 6))
    check_gradient(lambda: (depth(sources) * w).sum(), [depth.query, depth.norm.weight, *sources])


@check("swiglu")
def _():
    ffn = SwiGLU(4, 6)
    x = randt(2, 3, 4)
    check_gradient(lambda: (ffn(x) ** 2).sum(), [x, ffn.w1.weight, ffn.w2.weight, ffn.w3.weight])


@check("attention-gqa-rope")
def _():
    attn = Attention(8, n_heads=4, n_kv_heads=2)
    x = randt(2, 5, 8)
    params = [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight, attn.o_proj.weight]
    check_gradient(lambda: (attn(x) ** 2).sum(), [x, *params])


@check("attention-head-dim-8")
def _():
    # head_dim 8, the smallest the Metal kernels take: tests/backends.py runs this through them
    attn = Attention(16, n_heads=2, n_kv_heads=1)
    x = randt(1, 3, 16)
    params = [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight, attn.o_proj.weight]
    check_gradient(lambda: (attn(x) ** 2).sum(), [x, *params])


@check("attention-left-padding")
def _():
    # the key mask every GRPO update backpropagates through: batch row 1's first two positions are padding. The loss
    # covers real positions only, as every training loss here does; a padding query's scores sit near -1e9, where
    # float64's spacing (~1e-7) is too coarse for a finite difference, and its output is never used
    from minitorch.attention import scaled_dot_product_attention
    q, k, v = randt(2, 2, 4, 4), randt(2, 2, 4, 4), randt(2, 2, 4, 4)
    real = np.ones((2, 1, 4, 1))
    real[1, :, :2] = 0
    check_gradient(lambda: ((scaled_dot_product_attention(q, k, v, pad=np.array([0, 2])) ** 2) * real).sum(), [q, k, v])


@check("attention-cached-keys")
def _():
    # decoding: 2 new queries against 5 keys, so the causal mask is offset by S - T
    from minitorch.attention import scaled_dot_product_attention
    q, k, v = randt(1, 2, 2, 4), randt(1, 2, 5, 4), randt(1, 2, 5, 4)
    check_gradient(lambda: (scaled_dot_product_attention(q, k, v) ** 2).sum(), [q, k, v])


@check("attention-non-causal")
def _():
    # cross-attention, as Stable Diffusion's UNet uses it: every query sees all 6 keys
    from minitorch.attention import scaled_dot_product_attention
    q, k, v = randt(1, 2, 3, 4), randt(1, 2, 6, 4), randt(1, 2, 6, 4)
    check_gradient(lambda: (scaled_dot_product_attention(q, k, v, causal=False) ** 2).sum(), [q, k, v])


@check("lora-linear")
def _():
    from minitorch.lora import LoRALinear
    layer = LoRALinear(Linear(5, 3, bias=False), r=2, alpha=4)
    layer.B.data = np.random.randn(2, 3)  # B starts at zero, which would hide a wrong gradient for A
    x = randt(2, 4, 5)
    check_gradient(lambda: (layer(x) ** 2).sum(), [x, layer.A, layer.B])


@check("transformer-lm")
def _():
    lm = TransformerLM(vocab_size=11, context_length=8, d_model=8, n_layers=1, n_heads=2, d_ff=12)
    idx = np.array([[1, 4, 4, 9], [0, 2, 7, 3]])
    targets = np.array([[4, 4, 9, 1], [2, 7, 3, 3]])
    check_gradient(lambda: cross_entropy_loss(lm(idx), targets), lm.parameters())


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
        except Exception as e:  # noqa: BLE001 - report every failing check, not just the first
            print(f"[FAIL] {name}: {str(e).splitlines()[0]}")
            failed.append(name)
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} gradient checks passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
