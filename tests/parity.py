#!/usr/bin/env python3
"""Parity check: every minitorch op and layer against its PyTorch twin, outputs and grads, fp32.

    uv run python tests/parity.py

Each check runs the same inputs and weights through both, backprops the same
random upstream gradient, and compares. Exits nonzero if anything differs.
"""
import itertools

import numpy as np
import torch
import torch.nn.functional as TF

import minitorch as mt
from minitorch import functional as F

ATOL, RTOL = 1e-5, 1e-4
CHECKS = []


def check(name):
    def register(fn):
        CHECKS.append((name, fn))
        return fn
    return register


def close(name, a, b, atol=ATOL, rtol=RTOL):
    b = b.detach().numpy() if isinstance(b, torch.Tensor) else b
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(f"{name}: max diff {np.max(np.abs(a - b)):.3g}")


def compare(mt_fn, th_fn, *arrays, params=(), atol=ATOL, rtol=RTOL):
    """Run both, backprop the same random upstream grad, compare output, input grads, and (mt, torch) param grads."""
    xs = [mt.Tensor(a, requires_grad=True) for a in arrays]
    ts = [torch.tensor(a, requires_grad=True) for a in arrays]
    out, tout = mt_fn(*xs), th_fn(*ts)
    close("output", out.data, tout, atol, rtol)
    w = np.asarray(np.random.randn(*out.shape), dtype=np.float32)  # randn() with no dims returns a float
    (out * w).sum().backward()
    (tout * torch.tensor(w)).sum().backward()
    for i, (x, t) in enumerate(zip(xs, ts)):
        close(f"grad input {i}", x.grad, t.grad, atol, rtol)
    for i, (p, tp) in enumerate(params):
        close(f"grad param {i}", p.grad, tp.grad, atol, rtol)


def r(*shape):
    return np.random.randn(*shape).astype(np.float32)


def twin(p):
    """A torch leaf holding a copy of minitorch param p."""
    return torch.tensor(p.data.copy(), requires_grad=True)


# ops

@check("add/sub/mul/div broadcast")
def _():
    compare(lambda a, b: (a + b) * a - a / (b * b + 1), lambda a, b: (a + b) * a - a / (b * b + 1), r(3, 4), r(4))


@check("pow")
def _():
    compare(lambda a: a ** 3, lambda a: a ** 3, r(3, 4))


@check("matmul batched broadcast")
def _():
    compare(lambda a, b: a @ b, lambda a, b: a @ b, r(2, 3, 4, 5), r(5, 6))


@check("sum/mean/max over axes")
def _():
    compare(lambda a: a.sum(axis=1) + a.mean(axis=1) + a.max(axis=1),
            lambda a: a.sum(1) + a.mean(1) + a.amax(1), r(3, 5, 4))


@check("exp/log/sigmoid/tanh/relu")
def _():
    compare(lambda a: a.exp() + (a * a + 1).log() + a.sigmoid() + a.tanh() + a.relu(),
            lambda a: a.exp() + (a * a + 1).log() + a.sigmoid() + a.tanh() + a.relu(), r(3, 4))


@check("gelu/silu")
def _():
    compare(lambda a: F.gelu(a) + F.silu(a), lambda a: TF.gelu(a, approximate="tanh") + TF.silu(a), r(3, 4))


@check("softmax/log_softmax")
def _():
    compare(lambda a: F.softmax(a) + F.log_softmax(a), lambda a: a.softmax(-1) + a.log_softmax(-1), r(3, 7))


@check("reshape/permute/transpose/slice/gather/cat")
def _():
    idx = np.array([0, 2, 2, 1])
    compare(lambda a: mt.cat([a.permute(2, 0, 1).reshape(6, 4)[idx], a.transpose(0, 1)[:, ::2].reshape(3, 4)]),
            lambda a: torch.cat([a.permute(2, 0, 1).reshape(6, 4)[idx], a.transpose(0, 1)[:, ::2].reshape(3, 4)]),
            r(2, 3, 4))


# layers

@check("Linear")
def _():
    m = mt.Linear(5, 3)
    w, b = twin(m.weight), twin(m.bias)
    compare(m, lambda x: x @ w + b, r(2, 4, 5), params=[(m.weight, w), (m.bias, b)])


@check("LayerNorm")
def _():
    m = mt.LayerNorm(6)
    m.gamma.data, m.beta.data = r(6), r(6)
    g, b = twin(m.gamma), twin(m.beta)
    compare(m, lambda x: TF.layer_norm(x, (6,), g, b, eps=1e-5), r(3, 6), params=[(m.gamma, g), (m.beta, b)])


@check("GroupNorm (Stable Diffusion's 32 groups)")
def _():
    m = mt.GroupNorm(4, 8)
    m.weight.data, m.bias.data = r(8), r(8)
    w, b = twin(m.weight), twin(m.bias)
    compare(m, lambda x: TF.group_norm(x, 4, w, b, eps=1e-5), r(2, 8, 5, 3), params=[(m.weight, w), (m.bias, b)])


@check("RMSNorm")
def _():
    m = mt.RMSNorm(6)
    m.weight.data = r(6)
    g = twin(m.weight)
    compare(m, lambda x: TF.rms_norm(x, (6,), g, eps=1e-5), r(2, 3, 6), params=[(m.weight, g)])


@check("SwiGLU")
def _():
    m = mt.SwiGLU(4, 6)
    w1, w2, w3 = twin(m.w1.weight), twin(m.w2.weight), twin(m.w3.weight)
    compare(m, lambda x: (TF.silu(x @ w1) * (x @ w3)) @ w2, r(2, 3, 4),
            params=[(m.w1.weight, w1), (m.w2.weight, w2), (m.w3.weight, w3)])


@check("Embedding")
def _():
    m = mt.Embedding(10, 4)
    w = twin(m.weight)
    idx = np.array([[1, 3, 3], [9, 0, 1]])
    compare(lambda s: m(idx) * s, lambda s: w[torch.tensor(idx)] * s, r(1), params=[(m.weight, w)])


@check("Conv2d + MaxPool2d")
def _():
    conv, pool = mt.Conv2d(2, 3, 3, stride=1, padding=1), mt.MaxPool2d(2)
    w, b = twin(conv.weight), twin(conv.bias)
    compare(lambda x: pool(conv(x)), lambda x: TF.max_pool2d(TF.conv2d(x, w.reshape(3, 2, 3, 3), b, padding=1), 2),
            r(2, 2, 6, 6), params=[(conv.weight, w), (conv.bias, b)])


@check("BatchNorm1d (train)")
def _():
    m = mt.BatchNorm1d(4)
    g, b = twin(m.gamma), twin(m.beta)
    compare(m, lambda x: TF.batch_norm(x, None, None, g, b, training=True, eps=1e-5), r(8, 4),
            params=[(m.gamma, g), (m.beta, b)])


def torch_attention(m, x, n_heads, n_kv_heads, theta=10000.0):
    """The same attention written with torch's own SDPA and HF-style rotate_half RoPE."""
    B, T, _ = x.shape
    D = m.head_dim
    wq, wk, wv, wo = (twin(p) for p in (m.q_proj.weight, m.k_proj.weight, m.v_proj.weight, m.o_proj.weight))
    q = (x @ wq).reshape(B, T, n_heads, D).transpose(1, 2)
    k = (x @ wk).reshape(B, T, n_kv_heads, D).transpose(1, 2)
    v = (x @ wv).reshape(B, T, n_kv_heads, D).transpose(1, 2)
    inv_freq = 1.0 / theta ** (torch.arange(0, D, 2, dtype=torch.float64) / D)
    ang = torch.outer(torch.arange(T, dtype=torch.float64), inv_freq)
    cos, sin = torch.cat([ang.cos(), ang.cos()], -1).float(), torch.cat([ang.sin(), ang.sin()], -1).float()

    def rot(t):
        t1, t2 = t[..., :D // 2], t[..., D // 2:]
        return t * cos + torch.cat([-t2, t1], -1) * sin

    out = TF.scaled_dot_product_attention(rot(q), rot(k), v, is_causal=True, enable_gqa=n_kv_heads != n_heads)
    return out.transpose(1, 2).reshape(B, T, n_heads * D) @ wo, [wq, wk, wv, wo]


@check("Attention (RoPE, GQA) vs torch SDPA")
def _():
    m = mt.Attention(16, n_heads=4, n_kv_heads=2)
    holder = {}

    def th(x):
        out, holder["w"] = torch_attention(m, x, 4, 2)
        return out

    x = r(2, 7, 16)
    compare(m, th, x)
    # weight grads: rerun to pair them up (compare() above checked input grads and output)
    xs, xt = mt.Tensor(x), torch.tensor(x)
    for p in m.parameters():
        p.zero_grad()
    m(xs).sum().backward()
    th(xt).sum().backward()
    for p, t in zip([m.q_proj.weight, m.k_proj.weight, m.v_proj.weight, m.o_proj.weight], holder["w"]):
        close("attn weight grad", p.grad, t.grad)


@check("Attention without the causal mask vs torch SDPA")
def _():
    from minitorch.attention import scaled_dot_product_attention
    compare(lambda q, k, v: scaled_dot_product_attention(q, k, v, causal=False),
            lambda q, k, v: TF.scaled_dot_product_attention(q, k, v), r(2, 4, 6, 8), r(2, 4, 9, 8), r(2, 4, 9, 8))


@check("KV-cache decode == full forward")
def _():
    lm = mt.TransformerLM(vocab_size=20, context_length=16, d_model=16, n_layers=2, n_heads=4, d_ff=24, n_kv_heads=2)
    idx = np.random.randint(0, 20, size=(2, 9))
    with mt.no_grad():
        full = lm(idx).data
        caches = [mt.KVCache() for _ in lm.blocks]
        steps = [lm(idx[:, :5], caches).data] + [lm(idx[:, t:t + 1], caches).data for t in range(5, 9)]
    close("cached logits", np.concatenate(steps, axis=1), full)


# losses and optimizers

@check("cross_entropy (B, T, V) logits")
def _():
    labels = np.random.randint(0, 7, size=(2, 3))
    compare(lambda x: mt.cross_entropy_loss(x, labels),
            lambda x: TF.cross_entropy(x.reshape(-1, 7), torch.tensor(labels).reshape(-1)), r(2, 3, 7))


@check("AdamW 5 steps")
def _():
    p0 = r(4, 3)
    p, tp = mt.Tensor(p0.copy(), requires_grad=True), torch.tensor(p0.copy(), requires_grad=True)
    opt = mt.AdamW([p], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    topt = torch.optim.AdamW([tp], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    for _ in range(5):
        g = r(4, 3)
        p.grad, tp.grad = g.copy(), torch.tensor(g.copy())
        opt.step()
        topt.step()
    close("AdamW params", p.data, tp)


@check("clip_grad_norm")
def _():
    gs = [r(3, 4), r(5)]
    ps = [mt.Tensor(np.zeros_like(g), requires_grad=True) for g in gs]
    tps = [torch.zeros(g.shape, requires_grad=True) for g in gs]
    for p, tp, g in zip(ps, tps, gs):
        p.grad, tp.grad = g.copy(), torch.tensor(g.copy())
    n, tn = mt.clip_grad_norm(ps, 1.0), torch.nn.utils.clip_grad_norm_(tps, 1.0)
    close("total norm", np.float32(n), tn)
    for p, tp in zip(ps, tps):
        close("clipped grad", p.grad, tp.grad)


@check("cosine_lr matches CS336 schedule shape")
def _():
    lrs = [mt.cosine_lr(t, 1.0, 0.1, 10, 100) for t in range(120)]
    assert lrs[0] == 0.0 and lrs[10] == 1.0 and abs(lrs[100] - 0.1) < 1e-12 and lrs[119] == 0.1
    assert all(a >= b for a, b in itertools.pairwise(lrs[10:101])), "cosine phase must not increase"


@check("sampling: greedy, top_k=1, tiny top_p all pick argmax")
def _():
    from minitorch.sampling import sample
    logits = r(4, 50)
    best = logits.argmax(-1)
    assert (sample(logits, temperature=0) == best).all()
    assert (sample(logits, top_k=1) == best).all()
    assert (sample(logits, top_p=1e-6) == best).all()


@check("sampling on the device: greedy picks argmax, and draws follow softmax(logits / T)")
def _():
    from minitorch.sampling import sample_on_device
    logits = r(4, 50)
    assert (sample_on_device(mt.Tensor(logits), temperature=0) == logits.argmax(-1)).all()
    p = np.array([0.1, 0.2, 0.3, 0.4])
    ids = sample_on_device(mt.Tensor(np.log(np.tile(p, (200_000, 1)) ** 2)), temperature=2.0)  # T=2 undoes the square
    freq = np.bincount(ids, minlength=4) / len(ids)
    assert np.all(np.abs(freq - p) < 5 * np.sqrt(p * (1 - p) / len(ids))), freq


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    failed = []
    for name, fn in CHECKS:
        try:
            fn()
            print(f"[ok]   {name}")
        except Exception as e:  # noqa: BLE001 - report every failing check, not just the first
            print(f"[FAIL] {name}: {repr(e)[:300]}")
            failed.append(name)
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} parity checks passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
