"""A GPU backend against the NumPy reference.

    uv run python tests/backends.py --device metal
    uv run --no-dev python tests/backends.py --device cuda      # bqa3

1. Every check in test_grads.py runs twice, on cpu and on the device, from the
   same inputs; loss and every gradient must agree.
2. A TransformerLM trains for a few steps on both backends from identical
   weights and batches; the loss curves must agree.
3. The same model decodes token by token against a KV cache on both backends;
   the logits of every step must agree.
4. (metal) The model runs in bfloat16: every value in the graph stays bfloat16,
   and the logits stay close to float32.
5. (metal) Every attention kernel version against the float64 reference at the
   shapes the model checks never reach: head sizes 64 and 128, ragged T, T < S.
6. A 784-128-10 MLP trains on MNIST on both backends; the loss curves must
   agree within 1e-4. Skipped when scikit-learn is missing or MNIST cannot download.

The cpu side of part 1 runs in float64 (test_grads makes float64 inputs), so it
measures each backend's float32 error against near-exact values.
"""
import argparse
import json
import sys
import threading
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import test_grads
from minitorch import (
    Adam,
    AdamW,
    KVCache,
    Linear,
    Module,
    ReLU,
    Sequential,
    Tensor,
    TransformerLM,
    clip_grad_norm,
    cross_entropy_loss,
    no_grad,
)

RTOL = ATOL = 1e-4
RESULTS = {}  # what each check measured, written to bench/results/backends_<device>.json for the UI



def one_hot(labels, num_classes=10):
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def build_mlp():
    return Sequential(Linear(784, 128), ReLU(), Linear(128, 10))

def move(f, device):
    """Move every tensor and module the closure `f` reads onto `device`, in place."""
    for cell in f.__closure__ or ():
        v = cell.cell_contents
        if isinstance(v, Module):
            v.to(device)
        elif isinstance(v, Tensor):
            moved = v.to(device)
            v.data, v.device, v.grad = moved.data, moved.device, None


def loss_and_grads(f, inputs):
    for t in inputs:
        t.zero_grad()
    loss = f()
    loss.backward()
    return [loss.numpy()] + [Tensor(t.grad, device=t.device).numpy() for t in inputs]


def check_ops(device):
    failed = []

    def compare(f, inputs, **_):
        ref = loss_and_grads(f, inputs)
        move(f, device)
        for t in inputs:  # inputs reached through a Module's parameters() are not closure cells
            moved = t.to(device)
            t.data, t.device = moved.data, moved.device
        assert not any(isinstance(t.data, np.ndarray) for t in inputs), "an input stayed on the host"
        out = loss_and_grads(f, inputs)
        worst = max(float(np.max(np.abs(o - r) / (ATOL + RTOL * np.abs(r)))) for o, r in zip(out, ref))
        assert worst <= 1, f"off by {worst:.1f}x the tolerance"
        return worst

    test_grads.check_gradient = compare
    np.random.seed(0)
    for name, fn in test_grads.CHECKS:
        try:
            fn()
            print(f"[ok]   {name}")
        except Exception as e:  # noqa: BLE001 - report every failing check, not just the first
            where = traceback.extract_tb(e.__traceback__)[-1]
            print(f"[FAIL] {name}: {repr(e)[:200]} at {Path(where.filename).name}:{where.lineno}")
            failed.append(name)
    print(f"{len(test_grads.CHECKS) - len(failed)}/{len(test_grads.CHECKS)} ops agree with cpu on {device}\n")
    RESULTS["ops"] = {"agree": len(test_grads.CHECKS) - len(failed), "total": len(test_grads.CHECKS)}
    return failed


def check_training(device, steps=20):
    np.random.seed(0)
    config = {"vocab_size": 512, "context_length": 64, "d_model": 128, "n_layers": 2, "n_heads": 4, "d_ff": 344, "n_kv_heads": 2}
    models = {"cpu": TransformerLM(**config)}
    models[device] = TransformerLM(**config)
    models[device].load_state_dict(models["cpu"].state_dict())
    models[device].to(device)
    assert not isinstance(models[device].head.weight.data, np.ndarray)
    opts = {d: AdamW(m.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1) for d, m in models.items()}
    # a learnable rule (next token = this token + 1), so the curves being compared actually fall
    batches = [(x, (x + 1) % 512) for x in (np.random.randint(0, 512, (8, 64)) for _ in range(steps))]

    losses = {}
    for d, m in models.items():
        losses[d] = []
        for x, y in batches:
            loss = cross_entropy_loss(m(x), y)
            opts[d].zero_grad()
            loss.backward()
            clip_grad_norm(m.parameters(), 1.0)
            opts[d].step()
            losses[d].append(loss.item())
    gap = np.abs(np.array(losses[device]) - np.array(losses["cpu"]))
    print(f"TransformerLM, {steps} AdamW steps: loss {losses['cpu'][0]:.4f} -> {losses['cpu'][-1]:.4f}, "
          f"largest gap {gap.max():.2e} (step {gap.argmax() + 1})")
    RESULTS["training"] = {"steps": steps, "first_loss": losses["cpu"][0], "last_loss": losses["cpu"][-1], "max_gap": float(gap.max())}
    return [] if gap.max() <= 1e-3 else ["training"]


def check_decode(device, steps=4):
    np.random.seed(0)
    config = {"vocab_size": 64, "context_length": 32, "d_model": 64, "n_layers": 2, "n_heads": 4, "d_ff": 128, "n_kv_heads": 2}
    models = {"cpu": TransformerLM(**config), device: TransformerLM(**config)}
    models[device].load_state_dict(models["cpu"].state_dict())
    models[device].to(device)
    prompt, new = np.random.randint(0, 64, (2, 5)), np.random.randint(0, 64, (steps, 2, 1))
    logits = {}
    with no_grad():
        for d, m in models.items():
            caches = [KVCache() for _ in m.blocks]
            logits[d] = [m(prompt, caches).numpy()] + [m(tok, caches).numpy() for tok in new]
    gap = max(float(np.abs(a - b).max()) for a, b in zip(logits[device], logits["cpu"]))
    print(f"KV-cache decode, prefill + {steps} steps: largest logit gap {gap:.2e}")
    RESULTS["decode"] = {"steps": steps, "max_logit_gap": gap}
    return [] if gap <= ATOL * 10 else ["decode"]


def check_bfloat16(device):
    np.random.seed(0)
    config = {"vocab_size": 64, "context_length": 32, "d_model": 64, "n_layers": 2, "n_heads": 4, "d_ff": 128, "n_kv_heads": 2}
    ref = TransformerLM(**config)
    model = TransformerLM(**config)
    model.load_state_dict(ref.state_dict())
    ref.to(device)
    model.to(device, "bfloat16")
    idx = np.random.randint(0, 64, (2, 16))
    logits = model(idx)

    # walk the graph: any float32 node means something promoted silently
    seen, stack, promoted = set(), [logits], []
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        if "bfloat16" not in str(t.dtype):
            promoted.append((t._op or "leaf", str(t.dtype)))
        stack.extend(t._ctx.parents if t._ctx else [])
    gap = float(np.abs(logits.numpy() - ref(idx).numpy()).max() / np.abs(ref(idx).numpy()).max())

    # an explicit cast sends the gradient back in the source dtype (float32 master weights)
    w = Tensor(np.ones((4, 4), dtype=np.float32), requires_grad=True, device=device)
    (w.cast("bfloat16") * 3.0).sum().backward()
    grad_ok = "float32" in str(w.grad.dtype) and np.allclose(Tensor(w.grad, device=device).numpy(), 3.0)

    print(f"bfloat16: {len(seen)} graph nodes, {len(promoted)} not bfloat16 {promoted[:3]}, "
          f"logits within {gap:.1e} of float32 (relative to the largest), cast grad float32: {grad_ok}")
    RESULTS["bfloat16"] = {"nodes": len(seen), "promoted": len(promoted), "relative_logit_gap": gap, "cast_grad_float32": bool(grad_ok)}
    return [] if not promoted and gap < 0.05 and grad_ok else ["bfloat16"]


def check_kernels():
    import mlx.core as mx

    from minitorch.attention import scaled_dot_product_attention
    from minitorch.backend import metal_attention

    failed = []
    for B, H, Hkv, T, S, D in [(2, 4, 4, 64, 64, 32), (1, 4, 2, 37, 37, 64), (2, 8, 2, 1, 50, 128),
                               (1, 2, 1, 5, 70, 64), (1, 16, 8, 100, 300, 128)]:
        rng = np.random.default_rng(0)
        q, k, v = (rng.standard_normal(s) for s in [(B, H, T, D), (B, Hkv, S, D), (B, Hkv, S, D)])
        # the reference: mlops.Attention's own math on cpu in float64 (requires_grad so it keeps its log-sum-exp)
        ref = scaled_dot_product_attention(*(Tensor(a, requires_grad=True) for a in (q, k, v)))
        ref_out, ref_lse = ref.data, ref._ctx.saved[4]
        for version in ["v1", "v2", "v3"]:
            out, lse = metal_attention.attention(*(mx.array(a.astype(np.float32)) for a in (q, k, v)), version)
            err = max(float(np.abs(np.array(out) - ref_out).max()), float(np.abs(np.array(lse) - ref_lse).max()))
            if err > ATOL:
                failed.append(f"{version} B{B} H{H}/{Hkv} T{T} S{S} D{D}: {err:.1e}")
    print(f"attention kernels: {15 - len(failed)}/15 version x shape runs within {ATOL} of float64 {failed or ''}")
    RESULTS["kernels"] = {"within_tolerance": 15 - len(failed), "total": 15, "tolerance": ATOL}
    return ["kernels"] if failed else []


def check_triton_kernels():
    """The Triton FlashAttention-2 forward and backward against mlops.Attention's own math in float64."""
    import cupy as cp

    from minitorch.attention import scaled_dot_product_attention
    from minitorch.backend import cuda_attention

    failed, shapes = [], [(2, 4, 4, 64, 64, 32), (1, 4, 2, 37, 37, 64), (2, 8, 2, 1, 50, 128),
                          (1, 2, 1, 5, 70, 64), (1, 16, 8, 100, 300, 128)]
    for B, H, Hkv, T, S, D in shapes:
        rng = np.random.default_rng(0)
        q, k, v, dout = (rng.standard_normal(s) for s in [(B, H, T, D), (B, Hkv, S, D), (B, Hkv, S, D), (B, H, T, D)])
        leaves = [Tensor(a, requires_grad=True) for a in (q, k, v)]
        ref = scaled_dot_product_attention(*leaves)
        (ref * Tensor(dout)).sum().backward()
        out, lse = cuda_attention.attention(*(cp.asarray(a, cp.float32) for a in (q, k, v)))
        grads = cuda_attention.attention_backward(*(cp.asarray(a, cp.float32) for a in (q, k, v)), out, lse, cp.asarray(dout, cp.float32))
        err = max([float(np.abs(cp.asnumpy(out) - ref.data).max()), float(np.abs(cp.asnumpy(lse) - ref._ctx.saved[4]).max())]
                  + [float(np.abs(cp.asnumpy(g) - leaf.grad).max()) for g, leaf in zip(grads, leaves)])
        if err > ATOL:
            failed.append(f"B{B} H{H}/{Hkv} T{T} S{S} D{D}: {err:.1e}")
    print(f"Triton attention: forward and backward within {ATOL} of float64 on {len(shapes) - len(failed)}/{len(shapes)} shapes {failed or ''}")
    RESULTS["kernels"] = {"within_tolerance": len(shapes) - len(failed), "total": len(shapes), "tolerance": ATOL}
    return ["kernels"] if failed else []


def check_thread(device):
    """A model moved on one thread runs on another, as in the serving engine (MLX ties lazy work to a thread)."""
    np.random.seed(0)
    model = TransformerLM(vocab_size=64, context_length=32, d_model=64, n_layers=2, n_heads=4, d_ff=128)
    model.to(device, "bfloat16" if device == "metal" else None)
    out = {}

    def run():
        with no_grad():
            out["logits"] = model(np.arange(8)[None]).numpy()
    t = threading.Thread(target=run)
    t.start()
    t.join(timeout=60)
    ok = "logits" in out and np.isfinite(out["logits"]).all()
    print(f"another thread runs a model moved on this one: {'yes' if ok else 'NO'}")
    RESULTS["thread"] = {"ok": bool(ok)}
    return [] if ok else ["thread"]


def check_mnist(device, steps=100):
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    except (ImportError, OSError) as e:
        print(f"MNIST: skipped ({type(e).__name__}: {e})")
        return []

    X, y = mnist.data[:10_000].astype(np.float32) / 255.0, one_hot(mnist.target[:10_000].astype(np.int64))
    np.random.seed(0)
    models = {"cpu": build_mlp(), device: build_mlp()}
    models[device].load_state_dict(models["cpu"].state_dict())
    models[device].to(device)
    batches = [np.random.choice(len(X), 100, replace=False) for _ in range(steps)]
    losses = {}
    for d, m in models.items():
        opt = Adam(m.parameters(), lr=1e-3)
        losses[d] = []
        for b in batches:
            loss = cross_entropy_loss(m(Tensor(X[b], device=d)), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses[d].append(loss.item())
    gap = np.abs(np.array(losses[device]) - np.array(losses["cpu"]))
    print(f"MNIST MLP, {steps} Adam steps: loss {losses['cpu'][0]:.4f} -> {losses['cpu'][-1]:.4f}, largest gap {gap.max():.2e}")
    RESULTS["mnist"] = {"steps": steps, "first_loss": losses["cpu"][0], "last_loss": losses["cpu"][-1], "max_gap": float(gap.max())}
    return [] if gap.max() <= 1e-4 else ["mnist"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", required=True, choices=["metal", "cuda"])
    device = p.parse_args().device
    failed = check_ops(device) + check_training(device) + check_decode(device) + check_thread(device) + check_mnist(device)
    if device == "metal":  # NumPy, and so CuPy, has no bfloat16; the kernels are Metal-only
        failed += check_bfloat16(device) + check_kernels()
    else:
        failed += check_triton_kernels()
    out = Path(__file__).resolve().parent.parent / "bench" / "results" / f"backends_{device}.json"
    out.write_text(json.dumps({"device": device, "failed": failed, **RESULTS}, indent=1))
    if failed:
        print(f"FAILED: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
