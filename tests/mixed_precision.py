"""Mixed precision (minitorch.autocast): 16-bit multiplies, float32 weights and gradients, nearly the same numbers.

    uv run python tests/mixed_precision.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from minitorch import LossScaler, Tensor, TransformerLM, autocast, cross_entropy_loss


def main():
    np.random.seed(0)
    model = TransformerLM(vocab_size=50, context_length=32, d_model=32, n_layers=2, n_heads=4, d_ff=64)
    idx = np.random.randint(0, 50, (2, 16))
    runs = {}
    for dtype in (None, "float16"):
        for p in model.parameters():
            p.grad = None
        if dtype is None:
            loss = cross_entropy_loss(model(idx), (idx + 1) % 50)
        else:
            with autocast(dtype):
                loss = cross_entropy_loss(model(idx), (idx + 1) % 50)
        loss.backward()
        runs[dtype] = loss.item(), {n: p.grad for n, p in model.named_parameters()}
    (loss32, g32), (loss16, g16) = runs[None], runs["float16"]
    assert all(str(g.dtype) == "float32" for g in g16.values()) and all(str(p.data.dtype) == "float32" for p in model.parameters())
    rel = max(float(np.abs(g16[n] - g32[n]).max() / (np.abs(g32[n]).max() + 1e-12)) for n in g32)
    assert loss16 != loss32, "autocast changed nothing: the multiplies did not run in float16"
    assert abs(loss16 - loss32) / loss32 < 1e-2 and rel < 5e-2, (loss16, loss32, rel)
    print(f"[ok]   float16 multiplies: loss within {abs(loss16 - loss32) / loss32:.1e} of float32, gradients within {rel:.1e} "
          f"(relative to each one's largest), weights and gradients still float32")

    w = Tensor(np.ones(3, np.float32), requires_grad=True)
    scaler = LossScaler(scale=8.0)
    w.grad = np.array([8.0, np.inf, 16.0], np.float32)
    skipped = not scaler.unscale([w])
    assert skipped and scaler.scale == 4.0 and np.isinf(w.grad[1]), (skipped, scaler.scale, w.grad)  # skip, halve, leave grads
    w.grad = np.array([8.0, 4.0, 16.0], np.float32)
    assert scaler.unscale([w]) and np.allclose(w.grad, [2.0, 1.0, 4.0]), w.grad  # divided by the halved scale
    print("[ok]   loss scaler: an overflowed gradient skips the step and halves the scale; a finite one is divided back")


if __name__ == "__main__":
    main()
