"""Lazy tensors: writing an op records it, reading a result runs it.

Every primitive's forward is wrapped to count its runs, then:
- `a @ b` and a whole function returning a result run nothing, yet know their shape and dtype;
- reading the result runs each op it needs exactly once, and a second read runs nothing;
- an op nobody reads never runs;
- replacing an input's .data after an op was written does not change that op's result;
- comparisons and where() stay lazy too, and backward on a lazy loss gives the eager gradients.

    uv run python tests/lazy.py
"""
import inspect
import sys

import numpy as np

from minitorch import mlops
from minitorch.tensor import Tensor, where

RAN = []
for name, fn in inspect.getmembers(mlops, inspect.isclass):
    if issubclass(fn, mlops.Function) and fn is not mlops.Function and "forward" in fn.__dict__:
        def counted(self, *args, _forward=fn.forward, _name=name, **kwargs):
            RAN.append(_name)
            return _forward(self, *args, **kwargs)
        fn.forward = counted

failures = 0


def check(ok, what):
    global failures
    failures += not ok
    print(f"[{'ok' if ok else 'FAIL'}]   {what}")


rng = np.random.default_rng(0)
A, B = rng.standard_normal((64, 32)), rng.standard_normal((32, 16))
a, b = Tensor(A), Tensor(B)

c = a @ b
check(RAN == [] and c.shape == (64, 16) and c.dtype == np.float64 and c._buf is None,
      f"a @ b runs nothing and still knows its shape {c.shape} and dtype {c.dtype}")


def layer(x, w):
    return (x @ w + 1.0).relu().sum(axis=0)


out = layer(a, b)
unused = (a @ a.T).exp()
check(RAN == [], "a function that builds and returns a result runs nothing")

value = out.numpy()
ran = list(RAN)
check(sorted(ran) == sorted(["Matmul", "Add", "Less", "Where", "Sum"]) and len(ran) == 5,
      f"reading the result runs each op it needs once: {ran}")
check(np.array_equal(value, np.where(A @ B + 1.0 > 0, A @ B + 1.0, 0.0).sum(axis=0)), "the result equals NumPy's, bit for bit")
out.numpy()
check(RAN == ran, "a second read runs nothing")
check("Exp" not in RAN and unused._buf is None, "an op nobody reads never runs")

w = Tensor(B.copy())
y = a @ w
w.data = np.zeros_like(B)  # as an optimizer step replaces a weight's array
check(np.array_equal(y.numpy(), A @ B), "replacing an input's .data after the op was written leaves its result as written")

x = Tensor(rng.standard_normal(10), requires_grad=True)
mask = x > 0
check(mask._buf is None and where(mask, x, 0.0)._buf is None, "comparisons and where() are recorded, not run")

RAN.clear()
loss = (where(x > 0, x, 0.0) * x).sum()
check(RAN == [], "a loss is recorded, not run")
loss.backward()
check(np.allclose(x.grad, np.where(x.data > 0, 2 * x.data, 0.0)), "backward on a lazy loss runs it and gives d/dx of relu(x) * x")

print(f"\n{'all lazy checks passed' if failures == 0 else f'{failures} lazy checks failed'}")
sys.exit(1 if failures else 0)
