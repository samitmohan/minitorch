"""Tensor: an array plus a record of the op that produced it.

Every op goes through `_apply`, which records one primitive from mlops.py without
running it: `a @ b` returns a lazy tensor that knows its shape and dtype and holds
no numbers. The numbers are computed when something reads them: `.data`,
`.numpy()`, `.item()`, printing, `.backward()`, or `.realize()`. Realizing runs
every recorded op the tensor depends on, once each; an op nobody reads never runs.

When gradients are on, the record also goes in `out._ctx`, and `.backward()` walks
those records in reverse topological order, calling each primitive's backward.
Everything else in this file composes primitives, so it never defines a gradient
of its own.
"""
import math
import threading

import numpy as np

from . import backend, mlops

# global flag for no_grad context
_local = threading.local()  # per thread: the serving engine's no_grad() must not freeze tensors other threads build


def grad_enabled():
    """Whether operations in this thread record a graph for backward (False inside no_grad)."""
    return getattr(_local, "grad_enabled", True)
_profiler = None  # a minitorch.profile.profile while one is active
_autocast = None  # a dtype name while autocast is active


class no_grad:
    """Context manager to disable gradient tracking during inference."""
    def __enter__(self):
        self._prev = grad_enabled()
        _local.grad_enabled = False
        return self

    def __exit__(self, *args):
        _local.grad_enabled = self._prev


class autocast:
    """Mixed precision. Inside, every Linear multiplies in `dtype` ("bfloat16" or "float16") and hands back
    float32, while the weights and all other ops stay float32. The casts are differentiable, so the
    gradients land on the float32 weights. float16's narrow range needs the loss scaled (train_gpt.py --amp)."""
    def __init__(self, dtype):
        self.dtype = dtype

    def __enter__(self):
        global _autocast
        self._prev, _autocast = _autocast, self.dtype
        return self

    def __exit__(self, *args):
        global _autocast
        _autocast = self._prev


def _to_array(data, device):
    """Data as a float array on `device`. Float arrays keep their dtype; ints, bools, and Python numbers become float32."""
    xp = backend.get(device)
    if not hasattr(data, "dtype"):
        data = np.asarray(data, dtype=np.float32)
    data = xp.asarray(data)
    return data if xp.is_float(data.dtype) else data.astype(xp.float32)


def _shape(args):
    # reshape(2, 3) and reshape((2, 3)) both work
    return tuple(args[0]) if len(args) == 1 and isinstance(args[0], (tuple, list)) else tuple(args)


def _apply(fn, *parents, **kwargs):
    """Record primitive `fn` on `parents` without running it, and for backward too if any parent needs a gradient.

    Each input is taken as it is now: an input that already holds numbers is kept as that array, so replacing its
    `.data` later (an optimizer step) cannot change this result; an input still lazy is computed first when this one is.
    """
    device = parents[0].device
    assert all(p.device == device for p in parents), f"tensors on different devices: {[p.device for p in parents]}"
    xp = backend.get(device)
    ctx = fn(xp, *parents)
    dtype = ctx.out_dtype(*[p.dtype for p in parents], **kwargs)
    out = Tensor._recorded((ctx, kwargs, [p if p._buf is None else p._buf for p in parents]),
                           tuple(ctx.out_shape(*[tuple(p.shape) for p in parents], **kwargs)),
                           dtype if xp.is_float(dtype) else xp.float32, device)  # as Tensor() turns ints and bools to float32
    if grad_enabled() and fn.differentiable and any(ctx.needs_grad):
        out.requires_grad = True
        out._ctx = ctx
    return out


def _broadcast(tensors, shape=()):
    """Expand every tensor that needs a gradient to the common broadcast shape.

    Constants stay as they are: the backend broadcasts them for free, and
    nothing needs their gradient. Expand's backward is what sums gradients
    back down to the smaller shape.
    """
    shape = np.broadcast_shapes(shape, *(t.shape for t in tensors))
    return [t.expand(shape) if grad_enabled() and t.requires_grad and t.shape != shape else t
            for t in tensors]


class Tensor:
    __array_ufunc__ = None  # np.float32(2) * t calls t.__rmul__ instead of making an object array that drops the graph
    """An n-dimensional array that records operations for reverse-mode autodiff.

    Wraps an array of the backend named by `device` ("cpu", "metal", "cuda"). When `requires_grad=True`, every operation builds a node
    in the computation graph; calling `.backward()` on a scalar walks that graph
    in reverse and fills in `.grad` on the leaves.

    ```python
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x ** 2).sum()
    y.backward()
    x.grad  # array([2., 4., 6.])
    ```
    """
    def __init__(self, data, *, requires_grad=False, device="cpu"):
        self.data = _to_array(data, device)
        self.device = device
        self.requires_grad = requires_grad and grad_enabled()
        self.grad = None
        self._ctx = None

    @classmethod
    def _recorded(cls, lazy, shape, dtype, device):
        """A tensor an op has recorded but not run: `lazy` is (op, its keyword arguments, its inputs)."""
        out = cls.__new__(cls)
        out._buf, out._lazy, out._shape, out._dtype = None, lazy, shape, dtype
        out.device, out.requires_grad, out.grad, out._ctx = device, False, None, None
        return out

    # the numbers, computed on first read

    @property
    def data(self):
        """The backend array. Reading it runs the ops this tensor still owes (see realize)."""
        if self._buf is None:
            self.realize()
        return self._buf

    @data.setter
    def data(self, array):
        self._buf, self._lazy = array, None

    def realize(self):
        """Compute this tensor: run every recorded op it depends on, inputs first, once each. Returns self.

        On Metal this hands the work to MLX, which is lazy too: its arrays are computed when read or at `mx.eval`.
        """
        order, seen, stack = [], set(), [(self, False)]
        while stack:  # iterative post-order, so a deep graph cannot hit the recursion limit
            node, inputs_done = stack.pop()
            if inputs_done:
                order.append(node)
            elif node._buf is None and node not in seen:
                seen.add(node)
                stack.append((node, True))
                stack.extend((x, False) for x in node._lazy[2] if isinstance(x, Tensor))
        for node in order:
            ctx, kwargs, inputs = node._lazy
            args = [x._buf if isinstance(x, Tensor) else x for x in inputs]
            if _profiler is None:
                out = ctx.forward(*args, **kwargs)
            else:
                out = _profiler.time(type(ctx).__name__, node.device, ctx.forward, *args, **kwargs)
            if out.dtype != node._dtype:
                out = _to_array(out, node.device)  # ints and bools become float32, as Tensor() has always made them
            # the shape and dtype every later op was built on must be the ones that came out
            assert tuple(out.shape) == node._shape and out.dtype == node._dtype, \
                f"{type(ctx).__name__} recorded {node._shape} {node._dtype} but computed {tuple(out.shape)} {out.dtype}"
            node.data = out  # drops the record, so inputs nothing else holds can be freed
        return self

    # properties

    @property
    def shape(self):
        return self._shape if self._buf is None else self._buf.shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self._dtype if self._buf is None else self._buf.dtype

    @property
    def T(self):
        return self.transpose()

    # graph bookkeeping, read by the graph explorer (serve/api.py)

    @property
    def _op(self):
        return type(self._ctx).__name__.lower() if self._ctx else ''

    def _const(self, x):
        return x if isinstance(x, Tensor) else Tensor(backend.get(self.device).asarray(x, dtype=self.dtype), device=self.device)

    # arithmetic: sub, div, and neg are compositions of add, mul, and recip

    def __add__(self, other):
        return _apply(mlops.Add, *_broadcast([self, self._const(other)]))

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        return _apply(mlops.Mul, *_broadcast([self, self._const(other)]))

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + -self._const(other)

    def __rsub__(self, other):
        return -self + other

    def __truediv__(self, other):
        return self * self._const(other).recip()

    def __rtruediv__(self, other):
        return self.recip() * other

    def __pow__(self, exponent):
        if isinstance(exponent, Tensor):
            # a^b = e^(b ln a), so this needs a positive base
            return (exponent * self.log()).exp()
        return _apply(mlops.Pow, self, c=exponent)

    def __matmul__(self, other):
        return _apply(mlops.Matmul, self, other)

    # comparisons return constant 0/1 masks with no gradient

    def __lt__(self, other):
        return _apply(mlops.Less, self, self._const(other))

    def __gt__(self, other):
        return _apply(mlops.Less, self._const(other), self)

    # reductions

    def sum(self, axis=None, keepdims=False):
        """Sum over `axis` (or all elements). Differentiable."""
        return _apply(mlops.Sum, self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        """Mean over `axis` (or all elements). Differentiable."""
        out = self.sum(axis=axis, keepdims=keepdims)
        # out.size / self.size is 1 / (number of elements summed into each output); from shapes, so nothing runs yet
        return out * (math.prod(out.shape) / math.prod(self.shape))

    def max(self, axis=None, keepdims=False):
        return _apply(mlops.Max, self, axis=axis, keepdims=keepdims)

    def min(self, axis=None, keepdims=False):
        return -((-self).max(axis=axis, keepdims=keepdims))

    # shape ops

    def reshape(self, *shape):
        """Return a view with a new shape. Differentiable."""
        return _apply(mlops.Reshape, self, shape=_shape(shape))

    def permute(self, *order):
        return _apply(mlops.Permute, self, order=tuple(d % self.ndim for d in _shape(order)))

    def transpose(self, dim0=-2, dim1=-1):
        """Swap two axes (last two by default). Differentiable."""
        order = list(range(self.ndim))
        order[dim0], order[dim1] = order[dim1], order[dim0]
        return self.permute(order)

    def expand(self, *shape):
        """Broadcast to `shape`, adding leading dims if needed."""
        shape = _shape(shape)
        x = self.reshape((1,) * (len(shape) - self.ndim) + self.shape) if self.ndim < len(shape) else self
        return _apply(mlops.Expand, x, shape=shape)

    def squeeze(self, axis=None):
        axes = range(self.ndim) if axis is None else [a % self.ndim for a in np.atleast_1d(axis)]
        return self.reshape(tuple(d for i, d in enumerate(self.shape) if not (i in axes and d == 1)))

    def unsqueeze(self, axis):
        axis = axis if axis >= 0 else self.ndim + axis + 1
        return self.reshape(self.shape[:axis] + (1,) + self.shape[axis:])

    def __getitem__(self, key):
        # index arrays usually come from NumPy; they must live where the data lives
        xp = backend.get(self.device)

        def on_device(k):
            return xp.asarray(k) if isinstance(k, np.ndarray) else k
        key = tuple(map(on_device, key)) if isinstance(key, tuple) else on_device(key)
        return _apply(mlops.Index, self, key=key)

    def im2col(self, kh, kw, stride=1, padding=0):
        """(N, C, H, W) -> (N*OH*OW, C*kh*kw) patch rows, the input to a conv-as-matmul."""
        return _apply(mlops.Im2col, self, kh=kh, kw=kw, stride=stride, padding=padding)

    def cast(self, dtype):
        """Convert to `dtype` ("float32", "bfloat16", ...). Differentiable: the gradient comes back in the old dtype."""
        return _apply(mlops.Cast, self, dtype=backend.get(self.device).dtype(dtype))

    # elementwise ops

    def exp(self):
        """Elementwise `e ** x`. Differentiable."""
        return _apply(mlops.Exp, self)

    def log(self):
        return _apply(mlops.Log, self)

    def recip(self):
        return _apply(mlops.Recip, self)

    def sigmoid(self):
        return _apply(mlops.Sigmoid, self)

    def relu(self):
        return where(self > 0, self, 0.0)

    def tanh(self):
        # tanh(x) = 2 * sigmoid(2x) - 1
        return 2.0 * (2.0 * self).sigmoid() - 1.0

    def abs(self):
        return self * Tensor(backend.get(self.device).sign(self.data), device=self.device)

    def clamp(self, min_val=None, max_val=None):
        out = self
        if min_val is not None:
            out = where(out < min_val, min_val, out)
        if max_val is not None:
            out = where(out > max_val, max_val, out)
        return out

    # autograd

    def backward(self, keep_grads=False):
        """Run reverse-mode autodiff from this scalar, filling `.grad` on every leaf.

        keep_grads=True also leaves `.grad` on intermediate tensors (for inspecting
        the graph); call it once per graph, since a second pass would add to them.
        """
        assert self.data.size == 1, "backward() only works on scalar tensors - call .sum() or .mean() first"

        # iterative post-order DFS so deep graphs don't blow the recursion limit
        topo = []
        visited = set()
        stack = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            if node._ctx:
                for parent in node._ctx.parents:
                    if parent.requires_grad and parent not in visited:
                        stack.append((parent, False))

        self.grad = backend.get(self.device).ones_like(self.data)
        for node in reversed(topo):
            if node._ctx is None or node.grad is None:
                continue
            if _profiler is None:
                grads = node._ctx.backward(node.grad)
            else:
                grads = _profiler.time(type(node._ctx).__name__ + " backward", node.device, node._ctx.backward, node.grad)
            # An intermediate grad is only needed until it reaches the parents.
            # Freeing it also means a second backward() starts clean.
            if not keep_grads:
                node.grad = None
            for parent, grad in zip(node._ctx.parents, grads):
                if grad is not None and parent.requires_grad:
                    # never += : one grad array can be shared by two parents (see Add)
                    parent.grad = grad if parent.grad is None else parent.grad + grad

    def zero_grad(self):
        self.grad = None

    def detach(self):
        """Return a tensor sharing this data but cut out of the graph."""
        return Tensor(self.data, device=self.device)


    def to(self, device=None, dtype=None):
        """This tensor outside the graph, on `device` and in `dtype` ("bfloat16", ...); None keeps the current one.

        Moving or casting copies the data; with neither, the new tensor shares it.
        """
        device = device or self.device
        data = self.data if device == self.device else backend.get(device).asarray(self.numpy())
        if dtype is not None:
            data = data.astype(backend.get(device).dtype(dtype))
        return Tensor(data, device=device, requires_grad=self.requires_grad)

    def numpy(self):
        """The data as a NumPy array on the host (a copy)."""
        return backend.get(self.device).to_numpy(self.data)

    def item(self):
        return self.data.item()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __len__(self):
        return self.shape[0]

    def __float__(self):
        return float(self.data.item())


# free functions

def where(cond, x, y):
    """Elementwise `x if cond else y`. `cond` is a mask and gets no gradient."""
    ref = x if isinstance(x, Tensor) else y
    cond = cond if isinstance(cond, Tensor) else Tensor(backend.get(ref.device).asarray(cond), device=ref.device)
    x, y = _broadcast([ref._const(x), ref._const(y)], cond.shape)
    return _apply(mlops.Where, x, y, cond)


def cat(tensors, axis=0):
    """Concatenate tensors along an existing axis. Differentiable."""
    return _apply(mlops.Cat, *tensors, axis=axis)


def stack(tensors, axis=0):
    """Stack tensors along a new axis. Differentiable."""
    return cat([t.unsqueeze(axis) for t in tensors], axis=axis)
