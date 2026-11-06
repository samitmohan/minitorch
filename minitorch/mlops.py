"""Every gradient in minitorch, in one file.

Each Function is one primitive op. `forward` computes the result from raw backend
arrays and saves what the gradient needs. `backward` takes the gradient of the
output and returns one gradient per input (None for inputs that don't need one).

Everything else in the framework (sub, div, relu, softmax, layernorm, attention,
conv) is a composition of these, so its gradient comes for free.

`self.xp` is the array backend (see backend/). Python operators work on every
backend's arrays; anything else goes through `xp`.

Tensors are lazy (tensor.py): an op is recorded when it is written and runs when its
result is needed. `out_shape` and `out_dtype` tell a recorded result its shape and
dtype from its inputs' before anything runs; tensor.py checks them against the real
result when it does run.
"""
import math

import numpy as np


class Function:
    differentiable = True  # False: the result never carries a gradient (comparisons)

    def __init__(self, xp, *parents):
        self.xp = xp
        self.parents = parents
        self.needs_grad = [p.requires_grad for p in parents]

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def out_shape(self, *shapes, **kwargs):
        """The result's shape from the inputs' shapes. Elementwise ops broadcast; the others override this."""
        return np.broadcast_shapes(*shapes)

    def out_dtype(self, *dtypes, **kwargs):
        """The result's dtype: the inputs' promoted, as the backend promotes them (a 0-d probe asks it)."""
        out = dtypes[0]
        for d in dtypes[1:]:
            if d != out:
                out = (self.xp.asarray(0, dtype=out) + self.xp.asarray(0, dtype=d)).dtype
        return out


def _reduced(shape, axis, keepdims):
    """The shape a reduction over `axis` (None: every axis) leaves."""
    axes = range(len(shape)) if axis is None else [a % len(shape) for a in (axis if isinstance(axis, (tuple, list)) else (axis,))]
    return tuple(1 if i in axes else d for i, d in enumerate(shape) if keepdims or i not in axes)


def unbroadcast(grad, shape):
    """Sum `grad` down to `shape`. Broadcasting copies a value, so its backward adds the copies up."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, size in enumerate(shape):
        if size == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


# unary

class Recip(Function):
    def forward(self, x):
        self.out = 1.0 / x
        return self.out

    def backward(self, grad):
        # d(1/x)/dx = -1/x^2
        return (-grad * self.out * self.out,)


class Exp(Function):
    def forward(self, x):
        self.out = self.xp.exp(x)
        return self.out

    def backward(self, grad):
        return (grad * self.out,)


class Log(Function):
    def forward(self, x):
        self.x = x
        return self.xp.log(x)

    def backward(self, grad):
        return (grad / self.x,)


class Sigmoid(Function):
    # A primitive, not exp/recip composed: through the composition, a very
    # negative x gives exp(-x) = inf and the backward computes 0 * inf = nan.
    def forward(self, x):
        self.out = self.xp.sigmoid(x)
        return self.out

    def backward(self, grad):
        return (grad * self.out * (1.0 - self.out),)


class Pow(Function):
    """x ** c for a constant exponent c."""
    def forward(self, x, c):
        self.x, self.c = x, c
        return x ** c

    def backward(self, grad):
        return (grad * self.c * self.x ** (self.c - 1),)


class Cast(Function):
    def forward(self, x, dtype):
        self.dtype = x.dtype
        return x.astype(dtype)

    def backward(self, grad):
        return (grad.astype(self.dtype),)

    def out_dtype(self, x, dtype):
        return dtype


# binary: inputs that need a gradient arrive already expanded to the output shape

class Add(Function):
    def forward(self, x, y):
        return x + y

    def backward(self, grad):
        return grad, grad


class Mul(Function):
    def forward(self, x, y):
        self.x, self.y = x, y
        return x * y

    def backward(self, grad):
        return (grad * self.y if self.needs_grad[0] else None,
                grad * self.x if self.needs_grad[1] else None)


class Where(Function):
    """x where `cond` is true, else y. `cond`, the third input, is a mask and gets no gradient."""
    def forward(self, x, y, cond):
        self.cond = cond
        return self.xp.where(cond, x, y)

    def backward(self, grad):
        zero = self.xp.zeros_like(grad)
        return self.xp.where(self.cond, grad, zero), self.xp.where(self.cond, zero, grad)

    def out_dtype(self, x, y, cond):
        return super().out_dtype(x, y)


class Less(Function):
    """x < y as a float32 mask of 0s and 1s, the type comparisons have always returned. It has no gradient."""
    differentiable = False

    def forward(self, x, y):
        return (x < y).astype(self.xp.float32)

    def out_dtype(self, x, y):
        return self.xp.float32


# reduce

class Sum(Function):
    def forward(self, x, axis, keepdims):
        self.shape, self.axis, self.keepdims = x.shape, axis, keepdims
        return x.sum(axis=axis, keepdims=keepdims)

    def out_shape(self, x, axis, keepdims):
        return _reduced(x, axis, keepdims)

    def backward(self, grad):
        # every input element contributed once, so each one gets the output's grad
        if self.axis is not None and not self.keepdims:
            grad = self.xp.expand_dims(grad, self.axis)
        return (self.xp.broadcast_to(grad, self.shape),)


class Max(Function):
    def forward(self, x, axis, keepdims):
        self.x, self.axis, self.keepdims = x, axis, keepdims
        self.out = x.max(axis=axis, keepdims=True)
        return self.out if keepdims else x.max(axis=axis)

    def out_shape(self, x, axis, keepdims):
        return _reduced(x, axis, keepdims)

    def backward(self, grad):
        # only the max element(s) affect the output; ties split the gradient evenly
        mask = (self.x == self.out).astype(self.x.dtype)
        mask = mask / mask.sum(axis=self.axis, keepdims=True)
        if self.axis is not None and not self.keepdims:
            grad = self.xp.expand_dims(grad, self.axis)
        return (mask * grad,)


# movement

class Reshape(Function):
    def forward(self, x, shape):
        self.shape = x.shape
        return x.reshape(shape)

    def backward(self, grad):
        return (grad.reshape(self.shape),)

    def out_shape(self, x, shape):
        size = math.prod(x)
        known = math.prod(d for d in shape if d != -1)
        out = tuple(size // known if d == -1 else d for d in shape)
        assert math.prod(out) == size, f"cannot reshape {x} into {shape}"
        return out


class Permute(Function):
    def forward(self, x, order):
        self.order = order
        return x.transpose(order)

    def backward(self, grad):
        inverse = [self.order.index(i) for i in range(len(self.order))]
        return (grad.transpose(inverse),)

    def out_shape(self, x, order):
        return tuple(x[i] for i in order)


class Expand(Function):
    """Broadcast x to a larger shape (x already has the target rank)."""
    def forward(self, x, shape):
        self.shape = x.shape
        return self.xp.broadcast_to(x, shape)

    def backward(self, grad):
        return (unbroadcast(grad, self.shape),)

    def out_shape(self, x, shape):
        return tuple(shape)


class Index(Function):
    """x[key] for any NumPy-style key: slices, ints, or integer arrays (gather)."""
    def forward(self, x, key):
        self.shape, self.key = x.shape, key
        return x[key]

    def backward(self, grad):
        # scatter the gradient back; an index picked twice gets both gradients
        return (self.xp.scatter_add(self.shape, self.key, grad),)

    def out_shape(self, x, key):
        """Index a stand-in of shape x whose every element shares one byte. An index array's values do not change the
        result's shape, so it becomes zeros, except a boolean mask, whose count of True does."""
        def stand_in(k):
            if not hasattr(k, "dtype"):
                return k  # an int, a slice, None, or Ellipsis
            if "bool" in str(k.dtype):
                return self.xp.to_numpy(k)
            return np.zeros(tuple(k.shape), dtype=np.intp)
        key = tuple(map(stand_in, key)) if isinstance(key, tuple) else stand_in(key)
        # shortcut: an integer-array index (a gather) allocates one byte per result element here; derive the shape from
        # NumPy's indexing rules instead if a large gather ever shows up in a profile
        return np.lib.stride_tricks.as_strided(np.zeros(1, np.uint8), x, (0,) * len(x))[key].shape


class Cat(Function):
    def forward(self, *xs, axis):
        self.axis = axis
        self.bounds = []
        end = 0
        for x in xs:
            self.bounds.append((end, end + x.shape[axis]))
            end += x.shape[axis]
        return self.xp.concatenate(xs, axis=axis)

    def backward(self, grad):
        index = [slice(None)] * grad.ndim
        grads = []
        for start, end in self.bounds:
            index[self.axis] = slice(start, end)
            grads.append(grad[tuple(index)])
        return tuple(grads)

    def out_shape(self, *shapes, axis):
        axis %= len(shapes[0])
        return shapes[0][:axis] + (sum(s[axis] for s in shapes),) + shapes[0][axis + 1:]


class Im2col(Function):
    """Unroll every (kh, kw) patch of an (N, C, H, W) image into one row, so conv becomes a matmul."""
    def forward(self, x, kh, kw, stride, padding):
        self.args = (x.shape, kh, kw, stride, padding)
        return self.xp.im2col(x, kh, kw, stride, padding)

    def backward(self, grad):
        # col2im adds each row back onto the pixels its patch came from
        return (self.xp.col2im(grad, *self.args),)

    def out_shape(self, x, kh, kw, stride, padding):
        n, c, h, w = x
        oh, ow = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
        return (n * oh * ow, c * kh * kw)


# matmul stays a primitive: decomposing it into broadcast-multiply-sum would throw away BLAS

class Matmul(Function):
    def forward(self, x, y):
        self.x, self.y = x, y
        return x @ y

    def backward(self, grad):
        # swapaxes(-1, -2) transposes the matrix dims and keeps batch dims;
        # unbroadcast sums over batch dims that were broadcast in the forward
        return (unbroadcast(grad @ self.y.swapaxes(-1, -2), self.x.shape) if self.needs_grad[0] else None,
                unbroadcast(self.x.swapaxes(-1, -2) @ grad, self.y.shape) if self.needs_grad[1] else None)

    def out_shape(self, x, y):
        """NumPy's matmul rule: (..., n, k) @ (..., k, m) -> (broadcast ..., n, m); a 1-d operand loses its matrix dim."""
        x1, y1 = len(x) == 1, len(y) == 1
        x, y = (1,) + x if x1 else x, y + (1,) if y1 else y
        assert x[-1] == y[-2], f"cannot multiply {x} by {y}"
        out = np.broadcast_shapes(x[:-2], y[:-2]) + (x[-2], y[-1])
        out = out[:-2] + out[-1:] if x1 else out
        return out[:-1] if y1 else out


class Attention(Function):
    """Causal softmax(q k^T / sqrt(D)) v. q is (B, H, T, D); k and v are (B, Hkv, S, D), each kv head serving H/Hkv query heads.

    A primitive, though it could be composed, so a backend can run it as one fused
    kernel that never writes the (T, S) score matrix to memory (metal_attention.py).
    After a fused forward, the backward rebuilds the probabilities from q, k and the
    saved log-sum-exp; the reference forward already has them, so it keeps them.
    """
    def scores(self, q, k):
        xp = self.xp
        T, S = q.shape[2], k.shape[2]
        k = xp.repeat(k, q.shape[1] // k.shape[1], axis=1)
        s = q @ k.swapaxes(-1, -2) * q.shape[-1] ** -0.5
        if self.causal:  # query i sits at absolute position S - T + i and may see keys 0 .. S - T + i
            s = s + xp.triu(xp.full((T, S), -xp.inf, dtype=q.dtype), k=S - T + 1)
        if self.key_bias is not None:
            s = s + xp.asarray(self.key_bias, dtype=q.dtype)[:, None, None, :]
        return s

    def forward(self, q, k, v, key_bias=None, causal=True):
        """key_bias: None, or a (B, S) array added to every query's score for each key (a mask, for left padding).
        causal=False lets every query see every key, as image and cross-attention do."""
        self.key_bias, self.causal = key_bias, causal
        fused = getattr(self.xp, "fused_attention", None) if key_bias is None and causal else None  # the kernels take no mask
        out = fused(q, k, v) if fused else None
        self.p = None
        if out is None:
            s = self.scores(q, k)
            m = s.max(axis=-1, keepdims=True)
            p = self.xp.exp(s - m)
            l = p.sum(axis=-1, keepdims=True)
            self.p = p / l
            out = self.p @ self.xp.repeat(v, q.shape[1] // k.shape[1], axis=1), (m + self.xp.log(l))[..., 0]
        self.saved = q, k, v, out[0], out[1].astype(q.dtype)
        return out[0]

    def backward(self, grad):
        q, k, v, out, lse = self.saved
        xp = self.xp
        fused = getattr(xp, "fused_attention_backward", None)
        if fused is not None and self.p is None:  # the forward ran fused (CUDA): the kernels' backward
            return fused(q, k, v, out, lse, grad)
        B, Hkv, S, D = k.shape
        rep = q.shape[1] // Hkv
        p = self.p if self.p is not None else xp.exp(self.scores(q, k) - lse[..., None])
        dv = p.swapaxes(-1, -2) @ grad
        dp = grad @ xp.repeat(v, rep, axis=1).swapaxes(-1, -2)
        # softmax backward: ds = p * (dp - rowsum(p * dp)), and rowsum(p * dp) = rowsum(grad * out)
        ds = p * (dp - (grad * out).sum(axis=-1, keepdims=True)) * D ** -0.5
        dq = ds @ xp.repeat(k, rep, axis=1)
        dk = ds.swapaxes(-1, -2) @ q
        # each kv head served `rep` query heads, so it collects all their gradients
        return dq, dk.reshape(B, Hkv, rep, S, D).sum(axis=2), dv.reshape(B, Hkv, rep, S, D).sum(axis=2)

    def out_shape(self, q, k, v, key_bias=None, causal=True):
        return q[:-1] + v[-1:]

    def out_dtype(self, q, k, v, key_bias=None, causal=True):
        return q
