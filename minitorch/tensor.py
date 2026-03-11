import numpy as np
from .backend import get_array_module, to_device, gpu_available

# global flag for no_grad context
_grad_enabled = True


class no_grad:
    """Context manager to disable gradient tracking during inference."""
    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, *args):
        global _grad_enabled
        _grad_enabled = self._prev


def _sum_to_shape(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, (g_dim, s_dim) in enumerate(zip(grad.shape, shape)):
        if s_dim == 1 and g_dim != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


def _accum_grad(tensor, grad):
    """Accumulate gradient into tensor, initializing if needed."""
    if tensor.grad is None:
        tensor.grad = np.zeros_like(tensor.data)
    tensor.grad += grad


class Tensor:
    def __init__(self, data, *, requires_grad=False):
        if isinstance(data, np.ndarray):
            self.data = data if data.dtype in (np.float32, np.float64) else data.astype(np.float32)
        elif isinstance(data, np.floating):
            self.data = np.array(data, dtype=data.dtype)
        elif gpu_available() and hasattr(data, '__cuda_array_interface__'):
            self.data = data if data.dtype in (np.float32, np.float64) else data.astype(np.float32)
        else:
            self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad and _grad_enabled
        self.grad = None
        self._backward = lambda: None
        self._prev = set()
        self._op = ''

    # properties

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return self.transpose()

    @property
    def device(self):
        xp = get_array_module(self.data)
        return "cuda" if xp is not np else "cpu"

    def size(self, dim=None):
        if dim is None:
            return self.data.size
        return self.data.shape[dim]

    def to(self, device):
        new_data = to_device(self.data, device)
        t = Tensor(new_data, requires_grad=self.requires_grad)
        if self.grad is not None:
            t.grad = to_device(self.grad, device)
        return t

    # static constructors

    @staticmethod
    def zeros(*shape, requires_grad=False, device="cpu"):
        t = Tensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)
        return t.to(device) if device != "cpu" else t

    @staticmethod
    def ones(*shape, requires_grad=False, device="cpu"):
        t = Tensor(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)
        return t.to(device) if device != "cpu" else t

    @staticmethod
    def randn(*shape, requires_grad=False, device="cpu"):
        t = Tensor(np.random.randn(*shape).astype(np.float32), requires_grad=requires_grad)
        return t.to(device) if device != "cpu" else t

    @staticmethod
    def eye(n, requires_grad=False, device="cpu"):
        t = Tensor(np.eye(n, dtype=np.float32), requires_grad=requires_grad)
        return t.to(device) if device != "cpu" else t

    # arithmetic ops

    def _should_track(self, *others):
        return _grad_enabled and any(
            t.requires_grad for t in (self, *others)
        )

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self._should_track(other))

        def _backward():
            if self.requires_grad:
                _accum_grad(self, _sum_to_shape(out.grad, self.data.shape))
            if other.requires_grad:
                _accum_grad(other, _sum_to_shape(out.grad, other.data.shape))

        out._backward = _backward
        out._prev = {self, other}
        out._op = 'add'
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, requires_grad=self._should_track(other))

        def _backward():
            if self.requires_grad:
                _accum_grad(self, _sum_to_shape(out.grad, self.data.shape))
            if other.requires_grad:
                _accum_grad(other, _sum_to_shape(-out.grad, other.data.shape))

        out._backward = _backward
        out._prev = {self, other}
        out._op = 'sub'
        return out

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other.__sub__(self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self._should_track(other))

        def _backward():
            if self.requires_grad:
                _accum_grad(self, _sum_to_shape(other.data * out.grad, self.data.shape))
            if other.requires_grad:
                _accum_grad(other, _sum_to_shape(self.data * out.grad, other.data.shape))

        out._backward = _backward
        out._prev = {self, other}
        out._op = 'mul'
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, requires_grad=self._should_track(other))

        def _backward():
            if self.requires_grad:
                _accum_grad(self, _sum_to_shape(out.grad / other.data, self.data.shape))
            if other.requires_grad:
                # d(a/b)/db = -a / b^2
                _accum_grad(other, _sum_to_shape(-self.data / (other.data ** 2) * out.grad, other.data.shape))

        out._backward = _backward
        out._prev = {self, other}
        out._op = 'div'
        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other.__truediv__(self)

    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data ** other.data, requires_grad=self._should_track(other))

        def _backward():
            if self.requires_grad:
                # d(a^b)/da = b * a^(b-1)
                _accum_grad(self, _sum_to_shape(other.data * (self.data ** (other.data - 1)) * out.grad, self.data.shape))
            if other.requires_grad:
                # d(a^b)/db = a^b * ln(|a|), only valid for positive base
                safe_base = np.abs(self.data) + 1e-12
                _accum_grad(other, _sum_to_shape(out.data * np.log(safe_base) * out.grad, other.data.shape))

        out._backward = _backward
        out._prev = {self, other}
        out._op = 'pow'
        return out

    def __neg__(self):
        return self * -1

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, requires_grad=self._should_track(other))

        def _backward():
            if self.requires_grad:
                _accum_grad(self, out.grad @ other.data.T)
            if other.requires_grad:
                _accum_grad(other, self.data.T @ out.grad)

        out._backward = _backward
        out._prev = {self, other}
        out._op = 'matmul'
        return out

    # reduction ops

    def sum(self, axis=None, keepdims=False):
        data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(data, requires_grad=self.requires_grad and _grad_enabled)

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                _accum_grad(self, np.broadcast_to(grad, self.data.shape))

        out._backward = _backward
        out._prev = {self}
        out._op = 'sum'
        return out

    def mean(self, axis=None, keepdims=False):
        data = self.data.mean(axis=axis, keepdims=keepdims)
        out = Tensor(data, requires_grad=self.requires_grad and _grad_enabled)

        def _backward():
            if self.requires_grad:
                if axis is None:
                    count = self.data.size
                elif isinstance(axis, int):
                    count = self.data.shape[axis]
                else:
                    count = 1
                    for a in axis:
                        count *= self.data.shape[a]
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                _accum_grad(self, np.broadcast_to(grad, self.data.shape) / count)

        out._backward = _backward
        out._prev = {self}
        out._op = 'mean'
        return out

    def var(self, axis=None, keepdims=False):
        # var = mean((x - mean(x))^2)
        m = self.mean(axis=axis, keepdims=True)
        return ((self - m) ** 2).mean(axis=axis, keepdims=keepdims)

    def std(self, axis=None, keepdims=False):
        return (self.var(axis=axis, keepdims=keepdims) + 1e-12) ** 0.5

    def max(self, axis=None, keepdims=False):
        data = self.data.max(axis=axis, keepdims=keepdims)
        out = Tensor(data, requires_grad=self.requires_grad and _grad_enabled)

        def _backward():
            if self.requires_grad:
                max_val = self.data.max(axis=axis, keepdims=True)
                mask = (self.data == max_val).astype(self.data.dtype)
                mask = mask / mask.sum(axis=axis, keepdims=True)
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                _accum_grad(self, np.broadcast_to(grad, self.data.shape) * mask)

        out._backward = _backward
        out._prev = {self}
        out._op = 'max'
        return out

    def min(self, axis=None, keepdims=False):
        data = self.data.min(axis=axis, keepdims=keepdims)
        out = Tensor(data, requires_grad=self.requires_grad and _grad_enabled)

        def _backward():
            if self.requires_grad:
                min_val = self.data.min(axis=axis, keepdims=True)
                mask = (self.data == min_val).astype(self.data.dtype)
                mask = mask / mask.sum(axis=axis, keepdims=True)
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                _accum_grad(self, np.broadcast_to(grad, self.data.shape) * mask)

        out._backward = _backward
        out._prev = {self}
        out._op = 'min'
        return out

    # shape ops

    def reshape(self, *shape):
        data = self.data.reshape(*shape)
        out = Tensor(data, requires_grad=self.requires_grad and _grad_enabled)

        def _backward():
            if self.requires_grad:
                _accum_grad(self, out.grad.reshape(self.data.shape))

        out._backward = _backward
        out._prev = {self}
        out._op = 'reshape'
        return out

    def transpose(self, dim0=-2, dim1=-1):
        axes = list(range(self.data.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        data = self.data.transpose(axes)
        out = Tensor(data, requires_grad=self.requires_grad and _grad_enabled)

        def _backward():
            if self.requires_grad:
                _accum_grad(self, out.grad.transpose(axes))

        out._backward = _backward
        out._prev = {self}
        out._op = 'transpose'
        return out

    def squeeze(self, axis=None):
        data = self.data.squeeze(axis=axis)
        orig_shape = self.data.shape
        out = Tensor(data, requires_grad=self.requires_grad and _grad_enabled)

        def _backward():
            if self.requires_grad:
                _accum_grad(self, out.grad.reshape(orig_shape))

        out._backward = _backward
        out._prev = {self}
        out._op = 'squeeze'
        return out

    def unsqueeze(self, axis):
        data = np.expand_dims(self.data, axis=axis)
        out = Tensor(data, requires_grad=self.requires_grad and _grad_enabled)

        def _backward():
            if self.requires_grad:
                _accum_grad(self, out.grad.squeeze(axis=axis))

        out._backward = _backward
        out._prev = {self}
        out._op = 'unsqueeze'
        return out

    def __getitem__(self, key):
        data = self.data[key]
        out = Tensor(data, requires_grad=self.requires_grad and _grad_enabled)

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                np.add.at(self.grad, key, out.grad)

        out._backward = _backward
        out._prev = {self}
        out._op = 'getitem'
        return out

    # elementwise ops

    def exp(self):
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad and _grad_enabled)

        def _backward():
            if self.requires_grad:
                _accum_grad(self, out.data * out.grad)

        out._backward = _backward
        out._prev = {self}
        out._op = 'exp'
        return out

    def log(self):
        data = np.log(np.maximum(self.data, 1e-12))
        out = Tensor(data, requires_grad=self.requires_grad and _grad_enabled)

        def _backward():
            if self.requires_grad:
                _accum_grad(self, (1.0 / np.maximum(self.data, 1e-12)) * out.grad)

        out._backward = _backward
        out._prev = {self}
        out._op = 'log'
        return out

    def clamp(self, min_val=None, max_val=None):
        data = self.data.copy()
        if min_val is not None:
            data = np.maximum(data, min_val)
        if max_val is not None:
            data = np.minimum(data, max_val)
        out = Tensor(data, requires_grad=self.requires_grad and _grad_enabled)

        def _backward():
            if self.requires_grad:
                mask = np.ones_like(self.data)
                if min_val is not None:
                    mask *= (self.data >= min_val)
                if max_val is not None:
                    mask *= (self.data <= max_val)
                _accum_grad(self, out.grad * mask)

        out._backward = _backward
        out._prev = {self}
        out._op = 'clamp'
        return out

    def abs(self):
        out = Tensor(np.abs(self.data), requires_grad=self.requires_grad and _grad_enabled)

        def _backward():
            if self.requires_grad:
                _accum_grad(self, np.sign(self.data) * out.grad)

        out._backward = _backward
        out._prev = {self}
        out._op = 'abs'
        return out

    # autograd

    def backward(self):
        assert self.data.size == 1, "backward() only works on scalar tensors - call .sum() or .mean() first"
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        self.grad = None

    def detach(self):
        return Tensor(self.data, requires_grad=False)

    def clone(self):
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __len__(self):
        return self.data.shape[0]

    def __float__(self):
        return float(self.data)


# free functions

def cat(tensors, axis=0):
    data = np.concatenate([t.data for t in tensors], axis=axis)
    any_grad = any(t.requires_grad for t in tensors)
    out = Tensor(data, requires_grad=any_grad and _grad_enabled)

    def _backward():
        sizes = [t.data.shape[axis] for t in tensors]
        grads = np.split(out.grad, np.cumsum(sizes[:-1]), axis=axis)
        for t, g in zip(tensors, grads):
            if t.requires_grad:
                _accum_grad(t, g)

    out._backward = _backward
    out._prev = set(tensors)
    out._op = 'cat'
    return out


def stack(tensors, axis=0):
    data = np.stack([t.data for t in tensors], axis=axis)
    any_grad = any(t.requires_grad for t in tensors)
    out = Tensor(data, requires_grad=any_grad and _grad_enabled)

    def _backward():
        for i, t in enumerate(tensors):
            if t.requires_grad:
                slices = [slice(None)] * out.grad.ndim
                slices[axis] = i
                _accum_grad(t, out.grad[tuple(slices)])

    out._backward = _backward
    out._prev = set(tensors)
    out._op = 'stack'
    return out
