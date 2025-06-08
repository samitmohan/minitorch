import numpy as np


def _sum_to_shape(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Sum `grad` over any broadcasted dimensions so that the result
    has shape `shape`.
    """
    # 1) collapse leading extra dims
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # 2) sum along axes where original dim was 1
    for i, (g_dim, s_dim) in enumerate(zip(grad.shape, shape)):
        if s_dim == 1 and g_dim != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    def __init__(self, data, *, requires_grad=False):
        self.data = np.array(data, dtype=data.dtype if isinstance(data, np.ndarray) else np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set()
        self._op = ''

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = _sum_to_shape(out.grad, self.data.shape)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += grad_self
            if other.requires_grad:
                grad_other = _sum_to_shape(out.grad, other.data.shape)
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += grad_other

        out._backward = _backward
        out._prev = {self, other}
        out._op = 'add'
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = _sum_to_shape(other.data * out.grad, self.data.shape)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += grad_self
            if other.requires_grad:
                grad_other = _sum_to_shape(self.data * out.grad, other.data.shape)
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += grad_other

        out._backward = _backward
        out._prev = {self, other}
        out._op = 'mul'
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = out.grad @ other.data.T
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += grad_self
            if other.requires_grad:
                grad_other = self.data.T @ out.grad
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += grad_other

        out._backward = _backward
        out._prev = {self, other}
        out._op = 'matmul'
        return out

    def sum(self, axis=None, keepdims=False):
        data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(data, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = out.grad
                # Broadcast back to original shape
                grad_self = np.broadcast_to(grad_self, self.data.shape)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += grad_self

        out._backward = _backward
        out._prev = {self}
        out._op = 'sum'
        return out

    def reshape(self, *shape):
        data = self.data.reshape(*shape)
        out = Tensor(data, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = out.grad.reshape(self.data.shape)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += grad_self

        out._backward = _backward
        out._prev = {self}
        out._op = 'reshape'
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = out.data * out.grad
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += grad_self

        out._backward = _backward
        out._prev = {self}
        out._op = 'exp'
        return out

    def log(self):
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = (1.0 / self.data) * out.grad
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += grad_self

        out._backward = _backward
        out._prev = {self}
        out._op = 'log'
        return out

    def __neg__(self):
        # Unary negation
        return self * -1

    def backward(self):
        # Initialize root gradient for scalar outputs
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
        """Clears the gradient of this tensor."""
        self.grad = None

    def detach(self):
        """Returns a new Tensor with the same data but no gradient history."""
        return Tensor(self.data.copy(), requires_grad=False)

    def clone(self):
        """Returns a copy of this Tensor, preserving requires_grad."""
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
