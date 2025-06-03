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
    def __init__(self, data, requires_grad=False):
        # Wrap numbers, lists, or numpy arrays
        self.data = np.array(data, dtype=data.dtype if isinstance(data, np.ndarray) else np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = ()
        self._op = ''

    @staticmethod
    def zeros(shape, requires_grad=False, dtype=np.float32):
        return Tensor(np.zeros(shape, dtype=dtype), requires_grad)

    @staticmethod
    def eye(n, requires_grad=False, dtype=np.float32):
        return Tensor(np.eye(n, dtype=dtype), requires_grad)

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)
        out._backward, out._prev = _backward, (self,)
        out._op = 'reshape'
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                grad = out.grad
                self.grad += np.broadcast_to(grad, self.data.shape)
        out._backward, out._prev = _backward, (self,)
        out._op = 'sum'
        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad)
        def _backward():
            if self.requires_grad:
                grad_self = out.grad
                if self.data.shape != out.data.shape:
                    grad_self = _sum_to_shape(out.grad, self.data.shape)
                self.grad += grad_self
            if other.requires_grad:
                grad_other = out.grad
                if other.data.shape != out.data.shape:
                    grad_other = _sum_to_shape(out.grad, other.data.shape)
                other.grad += grad_other
        out._backward, out._prev, out._op = _backward, (self, other), 'add'
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad)
        def _backward():
            if self.requires_grad:
                grad_self = other.data * out.grad
                if self.data.shape != out.data.shape:
                    grad_self = _sum_to_shape(grad_self, self.data.shape)
                self.grad += grad_self
            if other.requires_grad:
                grad_other = self.data * out.grad
                if other.data.shape != out.data.shape:
                    grad_other = _sum_to_shape(grad_other, other.data.shape)
                other.grad += grad_other
        out._backward, out._prev, out._op = _backward, (self, other), 'mul'
        return out

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data @ other.data
        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        out._backward, out._prev, out._op = _backward, (self, other), 'matmul'
        return out

    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build(parent)
                topo.append(v)
        build(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"