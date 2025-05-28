import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
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
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad
        out._backward, out._prev = _backward, (self, other)
        out._op = 'add'
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad
        out._backward, out._prev = _backward, (self, other)
        out._op = 'mul'
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data @ other.data
        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        out._backward, out._prev = _backward, (self, other)
        out._op = 'matmul'
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
