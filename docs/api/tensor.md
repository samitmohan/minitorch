# Tensor

The core `Tensor` class wraps a NumPy array and tracks gradients for backpropagation.


## Key Methods

- `Tensor(data, requires_grad=False)`: create a tensor.
- `tensor.sum(axis=None, keepdims=False)`: sum elements.
- `tensor.reshape(*shape)`: reshape tensor.
- `tensor.backward()`: compute gradients.

## Example

```python
from minitorch import Tensor

x = Tensor([[2.0, 0.0, -2.0]], requires_grad=True)
I = Tensor.eye(3, requires_grad=True)
z = x.matmul(I).sum()
z.backward()

print(I.grad)
print(x.grad)
```
