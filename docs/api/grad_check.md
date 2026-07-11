# Gradient Checking

Verify analytic gradients from `backward()` against numerical central differences.

```python
from minitorch import Tensor, check_gradient

a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
check_gradient(lambda: (a * b).sum(), [a, b])
```

::: minitorch.grad_check.check_gradient

::: minitorch.grad_check.numerical_gradient
