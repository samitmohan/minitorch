# Gradient Checking

Utilities for verifying analytic gradients against numerical (finite difference) gradients.

## `check_gradient(f, inputs, eps=1e-5, atol=1e-4, rtol=1e-3)`

Computes analytic gradients via `backward()` and compares against central differences. Raises `AssertionError` on mismatch.

```python
from minitorch import Tensor, check_gradient

a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
b = Tensor([4.0, 5.0, 6.0], requires_grad=True)

check_gradient(lambda: (a * b).sum(), [a, b])
print("Passed!")
```

### Parameters

- `f` - callable returning a scalar Tensor
- `inputs` - list of Tensors to check gradients for
- `eps` - perturbation size for finite differences
- `atol` - absolute tolerance
- `rtol` - relative tolerance

## `numerical_gradient(f, inputs, eps=1e-5)`

Returns a list of numpy arrays with numerical gradients for each input, computed via central differences: `(f(x+h) - f(x-h)) / 2h`.
