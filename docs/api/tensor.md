# Tensor

The core `Tensor` class wraps a NumPy (or CuPy) array and tracks gradients for backpropagation.

## Constructor

```python
Tensor(data, requires_grad=False)
```

## Properties

- `shape`, `ndim`, `dtype` - standard array properties
- `T` - transpose (swaps last two dims)
- `device` - `"cpu"` or `"cuda"`
- `size(dim=None)` - total elements or size of a specific dimension

## Static Constructors

```python
Tensor.zeros(2, 3)
Tensor.ones(2, 3, requires_grad=True)
Tensor.randn(4, 5, device="cuda")
Tensor.eye(3)
```

## Arithmetic

`+`, `-`, `*`, `/`, `**` with broadcasting. Right-hand variants (`5.0 + tensor`) supported. `@` for matrix multiplication.

## Reductions

- `sum(axis, keepdims)`, `mean(axis, keepdims)`
- `max(axis, keepdims)`, `min(axis, keepdims)`
- `var(axis, keepdims)`, `std(axis, keepdims)`

## Shape Operations

- `reshape(*shape)`, `transpose(dim0, dim1)`
- `squeeze(axis)`, `unsqueeze(axis)`
- `__getitem__` for indexing/slicing

## Elementwise

- `exp()`, `log()`, `abs()`, `clamp(min_val, max_val)`

## Autograd

- `backward()` - reverse-mode autodiff (scalar tensors only)
- `zero_grad()`, `detach()`, `clone()`

## Device Transfer

```python
x = Tensor.randn(3, 4)
x_gpu = x.to("cuda")
```

## no_grad Context Manager

Disable gradient tracking for inference:

```python
from minitorch import no_grad

with no_grad():
    y = model(x)  # no gradients tracked
```

## Free Functions

```python
from minitorch import cat, stack

c = cat([a, b], axis=0)
s = stack([a, b], axis=0)
```
