# Performance

MiniTorch is slow compared to PyTorch. That's by design - the goal is readable code, not speed. This page explains where the time goes and what PyTorch does differently.

---

## Benchmark

```bash
uv run python benchmark.py
```

On a typical machine, for 100 epochs of linear regression on 100 samples:

| Framework | Time |
|-----------|------|
| MiniTorch | ~25ms |
| NumPy (manual) | ~12ms |
| PyTorch | ~5ms |

MiniTorch is roughly 2x slower than hand-written NumPy and 5x slower than PyTorch. The gap gets worse on larger models.

---

## Where the time goes

### 1. Python object creation

Every operation creates a new `Tensor` object with a closure for its backward function. A simple `y = x * w + b` creates 3 new Tensor objects, 3 closures, 3 sets for `_prev`, and 3 strings for `_op`.

PyTorch does this in C++. Object creation is nearly free.

### 2. One NumPy call per operation

`x @ w + b` makes two separate NumPy calls: one for matmul, one for add. Each call has Python-to-C overhead, and NumPy allocates a new array for each result.

PyTorch fuses operations into optimized kernels. A fused matmul+bias is significantly faster than two separate calls.

### 3. No in-place operations

MiniTorch always creates new arrays. There's no way to do `x += 1` without creating a new tensor. This means extra memory allocation and copying.

PyTorch has in-place variants (like `add_`, `mul_`) that modify tensors without allocation.

### 4. Autograd overhead

The backward pass walks a Python graph of Tensor objects, calling Python closures at each step. For a model with 100 operations, that's 100 Python function calls.

PyTorch's autograd is written in C++ and processes the graph much more efficiently.

### 5. No BLAS tuning

NumPy uses whatever BLAS library is installed (OpenBLAS, MKL, etc.) but doesn't tune for specific matrix sizes. PyTorch ships with optimized BLAS and uses cuBLAS on GPU, with tuning for common shapes.

---

## What you can do about it

For MiniTorch specifically:

- **Use batched operations**: `x @ w` on a (100, 784) matrix is much faster per-sample than looping over individual samples
- **Keep tensors in float32**: float64 uses 2x memory and is slower on most hardware
- **Use Adam**: it converges in fewer steps than SGD, so you need fewer epochs

For real work, use PyTorch. MiniTorch is for learning.

---

## The trade-off

Everything in MiniTorch that makes it slow is also what makes it readable:

| Design choice | Cost | Benefit |
|--------------|------|---------|
| Pure Python autograd | Slow backward pass | You can read every line |
| Closures for _backward | Memory overhead | Each op's gradient logic is right next to its forward logic |
| NumPy as the only backend | No GPU, no fusion | One dependency, works everywhere |
| New Tensor per operation | Extra allocation | No aliasing bugs, simple mental model |

PyTorch makes the opposite trade-offs: fast but the source code is 3 million lines of C++ and Python spread across hundreds of files.
