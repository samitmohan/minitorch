[![CI](https://github.com/samitmohan/minitorch/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/samitmohan/minitorch/actions/workflows/ci.yml)

# MiniTorch

> A PyTorch clone built from scratch to understand how autograd and neural networks actually work.

<p align="center">
  <img src="assets/banner.png" alt="MiniTorch Banner" width="300">
</p>

---

## What is this?

MiniTorch is a small deep learning framework I wrote in Python + NumPy. Everything is implemented from first principles: the autograd engine, the layers, the optimizers.

It has:

- A `Tensor` class with automatic gradient tracking and broadcasting
- Reverse-mode autodiff using topological sort
- A `Module` system for building networks (like `nn.Module` in PyTorch)
- Common layers: Linear, Conv2d, MaxPool2d, BatchNorm1d, Dropout, ReLU, Sigmoid, Tanh, Softmax
- SGD (with momentum) and Adam, with LR schedulers (StepLR, CosineAnnealingLR)
- MSE, cross-entropy, and binary cross-entropy loss
- Computation graph visualization with graphviz
- DataLoader, gradient clipping, numerical gradient checking
- Optional CUDA support via CuPy

---

## Install

```bash
git clone https://github.com/samitmohan/minitorch.git
cd minitorch
uv pip install -e .
```

Optional extras:

```bash
uv pip install -e ".[dev]"    # pytest
uv pip install -e ".[docs]"   # mkdocs
uv pip install -e ".[app]"    # streamlit
uv pip install -e ".[cuda]"   # cupy (needs CUDA 12.x)
uv pip install -e ".[all]"    # everything
```

---

## Quick Example

```python
from minitorch import Tensor, Sequential, Linear, ReLU, SGD, mse_loss

x = Tensor([[1.0], [2.0], [3.0]])
y = Tensor([[3.0], [5.0], [7.0]])

model = Sequential(Linear(1, 1))
opt = SGD(model.parameters(), lr=0.1)

for epoch in range(50):
    pred = model(x)
    loss = mse_loss(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
```

---

## Running Examples

```bash
# linear regression
uv run python train.py

# MNIST with MLP
uv run python mnist_example.py --model mlp --epochs 15

# MNIST with CNN
uv run python mnist_example.py --model cnn --epochs 10 --n-train 2000

# benchmark against numpy/pytorch
uv run python benchmark.py
```

### Streamlit Playground

Interactive app with autograd visualization, MNIST training, and gradient checking:

```bash
uv run --extra app streamlit run app.py
```

---

## Tests

```bash
# all tests
uv run --extra dev pytest tests/ -v

# just gradient checks
uv run --extra dev pytest tests/test_gradients.py -v

# just tensor ops
uv run --extra dev pytest tests/test_tensor_ops.py -v

# legacy sanity check
uv run python test_cases.py
```

---

## Docs

To serve these docs locally:

```bash
uv run --extra docs mkdocs serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

To build static HTML into `site/`:

```bash
uv run --extra docs mkdocs build
```

---

## Project Layout

```
minitorch/
    tensor.py       # Tensor + autograd engine
    module.py       # Module base class, Sequential
    layers.py       # Linear, activations, Dropout, BatchNorm1d
    conv.py         # Conv2d (im2col), MaxPool2d (stride tricks), Flatten
    loss.py         # mse_loss, cross_entropy_loss, bce_loss
    optim.py        # SGD, Adam, LR schedulers, gradient clipping
    functional.py   # stateless activation functions
    viz.py          # computation graph visualization
    data.py         # DataLoader
    backend.py      # CPU/CUDA switching
    grad_check.py   # numerical gradient verification
tests/
    test_tensor_ops.py
    test_gradients.py
    test_new_features.py
```
