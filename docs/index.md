[![Build Status](https://github.com/samitmohan/minitorch/actions/workflows/ci.yml/badge.svg)](https://github.com/samitmohan/minitorch/actions)
# Getting Started

> A minimalist PyTorch–style deep learning engine built from scratch

![MiniTorch Banner](assets/banner.png)

---

## Overview

MiniTorch is a **small-footprint** deep‐learning framework implemented entirely in Python. It provides:

- **`Tensor`**: a NumPy-backed tensor with full broadcasting and automatic gradient tracking.  
- **`Linear`, `ReLU`, `BatchNorm1d`**: fundamental layer types.  
- **Loss functions**: Mean Squared Error (MSE) and Cross-Entropy.  
- **Optimizers**: SGD and Adam.  
- **Demos**: a toy regression (learn *y = 2x + 1*), a small MNIST example, and benchmarking scripts.

---

## Quickstart

```python
from minitorch import Tensor, Linear, SGD, mse_loss

# Sample data
x = Tensor([[1.0], [2.0], [3.0]], requires_grad=False)
y = Tensor([[3.0], [5.0], [7.0]], requires_grad=False)

# Build model
model = [Linear(1, 1)]
params = []
for layer in model:
    params += layer.parameters()

optimizer = SGD(params, lr=0.1)

# Training loop
for epoch in range(5):
    pred = model[0](x)
    loss = mse_loss(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss {loss.data:.4f}")
```



