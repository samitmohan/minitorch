# Getting Started

Welcome to **MiniTorch**, a minimalist PyTorch-like deep learning engine built from scratch.

## Installation

Install MiniTorch and documentation dependencies:

```bash
pip install -e .
pip install mkdocs-material streamlit
```

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
