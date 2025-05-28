# MiniTorch

A minimalist PyTorch-like deep learning engine built from scratch.  
Features:
- Autograd with NumPy backend and full broadcasting  
- `Tensor` class: arithmetic ops, `sum`, `reshape`, `.backward()`  
- Layers: `Linear`, `ReLU`, `BatchNorm1d`  
- Losses: MSE, Cross-Entropy  
- Optimizers: SGD (with momentum), Adam  
- Toy regression demo: learn y=2x+1, with timing and loss plot  
- Benchmark script: compare MiniTorch vs NumPy vs PyTorch performance  

## Installation

```bash
pip install -e .

mkdocs serve
```

## Quickstart

```python
from minitorch import Tensor, Linear, SGD, mse_loss

# Create data
x = Tensor([[1.0], [2.0], [3.0]], requires_grad=False)
y = Tensor([[3.0], [5.0], [7.0]], requires_grad=False)

# Build model
model = [Linear(1, 1)]
params = []
for layer in model:
    params += layer.parameters()

optimizer = SGD(params, lr=0.1)

# Training loop
for epoch in range(10):
    pred = model[0](x)
    loss = mse_loss(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss {loss.data:.4f}")
```

## Demo: Toy Regression

Run:

```bash
python train.py
```

to see training time and loss plot.

## Benchmarking

You can benchmark MiniTorch against pure NumPy and PyTorch implementations using the provided `benchmark.py` script:

```bash
python benchmark.py
```

Example results (your times may vary):

```
Benchmark results for 100 epochs on 100 samples:
  MiniTorch: 0.0251s
    NumPy: 0.0123s
  PyTorch: 0.0056s
```

Make sure you have PyTorch installed (for comparisions)

```bash
pip install torch
```

![image](https://i.ibb.co/gLG4HMHc/Screenshot-2025-05-28-at-11-19-04-PM.png)


Inspired by karpathy and george hotz.


## License

MIT