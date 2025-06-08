# Tutorial

Welcome to MiniTorch! This tutorial will guide you through:

## 1. Hello, Autograd

```python
from minitorch.tensor import Tensor

x = Tensor(2.0, requires_grad=True)
y = x * x + Tensor(3.0) * x

y.backward()

print(x.grad)  # should be 2*x + 3 = 7
```

## 2. Regression Demo

Learn a simple linear relationship y = 2x + 1:

```bash
python train.py
```

This runs 100 epochs on synthetic data and plots training loss.

## 3. MNIST Demo

Train a small 2-Layer neural network on MNIST:
```bash
python mnist_example.py
```

## 4. Benchmarking

Compare MiniTorch performance to NumPy and PyTorch:

```bash
python benchmark.py
```

See timing output in the console.