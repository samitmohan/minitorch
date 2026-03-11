# MiniTorch

A PyTorch clone I built from scratch to understand how autograd and neural networks work under the hood.

[Docs](https://samitmohan.github.io/minitorch/)

[![CI](https://github.com/samitmohan/minitorch/actions/workflows/ci.yml/badge.svg)](https://github.com/samitmohan/minitorch/actions/workflows/ci.yml)

## What's in here

- Reverse-mode autograd with topological sort
- Tensor with broadcasting, slicing, reductions
- Module system (parameters, train/eval, state_dict)
- Layers: Linear (kaiming/xavier init), Conv2d, MaxPool2d, BatchNorm1d, Dropout, ReLU, Sigmoid, Tanh, Softmax
- SGD (with momentum) and Adam optimizers
- LR schedulers: StepLR, CosineAnnealingLR
- Cross-entropy, MSE, and binary cross-entropy loss
- Computation graph visualization with graphviz
- DataLoader, gradient clipping, numerical gradient checking
- Optional CUDA via CuPy

## Setup

```bash
git clone https://github.com/samitmohan/minitorch.git
cd minitorch
uv pip install -e .
```

## Run

```bash
# regression demo
uv run python train.py

# MNIST
uv run python mnist_example.py --model mlp --epochs 15
uv run python mnist_example.py --model cnn --epochs 10 --n-train 2000

# tests
uv run --extra dev pytest tests/ -v

# streamlit playground
uv run --extra app streamlit run app.py

# docs
uv run --extra docs mkdocs serve
```

## License

MIT
