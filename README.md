# MiniTorch

A minimalist PyTorch-like deep learning engine built from scratch.  
[Link](https://samitmohan.github.io/minitorch/)

![image](https://i.ibb.co/bgyGNtX9/Screenshot-2025-06-04-at-4-11-30-AM.png)
## Features

- **`Tensor`**: NumPy-backed arrays with full broadcasting, `.grad`, and `.backward()`.  
- **Layers**:  
  - `Linear(in_features, out_features, bias=True)`  
  - `ReLU()`  
  - `BatchNorm1d(num_features)` (forward only; backward is a placeholder)  
- **Losses**:  
  - `mse_loss(input, target)`  
  - `cross_entropy_loss(input, target)`  
- **Optimizers**:  
  - `SGD(params, lr, momentum=0.0)`  
  - `Adam(params, lr, betas=(0.9,0.999), eps=1e-8)`  
- **Demos & Examples**:  
  - **Toy Regression** (`train.py`) to fit *y = 2x + 1* (with loss plot).  
  - **MNIST Example** (`mnist_example.py`): small MLP on MNIST subset.  
  - **Benchmark** (`benchmark.py`): compare MiniTorch, NumPy, and PyTorch runtimes.  
  - **Autograd Tests** (`test_cases.py`): ensure gradient correctness and broadcasting.

## Installation

Clone this repo and install.

```bash
git clone https://github.com/samitmohan/minitorch.git
cd minitorch
pip install -e .
```

## Demo

**Regression**

- This script will:
	- Generate 100 random (x, y) pairs with noise.
	- Train a single Linear(1,1) model for 100 epochs.
	- Print elapsed training time and final MSE loss.
	- Display a plot of loss over epochs.

**MNIST**

- This demo:
	- Fetches the MNIST dataset (using sklearn.datasets.fetch_openml).
	- Uses 1,000 training samples and 200 test samples.
	- Builds a 2‐layer MLP (784 → 128 → 10) with ReLU activation.
	- Trains for 10 epochs with SGD (momentum).
	- Reports average loss per epoch and final test accuracy.

Run:

```bash
python train.py
python mnist_example.py
```

To see training loss and final test accuracy.

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

## TODO
- _backward() for BatchNormalisation
- Add /tests in Docs
- Why MiniTorch is slow? Tradeoffs (Add to Docs)


## License

MIT