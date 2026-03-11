"""Benchmark MiniTorch vs NumPy vs PyTorch on simple linear regression."""
import time
import numpy as np
from minitorch import Tensor, Linear, SGD, mse_loss


def benchmark_minitorch(x_np, y_np, epochs=100):
    x = Tensor(x_np)
    y = Tensor(y_np)
    model = Linear(1, 1)
    opt = SGD(model.parameters(), lr=0.1, momentum=0.9)
    start = time.perf_counter()
    for _ in range(epochs):
        pred = model(x)
        loss = mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return time.perf_counter() - start


def benchmark_numpy(x_np, y_np, epochs=100):
    w = np.random.randn(1, 1).astype(np.float32)
    b = np.zeros((1,), dtype=np.float32)
    lr = 0.1
    start = time.perf_counter()
    for _ in range(epochs):
        y_pred = x_np @ w + b
        grad_w = 2 * (x_np.T @ (y_pred - y_np)) / x_np.shape[0]
        grad_b = 2 * np.sum(y_pred - y_np, axis=0) / x_np.shape[0]
        w -= lr * grad_w
        b -= lr * grad_b
    return time.perf_counter() - start


def benchmark_torch(x_np, y_np, epochs=100):
    try:
        import torch
    except ImportError:
        return None
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    w = torch.randn(1, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([w, b], lr=0.1, momentum=0.9)
    start = time.perf_counter()
    for _ in range(epochs):
        pred = x @ w + b
        loss = torch.mean((pred - y) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return time.perf_counter() - start


if __name__ == "__main__":
    np.random.seed(0)
    x_np = np.random.rand(100, 1).astype(np.float32)
    y_np = 2 * x_np + 1 + 0.1 * np.random.randn(100, 1).astype(np.float32)

    t_mt = benchmark_minitorch(x_np, y_np)
    t_np = benchmark_numpy(x_np, y_np)
    t_torch = benchmark_torch(x_np, y_np)

    print("Benchmark results for 100 epochs on 100 samples:")
    print(f"  MiniTorch: {t_mt:.4f}s")
    print(f"    NumPy:   {t_np:.4f}s")
    if t_torch is not None:
        print(f"  PyTorch:   {t_torch:.4f}s")
    else:
        print("  PyTorch:   (not installed, skipped)")
