"""Learn y = 2x + 1 with a single Linear layer."""
import time
import numpy as np
import matplotlib.pyplot as plt
from minitorch import Tensor, Sequential, Linear, SGD, mse_loss

np.random.seed(0)
x_np = np.random.rand(100, 1).astype(np.float32)
y_np = 2 * x_np + 1 + 0.1 * np.random.randn(100, 1).astype(np.float32)

x = Tensor(x_np)
y = Tensor(y_np)

model = Sequential(Linear(1, 1))
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)

loss_history = []
start = time.perf_counter()
for epoch in range(100):
    pred = model(x)
    loss = mse_loss(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_history.append(float(loss.data))
elapsed = time.perf_counter() - start

print(f"Training time: {elapsed:.4f}s")
print(f"Final loss: {loss_history[-1]:.4f}")

plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()
