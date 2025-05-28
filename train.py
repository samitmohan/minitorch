import time
import numpy as np
import matplotlib.pyplot as plt
from minitorch.tensor import Tensor
from minitorch.layers import Linear, ReLU
from minitorch.loss import mse_loss
from minitorch.optim import SGD

# Toy regression: learn y = 2x + 1
np.random.seed(0)
x_np = np.random.rand(100, 1).astype(np.float32)
y_np = 2 * x_np + 1 + 0.1 * np.random.randn(100, 1).astype(np.float32)

# Convert to Tensors
x = Tensor(x_np, requires_grad=False)
y = Tensor(y_np, requires_grad=False)

# Model
model = [Linear(1, 1)]
params = []
for layer in model:
    params += layer.parameters()

optimizer = SGD(params, lr=0.1, momentum=0.9)

loss_history = []
start = time.perf_counter()
for epoch in range(100):
    # forward
    y_pred = model[0](x)
    loss = mse_loss(y_pred, y)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_history.append(loss.data)
end = time.perf_counter()

print(f"Training time: {end - start:.4f}s")
print(f"Final loss: {loss_history[-1]:.4f}")

# Plot loss over epochs
plt.plot(loss_history)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.show()
