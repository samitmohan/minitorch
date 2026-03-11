# Tutorial

## 1. Hello, Autograd

```python
from minitorch import Tensor

x = Tensor(2.0, requires_grad=True)
y = x ** 2 + 3.0 * x
y.backward()
print(x.grad)  # 2*x + 3 = 7
```

## 2. Tensor Operations

```python
from minitorch import Tensor, cat, stack

a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

# reductions
print(a.mean())        # scalar mean
print(a.var())         # variance
print(a.min(axis=1))   # min along axis

# shape ops
b = a.unsqueeze(0)     # (1, 2, 2)
c = a.squeeze()        # removes size-1 dims
d = a.clamp(0, 3)      # clamp values

# concatenation
e = cat([a, a], axis=0)   # (4, 2)
f = stack([a, a], axis=0) # (2, 2, 2)
```

## 3. Module Subclassing

```python
from minitorch import Module, Linear, ReLU, Tensor

class MyNet(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 128)
        self.relu = ReLU()
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = MyNet()
print(len(model.parameters()))  # 4 (2 weights + 2 biases)
```

## 4. Functional API

```python
import minitorch.functional as F
from minitorch import Tensor

x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
y = F.relu(x)         # [0, 0, 1]
z = F.softmax(x)      # normalized probabilities
w = F.log_softmax(x)  # numerically stable log-softmax
```

## 5. Training with no_grad and Gradient Clipping

```python
from minitorch import no_grad, clip_grad_norm

# training loop
loss.backward()
clip_grad_norm(model.parameters(), max_norm=1.0)
optimizer.step()

# inference
model.eval()
with no_grad():
    predictions = model(test_data)
```

## 6. Learning Rate Scheduling

```python
from minitorch import Adam, StepLR, CosineAnnealingLR

opt = Adam(model.parameters(), lr=0.01)

# drop lr by half every 10 epochs
scheduler = StepLR(opt, step_size=10, gamma=0.5)

# or decay smoothly with cosine
scheduler = CosineAnnealingLR(opt, T_max=50)

for epoch in range(50):
    # ... train ...
    scheduler.step()
```

## 7. Save and Load Models

```python
import numpy as np

# save
state = model.state_dict()
np.savez("model.npz", **state)

# load into new model
model2 = MyNet()
data = np.load("model.npz")
model2.load_state_dict(dict(data))
```

## 8. CNN Example

```python
from minitorch import Sequential, Conv2d, MaxPool2d, Flatten, Linear, ReLU

cnn = Sequential(
    Conv2d(1, 16, 3, padding=1),
    ReLU(),
    MaxPool2d(2),
    Flatten(),
    Linear(16 * 14 * 14, 10),
)
```

## 9. MNIST Demo

```bash
# MLP
uv run python mnist_example.py --model mlp --epochs 15

# CNN
uv run python mnist_example.py --model cnn --epochs 10 --n-train 2000
```

## 10. Binary Classification

```python
from minitorch import Tensor, Linear, Sigmoid, bce_loss, Adam
import minitorch.functional as F

model = Linear(4, 1, init='xavier')
opt = Adam([model.weight, model.bias], lr=0.01)

x = Tensor(...)  # (batch, 4)
y = Tensor(...)  # (batch, 1), values 0 or 1

probs = F.sigmoid(model(x))
loss = bce_loss(probs, y)
opt.zero_grad()
loss.backward()
opt.step()
```

## 11. Visualizing the Graph

```python
from minitorch import Tensor, draw_graph

x = Tensor(2.0, requires_grad=True)
y = x ** 2 + 3.0 * x

dot = draw_graph(y)
dot.render('graph', format='png')
```

Requires `graphviz`: `uv pip install graphviz`

## 12. Gradient Checking

```python
from minitorch import Tensor, check_gradient

a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
check_gradient(lambda: (a ** 2).sum(), [a])
```

## 13. Streamlit Playground

```bash
uv run --extra app streamlit run app.py
```
