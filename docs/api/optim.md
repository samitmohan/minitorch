# Optimizers

## `SGD(params, lr, momentum=0.0)`

Stochastic Gradient Descent with momentum:

```python
from minitorch.optim import SGD
opt = SGD(model.parameters(), lr=0.01, momentum=0.9)
opt.zero_grad()
loss.backward()
opt.step()
```

## `Adam(params, lr, betas=(0.9,0.999), eps=1e-8)`

Adam optimizer.
```python
from minitorch.optim import Adam
opt = Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8)
opt.zero_grad()
loss.backward()
opt.step()
```