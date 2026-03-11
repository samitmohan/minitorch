# Optimizers

## `SGD(params, lr, momentum=0.0, weight_decay=0.0)`

Stochastic Gradient Descent with optional momentum:

```python
from minitorch import SGD
opt = SGD(model.parameters(), lr=0.01, momentum=0.9)
```

## `Adam(params, lr, betas=(0.9,0.999), eps=1e-8, weight_decay=0.0)`

Adam optimizer with bias correction:

```python
from minitorch import Adam
opt = Adam(model.parameters(), lr=0.001)
```

## Gradient Clipping

```python
from minitorch import clip_grad_norm, clip_grad_value

# clip by total norm
loss.backward()
clip_grad_norm(model.parameters(), max_norm=1.0)
opt.step()

# clip by value
loss.backward()
clip_grad_value(model.parameters(), clip_value=0.5)
opt.step()
```

## Learning Rate Schedulers

### `StepLR(optimizer, step_size, gamma=0.1)`

Multiplies the learning rate by `gamma` every `step_size` epochs:

```python
from minitorch import Adam, StepLR

opt = Adam(model.parameters(), lr=0.01)
scheduler = StepLR(opt, step_size=10, gamma=0.5)

for epoch in range(30):
    train_one_epoch()
    scheduler.step()  # lr: 0.01 -> 0.005 -> 0.0025
```

### `CosineAnnealingLR(optimizer, T_max, eta_min=0)`

Decays the learning rate following a cosine curve from the initial lr down to `eta_min` over `T_max` epochs:

```python
from minitorch import Adam, CosineAnnealingLR

opt = Adam(model.parameters(), lr=0.01)
scheduler = CosineAnnealingLR(opt, T_max=50)

for epoch in range(50):
    train_one_epoch()
    scheduler.step()  # smooth decay from 0.01 to 0
```
