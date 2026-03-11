# Module

Base class for all neural network layers.

## `Module`

```python
from minitorch import Module, Linear, ReLU

class MyModel(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 128)
        self.relu = ReLU()
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
```

### Methods

- `forward(*args)` - override in subclass
- `parameters()` - auto-collects all trainable Tensors
- `train()` / `eval()` - switch training/eval mode
- `state_dict()` - returns dict of parameter names to numpy arrays
- `load_state_dict(state)` - loads parameters from dict

### Save and Load

```python
import numpy as np

# save
state = model.state_dict()
np.savez("model.npz", **state)

# load
data = np.load("model.npz")
model.load_state_dict(dict(data))
```

## `Sequential(*layers)`

Container that chains layers in order:

```python
from minitorch import Sequential, Linear, ReLU

model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10),
)
```
