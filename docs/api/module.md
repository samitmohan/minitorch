# Module

Subclass `Module`, assign layers as attributes, and implement `forward`:

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

::: minitorch.module.Module

::: minitorch.module.Sequential
