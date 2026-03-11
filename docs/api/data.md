# DataLoader

Batched iteration over datasets with optional shuffling.

## `DataLoader(x, y, batch_size=32, shuffle=True, drop_last=False)`

```python
from minitorch import DataLoader
import numpy as np

X = np.random.randn(1000, 784).astype(np.float32)
Y = np.random.randn(1000, 10).astype(np.float32)

loader = DataLoader(X, Y, batch_size=64, shuffle=True)

for x_batch, y_batch in loader:
    # x_batch: (64, 784), y_batch: (64, 10)
    pass

print(len(loader))  # number of batches
```

### Parameters

- `x` - numpy array of inputs
- `y` - numpy array of targets
- `batch_size` - samples per batch (default 32)
- `shuffle` - shuffle indices each epoch (default True)
- `drop_last` - if True, drop the last batch when it is smaller than `batch_size` (default False)
