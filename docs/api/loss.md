# Loss Functions

## `mse_loss(input, target)`

Mean Squared Error:

```python
from minitorch.loss import mse_loss
loss = mse_loss(predictions, targets)
```

## `cross_entropy_loss(input, target)`

Cross-entropy for classification.

```python
from minitorch.loss import cross_entropy_loss
loss = cross_entropy_loss(predictions, targets)
```