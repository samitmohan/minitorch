# Loss Functions

## `mse_loss(input, target)`

Mean Squared Error: `((input - target) ** 2).mean()`.

```python
from minitorch import mse_loss
loss = mse_loss(predictions, targets)
```

## `cross_entropy_loss(input, target)`

Cross-entropy for classification. Accepts either one-hot targets or class indices:

```python
from minitorch import cross_entropy_loss, Tensor

# with one-hot targets
target_oh = Tensor([[1, 0, 0], [0, 1, 0]])
loss = cross_entropy_loss(logits, target_oh)

# with class indices
target_idx = Tensor([0, 1])
loss = cross_entropy_loss(logits, target_idx)
```

Uses log-softmax internally for numerical stability.

## `bce_loss(input, target)`

Binary cross-entropy for binary classification. Input should be probabilities (after sigmoid), not raw logits:

```python
from minitorch import bce_loss, Tensor
import minitorch.functional as F

logits = model(x)
probs = F.sigmoid(logits)
loss = bce_loss(probs, labels)  # labels are 0 or 1
```

Formula: `-[y * log(p) + (1-y) * log(1-p)]`, averaged over all elements.
