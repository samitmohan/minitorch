# Layers

All layers inherit from `Module` and implement `forward()`.

## `Linear(in_features, out_features, bias=True, init='kaiming')`

Fully-connected layer. Supports two weight initialization schemes:

- `init='kaiming'` (default) - good for ReLU networks: `scale = sqrt(2 / fan_in)`
- `init='xavier'` - good for sigmoid/tanh networks: `scale = sqrt(2 / (fan_in + fan_out))`

```python
from minitorch import Linear

# default kaiming init (for ReLU nets)
fc = Linear(784, 128)

# xavier init (for sigmoid/tanh nets)
fc = Linear(784, 128, init='xavier')
```

## `ReLU()`

Rectified Linear Unit: `max(0, x)`.

## `Sigmoid()`

Logistic sigmoid: `1 / (1 + exp(-x))`.

## `Tanh()`

Hyperbolic tangent: values in (-1, 1).

## `Softmax(axis=-1)`

Softmax activation with proper Jacobian backward.

```python
from minitorch import Softmax
sm = Softmax()
probs = sm(logits)  # sums to 1 along last axis
```

## `Dropout(p=0.5)`

Randomly zeros elements during training, scales by `1/(1-p)`. Passthrough during eval.

## `BatchNorm1d(num_features, eps=1e-5, momentum=0.1)`

Batch normalization with full backward. Uses running stats in eval mode.

## `Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)`

2D convolution using im2col. Kaiming weight initialization.

## `MaxPool2d(kernel_size, stride=None)`

2D max pooling. Vectorized using `as_strided`.

## `Flatten()`

Flattens spatial dimensions: `(N, C, H, W) -> (N, C*H*W)`.

## Functional API

All activations also available as functions:

```python
import minitorch.functional as F

y = F.relu(x)
y = F.sigmoid(x)
y = F.tanh(x)
y = F.softmax(x, axis=-1)
y = F.log_softmax(x, axis=-1)
```
