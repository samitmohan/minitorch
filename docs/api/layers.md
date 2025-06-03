# Layers

## `Linear(in_features, out_features, bias=True)`

Fully-connected layer:

```python
from minitorch.layers import Linear
from minitorch import Tensor

layer = Linear(5, 2)
x = Tensor.zeros((3,5), requires_grad=True)
y = layer(x)
```

## `ReLU()`

Activation layer:

```python
from minitorch.layers import ReLU
act = ReLU()
y = act(x)
```

## `BatchNorm1d(num_features)`

Batch normalization for 1D features.
TODO: _backward function