# Layers

Every layer subclasses `Module`. Activations are also available as plain
functions under `minitorch.functional` (aliased as `F`).

## Linear

::: minitorch.layers.Linear

## Activations

::: minitorch.layers.ReLU

::: minitorch.layers.GELU

::: minitorch.layers.Sigmoid

::: minitorch.layers.Tanh

::: minitorch.layers.Softmax

## Normalization and regularization

::: minitorch.layers.BatchNorm1d

::: minitorch.layers.LayerNorm

::: minitorch.layers.Dropout

## Embedding

::: minitorch.layers.Embedding

## Convolution

::: minitorch.conv.Conv2d

::: minitorch.conv.MaxPool2d

::: minitorch.conv.Flatten

## Functional

::: minitorch.functional
    options:
      members: [relu, gelu, sigmoid, tanh, softmax, log_softmax]
