# Reading the Code

The whole library is about 1200 lines of NumPy (`python sz.py` prints the
breakdown). It is meant to be read, not just imported. Here is the order I would
read it in.

## 1. `minitorch/tensor.py` (the engine, ~410 lines)

Start here and read top to bottom. This is the whole idea. A `Tensor` wraps a
NumPy array and, for every operation, stores two things: a `_backward` closure
that knows how to push gradient to its inputs, and a `_prev` set of the tensors
that produced it. `backward()` topologically sorts that graph and walks it in
reverse, calling each closure. Once `__add__`, `__mul__`, and `__matmul__` click,
the rest of the ops are variations on the same pattern.

## 2. `minitorch/module.py`

How models are built. A `Module` finds its parameters by walking `__dict__` for
anything that is a `Tensor` or a nested `Module`. Same walk drives `train`/`eval`
and `state_dict`. `Sequential` is just a list of layers.

## 3. `minitorch/layers.py` and `functional.py`

The layers are thin. `ReLU` calls `F.relu`; `Linear` is `x @ W + b`. `LayerNorm`
is written with `mean`, `**`, and broadcasting, so autograd handles its backward
for free. Read `functional.py` alongside to see the activations as plain
functions.

## 4. `minitorch/loss.py` and `optim.py`

Loss functions (`mse_loss`, `cross_entropy_loss`) and the optimizers (`SGD`,
`Adam`). Short and standard once the engine makes sense.

## 5. `minitorch/conv.py`

The one place with a real trick: `im2col` unrolls image patches into columns so a
convolution becomes a single matmul. Worth reading slowly.

## 6. `minitorch/transformer.py`

Attention, layernorm, and a GPT block built entirely from the ops above. Nothing
here defines a custom backward; the engine differentiates all of it.

## Then run it

```bash
uv run python train.py                    # linear regression
uv run python mnist_example.py --model mlp
uv run python char_transformer.py --iters 2000
```
