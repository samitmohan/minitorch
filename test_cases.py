import torch
from tinygrad.engine import Value


def test_check():
    a = Value(-4.0)
    b = 2 * a + 2 + a
    c = b.tanh() + b * a
    d = (b * b).tanh()
    e = d + c + c * a
    e.backward()
    x_tinygrad, y_tinygrad = a, e

    # Doing the same with pytorch
    a = torch.Tensor([-4.0]).double()
    a.requires_grad = True
    b = 2 * a + 2 + a
    c = b.tanh() + b * a
    d = (b * b).tanh()
    e = d + c + c * a
    e.backward()
    x_pytorch, y_pytorch = a, e

    # forward pass
    assert y_tinygrad.data == y_pytorch.data.item()
    # backward pass
    assert x_tinygrad.grad == x_pytorch.grad.item()


print(test_check())
