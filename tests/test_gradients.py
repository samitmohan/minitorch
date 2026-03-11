import numpy as np
import pytest
from minitorch.tensor import Tensor
from minitorch.grad_check import check_gradient


def make_inputs(*shapes, seed=42):
    rng = np.random.RandomState(seed)
    return [Tensor(rng.randn(*s).astype(np.float32), requires_grad=True) for s in shapes]


class TestElementwiseGradients:
    def test_add(self):
        a, b = make_inputs((3, 4), (3, 4))
        assert check_gradient(lambda: (a + b).sum(), [a, b])

    def test_sub(self):
        a, b = make_inputs((3, 4), (3, 4))
        assert check_gradient(lambda: (a - b).sum(), [a, b])

    def test_mul(self):
        a, b = make_inputs((3, 4), (3, 4))
        assert check_gradient(lambda: (a * b).sum(), [a, b])

    def test_div(self):
        a, b = make_inputs((3, 4), (3, 4))
        b.data = np.abs(b.data) + 0.5  # avoid division by near-zero
        assert check_gradient(lambda: (a / b).sum(), [a, b])

    def test_pow(self):
        a = make_inputs((3, 4))[0]
        a.data = np.abs(a.data) + 0.1  # positive for pow grad
        assert check_gradient(lambda: (a ** 2).sum(), [a])

    def test_exp(self):
        a = make_inputs((3, 4))[0]
        a.data = a.data * 0.5  # keep values moderate
        assert check_gradient(lambda: a.exp().sum(), [a])

    def test_log(self):
        a = make_inputs((3, 4))[0]
        a.data = np.abs(a.data) + 0.1
        assert check_gradient(lambda: a.log().sum(), [a])

    def test_neg(self):
        a = make_inputs((3, 4))[0]
        assert check_gradient(lambda: (-a).sum(), [a])


class TestReductionGradients:
    def test_sum(self):
        a = make_inputs((3, 4))[0]
        assert check_gradient(lambda: a.sum(), [a])

    def test_sum_axis(self):
        a = make_inputs((3, 4))[0]
        assert check_gradient(lambda: a.sum(axis=1).sum(), [a])

    def test_mean(self):
        a = make_inputs((3, 4))[0]
        assert check_gradient(lambda: a.mean(), [a])

    def test_mean_axis(self):
        a = make_inputs((3, 4))[0]
        assert check_gradient(lambda: a.mean(axis=0).sum(), [a])


class TestMatmulGradient:
    def test_matmul(self):
        a, b = make_inputs((3, 4), (4, 5))
        assert check_gradient(lambda: (a @ b).sum(), [a, b])


class TestBroadcastingGradients:
    def test_add_broadcast(self):
        a = make_inputs((3, 4))[0]
        b = make_inputs((1, 4), seed=99)[0]
        assert check_gradient(lambda: (a + b).sum(), [a, b])

    def test_mul_broadcast(self):
        a = make_inputs((3, 4))[0]
        b = make_inputs((4,), seed=99)[0]
        assert check_gradient(lambda: (a * b).sum(), [a, b])


class TestActivationGradients:
    def test_relu(self):
        from minitorch.layers import ReLU
        a = make_inputs((3, 4))[0]
        relu = ReLU()
        assert check_gradient(lambda: relu(a).sum(), [a])

    def test_sigmoid(self):
        from minitorch.layers import Sigmoid
        a = make_inputs((3, 4))[0]
        sig = Sigmoid()
        assert check_gradient(lambda: sig(a).sum(), [a])

    def test_tanh(self):
        from minitorch.layers import Tanh
        a = make_inputs((3, 4))[0]
        tanh = Tanh()
        assert check_gradient(lambda: tanh(a).sum(), [a])


class TestLayerGradients:
    def test_linear(self):
        from minitorch.layers import Linear
        x = make_inputs((4, 5))[0]
        layer = Linear(5, 3)
        assert check_gradient(lambda: layer(x).sum(), [x, layer.weight, layer.bias])

    def test_batchnorm(self):
        from minitorch.layers import BatchNorm1d
        x = make_inputs((8, 4))[0]
        bn = BatchNorm1d(4)
        assert check_gradient(lambda: bn(x).sum(), [x, bn.gamma, bn.beta], atol=1e-3, rtol=1e-2)


class TestConvGradients:
    def test_conv2d(self):
        from minitorch.conv import Conv2d
        x = Tensor(np.random.randn(2, 1, 8, 8).astype(np.float32), requires_grad=True)
        conv = Conv2d(1, 4, 3, padding=1)
        assert check_gradient(lambda: conv(x).sum(), [x, conv.weight, conv.bias])

    def test_maxpool2d(self):
        from minitorch.conv import MaxPool2d
        x = Tensor(np.random.randn(2, 1, 4, 4).astype(np.float32), requires_grad=True)
        pool = MaxPool2d(2)
        assert check_gradient(lambda: pool(x).sum(), [x], atol=1e-3)


class TestLossGradients:
    def test_mse(self):
        from minitorch.loss import mse_loss
        pred = make_inputs((4, 3))[0]
        target = Tensor(np.random.randn(4, 3).astype(np.float32), requires_grad=False)
        assert check_gradient(lambda: mse_loss(pred, target), [pred])

    def test_cross_entropy(self):
        from minitorch.loss import cross_entropy_loss
        logits = make_inputs((4, 5))[0]
        target = np.zeros((4, 5), dtype=np.float32)
        target[np.arange(4), np.random.randint(0, 5, 4)] = 1.0
        target = Tensor(target, requires_grad=False)
        assert check_gradient(lambda: cross_entropy_loss(logits, target), [logits])

    def test_bce(self):
        from minitorch.loss import bce_loss
        # inputs must be in (0,1) range for BCE
        pred_raw = make_inputs((4, 3))[0]
        pred_raw.data = 1.0 / (1.0 + np.exp(-pred_raw.data))  # sigmoid to get probs
        target = Tensor(np.random.randint(0, 2, (4, 3)).astype(np.float32), requires_grad=False)
        assert check_gradient(lambda: bce_loss(pred_raw, target), [pred_raw])
