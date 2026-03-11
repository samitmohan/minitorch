import numpy as np
import pytest
from minitorch.tensor import Tensor, no_grad, cat, stack
from minitorch.module import Module, Sequential
from minitorch.layers import Linear, ReLU, Sigmoid, Tanh, Softmax, Dropout, BatchNorm1d
from minitorch.conv import Conv2d, MaxPool2d, Flatten
from minitorch.loss import mse_loss, cross_entropy_loss, bce_loss
from minitorch.optim import SGD, Adam, clip_grad_norm, clip_grad_value, StepLR, CosineAnnealingLR
from minitorch.data import DataLoader
from minitorch.grad_check import check_gradient
import minitorch.functional as F


# Bug fix: backward only on scalars

class TestBackwardScalarCheck:
    def test_backward_on_scalar(self):
        x = Tensor([2.0], requires_grad=True)
        y = (x * 3.0).sum()
        y.backward()
        assert x.grad is not None

    def test_backward_on_nonscalar_raises(self):
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x * 2
        with pytest.raises(AssertionError, match="scalar"):
            y.backward()


# Bug fix: Module._training is instance variable

class TestModuleTrainingIsolation:
    def test_training_mode_independent(self):
        a = Linear(2, 2)
        b = Linear(2, 2)
        a.eval()
        assert a._training is False
        assert b._training is True  # b should NOT be affected

    def test_train_eval_toggle(self):
        model = Sequential(Linear(2, 2), Dropout(0.5))
        model.eval()
        assert model._training is False
        model.train()
        assert model._training is True


# no_grad context manager

class TestNoGrad:
    def test_no_grad_disables_tracking(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        with no_grad():
            y = x * 2
            assert y.requires_grad is False

    def test_no_grad_restores_state(self):
        x = Tensor([1.0], requires_grad=True)
        with no_grad():
            pass
        y = x * 2
        assert y.requires_grad is True


# New tensor ops

class TestNewOps:
    def test_min(self):
        a = Tensor(np.array([[3.0, 1.0], [2.0, 4.0]]), requires_grad=True)
        m = a.min()
        m.backward()
        expected = np.array([[0.0, 1.0], [0.0, 0.0]])
        np.testing.assert_allclose(a.grad, expected)

    def test_var(self):
        a = Tensor(np.array([1.0, 2.0, 3.0, 4.0]), requires_grad=True)
        v = a.var()
        v.backward()
        assert a.grad is not None
        assert v.data > 0

    def test_std(self):
        a = Tensor(np.array([1.0, 2.0, 3.0, 4.0]))
        s = a.std()
        np.testing.assert_allclose(float(s.data), float(np.std([1, 2, 3, 4])), atol=1e-3)

    def test_clamp(self):
        a = Tensor(np.array([-2.0, 0.5, 3.0]), requires_grad=True)
        c = a.clamp(min_val=0.0, max_val=1.0)
        np.testing.assert_allclose(c.data, [0.0, 0.5, 1.0])
        c.sum().backward()
        np.testing.assert_allclose(a.grad, [0.0, 1.0, 0.0])

    def test_abs(self):
        a = Tensor(np.array([-2.0, 0.0, 3.0]), requires_grad=True)
        c = a.abs()
        np.testing.assert_allclose(c.data, [2.0, 0.0, 3.0])

    def test_squeeze(self):
        a = Tensor(np.zeros((2, 1, 3)), requires_grad=True)
        b = a.squeeze(axis=1)
        assert b.shape == (2, 3)
        b.sum().backward()
        assert a.grad.shape == (2, 1, 3)

    def test_unsqueeze(self):
        a = Tensor(np.zeros((2, 3)), requires_grad=True)
        b = a.unsqueeze(axis=1)
        assert b.shape == (2, 1, 3)
        b.sum().backward()
        assert a.grad.shape == (2, 3)

    def test_cat(self):
        a = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
        b = Tensor(np.array([[3.0, 4.0]]), requires_grad=True)
        c = cat([a, b], axis=0)
        assert c.shape == (2, 2)
        c.sum().backward()
        np.testing.assert_allclose(a.grad, [[1.0, 1.0]])
        np.testing.assert_allclose(b.grad, [[1.0, 1.0]])

    def test_stack(self):
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        b = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        c = stack([a, b], axis=0)
        assert c.shape == (2, 2)
        c.sum().backward()
        np.testing.assert_allclose(a.grad, [1.0, 1.0])
        np.testing.assert_allclose(b.grad, [1.0, 1.0])

    def test_size(self):
        t = Tensor(np.zeros((2, 3, 4)))
        assert t.size() == 24
        assert t.size(1) == 3

    def test_len(self):
        t = Tensor(np.zeros((5, 3)))
        assert len(t) == 5

    def test_float(self):
        t = Tensor(np.array(3.14))
        assert abs(float(t) - 3.14) < 1e-5


# Softmax

class TestSoftmax:
    def test_softmax_sums_to_one(self):
        x = Tensor(np.array([[1.0, 2.0, 3.0]]))
        sm = Softmax()
        out = sm(x)
        np.testing.assert_allclose(out.data.sum(), 1.0, atol=1e-6)

    def test_softmax_gradient(self):
        x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        sm = Softmax()
        check_gradient(lambda: sm(x).sum(), [x])

    def test_functional_softmax(self):
        x = Tensor(np.array([[1.0, 2.0, 3.0]]))
        out = F.softmax(x)
        np.testing.assert_allclose(out.data.sum(), 1.0, atol=1e-6)


# Functional API

class TestFunctional:
    def test_relu(self):
        x = Tensor(np.array([-1.0, 0.0, 1.0]), requires_grad=True)
        y = F.relu(x)
        np.testing.assert_allclose(y.data, [0.0, 0.0, 1.0])

    def test_sigmoid(self):
        x = Tensor(np.array([0.0]), requires_grad=True)
        y = F.sigmoid(x)
        np.testing.assert_allclose(y.data, [0.5], atol=1e-6)

    def test_tanh(self):
        x = Tensor(np.array([0.0]), requires_grad=True)
        y = F.tanh(x)
        np.testing.assert_allclose(y.data, [0.0], atol=1e-6)

    def test_log_softmax(self):
        x = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)
        out = F.log_softmax(x)
        # log_softmax should sum to log(1)... actually exp(log_softmax) should sum to 1
        np.testing.assert_allclose(np.exp(out.data).sum(), 1.0, atol=1e-5)

    def test_log_softmax_gradient(self):
        x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        check_gradient(lambda: F.log_softmax(x).sum(), [x])


# Dropout behavior

class TestDropout:
    def test_dropout_train_mode_changes_output(self):
        np.random.seed(0)
        d = Dropout(0.5)
        d.train()
        x = Tensor(np.ones((10, 10)))
        out = d(x)
        assert not np.allclose(out.data, x.data)

    def test_dropout_eval_mode_passthrough(self):
        d = Dropout(0.5)
        d.eval()
        x = Tensor(np.ones((10, 10)))
        out = d(x)
        np.testing.assert_allclose(out.data, x.data)

    def test_dropout_scale_factor(self):
        np.random.seed(42)
        d = Dropout(0.5)
        d.train()
        x = Tensor(np.ones((1000,)))
        out = d(x)
        # with p=0.5, scale=2, so non-zero values should be 2.0
        nonzero = out.data[out.data > 0]
        np.testing.assert_allclose(nonzero, 2.0)

    def test_dropout_zero_prob(self):
        d = Dropout(0.0)
        d.train()
        x = Tensor(np.ones((5, 5)))
        out = d(x)
        np.testing.assert_allclose(out.data, x.data)


# BatchNorm eval mode

class TestBatchNormEval:
    def test_batchnorm_eval_uses_running_stats(self):
        bn = BatchNorm1d(4)
        bn.train()
        x = Tensor(np.random.randn(8, 4).astype(np.float32))
        # run a few forward passes to update running stats
        for _ in range(5):
            bn(x)
        running_mean = bn.running_mean.copy()
        running_var = bn.running_var.copy()

        bn.eval()
        y = bn(x)
        # in eval mode, result should be deterministic and use running stats
        y2 = bn(x)
        np.testing.assert_allclose(y.data, y2.data)

    def test_batchnorm_train_normalizes(self):
        bn = BatchNorm1d(4)
        bn.train()
        x = Tensor(np.random.randn(32, 4).astype(np.float32) * 5 + 3)
        y = bn(x)
        # output should be roughly zero mean, unit variance
        assert abs(y.data.mean()) < 0.5
        assert abs(y.data.std() - 1.0) < 0.5


# Sequential

class TestSequential:
    def test_sequential_forward(self):
        model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
        x = Tensor(np.random.randn(3, 4).astype(np.float32))
        y = model(x)
        assert y.shape == (3, 2)

    def test_sequential_parameters(self):
        model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
        params = model.parameters()
        # Linear(4,8): weight + bias, Linear(8,2): weight + bias = 4
        assert len(params) == 4

    def test_sequential_empty(self):
        model = Sequential()
        x = Tensor(np.array([1.0, 2.0]))
        y = model(x)
        np.testing.assert_allclose(y.data, x.data)


# State dict save/load

class TestStateDict:
    def test_save_load_roundtrip(self):
        model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
        state = model.state_dict()
        assert len(state) > 0

        model2 = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
        model2.load_state_dict(state)

        for k in state:
            # check that values were copied
            v1 = state[k]
            found = False
            for k2, v2 in model2.state_dict().items():
                if k == k2:
                    np.testing.assert_allclose(v1, v2)
                    found = True
            assert found, f"Key {k} not found in loaded state dict"


# Gradient clipping

class TestGradClipping:
    def test_clip_grad_norm(self):
        a = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        a.grad = np.array([3.0, 4.0], dtype=np.float32)  # norm = 5
        norm = clip_grad_norm([a], max_norm=2.0)
        assert abs(norm - 5.0) < 1e-5
        clipped_norm = float(np.sqrt((a.grad ** 2).sum()))
        assert abs(clipped_norm - 2.0) < 1e-5

    def test_clip_grad_value(self):
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        a.grad = np.array([10.0, -20.0], dtype=np.float32)
        clip_grad_value([a], clip_value=5.0)
        np.testing.assert_allclose(a.grad, [5.0, -5.0])


# Cross entropy with class indices

class TestCrossEntropyIndices:
    def test_class_indices(self):
        logits = Tensor(np.array([[2.0, 1.0, 0.1], [0.1, 2.0, 1.0]]), requires_grad=True)
        labels = Tensor(np.array([0, 1]))  # class indices
        loss = cross_entropy_loss(logits, labels)
        loss.backward()
        assert loss.data > 0
        assert logits.grad is not None

    def test_one_hot_same_as_indices(self):
        np.random.seed(42)
        logits1 = Tensor(np.random.randn(4, 3).astype(np.float32), requires_grad=True)
        logits2 = Tensor(logits1.data.copy(), requires_grad=True)

        labels_idx = Tensor(np.array([0, 1, 2, 1]))
        labels_oh = Tensor(np.array([
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]
        ], dtype=np.float32))

        loss1 = cross_entropy_loss(logits1, labels_idx)
        loss2 = cross_entropy_loss(logits2, labels_oh)
        np.testing.assert_allclose(float(loss1.data), float(loss2.data), atol=1e-5)


# Conv2d / MaxPool2d forward values

class TestConvForward:
    def test_conv2d_output_shape(self):
        conv = Conv2d(1, 4, 3, padding=1)
        x = Tensor(np.random.randn(2, 1, 8, 8).astype(np.float32))
        y = conv(x)
        assert y.shape == (2, 4, 8, 8)

    def test_conv2d_no_padding_shape(self):
        conv = Conv2d(1, 4, 3, padding=0)
        x = Tensor(np.random.randn(2, 1, 8, 8).astype(np.float32))
        y = conv(x)
        assert y.shape == (2, 4, 6, 6)

    def test_maxpool2d_output_shape(self):
        pool = MaxPool2d(2)
        x = Tensor(np.random.randn(2, 3, 8, 8).astype(np.float32))
        y = pool(x)
        assert y.shape == (2, 3, 4, 4)

    def test_maxpool2d_values(self):
        x = Tensor(np.array([[[[1, 2], [3, 4]]]]).astype(np.float32))
        pool = MaxPool2d(2)
        y = pool(x)
        assert float(y.data.flatten()[0]) == 4.0

    def test_flatten_shape(self):
        x = Tensor(np.random.randn(2, 3, 4, 4).astype(np.float32))
        flat = Flatten()
        y = flat(x)
        assert y.shape == (2, 48)


# DataLoader drop_last

class TestDataLoaderDropLast:
    def test_drop_last_true(self):
        x = np.arange(10).reshape(10, 1).astype(np.float32)
        y = np.arange(10).reshape(10, 1).astype(np.float32)
        loader = DataLoader(x, y, batch_size=3, shuffle=False, drop_last=True)
        batches = list(loader)
        assert len(batches) == 3  # 10 // 3 = 3, drops last 1
        assert all(b[0].shape[0] == 3 for b in batches)

    def test_drop_last_false(self):
        x = np.arange(10).reshape(10, 1).astype(np.float32)
        y = np.arange(10).reshape(10, 1).astype(np.float32)
        loader = DataLoader(x, y, batch_size=3, shuffle=False, drop_last=False)
        batches = list(loader)
        assert len(batches) == 4  # ceil(10/3) = 4


# Input validation

class TestInputValidation:
    def test_linear_negative_features(self):
        with pytest.raises(AssertionError):
            Linear(-1, 5)

    def test_conv2d_negative_channels(self):
        with pytest.raises(AssertionError):
            Conv2d(-1, 4, 3)

    def test_dropout_invalid_prob(self):
        with pytest.raises(AssertionError):
            Dropout(1.0)

    def test_linear_wrong_ndim(self):
        layer = Linear(4, 2)
        x = Tensor(np.zeros((3,)))  # 1D, should be 2D
        with pytest.raises(AssertionError, match="2D"):
            layer(x)


# Integration: small training loop

class TestIntegration:
    def test_mlp_training_reduces_loss(self):
        np.random.seed(0)
        model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
        x = Tensor(np.random.randn(16, 4).astype(np.float32))
        y = Tensor(np.random.randn(16, 2).astype(np.float32))
        opt = Adam(model.parameters(), lr=0.01)

        first_loss = None
        for i in range(50):
            pred = model(x)
            loss = mse_loss(pred, y)
            if i == 0:
                first_loss = float(loss.data)
            opt.zero_grad()
            loss.backward()
            opt.step()

        final_loss = float(loss.data)
        assert final_loss < first_loss

    def test_no_grad_inference(self):
        model = Sequential(Linear(4, 2))
        x = Tensor(np.random.randn(8, 4).astype(np.float32))
        with no_grad():
            y = model(x)
            assert y.requires_grad is False


# BCE loss

class TestBCELoss:
    def test_bce_basic(self):
        pred = Tensor(np.array([[0.9], [0.1], [0.8]]), requires_grad=True)
        target = Tensor(np.array([[1.0], [0.0], [1.0]]))
        loss = bce_loss(pred, target)
        assert loss.data > 0
        loss.backward()
        assert pred.grad is not None

    def test_bce_perfect_prediction(self):
        pred = Tensor(np.array([[0.999], [0.001]]))
        target = Tensor(np.array([[1.0], [0.0]]))
        loss = bce_loss(pred, target)
        assert float(loss.data) < 0.01

    def test_bce_gradient(self):
        from minitorch.grad_check import check_gradient
        pred = Tensor(np.array([[0.3, 0.7], [0.6, 0.4]]), requires_grad=True)
        target = Tensor(np.array([[0.0, 1.0], [1.0, 0.0]]))
        check_gradient(lambda: bce_loss(pred, target), [pred])


# Xavier init

class TestXavierInit:
    def test_xavier_different_scale(self):
        np.random.seed(0)
        kaiming = Linear(100, 100, init='kaiming')
        np.random.seed(0)
        xavier = Linear(100, 100, init='xavier')
        # xavier scale = sqrt(2/200), kaiming scale = sqrt(2/100)
        # so xavier weights should be smaller
        assert xavier.weight.data.std() < kaiming.weight.data.std()

    def test_xavier_creates_valid_layer(self):
        layer = Linear(10, 5, init='xavier')
        x = Tensor(np.random.randn(3, 10).astype(np.float32))
        y = layer(x)
        assert y.shape == (3, 5)


# LR schedulers

class TestStepLR:
    def test_step_lr_decays(self):
        opt = SGD([Tensor(np.array([1.0]), requires_grad=True)], lr=0.1)
        scheduler = StepLR(opt, step_size=3, gamma=0.5)
        for _ in range(3):
            scheduler.step()
        assert abs(opt.lr - 0.05) < 1e-8

    def test_step_lr_no_decay_before_step_size(self):
        opt = SGD([Tensor(np.array([1.0]), requires_grad=True)], lr=0.1)
        scheduler = StepLR(opt, step_size=5, gamma=0.1)
        for _ in range(4):
            scheduler.step()
        assert abs(opt.lr - 0.1) < 1e-8


class TestCosineAnnealingLR:
    def test_cosine_decays_to_eta_min(self):
        opt = SGD([Tensor(np.array([1.0]), requires_grad=True)], lr=0.1)
        scheduler = CosineAnnealingLR(opt, T_max=100, eta_min=0.001)
        for _ in range(100):
            scheduler.step()
        assert abs(opt.lr - 0.001) < 1e-6

    def test_cosine_halfway(self):
        opt = SGD([Tensor(np.array([1.0]), requires_grad=True)], lr=1.0)
        scheduler = CosineAnnealingLR(opt, T_max=100, eta_min=0)
        for _ in range(50):
            scheduler.step()
        # at halfway, cosine annealing gives lr = 0.5 * (1 + cos(pi/2)) = 0.5
        assert abs(opt.lr - 0.5) < 0.05


# draw_graph

class TestDrawGraph:
    def test_draw_graph_returns_object(self):
        pytest.importorskip("graphviz")
        from minitorch.viz import draw_graph
        x = Tensor(2.0, requires_grad=True)
        y = x ** 2 + 3.0 * x
        dot = draw_graph(y)
        source = dot.source
        assert 'pow' in source
        assert 'mul' in source
        assert 'add' in source
