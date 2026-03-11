import numpy as np
import pytest
from minitorch.tensor import Tensor


class TestProperties:
    def test_shape(self):
        t = Tensor(np.zeros((2, 3)))
        assert t.shape == (2, 3)

    def test_ndim(self):
        t = Tensor(np.zeros((2, 3, 4)))
        assert t.ndim == 3

    def test_dtype(self):
        t = Tensor([1.0])
        assert t.dtype == np.float32


class TestStaticConstructors:
    def test_zeros(self):
        t = Tensor.zeros(2, 3)
        assert t.shape == (2, 3)
        np.testing.assert_allclose(t.data, 0.0)

    def test_ones(self):
        t = Tensor.ones(2, 3)
        assert t.shape == (2, 3)
        np.testing.assert_allclose(t.data, 1.0)

    def test_randn(self):
        t = Tensor.randn(100, 100)
        assert t.shape == (100, 100)
        assert abs(t.data.mean()) < 0.2

    def test_eye(self):
        t = Tensor.eye(3)
        np.testing.assert_allclose(t.data, np.eye(3))


class TestArithmeticOps:
    def test_sub(self):
        a = Tensor(np.array([5.0, 3.0]), requires_grad=True)
        b = Tensor(np.array([2.0, 1.0]), requires_grad=True)
        c = a - b
        c.sum().backward()
        np.testing.assert_allclose(a.grad, np.array([1.0, 1.0]))
        np.testing.assert_allclose(b.grad, np.array([-1.0, -1.0]))

    def test_div(self):
        a = Tensor(np.array([6.0, 8.0]), requires_grad=True)
        b = Tensor(np.array([2.0, 4.0]), requires_grad=True)
        c = a / b
        np.testing.assert_allclose(c.data, [3.0, 2.0])
        c.sum().backward()
        np.testing.assert_allclose(a.grad, [0.5, 0.25])
        np.testing.assert_allclose(b.grad, [-1.5, -0.5])

    def test_pow(self):
        a = Tensor(np.array([2.0, 3.0]), requires_grad=True)
        c = a ** 2
        c.sum().backward()
        np.testing.assert_allclose(a.grad, [4.0, 6.0])

    def test_radd(self):
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        c = 5.0 + a
        c.sum().backward()
        np.testing.assert_allclose(a.grad, [1.0, 1.0])

    def test_rmul(self):
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        c = 3.0 * a
        c.sum().backward()
        np.testing.assert_allclose(a.grad, [3.0, 3.0])

    def test_rsub(self):
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        c = 5.0 - a
        c.sum().backward()
        np.testing.assert_allclose(a.grad, [-1.0, -1.0])

    def test_rtruediv(self):
        a = Tensor(np.array([2.0, 4.0]), requires_grad=True)
        c = 8.0 / a
        np.testing.assert_allclose(c.data, [4.0, 2.0])
        c.sum().backward()
        np.testing.assert_allclose(a.grad, [-2.0, -0.5])


class TestReductionOps:
    def test_mean(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        m = a.mean()
        m.backward()
        np.testing.assert_allclose(a.grad, np.full((2, 2), 0.25))

    def test_mean_axis(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        m = a.mean(axis=1)
        m.sum().backward()
        np.testing.assert_allclose(a.grad, np.full((2, 2), 0.5))

    def test_max(self):
        a = Tensor(np.array([[1.0, 3.0], [4.0, 2.0]]), requires_grad=True)
        m = a.max()
        m.backward()
        expected = np.array([[0.0, 0.0], [1.0, 0.0]])
        np.testing.assert_allclose(a.grad, expected)

    def test_max_axis(self):
        a = Tensor(np.array([[1.0, 3.0], [4.0, 2.0]]), requires_grad=True)
        m = a.max(axis=1)
        m.sum().backward()
        expected = np.array([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_allclose(a.grad, expected)


class TestShapeOps:
    def test_transpose(self):
        a = Tensor(np.arange(6).reshape(2, 3).astype(np.float32), requires_grad=True)
        b = a.transpose()
        assert b.shape == (3, 2)
        b.sum().backward()
        np.testing.assert_allclose(a.grad, np.ones((2, 3)))

    def test_T_property(self):
        a = Tensor(np.arange(6).reshape(2, 3).astype(np.float32))
        assert a.T.shape == (3, 2)

    def test_getitem(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        b = a[0]
        b.sum().backward()
        expected = np.array([[1.0, 1.0], [0.0, 0.0]])
        np.testing.assert_allclose(a.grad, expected)

    def test_getitem_slice(self):
        a = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
        b = a[:, 1:]
        b.sum().backward()
        expected = np.array([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
        np.testing.assert_allclose(a.grad, expected)
