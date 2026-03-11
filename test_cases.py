"""Basic autograd sanity checks."""
import numpy as np
from minitorch import Tensor


def test_autograd():
    x = Tensor(np.array(3.0), requires_grad=True)
    y = Tensor(np.array(4.0), requires_grad=True)
    c = x * y + x
    c.backward()
    assert x.grad == 4.0 + 1.0  # dc/dx = y + 1
    assert y.grad == 3.0        # dc/dy = x

    a = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    b = Tensor(np.array([[1.0], [1.0]]), requires_grad=True)
    d = a + b  # shapes: (2,) + (2,1) -> (2,2)
    z = d.sum()
    z.backward()
    np.testing.assert_allclose(a.grad, np.array([2.0, 2.0]))
    np.testing.assert_allclose(b.grad, np.array([[2.0], [2.0]]))

    print("All tests passed!")


if __name__ == "__main__":
    test_autograd()
