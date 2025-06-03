import sys
import numpy as np

# Prepend project root to path so `minitorch` imports work
sys.path.insert(0, "")

from minitorch.tensor import Tensor

def test_autograd():
    # Test simple scalar addition/multiplication
    x = Tensor(np.array(3.0), requires_grad=True)
    y = Tensor(np.array(4.0), requires_grad=True)
    c = x * y + x
    c.backward()
    assert x.grad == 4.0 + 1.0  # dc/dx = y + 1
    assert y.grad == 3.0        # dc/dy = x

    # Test broadcasting add
    a = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    b = Tensor(np.array([[1.0], [1.0]]), requires_grad=True)
    d = a + b  # shapes: (2,) + (2,1) → (2,2)
    z = d.sum()
    z.backward()
    # z = (a[0]+b[0,0]) + (a[1]+b[0,0]) + (a[0]+b[1,0]) + (a[1]+b[1,0])
    # dz/da = 2 for each element; dz/db = 2 for each element
    np.testing.assert_allclose(a.grad, np.array([2.0, 2.0]))
    np.testing.assert_allclose(b.grad, np.array([[2.0], [2.0]]))

    print("Test passed!")

if __name__ == "__main__":
    test_autograd()