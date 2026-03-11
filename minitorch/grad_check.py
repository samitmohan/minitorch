import numpy as np
from .tensor import Tensor


def numerical_gradient(f, inputs, eps=1e-5):
    grads = []
    # temporarily convert all inputs to float64 for precision
    orig_data = [inp.data.copy() for inp in inputs]
    for inp in inputs:
        inp.data = inp.data.astype(np.float64)
    # also stash float64 copies for perturbation
    f64_data = [inp.data.copy() for inp in inputs]

    for k, inp in enumerate(inputs):
        grad = np.zeros(inp.data.shape, dtype=np.float64)
        it = np.nditer(f64_data[k], flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            old_val = f64_data[k][idx]

            inp.data = f64_data[k].copy()
            inp.data[idx] = old_val + eps
            loss_plus = float(f().data)

            inp.data = f64_data[k].copy()
            inp.data[idx] = old_val - eps
            loss_minus = float(f().data)

            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            it.iternext()
        inp.data = f64_data[k].copy()
        grads.append(grad.astype(np.float32))

    # restore original float32 data
    for inp, od in zip(inputs, orig_data):
        inp.data = od
    return grads


def check_gradient(f, inputs, eps=1e-5, atol=1e-4, rtol=1e-3):
    for inp in inputs:
        inp.zero_grad()
    loss = f()
    loss.backward()
    analytic = [inp.grad.copy() for inp in inputs]

    numerical = numerical_gradient(f, inputs, eps)

    for i, (a, n) in enumerate(zip(analytic, numerical)):
        if not np.allclose(a, n, atol=atol, rtol=rtol):
            max_diff = np.max(np.abs(a - n))
            raise AssertionError(
                f"Gradient check failed for input {i}: max diff = {max_diff}\n"
                f"Analytic:\n{a}\nNumerical:\n{n}"
            )
    return True
