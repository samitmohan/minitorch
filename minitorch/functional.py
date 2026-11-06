"""Activations as compositions of Tensor ops. None of them define a gradient."""
import math


def relu(x):
    """Rectified linear unit, `max(0, x)`."""
    return x.relu()


def sigmoid(x):
    """Logistic sigmoid, `1 / (1 + e**-x)`."""
    return x.sigmoid()


def tanh(x):
    """Hyperbolic tangent."""
    return x.tanh()


def gelu(x):
    """Gaussian error linear unit (tanh approximation, as in GPT-2)."""
    c = math.sqrt(2.0 / math.pi)
    inner = (x + x ** 3 * 0.044715) * c
    return x * 0.5 * (tanh(inner) + 1.0)


def softmax(x, axis=-1):
    """Softmax over `axis`. Numerically stable via max-subtraction."""
    # softmax is shift-invariant, so the subtracted max needs no gradient
    e = (x - x.max(axis=axis, keepdims=True).detach()).exp()
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    """Log of softmax over `axis`. More stable than `softmax(x).log()`."""
    shifted = x - x.max(axis=axis, keepdims=True).detach()
    return shifted - shifted.exp().sum(axis=axis, keepdims=True).log()


def silu(x):
    """x * sigmoid(x), the gate activation in SwiGLU."""
    return x * x.sigmoid()
