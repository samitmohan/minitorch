import numpy as np

from . import functional as F
from .tensor import Tensor


def mse_loss(input, target):
    """Mean squared error between predictions and targets."""
    return ((input - target) ** 2).mean()


def bce_loss(input, target):
    """Binary cross-entropy. Input should be probabilities."""
    p = input.clamp(1e-7, 1 - 1e-7)
    return -(target * p.log() + (1.0 - target) * (1.0 - p).log()).mean()


def cross_entropy_loss(input, target):
    """Softmax cross-entropy over the last axis, averaged over everything else.

    `target` holds class indices (shape `input.shape[:-1]`, as an int array or
    Tensor) or one-hot rows (same shape as `input`).
    """
    log_probs = F.log_softmax(input, axis=-1)
    labels = target.numpy() if isinstance(target, Tensor) else np.asarray(target)
    if labels.shape == input.shape:
        return -(log_probs * target).sum() / (input.data.size // input.shape[-1])
    # pick each row's log-prob at its label (a gather) instead of building a one-hot matrix
    labels = labels.reshape(-1).astype(np.int64)
    log_probs = log_probs.reshape(-1, input.shape[-1])
    return -log_probs[np.arange(len(labels)), labels].mean()
