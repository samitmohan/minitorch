import numpy as np
from .tensor import Tensor
from . import functional as F


def mse_loss(input, target):
    return ((input - target) ** 2).mean()


def bce_loss(input, target):
    """Binary cross-entropy. Input should be probabilities."""
    p = input.clamp(1e-7, 1 - 1e-7)
    return -(target * p.log() + (1.0 - target) * (1.0 - p).log()).mean()


def cross_entropy_loss(input, target):
    if target.data.ndim == 1 or (target.data.ndim == 2 and target.data.shape[1] == 1):
        labels = target.data.flatten().astype(np.int64)
        one_hot = np.zeros((input.data.shape[0], input.data.shape[1]), dtype=np.float32)
        one_hot[np.arange(len(labels)), labels] = 1.0
        target = Tensor(one_hot, requires_grad=False)

    log_probs = F.log_softmax(input, axis=1)
    N = input.data.shape[0]
    return (-log_probs * target).sum() / N
