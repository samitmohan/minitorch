import numpy as np
from .tensor import Tensor

def mse_loss(input: Tensor, target: Tensor) -> Tensor:
    diff = input + (target * -1.0)
    return (diff * diff).sum() * Tensor(1.0 / diff.data.size)

def cross_entropy_loss(input: Tensor, target: Tensor) -> Tensor:
    # input: (batch, classes), target: one-hot or probabilities
    # compute log-softmax
    exps = np.exp(input.data)
    sum_exps = Tensor(exps.sum(axis=1, keepdims=True), requires_grad=False)
    log_prob = Tensor(input.data, requires_grad=False) + Tensor(np.log(1.0 / sum_exps.data), requires_grad=False)
    loss = (-log_prob * target).sum() * Tensor(1.0 / input.data.shape[0])
    return loss
