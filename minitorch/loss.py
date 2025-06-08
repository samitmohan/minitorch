import numpy as np
from .tensor import Tensor

def mse_loss(input: Tensor, target: Tensor) -> Tensor:
    """
    Computes the mean squared error (MSE) loss:
        loss = (1/N) * sum_i (input_i - target_i)^2
    """
    diff = input + (target * -1.0)
    return (diff * diff).sum() * Tensor(1.0 / diff.data.size)

def cross_entropy_loss(input: Tensor, target: Tensor) -> Tensor:
    """
    Computes the mean cross-entropy loss with a numerically stable implementation:
        ℓ = -1/N ∑_i ∑_c y_{ic} * (x_{ic} - log ∑_j exp(x_{ij}))
    Assumes `target` is one-hot encoded.
    """
    # input: (batch, classes), target: one-hot
    # 1) Compute max per sample (detached—no grad needed):
    max_logits = Tensor(input.data.max(axis=1, keepdims=True), requires_grad=False)
    # 2) Shift logits so the largest is 0:
    shifted = input + (max_logits * -1.0)
    # 3) exp
    exps     = shifted.exp()
    sum_exps = exps.sum(axis=1, keepdims=True)
    logsum   = sum_exps.log()
    # 4) log‐softmax:
    log_probs = shifted + (logsum * -1.0)

    loss = (-log_probs * target).sum() * Tensor(1.0 / input.data.shape[0])
    return loss
