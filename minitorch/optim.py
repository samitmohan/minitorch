import math
import numpy as np
from .backend import get_array_module


class Optimizer:
    def __init__(self, params, lr=1e-3, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self.velocities):
            if not p.requires_grad or p.grad is None:
                continue
            grad = p.grad.copy()
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data
            v[:] = self.momentum * v + grad
            p.data -= self.lr * v


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if not p.requires_grad or p.grad is None:
                continue
            xp = get_array_module(p.data)
            grad = p.grad.copy()
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad * grad)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)


def clip_grad_norm(params, max_norm):
    """Clip gradient norm across all parameters (like torch.nn.utils.clip_grad_norm_)."""
    total_norm_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_norm_sq += float((p.grad ** 2).sum())
    total_norm = total_norm_sq ** 0.5
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-12)
        for p in params:
            if p.grad is not None:
                p.grad *= scale
    return total_norm


def clip_grad_value(params, clip_value):
    """Clip all gradients element-wise to [-clip_value, clip_value]."""
    for p in params:
        if p.grad is not None:
            np.clip(p.grad, -clip_value, clip_value, out=p.grad)


# learning rate schedulers


class StepLR:
    """Multiply lr by gamma every step_size epochs."""
    def __init__(self, optimizer, step_size, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.epoch = 0

    def step(self):
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma


class CosineAnnealingLR:
    """Cosine annealing from base lr down to eta_min over T_max epochs."""
    def __init__(self, optimizer, T_max, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self.epoch = 0

    def step(self):
        self.epoch += 1
        self.optimizer.lr = self.eta_min + (self.base_lr - self.eta_min) * \
            (1 + math.cos(math.pi * self.epoch / self.T_max)) / 2
