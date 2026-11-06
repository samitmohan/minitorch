import math

from . import backend


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

    def _realize(self, *state):
        # lazy backends (MLX) compute the new weights here, so the next step's graph starts from arrays, not this step's ops
        backend.get(self.params[0].device).realize(*[p.data for p in self.params], *state)


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum and weight decay."""
    def __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum
        self.velocities = [0.0] * len(self.params)  # a scalar 0 broadcasts on any backend

    def step(self):
        for i, p in enumerate(self.params):
            if not p.requires_grad or p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data
            self.velocities[i] = self.momentum * self.velocities[i] + grad
            p.data = p.data - self.lr * self.velocities[i]
        self._realize(*self.velocities)


class Adam(Optimizer):
    """Adam optimizer with bias-corrected first and second moment estimates."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [0.0] * len(self.params)  # a scalar 0 broadcasts on any backend
        self.v = [0.0] * len(self.params)

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if not p.requires_grad or p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad * grad)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data = p.data - self.lr * m_hat / (v_hat ** 0.5 + self.eps)
        self._realize(*self.m, *self.v)


class AdamW(Adam):
    """Adam with decoupled weight decay: shrink the weights directly instead of adding wd * p to the gradient."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr, betas, eps, weight_decay=0.0)
        self.decay = weight_decay

    def step(self):
        for p in self.params:
            if p.requires_grad and p.grad is not None:
                p.data = p.data * (1 - self.lr * self.decay)
        super().step()


def cosine_lr(step, max_lr, min_lr, warmup_steps, total_steps):
    """Linear warmup to max_lr, cosine decay to min_lr at total_steps, then flat."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)  # warmup == total: the last step is the peak
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)


def clip_grad_norm(params, max_norm):
    """Clip gradient norm across all parameters (like torch.nn.utils.clip_grad_norm_)."""
    # one sum on the device and one transfer, instead of a host sync per parameter
    total_norm = float(sum((p.grad * p.grad).sum() for p in params if p.grad is not None)) ** 0.5
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-12)
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * scale
    return total_norm


class LossScaler:
    """Dynamic loss scaling, for float16 multiplies (autocast("float16")).

    A gradient below float16's smallest value (about 6e-8) becomes zero. Multiplying the loss by a
    large scale multiplies every gradient too, which keeps them in range; unscale() divides them
    back before the optimizer sees them. When a gradient overflows instead, the step is skipped and
    the scale halves; after `interval` clean steps it doubles again.
    """
    def __init__(self, scale=2.0 ** 15, interval=1000):
        self.scale, self.interval, self.clean = scale, interval, 0

    def unscale(self, params):
        """Divide every gradient by the scale. Returns False, and halves the scale, if any of them overflowed."""
        total = float(sum((p.grad * p.grad).sum() for p in params if p.grad is not None))
        if not math.isfinite(total):
            self.scale, self.clean = self.scale / 2, 0
            return False
        for p in params:
            if p.grad is not None:
                p.grad = p.grad / self.scale
        self.clean += 1
        if self.clean % self.interval == 0:
            self.scale *= 2
        return True
