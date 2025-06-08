import numpy as np

class Optimizer:
    def __init__(self, params, lr=1e-3, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self):
        for p in self.params:
            if getattr(p, "grad", None) is not None:
                # reset gradient array
                p.grad = np.zeros_like(p.data)

    def step(self):
        raise NotImplementedError("Must override step() in subclass.")


class SGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum
        # one velocity buffer per parameter
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self.velocities):
            if not getattr(p, "requires_grad", False):
                continue

            grad = p.grad 
            # optional L2 regularization
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data

            # momentum update
            v[:] = self.momentum * v + grad
            p.data -= self.lr * v

        # zeroing grads every step (optional, but common)
        self.zero_grad()


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
            if not getattr(p, "requires_grad", False):
                continue

            grad = p.grad 
            # optional L2 regularization
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data

            # update biased moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad * grad)

            # bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # parameter update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        # zero out gradients after update
        self.zero_grad()