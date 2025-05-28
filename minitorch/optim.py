import numpy as np

class SGD:
    def __init__(self, params, lr=1e-3, momentum=0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self.velocities):
            if not getattr(p, 'requires_grad', False):
                continue
            v[:] = self.momentum * v + p.grad
            p.data -= self.lr * v
            p.grad = np.zeros_like(p.data)

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)

class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if not getattr(p, 'requires_grad', False):
                continue
            grad = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad * grad)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p.grad = np.zeros_like(p.data)

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)
