import numpy as np

from .module import Module
from .tensor import Tensor


class Conv2d(Module):
    """2D convolution over `(N, C, H, W)` input, implemented with im2col."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        assert in_channels > 0 and out_channels > 0, "channels must be positive"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        kh, kw = self.kernel_size
        assert kh > 0 and kw > 0, "kernel_size must be positive"
        fan_in = in_channels * kh * kw
        scale = np.sqrt(2.0 / fan_in)
        self.weight = Tensor(
            (np.random.randn(out_channels, in_channels * kh * kw) * scale).astype(np.float32),
            requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_channels, dtype=np.float32), requires_grad=True)

    def forward(self, x):
        assert x.data.ndim == 4, f"Conv2d expects 4D input (N,C,H,W), got {x.data.ndim}D"
        N, _, H, W = x.data.shape
        kh, kw = self.kernel_size
        OH = (H + 2 * self.padding - kh) // self.stride + 1
        OW = (W + 2 * self.padding - kw) // self.stride + 1

        # every output pixel is one patch row dotted with every filter
        cols = x.im2col(kh, kw, self.stride, self.padding)    # (N*OH*OW, C*kh*kw)
        out = cols @ self.weight.T + self.bias               # (N*OH*OW, out_channels)
        return out.reshape(N, OH, OW, self.out_channels).permute(0, 3, 1, 2)


class MaxPool2d(Module):
    """Max pooling over `(N, C, H, W)` input: im2col, then max over each patch."""
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size[0]

    def forward(self, x):
        assert x.data.ndim == 4, f"MaxPool2d expects 4D input (N,C,H,W), got {x.data.ndim}D"
        N, C, H, W = x.data.shape
        kh, kw = self.kernel_size
        OH = (H - kh) // self.stride + 1
        OW = (W - kw) // self.stride + 1

        cols = x.im2col(kh, kw, self.stride, 0)              # (N*OH*OW, C*kh*kw)
        out = cols.reshape(N, OH, OW, C, kh * kw).max(axis=-1)
        return out.permute(0, 3, 1, 2)


class Flatten(Module):
    """Flatten all dims except the batch dim into one."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size = x.data.shape[0]
        return x.reshape(batch_size, -1)
