import numpy as np
from .tensor import Tensor, _accum_grad
from .module import Module


def im2col(input_data, kh, kw, stride, padding):
    N, C, H, W = input_data.shape
    if padding > 0:
        input_data = np.pad(input_data, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    _, _, H_pad, W_pad = input_data.shape
    OH = (H_pad - kh) // stride + 1
    OW = (W_pad - kw) // stride + 1

    cols = np.zeros((N, C, kh, kw, OH, OW), dtype=input_data.dtype)
    for y in range(kh):
        y_max = y + stride * OH
        for x in range(kw):
            x_max = x + stride * OW
            cols[:, :, y, x, :, :] = input_data[:, :, y:y_max:stride, x:x_max:stride]

    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, -1)


def col2im(cols, input_shape, kh, kw, stride, padding):
    N, C, H, W = input_shape
    H_pad = H + 2 * padding
    W_pad = W + 2 * padding
    OH = (H_pad - kh) // stride + 1
    OW = (W_pad - kw) // stride + 1

    cols = cols.reshape(N, OH, OW, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H_pad, W_pad), dtype=cols.dtype)

    for y in range(kh):
        y_max = y + stride * OH
        for x in range(kw):
            x_max = x + stride * OW
            img[:, :, y:y_max:stride, x:x_max:stride] += cols[:, :, y, x, :, :]

    if padding > 0:
        return img[:, :, padding:-padding, padding:-padding]
    return img


class Conv2d(Module):
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
        N, C, H, W = x.data.shape
        kh, kw = self.kernel_size
        OH = (H + 2 * self.padding - kh) // self.stride + 1
        OW = (W + 2 * self.padding - kw) // self.stride + 1

        cols = im2col(x.data, kh, kw, self.stride, self.padding)
        out_data = cols @ self.weight.data.T + self.bias.data
        out_data = out_data.reshape(N, OH, OW, self.out_channels).transpose(0, 3, 1, 2)
        out = Tensor(out_data, requires_grad=x.requires_grad or self.weight.requires_grad)

        def _backward():
            dout = out.grad.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

            if self.weight.requires_grad:
                _accum_grad(self.weight, dout.T @ cols)

            if self.bias.requires_grad:
                _accum_grad(self.bias, dout.sum(axis=0))

            if x.requires_grad:
                dcols = dout @ self.weight.data
                dx = col2im(dcols, x.data.shape, kh, kw, self.stride, self.padding)
                _accum_grad(x, dx)

        out._backward = _backward
        out._prev = {x, self.weight, self.bias}
        out._op = 'conv2d'
        return out


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size[0]

    def forward(self, x):
        assert x.data.ndim == 4, f"MaxPool2d expects 4D input (N,C,H,W), got {x.data.ndim}D"
        N, C, H, W = x.data.shape
        kh, kw = self.kernel_size
        s = self.stride
        OH = (H - kh) // s + 1
        OW = (W - kw) // s + 1

        # vectorized: use as_strided to extract all patches at once
        strides = x.data.strides
        patches = np.lib.stride_tricks.as_strided(
            x.data,
            shape=(N, C, OH, OW, kh, kw),
            strides=(strides[0], strides[1], strides[2] * s, strides[3] * s, strides[2], strides[3])
        )
        out_data = patches.max(axis=(4, 5))
        out = Tensor(out_data, requires_grad=x.requires_grad)

        # stash patches for backward
        saved_patches = patches.copy()

        def _backward():
            if x.requires_grad:
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                max_vals = out.data[:, :, :, :, None, None]
                mask = (saved_patches == max_vals).astype(np.float32)
                mask_sum = mask.sum(axis=(4, 5), keepdims=True)
                mask = mask / np.maximum(mask_sum, 1.0)
                weighted = mask * out.grad[:, :, :, :, None, None]
                # scatter back
                for i in range(kh):
                    for j in range(kw):
                        x.grad[:, :, i:i+s*OH:s, j:j+s*OW:s] += weighted[:, :, :, :, i, j]

        out._backward = _backward
        out._prev = {x}
        out._op = 'maxpool2d'
        return out


class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size = x.data.shape[0]
        return x.reshape(batch_size, -1)
