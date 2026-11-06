"""Matmul against group-quantized weights as one Metal kernel, for decoding (few rows of x).

    x (M, K) @ W.T, with W (N, K) stored packed as quant.py describes   ->   (M, N)

Decode is a matrix-vector product: every weight is read once and used once, so the
kernel's job is to read as few bytes as possible. One simdgroup owns one output column;
its 32 lanes stride over that row's packed words (neighbouring lanes read neighbouring
words, one coalesced load), unpack each word in registers, and meet in a simd_sum.
Within a group, sum_k x_k (s q_k + b) = s * sum_k x_k q_k + b * sum_k x_k, so the scale
and bias are applied once per word instead of once per weight.
"""
import mlx.core as mx

SIMDGROUPS = 8  # output columns per threadgroup

_SOURCE = """
    constexpr int PER_WORD = 32 / BITS, MASK = (1 << BITS) - 1;
    const int M = x_shape[0], K = x_shape[1], N = w_shape[0];
    const int words = K / PER_WORD, groups = K / GROUP;
    const int n = threadgroup_position_in_grid.x * SG + simdgroup_index_in_threadgroup;
    const int lane = thread_index_in_simdgroup;
    if (n >= N) return;                                    // the whole simdgroup leaves together
    for (int m = 0; m < M; m++) {
        float acc = 0.0f;
        for (int wi = lane; wi < words; wi += 32) {
            const uint word = w[n * words + wi];
            const int k0 = wi * PER_WORD, g = k0 / GROUP;
            float xq = 0.0f, xs = 0.0f;
            for (int j = 0; j < PER_WORD; j++) {
                const float xv = float(x[m * K + k0 + j]);
                xq += xv * float((word >> (j * BITS)) & MASK);
                xs += xv;
            }
            acc += float(scales[n * groups + g]) * xq + float(biases[n * groups + g]) * xs;
        }
        acc = simd_sum(acc);
        if (lane == 0) out[m * N + n] = InT(acc);
    }
"""

_KERNEL = mx.fast.metal_kernel(name="minitorch_quant_matmul", input_names=["x", "w", "scales", "biases"],
                               output_names=["out"], source=_SOURCE)


def quant_matmul(x, packed, scales, biases, bits, group):
    M, N = x.shape[0], packed.shape[0]
    return _KERNEL(
        inputs=[x, packed, scales, biases],
        template=[("InT", x.dtype), ("BITS", bits), ("GROUP", group), ("SG", SIMDGROUPS)],
        grid=(-(-N // SIMDGROUPS) * SIMDGROUPS * 32, 1, 1),
        threadgroup=(SIMDGROUPS * 32, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )[0]
