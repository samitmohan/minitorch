"""Causal attention forward as Metal kernels, in three versions that each fix the previous one's bottleneck.

    q (B, H, T, D), k and v (B, Hkv, S, D)  ->  out (B, H, T, D), lse (B, H, T)

Query row i sits at absolute position S - T + i and sees keys 0 .. S - T + i, so
the same kernel serves training (T == S) and decoding against a KV cache (T < S).
Query head h reads kv head h // (H / Hkv): grouped-query attention without
copying K and V. Scale is 1/sqrt(D). `lse` is each row's log-sum-exp of scores,
which is all a backward pass needs to rebuild the softmax.

None of them ever writes the (T, S) score matrix to memory. Each keeps a running
max m, a running sum l, and an output accumulator, and rescales them by
exp(m_old - m_new) whenever a larger score shows up (the online softmax).

v1  one thread per query row, reading K and V straight from device memory. Every
    thread re-reads every key, so it is bound by memory traffic.
v2  a threadgroup of 32 query rows loads each K/V tile into threadgroup memory
    once and all 32 rows reuse it: 32x less device traffic.
v3  v2's tiles, with QK^T and PV done by simdgroup 8x8 matrix multiplies (the
    GPU's matrix hardware) instead of one scalar dot product per thread.
"""
import mlx.core as mx

_SHAPES = """
    const int T = q_shape[2], S = k_shape[2], H = q_shape[1], Hkv = k_shape[1];
    const float scale = rsqrt(float(D));
"""

_V1 = _SHAPES + """
    const uint row = thread_position_in_grid.x;            // over B * H * T query rows
    if (row >= uint(q_shape[0] * H * T)) return;
    const int i = row % T, bh = row / T, b = bh / H, h = bh % H;
    const int kv = (b * Hkv + h / (H / Hkv)) * S * D;

    float qr[D], acc[D];
    for (int d = 0; d < D; d++) { qr[d] = float(q[row * D + d]) * scale; acc[d] = 0.0f; }
    float m = -INFINITY, l = 0.0f;
    for (int j = 0; j <= S - T + i; j++) {
        float s = 0.0f;
        for (int d = 0; d < D; d++) s += qr[d] * float(k[kv + j * D + d]);
        const float m_new = max(m, s), corr = exp(m - m_new), p = exp(s - m_new);
        l = l * corr + p;
        for (int d = 0; d < D; d++) acc[d] = acc[d] * corr + p * float(v[kv + j * D + d]);
        m = m_new;
    }
    for (int d = 0; d < D; d++) out[row * D + d] = InT(acc[d] / l);
    lse[row] = m + log(l);
"""

_V2 = _SHAPES + """
    constexpr int BQ = 32;                                  // query rows (= threads) per threadgroup
    threadgroup float Ks[BK * D], Vs[BK * D];
    const int tid = thread_position_in_threadgroup.x;
    const int n_qblocks = (T + BQ - 1) / BQ;
    const int bh = threadgroup_position_in_grid.x / n_qblocks, qb = threadgroup_position_in_grid.x % n_qblocks;
    const int b = bh / H, h = bh % H, kv = (b * Hkv + h / (H / Hkv)) * S * D;
    const int i = qb * BQ + tid, last = S - T + i;          // this row sees keys 0 .. last
    const int group_last = min(S - T + qb * BQ + BQ - 1, S - 1);

    float qr[D], acc[D];
    for (int d = 0; d < D; d++) { qr[d] = i < T ? float(q[(bh * T + i) * D + d]) * scale : 0.0f; acc[d] = 0.0f; }
    float m = -INFINITY, l = 0.0f;
    for (int j0 = 0; j0 <= group_last; j0 += BK) {
        for (int e = tid; e < BK * D; e += BQ) {           // the 32 threads load the tile together
            const bool in = j0 + e / D < S;
            Ks[e] = in ? float(k[kv + j0 * D + e]) : 0.0f;
            Vs[e] = in ? float(v[kv + j0 * D + e]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s[BK], m_tile = -INFINITY;
        for (int jj = 0; jj < BK; jj++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) dot += qr[d] * Ks[jj * D + d];
            s[jj] = j0 + jj <= last ? dot : -INFINITY;
            m_tile = max(m_tile, s[jj]);
        }
        const float m_new = max(m, m_tile), corr = exp(m - m_new);
        l *= corr;
        for (int d = 0; d < D; d++) acc[d] *= corr;
        for (int jj = 0; jj < BK; jj++) {
            const float p = exp(s[jj] - m_new);
            l += p;
            for (int d = 0; d < D; d++) acc[d] += p * Vs[jj * D + d];
        }
        m = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);  // everyone is done with this tile
    }
    if (i < T) {
        for (int d = 0; d < D; d++) out[(bh * T + i) * D + d] = InT(acc[d] / l);
        lse[bh * T + i] = m + log(l);
    }
"""

# 4 simdgroups x 8 query rows. Within a simdgroup, lanes 4r..4r+3 share row r of the
# softmax, each taking a quarter of the tile's columns.
_V3 = _SHAPES + """
    constexpr int BQ = 32, TILE = (32 * D > 2 * BK * D) ? 32 * D : 2 * BK * D;
    threadgroup float smem[TILE];                           // Q staging, then K/V tiles, then O staging
    threadgroup float Ps[4][8 * BK];                        // each simdgroup's scores, then probabilities
    threadgroup float Dg[4][64];                            // each simdgroup's diag(corr), for rescaling O
    threadgroup float* Ks = smem;
    threadgroup float* Vs = smem + BK * D;
    const int tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const int n_qblocks = (T + BQ - 1) / BQ;
    const int bh = threadgroup_position_in_grid.x / n_qblocks, qb = threadgroup_position_in_grid.x % n_qblocks;
    const int b = bh / H, h = bh % H, kv = (b * Hkv + h / (H / Hkv)) * S * D;
    const int group_last = min(S - T + qb * BQ + BQ - 1, S - 1);
    const int r = lane / 4, part = lane % 4;                // this lane's softmax row and column quarter
    const int i = qb * BQ + sg * 8 + r, last = S - T + i;

    // stage this block's Q (scaled) and load each simdgroup's 8 rows into matrix registers
    for (int e = tid; e < BQ * D; e += 128) {
        const int qi = qb * BQ + e / D;
        smem[e] = qi < T ? float(q[(bh * T + qb * BQ) * D + e]) * scale : 0.0f;
    }
    for (int e = lane; e < 64; e += 32) Dg[sg][e] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 Q[D / 8], O[D / 8];
    for (int dc = 0; dc < D / 8; dc++) {
        simdgroup_load(Q[dc], smem + sg * 8 * D + dc * 8, D);
        O[dc] = simdgroup_float8x8(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY, l = 0.0f;
    for (int j0 = 0; j0 <= group_last; j0 += BK) {
        for (int e = tid; e < BK * D; e += 128) {
            const bool in = j0 + e / D < S;
            Ks[e] = in ? float(k[kv + j0 * D + e]) : 0.0f;
            Vs[e] = in ? float(v[kv + j0 * D + e]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // scores: (8 x D) @ (D x BK), one 8x8 output block per 8 keys
        for (int kb = 0; kb < BK / 8; kb++) {
            simdgroup_float8x8 acc(0.0f), Kt;
            for (int dc = 0; dc < D / 8; dc++) {
                simdgroup_load(Kt, Ks + kb * 8 * D + dc * 8, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, Q[dc], Kt, acc);
            }
            simdgroup_store(acc, Ps[sg] + kb * 8, BK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax on row r, columns part*BK/4 .. (part+1)*BK/4, reduced across the row's 4 lanes
        constexpr int C = BK / 4;
        float m_tile = -INFINITY;
        for (int c = part * C; c < (part + 1) * C; c++) {
            const float s = j0 + c <= last ? Ps[sg][r * BK + c] : -INFINITY;
            Ps[sg][r * BK + c] = s;
            m_tile = max(m_tile, s);
        }
        m_tile = max(m_tile, simd_shuffle_xor(m_tile, 1));
        m_tile = max(m_tile, simd_shuffle_xor(m_tile, 2));
        const float m_new = max(m, m_tile), corr = exp(m - m_new);
        float psum = 0.0f;
        for (int c = part * C; c < (part + 1) * C; c++) {
            const float p = exp(Ps[sg][r * BK + c] - m_new);
            Ps[sg][r * BK + c] = p;
            psum += p;
        }
        psum += simd_shuffle_xor(psum, 1);
        psum += simd_shuffle_xor(psum, 2);
        l = l * corr + psum;
        m = m_new;
        if (part == 0) Dg[sg][r * 8 + r] = corr;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(corr) @ O + P @ V_tile
        simdgroup_float8x8 Corr, P, Vm;
        simdgroup_load(Corr, Dg[sg], 8);
        for (int dc = 0; dc < D / 8; dc++) {
            simdgroup_multiply(O[dc], Corr, O[dc]);
            for (int kb = 0; kb < BK / 8; kb++) {
                simdgroup_load(P, Ps[sg] + kb * 8, BK);
                simdgroup_load(Vm, Vs + kb * 8 * D + dc * 8, D);
                simdgroup_multiply_accumulate(O[dc], P, Vm, O[dc]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // everyone is done with this tile
    }

    // stage O through threadgroup memory, then each lane writes its row's quarter divided by l
    for (int dc = 0; dc < D / 8; dc++) simdgroup_store(O[dc], smem + sg * 8 * D + dc * 8, D);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (i < T) {
        for (int d = part * (D / 4); d < (part + 1) * (D / 4); d++)
            out[(bh * T + i) * D + d] = InT(smem[(sg * 8 + r) * D + d] / l);
        if (part == 0) lse[bh * T + i] = m + log(l);
    }
"""

_KERNELS = {
    name: mx.fast.metal_kernel(name=f"minitorch_attention_{name}", input_names=["q", "k", "v"],
                               output_names=["out", "lse"], source=src)
    for name, src in [("v1", _V1), ("v2", _V2), ("v3", _V3)]
}


def attention(q, k, v, version="v3"):
    """Causal attention forward. Returns (out, lse); see the module docstring for shapes."""
    B, H, T, D = q.shape
    assert k.shape[1] and H % k.shape[1] == 0 and D % 8 == 0 and D <= 128, (q.shape, k.shape)
    bk = 32 if D <= 64 else 16  # keeps the K/V tiles within 32 KB of threadgroup memory
    if version == "v1":
        grid, group = (B * H * T, 1, 1), (min(256, B * H * T), 1, 1)
    else:
        threads = 32 if version == "v2" else 128
        grid, group = (B * H * -(-T // 32) * threads, 1, 1), (threads, 1, 1)
    return _KERNELS[version](
        inputs=[q, k, v],
        template=[("InT", q.dtype), ("D", D), ("BK", bk)],
        grid=grid,
        threadgroup=group,
        output_shapes=[q.shape, (B, H, T)],
        output_dtypes=[q.dtype, mx.float32],
    )
