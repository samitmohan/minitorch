# Transformer

MiniTorch's autograd engine can train a GPT-style transformer. Attention,
layernorm, and the residual stream run on the same `Tensor` graph as the MNIST
examples, with no special cases.

## Run it

```bash
uv run python char_transformer.py --iters 2000 --plot loss.png
```

This trains a character-level model on an embedded snippet of *Alice in
Wonderland*, so nothing downloads. A 107k-parameter model drops cross-entropy
from 5.2 to under 0.3 in 2000 numpy-only iterations, then samples text that
echoes the source. It works at the character level, so a few words come out
mangled:

```
Alice taros the field after
it, and fortunately was just in time to seee it pop down a large rabbit-hole under the
hedered tofer feet, for it flashed acrosss
her mind that she had never before seen a r
```

## What it's built from

Each piece is a `Tensor` op, so autograd differentiates it without hand-written
backward code:

- `Embedding`: token and position lookups. Gradients scatter back through
  `Tensor.__getitem__` with `np.add.at`.
- `LayerNorm`: mean and variance over the last dim, written with `mean`, `**`,
  and broadcasting.
- `MultiHeadAttention`: `q @ k.transpose(-2, -1)`, a causal mask, `softmax`,
  then `attn @ v`. This needs the batched-matmul fix so the leading
  `(batch, head)` dims broadcast through `@`.
- `GELU`: the tanh approximation, written with `tanh` and arithmetic.

## The block

Pre-norm, GPT-2 style:

```python
def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x
```

## Attention in full

```python
def forward(self, x):
    B, T, C = x.shape
    q = self._split_heads(self.q_proj(x), B, T)   # (B, n_head, T, head_dim)
    k = self._split_heads(self.k_proj(x), B, T)
    v = self._split_heads(self.v_proj(x), B, T)

    scores = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(self.head_dim))
    mask = np.triu(np.ones((T, T), dtype=np.float32), k=1) * -1e9
    scores = scores + Tensor(mask)                # block the future

    attn = F.softmax(scores, axis=-1)
    out = attn @ v
    out = out.transpose(1, 2).reshape(B, T, C)
    return self.out_proj(out)
```

The engine differentiates matmul, softmax, and transpose, so attention needs no
custom gradient.

## Correctness

Batched matmul, layernorm, softmax, and cross-entropy each have a gradient-parity
test against PyTorch in `tests/test_torch_parity.py`. The transformer has an
overfit test (`tests/test_transformer.py`) that drives loss down on a fixed
batch. For a full training run, `parity_demo.py` trains the same MLP in minitorch
and PyTorch from identical weights and confirms the loss curves stay within 3e-8
of each other.
