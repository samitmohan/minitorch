"""A small GPT-style transformer built on the minitorch autograd engine.

Attention, layernorm, and the residual stream are all Tensor ops, so backprop
through them comes from the same engine that powers the MNIST examples.
"""
import numpy as np

from .tensor import Tensor
from .module import Module, Sequential
from .layers import Linear, LayerNorm, Embedding, GELU, Dropout
from . import functional as F


class MultiHeadAttention(Module):
    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.q_proj = Linear(n_embd, n_embd)
        self.k_proj = Linear(n_embd, n_embd)
        self.v_proj = Linear(n_embd, n_embd)
        self.out_proj = Linear(n_embd, n_embd)
        self.attn_drop = Dropout(dropout)

    def _split_heads(self, x, B, T):
        # (B, T, C) -> (B, n_head, T, head_dim)
        return x.reshape(B, T, self.n_head, self.head_dim).transpose(1, 2)

    def forward(self, x):
        B, T, C = x.shape
        q = self._split_heads(self.q_proj(x), B, T)
        k = self._split_heads(self.k_proj(x), B, T)
        v = self._split_heads(self.v_proj(x), B, T)

        scores = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(self.head_dim))

        # causal mask: block attention to future positions
        mask = np.triu(np.ones((T, T), dtype=np.float32), k=1) * -1e9
        scores = scores + Tensor(mask)

        attn = F.softmax(scores, axis=-1)
        attn = self.attn_drop(attn)
        out = attn @ v                              # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, C)  # (B, T, C)
        return self.out_proj(out)


class TransformerBlock(Module):
    """Pre-norm block: x + attn(ln1(x)), then x + mlp(ln2(x))."""
    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        self.ln1 = LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ln2 = LayerNorm(n_embd)
        self.mlp = Sequential(
            Linear(n_embd, 4 * n_embd),
            GELU(),
            Linear(4 * n_embd, n_embd),
            Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(Module):
    def __init__(self, vocab_size, block_size, n_embd=64, n_head=4, n_layer=2, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = Embedding(vocab_size, n_embd)
        self.pos_emb = Embedding(block_size, n_embd)
        self.drop = Dropout(dropout)
        self.blocks = [TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)]
        self.ln_f = LayerNorm(n_embd)
        self.lm_head = Linear(n_embd, vocab_size)

    def forward(self, idx):
        # idx: int array (B, T)
        idx = np.asarray(idx)
        B, T = idx.shape
        assert T <= self.block_size, f"sequence length {T} exceeds block_size {self.block_size}"
        tok = self.tok_emb(idx)                    # (B, T, C)
        pos = self.pos_emb(np.arange(T))           # (T, C), broadcasts over batch
        x = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)                      # (B, T, vocab)

    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Autoregressively sample tokens. idx: int array (B, T0)."""
        idx = np.asarray(idx)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self.forward(idx_cond).data[:, -1, :] / temperature
            logits = logits - logits.max(axis=-1, keepdims=True)
            probs = np.exp(logits)
            probs = probs / probs.sum(axis=-1, keepdims=True)
            next_ids = np.array([np.random.choice(probs.shape[-1], p=p) for p in probs])
            idx = np.concatenate([idx, next_ids[:, None]], axis=1)
        return idx
