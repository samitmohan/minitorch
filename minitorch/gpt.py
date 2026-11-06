"""A modern decoder-only language model: the CS336 assignment 1 architecture.

Pre-norm blocks with RMSNorm, RoPE attention (optionally grouped-query), and a
SwiGLU feed-forward. No biases, untied output head. attn_res=True swaps the residual
sums for Block Attention Residuals (see DepthAttention in layers.py).
"""
import numpy as np

from .attention import Attention, KVCache
from .checkpoint import checkpoint
from .layers import DepthAttention, Embedding, Linear, RMSNorm, SiLUMLP, SwiGLU
from .module import Module
from .sampling import sample, sample_on_device
from .tensor import no_grad


def trunc_normal(shape, std):
    """N(0, std^2) resampled until every value lies within 3 std."""
    x = np.random.randn(*shape)
    while (bad := np.abs(x) > 3).any():
        x[bad] = np.random.randn(bad.sum())
    return (x * std).astype(np.float32)


class Block(Module):
    """norm, rope, and ffn set the CS336 ablations; the defaults are the modern block."""
    def __init__(self, d_model, n_heads, n_kv_heads, d_ff, rope_theta, head_dim=None, qk_norm=False, norm_eps=1e-5,
                 norm="pre", rope=True, ffn="swiglu"):
        super().__init__()
        self.norm = norm
        self.attn_norm = None if norm == "none" else RMSNorm(d_model, norm_eps)
        self.attn = Attention(d_model, n_heads, n_kv_heads, rope_theta, head_dim, qk_norm, norm_eps, rope)
        self.ffn_norm = None if norm == "none" else RMSNorm(d_model, norm_eps)
        self.ffn = SwiGLU(d_model, d_ff) if ffn == "swiglu" else SiLUMLP(d_model, d_ff)

    def forward(self, x, cache=None, pad=None):
        if self.norm == "post":  # the original transformer's order: normalize after each residual add
            x = self.attn_norm(x + self.attn(x, cache, pad))
            return self.ffn_norm(x + self.ffn(x))
        if self.norm == "none":
            x = x + self.attn(x, cache, pad)
            return x + self.ffn(x)
        x = x + self.attn(self.attn_norm(x), cache, pad)
        return x + self.ffn(self.ffn_norm(x))

    def sublayers(self, cache=None, pad=None):
        """The block's attention and feed-forward as two functions of their input, without the residual adds that
        forward() puts around them. Attention Residuals mixes their outputs itself."""
        return [lambda x: self.attn(self.attn_norm(x), cache, pad), lambda x: self.ffn(self.ffn_norm(x))]


class TransformerLM(Module):
    """The CS336 model by default. Qwen3 is the same model with head_dim, qk_norm, norm_eps,
    and tie_embeddings set (see qwen3.py): the output head reuses the embedding table."""
    def __init__(self, vocab_size, context_length, d_model, n_layers, n_heads, d_ff,
                 n_kv_heads=None, rope_theta=10000.0, head_dim=None, qk_norm=False, norm_eps=1e-5, tie_embeddings=False,
                 norm="pre", rope=True, ffn="swiglu", attn_res=False, attn_res_block=8):
        """attn_res=True: Block Attention Residuals. Before every attention and feed-forward, a DepthAttention mixes
        the finished block sums and the current block's partial sum, where a block is attn_res_block sublayers (two
        per Block); one more mixes them before the final norm."""
        super().__init__()
        assert not attn_res or norm == "pre", "Attention Residuals is defined on pre-norm blocks"
        self.context_length = context_length
        self.attn_res_block = attn_res_block
        self.depth = [DepthAttention(d_model, norm_eps) for _ in range(2 * n_layers + 1)] if attn_res else None
        self.embed = Embedding(vocab_size, d_model)
        self.blocks = [Block(d_model, n_heads, n_kv_heads, d_ff, rope_theta, head_dim, qk_norm, norm_eps, norm, rope, ffn)
                       for _ in range(n_layers)]
        self.norm = None if norm == "none" else RMSNorm(d_model, norm_eps)  # "none" removes every RMSNorm, the final one too
        self.head = None if tie_embeddings else Linear(d_model, vocab_size, bias=False)
        self.checkpoint_blocks = False  # True: recompute each block in backward instead of storing its activations

        # CS336 init: linear weights N(0, 2 / (fan_in + fan_out)), embeddings N(0, 1), both truncated at 3 std
        for p in self.parameters():
            if p.ndim == 2:
                p.data = trunc_normal(p.shape, np.sqrt(2.0 / sum(p.shape)))
        self.embed.weight.data = trunc_normal(self.embed.weight.shape, 1.0)

    def forward(self, idx, caches=None, pad=None):
        """idx: int array (B, T) -> logits (B, T, vocab). Pass one KVCache per block to decode incrementally,
        and pad (B,) when rows are left-padded to a common length."""
        return self.project(self.hidden(idx, caches, pad))

    def hidden(self, idx, caches=None, pad=None):
        """The final normalized hidden states (B, T, d_model): the whole model but the output projection."""
        x = self.embed(idx)
        if self.depth is not None:
            x = self.attn_res(x, caches, pad)
            return x if self.norm is None else self.norm(x)
        for i, block in enumerate(self.blocks):
            if self.checkpoint_blocks and caches is None and pad is None:
                x = checkpoint(block, x)
            else:
                x = block(x, None if caches is None else caches[i], pad)
        return x if self.norm is None else self.norm(x)

    def attn_res(self, x, caches=None, pad=None):
        """Block Attention Residuals over the embedding x: the hidden states before the final norm."""
        # shortcut: no activation checkpointing here; add it per block if attn_res ever trains a model too big to fit
        blocks, partial = [x], None  # finished block sums (the embedding first), and the running sum of this block
        layers = [f for i, b in enumerate(self.blocks) for f in b.sublayers(None if caches is None else caches[i], pad)]
        for i, f in enumerate(layers):
            out = f(self.depth[i](blocks if partial is None else blocks + [partial]))
            partial = out if partial is None else partial + out
            if (i + 1) % self.attn_res_block == 0:
                blocks, partial = blocks + [partial], None
        return self.depth[-1](blocks if partial is None else blocks + [partial])

    def project(self, h):
        """Hidden states -> logits, through the output head or, when tied, the embedding table.
        Callers that need only some positions project only those: a row of logits is vocab-size floats."""
        return self.head(h) if self.head is not None else h @ self.embed.weight.T

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, eos_id=None, pad=None):
        """Extend int array idx (B, T0) by up to max_new_tokens, stopping once every row has emitted eos_id.

        Prompts of different lengths share a batch left-padded: pad (B,) counts each row's
        padding, which attention hides. RoPE scores depend only on the distance between two
        positions, so shifting a prompt right changes nothing it computes.
        """
        idx = np.asarray(idx)
        assert idx.shape[1] + max_new_tokens <= self.context_length, "prompt + new tokens exceed context_length"
        caches = [KVCache() for _ in self.blocks]
        done = np.zeros(len(idx), dtype=bool)
        with no_grad():
            logits = self.project(self.hidden(idx, caches, pad)[:, -1])  # prefill: the whole prompt in one pass
            for _ in range(max_new_tokens):
                if top_k is None and top_p is None:
                    next_id = sample_on_device(logits, temperature)
                else:
                    next_id = sample(logits.numpy(), temperature, top_k, top_p)
                idx = np.concatenate([idx, next_id[:, None]], axis=1)
                if eos_id is not None:
                    done |= next_id == eos_id
                    if done.all():
                        break
                logits = self.project(self.hidden(next_id[:, None], caches, pad)[:, -1])  # decode: one new token per pass
        return idx
