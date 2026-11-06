"""GPT-2 in minitorch: OpenAI's 124M-parameter model from 2019, loaded from Hugging Face.

    uv run python examples/gpt2.py
    uv run python examples/gpt2.py --device metal --prompt "The meaning of life is"

One file, like tinygrad's examples/gpt2.py. GPT-2 predates the pieces TransformerLM uses
(RMSNorm, RoPE, SwiGLU), so it gets its own small model: learned position embeddings,
LayerNorm, a bias on every projection, and a GELU feed-forward. Module names follow the
checkpoint's, so loading is a name lookup. tests/gpt2.py checks it against transformers.
Weights: huggingface_hub.snapshot_download("openai-community/gpt2", local_dir="data/gpt2").
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from minitorch import no_grad
from minitorch.attention import KVCache, scaled_dot_product_attention
from minitorch.functional import gelu
from minitorch.layers import Embedding, LayerNorm, Linear
from minitorch.module import Module
from minitorch.sampling import sample_on_device
from minitorch.state import load_safetensors
from minitorch.tokenizer import Tokenizer


class Attention(Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.c_attn = Linear(d, 3 * d)  # queries, keys, and values from one matrix
        self.c_proj = Linear(d, d)

    def forward(self, x, cache=None):
        B, T, C = x.shape
        qkv = self.c_attn(x).reshape(B, T, 3, self.n_heads, C // self.n_heads)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))  # each (B, heads, T, head width)
        if cache is not None:
            k, v = cache.update(k, v)
        out = scaled_dot_product_attention(q, k, v)
        return self.c_proj(out.transpose(1, 2).reshape(B, T, C))


class MLP(Module):
    def __init__(self, d):
        super().__init__()
        self.c_fc = Linear(d, 4 * d)
        self.c_proj = Linear(4 * d, d)

    def forward(self, x):
        return self.c_proj(gelu(self.c_fc(x)))


class Block(Module):
    def __init__(self, d, n_heads, eps):
        super().__init__()
        self.ln_1 = LayerNorm(d, eps)
        self.attn = Attention(d, n_heads)
        self.ln_2 = LayerNorm(d, eps)
        self.mlp = MLP(d)

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln_1(x), cache)
        return x + self.mlp(self.ln_2(x))


class GPT2(Module):
    def __init__(self, vocab_size, n_positions, d, n_layers, n_heads, eps):
        super().__init__()
        self.wte = Embedding(vocab_size, d)
        self.wpe = Embedding(n_positions, d)
        self.h = [Block(d, n_heads, eps) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d, eps)

    def forward(self, idx, caches=None):
        """idx (B, T) -> logits (B, T, vocab). The output layer reuses the token embedding table."""
        start = 0 if caches is None else len(caches[0])
        x = self.wte(idx) + self.wpe(np.arange(start, start + idx.shape[1]))
        for i, block in enumerate(self.h):
            x = block(x, None if caches is None else caches[i])
        return self.ln_f(x) @ self.wte.weight.T


def load_gpt2(path):
    """GPT-2 from a Hugging Face directory; refuses to run if any name or shape differs from the checkpoint's."""
    c = json.loads((Path(path) / "config.json").read_text())
    model = GPT2(c["vocab_size"], c["n_positions"], c["n_embd"], c["n_layer"], c["n_head"], c["layer_norm_epsilon"])
    # h.<i>.attn.bias (and masked_bias in older files) is the causal mask, saved as a buffer, not a weight
    weights = {k: v for k, v in load_safetensors(Path(path) / "model.safetensors").items()
               if not re.fullmatch(r"h\.\d+\.attn\.(masked_)?bias", k)}
    params = dict(model.named_parameters())
    hf_name = {name: name.replace(".gamma", ".weight").replace(".beta", ".bias") for name in params}  # LayerNorm's names
    assert sorted(hf_name.values()) == sorted(weights), sorted(set(weights) ^ set(hf_name.values()))
    for name, p in params.items():
        w = weights[hf_name[name]]
        assert w.shape == p.shape, (name, w.shape, p.shape)
        p.data = w
    return model


def load_tokenizer(path):
    return Tokenizer.from_files(Path(path) / "vocab.json", Path(path) / "merges.txt", ["<|endoftext|>"])


def generate(model, ids, max_new, temperature, eos):
    """ids extended by up to max_new sampled tokens, stopping at eos. temperature=0 is greedy."""
    assert len(ids) + max_new <= model.wpe.weight.shape[0], "GPT-2 sees at most 1,024 positions"
    caches = [KVCache() for _ in model.h]
    out = []
    with no_grad():
        logits = model(np.array([ids]), caches)[:, -1]
        for _ in range(max_new):
            token = int(sample_on_device(logits, temperature)[0])
            if token == eos:
                break
            out.append(token)
            logits = model(np.array([[token]]), caches)[:, -1]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/gpt2")
    p.add_argument("--device", default="cpu", choices=["cpu", "metal", "cuda"])
    p.add_argument("--prompt", default="The meaning of life is")
    p.add_argument("--max-new", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)
    tok = load_tokenizer(args.model)
    model = load_gpt2(args.model).to(args.device)
    ids = tok.encode(args.prompt) or [tok.ids[b"<|endoftext|>"]]  # an empty prompt starts from end-of-text
    print(args.prompt + tok.decode(generate(model, ids, args.max_new, args.temperature, tok.ids[b"<|endoftext|>"])))


if __name__ == "__main__":
    main()
