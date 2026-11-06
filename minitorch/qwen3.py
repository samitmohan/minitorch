"""Qwen3 in minitorch: TransformerLM with Qwen3's options, loaded from a Hugging Face checkpoint.

    model = load_qwen3("data/Qwen3-0.6B")    # config.json + model.safetensors

Qwen3 differs from the CS336 model in four settings, not in structure: a head_dim
set apart from d_model / n_heads, RMSNorm on each head's queries and keys, an
RMSNorm epsilon of 1e-6, and an output head that reuses the embedding table.
"""
import json
from pathlib import Path

import numpy as np
import regex as re

from .gpt import TransformerLM
from .state import iter_safetensors
from .tokenizer import Tokenizer

# Qwen's pre-tokenizer (GPT-4 style): case-insensitive contractions, one digit per token
PAT = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def config(path):
    """TransformerLM arguments from a Hugging Face Qwen3 config.json."""
    c = json.loads((Path(path) / "config.json").read_text())
    return {"vocab_size": c["vocab_size"], "context_length": c["max_position_embeddings"], "d_model": c["hidden_size"],
            "n_layers": c["num_hidden_layers"], "n_heads": c["num_attention_heads"], "d_ff": c["intermediate_size"],
            "n_kv_heads": c["num_key_value_heads"], "rope_theta": c["rope_theta"], "head_dim": c["head_dim"],
            "qk_norm": True, "norm_eps": c["rms_norm_eps"], "tie_embeddings": c["tie_word_embeddings"]}


def hf_names(n_layers):
    """minitorch parameter name -> (Hugging Face tensor name, transpose?). HF Linear weights are (out, in); minitorch's are (in, out)."""
    names = {"embed.weight": ("model.embed_tokens.weight", False), "norm.weight": ("model.norm.weight", False)}
    for i in range(n_layers):
        mine, hf = f"blocks.{i}.", f"model.layers.{i}."
        names[mine + "attn_norm.weight"] = (hf + "input_layernorm.weight", False)
        names[mine + "ffn_norm.weight"] = (hf + "post_attention_layernorm.weight", False)
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            names[f"{mine}attn.{proj}.weight"] = (f"{hf}self_attn.{proj}.weight", True)
        names[mine + "attn.q_norm.weight"] = (hf + "self_attn.q_norm.weight", False)
        names[mine + "attn.k_norm.weight"] = (hf + "self_attn.k_norm.weight", False)
        for mine_ffn, hf_ffn in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
            names[f"{mine}ffn.{mine_ffn}.weight"] = (f"{hf}mlp.{hf_ffn}.weight", True)
    return names


def load_qwen3(path, context_length=4096):
    cfg = config(path) | {"context_length": context_length}
    # shortcut: the model draws a random init (14.7 of a 19.6 s load) that the checkpoint overwrites; skip it if loads start to hurt
    model = TransformerLM(**cfg)
    params = dict(model.named_parameters())
    names = hf_names(cfg["n_layers"])
    # fail loudly on any name the mapping misses or invents: a skipped tensor would keep its random init
    assert params.keys() == names.keys(), (sorted(params.keys() - names.keys())[:3], sorted(names.keys() - params.keys())[:3])
    unplaced = {hf: (name, transpose) for name, (hf, transpose) in names.items()}
    lm_head = None
    # one tensor at a time, each replacing its random init as it lands: for Qwen3-1.7B, holding every
    # float32 weight beside the init at once would take 16 GB. Large checkpoints split into several files
    for shard in sorted(Path(path).glob("*.safetensors")):
        for hf, w in iter_safetensors(shard):
            hf = hf if hf.startswith(("model.", "lm_head.")) else "model." + hf  # Qwen3-Embedding saves its names without the prefix
            if hf == "lm_head.weight" and cfg["tie_embeddings"]:
                lm_head = w
                continue
            name, transpose = unplaced.pop(hf)  # a KeyError: the checkpoint has a tensor the mapping doesn't know
            w = w.T if transpose else w
            assert w.shape == params[name].shape, (name, w.shape, params[name].shape)
            params[name].data = np.ascontiguousarray(w, dtype=np.float32)
    assert not unplaced, f"the checkpoint lacks {sorted(unplaced)[:3]}"
    if lm_head is not None:
        assert np.array_equal(lm_head, params["embed.weight"].data), "lm_head differs from the tied embedding"
    return model


def load_tokenizer(path):
    """Qwen's vocab.json and merges.txt in minitorch's BPE, with its special tokens at their own ids."""
    added = json.loads((Path(path) / "tokenizer.json").read_text())["added_tokens"]
    tok = Tokenizer.from_files(Path(path) / "vocab.json", Path(path) / "merges.txt", pattern=PAT, nfc=True)
    tok = Tokenizer({**tok.vocab, **{a["id"]: a["content"].encode() for a in added}},
                    [p for p, _ in sorted(tok.ranks.items(), key=lambda kv: kv[1])],
                    [a["content"] for a in added], pattern=PAT, nfc=True)
    assert all(tok.ids[a["content"].encode()] == a["id"] for a in added)
    return tok


_TOOLS_HEADER = ("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                 "You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
_TOOLS_FOOTER = ("\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
                 "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
                 "</tool_call><|im_end|>\n")


def chat_prompt(messages, think=False, tools=None):
    """Qwen3's chat template: [{"role": ..., "content": ...}] -> prompt text that ends where the reply starts.

    think=False closes an empty <think> block, which is how Qwen3 switches its reasoning off. tools lists
    function schemas ({"type": "function", "function": {...}}); Qwen3 calls one by writing
    <tool_call>{"name": ..., "arguments": {...}}</tool_call> and ending its turn, an assistant message
    carries that text back as it was, and a "tool" message carries the result.
    """
    # shortcut: earlier assistant turns keep any <think>...</think> text (HF's template strips it); strip it if a long agent conversation needs the room
    text = ""
    if tools:
        system = messages[0]["content"] + "\n\n" if messages[0]["role"] == "system" else ""
        text = "<|im_start|>system\n" + system + _TOOLS_HEADER + "".join("\n" + json.dumps(t, ensure_ascii=False) for t in tools) + _TOOLS_FOOTER
        messages = messages[1:] if system else messages
    for i, m in enumerate(messages):
        if m["role"] != "tool":
            text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            continue
        # consecutive tool results share one user turn, each in its own <tool_response> block
        if i == 0 or messages[i - 1]["role"] != "tool":
            text += "<|im_start|>user"
        text += f"\n<tool_response>\n{m['content']}\n</tool_response>"
        if i == len(messages) - 1 or messages[i + 1]["role"] != "tool":
            text += "<|im_end|>\n"
    return text + "<|im_start|>assistant\n" + ("" if think else "<think>\n\n</think>\n\n")
