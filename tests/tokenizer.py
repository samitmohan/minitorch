#!/usr/bin/env python3
"""Tokenizer checks: GPT-2 parity with tiktoken, round trips, and the trainer against a naive reference.

    uv run python tests/tokenizer.py
"""
import itertools
import os
import tempfile
from collections import Counter
from pathlib import Path

import tiktoken
from huggingface_hub import hf_hub_download, snapshot_download

from minitorch.tokenizer import PAT, Tokenizer, _merge, train_bpe

EOT = "<|endoftext|>"
CORPUS = [
    "", "Hello, world!", "  leading and trailing spaces  ", "tabs\tand\nnewlines\n\n\nruns",
    "def f(x):\n    return x ** 2  # code", "numbers 12345 3.14159 1e-9 007",
    "unicode: naïve café 東京 😀🚀 ü", "it's they're we'll I've don't",
    f"two docs{EOT}second doc{EOT}", f"{EOT}{EOT}", "a" * 300,
]
TEXT = "Once upon a time, a little dog named Max found a red ball. He liked it! " * 3 + \
    "The cat said: 'I'm hungry.' 123 apples, 45 pears.\n\nThe end.\n"


def naive_train(text, vocab_size):
    """Recount every pair on every merge: slow, obviously correct."""
    words = Counter(m.group() for m in PAT.finditer(text))
    words = {tuple(bytes([b]) for b in w.encode()): c for w, c in words.items()}
    merges = []
    while 256 + 1 + len(merges) < vocab_size:
        pairs = Counter()
        for w, c in words.items():
            for p in itertools.pairwise(w):
                pairs[p] += c
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))
        merges.append(best)
        words = {tuple(_merge(list(w), *best)): c for w, c in words.items()}
    return merges


def check_gpt2_parity():
    vocab = hf_hub_download("openai-community/gpt2", "vocab.json")
    merges = hf_hub_download("openai-community/gpt2", "merges.txt")
    mine = Tokenizer.from_files(vocab, merges, [EOT])
    ref = tiktoken.get_encoding("gpt2")
    for text in CORPUS:
        got, want = mine.encode(text), ref.encode(text, allowed_special={EOT})
        assert got == want, f"{text[:30]!r}: {got[:10]} != {want[:10]}"
        assert mine.decode(got) == text, f"round trip failed on {text[:30]!r}"


def check_trainer_matches_naive():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(TEXT)
    try:
        vocab, merges = train_bpe(f.name, 320, [EOT], num_workers=2)
    finally:
        os.unlink(f.name)
    assert merges == naive_train(TEXT, 320), "incremental trainer diverged from naive recount"
    assert len(vocab) == 256 + 1 + len(merges)
    tok = Tokenizer(vocab, merges, [EOT])
    for text in [TEXT, *CORPUS]:
        assert tok.decode(tok.encode(text)) == text


def check_encode_iterable():
    tok = Tokenizer.from_files(hf_hub_download("openai-community/gpt2", "vocab.json"),
                               hf_hub_download("openai-community/gpt2", "merges.txt"), [EOT])
    text = "\n".join(CORPUS) + TEXT
    lines = text.splitlines(keepends=True)
    assert list(tok.encode_iterable(lines)) == tok.encode(text)


QWEN_EDGES = [
    "I'LL say it's fine, DON'T we'Re sure?",
    "12345 + 3.14159 = x^2+y^2 and 1,000,000",
    "cafe\u0301 and caf\u00e9 should tokenize the same after NFC",
    "\u4f60\u597d\uff0c\u4e16\u754c\uff01 \u6570\u5b66\u5f88\u6709\u8da3\u3002",
    "robot \U0001f916 thumbs \U0001f44d\U0001f3fd check \u2713",
    "def f(x):\n    return x**2  # square\n\n\n",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n4<|im_end|>",
    "  leading\ttabs\r\nwindows lines   \n\n trailing   ",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.",
]


def check_qwen_parity():
    from transformers import AutoTokenizer

    from minitorch.qwen3 import chat_prompt, load_tokenizer

    path = "data/Qwen3-0.6B"
    if not Path(path).exists():  # CI: fetch only the tokenizer files (15 MB), not the weights
        path = snapshot_download("Qwen/Qwen3-0.6B", allow_patterns=["vocab.json", "merges.txt", "tokenizer*.json"])
    mine, hf = load_tokenizer(path), AutoTokenizer.from_pretrained(path)
    corpus = []
    if Path("data/TinyStoriesV2-GPT4-valid.txt").exists():
        with open("data/TinyStoriesV2-GPT4-valid.txt", encoding="utf-8") as f:
            corpus = [next(f) for _ in range(1000)]
    chat = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"}, {"role": "user", "content": "And 3+3?"}]
    for think in [False, True]:
        want = hf.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=think)
        assert chat_prompt(chat, think) == want, f"chat template differs (think={think})"
    # a tool conversation: the tool schema in the system turn, the model's <tool_call> sent back as written,
    # and two results, which share one user turn
    tools = [{"type": "function", "function": {"name": "calculator", "description": "Evaluate arithmetic, e.g. 3 * (4 + 5)",
                                               "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                                                              "required": ["expression"]}}}]
    agent = [{"role": "system", "content": "Use the calculator."}, {"role": "user", "content": "What is 17 * 23 + 9?"},
             {"role": "assistant", "content": '<tool_call>\n{"name": "calculator", "arguments": {"expression": "17 * 23"}}\n</tool_call>'},
             {"role": "tool", "content": "391"}, {"role": "tool", "content": "400"}]
    for chat_ in (agent, agent[1:]):
        want = hf.apply_chat_template(chat_, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert chat_prompt(chat_, tools=tools) == want, "chat template with tools differs"
    for text in QWEN_EDGES + corpus:
        want = hf(text, add_special_tokens=False)["input_ids"]
        got = mine.encode(text)
        assert got == want, f"{text[:40]!r}: mine {got[:12]} vs HF {want[:12]}"
        assert mine.decode(got) == hf.decode(want), f"decode differs on {text[:40]!r}"


def main():
    failed = []
    checks = [("gpt2 merges == tiktoken + round trip", check_gpt2_parity),
              ("trainer == naive recount, round trip", check_trainer_matches_naive),
              ("encode_iterable == encode", check_encode_iterable),
              ("Qwen3 vocab in my BPE == HF tokenizer, chat template too", check_qwen_parity)]
    for name, fn in checks:
        try:
            fn()
            print(f"[ok]   {name}")
        except Exception as e:  # noqa: BLE001 - report every failing check, not just the first
            print(f"[FAIL] {name}: {repr(e)[:300]}")
            failed.append(name)
    print(f"\n{len(checks) - len(failed)}/{len(checks)} tokenizer checks passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
