"""Speculative decoding on the Apple GPU: Qwen3-0.6B drafts for Qwen3-1.7B (minitorch/speculative.py).

    uv run python bench/speculative.py

Greedy, bfloat16, on GSM8K prompts in the chat template: Qwen3-1.7B decoding alone, Qwen3-0.6B
alone, and the pair at several draft lengths k. Reports tokens per second, how many drafts
the target kept, and where each output first differs from the 1.7B's own greedy text
(bfloat16 rounds a k-position pass differently from k single steps, so a near-tie can flip).
Writes bench/results/speculative_metal.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.gsm8k import PROMPT
from minitorch.qwen3 import chat_prompt, load_qwen3, load_tokenizer
from minitorch.speculative import speculative_generate

RESULTS = Path(__file__).resolve().parent / "results"
MAX_NEW = 128


def main():
    tok = load_tokenizer("data/Qwen3-0.6B")
    target = load_qwen3("data/Qwen3-1.7B").to("metal", "bfloat16")
    draft = load_qwen3("data/Qwen3-0.6B").to("metal", "bfloat16")
    with open("data/gsm8k/test.jsonl") as f:
        questions = [json.loads(line)["question"] for line in f][:4]
    prompts = [tok.encode(chat_prompt([{"role": "user", "content": PROMPT.format(question=q)}])) for q in questions]

    def alone(model, ids):
        return model.generate(np.array([ids]), MAX_NEW, temperature=0)[0, len(ids):].tolist()

    alone(target, prompts[0][:16])  # warm up: the first calls allocate
    rows = []
    for name, model in (("Qwen3-1.7B alone", target), ("Qwen3-0.6B alone", draft)):
        start, outs = time.perf_counter(), [alone(model, ids) for ids in prompts]
        rows.append({"method": name, "k": 0, "tokens_per_s": round(sum(map(len, outs)) / (time.perf_counter() - start), 2)})
        if model is target:
            reference = outs
        print(rows[-1], flush=True)
    for k in (1, 2, 3, 4, 6):
        start, tokens, proposed, accepted, first_diff = time.perf_counter(), 0, 0, 0, []
        for ids, ref in zip(prompts, reference):
            out, p, a = speculative_generate(target, draft, ids, MAX_NEW, k=k)
            tokens, proposed, accepted = tokens + len(out), proposed + p, accepted + a
            first_diff.append(next((i for i, (x, y) in enumerate(zip(out, ref)) if x != y), None))
        rows.append({"method": "speculative", "k": k, "tokens_per_s": round(tokens / (time.perf_counter() - start), 2),
                     "acceptance": round(accepted / proposed, 3), "first_diff_from_target": first_diff})
        print(rows[-1], flush=True)
    base = rows[0]["tokens_per_s"]
    for r in rows:
        r["speedup"] = round(r["tokens_per_s"] / base, 3)
    result = {"device": "metal", "dtype": "bfloat16", "max_new": MAX_NEW, "prompts": len(prompts), "rows": rows}
    (RESULTS / "speculative_metal.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
