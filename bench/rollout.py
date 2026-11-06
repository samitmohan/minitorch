"""How fast GRPO's rollouts sample: Qwen3-0.6B decoding 1, 8, and 64 rows at once (SPEC.md A5).

    CUPY_TF32=1 uv run --no-dev python bench/rollout.py --device cuda    # bqa3

For each batch size: the prefill of left-padded GSM8K prompts, then the time per decode
step and the reply tokens per second. At 64 rows, one decode step split into the model,
copying the logits to the host, and drawing the tokens on the host or on the device.
Writes bench/results/rollout_<device>.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.gsm8k import PROMPT
from minitorch import backend, no_grad
from minitorch.attention import KVCache
from minitorch.qwen3 import chat_prompt, load_qwen3, load_tokenizer
from minitorch.sampling import sample, sample_on_device
from train_grpo import load_questions

RESULTS = Path(__file__).resolve().parent / "results"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", choices=["cuda", "metal"])
    p.add_argument("--model", default="data/Qwen3-0.6B")
    p.add_argument("--steps", type=int, default=64, help="decode steps timed per batch size")
    args = p.parse_args()

    tok = load_tokenizer(args.model)
    model = load_qwen3(args.model).to(args.device)
    questions = load_questions("data/gsm8k/train.jsonl")
    prompts = [tok.encode(chat_prompt([{"role": "user", "content": PROMPT.format(question=q)}])) for q, _ in questions[:64]]

    def timed(fn, *fn_args, **fn_kwargs):
        """fn's result and its wall time, waiting for the GPU's queued work: kernels launch asynchronously."""
        start = time.perf_counter()
        out = fn(*fn_args, **fn_kwargs)
        if args.device == "cuda":
            import cupy
            cupy.cuda.Device().synchronize()
        else:
            backend.get(args.device).realize(*([out.data] if hasattr(out, "data") else []))
        return out, time.perf_counter() - start

    rows = []
    for batch in (1, 8, 64):
        chosen = prompts[:batch]
        width = max(len(q) for q in chosen)
        pad = np.array([width - len(q) for q in chosen])
        ids = np.array([[0] * n + q for n, q in zip(pad, chosen)])
        model.generate(ids, 4, pad=pad)  # warm up: the first calls allocate memory
        _, prefill = timed(model.generate, ids, 1, pad=pad)
        _, total = timed(model.generate, ids, args.steps + 1, pad=pad)
        step = (total - prefill) / args.steps
        rows.append({"rows": batch, "prompt_tokens": int(width), "prefill_ms": round(prefill * 1000, 1),
                     "decode_ms_per_step": round(step * 1000, 1), "tokens_per_s": round(batch / step, 1)})
        print(rows[-1], flush=True)

    # one decode step of the last batch (64 rows), piece by piece; medians of 10 after 2 warm-ups
    pieces = {"model": [], "logits_to_host": [], "sample_on_host": [], "sample_on_device": []}
    caches = [KVCache() for _ in model.blocks]

    def decode_logits(next_id):
        return model.project(model.hidden(next_id, caches, pad)[:, -1])

    with no_grad():
        model.hidden(ids, caches, pad)
        for i in range(12):
            logits, t_model = timed(decode_logits, np.random.randint(0, 150_000, (len(ids), 1)))
            host, t_copy = timed(logits.numpy)
            _, t_host = timed(sample, host, 1.0)
            _, t_device = timed(sample_on_device, logits, 1.0)
            if i >= 2:
                for name, seconds in zip(pieces, (t_model, t_copy, t_host, t_device)):
                    pieces[name].append(seconds * 1000)
    step_ms = {name: round(float(np.median(ms)), 1) for name, ms in pieces.items()}
    print("one 64-row decode step, ms:", step_ms)

    result = {"device": args.device, "decode_steps_timed": args.steps, "rows": rows, "step_ms_64_rows": step_ms}
    (RESULTS / f"rollout_{args.device}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
