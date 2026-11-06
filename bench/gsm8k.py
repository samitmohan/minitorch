"""GSM8K accuracy of Qwen3-0.6B running in minitorch: before and after LoRA (SPEC.md M3).

    uv run python bench/gsm8k.py                                   # the base model
    uv run python bench/gsm8k.py --adapter runs/lora_gsm8k/adapter.npz

The protocol is fixed so that the two numbers compare: the first --n questions of the
test split, the same prompt, greedy decoding, thinking off, at most 320 new tokens.
A reply scores if its final answer equals the reference: the number after "####", else the
one in \\boxed{} (the base model's habit), else the reply's last number.
Writes bench/results/gsm8k_<name>.json, and every reply alongside it.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from minitorch.lora import merge_adapter
from minitorch.qwen3 import chat_prompt, load_qwen3, load_tokenizer

RESULTS = Path(__file__).resolve().parent / "results"
PROMPT = "{question}\nSolve it step by step, then give the final answer on its own line as: #### <number>"
NUMBER = r"-?\$?\d[\d,]*(?:\.\d+)?"


def final_number(text):
    """The number after "####", else inside \\boxed{}, else the last number in the text, as a float (None if none)."""
    m = re.search(rf"####\s*({NUMBER})", text) or re.search(rf"\\boxed\{{\s*({NUMBER})", text)
    found = [m.group(1)] if m else re.findall(NUMBER, text)
    if not found:
        return None
    try:
        return float(found[-1].replace(",", "").replace("$", ""))
    except ValueError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/Qwen3-0.6B")
    p.add_argument("--adapter", default=None, help="LoRA adapter .npz from train_lora.py, merged into the weights")
    p.add_argument("--n", type=int, default=250)
    p.add_argument("--max-new", type=int, default=320)
    p.add_argument("--device", default="metal")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--name", default=None, help="results file name; defaults to base or the adapter's run name")
    args = p.parse_args()

    name = args.name or (Path(args.adapter).parent.name if args.adapter else "base")
    with open("data/gsm8k/test.jsonl") as f:
        questions = [json.loads(line) for line in f][:args.n]
    tok = load_tokenizer(args.model)
    model = load_qwen3(args.model)
    if args.adapter:
        merge_adapter(model, args.adapter)
    model.to(args.device, args.dtype)
    eos = tok.ids[b"<|im_end|>"]

    rows, correct, new_tokens, start = [], 0, 0, time.time()
    for i, q in enumerate(questions):
        ids = np.array([tok.encode(chat_prompt([{"role": "user", "content": PROMPT.format(question=q["question"])}]))])
        out = model.generate(ids, args.max_new, temperature=0, eos_id=eos)[0, ids.shape[1]:].tolist()
        reply = tok.decode([t for t in out if t != eos])
        gold = float(q["answer"].split("####")[-1].strip().replace(",", ""))
        got = final_number(reply)
        ok = got is not None and abs(got - gold) < 1e-6
        correct += ok
        new_tokens += len(out)
        rows.append({"i": i, "gold": gold, "got": got, "correct": ok, "tokens": len(out), "reply": reply})
        if (i + 1) % 25 == 0:
            print(f"{i + 1:4d}/{len(questions)}  accuracy so far {correct / (i + 1):.3f}  ({new_tokens / (time.time() - start):.1f} tok/s)", flush=True)

    seconds = time.time() - start
    summary = {"name": name, "adapter": args.adapter, "n": len(questions), "correct": correct, "accuracy": correct / len(questions),
               "device": args.device, "dtype": args.dtype, "max_new": args.max_new, "prompt": PROMPT,
               "new_tokens": new_tokens, "seconds": round(seconds, 1), "tokens_per_s": round(new_tokens / seconds, 1),
               "hit_token_limit": sum(r["tokens"] >= args.max_new for r in rows)}
    (RESULTS / f"gsm8k_{name}.json").write_text(json.dumps(summary, indent=1))
    with open(RESULTS / f"gsm8k_{name}_replies.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
