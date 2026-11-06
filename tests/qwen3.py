"""Qwen3-0.6B in minitorch against Hugging Face transformers (the oracle), from the same checkpoint.

    uv run python tests/qwen3.py            # needs data/Qwen3-0.6B (see minitorch/qwen3.py)

1. Logits for a prompt: minitorch float32 on NumPy and on Metal against HF float32 on CPU,
   then minitorch bfloat16 on Metal (looser; the gap is printed).
2. Greedy decoding through minitorch's KV cache picks the same 24 tokens as HF's generate.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from minitorch import no_grad
from minitorch.qwen3 import config, load_qwen3

PATH = "data/Qwen3-0.6B"
PROMPT = "The derivative of x^3 is 3x^2 because the power rule says"
NEW = 24


def main():
    tok = AutoTokenizer.from_pretrained(PATH)
    ids = tok(PROMPT, return_tensors="np")["input_ids"]
    hf = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float32).eval()
    with torch.no_grad():
        ref = hf(torch.tensor(ids)).logits.numpy()
        ref_gen = hf.generate(torch.tensor(ids), max_new_tokens=NEW, do_sample=False)[0, ids.shape[1]:].tolist()
    del hf
    scale = float(np.abs(ref).max())

    model = load_qwen3(PATH)
    failed, rows = [], []
    for device, dtype, tol in [("cpu", None, 1e-3), ("metal", None, 1e-3), ("metal", "bfloat16", 5e-2)]:
        model.to(device, dtype)
        with no_grad():
            got = model(ids).numpy()
        gap = float(np.abs(got - ref).max())
        same_argmax = float((got.argmax(-1) == ref.argmax(-1)).mean())
        ok = gap / scale <= tol
        print(f"[{'ok' if ok else 'FAIL'}]   logits on {device} {dtype or 'float32'}: largest gap {gap:.2e} "
              f"({gap / scale:.1e} of the largest logit, {scale:.1f}); argmax agrees at {same_argmax:.0%} of {ids.shape[1]} positions")
        failed += [] if ok else [f"logits {device} {dtype}"]
        rows.append({"device": device, "dtype": dtype or "float32", "max_gap": gap, "relative_gap": gap / scale, "argmax_agree": same_argmax})
        if (device, dtype) == ("metal", None):  # before the bfloat16 row: .to() casts in place, and casting back keeps the rounding
            gen = model.generate(ids, NEW, temperature=0)[0, ids.shape[1]:].tolist()
            gen_ok = gen == ref_gen
            print(f"[{'ok' if gen_ok else 'FAIL'}]   greedy {NEW} tokens on metal float32 match HF generate: {tok.decode(gen)!r}")
            failed += [] if gen_ok else ["generate"]
    Path("bench/results/qwen3_parity.json").write_text(json.dumps(
        {"prompt_tokens": int(ids.shape[1]), "largest_logit": scale, "logits": rows, "greedy_tokens": NEW, "greedy_match": gen_ok,
         "params": int(sum(p.data.size for p in model.parameters())), "config": config(PATH), "failed": failed}, indent=1))
    if failed:
        raise SystemExit(f"FAILED: {failed}")


if __name__ == "__main__":
    main()
