"""GPT-2 in minitorch (examples/gpt2.py) against Hugging Face's transformers.

    uv run python tests/gpt2.py

Logits for one prompt on the CPU and the Apple GPU, and 40 greedy tokens, all in float32.
Writes bench/results/gpt2_parity.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from examples.gpt2 import generate, load_gpt2, load_tokenizer
from minitorch import no_grad

PATH = ROOT / "data/gpt2"
PROMPT = "Hello, I'm a language model, and the first thing I learned was"


def main():
    tok = load_tokenizer(PATH)
    ids = tok.encode(PROMPT)
    ref_model = GPT2LMHeadModel.from_pretrained(PATH, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        ref = ref_model(torch.tensor([ids])).logits[0].numpy()
        ref_greedy = ref_model.generate(torch.tensor([ids]), max_new_tokens=40, do_sample=False,
                                        pad_token_id=50256)[0, len(ids):].tolist()

    rows, failed = [], False
    for device in ("cpu", "metal"):
        model = load_gpt2(PATH).to(device)
        with no_grad():
            got = model(np.array([ids])).numpy()[0]
        gap = float(np.abs(got - ref).max())
        agree = float((got.argmax(-1) == ref.argmax(-1)).mean())
        greedy = generate(model, ids, 40, 0, eos=50256)
        rows.append({"device": device, "max_gap": gap, "relative_gap": gap / float(np.abs(ref).max()),
                     "argmax_agree": agree, "greedy_match": greedy == ref_greedy[:len(greedy)] and len(greedy) == len(ref_greedy)})
        ok = gap < 1e-3 and agree == 1.0 and rows[-1]["greedy_match"]
        failed |= not ok
        print(f"[{'ok' if ok else 'FAIL':4s}] {device}: largest logit gap {gap:.1e}, same next token at {agree:.0%} of "
              f"{len(ids)} positions, 40 greedy tokens {'match' if rows[-1]['greedy_match'] else 'differ'}")
    print("greedy:", repr(PROMPT + tok.decode(ref_greedy)))

    params = sum(p.data.size for p in load_gpt2(PATH).parameters())
    result = {"prompt_tokens": len(ids), "largest_logit": float(np.abs(ref).max()), "params": int(params),
              "greedy_tokens": 40, "logits": rows}
    (ROOT / "bench/results/gpt2_parity.json").write_text(json.dumps(result, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
