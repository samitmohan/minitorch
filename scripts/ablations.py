"""The CS336 A1 ablations, summarized from their runs: what each variant changes and where its loss ends.

    uv run python scripts/ablations.py      # reads runs/abl_*, writes bench/results/ablations.json

The runs come from train_gpt.py on bqa3 (gpu_queue.sh): the story model for 10,000 steps, once as it
is and once per variant. A run that diverged records the step where its loss first stopped being a
number; its validation loss after that is NaN, written here as null.
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CHANGES = {"abl_baseline": "none: pre-norm RMSNorm, RoPE, SwiGLU", "abl_no_rmsnorm": "no RMSNorm",
           "abl_no_rmsnorm_lr3e-4": "no RMSNorm, learning rate 3e-4", "abl_post_norm": "RMSNorm after each sublayer",
           "abl_nope": "no position information (NoPE)", "abl_silu": "SiLU feed-forward, widened to SwiGLU's size"}


def finite(x):
    return x if x is not None and math.isfinite(x) else None


def summarize(name):
    rows = [json.loads(line) for line in (ROOT / "runs" / name / "metrics.jsonl").read_text().splitlines()]
    config = json.loads((ROOT / "runs" / name / "config.json").read_text())
    val = [(r["step"], r["val_loss"]) for r in rows if "val_loss" in r]
    diverged = next((r["step"] for r in rows if not math.isfinite(r["loss"])), None)
    return {"run": name, "change": CHANGES[name], "lr": config["lr"], "warmup": config["warmup"], "steps": rows[-1]["step"],
            "final_val_loss": finite(val[-1][1]), "best_val_loss": min((v for _, v in val if math.isfinite(v)), default=None),
            "diverged_at_step": diverged, "tokens_per_s": round(float(np.median([r["tokens_per_s"] for r in rows]))),
            "val_curve": [[s, round(v, 4) if math.isfinite(v) else None] for s, v in val]}


def main():
    names = [n for n in CHANGES if (ROOT / "runs" / n / "metrics.jsonl").exists()]
    rows = [summarize(n) for n in names]
    (ROOT / "bench/results/ablations.json").write_text(json.dumps({"runs": rows}, indent=1))
    for r in rows:
        print(f"{r['run']:24s} val {r['final_val_loss']}  diverged at {r['diverged_at_step']}  {r['tokens_per_s']:,} tok/s")


if __name__ == "__main__":
    main()
