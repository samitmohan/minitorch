"""Scaling laws from my own runs (CS336 A3, Chinchilla's IsoFLOP method): how big a model each compute budget wants.

    CUPY_TF32=1 uv run --no-dev python scripts/scaling.py run --device cuda    # bqa3: trains the grid
    uv run python scripts/scaling.py fit                                        # bench/results/scaling.json

For each compute budget C (FLOPs), TinyStories models of several sizes N train on exactly
C / (6 N) tokens, each with its own warmup and cosine schedule, and report a final validation
loss. N counts the parameters in matrix multiplies, the ones that cost 6 FLOPs per token.
Along one budget, loss against log N is close to a parabola, and its bottom is the budget's
compute-optimal size. A line through those optima on log-log axes, N_opt = k C^a, says how to
grow the model as compute grows, and D_opt = C / (6 N_opt) gives the tokens.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUDGETS = [3e14, 1e15, 3e15]
D_MODELS = [128, 192, 256, 384, 512]
LAYERS, VOCAB, BATCH, CONTEXT = 4, 10_000, 32, 256


def shape(d_model):
    """(heads, d_ff, matmul parameters) at d_model: 32-wide heads and SwiGLU's d_ff = 8/3 d_model, to a multiple of 64."""
    d_ff = round(8 / 3 * d_model / 64) * 64
    return d_model // 32, d_ff, LAYERS * (4 * d_model ** 2 + 3 * d_model * d_ff) + VOCAB * d_model


CS336_RUN = 6 * shape(512)[2] * 327_680_000  # the story model of §6: its matmul parameters x its tokens x 6


def run(device):
    for budget in BUDGETS:
        for d in D_MODELS:
            heads, d_ff, n = shape(d)
            steps = round(budget / (6 * n) / (BATCH * CONTEXT))
            name = f"scaling/C{budget:.0e}_d{d}"
            if (ROOT / "runs" / name / "metrics.jsonl").exists():
                continue  # a finished or interrupted run keeps its place; delete it to retrain
            print(f"{name}: {n / 1e6:.1f}M parameters, {steps:,} steps", flush=True)
            subprocess.run([sys.executable, "train_gpt.py", "--run", name, "--device", device, "--d-model", str(d),
                            "--heads", str(heads), "--d-ff", str(d_ff), "--layers", str(LAYERS), "--steps", str(steps),
                            "--warmup", str(max(steps // 10, 1)), "--eval-every", str(steps), "--eval-batches", "50",
                            "--sample-every", str(10 * steps)], cwd=ROOT, check=True)


def fit():
    rows = []
    for budget in BUDGETS:
        for d in D_MODELS:
            path = ROOT / "runs" / f"scaling/C{budget:.0e}_d{d}" / "metrics.jsonl"
            evals = [r for r in map(json.loads, path.read_text().splitlines()) if "val_loss" in r] if path.exists() else []
            if evals:
                rows.append({"budget": budget, "d_model": d, "params": shape(d)[2], "val_loss": evals[-1]["val_loss"]})
    optima = []
    for budget in BUDGETS:
        pts = [r for r in rows if r["budget"] == budget]
        if len(pts) < 3:
            continue
        a, b, c = np.polyfit(np.log([r["params"] for r in pts]), [r["val_loss"] for r in pts], 2)
        if a > 0:  # a parabola that opens upward has a bottom
            optima.append({"budget": budget, "params": float(np.exp(-b / (2 * a))), "loss": float(c - b * b / (4 * a))})
    result = {"budgets": BUDGETS, "runs": rows, "optima": optima}
    if len(optima) >= 2:
        slope, intercept = np.polyfit(np.log([o["budget"] for o in optima]), np.log([o["params"] for o in optima]), 1)
        n_cs336 = float(np.exp(intercept) * CS336_RUN ** slope)
        result |= {"exponent": float(slope), "coefficient": float(np.exp(intercept)), "cs336_budget": CS336_RUN,
                   "cs336_params_predicted": n_cs336, "cs336_tokens_predicted": CS336_RUN / (6 * n_cs336)}
    (ROOT / "bench/results/scaling.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["run", "fit"])
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run(args.device) if args.command == "run" else fit()


if __name__ == "__main__":
    main()
