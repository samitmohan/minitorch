"""Mixed-precision training against float32 (CS336 A2): the story model, same weights, same batches.

    uv run python bench/mixed_precision.py --device metal                          # bfloat16 multiplies
    CUPY_TF32=1 uv run --no-dev python bench/mixed_precision.py --device cuda      # float16, loss scaled

Each precision trains the CS336 configuration for --steps steps from one starting point on one
sequence of batches. Reports the median step time and both loss curves. Writes
bench/results/mixed_precision_<device>.json.
"""
import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from minitorch import (
    AdamW,
    LossScaler,
    TransformerLM,
    autocast,
    clip_grad_norm,
    cross_entropy_loss,
)

RESULTS = Path(__file__).resolve().parent / "results"


def train(device, amp, state, batches):
    model = TransformerLM(10_000, 256, 512, 4, 16, 1344)
    model.load_state_dict(state)
    model.to(device)
    params = model.parameters()
    opt = AdamW(params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = LossScaler() if amp == "float16" else None
    losses, ms, skipped = [], [], 0
    for x in batches:
        start = time.perf_counter()
        with autocast(amp) if amp else nullcontext():
            loss = cross_entropy_loss(model(x[:, :-1]), x[:, 1:])
        opt.zero_grad()
        (loss * scaler.scale if scaler else loss).backward()
        if scaler and not scaler.unscale(params):
            skipped += 1
            continue
        clip_grad_norm(params, 1.0)
        opt.step()
        losses.append(loss.item())  # waits for the device, so the clock below includes the work
        ms.append((time.perf_counter() - start) * 1000)
    return {"precision": amp or "float32", "step_ms": round(float(np.median(ms[5:])), 1), "losses": [round(v, 4) for v in losses],
            "skipped_steps": skipped}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="metal", choices=["metal", "cuda"])
    p.add_argument("--steps", type=int, default=150)
    args = p.parse_args()

    np.random.seed(0)
    state = TransformerLM(10_000, 256, 512, 4, 16, 1344).state_dict()
    data = np.memmap("data/tinystories_train.bin", dtype=np.uint16, mode="r")
    starts = np.random.randint(0, len(data) - 257, (args.steps, 32))
    batches = [np.stack([data[s:s + 257] for s in row]).astype(np.int64) for row in starts]
    low = "bfloat16" if args.device == "metal" else "float16"
    rows = [train(args.device, amp, state, batches) for amp in (None, low)]
    for r in rows:
        print(f"{r['precision']:9s} {r['step_ms']:7.1f} ms per step, loss {r['losses'][0]:.3f} -> {r['losses'][-1]:.3f}, "
              f"{r['skipped_steps']} steps skipped")
    tail = [np.mean(r["losses"][-50:]) for r in rows]
    result = {"device": args.device, "steps": args.steps, "rows": rows, "last_50_loss_gap": round(float(tail[1] - tail[0]), 4)}
    (RESULTS / f"mixed_precision_{args.device}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
