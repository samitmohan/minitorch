"""Where one training step of the story model goes, op by op (CS336 A2's profiling, with minitorch's own profiler).

    uv run python bench/profile_step.py --device metal
    CUPY_TF32=1 uv run --no-dev python bench/profile_step.py --device cuda    # bqa3

The CS336 configuration (22.7M parameters) at batch 32 x 256 tokens: the step's normal time,
then forward and backward under minitorch.profile, which finishes each op before timing the
next. Writes bench/results/profile_<device>.json with every op's calls and milliseconds.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from minitorch import AdamW, TransformerLM, cross_entropy_loss
from minitorch.profile import profile

RESULTS = Path(__file__).resolve().parent / "results"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="metal", choices=["cpu", "metal", "cuda"])
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()

    np.random.seed(0)
    model = TransformerLM(10_000, 256, 512, 4, 16, 1344).to(args.device)
    opt = AdamW(model.parameters(), lr=1e-3)
    idx = np.random.randint(0, 10_000, (args.batch, 257))

    def step():
        loss = cross_entropy_loss(model(idx[:, :-1]), idx[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()  # waits for the device

    for _ in range(3):
        step()
    start = time.perf_counter()
    for _ in range(5):
        step()
    step_ms = (time.perf_counter() - start) / 5 * 1000

    with profile() as prof:
        loss = cross_entropy_loss(model(idx[:, :-1]), idx[:, 1:])
        opt.zero_grad()
        loss.backward()
    rows = [{"op": op, "calls": calls, "ms": round(ms, 3)} for op, calls, ms in prof.rows()]
    print(f"{args.device}: a step takes {step_ms:.1f} ms; forward + backward profiled op by op: {prof.wall_ms:.1f} ms")
    print(prof.table())
    result = {"device": args.device, "batch": args.batch, "context": 256, "step_ms": round(step_ms, 1),
              "profiled_ms": round(prof.wall_ms, 1), "rows": rows}
    (RESULTS / f"profile_{args.device}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
