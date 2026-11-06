"""What lazy tensors cost and skip: one training step of a small story model, lazy against eager.

Eager is minitorch's _apply as it was before tensors were lazy: every op runs the moment it is written. The two
alternate within one process, trial by trial, so drift in the machine's speed hits both alike. Also counts the ops a
step records and the ops it runs (the difference is work no one read), and compares the losses: the largest gap between
an eager and a lazy run against the largest gap between two eager runs, which is not zero on a GPU that sums in a
different order from run to run.

    uv run python bench/lazy.py        # writes bench/results/lazy.json
"""
import inspect
import json
import time
from pathlib import Path

import numpy as np

from minitorch import attention, backend, mlops
from minitorch import tensor as T
from minitorch.gpt import TransformerLM
from minitorch.loss import cross_entropy_loss
from minitorch.optim import AdamW

OUT = Path(__file__).resolve().parent / "results" / "lazy.json"
CONFIG = {"vocab_size": 1000, "context_length": 64, "d_model": 128, "n_layers": 2, "n_heads": 4, "d_ff": 384}
BATCH, STEPS, TRIALS, WARMUP = 8, 20, 5, 3
LAZY = T._apply


def eager_apply(fn, *parents, **kwargs):
    """_apply before tensors were lazy: run primitive `fn` on `parents` now."""
    device = parents[0].device
    ctx = fn(backend.get(device), *parents)
    out = T.Tensor(ctx.forward(*[p.data for p in parents], **kwargs), device=device)
    if T.grad_enabled() and fn.differentiable and any(ctx.needs_grad):
        out.requires_grad, out._ctx = True, ctx
    return out


def use(apply):
    """Make every op go through `apply`; attention.py imported _apply by name, so it is swapped there too."""
    T._apply = attention._apply = apply


def train(device, apply, steps):
    """`steps` training steps from the same seed: per-step milliseconds and losses."""
    use(apply)
    try:
        np.random.seed(0)
        model = TransformerLM(*CONFIG.values()).to(device)
        opt = AdamW(model.parameters(), lr=1e-3)
        xp = backend.get(device)
        ms, losses = [], []
        for _ in range(steps):
            ids = np.random.randint(0, CONFIG["vocab_size"], (BATCH, CONFIG["context_length"] + 1))
            start = time.perf_counter()
            loss = cross_entropy_loss(model(ids[:, :-1]), ids[:, 1:])
            opt.zero_grad()
            loss.backward()
            opt.step()
            xp.realize(*[p.data for p in model.parameters()])
            ms.append(1000 * (time.perf_counter() - start))
            losses.append(loss.item())
        return ms, losses
    finally:
        use(LAZY)


def count_ops(device):
    """Ops one training step records, and ops it runs."""
    counts = {"recorded": 0, "run": 0}

    def recording(*args, **kwargs):
        counts["recorded"] += 1
        return LAZY(*args, **kwargs)

    forwards = {}
    for _, fn in inspect.getmembers(mlops, inspect.isclass):
        if issubclass(fn, mlops.Function) and "forward" in fn.__dict__:
            forwards[fn] = fn.forward

            def running(self, *args, _forward=fn.forward, **kwargs):
                counts["run"] += 1
                return _forward(self, *args, **kwargs)
            fn.forward = running
    try:
        train(device, recording, 1)
    finally:
        for fn, forward in forwards.items():
            fn.forward = forward
    return counts


def main():
    rows = []
    for device in ["cpu", "metal"]:
        times = {"eager": [], "lazy": []}
        losses = {"eager": [], "lazy": []}
        for _ in range(TRIALS):
            for mode, apply in [("eager", eager_apply), ("lazy", LAZY)]:
                ms, run = train(device, apply, STEPS)
                times[mode] += ms[WARMUP:]
                losses[mode].append(np.array(run))
        gap = max(float(np.abs(e - z).max()) for e in losses["eager"] for z in losses["lazy"])
        noise = max(float(np.abs(e - f).max()) for e in losses["eager"] for f in losses["eager"])
        ops = count_ops(device)
        for mode in ["eager", "lazy"]:
            rows.append({"device": device, "mode": mode, "ms_per_step": round(float(np.median(times[mode])), 2)})
        rows.append({"device": device, "mode": "both", "loss_gap_eager_lazy": gap, "loss_gap_eager_eager": noise,
                     "ops_recorded_per_step": ops["recorded"], "ops_run_per_step": ops["run"]})
        print(rows[-3:])
    primitives = sum(1 for _, fn in inspect.getmembers(mlops, inspect.isclass)
                     if issubclass(fn, mlops.Function) and fn is not mlops.Function)
    OUT.write_text(json.dumps({"config": CONFIG, "batch": BATCH, "steps": STEPS, "trials": TRIALS,
                               "primitives": primitives, "rows": rows}, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
