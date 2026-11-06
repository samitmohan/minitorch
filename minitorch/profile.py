"""Where the time goes: each primitive's forward and backward, timed one by one.

    with profile() as p:
        loss = model(x)...; loss.backward()
    print(p.table())

The device finishes each op before the clock stops (GPUs otherwise run ops asynchronously,
and a lazy backend like MLX would run them all at the end), so the times add up honestly but
the total runs slower than an unprofiled step. Wall time minus the ops' sum is everything
between them: Python, the autograd bookkeeping, and starting each op.
"""
import time
from collections import defaultdict

from . import backend
from . import tensor as T


def _wait(device, out):
    """Block until the device has finished computing out."""
    if device == "cuda":
        import cupy
        cupy.cuda.Device().synchronize()
    else:
        backend.get(device).realize(*(out if isinstance(out, tuple) else (out,)))


class profile:
    def __init__(self):
        self.ms = defaultdict(float)
        self.calls = defaultdict(int)

    def __enter__(self):
        T._profiler = self
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.wall_ms = (time.perf_counter() - self.start) * 1000
        T._profiler = None

    def time(self, name, device, fn, *args, **kwargs):
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        _wait(device, tuple(o for o in out if o is not None) if isinstance(out, tuple) else out)
        self.ms[name] += (time.perf_counter() - start) * 1000
        self.calls[name] += 1
        return out

    def rows(self):
        """(op, calls, ms) sorted by time, then the time between ops."""
        rows = sorted(((name, self.calls[name], self.ms[name]) for name in self.ms), key=lambda r: -r[2])
        return rows + [("between ops", 0, self.wall_ms - sum(self.ms.values()))]

    def table(self):
        return "\n".join(f"{name:28s} {calls:6d} calls {ms:9.2f} ms {ms / self.wall_ms:6.1%}" for name, calls, ms in self.rows())
