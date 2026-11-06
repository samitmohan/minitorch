"""The serving core: block accounting, and continuous batching that changes nothing about the output.

    uv run python tests/serve.py [--device metal]

With greedy sampling, requests run concurrently through the Scheduler must produce
exactly the tokens model.generate produces for each one alone: with room for
everyone, with a batch cap that makes requests queue, and with a KV budget so small
that requests get preempted and recomputed. Then the same through the Engine, with
every request submitted from its own thread at once, as an HTTP server would.
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from minitorch import TransformerLM, no_grad
from minitorch.serve.engine import Engine
from minitorch.serve.kv_cache import BlockManager
from minitorch.serve.runner import MinitorchRunner
from minitorch.serve.scheduler import Request, Scheduler


def check_blocks():
    b = BlockManager(num_blocks=4, block_size=4)
    assert b.append("a", 5) == [0, 1]                       # 5 tokens -> 2 blocks
    assert b.blocks_needed("a", 3) == 0 and b.blocks_needed("a", 4) == 1
    assert b.append("a", 3) == [0, 1] and b.append("a", 1) == [0, 1, 2]
    assert b.append("b", 4) == [3] and not b.can_append("b", 1)
    try:
        b.append("b", 1)
        raise AssertionError("allocated past the pool")
    except MemoryError:
        pass
    b.release("a")
    assert b.used() == 1 and b.append("b", 1) == [3, 0]  # freed blocks are reused
    print("[ok]   block manager: allocation, reuse, exhaustion")


def run(model, requests, num_blocks, max_batch):
    sched = Scheduler(MinitorchRunner(model), BlockManager(num_blocks, block_size=4), max_batch)
    for r in requests:
        sched.add(r)
    steps = 0
    while sched.has_work():
        sched.step()
        steps += 1
    assert sched.blocks.used() == 0 and not sched.runner.caches, "blocks or caches leaked"
    return {r.id: r.out for r in requests}, steps, sched.preemptions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    device = p.parse_args().device

    check_blocks()
    np.random.seed(0)
    model = TransformerLM(vocab_size=97, context_length=64, d_model=32, n_layers=2, n_heads=4, d_ff=64, n_kv_heads=2).to(device)
    rng = np.random.default_rng(0)
    specs = [(rng.integers(0, 97, n).tolist(), m) for n, m in [(5, 12), (11, 4), (3, 20), (17, 9), (8, 15), (2, 6)]]

    reference = {}
    for i, (prompt, max_tokens) in enumerate(specs):
        ids = model.generate(np.array([prompt]), max_tokens, temperature=0)
        reference[i] = ids[0, len(prompt):].tolist()
    # request 2 stops early on a stop token: the third token it would generate
    eos = {2: reference[2][2]}
    reference[2] = reference[2][:reference[2].index(eos[2]) + 1]

    failed, results = [], []
    for name, num_blocks, max_batch in [("room for all", 64, 8), ("batch cap 2", 64, 2), ("tiny KV budget", 9, 8)]:
        requests = [Request(i, prompt, m, temperature=0, eos_id=eos.get(i)) for i, (prompt, m) in enumerate(specs)]
        out, steps, preemptions = run(model, requests, num_blocks, max_batch)
        ok = out == reference
        print(f"[{'ok' if ok else 'FAIL'}]   {name}: {len(specs)} requests in {steps} steps, {preemptions} preemptions, "
              f"tokens {'identical to' if ok else 'DIFFER from'} one-at-a-time generate")
        failed += [] if ok else [name]
        results.append({"scenario": name, "requests": len(specs), "num_blocks": num_blocks, "max_batch": max_batch,
                        "steps": steps, "preemptions": preemptions, "identical": ok})
        if name == "tiny KV budget":
            assert preemptions > 0, "the tiny budget never forced a preemption"
    # the engine: one thread per caller, all submitting at once
    engine = Engine(Scheduler(MinitorchRunner(model), BlockManager(9, block_size=4), max_batch=8), threading.Lock())
    streams, threads = {}, []
    for i, (prompt, m) in enumerate(specs):
        def call(i=i, prompt=prompt, m=m):
            streams[i] = list(iter(engine.submit(prompt, m, temperature=0, eos_id=eos.get(i)).get, None))
        threads.append(threading.Thread(target=call))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    ok = streams == reference
    print(f"[{'ok' if ok else 'FAIL'}]   engine: {len(specs)} threads submitting at once get the same tokens as generate alone")
    failed += [] if ok else ["engine"]
    try:
        engine.submit(list(range(60)), 10)
        failed.append("engine accepted a request longer than the context")
    except ValueError:
        print("[ok]   engine: a request that can never fit is refused at submit")

    # a step that fails: its request and one queued behind it hear about it, and the engine keeps serving
    class FailOnce(MinitorchRunner):
        failed = False

        def prefill(self, seq_id, tokens):
            if not self.failed:
                self.failed = True
                time.sleep(0.2)  # long enough for a second request to queue behind this one
                raise RuntimeError("simulated GPU failure")
            return super().prefill(seq_id, tokens)
    engine = Engine(Scheduler(FailOnce(model), BlockManager(64, block_size=4)), threading.Lock())
    first = engine.submit(specs[0][0], 4, temperature=0)
    time.sleep(0.05)
    queued = engine.submit(specs[1][0], 4, temperature=0)
    errors = [first.get(timeout=30), queued.get(timeout=30)]
    out = engine.submit(specs[2][0], specs[2][1], temperature=0, eos_id=eos.get(2))
    after = list(iter(lambda: out.get(timeout=30), None))  # a dead engine fails this with queue.Empty, not a hang
    ok = all(isinstance(e, RuntimeError) for e in errors) and after == reference[2]
    print(f"[{'ok' if ok else 'FAIL'}]   engine: a failed step reaches its caller and the one queued behind, then serving goes on")
    failed += [] if ok else ["engine failure"]

    # imported here, not at the top: the API module pulls in FastAPI
    import os
    import tempfile

    from fastapi.testclient import TestClient

    from minitorch.serve import api
    from minitorch.serve.api import _parse
    with tempfile.TemporaryDirectory() as tmp:
        api.RUNS = Path(tmp)
        (Path(tmp) / "done").mkdir()
        (Path(tmp) / "done" / "metrics.jsonl").write_text('{"step": 1, "loss": 2.5}\n{"step": 2, "loss": NaN}\n')
        os.utime(Path(tmp) / "done" / "metrics.jsonl", (0, 0))  # last written in 1970: a finished run
        body = TestClient(api.app).get("/api/runs/done/stream").text  # returns only if the server ends the stream
    ok = body.count('data: {"step"') == 2 and '"loss": null' in body and body.rstrip().endswith("event: end\ndata: {}")
    print(f"[{'ok' if ok else 'FAIL'}]   api: a finished run's stream sends its history, then ends instead of holding the connection")
    failed += [] if ok else ["api stream end"]

    r = TestClient(api.app).post("/api/graph", json={"expr": "log(x)", "variables": {"x": -1.0}})
    ok = r.status_code == 200 and None in [n.get("value") for n in r.json()["nodes"]]
    print(f"[{'ok' if ok else 'FAIL'}]   api: log(-1) in the graph explorer comes back as null, not a server error")
    failed += [] if ok else ["api graph NaN"]

    inside, release = threading.Event(), threading.Event()

    def hold_no_grad():  # the serving engine sits inside no_grad while it decodes
        with no_grad():
            inside.set()
            release.wait()
    holder = threading.Thread(target=hold_no_grad)
    holder.start()
    inside.wait()
    frozen = sum(not p.requires_grad for p in TransformerLM(97, 16, 32, 1, 4, 64).parameters())
    release.set()
    holder.join()
    ok = frozen == 0
    print(f"[{'ok' if ok else 'FAIL'}]   no_grad in one thread leaves a model built in another trainable ({frozen} frozen)")
    failed += [] if ok else ["thread-local no_grad"]
    ok = _parse('{"step": 626, "loss": NaN, "grad_norm": Infinity}') == {"step": 626, "loss": None, "grad_norm": None}
    print(f"[{'ok' if ok else 'FAIL'}]   api: a diverged run's NaN metrics reach the browser as null, not as invalid JSON")
    failed += [] if ok else ["api NaN"]

    out = Path(__file__).resolve().parent.parent / "bench" / "results" / f"serve_{device}.json"
    out.write_text(json.dumps({"device": device, "block_size": 4, "scenarios": results, "failed": failed}, indent=1))
    if failed:
        raise SystemExit(f"FAILED: {failed}")


if __name__ == "__main__":
    main()
