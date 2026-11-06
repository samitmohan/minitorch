"""The serving engine: a background thread runs the scheduler, and each request's tokens stream back through a queue.

    engine = Engine(Scheduler(MinitorchRunner(model), BlockManager(512)), lock)
    tokens = engine.submit(prompt_ids, max_tokens=200, eos_id=eos)   # a queue.Queue
    for token in iter(tokens.get, None): ...

Requests from many callers interleave at the granularity of one scheduler step, which is
continuous batching (scheduler.py) seen from the outside.
"""
import itertools
import queue
import threading

from .scheduler import Request


class Engine:
    def __init__(self, scheduler, lock):
        """`lock` is held around every step: the model and its caches are not safe to run from two threads at once."""
        self.scheduler, self.lock = scheduler, lock
        self.inbox, self.outputs, self.ids = queue.Queue(), {}, itertools.count()
        threading.Thread(target=self._loop, daemon=True, name="engine").start()

    def submit(self, prompt, max_tokens, temperature=0.7, top_p=0.95, eos_id=None):
        """Queue a request. Returns a queue that receives each new token, then None (or an exception if a step failed)."""
        req = Request(next(self.ids), list(prompt), max_tokens, temperature, top_p, eos_id)
        if not self.scheduler.fits(req):
            raise ValueError(f"{len(prompt)} prompt tokens plus {max_tokens} new ones do not fit")
        out = self.outputs[req.id] = queue.Queue()
        self.inbox.put(req)
        return out

    def _loop(self):
        s = self.scheduler
        while True:
            s.add(self.inbox.get())  # sleep until there is work
            while s.has_work():
                while not self.inbox.empty():  # newcomers join between steps
                    s.add(self.inbox.get_nowait())
                try:
                    with self.lock:
                        emitted = s.step()
                except Exception as e:  # noqa: BLE001 - every caller must hear about it, or it waits forever
                    # every live request, including one caught between the queue and the batch when prefill failed
                    for rid in list(self.outputs):  # a snapshot: submit() may add one meanwhile, from another thread
                        s.blocks.release(rid)
                        s.runner.release(rid)
                        self.outputs.pop(rid).put(e)
                    s.waiting.clear()
                    s.running.clear()
                    while not self.inbox.empty():  # submitted while the step failed: tell any still listening, then drop
                        out = self.outputs.pop(self.inbox.get_nowait().id, None)
                        if out is not None:
                            out.put(e)
                    continue
                for req, token in emitted:
                    out = self.outputs.get(req.id)
                    if out is None:  # submitted while a step failed: its caller already has the error
                        continue
                    out.put(token)
                    if req.done:
                        self.outputs.pop(req.id).put(None)
