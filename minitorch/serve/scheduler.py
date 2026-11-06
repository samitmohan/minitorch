"""Continuous batching: requests join and leave the running batch between decode steps.

A static batch waits for its longest sequence, so short requests sit idle. Here
every `step()` first admits waiting requests (one prefill pass each), then runs one
decode step for everything running. A request leaves the moment it emits its stop
token or hits max_tokens, and its KV blocks go straight to the next request.

When a decode step needs a block and none is free, the newest running request is
preempted: its blocks and cache are dropped and it goes back to the front of the
queue. On readmission it prefills prompt + everything it has generated so far,
which rebuilds the same cache (recompute instead of swapping to host memory).
"""
from collections import deque
from dataclasses import dataclass, field

from ..sampling import sample


@dataclass(eq=False)  # requests compare by identity: two can hold the same tokens
class Request:
    id: int
    prompt: list
    max_tokens: int
    temperature: float = 1.0
    top_p: float | None = None
    eos_id: int | None = None
    out: list = field(default_factory=list)
    done: bool = False


class Scheduler:
    def __init__(self, runner, blocks, max_batch=8):
        """`runner` runs the model (see runner.py); `blocks` is the BlockManager that budgets its KV cache."""
        self.runner, self.blocks, self.max_batch = runner, blocks, max_batch
        self.waiting, self.running = deque(), []
        self.preemptions = 0

    def fits(self, request):
        """Whether the request could ever run: its prompt plus all its new tokens within the context and the cache."""
        total = len(request.prompt) + request.max_tokens
        return total <= self.runner.context_length and -(-total // self.blocks.block_size) <= self.blocks.num_blocks

    def add(self, request):
        if not self.fits(request):
            raise ValueError(f"request {request.id}: {len(request.prompt) + request.max_tokens} tokens can never fit")
        self.waiting.append(request)

    def has_work(self):
        return bool(self.waiting or self.running)

    def step(self):
        """Admit, prefill, decode once. Returns the (request, token) pairs produced this step."""
        emitted = []
        # admit in arrival order while a batch slot and the whole prefix's blocks are free
        while self.waiting and len(self.running) < self.max_batch:
            req = self.waiting[0]
            tokens = req.prompt + req.out  # a preempted request also replays what it already generated
            if not self.blocks.can_append(req.id, len(tokens)):
                break
            self.waiting.popleft()
            self.blocks.append(req.id, len(tokens))
            emitted.append(self._emit(req, self.runner.prefill(req.id, tokens)[None]))
            self.running.append(req)
        self._retire()

        # requests admitted this step already have their token; the rest decode once
        fresh = {id(r) for r, _ in emitted}
        decode = [r for r in self.running if id(r) not in fresh]
        for req in list(decode):
            # room for the token about to be fed back in, preempting the newest request until it fits
            while req in self.running and not self.blocks.can_append(req.id, 1):
                victim = self.running[-1]
                self._preempt(victim)
                if victim in decode:
                    decode.remove(victim)
            if req in self.running:
                self.blocks.append(req.id, 1)
        if decode:
            logits = self.runner.decode([r.id for r in decode], [r.out[-1] for r in decode])
            emitted += [self._emit(r, logits[i:i + 1]) for i, r in enumerate(decode)]
        self._retire()
        return emitted

    def _retire(self):
        for req in [r for r in self.running if r.done]:
            self.running.remove(req)
            self.blocks.release(req.id)
            self.runner.release(req.id)

    def _emit(self, req, logits):
        token = int(sample(logits, req.temperature, top_p=req.top_p)[0])
        req.out.append(token)
        req.done = token == req.eos_id or len(req.out) >= req.max_tokens
        return req, token

    def _preempt(self, req):
        self.running.remove(req)
        self.blocks.release(req.id)
        self.runner.release(req.id)
        self.waiting.appendleft(req)
        self.preemptions += 1
