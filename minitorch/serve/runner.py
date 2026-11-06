"""Model runners: the scheduler's only view of the model.

    prefill(seq_id, tokens) -> logits (V,)     run a whole prefix, keep its KV cache
    decode(seq_ids, tokens) -> logits (B, V)   one new token for each sequence
    release(seq_id)                            drop that sequence's cache
"""
import numpy as np

from ..attention import KVCache
from ..tensor import no_grad


class MinitorchRunner:
    def __init__(self, model):
        self.model = model
        self.context_length = model.context_length
        self.caches = {}

    def prefill(self, seq_id, tokens):
        self.caches[seq_id] = [KVCache() for _ in self.model.blocks]
        with no_grad():
            h = self.model.hidden(np.array([tokens]), self.caches[seq_id])
            return self.model.project(h[:, -1])[0].numpy()  # the last position only: the vocabulary-wide logits of the rest go unread

    def decode(self, seq_ids, tokens):
        # shortcut: one forward per sequence, since their caches differ in length; a paged attention kernel that reads each row's K/V through its block table batches them (M4)
        with no_grad():
            return np.stack([self.model(np.array([[t]]), self.caches[s])[0, -1].numpy() for s, t in zip(seq_ids, tokens)])

    def release(self, seq_id):
        self.caches.pop(seq_id, None)
