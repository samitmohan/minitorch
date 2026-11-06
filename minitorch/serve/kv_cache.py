"""Paged KV-cache bookkeeping: which fixed-size blocks of cache memory each sequence owns.

The cache is a pool of `num_blocks` blocks of `block_size` token slots. A sequence
takes blocks one at a time as it grows and returns them all when it finishes, so
memory goes to sequences in proportion to their length instead of every sequence
reserving the maximum up front. A sequence's block table lists its blocks in order:
token t lives in block table[t // block_size], slot t % block_size.

This file only does the accounting. The scheduler asks it whether a sequence fits
before running it; a paged attention kernel would read K and V through the tables.
"""


class BlockManager:
    def __init__(self, num_blocks, block_size=16):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free = list(range(num_blocks - 1, -1, -1))  # pop() hands out block 0 first
        self.tables = {}   # seq_id -> [block ids]
        self.lengths = {}  # seq_id -> tokens stored

    def blocks_needed(self, seq_id, n_new):
        """How many more blocks `seq_id` needs to store `n_new` more tokens."""
        have = len(self.tables.get(seq_id, ()))
        need = -(-(self.lengths.get(seq_id, 0) + n_new) // self.block_size)  # ceil
        return max(0, need - have)

    def can_append(self, seq_id, n_new):
        return self.blocks_needed(seq_id, n_new) <= len(self.free)

    def append(self, seq_id, n_new):
        """Record `n_new` more tokens for `seq_id`, taking blocks as needed. Returns its block table."""
        n_blocks = self.blocks_needed(seq_id, n_new)
        if n_blocks > len(self.free):
            raise MemoryError(f"sequence {seq_id} needs {n_blocks} blocks, {len(self.free)} free")
        table = self.tables.setdefault(seq_id, [])
        table.extend(self.free.pop() for _ in range(n_blocks))
        self.lengths[seq_id] = self.lengths.get(seq_id, 0) + n_new
        return table

    def release(self, seq_id):
        """Return every block of `seq_id` to the pool."""
        self.free.extend(reversed(self.tables.pop(seq_id, [])))
        self.lengths.pop(seq_id, None)

    def used(self):
        return self.num_blocks - len(self.free)
