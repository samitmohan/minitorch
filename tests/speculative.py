"""Speculative decoding (minitorch/speculative.py): greedy output equals the target's, and samples follow its distribution.

    uv run python tests/speculative.py
"""
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from minitorch import TransformerLM, no_grad
from minitorch.speculative import speculative_generate


def model(vocab, seed, d_model):
    np.random.seed(seed)
    return TransformerLM(vocab_size=vocab, context_length=64, d_model=d_model, n_layers=2, n_heads=4, d_ff=2 * d_model, n_kv_heads=2)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def main():
    target, draft = model(50, 0, 32), model(50, 1, 16)
    prompt = [3, 14, 15, 9, 2]
    alone = target.generate(np.array([prompt]), 20, temperature=0)[0, len(prompt):].tolist()
    for k in (1, 2, 3, 4):
        got, _, _ = speculative_generate(target, draft, prompt, 20, k=k, temperature=0)
        assert got == alone, (k, got, alone)
    print("[ok]   greedy: 20 tokens equal the target's own greedy output for k = 1, 2, 3, 4")

    # sampling: the first two tokens must follow the target's joint distribution, which a 6-token vocabulary makes exact
    target, draft = model(6, 2, 32), model(6, 3, 16)
    prompt = [1, 4, 2]
    with no_grad():
        def first_two(m):
            p1 = softmax(m(np.array([prompt])).numpy()[0, -1].astype(np.float64))
            return {(a, b): p1[a] * softmax(m(np.array([prompt + [a]])).numpy()[0, -1].astype(np.float64))[b]
                    for a, b in product(range(6), repeat=2)}
        exact, draft_exact = first_two(target), first_two(draft)
    np.random.seed(0)
    n = 4000
    counts = dict.fromkeys(exact, 0)
    for _ in range(n):
        got, _, _ = speculative_generate(target, draft, prompt, 2, k=2, temperature=1.0)
        counts[tuple(got)] += 1
    worst = max(abs(counts[c] / n - exact[c]) / np.sqrt(exact[c] * (1 - exact[c]) / n) for c in exact)
    apart = max(abs(draft_exact[c] - exact[c]) / np.sqrt(exact[c] * (1 - exact[c]) / n) for c in exact)
    assert worst < 4.5, worst
    assert apart > 10, apart  # the draft's own distribution is far off, so the check would catch a sampler that echoes it
    print(f"[ok]   sampling: {n} two-token samples match the target's exact distribution over 36 pairs "
          f"(worst cell {worst:.1f} standard errors; the draft's own distribution is {apart:.0f} away)")


if __name__ == "__main__":
    main()
