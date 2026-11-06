"""Turn next-token logits into token ids."""
import numpy as np

from . import backend


def sample(logits, temperature=1.0, top_k=None, top_p=None):
    """Pick one token id per row of `logits` (B, V). temperature=0 is greedy.

    top_k keeps the k likeliest tokens. top_p keeps the smallest set of likeliest
    tokens whose probability adds up to at least p.
    """
    if temperature == 0:
        return logits.argmax(axis=-1)
    logits = logits / temperature
    if top_k is not None:
        kth = np.sort(logits, axis=-1)[:, -top_k][:, None]
        logits = np.where(logits < kth, -np.inf, logits)
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    if top_p is not None:
        order = np.argsort(-probs, axis=-1)
        sorted_probs = np.take_along_axis(probs, order, axis=-1)
        # drop a token once the likelier tokens before it already reach top_p
        sorted_probs[np.cumsum(sorted_probs, axis=-1) - sorted_probs >= top_p] = 0.0
        probs = np.zeros_like(probs)
        np.put_along_axis(probs, order, sorted_probs, axis=-1)
        probs /= probs.sum(axis=-1, keepdims=True)
    return np.array([np.random.choice(len(p), p=p) for p in probs])


def sample_on_device(logits, temperature=1.0):
    """sample() without top-k or top-p, for a Tensor of logits (B, V), run where the logits live.

    Only B ids reach the host instead of B * V floats. At 64 rows of Qwen3's 151,936-entry
    vocabulary on the 3090, the host path costs 99 ms per decode step, more than the model.
    Inverse-CDF sampling: the first token whose running total of probability reaches a
    uniform draw scaled to the total, so the probabilities never need dividing by their sum.
    "First" matters. A GPU computes the running total in parallel, in float32, and it is not
    monotone: on the 3090 it decreased in 1,344 places across 64 rows of Qwen3's logits.
    Counting the positions below the draw also counts the dips after the crossing. Over one
    GRPO rollout (20,480 draws) counting disagreed with an exact float64 sampler 28 times, 25
    of them on tokens below 1e-6, enough to derail replies; the first crossing disagreed twice.
    """
    xp = backend.get(logits.device)
    z = logits.data.astype(xp.float32)
    if temperature == 0:
        return xp.to_numpy(z.argmax(axis=-1))
    z = z / temperature
    cdf = xp.cumsum(xp.exp(z - z.max(axis=-1, keepdims=True)), axis=-1)
    u = xp.asarray(np.random.rand(len(z), 1).astype(np.float32)) * cdf[:, -1:]
    # u is below the total, which is cdf's last entry, so every row has a crossing
    return xp.to_numpy((cdf >= u).argmax(axis=-1))
