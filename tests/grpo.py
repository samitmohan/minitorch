"""GRPO's pieces: batched sampling across prompt lengths, group advantages, and the update.

    uv run python tests/grpo.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from minitorch import SGD, TransformerLM, no_grad
from train_grpo import group_advantages, policy_loss, reward
from train_lora import make_batch, reply_loss

CONFIG = {"vocab_size": 50, "context_length": 64, "d_model": 32, "n_layers": 2, "n_heads": 4, "d_ff": 64, "n_kv_heads": 2}


def main():
    np.random.seed(0)
    model = TransformerLM(**CONFIG)

    prompts = [list(np.random.randint(1, 50, n)) for n in (3, 7, 12)]
    width = max(len(p) for p in prompts)
    pad = np.array([width - len(p) for p in prompts])
    alone = [model.generate(np.array([p]), 10, temperature=0)[0, len(p):] for p in prompts]
    for pad_id in (0, 49):  # the padding's ids must not matter: attention hides those positions
        batch = np.array([[pad_id] * n + p for n, p in zip(pad, prompts)])
        together = model.generate(batch, 10, temperature=0, pad=pad)[:, width:]
        assert all(np.array_equal(a, t) for a, t in zip(alone, together)), (pad_id, alone, together)
    with no_grad():
        last = model(batch, pad=pad).numpy()[:, -1]
        gap = max(float(np.abs(model(np.array([p])).numpy()[0, -1] - row).max()) for p, row in zip(prompts, last))
    assert gap < 1e-4, gap
    print(f"[ok]   prompts of 3, 7, and 12 tokens left-padded into one batch generate what each does alone "
          f"(last logits within {gap:.1e}), whatever the padding ids")

    assert reward("so the total is\n#### 1,250", 1250, stopped=True) == 1.0
    assert reward("#### 1250 ---", 1250, stopped=False) == 0.0  # right number, but cut off at the token limit
    assert reward("#### 1249", 1250, stopped=True) == 0.0
    print("[ok]   reward: 1 only for a reply that stops on its own with the right final number")

    adv = group_advantages([1, 0, 0, 0, 1, 1, 1, 1], group=4)
    assert np.allclose(adv[:4], [3 ** 0.5, -(3 ** -0.5), -(3 ** -0.5), -(3 ** -0.5)], atol=1e-4), adv
    assert np.all(adv[4:] == 0), adv
    print("[ok]   advantages are (reward - group mean) / group std, and 0 for a group that scored the same everywhere")

    prompt = list(np.random.randint(1, 50, 5))
    rows = [(prompt + list(np.random.randint(1, 50, 6)), 5), (prompt + list(np.random.randint(1, 50, 3)), 5)]
    budget = 8
    with no_grad():
        both = policy_loss(model, rows, [1.0, 1.0], 2, pad=0, budget=budget).item()
        each = [reply_loss(model, *make_batch([r], 0)).item() for r in rows]
    lengths = [len(ids) - n for ids, n in rows]
    expected = np.mean([ce * n / budget for ce, n in zip(each, lengths)])
    assert abs(both - expected) < 1e-5, (both, expected)
    print("[ok]   with every advantage 1, the loss is the mean over replies of each reply's summed cross-entropy / budget")

    opt = SGD(model.parameters(), lr=0.5)
    loss = policy_loss(model, rows, [1.0, -1.0], 2, pad=0, budget=budget)
    opt.zero_grad()
    loss.backward()
    opt.step()
    with no_grad():
        after = [reply_loss(model, *make_batch([r], 0)).item() for r in rows]
    assert after[0] < each[0] and after[1] > each[1], (each, after)
    print("[ok]   one update makes the reply with a positive advantage likelier and the other less likely")


if __name__ == "__main__":
    main()
