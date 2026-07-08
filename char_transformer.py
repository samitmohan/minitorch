"""Train a small GPT on character-level text using the minitorch engine.

Attention backprops through minitorch's autograd, same as every other layer.
Trains on an embedded text snippet, so there is nothing to download.

    uv run python char_transformer.py --iters 2000
"""
import argparse

import numpy as np

from minitorch import GPT, Adam
from minitorch.loss import cross_entropy_loss
from minitorch.tensor import Tensor, no_grad

# opening of "Alice's Adventures in Wonderland" (public domain)
TEXT = """Alice was beginning to get very tired of sitting by her sister on the bank,
and of having nothing to do: once or twice she had peeped into the book her sister
was reading, but it had no pictures or conversations in it, and what is the use of a
book, thought Alice, without pictures or conversations? So she was considering in her
own mind, whether the pleasure of making a daisy-chain would be worth the trouble of
getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran
close by her. There was nothing so very remarkable in that; nor did Alice think it so
very much out of the way to hear the Rabbit say to itself, Oh dear! Oh dear! I shall
be late! But when the Rabbit actually took a watch out of its waistcoat-pocket, and
looked at it, and then hurried on, Alice started to her feet, for it flashed across
her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a
watch to take out of it, and burning with curiosity, she ran across the field after
it, and fortunately was just in time to see it pop down a large rabbit-hole under the
hedge."""


def get_batch(data, block_size, batch_size):
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--plot", type=str, default=None, help="save loss curve to this path")
    args = parser.parse_args()

    np.random.seed(args.seed)

    chars = sorted(set(TEXT))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = np.array([stoi[c] for c in TEXT], dtype=np.int64)
    vocab_size = len(chars)
    print(f"corpus: {len(data)} chars, vocab: {vocab_size}")

    model = GPT(vocab_size, args.block_size, args.n_embd, args.n_head, args.n_layer, dropout=0.1)
    n_params = sum(p.data.size for p in model.parameters())
    print(f"model: {n_params} params, {args.n_layer} layers, {args.n_head} heads")

    opt = Adam(model.parameters(), lr=args.lr)
    model.train()

    losses = []
    for it in range(args.iters):
        x, y = get_batch(data, args.block_size, args.batch_size)
        logits = model(x)
        B, T, V = logits.shape
        loss = cross_entropy_loss(logits.reshape(B * T, V), Tensor(y.reshape(-1)))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
        if it % max(1, args.iters // 20) == 0 or it == args.iters - 1:
            print(f"iter {it:5d} | loss {float(loss.data):.4f}")

    # sample
    model.eval()
    with no_grad():
        start = np.array([[stoi["A"]]])
        out = model.generate(start, max_new_tokens=200, temperature=0.8)[0]
    print("\nsample:")
    print("".join(itos[i] for i in out))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot(losses, lw=0.8)
        plt.xlabel("iteration")
        plt.ylabel("cross-entropy loss")
        plt.title("char-level GPT on minitorch")
        plt.tight_layout()
        plt.savefig(args.plot, dpi=120)
        print(f"\nsaved loss curve to {args.plot}")


if __name__ == "__main__":
    main()
