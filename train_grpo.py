"""GRPO on GSM8K (CS336 assignment 5): Qwen3-0.6B learns from its own sampled solutions (SPEC.md).

    CUPY_TF32=1 uv run --no-dev python train_grpo.py --run grpo_gsm8k --device cuda    # bqa3
    CUPY_TF32=1 uv run --no-dev python train_grpo.py --run grpo_smoke --device cuda --steps 2

Each step samples --group replies to each of --questions training problems from the current
model, all in one batch, and scores a reply 1 when it stops on its own and its final number
equals the reference (bench/gsm8k.py's rule), and 0 otherwise. A reply's advantage is its score minus its group's
mean, over the group's standard deviation. The update raises the log-probability of replies
with a positive advantage and lowers the rest. The model trains through a LoRA adapter, as in
train_lora.py. Writes runs/<run>/config.json, metrics.jsonl, samples.jsonl, and adapter.npz.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from bench.gsm8k import PROMPT, final_number
from minitorch import AdamW, clip_grad_norm
from minitorch import functional as F
from minitorch.lora import apply_lora, save_adapter
from minitorch.qwen3 import chat_prompt, load_qwen3, load_tokenizer
from train_lora import make_batch, reply_logits


def load_questions(path):
    """(question, reference answer) for every problem in a GSM8K .jsonl file."""
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return [(r["question"], float(r["answer"].split("####")[-1].strip().replace(",", ""))) for r in rows]


def reward(text, gold, stopped):
    """1 for a reply that stops on its own (emits <|im_end|>) with the right final number, else 0.

    Stopping is part of the answer. A run that scored the final number alone (runs/grpo_no_stop)
    learned by step 49 to never stop: every reply ran to the token limit, a correct answer
    followed by "---" separators, because nothing cost it anything to keep writing.
    """
    got = final_number(text)
    return float(stopped and got is not None and abs(got - gold) < 1e-6)


def sample_replies(model, tok, questions, group, max_new, temperature, eos):
    """group replies to each question, sampled in one left-padded batch.

    Returns (prompt ids, reply ids) per reply, question by question. A reply ends at its first
    eos, which it keeps, so training also teaches the model when to stop.
    """
    prompts = [tok.encode(chat_prompt([{"role": "user", "content": PROMPT.format(question=q)}])) for q, _ in questions]
    prompts = [p for p in prompts for _ in range(group)]
    width = max(len(p) for p in prompts)
    pad = np.array([width - len(p) for p in prompts])
    ids = np.array([[eos] * n + p for n, p in zip(pad, prompts)])  # any id works as padding: attention hides it
    out = model.generate(ids, max_new, temperature=temperature, eos_id=eos, pad=pad)[:, width:].tolist()
    replies = [r[:r.index(eos) + 1] if eos in r else r for r in out]
    return prompts, replies


def group_advantages(rewards, group, eps=1e-6):
    """Each reward minus its group's mean, over the group's standard deviation.

    A group that scored the same everywhere gets zeros: it says nothing about which replies were better.
    """
    r = np.asarray(rewards, dtype=np.float32).reshape(-1, group)
    return ((r - r.mean(axis=1, keepdims=True)) / (r.std(axis=1, keepdims=True) + eps)).reshape(-1)


def policy_loss(model, rows, advantages, n_replies, pad, budget):
    """-(1 / n_replies) * the sum over rows of advantage * (the sum of the reply's token log-probs) / budget.

    rows: (prompt + reply ids, prompt length) pairs. Summed over micro-batches, with n_replies the
    whole step's count, the gradient is the step's mean. Every reply divides by the same budget,
    the reply token limit, not by its own length (Dr. GRPO, Liu et al. 2025): dividing by its own
    length gives a long wrong reply a smaller push per token than a short one. A 40-step run that
    divided by length (runs/grpo_mean_length) saw the share of replies cut off rise from 22% to 40%
    by steps 17-24 and fall back to 12% by steps 33-40, so those 40 steps do not settle which is better.
    """
    inputs, targets, mask = make_batch(rows, pad)
    logits, at = reply_logits(model, inputs, mask)
    logp = F.log_softmax(logits, axis=-1)[np.arange(len(at)), targets.reshape(-1)[at]]
    # each reply's tokens sit next to each other in `at`, row after row, so its weight repeats once per token
    weight = np.repeat(np.asarray(advantages) / budget / n_replies, mask.sum(axis=1)).astype(np.float32)
    return -(logp * weight).sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--device", default="cuda", choices=["cpu", "metal", "cuda"])
    p.add_argument("--model", default="data/Qwen3-0.6B")
    p.add_argument("--data", default="data/gsm8k/train.jsonl")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--questions", type=int, default=8, help="problems per step")
    p.add_argument("--group", type=int, default=8, help="replies sampled per problem")
    p.add_argument("--max-new", type=int, default=320, help="the evaluation's reply limit (bench/gsm8k.py)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32)
    # CS336's GRPO fine-tunes every weight at 1e-5; LoRA trains best near 10x the full fine-tuning rate
    # (Schulman et al., "LoRA Without Regret", Thinking Machines, 2025)
    p.add_argument("--lr", type=float, default=1e-4)
    # shortcut: one reply per pass. A reply runs to 320 tokens after a ~150-token prompt, and LoRA's update peaked at
    # 17.9 GB on the 3090 for 2 shorter examples (train_lora.py), so 2 long replies may not fit; activation checkpointing would
    p.add_argument("--micro", type=int, default=1, help="replies per forward pass in the update")
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)
    out = Path("runs") / args.run
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    tok = load_tokenizer(args.model)
    data = load_questions(args.data)
    model = load_qwen3(args.model).to(args.device)
    apply_lora(model, args.rank, args.alpha)
    params = model.parameters()
    opt = AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    eos, pad = tok.ids[b"<|im_end|>"], tok.ids[b"<|endoftext|>"]
    order = list(np.random.permutation(len(data)))
    print(f"{args.steps} steps of {args.questions} problems x {args.group} replies -> {out}", flush=True)

    start = time.time()
    with open(out / "metrics.jsonl", "w") as metrics, open(out / "samples.jsonl", "w") as samples:
        for step in range(1, args.steps + 1):
            questions = [data[order.pop()] for _ in range(args.questions)]
            t0 = time.perf_counter()
            prompts, replies = sample_replies(model, tok, questions, args.group, args.max_new, args.temperature, eos)
            t1 = time.perf_counter()
            texts = [tok.decode([t for t in r if t != eos]) for r in replies]
            golds = np.repeat([gold for _, gold in questions], args.group)
            rewards = [reward(text, gold, stopped=r[-1] == eos) for text, gold, r in zip(texts, golds, replies)]
            adv = group_advantages(rewards, args.group)

            # a reply with zero advantage adds nothing to the gradient, so the update skips it;
            # sorted by length, so each micro-batch pads to a similar length
            keep = sorted(np.flatnonzero(adv), key=lambda i: len(prompts[i]) + len(replies[i]))
            opt.zero_grad()
            for j in range(0, len(keep), args.micro):
                chunk = keep[j:j + args.micro]
                loss = policy_loss(model, [(prompts[i] + replies[i], len(prompts[i])) for i in chunk], adv[chunk], len(adv), pad, args.max_new)
                loss.backward()
                del loss  # drop this graph's saved activations before the next forward builds its own
            grad_norm = clip_grad_norm(params, args.clip)
            opt.step()
            t2 = time.perf_counter()

            reply_tokens = sum(len(r) for r in replies)
            by_group = np.reshape(rewards, (-1, args.group))
            row = {"step": step, "reward": round(float(np.mean(rewards)), 4),
                   "groups_with_signal": round(float(np.mean(by_group.max(axis=1) > by_group.min(axis=1))), 3),
                   "reply_tokens": round(reply_tokens / len(replies), 1),
                   "hit_limit": round(float(np.mean([r[-1] != eos for r in replies])), 3),  # cut off: never emitted <|im_end|>
                   "grad_norm": round(float(grad_norm), 4), "rollout_s": round(t1 - t0, 1), "update_s": round(t2 - t1, 1),
                   "rollout_tokens_per_s": round(reply_tokens / (t1 - t0)), "time": round(time.time() - start, 1)}
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            samples.write(json.dumps({"step": step, "question": questions[0][0], "text": texts[0], "reward": rewards[0]}) + "\n")
            samples.flush()
            print(f"step {step:4d}/{args.steps} | reward {row['reward']:.3f} | reply {row['reply_tokens']:.0f} tok | "
                  f"rollout {row['rollout_s']:.0f} s ({row['rollout_tokens_per_s']:,} tok/s) | update {row['update_s']:.0f} s", flush=True)
            if step % 10 == 0 or step == args.steps:
                save_adapter(model, out / "adapter.npz", args.rank, args.alpha)


if __name__ == "__main__":
    main()
