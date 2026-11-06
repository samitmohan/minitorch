"""Retrieval on minitorch (SPEC.md M5): Qwen3-Embedding-0.6B finds the passage, Qwen3-0.6B on the server answers.

    uv run uvicorn minitorch.serve.api:app --port 8000     # the server, if it is not already up
    uv run python apps/rag.py                               # writes bench/results/rag.json

The corpus is SQuAD v1.1's dev set, 2,067 Wikipedia paragraphs. minitorch embeds each paragraph
once: Qwen3-Embedding is Qwen3's architecture, and its embedding is the last token's hidden
state, scaled to length 1. The index is a NumPy matrix, and a query's nearest paragraphs are
its largest dot products. For --n questions drawn at random, the script measures retrieval (is
the paragraph the question was written from among the top k?) and asks the server for an
answer twice, closed-book and with the top k paragraphs in the prompt. Answers score by SQuAD's
rules: exact match and token F1 against any reference answer, after normalizing both, plus whether
the answer contains a reference as whole words, since a chat model answers in sentences.
"""
import argparse
import collections
import json
import re
import string
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from minitorch import backend, no_grad
from minitorch.qwen3 import load_qwen3, load_tokenizer

EMBEDDER = ROOT / "data/Qwen3-Embedding-0.6B"
INDEX = ROOT / "data/squad/qwen3_embedding_index.npy"
INSTRUCT = "Instruct: Given a question, retrieve Wikipedia passages that answer the question\nQuery:"


def embed(model, tok, texts, device, batch=16):
    """Unit vectors for texts: the last token's final hidden state, Qwen3-Embedding's pooling."""
    eos = tok.ids[b"<|endoftext|>"]
    ids = [tok.encode(t) + [eos] for t in texts]
    out = np.zeros((len(texts), model.embed.weight.shape[1]), np.float32)
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))  # similar lengths share a batch, so little padding
    with no_grad():
        for start in range(0, len(order), batch):
            rows = [ids[i] for i in order[start:start + batch]]
            width = max(len(r) for r in rows)
            pad = np.array([width - len(r) for r in rows])  # left padding: every row's last token sits in the last column
            h = model.hidden(np.array([[eos] * n + r for n, r in zip(pad, rows)]), None, pad if pad.any() else None)
            out[order[start:start + batch]] = h[:, -1].numpy()
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def corpus():
    """SQuAD v1.1 dev: (article title, paragraph) pairs, and every question as (question, reference answers, paragraph index)."""
    articles = json.loads((ROOT / "data/squad/dev-v1.1.json").read_text())["data"]
    paragraphs = [(a["title"].replace("_", " "), para["context"]) for a in articles for para in a["paragraphs"]]
    questions = [(q["question"], [g["text"] for g in q["answers"]], i)
                 for i, para in enumerate(p for a in articles for p in a["paragraphs"]) for q in para["qas"]]
    return paragraphs, questions


def rag_prompt(question, passages):
    """The answer prompt: the retrieved paragraphs, numbered, then the question."""
    numbered = "\n\n".join(f"[{j + 1}] {text}" for j, text in enumerate(passages))
    return f"Answer with a short phrase copied from the passages, no explanation.\n\n{numbered}\n\nQuestion: {question}"


def ask(server, prompt):
    """The server's greedy reply to one user message, and the seconds it took."""
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 32}).encode()
    request = urllib.request.Request(f"{server}/v1/chat/completions", body, {"content-type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"], time.perf_counter() - start


def normalize(text):
    """SQuAD's answer normalization: lowercase, no punctuation, no articles, single spaces."""
    text = "".join(c for c in text.lower() if c not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def scores(answer, golds):
    """(exact match, token F1, contains) against the best of the reference answers.

    Contains asks whether a reference appears in the answer as whole words: a chat model that answers
    "It is compared to the mitochondrial double membrane." is right, and exact match scores it 0.
    """
    def f1(gold):
        a, g = normalize(answer).split(), normalize(gold).split()
        common = sum((collections.Counter(a) & collections.Counter(g)).values())
        return 0.0 if common == 0 else 2 * common / (len(a) + len(g))
    contains = max(float(f" {normalize(g)} " in f" {normalize(answer)} ") for g in golds)
    return max(float(normalize(answer) == normalize(g)) for g in golds), max(f1(g) for g in golds), contains


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://localhost:8000")
    p.add_argument("--device", default="metal")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--k", type=int, default=3)
    args = p.parse_args()

    paragraphs, questions = corpus()
    questions = [questions[i] for i in np.random.default_rng(0).choice(len(questions), args.n, replace=False)]

    tok = load_tokenizer(EMBEDDER)
    model = load_qwen3(EMBEDDER).to(args.device)
    build_time = INDEX.with_suffix(".json")  # the corpus never changes, so it is embedded once, and that time kept beside it
    if INDEX.exists():
        index, index_s = np.load(INDEX), json.loads(build_time.read_text())["seconds"]
    else:
        start = time.perf_counter()
        index = embed(model, tok, [text for _, text in paragraphs], args.device)
        index_s = time.perf_counter() - start
        np.save(INDEX, index)
        build_time.write_text(json.dumps({"seconds": round(index_s, 1)}))
    start = time.perf_counter()
    queries = embed(model, tok, [INSTRUCT + q for q, _, _ in questions], args.device)
    query_ms = (time.perf_counter() - start) / len(questions) * 1000
    ranked = np.argsort(-(queries @ index.T), axis=1)[:, :args.k]  # the search: one matrix product, then a sort
    del model  # the server loads its own model on the first question; one model in memory at a time
    backend.get(args.device).clear_cache()
    print(f"embedded {len(paragraphs):,} paragraphs in {index_s:.0f} s, {query_ms:.0f} ms per question", flush=True)

    rows = []
    for (question, golds, source), top in zip(questions, ranked):
        closed, closed_s = ask(args.server, f"Answer with a short phrase, no explanation.\nQuestion: {question}")
        opened, open_s = ask(args.server, rag_prompt(question, [paragraphs[i][1] for i in top]))
        rows.append({"question": question, "golds": golds, "source_rank": int(np.flatnonzero(top == source)[0]) + 1 if source in top else None,
                     "closed": closed, "closed_scores": scores(closed, golds), "rag": opened, "rag_scores": scores(opened, golds),
                     "closed_s": round(closed_s, 2), "rag_s": round(open_s, 2)})

    def mean(key, i):
        return round(float(np.mean([r[key][i] for r in rows])), 4)
    summary = {"questions": len(rows), "paragraphs": len(paragraphs), "k": args.k,
               "recall_at_1": round(float(np.mean([r["source_rank"] == 1 for r in rows])), 4),
               f"recall_at_{args.k}": round(float(np.mean([r["source_rank"] is not None for r in rows])), 4),
               "closed_exact": mean("closed_scores", 0), "closed_f1": mean("closed_scores", 1), "closed_contains": mean("closed_scores", 2),
               "rag_exact": mean("rag_scores", 0), "rag_f1": mean("rag_scores", 1), "rag_contains": mean("rag_scores", 2),
               "index_seconds": round(index_s, 1), "query_embed_ms": round(query_ms, 1),
               "closed_answer_s": round(float(np.mean([r["closed_s"] for r in rows])), 2),
               "rag_answer_s": round(float(np.mean([r["rag_s"] for r in rows])), 2)}
    (ROOT / "bench/results/rag.json").write_text(json.dumps(summary, indent=1))
    with open(ROOT / "bench/results/rag_answers.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
