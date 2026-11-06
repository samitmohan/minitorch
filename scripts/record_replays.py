"""Record the post's live figures against a running server, for a static copy of the post that replays them.

Start the server first (uv run uvicorn minitorch.serve.api:app --port 8000), then:
    uv run python scripts/record_replays.py
Writes bench/replays/<figure>.json. Each request is the one the live figure sends; the chat records when each streamed
chunk arrived, so a replay can show the reply at the speed it was written.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "bench" / "replays"

GRAPHS = [
    ("relu(x * w + b) ** 2", {"x": 2, "w": -0.5, "b": 3}),
    ("x * x + x", {"x": 3}),
    ("log(exp(a) + exp(b))", {"a": 1, "b": 2}),
    ("sigmoid(sum(w * v))", {"w": [0.5, -1, 2], "v": [1, 2, 3]}),
]
BPE = [
    "Once upon a time, a curious little robot found a shiny gear.",
    "The dragon Zorblax ate 1234 cookies and said, \"Yum!\"",
    "Lily and Ben were best friends. They played in the garden every day.",
]
ATTENTION = [
    "Once upon a time, there was a little girl named Lily. Lily loved to play with her red ball.",
    "Tom had a big dog named Max. Every day, Tom and Max went to the park. One day, Tom and Max",
]
STORY_PROMPTS = ["Once upon a time, there was a little", "Tom and his dog went to the park. They saw", "The little bird was sad because"]
TEMPERATURES = [0.0, 0.8, 1.5]
CHATS = [
    "What is 17 times 23? Show the steps briefly.",
    "Write a haiku about a GPU.",
]
RAG = [  # RagDemo's examples: questions from the SQuAD development set the index covers
    "What language did Tesla study while in school?",
    "Where was the Charles Porter steam engine indicator shown?",
    "What is the chloroplast double membrane sometimes compared to?",
]
AGENT = [
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
     "How many clips did Natalia sell altogether in April and May?"),
]


def post(url, body, timeout=600):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as res:
        out = json.loads(res.read())
    return out, time.perf_counter() - start


def chat(url, text):
    """One streamed reply, as the Chat figure sends it: [(seconds since the request, text added)], and the usage."""
    body = {"messages": [{"role": "user", "content": text}], "stream": True, "max_tokens": 400, "temperature": 0.7}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    start = time.perf_counter()
    chunks, usage = [], None
    with urllib.request.urlopen(req, timeout=600) as res:
        for line in res:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            if "error" in data:
                raise RuntimeError(data["error"]["message"])
            delta = data["choices"][0]["delta"].get("content")
            if delta:
                chunks.append([round(time.perf_counter() - start, 3), delta])
            usage = data.get("usage", usage)
    return {"message": text, "chunks": chunks, "completion_tokens": usage["completion_tokens"],
            "seconds": round(time.perf_counter() - start, 3)}


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False) + "\n")
    print(f"{name}: {len(json.dumps(data)) / 1024:.0f} KB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--only", nargs="*", help="record these figures only")
    args = p.parse_args()
    u = args.url
    want = set(args.only or ["graph", "gradcheck", "bpe", "attention", "generate", "chat", "rag", "agent"])

    if "graph" in want:
        save("graph", [{"expr": e, "variables": v, "graph": post(u + "/api/graph", {"expr": e, "variables": v})[0]} for e, v in GRAPHS])
    if "gradcheck" in want:
        results, seconds = post(u + "/api/gradcheck", {})
        save("gradcheck", {"results": results, "seconds": round(seconds, 2)})
    if "bpe" in want:
        save("bpe", [{"text": t, **post(u + "/api/bpe", {"text": t})[0]} for t in BPE])
    if "attention" in want:
        save("attention", [post(u + "/api/attention", {"text": t})[0] for t in ATTENTION])
    if "generate" in want:
        post(u + "/api/generate", {"run": "tinystories", "prompt": "Once", "max_tokens": 8})  # loads the checkpoint
        rows = []
        for prompt in STORY_PROMPTS:
            for t in TEMPERATURES:
                req = {"run": "tinystories", "prompt": prompt, "max_tokens": 120, "temperature": t, "top_p": 0.95}
                out, seconds = post(u + "/api/generate", req)
                rows.append(req | out | {"seconds": round(seconds, 3)})
        save("generate", rows)
    if "chat" in want:
        chat(u, "Hi")  # the first request loads Qwen3-0.6B onto the GPU; the figure's note says so
        save("chat", [chat(u, m) for m in CHATS])
    if "rag" in want:
        post(u + "/api/rag", {"question": RAG[0]})  # loads the embedder and the index
        save("rag", [{"question": q, **post(u + "/api/rag", {"question": q})[0]} for q in RAG])
    if "agent" in want:
        rows = []
        for q in AGENT:
            out, seconds = post(u + "/api/agent", {"question": q})
            rows.append({"question": q, **out, "seconds": round(seconds, 2)})
        save("agent", rows)


if __name__ == "__main__":
    main()
