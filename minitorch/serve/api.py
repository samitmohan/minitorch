"""HTTP API behind the UI: training runs, autograd graphs, BPE traces, gradient checks, generation,
and an OpenAI-compatible chat endpoint serving Qwen3-0.6B through minitorch's engine.

    uv run uvicorn minitorch.serve.api:app --port 8000

Serves ui/dist at / when it has been built (cd ui && npm run build).
"""
import ast
import asyncio
import json
import operator
import sys
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from minitorch import Tensor, TransformerLM, no_grad
from minitorch.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
IDLE_S = 120  # seconds without a new metrics line before a run counts as finished
MAX_PROMPT_CHARS = 64_000  # far past what a 4k-token context holds
EOT = "<|endoftext|>"
app = FastAPI(title="minitorch")
# no_grad flips one process-wide flag, so requests that run a model must not overlap
_model_lock = threading.Lock()


# training runs

def _run_dir(name):
    path = (RUNS / name).resolve()
    if path.parent != RUNS.resolve() or not (path / "metrics.jsonl").exists():
        raise HTTPException(404, f"no run named {name!r}")
    return path


def _parse(line):
    """One metrics line. A diverged run logs NaN, which Python's json writes but browsers and strict JSON reject: it becomes null."""
    return json.loads(line, parse_constant=lambda _: None)


def _read_jsonl(path):
    return [_parse(line) for line in path.open()] if path.exists() else []


@app.get("/api/runs")
def runs():
    out = []
    for d in sorted(RUNS.glob("*/metrics.jsonl")):
        rows = _read_jsonl(d)
        out.append({"name": d.parent.name, "config": json.loads((d.parent / "config.json").read_text()),
                    "steps": len(rows), "last": rows[-1] if rows else None,
                    "has_checkpoint": (d.parent / "ckpt.npz").exists()})
    return out


@app.get("/api/runs/{name}/samples")
def samples(name: str):
    return _read_jsonl(_run_dir(name) / "samples.jsonl")


@app.get("/api/runs/{name}/stream")
async def stream(name: str, offset: int = 0):
    """Server-sent events: every metrics line from byte `offset` on, as the training script appends them."""
    path = _run_dir(name) / "metrics.jsonl"

    async def lines():
        pos = offset
        while True:
            with path.open() as f:
                f.seek(pos)
                chunk = f.read()
            # only send complete lines; a partial one waits for its newline
            end = chunk.rfind("\n") + 1
            for line in chunk[:end].splitlines():
                yield f"data: {json.dumps(_parse(line))}\n\n"
            pos += len(chunk[:end].encode())
            # a run that stopped writing is finished: end the stream, so a page that shows several finished runs does
            # not hold the browser's six connections per host open forever (the slowest live run, GRPO, writes every ~47 s)
            if time.time() - path.stat().st_mtime > IDLE_S:
                yield "event: end\ndata: {}\n\n"
                return
            await asyncio.sleep(1.0)

    return StreamingResponse(lines(), media_type="text/event-stream")


# autograd graph for a typed expression

BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
          ast.Pow: operator.pow, ast.MatMult: operator.matmul}
FUNCS = {"exp": Tensor.exp, "log": Tensor.log, "relu": Tensor.relu, "sigmoid": Tensor.sigmoid,
         "tanh": Tensor.tanh, "sum": Tensor.sum, "mean": Tensor.mean}


def _eval(node, env):
    """Evaluate an expression AST using only Tensor ops: no names, calls, or attributes outside the allowlist."""
    if isinstance(node, ast.BinOp) and type(node.op) in BINOPS:
        left, right = _eval(node.left, env), _eval(node.right, env)
        if isinstance(node.op, ast.Pow) and not isinstance(left, Tensor):
            left = Tensor(left)  # 2 ** x: a constant base needs the tensor-exponent path
        return BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand, env)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS \
            and len(node.args) == 1 and not node.keywords:
        arg = _eval(node.args[0], env)
        return FUNCS[node.func.id](arg if isinstance(arg, Tensor) else Tensor(arg))  # exp(2) as well as exp(x)
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)  # stays a float so x ** 2 is one pow node, not exp(2 log x)
    raise HTTPException(400, f"unsupported: {ast.unparse(node)}")


class GraphRequest(BaseModel):
    expr: str = Field(max_length=300)
    variables: dict[str, float | Annotated[list[float], Field(max_length=64)]] = Field(max_length=8)


def _small(a):
    """A small array as JSON. log(-1) or x / 0 make NaN and infinities, which JSON cannot carry: they become null."""
    if a is None or a.size > 16:
        return None
    a = np.asarray(a, dtype=float)
    return np.where(np.isfinite(a), np.round(a, 6), None).tolist()


@app.post("/api/graph")
def graph(req: GraphRequest):
    try:
        with _model_lock:
            env = {k: Tensor(v, requires_grad=True) for k, v in req.variables.items() if k.isidentifier()}
            out = _eval(ast.parse(req.expr, mode="eval").body, env)
            if not isinstance(out, Tensor):
                raise TypeError("the expression must use at least one variable")
            root = out if out.data.size == 1 else out.sum()
            root.backward(keep_grads=True)
    except (SyntaxError, ValueError, TypeError, ArithmeticError, AttributeError) as e:  # AttributeError: x @ 2
        raise HTTPException(400, str(e)) from e

    names = {id(t): k for k, t in env.items()}
    order = _walk(root)
    ids = {id(t): i for i, t in enumerate(order)}
    nodes, edges = [], []
    for t in order:
        label = names.get(id(t)) or (t._op if t._ctx else "const")
        nodes.append({"id": ids[id(t)], "label": label, "op": t._op, "shape": list(t.shape),
                      "value": _small(t.data), "grad": _small(t.grad), "leaf": t._ctx is None,
                      "requires_grad": t.requires_grad})
        for p in t._ctx.parents if t._ctx else []:
            edges.append({"source": ids[id(p)], "target": ids[id(t)]})
    return {"nodes": nodes, "edges": edges, "root": ids[id(root)]}


def _walk(root):
    seen, order, stack = set(), [], [root]
    while stack:
        t = stack.pop()
        if id(t) not in seen:
            seen.add(id(t))
            order.append(t)
            stack.extend(t._ctx.parents if t._ctx else [])
    return order


# tokenizer

@lru_cache(maxsize=1)
def _tokenizer():
    bpe = ROOT / "data" / "tinystories_bpe"
    if (bpe / "vocab.json").exists():
        return "tinystories-10k", Tokenizer.from_files(bpe / "vocab.json", bpe / "merges.txt", [EOT])
    from huggingface_hub import hf_hub_download
    return "gpt2", Tokenizer.from_files(hf_hub_download("openai-community/gpt2", "vocab.json"),
                                         hf_hub_download("openai-community/gpt2", "merges.txt"), [EOT])


class TextRequest(BaseModel):
    text: str = Field(max_length=2000)


@app.post("/api/bpe")
def bpe(req: TextRequest):
    name, tok = _tokenizer()
    ids = tok.encode(req.text)
    return {"tokenizer": name, "ids": ids,
            "tokens": [tok.vocab[i].decode("utf-8", errors="replace") for i in ids],
            "trace": tok.trace(req.text)}


# gradient checks: the same registry test_grads.py runs

@app.post("/api/gradcheck")
def gradcheck():
    sys.path.insert(0, str(ROOT))
    import test_grads
    np.random.seed(0)
    results = []
    with _model_lock:
        for name, fn in test_grads.CHECKS:
            try:
                fn()
                results.append({"name": name, "ok": True})
            except Exception as e:  # noqa: BLE001 - a crashing check is a failed check
                results.append({"name": name, "ok": False, "error": repr(e)[:300]})
    return results


# generation from a run's checkpoint

@lru_cache(maxsize=2)
def _model(name, mtime):
    cfg = json.loads((RUNS / name / "config.json").read_text())
    model = TransformerLM(cfg["vocab_size"], cfg["context"], cfg["d_model"], cfg["layers"], cfg["heads"], cfg["d_ff"],
                          norm=cfg.get("norm", "pre"), rope=not cfg.get("no_rope", False), ffn=cfg.get("ffn", "swiglu"))
    return model.load(RUNS / name / "ckpt.npz").eval()


class GenerateRequest(BaseModel):
    run: str
    prompt: str = Field(max_length=1000)
    max_tokens: int = Field(100, ge=1, le=512)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_p: float = Field(0.95, gt=0.0, le=1.0)


@app.post("/api/generate")
def generate(req: GenerateRequest):
    ckpt = _run_dir(req.run) / "ckpt.npz"
    if not ckpt.exists():
        raise HTTPException(404, f"run {req.run!r} has no checkpoint yet")
    model = _model(req.run, ckpt.stat().st_mtime)
    _, tok = _tokenizer()
    prompt = tok.encode(req.prompt) or [tok.ids[EOT.encode()]]
    max_new = min(req.max_tokens, model.context_length - len(prompt))
    if max_new < 1:
        raise HTTPException(400, "prompt fills the whole context")
    with _model_lock:
        ids = model.generate(np.array([prompt]), max_new, req.temperature, top_p=req.top_p, eos_id=tok.ids[EOT.encode()])
    return {"text": tok.decode(ids[0, len(prompt):].tolist()), "prompt_tokens": len(prompt),
            "new_tokens": int(ids.shape[1] - len(prompt))}


# chat: Qwen3-0.6B on the Apple GPU, through the scheduler and engine (minitorch/serve)

QWEN = ROOT / "data" / "Qwen3-0.6B"
_chat = None
_chat_lock = threading.Lock()


def _qwen():
    """(tokenizer, engine), loaded on the first chat request: about 20 s, once."""
    global _chat
    with _chat_lock:
        if _chat is None:
            if not QWEN.exists():
                raise HTTPException(503, "Qwen3-0.6B is not downloaded into data/Qwen3-0.6B")
            from minitorch.qwen3 import load_qwen3, load_tokenizer
            from minitorch.serve.engine import Engine
            from minitorch.serve.kv_cache import BlockManager
            from minitorch.serve.runner import MinitorchRunner
            from minitorch.serve.scheduler import Scheduler
            model = load_qwen3(QWEN).to("metal", "bfloat16")
            _chat = load_tokenizer(QWEN), Engine(Scheduler(MinitorchRunner(model), BlockManager(512, 16)), _model_lock)
    return _chat


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(max_length=16000)  # about 4,000 tokens: three of SQuAD's longest paragraphs and a question


class ChatRequest(BaseModel):
    model: str = "qwen3-0.6b"
    messages: list[Message] = Field(min_length=1, max_length=50)
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.95, gt=0, le=1)
    stream: bool = False
    # function schemas the model may call; its <tool_call> blocks come back inside the message text, as Qwen3 writes them
    tools: list[dict] | None = Field(default=None, max_length=16)


def _replies(tok, tokens, eos):
    """(text so far, finish reason) each time the reply grows by whole characters; the finish reason comes last."""
    ids, finish = [], "length"
    for t in iter(tokens.get, None):
        if isinstance(t, Exception):
            raise t
        if t == eos:
            finish = "stop"
            continue
        ids.append(t)
        text = tok.decode(ids)
        if not text.endswith("\ufffd"):  # a character split across tokens waits for its other bytes
            yield text, None, len(ids)
    yield tok.decode(ids), finish, len(ids)


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    from minitorch.qwen3 import chat_prompt

    tok, engine = _qwen()
    text = chat_prompt([m.model_dump() for m in req.messages], tools=req.tools)
    if len(text) > MAX_PROMPT_CHARS:  # checked before encoding: BPE on 16k characters of one word takes seconds
        raise HTTPException(400, f"the prompt has {len(text):,} characters; the limit is {MAX_PROMPT_CHARS:,}")
    prompt = tok.encode(text)
    eos = tok.ids[b"<|im_end|>"]
    try:
        tokens = engine.submit(prompt, req.max_tokens, req.temperature, req.top_p, eos)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    head = {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "created": int(time.time()), "model": req.model}

    def usage(n):
        return {"prompt_tokens": len(prompt), "completion_tokens": n, "total_tokens": len(prompt) + n}

    if not req.stream:
        text, finish, n = list(_replies(tok, tokens, eos))[-1]
        return head | {"object": "chat.completion", "usage": usage(n),
                       "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}]}

    def events():
        # shortcut: a client that disconnects leaves its request generating to max_tokens; cancel it in the engine if that wastes real time
        def chunk(delta, finish=None, **extra):
            body = head | {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]} | extra
            return f"data: {json.dumps(body)}\n\n"

        yield chunk({"role": "assistant"})
        sent = ""
        try:
            for text, finish, n in _replies(tok, tokens, eos):
                if text != sent:
                    yield chunk({"content": text[len(sent):]})
                    sent = text
                if finish:
                    yield chunk({}, finish, usage=usage(n))
        except Exception as e:  # noqa: BLE001 - the stream has started, so the error travels in it
            yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


class AttentionRequest(BaseModel):
    run: str = "tinystories"
    text: str = Field(max_length=400)


@app.post("/api/attention")
def attention_patterns(req: AttentionRequest):
    """Every head's attention weights for a text: softmax(q k^T / sqrt(d)) with the causal mask, per layer."""
    ckpt = _run_dir(req.run) / "ckpt.npz"
    if not ckpt.exists():
        raise HTTPException(404, f"run {req.run!r} has no checkpoint yet")
    model = _model(req.run, ckpt.stat().st_mtime)
    _, tok = _tokenizer()
    ids = tok.encode(req.text)[:40] or [tok.ids[EOT.encode()]]
    blocks = [b.attn for b in model.blocks]
    with _model_lock, no_grad():
        for a in blocks:
            a.record = True
        try:
            model(np.array([ids]))
            recorded = [a.recorded for a in blocks]
        finally:  # left on, every later generate would copy each layer's queries and keys
            for a in blocks:
                a.record, a.recorded = False, None
    layers = []
    for q, k in recorded:  # (1, H, T, D), (1, Hkv, T, D)
        k = np.repeat(k, q.shape[1] // k.shape[1], axis=1)
        s = q[0] @ k[0].swapaxes(-1, -2) / np.sqrt(q.shape[-1]) + np.triu(np.full((len(ids),) * 2, -np.inf), k=1)
        p = np.exp(s - s.max(-1, keepdims=True))
        layers.append(np.round(p / p.sum(-1, keepdims=True), 3).tolist())
    return {"tokens": [tok.decode([i]) for i in ids], "layers": layers}


# the post's two applications, live: they run apps/rag.py's and apps/agent.py's own code on this server's engine

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


def _complete(messages, max_tokens, tools=None):
    """One greedy reply from the chat engine, in process: (text, prompt tokens, reply tokens, seconds)."""
    from minitorch.qwen3 import chat_prompt

    tok, engine = _qwen()
    prompt = tok.encode(chat_prompt(messages, tools=tools))
    eos = tok.ids[b"<|im_end|>"]
    start = time.perf_counter()
    try:
        tokens = engine.submit(prompt, max_tokens, 0.0, 1.0, eos)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    text, _, n = list(_replies(tok, tokens, eos))[-1]
    return text, len(prompt), n, time.perf_counter() - start


@lru_cache(maxsize=1)
def _retriever():
    """(tokenizer, Qwen3-Embedding, SQuAD paragraphs, index), loaded on the first question."""
    # the apps sit on top of minitorch, so they are imported here, not at the top of this module
    from apps.rag import EMBEDDER, INDEX, corpus
    from minitorch.qwen3 import load_qwen3, load_tokenizer

    if not INDEX.exists():
        raise HTTPException(503, "no index yet: run uv run python apps/rag.py once to build it")
    return load_tokenizer(EMBEDDER), load_qwen3(EMBEDDER).to("metal"), corpus()[0], np.load(INDEX)


@app.post("/api/rag")
def rag(req: AskRequest):
    """Embed the question, take the three nearest SQuAD paragraphs, and answer from them."""
    from apps.rag import INSTRUCT, embed, rag_prompt

    with _model_lock:
        tok, model, paragraphs, index = _retriever()
        start = time.perf_counter()
        query = embed(model, tok, [INSTRUCT + req.question], "metal")[0]
        embed_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        scores = index @ query  # the search: one matrix-vector product and a sort
        top = np.argsort(-scores)[:3]
        search_ms = (time.perf_counter() - start) * 1000
    answer, _, _, seconds = _complete([{"role": "user", "content": rag_prompt(req.question, [paragraphs[i][1] for i in top])}], 32)
    return {"passages": [{"title": paragraphs[i][0], "text": paragraphs[i][1], "score": round(float(scores[i]), 4)} for i in top],
            "answer": answer, "embed_ms": round(embed_ms, 1), "search_ms": round(search_ms, 2), "answer_s": round(seconds, 2)}


@app.post("/api/agent")
def agent(req: AskRequest):
    """apps/agent.py's loop on this server: the model calls the calculator until it answers without a call."""
    from apps.agent import TOOLS, solve
    from bench.gsm8k import final_number

    def ask(messages):
        return _complete(messages, 320, tools=TOOLS)
    reply, trace = solve(ask, req.question)
    return {"reply": reply, "answer": final_number(reply), "trace": trace}


# benchmark results

@app.get("/api/bench")
def bench():
    return {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "bench" / "results").glob("*.json"))}


app.mount("/results", StaticFiles(directory=ROOT / "bench" / "results"), name="results")  # the post's images
if (ROOT / "ui" / "dist").exists():
    app.mount("/", StaticFiles(directory=ROOT / "ui" / "dist", html=True), name="ui")
