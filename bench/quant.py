"""What quantizing Qwen3-0.6B's weights costs in quality, and what it buys in decode speed.

    uv run python bench/quant.py

For bfloat16, int8, and int4 weights (group size 64) on the M4 GPU:
- perplexity on two held-out texts: GSM8K reference solutions and TinyStories stories,
  in 512-token windows, so the quality cost shows up as one number per text;
- decode speed: the median time of one single-token step against a KV cache, with the three\n  precisions (and int8/int4 on MLX's kernel) taking turns step by step;
- the bandwidth ceiling: decoding reads every weight once per token, so no decoder can beat
  (memory bandwidth) / (bytes of weights). The bandwidth is measured here, with a large copy;
- the matmuls alone, for one token, on Qwen3's largest and most common shapes: bfloat16, then
  int8 and int4 through minitorch's Metal kernel and through MLX's (the oracle).
Writes bench/results/quant.json.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from minitorch import cross_entropy_loss, no_grad
from minitorch.attention import KVCache
from minitorch.backend import metal
from minitorch.backend.metal_quant import quant_matmul
from minitorch.quant import QuantLinear, quantize, quantize_model
from minitorch.qwen3 import load_qwen3, load_tokenizer

RESULTS = Path(__file__).resolve().parent / "results"
MODEL = "data/Qwen3-0.6B"
WINDOW, WINDOWS = 512, 12  # 6,144 tokens per text


def texts(tok):
    with open("data/gsm8k/test.jsonl") as f:
        gsm = "\n\n".join(json.loads(line)["answer"] for line in f)
    with open("data/TinyStoriesV2-GPT4-valid.txt", encoding="utf-8") as f:
        stories = f.read(400_000)
    return {name: tok.encode(t)[:WINDOW * WINDOWS] for name, t in [("gsm8k", gsm), ("tinystories", stories)]}


def perplexity(model, ids):
    with no_grad():
        # logits in float32 for the loss: a bfloat16 softmax over 151,936 entries would round away the answer
        losses = [cross_entropy_loss(model(np.array([ids[i:i + WINDOW]])).cast("float32"), np.array(ids[i + 1:i + WINDOW + 1])[None]).item()
                  for i in range(0, len(ids) - WINDOW, WINDOW)]
    return float(np.exp(np.mean(losses)))


def weight_bytes(model):
    """Bytes a decode step reads from weight matrices (embedding lookup aside: one row per token)."""
    total = 0
    for block in model.blocks:
        for parent in (block.attn, block.ffn):
            for val in parent.__dict__.values():
                if isinstance(val, QuantLinear):
                    total += val.nbytes()
                elif hasattr(val, "weight") and val.weight.ndim == 2:
                    total += val.weight.data.nbytes
    head = model.head
    total += head.nbytes() if isinstance(head, QuantLinear) else model.embed.weight.data.nbytes
    return total


def per_call(fn, n=50):
    mx.eval(fn())
    t0 = time.perf_counter()
    mx.eval([fn() for _ in range(n)])
    return (time.perf_counter() - t0) / n


def kernel_rows():
    rows = []
    for name, n_out, n_in in [("output head", 151936, 1024), ("feed-forward up", 3072, 1024), ("feed-forward down", 1024, 3072)]:
        w = np.random.default_rng(0).standard_normal((n_out, n_in)).astype(np.float32) * 0.02
        x = mx.array(np.random.default_rng(1).standard_normal((1, n_in)).astype(np.float32)).astype(mx.bfloat16)
        wb = mx.array(w).astype(mx.bfloat16)
        t = per_call(lambda x=x, wb=wb: x @ wb.T)
        rows.append({"matrix": name, "impl": "bfloat16", "us": round(t * 1e6, 1), "gb_s": round(wb.nbytes / t / 1e9, 1)})
        for bits in (8, 4):
            p, s, b = quantize(w, bits, 64)
            p, s, b = mx.array(p), mx.array(s).astype(mx.bfloat16), mx.array(b).astype(mx.bfloat16)
            nbytes = p.nbytes + s.nbytes + b.nbytes
            args = {"x": x, "p": p, "s": s, "b": b, "bits": bits}
            for impl, fn in [("minitorch", lambda a=args: quant_matmul(a["x"], a["p"], a["s"], a["b"], a["bits"], 64)),
                             ("MLX", lambda a=args: mx.quantized_matmul(a["x"], a["p"], a["s"], a["b"], transpose=True,
                                                                        group_size=64, bits=a["bits"]))]:
                t = per_call(fn)
                rows.append({"matrix": name, "impl": f"int{bits} {impl}", "us": round(t * 1e6, 1), "gb_s": round(nbytes / t / 1e9, 1)})
    return rows


def mlx_kernel(x, packed, scales, biases, bits, group):
    """MLX's quantized matmul, with minitorch's quant_matmul signature: the oracle to swap in."""
    return mx.quantized_matmul(x, packed, scales, biases, transpose=True, group_size=group, bits=bits)


def decode_ms(models, steps=40):
    """Median milliseconds of one single-token decode step, for each (name, model, kernel), interleaved.

    The models take turns, step by step, so they all see the same machine: timing them one after
    another lets the chip's temperature and other work move the numbers by a quarter.
    """
    caches = {name: [KVCache() for _ in model.blocks] for name, model, _ in models}
    times = {name: [] for name, _, _ in models}
    mine = metal.quant_matmul
    with no_grad():
        for name, model, _ in models:
            model(np.arange(1, 17)[None], caches[name]).numpy()
        for t in range(steps):
            for name, model, kernel in models:
                metal.quant_matmul = kernel or mine
                t0 = time.perf_counter()
                model(np.array([[100 + t]]), caches[name])[:, -1].numpy()  # .numpy() waits for the GPU, as sampling does
                times[name].append(time.perf_counter() - t0)
    metal.quant_matmul = mine
    return {name: 1000 * float(np.median(ts[5:])) for name, ts in times.items()}  # the first steps compile kernels


def bandwidth_gb_s():
    a = mx.random.normal((256 * 2**20,))  # 1 GiB of float32
    mx.eval(a)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        mx.eval(a + 1)  # reads 1 GiB and writes 1 GiB
        times.append(time.perf_counter() - t0)
    return 2 * a.nbytes / min(times) / 1e9


def main():
    tok = load_tokenizer(MODEL)
    data = texts(tok)
    bw = bandwidth_gb_s()
    print(f"measured bandwidth {bw:.1f} GB/s")
    models = {}
    for precision, bits in [("bfloat16", None), ("int8", 8), ("int4", 4)]:
        models[precision] = load_qwen3(MODEL).to("metal", "bfloat16")
        if bits:
            quantize_model(models[precision], bits, 64)
    ms = decode_ms([(p, m, None) for p, m in models.items()] + [(f"{p} with MLX's kernel", models[p], mlx_kernel) for p in ("int8", "int4")])
    print({k: round(v, 1) for k, v in ms.items()})
    rows = []
    for precision, model in models.items():
        nbytes = weight_bytes(model)
        row = {"precision": precision, "weight_mb": round(nbytes / 1e6, 1), "decode_ms_per_token": round(ms[precision], 1),
               "decode_tokens_per_s": round(1000 / ms[precision], 1), "ceiling_tokens_per_s": round(bw * 1e9 / nbytes, 1),
               "perplexity": {name: round(perplexity(model, ids), 3) for name, ids in data.items()}}
        if precision != "bfloat16":
            row["decode_ms_per_token_mlx_kernel"] = round(ms[f"{precision} with MLX's kernel"], 1)
        rows.append(row)
        print(row, flush=True)
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], check=False, capture_output=True, text=True).stdout.strip()
    out = {"model": "Qwen3-0.6B", "device": f"{chip} GPU", "group": 64, "window": WINDOW,
           "tokens_per_text": WINDOW * WINDOWS, "bandwidth_gb_s": round(bw, 1), "rows": rows, "kernels": kernel_rows()}
    (RESULTS / "quant.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
