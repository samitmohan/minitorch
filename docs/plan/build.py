"""Inline the rendered Excalidraw PNGs and scene JSON into one self-contained HTML page."""
import base64
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
D = HERE / "diagrams"  # gen.py writes here; render with the excalidraw CLI
# Default: the Desktop copy. Pass a path to also write elsewhere (e.g. the portfolio site).
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Desktop" / "minitorch-plan.html"


def svg(name):
    """Embed the rendered PNG; the renderer's SVG output loses text alignment in browsers."""
    b64 = base64.b64encode((D / "png" / f"{name}.png").read_bytes()).decode()
    return f'<img alt="{name} diagram" src="data:image/png;base64,{b64}">'


def scene_b64(name):
    return base64.b64encode((D / f"{name}.excalidraw").read_bytes()).decode()


M = [
    {
        "n": 1, "title": "Core, tokenizer, GPT", "phase": "Study Phase 1: build the transformer",
        "study": ["CS336 Assignment 1, written in minitorch instead of torch",
                  "Karpathy Zero to Hero: nanoGPT, reproduce GPT-2",
                  "Life of a Token", "Notes on essential LLM"],
        "why": "This milestone shows you understand backprop, transformers, and tokenization "
               "well enough to write them with no framework underneath. Today "
               "<code>tensor.py</code> has 35 hand-written <code>_backward</code> closures, and "
               "most of them are sugar: <code>relu</code> is <code>maximum(x, 0)</code>, "
               "<code>sub</code> is <code>add</code> plus <code>neg</code>. Once about 20 "
               "primitives own every gradient, each new layer is just composition.",
        "build": ["Every gradient moves into one file: <code>minitorch/mlops.py</code>",
                  "RMSNorm, RoPE, GQA with KV cache, SwiGLU, AdamW with cosine warmup",
                  "A byte-level BPE tokenizer, trained and run by your own code (CS336 A1 has this)",
                  "<code>tests/parity.py</code>: every layer against its torch twin, outputs and grads",
                  "<code>examples/train_gpt.py</code>: a 7M-parameter GPT on your tokenizer"],
        "read": ["tinygrad <code>git show 91d97c831:tinygrad/mlops.py</code> (211 lines, every gradient)",
                 "MyTorch <code>mytorch/tensor.py</code> for contrast: 2,229 lines doing the same job"],
        "done": ["<code>test_grads.py</code> and <code>tests/parity.py</code> pass in CI",
                 "7M GPT loss curve matches the PyTorch version with the same seed",
                 "Your BPE encodes and decodes a held-out file round-trip exactly"],
        "blog": ["Twenty primitives: every gradient in a neural network fits in one file",
                 "Matching PyTorch's loss curve to the decimal"],
        "extra": "",
    },
    {
        "n": 2, "title": "Metal kernels + start serving", "phase": "Study Phase 2: optimisation",
        "study": ["Drepper, What Every Programmer Should Know About Memory",
                  "FlashAttention paper, then FlashAttention 1-4",
                  "CUDA for Deep Learning, read for concepts only (tiling, shared memory, warps)",
                  "How to Scale Your Model", "tinyserve video"],
        "why": "This milestone shows you understand hardware: memory bandwidth, tiling, and "
               "why attention is memory-bound. MLX supplies array storage and kernel "
               "launch. The kernels themselves are yours. Shared memory maps to Metal "
               "threadgroup memory, and a warp maps to a 32-lane SIMD group. The tinyserve "
               "port starts here, on torch, because it doesn't depend on minitorch at all.",
        "build": [("<code>minitorch/backend/numpy.py</code> (reference) and "
                   "<code>minitorch/backend/mlx.py</code> (Metal), picked by <code>device=\"metal\"</code>"),
                  "bf16 as a real dtype on Tensor",
                  "FlashAttention forward via <code>mx.fast.metal_kernel</code>, the learning kernel",
                  "Side track: tinyserve's <code>KVCacheManager</code> (146-480) and <code>Scheduler</code> (505-680) on PyTorch MPS"],
        "read": ["MyTorch <code>mytorch/_array.py</code>: how one wrapper switches backends",
                 "MyTorch <code>fused_ops/flash_attention.py</code>, the algorithm only, after you write yours",
                 "Use <code>mx.fast.scaled_dot_product_attention</code> as the oracle, never in the product path"],
        "done": ["MNIST and the GPT train on Metal and match NumPy within 1e-4",
                 "Kernel table: your FlashAttention vs <code>mx.fast</code>, measured GB/s",
                 "Paged KV cache demo runs on torch (two requests sharing prefix blocks)"],
        "blog": ["FlashAttention in Metal, from a naive kernel to within X of mx.fast",
                 "Why my GPU backend was slower than NumPy"],
        "extra": "<div class='math'><b>Why flash attention.</b> Naive attention writes an N x N score "
                 "matrix to memory. N = 2048 in fp32 is 2048 x 2048 x 4 B = 16.8 MB per head per "
                 "layer. FlashAttention keeps tiles on chip and runs an online softmax, so that "
                 "matrix never exists in memory.</div>",
        "kv": True,
    },
    {
        "n": 3, "title": "Real weights, LoRA, quantization", "phase": "Study Phase 3: fine-tuning",
        "study": ["Raschka: LoRA, PEFT and adapters, grad accumulation on one GPU",
                  "LoRA paper, QLoRA paper", "Unsloth + HuggingFace docs"],
        "why": "This milestone shows your layers are exactly right, not just close. A wrong RoPE "
               "or norm turns real weights into garbage text. LoRA and quantization then show "
               "you understand parameter efficiency and numerics.",
        "build": ["Safetensors parser: 8-byte little-endian header length, then a JSON header, then raw bytes",
                  "Map Qwen3-0.6B's HF weight names onto your layers; sampling with temperature, top-k, top-p",
                  "<code>nn/lora.py</code>: frozen W plus a trainable B @ A at rank r",
                  "Group-wise int8 and int4 weights, dequantized inside the matmul",
                  "Finish the tinyserve torch port: <code>ModelRunner</code>, <code>Engine</code>, FastAPI"],
        "read": ["tinygrad <code>git show 718ced296:tinygrad/nn/state.py</code> (120 lines)",
                 "Qwen3-0.6B <code>config.json</code>: take shapes from it, not from memory"],
        "done": ["Qwen3-0.6B writes coherent text, and its logits match HF within tolerance",
                 "A LoRA fine-tune shows a measured eval improvement",
                 "Perplexity table: bf16 vs int8 vs int4 on the same text"],
        "blog": ["Loading Qwen3 by hand: safetensors is 8 bytes and a JSON header",
                 "Your decode speed is a division problem",
                 "LoRA from scratch: 1.6% of the weights, measured"],
        "extra": "<div class='math'><b>LoRA arithmetic.</b> A d x d weight has d<sup>2</sup> params. "
                 "LoRA at rank r trains 2dr. With d = 1024 and r = 8 that's 16,384 against 1,048,576, "
                 "or 1.6%.<br><b>Decode ceiling.</b> Every generated token reads every weight once. "
                 "At the M4's 120 GB/s spec bandwidth, 0.6B params give a ceiling of about 100 tok/s "
                 "in bf16 (1.2 GB), 200 in int8, and 400 in int4. Measure where the real numbers land.</div>",
    },
    {
        "n": 4, "title": "The engine runs on minitorch", "phase": "Study Phase 4: inference",
        "study": ["How GPT is trained and served", "100 days of inference",
                  "Engineering behind LLM inference", "FastAPI for AI engineers, Python threading",
                  "Speculative decoding"],
        "why": "This milestone shows you can build a production-shaped system, not just a "
               "model. The torch serving code from M2-M3 gets your framework underneath. That "
               "needs a second kernel: without paged attention, every decode step has to copy "
               "the scattered KV blocks into one contiguous array, which throws away the "
               "benefit of paging.",
        "build": ["Swap torch for minitorch inside <code>ModelRunner</code>",
                  "Paged decode-attention kernel: reads K/V through the block table, no gather",
                  "<code>mx.compile</code> on the decode step, the Metal stand-in for CUDA graphs",
                  "Quantized serving: the int8/int4 weights from M3 in the engine",
                  "Stretch: speculative decoding, Qwen3-0.6B drafting for Qwen3-1.7B"],
        "read": ["tinyserve <code>serve.py</code> <code>ModelRunner</code> (1052-1365) and <code>Engine</code> (1462)",
                 "mlx-lm's server, which is your real baseline"],
        "done": ["req/s table: HF static, mlx-lm, yours on torch, yours on minitorch (bf16 / int8 / int4)",
                 "<code>mx.compile</code> on vs off, measured",
                 "Two clients stream from your server at once, recorded as the README demo"],
        "blog": ["Building a vLLM in 2,000 lines on a MacBook",
                 "Prefix caching, explained with two block tables",
                 "Why paging needs its own attention kernel"],
        "extra": "<div class='math'><b>KV cache size.</b> bytes per token = 2 (K and V) x layers x "
                 "kv_heads x head_dim x 2 B (bf16). Take the values from <code>config.json</code>. "
                 "That number times your page size tells you how many pages fit in the memory the "
                 "weights leave free.</div>",
    },
    {
        "n": 5, "title": "End-to-end apps", "phase": "Study Phase 5: RAG and agents",
        "study": ["How to build an agent (300 lines, build it first)", "Building effective agents",
                  "Prod RAG, RAG design and eval guide", "LLM evals"],
        "why": "This is the step that makes it one end-to-end project. Text goes through your "
               "tokenizer, your model, your kernels, and your server, and comes out in your app. "
               "Nothing between the HTTP request and the GPU belongs to someone else.",
        "build": ["The 300-line agent calls <code>http://localhost:8000/v1/chat/completions</code>",
                  "A RAG pipeline on the same endpoint",
                  "Latency tracing: time spent in retrieval, prefill, and decode for each request"],
        "read": ["Your own <code>serve/</code> logs"],
        "done": ["One RAG eval and one agent task, end-to-end latency on your server vs mlx-lm",
                 "A latency breakdown chart (retrieval / prefill / decode) in the README"],
        "blog": ["My agent runs on an inference engine I wrote"],
        "extra": "",
    },
    {
        "n": 6, "title": "Stable Diffusion (optional)", "phase": "Study plan: Computer Vision",
        "study": ["CNN recap, then ViT", "CLIP, VAE, DDPM", "Implement UNet"],
        "why": "This one is optional and goes last. It proves conv2d and cross-attention under "
               "real load, but it adds less for an LLM-focused role than M1-M5. Do it only "
               "after the capstone post ships.",
        "build": ["CLIP text encoder, UNet, VAE decoder",
                  "DDIM sampler with classifier-free guidance (two forward passes per step)",
                  "Reuse <code>minitorch/conv.py</code>"],
        "read": ["tinygrad <code>git show c01a8c5c2:examples/stable_diffusion.py</code>"],
        "done": ["<code>examples/stable_diffusion.py --prompt \"a car in a box\"</code> writes a recognizable image",
                 "Seconds per step measured"],
        "blog": ["Bugs you can see: debugging conv2d with Stable Diffusion"],
        "extra": "",
    },
]

for m in M:
    m["svg"] = svg(f"m{m['n']}")
    m["scene"] = scene_b64(f"m{m['n']}")
KV = {"svg": svg("kv"), "scene": scene_b64("kv")}

html = (HERE / "template.html").read_text()
html = html.replace("/*DATA*/", "const M = " + json.dumps(M) + ";\nconst KV = " + json.dumps(KV) + ";")
OUT.write_text(html)
print(OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")
