# minitorch: SPEC

## Goal

Prove I can write every fundamental of a modern LLM stack myself, then wire it into
one end-to-end system. Text enters through my BPE tokenizer and runs through my
autograd, my layers, and my Metal kernels on real Qwen3 weights. A paged-KV inference
server I wrote serves it to a RAG app and an agent I wrote. The work runs alongside
`~/personal/learning/study_plan.md`, one milestone per phase, and covers the CS336
assignments listed below.

The blog post tells it as one story (2026-11-05). Qwen3-0.6B solves 60.8% of a fixed
250-problem GSM8K subset; the post builds, from a single derivative upward, the code that
loads that model, serves it, and trains it to solve more. Each section opens on the problem
the previous result left open and ends on a measured answer. Story order:
1. Learning: gradients, autograd.
2. Writing: tokens, the model (with the A1 ablations), training on TinyStories.
3. Speed: GPU backends, profiling, mixed precision, attention kernels (Metal, then Triton
   forward and backward), then scaling laws, which fast training makes affordable.
4. A real model: Qwen3 loading and parity (GPT-2 beside it, M6), serving (KV cache, batching, chat),
   quantization and speculative decoding (both answers to decode being bandwidth-bound).
5. Teaching it math: LoRA, its memory wall and activation checkpointing, GRPO with
   rollouts from my engine.
6. Using it: RAG and an agent on my server, with tracing.
7. Last section: Stable Diffusion, the same framework drawing an image (M6).

Interactive version with diagrams: `docs/plan/` (run `gen.py`, then render with the
excalidraw CLI, then `build.py`).

## Rules

- Backends: NumPy (reference), MLX (Metal), and CuPy (CUDA) for training throughput
  on bqa3 (RTX 3090). Custom kernels: Metal via `mx.fast.metal_kernel` on the Mac, and
  Triton on the NVIDIA path (2026-11-05: CS336 A2 grades a Triton FlashAttention-2).
  Triton kernels are Python (`@triton.jit`), no CUDA C: ruff TID251 bans CuPy's raw kernel
  APIs, which take C source. Open question for A2 start: launching Triton on CuPy
  pointers without importing torch.
- UI: FastAPI + React/Vite/TS in `ui/`: one blog post (2026-11-05), every experiment a
  section with live figures (chat, training runs, autograd graph, BPE, attention heads,
  benchmarks). It replaced the Streamlit `app.py` (deleted 2026-11-06) and grows one section per milestone.
- What I write: autograd, every gradient, layers, optimizers, BPE tokenizer,
  safetensors parser, sampling, Metal kernels, LoRA, quantization, KV cache manager,
  scheduler, engine, server, RAG pipeline, agent loop.
- What I borrow: NumPy/MLX arrays and kernel launch, HF checkpoints (weights only),
  FastAPI for HTTP parsing. The retrieval embeddings run in minitorch too (Qwen3-Embedding-0.6B
  is Qwen3's architecture, 2026-11-05). `mx.fast.*` and PyTorch serve as test oracles only,
  never in the product path.
- Tensors are lazy (2026-11-06): writing an op records it, and it runs when its result is read (`.data`,
  `.numpy()`, `.item()`, printing, `.backward()`, `.realize()`), each op once; an op nobody reads never runs. Every
  primitive in mlops.py states its output shape and dtype (`out_shape`, `out_dtype`), and realize() checks them
  against what it computed. Scope is lazy evaluation only: no fusion, scheduling, or kernel generation. Measured on a
  2-layer story model, 25 steps: identical losses to eager; about +0.8 ms per step on NumPy (9.3 to 10.2 ms) and
  within run-to-run noise on Metal (4.3 to 5.4 ms eager, 4.6 to 5.2 ms lazy). `tests/lazy.py` checks it.
- `TransformerLM(attn_res=True)`: Block Attention Residuals as an option (2026-11-06). Not trained or measured on
  purpose; the user's Attention Residuals post covers it. tests/ablations.py checks it builds, trains, starts as a
  plain average, and leaves attn_res=False checkpoints loadable.
- Every milestone ends on a measured number, and `bench/` writes it into the README.
- Models stay small: a 7M GPT, the CS336 A1 TinyStories model (22.7M params,
  trained on bqa3), Qwen3-0.6B, Qwen3-1.7B.

## Milestones

**M1. Core, tokenizer, GPT** (study Phase 1: CS336, nanoGPT)
- Every gradient moves into `minitorch/mlops.py` (about 20 primitives).
- RMSNorm, RoPE, GQA with KV cache, SwiGLU, AdamW, cosine warmup.
- Byte-level BPE tokenizer (CS336 A1).
- `tests/parity.py`: every layer against its torch twin, outputs and grads.
- Done when: CI runs `test_grads.py` and `tests/parity.py`, the 7M GPT loss matches
  the PyTorch version with the same seed, and BPE round-trips a held-out file.

**M2. Metal kernels + start serving** (study Phase 2: optimisation)
- `minitorch/backend/{cpu,metal,cuda}.py`, `device="metal"` / `"cuda"`, bf16 on metal.
- FlashAttention forward via `mx.fast.metal_kernel`.
- Side track: tinyserve's `KVCacheManager` and `Scheduler`, built against minitorch's own
  TransformerLM instead of PyTorch MPS (decided 2026-11-05: an HF model keeps its own KV
  cache, so it cannot exercise the block tables). The torch runner moves to M3 as the Qwen3 oracle.
- Done when: MNIST and the GPT on Metal match NumPy within 1e-4, a kernel table lists
  measured GB/s vs `mx.fast`, and the paged KV demo runs.

**M3. Real weights, LoRA, quantization** (study Phase 3: fine-tuning)
- Safetensors parser, Qwen3-0.6B name mapping, sampling.
- `nn/lora.py`, group-wise int8/int4 dequantized in the matmul.
- LoRA task (decided 2026-11-05): GSM8K train split; eval is exact match on the final
  number over a fixed 250-question test subset, greedy decoding, before vs after.
- Finish the tinyserve torch port: `ModelRunner`, `Engine`, FastAPI.
- Done when: logits match HF within tolerance, LoRA shows a measured eval gain, and a
  perplexity table covers bf16 / int8 / int4.

**M4. The engine runs on minitorch** (study Phase 4: inference)
- Swap torch for minitorch in `ModelRunner`.
- Paged decode-attention kernel that reads K/V through the block table. Without it,
  every decode step gathers the KV blocks into a contiguous copy.
- Batched decode across prompt lengths, built for GRPO (2026-11-05): left padding plus a key
  mask in mlops.Attention, so many problems' samples decode in one batch. Prefix sharing is
  skipped: a group's 8 copies of a ~150-token prompt cost one prefill pass, against ~320
  decode steps after it (prefill share to be measured).
- `mx.compile` on the decode step. Quantized serving.
- Speculative decoding (2026-11-05: required, no longer a stretch): Qwen3-0.6B drafts for
  Qwen3-1.7B. It is a decoding method, independent of the architecture.
- Done when: GRPO-shaped load (N prompts x G samples) is measured with and without batching
  and prefix sharing; speculative decoding's acceptance rate and tokens/s are measured;
  `mx.compile` on vs off is measured; one row compares against mlx-lm.
  Open 2026-11-06: `mx.compile` is unmeasured. The decode step's KV cache grows a token per step, so a
  compiled step retraces every token without a fixed-shape cache. Keep or cut: the user's call.

**M5. End-to-end apps** (study Phase 5: RAG, agents)
- The agent and the RAG pipeline call your server's OpenAI endpoint.
- Agent (apps/agent.py, 2026-11-05): GSM8K with a calculator tool through Qwen3's native tool
  calling (chat_prompt renders tools and tool results exactly as HF's template, tested); the
  calculator parses expressions and never executes model text (tests/agent.py). Scored on the
  same 250-problem protocol as bench/gsm8k.py.
- RAG (apps/rag.py, 2026-11-05): SQuAD v1.1 dev, 2,067 paragraphs embedded by Qwen3-Embedding-0.6B
  in minitorch (last-token pooling, matches HF to 5.5e-5), a NumPy dot-product index, and the
  server answering closed-book vs with the top 3 paragraphs; recall@k, exact match, and F1.
- Per-request latency tracing: each app records every model call's time and prompt/reply tokens.
- Done when: the RAG eval and the agent task are scored and their time breakdowns are in the post.
  Live in the post (2026-11-06): /api/rag and /api/agent run apps/rag.py's and apps/agent.py's own code on the
  server's engine; §13 shows both as figures, and §15 (Conclusion) answers §1.
  Done 2026-11-06 (post §13): RAG recall@1 73%, recall@3 91.5%, answers containing the reference 3% closed-book
  vs 55% with the paragraphs. Agent: with GSM8K's "step by step" prompt the model never called the tool (132/250
  vs 152, p = 0.015); asked for the tool in the user turn it called it before reasoning and fell to 42/250.

**CS336 coverage** (decided 2026-11-05; Spring 2026 handouts; each lands as a section of the post)
- A1 experiments: the four ablations (no RMSNorm, post-norm, NoPE, SiLU vs SwiGLU), in the
  post's model section. Optimizer state in checkpoints (code only, no section). FLOP and
  memory accounting inline where the post uses it. Cut 2026-11-05: the OpenWebText run and
  standalone learning-rate and batch-size sweeps (scaling laws carry the learning-rate sweep).
- A2 systems: a benchmarking and profiling harness, mixed-precision training, memory
  profiling, activation checkpointing, and FlashAttention-2 forward and backward in Triton
  on the CuPy path (against the Metal and composed versions). Cut 2026-11-05: distributed
  data parallel, optimizer state sharding, and FSDP (both models fit one 3090).
- A3: scaling laws fit from my own grid of small runs on bqa3 (Stanford's API is students-only).
- A5: GRPO on GSM8K with Qwen3-0.6B (train_grpo.py), rollouts from minitorch's batched
  generate; its variants (RFT, Dr. GRPO) as time allows. A4 (data filtering) is out.
  Decided 2026-11-05: start from the released model (60.8%), not the LoRA one (49.2%); train a
  LoRA adapter (rank 16, lr 1e-4, 10x CS336's full fine-tuning rate per "LoRA Without Regret");
  reward 1 when the final number matches; on-policy, one update per batch of rollouts, so
  GRPO-Clip's ratio is 1 and the loss is REINFORCE with the group-normalized baseline.
  Done when: the 250-problem protocol scores the GRPO adapter, and rollout tokens/s is
  measured at one problem's group per batch against many.
  Done 2026-11-06 (post §12): a reward without the stop requirement was hacked by step 51 (every reply
  ran to the cap); with it, 139/250 vs 152, fixed 25 / broke 38, p = 0.13 (bench/gsm8k_compare.py).
  A1, A2, A3 done the same day: ablations in §5, profiling and Triton FA2 in §7; scaling laws,
  mixed precision, and checkpointing in the appendix. Except: A1's optimizer state in checkpoints is not built
  (train_gpt.py saves weights only, no AdamW m/v/t or step, no --resume). Build it or cut it: the user's call.

**M6. Model examples, tinygrad style** (2026-11-05: GPT-2 and Stable Diffusion; not Mamba or
SDXL. Llama 3 dropped 2026-11-05, user decision)
- One file each, like tinygrad's `examples/`: `examples/gpt2.py`, `examples/llama3.py`,
  `examples/stable_diffusion.py`. Each loads the published weights through `state.py` and runs
  on minitorch's backends.
- GPT-2 (124M): LayerNorm, learned positions, GELU, biases; logit parity against HF.
  Done 2026-11-05: examples/gpt2.py, tests/gpt2.py (CPU 7.0e-7, Metal 1.1e-6 of the largest logit;
  40 greedy tokens match).
- Stable Diffusion v1.5: CLIP text encoder, UNet, VAE decoder, DDIM with classifier-free
  guidance; diffusers as the oracle.
  Done 2026-11-06: examples/stable_diffusion.py, tests/stable_diffusion.py (largest error 8.4e-6 of the
  output, CPU and Metal), bench/stable_diffusion.py (2.58 vs 1.16 s per step against diffusers on MPS;
  images within 3.4 of 255 levels). In the post's appendix (Other models).
- Post (user, 2026-11-06): both live in the appendix's Other models, GPT-2 as a parity table and Stable Diffusion
  as a tinygrad-style snippet with its parity table and image; the body mentions them in one sentence.
- Done when: GPT-2 matches HF logits within tolerance, and Stable Diffusion
  produces an image with a measured s/step and its gap to diffusers.

## Every milestone

- `uv run python test_grads.py` and `tests/parity.py` pass in CI.
- `bench/` regenerates the README tables.
- One blog post ships (list in the study plan's Blogs section).
- The mkdocs site holds the API reference and the study plan; the blog post holds the results.

## Open items

- Study plan: done 2026-11-06, `docs/guide.md` (replaced `docs/reading-guide.md`).
- Rename: "minitorch" collides with Cornell's minitorch course and an existing PyPI
  package.

## History

The earlier detailed v2 spec (tinygrad reading ladder, measured baseline table) lives
in `docs/spec-v2-detailed.md`.
