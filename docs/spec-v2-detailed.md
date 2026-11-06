# minitorch v2: SPEC

Turn minitorch from a readable autograd toy into a readable ML framework that runs
Llama and Stable Diffusion on Apple Metal.

Method: read tinygrad's git history forward from commit one, understand each idea,
close the tab, reimplement it in minitorch without looking, then diff.

Twelve weekends. Nine phases.

---

## What done looks like

```bash
uv run python examples/train_gpt.py                                    # trains a 7M GPT
uv run python examples/llama.py --prompt "the capital of France is"    # real 1B weights
uv run python examples/stable_diffusion.py --steps 15 --prompt "a car in a box"
```

Plus: every gradient in one readable file, a NumPy backend and a Metal backend,
four optimizers with measured curves, mixed precision and gradient checkpointing
with measured memory, and int4 quantization with a measured perplexity delta.

The claim this project supports: *I implemented every gradient, every modern
transformer component, quantization, and mixed precision myself, and I can run
Llama and Stable Diffusion on it.*

---

## The tradeoff, recorded so future-you remembers

An earlier version of this plan had you build a ShapeTracker, a lazy graph, and a
kernel scheduler. That is where fusion speed comes from, and it is also compiler
work rather than ML work. You cut the ShapeTracker and the scheduler on purpose.
The lazy graph came back in Phase 2.5 for its learning value, scoped so it does not
drag the other two in with it.

Consequence: minitorch will beat NumPy and match PyTorch on Metal, but MLX earns
that, not you. You get the speed without owning the reason for it. Accept this and
do not claim otherwise in the README.

If you ever want the systems layer back, the entry point is a `ShapeTracker` class
holding `(shape, strides, offset, mask)`. tinygrad's first version was 99 lines at
commit `6261a0639`.

---

## Measured baseline

Apple M4 base, 10-core GPU, 16GB unified memory, 4 performance + 6 efficiency cores.
20 elementwise ops on 8M float32, and matmul 2048x2048x2048. Minimum of 5 runs after
2 warmups, thermals uncontrolled.

| Backend | 20-op chain | matmul | GFLOP/s |
|---|---|---|---|
| minitorch today | 115.0 ms | 13.17 ms | 1304 |
| NumPy eager | 53.9 ms | 11.93 ms | 1440 |
| PyTorch CPU | 31.9 ms | 12.01 ms | 1431 |
| PyTorch MPS | 12.9 ms | 5.30 ms | 3243 |
| MLX (Metal) | 12.7 ms | 5.41 ms | 3173 |

Reproduce with `uv run python benchmark.py --throughput`. Run it at the end of every
phase and watch the minitorch row move.

Three more figures from a one-off probe, not in `benchmark.py` because they need a
fusion pass you are not building: MLX with `mx.compile` hit 2.64 ms on the chain,
numexpr 10.3 ms, and Numba with `parallel=True` 1.31 ms. That last one is 24x faster
than PyTorch CPU on the same graph. It is what the scheduler would have bought.

Two readings. First, your matmul is fast because NumPy hands it to Accelerate; you
contribute nothing there and neither will MLX beyond swapping in Metal. Second, the
elementwise gap between 92.0 ms and 14.0 ms is what the MLX backend buys you in
Phase 2, for roughly 150 lines of dispatch code.

---

## Architecture

```
Tensor            eager autograd, PyTorch-shaped API        yours, stays readable
  |
mlops             one gradient table, ~20 primitives        yours
  |
LazyBuffer        graph capture, topological realize        yours, ~200 lines
  |
array backend     numpy (reference) | mlx (Metal)           ~150 lines of dispatch
```

No ShapeTracker and no fusion pass sit between LazyBuffer and the backend. Movement
ops do real work at realize time and each node runs on its own. Those two absences
are the whole difference between this and tinygrad.

The dispatch layer is the only plumbing in the project. Everything above it is ML.

Repo layout after Phase 2:

```
minitorch/
  tensor.py       Tensor, autograd graph, backward
  mlops.py        the ~20 primitives and their gradients
  lazy.py         LazyBuffer, realize, DEBUG=1 schedule printing
  backend/
    numpy.py      reference implementation, readable
    mlx.py        Metal
  nn/
    layers.py     Linear, Conv2d, norms, activations
    attention.py  MHA, GQA, KV cache, RoPE
    moe.py        top-2 routing
    optim.py      SGD, Adam, AdamW, Muon
    init.py       Kaiming, scaled residual
    state.py      safetensors, state_dict, quantized load
  quant.py        int8 / int4 weight-only
examples/
  train_gpt.py
  llama.py
  stable_diffusion.py
  mnist.py
eager/            the current closure-based engine, frozen, still importable
```

`eager/` keeps the reading guide, the mkdocs site, and the Alice-in-Wonderland blog
post alive. It is the teaching artifact. Do not delete it and do not modify it.

---

## How to read tinygrad

```bash
git clone https://github.com/tinygrad/tinygrad
cd tinygrad
git log --reverse --oneline | head -40     # oldest first, which is what you want
git show <sha>:<path>                       # read a file as it existed then
git log --reverse --oneline --diff-filter=A -- <path>   # find when a file was born
```

`--reverse` gives oldest-first. Do not try to read backward from HEAD: today's
tinygrad is 35k core lines across 217 files plus 203k lines of generated bindings,
and none of it explains itself.

The method for each phase: read until you understand the idea. Close the tab. Write
your own version in minitorch's style. Then diff against theirs and write down where
you differed and why. The diff is the lesson. Skipping it wastes the exercise.

### Reading ladder

| Phase | File | Commit | Lines | Why |
|---|---|---|---|---|
| 1 | `tinygrad/mlops.py` | `91d97c831` | 211 | every gradient, in one file |
| 2 | `tinygrad/tensor.py` | `5216a1d9f` | - | how ops compose over primitives |
| 2.5 | `tinygrad/lazy.py` | `6a8fb5330` | 360 | graph capture and realize |
| 3 | `tinygrad/nn/__init__.py` | `b132de677` | 47 | Linear, Conv2d, BatchNorm, all of it |
| 4 | `tinygrad/nn/optim.py` | `b132de677` | 54 | SGD and Adam |
| 6 | `tinygrad/nn/state.py` | `718ced296` | 120 | safetensors, state dict |
| 6 | `extra/models/llama.py` | `7170a9a05` | 151 | the model |
| 8 | `examples/stable_diffusion.py` | `c01a8c5c2` | 125 | CLIP, UNet, VAE, sampler |

Context for scale: tinygrad ran Stable Diffusion when the framework was 1359 lines
(`c01a8c5c2`, Sept 2022) and Llama at 3085 lines (`4780f9a6d`, March 2023). Neither
number is out of reach.

---

## Phase 1: primitives and one gradient table

**Build.** Collapse the 35 hand-written `_backward` closures in `tensor.py` into
roughly 20 primitives with their gradients in a single `mlops.py`. Rederive
everything else as composition over those primitives.

Candidate primitive set, to be confirmed by what Phases 3 through 8 actually need:

- unary: `neg`, `recip`, `exp`, `log`, `sqrt`, `sin`
- binary: `add`, `mul`, `cmplt`, `maximum`
- reduce: `sum`, `max`
- movement: `reshape`, `permute`, `expand`, `pad`, `shrink`, `flip`
- special: `matmul`, `where`

`relu` is `maximum(x, 0)`. `sub` is `add` plus `neg`. `div` is `mul` plus `recip`. `softmax` is `exp`, `sum`,
`div` with a max subtraction. `sigmoid`, `tanh`, `gelu`, `silu` are all compositions.

**Keep matmul as a primitive.** tinygrad decomposes it into expand, mul, reduce and
rebuilds a tiled kernel in codegen. You have no codegen, so decomposing it would cost
you the 1473 GFLOP/s that BLAS gives you for free.

**Learn.** Which gradients are actually primitive. That most of a framework's op list is sugar. Where the chain rule stops being obvious: `max` needs an argmax
mask, `pad` and `shrink` are each other's gradient, `expand` backward is a sum.

**Read.** `git show 91d97c831:tinygrad/mlops.py`, 211 lines.

**Done when.** Every gradient definition lives in `mlops.py`. `test_grads.py` passes.
`benchmark.py --throughput` still runs and the minitorch row has not regressed.

---

## Phase 2: dtype and the Metal backend

**Build.** fp32, fp16, bf16 as a first-class field on Tensor, with explicit casting
rules and no silent promotion. Then split the array layer: `backend/numpy.py` stays
as the readable reference, `backend/mlx.py` runs on Metal. Same Tensor API,
selected by `minitorch.device('metal')`.

**Learn.** Why softmax overflows in fp16 and why subtracting the row max fixes it.
Why reductions accumulate in fp32 no matter what the storage dtype is. What bf16
trades away versus fp16 and why training prefers it. Where a host-to-device copy
costs you more than the compute you moved.

**Read.** `git show 5216a1d9f:tinygrad/tensor.py` for how a thin Tensor sits over a
backend.

**Done when.** `mnist.py` trains on Metal and matches the NumPy backend to 1e-4.
`bench.py` gains a Metal column. Expect the elementwise chain near 14 ms and matmul
near 5.5 ms based on the MLX numbers above; measure, do not assume.

**Note.** This is the only non-ML plumbing in the plan. Keep it under 200 lines.
If it grows, the abstraction is wrong.

---

## Phase 2.5: your own lazy graph

**Build.** Ops stop computing. They return a `LazyBuffer` holding an op, its sources,
a shape, a dtype, and a `realized` slot that starts empty. `.numpy()`, `.item()`, and
`__repr__` trigger `realize()`, which walks the DAG in topological order and runs each
node on the chosen backend. `backward()` builds graph too and computes nothing until
something reads `.grad`.

Plus `DEBUG=1`: print how many nodes the graph held, the order they realized in, and
how many buffers got allocated.

Explicitly not in this phase. No ShapeTracker, so movement ops still copy at realize
time. No fusion, so every node runs on its own. Those two are what turn a lazy graph
into a fast one, and they are the compiler work you cut.

**Learn.** Graph capture as a data structure instead of a concept. Topological realize
and why the order matters. Buffer lifetime, which is where you discover that a naive
lazy graph keeps every intermediate alive until realize and blows up memory on a long
chain. Why `backward()` can build graph without computing, and what that opens up in
Phase 5.

**Read.** `git show 6a8fb5330:tinygrad/lazy.py`, 360 lines. Start with
`LazyBuffer.realize` and `elementwise_op`. Skip the ShapeTracker interactions on the
first pass; they belong to a piece you are not building.

**Done when.** `(a + b) * c` allocates zero buffers until something reads it.
`DEBUG=1` prints node count and realize order. Every existing test and example runs
unchanged and matches the eager engine to 1e-5.

**Honest note.** On the MLX backend this buys no speed, because MLX already defers
underneath you. Measured: building a 10-op MLX graph costs 831 microseconds, and
`mx.eval` on it costs 42 ms. A 51x ratio. You are writing a deferral layer on top of
a deferral layer. Do it because you will understand it and because `DEBUG=1` shows
you what your framework does. For lazy to earn speed it needs a fusion pass into
Numba on the NumPy backend, and that is the scheduler you cut.

**Where the weekend comes from.** It displaces one of Phase 8's two weekends. If
Stable Diffusion then runs long, drop Phase 7 quantization, not this.

---

## Phase 3: modern layers, nanochat absorbed (two weekends)

**Build.** Port from nanochat, reimplemented on minitorch's autograd:

- RMSNorm, GroupNorm, and the LayerNorm you already have
- RoPE
- Multi-head attention, then GQA with separate Q and KV head counts
- KV cache
- SwiGLU
- QK-norm
- MoE with top-2 routing
- SiLU, plus `cat`, `chunk`, `interpolate` for Phase 8

Every one gradient-checked against finite differences and diffed against the
equivalent PyTorch module.

**Learn.** Backward through MoE routing, where gradients are sparse and the top-k
selection is not differentiable, so the gradient flows through the gate weights but
not the selection. RoPE as a complex rotation implemented in real arithmetic. How
the KV cache changes the shape math at every decode step, and why the attention
mask differs between prefill and decode.

**Read.** `git show b132de677:tinygrad/nn/__init__.py`, 47 lines. All of `nn` fits
in 47 lines. That is the readability bar.

**Done when.** `examples/train_gpt.py` trains nanochat's 7M model inside minitorch
and the loss curve matches the PyTorch version within run-to-run noise. Plot both.

**Note.** Delete nanochat's repo only after this phase passes. The loss-curve match
is the proof that nothing was lost in translation.

---

## Phase 4: optimizers and training dynamics

**Build.** AdamW with decoupled weight decay, cosine schedule with linear warmup,
gradient clipping by global norm, gradient accumulation, and Muon. Init schemes:
Kaiming, and scaled residual init at 1/sqrt(2L).

**Learn.** Why AdamW is not Adam plus L2, and what changes in the update rule. What
warmup actually fixes, which is Adam's second-moment estimate being garbage for the
first few hundred steps. Why residual branches scale by 1/sqrt(2L) and what happens
to activation variance when they do not. What Muon's Newton-Schulz orthogonalization
does to the update and why it only applies to 2D parameters.

**Read.** `git show b132de677:tinygrad/nn/optim.py`, 54 lines.

**Done when.** One plot, four loss curves, SGD versus Adam versus AdamW versus Muon
on the same 7M model with the same seed and step budget. Committed to the README.

---

## Phase 5: mixed precision and gradient checkpointing

**Build.** bf16 forward and backward with fp32 master weights. Loss scaling with
dynamic adjustment on overflow. Gradient checkpointing that drops activations in
forward and recomputes them in backward.

**Learn.** The memory-versus-compute tradeoff as a number you measured on your own
model instead of a claim you read. Why loss scaling exists for fp16 and less so for
bf16. Which activations are worth recomputing and which are cheaper to store.

**Done when.** A table in the README: peak memory and ms per step, for the 7M model,
with each feature off and on. Four rows.

**Note.** Peak memory needs a real measurement, not `sys.getsizeof`. Track allocated
bytes inside the backend.

---

## Phase 6: safetensors and Llama-3.2-1B (two weekends)

**Build.** A safetensors parser, roughly 50 lines: an 8-byte little-endian header
length, a JSON header mapping tensor names to dtype, shape, and byte offsets, then a
raw buffer. Map Hugging Face weight names onto your layers. Generation with KV cache
and real sampling: temperature, top-k, top-p, repetition penalty.

**Learn.** Weight layouts and why some checkpoints store transposed. bf16 unpacking
without native support. The sampling math, which is where top-p differs from top-k
in ways that matter. And the memory-bandwidth ceiling, derived on your own hardware:

```
tokens/sec ceiling = memory_bandwidth / bytes_of_weights_read_per_token
KV cache bytes/token = 2 * n_layers * n_kv_heads * head_dim * bytes_per_element
```

Read the real values out of the model's `config.json` rather than trusting memory.
Decode is memory bound, not compute bound, and this is the arithmetic that proves it.

**Read.** `git show 718ced296:tinygrad/nn/state.py` (120 lines) and
`git show 7170a9a05:extra/models/llama.py` (151 lines).

**Done when.** `examples/llama.py --prompt "the capital of France is"` emits coherent
text. Tokens per second reported for both backends, next to the computed ceiling.

**Note.** Tokenizer stays external. Use the HF tokenizer or tiktoken. Writing a BPE
implementation is a different project.

---

## Phase 7: quantization

**Build.** int8 and int4 weight-only quantization. Group-wise scales, typically 64
or 128 weights per group. Dequantize during the matmul rather than materializing the
fp16 weight.

**Learn.** Why per-channel beats per-tensor. How group size trades file size against
quality. Where quantization error concentrates, which is usually the outlier channels
in the attention projections. What symmetric versus asymmetric buys you.

**Done when.** Llama-3.2-1B runs under 1GB. Report a measured perplexity delta on a
held-out text sample: fp16 baseline versus int8 versus int4, three numbers.

**Note.** Perplexity needs the same tokenization and the same text for all three, or
the comparison is meaningless.

---

## Phase 8: Stable Diffusion (two weekends)

**Build.** CLIP text encoder, the UNet, the VAE decoder, a DDIM or Euler sampler, and
classifier-free guidance. SD 1.5 is roughly 860M UNet plus 123M VAE plus 123M CLIP;
verify against the actual checkpoint. At fp16 that is about 2.2GB, which fits 16GB.

**Learn.** Noise schedules and what beta scheduling controls. What CFG scale does to
the sample, and why it is two forward passes per step. Latent space and why the VAE
decode happens once at the end. Cross-attention, where K and V come from text and Q
from the image. And conv2d correctness under real load, where a bug produces a
visibly wrong image instead of a slightly wrong loss.

**Read.** `git show c01a8c5c2:examples/stable_diffusion.py`, 125 lines.

**Done when.** `examples/stable_diffusion.py --steps 15 --prompt "a car in a box"`
writes a recognizable image. Report seconds per step, measured.

---

## Examples backlog

Phases 1 through 8 are the spine. These are what to build after, or instead of
Phase 7 if quantization loses its weekend. Line counts are tinygrad's version at
HEAD, as a size estimate.

| Example | tinygrad lines | What it teaches | Depends on |
|---|---|---|---|
| `cartpole.py` | 126 | PPO: advantages, entropy bonus, clipped ratio. A whole domain the rest of this spec skips | Phase 4 |
| `rectified_flow.py` | 156 | Flow matching, which is what SD3 and Flux use instead of DDPM. Simpler than diffusion | Phase 6 |
| `gan.py` | 106 | Two optimizers alternating on opposing losses. A training dynamic nothing else here has | Phase 4 |
| `gpt2.py` | 246 | Real GPT-2 weights. Reuses the safetensors loader from Phase 6 | Phase 6 |
| `speculative.py` | - | Draft with the 7M model, verify with the 1B. Pure algorithm | Phase 6 |
| `scaling_laws.py` | - | Train 6 tiny GPTs across width and depth, fit the power law, plot it | Phase 4 |
| `lora.py` | - | A Linear subclass with two low-rank matrices. Compose with Phase 7 for QLoRA | Phase 3 |
| `dpo.py` | - | Roughly 30 lines of loss on top of the 7M model | Phase 4 |
| `vit.py` | - | Patch embedding plus the attention you already have | Phase 3 |
| `imagenet.py` | 170 | ResNet-50 with pretrained weights, top-1 on a validation subset. Less about learning than about having a hard accuracy number that catches conv2d and batchnorm bugs a pretty image would hide | Phase 8 |
| `mamba.py` | 323 | Selective state space. Non-transformer | Phase 3 |
| `whisper.py` | 332 | Audio. Needs mel spectrograms and a different tokenizer. Biggest lift here | Phase 6 |

Read tinygrad's version of each before writing yours, same blind method as the phases.

## Small features worth stealing

None of these are hard and each pays for itself.

- **`x.sequential([...])`.** Layers as a plain Python list, activations as bare
  functions passed by reference, no Module subclass needed to define a model. See
  `examples/beautiful_mnist.py`, 48 lines total. Roughly 5 lines in Tensor.
- **Counter-based RNG.** Seed in, deterministic stream out, no dependence on NumPy's
  global state. Matters the first time you compare two training runs.
- **`gather` and `scatter`.** Needed for MoE routing in Phase 3 regardless.
- **`topk`, `multinomial`, `argsort`.** Needed for sampling in Phase 6 regardless.
- **`einsum`.** Pure composition over the primitives. Makes attention readable.
- **`tril` and `triu`** for causal masks.
- **`safe_save` next to `safe_load`.** Twenty extra lines in Phase 6 and checkpoints
  become a real format instead of a pickle.
- **Blocked attention with online softmax.** The FlashAttention algorithm, running max
  and running sum, tiled over the sequence. Pure math, no CUDA. Measure the memory
  saving against the naive version.

## Where the study plan lands

The study plan files "build minitorch" under parallel tracks. It belongs on the main
line instead. Every phase there already ends in a build gate; point those gates here.

| Study plan item | Lands as |
|---|---|
| P0 gate, nanoGPT trains | Phase 3, `examples/train_gpt.py` |
| P1 gate, GPT-2 from an empty file | `examples/gpt2.py` |
| P1 architecture deltas: RoPE, RMSNorm, GQA, SwiGLU | Phase 3 |
| CS336 A1: tokenizer, architecture, optimizer, training | Phase 3 and Phase 4 |
| CS336 A2: FlashAttention | Blocked attention above. The Triton kernel stays in CS336 |
| CS336 A3: scaling laws | `examples/scaling_laws.py` |
| CS336 A5: SFT, DPO, RL | `examples/dpo.py` and `examples/cartpole.py` |
| P3: LoRA | `examples/lora.py` |
| P3: QLoRA | `lora.py` composed with Phase 7 |
| P4: KV cache | Phase 3 |
| P4: speculative decoding | `examples/speculative.py` |
| Vision track: CLIP, VAE, UNet | Phase 8 |
| Vision track: ViT | `examples/vit.py` |
| Vision track: DDPM | Or skip it and do `rectified_flow.py` |

Does not belong here and should not be forced in: Triton kernels and distributed
training from A2, CommonCrawl filtering from A4, the serving stack in P4, and RAG and
agents in P5. Those get their own repos.

---

## Verification

Three rules, applied every phase.

1. **Finite differences.** Keep `test_grads.py`. It catches math errors in gradients.
2. **Diff against PyTorch.** Every op and every layer, forward and backward, against
   the torch equivalent. Finite differences catch math errors; torch catches semantics
   errors: dtype promotion, reduce-axis conventions, broadcast corners, and the
   epsilon placement inside norms.
3. **Backend agreement.** NumPy and MLX must match to 1e-4 on every test. The NumPy
   backend is the reference; when they disagree, NumPy is right until proven otherwise.

Comparison rules: integers bitwise, floats with atol/rtol.

`uv run ruff check . --fix` on any Python touched.

---

## Out of scope

Listed so scope creep has to argue its case.

- ShapeTracker, kernel schedulers, codegen. Cut on purpose, see above. The lazy
  graph survived as Phase 2.5; fusion did not.
- CUDA and the RTX 5090. Revisit after Phase 8 if the backend interface held up.
- Writing a BPE tokenizer.
- Distributed training, optimizer state sharding, anything multi-device.
- Training Stable Diffusion or Llama. Inference only for both.
- A line-count CI gate. `sz.py` reports, it does not fail the build.

---

## Open questions

- Llama-3.2-1B config values (`n_layers`, `n_kv_heads`, `head_dim`) are unverified
  here. Read `config.json` in Phase 6 before doing the bandwidth arithmetic.
- SD 1.5 parameter counts above are approximate. Verify against the checkpoint.
- Seconds per diffusion step on an M4 needs measuring in Phase 8. Do not predict it.
- Whether the primitive set in Phase 1 survives Phase 8 unchanged. It probably gains
  one or two ops for conv and interpolate. Record what you added and why.
