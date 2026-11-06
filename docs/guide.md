# Study plan

minitorch was built in the order you would learn it: a gradient, then a model, then hardware, then a real model,
serving, fine-tuning, and applications. This page turns that order into a plan for rebuilding it yourself. The blog
post explains why each piece exists and what it measured. The code shows how. This page says in what order to take
them, how to work each piece, and how to think while you do.

Move to the next stage when two things hold: your own version passes the stage's check, and you can answer the
stage's questions at the end of this page without notes.

## Before you start

You need four things. If one is shaky, fix it first; every stage leans on them.

- **NumPy fluency.** Broadcasting, `reshape` against `transpose`, fancy indexing, and what `axis=-1` does. Most bugs in
  a framework are shape bugs.
- **Linear algebra.** What a matrix product computes, and its shapes: `(m, k) @ (k, n) -> (m, n)` costs `2mkn`
  operations.
- **The chain rule** for functions of many variables, and the idea of a gradient as "how much the output moves per unit
  of each input".
- **Probability basics.** Softmax, log-likelihood, and why cross-entropy is the loss for picking one token out of many.

Resources, in the order they help:

- Andrej Karpathy, [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html): the micrograd, makemore,
  GPT, and tokenizer lectures cover stages 1 to 5.
- 3Blue1Brown, [Essence of linear algebra](https://www.3blue1brown.com/topics/linear-algebra) and
  [Neural networks](https://www.3blue1brown.com/topics/neural-networks).
- T. Parr and J. Howard, [The Matrix Calculus You Need For Deep Learning](https://arxiv.org/abs/1802.01528).
- Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html): compute,
  memory, and overhead, the three costs behind every speed result in this project.
- [Stanford CS336](https://stanford-cs336.github.io/): the assignments this project follows for training, systems,
  scaling laws, and GRPO.
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/): the arithmetic of FLOPs, bytes, and bandwidth,
  and everything about many GPUs that this project leaves out.

## How to work each stage

The repository's tests compare every piece against a reference: PyTorch, Hugging Face's transformers, diffusers, or
finite differences. That makes the repo its own grader. Rebuild against it.

1. **Read the post's section** for the why: what problem the piece solves and what number it ended on.
2. **Read the files** for the how. Write the shape of every array in the margin.
3. **Delete and rebuild.** On a branch (`git checkout -b rebuild`, after committing), replace the bodies of the stage's
   functions with `raise NotImplementedError`, keep the signatures and docstrings, and write them again. Start with the
   smallest version that could pass.
4. **Run the stage's check** against your version. When it fails, find out why before you look at the original.
5. **Predict, then measure.** Each stage names a number. Work it out on paper first, then run the benchmark. If the
   measurement differs from your prediction, the difference is the lesson: explain it before moving on.
6. **Explain it in three sentences**, out loud or in writing, to someone who knows none of it.

## How to think about the problems

These are the habits that mattered most in this project. Each one came from a bug or a wrong guess it made, so each
comes with that story.

**Count bytes and FLOPs before you run.** The first version of prefill computed logits for every prompt token: 151,936
numbers per token, 4 bytes each. On a long prompt that is gigabytes for rows nobody reads, and the arithmetic said so
before the out-of-memory error did. The fix split the model into `hidden()` and `project()` and projected only the last
position. Before running anything big, write down its memory and its operation count.

**Know how your runtime executes.** MLX is lazy: an operation records itself and runs when a value is read. Activation
checkpointing saved no memory on the Apple GPU until it forced evaluation at each block's boundary, because the lazy
graph kept everything alive anyway. Later, Stable Diffusion's decoder had 5.6 GB of live arrays but a 12.4 GB process
footprint, because MLX keeps freed buffers for reuse. Capping that cache brought it to 6.0 GB at the same speed.

**Watch dtypes like shapes.** Under NumPy 2, dividing a float32 array by `np.sqrt(n)`, a float64, gives float64. One
LoRA initializer did that, every later layer became float64, and the RTX 3090 ran the whole fine-tune at its float64
rate: 136 minutes instead of 14.8. Check the dtype of what comes out, not only the shape.

**Parallel reductions are not exact.** On the GPU, a float32 running sum computed in parallel is not guaranteed to
increase, so "count the entries below a random number" picked wrong tokens. Taking the first entry that crosses the
number instead disagreed with exact sampling in 2 of 20,480 draws, against 714 for counting
(`bench/results/sampling_cuda.json`).

**Use a large finite number for masks, never minus infinity.** A softmax over a row that is all minus infinity is NaN,
and NaN times zero is NaN, so one fully masked padding row poisons real rows. minitorch masks with -1e9. The same choice
has a cost worth knowing: near -1e9, float64's spacing is about 1e-7, too coarse for a finite-difference check, so the
padding gradient test scores only real positions.

**Suspect the metric before the model.** GRPO's first reward counted the final number alone. The model learned to
write its answer, then repeat "The final answer is ..." until the token limit, and the reward could not tell. LoRA's
training loss fell while test accuracy dropped. When an objective improves and the thing you care about does not, read
the outputs.

**Compare paired, not totals.** After GRPO the model solved 139 problems against 152 before. It fixed 25 and broke 38,
and a split that uneven comes up 13% of the time by chance alone. Totals hide churn; a paired sign test
(`bench/gsm8k_compare.py`) says whether a difference is real.

**Predict, then measure, then explain the gap.** Counting matrix products predicts attention's backward at 4.4 times its
forward. The profiler measured 14.5. The gap is memory traffic the count ignores, and closing it is what FlashAttention
is for. Speculative decoding gained only 1.07x because the draft model costs 61% of the target per step; the arithmetic
of launches predicts that, once you count launches.

**Measure on a quiet machine.** The same Stable Diffusion step took 2.55 s on an idle laptop and 18.7 to 21.6 s while
the system was swapping. An API server stopped by killing its `uv run` wrapper left its Python child holding 2.4 GB for nine
hours. Before trusting a timing, check swap and what else is running.

**Test the reference before blaming your code.** The agent never called its calculator in 250 problems. Before
debugging minitorch, the same prompt went through Hugging Face's own model, which ignored the tool too: the prompt said
"solve it step by step", and the model obeyed that instead. One check against the reference saved a long search.

**Shared state breaks under threads.** `no_grad` began as one process-wide flag. While the chat engine sat inside it,
a model built by another request thread came out frozen, kept its random weights, and stayed cached. The flag is now per
thread. Anything global in a server is a bug waiting for a second request.

**A test that cannot fail is not a test.** The LayerNorm gradient check summed the output, and with a plain sum the true
input gradient of LayerNorm is exactly zero, so a wrong gradient passed. Multiplying by random weights first made it a
real check. When you write a test, break the code on purpose and watch the test fail.

## The stages

Stages follow the post's sections. Each lists what to read, what to rebuild, how to check it, a number to predict, and
where to read further.

### 1. Autograd (post §2)

Every gradient in minitorch lives in `minitorch/mlops.py`, in nineteen operations with a `forward` and a `backward`.
Everything else composes them.

- Read `minitorch/tensor.py` (the `Tensor`, `_apply`, and `backward`'s walk over the graph), then `minitorch/mlops.py`.
- Rebuild `Add`, `Mul`, `Sum`, `Expand`, `Reshape`, `Matmul`, and `backward()`'s topological walk. Then broadcasting:
  the gradient of a broadcast input sums over the broadcast axes.
- Check with `uv run python test_grads.py` (finite differences) and `uv run python tests/parity.py` (PyTorch).
- Predict the gradient of the post's one-weight example, `L = (2w - 6)^2` at `w = 1`, and the value of `w` after one
  step at learning rate 0.1.
- Read further: Karpathy's micrograd lecture; Rumelhart, Hinton, and Williams,
  [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0).

### 2. Modules, layers, losses, optimizers (post §2 and §4)

- Read `minitorch/module.py`, `layers.py`, `functional.py`, `loss.py`, `optim.py`.
- Rebuild `Linear`, `LayerNorm`, `RMSNorm`, softmax with the max subtracted, cross-entropy, and AdamW.
- Check with `uv run python tests/parity.py`.
- Predict what softmax returns for scores of 1000 and 1001 without subtracting the max, and with it.
- Read further: [Adam](https://arxiv.org/abs/1412.6980) and [AdamW](https://arxiv.org/abs/1711.05101).

### 3. Tokenizer (post §3)

- Read `minitorch/tokenizer.py` and `scripts/tinystories.py`.
- Rebuild byte-level BPE training: count adjacent pairs, merge the most frequent, update the counts without recounting
  everything. Then `encode` and `decode`.
- Check with `uv run python tests/tokenizer.py`: GPT-2's merges loaded into your tokenizer must match tiktoken, and
  Qwen3's must match Hugging Face's.
- Predict how many merges a 10,000-token vocabulary needs, given 256 byte tokens and the special tokens.
- Read further: Sennrich et al., [BPE](https://arxiv.org/abs/1508.07909); Karpathy's tokenizer lecture.

### 4. The transformer (post §4)

- Read `minitorch/attention.py` (RoPE, grouped-query attention, the KV cache, the padding mask) and `minitorch/gpt.py`.
- Rebuild causal attention from the formula, then RoPE, then a block, then the model.
- Check with `uv run python test_grads.py` (the attention checks) and `uv run python tests/parity.py`.
- Predict the story model's parameter count from its configuration: vocabulary 10,000, width 512, 4 layers, feed-forward
  width 1,344, untied output layer. The post's Table 2 has the answer.
- Read further: [Attention Is All You Need](https://arxiv.org/abs/1706.03762),
  [RoFormer](https://arxiv.org/abs/2104.09864), [RMSNorm](https://arxiv.org/abs/1910.07467),
  [GLU variants](https://arxiv.org/abs/2002.05202), [GQA](https://arxiv.org/abs/2305.13245).

### 5. Training (post §5, appendix: scaling laws)

- Read `train_gpt.py`, `minitorch/optim.py` (`cosine_lr`, `clip_grad_norm`), and `bench/m1_parity.py`.
- Rebuild the training loop: warmup and cosine schedule, clipping, evaluation, checkpoints.
- Check with `uv run python bench/m1_parity.py --steps 200` (step-for-step against PyTorch) and
  `uv run python tests/ablations.py`. Then a smoke run:
  `uv run python train_gpt.py --run smoke --steps 300 --d-model 128 --layers 2 --heads 4 --d-ff 384`.
- Predict the loss at step 1 for a 10,000-token vocabulary, and the training FLOPs of the full run from `6 N D`.
- Read further: [TinyStories](https://arxiv.org/abs/2305.07759); CS336 assignment 1; for the appendix,
  [Chinchilla](https://arxiv.org/abs/2203.15556) and `scripts/scaling.py`.

### 6. Backends (post §6, appendix: mixed precision, checkpointing)

- Read `minitorch/backend/cpu.py`, `metal.py`, `cuda.py`, then `minitorch/checkpoint.py` and `autocast` in `tensor.py`.
- Rebuild the MLX backend: most of it is re-exports, and the two places it differs (immutable arrays, lazy evaluation)
  are the lesson.
- Check with `uv run python tests/backends.py --device metal`, `uv run python tests/mixed_precision.py`, and
  `uv run python tests/checkpoint.py`.
- Predict the array size at which the GPU beats NumPy for an elementwise multiply, then run
  `uv run python bench/backend_overhead.py` and read Figure 9.
- Read further: Horace He's post; [Mixed Precision Training](https://arxiv.org/abs/1710.03740);
  [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174).

### 7. Kernels (post §7)

- Read `minitorch/backend/metal_attention.py` (three versions, each fixing the last one's bottleneck),
  `minitorch/backend/cuda_attention.py` (FlashAttention-2 in Triton), and `minitorch/profile.py`.
- Rebuild the one-pass softmax on paper first (the post's worked example with scores 1, 3, 2), then kernel v1, then v2.
- Check with `uv run python tests/backends.py --device metal` (every kernel against a 64-bit reference), then
  `uv run python bench/flash_attention.py` and `uv run python bench/profile_step.py --device metal`.
- Predict attention's backward-to-forward cost ratio by counting matrix products, before reading the profile.
- Read further: [online softmax](https://arxiv.org/abs/1805.02867), [FlashAttention](https://arxiv.org/abs/2205.14135),
  [FlashAttention-2](https://arxiv.org/abs/2307.08691), the
  [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html).

### 8. A real model (post §8, appendix: other models)

- Read `minitorch/state.py` (the safetensors reader), `minitorch/qwen3.py`, then `examples/gpt2.py`.
- Rebuild the safetensors reader and the name mapping from Hugging Face's tensor names to minitorch's parameters.
- Check with `uv run python tests/qwen3.py` and `uv run python tests/gpt2.py`.
- Predict the bytes of Qwen3-0.6B's weights in bfloat16, and why converting bfloat16 to float32 loses nothing.
- Read further: the [safetensors format](https://github.com/huggingface/safetensors);
  [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388).

### 9. Serving (post §9)

- Read `minitorch/serve/kv_cache.py`, `scheduler.py`, `runner.py`, `engine.py`, then `api.py`'s chat endpoint.
- Rebuild the block manager and continuous batching: admit between steps, release on finish, preempt the newest when
  blocks run out.
- Check with `uv run python tests/serve.py`: every request must get the tokens it would get running alone.
- Predict the KV cache size per token of Qwen3-0.6B: 28 layers, 8 key-value heads, 128 numbers per head, keys and
  values, bfloat16.
- Read further: [PagedAttention](https://arxiv.org/abs/2309.06180),
  [Orca](https://www.usenix.org/conference/osdi22/presentation/yu).

### 10. Decoding faster (post §10, appendix: decoding speed-ups)

- Read `minitorch/quant.py`, `minitorch/backend/metal_quant.py`, `minitorch/sampling.py`, `minitorch/speculative.py`.
- Rebuild group-wise quantization, then speculative decoding's acceptance rule.
- Check with `uv run python tests/quant.py` and `uv run python tests/speculative.py`.
- Predict the decode speedup from int4 if weight bytes were the whole cost, then compare with
  `uv run python bench/quant.py` and Table A8.
- Read further: [speculative decoding](https://arxiv.org/abs/2211.17192),
  [speculative sampling](https://arxiv.org/abs/2302.01318).

### 11. LoRA (post §11)

- Read `minitorch/lora.py`, `train_lora.py`, `bench/gsm8k.py`.
- Rebuild `LoRALinear`, merging, and the masked loss over the solution tokens.
- Check with `uv run python tests/lora.py`.
- Predict LoRA's trainable parameter count for Qwen3-0.6B at rank 16 across its seven matrices per layer.
- Read further: [LoRA](https://arxiv.org/abs/2106.09685), [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/),
  [GSM8K](https://arxiv.org/abs/2110.14168).

### 12. GRPO (post §12)

- Read `train_grpo.py` (`reward`, `group_advantages`, `policy_loss`), `bench/rollout.py`, `bench/gsm8k_compare.py`.
- Rebuild the group advantages and the loss, then the batched sampling with left padding.
- Check with `uv run python tests/grpo.py`.
- Predict the advantages in a group of four replies where one is right.
- Read further: [REINFORCE](https://link.springer.com/article/10.1007/BF00992696),
  [DeepSeekMath](https://arxiv.org/abs/2402.03300), [Dr. GRPO](https://arxiv.org/abs/2503.20783); CS336 assignment 5.

### 13. Retrieval and an agent (post §13)

- Read `apps/rag.py`, `apps/agent.py`, and the `/api/rag` and `/api/agent` endpoints in `minitorch/serve/api.py`.
- Rebuild the embedder's pooling, the index, the search, and the tool loop. Rebuild the calculator last, and make it
  parse, never execute.
- Check with `uv run python tests/agent.py`, then run both apps against the server
  (`uv run python apps/rag.py`, `uv run python apps/agent.py`).
- Predict the operation count of one search over 2,067 paragraph vectors of 1,024 numbers, and so its time.
- Read further: [RAG](https://arxiv.org/abs/2005.11401), [SQuAD](https://arxiv.org/abs/1606.05250),
  [Qwen3 Embedding](https://arxiv.org/abs/2506.05176), [ReAct](https://arxiv.org/abs/2210.03629).

### 14. Optional: Stable Diffusion (appendix: other models)

- Read `examples/stable_diffusion.py`: CLIP's tokenizer and text encoder, the UNet, the VAE decoder, DDIM.
- Check with `uv run python tests/stable_diffusion.py` (with nothing else loaded: it needs several GB).
- Read further: [DDPM](https://arxiv.org/abs/2006.11239), [DDIM](https://arxiv.org/abs/2010.02502),
  [latent diffusion](https://arxiv.org/abs/2112.10752), [CLIP](https://arxiv.org/abs/2103.00020),
  [classifier-free guidance](https://arxiv.org/abs/2207.12598), and Lilian Weng's
  [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/).

## Questions to answer without notes

An interviewer will ask some version of these. Answer each in a few sentences, with the numbers.

1. Why does one backward pass cost about two forward passes, whatever the number of weights?
2. What is the gradient of a broadcast addition with respect to the smaller input, and why?
3. Why subtract the maximum inside softmax, and why does that leave the result unchanged?
4. How does BPE choose its merges, and why does the same text encode to the same ids every time?
5. Why does RoPE make the score between two positions depend only on their distance?
6. What does pre-norm change against post-norm, and what happened to training without normalization at all?
7. Why can a GPU be slower than NumPy, and at what array size does that flip?
8. How does the one-pass softmax rescale a running sum, and why does FlashAttention never write the score matrix?
9. Why does decode speed depend on memory bandwidth more than on arithmetic, and what sets the ceiling?
10. What does a paged KV cache save over reserving the maximum length per request, and what does it cost?
11. Why did int4 weights speed up the decode step by much less than they cut its bytes?
12. Why does speculative decoding leave the output distribution unchanged, and why did it gain only 1.07x here?
13. Why does LoRA start from exactly the base model, and why did it make the model worse at GSM8K?
14. What is a group-normalized advantage, and how did the first GRPO reward get gamed?
15. How does a paired sign test work, and what did it say about GRPO's 139 against 152?
16. How does retrieval find a paragraph, and why is closed-book SQuAD a floor rather than a baseline?
17. Why did asking for a calculator make the model solve fewer problems?
18. What would you change first to make this framework faster, and how would you measure it?

## What this project leaves out

Learn these elsewhere; they are where the field's hard engineering lives.

- **Training on many GPUs.** Data parallelism, sharded optimizer state (ZeRO, FSDP), tensor and pipeline parallelism.
  CS336 assignment 2's distributed part and the sharding chapters of How to Scale Your Model cover it.
- **Data.** Filtering, deduplication, and mixing a pretraining corpus: CS336 assignment 4.
- **Mixture-of-experts models**, and the routing and load balancing they need.
- **Inference at scale.** A decode-attention kernel that reads the KV cache through block tables, so a server can batch
  conversations of different lengths; read vLLM's source.
- **Evaluation at scale.** Many benchmarks, many seeds, and confidence intervals on every number.

## Checklist

- [ ] 1. Autograd: `test_grads.py` and `tests/parity.py` pass on your rebuild
- [ ] 2. Modules, layers, losses, optimizers
- [ ] 3. Tokenizer: matches tiktoken and Hugging Face
- [ ] 4. The transformer
- [ ] 5. Training: loss curve matches PyTorch step for step
- [ ] 6. Backends: Metal agrees with NumPy
- [ ] 7. Kernels: within 1e-4 of the 64-bit reference
- [ ] 8. A real model: Qwen3 logits match Hugging Face
- [ ] 9. Serving: batched requests get their solo tokens
- [ ] 10. Decoding faster
- [ ] 11. LoRA
- [ ] 12. GRPO
- [ ] 13. Retrieval and an agent
- [ ] 14. Optional: Stable Diffusion
- [ ] All eighteen questions answered without notes
