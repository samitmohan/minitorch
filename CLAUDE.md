# minitorch

- Backends: NumPy (reference), CuPy (CUDA, training throughput on bqa3), MLX (Metal).
- Custom kernels: Metal through `mx.fast.metal_kernel` on the Mac, and Triton on the NVIDIA
  path (decided 2026-11-05, for CS336 A2's FlashAttention-2). Ruff TID251 bans CuPy's raw
  kernel APIs (`cupy.RawKernel`, `RawModule`, `ElementwiseKernel`, `ReductionKernel`).
- PyTorch, tiktoken, and HF transformers are test oracles and comparison baselines
  only (dev dependency group). Never import them from `minitorch/`.
- UI: FastAPI (`minitorch/serve/api.py`) + React/Vite/TS in `ui/`.
  The UI is one blog post and nothing else (decided 2026-11-05): no dashboard pages.
  Every experiment and result lives in the post, as prose plus live figures from
  bench/results and runs/ (reference style: rikiyat.github.io/blog/weight-space-programming).
  Every number in it comes from those files. The whole post reads as one story (decided
  2026-11-05): each section opens on the problem the previous section's result left open and
  ends on a measured answer. A section, figure, or experiment the chain does not need stays
  out of the post and out of the plan. Write for a reader who knows none of this:
  explain each idea from first principles before using it. The prose passes the stop-slop
  rules (~/.claude/skills/stop-slop), and the design stays as plain as the reference: white
  page, one text column, light headings, centered captions, no decoration.
  The post reads as a research paper (user, 2026-11-06): Abstract, numbered sections, a Conclusion with
  limitations, References, Appendix. Its goal, stated in §1: a from-scratch PyTorch/tinygrad built in study-plan
  order up to RAG and an agent on its own server, both live in §13; §14 is the conclusion, its opening paragraph only (user, 2026-11-06: related work and the
  conclusion's subsections were cut; the acknowledgements are one line in the user's words). Section titles are the questions each section answers. GPT-2 and Stable Diffusion appear only in the
  appendix (Other models), as tinygrad-style examples with a snippet; side results (quantization tables,
  speculative decoding, scaling laws) go to the appendix with one sentence in the body.
- The post also runs as a static page at samitmohan.in/blog/minitorch (user, 2026-11-06), titled "Modern AI: Autograd to Agents"
  and set in the site's type and colours (its own structure and prose kept). Its source is a port of ui/src in
  ../portfolio/web/src/paper/minitorch: sections render on the server, and the live figures replay recordings. With the
  server running, `uv run python scripts/record_replays.py` records them into bench/replays/; then
  `uv run python scripts/export_paper.py ../portfolio/web` copies the results and replays into the site. A prose change
  here has to be made there too.
- Every gradient lives in `minitorch/mlops.py`. Everything else composes those ops.
- `TransformerLM(attn_res=True)` is Block Attention Residuals (DepthAttention in layers.py), an option only (user,
  2026-11-06): no retraining or measuring it. The paper's §5 says so in one paragraph and links the user's Attention
  Residuals post.
- Tensors are lazy (user, 2026-11-06: "the realize thing is what i want just that"): `_apply` records an op, and
  `.data` / `.numpy()` / `.item()` / `.backward()` / `.realize()` run it. A new primitive needs `out_shape` (and
  `out_dtype` when it changes the type) or every result built on it gets the wrong shape. No fusion or codegen: the
  user scoped it to lazy evaluation alone. Composite ops in tensor.py must not read `.data` (that runs the graph
  early); use `.shape` and `.dtype`, which never run anything.
- Ponytail mode (full) for all code: the smallest complete change, no unrequested
  options or abstractions, and every deliberate limit marked
  `# shortcut: <the limit>, <when to upgrade>`. Milestone checkpoints get a
  ponytail-review pass (correct, safe, holds under load, tested, fast, lean).
- No code golf (tinygrad's rule, adopted 2026-11-05). Low line count guides the project, but
  the goal is less complexity and more readability. In the UI, `npm run lint` (oxlint, config
  `ui/.oxlintrc.json`) bans the comma operator, `cond && f()` as a statement, and nested
  ternaries. Ruff (pyproject extend-select E701-E703, E731) bans semicolon statements and assigned lambdas. Lint can't
  catch the rest, so write the plain `if` instead of a dense one-liner a reader has to decode.
- Git: do not commit. The user reviews diffs and commits at milestone checkpoints.
- On the Mac, run one model-loading job at a time (user, 2026-11-05): 16 GB of unified memory overflowed
  with an eval, the API server, and the RAG embedder up together. Wait for one to finish before starting the next.
  Stop a server with `pkill -f 'uvicorn minitorch.serve.api'`, then confirm with `pgrep -f` that it is gone: killing
  the `uv run` wrapper's pid leaves its Python child running (an orphan server held 2.4 GB for 9 hours on 2026-11-05).
- Never use bqa1 for this project (user, 2026-11-05). All GPU work runs on bqa3.
- CuPy runs on bqa3 (`ssh bqa3`, RTX 3090): `uv sync --extra cuda --no-dev` (the dev group
  pulls torch, gigabytes on a 98%-full disk), code in `~/samit_workspace/minitorch`.
  `tests/backends.py --device cuda` must pass after any change to mlops or the backends.
- Plan and milestones: SPEC.md.
- Check: `uv run python test_grads.py`, `uv run python tests/parity.py`,
  `uv run python tests/serve.py`, `uv run python tests/lora.py`, `uv run python tests/grpo.py`, `uv run python tests/speculative.py`,
  `uv run python tests/checkpoint.py`, `uv run python tests/ablations.py`, `uv run python tests/agent.py`, `uv run python tests/mixed_precision.py`,
  `uv run python tests/tokenizer.py`, `uv run python tests/lazy.py`,
  `uv run python tests/backends.py --device metal`; with the Metal GPU and data/Qwen3-0.6B:
  `uv run python tests/quant.py`, `uv run python tests/qwen3.py`; with data/gpt2: `uv run python tests/gpt2.py`; with data/sd15 (and nothing else loaded): `uv run python tests/stable_diffusion.py`. UI: `cd ui && npm run lint && npx vitest run && npx playwright test`.
  Lint: `uv run ruff check minitorch tests bench scripts examples apps train_gpt.py train_lora.py train_grpo.py test_grads.py`.
