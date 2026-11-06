# minitorch

A deep learning framework written from scratch, and the stack on top of it: from a single derivative to a server that
answers questions from Wikipedia and runs a tool-using agent. Reverse-mode autodiff with every gradient in one file
(`minitorch/mlops.py`), three array backends (NumPy, Apple's GPU through MLX, NVIDIA's through CuPy), attention kernels
in Metal and Triton, a BPE tokenizer, a language model trained from nothing, Qwen3-0.6B loaded and served behind an
OpenAI-style endpoint, LoRA and GRPO fine-tuning, and retrieval-augmented generation and an agent on that server.

The write-up is one paper with live figures: build the UI and start the server (see Run), then open
http://localhost:8000.

## Results

Every number comes from a file the named script writes.

| | Result | Source |
|---|---|---|
| Size | the framework is 3,141 lines of Python (10,552 for the whole repository, tests and write-up included) | `scripts/lines.py` |
| Gradients | a model trained beside PyTorch from the same weights differs from it by 1.8e-15 in loss at the first step (float64) | `bench/m1_parity.py` |
| Training | a 22.7M-parameter model trains on TinyStories (327.7M tokens) to a validation loss of 1.399 in 71 minutes on one RTX 3090, at 23.9% MFU | `train_gpt.py` |
| A real model | Qwen3-0.6B's logits match Hugging Face's to 5.1e-5 in float32, and greedy decoding reproduces its first 24 tokens | `tests/qwen3.py` |
| Retrieval | on SQuAD v1.1, the source paragraph is among the top three 91.5% of the time, and 55% of answers then contain the reference | `apps/rag.py` |

Also measured: the Metal attention kernel runs 6.7x faster than MLX's own at the training shape
(`bench/flash_attention.py`), and FlashAttention-2 in Triton runs forward and backward 8.5x faster than the composed
version on the RTX 3090 (`bench/flash_attention_cuda.py`).

What did not work, on 250 GSM8K problems (greedy, thinking off, 320 new tokens), against the released model's 152:

| Attempt | Solved | Paired sign test |
|---|---|---|
| LoRA fine-tuning on GSM8K's solutions | 123 | p = 0.0011 |
| GRPO, reward requiring a finished reply | 139 | p = 0.13 |
| A calculator tool, asked for in the prompt | 42 | p < 0.0001 |

(`bench/gsm8k.py`, `bench/gsm8k_compare.py`)

## Layout

```
minitorch/        the framework: tensor.py and mlops.py (autograd), backend/ (NumPy, MLX, CuPy, Metal and Triton
                  kernels), layers, optimizers, attention, gpt.py, qwen3.py, tokenizer.py, lora.py, quant.py, serve/
apps/             rag.py and agent.py, clients of the server
train_*.py        TinyStories pretraining, LoRA, GRPO
examples/         GPT-2 and Stable Diffusion v1.5, one file each, in the style of tinygrad's examples
tests/            each piece against its reference (PyTorch, transformers, diffusers, finite differences)
bench/, scripts/  the measurements; results land in bench/results/
ui/               the write-up: React, reading bench/results/ and runs/ through the server
```

## Setup

```bash
git clone https://github.com/samitmohan/minitorch.git
cd minitorch
uv sync --extra metal --extra serve        # Apple silicon; NVIDIA: uv sync --extra cuda --no-dev
```

## Run

```bash
uv run python test_grads.py                 # every gradient against finite differences
uv run python tests/parity.py               # every op and layer against PyTorch
uv run python scripts/tinystories.py download   # then: train, encode (the data and the 10k BPE, once)
uv run python train_gpt.py --run smoke --steps 300 --d-model 128 --layers 2 --heads 4 --d-ff 384
uv run python scripts/lines.py              # the size table above

cd ui && npm install && npm run build && cd ..
uv run uvicorn minitorch.serve.api:app --port 8000   # the write-up, at http://localhost:8000

uv run --extra docs mkdocs serve            # the study plan (docs/guide.md) and the API reference
```

[SPEC.md](SPEC.md) holds the plan and the decisions behind it.

## License

MIT
