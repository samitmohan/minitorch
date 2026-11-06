# minitorch

A deep learning framework written from scratch in Python: an autograd engine with every gradient in
one file, three array backends (NumPy, Apple's GPU through MLX, NVIDIA's through CuPy), and custom
attention kernels in Metal and Triton. It trains a language model, loads and serves Qwen3-0.6B,
fine-tunes it with LoRA and GRPO, and runs retrieval and a tool-using agent on its own server.

- The results, written up as one paper with live figures: run the server
  (`uv run uvicorn minitorch.serve.api:app --port 8000` after `cd ui && npm run build`) and open
  `http://localhost:8000`.
- The headline numbers and how to reproduce them: the repository's README.
- The study plan for rebuilding it yourself, stage by stage: the Study plan page.
- The API reference: the pages in the menu.
