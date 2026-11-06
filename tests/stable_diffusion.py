"""Stable Diffusion in minitorch (examples/stable_diffusion.py) against transformers and diffusers.

    uv run python tests/stable_diffusion.py

One piece at a time, so only one network sits in memory: the tokenizer's ids and the DDIM schedule
and step (exact), then CLIP's 77 vectors, the UNet's noise prediction, and the VAE's decode, all
float32, on the CPU and the Apple GPU. The UNet sees a 32x32 latent and the VAE a 16x16 one: every
layer still runs, and the reference's attention stays small. Writes bench/results/sd_parity.json.
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from examples.stable_diffusion import (
    ClipTokenizer,
    ddim_step,
    ddim_timesteps,
    load_clip,
    load_unet,
    load_vae,
)
from minitorch import Tensor, no_grad

PATH = ROOT / "data/sd15"
PROMPTS = ["a watercolor painting of a lighthouse on a cliff at dusk", "", "A photo of an astronaut riding a horse on Mars!!",
           "it's 3:00pm; the cat's toys cost $12.50 -- naïve café", "tiny " * 100]
TOL = 1e-3  # largest |mine - reference|, as a fraction of the reference's largest value


def check_tokenizer():
    ref = CLIPTokenizer.from_pretrained(PATH / "tokenizer")
    tok = ClipTokenizer(PATH / "tokenizer")
    for prompt in PROMPTS:
        want = ref(prompt, padding="max_length", max_length=77, truncation=True).input_ids
        assert tok.encode(prompt) == want, (prompt, tok.encode(prompt), want)
    print(f"tokenizer    ids match on {len(PROMPTS)} prompts")
    return tok


def check_ddim():
    ref = DDIMScheduler.from_pretrained(PATH / "scheduler")  # SD 1.5 ships a PNDM config; DDIM reads the same keys
    ref.set_timesteps(25)
    times = ddim_timesteps(25)
    assert ref.timesteps.tolist() == times, (ref.timesteps.tolist(), times)
    rng = np.random.default_rng(0)
    x, noise = rng.standard_normal((2, 1, 4, 8, 8)).astype(np.float32)
    worst = 0.0
    for t, t_prev in zip(times, [*times[1:], -1]):
        want = ref.step(torch.tensor(noise), t, torch.tensor(x)).prev_sample.numpy()
        worst = max(worst, float(np.abs(ddim_step(x, noise, t, t_prev) - want).max()))
    assert worst < 1e-5, worst
    print(f"ddim         25 timesteps match, step error {worst:.1e}")
    return worst


def compare(name, ref, model, run, results):
    """run(model, device) on the CPU and the GPU; record each output's error against ref, and the model's size."""
    results[f"{name}_params"] = sum(int(np.prod(p.shape)) for _, p in model.named_parameters())
    for device in ("cpu", "metal"):
        model.to(device)
        with no_grad():
            out = run(model, device)
        error = float(np.abs(out - ref).max() / np.abs(ref).max())
        results[f"{name}_{device}"] = error
        print(f"{name:12s} {device:5s} error {error:.1e} of the largest value {np.abs(ref).max():.2f}")


def main():
    results = {"tokenizer_prompts": len(PROMPTS)}
    tok = check_tokenizer()
    results["ddim_step_error"] = check_ddim()
    rng = np.random.default_rng(0)

    ids = np.array([tok.encode(""), tok.encode(PROMPTS[0])])
    ref = CLIPTextModel.from_pretrained(PATH / "text_encoder", variant="fp16", torch_dtype=torch.float32).eval()
    with torch.no_grad():
        context = ref(torch.tensor(ids)).last_hidden_state.numpy()
    del ref
    compare("clip", context, load_clip(PATH, "cpu"), lambda m, d: m(ids).numpy(), results)
    gc.collect()

    latents = rng.standard_normal((2, 4, 32, 32)).astype(np.float32)
    ref = UNet2DConditionModel.from_pretrained(PATH / "unet", variant="fp16", torch_dtype=torch.float32).eval()
    with torch.no_grad():
        want = ref(torch.tensor(latents), 501, encoder_hidden_states=torch.tensor(context)).sample.numpy()
    del ref
    gc.collect()
    compare("unet", want, load_unet(PATH, "cpu"),
            lambda m, d: m(Tensor(latents, device=d), 501, Tensor(context, device=d)).numpy(), results)
    gc.collect()

    z = rng.standard_normal((1, 4, 16, 16)).astype(np.float32)
    ref = AutoencoderKL.from_pretrained(PATH / "vae", variant="fp16", torch_dtype=torch.float32).eval()
    with torch.no_grad():
        want = ref.decode(torch.tensor(z) / ref.config.scaling_factor).sample.numpy()
    del ref
    gc.collect()
    compare("vae", want, load_vae(PATH, "cpu"), lambda m, d: m.decode(Tensor(z, device=d)).numpy(), results)

    (ROOT / "bench/results/sd_parity.json").write_text(json.dumps(results, indent=1))
    failed = {k: v for k, v in results.items() if k.endswith(("_cpu", "_metal")) and v > TOL}
    assert not failed, f"over {TOL}: {failed}"
    print("all pieces within", TOL)


if __name__ == "__main__":
    main()
