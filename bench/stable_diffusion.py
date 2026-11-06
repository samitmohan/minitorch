"""Stable Diffusion in minitorch against diffusers on the same Mac: speed per step, and how far apart the images land.

    uv run python bench/stable_diffusion.py

Both pipelines turn the same prompt and the same starting noise into a 512x512 image with 25 DDIM
steps and guidance 7.5, in float32 on the Apple GPU: minitorch through MLX, diffusers through
PyTorch's MPS backend. They run one after the other, so only one sits in memory. A step is one
UNet call on the prompt and the empty prompt together, timed until the GPU finishes. Writes
bench/results/stable_diffusion.json and both images, then profiles one UNet call op by op.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from examples.stable_diffusion import generate, load_unet
from minitorch import Tensor, backend, no_grad
from minitorch.profile import profile

PATH = ROOT / "data/sd15"
RESULTS = ROOT / "bench/results"
PROMPT = "a watercolor painting of a lighthouse on a cliff at dusk"
STEPS, GUIDANCE, SIZE = 25, 7.5, 512


def diffusers_generate(noise):
    """The reference pipeline, assembled from the same five folders (the download has no safety checker)."""
    def part(cls, sub):
        return cls.from_pretrained(PATH / sub, variant="fp16", torch_dtype=torch.float32)
    pipe = StableDiffusionPipeline(
        vae=part(AutoencoderKL, "vae"), text_encoder=part(CLIPTextModel, "text_encoder"),
        tokenizer=CLIPTokenizer.from_pretrained(PATH / "tokenizer"), unet=part(UNet2DConditionModel, "unet"),
        scheduler=DDIMScheduler.from_pretrained(PATH / "scheduler"), safety_checker=None, feature_extractor=None,
        requires_safety_checker=False).to("mps")
    marks = []

    def mark(pipe, step, t, kwargs):
        torch.mps.synchronize()  # MPS queues work; wait so each mark lands when the step's work is done
        marks.append(time.perf_counter())
        return kwargs
    start = time.perf_counter()
    image = pipe(PROMPT, num_inference_steps=STEPS, guidance_scale=GUIDANCE, latents=torch.tensor(noise).to("mps"),
                 output_type="np", callback_on_step_end=mark).images[0]
    seconds = np.diff([start, *marks])  # the first interval also holds the prompt's encoding, a few ms
    return image, seconds


def profile_unet():
    """One guided UNet call (batch 2, a 64x64 latent) on the Apple GPU, timed one primitive at a time."""
    unet = load_unet(PATH, "metal")
    x = Tensor(np.zeros((2, 4, SIZE // 8, SIZE // 8), np.float32), device="metal")
    context = Tensor(np.zeros((2, 77, 768), np.float32), device="metal")  # its values don't change the work
    with no_grad():
        unet(x, 500, context).numpy()  # warm-up: compiles the kernels
        with profile() as prof:
            unet(x, 500, context).numpy()
    return {"wall_ms": round(prof.wall_ms, 1), "rows": [{"op": op, "calls": n, "ms": round(ms, 2)} for op, n, ms in prof.rows()]}


def main():
    noise = np.random.default_rng(0).standard_normal((1, 4, SIZE // 8, SIZE // 8)).astype(np.float32)
    mine, mine_s = generate(PATH, PROMPT, noise, STEPS, GUIDANCE, "metal")
    backend.get("metal").clear_cache()  # hand MLX's cached buffers back before PyTorch loads its copy
    ref, ref_s = diffusers_generate(noise)
    for name, image in (("minitorch", mine), ("diffusers", ref)):
        Image.fromarray((image * 255).round().astype(np.uint8)).save(RESULTS / f"stable_diffusion_{name}.png")
    unet_profile = profile_unet()
    gap = np.abs(mine - ref)
    result = {"prompt": PROMPT, "steps": STEPS, "guidance": GUIDANCE, "size": SIZE, "dtype": "float32",
              "minitorch_metal_s_per_step": round(float(np.median(mine_s)), 3),
              "diffusers_mps_s_per_step": round(float(np.median(ref_s)), 3),
              "max_pixel_gap": round(float(gap.max()), 4), "mean_pixel_gap": round(float(gap.mean()), 5),
              "pixels_off_by_more_than_1_of_255": round(float((gap > 1 / 255).mean()), 5), "unet_profile": unet_profile}
    (RESULTS / "stable_diffusion.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
