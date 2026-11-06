"""The CS336 ablation switches on TransformerLM, and attn_res: each variant builds, trains, and SiLU matches SwiGLU's size.

    uv run python tests/ablations.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from minitorch import AdamW, Tensor, TransformerLM, cross_entropy_loss
from minitorch.layers import DepthAttention

D = 48
VARIANTS = {"baseline": {}, "no RMSNorm": {"norm": "none"}, "post-norm": {"norm": "post"},
            "NoPE": {"rope": False}, "SiLU": {"ffn": "silu", "d_ff": 4 * D},
            "AttnRes": {"attn_res": True, "attn_res_block": 2}}  # blocks of 2 sublayers, so 2 layers cross a boundary


def main():
    np.random.seed(0)
    idx = np.random.randint(0, 40, (4, 32))
    sizes, first = {}, {}
    for name, opts in VARIANTS.items():
        opts = {"d_ff": 128} | opts  # SwiGLU's 128 is 8/3 x 48, the CS336 ratio
        np.random.seed(1)  # the same draws for every variant, so a switch the model ignored shows as an identical first loss
        model = TransformerLM(vocab_size=40, context_length=64, d_model=D, n_layers=2, n_heads=4, **opts)
        sizes[name] = sum(p.data.size for p in model.parameters())
        opt = AdamW(model.parameters(), lr=3e-3)
        losses = []
        for _ in range(40):  # memorize one batch: every variant should manage that
            loss = cross_entropy_loss(model(idx), np.roll(idx, -1, axis=1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < 0.8 * losses[0], (name, losses[0], losses[-1])
        first[name] = losses[0]
        print(f"[ok]   {name:11s} {sizes[name]:,} parameters, loss {losses[0]:.2f} -> {losses[-1]:.2f} in 40 steps")
    same = [name for name in VARIANTS if name != "baseline" and first[name] == first["baseline"]]
    assert not same, f"these variants compute exactly what the baseline does: {same}"
    print(f"[ok]   from identical weights, every variant's first loss differs from the baseline's {first['baseline']:.4f}")
    ffn_gap = abs(sizes["SiLU"] - sizes["baseline"]) / sizes["baseline"]
    assert ffn_gap < 0.02, ffn_gap
    print(f"[ok]   SiLU at d_ff = 4 d_model is within {ffn_gap:.1%} of SwiGLU's parameter count")

    # a variant's checkpoint must not load into another variant: it would run in part on random weights
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "silu.npz"
        TransformerLM(97, 16, 32, 2, 4, 64, ffn="silu").save(path)
        try:
            TransformerLM(97, 16, 32, 2, 4, 64).load(path)
            raise AssertionError("a SiLU checkpoint loaded into a SwiGLU model")
        except AssertionError as e:
            assert "does not match" in str(e), e
    print("[ok]   a SiLU checkpoint is refused by a SwiGLU model, by name")

    # Attention Residuals starts from an average: with its query at zero, every source gets the same weight
    sources = [Tensor(np.random.randn(2, 3, D)) for _ in range(3)]
    gap = np.abs(DepthAttention(D)(sources).numpy() - np.mean([s.numpy() for s in sources], axis=0)).max()
    assert gap < 1e-12, gap
    print(f"[ok]   a fresh DepthAttention mixes its sources as a plain average (largest gap {gap:.0e})")
    with tempfile.TemporaryDirectory() as tmp:  # the option off leaves the model's parameters, so old checkpoints load
        path = Path(tmp) / "base.npz"
        TransformerLM(97, 16, 32, 2, 4, 64).save(path)
        TransformerLM(97, 16, 32, 2, 4, 64, attn_res=False).load(path)
    print("[ok]   attn_res=False is the model checkpoints were saved from")


if __name__ == "__main__":
    main()
