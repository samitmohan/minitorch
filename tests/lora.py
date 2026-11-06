"""LoRA: only the adapter trains, the base weights never change, and merging changes nothing.

    uv run python tests/lora.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from minitorch import AdamW, TransformerLM, cross_entropy_loss, no_grad
from minitorch.lora import apply_lora, merge_adapter, save_adapter

CONFIG = {"vocab_size": 50, "context_length": 16, "d_model": 32, "n_layers": 2, "n_heads": 4, "d_ff": 64, "n_kv_heads": 2}


def main():
    np.random.seed(0)
    base_state = TransformerLM(**CONFIG).state_dict()
    model = TransformerLM(**CONFIG)
    model.load_state_dict(base_state)
    base = dict(model.named_parameters())  # the same Tensor objects stay inside the LoRA wrappers
    apply_lora(model, r=4, alpha=8)

    names = [n for n, _ in model.named_parameters()]
    assert names and all(n.endswith((".A", ".B")) for n in names), names[:4]
    assert len(names) == 2 * 7 * CONFIG["n_layers"], f"{len(names)} adapter tensors"
    print(f"[ok]   only the adapter trains: {len(names)} tensors, {sum(p.data.size for p in model.parameters()):,} numbers")

    idx = np.random.randint(0, 50, (2, 12))
    opt = AdamW(model.parameters(), lr=1e-2)
    for _ in range(3):  # a few steps, so B moves off zero and the merge has something to merge
        loss = cross_entropy_loss(model(idx), (idx + 1) % 50)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert all(np.array_equal(base[n].data, base_state[n]) for n in base_state), "a frozen weight changed"
    print("[ok]   three AdamW steps leave every base weight bit-identical")

    # one float64 adapter would turn every activation after it into float64, which an RTX 3090 runs at 1/64 of float32's rate
    dtypes = {str(p.data.dtype) for p in model.parameters()} | {str(loss.dtype)}
    assert dtypes == {"float32"}, f"adapter or loss promoted to {dtypes}"
    print("[ok]   the adapter, and the loss it feeds, stay float32 through training")

    with no_grad():
        wrapped = model(idx).numpy()
    path = Path(tempfile.mkdtemp()) / "adapter.npz"
    save_adapter(model, path, r=4, alpha=8)
    merged = TransformerLM(**CONFIG)
    merged.load_state_dict(base_state)
    n = merge_adapter(merged, path)
    with no_grad():
        gap = float(np.abs(merged(idx).numpy() - wrapped).max())
    assert n == 2 * 7 and gap < 1e-5, (n, gap)
    print(f"[ok]   merging {n} adapted layers reproduces the wrapped model's logits (largest gap {gap:.1e})")


if __name__ == "__main__":
    main()
