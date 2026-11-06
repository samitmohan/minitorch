"""How big minitorch is: lines per component, the way tinygrad and nanoGPT state their size.

    uv run python scripts/lines.py      # writes bench/results/lines.json

A line counts when it holds code or a docstring: blank lines and comment-only lines do not.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "minitorch"
GROUPS = {
    "engine": [PKG / f for f in ["__init__.py", "tensor.py", "mlops.py", "module.py", "layers.py", "functional.py", "loss.py",
                                 "optim.py", "conv.py", "data.py", "grad_check.py", "checkpoint.py", "profile.py"]],
    "backends and kernels": sorted((PKG / "backend").glob("*.py")),
    "models": [PKG / f for f in ["attention.py", "gpt.py", "qwen3.py", "tokenizer.py", "state.py", "lora.py", "quant.py",
                                 "sampling.py", "speculative.py"]],
    "serving": sorted((PKG / "serve").glob("*.py")),
    "apps (RAG, agent)": sorted((ROOT / "apps").glob("*.py")),
    "training scripts": [ROOT / f for f in ["train_gpt.py", "train_lora.py", "train_grpo.py"]],
    "examples (GPT-2, Stable Diffusion)": sorted((ROOT / "examples").glob("*.py")),
    "tests": [ROOT / "test_grads.py", *sorted((ROOT / "tests").glob("*.py"))],
    "benchmarks and scripts": [*sorted((ROOT / "bench").glob("*.py")), *sorted((ROOT / "scripts").glob("*.py"))],
    "the post (TypeScript)": sorted(p for p in (ROOT / "ui" / "src").rglob("*") if p.suffix in (".ts", ".tsx")),
}
PACKAGE = ["engine", "backends and kernels", "models", "serving"]


def code_lines(path):
    comment = "//" if path.suffix in (".ts", ".tsx") else "#"
    return sum(1 for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith(comment))


def main():
    groups = [{"name": name, "files": len(files), "lines": sum(code_lines(f) for f in files)} for name, files in GROUPS.items()]
    covered = {f for files in GROUPS.values() for f in files}
    missing = [str(p.relative_to(ROOT)) for p in PKG.rglob("*.py") if p not in covered]
    assert not missing, f"package files in no group: {missing}"  # a new module must be counted somewhere
    by_name = {g["name"]: g["lines"] for g in groups}
    result = {"groups": groups, "package": sum(by_name[n] for n in PACKAGE), "repository": sum(by_name.values())}
    (ROOT / "bench/results/lines.json").write_text(json.dumps(result, indent=1))
    for g in groups:
        print(f"{g['name']:36s} {g['files']:3d} files {g['lines']:6,d} lines")
    print(f"{'minitorch package':36s} {result['package']:16,d} lines\n{'whole repository':36s} {result['repository']:16,d} lines")


if __name__ == "__main__":
    main()
