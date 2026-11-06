"""Freeze the post's data for a static copy of it, such as the one on samitmohan.in/blog/minitorch.

    uv run python scripts/export_paper.py ../portfolio/web

Writes, under that site's root:
  src/paper/minitorch/data/results.json  what /api/bench, /api/runs and the metric streams give the live post
  src/paper/minitorch/data/replays.json  bench/replays/ (scripts/record_replays.py), for the figures that call the server
  public/paper/minitorch/                the attention patterns (fetched by their figure) and Stable Diffusion's images
"""

import argparse
import json
import shutil
from pathlib import Path

from minitorch.serve.api import RUNS, _read_jsonl, bench, runs

ROOT = Path(__file__).resolve().parent.parent
CURVE_POINTS = 700  # RunCurve draws at most this many points; sending more to a static page only adds bytes


def downsample(rows, n):
    """Every k-th row and the last, as ui/src/lib.ts's downsample picks them."""
    if len(rows) <= n:
        return rows
    k = -(-len(rows) // n)
    out = rows[::k]
    return out if out[-1] is rows[-1] else out + [rows[-1]]


def median(xs):
    return sorted(xs)[len(xs) // 2]  # kit.tsx's median: the upper middle of an even count


def slim(rows, fields):
    return [{k: r[k] for k in fields if r.get(k) is not None} for r in rows]


def story_run():
    """The TinyStories run, cut to the curve the post draws, with the medians the prose quotes taken over every step."""
    rows = _read_jsonl(RUNS / "tinystories" / "metrics.jsonl")
    evals = [r for r in rows if r.get("val_loss") is not None]
    train = downsample([r for r in rows if r.get("val_loss") is None], CURVE_POINTS - len(evals))
    kept = sorted({id(r): r for r in [rows[0], *train, *evals, rows[-1]]}.values(), key=lambda r: r["step"])
    stats = {"tokS": median([r["tokens_per_s"] for r in rows]), "tflops": median([r["tflops"] for r in rows])}
    if rows[0].get("mfu") is not None:
        stats["mfu"] = median([r.get("mfu") or 0 for r in rows])
    return slim(kept, ["step", "loss", "val_loss", "time"]), stats


def attention(patterns):
    """Each head's causal weights as integers in thousandths, lower triangle only: the server rounds them to 3 decimals."""
    return [{"tokens": p["tokens"], "layers": [[[[round(w * 1000) for w in row[: i + 1]] for i, row in enumerate(head)]
                                                 for head in layer] for layer in p["layers"]]} for p in patterns]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("site", type=Path, help="the site's root, holding src/ and public/")
    site = p.parse_args().site
    data, public = site / "src" / "paper" / "minitorch" / "data", site / "public" / "paper" / "minitorch"
    data.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)

    ts, stats = story_run()
    samples = _read_jsonl(RUNS / "tinystories" / "samples.jsonl")
    results = {
        "bench": bench(), "runs": runs(), "ts": ts, "tsStats": stats,
        "lora": slim(_read_jsonl(RUNS / "lora_gsm8k" / "metrics.jsonl"), ["step", "loss", "val_loss"]),
        "grpo": _read_jsonl(RUNS / "grpo_gsm8k" / "metrics.jsonl"),
        "grpoNoStop": _read_jsonl(RUNS / "grpo_no_stop" / "metrics.jsonl"),
        "sample": samples[-1]["text"] if samples else None,
    }
    (data / "results.json").write_text(json.dumps(results, separators=(",", ":")) + "\n")

    replays = {f.stem: json.loads(f.read_text()) for f in sorted((ROOT / "bench" / "replays").glob("*.json"))}
    (public / "attention.json").write_text(json.dumps(attention(replays.pop("attention")), separators=(",", ":")) + "\n")
    (data / "replays.json").write_text(json.dumps(replays, ensure_ascii=False, separators=(",", ":")) + "\n")
    for name in ["stable_diffusion_minitorch.png", "stable_diffusion_diffusers.png"]:
        shutil.copy(ROOT / "bench" / "results" / name, public / name)

    for f in [data / "results.json", data / "replays.json", public / "attention.json"]:
        print(f"{f.relative_to(site)}: {f.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
