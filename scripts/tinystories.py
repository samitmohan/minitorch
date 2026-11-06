"""TinyStories data: download, train the 10k BPE, encode to uint16 token files.

    uv run python scripts/tinystories.py download
    uv run python scripts/tinystories.py train    # -> data/tinystories_bpe/{vocab.json,merges.txt}
    uv run python scripts/tinystories.py encode   # -> data/tinystories_{train,valid}.bin

Every measurement lands in bench/results/bpe.json.
"""
import json
import os
import resource
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from minitorch.tokenizer import Tokenizer, chunk_bounds, save, train_bpe

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BPE = DATA / "tinystories_bpe"
RESULTS = ROOT / "bench" / "results" / "bpe.json"
EOT = "<|endoftext|>"
FILES = {"train": "TinyStoriesV2-GPT4-train.txt", "valid": "TinyStoriesV2-GPT4-valid.txt"}


def record(**values):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    results.update(values)
    RESULTS.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(values, indent=2))


def peak_rss_mb(who):
    # macOS reports ru_maxrss in bytes, Linux in KiB
    rss = resource.getrusage(who).ru_maxrss
    return round(rss / 2**20 if sys.platform == "darwin" else rss / 2**10)


def download():
    for name in FILES.values():
        hf_hub_download("roneneldan/TinyStories", name, repo_type="dataset", local_dir=DATA)


def train():
    BPE.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    vocab, merges = train_bpe(DATA / FILES["train"], 10_000, [EOT])
    seconds = time.perf_counter() - t0
    save(vocab, merges, BPE / "vocab.json", BPE / "merges.txt")
    longest = sorted(vocab.values(), key=len, reverse=True)[:10]
    record(bpe_train_seconds=round(seconds, 1), bpe_vocab_size=len(vocab),
           bpe_train_peak_rss_mb_parent=peak_rss_mb(resource.RUSAGE_SELF),
           bpe_train_peak_rss_mb_largest_worker=peak_rss_mb(resource.RUSAGE_CHILDREN),
           bpe_longest_tokens=[t.decode("utf-8", errors="replace") for t in longest])


_tok = None


def _load_tokenizer():
    global _tok
    _tok = Tokenizer.from_files(BPE / "vocab.json", BPE / "merges.txt", [EOT])


def _encode_chunk(args):
    path, start, end = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")
    return np.array(_tok.encode(text), dtype=np.uint16)


def encode():
    _load_tokenizer()
    assert len(_tok.vocab) <= 2**16, "uint16 token files need vocab <= 65536"
    for split, name in FILES.items():
        path, out = DATA / name, DATA / f"tinystories_{split}.bin"
        bounds = chunk_bounds(path, 64)
        t0 = time.perf_counter()
        n_tokens = 0
        with Pool(os.cpu_count(), initializer=_load_tokenizer) as pool, open(out, "wb") as f:
            for ids in pool.imap(_encode_chunk, zip([path] * len(bounds), bounds, bounds[1:])):
                ids.tofile(f)
                n_tokens += len(ids)
        seconds = time.perf_counter() - t0
        size = os.path.getsize(path)
        record(**{f"{split}_tokens": n_tokens, f"{split}_encode_mb_per_s": round(size / 2**20 / seconds, 1),
                  f"{split}_bytes_per_token": round(size / n_tokens, 3)})

    # held-out round trip: valid text -> ids -> text must be byte-identical
    text = (DATA / FILES["valid"]).read_text(encoding="utf-8")
    ids = np.fromfile(DATA / "tinystories_valid.bin", dtype=np.uint16).tolist()
    record(valid_round_trip_exact=_tok.decode(ids) == text)


if __name__ == "__main__":
    {"download": download, "train": train, "encode": encode}[sys.argv[1]]()
