"""Byte-level BPE (the GPT-2 scheme): train merges on a corpus, encode and decode with them.

Text is cut into pre-tokens (a word with its leading space, a digit run, a
punctuation run, whitespace) by the GPT-2 regex. Merges never cross a pre-token
or a special token, so each pre-token is encoded on its own and cached.
"""
import itertools
import json
import os
import unicodedata
from collections import Counter, defaultdict
from multiprocessing import Pool

import regex as re

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def _special_splitter(special_tokens):
    # longest first, so "<|eot|><|eot|>" registered as one token wins over two "<|eot|>"
    alternatives = "|".join(re.escape(s) for s in sorted(special_tokens, key=len, reverse=True))
    return re.compile(f"({alternatives})") if special_tokens else None


def chunk_bounds(path, n_chunks, boundary=b"<|endoftext|>"):
    """Byte offsets that cut the file into ~n_chunks pieces, each cut moved forward to the next `boundary`."""
    size = os.path.getsize(path)
    bounds = [size * i // n_chunks for i in range(n_chunks)] + [size]
    with open(path, "rb") as f:
        for i in range(1, n_chunks):
            f.seek(bounds[i])
            # shortcut: assumes a boundary occurs within 1 MiB of every cut, true for TinyStories; scan further if not
            found = f.read(1 << 20).find(boundary)
            bounds[i] = size if found == -1 else bounds[i] + found
    return sorted(set(bounds))


def _count_pretokens(args):
    path, start, end, special_tokens = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    splitter = _special_splitter(special_tokens)
    pieces = splitter.split(text)[::2] if splitter else [text]  # even slots are the text between specials
    return Counter(m.group() for piece in pieces for m in PAT.finditer(piece))


def train_bpe(input_path, vocab_size, special_tokens, num_workers=None):
    """Learn merges until the vocab (256 bytes + specials + merges) reaches vocab_size."""
    num_workers = num_workers or os.cpu_count()
    bounds = chunk_bounds(input_path, num_workers * 4)
    jobs = [(input_path, s, e, special_tokens) for s, e in itertools.pairwise(bounds)]
    counts = Counter()
    with Pool(num_workers) as pool:
        for c in pool.imap_unordered(_count_pretokens, jobs):
            counts.update(c)

    # each distinct pre-token becomes a list of single-byte tokens, weighted by how often it occurs
    words = [[bytes([b]) for b in w.encode("utf-8")] for w in counts]
    freqs = list(counts.values())
    pair_counts = Counter()
    where = defaultdict(set)  # pair -> ids of words containing it
    for i, w in enumerate(words):
        for pair in itertools.pairwise(w):
            pair_counts[pair] += freqs[i]
            where[pair].add(i)

    vocab = {i: bytes([i]) for i in range(256)}
    for s in special_tokens:
        vocab[len(vocab)] = s.encode("utf-8")
    merges = []
    while len(vocab) < vocab_size and pair_counts:
        # most frequent pair, ties broken by the lexicographically greater pair
        # shortcut: O(distinct pairs) scan per merge; switch to a lazy max-heap if training time matters
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        a, b = best
        merges.append(best)
        vocab[len(vocab)] = a + b
        for i in list(where[best]):
            w, f = words[i], freqs[i]
            for pair in itertools.pairwise(w):
                pair_counts[pair] -= f
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
            words[i] = w = _merge(w, a, b)
            for pair in itertools.pairwise(w):
                pair_counts[pair] += f
                where[pair].add(i)
        del where[best]
    return vocab, merges


def _merge(word, a, b):
    out, i = [], 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == a and word[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


class Tokenizer:
    """Byte-level BPE. `pattern` splits text into pre-tokens (GPT-2's PAT by default); `nfc`
    normalizes text to Unicode NFC first, as Qwen's tokenizer does."""
    def __init__(self, vocab, merges, special_tokens=None, pattern=PAT, nfc=False):
        self.pattern, self.nfc = pattern, nfc
        self.vocab = dict(vocab)
        self.special_tokens = list(special_tokens or [])
        for s in self.special_tokens:
            if s.encode("utf-8") not in self.vocab.values():
                self.vocab[len(self.vocab)] = s.encode("utf-8")
        self.ids = {tok: i for i, tok in self.vocab.items()}
        self.ranks = {pair: r for r, pair in enumerate(merges)}
        self.splitter = _special_splitter(self.special_tokens)
        self.cache = {}

    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None, **options):
        """Load GPT-2 format files: vocab.json and merges.txt, with bytes spelled via bytes_to_unicode."""
        decode = {c: b for b, c in bytes_to_unicode().items()}
        with open(vocab_path, encoding="utf-8") as f:
            vocab = {i: bytes(decode[c] for c in s) for s, i in json.load(f).items()}
        with open(merges_path, encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip() and not ln.startswith("#version")]
        merges = [tuple(bytes(decode[c] for c in part) for part in ln.split(" ")) for ln in lines]
        return cls(vocab, merges, special_tokens, **options)

    def _bpe(self, token, steps=None):
        """Merge the bytes of one pre-token, lowest-rank pair first. Records each stage in `steps` if given."""
        word = [bytes([b]) for b in token]
        while True:
            if steps is not None:
                steps.append(word)
            pair = min(itertools.pairwise(word), key=lambda p: self.ranks.get(p, float("inf")), default=None)
            if pair not in self.ranks:
                return word
            word = _merge(word, *pair)

    def _encode_pretoken(self, s):
        ids = self.cache.get(s)
        if ids is None:
            ids = self.cache[s] = [self.ids[t] for t in self._bpe(s.encode("utf-8"))]
        return ids

    def encode(self, text):
        if self.nfc:
            text = unicodedata.normalize("NFC", text)
        ids = []
        pieces = self.splitter.split(text) if self.splitter else [text]
        for k, piece in enumerate(pieces):
            if k % 2:  # odd slots are the special tokens the splitter captured
                ids.append(self.ids[piece.encode("utf-8")])
            else:
                for m in self.pattern.finditer(piece):
                    ids.extend(self._encode_pretoken(m.group()))
        return ids

    def encode_iterable(self, iterable):
        """Encode lazily, one string (e.g. a file line) at a time, in constant memory.

        Each string's last pre-token waits for the next string: more text can
        change how it splits (a trailing "  \\n" becomes "  " + "\\n" if a word follows).
        """
        carry = ""
        for text in iterable:
            text = carry + text
            tail = self.splitter.split(text)[-1] if self.splitter else text
            starts = [m.start() for m in self.pattern.finditer(tail)]
            cut = len(text) - len(tail) + starts[-1] if starts else len(text)
            yield from self.encode(text[:cut])
            carry = text[cut:]
        yield from self.encode(carry)

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def trace(self, text):
        """Every merge step for every pre-token, for showing BPE at work."""
        out = []
        for m in self.pattern.finditer(text):
            steps = []
            self._bpe(m.group().encode("utf-8"), steps)
            out.append({"pretoken": m.group(),
                        "steps": [[t.decode("utf-8", errors="replace") for t in w] for w in steps]})
        return out


def bytes_to_unicode():
    """GPT-2's map from each byte to a printable character, so vocab files are plain text."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    chars, n = {}, 0
    for b in range(256):
        if b in printable:
            chars[b] = chr(b)
        else:
            chars[b] = chr(256 + n)
            n += 1
    return chars


def save(vocab, merges, vocab_path, merges_path):
    """Write vocab.json and merges.txt in the GPT-2 format that `Tokenizer.from_files` reads."""
    enc = bytes_to_unicode()

    def spell(token):
        return "".join(enc[b] for b in token)

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({spell(t): i for i, t in vocab.items()}, f, ensure_ascii=False)
    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        f.writelines(f"{spell(a)} {spell(b)}\n" for a, b in merges)
