#!/usr/bin/env python3
"""Line counter for the minitorch package, in the spirit of tinygrad's sz.py.

Prints total lines and code lines (blanks and comments stripped) per file, so
you can watch the whole thing stay small.

    uv run python sz.py
"""
import os


def count(path):
    with open(path) as f:
        lines = f.readlines()
    code = sum(1 for ln in lines if ln.strip() and not ln.strip().startswith("#"))
    return len(lines), code


def main():
    root = os.path.join(os.path.dirname(__file__), "minitorch")
    rows = []
    for name in os.listdir(root):
        if name.endswith(".py"):
            total, code = count(os.path.join(root, name))
            rows.append((name, total, code))
    rows.sort(key=lambda r: -r[2])

    width = max(len(r[0]) for r in rows)
    line = "-" * (width + 18)
    print(f"{'file':<{width}}  {'lines':>7}  {'code':>7}")
    print(line)
    for name, total, code in rows:
        print(f"{name:<{width}}  {total:>7}  {code:>7}")
    print(line)
    print(f"{'total':<{width}}  {sum(r[1] for r in rows):>7}  {sum(r[2] for r in rows):>7}")


if __name__ == "__main__":
    main()
