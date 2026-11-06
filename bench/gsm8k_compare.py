"""Two GSM8K evaluations on the same problems, compared problem by problem.

    uv run python bench/gsm8k_compare.py gsm8k_base gsm8k_grpo_gsm8k

Counts the problems the second model fixed (the first got wrong) and broke (the first got right).
Only those disagreements say which model is better. If the two were equally good, each disagreement
would go either way with probability 1/2, so the exact two-sided sign test gives the chance of a
split at least this uneven. Writes bench/results/<first>_vs_<second>.json.
"""
import json
import math
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def sign_test(fixed, broke):
    """Two-sided exact p-value: the chance a fair coin splits fixed + broke flips at least this unevenly."""
    n, k = fixed + broke, min(fixed, broke)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    first, second = sys.argv[1:3]
    a = [json.loads(line) for line in (RESULTS / f"{first}_replies.jsonl").read_text().splitlines()]
    b = [json.loads(line) for line in (RESULTS / f"{second}_replies.jsonl").read_text().splitlines()]
    assert [x["i"] for x in a] == [x["i"] for x in b], "the two runs must cover the same problems"
    fixed = sum(y["correct"] and not x["correct"] for x, y in zip(a, b))
    broke = sum(x["correct"] and not y["correct"] for x, y in zip(a, b))
    result = {"first": first, "second": second, "n": len(a), "fixed": fixed, "broke": broke, "p_value": round(sign_test(fixed, broke), 4)}
    assert sign_test(5, 5) == 1.0 and abs(sign_test(0, 10) - 2 / 1024) < 1e-12  # a fair split and the extreme one
    (RESULTS / f"{first}_vs_{second}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
