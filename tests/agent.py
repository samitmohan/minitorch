"""The agent's calculator (apps/agent.py): arithmetic works, and model-written text never runs as code.

    uv run python tests/agent.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps"))
from agent import calculate


def main():
    cases = {"17 * 23 + 9": "400", "$1,250 * 20%": "250", "7 // 2": "3", "10 / 4": "2.5", "-(2 - 5) ** 2": "-9"}
    for expression, want in cases.items():
        assert calculate(expression) == want, (expression, calculate(expression))
    print(f"[ok]   {len(cases)} expressions, GSM8K's $ , and % included")
    for hostile in ['__import__("os").system("ls")', "open('x')", "(1).__class__", "[1] * 10", "2 ** 10000", "9 ** 9 ** 9"]:
        assert calculate(hostile).startswith("error"), hostile
    print("[ok]   names, calls, attributes, lists, and huge powers come back as errors, never run")
    for hostile in ["((((9**64)**64)**64)**64)", "-" * 1000 + "1", "(-8) ** 0.5"]:  # minutes of bignum, deep recursion, a complex root
        start = time.perf_counter()
        result = calculate(hostile)
        assert result.startswith("error") and time.perf_counter() - start < 1, (hostile[:30], result)
    print("[ok]   towers of powers, 1,000 nested minus signs, and a complex root come back as errors in under a second")


if __name__ == "__main__":
    main()
