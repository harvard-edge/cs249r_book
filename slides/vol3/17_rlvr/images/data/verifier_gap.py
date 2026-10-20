"""Lecture 17: run open-r1's code_reward checker on three candidates.

evaluate_code below is copied verbatim from huggingface/open-r1,
src/open_r1/rewards.py, commit 5b6ff22b3fb7 (2026-10-02), the template inside
code_reward(). It runs locally here instead of in an E2B sandbox.
Task: read n, print 1..n one per line. Writes verifier_gap.txt.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def evaluate_code(code, test_cases):
    passed = 0
    total = len(test_cases)
    exec_timeout = 5

    for case in test_cases:
        process = subprocess.run(
            ["python3", "-c", code],
            input=case["input"],
            text=True,
            capture_output=True,
            timeout=exec_timeout
        )

        if process.returncode != 0:  # Error in execution
            continue

        output = process.stdout.strip()

        # TODO: implement a proper validator to compare against ground truth. For now we just check for exact string match on each line of stdout.
        all_correct = True
        for line1, line2 in zip(output.split('\n'), case['output'].split('\n')):
            all_correct = all_correct and line1.strip() == line2.strip()

        if all_correct:
            passed += 1

    success_rate = (passed / total)
    return success_rate


TESTS = [{"input": "3\n", "output": "1\n2\n3"}, {"input": "5\n", "output": "1\n2\n3\n4\n5"}]
CANDIDATES = {
    "correct: print 1..n": "n=int(input())\nfor i in range(1,n+1): print(i)",
    "wrong: print only 1": "input()\nprint(1)",
    "wrong: print 1..100": "input()\nfor i in range(1,101): print(i)",
}

if __name__ == "__main__":
    lines = [f"{name:24s} reward {evaluate_code(code, TESTS):.1f}" for name, code in CANDIDATES.items()]
    lines.append(f"python {sys.version.split()[0]}")
    (HERE / "verifier_gap.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
