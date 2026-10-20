"""Lecture 13: a real, small agent evaluation, run for this lecture.

Twelve repair tasks. Each gives a buggy Python function, a one-line
description, and one visible test. A sealed grader holds the hidden tests:
it writes the candidate to a fresh temporary directory, runs the hidden
tests in a separate interpreter (python -I) with a 10 s cutoff, and records
one verdict per trial: pass, fail, or unknown (no result at the cutoff).

Two configurations of the same local model are compared on the same tasks,
run by run (paired):
  baseline_v1   one model call; the reply is the submission.
  candidate_v2  the same call, then the harness runs the visible test; if it
                fails, the error is sent back once and the second reply is
                the submission.
Every trial records tokens and seconds, so cost per accepted task is real.

Usage: python eval_run.py OUT.json [RUNS]
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.request

MODEL = "qwen2.5:7b"
TEMP = 0.8
CUTOFF_S = 10

TASKS = [
    dict(id="median", desc="Return the median of a non-empty list of numbers.",
         code="def median(xs):\n    return xs[len(xs) // 2]\n",
         visible="assert median([3, 1, 2]) == 2",
         hidden=["assert median([1, 2, 3, 4]) == 2.5", "assert median([5, 1, 4, 2, 3]) == 3",
                 "assert median([-1.5, 2.5]) == 0.5", "assert median([7]) == 7"]),
    dict(id="merge_intervals", desc="Merge overlapping or touching closed intervals; return them sorted.",
         code=("def merge_intervals(iv):\n    out = []\n    for a, b in iv:\n"
               "        if out and a < out[-1][1]:\n            out[-1] = (out[-1][0], max(b, out[-1][1]))\n"
               "        else:\n            out.append((a, b))\n    return out\n"),
         visible="assert merge_intervals([(1, 3), (2, 6)]) == [(1, 6)]",
         hidden=["assert merge_intervals([(1, 2), (2, 3)]) == [(1, 3)]",
                 "assert merge_intervals([(5, 6), (1, 2)]) == [(1, 2), (5, 6)]",
                 "assert merge_intervals([(1, 10), (2, 3)]) == [(1, 10)]",
                 "assert merge_intervals([]) == []"]),
    dict(id="roman_to_int", desc="Convert a valid Roman numeral string to an integer.",
         code=("def roman_to_int(s):\n    v = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n"
               "    return sum(v[c] for c in s)\n"),
         visible="assert roman_to_int('III') == 3",
         hidden=["assert roman_to_int('IV') == 4", "assert roman_to_int('MCMXCIV') == 1994",
                 "assert roman_to_int('XL') == 40", "assert roman_to_int('LVIII') == 58"]),
    dict(id="is_balanced", desc="Return True if every (), [], {} bracket in s is closed in the right order; other characters are ignored.",
         code=("def is_balanced(s):\n    depth = 0\n    for c in s:\n        if c == '(':\n            depth += 1\n"
               "        elif c == ')':\n            depth -= 1\n    return depth == 0\n"),
         visible="assert is_balanced('(())') is True",
         hidden=["assert is_balanced('([)]') is False", "assert is_balanced('{[]}') is True",
                 "assert is_balanced(')(') is False", "assert is_balanced('a(b)c') is True",
                 "assert is_balanced('') is True"]),
    dict(id="rle_decode", desc="Decode run-length text such as '3a2b' into 'aaabb'; counts may have several digits.",
         code=("def rle_decode(s):\n    out = ''\n    for i in range(0, len(s), 2):\n"
               "        out += s[i + 1] * int(s[i])\n    return out\n"),
         visible="assert rle_decode('3a2b') == 'aaabb'",
         hidden=["assert rle_decode('12a') == 'a' * 12", "assert rle_decode('1a1b1c') == 'abc'",
                 "assert rle_decode('') == ''", "assert rle_decode('2x10y') == 'xx' + 'y' * 10"]),
    dict(id="compare_versions", desc="Compare dotted version strings numerically; return -1, 0, or 1. Missing parts count as 0.",
         code="def compare_versions(a, b):\n    return (a > b) - (a < b)\n",
         visible="assert compare_versions('1.0', '1.0') == 0",
         hidden=["assert compare_versions('1.10', '1.2') == 1", "assert compare_versions('1.0', '1') == 0",
                 "assert compare_versions('2.0.1', '2.0') == 1", "assert compare_versions('0.9', '1.0') == -1"]),
    dict(id="chunk", desc="Split a list into consecutive chunks of size n (n >= 1); the last chunk may be shorter.",
         code="def chunk(xs, n):\n    return [xs[i:i + n] for i in range(0, len(xs) - n + 1, n)]\n",
         visible="assert chunk([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]",
         hidden=["assert chunk([1, 2, 3], 2) == [[1, 2], [3]]", "assert chunk([], 3) == []",
                 "assert chunk([1, 2], 5) == [[1, 2]]", "assert chunk([1, 2, 3], 1) == [[1], [2], [3]]"]),
    dict(id="word_count", desc="Count words case-insensitively; words are runs of letters, digits, or apostrophes. Return a dict.",
         code="def word_count(text):\n    d = {}\n    for w in text.split(' '):\n        d[w] = d.get(w, 0) + 1\n    return d\n",
         visible="assert word_count('a b a') == {'a': 2, 'b': 1}",
         hidden=["assert word_count('Hi, hi!') == {'hi': 2}", "assert word_count('one  two\\nthree') == {'one': 1, 'two': 1, 'three': 1}",
                 "assert word_count('') == {}", "assert word_count(\"don't stop\") == {\"don't\": 1, 'stop': 1}"]),
    dict(id="bounded_pool", desc=("A connection pool of fixed size. acquire() returns a free connection id or None if none is free; "
                                  "release(c) returns it, and raises ValueError if c is not currently held."),
         code=("class BoundedPool:\n    def __init__(self, size):\n        self.free = list(range(size))\n\n"
               "    def acquire(self):\n        return self.free.pop() if self.free else None\n\n"
               "    def release(self, c):\n        self.free.append(c)\n"),
         visible="p = BoundedPool(1); c = p.acquire(); assert p.acquire() is None; p.release(c); assert p.acquire() == c",
         hidden=["p = BoundedPool(1); c = p.acquire(); p.release(c)\ntry:\n    p.release(c); raise SystemExit(1)\nexcept ValueError:\n    pass",
                 "p = BoundedPool(2); a = p.acquire(); b = p.acquire(); assert {a, b} == {0, 1}; assert p.acquire() is None",
                 "p = BoundedPool(1)\ntry:\n    p.release(5); raise SystemExit(1)\nexcept ValueError:\n    pass",
                 "p = BoundedPool(2); a = p.acquire(); p.release(a); b = p.acquire(); c = p.acquire(); assert None not in (b, c) and b != c"]),
    dict(id="parse_duration", desc="Parse a duration such as '1h30m' or '45s' into seconds; units h, m, s may appear in that order, each optional.",
         code="def parse_duration(s):\n    n, unit = int(s[:-1]), s[-1]\n    return n * {'h': 3600, 'm': 60, 's': 1}[unit]\n",
         visible="assert parse_duration('45s') == 45",
         hidden=["assert parse_duration('1h30m') == 5400", "assert parse_duration('2h') == 7200",
                 "assert parse_duration('1h0m5s') == 3605", "assert parse_duration('90m') == 5400"]),
    dict(id="binary_search", desc="Return the index of target in the sorted list xs, or -1 if it is absent.",
         code=("def binary_search(xs, target):\n    lo, hi = 0, len(xs) - 1\n    while lo < hi:\n"
               "        mid = (lo + hi) // 2\n        if xs[mid] == target:\n            return mid\n"
               "        if xs[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid\n    return -1\n"),
         visible="assert binary_search([1, 3, 5, 7], 3) == 1",
         hidden=["assert binary_search([1, 3, 5, 7], 7) == 3", "assert binary_search([], 1) == -1",
                 "assert binary_search([2], 2) == 0", "assert binary_search([1, 3, 5], 4) == -1"]),
    dict(id="flatten", desc="Flatten arbitrarily nested lists into one flat list, keeping order.",
         code=("def flatten(x):\n    out = []\n    for e in x:\n        if isinstance(e, list):\n"
               "            out.extend(e)\n        else:\n            out.append(e)\n    return out\n"),
         visible="assert flatten([1, [2, 3]]) == [1, 2, 3]",
         hidden=["assert flatten([1, [2, [3, [4]]]]) == [1, 2, 3, 4]", "assert flatten([]) == []",
                 "assert flatten([[], [[]], 5]) == [5]", "assert flatten([[1, 2], 3, [[4], 5]]) == [1, 2, 3, 4, 5]"]),
]

SYSTEM = ("You repair Python code. Reply with the corrected code only, in one ```python block, "
          "with no explanation.")


def ask(messages):
    body = json.dumps({"model": MODEL, "messages": messages, "stream": False,
                       "options": {"temperature": TEMP, "num_predict": 400}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    return (d["message"]["content"], d.get("prompt_eval_count", 0),
            d.get("eval_count", 0), time.time() - t0)


def extract(reply):
    m = re.findall(r"```(?:python)?\n(.*?)```", reply, re.S)
    return m[0] if m else None


def run_checks(code, checks):
    """Run each check in a fresh interpreter in a fresh directory.
    Returns 'pass', 'fail', or 'unknown' (cutoff reached), plus detail."""
    if code is None:
        return "fail", "no code block"
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "candidate.py"), "w") as f:
            f.write(code)
        for i, chk in enumerate(checks):
            src = "import sys\nsys.path.insert(0, '.')\nfrom candidate import *\n" + chk + "\nprint('CHECK_OK')\n"
            with open(os.path.join(d, f"check_{i}.py"), "w") as f:
                f.write(src)
            try:
                p = subprocess.run([sys.executable, "-I", f"check_{i}.py"], cwd=d,
                                   capture_output=True, text=True, timeout=CUTOFF_S)
            except subprocess.TimeoutExpired:
                return "unknown", f"check {i}: no result at {CUTOFF_S} s cutoff"
            if p.returncode != 0 or "CHECK_OK" not in p.stdout:
                tail = (p.stderr.strip().splitlines() or ["exit " + str(p.returncode)])[-1]
                return "fail", f"check {i}: {tail[:160]}"
    return "pass", ""


def prompt_for(t):
    return (f"Task: {t['desc']}\nThis code is wrong:\n```python\n{t['code']}```\n"
            f"A visible test:\n```python\n{t['visible']}\n```")


def trial(t, config):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt_for(t)}]
    reply, pin, pout, secs = ask(msgs)
    calls, tin, tout, wall = 1, pin, pout, secs
    code = extract(reply)
    vis, vdetail = run_checks(code, [t["visible"]])
    if config == "candidate_v2" and vis != "pass":
        msgs += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": f"The visible test failed: {vdetail}. Fix it."}]
        reply, pin, pout, secs = ask(msgs)
        calls, tin, tout, wall = calls + 1, tin + pin, tout + pout, wall + secs
        code = extract(reply)
        vis, vdetail = run_checks(code, [t["visible"]])
    verdict, detail = run_checks(code, t["hidden"])
    return dict(task=t["id"], config=config, visible=vis, verdict=verdict, detail=detail,
                calls=calls, input_tokens=tin, output_tokens=tout, seconds=round(wall, 2),
                code=code)


def main(out, runs):
    rows = []
    start = time.time()
    for r in range(runs):
        for t in TASKS:
            for config in ("baseline_v1", "candidate_v2"):
                row = trial(t, config)
                row["run"] = r
                rows.append(row)
                print(r, t["id"], config, row["visible"], row["verdict"], row["calls"],
                      row["output_tokens"], row["seconds"], flush=True)
        with open(out, "w") as f:
            json.dump({"model": f"{MODEL} (Ollama, Q4_K_M)", "temperature": TEMP,
                       "cutoff_s": CUTOFF_S, "measured": time.strftime("%Y-%m-%d"),
                       "runs_done": r + 1, "elapsed_s": round(time.time() - start),
                       "tasks": [dict(id=t["id"], desc=t["desc"], visible=t["visible"],
                                      hidden=t["hidden"], code=t["code"]) for t in TASKS],
                       "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
