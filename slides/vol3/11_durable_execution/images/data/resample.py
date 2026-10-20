"""Lecture 11: does restarting from the prompt reproduce the next step?

Sends the repair agent's original task prompt to a local model N times
at temperature 0.8 (a sampled agent) and N times at temperature 0, and
records each reply's first line. Run on one machine with one request at a
time, so the temperature-0 case shows the best case for reproduction.

Usage: python resample.py OUT.json
"""

import collections
import json
import sys
import time
import urllib.request

MODEL = "qwen2.5:7b"
PROMPT = ("You are a repair agent. Task: fix the intermittent failure in "
          "tests/test_pool.py::test_concurrent_acquire, a race in pool.py. "
          "Reply with exactly one line: your first tool call, as "
          "tool_name(arguments). Tools: run_tests(path), read_file(path), "
          "grep(pattern), edit_file(path, change), deploy(snapshot).")

N = 30


def ask(temp, seed=None):
    opts = {"temperature": temp, "num_predict": 40}
    if seed is not None:
        opts["seed"] = seed
    body = json.dumps({"model": MODEL, "prompt": PROMPT, "stream": False,
                       "options": opts}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["response"].strip().splitlines()[0]


def main(out):
    res = {}
    for temp in (0.8, 0.0):
        replies = [ask(temp) for _ in range(N)]
        res[str(temp)] = {"n": N, "distinct": len(set(replies)),
                          "counts": collections.Counter(replies).most_common()}
        print(temp, res[str(temp)]["distinct"], res[str(temp)]["counts"][:6])
    data = {"model": f"{MODEL} (Ollama, Q4_K_M)", "measured": time.strftime("%Y-%m-%d"),
            "prompt": PROMPT, **res}
    with open(out, "w") as f:
        json.dump(data, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
