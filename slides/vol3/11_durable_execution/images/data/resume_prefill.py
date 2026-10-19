"""Lecture 11: what does the next model call cost after a resume?

A recovered trajectory's next call carries its whole reconstructed context.
This script measures that call on a local model (Ollama) three ways:
  warm     the same server already holds the prefix (the previous turn's
           request was identical up to the new suffix);
  cold     the model was unloaded (a different server, or a restarted one):
           the full context is prefilled again;
Each case is run REPS times. The context is a synthetic 15-turn repair
transcript (tool calls and test-log lines), about 10,000 tokens.

Usage: python resume_prefill.py OUT.json
"""

import json
import subprocess
import sys
import time
import urllib.request

MODEL = "qwen2.5:7b"
NUM_CTX = 16384
REPS = 3


def transcript(turns=15):
    lines = ["System: you are a repair agent for a connection-pool race in pool.py."]
    for t in range(1, turns + 1):
        lines.append(f"Turn {t}. Assistant: run_tests(ws_r{t % 20 + 1})")
        lines.append(f"Tool result (turn {t}):")
        for k in range(20):
            lines.append(f"  tests/test_pool.py::test_acquire_release_{t}_{k} PASSED "
                         f"[{(k + 1) * 5}%] elapsed={0.01 * (k + 1):.2f}s worker=gw{k % 4}")
    return "\n".join(lines)


def generate(prompt):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "raw": True,
                       "options": {"temperature": 0, "num_predict": 1,
                                   "num_ctx": NUM_CTX}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    out["wall_s"] = time.time() - t
    return {k: out.get(k) for k in ("prompt_eval_count", "prompt_eval_duration",
                                    "load_duration", "total_duration", "wall_s")}


def unload():
    subprocess.run(["ollama", "stop", MODEL], capture_output=True)
    time.sleep(2)


def main(out):
    base = transcript()
    res = {"cold": [], "warm": []}
    for r in range(REPS):
        unload()
        cold = generate(base + f"\nTurn 31 (resume {r}). Assistant:")
        res["cold"].append(cold)
        warm = generate(base + f"\nTurn 31 (resume {r}). Assistant: run_tests(ws_r18)\n"
                               "Tool result: in doubt.")
        res["warm"].append(warm)
        print(r, cold, warm)
    hw = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                        capture_output=True, text=True).stdout.strip()
    data = {"model": f"{MODEL} (Ollama, Q4_K_M)", "machine": hw,
            "measured": time.strftime("%Y-%m-%d"), "num_ctx": NUM_CTX, **res}
    with open(out, "w") as f:
        json.dump(data, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
