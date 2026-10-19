"""Lecture 17: what an inference server records for one sampled invocation.

Asks a local Ollama server (gemma3:1b) to continue a prompt at three sampling
temperatures with the same seed, recording each sampled token and the log
probability the server reports, plus its top-3 alternatives.
Writes logprob_run.json beside this file. Run: python3 logprob_run.py
(needs `ollama serve` and `ollama pull gemma3:1b`).
"""

import json
import subprocess
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL = "http://localhost:11434/api/generate"


def call(temperature):
    body = {"model": "gemma3:1b", "prompt": "def add(a, b):", "raw": True, "stream": False,
            "logprobs": True, "top_logprobs": 3,
            "options": {"num_predict": 6, "temperature": temperature, "seed": 7}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    return {"temperature": temperature, "response": d["response"],
            "tokens": [{"token": t["token"], "logprob": t["logprob"],
                        "top": [[a["token"], a["logprob"]] for a in t["top_logprobs"]]} for t in d["logprobs"]]}


if __name__ == "__main__":
    version = subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip()
    runs = [call(t) for t in (0.2, 0.8, 1.5)]
    (HERE / "logprob_run.json").write_text(json.dumps({"server": version, "model": "gemma3:1b",
                                                       "runs": runs}, indent=1))
    for r in runs:
        print(r["temperature"], [(t["token"], round(t["logprob"], 3)) for t in r["tokens"][:4]])
