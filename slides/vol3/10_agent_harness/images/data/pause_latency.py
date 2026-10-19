#!/usr/bin/env python3
"""Measure how long an operator pause waits under two harness policies.

A local model server (Ollama, qwen2.5:7b, 4-bit) streams a reply with an
output allowance K (num_predict). An operator pause arrives 0.1 s after the
call is sent. Two policies:

  end-check: the harness checks pending_control only when the reply ends,
             so the pause takes effect when the stream finishes;
  cancel:    the harness closes the stream when the pause arrives
             (streaming cancellation); the pause takes effect when the
             connection is closed, and we record how many tokens arrived.

Writes pause_latency.json (one row per run). Run: python3 pause_latency.py
Measured on the author's laptop; the slide names the hardware.
"""

import json
import platform
import time
import urllib.request
from pathlib import Path

URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
PROMPT = ("Write the integers from 1 to 3000 in order, separated by commas. "
          "Do not stop early and do not add any other text.")
PAUSE_AT = 0.1
ALLOWANCES = [256, 1024, 2048, 4096]
REPS = 3


def stream(k, cancel_at=None):
    body = json.dumps({"model": MODEL, "prompt": PROMPT, "stream": True,
                       "options": {"num_predict": k, "temperature": 0}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    n = 0
    done = None
    resp = urllib.request.urlopen(req, timeout=600)
    try:
        for line in resp:
            now = time.perf_counter() - t0
            msg = json.loads(line)
            if msg.get("response"):
                n += 1
                if first is None:
                    first = now
            if msg.get("done"):
                done = msg
                break
            if cancel_at is not None and now >= cancel_at:
                break
    finally:
        resp.close()
    t_end = time.perf_counter() - t0
    out = {"k": k, "tokens_received": n, "t_first": first, "t_end": t_end}
    if done:
        out["prompt_eval_count"] = done.get("prompt_eval_count")
        out["eval_count"] = done.get("eval_count")
        out["eval_duration_s"] = done.get("eval_duration", 0) / 1e9
    return out


def main():
    stream(16)  # warm the model into memory
    rows = []
    idle = [stream(16)["t_first"] for _ in range(3)]
    for k in ALLOWANCES:
        for r in range(REPS):
            a = stream(k)
            a.update(policy="end-check", rep=r, pause_at=PAUSE_AT,
                     pause_effect_s=max(a["t_end"] - PAUSE_AT, 0.0))
            rows.append(a)
            b = stream(k, cancel_at=PAUSE_AT)
            b.update(policy="cancel", rep=r, pause_at=PAUSE_AT,
                     pause_effect_s=max(b["t_end"] - PAUSE_AT, 0.0))
            # Did the server stop generating? A short call sent right after
            # the cancel should start as fast as one sent to an idle server.
            c = stream(16)
            b["next_call_first_token_s"] = c["t_first"]
            rows.append(b)
            print(k, r, round(a["pause_effect_s"], 2), round(b["pause_effect_s"], 4),
                  a.get("eval_count"), b["tokens_received"], flush=True)
    meta = {"model": MODEL, "server": "Ollama (localhost)", "machine": platform.platform(),
            "processor": platform.processor(), "date": time.strftime("%Y-%m-%d"),
            "pause_at_s": PAUSE_AT, "idle_first_token_s": idle}
    Path(__file__).with_name("pause_latency.json").write_text(
        json.dumps({"meta": meta, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
