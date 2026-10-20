#!/usr/bin/env python3
"""Real model-call artifacts for Lecture 2 (The Foundation Model).

Runs one small open model (Qwen2.5-Coder-0.5B-Instruct, Apache 2.0) on the
chapter's repair-agent inspection request (Listing 1 of the chapter) and
records what the runtime actually sees: the serialized wire text, token
counts, a completed tool call, a truncated one, next-token probabilities at
the path argument, batch-dependent logits, and prefill and decode timings.

This script is NOT run by the slide build. It needs torch, transformers, and
tiktoken, and the model in the local Hugging Face cache. It writes
data/model_runs.json, which calcs.py and make_figs.py read.

    python3 run_model.py          (from this folder, in an environment with torch)
"""

import json
import os
import platform
import statistics
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import tiktoken  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "data" / "model_runs.json"
MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
REVISION = "ea3f2471cf1b1f0db85067f1ef93848e38e88c25"   # snapshot used for the deck
SNAP = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots" / REVISION
torch.manual_seed(0)
torch.set_num_threads(8)

res = {"model": MODEL, "revision": REVISION, "torch": torch.__version__,
       "transformers": transformers.__version__, "tiktoken": tiktoken.__version__,
       "machine": platform.platform(), "processor": platform.processor()}

# --------------------------------------------------------------- tiktoken (chapter 2.3.1)
enc = tiktoken.get_encoding("cl100k_base")
for key, s in [("flat", "def get_user_id():"), ("indented", "    def get_user_id():"),
               ("path", "src/pool.py")]:
    ids = enc.encode(s)
    res[f"cl100k_{key}"] = {"text": s, "ids": ids,
                            "pieces": [enc.decode([i]) for i in ids]}

# --------------------------------------------------------------- the request (chapter Listing 1)
tok = AutoTokenizer.from_pretrained(SNAP)
model = AutoModelForCausalLM.from_pretrained(SNAP, dtype=torch.float32)
model.eval()

tools = [{"type": "function", "function": {
    "name": "read_file",
    "description": "Read a file from the repair workspace.",
    "parameters": {"type": "object",
                   "properties": {"path": {"type": "string"}},
                   "required": ["path"], "additionalProperties": False}}}]
messages = [
    {"role": "system", "content": "Propose the next inspection for the assigned repair."},
    {"role": "user", "content": "Inspect the connection-pool locking logic in src/pool.py."},
]
wire = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                               tokenize=False)
ids = tok(wire, return_tensors="pt").input_ids
res["wire_text"] = wire
res["wire_tokens"] = int(ids.shape[1])
res["visible_tokens"] = {
    "system": len(tok(messages[0]["content"]).input_ids),
    "user": len(tok(messages[1]["content"]).input_ids),
}
res["vocab_size"] = int(model.config.vocab_size)
res["max_position_embeddings"] = int(model.config.max_position_embeddings)


def generate(max_new, **kw):
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             return_dict_in_generate=True, **kw)
    new = out.sequences[0, ids.shape[1]:]
    text = tok.decode(new, skip_special_tokens=False)
    ended_eos = bool(len(new) and int(new[-1]) in set(
        [tok.eos_token_id] + list(model.generation_config.eos_token_id
                                  if isinstance(model.generation_config.eos_token_id, list)
                                  else [model.generation_config.eos_token_id])))
    return {"text": text, "new_tokens": int(len(new)), "ended_on_eos": ended_eos,
            "max_new_tokens": max_new,
            "stop": "eos" if ended_eos else ("length" if len(new) >= max_new else "other")}


full = generate(128)
res["completed_call"] = full

# Truncate just inside the path argument: find the token count that ends mid-path.
full_ids = tok(full["text"]).input_ids
cut = None
for k in range(1, len(full_ids)):
    t = tok.decode(full_ids[:k])
    tail = t.split('"path": "')[-1] if '"path": "' in t else ""
    if '"path": "' in t and "/" in tail and len(tail.split("/")) >= 3 and '"' not in tail:
        cut = k
        break
res["truncated_call"] = generate(cut) if cut else None

# --------------------------------------------------------------- next-token probabilities
prefix_text = full["text"].split('"path": "')[0] + '"path": "'
pref_ids = tok(wire + prefix_text, return_tensors="pt").input_ids
with torch.no_grad():
    logits = model(pref_ids).logits[0, -1].double()
top = torch.topk(logits, 8)
res["path_logits"] = {
    "prefix_tail": prefix_text[-40:],
    "tokens": [tok.decode([int(i)]) for i in top.indices],
    "logits": [round(float(v), 4) for v in top.values],
    "p_T1": [float(v) for v in torch.softmax(logits, 0)[top.indices]],
    "p_T05": [float(v) for v in torch.softmax(logits / 0.5, 0)[top.indices]],
}
p_sorted = torch.softmax(logits, 0).sort(descending=True).values
res["path_logits"]["nucleus_090_size"] = int((p_sorted.cumsum(0) < 0.90).sum()) + 1
res["path_logits"]["nucleus_090_mass_of_top8"] = float(p_sorted[:8].sum())

# --------------------------------------------------------------- batch-dependent logits
other = [{"role": "user", "content": "Explain in detail how a thread pool schedules work, "
          "with examples in three languages, and compare the designs carefully. " * 20}]
wire_b = tok.apply_chat_template(other, add_generation_prompt=True, tokenize=False)
tok.padding_side = "left"
batch = tok([wire, wire_b], return_tensors="pt", padding=True)
res["batch_other_tokens"] = int(tok(wire_b, return_tensors="pt").input_ids.shape[1])
diffs = {}
for dtype in (torch.float32, torch.bfloat16):
    m = model.to(dtype)
    with torch.no_grad():
        alone = m(ids).logits[0, -1].float()
        both = m(**batch).logits[0, -1].float()
    d = (alone - both).abs()
    diffs[str(dtype).split(".")[-1]] = {
        "max_abs_diff": float(d.max()),
        "n_logits_changed": int((d > 0).sum()),
        "argmax_alone": tok.decode([int(alone.argmax())]),
        "argmax_batched": tok.decode([int(both.argmax())]),
        "pad_tokens_added": int(batch.input_ids.shape[1] - ids.shape[1]),
    }
model.to(torch.float32)
res["batch_logits"] = diffs

# --------------------------------------------------------------- prefill and decode timing
filler = ("    def acquire(self, timeout=None):\n        with self._lock:\n"
          "            while not self._free:\n                self._cond.wait(timeout)\n"
          "            return self._free.pop()\n")
long_prompt = tok(filler * 60, return_tensors="pt").input_ids[:, :2048]
S = int(long_prompt.shape[1])


def prefill_time():
    t = time.perf_counter()
    with torch.no_grad():
        model(long_prompt, use_cache=True)
    return time.perf_counter() - t


def decode_time(n=64):
    with torch.no_grad():
        out = model(long_prompt, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
        t = time.perf_counter()
        for _ in range(n):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
        return (time.perf_counter() - t) / n


prefill_time()  # warm up
pre = [prefill_time() for _ in range(5)]
dec = [decode_time() for _ in range(3)]
res["timing"] = {"prompt_tokens": S, "prefill_s_runs": pre, "decode_s_per_token_runs": dec,
                 "prefill_s_median": statistics.median(pre),
                 "decode_s_per_token_median": statistics.median(dec),
                 "device": "cpu float32, 8 threads", "batch": 1}

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "wire_text"}, indent=1))
print(wire)
