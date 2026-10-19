#!/usr/bin/env python3
"""Measure real attention weights from a pronoun in a trained model (Lecture 6b).

A real measurement, not a mock: Hugging Face bert-base-uncased (the published
weights) reads the slide's two sentences, and the script prints the weight
that each head's attention from the pronoun puts on every token. The head
shown on the slide is chosen by one rule, stated here and in the note: the
head, of all 144, that puts the largest weight from "it" on either candidate
noun ("trophy" or "suitcase") in the trophy sentence with "big". Writes data/attention_probe.txt with the
versions and the full rows for that head.

Needs torch and transformers (not in the MLSysBook venv):
  uv venv tvenv && uv pip install --python tvenv/bin/python torch transformers
  tvenv/bin/python attention_probe.py
"""
import platform
from pathlib import Path

import torch
import transformers
from transformers import AutoModel, AutoTokenizer

NAME = "bert-base-uncased"
SENTENCES = [
    ("The student did not finish all the homework because they were tired.", "they"),
    ("The trophy did not fit in the suitcase because it was too big.", "it"),
    ("The trophy did not fit in the suitcase because it was too small.", "it"),
]

tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModel.from_pretrained(NAME, attn_implementation="eager").eval()


def attention_from(sentence, word):
    enc = tok(sentence, return_tensors="pt")
    toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
    with torch.no_grad():
        att = torch.stack(model(**enc, output_attentions=True).attentions)[:, 0]
    i = toks.index(word)
    return toks, att[:, :, i, :]          # layers x heads x keys


# Pick the head: largest weight from "it" on either noun (trophy, big).
toks, a = attention_from(*SENTENCES[1])
nouns = [toks.index("trophy"), toks.index("suitcase")]
best = a[:, :, nouns].amax(dim=-1)
layer, head = divmod(int(best.argmax()), best.shape[1])

lines = [f"model: {NAME}; torch {torch.__version__}; transformers {transformers.__version__}; "
         f"{platform.platform()}",
         f"head chosen: layer {layer + 1}, head {head + 1} (1-indexed); "
         f"max weight from 'it' on a noun = {float(best.max()):.2f}"]
for sentence, word in SENTENCES:
    toks, a = attention_from(sentence, word)
    row = a[layer, head]
    lines.append(f"\n{sentence}  (from '{word}')")
    lines += [f"  {t:10s} {float(v):.2f}" for t, v in zip(toks, row)]
text = "\n".join(lines)
print(text)
out = Path(__file__).resolve().parent / "data" / "attention_probe.txt"
out.parent.mkdir(exist_ok=True)
out.write_text(text + "\n")
