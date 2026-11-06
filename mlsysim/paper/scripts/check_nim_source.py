#!/usr/bin/env python3
"""Check paper/data/nim-llama-70b-concurrency1.csv against NVIDIA's NIM benchmarking page.

Standard library only; does not import mlsysim. Reads the page from a saved copy
or downloads the Markdown rendering named in the provenance file, then checks:

* the page hash against the recorded ``source_sha256`` (a mismatch means the page
  has changed since retrieval; the row check still runs against the new text);
* that every concurrency-1 row under a 70B Llama heading appears in the CSV with
  identical cells and headings, and that the CSV holds no other rows.

    python paper/scripts/check_nim_source.py [saved-performance.md]
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
CSV_PATH = DATA / "nim-llama-70b-concurrency1.csv"
PROVENANCE = json.loads((DATA / "nim-llama-70b-concurrency1.provenance.json").read_text())


def page_rows(text: str) -> list[list[str]]:
    """Concurrency-1 rows of every 70B table, with the headings that describe them."""
    model = version = gpus = precision = lengths = None
    rows = []
    for line in text.splitlines():
        if line.startswith("## "):
            model = line[3:].replace(" Results", "").strip()
        elif line.startswith("### Version:"):
            version = line.split(":", 1)[1].strip()
        elif line.startswith("#### "):
            heading = line[5:].strip()
            if re.fullmatch(r"\d+ x .+", heading):
                gpus = heading
            elif re.fullmatch(r"(fp8|fp16|bf16) TP\d+", heading):
                precision = heading
            elif "Input Tokens" in heading:
                lengths = re.fullmatch(r"(\d+) Input Tokens / (\d+) Output Tokens", heading)
        elif line.startswith("| 1 ") and model and "70b" in model.lower():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            count, _, gpu = gpus.partition(" x ")
            dtype, tp = precision.split(" TP")
            rows.append([model, version, count, gpu, dtype, tp, lengths[1], lengths[2], *cells])
    return rows


def main() -> int:
    if len(sys.argv) > 1:
        raw = Path(sys.argv[1]).read_bytes()
    else:
        # The docs host refuses urllib's default User-Agent with HTTP 403.
        request = urllib.request.Request(PROVENANCE["url"], headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
    digest = hashlib.sha256(raw).hexdigest()
    same_page = digest == PROVENANCE["source_sha256"]
    print(f"page sha256 {digest} ({'matches' if same_page else 'differs from'} the recorded retrieval)")

    published = page_rows(raw.decode("utf-8"))
    with CSV_PATH.open() as stream:
        reader = csv.reader(stream)
        next(reader)
        recorded = [row for row in reader]
    missing = [r for r in published if r not in recorded]
    extra = [r for r in recorded if r not in published]
    print(f"{len(published)} concurrency-1 rows on the page, {len(recorded)} in the CSV")
    for r in missing:
        print("  on the page, not in the CSV:", ", ".join(r))
    for r in extra:
        print("  in the CSV, not on the page:", ", ".join(r))
    if missing or extra or len(recorded) != PROVENANCE["rows"]:
        return 1
    print("every CSV row matches the page")
    return 0


if __name__ == "__main__":
    sys.exit(main())
