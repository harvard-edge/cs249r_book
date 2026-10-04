#!/usr/bin/env python3
"""Re-pin Volume I's printed notation text after an approved erratum.

Usage: python3 binder/tools/audit/repin_vol1_notation.py "<reason, e.g. erratum E4: ...>"
The normalization must match _run_notation_vol1_pin in binder/cli/commands/validate.py.
"""
import datetime, hashlib, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WRAPPER = ROOT / "books" / "vol1" / "frontmatter" / "notation.qmd"
PIN = ROOT / "binder" / "tools" / "audit" / "data" / "vol1_notation_pin.json"


def digest() -> str:
    text = WRAPPER.read_text(encoding="utf-8")
    body = re.sub(r"^---.*?---\s*", "", text, flags=re.S)
    body = re.sub(r"^#\s+Notation[^\n]*\n", "", body)
    assembled = re.sub(r"```\{=latex\}\s*\\part\{key:mainmatter\}\s*```", "", body, flags=re.S)
    assembled = re.sub(r"<!--.*?-->", "", assembled, flags=re.S)
    assembled = re.sub(r"-{3,}", "---", assembled)
    return hashlib.sha256(re.sub(r"\s+", " ", assembled).strip().encode("utf-8")).hexdigest()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: repin_vol1_notation.py \"<reason>\"")
    PIN.write_text(json.dumps({"sha256": digest(), "pinned": datetime.date.today().isoformat(),
                               "reason": sys.argv[1]}, indent=2) + "\n", encoding="utf-8")
    print(f"pinned {PIN.relative_to(ROOT)}")
