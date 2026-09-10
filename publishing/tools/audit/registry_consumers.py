#!/usr/bin/env python3
"""Map MLSysIM registry symbols to the chapters that consume them.

The registry is the single source of truth for every physical number in the
book, which means editing an existing entry is a cross-volume act: Volume I is
in print, so a value it consumes must not move without an explicit decision.
This tool answers the question that gates any registry edit: *who reads this
today, and in which volume?*

Typical use before touching an existing entry::

    python3 publishing/tools/audit/registry_consumers.py --symbol Hardware.Cloud.H100.memory.bandwidth

It prints the consuming volumes first, so a symbol that Volume I reads is
visible before the edit rather than after the reprint.

Added 2026-09 while wiring Volume III to the registry, after the vol3 decode
economics table was found to carry three mutually inconsistent hand-typed
configurations. Growing the registry is the sanctioned fix; silently retuning a
shared entry is not.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOTS = ("Hardware", "Models", "Infrastructure", "Systems", "Ops",
         "Scenarios", "Literature", "Platforms", "Datasets")

SYMBOL_RE = re.compile(
    r"\b(?:%s)\.[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*" % "|".join(ROOTS)
)
CELL_RE = re.compile(r"^```\{python\}\n(.*?)^```$", re.S | re.M)


def contents_dir(start: Path) -> Path:
    for base in (start, *start.parents):
        candidate = base / "publishing"  / "books"
        if candidate.is_dir():
            return candidate
    sys.exit("could not locate books from %s" % start)


def volume_of(path: Path, contents: Path) -> str:
    try:
        rel = path.relative_to(contents)
    except ValueError:
        return "?"
    return rel.parts[0] if rel.parts else "?"


def scan(contents: Path):
    """Return {symbol: {volume: [(path, line)]}} for every registry read."""
    hits = defaultdict(lambda: defaultdict(list))
    for qmd in sorted(contents.rglob("*.qmd")):
        text = qmd.read_text(encoding="utf-8", errors="replace")
        vol = volume_of(qmd, contents)
        for cell in CELL_RE.finditer(text):
            offset = text.count("\n", 0, cell.start(1)) + 1
            for i, line in enumerate(cell.group(1).splitlines()):
                if line.lstrip().startswith("#"):
                    continue
                for symbol in SYMBOL_RE.findall(line):
                    hits[symbol][vol].append((qmd, offset + i))
    return hits


def prefixes(symbol: str):
    parts = symbol.split(".")
    return {".".join(parts[: i + 1]) for i in range(len(parts))}


def report_symbol(hits, symbol: str) -> int:
    """Report every consumer of `symbol` or of anything beneath it."""
    matched = {s: v for s, v in hits.items() if symbol in prefixes(s) or s.startswith(symbol + ".")}
    if not matched:
        print("No chapter reads %s today." % symbol)
        print("Safe to add or adjust: nothing in the book depends on it yet.")
        return 0

    volumes = sorted({v for per_vol in matched.values() for v in per_vol})
    print("Consumers of %s" % symbol)
    print("  volumes: %s" % ", ".join(volumes))
    if "vol1" in volumes:
        print("  WARNING: Volume I reads this symbol and Volume I is in print.")
        print("           Surface any change to the author before editing the entry.")
    print()
    for sym in sorted(matched):
        print("  %s" % sym)
        for vol in sorted(matched[sym]):
            for path, line in matched[sym][vol]:
                print("      %-5s %s:%d" % (vol, path, line))
    return 1 if "vol1" in volumes else 0


def report_all(hits, only_volume: str | None) -> None:
    rows = []
    for symbol, per_vol in hits.items():
        vols = sorted(per_vol)
        if only_volume and only_volume not in vols:
            continue
        rows.append((symbol, vols, sum(len(v) for v in per_vol.values())))
    rows.sort(key=lambda r: (-r[2], r[0]))
    print("%-58s %-22s %s" % ("symbol", "volumes", "reads"))
    for symbol, vols, count in rows:
        print("%-58s %-22s %d" % (symbol[:58], ",".join(vols), count))
    print("\n%d distinct symbols" % len(rows))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbol", help="registry path to check before editing it")
    ap.add_argument("--volume", help="restrict the full listing to one volume")
    ap.add_argument("--json", action="store_true", help="emit the full map as JSON")
    args = ap.parse_args()

    contents = contents_dir(Path.cwd().resolve())
    hits = scan(contents)

    if args.json:
        out = {
            symbol: {vol: ["%s:%d" % (p, ln) for p, ln in refs] for vol, refs in per_vol.items()}
            for symbol, per_vol in hits.items()
        }
        json.dump(out, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0

    if args.symbol:
        return report_symbol(hits, args.symbol)

    report_all(hits, args.volume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
