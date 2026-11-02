#!/usr/bin/env python3
"""Refresh and export the shared site data without rebuilding any pages.

The scheduled workflow calls this entry point. Metrics live in stats.json;
course and institution data live in adopters.json. Pages read these payloads
on load, with embedded build-time data as their offline fallback.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

SITE = Path(__file__).resolve().parents[1]


def refresh(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(SITE / "scripts/build_stats.py")], check=True)
    subprocess.run([sys.executable, str(SITE / "scripts/build_adopters.py"), "gather"], check=True)
    subprocess.run([sys.executable, str(SITE / "scripts/build_adopters.py"), "icons"], check=True)
    subprocess.run([sys.executable, str(SITE / "scripts/build_adopters.py"),
                    "json", str(output / "adopters.json")], check=True)
    cache = json.loads((SITE / "config/stats-cache.json").read_text(encoding="utf-8"))
    (output / "stats.json").write_text(
        json.dumps({"generated": cache.get("generated", ""),
                    "display": cache.get("display", {})}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"Shared site data refreshed in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    refresh(parser.parse_args().output_dir.resolve())
