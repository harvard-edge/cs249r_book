#!/usr/bin/env python3
"""Build chapters one-at-a-time in PDF-config order."""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BOOK = REPO  / "books"
BINDER = REPO / "binder" / "binder"
LEDGER = REPO / "binder/tools/audit/artifacts/chapter_build_sequence.json"
sys.path.insert(0, str(REPO / "binder/cli"))
from core.discovery import get_chapters_from_config

def _build(volume, stem, fmt, log_dir):
    """Build one chapter with `binder build` and write its combined output to `<log_dir>/<stem>.log`.

    Returns `(succeeded, elapsed_seconds, tail)`, where `tail` is the last
    12 lines of stdout (or of stderr when stdout is empty).
    """
    log_path = log_dir / f"{stem}.log"
    cmd = [str(BINDER), "build", fmt, f"{volume}/{stem}", f"--{volume}"]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    elapsed = time.monotonic() - t0
    log_path.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
    tail = "\n".join((proc.stdout or proc.stderr or "").splitlines()[-12:])
    return proc.returncode == 0, elapsed, tail

def main():
    """Parse arguments and build each selected volume's chapters in order.

    Builds vol1 and vol2 unless `--vol1` or `--vol2` narrows the set, in
    the chapter order from the volume config. `--from STEM` starts at that
    chapter when it is in the list. Logs go to
    `books/_build/chapter-sequence/<UTC run id>/<volume>/`. On the first
    failure it prints the output tail and a resume command and returns 1,
    unless `--continue-on-error` is set, in which case it returns 1 at the
    end if any chapter failed. Returns 0 when every build passes.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--vol1", action="store_true")
    p.add_argument("--vol2", action="store_true")
    p.add_argument("--format", default="html", choices=["html", "pdf"])
    p.add_argument("--from", dest="from_chapter")
    p.add_argument("--continue-on-error", action="store_true")
    args = p.parse_args()
    volumes = ["vol1"] if args.vol1 else []
    if args.vol2: volumes.append("vol2")
    if not volumes: volumes = ["vol1", "vol2"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_root = BOOK / "_build/chapter-sequence" / run_id
    log_root.mkdir(parents=True, exist_ok=True)
    failed = 0
    for volume in volumes:
        chapters = get_chapters_from_config(BOOK, volume)
        if args.from_chapter and args.from_chapter in chapters:
            chapters = chapters[chapters.index(args.from_chapter):]
        print(f"\n=== {volume}: {len(chapters)} chapters ===\n")
        vol_log = log_root / volume
        vol_log.mkdir(parents=True, exist_ok=True)
        for idx, stem in enumerate(chapters, 1):
            print(f"[{idx}/{len(chapters)}] {volume}/{stem} ...", flush=True)
            ok, secs, tail = _build(volume, stem, args.format, vol_log)
            print(f"  {'PASS' if ok else 'FAIL'} ({secs:.0f}s)")
            if not ok:
                failed += 1
                print(tail)
                if not args.continue_on_error:
                    print(f"\nResume: python3 binder/tools/audit/chapter_build_sequence.py --{volume} --format {args.format} --from {stem}")
                    return 1
    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main())
