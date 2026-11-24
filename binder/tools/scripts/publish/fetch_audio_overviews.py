#!/usr/bin/env python3
"""Copy the audio overviews listed in the manifest next to each volume's pages.

Usage:
  python3 binder/tools/scripts/publish/fetch_audio_overviews.py \
      --manifest books/shared/media/audio-overviews.json \
      --dest 'vol{vol}-site/assets/media'

Audio files are not kept in git. They are assets of the `audio-overviews` GitHub
Release (see docs/media-hosting.md), and the manifest lists each file with its
release URL, size, and SHA-256. At deploy time this script downloads every file
whose volume folder exists (for example vol1-site/), checks size and SHA-256, and
writes it to the destination. The pages then serve the audio from the same origin,
as audio/mp4, with the release URL kept as a fallback <source>.

A failed download is reported and skipped, since the page still plays from the
release URL. A size or SHA-256 mismatch exits non-zero, because it means the
release asset no longer matches what the manifest says was reviewed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path


def download(url: str, attempts: int = 3) -> bytes:
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mlsysbook-deploy"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # network errors are retried, then reported
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"download failed after {attempts} attempts: {last}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dest", required=True, help="destination pattern with {vol}, e.g. 'vol{vol}-site/assets/media'")
    args = ap.parse_args()

    m = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    base = m.get("base_url", "")
    mismatches = 0
    for key, item in m.get("items", {}).items():
        vol = str(item.get("volume", ""))
        dest_dir = Path(args.dest.format(vol=vol))
        if not vol or not dest_dir.parent.parent.exists():
            print(f"  - {key}: volume folder for vol{vol} not in this deploy; skipped")
            continue
        url = item.get("url") or base + item["file"]
        try:
            data = download(url)
        except RuntimeError as e:
            print(f"  ⚠️ {key}: {e}; the page falls back to the release URL")
            continue
        size_ok = len(data) == int(item["bytes"])
        sha = hashlib.sha256(data).hexdigest()
        if not size_ok or sha != item["sha256"]:
            print(f"  ❌ {key}: release asset does not match manifest (bytes {len(data)} vs {item['bytes']}, sha256 {sha})")
            mismatches += 1
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / item["file"]).write_bytes(data)
        print(f"  ✅ {key}: {dest_dir / item['file']} ({len(data) / 1e6:.1f} MB, sha256 ok)")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
