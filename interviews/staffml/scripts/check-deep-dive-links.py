#!/usr/bin/env python3
"""
Link checker for the StaffML → textbook chapter-URL manifest.

Walks src/data/chapter-urls.json (the 27-entry chapter-id → relative-path map
consumed by src/lib/refs.ts), prefixes each path with mlsysbook.ai, probes
each URL once via curl over a small concurrent worker pool, and emits a
structured JSON report at scripts/_deep_dive_link_report.json plus a
human-readable summary on stdout.

Background:
  The per-question `deep_dive_url` field was removed during the vault
  migration (Phase 1). StaffML now links to textbook chapters via this
  manifest. Topic-granular linking is a separate, deferred design
  (see interviews/vault/BOOK_LINKING_PLAN.md). Until that ships, the
  chapter-URL manifest IS the user-facing link surface — probing it keeps
  us honest about chapter-level link health.

Usage:
  python3 scripts/check-deep-dive-links.py                  # full check
  python3 scripts/check-deep-dive-links.py --hosts mlsysbook.ai
  python3 scripts/check-deep-dive-links.py --fail-on-broken # exit 1 if any URL is dead

Output report shape (keys stable for the workflow to parse):
  {
    "checked_at": "2026-04-16T18:42:00Z",
    "total_links": 27,
    "unique_urls": 27,
    "by_status": { "200": 27, "404": 0, ... },
    "by_host":   { "mlsysbook.ai": { "200": 27 }, ... },
    "broken":    [ { "url": "...", "status": 404, "occurrences": 1 }, ... ]
  }
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import subprocess
import shutil

if shutil.which("curl") is None:
    print("FATAL: curl is required (sudo apt install curl / brew install curl)", file=sys.stderr)
    sys.exit(2)

# ───────────────────────── Config ──────────────────────────
TIMEOUT_SECONDS = 6
MAX_WORKERS = 8
USER_AGENT = "StaffML-LinkChecker/1.0 (+https://staffml.ai)"
BASE_URL = "https://mlsysbook.ai"
SOURCE_PATH = Path(__file__).resolve().parent.parent / "src" / "data" / "chapter-urls.json"
REPORT_PATH = Path(__file__).resolve().parent / "_deep_dive_link_report.json"

# Hosts we know are broken (mark in report but don't even try to probe to save time)
KNOWN_DEAD_HOSTS = {
    "harvard-edge.github.io",
}


# ───────────────────── Probing logic ───────────────────────
def probe_url(url: str) -> dict:
    """Return {status, host} for a single URL via curl.

    Uses HEAD (-I -L --head) with --location-trusted to follow redirects.
    Returns the final HTTP status code, or a sentinel string like
    'timeout' / 'dns' / 'tls' / 'invalid' / 'curl-fail'.
    """
    parsed = urlparse(url)
    host = parsed.hostname or ""

    if host in KNOWN_DEAD_HOSTS:
        return {"status": "known-dead", "host": host}

    if parsed.scheme not in ("http", "https"):
        return {"status": "invalid-scheme", "host": host}

    try:
        result = subprocess.run(
            [
                "curl",
                "-sL",                          # silent + follow redirects
                "-o", os.devnull,
                "-A", USER_AGENT,
                "--max-time", str(TIMEOUT_SECONDS),
                "-w", "%{http_code}",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS + 2,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "host": host}
    except Exception as e:
        return {"status": f"error: {type(e).__name__}", "host": host}

    if result.returncode != 0:
        # curl error code -> sentinel
        # https://curl.se/libcurl/c/libcurl-errors.html
        stderr_lower = (result.stderr or "").lower()
        if result.returncode == 6 or "could not resolve" in stderr_lower:
            return {"status": "dns", "host": host}
        if result.returncode == 28:
            return {"status": "timeout", "host": host}
        if result.returncode in (35, 60):
            return {"status": "tls", "host": host}
        return {"status": f"curl-fail-{result.returncode}", "host": host}

    code_str = (result.stdout or "").strip()
    if not code_str.isdigit():
        return {"status": "no-status", "host": host}
    return {"status": int(code_str), "host": host}


# ───────────────────── Manifest walking ────────────────────
def collect_urls(source_path: Path) -> dict[str, int]:
    """Return {url: occurrence_count} from the chapter-url manifest.

    chapter-urls.json is a flat {chapter_id: relative_path} dict. Each entry
    is one user-facing destination, so occurrences=1 per URL. The relative
    path is joined with BASE_URL to form the probe target.
    """
    with source_path.open() as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise SystemExit(
            f"Expected a flat dict in {source_path}, got {type(data).__name__}"
        )

    counts: Counter[str] = Counter()
    for chapter_id, rel_path in data.items():
        if not isinstance(rel_path, str) or not rel_path:
            continue
        # Relative paths are absolute under the site root (start with '/'),
        # so a simple concatenation with BASE_URL is correct.
        url = BASE_URL.rstrip("/") + "/" + rel_path.lstrip("/")
        counts[url] += 1
    return dict(counts)


# ─────────────────────── Main flow ─────────────────────────
def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Check StaffML corpus deep_dive_url health.")
    parser.add_argument("--hosts", nargs="*", default=None,
                        help="Only probe URLs whose host is in this allowlist.")
    parser.add_argument("--fail-on-broken", action="store_true",
                        help="Exit with code 1 if any URL is dead (status >= 400 or sentinel).")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-URL progress.")
    args = parser.parse_args(argv)

    if not SOURCE_PATH.exists():
        print(f"FATAL: chapter-url manifest not found at {SOURCE_PATH}", file=sys.stderr)
        return 2

    print(f"Loading chapter-url manifest from {SOURCE_PATH}")
    occurrences = collect_urls(SOURCE_PATH)
    total_links = sum(occurrences.values())
    unique_urls = list(occurrences.keys())

    if args.hosts:
        allow = set(args.hosts)
        unique_urls = [u for u in unique_urls if (urlparse(u).hostname or "") in allow]
        print(f"Filtered by hosts {sorted(allow)}: {len(unique_urls)} URLs to probe.")

    print(f"Found {total_links} manifest entries → {len(occurrences)} unique URLs")
    if args.hosts:
        print(f"Probing {len(unique_urls)} after host filter")
    else:
        print(f"Probing {len(unique_urls)} unique URLs (HEAD with GET fallback, "
              f"timeout {TIMEOUT_SECONDS}s, {MAX_WORKERS} workers)")

    started = time.time()
    results: dict[str, dict] = {}
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_to_url = {ex.submit(probe_url, u): u for u in unique_urls}
        for fut in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[fut]
            try:
                results[url] = fut.result()
            except Exception as e:
                results[url] = {"status": f"exception: {type(e).__name__}", "host": urlparse(url).hostname or ""}
            completed += 1
            if not args.quiet and completed % 25 == 0:
                print(f"  ... {completed}/{len(unique_urls)} probed", file=sys.stderr)

    elapsed = time.time() - started

    # ────────── Aggregation ──────────
    by_status: Counter[str] = Counter()
    by_host: dict[str, Counter[str]] = defaultdict(Counter)
    broken = []

    SUCCESS_CODES = {200, 201, 204, 301, 302, 303, 307, 308}

    for url, info in results.items():
        status = info.get("status")
        host = info.get("host", "")
        status_str = str(status)
        by_status[status_str] += 1
        by_host[host][status_str] += 1

        # Broken = anything that isn't a 2xx/3xx success code.
        # Sentinel strings (timeout/dns/tls/known-dead/...) all count as broken.
        is_success = isinstance(status, int) and status in SUCCESS_CODES
        if not is_success:
            broken.append({
                "url": url,
                "status": status,
                "host": host,
                "occurrences": occurrences.get(url, 0),
            })

    broken.sort(key=lambda r: -r["occurrences"])

    report = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(elapsed, 1),
        "total_links": total_links,
        "unique_urls": len(occurrences),
        "probed": len(unique_urls),
        "by_status": dict(by_status),
        "by_host": {h: dict(c) for h, c in sorted(by_host.items())},
        "broken_count": len(broken),
        "broken": broken,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {REPORT_PATH}")

    # ────────── Human summary ──────────
    print(f"\n=== Summary ({elapsed:.1f}s) ===")
    print(f"Manifest entries:            {total_links}")
    print(f"Unique URLs:                 {len(occurrences)}")
    print(f"Probed:                      {len(unique_urls)}")
    print(f"\nBy status:")
    for s, n in sorted(by_status.items(), key=lambda kv: -kv[1]):
        print(f"  {s:>14}  {n}")

    print(f"\nTop 10 broken hosts (by unique URL count):")
    host_broken = sorted(
        [(h, sum(n for s, n in cs.items() if s not in ("200", "301", "302", "303", "307", "308"))) for h, cs in by_host.items()],
        key=lambda kv: -kv[1],
    )[:10]
    for h, n in host_broken:
        if n:
            print(f"  {h:>40}  {n} broken")

    print(f"\nTop 10 broken URLs (by user-impact = occurrence count):")
    for b in broken[:10]:
        print(f"  [{b['status']}] x{b['occurrences']:<4} {b['url'][:90]}")

    if args.fail_on_broken and broken:
        print(f"\n❌ {len(broken)} broken URLs — exiting 1", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
