#!/usr/bin/env python3
"""
Link checker for corpus deep_dive_url values.

Walks src/data/corpus.json, collects every unique deep_dive_url, probes each
once with a HEAD (then GET fallback) over a small concurrent worker pool, and
emits a structured JSON report at scripts/_deep_dive_link_report.json plus a
human-readable summary on stdout.

Why:
  ~3,000 of 4,159 deep_dive_urls in the corpus currently 404 or are flaky
  (mlsysbook.ai chapter routes return 404 even though the homepage links to
  them; the harvard-edge.github.io dev mirror is fully retired). This script
  surfaces the problem in CI so future regressions are caught at PR time.

Usage:
  python3 scripts/check-deep-dive-links.py                # full check
  python3 scripts/check-deep-dive-links.py --hosts arxiv.org pytorch.org
  python3 scripts/check-deep-dive-links.py --fail-on-broken    # exit 1 if any URL is dead

CI integration (suggested, not yet wired):
  - Run weekly via GitHub Action
  - Compare diff against the previous report and post a PR comment
  - Fail the build only on *new* breakage, not on the existing backlog

Output report shape:
  {
    "checked_at": "2026-04-07T18:42:00Z",
    "total_urls": 4159,
    "unique_urls": 612,
    "by_status": { "200": 423, "404": 180, "timeout": 6, ... },
    "by_host":   { "mlsysbook.ai": { "200": 0, "404": 117 }, ... },
    "broken":    [ { "url": "...", "status": 404, "occurrences": 23 }, ... ]
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
CORPUS_PATH = Path(__file__).resolve().parent.parent / "src" / "data" / "corpus.json"
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


# ───────────────────── Corpus walking ──────────────────────
def collect_urls(corpus_path: Path) -> dict[str, int]:
    """Return {url: occurrence_count} from corpus.json."""
    with corpus_path.open() as f:
        data = json.load(f)

    questions = data
    if isinstance(data, dict):
        # corpus.json is a dict that includes a 'questions' array under some key
        for key in ("questions", "items", "data"):
            if key in data and isinstance(data[key], list):
                questions = data[key]
                break

    if not isinstance(questions, list):
        raise SystemExit(
            f"Could not find a question list in {corpus_path}. Top-level keys: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}"
        )

    counts: Counter[str] = Counter()
    for q in questions:
        if not isinstance(q, dict):
            continue
        details = q.get("details") or {}
        url = details.get("deep_dive_url")
        if url:
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

    if not CORPUS_PATH.exists():
        print(f"FATAL: corpus not found at {CORPUS_PATH}", file=sys.stderr)
        return 2

    print(f"Loading corpus from {CORPUS_PATH}")
    occurrences = collect_urls(CORPUS_PATH)
    total_links = sum(occurrences.values())
    unique_urls = list(occurrences.keys())

    if args.hosts:
        allow = set(args.hosts)
        unique_urls = [u for u in unique_urls if (urlparse(u).hostname or "") in allow]
        print(f"Filtered by hosts {sorted(allow)}: {len(unique_urls)} URLs to probe.")

    print(f"Found {total_links} total references → {len(occurrences)} unique URLs")
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
    print(f"Total references in corpus:  {total_links}")
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
