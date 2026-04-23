#!/usr/bin/env python3
"""
Phase 0 audit for the deep_dive → resources migration.

Walks every question YAML under interviews/vault/questions/, extracts the
details.deep_dive field (when present), and emits counts that determine
the migration's blast radius:

  - total questions
  - questions with deep_dive present / absent
  - URL hostname distribution
  - book-host questions (mlsysbook.ai, harvard-edge.github.io) — these get
    dropped on the floor during migration per the resources-list plan
  - orphan questions: have deep_dive.url but no deep_dive.title (name fallback
    needed during migration)
  - "book-only" questions: their single deep_dive is a book URL, so post-migration
    they have zero resources — author review candidate list

Read-only. Writes one JSON report to scripts/_resources_migration_audit.json
and prints a human-readable summary to stdout.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import yaml

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_DIR = ROOT / "questions"
REPORT_PATH = Path(__file__).resolve().parent / "_resources_migration_audit.json"

BOOK_HOSTS = {"mlsysbook.ai", "harvard-edge.github.io"}


def host_of(url: str) -> str:
    try:
        h = urlparse(url).hostname or ""
        return h.lower().removeprefix("www.")
    except Exception:
        return ""


def main() -> int:
    yaml_files = sorted(QUESTIONS_DIR.rglob("*.yaml"))
    total = len(yaml_files)
    if total == 0:
        print(f"FATAL: no YAML files found under {QUESTIONS_DIR}", file=sys.stderr)
        return 2

    with_deep_dive = 0
    without_deep_dive = 0
    title_missing = 0               # has url, no title (name fallback needed)
    url_missing = 0                 # has title, no url (degenerate — we should log)
    hostname_counts: Counter[str] = Counter()
    book_host_count = 0
    book_only_ids: list[str] = []   # single deep_dive is a book URL -> zero refs post-migration
    orphans_without_title: list[dict] = []   # id + url, for fallback naming
    parse_failures: list[str] = []

    # Track by (track, level, zone) for distribution sanity
    by_track: Counter[str] = Counter()
    by_track_with_ref: Counter[str] = Counter()

    for fp in yaml_files:
        rel = fp.relative_to(ROOT).as_posix()
        # questions/<track>/<level>/<zone>/<id>.yaml
        parts = fp.relative_to(QUESTIONS_DIR).parts
        track = parts[0] if parts else "?"
        by_track[track] += 1

        try:
            data = yaml.safe_load(fp.read_text(encoding="utf-8"))
        except Exception as e:
            parse_failures.append(f"{rel}: {e}")
            continue

        if not isinstance(data, dict):
            parse_failures.append(f"{rel}: top-level not a dict")
            continue

        qid = data.get("id", fp.stem)
        details = data.get("details") or {}
        deep_dive = details.get("deep_dive") if isinstance(details, dict) else None

        if not deep_dive or not isinstance(deep_dive, dict):
            without_deep_dive += 1
            continue

        url = (deep_dive.get("url") or "").strip()
        title = (deep_dive.get("title") or "").strip()

        if not url and not title:
            without_deep_dive += 1
            continue

        with_deep_dive += 1
        by_track_with_ref[track] += 1

        if not url:
            url_missing += 1
            continue

        if not title:
            title_missing += 1
            orphans_without_title.append({"id": qid, "url": url, "path": rel})

        host = host_of(url)
        hostname_counts[host] += 1

        if host in BOOK_HOSTS:
            book_host_count += 1
            # book-only = this question's single deep_dive is a book URL → migration drops it
            book_only_ids.append(qid)

    # Compose report
    report = {
        "audited_at_iso": "2026-04-16",
        "total_questions": total,
        "with_deep_dive": with_deep_dive,
        "without_deep_dive": without_deep_dive,
        "deep_dive_coverage_pct": round(100.0 * with_deep_dive / total, 1) if total else 0.0,
        "title_missing_count": title_missing,
        "url_missing_count": url_missing,
        "book_host_count": book_host_count,
        "book_host_pct_of_refs": round(100.0 * book_host_count / with_deep_dive, 1) if with_deep_dive else 0.0,
        "top_hostnames": hostname_counts.most_common(25),
        "by_track_total": dict(by_track),
        "by_track_with_ref": dict(by_track_with_ref),
        "parse_failure_count": len(parse_failures),
        "parse_failures_sample": parse_failures[:10],
        "orphans_without_title_count": title_missing,
        "orphans_without_title_sample": orphans_without_title[:15],
        "book_only_questions_will_lose_ref_sample": book_only_ids[:15],
        "book_only_questions_total_to_lose_ref": book_host_count,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Human summary
    print()
    print("═══════════════════════════════════════════════════════════")
    print("  StaffML Phase 0 Audit — deep_dive → resources migration")
    print("═══════════════════════════════════════════════════════════")
    print(f"  Questions walked              : {total:>6}")
    print(f"  With deep_dive reference      : {with_deep_dive:>6}  ({report['deep_dive_coverage_pct']}%)")
    print(f"  Without any reference         : {without_deep_dive:>6}")
    print(f"  Has URL but no title          : {title_missing:>6}  (name-fallback needed)")
    print(f"  Has title but no URL          : {url_missing:>6}  (degenerate)")
    print()
    print(f"  Book-host references          : {book_host_count:>6}  ({report['book_host_pct_of_refs']}% of refs)")
    print(f"  → These drop during migration")
    print(f"  → Same {book_host_count} questions lose their only ref")
    print()
    print("  Track distribution:")
    for t, c in sorted(by_track.items()):
        wr = by_track_with_ref.get(t, 0)
        pct = round(100.0 * wr / c, 1) if c else 0.0
        print(f"    {t:<8} total={c:>5}  with_ref={wr:>5}  ({pct}%)")
    print()
    print("  Top 15 hostnames in existing refs:")
    for host, count in hostname_counts.most_common(15):
        flag = "  [BOOK]" if host in BOOK_HOSTS else ""
        pct = round(100.0 * count / with_deep_dive, 1) if with_deep_dive else 0.0
        print(f"    {host:<40} {count:>5}  ({pct:>5}%){flag}")
    print()
    if parse_failures:
        print(f"  ⚠️  Parse failures: {len(parse_failures)} (first 5 shown)")
        for line in parse_failures[:5]:
            print(f"     - {line}")
        print()
    print(f"  Full report → {REPORT_PATH.relative_to(ROOT)}")
    print("═══════════════════════════════════════════════════════════")
    return 0


if __name__ == "__main__":
    sys.exit(main())
