"""One-shot backfill: push every local .md file's author into Buttondown
as ``email.metadata.author``.

Why this exists: our posts historically stored the byline only in local
.md frontmatter. Now that ``news pull`` honors ``metadata.author`` as
its source of truth, we need to seed Buttondown with the authors we've
already curated in the repo so server and repo agree.

Idempotent: posts whose remote ``metadata.author`` already matches the
local frontmatter are skipped. Pass ``--apply`` to actually PATCH; by
default this prints the plan and exits.

Match rule: local post filename date (YYYY-MM-DD_...) is compared to
the remote email's publish_date (YYYY-MM-DD...). Unambiguous matches
are taken; ambiguities are reported and skipped.

Usage (run from site/newsletter/):
    python3 -m cli.scripts.backfill_metadata_author          # dry run
    python3 -m cli.scripts.backfill_metadata_author --apply  # do it
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Allow execution via `python3 -m cli.bin.backfill_metadata_author`
# or directly from within site/newsletter/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cli.core.buttondown import (  # noqa: E402
    ButtondownError,
    list_emails,
    update_email,
)
from cli.core.config import Config, load_api_key  # noqa: E402


FILENAME_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_")
AUTHOR_LINE_RE = re.compile(r'^author:\s*"([^"]+)"', re.MULTILINE)


def _read_local_author(path: Path) -> str | None:
    head = "".join(path.open("r", encoding="utf-8").readlines()[:30])
    m = AUTHOR_LINE_RE.search(head)
    return m.group(1) if m else None


def _date_of(filename: str) -> str | None:
    m = FILENAME_DATE_RE.match(filename)
    return m.group(1) if m else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually PATCH Buttondown. Default: dry run.",
    )
    args = parser.parse_args()

    cfg = Config.discover()
    api_key = load_api_key(cfg)

    posts: list[Path] = sorted(cfg.posts_dir.glob("*/*.md"))
    if not posts:
        print("No local posts found.")
        return 0

    print(f"Found {len(posts)} local posts. Fetching Buttondown catalog...")
    emails = list_emails(api_key, status="sent")
    by_date: dict[str, list[dict]] = {}
    for e in emails:
        pd = (e.get("publish_date") or "")[:10]
        if pd:
            by_date.setdefault(pd, []).append(e)

    print(f"{len(emails)} remote emails indexed by publish_date.\n")

    apply = args.apply
    planned = skipped_unchanged = skipped_nomatch = skipped_ambiguous = patched = failed = 0

    for post in posts:
        rel = post.relative_to(cfg.posts_dir)
        date = _date_of(post.name)
        local_author = _read_local_author(post)

        if not date or not local_author:
            print(f"  SKIP  {rel}: could not parse date or author")
            skipped_nomatch += 1
            continue

        candidates = by_date.get(date, [])
        if not candidates:
            print(f"  SKIP  {rel}: no remote email on {date}")
            skipped_nomatch += 1
            continue
        if len(candidates) > 1:
            subjects = [c.get("subject", "?")[:50] for c in candidates]
            print(f"  SKIP  {rel}: {len(candidates)} remote emails on {date}: {subjects}")
            skipped_ambiguous += 1
            continue

        email = candidates[0]
        current = (email.get("metadata") or {}).get("author", "")

        if current == local_author:
            skipped_unchanged += 1
            continue

        planned += 1
        print(f"  {'PATCH' if apply else 'PLAN '} {rel}")
        print(f"         id={email['id']}  subject={email.get('subject', '?')[:60]!r}")
        print(f"         {current!r}  ->  {local_author!r}")

        if not apply:
            continue

        merged = {**(email.get("metadata") or {}), "author": local_author}
        try:
            update_email(api_key, email["id"], {"metadata": merged})
            patched += 1
        except ButtondownError as exc:
            print(f"         ! PATCH failed: {exc}")
            failed += 1

    print()
    print(f"Summary: {planned} planned, {patched} patched, {failed} failed, "
          f"{skipped_unchanged} already in sync, "
          f"{skipped_nomatch} no match, {skipped_ambiguous} ambiguous")

    if not apply and planned:
        print("\nDry run. Re-run with --apply to execute.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
