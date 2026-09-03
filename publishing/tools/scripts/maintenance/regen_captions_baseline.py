#!/usr/bin/env python3
"""Regenerate book/tools/audit/baselines/captions_baseline.json.

Snapshots the current per-file counts of captionless-float violations
flagged by the `tables.caption-required`, `figures.label-required`, and
`listings.caption-required` scopes. New violations beyond the per-file
budget will fail the corresponding `book-check-*` pre-commit hook;
shrinking a count is always safe.

Run from the repo root:
    python3 book/tools/scripts/maintenance/regen_captions_baseline.py
"""
import datetime as _dt
import json
import sys
from collections import Counter
from pathlib import Path

BOOK_DIR = Path(__file__).resolve().parents[3]
REPO_ROOT = BOOK_DIR.parent
sys.path.insert(0, str(BOOK_DIR))

from cli.commands.validate import (  # noqa: E402
    CAPTIONS_BASELINE_PATH,
    ValidateCommand,
)


class _Config:
    book_dir = BOOK_DIR / "quarto"


def main() -> int:
    vc = ValidateCommand(config_manager=_Config(), chapter_discovery=None)
    vc._load_captions_baseline = lambda: {}
    root = _Config.book_dir / "contents"

    counts = {}
    for code, method in [
        ("table_caption_required", vc._run_table_caption_required),
        ("figure_label_required", vc._run_figure_label_required),
        ("listing_caption_required", vc._run_listing_caption_required),
    ]:
        result = method(root)
        per_file = Counter(issue.file for issue in result.issues)
        counts[code] = dict(sorted(per_file.items()))
        print(
            f"  {code}: "
            f"{sum(per_file.values())} violations across "
            f"{len(per_file)} files"
        )

    baseline = {
        "description": (
            "Per-file allow-list of pre-existing captionless-float "
            "violations grandfathered at baseline time. New violations "
            "beyond these counts block the commit. Regenerate via "
            "book/tools/scripts/maintenance/regen_captions_baseline.py."
        ),
        "generated": _dt.date.today().isoformat(),
        "counts": counts,
    }
    CAPTIONS_BASELINE_PATH.write_text(
        json.dumps(baseline, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {CAPTIONS_BASELINE_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
