# Retiring the `books/` mirror

**Retired at:** `2e8119556c3124c79f349569cfd572720eef99be` (the commit immediately before removal)

For most of this repository's life the book existed in **two tracked copies**:

| Path | Role |
|---|---|
| `publishing/quarto/contents/vol{1..4}/` | the tree every build and every check actually read |
| `books/vol{1..4}/` | a partial, drifting copy that fed nothing |

Nothing consumed `books/`. The only reference to it anywhere in the build was
change-detection in `.github/workflows/book-validate-dev.yml`, which used the
path to decide which volume to validate, never as a source.

The mirror was measurably behind. At retirement:

| Volume | `books/` | live tree |
|---|---|---|
| vol1 | 33 qmd | 36 qmd |
| vol2 | 35 qmd | 38 qmd |
| vol3 | 3 qmd | 30 qmd |
| vol4 | 35 qmd | 39 qmd |

## Why this mattered

Editorial work kept landing in the copy that fed nothing, where no render and
no check could see it. Nineteen commits touched `books/` without touching live
chapter content. Most were TinyTorch monograph work that legitimately lived
there, but these are real textbook edits that never reached the build:

- `3e5c5dd50e` 2026-09-10 — Reorder Part I: place Brain as Ch 3 and Nervous System as Ch 4
- `bed46ac0ee` 2026-09-09 — editorial(vol2): eliminate stray em-dashes across Volume II chapters
- `bca994a8ed` 2026-09-09 — editorial(vol2): surgical narrative tone and progressive flow pass
- `1b298dbeaf` 2026-09-07 — OUTLINE.md: link the OSR agentic-stack paper to its article DOI
- `7e4525e2db` 2026-09-07 — Vol III/IV READMEs: add audience, follow-along, license sections; retire superseded vol3
- `352a9d7d0d` 2026-09-07 — Simplify Vol III/IV READMEs, add per-volume book feedback issue forms
- `dd06f6abc2` 2026-09-06 — Standardize paper citations with full titles and authors in OUTLINE and references
- `8f42560843` 2026-09-05 — Prune vol4's dead callout taxonomy and stale xref artifacts
- `f3ee4c5735` 2026-09-05 — Eliminate ASCII box-drawing, format native tables, and add monograph-wide napkin math

The clearest case is the Part I reorder placing Brain at Chapter 3 and Nervous
System at Chapter 4. Every content file it changed was under `books/`, so the
live configuration still ordered the chapters the old way and every locator
figure still printed the old chapter numbers. The decision was made and
committed, and the book never saw it.

## Recovering anything from the mirror

The tree is fully preserved in history. To read a file as it stood:

```bash
git show <sha-above>:books/vol4/chapters/03-brain/03-brain.qmd
```

To restore the whole mirror into a scratch directory for comparison:

```bash
git worktree add /tmp/books-mirror <sha-above>
```

To port a specific stranded commit, diff it against the live tree first; the
two trees diverged in both directions, so these are not clean cherry-picks.

## What replaced it

`books/` is now the Quarto project root and the single home for book sources.
See `docs/REPO_LAYOUT.md`.
