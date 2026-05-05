# Book index audit tools

These scripts enforce the index conventions in `.claude/rules/index.md` and the formatting rules in `.claude/rules/book-prose.md`. They are designed to be wired into the project's pre-commit hook chain.

## Scripts

| Script | What it checks | Exit |
|---|---|---|
| `check_anti_patterns.py` | `\index{}` keys against §9 anti-pattern list (sub-subentries, author-year, et al., generic-bare, inline-Python, underscore/ampersand bugs, article-leading, plural duplicates, parenthetical-acronym, lowercase off allowlist) | 0 if clean, 1 on violation |
| `check_tag_placement.py` | `\index{}` placement: not inside opening-bold spans, not inside `code` spans, not on heading lines | 0 if clean, 1 on violation |
| `check_xref_resolves.py` | Every `\index{X\|see{Y}}` and `\index{X\|seealso{Y}}` target Y is a valid main entry | 0 if clean, 1 on broken |
| `audit.py` | Comprehensive audit: produces 4 CSV reports (coverage gaps, singleton classifier, subentry summary, see-ref check). Read-only. | 0 always |
| `format_audit.py` | Detailed formatting violation scan (V1–V7). Read-only. | 0 always |

## Wiring into pre-commit

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: book-index-anti-patterns
      name: "Book: Index anti-pattern audit"
      entry: python3 book/tools/audit/index/check_anti_patterns.py
      language: system
      pass_filenames: false
      files: 'book/quarto/contents/.*\.qmd$'

    - id: book-index-tag-placement
      name: "Book: Index tags not in bold/code/headings"
      entry: python3 book/tools/audit/index/check_tag_placement.py
      language: system
      pass_filenames: false
      files: 'book/quarto/contents/.*\.qmd$'

    - id: book-index-xref-resolves
      name: "Book: Index cross-references resolve"
      entry: python3 book/tools/audit/index/check_xref_resolves.py
      language: system
      pass_filenames: false
      files: 'book/quarto/contents/.*\.qmd$'
```

## Manual audit run

```bash
# Quick health check
python3 book/tools/audit/index/check_anti_patterns.py
python3 book/tools/audit/index/check_tag_placement.py
python3 book/tools/audit/index/check_xref_resolves.py

# Comprehensive audit (produces CSV reports)
python3 book/tools/audit/index/audit.py

# Detailed format violation scan
python3 book/tools/audit/index/format_audit.py
```

## Allowlist updates

When adding a new lowercase-allowed main entry (e.g., a new framework API like `tf.something`), update:
1. `check_anti_patterns.py` — `LOWERCASE_ALLOWLIST` set
2. `.claude/rules/index.md` — §3 lowercase main-entry allowlist

When adding a new parenthetical-acronym headword (rare; only for genuine disambiguation like `Precision (Metric)`), update:
1. `check_anti_patterns.py` — `PARENTHETICAL_ALLOWLIST` set
2. `.claude/rules/index.md` — note as exception

## How these relate to the rules files

- All anti-patterns in `check_anti_patterns.py` are codified in `.claude/rules/index.md` §9
- Tag-placement rules trace to `.claude/rules/index.md` §1 + `.claude/rules/book-prose.md` heading-case + the project's `book-check-index-placement` hook
- Cross-reference rules from `.claude/rules/index.md` §6
