# Post-Merge Build QA Notes

Working directory: `/Users/VJ/GitHub/Dev-build-post-merge`

Branch: `dev-build-post-merge`

Base state: this worktree was created from `dev` at commit `c5a3264336 Audit post-merge prose style consistency`, which includes merge commit `1011b74986 Merge branch 'codex-callouts' into dev`.

## Purpose

Validate the post-merge book state after the callout, capitalization, bold-label, caption, and principle-reference cleanup. The goal is to prove that every chapter renders cleanly by itself in HTML and PDF before attempting full-volume builds.

Do not use the main `/Users/VJ/GitHub/MLSysBook` dev worktree for this QA pass.

## Operating Rules

1. Run all commands with `workdir=/Users/VJ/GitHub/Dev-build-post-merge`.
2. Before any major phase, verify:
   - `pwd`
   - `git branch --show-current`
   - `git status --short --branch`
3. HTML comes first because it is faster and catches many QMD/math/render issues.
4. Per-chapter PDF builds come before full-volume PDF builds.
5. Fix only confirmed build/render/style regressions.
6. Use `apply_patch` for hand edits.
7. Do not bulk rewrite QMD files.
8. Commit after each major completed phase if fixes were made.
9. Preserve unrelated files and do not touch the main dev worktree.
10. Follow `.claude/rules/book-prose.md` as the source of truth for editorial decisions.
11. If a problem requires user/editorial judgment rather than a clear mechanical fix, add it to [Needs User Decision](#needs-user-decision) and continue with other build work.

## Needs User Decision

Status: none currently.

Use this section for questions that block only a local decision, not the entire build QA pass.

## Known Important Context

The UI footer may still show the ambient session directory (`~/GitHub`). That is not the command working directory. Every command for this pass must explicitly use `/Users/VJ/GitHub/Dev-build-post-merge`.

The intentional heading/context distinction:

- H4 heading uses sentence style: `#### Archetype B (DLRM at scale)`.
- Body prose, table-label cells, and lighthouse callout titles keep the canonical label: `Archetype B (DLRM at Scale)`.
- Binder heading-case check currently passes with the H4 sentence-style version.

## Pre-Build Style Regression Checklist

Run and keep clean:

```bash
rg -n "^<<<<<<<|^=======$|^>>>>>>>|icon=false|icon=\"false\"|#nte-|\\\\ref\\{nte-|\\*\\*[^*]+:\\*\\*|fig-cap=\"\\*\\*[^\\\"]+\\*\\*\\.|lst-cap=\"\\*\\*[^\\\"]+\\*\\*\\.|tbl-cap=\"\\*\\*[^\\\"]+\\*\\*\\." book/quarto/contents/vol1 book/quarto/contents/vol2 -g '*.qmd'
```

Expected result: no matches.

Meaning:

- no unresolved merge markers
- no removed `icon=false` residue
- no old `#nte-` / `\ref{nte-}` principle references
- no colon-inside-bold `**Label:**` pattern
- no malformed caption head period pattern

Also run:

```bash
pre-commit run --all-files
```

Expected result: pass before full volume builds. If hooks modify files, inspect the diff before committing.

## Chapter Inventory

Binder chapter list commands:

```bash
./book/binder list --vol1
./book/binder list --vol2
```

Current inventory from Binder:

- vol1: 33 entries
- vol2: 37 entries

Do not assume all entries are normal chapters. Some are frontmatter, backmatter, glossary, references, parts, or shelved content. Build exactly what Binder accepts, in Binder order, unless Binder debug excludes an entry.

## Binder Commands To Use

Help command:

```bash
./book/binder help
```

Known commands:

```bash
./book/binder build html --vol1
./book/binder build html --vol2
./book/binder build pdf --vol1
./book/binder build pdf --vol2
./book/binder build html --all
./book/binder build pdf --all
./book/binder debug html --vol1
./book/binder debug html --vol2
./book/binder debug pdf --vol1
./book/binder debug pdf --vol2
```

Specific chapter examples from help:

```bash
./book/binder build pdf vol1/intro
./book/binder build pdf intro
```

Need confirm exact accepted chapter identifiers before looping. Prefer Binder’s own list/debug mechanisms over guessing paths.

## Phase 1: Setup And Documentation

Status: completed.

Result:

- Worktree verified: `/Users/VJ/GitHub/Dev-build-post-merge`
- Branch verified: `dev-build-post-merge`
- Binder help checked.
- Binder inventory checked: vol1 has 33 entries, vol2 has 37 entries.
- QA plan committed as `be38463a1 Document post-merge build QA plan`.

Steps:

1. Verify worktree identity:
   ```bash
   pwd
   git branch --show-current
   git status --short --branch
   git log --oneline -3
   ```
2. Confirm Binder command behavior:
   ```bash
   ./book/binder help
   ./book/binder list --vol1
   ./book/binder list --vol2
   ```
3. Decide whether to use Binder debug directly or per-chapter build commands. If debug gives clear chapter-by-chapter output, use it. If not, build chapters individually.

## Phase 2: HTML Chapter Pass

Status: completed.

Result:

- `./book/binder debug html --vol1`: 25 passed, 0 failed.
- `./book/binder debug html --vol2`: 26 passed, 0 failed.
- Vol1 debug log directory: `book/quarto/_build/debug/vol1/html/20260509-220517`
- Vol2 debug log directory: `book/quarto/_build/debug/vol2/html/20260509-221120`
- Artifact scan found no `katex-error`, math-processing error, raw `{python}` output, or visible `Unresolved` marker in generated HTML artifacts.
- Style-regression scan for merge markers, disabled callout icons, manual note refs, bold-colon punctuation, and malformed caption-title endings returned no matches.
- Watch item: `data_engineering` emitted two Quarto warnings that a literal `:::` string was found. The source uses the required three-colon fenced-div syntax, and the generated HTML artifact contains no literal `:::`. Treat as a warning to revisit if it appears in PDF or full-volume builds.

Goal: every individual QMD entry that Binder treats as renderable should build as HTML.

Procedure:

1. Start with vol1 in Binder order.
2. Build one chapter at a time or use `./book/binder debug html --vol1` if it systematically isolates failing chapters.
3. For each failure:
   - read the relevant QMD context
   - read the render log
   - classify issue:
     - QMD syntax
     - inline Python
     - math/LaTeX
     - cross-reference/citation
     - figure/table/listing
     - callout/title/caption
   - fix manually with `apply_patch`
   - rerun the same chapter
4. Repeat for vol2.

HTML-specific inspection:

After a successful HTML build, search generated output/logs for obvious unresolved render artifacts:

- raw `{python}`
- raw `\ref{`
- `??`
- unresolved citations if visible
- raw display math that should have rendered
- missing image warnings

Exact output directory must be discovered from Binder/quarto output before scanning.

Commit checkpoint after all HTML chapter issues are fixed:

```bash
git status --short
git diff --stat
pre-commit run --all-files
git add <fixed files>
git commit -m "Fix post-merge HTML chapter build issues"
```

Only commit if there are fixes.

## Phase 3: HTML Reflection Pass

Status: completed once after the initial HTML pass.

Result:

- Rechecked generated HTML artifacts for obvious math/rendering failures.
- Rechecked the agreed post-merge style-regression patterns across all vol1/vol2 QMD files.
- No confirmed HTML-blocking issue found.
- Remaining watch item is the `data_engineering` fenced-div warning described in Phase 2.

Run at least one broad reflection sweep after HTML passes:

1. Re-run the pre-build style regression checklist.
2. Re-run Binder HTML debug for both volumes.
3. Scan HTML outputs/logs for unresolved math/crossref/inline-Python artifacts.
4. Inspect any suspicious hits manually.
5. Fix only confirmed issues.

Do not proceed to PDF until HTML is clean.

## Phase 4: PDF Chapter Pass

Status: pending.

Goal: every individual QMD entry that Binder treats as renderable should build as PDF.

Procedure:

1. Run vol1 PDF debug or chapter-by-chapter PDF builds:
   ```bash
   ./book/binder debug pdf --vol1
   ```
2. If a failure occurs:
   - inspect the LaTeX/log output
   - inspect QMD context
   - classify issue:
     - unescaped LaTeX special character
     - math syntax
     - Unicode/font
     - table/listing/figure layout
     - citation/ref
     - package limitation
   - patch manually
   - rerun failing chapter
3. Repeat for vol2:
   ```bash
   ./book/binder debug pdf --vol2
   ```

PDF-specific checks:

- no LaTeX fatal errors
- no raw QMD syntax in output/logs
- math compiles
- figures/tables/listings compile
- captions do not break LaTeX
- Unicode in titles/captions does not break PDF

Commit checkpoint after all PDF chapter fixes:

```bash
git status --short
git diff --stat
pre-commit run --all-files
git add <fixed files>
git commit -m "Fix post-merge PDF chapter build issues"
```

Only commit if there are fixes.

## Phase 5: Full Volume Builds

Status: pending.

Only start after:

- all HTML chapter builds pass
- all PDF chapter builds pass
- pre-commit passes

Commands:

```bash
./book/binder build html --vol1
./book/binder build html --vol2
./book/binder build pdf --vol1
./book/binder build pdf --vol2
```

If both volumes need one combined build:

```bash
./book/binder build html --all
./book/binder build pdf --all
```

If full volume fails but chapter builds pass:

1. Treat it as integration-level issue.
2. Inspect log around failure.
3. Identify whether issue is:
   - cross-chapter reference
   - bibliography
   - duplicated labels
   - front/backmatter
   - global LaTeX/PDF setting
4. Fix manually.
5. Rerun affected chapter if possible.
6. Rerun full volume.

Commit final build fixes:

```bash
git status --short
git diff --stat
pre-commit run --all-files
git add <fixed files>
git commit -m "Fix post-merge full volume build issues"
```

Only commit if there are fixes.

## Phase 6: Final Report

Status: pending.

Report must include:

1. Worktree and branch used.
2. Starting commit.
3. Chapter counts processed.
4. HTML status:
   - vol1 chapter pass result
   - vol2 chapter pass result
   - fixes made, if any
5. PDF status:
   - vol1 chapter pass result
   - vol2 chapter pass result
   - fixes made, if any
6. Full volume build status:
   - vol1 HTML
   - vol2 HTML
   - vol1 PDF
   - vol2 PDF
7. Pre-commit status.
8. Commit hashes created during QA.
9. Remaining risks or known benign warnings.

## Resume Checklist After Context Compaction

If context is compacted, resume by running:

```bash
cd /Users/VJ/GitHub/Dev-build-post-merge
pwd
git branch --show-current
git status --short --branch
git log --oneline -5
sed -n '1,260p' BUILD_QA_NOTES.md
```

Then continue from the first phase whose status is still `pending`.

Update this file after each completed phase by changing that phase’s status line from `pending` to `completed`, and add a short result note.
