# Reference check (hallucinator)

Validates bibliography entries in the book’s `.bib` files against academic databases (CrossRef, arXiv, DBLP, Semantic Scholar, etc.) using [hallucinator](https://github.com/gianlucasb/hallucinator). **Native Binder CLI only** — implementation lives in `book/cli/commands/reference_check.py`. Use **`./book/binder validate references`** (not a standalone script).

## Run

From repo root:

```bash
# Default: vol1 + vol2 references.bib
./book/binder validate references

# One .bib file, report to file, quick test (first 5 refs)
./book/binder validate references -f book/quarto/contents/vol1/backmatter/references.bib -o report.txt --limit 5

# Full run, save report
./book/binder validate references -o book/tools/scripts/reference_check_report.txt
```

Options: `-f` / `--file BIB`, `-o` / `--output FILE`, `--limit N`, `--skip-verified`, `--thorough`, `--refs-cache FILE`.

- **Output**: Each ref is printed with its citation key and status (✓ verified, ? not found, ~ author_mismatch, ! error). After the summary, a **Not verified** block lists every key that needs review.
- **Cache**: By default a cache file `.references_verified.json` (repo root) stores per-key status so future runs can skip refs already verified.
  - `--skip-verified`: Only validate refs that are not already verified in the cache (faster repeat runs).
  - `--thorough`: Revalidate all refs and ignore cache for filtering (cache is still updated after the run).
  - `--refs-cache FILE`: Use a different cache file (default: repo root `.references_verified.json`).

## Install

One of:

- `pip install -e ".[reference-check]"` (from repo root; optional extra)
- `pip install -r book/tools/dependencies/requirements.txt` (book tooling deps)
- `pip install hallucinator bibtexparser` (minimal)

Optional env: `OPENALEX_KEY`, `S2_API_KEY`.

## Rate limits

**Semantic Scholar (S2)** allows **1 request per second** (cumulative across endpoints). With `S2_API_KEY` set, full runs over many references will take longer because of this limit; the validator may back off when rate-limited. Use `--limit N` for quick checks, or run full validation when you can leave it running.

## Results

- **Verified** — Found in a database with matching authors.
- **Not found** — Not in any checked DB (may still be valid: reports, books, very new papers). Check manually.
- **Author mismatch** — Title matched but authors differ.
- **Error** — Validator crashed or timed out for that ref (resilient mode skips and continues).

Exit code: `0` if all verified; `1` if any not found, mismatch, or error.

## Using the report (do not auto-correct)

**We do not auto-correct or rewrite `.bib` from this check.** Reasons:

- **Not found** — Many valid sources are not in academic DBs: vendor docs, standards (IEEE, ISO), blog posts, manuals, reports. Auto-“fixing” would delete or overwrite them.
- **Author mismatch** — Often formatting (e.g. “Smith, J.” vs “J. Smith”) or multi-author ordering; DB metadata can be wrong.
- **Risk** — Applying DB metadata blindly can introduce wrong DOIs, wrong authors, or duplicate entries.

**Use the report as a manual review list:**

1. **Not found (186 in your run)**  
   - If the work has a DOI or arXiv ID, add it to the entry and re-run; many will then verify.  
   - If it’s a report, standard, or doc, leave as-is and optionally add a `note` that it’s not in academic DBs.

2. **Author mismatch (23 in your run)**  
   - Open the entry in the report and in your `.bib`; compare authors.  
   - Fix only if the `.bib` is clearly wrong (typo, wrong person); ignore harmless formatting differences.

3. **Getting keys for batch review**  
   From the report file you can pull citation keys, e.g.:
   ```bash
   # Keys that were not found (for grepping .bib or scripting)
   grep -E '^\s+\[[^]]+\]' report.txt | sed 's/.*\[\([^]]*\)\].*/\1/'
   ```
   Or use the “Not verified” block printed at the end of `binder validate references` (same keys + status + title).

## Pre-commit

The hallucinator reference check is **not** in pre-commit (it is slow and uses optional deps/API keys). Pre-commit’s “book-check-references” runs `binder check refs` (in-repo citation/label checks), not this. Run `binder validate references` manually or in CI. For a quick gate use `--limit 10` or `--skip-verified`.

## Betterbib

The cache (`.references_verified.json`) records which citation keys were verified; **betterbib does not read it**. To avoid overwriting entries you’ve already verified:

- Run reference check first, then run betterbib only on files or keys you’re editing, or
- Run reference check after betterbib and fix any newly introduced issues.

A future wrapper could run betterbib only on keys that are not in the cache or not verified.
