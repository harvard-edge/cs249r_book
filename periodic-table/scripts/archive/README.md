# Iteration Archive

These scripts were used during the 100-round LLM-driven iteration loop that
produced v0.2 of the Periodic Table of ML Systems. They are preserved here
for reproducibility and research provenance — they are **not** part of the
active build pipeline.

The current source of truth is [`periodic-table/table.yml`](../../table.yml).
Both the standalone `index.html` and the StaffML React data file are
regenerated from it via `make all` (see `../../Makefile`).

## Shell scripts

- `iterate.sh` — autonomous improvement loop. Each iteration asks Claude to
  pick one improvement from a priority list (fix misplacement, add missing
  element, fix broken bond, improve a `whyHere`, etc.) and edit `index.html`.
- `debate.sh` / `debate-continue.sh` / `run_100_rounds.sh` — panel-debate
  scripts that ask Gemini to simulate a 5-expert debate (Patterson, Lattner,
  Dean, Shannon, Mendeleev) about the table's structure. Output lands in
  `debate-log.md`.

## Python helpers

- `append_log.py`, `update_log.py` — append entries to `iteration-log.md`.
- `get_elements.py` — extract the `const elements = [...]` array from
  `index.html` for inspection.
- `patch_informal.py`, `patch_website.py` — one-shot find-and-replace scripts
  to apply a batch of changes to `index.html`.
- `run_claude_loop.py` — subprocess-driven iteration loop that writes to
  `refinement-log.md`.
- `run_iterations.py`, `run_iterations_13.py`, `run_iterations_16_20.py` —
  one-shot scripts that hardcode a 69-element snapshot of the table and
  perform a full `re.sub` replacement. These are stale (the current table
  has 90 elements) and are preserved only as a historical reference.

## Important caveats

The hardcoded paths in these scripts are **relative to the repo root**, not
to this archive directory. If you want to re-run any of them, run from the
repo root:

```bash
python3 periodic-table/scripts/archive/get_elements.py
```

Or copy the script back to where it originally lived and run from there. The
scripts are verbatim historical artifacts — they have not been modified to
work from their archived location.
