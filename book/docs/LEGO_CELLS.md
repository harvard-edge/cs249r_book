# LEGO Cell Contract

Inline `{python}` cells in MLSysBook chapters follow a **LEGO** pattern: a small
class computes scenario values once, produces formatted `*_str` fields, and prose
references them with `` `{python} Class.field_str` ``.

This document defines how to author and review LEGO cells. It complements the
fmt/notation audit lane (`book/tools/audit/fmt/README.md`), the agent verify
playbook (`.claude/rules/lego-verify.md`), and pre-commit checks
(`lego-dead-code`, `./book/binder check math --scope canonical`).

## Core rule

**One cell ≈ one narrative anchor** — a callout, table, or tight paragraph
cluster. Place the cell **immediately above** the first `{python}` reference that
uses its output values.

**Chapter-anchor exception:** When one scenario's numbers must stay consistent
across a problem callout, a later walkthrough, and a summary bullet, mark the
header with `# │ Scope: chapter-anchor` and a one-line rationale. Focal verify
then allows multi-section reference span; render and cell-exec gates still apply.
This is not the unrelated mega-class anti-pattern (different output values in
disconnected narratives hundreds of lines apart).

## Cell contract

Each LEGO cell should include:

1. **Header comment** — Context (section/callout ID), Goal, Show, and How.
2. **Class name** — Scenario-specific (`Gpt4ClusterMtbf`), not generic (`Calc`).
3. **Four blocks** (when applicable): LOAD → EXECUTE → GUARD → OUTPUT.
4. **Output values** — Formatted strings (`*_str`) via typed helpers such as `fmt_qty()`,
   `fmt_usd()`, `fmt_percent()`, and `fmt_multiple()`. **Units and fixed glyphs
   live in OUTPUT only**; prose must not repeat units, percent signs, currency
   symbols, or multiplier glyphs after `` `{python} *_str` ``. Multiplier outputs
   use a semantic `mult` token, for example `speedup_mult_str` or
   `speedup_range_mult_str`.

Example shape:

````markdown
```{python}
#| echo: false
# ┌── LEGO ───────────────────────────────────────────────
# │ Goal: A100 ridge point for the latency callout below.
# │ Show: The formatted ridge point consumed by the following prose.
# │ How: Read the registry value and format it for inline prose.
class A100RidgeExample:
    ridge = Hardware.Cloud.A100.ridge_fp16
    ridge_str = fmt_int(round(ridge.magnitude), commas=False)
```

The A100 ridge is `{python} A100RidgeExample.ridge_str` FLOP/byte.
````

## Do

- Keep **≤ 5–8 output values** per cell when possible.
- Use **registry paths** for shared specs — `Hardware.Cloud.H100`, `Models.Vision.ResNet50`,
  `Systems.Reliability.Gpu.mttf_hours`, `Literature.Training.MfuHigh`, and
  `mlsysim.physics.calc_*` for architecture formulas. Reserve `mlsysim.core.units`
  for physics/units only (`HOURS_PER_DAY`, `BYTES_FP16`, latency stack).
- Put `#| echo: false` as the **first line** after ` ```{python}` (required by
  `book-check-code`).
- Use **`fmt_int(round(x))`** for computed integers; **`precision=0`** only when
  the value is already integer-like at the source.

## Don't

- **Mega-classes** — one class whose output values appear in multiple distant sections
  or unrelated callouts (anti-pattern: `TrainingDimensions` spanning forward-pass
  prose and a wave-quantization table hundreds of lines apart). **Exception:**
  `# │ Scope: chapter-anchor` when the same scenario thread intentionally reuses
  the same output values (KWS case-study targets, a lighthouse profile + takeaway,
  GPT-3 household-year anchor, build/buy TCO + summary).
- **Cross-cell reads** — a later cell referencing `OtherClass.some_str` without
  redefining inputs in the same cell (hidden exec-order dependency).
- **Duplicate classes** — two cells for the same story (e.g. separate table and
  prose calcs for one latency budget); merge or split by narrative beat, not by
  output type.
- **Kitchen-sink output sections** — a cell emits ten fields used once each across
  the chapter; split by callout instead.
- **Legacy constant output names** — do not mirror removed `constants.py`
  symbols (`H100_FLOPS_FP16_TENSOR_val_str`, `GPUS_PER_HOST_str`, …). Use
  scenario-descriptive names (`h100_peak_fp16_val_str`, `dgx_h100_gpus_per_node_str`)
  with registry paths on the RHS. `./binder check registry --scope sources` enforces this.
- **Hardcoded walkthrough operands** — in callout **Problem** / **Setup** / **Step** prose that
  already uses ``{python} Class.field_str``, do not type scenario inputs or intermediate
  numbers (`100 GPUs`, `70B × 2`, `10×`, `/365`, `\$2/GPU-hour`). Provide setup inputs,
  operands, rates, and multipliers from the cell.
  `./binder check code --scope lego-prose-literals` flags common cases.

**Scope boundary (judgment):** The gate targets *computational* callouts—where a
LEGO cell derives ``{python} *_str`` from scenario inputs. Pure narrative
(``100--1,000× more expensive than arithmetic``), teaching asides, and footnotes
stay literal. If a number is an input to or step in worked math, compute it in the cell.

## Review checklist

When editing or auditing a chapter:

| Check | Question |
|-------|----------|
| Locality | Is the cell within ~50–100 lines of the first ref? |
| Span | Do all output values appear in the same callout / `##` section? |
| Coupling | Does any other cell read this class's attributes? |
| Dead code | Does `lego-dead-code` report unused output values? |
| Walkthrough literals | Does `lego-prose-literals` pass on touched callouts? |
| Fmt | Do prose preview + canonical audit pass (`fmt/` workflow)? |

## Migration (existing chapters)

Do **not** big-bang refactor large chapters. When a file is already open for
copyedit, fact-check, or fmt audit:

1. Split the mega-class at callout boundaries.
2. Rename classes to match the narrative beat.
3. Inline duplicated constants from `mlsysim` where it reduces drift.
4. Re-run `fmt/audit_prose.py` and `./book/binder check math --scope canonical` on that chapter.

Future optional lints (`lego-locality`, `lego-span`, `lego-cross-ref`) may
automate the checklist; until then, apply this contract in review.

## Related

- `book/tools/audit/fmt/README.md` — spurious `.0` / fmt precision workflow
- `./book/binder check math --scope canonical` — static fmt and suffix lint
- `./book/binder check code --scope lego-prose-literals` — walkthrough prose must not hardcode computed operands
- `./book/binder check refs --scope inline-python` — chapter exec validation
- `book/tools/audit/book_check_registry_sources.py` — legacy alias / constant import gate
- `mlsysim/tests/test_constants_allowlist.py` — CI lock on `constants.py` surface

## Audit runbook (last-minute cleanup)

Use this pass before chapter sign-off or pre-push `/precheck`.

### Phase 0 — Inventory

```bash
export PYTHONPATH=mlsysim
python3 book/tools/audit/lego_focal_verify.py book/quarto/contents/vol1 book/quarto/contents/vol2
python3 book/tools/audit/fmt/audit_fmt_usage.py --root book/quarto/contents
python3 book/tools/audit/fmt/fmt_prose_contract.py --root book/quarto/contents
```

Rank chapters by focal-verify failures + `{python}` ref density. Ledger:
`book/tools/audit/artifacts/lego_audit_ranked_chapters.md`.

### Phase 1 — Gates (per chapter or corpus)

```bash
CH=book/quarto/contents/vol1/introduction/introduction.qmd
./book/binder check code --scope lego-dead-code --path "$CH"
./book/binder check code --scope lego-prose-units --path "$CH"
python3 book/tools/audit/book_check_lego_prose_units.py "$CH"
python3 book/tools/audit/fmt/fmt_prose_contract.py "$CH"
python3 book/tools/audit/audit_math_canonical.py "$CH"
./book/binder check math --scope canonical --path "$CH"
python3 book/tools/scripts/maintenance/validate_inline_refs.py --path "$CH"
```

**Prose-unit contract (two lanes):** `lego-prose-units` / `book_check_lego_prose_units.py`
flags domain glyphs (`FLOP/byte`, `TFLOP/s`, `g/kWh`, `GB`, …) immediately after a
**closed** `` `{python} *_str` `` output. `fmt_prose_contract.py` flags `%`, `$`, scale,
and `×` duplication. Open `fmt()` outputs intentionally keep units in prose.
See `book/tools/audit/artifacts/lego_closed_prose_audit.md`.

### Phase 2 — Naming contract

Every `*_str` output must match its formatter (see `.claude/rules/lego-units.md`):

- **Closed-fixed:** `*_w_str`, `*_gb_per_s_str` → typed fmt with pinned `unit=`
- **Open:** generic `*_str` → `fmt()` and prose supplies unit
- **Scale:** `*_b_str` → `fmt_params(..., scale="B")`; `*_m_str` → `fmt_count(..., scale="M")`
- **Avoid false tokens:** `top_k_neighbors_str` not `top_k_str` when K is a dimension

### Phase 3 — Prose preview (after each fix batch)

```bash
PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_prose.py "$CH" --flagged-only
```

**Precision guard triage** (when exec fails with `Formatting Precision Error`):

| Error | Fix |
|-------|-----|
| `formatted as '0'` | Raise `precision` or change display unit |
| `not integer-like but precision=0` | Use `precision>=1` |
| `integer-like but precision=1 … '10.0'` | Use `precision=0` or omit `precision=` |

For **`fmt_arithmetic_intensity`**, prefer **no explicit `precision=`** (auto:
integer-like → 0, fractional → 1). Pass explicit precision only for pinned
scenario literals (e.g. 300 FLOP/byte). **`7.0 FLOP/byte`** is a smell on exact
integers; **`153.0 FLOP/byte`** is normal for fractional ridge points at
precision 1. Agent copy: `.claude/rules/lego-verify.md`.

**Corpus exec sweep** (all chapter QMDs, shared namespace):

```bash
python3 book/tools/audit/chapter_html_verify.py --vol1 training   # build + full lane
python3 book/tools/audit/chapter_html_verify.py --report
PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_lego_html.py
```

Do not commit truncated `book/quarto/config/_quarto-html-vol*.yml` after binder
builds — restore from git if the render list shrinks.

### Phase 4 — Per-chapter replayable verify (PASS bar)

One chapter at a time: binder HTML build, every LEGO cell, every inline ref in
rendered HTML prose, optional LLM coherence review. Writes a certificate under
`book/tools/audit/artifacts/lego_chapter_reports/`.

```bash
./book/tools/audit/verify_lego_chapter.sh vol1 introduction
./book/tools/audit/verify_lego_chapter.sh vol2 network_fabrics
```

**PASS requires:** HTML build clean; cells `N/N`; rendered prose refs `N/N` (value
+ HTML context); LLM coherence not `FAIL`. Re-run the same command after fixes.

Corpus sweep (sequential, resumable):

```bash
./book/tools/audit/run_all_lego_chapters.sh
# failures → book/tools/audit/artifacts/lego_chapter_failures.txt
# progress  → book/tools/audit/artifacts/lego_chapter_progress.md
```

### Phase 5 — HTML spot checks

```bash
python3 book/tools/audit/fmt/audit_html.py book/quarto/_build/html-audit/vol1/introduction.html
```

Spot-check certificates for substituted QMD→HTML prose; no literal `{python}`.

### Phase 6 — Capstone

```bash
./book/tools/audit/verify_lego_pipeline.sh   # or /precheck before push
```

Sign-off template: `book/tools/audit/artifacts/lego_audit_signoff.md`.

### Follow-up: distance / length outputs (P5)

Some cells still use **open** `fmt()` for meters while prose supplies the unit:

```python
distance_str = fmt(distance_m, precision=2, commas=False)
# prose: ... `{python} EdgeLatencyDistance.distance_str` meters ...
```

This pass deferred a corpus-wide decision on:

- **Comma rule** — no commas for small decimals (2.8 m, 3.33 m); commas when displaying km or values ≥ 1,000 m.
- **Closed vs open** — compare `LightLatency.distance_str` (`fmt_qty`, closed km) vs `BrakingDistance` / `EdgeLatencyDistance` (open m + prose “meters”).
- **`fmt_length` helper** — optional typed formatter (pinned `unit=` or auto m/km) if the pattern repeats; see P5 in `lego_audit_signoff.md`.
