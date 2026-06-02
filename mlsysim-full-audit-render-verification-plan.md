# MLSysIM Full Audit, Consumer Update, and Render Verification Plan

**Status:** Execution and audit record. Commit in logical chunks; push only in Phase 9 after local gates, merge, and final smoke checks are clean.
**Working tree:** `/Users/VJ/GitHub/MLSysBook-fmt-fix` on branch `fmt-fix`
**Created:** 2026-06-01
**Primary objective:** Make every MLSysIM-backed value, every LEGO cell, every corresponding prose reference, and every downstream consumer use a clean single-source-of-truth path, then verify the book with full HTML and PDF renders before any push.

This plan supersedes ad hoc cleanup ordering for the remaining work. It does not replace the narrower unit-hardening and SSOT plans; it coordinates them into one execution path.

---

## Overnight Execution Plan - 2026-06-02

This is the current overnight runbook after the `fmt-fix` branch absorbed the
margin-figure work. The order is deliberate:

1. prove the source layer and MLSysIM consumers are coherent;
2. prove every LEGO cell and prose reference is semantically correct;
3. prove the rendered HTML and PDF expose the expected values without leaks;
4. only then audit the margin SVG layout with PDF debug boxes.

Do not start the margin-layout phase until the material, math, and rendered
LEGO/prose checks are green. Layout work is allowed to be iterative and visual;
numeric/prose correctness is not.

### Nightly non-negotiables

- Work only in `/Users/VJ/GitHub/MLSysBook-fmt-fix` on `fmt-fix`.
- Treat `/Users/VJ/GitHub/MLSysBook` as the protected local `dev` reference.
  Do not edit it.
- Do not push to origin during verification. Final push happens only in Phase 9
  after all local gates, merge checks, and final smoke checks are clean.
- Commit in logical chunks when a stage is finished and verified. Stage explicit
  files only.
- The coordinator owns MLSysIM, shared formatters, shared checks, and rules.
  Chapter agents may edit QMD/prose/layout but must report missing shared
  values instead of inventing local registries.
- If a shared MLSysIM value, formatter, or audit rule changes, rerun the
  affected QMD checks and then the whole-book checks. A single-source-of-truth
  change invalidates earlier rendered evidence.
- Red margin/debug boxes are a temporary PDF inspection aid. `\MarginDebugtrue`
  must never be committed or left on for final production renders.

### Current state checkpoint

Before executing anything, record:

```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix
pwd
git rev-parse --show-toplevel
git branch --show-current
git status --short --branch
git log --oneline -12
git worktree list
```

Expected branch context at this writing:

- worktree: `/Users/VJ/GitHub/MLSysBook-fmt-fix`
- branch: `fmt-fix`
- status: clean
- branch is ahead of `origin/fmt-fix`
- recent history includes the margin-figure merge and `c9e41d3451 Align KV
  cache capacity example units`

If the tree is dirty, classify every file before continuing:

- source/docs/rules edits that are part of the plan;
- generated build artifacts that should be ignored or removed;
- temporary debug-box changes that must be restored;
- user/editorial changes that must be preserved.

### Phase 1 - Source and consumer inventory

Goal: prove the repo has no stale MLSysIM paths or copied constants before
rendering.

Run the broad MLSysIM consumer inventory:

```bash
rg -n "mlsysim|from mlsysim|import mlsysim|Hardware\.|Models\.|Systems\.|Infrastructure\.|Literature\.|Scenarios\.|Ops\." \
  book mlsysim slides labs site .claude \
  --glob '!**/_build/**' \
  --glob '!**/.quarto/**' \
  --glob '!**/__pycache__/**'
```

Check these surfaces, even if the search returns only a few hits:

- `book/quarto/contents/vol1/**/*.qmd`
- `book/quarto/contents/vol2/**/*.qmd`
- `book/quarto` support scripts and configs
- `mlsysim/mlsysim/**`
- `mlsysim/tests/**`
- `mlsysim/docs/**`, `mlsysim/tutorial/**`, `mlsysim/examples/**`
- `slides/**`
- `labs/**`
- `site/**`
- `.claude/rules/**`

For every hit:

- current semantic path is used;
- no old migrated names remain;
- no copied H100/A100/model/grid/storage/price value survives where an MLSysIM
  import should be used;
- no `BOOK_*`, `prov:book`, `MLSysBook`, `Volume I`, `Volume II`, or "worked
  example" naming exists inside `mlsysim/mlsysim`;
- `Literature.*` is used only for directly cited/report field figures, not as a
  generic provenance bucket.

Run the source/provenance gates:

```bash
PYTHONPATH=mlsysim pytest mlsysim/tests -o addopts=
PYTHONPATH=mlsysim python3 -m mlsysim.tools.audit_provenance --scope all --strict
python3 book/tools/audit/book_check_registry_sources.py
python3 book/tools/audit/book_check_lego_scenario_inputs.py book/quarto/contents --summary
```

If any of these fail because the margin merge restored stale paths or copied
values, fix those first and commit the source-layer correction before touching
chapter layout.

### Phase 2 - Static LEGO, formatter, and prose gates

Goal: every LEGO cell follows the Load, Execute, Guard, Output contract before
we trust rendered output.

Run:

```bash
python3 book/tools/scripts/lint_lego_units.py \
  --fail-on warning \
  --baseline book/tools/audit/lego_units_baseline.json

python3 book/tools/audit/book_check_lego_load_pint.py book/quarto/contents
python3 book/tools/audit/book_check_lego_prose_units.py book/quarto/contents
python3 book/tools/audit/book_check_lego_quantity_flow.py book/quarto/contents --summary
PYTHONPATH=mlsysim python3 book/tools/audit/lego_focal_verify.py book/quarto/contents

PYTHONPATH=mlsysim python3 book/tools/audit/fmt/fmt_prose_contract.py --root book/quarto/contents
PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_prose_semantics.py --root book/quarto/contents
```

Audit standards:

- unit-bearing outputs use domain formatters;
- fixed-unit names force the unit they claim, such as `_kwh_str`,
  `_gb_s_str`, `_tflop_s_str`, `_hours_str`;
- auto-scaled outputs have generic names;
- plain `fmt()` is used only for dimensionless values or explicitly open
  strings;
- duration-like values use `fmt_time` or a tighter domain helper when the prose
  needs unit validation;
- rates such as `tokens/s`, `samples/s`, QPS, requests/s use semantic rate
  helpers rather than manual suffix strings;
- `fmt_multiple` and prose do not combine into `7x \times` or `7x times`;
- compound Pint units are parenthesized for readability, for example
  `1.9 * (TB / second)` and `rate / (USD / GB)`;
- no LEGO cell depends on another LEGO class for a reusable value. Promote the
  shared value to MLSysIM or a shared helper.

If a recurring problem appears, add or tighten the checker before doing a broad
manual sweep. Checks should be written backwards from real drift incidents.

### Phase 3 - Chapter-by-chapter semantic read

Goal: read the book with the values substituted, not just with syntax passing.

Process chapters in book order. For each QMD with LEGO:

1. list every LEGO class and every prose reference to that class;
2. verify `LOAD` values come from MLSysIM when they are reusable facts;
3. verify remaining local literals are visibly local scenario assumptions;
4. verify `EXECUTE` keeps Pint quantities attached until output;
5. verify guards check units, dimensions, and the values prose depends on;
6. verify output names assert the rendered unit or stay generic if auto-scaled;
7. read the substituted prose for logic, grammar, and math coherence;
8. simplify repeated figure/table references when adjacent mentions do not add
   meaning;
9. merge or expand floating one-sentence paragraphs unless they are intentional
   transitions;
10. replace awkward standalone display equations with inline prose or aligned
    math when the current layout reads poorly.

Use the machine-assisted prose preview where useful:

```bash
PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_prose.py <chapter.qmd>
PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_prose_semantics.py <chapter.qmd> -v
```

High-risk chapters that must receive a manual read after any source change:

- `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd`
- `book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd`
- `book/quarto/contents/vol2/data_storage/data_storage.qmd`
- `book/quarto/contents/vol2/distributed_training/distributed_training.qmd`
- `book/quarto/contents/vol2/inference/inference.qmd`
- `book/quarto/contents/vol2/ops_scale/ops_scale.qmd`
- `book/quarto/contents/vol1/model_serving/model_serving.qmd`
- `book/quarto/contents/vol1/benchmarking/benchmarking.qmd`
- `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd`

Commit semantic/prose fixes in volume-sized or topic-sized batches, not as one
giant commit.

### Phase 4 - Full HTML render and rendered LEGO exposure

Goal: prove every inline Python value exposed in QMD appears in real rendered
HTML with the expected value.

Build both volumes:

```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix
./book/binder build html --vol1 -v
./book/binder build html --vol2 -v
```

If binder routing is unstable, use the direct Quarto fallback:

```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix/book/quarto
ln -sf config/_quarto-html-vol1.yml _quarto.yml
MPLBACKEND=Agg quarto render --to html -M jupyter:python3

ln -sf config/_quarto-html-vol2.yml _quarto.yml
MPLBACKEND=Agg quarto render --to html -M jupyter:python3
```

Post-render checks:

```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix
rg '\{python\}' book/quarto/_build/html-vol1 book/quarto/_build/html-vol2 --glob '*.html'
rg '\?@|Traceback|ImportError|NameError|AttributeError|ModuleNotFoundError' \
  book/quarto/_build/html-vol1 book/quarto/_build/html-vol2 --glob '*.html'

cd book/quarto
python3 scripts/verify_rendered_xrefs.py

cd /Users/VJ/GitHub/MLSysBook-fmt-fix
PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_lego_html.py \
  --report book/quarto/_build/html-audit/lego_html_verify_report.json
PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_html.py \
  book/quarto/_build/html-vol1 book/quarto/_build/html-vol2
```

The `audit_lego_html.py` gate is the explicit rendered-value exposure check:
it executes the LEGO cells, resolves `{python} Class.attr` references, and
verifies the rendered HTML contains those values. A raw render success is not
enough.

If `audit_lego_html.py` finds a missing rendered value:

- first check whether the value moved because the prose was rewritten;
- then check whether the formatter output changed;
- then inspect the HTML around the paragraph;
- fix the source or audit rule, rebuild the affected volume, and rerun the
  full rendered-value audit.

### Phase 5 - Full PDF render and LaTeX/PDF text checks

Goal: prove the production PDFs build without LaTeX errors and without rendered
reference/value leaks.

Build both volumes:

```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix
./book/binder build pdf --vol1 -v
./book/binder build pdf --vol2 -v
```

Direct Quarto fallback:

```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix/book/quarto
ln -sf config/_quarto-pdf-vol1.yml _quarto.yml
ln -sf index-vol1.qmd index.qmd
MPLBACKEND=Agg quarto render --to titlepage-pdf -M jupyter:python3

ln -sf config/_quarto-pdf-vol2.yml _quarto.yml
ln -sf index-vol2.qmd index.qmd
MPLBACKEND=Agg quarto render --to titlepage-pdf -M jupyter:python3
```

Never use `--to pdf`; use `titlepage-pdf` so the header includes are loaded.

Post-render checks:

```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix
ls -lh book/quarto/_build/pdf-vol1/*.pdf book/quarto/_build/pdf-vol2/*.pdf
rg '^!|Undefined control sequence|LaTeX Error|Emergency stop|Runaway argument|Missing \$ inserted|Traceback|ImportError|NameError|AttributeError|ModuleNotFoundError' \
  book/quarto/*.log book/quarto/_build/pdf-vol1 book/quarto/_build/pdf-vol2

./book/binder check pdf --vol1
./book/binder check pdf --vol2
./book/binder layout check book/quarto/_build/pdf-vol1/Machine-Learning-Systems-Vol1.pdf --skip-frontmatter
./book/binder layout check book/quarto/_build/pdf-vol2/Machine-Learning-Systems-Vol2.pdf --skip-frontmatter
./book/binder layout margins book/quarto/_build/pdf-vol1/Machine-Learning-Systems-Vol1.pdf --csv \
  > /private/tmp/mlsysbook-margin-overflow-vol1.csv
./book/binder layout margins book/quarto/_build/pdf-vol2/Machine-Learning-Systems-Vol2.pdf --csv \
  > /private/tmp/mlsysbook-margin-overflow-vol2.csv
```

`layout margins` catches margin content that runs off the page. It does not
catch figure-footnote overlap inside the margin. The visual phase below is still
required.

### Phase 6 - Rules and consumer documentation update

Goal: before layout work, encode only the durable lessons that should guide
future Claude/Codex work.

Update `.claude/rules` only with general rules that apply across future work:

- MLSysIM semantic homes and provenance separation;
- LEGO Load, Execute, Guard, Output discipline;
- fixed-unit output-name assertions;
- formatter ownership of units, glyphs, ratios, and durations;
- slides/labs/docs must source reusable measurable claims from MLSysIM;
- quantitative margin figures must be generated from MLSysIM-backed data or a
  recorded audited source;
- PDF margin overlap requires rendered inspection, not grep proximity.

Also check MLSysIM documentation, examples, tutorial, slides, labs, and site
copy for stale descriptions introduced by the registry/formatter changes.

Run after updates:

```bash
rg -n "Literature\.Scaling|Literature\.Overheads|Literature\.Energy|Literature\.Sustainability|BOOK_|prov:book|MLSysBook|Volume I|Volume II|worked example" \
  book mlsysim slides labs site .claude \
  --glob '!book/quarto/_build/**'
pre-commit run --all-files
```

Commit docs/rules/consumer updates separately from QMD numerical changes.

### Phase 7 - Margin SVG PDF layout audit

Goal: inspect margin figures in rendered PDF pages with alignment guides on and
fix only real layout problems: off-page content, figure-footnote overlap,
figure-caption overlap, unreadable labels, or margin content outside the debug
frame.

This phase starts only after Phases 1-6 are green.

#### 7.1 Prepare the layout ledger

Maintain a live ledger while working:

```text
book/tools/audit/artifacts/margin_layout_audit.md
book/tools/audit/artifacts/margin_layout_audit.json
```

Each finding should record:

- volume and chapter;
- QMD path and source line;
- asset path;
- PDF path and page label;
- screenshot path if captured;
- status: PASS, MOVE, RESIZE, PROMOTE_TO_BODY, REMOVE, or REGENERATE;
- reason;
- exact fix;
- validation command after fix.

Also maintain a learning note for future skill/rule extraction:

```text
mlsysbook-layout-pdf-skill-notes.md
```

Capture rules learned from actual fixes, not guesses.

#### 7.2 Enable debug boxes temporarily

In `book/quarto/tex/header-includes.tex`, change:

```tex
\MarginDebugfalse
```

to:

```tex
\MarginDebugtrue
```

Do not commit this debug toggle. Record that the tree is intentionally dirty
while inspecting layout.

#### 7.3 Inventory margin chapters

Current chapters with `.column-margin` content:

Vol I:

```text
introduction
ml_systems
ml_workflow
data_engineering
nn_computation
nn_architectures
frameworks
training
data_selection
model_compression
hw_acceleration
benchmarking
model_serving
ml_ops
responsible_engr
conclusion
```

Vol II:

```text
introduction
compute_infrastructure
network_fabrics
data_storage
distributed_training
collective_communication
fault_tolerance
fleet_orchestration
performance_engineering
inference
edge_intelligence
ops_scale
security_privacy
robust_ai
sustainable_ai
responsible_ai
conclusion
```

Regenerate the current list before execution:

```bash
rg -l "column-margin" book/quarto/contents -g '*.qmd' \
  > /private/tmp/mlsysbook-margin-chapters.txt
rg -n "column-margin|images/svg|marginfigure|marginnote" book/quarto/contents -g '*.qmd' \
  > /private/tmp/mlsysbook-margin-source-inventory.txt
```

#### 7.4 Build chapter PDFs in parallel batches

Use chapter-level PDF audit builds. They archive PDF and TeX artifacts under
`book/quarto/_build/pdf-audit/` and update the chapter audit ledger:

```bash
./book/binder audit chapter-pdf --vol1 training
./book/binder audit chapter-pdf --vol2 sustainable_ai
```

For a direct build when the audit wrapper is not needed:

```bash
./book/binder build pdf training --vol1 -v
./book/binder build pdf sustainable_ai --vol2 -v
```

Parallelization plan:

- Coordinator: owns debug toggle, shared fixes, final verification, and ledger
  merge.
- Agent A: Vol I chapters 1-4.
- Agent B: Vol I chapters 5-8.
- Agent C: Vol I chapters 9-12.
- Agent D: Vol I chapters 13-16.
- Agent E: Vol II chapters 1-5.
- Agent F: Vol II chapters 6-10.
- Agent G: Vol II chapters 11-17.

Each agent must:

1. build its chapter PDF with debug boxes on;
2. run `binder layout margins <pdf> --csv`;
3. locate each margin figure page;
4. rasterize pages with margin figures;
5. visually inspect the margin band;
6. fix only its assigned QMD/source assets;
7. rebuild and recheck after every fix;
8. update the ledger with findings and fixes.

Useful page raster command:

```bash
pdftoppm -png -r 250 -f <page> -l <page> <pdf> /private/tmp/mlsysbook-margin-<chapter>-p
```

Visual checklist for each margin figure:

- figure is inside the showframe margin column;
- no overlap with footnotes, sidenotes, body-figure captions, or another margin
  figure;
- caption wraps inside the margin;
- labels remain legible at print size;
- figure stays above the footer and below the header;
- figure is near the prose it supports but not forced into a crowded margin
  slot;
- quantitative labels trace to MLSysIM or an audited source;
- the figure is not redundant with an adjacent body table/figure.

#### 7.5 Fix taxonomy

Use the least invasive fix that preserves the teaching point:

- MOVE: move the `.column-margin` block to a footnote-clear anchor in the same
  section.
- RESIZE: adjust width/height only when the figure is inside the margin but too
  cramped.
- REGENERATE: rerun the margin SVG generator when labels or values are wrong.
- PROMOTE_TO_BODY: convert to a normal body figure when the reader must inspect
  details that are too dense for the margin.
- REMOVE: delete the margin figure when it duplicates nearby prose/table/body
  figure or cannot fit without harming the page.

Do not solve a collision by deleting a necessary footnote without a prose
reason. Footnotes and margin figures share the margin; the figure usually moves.

#### 7.6 Restore production mode and final layout gates

When all chapter-level layout findings are resolved, restore:

```tex
\MarginDebugfalse
```

Then rebuild final production PDFs:

```bash
./book/binder build pdf --vol1 -v
./book/binder build pdf --vol2 -v
./book/binder layout margins book/quarto/_build/pdf-vol1/Machine-Learning-Systems-Vol1.pdf --csv \
  > /private/tmp/mlsysbook-margin-overflow-final-vol1.csv
./book/binder layout margins book/quarto/_build/pdf-vol2/Machine-Learning-Systems-Vol2.pdf --csv \
  > /private/tmp/mlsysbook-margin-overflow-final-vol2.csv
```

Final layout acceptance:

- debug boxes are off;
- no `layout margins` overflow findings remain unless explicitly accepted and
  documented;
- visually audited collision ledger has no open MOVE/RESIZE/PROMOTE/REMOVE
  items;
- full PDFs have no LaTeX hard errors;
- final PDFs are built from clean production settings.

### Phase 8 - Final verification and sign-off package

Run the complete gate sequence after all substantive and layout changes:

```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix
git status --short --branch

PYTHONPATH=mlsysim pytest mlsysim/tests -o addopts=
PYTHONPATH=mlsysim python3 -m mlsysim.tools.audit_provenance --scope all --strict
python3 book/tools/scripts/lint_lego_units.py --fail-on warning --baseline book/tools/audit/lego_units_baseline.json
python3 book/tools/audit/book_check_lego_load_pint.py book/quarto/contents
python3 book/tools/audit/book_check_lego_prose_units.py book/quarto/contents
python3 book/tools/audit/book_check_lego_quantity_flow.py book/quarto/contents --summary
PYTHONPATH=mlsysim python3 book/tools/audit/lego_focal_verify.py book/quarto/contents
PYTHONPATH=mlsysim python3 book/tools/audit/fmt/fmt_prose_contract.py --root book/quarto/contents
PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_prose_semantics.py --root book/quarto/contents
pre-commit run --all-files

./book/binder build html --vol1 -v
./book/binder build html --vol2 -v
PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_lego_html.py \
  --report book/quarto/_build/html-audit/lego_html_verify_report.json

./book/binder build pdf --vol1 -v
./book/binder build pdf --vol2 -v
./book/binder check pdf --vol1
./book/binder check pdf --vol2
./book/binder layout margins book/quarto/_build/pdf-vol1/Machine-Learning-Systems-Vol1.pdf --csv
./book/binder layout margins book/quarto/_build/pdf-vol2/Machine-Learning-Systems-Vol2.pdf --csv
```

Prepare the sign-off summary:

- latest commit hash;
- list of commits made during the night;
- source/MLSysIM changes;
- QMD/prose changes;
- docs/slides/labs/site/rules changes;
- HTML render result;
- rendered LEGO exposure audit result;
- PDF render result;
- margin-layout audit result;
- any accepted residual risks;
- confirmation that the branch is ready for final `dev` promotion.

Only after this package is clean should the branch be considered ready for final
`dev` merge and push.

### Phase 9 - Merge to `dev`, push, and monitor GitHub workflows

Goal: promote the verified work to `dev`, push once, and watch the CI runs that
the push actually triggers.

Do not manually dispatch validate workflows as a substitute for the push-driven
signal. The push to `dev` should create the real integration runs. Manual
dispatch is only a fallback if a path filter unexpectedly prevents a required
workflow from starting.

#### 9.1 Final branch cleanliness check

In the task worktree:

```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix
git status --short --branch
git log --oneline origin/dev..HEAD
```

All intended work must be committed. There must be no temporary debug-box
change, generated audit JSON, or uncommitted QMD/rules/docs edit.

#### 9.2 Sync local `dev`

Use the protected main checkout only for `dev` sync and final merge:

```bash
cd /Users/VJ/GitHub/MLSysBook
pwd
git rev-parse --show-toplevel
git branch --show-current
git status --short --branch
git pull --ff-only origin dev
```

If `/Users/VJ/GitHub/MLSysBook` is dirty, stop and report. Do not overwrite the
main checkout.

#### 9.3 Rebase/merge current `dev` into `fmt-fix` if needed

If the pull advanced `dev`, bring it back into the task branch before promotion:

```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix
git fetch origin dev
git merge --no-ff origin/dev
```

Conflict policy:

- preserve Pint-safe, unit-hardened LEGO cells;
- preserve MLSysIM semantic-home decisions;
- preserve margin-layout fixes unless the incoming `dev` text is newer and
  compatible;
- never resolve a conflict by deleting checks, guards, or provenance;
- never revive old `Literature.*` catch-all paths or book-specific MLSysIM
  names.

After any conflict resolution, rerun Phases 1-8 as needed. At minimum, rerun all
static gates, full HTML, rendered LEGO exposure audit, full PDFs, and final
margin overflow checks.

Commit the merge resolution on `fmt-fix` before continuing.

#### 9.4 Merge `fmt-fix` into `dev`

From the main checkout:

```bash
cd /Users/VJ/GitHub/MLSysBook
git checkout dev
git status --short --branch
git merge --no-ff fmt-fix
```

If this reports conflicts, resolve them in `dev` using the same conflict policy
above, then rerun the affected local gates from the `dev` checkout. The preferred
state is that `fmt-fix` already contains current `dev`, so this merge is clean.

After the merge, verify the final `dev` tree:

```bash
git status --short --branch
git log --oneline -8
git diff --stat origin/dev..HEAD
```

Run the final smoke gate on `dev`:

```bash
PYTHONPATH=mlsysim pytest mlsysim/tests -o addopts=
python3 book/tools/scripts/lint_lego_units.py --fail-on warning --baseline book/tools/audit/lego_units_baseline.json
python3 book/tools/audit/book_check_lego_load_pint.py book/quarto/contents
python3 book/tools/audit/book_check_lego_prose_units.py book/quarto/contents
python3 book/tools/audit/book_check_lego_quantity_flow.py book/quarto/contents --summary
PYTHONPATH=mlsysim python3 book/tools/audit/fmt/fmt_prose_contract.py --root book/quarto/contents
PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_prose_semantics.py --root book/quarto/contents
pre-commit run --all-files
```

If the exact tree differs from the final `fmt-fix` render-verified tree, rebuild
HTML/PDF on `dev` too before pushing.

#### 9.5 Push `dev`

Push only after the merge commit and final smoke gate are clean:

```bash
git push origin dev
PUSH_SHA=$(git rev-parse HEAD)
echo "$PUSH_SHA"
```

Record the pushed SHA in the sign-off notes.

#### 9.6 Monitor actual workflows from the push

List the runs created by the pushed SHA:

```bash
gh run list --branch dev --commit "$PUSH_SHA" --limit 30
```

Expected workflow families depend on touched paths:

- book changes: `📚 Book · ✅ Validate (Dev)`, followed by `📚 Book · 👁️ Preview (Dev)` after validate succeeds;
- MLSysIM changes: `🧮 MLSysIM · ✅ Validate (Dev)` and related preview/build runs if configured by path filters;
- slides changes: `📊 Slides · ✅ Validate (Dev)` and `📊 Slides · 👁️ Preview (Dev)`;
- labs changes: `🔮 Labs · ✅ Validate (Dev)` and `🔮 Labs · 👁️ Preview (Dev)`;
- site changes: `🌐 Landing Site · ✅ Validate (Dev)` and `🌐 Landing Site · 👁️ Preview (Dev)`;
- workflow-file changes: `🧪 CI Sanity` plus any changed workflow's own path-triggered validation.

Watch every run associated with the pushed SHA until it is completed:

```bash
for id in $(gh run list --branch dev --commit "$PUSH_SHA" --json databaseId --jq '.[].databaseId'); do
  gh run watch "$id" --exit-status
done
```

If a run fails:

1. inspect the failing jobs and logs;
2. fix on `dev` only if the failure is caused by this merge and the fix is
   straightforward;
3. otherwise create a small follow-up branch/worktree from `dev`;
4. rerun the relevant local gate;
5. commit, push `dev` again only when the fix is verified;
6. monitor the new pushed SHA.

Completion requires all required push-triggered workflows for the pushed SHA to
be success or an explicitly documented non-required path-filter absence.

---

## Latest Validation Checkpoint - 2026-06-01

The current pass established these additional invariants:

- Full focal LEGO verification is clean across all LEGO-bearing QMDs: every cell executed by `lego_focal_verify.py` reports `cross=0` and `issues=0`.
- QMD code no longer uses `unit_label=`. Non-Pint display-label policies now belong in typed helpers such as `fmt_memory_capacity`, `fmt_sci_flops`, `fmt_decibel`, `fmt_illuminance`, and temperature helpers.
- Shared values needed by more than one LEGO cell must move to MLSysIM. Example fixed in this pass: `Scenarios.ClinicalImaging.RetinalPhotoSize` replaces a cross-cell dependency between `BandwidthCompute` and `DeploymentEconomics`.
- The static LEGO gates passed after the cleanup:
  - `book_check_lego_quantity_flow.py`
  - `book_check_lego_load_pint.py`
  - `book_check_lego_prose_units.py`
  - `lint_lego_units.py --fail-on warning --baseline book/tools/audit/lego_units_baseline.json`
- MLSysIM provenance audit passed with `--scope all --strict`.
- Focused tests passed: `mlsysim/tests/test_fmt.py`, `test_ops_registry.py`, `test_system_registry.py`, and `test_units_registry.py` (`178 passed`).

Remaining before final push/merge readiness:

- commit completed MLSysIM, LEGO, docs, slides, and rule updates in logical chunks;
- merge local `dev` into `fmt-fix` and resolve conflicts from a clean branch baseline;
- rerun LEGO/prose/static checks after the merge;
- render full HTML/PDF after the merge;
- push only in Phase 9 after the final local gate and `dev` merge are clean.

---

## 0. Current State and Guardrails

### 0.1 Commit and push policy

The user has approved committing completed work before merging local `dev` into
`fmt-fix`. Commit only coherent chunks, stage explicit paths, and keep unrelated
dirty files out of the commit.

Never push to `origin` until all acceptance gates in this document are green and
the Phase 9 `dev` merge checks have passed.

### 0.2 Worktree policy

Work only in:

```bash
/Users/VJ/GitHub/MLSysBook-fmt-fix
```

The protected main checkout is:

```bash
/Users/VJ/GitHub/MLSysBook
```

Use the main checkout only as the local `dev` reference and merge source. Do not make risky edits there. Before editing, committing, merging, or retiring anything, verify:

```bash
pwd
git rev-parse --show-toplevel
git branch --show-current
git worktree list
```

### 0.3 Historical dirty files at plan creation

These files were the initial dirty state when this plan was drafted. Do not use
this list as current status; use `git status --short` before any commit or
merge.

```text
M  book/tools/audit/fmt/audit_fmt_usage.py
M  book/tools/audit/fmt/fmt_prose_contract.py
M  mlsysim/mlsysim/core/loader.py
M  mlsysim/mlsysim/core/provenance_catalog.py
M  mlsysim/mlsysim/literature/data/batchsize.yaml
```

Original interpretation:

- `book/tools/audit/fmt/audit_fmt_usage.py` and `book/tools/audit/fmt/fmt_prose_contract.py` began as fmt-thread WIP and are now part of the semantic formatting/audit hardening once validated.
- `mlsysim/mlsysim/core/loader.py`, `mlsysim/mlsysim/core/provenance_catalog.py`, and `mlsysim/mlsysim/literature/data/batchsize.yaml` are drafted taxonomy/provenance hardening edits. They should be validated before keeping:
  - `Literature.BatchSize` now has explicit provenance.
  - sourced registries reject bare scalars.
  - `MCCANDLISH_LARGE_BATCH_TRAINING` was added to `provenance_catalog.py`.

### 0.4 Merge timing

Do not merge `dev` first. The current branch already contains major local unit/prose/registry work. Finish the source-of-truth cleanup and consumer updates on `fmt-fix`, then merge `dev` once the branch has a clean internal baseline. After the merge, rerun every LEGO/prose/render gate because conflicts can reintroduce stale values or prose.

---

## 1. Target Standard

### 1.1 MLSysIM is the single source of truth

Every reusable measurable fact must live in the correct MLSysIM semantic registry:

- hardware specs in `Hardware.*`
- model specs in `Models.*`
- infrastructure facts in `Infrastructure.*`
- systems and composed objects in `Systems.*`
- operational policies, thresholds, and run-overhead profiles in `Ops.*`
- scenario bundles or comparison anchors in `Scenarios.*`
- directly literature-sourced field figures in `Literature.*`

Book use is not a category. Do not encode chapter names, `BOOK_*`, `MLSysBook`, `Volume I`, `Volume II`, or "worked example" into MLSysIM registry names, provenance identifiers, or descriptions.

### 1.2 Provenance is metadata, not a semantic home

Keep the split:

- `core.provenance` defines the provenance data model and source contract.
- `core.provenance_catalog` stores reusable source records.
- semantic registries hold values.
- `Literature.*` is a semantic registry only for values whose category is a cited paper/report field figure.

Do not collapse `Provenance` and `Literature`. A value can be sourced from a paper while still belonging in `Hardware`, `Systems`, `Infrastructure`, or `Scenarios`.

### 1.3 LEGO stage contract

Every LEGO calculation should follow the same contract:

| Stage | Requirement |
|-------|-------------|
| L - Load | Pull reusable specs/scenarios from MLSysIM. Local literals are allowed only for truly local assumptions and must be labeled as scenario/illustrative/budget assumptions. |
| E - Execute | Keep Pint quantities attached. Use formulas/helpers where available. Avoid early `.magnitude` extraction except at explicit guard or output boundaries. |
| G - Guard | Add checks that make the prose hard to break: dimensions, representative values, ratios, and closed output expectations. |
| O - Output | Export typed strings only. Unit-bearing prose strings use domain formatters and names that assert the rendered unit. |

### 1.4 Output names are assertions

For fixed-unit output strings, the suffix must identify the value and rendered unit:

```python
facility_energy_kwh_str = fmt_energy(facility_energy, unit=kWh)
h100_tdp_w_str = fmt_power(h100.tdp, unit=watt)
peak_flops_tflop_s_str = fmt_flop_rate(peak_flops, unit=TFLOP / second)
params_b_str = fmt_params(params, scale="B")
decode_tokens_s_str = fmt_rate(decode_rate, "tokens/s")
```

If the formatter auto-scales, the name must not claim a fixed unit:

```python
facility_energy_str = fmt_energy(facility_energy)
```

Closed strings should be referenced bare in prose. Do not repeat the unit manually:

```markdown
`{python} CarbonEstimate.total_tonnes_str`
```

not:

```markdown
`{python} CarbonEstimate.total_tonnes_str` t CO2
```

### 1.5 Domain formatter policy

Anything unit-bearing should use a domain formatter where one exists or should get one if the domain recurs:

- memory: `fmt_memory`
- bandwidth: `fmt_bandwidth`
- energy: `fmt_energy`
- power: `fmt_power`
- emissions: `fmt_emissions`
- carbon intensity: `fmt_carbon_intensity`
- latency/time durations: `fmt_time`, `fmt_latency`, or a clarified domain-specific helper
- FLOPs and FLOP rates: `fmt_flops`, `fmt_flop_rate`
- arithmetic intensity: `fmt_arithmetic_intensity`
- compute efficiency: `fmt_compute_efficiency`
- parameters: `fmt_params`
- tokens: `fmt_tokens`
- rates such as tokens/s, samples/s, QPS: `fmt_rate`
- multipliers/speedups/ratios: `fmt_multiple` or a new `fmt_ratio` if audit shows recurring ambiguity

Plain `fmt()` is acceptable for dimensionless scalars such as PUE, counts that do not need semantic suffixes, or local values whose semantics are clear. If a value has a unit or a recurring semantic type, prefer a typed formatter.

---

## 2. Phase A - Freeze and Inventory

Goal: know exactly what will be touched before changing more files.

### A1. Confirm branch and dirty state

```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix
pwd
git rev-parse --show-toplevel
git branch --show-current
git status --short
git log --oneline -5
```

Record dirty files in this plan or a progress note before edits.

### A2. Inventory all MLSysIM consumers

Build a complete file list for surfaces that may import or copy MLSysIM values:

```bash
rg -n "mlsysim|from mlsysim|import mlsysim|Hardware\\.|Models\\.|Systems\\.|Infrastructure\\.|Literature\\.|Scenarios\\." \
  book mlsysim slides labs .claude \
  --glob '!**/_build/**' \
  --glob '!**/.quarto/**' \
  --glob '!**/__pycache__/**'
```

Expected surfaces:

- `book/quarto/contents/vol1/**/*.qmd`
- `book/quarto/contents/vol2/**/*.qmd`
- `book/quarto/**` support scripts and rendered config helpers
- `slides/**`
- `labs/**`
- `mlsysim/docs/**`
- `mlsysim/tutorials/**` or examples, if present
- `mlsysim/mlsysim/**`
- `mlsysim/tests/**`
- `.claude/rules/**`
- `.claude/docs/**`

### A3. Inventory every LEGO cell

Generate a machine-readable list of every QMD with LEGO cells and every exported string:

```bash
python3 book/tools/audit/fmt/audit_lego_cells.py book/quarto/contents --json \
  > /private/tmp/mlsysbook-lego-cells.json
```

If the existing audit tool is incomplete or unstable, write a small read-only scanner first. The output must include:

- file
- class/cell name
- line number range
- imports
- `LOAD` bindings
- `EXECUTE` assignments
- `GUARD` checks
- `OUTPUT` assignments ending in `_str`, `_math`, `_val`, `_tbl`, etc.
- inline prose references to those outputs

Do not start broad rewrites until this inventory exists.

---

## 3. Phase B - MLSysIM Taxonomy and Provenance Hardening

Goal: make the source registry honest before updating prose and downstream consumers.

### B1. Validate the drafted batch-size provenance change

Current draft:

- `Literature.BatchSize` gets explicit `MCCANDLISH_LARGE_BATCH_TRAINING` provenance.
- `load_sourced_registry` rejects bare scalars.

Validation:

```bash
PYTHONPATH=mlsysim pytest mlsysim/tests -o addopts=
PYTHONPATH=mlsysim python -m mlsysim.tools.audit_provenance --scope all --strict
```

Decision:

- Keep if all sourced registries can comply with `{value, provenance}`.
- If failures show legitimate non-sourced registries are using the sourced loader, split the loader contract rather than weakening sourced registry requirements.

### B2. Audit `Literature.*` for non-literature values

Run an introspection script to list every `Literature` leaf with provenance kind and description.

Known suspected migrations:

| Current path | Issue | Target direction |
|--------------|-------|------------------|
| `Literature.Scaling.*` | mixed: some values are convention tiers, one is a cited/empirical 8192-GPU anchor | do not move to a vague `Systems.Scaling` bucket. Either derive scaling from `Systems.Fleet` + engine, or create explicit scale-profile records such as `Scenarios.TrainingScaleProfiles.*` for scenario assumptions and keep truly cited field measurements in `Literature.*` only if they are used as literature facts |
| `Literature.Overheads.*` | operational goodput budget fractions, not physical systems | `Ops.TrainingRunOverheads.*` or a named `Ops.TrainingRunProfiles.*` object if reused as an operational policy profile; if the values are one scenario only, put the complete bundle in `Scenarios.*` |
| `Literature.Energy.*` | architecture-class effective pJ/FLOP and per-byte movement hierarchy; these describe technology behavior, not literature as a category | `Hardware.Tech.*` when they are technology-class facts, preferably as typed quantities; use `Scenarios.*` only for explicit comparison bundles that are not asserted as hardware/technology facts |
| `Literature.Sustainability.TransatlanticRoundTripCo2Kg` | sustainability comparison anchor, not a literature domain | `Scenarios.EmissionsAnchors.*` or `Scenarios.SustainabilityAnchors.*`, parallel to existing `Scenarios.EnergyAnchors.*` |

The decision rule:

- If the value describes a part, generation, operation, memory movement, or process technology, put it under `Hardware.Tech.*`.
- If the value describes a composed physical/logical system, cluster, rack, node, fabric, topology, or storage path, put it under `Systems.*`.
- If the value describes an operational policy, maintenance/recovery budget, monitoring threshold, or goodput loss profile, put it under `Ops.*`.
- If the value is a reusable scenario or comparison bundle, put it under `Scenarios.*`.
- If the value is directly a cited field figure from a paper/report, keep it under `Literature.*`.
- If the value can be derived from a system configuration and an engine model, prefer deriving it instead of storing another scalar.

### B3. Create neutral semantic homes

Add the smallest set of neutral registries needed. Do not add book-specific names.

Candidate additions:

- `Ops.TrainingRunOverheads` or `Ops.TrainingRunProfiles` for reusable operational goodput-loss fractions such as pipeline bubbles, checkpointing, recovery, and maintenance. Do not put these under `Systems` unless they are fields of a concrete fleet/system profile.
- `Scenarios.TrainingScaleProfiles` only for explicit scenario assumptions such as "32-GPU near-linear tier" or "1024-GPU teaching tier." Prefer an engine-derived value when possible. Keep true paper/report measurements in `Literature` only when they are used as cited literature facts, not as generic defaults.
- `Hardware.Tech.EffectiveOpEnergy` / `Hardware.Tech.MovementEnergy` or equivalent typed technology-class homes for effective pJ/FLOP and pJ/byte hierarchy values. Avoid `Scenarios.ComputeEnergyHierarchy` unless the value is explicitly an illustrative comparison rather than a hardware/technology fact.
- `Scenarios.EmissionsAnchors` or `Scenarios.SustainabilityAnchors` for reusable emissions comparison anchors, parallel to `Scenarios.EnergyAnchors`.
- `Systems.Clusters` entries for reusable cluster configurations that currently exist as local "dummy" or "frontier" bundles in LEGO cells, but only when the object is a composed fleet/node/fabric system. Scenario bundles that include workload, grid, utilization, or amortization should live in `Scenarios.*` and reference `Systems.Clusters.*`.
- existing storage homes should be used before adding a new namespace:
  - `Hardware.Tech.Storage` for generic storage technology bandwidth/latency tiers
  - `Systems.StorageSubsystem`, `NodeStorageConfig`, and `CheckpointStoragePath` for composed storage systems
  - `Infrastructure.Pricing.*` for storage prices and billing rates
  Add a new `Systems.Storage` registry only if the audit shows repeated reusable storage subsystems that cannot cleanly live in the existing storage types.

Avoid alias shims if all consumers are in-repo and can be migrated atomically. If a compatibility alias is temporarily necessary, mark it deprecated and add a grep/audit rule that forbids new uses.

### B4. Add source/invariant checks backwards from mistakes

For every mistake found, add a check before broad migration or in the same commit:

- no bare scalars in sourced registries
- no `BOOK_*`, `prov:book`, `MLSysBook`, `Volume I`, `Volume II`, or "worked example" inside `mlsysim/mlsysim`
- no non-literature convention values under `Literature.*`, except explicit allowlist during migration
- no old migrated paths such as `Literature.Scaling`, `Literature.Overheads`, `Literature.Energy`, `Literature.Sustainability` across book/slides/labs/docs after migration
- no local hardware/model/infrastructure specs in LEGO `LOAD`
- no quantity-to-float-to-quantity reattachment patterns in LEGO cells
- no unit-bearing `_str` outputs using plain `fmt()` unless explicitly allowed
- no closed-output prose that repeats the unit
- no `_kwh_str`, `_gb_s_str`, `_tflop_s_str`, etc. names whose formatter does not force that unit

These checks should be pre-commit capable where fast. Slower whole-book audits can be binder or CI gates first.

---

## 4. Phase C - Rules Update Before Broad Editing

Goal: encode the lessons that should guide every future agent before broad mechanical work starts.

Update local `.claude/rules` only with durable rules, not one-off project status.

### C1. `.claude/rules/mlsysim.md`

Add or tighten:

- provenance is metadata, not a semantic home
- `Literature` is only for directly cited field figures
- conventions, estimates, scenarios, hardware facts, and system compositions must live in their semantic homes even when sourced from a paper
- sourced registries require provenance-bearing records
- no book-specific registry/provenance naming inside MLSysIM
- if a value is reusable across QMD/slides/labs/docs, it belongs in MLSysIM first

### C2. `.claude/rules/lego-units.md`

Add or tighten:

- output names are assertions of rendered unit/scale
- fixed-unit output names must force the same unit in the formatter
- auto-scaling output names must stay generic
- unit-bearing outputs should use domain formatters
- `fmt()` is only for dimensionless values or values intentionally formatted without a unit
- local literals in `LOAD` must be marked as scenario/illustrative/budget assumptions
- any recurring local scenario should be promoted to `Scenarios` or `Systems`

### C3. `.claude/rules/slides.md`

Add:

- slides must not copy stale constants from book prose
- if a slide uses a measurable book claim, source it from MLSysIM or from generated artifacts derived from MLSysIM
- slide examples should use the current semantic registry path, not migrated/deprecated paths

### C4. `.claude/rules/labs.md`

Add:

- labs should import current MLSysIM registries rather than retyping hardware/model/system values
- lab constants should follow the same semantic-home distinction as the book
- lab output numbers should use MLSysIM formatters or lab-specific wrappers built on them

### C5. `.claude/rules/margin-figures.md`

Defer detailed margin-figure rules until after the margin-figure audit, but add one durable placeholder:

- margin figures that encode quantitative values must either source those values from MLSysIM or be regenerated from MLSysIM-backed data.

---

## 5. Phase D - Whole-QMD LEGO and Prose Audit

Goal: every QMD cell and nearby prose follows the same source, quantity, guard, and output convention.

### D1. Process order

Audit in book order:

Vol I:

1. `introduction`
2. `ml_systems`
3. `ml_workflow`
4. `data_engineering`
5. `nn_computation`
6. `nn_architectures`
7. `frameworks`
8. `training`
9. `data_selection`
10. `model_compression`
11. `hw_acceleration`
12. `benchmarking`
13. `model_serving`
14. `ml_ops`
15. `responsible_engr`
16. `conclusion`
17. appendices with LEGO

Vol II:

1. `introduction`
2. `compute_infrastructure`
3. `network_fabrics`
4. `data_storage`
5. `distributed_training`
6. `collective_communication`
7. `fault_tolerance`
8. `fleet_orchestration`
9. `performance_engineering`
10. `inference`
11. `edge_intelligence`
12. `ops_scale`
13. `security_privacy`
14. `robust_ai`
15. `sustainable_ai`
16. `responsible_ai`
17. `conclusion`
18. appendices with LEGO

### D2. Per-chapter checklist

For each QMD:

1. List all LEGO cells and classes.
2. Confirm `LOAD` sources:
   - hardware/model/grid/system/storage/fabric/price facts come from MLSysIM
   - local literals are only local assumptions and are marked
   - repeated assumptions are promoted to `Scenarios` or `Systems`
3. Confirm `EXECUTE` discipline:
   - Pint quantities remain quantities until output
   - use `Q_`, units, or registry quantities consistently
   - parenthesize compound unit construction, e.g. `1.9 * (TB / second)`
   - avoid ambiguous `R.value * GB/second`; prefer `R.value * (GB / second)` when the source is scalar
4. Confirm `GUARD` checks:
   - unit/dimension checks for physical results
   - value checks for critical textbook numbers
   - ratio/multiple checks where prose depends on comparison
   - prose-facing expectations for fixed-unit output names
5. Confirm `OUTPUT` naming:
   - every unit-bearing output uses a domain formatter
   - fixed-unit names force the matching formatter unit
   - auto-scaled names stay generic
   - no stale manual suffix pattern remains
6. Confirm prose:
   - closed strings are referenced bare
   - prose does not duplicate units
   - approximate/equality spacing is typographically clean
   - equations avoid awkward display lines where inline prose is clearer
   - repeated table/figure references are intentional
   - single-sentence floating paragraphs are either merged, expanded, or justified
7. Confirm no old path usage after registry migration.
8. Run the chapter through the headless LEGO executor.

### D3. Specific known patterns to include

Include these in the audit backlog:

- `fmt(hours, ...)` used for durations such as chargeback hours should become `fmt_time` or a duration-specific helper if it needs unit validation. Fixed-unit names such as `_hours_str` should force a time unit.
- `fmt(ratio, ...)` for speedups/ratios should be reviewed. If ratios recur, use `fmt_multiple` for `x`-style outputs and consider `fmt_ratio` for unitless ratios that should not imply speedup.
- Avoid `7x $\times$ speedup`; a formatter that emits `7x` should not be followed by a manual `\times`.
- Decide whether `fmt_multiple` should own the LaTeX multiplication symbol for math contexts or whether prose should consistently use `7x` without extra symbols.
- `samples/s`, `tokens/s`, QPS, requests/s should use a semantic rate formatter rather than raw `fmt()` plus a manual suffix.
- QMD code should not pass `unit_label=`. Add or use a semantic formatter helper in `mlsysim.fmt` for every deliberate label policy.
- LEGO code should not reference another LEGO class in Python. Promote the shared value to `Hardware`, `Models`, `Systems`, `Infrastructure`, `Ops`, `Scenarios`, or a helper, then have both cells load from that source.
- Parenthesize compound units in code for readability: `(USD / GB)`, `(GB / second)`, `(TFLOP / second)`.
- Replace awkward display equations like `T_load = 10 GB / 32 GB/s approx 312.5 ms` with either a proper aligned equation block or an inline sentence, depending on layout.
- Review table/figure references that repeat in adjacent sentences and simplify where the second reference does not add clarity.
- Review generic "see @sec-glossary" prose. If the target is too broad or not useful in context, replace with a more specific reference or remove it.
- Review HTML/PDF conditional text blocks that duplicate the same sentence unless the format-specific split materially improves layout.

### D4. Parallelization model

Parallelize review by chapter only after Phase B and C are complete. Central MLSysIM changes stay with one owner.

Each chapter agent gets:

- this plan
- `.claude/rules/mlsysim.md`
- `.claude/rules/lego-units.md`
- the generated LEGO inventory for that chapter
- the current allowed formatter list
- explicit instruction not to modify MLSysIM

Each chapter agent returns:

- patch proposal or direct file edits
- list of MLSysIM values they think are missing
- list of formatter gaps
- list of prose/layout concerns
- local validation commands and outputs

The central owner then:

- adds MLSysIM registry entries
- adds/updates domain formatters
- resolves cross-chapter naming
- runs whole-book checks

---

## 6. Phase E - Update Non-QMD Consumers

Goal: no stale MLSysIM path, copied constant, or old teaching number survives outside book chapters.

### E1. Slides

Audit:

```bash
rg -n "Hardware\\.|Models\\.|Systems\\.|Infrastructure\\.|Literature\\.|Scenarios\\.|mlsysim|GB/s|TFLOP|kWh|MWh|PUE|H100|A100|BatchSize|Scaling|Overheads|Energy" slides
```

Fix:

- migrated registry paths
- copied hardware/model/system values that should come from generated MLSysIM data
- stale units or suffixes
- deck speaker notes that cite old values

Validate:

```bash
cd slides
make check
```

If full slide builds are too slow, build touched decks first, then run `make check` before sign-off.

### E2. Labs

Audit:

```bash
rg -n "Hardware\\.|Models\\.|Systems\\.|Infrastructure\\.|Literature\\.|Scenarios\\.|mlsysim|GB/s|TFLOP|kWh|MWh|PUE|H100|A100|BatchSize|Scaling|Overheads|Energy" labs mlsysim/mlsysim/labs
```

Fix:

- stale imports
- hardcoded specs
- lab-specific constants that should be promoted to MLSysIM
- output formatting that bypasses MLSysIM formatters where consistency matters

Validate:

```bash
pytest labs/tests/test_static.py -v
```

### E3. MLSysIM docs/site/examples

Audit:

```bash
rg -n "Hardware\\.|Models\\.|Systems\\.|Infrastructure\\.|Literature\\.|Scenarios\\.|mlsysim|BatchSize|Scaling|Overheads|Energy|BOOK_|prov:book|MLSysBook|Volume I|Volume II|textbook|worked example" \
  mlsysim/docs mlsysim/examples mlsysim/tutorials mlsysim/mlsysim \
  --glob '!**/__pycache__/**'
```

Fix:

- stale registry paths
- stale API docs
- examples that teach local constants instead of registry imports
- docs that describe `Literature` as a catch-all source domain

If generated API docs exist, regenerate them after code changes rather than hand-editing generated stale output.

### E4. Book support files and tools

Audit:

```bash
rg -n "Literature\\.Scaling|Literature\\.Overheads|Literature\\.Energy|Literature\\.Sustainability|BOOK_|prov:book|MLSysBook|Volume I|Volume II|worked example" \
  book mlsysim .claude \
  --glob '!book/quarto/_build/**'
```

Fix tools/tests so new checks are enforced by local validation.

---

## 7. Phase F - Validation Before Merging Dev

Goal: establish that `fmt-fix` is internally clean before conflict resolution.

Run:

```bash
PYTHONPATH=mlsysim pytest mlsysim/tests -o addopts=
PYTHONPATH=mlsysim python -m mlsysim.tools.audit_provenance --scope all --strict
python3 book/tools/audit/book_check_lego_quantity_flow.py book/quarto/contents --summary
python3 book/tools/audit/book_check_lego_load_pint.py book/quarto/contents
python3 book/tools/scripts/lint_lego_units.py --fail-on warning --baseline book/tools/audit/lego_units_baseline.json
python3 book/tools/audit/book_check_lego_scenario_inputs.py book/quarto/contents --summary
pre-commit run --all-files
```

Run headless LEGO execution across all QMD files. If the existing command is unstable, use the established audit harness from the unit-hardening effort and record the exact command in the progress note.

Expected result:

- no high-severity scenario-input findings
- no stale migrated registry paths
- no unit/prose duplication warnings except explicit baseline items
- all QMD LEGO cells execute

---

## 8. Phase G - Merge Dev Into `fmt-fix`

Goal: reconcile with current local `dev` after the branch is internally clean.

### G1. Inspect main checkout

```bash
cd /Users/VJ/GitHub/MLSysBook
git status --short
git branch --show-current
git pull --ff-only
```

If the main checkout is dirty, stop and report. Do not overwrite it.

### G2. Merge into task branch

```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix
git fetch origin dev
git merge --no-ff origin/dev
```

Conflict rules:

- prefer unit-hardened, Pint-safe LEGO cells
- preserve current MLSysIM semantic-home decisions
- preserve user/editorial changes from `dev` unless they reintroduce stale constants or broken prose
- do not resolve conflicts by deleting checks
- do not revive deprecated registry paths

### G3. Post-merge re-audit

After conflict resolution, rerun all Phase F checks. Treat every failure as a possible merge regression, not as expected noise.

---

## 9. Phase H - Full HTML and PDF Verification

Goal: build the real artifacts and prove rendered output is clean.

### H1. Environment

Use the task worktree and force the correct Jupyter kernel:

```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix
python3 -c "import mlsysim; print(mlsysim.__file__)"
```

The printed path must point into `/Users/VJ/GitHub/MLSysBook-fmt-fix/mlsysim/...`.

If not:

```bash
pip install -e /Users/VJ/GitHub/MLSysBook-fmt-fix/mlsysim
```

Always render with:

```bash
-M jupyter:python3
```

### H2. Full HTML renders

```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix/book/quarto

ln -sf config/_quarto-html-vol1.yml _quarto.yml
MPLBACKEND=Agg quarto render --to html -M jupyter:python3

ln -sf config/_quarto-html-vol2.yml _quarto.yml
MPLBACKEND=Agg quarto render --to html -M jupyter:python3
```

Post-checks:

```bash
rg '\\{python\\}' _build/html-vol1 _build/html-vol2 --glob '*.html'
python3 scripts/verify_rendered_xrefs.py
rg -n '\\?@|Traceback|ImportError|NameError' _build/html-vol1 _build/html-vol2 --glob '*.html'
```

Any raw `{python}` leak, unresolved `?@`, traceback, import error, or name error blocks sign-off.

### H3. Full PDF renders

```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix/book/quarto

ln -sf config/_quarto-pdf-vol1.yml _quarto.yml
ln -sf index-vol1.qmd index.qmd
MPLBACKEND=Agg quarto render --to titlepage-pdf -M jupyter:python3

ln -sf config/_quarto-pdf-vol2.yml _quarto.yml
ln -sf index-vol2.qmd index.qmd
MPLBACKEND=Agg quarto render --to titlepage-pdf -M jupyter:python3
```

Do not use `--to pdf`; use `--to titlepage-pdf` so the PDF header includes are loaded.

Post-checks:

```bash
ls -lh _build/pdf-vol1/*.pdf _build/pdf-vol2/*.pdf
rg '^!|Undefined control sequence|LaTeX Error|Traceback|ImportError|NameError' index.log _build/pdf-vol1 _build/pdf-vol2
```

If Quarto saves TeX logs elsewhere, scan those logs too.

### H4. Rendered prose spot checks

Spot-check rendered HTML and PDF for:

- no duplicated units such as `W W`, `kWh kWh`, `t CO2 CO2`
- no awkward `7x x` or `7x \times` speedup phrasing
- no raw Pint reprs
- no fake unit labels after scalar conversion
- no unintentional display equations that should be prose
- no table/figure reference loops in adjacent sentences
- no stale glossary or section links

Spot checks must include:

- `sustainable_ai`
- `responsible_engr`
- `compute_infrastructure`
- `data_storage`
- `distributed_training`
- `model_serving`
- `benchmarking`
- any chapter touched by taxonomy migration

---

## 10. Phase I - Margin-Figure Audit Plan

Goal: after source/prose/rendering is stable and `dev` is merged, audit margin figures as a separate design/layout effort.

Do not start this before Phase H is green.

### I1. Inventory

List all margin figures, margin notes, and side graphics:

```bash
rg -n "marginfigure|marginnote|margin-|column-margin|aside|fig-margin|layout-ncol|layout-valign|includegraphics|\\.svg|\\.png" \
  book/quarto/contents \
  --glob '*.qmd'
```

Capture:

- file and line
- figure asset path
- page/section after PDF render
- whether it contains quantitative values
- whether values are sourced from MLSysIM or static drawing text

### I2. Automatic layout checks

Use LaTeX logs and PDF artifacts to find likely problems:

- overfull hbox/vbox warnings
- underfull warnings near margin content
- float placement warnings
- pages with dense footnotes and margin notes
- pages with multiple nearby side figures

Potential commands:

```bash
rg -n "Overfull|Underfull|marginpar|Float|too large|rerun|LaTeX Warning" book/quarto/*.log book/quarto/_build/pdf-vol*/**
```

This will not catch everything, but it provides a triage list.

### I3. Visual review

Review PDF spreads for:

- cramped margin notes
- overlapping margin figures and footnotes
- side graphics that fight main text
- labels too small to read
- diagrams whose style no longer matches the book
- quantitative labels not sourced from MLSysIM
- redundant figures where prose/table already carries the idea

### I4. Improvement candidates

Possible actions:

- move some dense footnotes into endnotes or prose
- move some margin notes earlier/later in the paragraph
- convert a marginal graphic into an inline figure when it needs inspection
- simplify a margin figure to one idea
- regenerate quantitative graphics from MLSysIM data
- add new margin figures only where they clarify a central systems idea

Do not treat margin-figure polish as part of numeric correctness. It is a separate final pass after the book is already correct.

---

## 11. Commit Plan After Review

When the user approves execution, commit in small stages. Do not use `git add -A`.

Candidate commit boundaries:

1. provenance/source-loader hardening
2. Literature taxonomy migration and consumer path updates
3. MLSysIM Systems/Scenarios additions for reusable cluster/storage/fleet assumptions
4. formatter/check improvements
5. Vol I QMD LEGO/prose cleanup
6. Vol II QMD LEGO/prose cleanup
7. slides/labs/docs consumer updates
8. `.claude/rules` updates
9. dev merge conflict resolution
10. render/preflight fixes

Each commit should include only files relevant to that stage. Keep unrelated fmt-thread WIP unstaged unless explicitly included.

---

## 12. Acceptance Criteria

Do not sign off until all are true:

- all reusable measurable values used by QMDs/slides/labs/docs come from MLSysIM
- no book-specific names remain inside `mlsysim/mlsysim`
- `Literature.*` contains only true literature/report field figures, or any remaining exceptions are documented and scheduled
- every sourced registry record has provenance
- every LEGO cell in Vol I and Vol II follows Load, Execute, Guard, Output discipline
- every unit-bearing output string uses a domain formatter or a documented exception
- fixed-unit output names force the matching formatter unit
- rendered prose does not duplicate units
- no stale migrated registry paths remain in book/slides/labs/docs
- all MLSysIM tests pass
- provenance audit passes strict mode
- LEGO quantity/load/lint/scenario audits pass
- headless LEGO execution passes all QMD files
- pre-commit passes
- `dev` has been merged into `fmt-fix`
- all checks pass again after the merge
- full Vol I and Vol II HTML render cleanly
- full Vol I and Vol II PDF render cleanly
- no raw `{python}`, unresolved `?@`, or render tracebacks remain
- PDF logs have no LaTeX errors
- margin-figure audit plan is ready to execute after correctness work
- `fmt-fix` has been merged cleanly into `dev`
- `dev` has been pushed to `origin`
- all workflows triggered by the pushed `dev` SHA have passed, or any
  non-triggered workflow is explained by path filters

---

## 13. Immediate Next Step After Review

If this plan is approved, start with Phase A and B:

1. preserve current dirty-state notes
2. validate or adjust the drafted batch-size provenance/source-loader edits
3. generate the whole-repo MLSysIM consumer inventory
4. generate the LEGO-cell/output/prose inventory
5. update the durable `.claude/rules` before broad QMD edits

Only after those are stable should broad chapter-by-chapter work begin.
