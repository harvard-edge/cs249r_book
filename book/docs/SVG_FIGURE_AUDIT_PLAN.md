# SVG Figure Quality Audit (Resumable Plan)

**Status**: Not started — plan only. Pick up after the camera-ready merge into `dev` lands.
**Target**: All 238 SVG figures under `book/quarto/contents/**/images/svg/*.svg`.
**Reviewer model**: `gemini-3.1-pro-preview` (hard pin — no fallback).
**Working branch**: Create a fresh branch off `dev` named `chore/svg-quality-audit-<RUN_TS>`.
**Worktree**: Create with `git worktree add ../MLSysBook-svg-audit-<RUN_TS> chore/svg-quality-audit-<RUN_TS>` so the regular `dev` worktree is unaffected. All work and commits stay inside that worktree.

---

## 0. Goal & success criteria

Every SVG in the book renders cleanly when viewed at the size it actually appears in the published PDF and HTML. Specifically, no figure should exhibit any of these defects:

| Defect category | Examples |
|---|---|
| Overlapping geometry | Two boxes sharing a border, a line crossing under a node label, an arrow head buried in a shape |
| Truncated / clipped content | Text running past the viewBox, an arrow ending inside a node, a label cut off by an edge |
| Mis-aligned annotations | A label not centered on its node, a number floating away from its bar, an axis tick missing |
| Illegible text | Font too small at the rendered size, low-contrast text on a coloured fill, text rotated past readability |
| Visual chartjunk | Stray strokes, orphaned anchors, duplicate shapes stacked, debug guides left in source |
| Brand inconsistencies | Colours not in the BlueLine/GreenLine/RedLine/OrangeLine/etc. palette defined in `header-includes.tex` and `diagram.yml` |

**Done when**: Every SVG either passes Gemini review OR is documented as a known-OK exception with reasoning. A summary report (`SVG_AUDIT_REPORT.md`) lists every figure, its verdict, fixes applied, and remaining open items.

---

## 1. Inventory & batching

```bash
# From worktree root
find book/quarto/contents -path '*/images/svg/*.svg' \
  | sort > .audit/svg-inventory.txt
wc -l .audit/svg-inventory.txt   # expect ~238
```

Group inventory by chapter (one batch per chapter so Gemini sees coherent visual context). Approximate batch sizes from the current count:

| Chapter | SVG count |
|---|---|
| vol2/fault_tolerance | 28 |
| vol2/distributed_training | 21 |
| vol2/security_privacy | 20 |
| vol2/sustainable_ai | 17 |
| vol2/robust_ai | 17 |
| vol2/inference | 16 |
| vol2/ops_scale | 13 |
| vol2/edge_intelligence | 13 |
| vol2/responsible_ai | 11 |
| vol2/fleet_orchestration | 11 |
| vol2/compute_infrastructure | 11 |
| vol2/performance_engineering | 10 |
| vol2/data_storage | 9 |
| vol2/network_fabrics | 8 |
| vol2/introduction | 7 |
| vol2/collective_communication | 7 |
| vol1/optimizations | 4 |
| vol1/nn_computation | 4 |
| vol1/responsible_engr | 3 |
| vol1/data_engineering | 2 |
| vol1/{ml_systems,introduction,hw_acceleration,frameworks,backmatter} | 1 each |

**Sub-batching**: Cap each Gemini call at **8 images** to keep prompts coherent and stay well under the model's context window. Chapters with > 8 figures get split into ceiling(N/8) sub-batches in inventory order.

---

## 2. Tooling

All tools confirmed available on the dev machine via `which`:

- **`rsvg-convert`** (`/opt/homebrew/bin/rsvg-convert`) — primary fast renderer.
- **`inkscape`** (`/opt/homebrew/bin/inkscape`) — fallback for SVGs that rsvg can't render (rare; usually fonts or filters).
- **`gemini`** (`/opt/homebrew/bin/gemini`) — invokes `gemini-3.1-pro-preview`.

Render command (one SVG at a time, but batch the loop):

```bash
# 600 px wide is roughly the size figures appear in the published HTML at 1.5× zoom;
# that's where rendering defects are most visually apparent.
rsvg-convert --width 600 --keep-aspect-ratio "$svg" \
  > ".audit/png/$(basename "$svg" .svg).png"
```

If `rsvg-convert` fails or output is empty, fall back to:

```bash
inkscape "$svg" --export-type=png --export-width=600 --export-filename=- \
  > ".audit/png/$(basename "$svg" .svg).png"
```

For each SVG, also render at the **actual published size** (look up `fig-width` in the chapter's `.qmd` to determine the production width, default to 480 px if not specified). Tiny figures hide defects — review at production size to see what the reader sees.

**Browser screenshot path (optional, for high-fidelity verification)**: install `chromium` via Homebrew (`brew install --cask chromium`) and use `chromium --headless --disable-gpu --window-size=1200,1200 --screenshot=out.png file:///path/to/figure.svg`. Browsers honour CSS, web fonts, and SVG filters more accurately than rsvg. Use this only when Gemini flags a font/filter rendering issue that rsvg may have misrendered.

---

## 3. Gemini CLI invocation pattern

Confirmed pattern (mirrors the camera-ready `gemini_math_check.py` wrapper):

```bash
gemini -m gemini-3.1-pro-preview \
  -o json \
  --approval-mode plan \
  -p "$prompt"
```

**Image attachment**: Gemini CLI's `-p` accepts `@path/to/file.png` references inline in the prompt text. Verify at runtime by sending one test prompt with one image and checking the model acknowledges the visual content. If `@`-attachment is not supported in the installed CLI version, fall back to either: (a) putting all images in a temp directory and including that directory via `--include-directories`, or (b) using the Python SDK directly (`google-genai` package) which has explicit multimodal support.

**Reference prompt template** (one batch of up to 8 figures):

```
You are a meticulous figure-quality reviewer for an MIT Press machine learning
systems textbook. Below are <N> SVG figures from chapter <chapter_name>, each
rendered as a PNG at the size readers will see in print/HTML. For EACH figure,
classify rendering quality and return ONLY a JSON array (no prose, no markdown
fence) with one object per figure in input order:

[
  {
    "figure": "<filename.svg>",
    "verdict": "clean" | "minor-issue" | "major-issue",
    "issues": [
      {
        "category": "overlap" | "clipping" | "alignment" | "legibility" |
                    "chartjunk" | "brand-inconsistency",
        "description": "<one sentence — be specific about which element>",
        "severity": "low" | "medium" | "high",
        "suggested_fix": "<one sentence — what edit would resolve this>",
        "fix_safe_to_auto_apply": true | false
      }
    ],
    "overall_assessment": "<one sentence summary>"
  },
  ...
]

Verdicts:
- "clean": ready to ship, no edits needed
- "minor-issue": one or more low/medium-severity issues, ship-able but worth fixing
- "major-issue": at least one high-severity issue (illegible text, broken layout,
  off-canvas content) — must fix before publication

fix_safe_to_auto_apply must be true ONLY if the fix is a clearly localised SVG
edit (move a text node by N px, swap a stroke colour to BlueLine, increase
font-size from 8 to 10) that preserves all other geometry. For any change
involving redrawing shapes or restructuring the figure, return false.

Figures attached:
@<path/to/fig1.png>
@<path/to/fig2.png>
...
```

---

## 4. Iterative fix loop

Per batch, per figure:

1. **Initial review** — render PNG, send batch to Gemini, parse JSON verdicts.
2. **Triage** — split flagged figures into:
   - `auto-fixable`: at least one issue with `fix_safe_to_auto_apply: true` and severity ≤ medium.
   - `needs-human`: any high-severity issue OR `fix_safe_to_auto_apply: false`.
3. **Auto-fix** (loop, max 3 iterations per figure):
   - Apply Gemini's suggested SVG edit using `xmllint --xpath`-based patches OR a small Python `xml.etree.ElementTree` script. NEVER blindly let Gemini rewrite the whole SVG file.
   - Re-render to PNG.
   - Send the BEFORE + AFTER pair back to Gemini with this prompt:
     ```
     Below are two renderings of the same figure: BEFORE (the original) and
     AFTER (with your suggested fix applied). Has the issue been resolved?
     Return ONLY: {"resolved": true|false, "still_present": "<description>",
     "regression": "<description or null>", "next_action": "accept" |
     "revert" | "iterate"}.
     @before.png
     @after.png
     ```
   - If `accept`: commit the SVG edit atomically and move on.
   - If `iterate` AND iteration count < 3: incorporate the new suggested fix and loop.
   - If `revert` OR iteration count == 3: `git checkout -- <svg>`, log to `needs-human`.
4. **Human queue** — `needs-human` figures get listed in the report with full Gemini reasoning and the rendered PNG path. They are NOT auto-edited.

---

## 5. Safety guardrails (STRICT)

- **NEVER** let Gemini emit a full replacement SVG to be written wholesale. Only structured patch instructions (move element X by Δx,Δy; change attribute Y from A to B) that a small Python helper applies via `xml.etree.ElementTree`.
- **Atomic commits**: one commit per SVG fix. Subject format: `style(<vol/chapter>/svg): <terse description> (<filename>)`.
- **No co-author / AI footer** in any commit message (project rule).
- **Validate after every fix**:
  - Re-render the SVG and check the output PNG is non-empty and has dimensions within ±5% of the original render.
  - If the chapter has a `quarto preview` or per-chapter PDF build path, run a quick smoke render to make sure the SVG still embeds correctly.
- **Revert-on-regression**: if Gemini's "is the issue resolved" pass returns `regression`, revert the working-tree change and queue for human review.
- **Per-figure cap**: max 3 iteration attempts. Hard-stop and queue for human after that.
- **Per-batch budget cap**: max 5 minutes wall-clock of Gemini time per batch. If exceeded, defer remaining figures in the batch to `needs-human`.
- **Forbidden paths**: do NOT touch `book/quarto/_build/`, `_freeze/`, `_extensions/`, or any `.pdf` / `.html` artifact. SVG source only.

---

## 6. Resumable state

Maintain a single state file at `.audit/state.json` in the worktree:

```json
{
  "run_ts": "20260420-103000",
  "model": "gemini-3.1-pro-preview",
  "total_figures": 238,
  "batches": [
    {
      "batch_id": "vol2-fault_tolerance-1of4",
      "figures": ["...svg", "...svg"],
      "status": "completed",
      "verdicts": {"figA.svg": "clean", "figB.svg": "minor-issue (resolved)"},
      "commits": ["abc1234", "def5678"]
    },
    ...
  ],
  "needs_human": [
    {"svg": "...", "issues": [...], "rendered_png": ".audit/png/..."}
  ],
  "totals": {"clean": 0, "minor": 0, "major": 0, "needs_human": 0}
}
```

Resume command (when restarting): script reads `state.json`, skips any batch with `status: completed`, and continues from the next pending one. Gemini wall-clock budget tracked separately at `.audit/gemini-budget.txt` (suggested cap: 4 hours total — at ~30 sec/batch and ~60 batches, budget ≈ 30 minutes; cap allows 8× headroom).

---

## 7. Suggested implementation: one Python driver + one shell wrapper

Two files in `book/tools/scripts/audit/`:

- **`svg_audit.py`** — orchestrator. Reads inventory, manages batches, calls render + Gemini + patch helpers, writes state, emits report.
- **`run-svg-audit.sh`** — entrypoint. Activates venv, sets `RUN_TS`, creates `.audit/` directory, invokes `svg_audit.py`, writes summary log to `.audit/logs/run.log`.

`svg_audit.py` modules:

| Module | Responsibility |
|---|---|
| `inventory.py` | Walk the contents tree; enumerate SVGs; group into batches (one per chapter, capped at 8). |
| `render.py` | Wrap `rsvg-convert` (with `inkscape` fallback). Produce PNG at the figure's published size. |
| `gemini.py` | Reuse the budget/retry pattern from `gemini_math_check.py`. Multimodal prompt builder with `@image` references. JSON parser with strict schema validation. |
| `patcher.py` | Apply structured SVG patches using `xml.etree.ElementTree` (move element, change attribute, set style key). NEVER overwrite full file. |
| `state.py` | Read/write `.audit/state.json` atomically. Idempotent batch progression. |
| `report.py` | Emit `SVG_AUDIT_REPORT.md` with per-batch tables, verdict counts, screenshots of needs-human items, commit SHAs. |

---

## 8. Estimated cost & duration

- **Render**: ~1 second per SVG via `rsvg-convert` → ~4 minutes for all 238.
- **Gemini calls**:
  - Initial review: 30 batches × ~30 sec = ~15 minutes wall-clock.
  - Re-review (only flagged): assume ~30% flagged → ~10 batches × ~30 sec × ~2 iterations avg = ~10 minutes.
  - Total Gemini wall-clock: **~30 minutes** (well under any reasonable budget).
- **Auto-fix application + commits**: ~1 minute per figure × ~50 figures × ~1 iteration = ~50 minutes of CPU + git work.
- **Total wall-clock**: ~2–3 hours end-to-end (mostly serialised by Gemini latency).

---

## 9. Final report

`SVG_AUDIT_REPORT.md` (committed at the end of the audit) must contain:

1. **Run metadata**: timestamp, model, branch, total figures, totals by verdict.
2. **Per-chapter summary table**: figure name → initial verdict → final verdict → commit SHA (if fixed) → notes.
3. **Needs-human queue**: full list with Gemini reasoning + rendered PNG paths so a human reviewer can triage in one sitting.
4. **Brand inconsistencies**: any figures using colours outside the documented palette (`BlueLine`, `GreenLine`, `RedLine`, `OrangeLine`, `BrownLine`, `VioletLine`, `GrayLine`) — these are usually low-effort manual fixes.
5. **Diff stats**: total SVG files modified, lines added/removed, total commits in the audit.
6. **Open follow-ups**: anything discovered that's out of scope (e.g., a figure that should be redrawn rather than patched) gets logged here for future work.

---

## 10. Pickup checklist

When resuming this work:

- [ ] Confirm dev branch contains the camera-ready merge (`git log --oneline | grep "Merge camera-ready"`).
- [ ] Confirm `gemini`, `rsvg-convert`, `inkscape` all on PATH.
- [ ] Confirm `gemini -m gemini-3.1-pro-preview -p "ping"` returns successfully (model still available).
- [ ] Verify Gemini CLI image attachment mechanism with one test call (`@<path>` syntax — fall back to `--include-directories` or Python SDK if needed).
- [ ] Decide on the audit's scope: full 238 figures, or Vol2 only, or a single chapter as a smoke test first.
- [ ] Create the audit worktree + branch, set `RUN_TS`, initialise `.audit/`.
- [ ] Run `svg_audit.py --batch <chapter>` for one chapter as a smoke test (recommend: `vol2/introduction` — only 7 SVGs).
- [ ] Inspect the smoke-test output (PNG renders, Gemini JSON, applied patches) by hand. If satisfactory, run the full audit.
- [ ] After each chapter completes, push commits to the audit branch (do NOT push to `dev` until human reviewer signs off on the report).

---

## 11. Out of scope

- TikZ figures (inline LaTeX in `.qmd` files) — separate audit, handled by chapter-specific QA in the camera-ready sweep.
- Screenshots and photographs (`*.png`, `*.jpg` under `images/png/`, `images/photos/`) — not vector, defects are different (resolution, compression).
- Bibliography entries, body prose, math — covered by the existing `book-prose-merged.md` rules and the math semantic QA.
- HTML/PDF rendering bugs caused by Quarto/LaTeX (not the SVG itself) — file as a separate issue if discovered during the audit.
