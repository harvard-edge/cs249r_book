# Standardized Lab Template

## Cell-by-Cell Specification

Every lab consists of exactly 22 named cell slots. Not every lab will use all 22 (some
may merge or omit optional cells), but the ordering and naming convention is fixed.
Students moving between labs should always encounter the same rhythm.

---

### ZONE A: OPENING (Cells 0-4) -- read time < 1 minute

```
Cell 0   SETUP              (hide_code=False)   Imports, path wiring, constants, ledger init
Cell 1   HEADER             (hide_code=True)     Dark gradient banner with lab identity
Cell 2   BRIEFING           (hide_code=True)     NEW: Objectives, prerequisites, core question, duration
Cell 3   READING            (hide_code=True)     Recommended chapter sections
Cell 4   CONTEXT_TOGGLE     (hide_code=True)     2-context radio + prior ledger load
```

**What changed:** Cell 2 (BRIEFING) is new. It replaces the jump from header straight
into recommended reading. The briefing tells students what they will learn, what they
need to know, and what tension the lab explores -- in under 45 seconds of reading.

The old RECOMMENDED READING cell (previously Cell 2) shifts to Cell 3.

---

### ZONE B: ACT I -- CALIBRATION (Cells 5-11) -- 12-15 minutes

```
Cell 5   ACT1_BANNER        (hide_code=True)     Act I number badge + title + "why this matters"
Cell 6   ACT1_STAKEHOLDER   (hide_code=True)     Stakeholder message setting the scenario
Cell 7   ACT1_CONCEPT       (hide_code=True)     Brief concept setup (formula, key idea)
Cell 8   ACT1_PREDICTION    (hide_code=True)     Structured prediction lock (radio or number)
Cell 9   ACT1_GATE          (hide_code=True)     mo.stop gate -- instruments hidden until prediction
Cell 10  ACT1_INSTRUMENTS   (hide_code=True)     Sliders + live chart(s) -- 1-2 controls max
Cell 11  ACT1_REVEAL        (hide_code=True)     Prediction-vs-reality overlay + reflection + MathPeek
```

**Cell 11 consolidation:** The current labs spread reveal, reflection, and MathPeek
across 3 separate cells. The template consolidates them into a single REVEAL cell
that stacks: (a) prediction overlay, (b) structured reflection radio, (c) feedback
callout, (d) MathPeek accordion. This reduces cell count and keeps the "aha moment"
in one place. If Marimo's dataflow requires splitting (e.g., reflection value needed
downstream), Cell 11 may split into 11a (REVEAL) and 11b (REFLECTION), but the visual
grouping should remain tight.

---

### ZONE C: ACT II -- DESIGN CHALLENGE (Cells 12-19) -- 20-25 minutes

```
Cell 12  ACT2_BANNER        (hide_code=True)     Act II number badge + title + "why this matters"
Cell 13  ACT2_STAKEHOLDER   (hide_code=True)     Stakeholder message (escalated scenario)
Cell 14  ACT2_CONCEPT       (hide_code=True)     Concept setup for design challenge
Cell 15  ACT2_PREDICTION    (hide_code=True)     Numeric or radio prediction
Cell 16  ACT2_GATE          (hide_code=True)     mo.stop gate
Cell 17  ACT2_INSTRUMENTS   (hide_code=True)     Full instrument set (2-3 charts, multiple controls)
Cell 18  ACT2_FAILURE       (hide_code=True)     Failure state display (OOM, SLA violation, etc.)
Cell 19  ACT2_REVEAL        (hide_code=True)     Prediction overlay + reflection + MathPeek
```

**Note on Cells 17-18:** In practice, the failure state is often embedded within the
instrument chart cell (bars turn red inline). Cell 18 is the *banner* that appears when
the failure state triggers. If the failure state is purely visual within the chart, Cell 18
can be a conditional callout rather than a separate instrument.

---

### ZONE D: CLOSING (Cells 20-21) -- read time < 2 minutes

```
Cell 20  SYNTHESIS          (hide_code=True)     NEW: Key takeaways + what's next + self-assessment
Cell 21  LEDGER_HUD         (hide_code=True)     Design Ledger save + HUD footer bar
```

**What changed:** Cell 20 (SYNTHESIS) is new as a dedicated, structured cell. The current
labs put takeaways inside the ledger cell. Separating them makes the closing a proper
pedagogical moment rather than an afterthought appended to data persistence.

---

## Zone A Detail: The BRIEFING Cell (Cell 2)

### Purpose

Students opening a lab should know within 30 seconds: (1) what they will learn,
(2) what they need to know already, (3) what question drives the exploration,
and (4) how long it will take. The BRIEFING cell delivers all four.

### Visual Design

A single white card with a thin left-border accent (BlueLine). Four compact sections
separated by subtle dividers. No dark background -- the header already provides the
visual weight. The briefing is clean, informational, low-visual-noise.

### HTML Template

```python
@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['BlueLine']};
                background: white; border-radius: 0 12px 12px 0;
                padding: 20px 28px; margin: 8px 0 16px 0;
                box-shadow: 0 1px 4px rgba(0,0,0,0.06);">

        <!-- LEARNING OBJECTIVES -->
        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Learning Objectives
            </div>
            <div style="font-size: 0.9rem; color: {COLORS['TextSec']}; line-height: 1.7;">
                <div style="margin-bottom: 3px;">1. <strong>[Measurable outcome using an action verb]</strong></div>
                <div style="margin-bottom: 3px;">2. <strong>[Measurable outcome using an action verb]</strong></div>
                <div style="margin-bottom: 3px;">3. <strong>[Measurable outcome using an action verb]</strong></div>
            </div>
        </div>

        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>

        <!-- PREREQUISITES + DURATION (side by side) -->
        <div style="display: flex; gap: 32px; margin-top: 16px; margin-bottom: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Prerequisites
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    [Concept A from @sec-...] &middot; [Concept B from @sec-...]
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>35-40 min</strong><br/>
                    Act I: ~12 min &middot; Act II: ~25 min
                </div>
            </div>
        </div>

        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>

        <!-- CORE QUESTION -->
        <div style="margin-top: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                "[The central tension phrased as a question the student cannot yet answer]"
            </div>
        </div>
    </div>
    """)
    return
```

### Content Rules

**Learning Objectives** -- exactly 3 (occasionally 4). Each must:
- Start with a measurable action verb: *Quantify, Diagnose, Identify, Predict, Design, Compare*
- Name the specific physical quantity or trade-off
- Be verifiable by the end of the lab

Good: "Quantify why a 6x compute upgrade yields only 8% latency reduction for memory-bound workloads."
Bad: "Understand the Iron Law." (not measurable, not specific)

**Prerequisites** -- the 2-3 concepts students must already know. Written as a compact
inline list, not a bulleted paragraph. Reference textbook sections with @sec- slugs.
If a student does not know these, the lab will not make sense; they should stop and read first.

**Duration** -- always "35-40 min" total with the Act I / Act II breakdown.

**Core Question** -- one sentence, phrased as a question, expressing the tension the lab
resolves. This is the "aha moment" stated as a mystery. The student should not be able to
answer it correctly before completing the lab.

### Example: Lab 02

```
Learning Objectives:
1. Decompose inference latency into memory, compute, and overhead terms using the Iron Law
2. Diagnose why a 6x compute upgrade yields only 8% latency improvement for memory-bound workloads
3. Calculate the propagation delay floor for a given datacenter distance and determine when edge
   deployment becomes physically necessary

Prerequisites:
Iron Law equation (T = D/BW + O/R + L) from @sec-introduction-iron-law  ·
Arithmetic Intensity definition from @sec-ml-systems-deployment-spectrum

Duration:
35-40 min  --  Act I: ~12 min  ·  Act II: ~25 min

Core Question:
"If you double the compute power of your inference server, why doesn't latency halve --
and when does the speed of light make cloud inference physically impossible?"
```

### Example: Lab 05

```
Learning Objectives:
1. Quantify the transistor cost ratio between ReLU and Sigmoid activation functions (50x)
2. Predict which memory hierarchy tier a given layer's activations will land in, given batch
   size and spatial dimensions
3. Identify the batch size threshold where activation memory spills from L2 cache to HBM,
   triggering a 2.5x latency penalty

Prerequisites:
Activation function definitions from @sec-neural-computation-artificial-neuron  ·
Memory hierarchy tiers (L1/L2/HBM/DRAM) from @sec-neural-computation-transistor-tax

Duration:
35-40 min  --  Act I: ~12 min  ·  Act II: ~25 min

Core Question:
"ReLU and Sigmoid produce similar accuracy -- so why does the choice of activation function
determine whether your model fits in cache or spills to memory 100x slower?"
```

---

## Zone B/C Detail: Act Banner Cells (Cells 5 and 12)

### Purpose

Each act opens with a compact banner that tells the student: (1) which act they are in,
(2) the act title, and (3) one sentence explaining why this act matters in the context
of the core question.

### HTML Template

```python
@app.cell(hide_code=True)
def _(mo, COLORS):
    # Act number: "I" or "II"
    # Act title: e.g., "The Memory Wall Revelation"
    # Why it matters: one sentence connecting to the core question
    _act_num = "I"
    _act_color = COLORS["BlueLine"]
    _act_title = "[Act Title]"
    _act_duration = "12-15 min"
    _act_why = "[One sentence: what wrong prior will be corrected and why that matters]"

    mo.Html(f"""
    <div style="margin: 32px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: {_act_color}; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;
                        flex-shrink: 0;">{_act_num}</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">
                Act {_act_num} &middot; {_act_duration}</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">
            {_act_title}
        </div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            {_act_why}
        </div>
    </div>
    """)
    return
```

### Content Rules for the "Why" Line

The "why" line is NOT a description of what students will do. It is a statement of
the tension or wrong prior that this act addresses.

Good: "You expect a faster GPU to halve latency. The data will show that memory bandwidth,
not compute, is the binding constraint -- and the $2M upgrade attacked the wrong term."

Bad: "In this act you will use sliders to explore the Iron Law."

For Act II, the "why" line should escalate: "Act I showed that memory dominates compute.
Now discover a constraint that even memory bandwidth cannot fix: the speed of light."

---

## Zone D Detail: The SYNTHESIS Cell (Cell 20)

### Purpose

After completing both acts, students need a structured moment to consolidate what they
learned. The SYNTHESIS cell provides: (1) key takeaways tied to the objectives, (2) a
forward pointer to the next lab, and (3) an optional self-assessment checkpoint.

This cell is NOT gated behind `mo.stop` -- it is always visible. Students who scroll
down before completing both acts will see it, which is fine; it serves as a preview
of what they should be able to articulate by the end.

However, some content within it (the self-assessment answers) CAN be gated behind act
completion if desired.

### HTML Template

```python
@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.vstack([
        mo.md("---"),

        # ── KEY TAKEAWAYS ──
        mo.Html(f"""
        <div style="background: {COLORS['Surface2']}; border: 1px solid {COLORS['Border']};
                    border-radius: 12px; padding: 24px 28px; margin: 16px 0;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px;">
                Key Takeaways
            </div>
            <div style="font-size: 0.92rem; color: {COLORS['Text']}; line-height: 1.75;">
                <div style="margin-bottom: 10px;">
                    <strong>1. [Takeaway tied to Objective 1].</strong>
                    [One sentence with the specific number or ratio discovered.]
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. [Takeaway tied to Objective 2].</strong>
                    [One sentence with the specific number or ratio discovered.]
                </div>
                <div>
                    <strong>3. [Takeaway tied to Objective 3].</strong>
                    [One sentence with the specific number or ratio discovered.]
                </div>
            </div>
        </div>
        """),

        # ── CONNECTIONS ──
        mo.Html(f"""
        <div style="display: flex; gap: 16px; margin: 8px 0 16px 0; flex-wrap: wrap;">

            <!-- What's Next -->
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    What's Next
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Lab NN:</strong> [Title] -- [one sentence on how this lab's
                    discovery creates the next lab's question].
                </div>
            </div>

            <!-- Textbook Connection -->
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    Textbook &amp; TinyTorch
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Read:</strong> @sec-[...] for the full derivation.<br/>
                    <strong>Build:</strong> TinyTorch Module NN -- [what they will implement].
                </div>
            </div>

        </div>
        """),
    ])
    return
```

### Content Rules

**Key Takeaways** -- exactly 3 (matching the 3 learning objectives). Each takeaway must:
- Reference the specific quantitative result the student discovered
- Not introduce new information -- only crystallize what the instruments showed
- Be memorable enough to recall without the lab open

Good: "The Memory Wall dominates transformer inference. At AI = 5 FLOPs/Byte (far below
the H100 ridge point of 590), the memory term accounts for >99% of latency. A 6x compute
upgrade yields only ~8% improvement."

Bad: "Memory bandwidth matters." (too vague, no numbers)

**What's Next** -- one sentence establishing the causal link to the next lab. The pattern
is: "This lab showed X. The next lab asks: what happens when Y?"

**Textbook and TinyTorch** -- a compact pointer to the relevant textbook section for deeper
reading and the TinyTorch module where students implement the concept from scratch.

### Self-Assessment Checkpoint (Optional)

If desired, a collapsible self-assessment can be added at the bottom of the SYNTHESIS cell:

```python
mo.accordion({
    "Self-Assessment: Can you answer these?": mo.md("""
    1. Why does doubling compute throughput not halve latency for a memory-bound workload?
    2. At what Arithmetic Intensity does the H100 transition from memory-bound to compute-bound?
    3. For a 10 ms SLA, what is the maximum datacenter distance for cloud inference?

    *If you cannot answer all three from memory, revisit Act I and Act II.*
    """)
})
```

The self-assessment questions should be directly answerable from the lab's instruments.
They are NOT new questions -- they restate the prediction questions in a form the student
should now be able to answer correctly.

---

## Zone D Detail: The LEDGER_HUD Cell (Cell 21)

This cell is unchanged from the current implementation. It:
1. Gates behind completion of both acts (all predictions + reflections non-null)
2. Saves the chapter's Design Ledger fields
3. Renders the dark monospace HUD footer bar

The only change: the Key Takeaways and Connections callouts currently embedded in this
cell MOVE to Cell 20 (SYNTHESIS). The LEDGER_HUD cell contains ONLY the ledger save
logic and the HUD bar.

---

## Complete Cell Map (Reference)

```
ZONE A: OPENING
  Cell 0   SETUP              imports, constants, ledger
  Cell 1   HEADER             dark gradient banner
  Cell 2   BRIEFING           objectives, prereqs, core question, duration    [NEW]
  Cell 3   READING            recommended chapter sections
  Cell 4   CONTEXT_TOGGLE     2-context radio + specs display

ZONE B: ACT I (CALIBRATION)
  Cell 5   ACT1_BANNER        act number badge + title + "why this matters"
  Cell 6   ACT1_STAKEHOLDER   stakeholder message
  Cell 7   ACT1_CONCEPT       formula / key idea setup
  Cell 8   ACT1_PREDICTION    structured prediction lock
  Cell 9   ACT1_GATE          mo.stop until prediction made
  Cell 10  ACT1_INSTRUMENTS   sliders + live charts (1-2 controls)
  Cell 11  ACT1_REVEAL        prediction overlay + reflection + MathPeek

ZONE C: ACT II (DESIGN CHALLENGE)
  Cell 12  ACT2_BANNER        act number badge + title + "why this matters"
  Cell 13  ACT2_STAKEHOLDER   stakeholder message (escalated)
  Cell 14  ACT2_CONCEPT       concept setup for design challenge
  Cell 15  ACT2_PREDICTION    numeric or radio prediction
  Cell 16  ACT2_GATE          mo.stop until prediction made
  Cell 17  ACT2_INSTRUMENTS   full instrument set (2-3 charts)
  Cell 18  ACT2_FAILURE       failure state banner (conditional)
  Cell 19  ACT2_REVEAL        prediction overlay + reflection + MathPeek

ZONE D: CLOSING
  Cell 20  SYNTHESIS          takeaways + what's next + self-assessment       [NEW]
  Cell 21  LEDGER_HUD         ledger save + HUD footer bar
```

Total: 22 cell slots. Typical actual cell count: 18-22 (some cells merge or split
based on Marimo dataflow requirements).

---

## Comment Conventions

Each cell is preceded by a comment block following this pattern:

```python
# ─── CELL N: CELL_NAME ──────────────────────────────────────────────────────
```

Zone boundaries use double-line separators:

```python
# ═════════════════════════════════════════════════════════════════════════════
# ZONE B: ACT I -- CALIBRATION
# ═════════════════════════════════════════════════════════════════════════════
```

---

## Migration Checklist for Existing Labs

Use this checklist when updating any existing lab to the standardized template.

### Phase 1: Add New Cells

- [ ] **Create Cell 2 (BRIEFING)** between the current HEADER and RECOMMENDED READING cells
  - Write 3 learning objectives (measurable, action-verb, specific)
  - Write prerequisites as compact inline list with @sec- references
  - Write the Core Question as a single interrogative sentence
  - Add duration breakdown (Act I / Act II)

- [ ] **Create Cell 20 (SYNTHESIS)** before the current LEDGER_HUD cell
  - Write 3 Key Takeaways (one per objective, with specific numbers)
  - Write "What's Next" pointer to the next lab
  - Write "Textbook and TinyTorch" connection
  - Optional: add self-assessment accordion

### Phase 2: Restructure Existing Cells

- [ ] **Renumber all cells** to match the 0-21 template numbering
  - Old Cell 2 (RECOMMENDED READING) becomes Cell 3 (READING)
  - Old Cell 3 (CONTEXT TOGGLE) becomes Cell 4 (CONTEXT_TOGGLE)

- [ ] **Update Act I banner** (Cell 5) to include the "why this matters" line
  - Add one sentence explaining the wrong prior being corrected
  - Keep the existing number badge + title + duration format

- [ ] **Update Act II banner** (Cell 12) to include the "why this matters" line
  - Add one sentence explaining the escalation from Act I

- [ ] **Extract takeaways from LEDGER_HUD** into the new SYNTHESIS cell (Cell 20)
  - Move `mo.callout` blocks containing "Key Takeaways" and "Connections"
  - Leave only ledger.save() logic and HUD HTML in Cell 21

### Phase 3: Consolidate Reveal Cells

- [ ] **Merge Act I reveal + reflection + MathPeek** into Cell 11 (ACT1_REVEAL)
  - Stack: prediction overlay, reflection radio, feedback callout, MathPeek accordion
  - If Marimo dataflow requires separation, keep as 11a/11b but use same visual grouping

- [ ] **Merge Act II reveal + reflection + MathPeek** into Cell 19 (ACT2_REVEAL)
  - Same consolidation pattern as Act I

### Phase 4: Update Comment Headers

- [ ] Replace all cell comment headers with standardized format:
  ```
  # ─── CELL N: CELL_NAME ───...
  ```

- [ ] Add zone boundary separators:
  ```
  # ═════════════════════════════════════════════════════════════════════════════
  # ZONE X: ZONE_NAME
  # ═════════════════════════════════════════════════════════════════════════════
  ```

### Phase 5: Verify

- [ ] All 4 zones present (Opening, Act I, Act II, Closing)
- [ ] BRIEFING cell has exactly 3 objectives, all measurable
- [ ] SYNTHESIS cell has exactly 3 takeaways, each with a specific number
- [ ] Core Question in BRIEFING cannot be answered without completing the lab
- [ ] Each takeaway maps to one objective (1-to-1 correspondence)
- [ ] "What's Next" names the next lab and establishes a causal link
- [ ] Act banners include "why this matters" line (not just title)
- [ ] Takeaways and connections are NOT in the LEDGER_HUD cell
- [ ] Total lab still fits 35-40 minute target
- [ ] No new cells add interaction time (BRIEFING and SYNTHESIS are read-only)

---

## Design Principles

### Why This Structure Works

1. **Cognitive framing.** The BRIEFING cell activates relevant prior knowledge before
   the first prediction. Students who know what they are looking for engage more deeply
   with the instruments. This is established pedagogy (advance organizers, Ausubel 1960).

2. **Prediction calibration.** The Core Question plants a seed of uncertainty. Students
   who read it think "I know the answer to this" -- and then discover in Act I that they
   do not. The gap between the briefing's question and the instrument's answer is the
   learning moment.

3. **Closure.** The SYNTHESIS cell provides resolution. The Core Question from the BRIEFING
   now has an answer. The takeaways crystallize what was vague. The "What's Next" transforms
   a finished lab into motivation for the next one.

4. **Consistency.** Every lab follows the same rhythm: brief, predict, discover, reflect,
   synthesize. Students who complete Lab 02 know exactly what to expect in Lab 05. This
   reduces cognitive overhead and lets students focus on content, not navigation.

### What This Template Does NOT Change

- The 2-Act structure (PROTOCOL invariant)
- The structured prediction lock pattern (radio/number, never free text)
- The failure state requirement in Act II
- The stakeholder message pattern
- The MathPeek accordion pattern
- The Design Ledger schema and save pattern
- The HUD footer bar
- The dark gradient header visual identity
- The deployment context toggle
- Any slider ranges, chart formulas, or physics constants

The template adds framing (BRIEFING, SYNTHESIS) and reorganizes existing content
(takeaways move out of LEDGER_HUD). It does not alter the pedagogical engine.
