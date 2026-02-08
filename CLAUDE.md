# Claude Instructions for mlsysbook-vols

Project-wide conventions for AI assistance.

## Textbook Prose Style Guide

This book must read like a **textbook**, not a report, white paper, or blog post. The introduction chapter sets the gold standard. Every chapter must match its narrative voice, pedagogical structure, and prose flow.

### Voice and Tone

Write in the voice of an experienced professor teaching a graduate-level course:

- **Explanatory, not declarative.** Don't just state facts; explain *why* they matter and *how* they connect to what the reader already knows. Every section should answer "why should I care?" before "what is this?"
- **Narrative, not encyclopedic.** Prose should flow as a continuous argument, not a disconnected collection of facts. Each paragraph should connect to the previous one.
- **Precise but accessible.** Use technical terms precisely, but define them on first use (preferably with etymological footnotes). Avoid jargon without explanation.
- **Active voice preferred.** "The optimizer updates weights" not "weights are updated by the optimizer."
- **Second person sparingly.** Use "we" (author + reader together) for shared reasoning. Use "you" only in learning objectives and checkpoint questions.

### Chapter Structure (Required Elements)

Every chapter MUST include these structural elements, in this order:

1. **Cover image + mini-TOC** (`\chapterminitoc`)
2. **Purpose section** (`## Purpose {.unnumbered}`) — A motivating question in italics, followed by 1-2 paragraphs grounding the chapter's importance
3. **Learning Objectives** — in a `.callout-tip` titled "Learning Objectives", with 5-8 bullet points using action verbs (Explain, Apply, Distinguish, Describe, Compare)
4. **Body sections** (`##`, `###`, `####`) — the chapter's content
5. **Fallacies and Pitfalls** — using the `**Fallacy:**`/`**Pitfall:**` format (Rule 4)
6. **Summary** — in a `.callout-takeaways` titled "Key Takeaways", with 4-7 bullet points
7. **Chapter Connection** — in a `.callout-chapter-connection` titled with a bridge phrase (e.g., "From Training to Deployment"), explaining what comes next and why

### Section Transitions (Critical)

**Every major section (`##`) must end with a transition paragraph** that bridges to the next section. This is the single most important difference between textbook prose and report prose. Readers should never encounter a `##` header without understanding why the book is moving to that topic.

```markdown
<!-- ❌ BAD — abrupt section boundary -->
...final paragraph of previous section about data pipelines.

## Model Training Architecture

The training loop consists of...

<!-- ✅ GOOD — explicit bridge paragraph -->
...final paragraph of previous section about data pipelines.

With data infrastructure established, the natural question becomes: how do we actually use this data to train models? The answer requires understanding not just the algorithms but the systems architecture that makes large-scale training possible.

## Model Training Architecture

The training loop consists of...
```

### Concrete-Before-Abstract Pattern

Introduce concrete examples BEFORE formal definitions. Let the reader build intuition from a familiar scenario, then formalize it.

```markdown
<!-- ❌ BAD — definition first, example later -->
## Batch Normalization

Batch normalization normalizes activations across the batch dimension...
[formal math]
For example, consider training a deep network...

<!-- ✅ GOOD — concrete scenario first -->
## Batch Normalization

Consider training a 50-layer ResNet. After a few epochs, gradients in early layers
either explode to infinity or vanish to zero, making the network untrainable...

This instability arises from a phenomenon called *internal covariate shift*...
[then formal definition]
```

### Callout Box Types and Usage

Use callout boxes consistently for their designated purposes:

| Callout Type | Purpose | When to Use |
|---|---|---|
| `.callout-tip` | Learning Objectives | Once per chapter, after Purpose section |
| `.callout-example` | Worked examples | Concrete scenarios with calculations or code |
| `.callout-perspective` | Deeper conceptual insights | "Why" discussions, historical context, philosophical points |
| `.callout-notebook` | Computational exercises | Step-by-step calculations with **bold labels** for structure |
| `.callout-checkpoint` | Self-assessment questions | 2-4 per chapter, after major conceptual blocks |
| `.callout-definition` | Formal definitions | Precise, quotable definitions of key concepts |
| `.callout-takeaways` | Chapter summary | Once per chapter, at the end |
| `.callout-chapter-connection` | Bridge to next chapter | Once per chapter, final element before quiz |

### Footnotes for Key Terms

First occurrences of important technical terms should get **etymological/contextual footnotes** that include:

- The term's origin (Greek/Latin roots, who coined it, when)
- A 1-2 sentence systems perspective on why it matters
- Cross-reference to the chapter where it's covered in depth

```markdown
<!-- ✅ GOOD — rich footnote -->
Stochastic gradient descent[^fn-sgd] estimates gradients from small random batches...

[^fn-sgd]: **Stochastic Gradient Descent (SGD)**: "Stochastic" derives from Greek
*stochastikos* (able to guess); "gradient" from Latin *gradiens* (stepping). Rather
than computing gradients over the entire dataset, SGD estimates them from small
random batches. See @sec-ai-training for details.
```

### Quantitative Grounding

Use computed Python values (via `physx.constants` and `physx.formatting`) instead of hardcoded numbers wherever possible. This ensures consistency across chapters and makes values updatable from a single source of truth.

```markdown
<!-- ❌ BAD — hardcoded number -->
GPT-3 has 175 billion parameters.

<!-- ✅ GOOD — computed from physx constants -->
GPT-3 has `{python} gpt3_params_billion_str` billion parameters.
```

### Table Captions

Every table MUST have a descriptive caption that explains not just what the table shows but *why it matters pedagogically*. Include a `{#tbl-...}` cross-reference ID.

```markdown
<!-- ❌ BAD — minimal caption -->
: Comparison of optimizers {#tbl-optimizers}

<!-- ✅ GOOD — rich caption -->
: **Optimizer Comparison**: Each optimizer trades convergence speed against memory
overhead. Adam's adaptive learning rates require 3× the memory of SGD but converge
2-5× faster on transformer architectures, explaining why it dominates modern
training despite its memory cost. {#tbl-optimizers}
```

### Prose Flow: Avoid Report-Style Writing

**Report style** fragments information into disconnected bullets and bold-labeled paragraphs. **Textbook style** weaves information into a narrative argument.

```markdown
<!-- ❌ BAD — report style (disconnected bullets) -->
The key challenges of distributed training are:

- **Communication overhead**: Gradients must be synchronized...
- **Load imbalance**: Heterogeneous hardware...
- **Fault tolerance**: GPU failures interrupt...

<!-- ✅ GOOD — textbook style (flowing prose) -->
Distributed training introduces three interrelated challenges. Communication overhead
dominates when gradients must be synchronized across thousands of GPUs, consuming up
to 40% of total training time for large models. This overhead compounds with load
imbalance: heterogeneous hardware configurations mean some nodes finish their
computations earlier and idle while waiting. Most critically, fault tolerance becomes
a first-class concern—at the scale of thousands of GPUs, hardware failures are not
exceptional events but statistical certainties.
```

**Exception**: Bulleted lists ARE appropriate for:
- Learning objectives
- Enumerated steps in a procedure
- Comparison tables
- Items in callout boxes
- List items with bold lead terms (Rule 5)

## QMD Inline Python: The Just-in-Time Pattern

When working with `.qmd` files, Python compute cells should follow the **just-in-time pattern**: small, focused cells placed immediately before the prose that uses their values.

### Cell Documentation Format

Every compute cell should have a documentation header box:

```python
```{python}
#| echo: false
#| label: descriptive-name
# ┌─────────────────────────────────────────────────────────────────────────────
# │ CELL NAME IN CAPS
# ├─────────────────────────────────────────────────────────────────────────────
# │ Context: Which section/callout/figure uses these values
# │
# │ Why: 2-3 sentences explaining the pedagogical purpose of this calculation
# │
# │ Imports: physx.constants (LIST_CONSTANTS), physx.formatting (fmt)
# │ Exports: var1_str, var2_str
# └─────────────────────────────────────────────────────────────────────────────
from physx.constants import CONSTANT1, CONSTANT2
from physx.formatting import fmt

# --- Inputs (description of source) ---
input_var = 100                          # description

# --- Outputs (formatted strings for prose) ---
output_value = input_var * 2
output_str = fmt(output_value, precision=0)  # e.g. "200" units
` ` `
```

### Inline Python with LaTeX Math

When combining Python values with mathematical notation, use **simple mixing**:

```markdown
<!-- ✅ GOOD - Mix inline Python with LaTeX $...$ -->
`{python} params_b_str` $\times 10^9$ $\times$ `{python} bytes_str` bytes = `{python} result_str` GB
```

**Key rules:**
- Python variables hold simple formatted strings (no LaTeX)
- LaTeX `$...$` wraps math symbols (`×`, `10^9`, `\times`)
- Use Unicode `×` in tables (tables don't process LaTeX)

### What NOT To Do

```markdown
<!-- ❌ BAD - Raw values inside $...$ math blocks -->
$T = `{python} value`$ ms  <!-- Decimals get stripped! -->

<!-- ❌ BAD - Inline f-string formatting -->
`{python} f"{value:.2f}"`

<!-- ❌ BAD - Function calls inline -->
`{python} fmt(value, "ms", 3)`

<!-- ❌ BAD - Plain text exponents -->
`{python} value` × 10^9 × 2  <!-- 10^9 won't render as superscript! -->
```

### LaTeX Formatting Rules

| Context | Correct Format | Wrong Format |
|---------|----------------|--------------|
| Prose | `$10^{12}$` | `10^12` |
| Tables | `×` (Unicode) | `\times` or `$\times$` |
| Python strings | `f"{value}"` | `f"$10^{value}$"` |

### Validation

Run validation to check all inline references resolve:
```bash
python3 book/quarto/physx/validate_inline_refs.py --verbose
```

## QMD Section Headers and Bold Text Style Guide

Consistent formatting of section headers and bold text across all `.qmd` chapters.

### Rule 1: Headers for Section Divisions — Never Substitute Bold Text

If a bold-start paragraph introduces a new topic that gets its own paragraph(s) of discussion, it **must** be a proper header (`####`, `#####`) at the appropriate level. Bold text must never act as a pseudo-header.

**The test:** "Does this introduce a distinct topic with its own paragraph(s) of discussion?" If yes, it needs a header.

```markdown
<!-- ❌ BAD — bold text acting as a section header -->
**Tensor Core Architecture.** Modern GPUs include dedicated...

<!-- ✅ GOOD — proper header -->
#### Tensor Core Architecture

Modern GPUs include dedicated...
```

```markdown
<!-- ❌ BAD — bold paragraph lead as sub-section -->
**Data Quality and Heterogeneity** present the first hurdle...

<!-- ✅ GOOD — proper header -->
##### Data Quality and Heterogeneity

Data quality and heterogeneity present the first hurdle...
```

Note: When converting `**Bold Title.**` or `**Bold Title**` to a header, remove the bold markers and any trailing period. The first sentence of the following paragraph should NOT repeat the header text — rewrite to flow naturally from the header.

**Important exception — too granular for headers:** If a parent section (e.g., `#### Data Challenges`) already names the theme, and the items within it are each only 1-2 paragraphs, do NOT create `#####` sub-headers for each item. Instead, write them as flowing prose with the key term in natural emphasis (italic or just the first word of the sentence). Headers should mark genuine navigational divisions, not every paragraph topic.

```markdown
<!-- ❌ BAD — headers too granular, fragmenting short prose -->
#### Data Challenges: Quality, Scale, and Drift
##### Data Quality and Heterogeneity
Real-world data is often noisy...
##### Scale and Infrastructure
Scale requirements compound...
##### Data Drift
Data drift creates...

<!-- ✅ GOOD — flowing prose under a single header -->
#### Data Challenges: Quality, Scale, and Drift
Real-world data is often noisy and inconsistent, presenting the first hurdle...

Scale and infrastructure requirements compound these challenges...

Data drift creates an ongoing operational burden...
```

### Rule 2: Bold Leads ARE Allowed Inside Callouts

Inside `.callout-*` boxes (notebooks, examples, perspectives), bold labels provide internal structure. These are contained pedagogical units, not chapter sections:

```markdown
<!-- ✅ GOOD — internal callout structure -->
::: {.callout-notebook title="Training GPT-3"}
**The Variables**: ...
**The Calculation**: ...
**The Systems Conclusion**: ...
:::
```

### Rule 3: Bold Leads ARE Allowed in Parallel Definition-Style Lists

When introducing 3+ parallel items in quick succession (each 1-2 sentences), bold leads work as a lightweight definition list:

```markdown
<!-- ✅ GOOD — parallel definition items, each 1-2 sentences -->
**Cloud ML** deploys models on datacenter GPUs with virtually unlimited compute.
**Edge ML** runs inference on local hardware near the data source.
**Mobile ML** targets smartphones with strict power and thermal budgets.
**TinyML** targets microcontrollers with kilobytes of memory.
```

**The test:** Each item is 1-2 sentences max, and they form a clearly parallel structure. If any item expands to a full paragraph or more, convert ALL items to headers.

### Rule 4: Fallacy/Pitfall Format Is Standard

The Fallacies and Pitfalls section at the end of every chapter uses this established format. Keep it:

```markdown
**Fallacy:** *One deployment paradigm solves all ML problems.*

Explanation paragraph...

**Pitfall:** *Minimizing computational resources minimizes total cost.*

Explanation paragraph...
```

### Rule 5: Bold Terms in List Items Are Fine

Standard list formatting with a bold lead term:

```markdown
<!-- ✅ GOOD — bold term in list item -->
1. **Checkpointing**: saves model state periodically...
- **Forward activations**: stored during the forward pass...
```

### Rule 6: No Bold-Start Paragraphs in Flowing Body Text

Outside of callouts, definition lists, Fallacy/Pitfall sections, and list items, do NOT start a paragraph with bold text. Options:

```markdown
<!-- ❌ BAD — bold lead in body text -->
**An important caveat.** The Iron Law assumes...

<!-- ✅ Option A — just write it as prose -->
An important caveat: the Iron Law assumes...

<!-- ✅ Option B — if it's truly important, use a callout -->
::: {.callout-important}
The Iron Law assumes...
:::

<!-- ✅ Option C — if it introduces a new topic, make it a header -->
#### Important Caveat
The Iron Law assumes...
```

### Rule 7: Header Hierarchy Must Be Strict

Never skip heading levels:

- `##` — Major chapter sections (appear in TOC)
- `###` — Sub-sections within a major section
- `####` — Sub-sub-sections
- `#####` — Fine-grained topics (use sparingly)
- `######` — Avoid; restructure instead

### Rule 8: No Lone Sub-Headers

If a section header has exactly ONE child header at the next level (no siblings), remove the child — it fragments prose without adding navigational value. The content should flow directly under the parent.

```markdown
<!-- ❌ BAD — lone child header -->
### The Load Balancer Layer
#### Impact on Queuing Analysis
Content here...
### Next Section

<!-- ✅ GOOD — content flows under parent -->
### The Load Balancer Layer
Content about queuing analysis here...
### Next Section
```

If the lone child has a `{#sec-...}` cross-reference ID, preserve it as an anchor: `[]{#sec-...}`.

### Rule 9: Deciding Header Level for Conversions

When converting a bold-start paragraph to a header, choose the level that fits the local hierarchy:

- If inside a `##` section with no `###` siblings, use `###`
- If inside a `###` section, use `####`
- If inside a `####` section, use `#####`
- Never introduce a header level that skips over its parent

### Rule 10: `##` Section Headers Must Be Concise Noun Phrases

All `##`-level headers (except the structural `Purpose`, `Fallacies and Pitfalls`, and `Summary` headers) must follow these constraints:

- **Maximum 6 words, 45 characters** (whichever is hit first)
- **Noun phrase, not a sentence** — no verbs in imperative/gerund form acting as the main clause
- **No articles at the start** — drop leading "The" unless it's part of a proper name (e.g., "The Iron Law")
- **No "Putting It All Together"** — this phrase adds no information; use a descriptive noun phrase instead
- **Colon subtitles are allowed** when the base topic alone would be too vague (e.g., "CNNs" alone is insufficient). Format: `Topic: Subtitle`. Both parts must be noun phrases.
- **No ampersands (`&`)** — use "and" (but prefer restructuring to avoid "and" entirely)
- **No stage/phase suffixes** — drop trailing "Stage", "Phase", "Step" when the context is obvious from the chapter

```markdown
<!-- ❌ BAD — too long, sentence-like -->
## Organizing ML Systems Engineering: The Five-Pillar Framework
## How ML Systems Differ from Traditional Software
## Putting It All Together: Anatomy of a Training Step

<!-- ✅ GOOD — concise noun phrases -->
## Five-Pillar Framework
## ML vs. Traditional Software
## Anatomy of a Training Step
```

**Decision tree for shortening:**

1. Drop a leading "The" (e.g., "The Serving Paradigm" → "Serving Paradigm")
2. Drop filler phrases ("Putting It All Together", "Introduction to", "Understanding the")
3. Remove "and" by restructuring or picking the primary concept
4. Abbreviate compound nouns (e.g., "Machine Learning Operations" → "ML Operations")
5. If still too long, extract the topic's core noun phrase

### Summary Decision Tree

```
Is the bold text inside a .callout-* box?
  → YES: Keep as bold label (Rule 2)
  → NO: Continue...

Is it a Fallacy/Pitfall label?
  → YES: Keep as **Fallacy:**/**Pitfall:** (Rule 4)
  → NO: Continue...

Is it a bold term in a numbered/bulleted list item?
  → YES: Keep as bold list item (Rule 5)
  → NO: Continue...

Is it one of 3+ parallel items, each 1-2 sentences?
  → YES: Keep as definition-style list (Rule 3)
  → NO: Continue...

Does it introduce a topic with its own paragraph(s)?
  → YES: Convert to header at appropriate level (Rule 1, 9)
  → NO: Remove bold or rewrite as plain prose (Rule 6)

After converting, check: is the new header a lone child?
  → YES: Remove it — let content flow under parent (Rule 8)

Are there 3+ parallel named concepts with multi-paragraph discussion?
  → YES: Give each its own header (sibling headers are justified)
  → Example: Covariate Shift, Label Shift, Concept Drift → each gets ####
```
