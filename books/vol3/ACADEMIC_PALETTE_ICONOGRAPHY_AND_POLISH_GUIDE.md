# Academic Art Direction: Color Palette, Systems Iconography, Visual Polish Audit & CMOS Integration
**To:** Author & Production Team, *Agentic Machine Learning Systems* (Volume III, MLSysBook Series)
**From:** Senior Visual Editor & Textbook Art Director, Premier Academic Press (MIT Press / Morgan Kaufmann Standard)
**Date:** September 16, 2026
**Subject:** Publication-Grade Academic Color Palette, Systems Iconography Standards, Visual Polish Audit of 5 "v2" Figures, and CMOS 17th/18th Quarto Guidelines

---

## 1. Academic Textbook Color Scheme: Flagship Press Standard vs. SaaS Web UI

### The Problem with the Current "v2" Palette
The current palette across the v2 figures relies heavily on **Tailwind CSS / SaaS web dashboard defaults**:
- Electric Sky Blue (`#0284c7`, `#38bdf8`)
- High-Saturation Emerald Green (`#15803d`, `#16a34a`)
- Candy Amber (`#d97706`, `#f59e0b`)
- Vivid Electric Violet (`#7c3aed`, `#6d28d9`)
- Bright Alert Red (`#dc2626`, `#ef4444`)
- Pastel fill tints (`#eff6ff`, `#f0fdf4`, `#faf5ff`, `#fffbeb`)

**Why this fails academic textbook publishing standards:**
1. **The "SaaS Dashboard" Aesthetic:** High-saturation pastels and vibrant primaries look like a commercial cloud console (e.g., AWS, Supabase, Vercel) or a software pitch deck. They lack the gravitas, permanence, and intellectual weight of graduate texts (*Hennessy & Patterson*, *Saltzer & Kaashoek*, *Cormen et al.*).
2. **CMYK Gamut Clipping in Print:** Electric RGB cyans, neon purples, and saturated bright ambers fall outside the standard ISO Coated / GRACoL CMYK print gamut. When printed on press, these colors shift unpredictably into muddy greens, dull grays, or desaturated browns.
3. **Monochrome / Grayscale Legibility Failure:** Academic textbooks are frequently reprinted in black-and-white (library editions, international student paperbacks, or e-ink readers). The current pastel tints (`#eff6ff`, `#f0fdf4`, `#faf5ff`) all collapse to an identical $\approx 92\%$ luminance gray, completely erasing the visual distinction between subsystems.
4. **Accessibility (WCAG AA/AAA):** Light text on pastel fills or medium-tone borders against white backgrounds frequently fail the 4.5:1 (AA) and 7:1 (AAA) contrast ratios required for academic publishing.

---

### The Canonical Publication-Grade Academic Palette (MLSysBook Standard)

Academic computer systems diagrams rely on a **restrained semantic color architecture**: deep, grounded, desaturated tones paired with intentional neutral grays, calibrated for both **RGB screen display** and **CMYK press sheet stability**.

```
========================================================================================================================
CANONICAL ACADEMIC SYSTEMS PALETTE (Print CMYK & Digital RGB / Web)
========================================================================================================================
Semantic Role          Name              Hex Code    RGB (sRGB)      CMYK Equivalent   Monochrome (L*) Usage & Semantics
------------------------------------------------------------------------------------------------------------------------
Primary Architecture   Deep Oxford Navy  #1E293B     (30, 41, 59)    C=85 M=70 Y=45 K=50   L* = 17%    Main borders, titles, compute cores
Secondary Datapath     Slate Steel Blue  #334155     (51, 65, 85)    C=75 M=55 Y=35 K=30   L* = 27%    Datapath lines, register outlines
Memory & Buffers       Muted Slate Teal  #0F766E     (15, 118, 110)  C=85 M=30 Y=55 K=25   L* = 44%    Memory hierarchy, caches, DRAM
Control & Interrupt    Deep Burgundy     #991B1B     (153, 27, 27)   C=25 M=95 Y=85 K=25   L* = 28%    Preemption, traps, abort buses
Interconnect / Fabric  Warm Ochre / Umber#92400E     (146, 64, 14)   C=30 M=75 Y=100 K=25  L* = 35%    Buses, RDMA fabric, network loops
Verified / State Gate  Muted Forest Sage #166534     (22, 101, 52)   C=85 M=30 Y=95 K=30   L* = 38%    Verified oracles, committed states
Neutral Container Fill Cool Parchment    #F8FAFC     (248, 250, 252) C=3  M=2  Y=1  K=0    L* = 98%    Box fill (subtle container depth)
Accent Sub-Block Fill  Pale Slate Tint   #F1F5F9     (241, 245, 249) C=6  M=4  Y=3  K=0    L* = 96%    Active functional block fill
Boundary Stroke Rule   Muted Slate Rim   #CBD5E1     (203, 213, 225) C=20 M=14 Y=12 K=0    L* = 85%    Structural sub-box borders (1px)
Dark Text / Labels     Midnight Charcoal #0F172A     (15, 23, 42)    C=80 M=70 Y=55 K=65   L* = 8%     Primary labels, math variables
Secondary Text         Muted Iron        #475569     (71, 85, 105)   C=65 M=50 Y=40 K=15   L* = 35%    Bus widths, timing rates, specs
========================================================================================================================
```

#### Monochromatic Separation Guarantee:
Notice the **$L^*$ (Luminance)** values:
- Deep Oxford Navy ($L^* = 17\%$), Slate Steel Blue ($L^* = 27\%$), Deep Burgundy ($L^* = 28\%$), and Muted Forest Sage ($L^* = 38\%$) each map to distinctly different, dark ink densities on a grayscale plate.
- Even without color, a reader can distinguish an interrupt bus (dashed line + heavy black/burgundy ink) from a memory page table (solid teal line).
- **Rule of Thumb:** If a figure cannot be understood when printed in black-and-white on a standard laser printer, the color palette is defective.

---

## 2. Systems Iconography & Schematic Glyphs: Academic Standard vs. Slide Clipart

### The Danger: Clipart Contamination
Adding emoji (`⏱`, `🛡`, `⚡`, `🚀`, `💡`, `❌`, `✅`) or generic marketing vector icons (FontAwesome / Lucide flat icons) degrades an academic diagram into an infographic blog post. Graduate textbooks at MIT Press never use emojis or cartoon icons.

### The Canonical Computer Engineering Iconography (IEEE / ACM / VLSI Standard)
Instead of clipart, systems textbooks use **precise, standardized schematic symbols**:

```
+---------------------------------------------------------------------------------------------------------+
| SCHEMATIC GLYPHS FOR COMPUTER SYSTEMS TEXTBOOKS                                                         |
+---------------------------------------------------------------------------------------------------------+
| Systems Entity       Prohibited Slide Clipart  Canonical Academic Engineering Symbol                    |
+---------------------------------------------------------------------------------------------------------+
| Clock / Period (\tau) Stopwatch emoji (⏱)      Rising-edge square-wave clock pulse with phase notation: |
|                                                ┌─┐   ┌─┐                                                |
|                                                ┘ └───┘ └──  \Phi, \tau \sim 1\text{ ns}                 |
+---------------------------------------------------------------------------------------------------------+
| Memory / Register    Stacked database disks    Segmented bit-field register or 2D address matrix:       |
|                                                +----+----+----+----+                                    |
|                                                |Tag |Index| Offset |  R[31:0]                           |
|                                                +----+----+----+----+                                    |
+---------------------------------------------------------------------------------------------------------+
| Trap / Interrupt     Comic lightning bolt (⚡) Active-low hardware interrupt bubble or pulse strobe:   |
|                                                ─────o  \overline{\text{SIG\_PREEMPT}}                   |
|                                                or sharp step trigger with threshold level V_{th}         |
+---------------------------------------------------------------------------------------------------------+
| Isolation / Sandbox  Medieval shield (🛡)      Double-walled boundary stroke (2px outer, 1px inner)    |
|                                                or grounded guard ring with diagonal hash pattern (▨▨)   |
+---------------------------------------------------------------------------------------------------------+
| Comparator / Gate    Checkmark/cross (✓/✗)     IEEE triangle comparator or differential circle:         |
|                                                    + \                                                  |
|                                                -----> \  Delta > 0                                      |
|                                                    -  /                                                 |
+---------------------------------------------------------------------------------------------------------+
| Queue Structure      Generic horizontal boxes  CS-standard open-ended FIFO register with pointers:      |
|                                                   H                     T                               |
|                                                [in] ---> [ | | | | | ] ---> [out]                       |
+---------------------------------------------------------------------------------------------------------+
| Bus & Interconnect   Curved arrow              Thick orthogonal trunk with tick-mark width slashes:     |
|                                                ══════════/══════════  [63:0] or [Tokens, d]             |
+---------------------------------------------------------------------------------------------------------+
| Multiplexer / Switch Switch toggle icon        Trapezoidal MUX/DEMUX symbol with select lines:          |
|                                                \ 0 |                                                    |
|                                                 \  |=== Out                                             |
|                                                / 1 |                                                    |
+---------------------------------------------------------------------------------------------------------+
```

#### Vector Implementation Guidelines:
1. **Draw glyphs as native SVG paths:** Do not embed external PNG/WebP icons or icon font glyphs (which fail if the font is missing).
2. **Stroke matching:** The stroke width of all schematic glyphs (clock pulses, queue brackets, comparator triangles) must match the diagram's standard signal stroke (`stroke-width="1.5"` or `1.2`).
3. **Restrained scale:** Glyphs should never exceed $16 \times 16\text{ px}$ or $20 \times 20\text{ px}$. They serve as visual anchors, not decorative centerpieces.

---

## 3. Visual Polish Audit: Specific Flaws in the 5 "v2" Figures

A forensic examination of the rendered SVGs and PNGs reveals critical layout bugs, arrow collisions, and typographic inconsistencies that violate publication quality:

---

### Audit 1: `evolution_execution_units_v2.svg`
1. **Typographic Anarchy (13 Distinct Font Sizes):**
   - Uses `6.2`, `6.5`, `6.8`, `7.0`, `7.2`, `7.5`, `7.8`, `8.0`, `8.5`, `8.8`, `9.0`, `9.5`, and `10.0pt`.
   - **Remedy:** Normalize to exactly **3 sizes**: Titles = `9pt bold`, Component Labels = `7.5pt`, Timing/Units = `6.5pt`.
2. **Column 5 Border Asymmetry:**
   - Columns 1–4 have a border width of `1.2px` (`#cbd5e1`), while Column 5 has a heavy purple border of `2px` (`#7c3aed`). This makes the fifth column look like a "selected pricing tier" on a SaaS landing page.
   - **Remedy:** All five columns must have identical structural border weights (`1.2px` neutral slate); semantic distinction should be indicated by a subtle top-cap colored rule (2px) or clean header label.
3. **Microarchitecture Clock Pulse Collision (Col 1):**
   - The red clock wave `CLK (3 GHz)` at $y=114$ sits only $4\text{px}$ from the sub-box bottom stroke ($y=120$), with text clipping the waveform line.
4. **RPC Clock Alignment (Col 3):**
   - The orange circle at $x=450, y=70$ representing the timeout clock has its clock hands (`stroke="#d97706"`) slightly off-center ($x=450, y=68$), creating an eccentric wobble.
5. **Arrowhead Bleed into Box Strokes (Col 5):**
   - The blue arrow from `Policy (f_\theta)` to `Actuation` connects at $x=85$ to $x=105$, but the marker `arr-blue` penetrates $2\text{px}$ into the `Actuation` rectangle stroke because `refX` is set to `5` instead of `8`.

---

### Audit 2: `prefill_decode_disaggregation_v2.svg`
1. **Arrowhead Penetrating Top Container Border:**
   - The vertical blue arrow from `GLOBAL CLUSTER ORCHESTRATOR` down to `PREFILL WORKER POOL` originates at $y=112$ and terminates at $y=140$. The marker `arr-blue` directly impales the border stroke of the Prefill container, overlapping the border by $2.5\text{px}$.
2. **Micro-Fonts Down to 5.5pt:**
   - Contains text nodes at `5.5pt` (`KV Transfer Tradeoff Invariant` formula text). At 100% scale in a print PDF, 5.5pt is below the legal threshold for legible book typesetting (minimum allowable is 6.5pt, standard is 7.5pt).
3. **RDMA Big Arrow / Box Stacking Collision:**
   - The horizontal purple arrow inside the RDMA container ($y=235$) is drawn with `stroke-width="6"`. It passes directly behind the three packet boxes (`KV_L0..19`, etc.). While the boxes have white fill, the purple arrow head protrudes abruptly past `KV_L40..79` at $x=600$ with an awkward gap before touching the Decode Worker Pool border.
4. **Bottom Return Loop Routing:**
   - The orange return arrow at the bottom exits `TOOL ACTUATION & RE-ROUTING LOOP` on the left, travels up to $y=180$, and enters `PREFILL WORKER POOL`. However, the vertical line at $x=18$ is positioned only $6\text{px}$ from the outer SVG canvas margin, making it visually precarious.

---

### Audit 3: `progressive_autonomy_spectrum_v2.svg`
1. **Unmasked Text on Connector Arrows:**
   - The horizontal arrows connecting Stage 1 $\to$ Stage 2 and Stage 2 $\to$ Stage 3 have text labels (`Promote`, `Scale Up`) positioned directly on the arrow path line. Because there is no background `<rect>` mask behind the text, the arrow line cuts through the descenders of `p` and `g`.
2. **Bottom Abort Bus Penetration:**
   - The dashed red abort bus (`stroke-dasharray="4,3"`) exits Stage 3 at $y=490$, runs down to the bottom box, then heads left to Stage 1. The upward-pointing arrowhead at $x=135, y=490$ penetrates into the red box `If p >= 0.05: REJECT CANDIDATE`, intersecting the text boundary.
3. **Unequal Stage Spacing:**
   - Stage 1 is at $x=25$, Stage 2 is at $x=275$ (gap = 250px). Stage 3 is at $x=525$ (gap = 250px). Stage 4 is at $x=775$ (gap = 250px). While column origins are equal, the inner padding of cards varies from 8px to 14px, creating irregular ragged gutters.

---

### Audit 4: `tokenomics_memory_hierarchy_caching_v2.svg`
1. **Crowded Double-Headed DMA Arrows:**
   - Between Tier 1 (HBM) and Tier 2 (DRAM), the bidirectional DMA arrow spans a vertical gap of only $24\text{px}$ ($y=206$ to $y=230$). Squeezing the text `PCIe Gen5 x16 DMA (64 GB/s) · ~195 ms paging latency` horizontally adjacent to this arrow leaves only $2\text{px}$ of clearance on either side.
2. **Right-Side Pre-Declared Batch Bypass Collision:**
   - The red dashed line on the right side of the MLFQ box runs vertically at $x=505$. It sits only $4\text{px}$ away from the right border of the parent container ($x=510$), creating visual vibration with the outer stroke.
3. **Queue Box Padding Collapse:**
   - In Priority Queue 0, the FIFO packet boxes `req_901`, `req_902`, `req_903` have height $20\text{px}$ and width $55\text{px}$. The text inside is centered, but in Priority Queue 2, the label `batch_crawl_01` has 14 characters, causing the text to touch the left and right borders of its $60\text{px}$ box (zero internal horizontal padding).

---

### Audit 5: `five_part_agent_runtime_architecture_v2.svg`
1. **Severe Canvas Overcrowding (565 Words!):**
   - The canvas is packed edge-to-edge. Vertical margins between major subsystems are reduced to $8–12\text{px}$, completely eliminating structural breathing room.
2. **Inter-Column Channel Squeeze:**
   - The horizontal spacing between Column 1 (SPU), Column 2 (TCU), and Column 3 (C-MMU) is only $18\text{px}$. Inside this tiny $18\text{px}$ gutter, bidirectional arrows and four text labels (`Proposals`, `Context`, `Alloc/Page`, `Addresses`) are crammed at `7.2pt`. The text is nearly illegible and collides with arrow strokes.
3. **Perimeter Feedback Bus Edge Hugging:**
   - The purple dashed line representing the `Model Weight Update Bus (\theta_k+1)` runs along the outer perimeter: left at $x=12$, bottom at $y=605$. Given the canvas size of $1020 \times 620$, a line at $y=605$ leaves only $15\text{px}$ from the physical edge of the SVG. If placed inside a Quarto PDF margin, this line risks being clipped by the page trim box.

---

## 4. Chicago Manual of Style (CMOS 17th & 18th Edition) Integration in Quarto

To ensure Volume III meets the editorial standards of MIT Press and Morgan Kaufmann, all figure presentations must strictly follow **CMOS Chapter 3 ("Illustrations and Tables")** as implemented in Quarto (`.qmd`).

### A. The Structural Anatomy of a CMOS Figure
CMOS establishes a clear hierarchy between the illustration, its title, its caption (legend), and the source note:

```
[Figure Canvas / Graphic: Pure visual geometry, zero burned-in titles]

Figure 11.4. Disaggregated prefill and decode cluster architecture.
Compute-specialized P-Nodes evaluate input prompts using parallel GEMM
tensors (AI > I*) and stream KV cache activations via zero-copy RDMA to
memory-bandwidth-specialized D-Nodes. D-Nodes manage PagedAttention page
tables for autoregressive token generation (AI < I*).
```

#### CMOS Caption Rules (CMOS 3.21–3.31):
1. **No Burned-in Titles in the Canvas:** Never include a title header inside the SVG artwork (e.g., no `"GLOBAL CLUSTER ORCHESTRATOR & DISPATCHER"` banner across the top). The title is typeset as part of the caption text in the book layout.
2. **Figure Numbering & Punctuation:**
   - Double-numbered by chapter: `Figure 11.4` (Volume III chapter-based standard).
   - In CMOS, the label is followed by a period: `Figure 11.4.`
3. **The Lead-in Phrase (Title):** A short descriptive phrase follows the figure number. In academic engineering texts, this is either bolded or set in sentence style ending with a period.
4. **The Explanatory Sentence (Walkthrough):** Full grammatical sentences explaining the mechanism, defining all mathematical symbols used in the figure, and directing the reader's eye.

---

### B. Canonical Quarto (`.qmd`) Implementation

In Quarto, academic figures must be implemented using **cross-referenceable div blocks** (`::: {#fig-...}`), which enable automated numbering, list of figures generation, LaTeX float placement, and responsive HTML figures:

```markdown
::: {#fig-prefill-decode}
![](images/svg/prefill_decode_disaggregation_v2.svg){fig-alt="Cluster schematic showing compute-bound prefill nodes transmitting KV cache tensors via zero-copy RDMA to memory-bound decode nodes" width=100%}

Disaggregated prefill-decode cluster architecture. Compute-specialized P-Nodes
process prompt tokens across parallel GEMM tiles ($\text{AI} > \mathcal{I}^*$)
and stream generated KV activations across the zero-copy RDMA network fabric to
decode worker nodes. D-Nodes ingest KV blocks into PagedAttention tables for
single-token autoregressive generation ($\text{AI} < \mathcal{I}^*$), eliminating
head-of-line blocking while preserving memory bandwidth.
:::
```

#### Why this Quarto syntax is mandatory:
- **`{#fig-prefill-decode}` on the div:** Tells Quarto to treat the block as a formal float. In PDF export, Quarto translates this directly into a LaTeX `\begin{figure}...\caption{...}\end{figure}` environment.
- **`fig-alt` attribute:** Essential for web accessibility (WCAG 2.1) and screen-reader accessibility in digital ePub/HTML formats.
- **`width=100%`:** Ensures the SVG scales responsively across both single-column web and double-column print page spreads.

---

### C. In-Text Callout Rules (CMOS 3.9 & 3.10)

1. **Every Figure Must Have an In-Text Callout:**
   A figure must never appear unannounced. In CMOS, every illustration must be explicitly introduced in the running prose *before* or *immediately adjacent to* its physical position.
2. **Never Use Positional References:**
   - ❌ *"As seen in the figure above..."*
   - ❌ *"The diagram below illustrates..."*
   - **Why:** In print publishing, page layout engines (LaTeX/InDesign) float figures to the top or bottom of pages to avoid awkward whitespace. A figure may end up on the facing page or the following spread.
3. **Correct CMOS Phrasing:**
   - Standard grammatical reference: *"As illustrated in Figure 11.4, the disaggregation of..."*
   - Parenthetical reference: *"...mitigating head-of-line blocking during long-context bursts (see fig. 11.4)."*
4. **Quarto Cross-Reference Syntax:**
   - Always use `@fig-label`:
     ```markdown
     As illustrated in @fig-prefill-decode, separating prompt processing from token
     generation restores compute efficiency...
     ```
   - Quarto automatically compiles `@fig-prefill-decode` to `"Figure 11.4"` in PDF and HTML, maintaining active hyperlinks.

---

### D. Multi-Panel Subfigures (CMOS 3.23)

For complex multi-part comparisons (such as Figure 1's CPU vs. OS vs. RPC vs. Model vs. Agent), CMOS recommends labeled sub-panels: *(a)*, *(b)*, *(c)*, *(d)*, *(e)*.

In Quarto, this is authored cleanly using layout grids:

```markdown
::: {#fig-execution-units layout="[[1,1,1], [1,1]]"}
![CPU instruction datapath](images/svg/unit_cpu.svg){#fig-unit-cpu}

![OS virtual memory translation](images/svg/unit_os.svg){#fig-unit-os}

![Distributed RPC interface](images/svg/unit_rpc.svg){#fig-unit-rpc}

![Transformer GEMM inference](images/svg/unit_model.svg){#fig-unit-model}

![Closed-loop agent trajectory](images/svg/unit_agent.svg){#fig-unit-agent}

Evolution of systems execution units across latency scales. *(a)* Silicon CPU ALU
operating at single clock edges ($\sim 1\text{ ns}$). *(b)* OS process context
virtualization mediated by MMU page tables ($\sim 1\ \mu\text{s}$). *(c)* Distributed
RPC framing with socket buffers and network timeouts ($\sim 10\text{ ms}$). *(d)* Transformer
matrix inference computing autoregressive token distributions ($\sim 100\text{ ms}$).
*(e)* Closed-loop agentic trajectory orchestrating iterative tool actuation,
deterministic verification, and state logging ($\sim 10^1\text{--}10^4\text{ s}$).
:::
```

---

## 5. Master Production Checklist for Volume III Figures

Before any SVG is signed off for final publication in Volume III, it must pass this 10-point Art Director Checklist:

- [ ] **1. Text Density:** Total canvas word count is $\le 50$ words (hard ceiling: 100 words).
- [ ] **2. Zero Prose Sentences:** No complete sentences, verbs, terminal periods, or bullet points (`•`) on canvas.
- [ ] **3. Palette Discipline:** Uses canonical MIT Press palette (Oxford Navy `#1E293B`, Slate Blue `#334155`, Slate Teal `#0F766E`, Burgundy `#991B1B`, Warm Ochre `#92400E`). Zero high-saturation Tailwind pastels.
- [ ] **4. Grayscale Verified:** Image verified under monochrome filter ($L^*$ luminance levels guarantee distinct components).
- [ ] **5. Strict Typographic Hierarchy:** Maximum of 3 font sizes used across the entire diagram (e.g., 9pt, 7.5pt, 6.5pt).
- [ ] **6. Arrow & Marker Clearance:** Arrow markers have `refX` calibrated to avoid penetrating target box strokes; zero text collisions on connector paths.
- [ ] **7. Systems Glyphs Only:** Uses IEEE/ACM schematic symbols (square-wave clock pulse, FIFO queue brackets, comparator triangles). Zero emoji or clipart.
- [ ] **8. No Section References:** No `(§11.5)` or chapter numbers drawn inside the SVG canvas.
- [ ] **9. No Transient Hardware SKUs:** No `H100`, `B200`, or spot dollar costs baked into vector paths; uses parameterized variables ($B_{\text{mem}}, T_{\text{FLOPS}}, C_{\text{rent}}$).
- [ ] **10. Quarto & CMOS Conformance:** Authored inside `::: {#fig-...}` with complete CMOS-compliant caption, `fig-alt` text, and `@fig-...` in-text callout.

---
*Signed,*
**Senior Visual Editor & Textbook Art Director**
*Academic Computer Systems Publishing Division*
