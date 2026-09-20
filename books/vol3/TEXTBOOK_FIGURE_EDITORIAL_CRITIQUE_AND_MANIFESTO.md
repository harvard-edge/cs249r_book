# Editorial Critique & Art Direction: Volume III "v2" Figures
**To:** Author, *Agentic Machine Learning Systems* (Volume III, MLSysBook Series)
**From:** Senior Visual Editor & Textbook Art Director, Premier Academic Press (MIT Press / Morgan Kaufmann Standard)
**Date:** September 16, 2026
**Subject:** Visual Audit of Volume III "v2" SVGs, Root-Cause Analysis of Canvas Text Density, and the *Text-Minimal Academic Figure Manifesto*

---

### Executive Verdict

> **The Author's Concern:**
> *"there is so much text in those images so that's what i am worried about... see if they are textbook cause there is so much text in those images"*

**Editorial Verdict: The author's instinct is 100% correct.**

The newly authored "v2" figures are **not** graduate textbook schematics. They are **PowerPoint presentation slides and executive summary cards masquerading inside an SVG container**.

While the previous audit correctly identified that "v1" lacked dataflow lines, the "v2" iterations made a classic textbook production error: **instead of drawing the underlying hardware and systems mechanisms, the author drew boxes and filled them with dense bullet points, contractual requirements, failure semantics, formulas, and historical citations.**

Across the five audited v2 SVGs, there are **407 text elements** containing **1,786 words**—literally two full printed book pages of prose trapped inside vector graphics. In several of these figures, **75% to 85% of the canvas area is occupied by typography**, leaving almost no room for actual schematic geometry, bus topologies, datapath circuits, or state transitions.

If printed in a flagship graduate textbook like Hennessy & Patterson's *Computer Architecture: A Quantitative Approach* or Saltzer & Kaashoek's *Principles of Computer System Design*, these figures would be immediately rejected by the editorial board as unreadable, unmaintainable, and pedagogically counterproductive.

---

## 1. Forensic Census: The 5 "v2" Figures

```
========================================================================================
FIGURE CANVAS FORENSIC CENSUS (Volume III v2 Batch)
========================================================================================
Figure Asset                               Text Nodes  Word Count  Text Area  Status
----------------------------------------------------------------------------------------
evolution_execution_units_v2.svg           102 nodes   301 words   ~75%       REJECT (Slide Deck)
prefill_decode_disaggregation_v2.svg        59 nodes   311 words   ~70%       REJECT (Card Deck)
progressive_autonomy_spectrum_v2.svg        80 nodes   323 words   ~85%       REJECT (Text Table)
tokenomics_memory_hierarchy_caching_v2.svg  50 nodes   286 words   ~65%       REJECT (Text Heavy)
five_part_agent_runtime_architecture_v2.svg 116 nodes   565 words   ~80%       REJECT (Syllabus Card)
----------------------------------------------------------------------------------------
TOTALS                                     407 nodes  1,786 words  ~75% avg   5 / 5 Unviable
========================================================================================
```

### Why Did This Happen?
When transitioning from v1 to v2, the visual author attempted to make the diagrams "rigorous" by transcribing the chapter's conceptual frameworks directly into the graphic. But **rigor in a textbook diagram comes from geometric precision, structural dataflow, and exact architectural topology—never from the word count inside the boxes.**

---

## 2. Why Real Textbook Figures Have Almost Zero Prose on Canvas

Flagship graduate computer engineering texts (*Hennessy & Patterson*, *Saltzer & Kaashoek*, *Silberschatz*, *Cormen et al.*) follow an iron discipline regarding canvas typography for four non-negotiable reasons:

### A. The Cognitive Science of Diagrams (Split-Attention Effect)
Educational psychology (Mayer's *Multimedia Principle* and Sweller's *Cognitive Load Theory*) proves that diagrammatic reasoning relies on the brain's spatial visual channel, while reading text relies on the phonological channel.
- When a diagram presents **geometry, arrows, and spatial hierarchy**, the visual cortex processes relationships in parallel.
- When a diagram forces the reader to read **15-word bullet points inside 12 adjacent boxes**, the visual channel is shut down. The reader is forced into sequential text parsing while constantly scanning across visual borders. This creates severe **split-attention friction**, blinding the reader to the actual architectural relationship.

### B. Hardware & Implementation Specification Rot
Consider `prefill_decode_disaggregation_v2.svg`, which has `"Hardware: High-TFLOPS Tensor Cores (H100 SXM5 / B200)"` and `tokenomics_memory_hierarchy_caching_v2.svg`, which specifies `"Holding Rent: $1.39 × 10⁻⁵ / GB·s ($4.00/hr lease equivalent)"`.
- Textbooks like Hennessy & Patterson are engineered for a **10- to 15-year shelf life**.
- Physical cloud SKUs (H100, B200) and spot prices ($4.00/hr) become obsolete within 18 months.
- When transient commercial details are baked into vector paths, the book ages prematurely. Canonical figures use invariant parameter symbols ($B_{\text{mem}}$, $T_{\text{FLOPS}}$, $C_{\text{rent}}$, $\tau_{\text{DMA}}$), reserving concrete hardware instances for accompanying comparison tables or captions.

### C. Typesetting, Responsive Layout & Internationalization
- **Vector Text is Dead Text:** Text inside SVG `<text>` elements cannot be searched reliably, cannot be parsed cleanly by screen readers, does not wrap dynamically, and ignores the LaTeX font metrics and baseline grid of the printed page.
- **Foreign Translation Barrier:** Academic press titles are licensed worldwide (e.g., Japanese, Chinese, Korean, and German translations). Translating an SVG with 565 words of English prose requires manual redraw by foreign typesetters, resulting in broken boxes and overlapping text. A diagram with clean mathematical labels ($\mathbf{x}, \mathbf{z}, a_t, Q_0$) requires **zero translation**.

### D. The Canonical Tripartite Division of Labor
In masterwork textbooks, content is partitioned into three distinct, non-overlapping media:

```
+-----------------------------------------------------------------------------------------+
| CANONICAL DIVISION OF LABOR IN FLAGSHIP COMPUTER SYSTEMS TEXTBOOKS                      |
+-----------------------------------------------------------------------------------------+
| 1. THE FIGURE CANVAS                                                                    |
|    • Sole Purpose: Structural topology, datapath, timing, state automata, physical geometry|
|    • Elements: Bounded hardware modules, registers, buses, signal lines, mathematical   |
|      variables, bit-widths, state bubbles, coordinate curves.                           |
|    • Strict Rule: ZERO complete sentences. NO bullet points. NO historical citations.    |
+-----------------------------------------------------------------------------------------+
| 2. THE FIGURE CAPTION                                                                   |
|    • Sole Purpose: The narrative walkthrough (50–150 words).                            |
|    • Elements: Guides the reader's eye ("Notice that the RDMA engine bypasses..."),     |
|      defines notation, explains the mechanism, and states the primary quantitative takeaway|
|    • Strict Rule: Never repeat the caption inside the canvas.                           |
+-----------------------------------------------------------------------------------------+
| 3. THE CHAPTER BODY TEXT                                                                |
|    • Sole Purpose: Derivation, rigorous theory, engineering trade-offs, and proofs.     |
|    • Elements: Formal contracts, failure models, historical citations, empirical data.  |
|    • Strict Rule: References the figure by label (e.g., "As shown in Figure 11.4...").   |
+-----------------------------------------------------------------------------------------+
```

---

## 3. Canonical Rules for Labels, Symbols, and Signal Names

When examining Hennessy & Patterson's legendary MIPS/RISC-V datapath diagrams (e.g., Figure 4.17 in *Computer Organization and Design*):
1. **Modules are labeled with compact acronyms or functional names:** `ALU`, `RegFile`, `PC`, `Instruction Memory`, `MUX`, `Data Memory`, `TLB`, `MMU`.
2. **Buses have explicit bit-widths or vector dimensions:** Shown by a slash across the line with a width annotation: `[31:0]`, `[63:0]`, `[B, d]`, `[1, S]`.
3. **Control signals are single capitalized words or camelCase:** `RegWrite`, `ALUSrc`, `MemtoReg`, `Branch`, `Preempt`, `Valid`, `Ready`.
4. **Timing and physical rates are compact numbers with units:** `1 ns`, `3.35 TB/s`, `100 µs`, `64 GB/s`, `τ = 2.0 s`.
5. **Formulas are isolated algebraic expressions:** $\mathbf{z} = \mathbf{W}\mathbf{x}$, $S_{\text{KV}} \propto 2L$, $\text{AI} > \mathcal{I}^*$, $p < 0.05$.
6. **No Section Numbers on Canvas:** Banners like `"GLOBAL CLUSTER ORCHESTRATOR & DISPATCHER (§11.5)"` or `"(§17.4)"` scream "slide deck handout." Section references belong exclusively in the body text.

---

## 4. Surgical Figure-by-Figure Redesign Blueprints

Here is the exact editorial prescription for transforming each of the five "v2" figures into true academic textbook schematics.

---

### Figure 1: `evolution_execution_units_v2.svg`
**Current State:** 102 text nodes, 301 words. A timeline over 5 vertical cards, each containing an "Architectural Contract" checklist, a "Failure Semantics" paragraph, and a historical citation box.

#### What to Strip Immediately:
- ❌ Strip all 5 "Architectural Contract" boxes (20 bullet points: *Determinism, Fault Model, Memory Scope, Primitives*).
- ❌ Strip all 5 "Failure Semantics" prose blocks (*"Instruction succeeds in single clock edge or hardware faults (SIGFPE)..."*).
- ❌ Strip all 5 historical reference footers (*"Maurice Wilkes (1949) Stored-program EDSAC"*, *"Birrell & Nelson (1984)"*, etc.). These belong in the chapter's "Historical Perspective" section.
- ❌ Strip the bottom thesis badge (*"The Stochastic Computer MLSysBook Volume III Core Thesis"*).
*(This eliminates 230 of the 301 words on canvas).*

#### What to Draw Instead (Pure Schematic Datapath):
Expand the middle diagram pane so it occupies 85% of each column's height. Replace prose with concrete microarchitectural schematics:
1. **Col 1 (CPU Instruction, $\tau \sim 1\text{ ns}$):** Classic Hennessy & Patterson datapath: Program Counter (`PC`), Register File (`R₀..R₃₁`), ALU with arithmetic inputs, single clock pulse $\Phi$, and bus width `[63:0]`.
2. **Col 2 (OS Process, $\tau \sim 1–10\ \mu\text{s}$):** Hardware MMU with `CR3` register pointing to two-level Page Table translating Virtual Address (`VA`) to Physical Address (`PA`), with Ring 0 / Ring 3 privilege boundary line.
3. **Col 3 (Distributed RPC, $\tau \sim 1–100\text{ ms}$):** Saltzer & Kaashoek client/service RPC interface: Client Stub $\to$ Socket Buffer $\to$ Network Wire $\to$ Server Skeleton, with an explicit hardware timeout deadline clock $\Delta t_{\text{to}}$.
4. **Col 4 (Model Inference, $\tau \sim 100\text{ ms}$):** Transformer systolic GEMM datapath: Activation vector $\mathbf{x}_{[1 \times d]} \times$ Weight matrix $\mathbf{W}_{[d \times d]}$, feeding Softmax logit distribution and appending to KV Cache tensor blocks ($K_t, V_t$).
5. **Col 5 (Agent Trajectory, $\tau \sim 10^1–10^4\text{ s}$):** Closed-loop control system: Stochastic Policy $\pi_\theta \xrightarrow{a_t}$ Isolated Sandbox Execution $\xrightarrow{o_t}$ Verifier / Oracle $\xrightarrow{r_t}$ WAL Trajectory State Journal, with an explicit rollback compensation loop ($T_i \to C_i$).

```
+---------------------------------------------------------------------------------------------------+
| FIGURE 1 REDESIGN: PURE SCHEMATIC EXECUTION UNITS                                                 |
+---------------------------------------------------------------------------------------------------+
|  10⁻⁹ s               10⁻⁶ s               10⁻³ s               10⁻¹ s               10¹–10⁴ s    |
|----+--------------------+--------------------+--------------------+--------------------+--------->|
| [CPU Instruction]   [OS Process]         [Distributed RPC]    [Model Inference]    [Agent Loop]   |
|                                                                                                   |
|    +-----+             +-------+            +----+    Wire       +-----+     K, V      +--------+     |
| PC |     |          CR3|PageTab|      Client|Buf |====[RPC]====> |GEMM |-->[Cache]     | Policy |     |
|    v     |             v       |            +----+    \Delta t   +-----+     Tokens      v a_t    |     |
|  [Regs]  |           [Virtual] |                                    v                  [Sandbox]  |
|    |     |             | MMU   |                                 [Logits]                 | o_t   |
|    v     |             v       |                                    v                     v       |
|  [ALU]   |           [Physical]|                                [Softmax]              [Verifier] |
|    |     |                     |                                                          | r_t   |
|    +--\Phi             +-------+                                                       [WAL Log]  |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

---

### Figure 2: `prefill_decode_disaggregation_v2.svg`
**Current State:** 59 text nodes, 311 words. Four large boxes packed with bullet points describing hardware types, disaggregation invariants, and a 3-step bottom numbered text banner.

#### What to Strip Immediately:
- ❌ Strip all hardware model enumerations (*"High-TFLOPS Tensor Cores (H100 SXM5 / B200)"*, *"H200 141GB / MI300X 192GB"*).
- ❌ Strip explanatory paragraphs (*"Prompt Evaluation & KV Generation"*, *"Ingests RDMA KV blocks directly into local page table"*).
- ❌ Strip the bottom 3-step numbered prose box (*"1. D-Node emits tool call... 2. External execution in Sandbox... 3. Updated context..."*).
- ❌ Strip textual invariants (*"Disaggregated State Invariant: Zero local decode execution on P-nodes..."*).
*(This eliminates 240 of the 311 words).*

#### What to Draw Instead (Physical Cluster & Memory Topology):
1. **P-Node Architecture Box:**
   - Draw the physical GPU board schematic: Large Compute Matrix / Tensor Core Grid labeled `GEMM Array [FP8/FP16]`.
   - Inbound prompt FIFO buffer labeled `Prompt Queue (S tokens)`.
   - Pipelined chunked prefill tiles: $T_0, T_1, \dots, T_k$.
   - **Roofline Inset (Upper Corner):** A clean, unlabelled 2-axis plot showing the operating point far to the right of the ridge point ($\text{AI} \gg \mathcal{I}^*$).
2. **Zero-Copy RDMA Interconnect Bus:**
   - Draw an actual packetized network bus with PCIe Gen5 NIC and GPUDirect RDMA engine.
   - Show discrete KV Cache Page Packets ($[K_l, V_l]$) in flight across the wire, with bandwidth label $B_{\text{net}} \ge 50\text{ GB/s}$.
3. **D-Node Architecture Box:**
   - Draw High-Capacity Memory Banks labeled `HBM3e / DRAM Pool`.
   - PagedAttention page table mapping: Virtual Token Blocks $\to$ Physical Memory Slots.
   - Single-token vector ALU labeled `GEMV Engine (Batch B)`.
   - Outbound autoregressive token stream arrow ($y_{t+1}$).
   - **Roofline Inset (Upper Corner):** A 2-axis plot showing the operating point on the sloped bandwidth ceiling ($\text{AI} \ll \mathcal{I}^*$).
4. **Tool Actuation Boundary:**
   - A clean return loop arrow passing through an isolated sandbox gate with DMA swap to Host DRAM ($M_{\text{KV}} \to \text{DRAM}$).

---

### Figure 3: `progressive_autonomy_spectrum_v2.svg`
**Current State:** 80 text nodes, 323 words. 4 vertical columns with 12 nested cards filled with statistical hypothesis formulas, eval gym details, and rollout parameters.

#### What to Strip Immediately:
- ❌ Strip all 12 card wrappers and bullet points (*"• N_eval = 2,500 held-out tasks"*, *"• Sandboxed microVM environment"*).
- ❌ Strip text descriptions of statistical tests (*"Null Hypothesis: \theta_cand \le \theta_prod"*, *"Paired t-test: p < 0.05 required"*).
- ❌ Strip bottom banner text (*"FAST TRAFFIC ABORT & ROLLBACK (MTTR < 10 s)"*).
*(This eliminates 250 of the 323 words).*

#### What to Draw Instead (Control Pipeline & Circuit Breakers):
Transform the table into a classic **Signal Flow Pipeline with Hardware Circuit Breakers**:
1. **Stage 1 (Hermetic Eval Gate):**
   - Candidate weights box ($\theta_{\text{cand}}$) feeding into a sandbox block with unit test oracle ($\mathcal{V}_{\text{test}}$).
   - A clean mathematical comparator triangle ($\Delta$): $\text{Score} \ge \text{Baseline} \land p < 0.05$.
   - Green pass line forward; red reject line downward to `Abort`.
2. **Stage 2 (Shadow Execution):**
   - Traffic Demultiplexer splitting incoming user stream: $100\%$ to Primary Fleet ($\theta_{\text{prod}}$), $100\%$ mirrored to Shadow Worker ($\theta_{\text{cand}}$).
   - Shadow Worker isolated by a grounded barrier (read-only, no write bus).
   - Differential Comparator ($\otimes$) measuring Action Semantic Divergence: $D_{\text{sem}}(a_{\text{cand}}, a_{\text{prod}}) < \epsilon_{\text{div}}$.
3. **Stage 3 (Canary Gate):**
   - Dynamic Potentiometer / Throttle icon showing $1\% \to 5\%$ live traffic split.
   - Circuit breaker tripwire switch monitored by telemetry sensor ($\Lambda_{\text{SPRT}}$).
   - Tripped signal triggers a fast-acting shunt switch reverting traffic back to $0\%$.
4. **Stage 4 (Production Fleet):**
   - Full traffic multiplexer ($100\%$) feeding multi-tenant cluster with continuous watchdog monitor loop.

---

### Figure 4: `tokenomics_memory_hierarchy_caching_v2.svg`
**Current State:** 50 text nodes, 286 words. Left side has MLFQ queues; right side has 3 large tiered memory boxes packed with dollar lease rates, rental equations, and scheduling rules.

#### What to Strip Immediately:
- ❌ Strip long prose in the bottom left controller (*"Scheduler Dispatch Arbiter & Preemption Controller: Enforces priority order Q0 > Q1 > Q2... Clamps steady-state cluster utilization to Kingman headroom boundary"*).
- ❌ Strip dollar holding rent text (*"Holding Rent: $1.39 × 10⁻⁵ / GB·s ($4.00/hr lease equivalent)"*). Pricing belongs in an economic case study in the body text.
- ❌ Strip section cross-references (*"(§17.4)", "(§07, §17)"*).
- ❌ Strip explanatory text blocks inside tiers (*"Absorbs tool-execution pauses (50 ms – 15 s) Prevents memory-tax leakage..."*).
*(This eliminates 200 of the 286 words).*

#### What to Draw Instead (Memory Hierarchy Pyramid + Queue State Machine):
1. **Left Side: Pure Computer Systems MLFQ Queue:**
   - Three standard FIFO shift-registers with entry/exit heads:
     - $Q_0$ [Interactive]: high-priority FIFO with short quantum $\tau_0$.
     - $Q_1$ [Standard]: round-robin FIFO with quantum $\tau_1$.
     - $Q_2$ [Batch]: pre-emptible FIFO.
   - Demotion arrows labeled simply with guard conditions: $\Delta t > \tau_0$ and $\Delta t > \tau_1$.
   - Anti-starvation promotion arrow: $t_{\text{wait}} > T_{\text{boost}}$.
   - Arbiter block showing preemption interrupt signal (`SIG_PREEMPT`) asserted when $\rho \ge \rho_{\text{crit}}$.
2. **Right Side: Hennessy & Patterson Classic Memory Pyramid / Hierarchy Stack:**
   - **Tier 1 (GPU HBM3):** Array of active KV Cache blocks ($B_0, B_1, \dots$). Spec label: $3.4\text{ TB/s} \cdot 100\text{ ns}$.
   - **Tier 2 (Host DRAM):** Radix Tree Prefix Cache and suspended session pages. Spec label: $64\text{ GB/s} \cdot 100\ \mu\text{s}$.
   - **Tier 3 (NVMe SSD / WAL):** Sequential append-only disk log and durable checkpoints. Spec label: $4\text{ GB/s} \cdot 10\text{ ms}$.
3. **The Eviction Bus:**
   - Draw an explicit DMA transfer path between HBM and DRAM labeled with the mathematical threshold:
     $$T_{\text{idle}} > \frac{C_{\text{swap}}}{R_{\text{HBM}} - R_{\text{DRAM}}}$$
   - Inset: A tiny 2-axis curve showing Cost vs. Idle Time with the break-even intersection point $T^*$.

---

### Figure 5: `five_part_agent_runtime_architecture_v2.svg`
**Current State:** 116 text nodes, 565 words! A massive syllabus outline in SVG format. Every single box contains 4 bullet points that literally recite the chapter sub-headings.

#### What to Strip Immediately:
- ❌ Strip ALL 36 bullet points across the 6 subsystems!
- ❌ Strip all section number tags (*"§02–§03"*, *"§09–§11"*, *"§04–§06"*, *"§07–§08"*, *"§12–§14"*, *"§15–§17"*).
- ❌ Strip explanatory paragraphs (*"Append-only trajectory state journal"*, *"Compensating actions: T_i -> C_i on failure"*, *"Strict boundary separating untrusted token generation from execution authority"*).
*(This eliminates over 450 of the 565 words, cutting word count by 80%).*

#### What to Draw Instead (The Capstone Stochastic Computer Block Diagram):
This is Chapter 18's grand capstone diagram. It must look like the **Central Microprocessor Architecture Block Diagram** in Hennessy & Patterson or Saltzer & Kaashoek's client/server operating system:
1. **The Central Trajectory System Bus:**
   - Draw a bold, unified 3-channel bus running through the center:
     - **Address / KV Page Bus** (purple)
     - **Data / Token Flow Bus** (blue)
     - **Control & Interrupt Bus** (orange/red)
   - Show tick-marks indicating bus widths: `Tokens [1, S]`, `Embeddings [d]`, `Interrupts [8]`.
2. **Subsystem 1: Stochastic Processing Unit (SPU):**
   - Context Register $\mathbf{c} \to$ GEMM Transformer Pipeline $\to$ Sampler/Logit Mask $\to$ Action Output Register $a_t$.
3. **Subsystem 2: Trajectory Control Unit (TCU):**
   - Agent Control Block (`ACB`) register array, Trajectory Program Counter ($t \in \mathbb{N}$), Token Budget Monotonic Counter ($B_t$), and Hardware Trap/Preemption Logic.
4. **Subsystem 3: Context Memory Management Unit (C-MMU):**
   - Radix Prefix Tree index walker, PagedAttention Page Table mapping virtual token offsets to physical GPU/Host memory blocks, and Eviction Controller.
5. **Subsystem 4: Peripheral Isolation Unit (PIU):**
   - Firecracker MicroVM enclave boundary with Capability Escrow Gate ($W \oplus X$) and Reversibility Barrier dividing ephemeral memory from external network RPCs.
6. **Subsystem 5: Policy Compiler & Reinforcement Unit:**
   - Feedback bus returning offline weight updates ($\Delta \theta$) from Ground-Truth Verifier / GRPO loss back to model storage.

```
+---------------------------------------------------------------------------------------------------+
| FIGURE 5 REDESIGN: THE STOCHASTIC COMPUTER MICROARCHITECTURE                                      |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  +-----------------------+     +-----------------------+     +-----------------------+            |
|  | SPU (Processor Core)  |     | TCU (Control Unit)    |     | C-MMU (Memory Unit)   |            |
|  | Context Reg -> [GEMM] |     | [ACB Registers]       |     | [Radix Page Table]    |            |
|  |         |             |     | Program Counter (t)   |     | Virtual -> Physical   |            |
|  |         v             |     | Budget Counter (B_t)  |     | PagedAttention Engine |            |
|  | Softmax -> Action Reg |     | Trap / Preempt Logic  |     | Eviction DMA Control. |            |
|  +-----------+-----------+     +-----------+-----------+     +-----------+-----------+            |
|              |                             |                             |                        |
|  ============#=============================#=============================#=============           |
|  SYSTEM BUS: Address Bus (L2/L3) | Data Bus [Tokens, d] | Control & Interrupt Bus (IRQ)           |
|  ============#=============================#=============================#=============           |
|              |                             |                             |                        |
|  +-----------+-----------+     +-----------+-----------+     +-----------+-----------+            |
|  | PIU (Peripheral Unit) |     | Verifier & Saga Unit  |     | Policy Compiler Engine|            |
|  | [W^X Isolation Wall]  |     | Write-Ahead Log (WAL) |     | AST Pruning & GRPO    |            |
|  | MicroVM Enclave Gate  |     | Compensating Saga C_i |     | Loss Mask \nabla L_CE |            |
|  | Capability Escrow     |     | Two-Phase Commit      |     | Return Bus (\Delta\theta)          |
|  +-----------------------+     +-----------------------+     +-----------------------+            |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

---

## 5. The "Text-Minimal Academic Figure Manifesto" for MLSysBook

To ensure that every future figure authored for Volume III (and across the entire MLSysBook series) adheres strictly to the highest academic press standards, we formally codify the following six rules:

### Rule 1: The 10% Typography Ceiling
- **Metric:** Under no circumstances may text elements occupy more than **10% of the total canvas surface area**.
- At least **90% of the canvas** must consist of architectural geometry, vector lines, memory layouts, coordinate plots, state bubbles, or negative whitespace.
- If an SVG's word count exceeds **50 words**, it is flagged for editorial review. If it exceeds **100 words**, it is rejected automatically.

### Rule 2: The "Zero Complete Sentences" Mandate
- No text element on canvas may contain a verb-predicate sentence structure, terminal punctuation (`.`), or bullet points (`•`, `-`).
- Canvas text is strictly restricted to:
  1. Component & module names (`ALU`, `C-MMU`, `RegFile`, `Sandbox`).
  2. Mathematical variables, dimensions, and formulas ($a_t, o_t, \mathbf{W}\mathbf{x}, [B, d]$).
  3. Signal, pin, and bus names (`Valid`, `Ready`, `CLK`, `SIG_PREEMPT`).
  4. Physical units and timing rates ($1\text{ ns}, 3.35\text{ TB/s}, 100\ \mu\text{s}$).
  5. State names (`CLOSED`, `HALF-OPEN`, `WAIT_TOOL`).

### Rule 3: Separation of Concerns (Canvas vs. Caption vs. Body)
- **If it explains *why*, it belongs in the Caption or Body Text.**
- **If it shows *what* and *where*, it belongs on the Canvas.**
- Every figure must have an accompanying textbook caption (75–125 words) that explains the mechanism, guides the reader through the dataflow, and states the architectural trade-off.

### Rule 4: Ban on Ephemeral Product Codes & Commercial Pricing
- Never etch transient commercial SKUs (e.g., `H100 SXM5`, `B200`, `GH200`) or transient cloud rental rates (e.g., `$4.00/hr`, `$1.39 × 10⁻⁵`) into vector graphics.
- Use canonical architectural parameters: $T_{\text{peak}}$, $B_{\text{mem}}$, $C_{\text{rent}}$, $\tau_{\text{DMA}}$.
- Grounding to current commercial hardware belongs in the chapter text or in a dedicated Markdown table, where it can be updated across editions without redrawing vector art.

### Rule 5: Schematic Integrity (Anti-Card Principle)
- A rectangle in an engineering textbook must represent a **physical or logical entity**: a silicon functional unit, a register, a memory buffer, a network packet, a software process boundary, or a state node.
- Rectangles must **never** be used as decorative presentation cards or slides with titles, subtitles, divider rules, and bulleted text summaries.

### Rule 6: Section Cross-References are Forbidden on Canvas
- Never place section markers like `(§11.5)` or `(§17.4)` inside a vector diagram.
- Figures must remain modular, self-contained, and relocatable across chapter revisions.

---

### Implementation Action Plan

| Priority | Action Item | Target File | Editorial Remedy |
| :---: | :--- | :--- | :--- |
| **P0** | **Strip Card Prose & Redraw Datapaths** | `evolution_execution_units_v2.svg` | Remove 230 words of contracts/failures; expand datapath schematics to 85% height. |
| **P0** | **Strip Specs & Draw Cluster Schematics** | `prefill_decode_disaggregation_v2.svg` | Remove H100 bullets and bottom text loop; draw physical boards, RDMA bus, and dual Roofline insets. |
| **P0** | **Convert Table to Signal/Gate Pipeline** | `progressive_autonomy_spectrum_v2.svg` | Remove 12 text cards; draw comparator triangles ($\Delta$), demux switches, and circuit breaker tripwires. |
| **P0** | **Clean Memory Pyramid & MLFQ Queues** | `tokenomics_memory_hierarchy_caching_v2.svg` | Remove prose blocks and lease prices; draw classic H&P memory pyramid with DMA break-even curve. |
| **P0** | **Convert Syllabus to Microprocessor Bus** | `five_part_agent_runtime_architecture_v2.svg` | Remove 450 words of outline bullets; draw unified central System Bus with SPU, TCU, and C-MMU blocks. |

---
*Signed,*
**Senior Visual Editor & Textbook Art Director**
*Academic Computer Systems Publishing Division*
