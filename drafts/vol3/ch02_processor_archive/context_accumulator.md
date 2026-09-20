# Context Accumulator: Chapter 02 - The Stochastic Processor Core

**Governing Systems Question:** *What does a foundation-model invocation compute, and what contract does the rest of the computer need to use its output?*

**Core Takeaway:** *One foundation-model invocation maps staged tokens to a candidate sequence through repeated next-token computation; an explicit caller contract is required to interpret its status, validity, and cost without confusing output with an authorized effect.*

**Canonical Systems Scenario:** The Configuration Parser Defect (`int(seconds) * 1000` truncating `2.5s` to `2000ms`).

## Running Narrative & Symbols

### Completed Step 2: Section 2.2: Tokens as the Processor Interface
- **File:** `02_sec_2_2.qmd` | **Word Count:** 1,861 words
- **Active Symbols Added:** `\mathcal{V}_0`, `\mathbf{t}`, `\mathbf{W}_E`, `t_i`, `\mathbf{e}_{755}`, `\mathbf{e}_{711}`, `\mathbf{e}_{2038}`, `M`
- **Terminal Bridge Handed Off:**
  _To guarantee mechanical invariant closure, the host agent runtime must treat tokenization as an explicit, bounded serialization protocol. The host must execute the exact BPE tokeni..._

### Completed Step 3: Section 2.3: Next-Token Computation
- **File:** `03_sec_2_3.qmd` | **Word Count:** 2,018 words
- **Active Symbols Added:** `M`, `\mathcal{V}_0`, `K`, `y_t`, `t`, `K_{\max}`, `T_{\max}`, `L`
- **Terminal Bridge Handed Off:**
  _However, grammar-directed masking operates strictly at the boundary of syntax, not semantics. An external mask can force the model to emit a structurally pristine JSON schema conta..._

### Completed Step 4: Section 2.4: Candidate Sequences Versus Valid Conclusions
- **File:** `04_sec_2_4.qmd` | **Word Count:** 2,306 words
- **Active Symbols Added:** `\Theta`, `\mathcal{V}`, `y_t`, `t`, `\mathbf{x}`, `\mathcal{V}_0`, `\mathbf{y}`
- **Terminal Bridge Handed Off:**
  _Because a stochastic core fails plausibly rather than fail-stopping, the external verification perimeter serves as the critical translation layer. It intercepts plausible counterfa..._

### Completed Step 5: Section 2.5: The Invocation Contract
- **File:** `05_sec_2_5.qmd` | **Word Count:** 1,911 words
- **Active Symbols Added:** `\mathcal{R}`, `M`, `\Theta`, `\mathcal{G}`, `K_{\max}`, `T_{\max}`, `L`, `H_{\text{kv}}`
- **Terminal Bridge Handed Off:**
  _Invariant closure is strictly external. The host runtime—using deterministic JSON parsers, compiler type checkers, AST linters, and explicit envelope flags—mechanically decides whe..._

### Completed Step 5: Section 2.5: The Invocation Contract
- **File:** `05_sec_2_5.qmd` | **Word Count:** 2,081 words
- **Active Symbols Added:** `\mathbf{x}`, `S_{\max}`, `\Theta_{\text{id}}`, `K_{\max}`, `T_{\max}`, `\mathcal{S}_{\text{stop}}`, `\mathcal{G}`, `M`
- **Terminal Bridge Handed Off:**
  _The invocation contract and normalized status envelope transform the stochastic core from an unpredictable streaming socket into a well-behaved coprocessor. Yet, while the contract..._

### Completed Step 6: Section 2.6: Constraining the Output Surface
- **File:** `06_sec_2_6.qmd` | **Word Count:** 1,575 words
- **Active Symbols Added:** `\mathcal{V}_0`, `t`, `\Sigma`, `Q`, `\bot`, `y_t`
- **Terminal Bridge Handed Off:**
  _By ensuring that the automaton contains valid transitions for error reports, diagnostic explanations, and uncertainty indicators, the runtime avoids artificially concentrating prob..._

### Completed Step 7: Section 2.7: The Cost of an Invocation
- **File:** `07_sec_2_7.qmd` | **Word Count:** 2,200 words
- **Active Symbols Added:** `T_{\text{call}}`, `T_{\text{queue}}`, `T_{\text{transport}}`, `T_{\text{prefill}}`, `K`, `T_{\text{validate}}`, `\Pi_{\text{peak}}`, `\beta_{\text{mem}}`
- **Terminal Bridge Handed Off:**
  _A common runtime bug is the injection of dynamic metadata—such as nonces, localized wall-clock timestamps (`Current Time: 21:07:59`), or randomized tool execution IDs—at the head o..._
