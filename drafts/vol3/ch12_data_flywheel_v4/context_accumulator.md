# Context Accumulator: Chapter 12 - Trajectory Harvesting

**Governing Systems Question:** *How do we harvest and distill chaotic execution failures so past mistakes become high-value training signal rather than context noise?*

**Core Takeaway:** *Execution traces become useful learning evidence only after capability-gap diagnosis, reproducible task fixtures, immutable provenance, staged verifier cascades, recovery-example curation, and strict evaluation split hygiene.*

## Running Narrative & Symbols

### Completed Step 1: Section 12.1: Capability Gap Diagnosis
- **File:** `01_sec_12_1.qmd` | **Word Count:** 2,590 words
- **Active Symbols Added:** `\Theta`, `\Delta`, `300`, `70\text{B}`, `\`, `4`, `12`, `2`
- **Terminal Bridge Handed Off:**
  _**Conclusion:** Strategy A resolves $90\%$ of failures deterministically with sub-hour deployment latency, zero compute overhead, and zero model regression risk. Premature fine-tun..._

### Completed Step 2: Section 12.2: Task Fixture Design
- **File:** `02_sec_12_2.qmd` | **Word Count:** 2,588 words
- **Active Symbols Added:** `F`, `S_0`, `\mathcal{M}_{\text{tool}}`, `P_{\text{task}}`, `N`, `\`, `\pi_\theta`, `K`
- **Terminal Bridge Handed Off:**
  _The systems lesson of SWE-bench and industrial harvesting platforms is unambiguous: trajectory quality is bounded by the precision of the environment harness. If the fixture fails ..._

### Completed Step 3: Section 12.3: Staged Verifier Cascades
- **File:** `03_sec_12_3.qmd` | **Word Count:** 2,628 words
- **Active Symbols Added:** `K`, `c_i`, `i`, `C_{\text{attempt}}`, `Y`, `c_3`, `c_5`, `p_1`
- **Terminal Bridge Handed Off:**
  _The mathematical reality captured in @eq-verifier-cost-cascade dictates the scheduling policy of distributed harvesting pipelines. By interposing microsecond syntactic checks and m..._

### Completed Step 4: Section 12.4: Recovery Demonstration Curation
- **File:** `04_sec_12_4.qmd` | **Word Count:** 2,358 words
- **Active Symbols Added:** `\pi_\theta`, `\epsilon`, `t`, `T`, `S_0`, `t_{\text{div}}`, `\mathcal{M}_{\text{tool}}`
- **Terminal Bridge Handed Off:**
  _At a standard float16 representation (2 bytes per parameter/token gradient footprint in data loaders), the dataset occupies over $1.028\text{ GB}$ of packed token storage. More cri..._

### Completed Step 5: Section 12.5: Collection Pipeline Architecture
- **File:** `05_sec_12_5.qmd` | **Word Count:** 3,348 words
- **Active Symbols Added:** `P_{50}`, `P_{99}`, `T_{\max}`, `T_{\text{lease}}`, `S_0`, `V_1`, `V_2`, `V_3`
- **Terminal Bridge Handed Off:**
  _***

As sanitized, validated trajectories are streamed into persistent columnar storage sinks, the engineering challenge pivots from operational availability to data integrity. A m..._

### Completed Step 6: Section 12.6: Trajectory Provenance Tracking
- **File:** `06_sec_12_6.qmd` | **Word Count:** 2,998 words
- **Active Symbols Added:** `\pi_\theta`, `t_0`, `\tau`, `p`, `k`, `H_{\text{lineage}}`, `K_{\text{pipe}}`, `H_{\text{trajectory}}`
- **Terminal Bridge Handed Off:**
  _***

Once trajectories are cryptographically enveloped, stripped of sensitive operational credentials, and classified by legal provenance, they constitute a trustworthy, reproducib..._

### Completed Step 7: Section 12.7: Split Hygiene Verification
- **File:** `07_sec_12_7.qmd` | **Word Count:** 2,495 words
- **Active Symbols Added:** `\pi_\theta`, `n`, `S_0`, `\mathcal{M}_{\text{tool}}`, `P_{\text{task}}`, `R`, `10^9`, `V_3`
- **Terminal Bridge Handed Off:**
  _***

Task fixtures define reproducible starting sandbox states; staged verifier cascades filter out specious completions; recovery demonstration curation injects behavioral resilie..._

### Completed Step 8: Section 12.8: End-to-End Trajectory Harvesting Synthesis
- **File:** `08_sec_12_8.qmd` | **Word Count:** 3,243 words
- **Active Symbols Added:** `\tau_{\text{raw}}`, `P_{\text{task}}`, `S_0`, `\mathcal{M}_{\text{tool}}`, `\Delta_{\text{fs}}`, `V_1`, `V_2`, `V_3`
- **Terminal Bridge Handed Off:**
  _By treating trajectory harvesting not as an incidental data-collection chore, but as an engineered, verifiable systems compiler, the host architecture establishes a self-sustaining..._

### Completed Step 9: Fallacies and Pitfalls
- **File:** `98_fallacies_pitfalls.qmd` | **Word Count:** 2,617 words
- **Active Symbols Added:** `\tau_{\text{raw}}`, `\pi_\theta`, `T_{\max}`, `70\text{B}`, `V_1`, `V_2`, `V_3`, `s_t`
- **Terminal Bridge Handed Off:**
  _The architectural defense requires *hierarchical, domain-isolated split hygiene*. Trajectory data must never be partitioned at the execution or task level. Instead, splits must be ..._

### Completed Step 10: Summary & Chapter Connection
- **File:** `99_summary_connection.qmd` | **Word Count:** 864 words
- **Active Symbols Added:** `S_0`
- **Terminal Bridge Handed Off:**
  _::: {.callout-chapter-connection title="From Trajectory Data to Supervised Adaptation"}
We have established how to diagnose agent failures, construct reproducible task fixtures, ex..._
