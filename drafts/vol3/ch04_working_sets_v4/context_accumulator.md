# Context Accumulator: Chapter 04 - Working Context

**Governing Systems Question:** *What information should the runtime stage when the trajectory's full history exceeds the model's useful context?*

**Core Takeaway:** *The context supplied to the next invocation is a deliberately selected logical working set; retention, ordering, compaction, and freshness determine whether the model sees the evidence needed for its next decision.*

## Running Narrative & Symbols

### Completed Step 1: Section 4.1: The Working-Set Decision
- **File:** `01_sec_4_1.qmd` | **Word Count:** 1,528 words
- **Active Symbols Added:** `S_{\max}`, `t`, `e_i`
- **Terminal Bridge Handed Off:**
  _Computing $\mathcal{W}(t)$ is an exercise in closed-loop runtime mediation governed by Edsger W. Dijkstra's classic verification principle: testing can reveal the presence of fault..._

### Completed Step 2: Section 4.2: Working Set Capacity
- **File:** `02_sec_4_2.qmd` | **Word Count:** 3,276 words
- **Active Symbols Added:** `M`, `S_{\max}`, `M_{\text{eff}}`, `i`, `L`, `d`, `Q`, `K`
- **Terminal Bridge Handed Off:**
  _The primary duty of the host supervisor's memory management subsystem is to enforce policies that keep the logical working set strictly bounded within Phase II. The runtime must no..._

### Completed Step 3: Section 4.3: Staging the Next Invocation
- **File:** `03_sec_4_3.qmd` | **Word Count:** 2,326 words
- **Active Symbols Added:** `0`, `L_{\text{root}}`, `L`, `1`, `P`, `2P`, `k`, `j`
- **Terminal Bridge Handed Off:**
  _Duplicating instructions at the end of the prompt reconciles prefix caching with attention recency: the static Root establishes the definitive, high-capacity system frame that rema..._

### Completed Step 4: Section 4.4: Context Compaction
- **File:** `04_sec_4_4.qmd` | **Word Count:** 2,695 words
- **Active Symbols Added:** `M`, `M_{\text{eff}}`, `\tau_{\text{high}}`, `\tau_{\text{low}}`, `O_t`, `t`, `S`, `k`
- **Terminal Bridge Handed Off:**
  _Compaction delivers a $7.78\times$ reduction in prefill latency (saving over 68 seconds of blocking host time per turn), primarily driven by the quadratic reduction in self-attenti..._

### Completed Step 5: Section 4.5: Context Checkpointing
- **File:** `05_sec_4_5.qmd` | **Word Count:** 2,491 words
- **Active Symbols Added:** `\mathcal{G}`, `\mathcal{C}`, `a_i`, `o_i`, `\rho_i`, `\mathcal{H}`, `\Omega`, `\mathcal{A}`
- **Terminal Bridge Handed Off:**
  _Through memory anchoring and structured resumption, the host runtime maintains the logical coherence of the agent across arbitrary task durations. The agent perceives a continuous,..._

### Completed Step 6: Section 4.6: Context Invalidation
- **File:** `06_sec_4_6.qmd` | **Word Count:** 3,144 words
- **Active Symbols Added:** `S`, `\Omega_{\text{env}}`, `t`, `S_t`, `V_1`, `V_2`, `\Omega`, `\lambda`
- **Terminal Bridge Handed Off:**
  _The tension between prefill compute efficiency and semantic purity represents a core architectural trade-off of agent context management. When an agent operates over short horizons..._

### Completed Step 7: Section 4.7: Working-Memory Evaluation
- **File:** `07_sec_4_7.qmd` | **Word Count:** 2,537 words
- **Active Symbols Added:** `S`, `\Omega_{\text{env}}`, `a_0`, `t`, `a_t`, `o_t`, `\mathcal{M}`, `\pi_{\text{ctx}}`
- **Terminal Bridge Handed Off:**
  _Evaluating an agent's working memory reveals that context management is not an aesthetic prompt-formatting challenge, but a disciplined systems resource allocation problem. A high-..._

### Completed Step 8: Fallacies and Pitfalls
- **File:** `98_fallacies_pitfalls.qmd` | **Word Count:** 1,682 words
- **Active Symbols Added:** `n_{\text{layers}}`, `n_{\text{kv\_heads}}`, `d_k`, `L`, `i`, `o_t`, `N`, `o_i`
- **Terminal Bridge Handed Off:**
  _To prevent stale-state corruption, the runtime supervisor must enforce explicit *context invalidation protocols* and *content versioning*. Every environment observation ingested in..._

### Completed Step 9: Summary & Chapter Connection
- **File:** `99_summary_connection.qmd` | **Word Count:** 765 words
- **Active Symbols Added:** `S_{\max}`, `M_{\text{eff}}`
- **Terminal Bridge Handed Off:**
  _::: {.callout-chapter-connection title="From Selected Tokens to Physical Attention State"}
Every logical token selected by the working-memory runtime ceases to be an abstract chara..._
