# Context Accumulator: Chapter 05 - Paged Attention Memory

**Governing Systems Question:** *How can a serving system allocate and reuse the attention state of active trajectories under finite accelerator memory?*

**Core Takeaway:** *The KV cache is physical attention state maintained by an inference service; its dynamic footprint, sharing, scheduling, and eviction determine how many long or branching trajectories the service can run.*

## Running Narrative & Symbols

### Completed Step 1: Section 5.1: From Context Tokens to KV State
- **File:** `01_sec_5_1.qmd` | **Word Count:** 2,099 words
- **Active Symbols Added:** `t`, `x_t`, `q_t`, `v_i`, `k_i`, `S`, `k_t`, `v_t`
- **Terminal Bridge Handed Off:**
  _DRAM or discard them entirely, forcing a computationally expensive prefill recomputation phase once the tool output arrives? The foundation model cannot resolve this dilemma. Follo..._

### Completed Step 2: Section 5.2: Memory Fragmentation
- **File:** `02_sec_5_2.qmd` | **Word Count:** 3,297 words
- **Active Symbols Added:** `S_{\max}`, `t`, `S_{\text{prompt}}`, `S_{\text{gen}}`, `m_{\text{token}}`, `\eta_{\text{static}}`, `S_{\text{alloc}}`, `A`
- **Terminal Bridge Handed Off:**
  _Empirical evaluations of production inference services corroborate these analytical limits. In their seminal characterization of LLM serving systems, Kwon et al. (2023) demonstrate..._

### Completed Step 3: Section 5.3: Paged KV Allocation
- **File:** `03_sec_5_3.qmd` | **Word Count:** 2,591 words
- **Active Symbols Added:** `L`, `H_{\text{kv}}`, `S_{\max}`, `d_{\text{head}}`, `B`, `p`, `b_{\text{elem}}`, `S_{\text{prompt}}`
- **Terminal Bridge Handed Off:**
  _By mapping logical token sequences to non-contiguous, reference-counted physical blocks, PagedAttention provides the foundational virtualization layer required to serve dynamic, br..._

### Completed Step 4: Section 5.4: Prefix Caching
- **File:** `04_sec_5_4.qmd` | **Word Count:** 2,869 words
- **Active Symbols Added:** `S_{\text{match}}`, `\mathbf{t}`, `t_{i}`, `B`, `2N`, `28\times`, `\tau_{\text{low}}`, `\mathbf{k}_t`
- **Terminal Bridge Handed Off:**
  _To maximize prefix hit rates, agent software architectures must adhere to a strict structural design pattern: place invariant, shared instructions and schemas at the absolute begin..._

### Completed Step 5: Section 5.5: Chunked Prefill Scheduling
- **File:** `05_sec_5_5.qmd` | **Word Count:** 2,655 words
- **Active Symbols Added:** `\mathcal{I}`, `T_{\text{peak}}`, `B_{\text{mem}}`, `P`, `2P`, `B_{\text{decode}}`, `S_{\text{prompt}}`, `C_{\text{chunk}}`
- **Terminal Bridge Handed Off:**
  _---

While chunked prefill balances active prefills against active decodes during continuous token generation, agentic systems exhibit a distinct operational pattern: *long idle pa..._

### Completed Step 6: Section 5.6: Retain, Evict, Recompute, or Offload
- **File:** `06_sec_5_6.qmd` | **Word Count:** 3,146 words
- **Active Symbols Added:** `B`, `M_{\text{KV}}`, `T_{\text{wait}}`, `B_{\text{xfer}}`, `T_{\text{resume}}`, `S`, `P`, `L`
- **Terminal Bridge Handed Off:**
  _---

The ability to dynamically retain, evict, recompute, or offload attention state ensures that an inference engine does not collapse under the erratic idle rhythms of external t..._

### Completed Step 7: Section 5.7: Cache Capacity Provisioning
- **File:** `07_sec_5_7.qmd` | **Word Count:** 3,251 words
- **Active Symbols Added:** `C_{\text{HBM}}`, `M`, `B_{\text{xfer}}`, `2P`, `M_{\text{weights}}`, `M_{\text{activations}}`, `M_{\text{runtime}}`, `M_{\text{KV}}`
- **Terminal Bridge Handed Off:**
  _---

The ability to accurately provision physical memory and govern dynamic cache allocations ensures that an individual serving node can sustain high-density agent trajectories wi..._

### Completed Step 8: Fallacies and Pitfalls
- **File:** `98_fallacies_pitfalls.qmd` | **Word Count:** 2,044 words
- **Active Symbols Added:** `i`, `32\text{k}`, `\bar{S}_{\text{prompt}}`, `\bar{S}_{\text{gen}}`, `\bar{m}_{\text{token}}`, `B`, `\lambda`, `W`
- **Terminal Bridge Handed Off:**
  _---

The fallacies and pitfalls detailed above underscore a fundamental systems truth: the physical KV cache is neither an autonomous knowledge base nor an elastic resource that ca..._

### Completed Step 9: Summary & Chapter Connection
- **File:** `99_summary_connection.qmd` | **Word Count:** 997 words
- **Active Symbols Added:** `L`, `H_{\text{kv}}`, `d_{\text{head}}`, `b_{\text{elem}}`, `S_{\max}`, `TTFT`, `ITL`
- **Terminal Bridge Handed Off:**
  _::: {.callout-chapter-connection title="From Serving State to Durable Information"}
Physical attention state is volatile and local by definition: it is bound to the lifespan of an ..._
