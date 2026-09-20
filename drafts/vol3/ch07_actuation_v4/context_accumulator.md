# Context Accumulator: Chapter 07 - Tool Execution

**Governing Systems Question:** *How does a candidate action become a controlled effect with an observable result?*

**Core Takeaway:** *A model output becomes an external effect only after the runtime parses, authorizes, dispatches, and observes it; typed tool contracts and idempotency make those transitions inspectable and recoverable.*

## Running Narrative & Symbols

### Completed Step 1: Section 7.1: Tool Subsystem Architecture
- **File:** `01_sec_7_1.qmd` | **Word Count:** 1,590 words
- **Active Symbols Added:** None
- **Terminal Bridge Handed Off:**
  _POSIX permissions (`rwxrwxrwx`), process UID/GID, and OS capability sets. | Explicit capability tokens, declarative access control policies, and Zero Ambient Authority ($A=0$). | |..._

### Completed Step 2: Section 7.2: Tool Interface Schemas
- **File:** `02_sec_7_2.qmd` | **Word Count:** 2,922 words
- **Active Symbols Added:** `S`, `n_i`, `d_i`, `v_{\text{default}}`, `\chi`, `P`, `5000`, `A`
- **Terminal Bridge Handed Off:**
  _JIT schema injection introduces an architectural trade-off between context efficiency and retrieval recall. If the host retriever fails to surface a necessary tool schema (a false ..._

### Completed Step 3: Section 7.3: Interoperable Tool Discovery
- **File:** `03_sec_7_3.qmd` | **Word Count:** 3,001 words
- **Active Symbols Added:** `N`, `M`, `\to`, `97\times`, `18\times`, `C_0`, `C_1`, `C_2`
- **Terminal Bridge Handed Off:**
  _This architectural division directly reflects the End-to-End Argument in System Design formulated by Saltzer, Reed, and Clark in 1984. The communication protocol (MCP) provides a u..._

### Completed Step 4: Section 7.4: Idempotent Action Execution
- **File:** `04_sec_7_4.qmd` | **Word Count:** 2,795 words
- **Active Symbols Added:** `\tau_{\text{timeout}}`, `S`, `\theta`, `t_0`, `\mathcal{C}`, `S_{\text{env}}`, `\sigma_{\text{drop}}`, `\sigma_{\text{crash}}`
- **Terminal Bridge Handed Off:**
  _By enforcing keyed deduplication on modern endpoints and structured reconciliation probes on legacy interfaces, the agent runtime insulates the underlying system from non-determini..._

### Completed Step 5: Section 7.5: Observation Stream Truncation
- **File:** `05_sec_7_5.qmd` | **Word Count:** 2,837 words
- **Active Symbols Added:** `S_{\max}`, `128\text{K}`, `10`, `50`, `200`, `N`, `B_{\text{max}}`, `B_{\text{head}}`
- **Terminal Bridge Handed Off:**
  _Through this combination of non-blocking kernel pipe drains, strict cumulative accounting, and `SIGPIPE` enforcement, the runtime provides complete structural protection for host s..._

### Completed Step 6: Section 7.6: Terminal Output Sanitization
- **File:** `06_sec_7_6.qmd` | **Word Count:** 2,845 words
- **Active Symbols Added:** `\text{FP16}`, `760\times`, `960\times`, `957\times`, `B_{\max}`
- **Terminal Bridge Handed Off:**
  _represents the head or the tail of the stream, and directs the model to the unexpurgated log file saved on disk (as illustrated in @lst-vol3-observation-sanitization). Armed with t..._

### Completed Step 7: Section 7.7: Asynchronous Tool Dispatch
- **File:** `07_sec_7_7.qmd` | **Word Count:** 3,215 words
- **Active Symbols Added:** `\text{GEMV}`, `\text{HBM}`, `t_{\text{spawn}}`, `\text{pid}`, `\sigma_{\text{state}}`, `\mathcal{P}_{\text{io}}`, `\tau_{\text{timeout}}`, `H`
- **Terminal Bridge Handed Off:**
  _By combining capability-secured job handles, interrupt-driven event loops, and strict process group lifecycle governance, the agent runtime establishes a robust, highly concurrent ..._

### Completed Step 8: Section 7.8: Toolkit Granularity Partitioning
- **File:** `08_sec_7_8.qmd` | **Word Count:** 2,151 words
- **Active Symbols Added:** `t_i`, `n_i`, `d_i`, `S_{\text{schema}}`, `N`, `\epsilon_{\text{select}}`, `\text{FP16}`, `A`
- **Terminal Bridge Handed Off:**
  _The defensive implementation above demonstrates how host runtimes absorb operational hazards. Rather than requiring the agent to manually verify file offsets, the host tool enforce..._

### Completed Step 9: Fallacies and Pitfalls
- **File:** `98_fallacies_pitfalls.qmd` | **Word Count:** 1,340 words
- **Active Symbols Added:** `S_{\text{schema}}`, `N`, `d_i`, `\epsilon_{\text{select}}`, `\tau_{\text{timeout}}`, `K_{\text{idem}}`, `10^6`, `S_{\max}`
- **Terminal Bridge Handed Off:**
  _Runtimes must enforce strict physical observation budgets ($B_{\max}$) directly at the kernel pipe level using non-blocking I/O and bounded ring buffers. The runtime supervisor mus..._

### Completed Step 10: Summary & Chapter Connection
- **File:** `99_summary_connection.qmd` | **Word Count:** 667 words
- **Active Symbols Added:** `K_{\text{idem}}`
- **Terminal Bridge Handed Off:**
  _correctness and integrity of an agentic system cannot rely on the model’s internal self-consistency or its probabilistic intent. Rather, invariant closure is achieved solely throug..._
