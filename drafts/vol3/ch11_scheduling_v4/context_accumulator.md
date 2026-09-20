# Context Accumulator: Chapter 11 - Fault Recovery

**Governing Systems Question:** *How does an autonomous system recover from cascading failures in an environment where actions have irrevocable side effects?*

**Core Takeaway:** *Long trajectories need recovery contracts that classify actions by reversibility and use verification, forward repair, compensation, or escalation according to the actual state left by partial effects.*

## Running Narrative & Symbols

### Completed Step 1: Section 11.1: The Transactional Boundary Collapse
- **File:** `01_sec_11_1.qmd` | **Word Count:** 1,328 words
- **Active Symbols Added:** `2\text{PC}`, `X`, `S`, `N`, `L`
- **Terminal Bridge Handed Off:**
  _with an application-level compensating transaction. A compensating transaction does not perform magic time-travel or rewrite historical reality; instead, it executes an active, sem..._

### Completed Step 2: Section 11.2: The Trajectory Saga Pattern
- **File:** `02_sec_11_2.qmd` | **Word Count:** 2,644 words
- **Active Symbols Added:** `T_i`, `C_i`, `T_4`, `\mathcal{T}`, `n`, `\alpha_i`, `\theta_i`, `\phi_i`
- **Terminal Bridge Handed Off:**
  _The latency and financial realities highlighted in the worked example reveal a fundamental engineering trade-off at the heart of runtime design. Unwinding a Saga via backward compe..._

### Completed Step 3: Section 11.3: Forward Recovery Versus Rollback
- **File:** `03_sec_11_3.qmd` | **Word Count:** 2,588 words
- **Active Symbols Added:** `\mathbf{s}_k`, `a_{\text{repair}}`, `T_t`, `s_t`, `C_i`, `T_i`, `o_t^{\text{err}}`, `\pi_\theta`
- **Terminal Bridge Handed Off:**
  _By enforcing the anti-spin invariant, the runtime bounds the worst-case blast radius of forward self-healing. If forward repair succeeds within $K_{\text{repair}}$ turns, the syste..._

### Completed Step 4: Section 11.4: Pivot Action Irreversibility
- **File:** `04_sec_11_4.qmd` | **Word Count:** 2,282 words
- **Active Symbols Added:** `T_i`, `C_i`, `T_{\text{pivot}}`, `\mathcal{L}_{\text{comp}}`, `T_1`, `\times`, `T_2`, `T_3`
- **Terminal Bridge Handed Off:**
  _The system design principle here reflects Saltzer and Kaashoek's end-to-end argument: when the software supervisor's internal recovery contracts are violated, safety must not depen..._

### Completed Step 5: Section 11.5: Semantic Watchdog Timers
- **File:** `05_sec_11_5.qmd` | **Word Count:** 2,581 words
- **Active Symbols Added:** `\text{GEMV}`, `\mathbf{s}_t`, `a_t`, `o_t`, `t`, `W`, `k`, `E`
- **Terminal Bridge Handed Off:**
  _`ERR_SEMANTIC_DEADLOCK`, revokes the agent's transient capability tokens, releases all acquired POSIX file and database locks, and dispatches the trajectory's Write-Ahead Log to hu..._

### Completed Step 6: Section 11.6: Tool Circuit Breakers
- **File:** `06_sec_11_6.qmd` | **Word Count:** 3,298 words
- **Active Symbols Added:** `\pi_\theta`, `o_t^{\text{err}}`, `C`, `\epsilon`, `N`, `\lambda_{\text{in}}`, `L_{\text{degraded}}`, `T_{\text{timeout}}`
- **Terminal Bridge Handed Off:**
  _By combining tri-state circuit breakers, full jitter backoff, and bulkhead partitioning, the host runtime constructs a resilient execution gateway. The agent's unprivileged, stocha..._

### Completed Step 7: Section 11.7: Blast Radius Quarantine
- **File:** `07_sec_11_7.qmd` | **Word Count:** 2,079 words
- **Active Symbols Added:** `T_i`, `C_i`, `\mathcal{T}_k`, `e_j`, `a`, `\text{TAINTED}`, `\mathcal{T}_{\text{peer}}`, `L_{\text{net}}`
- **Terminal Bridge Handed Off:**
  _By enforcing dynamic taint tracking, the supervisor guarantees that the boundary of failure remains strictly isolated to the misbehaving trajectory. Even in complex, multi-agent ar..._

### Completed Step 8: Section 11.8: Fault-Tolerant System Synthesis
- **File:** `08_sec_11_8.qmd` | **Word Count:** 2,761 words
- **Active Symbols Added:** `a_t`, `\mathcal{L}_{\text{comp}}`, `\mathcal{L}_{\text{wal}}`, `\mathcal{S}_{\text{saga}}`, `\mathcal{G}_{\text{gate}}`, `C_i`, `\mathbf{s}_t`, `o_t`
- **Terminal Bridge Handed Off:**
  _Only the fully synthesized runtime harness (Configuration 4)—uniting the control plane scheduler, WAL event storage, Saga compensators, idempotent reconciliation probes, and verifi..._

### Completed Step 9: Fallacies and Pitfalls
- **File:** `98_fallacies_pitfalls.qmd` | **Word Count:** 1,312 words
- **Active Symbols Added:** `2\text{PC}`, `T_i`, `C_i`, `\mathcal{L}_{\text{wal}}`, `\mathbf{s}_0`, `T_{\text{pivot}}`, `\mathcal{G}_{\text{gate}}`, `S_{\max}`
- **Terminal Bridge Handed Off:**
  _Tool execution layers must implement a three-state circuit breaker pattern (Closed, Open, Half-Open) operated directly by the supervisor plane. When a tool endpoint's failure rate ..._

### Completed Step 10: Summary & Chapter Connection
- **File:** `99_summary_connection.qmd` | **Word Count:** 974 words
- **Active Symbols Added:** `2\text{PC}`, `\mathcal{L}_{\text{wal}}`, `C_i`, `T_i`, `a_{\text{repair}}`, `S_{\max}`, `T_{\text{pivot}}`, `\pi_\theta`
- **Terminal Bridge Handed Off:**
  _Yet runtime resilience, however robust, is an operational mitigation rather than a cure. Continually invoking compensating transactions, tripping circuit breakers, and burning infe..._
