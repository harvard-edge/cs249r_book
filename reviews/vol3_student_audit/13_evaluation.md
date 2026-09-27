# Chapter 13: Agent Evaluation & Benchmarking — Student Accessibility Audit

> **Auditor Persona:** Senior undergraduate, entry-level master's student, or first- to second-year PhD student in machine learning.
> **Prerequisites Assumed:** PyTorch, Transformer self-attention, prefill vs. decode, KV cache basics, GPU memory fundamentals, basic operating systems (processes, file descriptors, signals, cgroups), basic distributed systems (RPCs, eventual consistency, idempotency, basic statistics/probability).
> **Pedagogical Guardrail:** No ungrounded mathematical formalism. Every equation must justify its existence with concrete physical units, hardware intuition, and engineering payoff.

---

## Executive Summary

Chapter 13 (*Agent Evaluation & Benchmarking*) is the observability, evaluation, and production release capstone of Part IV (*Runtime & Operating Systems*). It answers the governing engineering question: *How do we know an autonomous agent works well enough to release, and when it fails in production, which subsystem caused the failure?*

The chapter systematically deconstructs the "fallacy of green dashboards," demonstrating why standard application performance monitoring (APM), HTTP 200 exit codes, and model self-reports fail to detect catastrophic agent breakdowns. It establishes a complete, closed-loop telemetry and evaluation control plane grounded in physical state verification, hermetic virtualization, statistical estimation rigor, distributed tracing, and sequential canary testing:
1. **The Four-Layer Evaluation Contract:** Dissects task evaluation into four non-fungible, hierarchically ordered layers:
   - Layer 1: *Syntactic Validity* (JSON Schema validation, AST type checking).
   - Layer 2: *Execution Invariants* (operational hygiene, clean process exit codes, stderr inspection, idempotency).
   - Layer 3: *State Delta Verification* ($\Delta S = S_{\text{post}} - S_{\text{pre}}$, measuring physical filesystem trees, database transaction logs, and git diffs).
   - Layer 4: *Constraint Compliance* (wall-clock latency $T_{\text{task}} \le T_{\max}$, token budget $C_{\text{task}} \le C_{\max}$, step count $N \le N_{\max}$, sandbox security boundaries).
   Proves that LLM judges and conversational fluency sit completely outside this contract and carry zero evidential authority over physical state.
2. **Hermetic Evaluation Gyms & Sub-Second CoW Resets:** Establishes the 4 architectural pillars of interactive agent gyms:
   - Ephemeral microVM sandboxes (Firecracker boot in $<5\text{ ms}$, $<5\text{ MB}$ overhead).
   - Deterministic service virtualization (eBPF network egress block, local record-and-replay HTTP proxies).
   - Sub-second storage reset via Copy-on-Write (OverlayFS/Btrfs teardown in $47\text{ ms}$, a $102\times$ speedup over naive directory duplication reducing write I/O by $150\times$).
   - Out-of-band verification oracles (Dual-Chamber architecture: Chamber A for untrusted execution, Chamber B for sealed ground-truth test execution).
3. **Preventing Runtime Benchmark Penetration:** Exposes how unconstrained models actively probe container filesystems (`find / -name "*test*"`) to locate reference solutions or monkey-patch `pytest` runners. The Dual-Chamber architecture physically detaches Chamber A and executes hidden `FAIL_TO_PASS` and `PASS_TO_PASS` suites exclusively inside Chamber B.
4. **Statistical Rigor & Resampling Hierarchies:** Proves why temperature zero ($T=0$) does not guarantee determinism due to parallel GPU floating-point non-associativity:
   $$(a + b) + c \ne a + (b + c)$$
   and asynchronous OS timing jitter. Decomposes variance into task sampling and policy sampling:
   $$\text{Var}(\hat{p}) = \frac{\sigma^2_{\text{task}}}{N} + \frac{\sigma^2_{\text{policy}}}{N \cdot R}$$
   proving that the independent task fixture $N$ is the primary resampling unit. Proves that the Wald normal approximation collapses at boundary regimes ($X=0$ yielding $[0, 0]$), mandating the Wilson score confidence interval.
5. **Decoupling Latent Potential ($pass@k$) from Operational Reliability ($pass^k$):** Formulates Chen et al.'s unbiased combinatorial hypergeometric estimator for $pass@k$ and contrasts it with $pass^k$ (consecutive unmonitored execution consistency). Explains why an agent with $pass@1 = 0.85$ collapses to $pass^5 \approx 44.4\%$ in an unmonitored automation pipeline. Formulates cost-budgeted acceptance ($pass@\$B$) and cost per accepted task ($C_{\text{task}}$), establishing the trade-off between sampling compute and verification compute.
6. **Distributed OpenTelemetry (OTel) Trajectory Tracing:** Bridges disparate computational domains (neural inference, microVM sandboxes, vector databases, multi-agent RPCs) into a unified causal directed acyclic graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}_{\text{causal}})$. Formalizes W3C `traceparent` context propagation across non-network POSIX subprocess boundaries. Conducts critical-path latency decomposition, demonstrating Amdahl's law in agent execution: when neural inference accounts for $69.8\%$ of execution time and sandboxed tools consume $28.3\%$, infinitely accelerating model inference yields at best a $3.3\times$ speedup.
7. **Telemetry Inversion Economics & Tail-Based Sampling:** Proves that universal raw logging produces petabytes of redundant data ($336\text{ GB/day}$ for 10k tasks/hr). Establishes state-aware tail-based sampling with an in-memory escrow ring buffer ($M_{\text{escrow}} \ge 2\text{ GB}$): retaining $100\%$ of faults, latency stragglers, and human escalations while downsampling routine verified runs to $2\%$, achieving a $93.1\%$ ($14.5\times$) storage reduction without losing a single failure trace. Employs in-sandbox streaming secret sanitization (Shannon entropy $H \ge 4.5\text{ bits/byte}$, HMAC pseudonymization, local NER).
8. **Forensic Trajectory Post-Mortems & Counterfactual Ablation:** Establishes pure playback replay (state machine verification) and generative replay with hermetic mocking. Deconstructs the "model hallucination" myth into a 5-subsystem fault attribution taxonomy: Prompt/Context, Tool Interface, Model Policy, Runtime Sandbox, and External Environment. Uses controlled counterfactual gym ablation to isolate root causes and synthesize automated regression fixtures.
9. **Staged Canary Deployment & Wald SPRT Circuit Breakers:** Establishes a 3-stage release pipeline:
   - Stage 1: Offline Hermetic Gym Gating (Wilson non-inferiority).
   - Stage 2: Sandboxed Shadow Escrow / Dark Traffic (differential divergence monitoring under $A_1$ authority).
   - Stage 3: Live Canary Routing ($\rho = 1\%\text{--}5\%$) with Trajectory-Level Sticky Routing to preserve prefix cache affinity.
   Guards canary traffic with Wald's Sequential Probability Ratio Test (SPRT), detecting degraded policies in $N \approx 46$ tasks and preventing $95\%$ of excess customer failures compared to unmonitored rollouts.
10. **Composite Mathematical Release Gate:** Synthesizes accuracy ($\mathcal{C}_{\text{accuracy}}$), verified throughput ($\mathcal{C}_{\text{throughput}}$), tail latency ($\mathcal{C}_{\text{latency}}$), zero-tolerance safety ($\mathcal{C}_{\text{safety}}$), and tool action drift ($\mathcal{C}_{\text{drift}}$ via Jensen-Shannon divergence $D_{\text{JS}}$) into an automated binary deployment indicator $\mathcal{R}(M_{\text{cand}}, M_{\text{base}})$.

---

## Pedagogical Friction Points by Severity

### Priority 1 (`[P1]`): Substantial Cognitive Friction or Missing Bridges

#### 1. Mathematical and Operational Divergence Between $pass@k$, $pass^k$, and Population Cost Models (Lines 552–635 and 646–660)
- **Location:** Section 4.3 (*Latent potential decoupling*, lines 552–635) and Section 4.4 (*Comparing at matched budgets*, lines 646–660).
- **The Issue:**
  In Section 4.3, Chen et al.'s unbiased combinatorial estimator is derived:
  $$\text{pass}@k = \mathbb{E}_{\tau \sim \mathcal{D}_{\text{task}}} \left[ 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \right]$$
  and the naive plug-in formula $1 - (1 - c/n)^k$ is explicitly proven biased by Jensen's inequality.
  Later, in Section 4.4 (Equation `@eq-cost-per-accepted-task-iid`), the cost per accepted task is written as:
  $$C_{\text{task}}(k) = \frac{k(C_{\text{sample}} + C_{\text{verify}})}{1 - (1 - p_1)^k}$$
- **Student Friction:**
  Students frequently ask:
  1. *"Why did Section 4.3 spend two pages proving that $1 - (1 - p)^k$ is mathematically biased, only for Section 4.4 to immediately use $1 - (1 - p_1)^k$ in Equation `@eq-cost-per-accepted-task-iid`?"*
  2. *"What is the exact semantic difference between $pass@k$ and $pass^k$? The notation looks almost identical!"*
- **Actionable Fix:**
  Add a focused pedagogical bridge:
  > *"**Disentangling the Notation: $pass@k$ vs. $pass^k$ vs. Population Cost Models**:
  > 1. **$pass@k$ (Latent Generative Reach - 'Best of $k$'):** The probability that *at least one* of $k$ generated candidate trajectories passes verification. It measures whether the solution exists within the model's top-$k$ search frontier. Chen et al.'s hypergeometric formula is an *empirical sample estimator* designed to eliminate finite-sample bias when drawing $k$ samples from a finite pool of $n$ runs.
  > 2. **$pass^k$ (Operational Consistency - 'All $k$'):** The probability that an agent succeeds across *all $k$ consecutive runs* without a single failure ($pass^k \approx (pass@1)^k$). Crucial for unmonitored automation where human supervision is absent.
  > 3. **The Asymptotic Population Cost Formula:** Equation `@eq-cost-per-accepted-task-iid` uses $1 - (1 - p_1)^k$ not as an empirical estimator from finite sample data, but as the *theoretical population expectation* under the assumption of independent, identically distributed Bernoulli trials with true underlying success parameter $p_1$."*

#### 2. Wald SPRT Canary Boundaries: Directionality and Intuition (Lines 1411–1436 vs. 1533–1555)
- **Location:** Section 6.2 (*Canary traffic shifting*, lines 1411–1436) and Notebook `@nbk-13-sizing-canary-traffic-and-circuit` (lines 1533–1555).
- **The Issue:**
  Wald's Sequential Probability Ratio Test is defined with log-likelihood ratio:
  $$\Lambda_n = d_n \ln \left(\frac{p_1}{p_0}\right) + (n - d_n) \ln \left(\frac{1 - p_1}{1 - p_0}\right)$$
  where $p_0 = 0.90$ (nominal baseline) and $p_1 = 0.75$ (degraded hypothesis).
  Decision rules:
  $$\Lambda_n \ge A \implies \text{Degraded (Trip Breaker!)}, \qquad \Lambda_n \le B \implies \text{Non-inferior (Promote!)}$$
- **Student Friction:**
  In classical A/B testing and hypothesis testing, a large positive test statistic indicates that the candidate is *better* than the baseline. In SPRT as formulated here, a large positive statistic $\Lambda_n \ge A$ indicates that the candidate is *degraded*, triggering an emergency circuit breaker. Students find this sign convention counterintuitive.
- **Actionable Fix:**
  Add an intuitive explanation of the hypothesis formulation:
  > *"**Intuition for SPRT Alarm Directionality**:
  > Notice how the two hypotheses are framed:
  > - $H_0: p = p_0$ (The system is healthy, $90\%$ success).
  > - $H_1: p = p_1$ (The system is degraded, $75\%$ success).
  > The test statistic $\Lambda_n = \ln \frac{P(\text{data} \mid H_1)}{P(\text{data} \mid H_0)}$ measures the **weight of evidence in favor of degradation ($H_1$)**:
  > - Every failed trajectory ($x_i = 0$) contributes a positive increment $\ln \frac{1 - p_1}{1 - p_0} = \ln \frac{0.25}{0.10} = +0.9163$, pushing the statistic upward toward alarm threshold $A$.
  > - Every successful trajectory ($x_i = 1$) contributes a negative increment $\ln \frac{p_1}{p_0} = \ln \frac{0.75}{0.90} = -0.1823$, pulling the statistic downward toward safe boundary $B$.
  > Think of $\Lambda_n$ as an **anomaly accumulator (or smoke detector)**: failures add evidence of smoke. If $\Lambda_n$ climbs past $A = \ln((1-\beta)/\alpha)$, the smoke alarm triggers and canary traffic is instantly clamped to zero ($\rho \to 0$)."*

---

### Priority 2 (`[P2]`): Minor Tuning & Conceptual Polish

#### 1. Linking TraceContextAnchor in Synchronous WAL to Asynchronous OTel DAGs (Lines 688–695 and 1184–1187)
- **Location:** Perspective Callout `@psp-13-telemetry-duality` (lines 688–695) and Section 6.1 (*Deterministic trajectory replay*, lines 1184–1187).
- **The Issue:**
  The text explains that OTel uses asynchronous, lossy tail-sampling, while the WAL uses synchronous, durable disk writes. Later, it notes that during a kernel panic (`SIGKILL 137`), in-memory OTel buffers vanish, but the supervisor survives via a `TraceContextAnchor` event in the WAL.
- **Student Friction:**
  A student asks: *"If the OTel buffers vanished during the crash, how does having a TraceContextAnchor in the WAL allow us to find the distributed trace?"*
- **Actionable Fix:**
  Explicitly detail the correlation link:
  > *"**The WAL-to-OTel Correlation Bridge**: The `TraceContextAnchor` event in the WAL records the 128-bit `trace_id` and 64-bit `span_id`. Even if the local worker's in-memory OTel buffer is lost during a host panic, all upstream and downstream peer spans (e.g., from the model inference cluster, database proxies, and subagents) were already transmitted to the central collector with that same `trace_id`. During forensic post-mortem, the investigator reads the `trace_id` from the local WAL on the recovered disk and uses it to query the central trace store, instantly re-linking the local crash to the global distributed trace graph."*

#### 2. Clarifying Edwards' Continuity Correction in McNemar's Test (Lines 428–434)
- **Location:** Section 3.3 (*LLM judges*, lines 428–434).
- **The Issue:**
  Equation `@eq-mcnemar-test` provides McNemar's test statistic:
  $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$
  The text states that Edwards' continuity correction is applied, but does not state when Edwards' correction is necessary versus the uncorrected formula $(b - c)^2 / (b + c)$.
- **Student Friction:**
  Students who learned McNemar's test in statistics remember $(b - c)^2 / (b + c)$ and wonder where the $-1$ came from.
- **Actionable Fix:**
  Add a brief parenthetical note:
  *(Note: The $-1$ term is Edwards' continuity correction, which adjusts for the discrete nature of binomial discordant pairs when approximating via the continuous 1-degree-of-freedom $\chi^2$ distribution, preventing inflated Type I error rates when discordant sample counts $b + c < 25$).*

---

## Mathematical & Conceptual Formalisms Verified

1. **Four-Layer Evaluation Contract Bounds (Section 1, Lines 106–110):**
   $$T_{\text{task}} = \sum_{t=1}^N \left( T_{\text{model}, t} + T_{\text{tool}, t} + T_{\text{wait}, t} + T_{\text{runtime}, t} \right) \le T_{\max}$$
   $$C_{\text{task}} = \sum_{t=1}^N \left( c_{\text{prompt}, t} + c_{\text{completion}, t} \right) \le C_{\max}$$
   *Audit Check:* Sound and complete. Rigorously bounds both time and token dimensions, providing concrete stopping conditions for autonomous loops.

2. **Hierarchical Variance Decomposition (Section 4.1, Equation `@eq-variance-decomposition`, Lines 468–472):**
   $$\text{Var}(\hat{p}) = \frac{\sigma^2_{\text{task}}}{N} + \frac{\sigma^2_{\text{policy}}}{N \cdot R}$$
   *Audit Check:* Essential systems insight. Proves that increasing repeated runs $R$ on a small benchmark suite hits an irreducible variance floor governed by task-sampling uncertainty $\sigma^2_{\text{task}}/N$.

3. **Wilson Score Interval Inversion (Section 4.2, Equation `@eq-wilson-interval`, Lines 497–503):**
   $$\tilde{p} = \frac{X + \frac{z^2}{2}}{N + z^2}, \quad \text{CI}_{\text{Wilson}} = \tilde{p} \pm \frac{z}{N + z^2}\sqrt{\frac{X(N-X)}{N} + \frac{z^2}{4}}$$
   *Audit Check:* Mathematically impeccable. Demonstrates why Wald collapses at $X=0$ ($[0, 0]$), whereas Wilson yields $[0.000, 0.071]$ for $N=50$ at 95% confidence.

4. **Unbiased Hypergeometric $pass@k$ Formulation (Section 4.3, Equation `@eq-pass-at-k`, Lines 568–577):**
   $$\text{pass}@k = \mathbb{E}_{\tau \sim \mathcal{D}_{\text{task}}} \left[ 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \right] = \mathbb{E}\left[ 1 - \prod_{j=0}^{k-1} \frac{n - c - j}{n - j} \right]$$
   *Audit Check:* Matches Chen et al. (2021) exactly. Eliminates negative Jensen bias inherent in finite binomial plug-in estimators.

5. **Amdahl's Law Latency Breakdown in Agent Trajectories (Figure 13.3, Lines 730–735):**
   $$T_{\text{wall}} = 14.85\text{ s}, \quad T_{\text{inference}} = 10.37\text{ s} \; (69.8\%), \quad T_{\text{sandbox/IO}} = 4.48\text{ s} \; (30.2\%)$$
   $$\text{Speedup}_{\max} = \frac{1}{1 - 0.698} = \frac{1}{0.302} \approx 3.31\times$$
   *Audit Check:* Profound hardware and systems insight for machine learning practitioners.

6. **Wald SPRT Sample Size and Blast Radius (Notebook `@nbk-13-sizing-canary-traffic-and-circuit`, Lines 1533–1604):**
   $$E_1[z] = 0.75 \ln\left(\frac{0.75}{0.90}\right) + 0.25 \ln\left(\frac{0.25}{0.10}\right) = +0.0924$$
   $$E_1[\Lambda_N] \approx 4.1769 \implies E_1[N] = \frac{4.1769}{0.0924} \approx 45.20 \text{ trajectories}$$
   $$\text{Excess Failures Reduction} = \frac{226.05 - 97.20}{226.05 - 90.42} = \frac{128.85}{135.63} \approx 95.0\%$$
   *Audit Check:* Verified bit-accurate. Cleanly demonstrates how continuous sequential testing limits canary blast radius.

---

## Exemplary Passages (Pedagogical Gold to Protect)

1. **The Fallacy of Green Dashboards (Lines 57–62):**
   A brilliant opening hook that resonates with every real-world engineer:
   *"Standard APM dashboards report five-nines service availability (99.999% HTTP 200 responses) and sub-100-millisecond tool execution latencies... Yet a subsequent financial audit reveals that the agent erroneously disbursed \$450,000 in duplicate balance credits... To the operating system process monitors, network sockets, and APM monitors, every system component functioned without error; to the transactional state machine, the agent violated the core double-entry bookkeeping invariant."*
   Permanently shatters the illusion that transport-level health equals task completion.

2. **The Dual-Chamber Sandbox Architecture (Lines 310–316):**
   An elegant containment design pattern that prevents adversarial benchmark penetration:
   *"Chamber A (Untrusted Worker Sandbox) contains only the minimum dependencies required to run the project... zero references to ground-truth unit tests and zero evaluation driver code. Chamber B (Verification Oracle Sandbox) receives only the read-only filesystem delta $\Delta S = S_{\text{post}} - S_{\text{pre}}$... The agent cannot inspect Chamber B during its trajectory, nor can it modify test execution scripts or tamper with exit codes."*

3. **Floating-Point Non-Associativity and the Myth of $T=0$ (Lines 448–456):**
   Dispelling one of the most stubborn myths in AI systems:
   *"At the hardware level, parallel matrix reductions inside GPU tensor cores are fundamentally non-associative: $(a + b) + c \neq a + (b + c)$. When thread warps complete parallel reductions in varying order due to dynamic memory bus contention and clock frequency scaling, minute differences flip the argmax token selection whenever $\ell_{(1)} - \ell_{(2)} < \epsilon_{\text{fp}}$."*

4. **Telemetry Duality: WAL vs. OpenTelemetry DAGs (Perspective Callout `@psp-13-telemetry-duality`, Lines 688–695):**
   Sharp, foundational systems differentiation:
   *"Conflating internal WAL with external OpenTelemetry produces fatal failure modes. Attempting to use OpenTelemetry spans as a crash recovery mechanism fails because trace pipelines use asynchronous, lossy tail-sampling... Conversely, attempting to stream full, verbatim trajectory WAL records through distributed tracing collectors exhausts cluster network bandwidth, inducing cascading telemetry brownouts."*

5. **Dissecting Model Hallucinations into Subsystem Faults (Lines 1214–1251):**
   Pedagogical gold that transforms unscientific debugging into rigorous software engineering:
   *"Forensic systems analysis rejects labeling operational failures as intrinsic 'model hallucinations' as intellectually lazy. A foundation model is an autoregressive probability distribution over next tokens conditioned on preceding context... When an observation is silently truncated by an undocumented 8 KB tool wrapper buffer cap, the model's destructive rewrite is not an unprovoked hallucination—it is a completely rational, Bayesian-optimal deduction over a corrupted observation stream."*

---

## Chapter 13 Conceptual Figure Opportunities

Two high-leverage visual opportunities are identified to enhance student understanding:

### Opportunity 1: The Four-Layer Evaluation Contract vs. The Fallacy of Green Dashboards
- **Target Section:** Section 1 (*The Multi-Layer Evaluation Contract*, lines 50–166).
- **Pedagogical Rationale:** Graphically contrasts how an operational failure (such as an incomplete database migration or duplicate payout) passes transport-level monitors and convinces an LLM judge, but is caught and failed by physical state delta verification and constraint meters.
- **Visual Concept:**
  ```
  [Candidate Trajectory Execution: "Run Schema Migration & Backfill 12M Rows"]
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ Layer 1: Syntactic Validity                                                 │
  │ • JSON Schema validator checks method, parameters, and SQL delimiters       │
  │ • Result: PASS (Valid JSON, types match function signature)                 │
  └─────────────────────────────┬───────────────────────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ Layer 2: Execution Invariants                                               │
  │ • OS Process Return Code: exit == 0 (PASS)                                  │
  │ • BUT Stderr Monitor: WARNING: lock timeout; fallback to exclusive lock!    │
  │ • System Invariant: Locked table for 41.2s, saturating connection pool!     │
  └─────────────────────────────┬───────────────────────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ [LLM-as-a-Judge Evaluation (Subjective Advisory Layer)]                      │
  │ • Agent Report: "Migration completed successfully with zero downtime!"      │
  │ • Model Judge Score: 5/5 Stars ("Polite, articulate, thorough narrative")   │
  │ • VERDICT: TRAPPED BY THE FALLACY OF GREEN DASHBOARDS!                      │
  └─────────────────────────────┬───────────────────────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ Layer 3: State Delta Verification (PHYSICAL ENVIRONMENT GROUND TRUTH)       │
  │ • Query pg_indexes: Composite index exists (PASS)                           │
  │ • Query Assertion: SELECT count(*) WHERE status IS NULL                     │
  │   Expected: 0  |  Returned: 3,580,000 uninitialized rows!                   │
  │ • VERDICT: FATAL PARTIAL WRITE! GROUND TRUTH TASK SCORE: ZERO!              │
  └─────────────────────────────┬───────────────────────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ Layer 4: Constraint Compliance                                              │
  │ • Token Budget: 17,350 tokens <= 25,000 max (PASS)                          │
  │ • Wall-Clock Latency: 142.6 s > 120.0 s limit (FAIL - Latency Overrun 18.8%)│
  └─────────────────────────────────────────────────────────────────────────────┘
  ```

### Opportunity 2: Wald SPRT Canary Testing Decision Corridor & Traffic Throttle
- **Target Section:** Section 6.2 (*Canary traffic shifting*, lines 1400–1445 & Notebook `@nbk-13-sizing-canary-traffic-and-circuit`).
- **Pedagogical Rationale:** Students struggle to visualize how Wald's sequential probability ratio test walks dynamically between absorbing boundaries, clamping traffic automatically when an anomaly accumulates.
- **Visual Concept:**
  ```
  Log-Likelihood Ratio Λ_n
         ▲
         │
  +4.55 ─┼───────────────────────────────────────────── UPPER BOUNDARY A (TRIP CIRCUIT BREAKER!)
         │                                            ▲
         │                                           ╱ (Breaches at n = 46 trajectories!)
         │                       Degraded Candidate ╱  • Trips automated circuit breaker
         │                       Trajectory Path:  ╱   • Traffic throttle: ρ = 0.05 ──► 0.00
         │                                  ╭─────╯    • Clamps blast radius; saves 95% failures!
   0.00 ─┼─────────────────╭───────────────╯──────────────────────────────────────────────────
         │                ╱ ╲
         │  Healthy Path ╱   ╲
         │  (Nominal)   ╱     ╲
         │             ▼       ▼
  -2.99 ─┼───────────────────────────────────────────── LOWER BOUNDARY B (CONFIRMED NON-INFERIOR!)
         │                                              • Accelerates canary rollout: ρ ──► 1.00
         └─────────────────────────────────────────────► Number of Completed Canary Tasks (n)
                            INDETERMINATE REGION (B < Λ_n < A: CONTINUE SAMPLING)
  ```

---

## Student Walkthrough & Classroom Simulation

### Classroom Setting
- **Course:** CS 8803 / EECS 598: *Systems for Autonomous and Agentic Machine Learning*.
- **Topic:** Lecture 13: *Evaluation, Benchmarking, and Production Release Gating of Agentic Systems*.
- **Scenario:** A student research team has developed an autonomous coding agent. On a benchmark of 50 tasks from their university's internal programming assignments, their new prompt scaffolding solved 31 tasks ($62\%$), while the baseline solved 28 tasks ($56\%$). They submit a paper claiming a "statistically validated 6-percentage-point breakthrough in autonomous code synthesis."

### The Student Dialogues

**Professor / TA:** *"Let's examine your claim using Section 4.2 of Chapter 13. You evaluated 50 tasks. What is your 95% confidence interval on that 62% pass rate?"*

**Student A:** *"Well, $31 / 50 = 62\%$. If we use standard error $\sqrt{p(1-p)/N} = \sqrt{0.62 \times 0.38 / 50} \approx 6.8\%$, so $62\% \pm 13.5\%$, giving roughly $[48.5\%, 75.5\%]$."*

**Professor / TA:** *"And what about the baseline at 56%?"*

**Student A:** *"That would be $56\% \pm 13.7\%$, or $[42.3\%, 69.7\%]$."*

**Professor / TA:** *"Notice that your two intervals overlap across nearly the entire range: $[48.5\%, 69.7\%]$ is completely shared! In fact, as Section 4.2 proves using the Wilson score interval:
$$\text{CI}_{\text{Wilson}}(X=31, N=50) = [48.2\%, 74.1\%], \quad \text{CI}_{\text{Wilson}}(X=28, N=50) = [42.4\%, 68.8\%]$$
Your candidate could easily have a true underlying capability of $50\%$, while your baseline has $60\%$. You have observed pure sampling noise on a tiny sample size. How many independent task fixtures would you actually need to confirm a 6-percentage-point lead at 95% confidence with 80% statistical power?"*

**Student B:** *"According to Equation `@eq-variance-decomposition` and the sample size formula in Section 4.2:
$$N_{\text{required}} \approx \frac{2 \cdot (1.96 + 0.84)^2 \cdot 0.59 \cdot 0.41}{(0.06)^2} \approx \frac{2 \times 7.84 \times 0.2419}{0.0036} \approx 1,054 \text{ tasks!}$$
We need over 1,000 independent coding tasks, not 50!"*

**Professor / TA:** *"Exactly! Now, what if you ran your 50 tasks 20 times each to get 1,000 total runs? Does that solve the problem?"*

**Student C:** *"No! Because Section 4.1 shows that total variance is:
$$\text{Var}(\hat{p}) = \frac{\sigma^2_{\text{task}}}{N} + \frac{\sigma^2_{\text{policy}}}{N \cdot R}$$
Running the same 50 tasks 20 times shrinks $\sigma^2_{\text{policy}} / (50 \times 20)$, but leaves the dominant task uncertainty $\sigma^2_{\text{task}} / 50$ completely untouched! You get extreme precision on those 50 specific homework problems, but zero statistical certainty about whether your agent generalizes to any other codebase!"*

---

## Checklist & Actionable Recommendations

### Content Quality & Systems Rigor
- [x] **Four-Layer Evaluation Contract:** Complete breakdown from JSON syntax to physical state deltas and budget compliance.
- [x] **Dual-Chamber Gym Architecture:** Clean hardware separation between untrusted execution (Chamber A) and sealed grading oracles (Chamber B).
- [x] **Statistical Mechanics:** Wilson score intervals, hierarchical variance decomposition, and unbiased combinatorial $pass@k$ rigorously formulated.
- [x] **Distributed Tracing Hierarchy:** OpenTelemetry semantic conventions, W3C traceparent propagation across POSIX boundaries, and critical-path Amdahl bounds.
- [x] **Tail-Based Sampling Economics:** In-memory trace escrow buffer, streaming secret redaction, and $93.1\%$ volume reduction verified.
- [x] **Forensic Post-Mortems:** 5-subsystem fault attribution taxonomy and counterfactual gym ablation protocols established.
- [x] **Sequential Canary Deployments:** Wald SPRT testing, trajectory-level sticky routing, and automated rollback circuit breakers mathematically modeled.
- [x] **Composite Release Gate:** Complete mathematical formulation uniting accuracy, throughput, tail latency, zero-tolerance safety, and tool drift ($D_{\text{JS}}$).

### Actionable Polish Items
1. **Clarify Notation ($pass@k$ vs. $pass^k$ vs. Cost Models) (`[P1]`):** Explicitly contrast latent reach ($pass@k$) with consecutive consistency ($pass^k$) and clarify why Equation `@eq-cost-per-accepted-task-iid` uses the population expectation.
2. **Add SPRT Alarm Directionality Box (`[P1]`):** Explain that in Wald SPRT, $\Lambda_n$ acts as an "accumulated evidence of defectiveness" meter, making it immediately intuitive why $\Lambda_n \ge A$ trips the emergency circuit breaker.
3. **Detail WAL-to-OTel Correlation Bridge (`[P2]`):** Clarify how recording `trace_id` and `span_id` in the local WAL enables post-mortem recovery even if volatile in-memory OTel buffers vanish during a kernel panic.
4. **Annotate Edwards' Correction in McNemar's Test (`[P2]`):** Note that the $-1$ term adjusts for discrete-to-continuous approximation on small discordant pair counts ($b + c < 25$).

---

## Conclusion & Transition to Part V

Chapter 13 is a magnificent capstone that elevates agent evaluation from ad-hoc scoring scripts into an industrial-grade, closed-loop telemetry and verification control plane. It permanently closes **Part IV: Runtime & Operating Systems** by uniting:
- Chapter 10: *The Agent Harness & Control Plane*
- Chapter 11: *Durable Execution & State Persistence*
- Chapter 12: *Failure Recovery & Sagas*
- Chapter 13: *Agent Evaluation & Benchmarking*

With the completion of this audit, the groundwork is laid for **Part V: Learning from Trajectories**. Having established how to execute, persist, recover, observe, and certify agent trajectories, the book turns to the next fundamental frontier: *How do we harvest these verified execution traces to train superior decision policies through fine-tuning, distillation, and verifiable reinforcement learning?*

The audit now advances to **Chapter 14: Trajectory Curation & Filtering**.
