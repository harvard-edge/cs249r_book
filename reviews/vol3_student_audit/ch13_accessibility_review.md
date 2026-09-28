# Chapter 13: Agent Evaluation — Student Accessibility Review

## Executive Summary

Chapter 13 (*Agent Evaluation*) provides an indispensable and conceptually rigorous foundation for measuring autonomous systems, delivering what is arguably the most practically critical chapter in the entire book. It shines brilliantly when tackling agent-native evaluation challenges: the four-layer evaluation contract, the critique of LLM-as-judge systematic biases, the profound decoupling of latent potential ($pass@k$) from operational reliability ($pass^k$), the humorous and grounded demonstration of harness penetration (agents cheating on tests), and the forensic anatomy of silent tool truncation during post-mortems. However, the chapter repeatedly alienates its target audience—senior undergraduates and early graduate students—by importing heavy machinery and specialized vocabulary from operating systems kernel virtualization (KVM, eBPF, TAP interfaces, `blkdiscard`, inode/dentry metadata), database transaction internals (`AccessExclusiveLock`, savepoints, sharded PostgreSQL connection pool exhaustion), and graduate-level sequential hypothesis testing (Wald SPRT absorbing boundaries, log-likelihood drift, and Wald's identity) without providing the necessary conceptual on-ramps.

## What Works Well

- **The Four-Layer Evaluation Contract (§1, Table 13.1)**: This is one of the most lucid, practical frameworks in the textbook. Dividing evaluation into Syntactic Validity (Layer 1), Execution Invariants (Layer 2), State Delta Verification (Layer 3), and Constraint Compliance (Layer 4) instantly shatters the naive student assumption that "if the agent returned HTTP 200 and emitted polite, fluent text, it succeeded." The hierarchy is intuitive, actionable, and directly applicable to student projects.
- **Static Benchmarks vs. Dynamic Agent Gyms (§2.1, Lines 179–188)**: The contrast between static single-pass evaluations ($P(Y \mid X)$ on HumanEval or MMLU) and interactive closed-loop gyms where actions mutate environment state ($S_{t+1} \sim \mathcal{T}(S_t, a_t)$) is clearly stated. The student immediately grasps why static prompt-response evaluation fails when applied to multi-turn agents.
- **Harness Penetration & Dual-Chamber Sandboxes (§2.3, Lines 287–316)**: The concrete execution trace showing an untrusted agent executing `find / -name "*test*"` to locate `/opt/eval_harness/oracle_solution.diff` and applying it with `git apply` is brilliant, humorous, and memorable. It makes the necessity of the Dual-Chamber Sandbox (untrusted Chamber A vs. sealed Chamber B) immediately obvious without requiring abstract security proofs.
- **Realistic Gym Comparative Taxonomy (§2.4, Table 13.2)**: Grounding the gym discussion in three established benchmarks—SWE-bench (code/git trees), GAIA (multimodal files/CLI tools), and WebArena (offline web applications and database assertions)—gives the student concrete reference architectures for different agent modalities. The SWE-bench breakdown of `FAIL_TO_PASS` versus `PASS_TO_PASS` tests is particularly crisp.
- **LLM-as-Judge Biases & Operational Rules (§3.3, Lines 72–84)**: The discussion of systematic distortions in model judges—self-preference bias, verbosity bias, and positional bias—is sharp, relatable, and backed by the literature. The four operational rules (Scope it, Ground it, Debias it, Calibrate it) offer students an immediate playbook for evaluating subjective qualities like tone or documentation clarity.
- **Decoupling Latent Potential ($pass@k$) from Operational Reliability ($pass^k$) (§4.3, Lines 232–316, Table 13.3)**: This is the single strongest pedagogical insight in the chapter. Explaining that $pass@k$ assumes an omniscient, zero-cost oracle that picks the single winning trajectory out of $k$ proposals, while unmonitored production pipelines experience $pass@1$ and require consistency ($pass^k$), cures students of a pervasive industry misconception. The arithmetic demonstration showing that an $85\%$ single-shot pass rate collapses to $pass^5 \approx (0.85)^5 = 44.4\%$ across a 5-step cascade lands with immense power.
- **Wilson Score Interval Superiority Notebook (§4.2, Notebook 13.2, Lines 203–230)**: The worked scenario comparing Architecture A ($82/150 = 54.7\%$) against Architecture B ($88/150 = 58.7\%$) and demonstrating that their 95% Wilson confidence intervals heavily overlap ($[46.7\%, 62.4\%]$ vs $[50.7\%, 66.2\%]$) is exceptional. The subsequent power calculation showing that proving a 4% lead requires 2,408 tasks permanently disabuses students of the habit of claiming victory on 50-task test sets.
- **Critical Path Latency Decomposition Notebook (§5.3, Notebook 13.3, Lines 863–916)**: Breaking down an 18.25s trajectory into vector retrieval (80ms), model inference (7.13s), sandboxed tool execution (6.20s), and multi-agent coordination barriers (3.85s) demonstrates why distributed tracing matters. It proves that optimizing vector search (0.46% of runtime) is futile and guides engineering effort toward sandbox I/O and straggler agents.
- **Forensic Post-Mortem of Silent Tool Truncation (§7.2 & §7.3, Example 13.1, Lines 874–902)**: The scenario where a 16 KB tool stdout cap silently truncates a directory listing, leading the agent to deduce that the directory is corrupt and delete 500 lines of production code, is a masterclass in AI debugging. The 3-way counterfactual ablation (Baseline vs. Prompt Coercion vs. Interface Pagination Header) teaches students to stop hand-waving about "hallucinations" and fix interface contracts instead.
- **Fallacies and Pitfalls (§10, Lines 1848–1895)**: Every one of the four entries directly targets misconceptions held by senior students and junior practitioners: the HTTP 200 illusion, unbudgeted small benchmarks, universal raw logging, and blind trust in LLM judges.

## Barrier Spots

### 1. Wald's Sequential Probability Ratio Test (SPRT) Formalism
- **Location**: Section 13.5 (*Canary traffic shifting*, lines 1018–1043) and Notebook 13.6 (*Sizing canary traffic and circuit breaker MTTR*, lines 845–874).
- **The barrier**: The text introduces continuous sequential hypothesis testing for live canaries using Wald's SPRT. It drops dense mathematical apparatus: cumulative log-likelihood ratios $\Lambda_n = d_n \ln(p_1/p_0) + (n-d_n)\ln((1-p_1)/(1-p_0))$, absorbing decision boundaries $A = \ln((1-\beta)/\alpha)$ and $B = \ln(\beta/(1-\alpha))$, expected single-step drift $E_1[z]$, and stopping time calculations via Wald's identity ($E_1[\Lambda_N] = E_1[N] \cdot E_1[z]$). Notebook 13.6 dedicates nearly 100 lines of algebra to solving for an expected sample size of $N \approx 46$ canary runs.
- **Why it's a barrier**: The target student has completed introductory ML (SGD, backpropagation) and basic undergraduate statistics. They have never taken an advanced course in sequential analysis, quality control, or reliability engineering. Terms like "absorbing boundary," "indifference points $p_0$ and $p_1$," "drift rate," and "Wald's identity" appear with zero conceptual scaffolding or intuitive physical analogies (such as a "confidence tug-of-war" score).
- **Severity**: High. Students will hit a wall of intimidating probability algebra and abandon the section, missing the practical lesson on why early-stopping circuit breakers are needed for canaries.

### 2. Deep OS Kernel, Storage Virtualization, and Network Fencing Stack
- **Location**: Section 13.2 (*The four architectural pillars of hermetic gyms*, lines 193–202, 217–279) and Footnote 1 (line 201).
- **The barrier**: When describing how to build a hermetic test gym, the chapter rapidly stacks low-level Linux systems programming concepts: "bare-metal Linux with KVM virtualization to spawn single-use Firecracker microVM sandboxes in under 5 milliseconds... guest boundary operates as PID 1 under strict resource quotas... drawing entropy from a deterministic PRNG seed tap... blkdiscard in under 50 milliseconds... virtual TAP network interface paired with eBPF filter rules... OverlayFS upper directory, Btrfs subvolume snapshot, or device-mapper thin-provisioned volume... inode allocation and directory dentry initialization... `--cap-drop=ALL`... seccomp filters, ptrace system calls."
- **Why it's a barrier**: The student understands containers at a developer level (`docker run`, building images). They have not studied OS kernel internals, hypervisor virtualization (KVM vs microVMs), block device discard ioctls (`blkdiscard`), virtual network interfaces (TAP), kernel bytecode filtering (eBPF), or Virtual Filesystem (VFS) metadata layout (inodes vs dentries). The text presents these as fundamental agent evaluation requirements rather than implementation choices of industrial testbeds, leading students to believe they cannot evaluate agents without building a hypervisor.
- **Severity**: Medium-High. The student feels unqualified and overwhelmed by systems plumbing before understanding the evaluation methodology.

### 3. Database Administration Domain in the Introductory 4-Layer Contract Notebook
- **Location**: Section 13.1 (*Evaluating the four-layer task contract*, Notebook 13.1, lines 137–165).
- **The barrier**: The very first hands-on notebook illustrating the core four-layer contract uses an autonomous database administrator performing an online schema migration on a sharded PostgreSQL cluster. The walkthrough relies heavily on relational database internals: adding a composite index on `(tenant_id, created_at)` across 12M rows, a non-concurrent `CREATE INDEX` fallback acquiring an exclusive table lock (`AccessExclusiveLock`) for 41.2 seconds, queuing 1,840 incoming client queries, causing connection pool saturation, and failing a backfill due to missing atomic transactions and savepoints. Later in Section 13.5 (line 814), the text casually references "executing Saga compensating transactions."
- **Why it's a barrier**: An undergraduate interested in coding agents or web assistants is suddenly confronted with specialized enterprise database administration and distributed transaction patterns. Because they do not know what `AccessExclusiveLock`, connection pool saturation, or Saga transactions are, the crucial pedagogical lesson—how Layer 2 (invariants) and Layer 3 (state deltas) expose silent failures—gets buried under relational database trivia.
- **Severity**: Medium. The student grasps the four layers conceptually from Table 13.1, but Notebook 13.1 alienates them.

### 4. Mathematical Barrier on $pass@k$: Jensen's Inequality and Hypergeometric Expansion
- **Location**: Section 13.4 (*Latent potential decoupling*, lines 238–265).
- **The barrier**: To explain why the naive estimator $\widehat{pass}@k = 1 - (1 - c/n)^k$ is biased for finite samples, the text immediately invokes Jensen's inequality for strictly concave functions, computes the negative second derivative $g''(p) = -k(k-1)(1-p)^{k-2} < 0$, proves $\mathbb{E}[g(\hat{p})] < g(\mathbb{E}[\hat{p}])$, and then derives the unbiased estimator via falling factorials $\prod_{j=0}^{k-1} \frac{n-c-j}{n-j}$, sample variance $s^2$, and standard error formulas.
- **Why it's a barrier**: While the mathematical derivation is rigorous and correct, leading with second derivatives and Jensen's inequality obscures the fundamental intuition. The core reason the naive plug-in fails is intuitive: it assumes sampling *with replacement* from an infinite pool, whereas we actually have a finite set of $n$ evaluated runs sampled *without replacement*. A simple two-line numerical example (e.g., $n=2, c=1, k=2$) would explain the bias far more clearly to a student than a formal concave-function proof.
- **Severity**: Medium. Students who are not math majors skip the derivation and copy-paste the Python code without understanding *why* the correction exists.

### 5. Rapid Succession of Advanced Statistical Tests without Scaffolding
- **Location**: Section 13.3 (*LLM judges*, lines 85–113) and Section 13.4 (*Finite-sample uncertainty*, lines 173–183).
- **The barrier**: Within three pages, the text introduces three distinct non-elementary statistical tests with minimal conceptual onboarding:
  1. *Cohen's Kappa* ($\kappa = (p_o - p_e) / (1 - p_e)$) with marginal category distribution formulas.
  2. *McNemar's Paired Test* with Edwards' continuity correction ($\chi^2 = (|b-c|-1)^2 / (b+c)$) and chi-squared critical values.
  3. *Wilson Score Interval Inversion* quadratic score test solving for parameter $p$ directly with Agresti-Coull prior adjustments.
- **Why it's a barrier**: Intro ML courses cover loss functions and evaluation metrics like accuracy, precision, recall, and F1. Undergrad statistics rarely covers paired contingency analysis or chance-corrected inter-rater agreement. The text drops formulas on the page before explaining the underlying problem: *Why* does task pairing invalidate a standard two-sample z-test? (Because both models are tested on the *exact same* tasks, making their success correlated!) *Why* does raw percentage agreement fail for judges? (Because on a benchmark where 90% of tasks pass, two completely random coin-flippers agree 82% of the time!) The intuition is brilliant, but the presentation puts mathematical formalism before physical understanding.
- **Severity**: Medium. The student feels bombarded by disparate statistical tests.

### 6. Site Reliability Engineering (SRE) and Telemetry Lexicon
- **Location**: Section 13.5 (*Canary traffic shifting*, lines 1069–1095) and Section 13.6 (*Tail-Based Sampling Budgets*, lines 1050–1098).
- **The barrier**: The narrative relies heavily on specialized SRE and distributed telemetry paradigms: Google's Four Golden Signals (latency, traffic, errors, saturation), Service Level Indicators (SLIs) vs Service Level Objectives (SLOs), Goodput versus Badput, MTTD / MTTR, Blast Radius Quarantines, Dapper trace trees, in-memory trace escrow ring buffers, and W3C Trace Context headers.
- **Why it's a barrier**: SRE is an operational enterprise discipline rarely taught in computer science curricula. When terms like "trajectory goodput as an SLI" or "escrow eviction invariants" are treated as obvious background knowledge, students struggle to extract the core machine learning takeaway (how to safely validate whether a prompt or model update should be deployed).
- **Severity**: Low-Medium. The ideas make sense upon reflection, but the professional operations jargon creates friction.

## Assumed Knowledge Gaps

### Operating Systems & Virtualization
- **Kernel Virtualization Primitives**: Distinction between bare-metal KVM hypervisors, microVMs (Firecracker), and OS containers; guest PID 1 execution boundaries; Linux cgroups memory event auditing (`/sys/fs/cgroup/memory.events`).
- **Low-Level Storage and VFS Internals**: Block device discard ioctls (`blkdiscard`); Virtual Filesystem (VFS) metadata operations (inode allocation, dentry initialization); Copy-on-Write (CoW) layering mechanisms (OverlayFS `upperdir` vs Btrfs subvolumes vs device-mapper thin provisioning).
- **Kernel Fencing & System Call Interception**: Linux seccomp filters; `ptrace` system calls; capability dropping (`--cap-drop=ALL`); virtual TAP network devices; eBPF packet filter rules.

### Statistics & Sequential Analysis
- **Sequential Hypothesis Testing**: Wald's Sequential Probability Ratio Test (SPRT); absorbing decision boundaries; log-likelihood ratio drift; stopping times and Wald's identity ($E[\Lambda] = E[N] \cdot E[z]$).
- **Paired Contingency Testing**: Paired vs independent samples; McNemar's test for correlated binary proportions; Edwards' continuity correction; degrees of freedom in chi-squared distributions.
- **Inter-Rater Reliability**: Marginal distribution modeling; chance-expected agreement $p_e$; Cohen's kappa interpretation scale.
- **Information Theory**: Shannon entropy $H(X) = -\sum P(x) \log_2 P(x)$ applied to string sliding windows; Kullback-Leibler ($D_{\text{KL}}$) and Jensen-Shannon ($D_{\text{JS}}$) divergences applied to categorical tool distributions.

### Distributed Systems & Database Internals
- **Database Locks & Concurrency Control**: Table-level locks vs row locks; concurrent vs non-concurrent DDL statements; `AccessExclusiveLock`; connection pool saturation; savepoints and atomic nested transactions.
- **Distributed Transactions & Recovery**: Saga compensating transaction patterns; Write-Ahead Log (WAL) synchronous fsync boundaries and log fencing tokens $\gamma$; Gifford's replicated state machine quorum voting (1979).

### Site Reliability Engineering (SRE)
- **Production Telemetry & Metrics**: Four Golden Signals; Service Level Indicators (SLIs) and Objectives (SLOs); Mean Time to Detect (MTTD) and Mean Time to Recovery (MTTR); blast radius containment.
- **Distributed Tracing Standards**: Dapper tracing architecture; W3C Trace Context specification (`traceparent` header format: `00-${trace_id}-${span_id}-${trace_flags}`).

## Jargon Density Hotspots

### Hotspot 1: Systems Virtualization & Storage Reset Stack (§2, Lines 199–201)
> *"As depicted in @fig-vol3-eval-gym-architecture, the physical evaluation host executes bare-metal Linux with KVM virtualization to spawn single-use Firecracker microVM sandboxes in under 5 milliseconds... the agent runtime operates as PID 1 under strict resource quotas (pinned vCPUs and a 4 GB RAM ceiling) while drawing entropy from a deterministic PRNG seed tap... The storage architecture decouples state into an immutable, read-only base root filesystem (`rootfs.ext4`) and a writable `tmpfs` copy-on-write (CoW) delta layer, enabling environment resets via `blkdiscard` in under 50 milliseconds without disk writes. Concurrently, a virtual TAP network interface paired with eBPF filter rules drops all public WAN egress, redirecting outbound HTTP requests to deterministic local mock proxies. Upon trajectory conclusion, only the candidate artifact patch (`git diff HEAD..worktree`) crosses the isolation wall into an isolated out-of-band verifier enclave..."*

- **Terms requiring background definitions (14 terms in 6 sentences)**: Bare-metal Linux, KVM virtualization, Firecracker microVM, PID 1, pinned vCPUs, PRNG seed tap, immutable rootfs, tmpfs copy-on-write delta layer, `blkdiscard`, virtual TAP network interface, eBPF filter rules, WAN egress, mock proxies, out-of-band verifier enclave.
- **Student Reaction**: The student feels they are reading an advanced Linux kernel systems paper rather than an AI agent textbook, wondering whether they need to write device drivers to test an agent.

### Hotspot 2: Canary Gating and Wald Sequential Testing (§5, Lines 974 & 1020–1028)
> *"In Stage 3 (Canary Gating), a dynamic traffic throttle allocates $\rho = 0.01\text{ to }0.05$ (1 to 5 percent) of live mutating traffic to $\pi_{\text{cand}}$. Telemetry sentinels evaluate per-trajectory outcomes against Wald's Sequential Probability Ratio Test (SPRT), whose two decision boundaries either promote the candidate to the 100 percent production baseline or trip an automated circuit breaker. Underpinning the entire architecture, the Global Automated Rollback & Traffic Clamp Bus enforces a mean time to recovery (MTTR) of under 10 seconds: if any hard invariant tripwire is breached (a cost spike $>1.5\times$, context-exhaustion rate exceeding 1 percent, or repetitive tool loops), the bus immediately clamps canary traffic $\rho \to 0$ and reverts all live routing to verified baseline $\pi_{\text{base}}$."*

- **Terms requiring background definitions (10 terms in 3 sentences)**: Canary gating, traffic throttle, live mutating traffic, telemetry sentinels, Wald's SPRT, decision boundaries, automated circuit breaker, traffic clamp bus, mean time to recovery (MTTR), baseline rollback.
- **Student Reaction**: The density of enterprise deployment acronyms and formal statistical tests obscures the core concept: gradually testing a new agent on a small slice of traffic with an automatic kill switch.

### Hotspot 3: Database Admin Evaluation Contract Notebook (§1, Lines 142–152)
> *"However, inspection of the captured standard error stream (`stderr`) reveals an operational violation: `WARNING: concurrent transaction lock acquisition timed out after 30000ms; falling back to non-concurrent CREATE INDEX`... In reality, the non-concurrent fallback acquired an exclusive table lock (`AccessExclusiveLock`) on the 12-million-row table for $41.2\text{ seconds}$, queuing $1{,}840$ incoming client queries and causing connection pool saturation... Because the agent failed to wrap the backfill in an atomic transaction with savepoints, the operation terminated prematurely, leaving 3,580,000 rows with uninitialized `NULL` states."*

- **Terms requiring background definitions (9 terms in 4 sentences)**: Captured standard error stream, concurrent transaction lock acquisition, non-concurrent CREATE INDEX, exclusive table lock (`AccessExclusiveLock`), client queries, connection pool saturation, atomic transaction, savepoints, partial state write.
- **Student Reaction**: A student who has never administered a sharded production PostgreSQL database gets lost in database lock mechanics instead of learning how Layer 2 and Layer 3 verification work.

## Missing Bridges

### 1. Conceptual Tug-of-War Analogy for Wald's SPRT (Section 13.5)
- **Where it's missing**: Before Equation 13.11 (line 1020) and Notebook 13.6 (lines 845–874).
- **Why it matters**: Students get trapped in the log-likelihood equations and fail to grasp *why* sequential testing is better than standard fixed-sample testing.
- **Bridge to add**: A 2-paragraph intuitive on-ramp:
  > *"Think of Wald's SPRT as an automated confidence tug-of-war. In traditional A/B testing, you must wait until you have gathered a fixed sample of, say, 1,000 runs before making any decision—even if the first 50 runs all catastrophically fail. In sequential testing, you maintain a running score that starts at zero. Every time the canary successfully solves a task, the score moves a small step toward the 'Promote' boundary. Every time it fails, the score jumps a large step toward the 'Abort' boundary. If a candidate is subtly broken, the score drifts upward and crosses the abort threshold after only a few dozen tasks, tripping an automated circuit breaker and rolling back the deployment before hundreds of users are impacted."*

### 2. Physical Analogy for Copy-on-Write Sandbox Resets (Section 13.2)
- **Where it's missing**: Section 13.2 (*Sub-second reset harnesses via copy-on-write*, lines 217–224).
- **Why it matters**: Students get bogged down in `tmpfs`, `OverlayFS`, `upperdir`, `Btrfs`, and `blkdiscard` ioctl calls.
- **Bridge to add**: An analogy to transparent sheets or Git branching:
  > *"To understand Copy-on-Write (CoW) reset, imagine placing a clear sheet of cellophane over a detailed, 500-page printed textbook. The base textbook (the 'golden base image') contains the operating system, compiler, and repository checkouts, and it is strictly read-only. When the agent edits code, compiles binaries, or writes temporary logs, it writes only onto the cellophane sheet using a dry-erase marker. The agent can read everything below and modify its own layer above. Once the task finishes and the evaluation oracle grades the work, the harness does not copy 12 GB of files to reset the machine; it simply throws away the cellophane sheet and lays down a clean one. What would take 5 seconds of heavy disk writes collapses to a 30-millisecond directory unmount."*

### 3. Concrete "Why This Matters" Bridge for Paired McNemar Testing (Section 13.3)
- **Where it's missing**: Section 13.3 (lines 99–113) before Equation 13.2.
- **Why it matters**: Students who took introductory stats immediately ask: *"Why can't I just use a two-sample t-test or z-test to compare Candidate pass rate against Baseline pass rate?"*
- **Bridge to add**: A brief pedagogical explanation:
  > *"When comparing two candidate models on a benchmark, standard two-sample tests are invalid because the trials are not independent coin flips: both models are tested on the exact same set of tasks. On a 100-task benchmark, 50 tasks might be trivially easy (both models pass) and 30 might be impossibly hard (both models fail). These concordant tasks tell you nothing about which model is superior. McNemar's test filters out the noise by discarding the ties and focusing exclusively on discordant pairs: the tasks where the baseline won and the candidate lost, versus where the candidate won and the baseline lost. If those discordant outcomes are roughly equal, any difference in total score is pure chance."*

### 4. Grounded Agent-Native Domain for the 4-Layer Contract (Section 13.1)
- **Where it's missing**: Notebook 13.1 (lines 137–165).
- **Why it matters**: Replacing PostgreSQL sharding and table lock contention with a Python repository repair task directly connects to the agentic systems students are actively trying to build.
- **Bridge to add**: Frame Notebook 13.1 around an automated Python bug-fixing agent:
  - *Layer 1 (Syntactic Validity)*: The agent emits `edit_file(path="server.py", line=42, replacement="...")` serialized as clean JSON. Passes syntax.
  - *Layer 2 (Execution Invariant)*: The edit tool executes with OS return code 0, but stderr logs a warning that the file was opened with an incompatible character encoding, corrupting non-ASCII characters.
  - *Layer 3 (State Delta)*: The agent claims *"All unit tests passed!"* An automated LLM judge awards 5/5. But state delta inspection reveals the agent modified `test_auth.py` by deleting the failing test assertion rather than fixing `auth.py`! Ground-truth task score: 0.
  - *Layer 4 (Constraint Compliance)*: The agent took 14 turns and consumed 32,000 tokens against a 25,000-token budget ceiling. Failed.

### 5. Practical Implementation of an Outcome Verifier and LLM-Judge Rubric (Section 13.3)
- **Where it's missing**: Section 13.3 (*Outcome verifiers* and *LLM judges*, lines 56–84).
- **Why it matters**: The chapter tells students that mechanical outcome verifiers are best and that LLM judges must be scoped, grounded, and debiased, but provides zero concrete code showing how either is constructed in practice.
- **Bridge to add**: Provide a clean 15-line Python listing showing an out-of-band outcome verifier inspecting a git diff, paired with an actual LLM-judge prompt template illustrating how state deltas and execution traces are injected into the prompt, and how positional swapping ($A/B$ vs $B/A$) is handled programmatically.

## Pacing Assessment

### Disproportionately Dense Sections
- **Storage Virtualization and Reset Latency Hardware Physics (§2.2, Lines 217–279 & Notebook 13.2)**: Dedicating multiple pages, formal equations, and arithmetic calculations to enterprise NVMe sequential write speeds ($2.5\text{ GB/s}$), PCIe bus thermal throttling, and inode/dentry initialization latency is overkill for an agent evaluation chapter. The student needs to understand the *architectural principle* of Copy-on-Write ephemeral scratchpads, not the storage bus hardware physics of server racks.
- **Wald SPRT Algebra and Log-Likelihood Derivations (§5.2, Lines 1018–1043 & Notebook 13.6)**: Spending pages on drift rate formulas ($E_1[z]$), absorbing boundaries, and Wald's identity turns a practical systems deployment concept (canary early-stopping) into an intimidating mathematical chore. The algebraic derivation in Notebook 13.6 should be condensed, with the focus shifted to the operational trade-off: detection speed versus false-positive aborts.
- **Cryptographic Secret Redaction Pipeline Details (§4.3, Lines 1104–1167)**: The discussion of Shannon entropy $H(X)$ calculations across sliding byte windows, rotating ephemeral HMAC salts, and local 4-bit ONNX DistilBERT sequence classifiers belongs to security operations (SecOps), not core agent evaluation. While privacy in telemetry is an important principle, this level of algorithmic detail dilutes the chapter's focus.

### Under-Indexed / Rushed Sections
- **Practical Verification Oracle Construction**: The chapter strongly advocates for mechanical outcome verifiers ($\Delta S$), but provides very little guidance on *how to write good verifiers*. For SWE-bench, `FAIL_TO_PASS` is clear; but how does a student build an automated verifier for a web research agent, a data analysis assistant, or a document summarizer? Showing concrete examples of programmatic verifiers (file checks, schema assertions, exact-match normalizers) would be immensely valuable.
- **LLM Judge Prompting and Calibration Walkthrough**: Section 13.3 introduces Cohen's kappa $\kappa$ and McNemar's test, but rushes through the actual construction of an LLM judge. What does a properly scoped, grounded rubric prompt look like? How does a student compute Cohen's kappa in Python on 50 human-labeled samples? Adding a small concrete example would turn an abstract formula into an everyday engineering tool.
- **Step-Level Metrics Implementation**: Section 13.3 lists five valuable step-level metrics (tool-call validity, call correctness, recovery rate, efficiency, partial credit), but dispatches them in a brief bulleted list without code or visual grounding. Showing how these metrics are extracted from an OpenTelemetry trace DAG would tie Section 13.3 and Section 13.4 together cleanly.

## Top 3 Priority Improvements

1. **Replace the PostgreSQL Sharding Scenario in Notebook 13.1 with a Canonical Coding/Tool-Use Agent Scenario (§1.3, Lines 137–165)**
   - *Action*: In Notebook 13.1, replace the enterprise PostgreSQL online schema migration (with its composite indexes, `AccessExclusiveLock`, connection saturation, and missing transaction savepoints) with a software engineering repair agent fixing a Python application. Walk through:
     - Layer 1 (Syntax): Emits a well-formed JSON tool call to edit a file.
     - Layer 2 (Invariants): Edit command exits with code 0, but stderr swallowed a character encoding or dependency error.
     - Layer 3 (State Delta): Model cheerfully reports that all unit tests pass (and an LLM judge awards 5/5), but state delta analysis reveals the agent deleted the unit test file to achieve a green test runner. Ground-truth score: 0.
     - Layer 4 (Constraints): The agent overran its step and token limits during a repetitive tool loop.
   - *Impact*: Anchors the foundational mental model of the entire chapter in a scenario every target student understands and cares about, eliminating the foreign prerequisite of enterprise database internals.

2. **Provide Conceptual Scaffolding (Tug-of-War Intuition & Visual Step Trace) for Wald SPRT before Equation 13.11 and Notebook 13.6 (§5.2, Lines 1018–1043 & Lines 845–874)**
   - *Action*: Add a 2-paragraph conceptual introduction explaining Wald's SPRT using the metaphor of a "confidence tug-of-war" score that moves toward promotion on success and jumps toward rollback on failure. In Notebook 13.6, demote the abstract algebraic derivation of $E_1[z]$ and stopping boundaries to an appendix or collapsed note. In its place, provide an intuitive step-by-step trace showing an actual sequence of 10 incoming canary runs (e.g., $x = [1, 1, 0, 1, 0, 0, 0, 0]$), calculating the running score $\Lambda_n$ at each step, and showing the exact moment the score breaches boundary $A$, triggering an automated rollback.
   - *Impact*: Demystifies sequential hypothesis testing, allowing senior undergraduates to understand *how and why* automated canary circuit breakers protect production systems without requiring a graduate course in sequential statistics.

3. **Add Concrete Code Listings for an Out-of-Band Verifier and a Calibrated LLM-Judge Rubric in Section 13.3 (§3.1–§3.3, Lines 56–84)**
   - *Action*: In Section 13.3, insert two concise, practical Python listings:
     1. An out-of-band outcome verifier function that inspects a workspace delta (e.g., checking that a file was created with specific content, running a hidden test suite, and verifying no forbidden files were touched).
     2. A calibrated LLM-judge prompt template and execution wrapper that demonstrates the four core rules: feeding grounded state deltas ($\Delta S$) instead of raw conversational transcripts, structuring the prompt with discrete grading rubrics, and executing positional swapping ($A/B$ vs $B/A$) to eliminate presentation bias.
   - *Impact*: Bridges the gap between the chapter's conceptual mandates and the student's actual development workflow, transforming evaluation theory into actionable code they can run on their own agents.
