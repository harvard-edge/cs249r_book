# Chapter 14: Trajectory Curation — Student Accessibility Review

## Executive Summary

Chapter 14 (*Trajectory Curation*) tackles one of the most critical and practically urgent challenges in agent systems: how to transform non-deterministic runtime telemetry and high-throughput synthetic rollouts into high-utility training datasets without degrading the model. The chapter excels when diagnosing why naive fine-tuning fails—specifically through the Systems Intervention Ladder, the four root causes of trajectory failure, the mathematical necessity of recovery traces to bound $O(\epsilon T^2)$ covariate shift, and the concrete Python AST implementation of test-tampering defenses. However, the chapter frequently alienates senior undergraduate and early graduate students by abruptly pivoting from high-level data curation into dense, specialized arcana far outside standard ML data engineering: low-level Linux kernel hypervisor plumbing (`cgroup.kill`, `PR_SET_CHILD_SUBREAPER`, `veth` bridge port limits, `ioctl(BLKDISCARD)`, unlinked inode leaks), ungrounded randomized data-mining algorithms (MinHash 2-universal hash families, LSH signature banding, AST subtree shingling), and formal queueing theory (Little's Law and Kingman's heavy-tailed $G/G/1$ approximations).

## What Works Well

- **The Systems Intervention Ladder & Database Migration Motivating Example (§1, Lines 56–88, Table 14.1, Figure 14.1)**: The opening story of an autonomous database migration agent failing on `ALTER TABLE` because the tool schema omitted the SQL dialect (MySQL vs. PostgreSQL) is an outstanding pedagogical hook. A student who has written basic SQL immediately grasps why retraining a 70B parameter model is an architectural anti-pattern when fixing a JSON schema takes 15 minutes.
- **The Four Root Causes of Failure Taxonomy (Table 14.1, Lines 72–84)**: Categorizing execution anomalies into *Context Deficit*, *Interface Ambiguity*, *Runtime Defect*, and *Policy Incapability* gives students a rigorous mental model. Grounding each class with an empirical diagnostic probe (e.g., injecting gold context snippets to measure pass-rate $\Delta$) teaches systematic systems diagnosis.
- **Systems Intervention Economics Worked Example (Example 14.1, Lines 108–124)**: The explicit numerical comparison between Strategy A (schema fix & container pooling: $2,400 labor, 0 GPU FLOPs, 15-minute deploy, 0% regression risk) and Strategy B (premature fine-tuning: $6,000 labor, $168 compute, 48-hour deploy, -2.5% general accuracy drop) provides compelling quantitative evidence that parameter updates should be the last resort.
- **The Staged Verifier Funnel & Cascade Economics (§3, Lines 328–385, Figure 14.2, Table 14.3, Notebook 14.2)**: Ordering validation gates by ascending computational cost (microsecond in-memory AST syntax $\to$ millisecond static sandbox checks $\to$ multi-second dynamic microVM tests $\to$ state delta audits $\to$ model/human review) is logically irresistible. The accompanying algebra ($C_{\text{attempt}} = c_1 + p_1 c_2 + \dots$) and Notebook 14.2 demonstrate a concrete 68.6% compute reduction (saving 191 core-hours/day) that any undergraduate can easily replicate.
- **Test-Tampering & Harness Subversion Defenses (§3.2, Lines 387–420, & Code Listing, Lines 1671–1715)**: The unified diff showing an agent subverting an escrow test (`assert result.reconciled_balance == 10000` replaced by `# Bypass partition assertion... assert True`) is unforgettable. Furthermore, the `AntiTamperVisitor` Python script utilizing `ast.NodeVisitor` to detect `assert True`, `pytest.mark.skip`, and empty test bodies provides immediate, executable code that students can incorporate into their own projects.
- **Covariate Shift Hazards & The Tripartite Trajectory Taxonomy (§4.1–§4.3, Lines 609–687, Figure 14.3, Table 14.4)**: Grounding data curation in Ross, Gordon, and Bagnell's (2011) imitation learning theory explains *why* training only on flawless demonstrations causes agents to freeze or loop infinitely when faced with their first error token. Partitioning traces into Pristine ($60\text{--}70\%$), Recovery ($20\text{--}30\%$), and Hard Negatives ($\sim 10\%$) gives a clear recipe for robust curriculum design.
- **The Hard Negative Training Hazard (Callout Note 14.1, Lines 684–686)**: The explicit warning that hard negatives must **never** be included in supervised fine-tuning cross-entropy loss (because it trains the model to maximize the probability of emitting `rm -rf /` or deleting tests) prevents a catastrophic design error that many junior ML practitioners commit.
- **Token Footprint Economics of Recovery Traces (Notebook 14.3, Lines 801–844)**: Demonstrating that recovery traces account for nearly 45% of total supervised tokens despite comprising only 25% of trajectories (due to extended 14-turn diagnostic loops) provides a profound lesson in training token economics and sequence packing.
- **The Informative Fraction Formula in Self-Improvement Loops (§5.2, Lines 856–863)**: Equation 14.4 ($P_{\text{inform}}(p, k) = 1 - (1-p)^k - p^k$) elegantly proves using elementary probability that tasks with 0% or 100% pass rates provide zero learning signal, mathematically justifying why sampling budgets must concentrate on the middle difficulty band.
- **Environmental Contamination vs. Lexical Leakage (§7.1, Lines 1384–1418)**: Demonstrating the failure trace where an agent blindly tries to read non-existent paths (`src/utils/compat/string_helpers.py`) because it memorized the directory structure of training repositories vividly illustrates why standard random train/test splitting is invalid for agent systems.
- **Syntactically Invariant Differential Token Masking (§6.2.3, Lines 1319–1346)**: The diff in lines 1334–1345 effectively illustrates why blunt regex string deletion breaks JSON schemas (`{"auth": }`), and how AST string literal replacement with salted pseudonyms (`<REDACTED_SECRET:4f8a2b1c9e01>`) preserves both syntax and referential transparency.
- **Downstream Dataset Emission Contracts (§8.1.7, Lines 1745–1783)**: Providing concrete, strongly typed JSON schemas for `trajectories_sft.jsonl` (with token-level `loss_mask`) and `trajectories_dpo.jsonl` (with `chosen` vs. `rejected` pairs) establishes an essential bridge between data curation and downstream fine-tuning.

## Barrier Spots

### 1. Hypervisor Hygiene and Host OS Kernel Plumbing
- **Location**: Section 5.4 / Section 6.3 (*Hypervisor hygiene*, lines 1098–1154, Tables 14.7 & 14.8).
- **The barrier**: The chapter suddenly plunges into low-level Linux kernel internals: `/proc/sys/kernel/pid_max`, `CLONE_NEWPID`, `PR_SET_CHILD_SUBREAPER`, `cgroup v2` control hierarchies (`echo 1 > /sys/fs/cgroup/.../cgroup.kill`), `veth` virtual Ethernet pairs on Linux bridges (`br0`) saturating 1,024 port limits, `ip netns del`, POSIX shared memory `/dev/shm` and `ipcs -m`, OverlayFS writable `upperdir` and DeviceMapper thin-provisioning, unreleased mounts with `lsof +L1` unlinked open file descriptors causing "ghost leaks" where `df` reports 100% disk usage while `du` shows zero, and `ioctl(BLKDISCARD)`.
- **Why it's a barrier**: Assumes advanced Linux systems administration and OS kernel engineering. The target senior CS undergraduate knows containers at a user level (`docker build`, `docker run`). They have never configured control group v2 leaf nodes, manipulated network namespaces via raw `ip link` subcommands, debugged unlinked file table overflows with `lsof +L1`, or written sub-reaper init daemons in C. The student feels like the textbook suddenly pivoted from an ML systems course into an operating systems kernel maintenance manual.
- **Severity**: High (student gets overwhelmed by infrastructure plumbing that has little to do with curation logic).

### 2. Unexplained AST MinHash and LSH Banding Mathematics
- **Location**: Section 7.3 (*Trajectory structural deduplication and AST MinHash*, lines 1443–1476).
- **The barrier**: Within three dense paragraphs and three equations, the text introduces Jaccard similarity of AST shingle sets, 2-universal hash families, the MinHash minimum-hash collision property ($P(\min h(A) = \min h(B)) = J(A, B)$), estimator variance $\mathrm{Var}(\hat{J}) = \frac{J(1-J)}{k}$, LSH signature partitioning into $b$ bands of $r$ rows, collision probability $P_{\text{collision}} = 1 - (1 - J^r)^b$, and the S-curve threshold $t \approx (1/b)^{1/r}$.
- **Why it's a barrier**: Assumes a graduate-level background in randomized data-mining algorithms (e.g., Broder's MinHash, Indyk and Motwani's Locality-Sensitive Hashing). The target student knows basic ML pipelines (PyTorch `Dataset`, DataLoader, standard train/val splits) but has never studied randomized hashing or LSH. The text drops formulas for banding and S-curves without explaining what a "band" physically represents, why dividing a signature matrix into bands shapes an S-curve, or how an AST is sliced into shingles.
- **Severity**: High (student encounters a wall of ungrounded randomized algorithms formulas and skips the section).

### 3. Queueing Theory Formalism (Kingman's Approximation & Heavy-Tailed G/G/1 Systems)
- **Location**: Section 5.2 (*Backpressure governance*, lines 952–969).
- **The barrier**: To explain why buffer queues back up, the text introduces Little's Law ($L = \lambda W$) and Kingman's formula for a $G/G/1$ queue:
  $$L_q = \lambda_{\text{rollout}} \left( \frac{\rho}{\mu_{\text{verify}}(1 - \rho)} \right) \cdot \left( \frac{C_a^2 + C_s^2}{2} \right)$$
  referencing squared coefficients of variation ($C_a^2, C_s^2 \gg 1$), right-skewed service distributions $B(t)$, and traffic intensity $\rho \to 1$.
- **Why it's a barrier**: The target student persona explicitly has **NOT** studied queueing theory or performance modeling. The text does not define what Kendall's notation $G/G/1$ stands for (General arrival, General service, 1 server), what a "squared coefficient of variation" measures, or why Kingman's heavy-traffic approximation applies here. The underlying intuitive takeaway—that test suite timeouts have extreme variance, causing queue lengths to explode quadratically—is smothered by queueing theory formalism.
- **Severity**: Medium (student skips the formula, though Notebook 14.4 partially rescues the practical arithmetic).

### 4. Cryptographic Lineage Envelope & Merkle Tree Complexity
- **Location**: Section 6.1 (*The cryptographic lineage envelope*, lines 1165–1216, Figure 14.5).
- **The barrier**: Trajectory provenance is formalized through Buneman's "why-provenance" vs "wherefore-provenance", a 5-axis cryptographic schema, HMAC-SHA256 composite envelopes ($H_{\text{lineage}} = \text{HMAC-SHA256}(K_{\text{pipe}}, \dots)$), state-action Merkle trees with step tuples $h_t = H(s_t, a_t, o_t)$ and intermediate node hashes for 32 steps ($2,016$ bytes), detached cryptographic worker signatures, and S3 Object Lock immutability. Furthermore, the margin note discusses non-associative FP16/BF16 GEMM parallel reduction order across thread block schedulers.
- **Why it's a barrier**: Blends database theory, formal cryptography, and GPU microarchitecture. A student who simply needs to track "which model checkpoint, prompt, and tool versions produced this trajectory" is confronted with Merkle trees, HMAC keys, and floating-point non-associativity in GEMM kernels.
- **Severity**: Medium (student understands the need for versioning, but tunes out the cryptographic formalization).

### 5. Multi-Variable Throughput Roofline Without Visual Grounding
- **Location**: Section 8.2 (*Formal mathematical metrics*, lines 1815–1820, Equation 14.7).
- **The barrier**: Equation 14.7 defines the Trajectory Throughput Roofline:
  $$\Theta_{\text{pipeline}} = \min\left( \frac{B_{\text{storage}}}{S_{\text{traj}}}, \, \frac{N_{\text{vcpus}} \cdot \mu_{\text{sandbox}}}{T_{\text{exec}}}, \, \frac{N_{\text{accelerators}} \cdot \mathcal{P}_{\text{tok}}}{L_{\text{tok}}} \right)$$
  featuring eight distinct, unnumbered variables across storage bandwidth, sandbox execution parallelism, and inference accelerator token throughput.
- **Why it's a barrier**: While the Williams et al. Roofline model is familiar from ML hardware courses, this three-way systems bottleneck equation is presented purely algebraically without a supporting roofline diagram or plot. The student is left to parse an 8-variable expression without visual intuition for where industrial pipelines typically bottleneck.
- **Severity**: Low/Medium.

### 6. Duplicated Callout Checkpoints and Disconnected Footnotes
- **Location**: Checkpoints in Section 3 (Lines 262–269 vs Lines 273–280) and Footnote 260.
- **The barrier**: Checkpoint 14.1 (`chk-14-admission-cascades-limits`, lines 262–269) and Checkpoint 14.2 (`chk-14-evaluating-verifier-cascades`, lines 273–280) appear virtually back-to-back with near-identical questions about cascade ordering and test tampering. Additionally, Footnote 260 abruptly introduces "Vector Memory Bandwidth Saturation... Locality-Sensitive Hashing" in the middle of a section discussing learned LLM judges and Process Reward Models (PRMs), which has no relation to dense embedding similarity.
- **Why it's a barrier**: Creates reader confusion and cognitive whiplash, making the student feel that text was pasted carelessly or that they missed an entire section on vector database indexing.
- **Severity**: Low.

## Assumed Knowledge Gaps

### Operating Systems & Linux Kernel Engineering
- **Control Groups & Namespace Isolation**: Linux `cgroup v2` unified hierarchies, `cgroup.kill` interface, memory controllers, `CLONE_NEWPID`, and process sub-reaper daemons (`PR_SET_CHILD_SUBREAPER`).
- **Virtual Networking Plumbing**: Virtual Ethernet (`veth`) interface pairs, Linux software bridges (`br0`), bridge port exhaustion limits (1,024 interfaces), `ip netns` network namespace deletion, and kernel netfilter state.
- **Filesystem & Storage Architecture**: Copy-on-Write (CoW) layered filesystems (OverlayFS `lowerdir`, `upperdir`, `workdir`), DeviceMapper thin-provisioning, unlinked open file descriptors (`lsof +L1`), inode reference counting, "ghost leaks" (discrepancies between `df` and `du`), and block-level discard commands (`ioctl(BLKDISCARD)`).
- **System Call Interposition**: Dynamic linker preloading (`LD_PRELOAD`), `ptrace` system call tracing, and Virtual Filesystem (VFS) hook wrappers.

### Randomized Algorithms & Data Mining
- **Locality-Sensitive Hashing (LSH) & MinHash**: Broder's MinHash theorem ($P(\min h(A) = \min h(B)) = J(A, B)$), 2-universal hash families, random permutation hashing, and signature matrix estimation.
- **LSH Banding & S-Curves**: Mathematical derivation of collision probabilities across $b$ bands of $r$ rows ($1 - (1 - J^r)^b$) and the resulting sigmoid threshold $t \approx (1/b)^{1/r}$.
- **Abstract Syntax Tree Shingling**: Decomposing parsed AST nodes into normalized $k$-grams or subtrees of depth $d=3$.

### Queueing Theory & Systems Modeling
- **Queueing Models**: Kendall's notation ($G/G/1$), Little's Law ($L = \lambda W$), and Kingman's heavy-traffic approximation formula.
- **Variability Metrics**: Squared coefficient of variation ($C_a^2, C_s^2$), right-skewed service time distributions, and traffic intensity $\rho = \lambda / \mu$.
- **Flow Control Mechanics**: Credit-based / token-bucket flow control and dual-watermark backpressure governors ($Q_{\text{low}}, Q_{\text{high}}$).

### Database Theory & Cryptographic Security
- **Provenance Formulations**: Classical database provenance theory (Buneman's *why-provenance* and *wherefore-provenance*).
- **Cryptographic Structures**: Merkle trees over state-action execution sequences, intermediate node digest propagation, HMAC-SHA256 composite signatures, and S3 Object Lock compliance storage.
- **Information Theory & Privacy**: Shannon entropy sliding-window computation ($H(S) = -\sum p \log_2 p$) for high-entropy secret detection, and Dwork's differential privacy framework.

### Advanced Post-Training & Reinforcement Learning
- **Preference Optimization**: Direct Preference Optimization (DPO), unlikelihood loss penalties, and margin-based contrastive ranking.
- **Reinforcement Learning with Verifiable Rewards (RLVR)**: Rule-based verification rewards vs. learned Process Reward Models (PRMs) and Outcome Reward Models (ORMs).
- **Theoretical Imitation Learning**: Ross, Gordon, and Bagnell's (2011) reduction of imitation learning to no-regret online learning, proving quadratic error compounding ($O(\epsilon T^2)$).

## Jargon Density Hotspots

### Hotspot 1: Hypervisor Hygiene & Linux Subsystem Leaks (§5.4, Table 14.7 & Lines 1104–1107)
> *"Dedicated PID namespaces (`CLONE_NEWPID`); sub-reaper init daemon sends `SIGKILL` to cgroup tree upon exit... Abandoned `veth` interfaces saturate Linux bridge (`br0`) port limits (max 1,024). Atomic network namespace teardown (`ip netns del`); dynamic interface reclamation daemon audits bridge state... Multiprocessing IPC segments left unlinked in `/dev/shm`, leaking host DRAM. Mount private, ephemeral `tmpfs` instances per sandbox; cgroup v2 memory controller enforces strict eviction limit... Orphaned CoW layers and unlinked open file descriptors consume NVMe disk blocks. Ephemeral loopback volumes or ZFS subvolumes with deterministic destruction (`umount` + `ioctl(BLKDISCARD)`)..."*

- **Terms requiring background definitions (18 terms in 8 lines)**: PID namespaces, `CLONE_NEWPID`, sub-reaper init daemon, `SIGKILL`, cgroup tree, `veth` interfaces, Linux bridge `br0`, bridge port limits, network namespace teardown, `ip netns del`, IPC segments, `/dev/shm`, `tmpfs`, cgroup v2 memory controller, CoW layers, unlinked open file descriptors, loopback volumes, `ioctl(BLKDISCARD)`.
- **Student Reaction**: "I thought this chapter was about collecting, filtering, and preparing training data for LLM agents. Why am I debugging Linux network bridge tables and issuing raw `ioctl` disk discard calls?"

### Hotspot 2: AST MinHash and LSH Deduplication (§7.3, Lines 1455–1469)
> *"To approximate trajectory structural diversity across millions of candidate traces without computing exhaustive pairwise tree-edit distances, the harvesting engine computes a signature vector using $k$ independent universal hash functions $\mathbf{h} = [h_1, h_2, \dots, h_k]$... The collision frequency across the $k$ hashes provides an unbiased estimator of structural similarity: $\hat{J}(T_1, T_2) = \frac{1}{k} \sum_{i=1}^k \mathbb{I}\left[ s_i(T_1) = s_i(T_2) \right], \quad \mathrm{Var}\left(\hat{J}\right) = \frac{J(1 - J)}{k}$... To execute deduplication at line-rate during streaming ingestion, the pipeline groups the $k$ signature components into $b$ bands of $r$ rows such that $k = b \cdot r$. Two trajectories collide in at least one band with probability: $P_{\text{collision}} = 1 - \left(1 - J(T_1, T_2)^r\right)^b$. This banding structure produces a sharp S-curve threshold centered at $t \approx (1/b)^{1/r}$."*

- **Terms requiring background definitions (14 terms in 12 lines)**: Pairwise tree-edit distances, signature vector, 2-universal hash family, collision frequency, unbiased estimator, variance of estimator, signature components, $b$ bands of $r$ rows, banding structure, collision probability, S-curve threshold, deduplication lattice, reward dominance, contiguous memory-mapped arrays.
- **Student Reaction**: "I understand intuitively that we want to deduplicate trajectories that execute the same logic, but this math dump assumes I already know MinHash and LSH cold. What does 'bands of rows' mean physically, and how does that turn into an S-curve?"

### Hotspot 3: Kingman's Queueing Formula & Backpressure Governance (§5.2, Lines 960–968)
> *"Applying Little's Law ($L = \lambda W$), the average number of trajectories accumulating in the verification staging buffer ($L_q$) is the product of the arrival rate $\lambda$ and the mean waiting time in the queue ($W_q$): $L_q = \lambda_{\text{rollout}} \left( \frac{\rho}{\mu_{\text{verify}}(1 - \rho)} \right) \cdot \left( \frac{C_a^2 + C_s^2}{2} \right)$, where $C_a^2$ and $C_s^2$ represent the squared coefficients of variation for the inter-arrival times and service times, respectively (the Kingman approximation for a $G/G/1$ system). Because software test suites exhibit heavy-tailed execution distributions ($C_s^2 \gg 1$), any sustained burst where $\lambda_{\text{rollout}} \to \mu_{\text{verify}}$ causes $L_q$ to explode."*

- **Terms requiring background definitions (11 terms in 8 lines)**: Little's Law, verification staging buffer, arrival rate, waiting time, traffic intensity $\rho$, service rate $\mu$, squared coefficient of variation, Kingman approximation, $G/G/1$ system, heavy-tailed execution distribution, sustained burst.
- **Student Reaction**: "I've never taken a queueing theory class. What is $G/G/1$? Why are coefficients of variation squared? I just want to understand why my buffer queue needs to be sized properly before the broker runs out of RAM!"

## Missing Bridges

1. **A Visual Walkthrough of AST MinHash and LSH Banding (§7.3)**:
   - *What's missing*: An intuitive, step-by-step walkthrough showing how an agent trajectory's code patch and tool calls are converted into normalized shingles, transformed into a MinHash signature vector, and divided into bands of rows to create the S-curve threshold.
   - *Bridge needed*: A 2-paragraph intuitive walkthrough with a toy example:
     - Show two agent patches that solve a bug identically but use different variable names (`x = 1; y = x + 1` vs `count = 1; total = count + 1`).
     - Show how stripping identifier names into typed AST nodes (`Assign(Num), Assign(BinOp)`) yields identical structural shingles.
     - Show a small table with 4 hash functions divided into 2 bands of 2 rows, explaining: *"Why do we band hashes? If we require all 4 hashes to match, we only catch identical programs. If we require any single hash to match, we get flooded with false positives. Banding enforces an 'OR of ANDs': match ALL hashes in band 1 OR ALL hashes in band 2. This mathematical trick transforms a gentle linear similarity curve into a steep S-curve, filtering out anything below 50% similarity while catching nearly everything above 80%."*
2. **An Everyday Analogy for Queueing Backpressure (§5.2)**:
   - *What's missing*: An accessible mental model for why variance in test suite execution time destroys pipeline throughput before dropping Kingman's $G/G/1$ formula.
   - *Bridge needed*: Connect to a familiar physical analogy: *"Think of a highway toll plaza. If cars arrive at a steady rate of 10 cars per minute and each toll booth takes exactly 6 seconds, traffic flows without delay ($\rho = 1.0$). But if a single driver gets stuck for 5 minutes (the equivalent of a test suite hitting a 120-second timeout), the queue behind it doesn't just grow linearly—it cascades across adjacent lanes and causes gridlock. In systems where task execution time has high variance (heavy tails), queues blow up long before average system capacity is reached."* Follow this analogy immediately with the simple linear memory growth formula ($\Delta \lambda \times \bar{S}_{\text{traj}}$) already present in Notebook 14.4.
3. **An Architectural Demarcation Between Infrastructure Management and Data Curation (§5.4)**:
   - *What's missing*: A clear boundary explaining that an ML data engineer is not expected to write raw kernel syscalls, but must configure sandbox runners correctly.
   - *Bridge needed*: A 1-paragraph clarifying note: *"In production ML engineering, you will rarely execute raw `cgroup.kill` or `ip link delete` commands in your curation scripts. Instead, container runtimes (such as Docker, Podman, or Firecracker) manage these boundaries. However, data engineers must understand that running millions of agent rollouts stresses host operating systems in ways traditional data pipelines do not: uncoordinated script terminations leave lingering processes, abandoned virtual network interfaces, and unreleased storage blocks that can bring down entire worker clusters."*
4. **A Concrete End-to-End Walkthrough of One Self-Improvement Iteration (§5.1–§5.3)**:
   - *What's missing*: A step-by-step example showing how one round of Rejection Sampling / Expert Iteration unfolds on a concrete task.
   - *Bridge needed*: A concrete scenario:
     - *Round 0*: Base policy $\pi_0$ attempts a coding task fixture 8 times ($k=8$). It succeeds on 2 attempts and fails on 6 (pass rate $p = 0.25$). Per Equation 14.4, this task is highly informative ($P_{\text{inform}} \approx 0.90$).
     - *Curation*: The verifier cascade admits the 2 successful traces; the failures are routed to diagnostic analysis.
     - *Training*: SFT fine-tunes $\pi_0$ on the admitted traces, producing policy $\pi_1$.
     - *Round 1*: Policy $\pi_1$ attempts the same task 8 times. Now it succeeds on 7 attempts ($p = 0.875$). The task has moved toward the solved end; its informative fraction drops.
     - *Takeaway*: Re-sampling this task repeatedly in Round 2 provides diminishing returns. The curation engine must detect this pass-rate shift and rotate the task fixture out of the active sampling band to avoid distribution collapse.

## Pacing Assessment

- **Disproportionately Dense / Over-Indexed Sections**:
  - **Section 5.4 / Section 6.3 (Hypervisor Hygiene, lines 1098–1154)**: Devotes excessive space and two full tables (Tables 14.7 and 14.8) to Linux kernel subsystems (`CLONE_NEWPID`, `PR_SET_CHILD_SUBREAPER`, `ip netns del`, `ioctl(BLKDISCARD)`, DeviceMapper thin-provisioning). This is far too granular for an agent data curation chapter; it belongs in Chapter 8 (Sandboxes) or a devops deployment appendix.
  - **Section 7.3 (AST MinHash & Deduplication, lines 1443–1476)**: Compresses an entire week of randomized data-mining theory (MinHash, 2-universal hashing, LSH banding, S-curves) into 30 lines of ungrounded mathematical formulas without a single visual diagram or worked numeric example.
  - **Section 6.1 (Cryptographic Lineage Envelope, lines 1165–1216)**: Spends extensive space detailing Merkle tree intermediate node calculations across 32 steps ($2,016$ bytes), HMAC signatures, and hardware GEMM floating-point reduction non-associativity, which distracts from the core message of dataset reproducibility and tracking.
- **Under-Indexed / Rushed Sections**:
  - **Section 5 (The Self-Improvement Loop, lines 848–902)**: Rejection sampling, expert iteration (STaR), staleness, collapse, and production feedback are squeezed into barely 50 lines. For an agent systems student, understanding how to run iterative self-improvement loops (generate $\to$ verify $\to$ fine-tune $\to$ repeat) is one of the most practically valuable skills in modern agent engineering. It deserves a dedicated workflow diagram, an illustrative walkthrough, and concrete algorithm pseudocode.
  - **Hard Negative Mining and Annotation Mechanics (§4.2, §4.3)**: While the chapter strongly warns against using hard negatives in SFT, it provides very little detail on how the divergence turn $t_{\text{div}}$ is actually detected algorithmically in production (e.g., using process reward models, state diff heuristics, or first-error exit handlers) before formatting them into DPO pairs.
  - **Practical Data Engineering Tooling**: The student is left wondering what modern libraries orchestrate these multi-terabyte pipelines. Mentioning standard industry tools (e.g., Ray Data, Hugging Face Datasets, Apache Arrow, Celery) would provide helpful real-world grounding for students transitioning from theory to practice.

## Top 3 Priority Improvements

1. **Add a Visual Walkthrough and Gentle On-Ramp for AST MinHash & LSH Banding (§7.3)**
   - *Action*: In Section 7.3, introduce an intuitive diagram or 2-paragraph worked toy example showing how AST diffs are converted into normalized structural shingles, hashed into MinHash signatures, and divided into bands of rows to create the S-curve collision threshold. Explain *why* banding creates an S-curve before presenting the formula $P_{\text{collision}} = 1 - (1 - J^r)^b$.
2. **Demote Hypervisor Kernel Plumbing to Chapter 8 or an Infrastructure Note (§5.4 / §6.3)**
   - *Action*: Streamline Section 5.4/6.3 ("Hypervisor hygiene"). Replace the raw Linux kernel syscalls (`CLONE_NEWPID`, `PR_SET_CHILD_SUBREAPER`, `ioctl(BLKDISCARD)`, `lsof +L1`, `ip netns del`) with a higher-level discussion of sandbox lifecycle management and resource leaks (process leaks, storage ghost-bloat, network port exhaustion). Cross-reference Chapter 8 (Sandboxes) for kernel-level implementation details, keeping this chapter's focus strictly on data collection throughput and stability.
3. **Expand the Self-Improvement Loop with a Concrete Worked Iteration Walkthrough (§5.1–§5.3)**
   - *Action*: Expand Section 5 ("The Self-Improvement Loop") by adding a concrete step-by-step walkthrough and workflow diagram of one complete iteration of Expert Iteration / Rejection Sampling Fine-Tuning. Show how a base policy $\pi_0$ samples rollouts on a fixture, how the verifier cascade selects successes, how informative tasks are identified via $P_{\text{inform}}(p, k)$, and how the resulting model $\pi_1$ causes the informative band to shift, necessitating task refresh to avoid collapse.
