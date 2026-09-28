# Chapter 12: Failure Recovery — Student Accessibility Review

## Executive Summary

Chapter 12 (*Failure Recovery*) is an intellectually rich and vital chapter that tackles one of the hardest practical problems in agent systems: how to build runtime supervisors that survive partial failures, avoid infinite repair loops, and gracefully handle irreversible actions without corrupting external environments. The chapter shines brightest when grounding reliability engineering in vivid, agent-specific failure modes—such as an agent oscillating cyclically between Python imports while debugging, becoming an accidental distributed denial-of-service (DDoS) bot by rephrasing failing API queries, or accidentally executing an irreversible action (like charging a credit card or flipping production DNS) before verifying upstream dependencies. However, the chapter frequently alienates senior undergraduate and early graduate students by dropping them into an unforgiving thicket of unintroduced graduate prerequisites: classical database concurrency theory (Two-Phase Commit, strict Two-Phase Locking, ARIES crash recovery), Linux kernel plumbing (cgroups v2 freezer controller, eBPF Traffic Control blackholes, VFS unmounting), and queueing/reliability engineering formalisms (Markovian random walks, Dirac delta spikes, MTBF/MTTR/RPO/RTO equations). If these systems concepts are anchored with intuitive software analogies and accompanied by concrete Python tool-registration code, this chapter will serve as the definitive guide for students transitioning from toy prompt scripts to robust, production-grade agent runtimes.

## What Works Well

- **The Pytest Repair Spin Loop Trace (§3.3, Lines 603–613)**: This is one of the most effective pedagogical examples in the textbook. Presenting a concrete 4-step execution trace where an agent oscillates between `ImportError` and `AttributeError` (fixing an import, attempting an `hasattr` workaround, causing a `TypeError`, and reverting to the original import) is immediately recognizable to any student who has prompted an LLM to fix failing unit tests. It provides an unassailable justification for why runtime supervisors must enforce anti-spin invariants and semantic error hashing.
- **The Anatomy of an Agent-Induced Retry Storm (§5.1, Lines 1003–1024)**: The diagnosis of why LLMs turn into "adversarial load generators" against struggling microservices is brilliant. Explaining that an autoregressive model does not treat an HTTP 503/504 status as a network fault, but rather as conversational feedback to be overcome by syntactically varying parameters (`query="v2 auth patch"` vs. `query="auth patch v2"`), brilliantly bridges model behavior with distributed systems failure modes.
- **Sunk Cost Trade-offs and the Break-Even Repair Notebook (§3.1 & Notebook 12.1, Lines 389, 501–593)**: The motivating scenario—where an agent fails on step 6 of an 8-step cloud migration due to a malformed JSON escape quote or a port passed as a string (`"8080"` vs. `8080`)—dramatically illustrates the folly of unconditional rollback. Unwinding would tear down an expensive VPC and database replica that took minutes and dollars to build. The break-even repair equation ($p^* = c_r / (U+R)$) gives students a practical, quantitative decision rule for runtime arbitration.
- **Tripartite Action Reversibility Taxonomy & The Pivot Boundary (§4.1–4.2, Lines 649–724)**: Classifying tool actions into Class I (Invertible), Class II (Compensable), and Class III (Irreversible / Pivot) provides a clean, highly actionable mental model. The "Dual-Commit Trap" case study (deploying an inference cluster, warming KV caches, flipping global BGP routing, running a DB migration, and emitting an audit webhook) vividly demonstrates why irreversible pivot actions must be structurally ordered after all fallible and compensable steps.
- **Fuel Gauge vs. Velocity Sensor Analogy (§5.1, Lines 827–831)**: The conceptual distinction between Resource Depletion Budgets (the fuel gauge tracking cumulative steps/tokens) and Semantic Progress Watchdogs (the velocity sensor measuring the directional derivative of progress) cleanly demystifies why a 50-step budget fails to prevent an agent from wasting 40 turns thrashing on the same configuration file.
- **The Quadratic Cost of Undetected Loops (Notebook 12.2, Lines 938–958)**: Deriving the token expenditure of a looping agent as an arithmetic series ($n \cdot s_0 + \Delta \cdot n(n-1)/2$) proves that unmonitored loops impose an $O(n^2)$ token penalty due to monotonic prompt context expansion. This grounds early cycle detection in inescapable economic reality.
- **Fail-Safe Containment in Prompt Injection Scenarios (§6.1, Lines 1245–1252)**: Pointing out that an agent compromised by indirect prompt injection cannot be trusted to execute its own compensating transactions or cleanup routines (since an adversary could hijack the cleanup to delete tables or exfiltrate secrets) is a critical systems insight that prevents students from building naive self-cleaning agent loops.

## Barrier Spots

### 1. Classical ACID & Two-Phase Commit Opening Salvo
- **Location**: Section 12.1 (*The Transactional Boundary Collapse*, lines 61–76, Table 12.1).
- **The Barrier**: The chapter opens its core technical argument by critiquing classical database transaction theory: Atomicity, Isolation, Jim Gray 1981, private undo logs, shadow pages, Two-Phase Commit ($2\text{PC}$), prepare phases, Long-Lived Transactions (LLTs), exclusive write locks ($X$-locks), shared read locks ($S$-locks), strict two-phase locking (2PL), serializability, and Gray's deadlock growth formula $O(N^2 \cdot L^4)$.
- **Why It's a Barrier**: Assumes an advanced undergraduate or graduate course in Database Management Systems and Concurrency Control. The target student reader (who understands REST APIs, JSON payloads, and basic SQL) has never studied 2PC coordinator protocols, prepare phases, or strict 2PL lock managers. Critiquing an architecture the student has never seen creates acute cognitive friction and alienates them before the text introduces what an agent Saga actually is.
- **Severity**: High (student feels unqualified to understand agent recovery because they lack classical database internals).

### 2. Linux Kernel Storage & Network Plumbery in "The Two Spheres of Rollback"
- **Location**: Section 12.2 (*The two spheres of rollback: Sandbox copy-on-write vs. distributed Sagas*, Callout Perspective 12.1, lines 97–109).
- **The Barrier**: The text explains the boundary between local sandboxes and external APIs by diving into kernel storage and networking primitives: OverlayFS `upperdir`, ZFS clones, Btrfs extent reflinks, eBPF traffic control (`tc`) filters, cgroup socket hooks, VFS unmount (`umount -l`), and backing extent pool purges.
- **Why It's a Barrier**: Assumes low-level Linux systems programming and container runtime internals. An undergraduate understands containers at the level of "Docker isolates a file system," but has no background in kernel virtual filesystem (VFS) extent trees or eBPF socket mediation. The distinction between mechanical file discard and semantic API compensation is obscured by kernel jargon.
- **Severity**: Medium (student skips the callout perspective and misses the vital distinction between local workspace rollbacks and external API compensations).

### 3. Formal Transitive Closures and Topological Cuts for Causal Dependency DAGs
- **Location**: Section 12.2 (*The causal resource dependency DAG versus LIFO saga freezing*, lines 273–299).
- **The Barrier**: To explain why a failure to delete a DNS record shouldn't prevent the runtime from tearing down a \$5.50/min GPU cluster, the text introduces formal graph theory and set-theoretic closures: directed graphs $\mathcal{G}_{\text{comp}} = (\mathcal{V}_{\text{comp}}, \mathcal{E}_{\text{comp}})$, transitive forward dependency closures ($\mathcal{V}_{\text{frozen}} = \{ C_k \in \mathcal{V}_{\text{comp}} \mid C_i \xrightarrow{*} C_k \} \cup \{C_i\}$), set difference cuts ($\mathcal{V}_{\text{indep}} = \mathcal{V}_{\text{comp}} \setminus \mathcal{V}_{\text{frozen}}$), dirty-read taint broadcasts, and administrative reconciliation tokens.
- **Why It's a Barrier**: The motivation (preventing capital bleed from head-of-line blocking on unrelated resources) is outstanding. However, dressing it up in formal discrete math and set notation makes a straightforward scheduling concept feel like an arcane research paper on distributed consensus.
- **Severity**: Medium (student struggles through the math notation when a simple diagram and intuitive explanation would suffice).

### 4. Two-Phase Pivot Settlement Protocol & Outbox Staging
- **Location**: Section 12.4 (*The two-phase pivot settlement protocol*, lines 735–760).
- **The Barrier**: When addressing the "network settlement cliff" (handling an ambiguous HTTP timeout on an irreversible tool call like a bank transfer or DNS cutover), the text introduces the Transactional Outbox pattern, HMAC-SHA256 nonces, `fdatasync()` storage barriers, and status reconciliation probes without explaining what an Outbox pattern is or how a client actually executes a reconciliation probe.
- **Why It's a Barrier**: Assumes knowledge of distributed systems consistency patterns (the Outbox Pattern, write barriers, the Two Generals problem). The student understands that a network timeout leaves the outcome ambiguous, but is left wondering: *What is an outbox table in an agent runtime? How does an external API recognize my HMAC key? What does a reconciliation probe actually send over HTTP?*
- **Severity**: High (leaves students with a gaping conceptual hole right at the most dangerous point of agent execution: irreversible actions).

### 5. Thundering Herds, Dirac Deltas, and Markovian Random Walks for Jitter
- **Location**: Section 12.6 (*Jittered backoff protocols*, lines 1068–1114).
- **The Barrier**: To explain retry backoff with jitter, the text invokes queueing theory and continuous probability abstractions: synchronized Dirac delta spikes, resonance patterns, continuous uniform distributions over $[0, \text{ceiling}]$, and a "Markovian random walk" for Decorrelated Jitter ($t_{j+1} = \min(t_{\max}, \text{Uniform}(t_{\text{base}}, 3 t_j))$).
- **Why It's a Barrier**: While senior undergraduates know basic probability, presenting a 5-line random delay function in terms of Dirac delta distributions and Markovian state updates creates unnecessary mathematical intimidation for what is conceptually simple: adding randomized delays to retries so clients don't strike a server at the exact same millisecond.
- **Severity**: Low/Medium (student grasps the Python code easily, but is intimidated by the surrounding text).

### 6. Kernel-Level Quarantine Mechanics (cgroups v2 & eBPF Blackholes)
- **Location**: Section 12.7 (*The four-stage quarantine protocol*, lines 1264–1288, Listing 12.3).
- **The Barrier**: When an agent trajectory is compromised by prompt injection, the text specifies a 4-stage quarantine pipeline executed via Linux kernel primitives: `cgroup.freeze` controller, `SIGSTOP`, `ip link set dev down`, and replacing the interface queueing discipline with `tc qdisc replace dev veth_iface root blackhole`.
- **Why It's a Barrier**: Assumes systems administration and Linux network engineering experience. An agent application developer or ML student has never interacted with `tc` (traffic control) or modified cgroup v2 sysfs nodes directly. They understand the need to freeze a process and sever networking, but the specific kernel commands distract from the architectural principles.
- **Severity**: Medium (student feels this is a DevOps/Sysadmin text rather than an agent systems book).

### 7. Reliability Engineering Acronym Dump (MTBF, MTTR, RPO, RTO)
- **Location**: Section 12.8 (*Fault injection benchmarks*, lines 1436–1476, Notebook 12.3).
- **The Barrier**: The section suddenly introduces classical Site Reliability Engineering (SRE) and disaster recovery metrics: $\text{MTBF} = \frac{\sum T_{\text{op}}}{N}$, $\text{MTTR} = \frac{\sum T_{\text{rec}}}{N}$, $\text{RPO} = \text{Lost Turns}$, $\text{RTO} = T_{\text{restart}} - T_{\text{crash}}$, alongside Jim Gray's taxonomy of "Bohrbugs vs. Heisenbugs".
- **Why It's a Barrier**: Assumes background in enterprise IT disaster recovery and reliability engineering. The crucial distinction between RPO (data currency / how many turns of work you lost) and RTO (downtime / how long until the agent is running again) is never introduced intuitively before being calculated across NVMe disk bandwidths and GPU prefill throughput.
- **Severity**: Medium (student treats the formulas as arbitrary math without appreciating their practical operational meaning).

## Assumed Knowledge Gaps

### Database Internals & Transaction Processing
- **Classical Transaction Semantics**: ACID definitions, Two-Phase Commit ($2\text{PC}$) coordinator/participant protocols, X/Open XA standard, prepare phase vs. commit phase, and the assumption of non-blocking distributed locks.
- **Storage Engine Concurrency & Recovery**: Write-Ahead Logging (WAL) and ARIES crash recovery (`BEFORE-IMAGE` undo logs vs. `AFTER-IMAGE` redo logs), shadow paging, exclusive ($X$) and shared ($S$) locks, Strict Two-Phase Locking (2PL), serializability, Long-Lived Transactions (LLTs), Gray's deadlock growth equation $O(N^2 \cdot L^4)$, and the Transactional Outbox pattern.

### Distributed Systems Theory
- **Saga Formulations**: Garcia-Molina & Salem (1987) long-lived transactions, semantic compensation vs. physical rollback, eventual consistency, and dirty read anomalies ($I=0$).
- **Distributed Network Faults**: The Two Generals problem, network timeout ambiguity (packet lost going out vs. response lost coming back), idempotency keys / nonces (Stripe/RFC 8474), and distributed deadlock cycles.
- **Graph & Provenance Tracking**: Transitive closures in directed acyclic graphs (DAGs), topological cuts, and dynamic taint tracking.

### Operating Systems & Linux Kernel
- **Storage Subsystems**: Copy-on-Write (CoW) filesystems (OverlayFS `upperdir`, ZFS clones, Btrfs reflinks), VFS unmount mechanics (`umount -l`), backing extent pools, and storage barrier syscalls (`fsync` vs. `fdatasync`).
- **Kernel Process & Network Isolation**: cgroups v2 freezer subsystem (`cgroup.freeze`), POSIX signals (`SIGSTOP` vs. `SIGKILL`), eBPF Traffic Control (`tc`) filters, queueing disciplines (`qdisc blackhole`), virtual Ethernet pairs (`veth`), POSIX file locks, and systemd watchdog signals (`sd_notify("WATCHDOG=1")`).

### Reliability Engineering & Queueing Theory
- **High Availability & Disaster Recovery**: Mean Time Between Failures (MTBF), Mean Time to Repair (MTTR), Recovery Point Objective (RPO), Recovery Time Objective (RTO), and Chaos Engineering principles.
- **Software Fault Taxonomies**: Jim Gray's classification of Bohrbugs vs. Heisenbugs.
- **Traffic Shaping & Queueing**: Thundering herd problem, Poisson arrival rates ($\lambda$), Dirac delta distribution spikes, and Markovian random walks for backoff decorrelation.

## Jargon Density Hotspots

### Hotspot 1: Classical Database Transaction Arcana (§12.1, Lines 61–65)
> *"Under classical database semantics, the twin guarantees of Atomicity and Isolation govern state mutation: either an entire batch of modifications commits successfully, or the runtime executes an abort, relying on private undo logs, shadow pages, or write locks to erase every intermediate write and render the aborted attempt completely invisible to concurrent observers. When transactions span multiple independent storage nodes, distributed systems deploy Two-Phase Commit ($2\text{PC}$) protocols, wherein a central coordinator establishes a prepare phase to force participants into a pre-committed, locked state... Under strict two-phase locking (2PL), serializability is preserved because the duration of lock holding is limited... As Gray showed, the probability of deadlock in a system with $N$ concurrent transactions grows as $O(N^2 \cdot L^4)$..."*

- **Terms requiring background definitions (15 terms in 10 lines)**: Atomicity, Isolation, abort, private undo logs, shadow pages, write locks, Two-Phase Commit ($2\text{PC}$), prepare phase, pre-committed locked state, Long-Lived Transactions (LLTs), exclusive write locks ($X$-locks), shared read locks ($S$-locks), strict two-phase locking (2PL), serializability, Gray's deadlock growth.
- **Student Reaction**: The student feels like they opened a 1980s database systems textbook rather than a modern guide to building AI agents.

### Hotspot 2: Kernel Isolation Plumbing (§12.2, Lines 106–108)
> *"To close this leak, guest network namespaces must enforce a **default-deny egress policy** via eBPF traffic control (`tc`) filters or cgroup socket hooks. Any outbound network transmission must be trapped by the host Tool Execution Gateway, assigned a deterministic idempotency key $\kappa_i$, and logged to the WAL before traversing the physical NIC... Rollback is executed strictly out-of-band by the host supervisor unmounting the VFS layer (`umount -l`) and purging the backing extent pool."*

- **Terms requiring background definitions (10 terms in 6 lines)**: Guest network namespaces, default-deny egress policy, eBPF traffic control (`tc`) filters, cgroup socket hooks, Tool Execution Gateway, idempotency key $\kappa_i$, WAL, physical NIC, VFS layer unmount (`umount -l`), backing extent pool.
- **Student Reaction**: An ML student who interacts with Docker purely via `docker run` has no frame of reference for eBPF qdiscs or VFS extent pools.

### Hotspot 3: The Pivot Settlement Protocol (§12.4, Lines 748–759)
> *"Prior to dispatching any network packets across the physical interface, the supervisor stages the pivot transaction in a durable, non-volatile Outbox table within the local write-ahead log... The supervisor forces an explicit `fdatasync()` storage barrier, transitioning the trajectory state to `STATUS_PIVOT_STAGED`... If the connection severs or times out, the supervisor enters a dedicated `STATUS_PIVOT_RECONCILING` state... launches an out-of-band **Reconciliation Probe**, querying the external service's authoritative status endpoint using $\kappa_{\text{pivot}}$..."*

- **Terms requiring background definitions (9 terms in 8 lines)**: Outbox table, write-ahead log, HMAC-SHA256, `fdatasync()` storage barrier, `STATUS_PIVOT_STAGED`, `STATUS_PIVOT_RECONCILING`, Reconciliation Probe, authoritative status endpoint, idempotency key $\kappa_{\text{pivot}}$.
- **Student Reaction**: The student understands why an ambiguous timeout is bad, but has no clear mental model of how an "Outbox table" and a "Reconciliation Probe" are implemented in Python.

## Missing Bridges

1. **An Agent-First "Flight and Hotel" Analogy for Sagas (§12.1)**:
   - *What's missing*: An intuitive, relatable bridge explaining why distributed Sagas are needed before launching into the historical failures of 2PC and database locking.
   - *Bridge needed*: Before mentioning Jim Gray or Two-Phase Commit, introduce a simple agent scenario: *"Imagine an agent booking a vacation. In Step 1, it calls an airline API to reserve a flight. In Step 2, it attempts to book a hotel room, but the hotel is sold out. Airlines do not provide an 'undo' button; they don't participate in a shared database transaction with the hotel. Instead, the airline exposes a cancellation endpoint. A Saga is simply this pattern: an ordered chain of operations where every forward action that alters external state is paired with an explicit, pre-registered compensating action (a cancellation or cleanup call) that runs if a later step fails."*
2. **A Concrete Python Example of Tool Registration with Compensators (§12.2 & §12.3)**:
   - *What's missing*: Showing how a developer actually writes an agent tool that registers a forward action, its reversibility tier, and its compensating action.
   - *Bridge needed*: A 15-line executable Python snippet showing a tool definition:
     ```python
     @register_tool(reversibility=Tier.COMPENSABLE)
     def create_cloud_vm(instance_name: str, acb: AgentControlBlock) -> str:
         vm_id = cloud_provider.launch_instance(name=instance_name)
         # Register compensating action to the ACB's LIFO stack before returning
         acb.register_compensator(
             compensator=lambda: cloud_provider.terminate_instance(vm_id),
             description=f"Terminate VM {vm_id}"
         )
         return vm_id
     ```
3. **An "E-Commerce Checkout" Analogy for the Network Settlement Cliff (§12.4)**:
   - *What's missing*: An intuitive explanation for why an ambiguous HTTP timeout on an irreversible tool call is dangerous and how reconciliation probes solve it.
   - *Bridge needed*: Connect to everyday online shopping: *"Anyone who has bought tickets online has experienced the anxiety of clicking 'Submit Order', watching the browser spinner freeze for 30 seconds, and getting a '504 Gateway Timeout'. Did your credit card get charged? If you click submit again, you might buy two sets of tickets; if you close the tab, you might lose your seats. An idempotency key acts like your unique order reference: it allows the client to send a safe, read-only status probe asking 'What is the status of Order #8492?' rather than blindly re-submitting the charge."*
4. **An Electrical Breaker Analogy for Tool Circuit Breakers (§12.6)**:
   - *What's missing*: An intuitive on-ramp explaining why circuit breakers trip from Closed to Open before presenting formal sliding-window error fraction equations.
   - *Bridge needed*: A physical analogy: *"In a household electrical panel, a circuit breaker trips open when current spikes, cutting off power before the wires overheat and cause a fire. In an agent system, when an external API begins failing with 503 errors, an LLM's natural reflex is to keep re-trying with slight variations, pouring more traffic onto a dying server. A software circuit breaker acts like that electrical switch: once error rates exceed a threshold, it trips 'Open', failing incoming calls immediately in 0 ms at the host gateway so the remote service has breathing room to recover."*

## Pacing Assessment

- **Disproportionately Dense Sections**:
  - **Section 12.1 (Lines 61–76)**: Devotes excessive space to a historical critique of database 2PC, strict 2PL, ARIES, and Gray's $O(N^2 \cdot L^4)$ deadlock equation. The student needs to understand why distributed APIs lack transactions, not study a 1980s relational database textbook.
  - **Section 12.2 (Lines 97–109 & 273–299)**: Dives into kernel storage arcana (OverlayFS, Btrfs reflinks) and formal set-theoretic notation for Causal Dependency DAGs ($\mathcal{V}_{\text{frozen}} = \{ C_k \in \mathcal{V}_{\text{comp}} \mid C_i \xrightarrow{*} C_k \} \cup \{C_i\}$). Explaining that independent resources (like GPUs) should be de-allocated even if an unrelated DNS cleanup gets stuck is great, but the math notation is unnecessarily heavy.
  - **Section 12.6 & 12.7**: Heavy operating systems and networking plumbery (Linux cgroups v2 freezer, eBPF TC blackholes, Dirac delta distributions, Markovian walks). The conceptual principles of process freeze, network disconnect, and jittered retries are overshadowed by kernel implementation minutiae.
- **Under-Indexed / Rushed Sections**:
  - **Tool Registration Developer Experience**: The chapter tells the reader that tool authors must register forward-compensating pairs, but gives zero examples of what a tool registration schema looks like.
  - **Handling Legacy APIs without Idempotency Support**: The text assumes all external APIs accept idempotency keys and expose reconciliation endpoints. In reality, most third-party REST APIs do not support `Idempotency-Key` headers or status query probes. How should an agent runtime handle these common real-world tools?
  - **Error-Feedback Prompt Engineering**: The text formalizes $a_{\text{repair}} \sim \pi_\theta(a \mid \tau_{<t}, T_t, o_t^{\text{err}})$ and shows that feedback resets epistemic decay, but never shows an actual prompt template for how error frames, stack traces, and repair instructions are structured for the LLM.

## Top 3 Priority Improvements

1. **Replace the Classical Database Opening in Section 12.1 with an Agent-First "Flight + Hotel" Saga Motivation**
   - *Action*: In Section 12.1, condense the lengthy academic discussion of 2PC, ARIES, X/S locks, strict 2PL, and Gray's $O(N^2 \cdot L^4)$ deadlock equation. Replace it with an intuitive 2-paragraph agent scenario: an agent booking a flight and a hotel across independent REST APIs. Show that when the hotel booking fails, the flight cannot be "rolled back" via database ACID mechanics because airlines don't share locks with hotels; the flight must be canceled via a compensating API call. Define the Saga pattern simply and directly. Relocate the technical comparison of 2PC vs. Sagas into Table 12.1 or an optional side callout for students with a database background.
2. **Provide a Concrete, End-to-End Python Code Listing for Defining Tools with Compensators and Reversibility Tiers (§12.2 & §12.4)**
   - *Action*: Add a 20-line Python code listing in Section 12.2 demonstrating how a developer actually writes an agent tool with a paired compensator and reversibility metadata. Show a `@compensable_tool` decorator where the forward function creates an AWS EC2 instance or S3 bucket, returns the resource ID, and automatically registers a corresponding teardown lambda on the ACB's LIFO stack. This grounds the abstract Agent Control Block and Saga Orchestrator in concrete code that students can visualize and adapt in their own agent projects.
3. **Demystify the Network Settlement Cliff with an E-Commerce Analogy and a Visual Flowchart (§12.4)**
   - *Action*: In Section 12.4, replace the formal HMAC equations and `fdatasync()` jargon with the relatable "frozen credit card checkout" analogy (did the payment go through, or will clicking submit charge me twice?). Provide a clear 3-branch visual flowchart or ASCII diagram showing what the Reconciliation Probe does when an HTTP timeout occurs on a pivot action: (1) Probe returns "Committed" $\to$ advance to finalization; (2) Probe returns "Not Found" $\to$ safely retry with identical idempotency key; (3) Probe returns "Unknown/Timeout" $\to$ freeze ACB and escalate to human. This makes the Two-Phase Pivot Settlement immediately actionable without requiring a background in distributed consensus.
