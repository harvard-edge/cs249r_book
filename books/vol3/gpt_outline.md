# Machine Learning Systems, Volume III: Agentic Systems

## Independent Outline Proposal

Agentic machine learning systems turn models from passive prediction components into participants in stateful feedback loops. An agent observes an environment, constructs state, selects actions, receives feedback, and repeats the cycle until it reaches a termination condition. Engineering such a system therefore requires more than model capability. It requires explicit control over state, interfaces, resources, failure propagation, permissions, and adaptation.

The volume should teach a durable systems discipline for this emerging class of software. It should not organize itself around current agent frameworks, collections of prompt patterns, or claims that greater autonomy always produces a better system. Its central argument should be that constraints determine the appropriate form and degree of agency.

## Unifying Systems Model

Every agentic system can be analyzed through six coupled dimensions:

| Dimension | Systems question |
|:---|:---|
| Objective | What outcome must the system achieve, and how is success measured? |
| Environment | What can the system observe, and what can change independently of it? |
| State | What information persists across decisions, where does it reside, and when does it become invalid? |
| Actions | What can the system do, with which permissions, costs, and consequences? |
| Control | What constrains, approves, interrupts, and terminates execution? |
| Feedback | How does the system detect progress, failure, and opportunities to adapt? |

These dimensions provide a common vocabulary for comparing an assistant that makes one tool call, a coding agent that works for an hour, and an embodied agent that operates continuously. They also expose the defining systems tension of agency: each additional degree of freedom can increase capability while expanding cost, uncertainty, and blast radius.

The recurring execution loop is:

```text
Observe → Construct state → Decide → Act → Verify → Update state → Stop or repeat
```

Verification belongs inside the loop rather than after it. Without verification, an incorrect observation or action becomes state for the next decision, allowing errors to compound across the trajectory.

# Part I: From Models to Agents

Part I establishes why agentic systems form a distinct systems class and gives readers the abstractions used throughout the volume.

## Chapter 1: From Prediction to Action

Conventional machine learning systems map inputs to predictions. Generative systems extend this mapping to open-ended outputs, while agentic systems place generation inside a feedback loop that changes the environment. This chapter defines agency as a spectrum characterized by persistence, goal-directed behavior, environmental interaction, and authority to act. It distinguishes agents from chat interfaces, fixed workflows, and ordinary automation, then develops criteria for deciding when agency is warranted.

The distinction matters because an incorrect prediction and an incorrect action have different consequences. Once model output can modify files, spend money, contact people, or control machinery, interface design and execution policy become part of the machine learning system.

## Chapter 2: Anatomy of an Agentic System

An agent combines a model with instructions, context construction, memory, planning, tools, environment interfaces, an execution runtime, evaluation, and human oversight. This chapter decomposes a complete system into those components and traces one task through the full observation–action loop. It separates the model's internal computation from externally represented state and distinguishes the agent policy from the runtime that enforces it.

The decomposition prevents model-centric reasoning from obscuring failures elsewhere in the stack. A poor result may originate in stale context, an ambiguous tool schema, a lost state transition, an unavailable dependency, or an unsafe authorization rule even when the model behaves as designed.

## Chapter 3: Objectives, Constraints, and Degrees of Autonomy

Agent design begins with a task contract: the desired outcome, acceptable process, resource budget, authority boundary, and stopping condition. This chapter examines the trade-offs between capability and predictability, autonomy and control, quality and cost, and task duration and recoverability. It introduces a risk-based autonomy ladder ranging from suggestions and action previews to bounded execution and long-running independent operation.

The chapter also establishes a negative design principle: many applications should remain deterministic workflows. Agency earns its place when the environment cannot be fully anticipated and model-directed decisions add enough value to justify their uncertainty and operational cost.

# Part II: The Agent Loop

Part II develops the components that allow a single agent to perceive, decide, act, and retain state.

## Chapter 4: Models as Decision Components

Models can serve as policies, planners, routers, critics, extractors, and interfaces. This chapter explains how those roles impose different requirements for accuracy, latency, context length, structured output, and calibration. It covers model routing, cascades, reasoning-time computation, sampling, local and hosted execution, and the division of labor between learned decisions and deterministic code.

Treating every model call as unconstrained text generation wastes resources and weakens reliability. A systems design should assign each decision to the least costly component that can satisfy its contract, then reserve more capable models for decisions whose ambiguity warrants them.

## Chapter 5: Context Engineering and Perception

An agent acts on the environment represented in its current context. This chapter treats context construction as a perception pipeline that selects, ranks, transforms, and labels observations before a model receives them. It covers instruction hierarchy, retrieval, context budgeting, compression, provenance, freshness, multimodal inputs, and the separation of trusted control information from untrusted environmental content.

The perception framing explains why context errors resemble corrupted state more than weak prompting. Missing observations produce blind action, irrelevant observations consume finite attention, and malicious observations can redirect execution unless the system preserves trust boundaries.

## Chapter 6: Memory and Persistent State

Long-running agents require state that survives individual model calls. This chapter distinguishes working, episodic, semantic, and procedural memory from conversation history and authoritative application state. It examines storage and retrieval policies, consolidation, forgetting, contradiction resolution, temporal validity, event sourcing, replay, user inspection, retention, and deletion.

Memory increases continuity at the cost of hidden dependencies. A durable memory can preserve a useful lesson, but it can also preserve an incorrect inference long after the evidence has expired. The systems problem is therefore selective persistence with explicit provenance and lifecycle control.

## Chapter 7: Tools and Action Interfaces

Tools define the agent's effective action space. This chapter develops principles for schemas, preconditions, postconditions, typed results, error semantics, idempotency, timeouts, retries, transactions, compensating actions, and result verification. It compares interfaces to databases, search systems, browsers, code executors, services, and physical actuators while keeping the treatment independent of any specific framework.

A model can select only among the actions its interfaces expose, and it can reason only from the feedback those interfaces return. Tool design therefore determines capability, observability, and safety simultaneously.

## Chapter 8: Planning, Search, and Termination

Reactive policies work for short tasks, but long horizons require control over decomposition, dependencies, progress, and resource consumption. This chapter compares reactive execution, plan-and-execute architectures, hierarchical task decomposition, search, critique, reflection, and replanning. It gives special attention to termination conditions, loop detection, budget exhaustion, and the cases in which explicit planning adds latency without improving outcomes.

Planning is valuable only when it reduces uncertainty or coordinates dependencies. The runtime must remain able to reject, revise, or terminate a plan because plans are hypotheses about an environment that may change during execution.

# Part III: Architectures of Agency

Part III composes the agent loop into system-level patterns and assigns agency across software components and people.

## Chapter 9: Workflows, Agents, and Hybrid Control

This chapter compares deterministic pipelines, finite-state machines, graphs, event-driven workflows, model-directed loops, and hybrid architectures. It shows how to place learned decisions inside deterministic control structures and how to escalate from a fixed path to model-directed behavior only when the task requires it.

Hybrid control often provides the best engineering boundary. Deterministic code can enforce invariants and predictable transitions, while models resolve ambiguity at selected decision points.

## Chapter 10: Single-Agent Design Patterns

Recurring single-agent structures include routing, tool use, planner–executor separation, generator–critic iteration, research and synthesis, checkpointed execution, and asynchronous background work. This chapter derives each pattern from the constraint it resolves, compares its state and control requirements, and identifies its common failure modes.

Pattern names matter less than the decisions they isolate. The durable lesson is how each architecture assigns observation, choice, execution, and verification across components.

## Chapter 11: Multi-Agent Systems

Multiple agents can specialize, explore alternatives in parallel, review one another, or form hierarchical teams. This chapter examines supervisor–worker systems, delegation, debate, ensembles, shared and private state, communication protocols, task allocation, and consensus. It also analyzes coordination overhead, duplicated work, conflicting goals, deadlock, correlated errors, and the difficulty of assigning credit.

Multi-agent systems should be treated as distributed systems with learned participants. They justify their added complexity only when specialization, isolation, or parallel exploration produces measurable gains over a single agent with equivalent computation.

## Chapter 12: Humans in the Control Loop

Human participation can provide intent, approval, correction, escalation, and accountability. This chapter covers mixed-initiative interaction, confidence- and risk-based escalation, action previews, interruptibility, review interfaces, undo, and feedback collection. It distinguishes meaningful control from approval fatigue and examines how interface design shapes automation bias.

Human oversight is part of the control architecture rather than a generic fallback. The system must decide which decisions require review, what evidence the reviewer needs, and whether intervention remains possible before consequences become irreversible.

# Part IV: Runtime and Infrastructure

Part IV addresses the infrastructure required when an agent persists beyond one request and coordinates models, tools, data, and external services.

## Chapter 13: Durable Agent Runtimes

Agent tasks may run for minutes, hours, or longer while depending on services that fail independently. This chapter covers state machines, event logs, queues, schedulers, checkpoints, concurrency, cancellation, leases, resumable execution, dependency tracking, isolation, and recovery after process failure. It distinguishes logical agent state from transient runtime state.

Ordinary request–response infrastructure assumes that work completes within one process lifetime. Durable execution replaces that assumption with explicit state transitions that can be reconstructed, inspected, and resumed.

## Chapter 14: Retrieval and Knowledge Infrastructure

Agents need information that is relevant, current, authorized, and actionable. This chapter covers ingestion, indexing, hybrid retrieval, ranking, structured databases, knowledge graphs, active information gathering, caching, provenance, freshness, invalidation, and access control. It explains how retrieval policy interacts with planning and memory rather than treating retrieval as an isolated preprocessing step.

An agent can amplify a small retrieval error by acting on it repeatedly. Knowledge infrastructure must therefore expose uncertainty, source boundaries, and temporal validity to the decision loop.

## Chapter 15: Scheduling, Scaling, and Economics

An apparent user request can produce dozens of model calls, tool invocations, retrieval operations, and verification steps. This chapter decomposes end-to-end latency and cost, then examines batching, caching, parallel and speculative execution, rate limits, backpressure, model routing, search budgets, admission control, and capacity planning. It introduces quality–latency–cost–risk frontiers for comparing architectures.

Agentic workloads make resource demand data dependent because the trajectory length is unknown in advance. Schedulers must control both instantaneous load and the expanding work that one task may generate through retries, delegation, or search.

# Part V: Evaluation and Reliability

Part V develops methods for measuring and improving systems whose outputs, trajectories, and environments can all vary between runs.

## Chapter 16: Evaluating Outcomes and Trajectories

Agent evaluation must measure task success, process compliance, efficiency, and side effects. This chapter covers outcome- and trajectory-level metrics, partial credit, benchmark construction, environment simulators, offline and online evaluation, human assessment, model-based judges, statistical uncertainty, contamination, and reproducibility under nondeterminism.

The final answer alone cannot reveal whether an agent violated a constraint, consumed excessive resources, or succeeded through an unsafe action. Evaluation must assess both what the system achieved and how it achieved it.

## Chapter 17: Testing and Verification

This chapter joins software testing with machine learning evaluation. It covers unit and contract tests for tools, scenario suites, trajectory assertions, invariants, property-based testing, simulation, fault injection, regression analysis, shadow execution, canary releases, and formal verification for constrained components.

Nondeterminism changes the meaning of a passing test. Reliable release decisions require distributions over behavior, explicit invariants for unacceptable outcomes, and tests that isolate failures across model, runtime, tool, and environment boundaries.

## Chapter 18: Observability and Causal Debugging

A complete agent trace includes observations, constructed context, model decisions, tool calls, state transitions, policy checks, costs, and external effects. This chapter covers structured traces, semantic logging, distributed tracing, privacy-preserving telemetry, replay, counterfactual analysis, failure taxonomies, and cross-version trajectory comparison.

Observability must support causal reconstruction rather than transcript collection. Engineers need to determine which state or decision first diverted the trajectory and why later safeguards failed to contain it.

## Chapter 19: Robustness and Recovery

Long action chains encounter model errors, invalid tool results, stale observations, unavailable services, partial side effects, and changing environments. This chapter examines bounded retries, fallback policies, checkpointing, rollback, compensating actions, circuit breakers, graceful degradation, safe termination, and recovery-oriented design.

Error probability compounds with trajectory length unless the system verifies intermediate state and limits propagation. Reliability therefore depends on containing local failures before they become global trajectory failures.

# Part VI: Security, Safety, and Governance

Part VI treats authority and consequence as first-class systems resources.

## Chapter 20: Security Boundaries for Agents

This chapter develops an agent-specific threat model spanning prompt injection, indirect prompt injection, tool compromise, malicious content, data exfiltration, credential theft, confused-deputy behavior, and dependency attacks. It covers least privilege, capability-based authorization, sandboxing, network and filesystem isolation, secrets management, tenant boundaries, audit records, and policy enforcement outside the model.

Agentic execution joins control and data channels in ways conventional applications try to avoid. An untrusted document can contain language that resembles an instruction, so the surrounding system must enforce boundaries that the model cannot reliably infer on its own.

## Chapter 21: Safety, Alignment, and Governance

Agentic failures can arise from misspecified goals, unsafe optimization, distribution shift, or correct execution of an inappropriate request. This chapter covers action risk classification, authorization thresholds, reversibility, policy layers, accountability, privacy, fairness, incident response, documentation, and organizational governance. It distinguishes model alignment from system-level control.

No single safeguard governs the full loop. Models, tools, runtimes, interfaces, operators, and institutions each control different failure paths, so a defensible design must assign responsibility across layers.

# Part VII: Adaptation and Open Problems

Part VII completes the lifecycle by examining how agents learn from operation and how the same principles extend to more demanding environments.

## Chapter 22: Learning from Experience

Agent trajectories create data about successful decisions, failed actions, human corrections, and environmental response. This chapter covers feedback collection, trajectory curation, prompt and policy updates, fine-tuning, preference optimization, distillation, memory-based adaptation, online learning, and exploration. It examines feedback loops, evaluation drift, rollback, and approval of learned changes.

An adaptive agent changes the process that generates its future training data. The system must preserve stable evaluation and governance boundaries while allowing selected components to improve.

## Chapter 23: Production Lifecycle and Organizations

This chapter follows an agentic system from prototype through release, operation, migration, and retirement. It covers versioning models, instructions, tools, policies, and data; release gates; service-level objectives; incident management; ownership boundaries; vendor migration; technical debt; and build-or-buy decisions.

Agent quality depends on coordinated ownership across teams that traditionally manage models, applications, infrastructure, security, and policy separately. Lifecycle discipline makes those dependencies explicit and keeps a prototype's implicit assumptions from becoming production liabilities.

## Chapter 24: Embodied and Open-Ended Agents

The closing chapter applies the volume's abstractions to computer-use agents, software engineering agents, scientific discovery systems, robots, and agents that operate over long time horizons. It examines partial observability, real-time deadlines, physical consequences, continual learning, open environments, and emergent coordination. The chapter closes by distinguishing established engineering principles from open research questions.

These domains stretch every dimension of the systems model while preserving the same core problem. An agent must convert incomplete observations into bounded actions while managing state, resources, uncertainty, and consequence.

## Recurring Lighthouse Systems

Four recurring systems can give the volume a continuous empirical thread:

| Lighthouse system | Dominant constraint regime | Concepts it exposes |
|:---|:---|:---|
| Research agent | Evidence quality and information freshness | Retrieval, provenance, planning, citation verification, and evaluation |
| Software engineering agent | Long-horizon state and verifiable action | Tools, sandboxing, testing, recovery, and human review |
| Enterprise operations agent | Authority and irreversible consequence | Access control, privacy, auditability, escalation, and governance |
| Embodied agent | Partial observability and real-time interaction | Perception, latency, safety envelopes, continual state, and physical recovery |

Each chapter should revisit a subset of these systems rather than inventing a disconnected example. Their differences reveal how the same architectural choice changes under different constraints.

## Recommended Chapter Contract

Each chapter should make one systems argument and support it with a consistent pedagogical sequence:

1. Open with a concrete system failure or design tension.
2. Identify the constraint that produces the problem.
3. Develop the governing abstraction or quantitative model.
4. Compare architectural responses and their trade-offs.
5. Trace the blast radius of each major optimization.
6. Apply the reasoning to one or more lighthouse systems.
7. Evaluate failure modes, metrics, and operational consequences.
8. Close with durable principles and open limitations.

The contract prevents the volume from becoming a taxonomy of components. Each chapter should explain why a mechanism exists, what constraint it resolves, when it fails, and how its cost can be measured.

## Scope Boundaries

The volume should avoid organizing chapters around current frameworks, vendor products, prompt recipes, or speculative claims about general intelligence. Specific systems can serve as evidence, but the teaching unit should remain the durable mechanism. Multi-agent designs should receive the same cost and coordination scrutiny as any distributed system, and security should recur across tools, memory, runtime, and evaluation rather than appearing only in a closing checklist.

The proposed structure deliberately gives evaluation, reliability, runtime design, and security as much weight as reasoning and planning. That balance is what makes the subject machine learning systems rather than an application guide for language models.
