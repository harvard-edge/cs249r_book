# Volume III: Detailed Chapter and Section Plan

**Status:** Proposed curriculum and revision specification. The chapter manuscripts and images have not yet been rewritten to this plan.

**Scope:** Eighteen chapters organized around the stochastic computer: processor, memory, I/O, operating system, learning, and distributed execution. This document expands [the architectural blueprint](ARCHITECTURE_FLOW.md). Its section sequences and coverage decisions are the more detailed specification. The companion [figure reuse audit](FIGURE_REUSE_AUDIT.md) identifies existing visual assets worth retaining, revising, relocating, or retiring from the narrative.

## 1. The educational contract

The graduate should be able to design, measure, diagnose, and improve a digital agentic ML system that uses a learned model to carry a task through a sequence of decisions and interactions. The unit being engineered is the complete trajectory: the observations, decisions, actions, effects, waiting, recovery, and outcome of that task.

The student should finish with an architectural mental model and the ability to make justified design choices. A collection of mechanisms, fashionable terms, or impressive equations does not establish that capability. The stochastic computer organizes the responsibilities that a complete system must fulfill. Its components can use different implementations as models, tools, and infrastructure change.

### Reader prerequisites

Assume working knowledge of ML systems: training and inference, generalization and evaluation, basic neural architectures, accelerator memory and computation, distributed execution, and the latency/throughput distinction. Also assume undergraduate probability, algorithms, systems programming, operating systems, and computer architecture. Prior completion of another volume is unnecessary.

Re-establish the specific assumption needed at the point of use. For example, briefly explain prefill and decode when their costs affect a model invocation, then develop their agent-specific memory and scheduling consequences later. Do not reteach backpropagation, virtual memory, database transactions, or an entire reinforcement-learning course. A short prerequisite reminder or optional reference can support readers without interrupting the main argument.

### Backward design

For every chapter, design in this order:

1. **Lasting takeaways:** the engineering judgments that should remain after implementation details fade.
2. **Learning objectives:** the capabilities that would demonstrate those judgments.
3. **Assessment evidence:** a calculation, diagnosis, comparison, or complete design that would establish the capability.
4. **Section sequence:** the prerequisites and mechanisms that make that assessment possible.
5. **Examples and figures:** representations selected to support the sequence.

This is the author's design order. The published chapter retains the normal teaching order: Purpose, learning objectives, body, fallacies and pitfalls, summary takeaways, and chapter connection. A chapter's conclusion resolves its opening problem and makes the next unresolved responsibility visible.

Section count follows distinct conceptual work. A section deserves its own heading when it changes the question being answered, introduces a necessary mechanism, or requires a substantial comparison. A subsection handles a local distinction within that argument. A worked example can conclude an existing section. Repetition, technologies with the same teaching role, and additional examples do not automatically justify new top-level sections.

## 2. Lasting principles of the volume

The identifiers below are editorial references for alignment. They are not proposed new named laws, acronyms for students to memorize, or claims of universal optimality.

| ID | Lasting principle | What the student must be able to do | Main development |
|---|---|---|---|
| P1 | Engineer the complete task and its effects. | Define the system boundary and judge an outcome using the work required to obtain it. | 1, 17, 18 |
| P2 | Decisions depend on available information and feedback. | Explain uncertainty, missing observations, and how a bounded computation strategy changes the next decision. | 2–4 |
| P3 | Component contracts make learned computation usable. | Separate output form, task correctness, authorization, execution, and evidence of completion. | 2, 7–8 |
| P4 | State has a representation, lifetime, and validity condition. | Choose what to retain, where to retain it, when it can be reused, and what invalidates it. | 4–6, 9 |
| P5 | Authority and intervention require explicit enforcement. | Bound accessible resources and effects, define supervision, and describe remaining limitations. | 8, 10 |
| P6 | Recovery must reconcile recorded intent with external reality. | Handle interrupted work and ambiguous completion without assuming earlier effects have disappeared. | 7, 9–10 |
| P7 | Shared resources require whole-system allocation. | Allocate compute, memory, tool capacity, time, and human attention under workload variation. | 5, 11, 17 |
| P8 | Improvement requires evidence and task-relevant feedback. | Diagnose a limitation, choose an intervention, and evaluate it on independent tasks. | 3, 12–14, 16 |
| P9 | Coordination creates integration work and shared failure modes. | Justify delegation using task structure, authority, communication, and integration costs. | 15 |

These principles recur through increasingly capable designs. They should not become an identical taxonomy paragraph at the beginning and end of every chapter. The existing H-S-A-C lens can remain a compact workload description: horizon, state, authority, and the properties for which completion evidence exists. Its final dimension must describe the scope and limits of checks; it must not imply that a deterministic test establishes complete task correctness or makes all future decisions reliable.

## 3. Content first; archetypes deferred

The active work is the curriculum and chapter content: lasting takeaways, assessable objectives, prerequisite order, mechanisms, quantitative reasoning, and conclusions. No recurring archetype roster is selected or required for this drafting pass. Choose any book-wide reference workloads only after the content establishes where they would improve learning.

Small, self-contained examples remain useful. A code edit, conflicting source, memory allocation, or interrupted request can explain a local concept without becoming a recurring protagonist. The examples suggested in chapter cards are replaceable illustrations or assessment inputs. Their application details must not determine the chapter's order or require an evolving application narrative.

### Local examples and calculations

Supply the assumptions a particular example needs: relevant inputs, state, permitted operations, outcome criteria, evidence limits, and resource constraints. Workload specification remains a systems competency even though named archetypes are deferred. A student should be able to understand each calculation or failure trace from the information supplied locally.

Use parameterized scenarios for calculations: context size, number of calls, tool duration, memory capacity, review time, or evaluation outcomes. During manuscript implementation, physical constants and specifications come from the calculation registry; explicitly hypothetical scenario inputs are labeled and computed through the book's normal calculation cells. Do not attach timeless lessons to a current vendor's token price, a particular benchmark score, or an unsupported production anecdote.

### Durability test

For each proposed subsection, ask whether its engineering question remains useful if the model, framework, tool protocol, and accelerator generation are replaced. Keep the responsibility, constraint, assumptions, mechanism, and tradeoff in the body. Use an implementation as a bounded example when it reveals those relationships. Put version-specific setup and extended implementation detail in supporting material. Remove material whose only purpose is to report a current trend.

This does not prohibit concrete mechanisms. Autoregressive generation, paged inference state, adapter tuning, and group-based policy updates can earn their place. Their scope must be explicit so the argument remains useful when a different implementation becomes appropriate.

## 4. Book sequence and progressive disclosure

**Meet the complete machine → understand its subsystems → design the complete machine.**

| Position | Responsibility | Chapters |
|---|---|---|
| Introduction | Establish the task, execution loop, architectural map, workload contracts, and basic evaluation contract. | 1 |
| Part I: The Stochastic Processor | Invoke the learned core and organize bounded inference-time computation. | 2–3 |
| Part II: Memory and Storage | Select working information, manage derived inference state, and retain useful records. | 4–6 |
| Part III: Tools and I/O | Express actions, interpret observations, and enforce authority and isolation. | 7–8 |
| Part IV: The Agent Operating System | Recover execution, govern its lifecycle, and allocate resources. | 9–11 |
| Part V: Learning and Policy Adaptation | Turn experience into data and evaluate changes to learned behavior. | 12–14 |
| Part VI: System Integration and Operation | Coordinate agents, evaluate and operate the complete system, and optimize its cost. | 15–17 |
| Synthesis | Defend a complete architecture against workload requirements and evidence. | 18 |

Part V adds optional policy adaptation. Part VI studies integration and operation, including the decision to coordinate multiple agents. A useful deployment may use an existing model and a single agent. The revised Part VI title makes evaluation and economics visibly applicable to that system too; they are not subordinate topics of multi-agent execution. Their minimum concepts appear in Chapter 1 and deepen throughout the book before the full treatment in Chapters 16–17.

### Concept ownership and prerequisite map

| Concept | First encounter | Main teaching home | Later use and boundary |
|---|---|---|---|
| Goal, trajectory, observation, action, environment | 1 | 1 | 2–18 use the same vocabulary. Observation is only the information available to the agent. |
| Requirements, ambiguity, completion, autonomy | 1 | 1; supervision in 10 | 16 measures outcomes; 17 accounts for the resources needed; 18 synthesizes. |
| Baseline, repeated trials, development/evaluation separation | 1 | 16 develops experimental design | 3 and 12–15 use a small evaluation contract already taught. |
| Model invocation and output contract | 1 | 2 | 3 composes invocations; 7 interprets executable requests. |
| Instruction design and program control | 1–2 | 2–3 | 10 governs execution; no late prerequisite for elementary single-agent planning. |
| Planning, information gathering, feedback, stopping | 1 | 3 | 10 enforces lifecycle limits; 14 teaches learning credit assignment; 15 extends decomposition across workers. |
| Working context and compaction | 1–2 | 4 | 6 supplies retrieved records; 9 preserves durable execution state. |
| Derived inference state and cache validity | 2 names its role | 5 | 11 selects residency and placement policies across tasks. |
| Persistent knowledge, provenance, freshness | 1 names retention | 6 | 12 uses execution evidence as training data; learning changes weights rather than merely storing records. |
| Action schemas, observations, tool failure contracts | 1–2 names interface roles | 7 | 9 implements recovery for uncertain outcomes; 10 handles delayed results. |
| Untrusted content, permissions, isolation, data access | 1 names the action gate | 8 | 10 applies review and delegation; 12 and 16 apply retention/privacy boundaries. |
| Durable execution identity, pending effects, versions | 1 names task continuity | 9 | 10 extends lifecycle; 11 schedules records; 15 coordinates ownership. |
| Waiting, cancellation, supervision, changed instructions | 1 names intervention | 10 | 11 accounts for wait-state resources; 16 validates operational behavior. |
| Shared serving capacity, admission, fairness | 2 names contention | 11 | 14 and 17 apply the mechanisms without reteaching serving infrastructure. |
| Data lineage, coverage, feedback quality | 1 names evidence | 12 | 13–14 construct and evaluate policy updates. |
| Supervised adaptation and distribution shift | ML prerequisite | 13 in the trajectory setting | 14 contrasts outcome-driven learning. Adapter serving detail follows introduction of adapters. |
| Rewards, exploration, delayed credit, policy freshness | Learning purpose in 12 | 14 | 16 evaluates independently of the training reward. |
| Delegation, shared artifacts, coordination, correlated error | 1 names multiple workers | 15 | 16 traces failures; 17 measures useful speedup and cost. |
| Release decisions, regression monitoring, diagnosis | Basic checks in 1 | 16 | 17 optimizes from evidence; 18 applies release criteria. |
| Task cost, critical path, useful throughput | 1–2 | 17 | Earlier chapters compute local costs; 18 reuses those calculations. |

A forward mention may identify a responsibility without teaching its internals. It may not require an unexplained mechanism to solve the current chapter's problem. Names such as GRPO, distributed sagas, a specific sandbox, or a named protocol should appear only when the reader has the conceptual prerequisites to interpret them.


### Chapter navigation and proposed body-section counts

| Chapter | Main body sections |
|---|---|
| [1. The Stochastic Computer](#chapter-1-the-stochastic-computer) | 7 |
| [2. The Stochastic Processor](#chapter-2-the-stochastic-processor) | 6 |
| [3. Inference-Time Deliberation](#chapter-3-inference-time-deliberation) | 7 |
| [4. Context Management](#chapter-4-context-management) | 6 |
| [5. Inference Memory and Caching](#chapter-5-inference-memory-and-caching) | 7 |
| [6. Persistent Memory and Retrieval](#chapter-6-persistent-memory-and-retrieval) | 7 |
| [7. Tool Interfaces and Actions](#chapter-7-tool-interfaces-and-actions) | 6 |
| [8. Authority and Execution Isolation](#chapter-8-authority-and-execution-isolation) | 6 |
| [9. Durable Execution and Recovery](#chapter-9-durable-execution-and-recovery) | 7 |
| [10. Supervision and Runtime Control](#chapter-10-supervision-and-runtime-control) | 6 |
| [11. Scheduling and Resource Management](#chapter-11-scheduling-and-resource-management) | 6 |
| [12. Trajectory Data and Feedback](#chapter-12-trajectory-data-and-feedback) | 6 |
| [13. Learning from Agent Trajectories](#chapter-13-learning-from-agent-trajectories) | 6 |
| [14. Reinforcement Learning from Environment Feedback](#chapter-14-reinforcement-learning-from-environment-feedback) | 6 |
| [15. Multi-Agent Coordination](#chapter-15-multi-agent-coordination) | 6 |
| [16. Evaluation and Observability](#chapter-16-evaluation-and-observability) | 7 |
| [17. Performance and Cost Engineering](#chapter-17-performance-and-cost-engineering) | 6 |
| [18. Designing the Stochastic Computer](#chapter-18-designing-the-stochastic-computer) | 6 |

Counts exclude Purpose, learning objectives, fallacies, summary, chapter connection, and optional worked subsections. They describe the proposed argument and may change if a later draft reveals a genuinely missing competency or a redundant division.

## 5. Chapter plans

The section numbers below are proposed teaching positions, not replacement Quarto anchors. Preserve existing anchors where the concept survives and review dependent references when content moves. Chapter cards distinguish intended reader outcomes from implementation choices; the latter remain subject to evidence during rewriting.

### Chapter 1. The Stochastic Computer

**Current manuscript:** [01_introduction](01_introduction/01_introduction.qmd)

**Central point:** An agentic ML system coordinates learned decisions, state, actions, and execution management to complete a task; its architecture must be judged over the whole trajectory.

**Prerequisites and placement:** Start from the reader's ML systems background. Define the agent-specific unit of work before introducing the subsystems. The chapter establishes the vocabulary and minimal evaluation contract that all later chapters use.

**Lasting takeaways**

- **The task is the engineering boundary.** A model response is one event within a trajectory that also includes observations, effects, waiting, and completion evidence.
- **The architecture assigns responsibilities.** Processor, memory, I/O, and operating system provide a useful functional map; the implementation must establish the actual contracts between them.
- **Workload requirements determine the needed machinery.** Horizon, state, authority, and available checks reveal different constraints across otherwise similar-looking agents.
- **Capability requires task-level evidence.** A plausible output or completed request cannot establish every property of the requested result; comparisons need explicit criteria and resource accounting.

**Learning objectives and assessment alignment**

| Objective | Supporting sections | Evidence of learning |
|---|---|---|
| Trace a task through model calls, observations, actions, and completion decisions. | 1.1–1.2 | Annotated execution trace with task/environment boundaries. |
| Assign component responsibilities within a stochastic computer architecture. | 1.3, 1.7 | Architecture sketch identifying the owner of state, authority, and effects. |
| Compare workload requirements using horizon, state, authority, and completion evidence. | 1.4–1.5 | A comparison of two supplied task specifications. |
| Calculate task resource use from a simple execution record. | 1.6 | A checked accounting with explicit treatment of waiting and failed work. |
| Justify a fixed workflow or model-directed loop for a bounded task. | 1.6–1.7 | A design choice supported by requirements and a baseline. |

**Ordered body sections**

#### 1.1 A task carried through execution

Trace a small file-editing task from a delegated request through an initial inspection, an unsuccessful proposal, a revealing check, and a reviewable result. Name the deliverable and show why the intermediate response alone does not complete the task. Keep the example small enough that a student can trace every event; it establishes the execution loop without committing the rest of the book to this application.

**Contribution and transition:** Establish a concrete system to explain. The sequence exposes the need to distinguish the model's decision from what the environment actually did.

#### 1.2 Goals, observations, decisions, and effects

Define goal, trajectory, context, model invocation, action proposal, runtime dispatch, observation, and completion. Explain that observations give partial and sometimes stale information about the environment. Show that actions can change future inputs, and that an ambiguous goal may require clarification or a bounded assumption. Give a minimal execution-loop sketch with explicit continuation and stopping conditions.

**Contribution and transition:** Establish the unit and vocabulary without a full decision-process derivation. The loop now has identifiable responsibilities that can be assigned to components.

#### 1.3 The computer architecture

Map the learned core, working information, persistent storage, action/observation interfaces, and runtime management to processor, memory, I/O, and operating system. Locate the host OS and model-serving service beneath or beside the agent runtime. Introduce learning and distribution as available extensions. Explain the interfaces through the example; a model-generated proposal still passes through an execution boundary.

**Contribution and transition:** Give students the book-wide map. Different requirements put different demands on this same arrangement, motivating the task specification that follows.

#### 1.4 Tasks and their requirements

Specify a task through its goal, observations, permitted actions, environment, and completion criteria. Use short contrasts among executable checks, source-based judgment, and external acknowledgments to explain the uncertainty each leaves unresolved. Establish requirements without selecting recurring application archetypes or treating an example as permanently compute-bound, memory-bound, or fully verifiable.

**Contribution and transition:** Establish the requirements vocabulary used in later chapters. Different evidence and action boundaries make the engineering envelope concrete.

#### 1.5 The engineering envelope

Develop horizon as decision and execution duration, state as information and mutable resources, authority as permitted effects, and closure as evidence for specified properties. Use these as separate engineering questions whose answers can interact: read-only work can have weak completion evidence, and a short operation can carry substantial authority. Do not assert formal independence or orthogonality among them. Introduce deadlines, resource budgets, and human intervention as requirements with operational consequences. Identify affected users, unacceptable outcomes, and information-access limits; specify when ambiguity requires clarification, abstention, or escalation.

**Contribution and transition:** Convert the examples into design constraints. The next step is deciding whether a proposed agent satisfies those constraints.

#### 1.6 Success, cost, and appropriate autonomy

Compare a fixed workflow with a model-directed loop on a bounded version of the opening task. Establish a baseline, comparable task conditions, repeated trials, separate development and evaluation tasks, and an explicit acceptable outcome. Count failed attempts and human intervention. Use one small serial trace to account for model work, tools, and waiting; defer statistical methods and optimization models to their owners.

**Contribution and transition:** Establish the evaluation contract needed by every subsequent design comparison. Students can now specify a minimum complete system.

#### 1.7 A minimum complete stochastic computer

Assemble component responsibilities, input/output contracts, completion criteria, and a budget for the opening task. Change one observation requirement: the relevant source exceeds what one invocation can receive and arrives in bounded pieces. Preserve the goal, permitted actions, and completion criteria while selecting which pieces enter the current context, retaining persistent references to omitted material, defining an I/O interface for obtaining additional pieces, and budgeting the extra calls and processing. Close with the architectural tour: computation, information, interaction, execution management, adaptation, coordination, and whole-system assessment.

**Contribution and closure:** The chapter produces an assessable architecture and task contract. The processor is the first component whose invocation behavior and costs require deeper analysis.

**Quantitative work:** In 1.6, use a deliberately serial trace with disjoint, nonoverlapping duration totals: `T_task = T_model_total + T_tool_total + T_standalone_wait_total + T_runtime_total`. Each term totals all occurrences of that activity in the trace. Model duration includes input processing and generation. Tool duration includes the tool's I/O waits; standalone waiting covers only intervals not already counted in a model call, tool operation, or runtime work. Runtime duration covers the remaining active orchestration work. Concurrent events require a critical-path treatment later. Distinguish elapsed time and occupied compute from retained memory: a waiting task may retain memory while using no accelerator computation.

Use one explicitly hypothetical fixture: model work takes 10 s, tool execution takes 20 s, standalone waiting takes 5 s, and runtime work takes 5 s, giving `10 + 20 + 5 + 5 = 40 s`. Halving model duration, with the other durations and outcome requirements unchanged, gives `5 + 20 + 5 + 5 = 35 s`, or `35/40 = 7/8` of the original duration. Compute observed acceptable completions divided by attempted tasks and report the attempted-task count. Describe repeated trials without a claim of statistical certainty. All scenario values remain explicit inputs to the manuscript's calculation cells.

**Culminating assessment:** Given the opening execution trace and the changed observation requirement that relevant source exceeds one invocation and arrives in bounded pieces, submit an architecture sketch, task contract, simple serial resource account, and justified autonomy choice. Specify context selection, persistent source references, the I/O contract for obtaining more pieces, and the budget for additional work. Preserve the task's original goal, permitted actions, and completion criteria. The result must identify who decides completion and what the available evidence cannot establish.

**Coverage decisions:** Add explicit workload contracts, goal ambiguity, observation/state distinction, and early evaluation prerequisites. Consolidate the current evolution and tripartite-paradigm sections into the motivation for 1.1–1.3. Reuse the conceptual trajectory material from `sec-vol3-intro-formal-definition-of-an`. Replace the literal guarantees in `sec-vol3-intro-the-reliability-wall` with the bounded envelope in 1.5. Remove the current frontier survey from the introduction, and reduce `sec-vol3-intro-book-organization` to the functional architectural tour. Detailed cache, recovery, protocol, and learning mechanisms remain in their owning chapters.

**Visual plan:** Reuse the trajectory, loop, and computer-map compositions after correcting labels and simplifying premature mechanisms; use one compact workload comparison table. The figure audit gives asset-specific decisions. The current cover's base artwork is reusable with revised labeling.

### Part I — The Stochastic Processor

The first chapter establishes a usable invocation contract; the second organizes invocations and feedback into bounded decision procedures. Their intermediate state motivates memory management.

### Chapter 2. The Stochastic Processor

**Current manuscript:** [02_processor](02_processor/02_processor.qmd)

**Lasting takeaways: design backward from these results**

1. **The model interface defines its systems role.** The model generates responses from supplied information; the runtime determines how a response becomes further execution. A proposed operation, a permitted operation, and a successful operation are distinct.
2. **Reproducibility and correctness require different evidence.** Output can vary with inputs, model versions, generation settings, and execution conditions. Eliminating variation does not establish that the model solved the task.
3. **Output constraints provide bounded guarantees.** Constraining the generated format can prevent some malformed responses. It does not establish that values are appropriate, an action is authorized, or the task is complete.
4. **Invocation cost depends on the workload.** Input processing and sequential output generation consume time and memory differently. A caller should measure its actual requests before deciding which expense to reduce.

**Assessable learning objectives**

| Objective | Sections that teach and assess it |
|---|---|
| Trace how supplied context becomes a model response and then a runtime decision. | 2.1–2.2; culminating design in 2.6 |
| Specify an invocation contract covering configuration, completion, unacceptable outputs, and recorded evidence. | 2.3; 2.6 |
| Diagnose differences among output variability, malformed responses, and task failures. | 2.3–2.4; 2.6 |
| Calculate the latency contribution of input processing and output generation from an execution trace. | 2.5; calculation A |
| Select a generation configuration using interface validity, task outcomes, and resource measurements. | 2.4–2.6; calculation B |

**Governing question and teaching claim**

**Question:** What can a caller depend on when it uses a language model as the computer's computational core?

**Claim:** A model supplies computational capability through an interface whose outputs, limitations, variability, and costs must be understood by its caller.

**Entry prerequisites:** Chapter 1's task/trajectory vocabulary, full execution loop, distinction between observations and environment state, basic authority gate, and elementary evaluation contract: a baseline, repeated trials, fixed task criteria, and a separate development/evaluation task set. Ordinary systems interfaces and elementary probability are assumed; transformer internals, tool-protocol implementations, and KV paging are not.

**Exit capability:** Given a task, a model endpoint, and several candidate configurations, the student can specify the invocation and response-handling contract, diagnose what a failure tells them, and justify a configuration using measured results.

**Body sequence**

#### 2.1 The model's role in the computer

**Concrete opening:** A caller supplies observations to a model. Its response could request more evidence, propose an operation, or stop without a solution. Trace where each response goes, using one short request/response example.

**Subtopics:**

- Computational capability expressed as text, code, a requested operation, or an explicit inability to complete the request.
- The boundary between a model proposal and runtime dispatch.
- Supplied observations versus the current environment; the model does not directly possess an authoritative view of external state.

**Purpose:** Establish the model's useful functional role before inspecting generation. Apply Chapter 1's whole-system loop to one component; do not repeat the entire architectural tour.

**Why this comes first / handoff:** An identifiable request and response give the reader a reason to learn what occurs between them. End with the fact that generating a useful response requires interpreting the supplied context.

#### 2.2 From context to response

**Concrete opening:** Use a short input and the first few generated tokens to show which information is supplied once and which information grows during a response. No application backstory is needed.

**Subtopics:**

- Tokenization as a representation and accounting choice; token boundaries need not match words, code symbols, or tool fields.
- Learned next-token distributions conditioned on the available prefix; distinguish a likely continuation from a verified answer.
- Autoregressive generation, stop conditions, and the difference between processing an existing prefix and producing a continuation.
- Minimal transformer orientation: attention uses supplied context and prior tokens; reusable inference state avoids repeating some work, with its implementation deferred to Chapter 5.

**Purpose:** Provide the mechanistic foundation needed to reason about variable-length responses, context limits, and cost. One small conditional-generation expression is sufficient if it helps readers interpret the mechanism.

**Why it follows / handoff:** Having located the interface, explain how output is formed. The generation mechanism then motivates explicit limits and completion semantics in the invocation contract.

#### 2.3 The invocation contract

**Concrete opening:** Two calls return different patches; a third returns an incomplete patch because its output limit is reached. Determine what the caller must record and distinguish.

**Subtopics:**

- Input context, model/version identifier, generation configuration, requested output form, and resource limits. Specify instructions, examples, and response requirements as part of the caller contract; context selection is developed in Chapter 4.
- Completed output, incomplete output, refusal or unavailable capability, transport failure, and caller-visible status; keep vendor API vocabulary out of the durable explanation.
- Correctness, task capability, and reproducibility as separate properties; sampling variability and execution variation without universal hardware nondeterminism claims.
- Digital multimodal inputs where the endpoint supports them: preserve document/image identity, representation limits, and the distinction between supplied observations and inferred claims; observation payload design is developed in Chapter 7.
- Minimal invocation record for comparison and diagnosis: inputs, settings, returned output/status, versions, and measured timing. Logging/storage implementation belongs later.

**Purpose:** Turn generation into a usable systems boundary. Clarify that the model's probability for a continuation is not a calibrated probability that the proposed action will accomplish the task.

**Why it follows / handoff:** The caller now knows what generation does but needs to decide which outcomes can enter the next stage. End with the need to constrain and validate the output interface.

#### 2.4 Shaping and constraining outputs

**Concrete opening:** Compare an explanatory paragraph, a malformed edit request, a well-formed request for the wrong file, and a well-formed applicable edit.

**Subtopics:**

- Generation controls such as greedy selection, sampling, and output limits; compare observable behavior instead of equating temperature with creativity or reliability.
- Requested output schemas and the intuition of excluding invalid continuations during generation.
- Valid prefix versus successfully completed output; truncation, unsupported constraints, and caller-side validation.
- Syntax, field validity, environment preconditions, permission, and task evidence as different acceptance questions.

**Purpose:** Teach the strongest bounded guarantee a particular mechanism actually supplies. Keep a small before/after interface example; put automata construction and protocol detail in Chapter 7 or supporting material.

**Why it follows / handoff:** The contract identifies acceptable outcomes; these mechanisms improve the chance of obtaining one or enforce part of its structure. The remaining choice is what those controls cost and whether they improve useful output.

#### 2.5 The cost of an invocation

**Concrete opening:** A full test log and a filtered diagnostic produce different input lengths; a compact edit request and a long explanation produce different output lengths.

**Subtopics:**

- Input-processing time and sequential generation time, with time to first token separated from full-response completion.
- Existing context, output headroom, and derived inference-state memory as demands on the service.
- Queueing and concurrency as conditions under which latency measurements change; avoid a universal fixed token rate.
- What a caller can change: supplied content, requested output, model/configuration, and the need for measurement under representative requests.

**Purpose:** Establish enough cost reasoning to design an invocation. Chapter 4 owns information selection, Chapter 5 memory sizing, Chapter 11 scheduling, and Chapter 17 complete-system optimization.

**Why it follows / handoff:** Response restrictions and configurations have resource consequences. End by combining validity, task usefulness, and time in the final interface design.

#### 2.6 Designing and evaluating the processor interface

**Concrete opening:** Compare candidate invocation contracts for the same bounded set of caller requests, using their returned outputs and status records.

**Subtopics:**

- Select the context, response form, generation limits, and caller-visible completion conditions.
- Specify treatment of malformed, unavailable, incomplete, and valid-but-unsuccessful outputs.
- Compare configurations on the same held-out tasks and repeat trials under recorded conditions.
- Identify what remains unresolved after one valid invocation and whether additional evidence or computation might help.

**Purpose:** Resolve the opening problem with a concrete contract and measured justification. This is synthesis of the chapter, not another mechanism catalog.

**Why it follows / handoff:** All inputs to an interface decision are now available. The next chapter begins where a single invocation's computational effort is insufficient.

**How the argument closes**

The completed artifact is a processor-interface specification, a small acceptance decision table, and a justified configuration choice. The chapter has succeeded if the student can explain why a valid generated request is only one stage of completing a task. One optional transfer question can contrast a well-formed edit with a well-formed citation record: neither establishes its intended outcome by syntax alone. Approval and ambiguous remote completion are taught later; no three-workload tour is needed here.

**Calculation designs**

**A. Invocation latency and the value of a shorter response — section 2.5.**

- Variables: measured queue time `Tq`, input-processing time `Tp`, per-token generation durations `d1…dN`, and response validation time `Tv`.
- Relationship: `Tcall = Tq + Tp + sum(di) + Tv`. A constant per-token duration may be used only as a stated approximation over the specific measured workload.
- Inputs: a trace or explicitly labeled pedagogical timings, obtained through the book's calculation infrastructure. No invented hardware throughput or provider price.
- Assumptions: define whether timings overlap and which phases the endpoint exposes. If only total latency is available, do not infer a precise internal decomposition.
- Decision: determine whether reducing input, reducing output, or changing queue conditions can materially improve the caller's deadline. Show sensitivity to generation length without asserting that shorter responses always preserve task quality.

**B. Selecting an invocation configuration — sections 2.4–2.6.**

- Variables: tasks `M`, repeats per task `R`, counts of complete and syntactically valid responses, counts satisfying task checks, total latency, and generated-token counts for each configuration.
- Calculations: report valid-response fraction and acceptable-task fraction with clearly stated denominators; compare median and tail timing from supplied observations without fitting an unnecessary model.
- Assumptions: task criteria and model version are held fixed; configuration tuning uses a separate development set; failed and incomplete calls remain in the denominator.
- Decision: select a configuration that satisfies the interface requirements and resource envelope. More repeatable or more often valid output does not automatically win if task outcomes deteriorate.

**Culminating assessed problem**

Provide a bounded editing task suite, tool-request schema, caller deadline, and execution results for three generation configurations. Include an incomplete response, a malformed request, a permitted edit to the wrong location, and a plausible edit that fails the relevant test. Ask the student to:

1. Specify the invocation contract and response-handling decision table (LO1–LO3).
2. Compute latency contributions from the supplied traces and identify the largest actionable term (LO4).
3. Select a configuration using both interface and task evidence, name an unresolved uncertainty, and specify one additional comparison (LO5).

The answer is assessed by consistency of the contract, correct denominators, appropriate use of evidence, and a defensible decision. Do not award full credit merely for choosing the smallest model or shortest output.

**Missing concepts / additions**

- **Explicit completion status and output truncation.** The current grammar discussion overstates validity without making successful termination central.
- **Probability of text versus evidence of correctness.** This is more durable than the current path-entropy treatment.
- **Controlled variability and deterministic-but-wrong behavior.** The current model contract incorrectly makes pervasive hardware nondeterminism its foundation.
- **Model access limits.** The model reasons from supplied observations and learned parameters; fresh or missing information must reach it through the system.
- **A single worked contract that closes the chapter.** The current summary recaps mechanisms without demonstrating that students can specify the caller's interface.

**Consolidation and removal from current sources**

| Current source anchor | Editorial action |
|---|---|
| `sec-vol3-processor-intro` | Replace four-stage IF/ID/EX/WB with the proposal-to-runtime boundary in 2.1; remove ACB program-counter terminology. |
| `sec-vol3-processor-analogy` | Keep short autoregressive explanation for 2.2 and input/output phase distinction for 2.5. Move KV geometry to Chapter 5, ACB/state lifecycle to Chapters 9–10, and offload policy to Chapter 11. Put necessary transformer foundations in a compact optional treatment. |
| `sec-vol3-processor-contract` | Rebuild as 2.3. Consolidate sampling into 2.4; remove enormous path-space numbers and impossible-replay claims. A change to a common positive softmax denominator cannot alone reverse its argmax ordering; do not migrate that erroneous explanation. |
| `sec-vol3-processor-grammar-constrained` | Retain mechanism and bounded contract in 2.4. Move grammar/parser implementation to Chapter 7/supporting material. Remove unsupported latency numbers and claims that every semantic restriction lies outside the grammar's expressiveness. |
| `sec-vol3-processor-isolation-boundary` | Move actual threat/enforcement discussion to Chapter 8. Remove prompt-injection = W⊕X equivalence rather than relocate it. Keep only the high-level caller authority boundary in 2.1/2.4. |
| `sec-vol3-processor-speculative-decoding` | Move a bounded optional generation-optimization example to Chapter 17. Distinguish preserving a target decoding distribution from checking task correctness. |
| `sec-vol3-processor-fallacies`; `sec-vol3-processor-summary` | Rebuild against the four takeaways. Do not retain attention-intensity derivations as the principal end-of-chapter assessment. |

### Chapter 3. Inference-Time Deliberation

**Current manuscript:** [03_deliberation](03_deliberation/03_deliberation.qmd)

**Lasting takeaways: design backward from these results**

1. **More computation needs a useful destination.** Additional generation, alternatives, and feedback can improve a decision; repeated elaboration alone provides no assurance of progress.
2. **Selection is part of the algorithm.** Useful candidates are insufficient when the selection procedure cannot distinguish them. Every checking mechanism has a scope, cost, and possible errors.
3. **Plans remain revisable.** A plan records intended work and dependencies under current information. Feedback can require changing the plan, acquiring evidence, or abandoning an approach.
4. **Search trades alternatives against resources.** Breadth and depth consume work, elapsed time, and retained state. Pruning saves resources but may remove a useful candidate.
5. **Deliberation is justified by outcomes.** A strategy earns its budget by improving task results under the same evaluation conditions, including failed attempts, assessment cost, and stopping behavior.

**Assessable learning objectives**

| Objective | Sections that teach and assess it |
|---|---|
| Select among additional generation, candidate sampling, and feedback-driven revision for an observed capability gap. | 3.1–3.2; 3.7 |
| Evaluate a candidate-selection procedure using its evidence, errors, and assessment cost. | 3.3; calculation B; 3.7 |
| Construct and revise a plan from task dependencies and new observations. | 3.4; 3.7 |
| Compare bounded search procedures by retained alternatives, pruning errors, work, and elapsed time. | 3.5–3.6; calculation A |
| Design and assess a deliberation strategy under explicit task and resource constraints. | 3.6–3.7; culminating problem |

**Governing question and teaching claim**

**Question:** When should the computer spend more computation on a decision, and how should it organize that work?

**Claim:** Additional inference-time computation helps when it produces useful alternatives or feedback at a cost justified by improved task outcomes.

**Entry prerequisites:** Chapter 2's model-invocation and acceptance contract; Chapter 1's success criteria, baseline and elementary repeated-comparison discipline; distinction between a proposal and an executed external action. No RL, process-reward-model training, environment virtualization, KV pages, multi-agent control, or distributed routing prerequisite.

**Exit capability:** The student can choose an evidence-aware, bounded single-agent deliberation strategy, explain its state requirements, and test whether its benefit warrants the resources spent.

**Body sequence**

#### 3.1 When one response is insufficient

**Concrete opening:** A failure has two plausible diagnoses. An initial patch fails one test, and the remaining evidence does not identify the cause.

**Subtopics:**

- Missing evidence, insufficient reasoning, incorrect assumptions, and inadequate tools as different reasons a response fails.
- Difference between producing more text and performing useful additional work.
- A baseline response with explicit task and resource criteria.

**Purpose:** Establish the need for a deliberation decision without claiming that greedy or single-path generation inevitably fails.

**Why it comes first / handoff:** Chapter 2 supplies a valid computational interface; this section identifies the unresolved task-level problem. End by asking what additional work could discriminate between the candidate explanations.

#### 3.2 Ways to spend additional computation

**Concrete opening:** On the same repair, compare a longer attempt, several candidate patches, and one attempt revised after a focused test.

**Subtopics:**

- Extended generation within an invocation, with intermediate work only where the interface exposes it.
- Multiple candidate responses; diversity of useful approaches versus repeated variants of the same mistake.
- Revision from external feedback; distinguish new observations from the model's restatement of its own confidence.
- Distinction between additional task computation, training model weights, and generation acceleration such as speculative decoding.

**Purpose:** Give students the small set of durable ways to allocate inference effort before introducing advanced search names.

**Why it follows / handoff:** Each strategy addresses a different gap in the baseline. Candidate generation and revision immediately raise the question of what evidence should guide them.

#### 3.3 Feedback and candidate selection

**Concrete opening:** Two patches pass the supplied test, but one deletes the tested behavior; an additional requirement distinguishes them. A short contrast with a fluent but unsupported summary can expose another limitation of candidate assessment.

**Subtopics:**

- Executable checks, learned scoring, and human assessment: what property each assesses and which assumptions it needs.
- Outcome assessment versus intermediate progress feedback; introduce learned intermediate assessors as one option without teaching their training objective.
- False acceptance, false rejection, incomplete specifications, and correlated generator/assessor errors.
- Search against an imperfect evaluator: increased opportunity to find its blind spots; independently held-out checks and refusal to overinterpret a score.

**Purpose:** Make assessment part of the system design. Correctness is established relative to task requirements and evidence, not by declaring a software oracle immune to exploitation.

**Why it follows / handoff:** We now have ways to generate and assess work. The next step organizes these decisions into a plan that can change when feedback arrives.

#### 3.4 Planning and revising work

**Concrete opening:** A repair requires reproducing the failure, distinguishing two causes, changing the relevant code, and testing for regressions. A newly discovered dependency changes that order.

**Subtopics:**

- A plan as an explicit representation of subgoals, dependencies, available evidence, and completion conditions.
- Single-agent task decomposition and the distinction between independent work and work blocked by a missing observation.
- Progress tracking, assumption revision, and replanning when an approach or precondition fails.
- Deciding to obtain evidence, stop, or escalate; do not substitute textual plan completion for task completion.

**Purpose:** Teach basic planning on its own terms before making it a graph-search or multi-agent problem.

**Why it follows / handoff:** Feedback has meaning only relative to the desired result and current approach. Once several plausible plans remain, the system needs an explicit policy for retaining and exploring alternatives.

#### 3.5 Bounded search over alternatives

**Concrete opening:** Retain two repair approaches while a discriminating test runs; show the consequence of discarding one too soon.

**Subtopics:**

- Generate-and-select, iterative revision, and a bounded branching frontier, with one simple worked search trace.
- Candidate representation: proposed action/patch, relevant evidence, accumulated cost, and status. A search state is an information record, not automatically a complete world snapshot.
- Expansion, candidate evaluation, pruning, and revisiting earlier choices; imperfect feedback may eliminate the eventual solution.
- Proposed changes, locally tested changes, and committed external effects. Use the interface abstraction established earlier; defer sandbox and recovery implementations.

**Purpose:** Explain the systems choices made by search without requiring PUCT, tree-cache management, or a catalog of named prompting methods.

**Why it follows / handoff:** Plans make alternatives concrete. Maintaining those alternatives exposes costs that require an explicit budget. MCTS may appear as one advanced supporting example after the main argument, not as its default implementation.

#### 3.6 Allocating and stopping deliberation

**Concrete opening:** The repair budget permits either several quick candidates, one deeper revision, or an expensive discriminating test. Compare what evidence each buys.

**Subtopics:**

- Breadth, depth, and feedback expense; use separate accounting for model work and environment checks.
- Total work versus elapsed time, including limited concurrency and waits, without teaching the scheduler implementation.
- Diminishing improvement, correlated candidates, stopping after sufficient evidence, and a hard external resource limit.
- Budget assignment based on observed task difficulty or failed attempts, followed by measured comparison with a simple baseline.

**Purpose:** Turn search from an algorithm diagram into a resource allocation decision. Budget enforcement is an OS responsibility taught in Chapter 10; here the student defines the needed policy.

**Why it follows / handoff:** The alternatives and their evidence are known; the system can now choose how much effort to invest. The closing design tests whether that choice improved useful outcomes.

#### 3.7 Designing a deliberation strategy

**Concrete opening:** Complete the opening repair using two competing strategies under the same limits and task checks.

**Subtopics:**

- Select a plan, candidate strategy, assessor, and stopping rule.
- Record all attempts and feedback, including rejected candidates and failure to find an acceptable result.
- Compare task outcomes, work, elapsed time, and assessment overhead on independent evaluation tasks.
- List the information that must survive between decisions, while distinguishing useful evidence from discardable intermediate material.

**Purpose:** Close with a justified single-agent deliberation design and identify its information requirements.

**Why it follows / handoff:** It integrates every preceding choice and creates the specific question for Chapter 4: which information must be retained in the next decision's context?

**How the argument closes**

The student leaves with a deliberation policy, a revised plan, a candidate-selection trace, and a measured comparison against one-pass execution. The student must also identify a task for which the policy is unnecessary. The explanation distinguishes incomplete checks from complete task evidence and imagined alternatives from committed external effects; permission and remote recovery remain later topics.

**Calculation designs**

**A. Choosing work under a deliberation budget — sections 3.5–3.6.**

- Variables: number of candidates `N`, generation work `G_i`, checking work `V_i`, generation duration `g_i`, checking duration `v_i`, maximum concurrent candidates `K`, and resource/time limits `B` and `D`.
- Accounting: `W = sum(G_i + V_i)` when work has a common defined unit. If model and CPU work do not share a meaningful unit, report them separately or convert using explicitly supplied resource prices. Do not add unlike FLOPs or treat time and currency as interchangeable.
- Elapsed time: for a serial design sum its stages; for a fully parallel independent stage use the maximum duration only if capacity and independence assumptions permit it. Otherwise give a small explicit execution schedule.
- Decision: compare wider candidate generation with deeper sequential revision and an additional test. Include failed candidates and checking expense. The winning design must also meet the same outcome criteria.

**B. Does the selection procedure find the useful candidate? — sections 3.3 and 3.7.**

- Inputs: small task-by-candidate tables containing assessor scores, selected candidates, independently checked task outcomes, and generation/checking cost.
- Variables: task count `M`, candidate budget `N`, candidate correctness labels `y_ij`, scores `s_ij`, selected candidate index `j*`, and total task resource expense.
- Calculations: distinguish fraction of tasks with at least one acceptable generated candidate from fraction whose selected candidate is acceptable; count false acceptance/rejection using a stated definition and report cost per acceptable completion.
- Assumptions: labels come from a specified check with its limitations; use independent evaluation cases and repeated runs. Do not assume candidates are independent merely because generation calls were separate.
- Decision: determine whether more candidates, better assessment, or improved generation is the binding need. A simple plot/table of observed outcomes at successive budgets may demonstrate diminishing returns without fitting a universal power law.

**Culminating assessed problem**

Give students a diagnostic task with two plausible causes, a dependency between subgoals, a bounded set of tool observations, several candidate corrections, incomplete checks, and a fixed resource envelope. Supply traces for three strategies: a longer single attempt, generate-and-select, and feedback-driven revision with a small frontier.

Require a plan and its revision after new evidence (LO3), an explanation of what the candidate assessor can and cannot establish (LO2), a resource comparison (LO4), and a final strategy/stopping choice justified against the baseline (LO1 and LO5). Include a transfer question in which no binary checker fully specifies success. Full credit requires recognizing a checking limitation and defining the evidence needed before accepting the result.

**Missing concepts / additions**

- **Basic single-agent planning before named search algorithms.** Explicit subgoals, dependencies, information gaps, and plan revision are student necessities currently buried under topologies.
- **Self-generated confidence versus new evidence.** Extra reflection and a new environment observation carry different information; neither should be equated automatically with verified progress.
- **Quality of selection, not only candidate generation.** Distinguish producing an acceptable candidate from selecting it and presenting it as the result.
- **Correlated alternatives and checks.** Separate runs can repeat the same blind spot; more search is not an independent-trial guarantee.
- **An explicit boundary between search state and world state.** Candidate backtracking does not undo arbitrary external effects.
- **Recorded evidence rather than assumed insight into internals.** Do not claim that a generated rationale faithfully exposes every internal computation or that unavailable intermediate tokens can be inspected.

**Consolidation and removal from current sources**

| Current source anchor | Editorial action |
|---|---|
| `sec-vol3-deliberation-intro` | Replace the mandatory internal CPU-speculation engine with the task-level decision in 3.1. |
| `sec-vol3-deliberation-limits-greedy` | Retain a concrete early-mistake example and the distinction between local probability and task success. Remove inevitability of irrecoverable error, tokenwise semantic reliability claims, and cached-vector explanations of inevitable rationalization. Put teacher forcing/distribution shift in Chapter 13. |
| `sec-vol3-deliberation-scaling-laws` | Keep resource-versus-outcome comparison in 3.6/3.7. Remove universal inference scaling invariants, invented benchmark inversions, and the claim that each thinking token is a virtual transformer layer. |
| `sec-vol3-deliberation-topologies` | Consolidate single-agent approaches into 3.2, 3.4, and 3.5. Move basic autonomy selection to Chapter 1 and multi-agent designs to Chapter 15. Retire the universal anti-swarm prescription. |
| `sec-vol3-deliberation-prm-vs-orm` | Use feedback/assessment distinctions in 3.3. Move training credit assignment to Chapter 14 and transactional/saga implementation to Chapters 8–9. Remove deterministic-oracle immunity and the identification of false-positive incidence with exploit selection probability. |
| `sec-vol3-deliberation-tree-search` | Keep one simple bounded-search trace and cost accounting. Put detailed MCTS/PUCT in optional supporting material. Remove radix page-table machinery and automatic asymptotic guarantees for an arbitrary agent environment. Do not transplant the checkpoint square-root result without matching its assumptions to the actual model. |
| `sec-vol3-deliberation-compaction` | Move information compaction to Chapter 4, cache sharing to Chapter 5, watchdog/cycle detection to Chapter 10, and distributed routing to Chapter 11. End 3.7 by identifying the need for retained information rather than implementing all those systems. |
| `sec-vol3-deliberation-fallacies`; `sec-vol3-deliberation-summary` | Rebuild against the five lasting takeaways and the final decision. Replace 'search makes a resilient self-correcting problem solver' with the conditions established by the evaluation. |

### Part II — Memory and Storage

Context selection, derived inference state, and persistent records answer distinct questions about information and computation. The part closes with a complete write, retrieve, update, and use policy.

### Chapter 4. Context Management

**Current manuscript:** [04_working_sets](04_working_sets/04_working_sets.qmd)

**Question, claim, entry, and exit**

**Question:** How can an agent retain the information needed to finish a task as observations and intermediate work accumulate?

**Teaching claim:** Context is a selected representation of the task; its usefulness depends on what the system preserves, omits, transforms, and later verifies.

**Entry prerequisites:** A model invocation accepts a finite input representation and produces a response; a trajectory contains observations and proposed or executed actions; task checks provide bounded evidence; more deliberation can create more intermediate work. No attention-head specialization, paging, or vector-index knowledge is required.

**Exit capability:** Given a task history and a finite context contract, the student can specify and test a context policy, identify its information-loss risks, and explain how the next invocation reconstructs task continuity.

**Lasting takeaways**

1. **The next decision determines useful context.** A complete interaction history and an effective working representation serve different purposes; the system should preserve access to relevant originals while selecting what the model receives now.
2. **Compaction changes the information available.** Filtering, excerpts, structural representations, and generated summaries have different failure modes. Reducing tokens does not establish that task-critical information survived.
3. **Provenance preserves the meaning of evidence.** The model should be able to distinguish instructions, observations, hypotheses, verified results, and unresolved questions after a context transformation.
4. **Context policy is part of system behavior.** Compare policies through completed tasks, missed constraints, and additional work as well as token count and latency. Preserving instructions in a prompt does not enforce them.

**Learning objectives**

| ID | Assessable objective | Sections |
|---|---|---|
| 4-LO1 | Classify task information by its role, source, and relevance to the next decision. | 4.1, 4.3 |
| 4-LO2 | Calculate an invocation's input budget while reserving the output and protocol space required by its contract. | 4.2 |
| 4-LO3 | Select context representations that preserve needed evidence within a bounded input budget. | 4.3, 4.4 |
| 4-LO4 | Diagnose loss of constraints, provenance, or task continuity across successive compaction cycles. | 4.4, 4.5 |
| 4-LO5 | Evaluate competing context policies on comparable tasks and justify a design using outcome and resource evidence. | 4.6 |

**Ordered body sections**

#### 4.1 Information for the next decision

**Subtopics**

- Start with a decision whose needed observation is absent from a long history. Classify the missing evidence, active constraint, and irrelevant material before proposing a context policy.
- Distinguish instructions, task goals, observed environment information, intermediate hypotheses, and evidence of completed work.
- Distinguish the full history from the selected working context; introduce external artifacts as sources that remain available without teaching their retrieval implementation.
- Establish that the model's observation of an environment may be incomplete or stale.

**Purpose:** Give context management an information problem before a memory-budget problem. Students should first identify what would make the next decision sound.

**Why first:** It uses the execution loop established in Part I and supplies the semantic categories needed by every later section. Avoid opening with another tour of the entire computer.

**Introduction/application:** First explicit context-role taxonomy; applies the earlier trajectory and evidence distinctions.

#### 4.2 The context budget

**Subtopics**

- Account for instructions, tool descriptions, selected observations, active task state, and representation overhead.
- Reserve output or reasoning capacity according to the actual invocation contract; distinguish model limits from an application's chosen operating budget.
- Explain why nominal capacity and successful use are different, using a controlled example with a needed detail placed among distractors.
- Compare growth under append-only history with bounded selection without deriving transformer internals.

**Purpose:** Establish the constraint that forces selection and show that capacity has both interface and workload consequences.

**Why follows:** Students now know what the information is for; they can decide where a finite budget should go.

**Introduction/application:** First context-budget calculation. Invocation cost is applied from Chapter 2; physical KV bytes remain Chapter 5 material.

#### 4.3 Constructing the working context

**Subtopics**

- Select source material according to the current subtask, dependencies, and missing evidence.
- Order and delimit instructions, source evidence, observations, and active task records so their roles remain explicit.
- Bound observations at their source when possible; retain omission notices and paths or identifiers for full artifacts.
- Refresh changed information instead of accumulating contradictory snapshots; do not mistake retrieved content for a new instruction.

**Purpose:** Teach the first practical design alternative to indiscriminately appending history.

**Why follows:** Budgeting alone does not tell the runtime what to include. This section provides a policy for allocating that budget before any lossy summarization becomes necessary.

**Introduction/application:** First assembly policy. Retrieval mechanisms are deferred to Chapter 6; observation schema and authority enforcement to Chapters 7–8.

#### 4.4 Compaction and continuity

**Subtopics**

- Compare omission, extractive selection, structural representations, and generated summaries by what information each changes.
- Show a compact task record containing the goal, constraints, completed work with evidence, rejected hypotheses, outstanding questions, and artifact references.
- Distinguish disposable intermediate work from intermediate results that later decisions still require; honor the model invocation contract for any opaque continuation state.
- Choose a compaction trigger with headroom for the transformation itself and the next useful observation.

**Purpose:** Explain how a task continues beyond a single fixed working representation.

**Why follows:** Assembly can limit incoming material, but a long task still accumulates useful history. Compaction must preserve continuity after selection alone becomes insufficient.

**Introduction/application:** First semantic compaction lifecycle. This is not a transaction protocol and does not mandate deleting all intermediate reasoning.

#### 4.5 Preserving constraints and evidence

**Subtopics**

- Keep original task requirements and relevant source evidence distinguishable from paraphrases and model-generated assertions.
- Preserve provenance, revision identifiers, uncertainty, and verification status in the compacted representation.
- Check summaries against explicit requirements and sampled originals; recognize the limits of automated fidelity checks.
- Repair a failed compaction by retrieving originals, revising the task record, or escalating when needed information is unavailable.

**Purpose:** Establish the correctness obligations of a context transformation and the recovery path when those obligations are not met.

**Why follows:** Students have seen how representations shrink; they can now inspect what that transformation must preserve and what cannot be guaranteed.

**Introduction/application:** First detailed compaction-fidelity evaluation; applies evidence boundaries from Chapter 1. Runtime enforcement is explicitly reserved for Chapters 8–10.

#### 4.6 Evaluating a context policy

**Subtopics**

- Compare append-only history, bounded excerpts, and a compact task record on the same supplied task sequence.
- Introduce successive compaction cycles and a later need for an earlier negative constraint or rejected hypothesis.
- Measure task outcome, violated constraints, retrieval/reinspection work, context size, and latency under comparable conditions.
- Test whether citations, source disagreement, and unresolved uncertainty survive compaction as well as discrete tool results do.

**Purpose:** Close the chapter by requiring a justified policy rather than a technique list.

**Why last:** Evaluation needs the categories, budget, alternatives, and failure modes developed above. The chapter can now answer its opening question with a complete design.

**Introduction/application:** Applies the book's initial evaluation contract; does not introduce advanced experiment design reserved for Chapter 16.

**Argument closure and culminating assessment**

Apply the policy to a supplied history after several rounds of work and a compaction event. The student delivers: (1) a context-role inventory; (2) a budget; (3) an assembly and compaction policy; (4) a compact task record linked to originals; and (5) a comparison against a baseline. A deliberately omitted interface constraint and a contradicted hypothesis test whether the policy preserves the information needed to finish. The explanation must identify which properties are checked and which remain empirical. This assesses 4-LO1 through 4-LO5. Chapter 5 can reuse the history's token counts in a self-contained memory calculation without continuing the application's narrative.

**Calculation designs**

1. **Context budget and trigger.** Variables: invocation capacity `C`, reserved output/continuation allowance `R`, fixed input `F`, active task record `S`, selected evidence `E`, observation allowance `O`, and chosen margin `M`, all in the model's accounting units. Assumption: explicitly state whether the endpoint shares one capacity between input and output. Evaluate feasibility and the remaining allowance rather than asserting a universal compaction percentage. Decision: which representation or observation must change before the next call?
2. **Policy comparison over a task history.** Variables: per-turn input length, task success indicator, constraint violations, repeated inspections, and measured latency under each policy. Hold model/configuration and task conditions comparable. Decision: choose a policy that meets the task's quality requirements at acceptable cost; do not infer quality from compression ratio.

**Substantive additions and consolidation ledger**

| Need | Existing location | Disposition and reason |
|---|---|---|
| Context roles and provenance as the starting model | `sec-vol3-workingsets-intro`, `sec-vol3-workingsets-theory` | Rebuild into 4.1. The current introduction starts from physical activation retention, so the student's semantic design decision arrives too late. |
| Explicit context contract and headroom | `sec-vol3-workingsets-theory`, `sec-vol3-workingsets-budgeting` | Add application-level accounting in 4.2 and transformation headroom in 4.4. Avoid an unexplained fixed percentage threshold. |
| Source freshness, omissions, and artifact references | `sec-vol3-workingsets-budgeting` | Develop in 4.3 and 4.5. Current examples contain useful logs/diffs but overstate what deterministic extraction preserves. |
| Opaque model continuation state | Current scratchpad subsections in `sec-vol3-workingsets-budgeting` | Add a bounded interface lesson: runtime policies must respect the model contract and cannot assume every intermediate state is visible text that can be removed safely. Do not teach product-specific APIs. |
| KV storage and bandwidth | `sec-vol3-workingsets-theory` | Move to Chapter 5. Keep only the conceptual resource motivation needed by 4.2. |
| Attention entropy and RoPE causation | `sec-vol3-workingsets-density` | Reduce to empirically tested useful-context limitations; demote attention-mechanism derivations. An attention-weight statistic is not a general proof that semantic information can be discarded. |
| Token-cache eviction and model architecture | `sec-vol3-workingsets-density`, `sec-vol3-workingsets-pruning` | Move selected mechanisms to 5.6; demote pruning sensitivity and MLA associativity to supporting material. |
| Scratchpad lifecycle | `sec-vol3-workingsets-budgeting` | Consolidate into 4.4; retire two-phase-commit naming, mandatory purge claims, guaranteed zero information loss, and universal prefix-hit guarantees. |
| Compaction fidelity | `sec-vol3-workingsets-budgeting` | Retain source-versus-summary distinction; replace physical “invariant enclosure” and universal semantic-decay claims with 4.5's explicit obligations and test design. |
| Actual policy evaluation | No substantial dedicated section | Add 4.6 and an assessable end-to-end problem; it is the evidence needed to conclude the chapter. |

### Chapter 5. Inference Memory and Caching

**Current manuscript:** [05_virtual_memory](05_virtual_memory/05_virtual_memory.qmd)

**Question, claim, entry, and exit**

**Question:** How can the serving system retain and reuse inference state without exhausting the memory needed for useful concurrent work?

**Teaching claim:** Reusable inference state saves computation when its validity, allocation, and lifetime fit the workload's memory budget.

**Entry prerequisites:** Model invocation and autoregressive generation; prefill/decode roles; the context policy from Chapter 4; branching computation from Chapter 3. Basic memory allocation and bandwidth are part of the stated systems prerequisites, but KV representation and allocation are taught here.

**Exit capability:** Given a model configuration, prompt histories, memory budget, and measured transfer/recomputation costs, the student can estimate capacity and justify an inference-state policy. The student can separate policies that preserve the model computation from approximations requiring quality evaluation.

**Lasting takeaways**

1. **Inference caches are derived execution state.** A prompt and its provenance describe the information supplied; KV tensors store model-dependent work that may be regenerated under a compatible configuration.
2. **Memory demand depends on the workload.** Context length, model geometry, precision, shared prefixes, concurrent requests, and non-cache allocations jointly determine capacity.
3. **Allocation and reuse solve different problems.** Blocks reduce stranded allocation; prefix caching avoids repeating compatible computation. Either can be valuable without implying the other is sufficient.
4. **Cache lifetime is a resource trade-off.** Retention spends scarce memory, offload spends transfer capacity, and recomputation spends inference work. The best choice depends on reuse and competing demand.
5. **Approximation needs outcome evidence.** Reducing the representation can change model behavior; memory savings alone do not establish an acceptable system improvement.

**Learning objectives**

| ID | Assessable objective | Sections |
|---|---|---|
| 5-LO1 | Distinguish logical context, derived inference state, and durable records through a multi-turn execution. | 5.1 |
| 5-LO2 | Calculate cache demand and concurrent capacity under explicit model, workload, and allocation assumptions. | 5.2, 5.3 |
| 5-LO3 | Trace valid prefix sharing and identify changes that invalidate reusable inference state. | 5.4 |
| 5-LO4 | Compare retention, offload, and recomputation using memory occupancy and measured execution costs. | 5.5 |
| 5-LO5 | Evaluate an approximate memory representation against resource savings and task-quality requirements. | 5.6, 5.7 |
| 5-LO6 | Design an inference-memory policy that satisfies a specified workload and capacity budget. | 5.7 |

**Ordered body sections**

#### 5.1 Why inference retains state

**Subtopics**

- Follow one generated token and the past representations it uses; establish the KV cache's role without re-deriving all transformer operations.
- Distinguish logical prompt text, tokenized model input, derived KV tensors, and durable task records.
- Explain the difference between reusing a computed representation and retrieving source information.
- Follow what can be reconstructed after cache eviction, assuming required inputs and compatible configuration remain available.

**Purpose:** Establish the exact object the chapter manages and prevent the L1/L2/L3 category error.

**Why first:** Chapter 4 has specified the information; this chapter now explains a representation of computation over it.

#### 5.2 Sizing memory demand

**Subtopics**

- Explain bytes per cached token using layer count, KV dimensions, and representation precision.
- Account separately for weights, cache, temporary workspaces, allocator metadata, and other reserved memory.
- Compare heterogeneous active context lengths and concurrent requests; state sharding assumptions when using multiple accelerators.
- Relate memory capacity and memory traffic without treating a capacity estimate as a latency prediction.

**Purpose:** Give students the minimum model needed to expose the limiting resource.

**Why follows:** The managed state is now defined; its cost determines why allocation, sharing, and lifetime matter.

#### 5.3 Allocating growing sequences

**Subtopics**

- Compare maximum-length reservation with incremental allocation for requests whose eventual length is unknown.
- Show logical token blocks, a block table, and physical cache blocks with one small trace.
- Explain tail waste, free-block reuse, metadata overhead, and the block-size trade-off.
- Distinguish allocation-pool fragmentation from actual hardware virtual-memory/address-translation semantics.

**Purpose:** Teach paging as a useful implementation mechanism, not a claim that the agent recreates a hardware memory hierarchy.

**Why follows:** A nominal cache-size estimate can still overstate usable capacity when allocation strands memory.

#### 5.4 Reusing prefixes and sharing branches

**Subtopics**

- Locate the common prefix across the prompt histories generated by Chapter 4 and across alternative candidate branches.
- Establish validity conditions: matching relevant model/configuration, token sequence, position-dependent state, and permitted reuse scope; include non-text inputs when the invocation contract has them.
- Explain block sharing, private suffixes, and ownership/reference tracking through a trace rather than a full production trie implementation.
- Show the consequences of early prompt edits, changed tools or instructions, model/configuration updates, and compaction; explain that a cache miss changes cost rather than deleting the underlying source record.

**Purpose:** Teach the validity contract that makes reuse safe and useful.

**Why follows:** Shared representations need the incremental allocation and ownership model established in 5.3.

#### 5.5 Retaining, offloading, or recomputing state

**Subtopics**

- Distinguish state actively required by a request from inactive reusable state; separate releasing ownership from immediately deleting a reusable cache entry.
- Compare retaining in accelerator memory, transferring to another storage location, and recreating state from preserved inputs.
- Account for transfer latency, restore latency, reuse likelihood, competing memory demand, and unsuccessful prefetch.
- Explain cache pressure and a simple recency/cost-aware decision; reserve system-wide scheduling and placement policy for Chapter 11.

**Purpose:** Make cache lifetime an explicit engineering decision rather than an unconditional “always swap” rule.

**Why follows:** Once students know what is valid and shareable, they can decide how long and where to keep it.

#### 5.6 Reducing the representation

**Subtopics**

- Distinguish architecture-defined compact state from runtime changes to precision or retained cached information.
- Explain capacity, bandwidth, conversion overhead, and possible quality changes with one chosen mechanism.
- Compare exact allocation/reuse with approximate cache modification; define a baseline and task-quality acceptance criterion.
- Keep detailed attention-head pruning, latent-attention algebra, and numerical format catalogs in supporting material.

**Purpose:** Provide the additional choice needed when allocation and reuse do not meet the budget, while preserving the distinction between implementation and approximation.

**Why follows:** Students should exhaust the explanation of preserving and relocating state before considering changing the representation itself.

#### 5.7 Choosing an inference-memory policy

**Subtopics**

- Apply the chapter's mechanisms to a supplied request trace with a shared candidate prefix, one changed early instruction, and an inactive interval. Supply the token counts locally for the prefix-sharing calculation.
- Compute occupied capacity and trace state ownership as branches start and finish.
- Compare resume cost and quality when optional compression is enabled.
- Specify which workload changes would require revisiting the design; provide the later scheduler with measured costs and capacity constraints.

**Purpose:** Close with a complete policy whose assumptions and limits the student can defend.

**Why last:** It requires sizing, allocation, validity, lifetime, and approximation distinctions. It ends at the boundary with Chapter 11 instead of teaching scheduling twice.

**Argument closure and culminating assessment**

Provide a model configuration, a finite memory budget, a prompt-history trace, and measured transfer/recomputation functions. The trace contains shared prefixes, divergent suffixes, a prompt revision, and idle intervals. The student supplies an allocation/ownership trace, peak occupied memory, cache-validity decisions, a retain/offload/recompute choice, and an acceptance test for an optional compressed representation. This assesses all objectives without requiring a custom attention kernel or a production scheduler. The chapter concludes that physical execution-state reuse is valuable but does not decide what knowledge deserves durable storage.

**Calculation designs**

1. **Capacity and allocation.** Variables: layers `L`, KV heads `Hkv`, cached dimensions per head `d`, bytes per element `b`, per-request lengths `n_i`, block capacity `B`, physical budget `M`, and non-cache allocations `M_other`. For a specified standard KV geometry, explain the factor for key plus value and calculate per-token cache size; other geometries use their specified representation. Compute rounded block occupancy for a heterogeneous request set. Decision: feasible concurrency and block-size sensitivity, with overhead and sharding assumptions explicit.
2. **Shared-prefix savings.** Variables: common prefix length `p`, branch-specific suffixes `s_i`, per-token storage `k`, and block rounding. Compare independent storage with shared storage; include a partially filled last common block if copy-on-write requires it. Decision: how much candidate breadth fits. State that cache sharing does not isolate external side effects.
3. **Retain/offload/recompute.** Variables: cache bytes `K`, effective outward/inward bandwidth, setup costs, measured prefill function `T_recompute(n)`, expected idle/reuse pattern, and required resume delay. Compare both transfer directions and available overlap; account for freed memory and uncertain reuse. Decision: choose a policy under an explicit capacity/responsiveness objective, not from idle duration alone.

**Substantive additions and consolidation ledger**

| Need | Existing location | Disposition and reason |
|---|---|---|
| Representation identity and reconstruction assumptions | `sec-vol3-virtualmem-intro`, `sec-vol3-virtualmem-geometry` | Make explicit in 5.1. The current phrasing repeatedly treats tokens, memory addresses, and tensor state as interchangeable. |
| Complete capacity accounting | `sec-vol3-virtualmem-geometry` | Consolidate duplicated formulas from Chapters 4–5 into 5.2; add non-cache allocations, heterogeneous lengths, and explicit sharding assumptions. |
| Allocation-pool versus hardware address semantics | `sec-vol3-virtualmem-pagedattention` | Explain in 5.3; keep block mapping and tail waste, remove overly literal MMU claims and universal physical-fragmentation guarantees. |
| Cache validity and invalidation | `sec-vol3-virtualmem-defrag`, `sec-vol3-vm-radix-tree` | Strengthen in 5.4. Common token content is not the only compatibility condition; prompt revisions and model changes are useful first-principles cases. |
| Prefix index versus allocation mechanism | `sec-vol3-virtualmem-pagedattention`, `sec-vol3-virtualmem-defrag` | Separate into 5.3 and 5.4. Consolidate radix lookup, hash indexing, and multiple repeated algorithm explanations into one illustrative implementation. |
| Inactive cache ownership and lifetime | `sec-vol3-virtualmem-defrag` | Develop in 5.5. Releasing active ownership, remaining reusable, and physical reclamation must not collapse into one event. |
| Approximation boundary and validation | `sec-vol3-virtualmem-compression`, Chapter 4 `sec-vol3-workingsets-pruning` | Consolidate in 5.6. Preserve only mechanisms needed for an engineering choice and require task evidence. |
| Scheduling and placement | Chunked-prefill and swapping subsections in `sec-vol3-virtualmem-defrag`; `sec-vol3-virtualmem-disaggregation` | Keep transfer/recompute mechanisms in 5.5; move interference management, resource allocation, disaggregation, and cluster sizing to Chapter 11. |
| Specialized math and catalogs | `sec-vol3-virtualmem-geometry`, `sec-vol3-virtualmem-compression` | Demote full attention-architecture catalogs, kernel instruction details, FP8 bit formats, and SVD proof. Retire unsupported guarantees of zero overhead, universal optimal sizes, and accuracy preservation. |

### Chapter 6. Persistent Memory and Retrieval

**Current manuscript:** [06_episodic_memory](06_episodic_memory/06_episodic_memory.qmd)

**Question, claim, entry, and exit**

**Question:** What should an agent remember across invocations, and how can it find information that remains relevant and trustworthy enough for the next decision?

**Teaching claim:** Persistent information is useful when the system can recover relevant, current, attributable records and evaluate their status.

**Entry prerequisites:** Information roles and context policy from Chapter 4; the distinction between derived inference state and source records from Chapter 5; the initial task-evaluation contract. Basic records, indexes, and database queries are systems prerequisites; semantic retrieval and agent memory lifecycle are taught here.

**Exit capability:** The student can design a write/retrieve/update policy, justify retrieval mechanisms for specific questions, preserve provenance and scope, and evaluate whether memory helps task completion without misleading the agent.

**Lasting takeaways**

1. **Different records need different contracts.** Source evidence, current authoritative state, execution history, and generated lessons differ in ownership, lifetime, and what the system may conclude from them.
2. **Retrieval supplies evidence to evaluate.** An exact lookup establishes which record was returned; semantic similarity establishes a retrieval relationship. Neither alone establishes that the record is current or its claim is true.
3. **Persistence requires a lifecycle.** Identity, versions, provenance, supersession, retention, and deletion determine whether stored information remains usable as the environment changes.
4. **Retrieval and context are one pipeline.** The cost includes locating, selecting, and incorporating information; retrieving more material can increase expense and reduce useful signal.
5. **Memory earns its place through task outcomes.** Measure useful recall and misleading recall, freshness, latency, and maintenance cost against an agent that uses simpler records or fresh observations.

**Learning objectives**

| ID | Assessable objective | Sections |
|---|---|---|
| 6-LO1 | Classify persistent records by authority, provenance, ownership, and required lifetime. | 6.1, 6.2 |
| 6-LO2 | Select retrieval methods that match a question's identity, semantic, or structural requirements. | 6.3 |
| 6-LO3 | Construct a bounded evidence package that preserves source identity, uncertainty, and contradictions. | 6.4 |
| 6-LO4 | Design update, invalidation, retention, and deletion policies for changing source information. | 6.5 |
| 6-LO5 | Allocate retrieval resources using latency, capacity, and task-quality requirements. | 6.6 |
| 6-LO6 | Evaluate whether a memory design improves task completion without increasing misleading or stale recall. | 6.7 |

**Ordered body sections**

#### 6.1 What needs to survive

**Subtopics**

- Contrast an original source, an observed result, a generated summary, and a prior hypothesis after the invocation that created them has ended. Establish which records need to survive and what each can establish.
- Separate authoritative external state, durable evidence, execution history, and generated reusable lessons.
- Distinguish persistence across invocations from adapting model weights, and persistent information from disposable inference caches.
- Define the questions a store must answer before choosing a database or index.

**Purpose:** Replace the old hardware-tier hierarchy with information responsibilities and retrieval requirements.

**Why first:** The student has just learned that caches may be regenerated or evicted; the book must now establish what should persist independently of those caches.

#### 6.2 Writing useful records

**Subtopics**

- Assign stable identity, origin, source version, capture time, relevant entity/task identifiers, and ownership or permitted scope.
- Preserve originals and link derived chunks, embeddings, summaries, and extracted claims to them.
- Choose meaningful record boundaries; include enough surrounding context to interpret a retrieved excerpt.
- Separate observed facts, inferred claims, and verification status; introduce controlled writes and deduplication without implying a log entry makes an external action atomic.

**Purpose:** Make retrieval correctness depend on what was recorded, not only on the search algorithm.

**Why follows:** Each record category in 6.1 now receives the identity and metadata needed to be retrieved and maintained.

#### 6.3 Finding information

**Subtopics**

- Compare identifier lookup and lexical retrieval with semantic similarity search; teach each through a query it serves and a query it can mishandle.
- Introduce structural retrieval where relationships matter, using a code dependency example with explicit limits of static analysis.
- Explain candidate generation, filters, and ranking as separate decisions; a hybrid is an option when requirements justify it.
- Introduce approximate indexing only after the retrieval objective is clear; describe the memory/latency/recall trade-off without full index derivations.

**Purpose:** Let the query contract choose the retrieval mechanism.

**Why follows:** Search depends on record identities, representations, and scope already established. It should not open with hypersphere geometry or an index catalog.

#### 6.4 Turning retrieval into context

**Subtopics**

- Filter candidates by task/entity, authorized scope, source version, relevance, and status.
- Compare candidate ranking, deduplication, diversity, and evidence coverage under a fixed context budget.
- Preserve citations, disagreements, and missing information when assembling a compact evidence package.
- Decide when the agent should fetch an original or obtain a fresh observation instead of trusting an excerpt or memory summary.

**Purpose:** Connect the retrieval system to the selected context that actually affects a decision.

**Why follows:** Candidate retrieval produces possible evidence; the next step is deciding which candidates belong in the invocation and what conclusions they support.

**Introduction/application:** Applies Chapter 4's context policy; first detailed retrieval-to-context pipeline. Authority enforcement remains Chapter 8.

#### 6.5 Keeping memory current

**Subtopics**

- Track source changes, invalidation, and supersession; distinguish a remembered status from the currently versioned external record.
- Preserve contradictory observations with provenance instead of silently choosing the latest prose sentence as truth.
- Treat generated consolidation and reusable lessons as derived claims; retain links to evidence and evaluate whether the lesson generalizes.
- Define retention, archival, deletion, and rebuilding of derived indexes; separate deleting a searchable summary from deleting its source record or audit obligation.

**Purpose:** Teach how useful memory survives a changing environment without accumulating misleading certainty.

**Why follows:** Students know how records influence context, so they can see why stale or contradicted records must change before future retrieval.

#### 6.6 Bounding retrieval cost

**Subtopics**

- Decompose the relevant read path into query preparation, search, filtering/ranking, context assembly, and added model processing.
- Compare index choices and placement through measured recall/latency/capacity trade-offs; distinguish rank fusion from learned reranking rather than declaring them interchangeable.
- Cache only under a stated validity and invalidation contract; distinguish exact query identity from unchanged environment state.
- Allocate a deadline and specify partial-result, fresh-query, or defer/escalate behavior according to the task's evidence requirements.

**Purpose:** Explain the systems costs that can make a useful retrieval design impractical.

**Why follows:** Optimization requires the fidelity, freshness, and scope contracts already established; latency targets must not silently weaken them.

#### 6.7 Evaluating the memory system

**Subtopics**

- Compare a minimal record/fresh-observation baseline with selective persistent retrieval on the same set of information requests.
- Test entity confusion, stale versions, duplicate evidence, contradictory reports, and unsupported generated lessons.
- Assess evidence coverage and citation fidelity where no executable checker establishes the complete answer.
- Measure retrieval usefulness, misleading recall, task outcomes, latency, and storage/maintenance cost over the complete write/read/update lifecycle.

**Purpose:** Establish that the chosen memory architecture improves the system, with explicit failure boundaries.

**Why last:** A retrieval benchmark alone omits ingestion, context assembly, changes over time, and the effect on the task; the full lifecycle is now available to assess.

**Argument closure and culminating assessment**

The student receives a small, versioned record collection: original sources, two revisions, conflicting evidence, a rejected hypothesis, a derived summary, and a stale record. Deliverables are a record schema, write/update policy, query-by-query retrieval choice, bounded evidence package, invalidation/deletion trace, and evaluation against a simpler baseline. The design must show when it needs a fresh external observation, establishing the handoff to Tools and I/O. This assesses 6-LO1 through 6-LO6 without requiring continuity with an earlier application exercise.

**Calculation designs**

1. **Storage growth and retention.** Variables: record arrival rate `lambda`, mean original bytes `s`, derived-index bytes per record `d`, retention interval `T`, deduplication/archival policy, and metadata/replication overhead. State workload stationarity only when used; separate originals and indexes. Decision: which representation and retention policy fit capacity while preserving required evidence and rebuildability.
2. **Retrieval and incorporation budget.** Variables: measured query preparation, search, ranking, assembly, and incremental model-processing costs as functions of candidate counts and selected tokens. Add sequential stages only when execution is actually sequential; do not add stage p99 values and label the sum the end-to-end p99. Decision: candidate count, reranking/placement choice, or deadline fallback that satisfies both evidence and response requirements.
3. **Useful versus misleading recall.** Variables: known relevant records, retrieved relevant records, stale/wrong-entity records, source coverage, and task outcomes on held-out comparable tasks. Contrast candidate recall with whether the selected evidence changed the final result correctly. Decision: justify an index/selection policy and a freshness filter; explicitly distinguish search approximation error from a wrong assertion in the store.

**Substantive additions and consolidation ledger**

| Need | Existing location | Disposition and reason |
|---|---|---|
| Information responsibilities rather than hardware levels | `sec-vol3-episodic-intro`, `sec-vol3-episodic-hierarchy` | Rebuild into 6.1. Remove automatic L1→L2→L3 paging language for semantic information. |
| Explicit write path, stable identity, and provenance | Fragmented across `sec-vol3-episodic-hierarchy`, `sec-vol3-episodic-graphs`, `sec-vol3-episodic-consolidation` | Consolidate and expand into 6.2. These are prerequisites for correct retrieval, not late implementation details. |
| Source/derived record lifecycle and deletion | `sec-vol3-episodic-consolidation` | Add into 6.2 and 6.5. Deleting an index entry, retracting a claim, archiving evidence, and deleting source content have different effects. |
| Exact/lexical/semantic/structural query choice | `sec-vol3-episodic-vector-ann`, `sec-vol3-episodic-graphs` | Reorder into 6.3, starting from query requirements. Move exact lookup and the modality comparison ahead of ANN details. |
| Retrieval-to-context evidence package | `sec-vol3-episodic-graphs` | Rebuild as 6.4, applying Chapter 4's budget; preserve disagreement and source links. Avoid a new general knapsack theorem. |
| Freshness and source-of-truth boundary | `sec-vol3-episodic-vector-ann`, `sec-vol3-episodic-consolidation` | Strengthen in 6.5. Exact lookup does not establish current truth, and a strict cosine threshold does not establish an exact semantic match. |
| Retrieval cost and quality | `sec-vol3-episodic-sla` | Rebuild as 6.6. Keep decomposition and the added model-processing cost; remove categorical local/remote and RRF/cross-encoder recommendations. |
| Durability of actions and recovery | WAL and sleep subsections under `sec-vol3-episodic-hierarchy` | Move the protocol to Chapter 9; leave only the distinction between execution records and knowledge retrieval. Suspension mechanics belong in Chapters 10–11. |
| Distributed ordering | Lamport subsections under `sec-vol3-episodic-consolidation` | Move to Chapter 15. Single-agent record freshness and provenance should not require this prerequisite. |
| Algorithmic detail | `sec-vol3-episodic-vector-ann`, `sec-vol3-episodic-graphs`, `sec-vol3-episodic-consolidation` | Demote hypersphere proofs, IVF-PQ mechanics, PageRank convergence, detailed bitemporal SQL, and neurobiological forgetting analogies. Remove guarantees of syntactic completeness, factual soundness, or optimal context packing not established by the mechanism. |
| Lifecycle evaluation | No substantial dedicated section | Add 6.7; a memory chapter needs to demonstrate benefit and misleading-memory failure, not just describe storage and retrieval. |

### Part III — Tools and I/O

The action/observation contract comes before the mechanisms that enforce permitted effects. Interface design and authority are complementary responsibilities.

### Chapter 7. Tool Interfaces and Actions

**Current manuscript:** [07_actuation](07_actuation/07_actuation.qmd)

**Backward design**

**Governing question:** How should an agent's action and observation interfaces be designed so the runtime can execute requests and the model can make useful subsequent decisions?

**Teaching claim:** Tool interfaces determine which actions an agent can express, what the runtime can check, and what the agent learns from execution.

**Assumed prerequisites:** The complete execution loop and initial evaluation contract from Chapter 1; proposal versus execution and structured model outputs from Chapter 2; bounded candidate evaluation from Chapter 3; context budgets, inference-cache validity, and retrieved evidence from Chapters 4–6. Students need ordinary systems knowledge of functions, processes, RPC, and errors. They do not yet need the book's security, recovery, or scheduling implementations.

**Exit capability:** Specify an action/observation interface for a task, explain its error and retry semantics, and justify its granularity and information budget using task-level evidence.

**Lasting takeaways**

1. **An interface shapes behavior.** The choice between narrow operations and expressive tools changes action count, model burden, inspection opportunities, and enforceable restrictions. Interface size alone does not identify the better design.
2. **Well-formed is one level of acceptance.** Parsing, schema checks, preconditions, authorization, and task correctness establish different properties. Passing one check does not imply the others.
3. **Observations are part of the interface.** Bounded, attributable results help the next decision; overcompression can discard the evidence needed to diagnose failure or establish completion.
4. **A timeout leaves a question.** The interface must distinguish unsuccessful execution from unknown completion and expose the information recovery will need.
5. **Interface quality is empirical.** Tool catalogs, result formats, and discovery policies should be compared on complete tasks under stated context, latency, and authority constraints.

**Assessable learning objectives**

| ID | Objective | Sections | Assessment evidence |
|---|---|---|---|
| 7-L1 | Compare tool granularities using task coverage, model burden, action count, and enforceable restrictions. | 7.1, 7.6 | Justified choice between two interfaces for the same task. |
| 7-L2 | Design a tool request contract that distinguishes structural validity, preconditions, authorization, and task correctness. | 7.2, 7.3 | Annotated request and dispatch sequence with rejection outcomes. |
| 7-L3 | Construct bounded observations that preserve evidence, provenance, freshness, and actionable failure information. | 7.4 | Result schema and context budget applied to a large test output. |
| 7-L4 | Diagnose tool outcomes and specify when retry, outcome lookup, or escalation is appropriate. | 7.3, 7.5 | Classification of failed, rejected, pending, and unknown requests. |
| 7-L5 | Evaluate tool exposure and discovery strategies under context limits and changing interface versions. | 7.6 | Repeated task comparison with reported omissions and selection errors. |

**Section arc**

#### 7.1 Designing the action space

- Compare tool boundaries for inspecting a failure, changing a file, and running a check.
- Compare a general execution tool, separate narrow file/test operations, and a higher-level repair operation.
- Explain expressive power, discoverability, argument burden, intermediate feedback, and what the interface can restrict.
- Contrast a retrieval interface whose useful observations are source identities and attributable excerpts.

**Purpose:** Establish that tool design is a systems choice shaping what the learned core can accomplish. A compact shell schema can shift complexity into generated commands; a larger catalog can provide useful guidance.

**Why first:** Students have learned computation and memory. They now need a concrete interface requirement before discussing serialization or protocols.

**Section result:** A small requirements list and two plausible interfaces to carry through the chapter.

#### 7.2 From a proposal to an executable request

- Apply structured generation already introduced in Chapter 2 to a named tool and its arguments.
- Separate parsing/schema checks, resource lookup, semantic preconditions, and permission checks.
- Include rejection, incomplete generation, unavailable tool, and successful dispatch as explicit outcomes.
- Explain schema expressiveness and validation limits using a valid edit that targets the wrong file.

**Purpose:** Build the request contract without claiming syntax implies safe or correct work. A short schema and a dispatch diagram are sufficient; parser automata are optional supporting material.

**Why here:** The chosen action space now needs a form the runtime can interpret. This section establishes the contract that the transport must carry.

**Section result:** One annotated request and acceptance path. The runtime authorization check is a named boundary; its implementation remains Chapter 8.

#### 7.3 Connecting runtimes and tools

- Assign responsibility to model-facing definitions, runtime adapters, transport, and tool implementations.
- Preserve request identity, tool/schema version, caller context, and response correlation.
- Compare local and remote execution, short calls and long-running operations, and progress versus completion.
- Show interface compatibility and initialization with a protocol-neutral request/result exchange; use a named standard only as a dated example if the editorial policy permits it.

**Purpose:** Explain interoperability and lifecycle semantics. A wire protocol provides agreed messages; the participating services must still implement their promised behavior.

**Why here:** Once the action has a contract, the reader can reason about carrying that contract across process and service boundaries.

**Section result:** A request lifecycle with pending, completed, rejected, failed, and unknown outcomes. Detailed asynchronous scheduling is deferred.

#### 7.4 Designing useful observations

- Define result status, useful payload, source/version identifiers, truncation markers, and error details.
- Compare full results, pagination, bounded excerpts, diffs, and out-of-band artifacts.
- Preserve originals or references when summaries remove detail; separate model-facing observations from full operational records.
- Explain result caching using input, environment, version, caller permissions, and freshness dependencies; apply Chapter 5's cache-validity distinction without repeating KV implementation.

**Purpose:** Teach the incoming half of I/O. Observation design determines both the next decision and the evidence available for later diagnosis.

**Why here:** Completing the outgoing request path reveals the next constraint: external systems can return much more information than the model should receive.

**Section result:** A bounded test-result schema and an attributable evidence excerpt. An explicit flag says when additional data exists.

#### 7.5 Failures and retry contracts

- Distinguish model choice errors, rejected preconditions, tool failures, and lost acknowledgments.
- Introduce idempotence with concrete repeated operations; distinguish repeated execution from repeated externally visible effects.
- Define stable operation identity across retries and the value of outcome queries or versioned conditional updates.
- Specify the behavior when the service offers no deduplication or authoritative outcome lookup.

**Purpose:** Give Chapter 9 the interface facts it needs for recovery. The runtime should receive an honest unknown state rather than fabricate success or failure.

**Why here:** After students understand request and response paths, a missing response has a precise meaning. The problem can be explained without FLP or a universal reliability law.

**Section result:** A retry-semantics table for reading a file, setting a versioned record field, and creating a new external record.

#### 7.6 Scaling and evaluating the interface

- Compare static exposure, task-specific subsets, explicit discovery, and retrieval-assisted selection.
- Explain that discovery introduces omissions and selection errors; an absent tool and an unauthorized tool are different cases.
- Handle catalog and schema evolution, including policy/model familiarity with an older interface.
- Evaluate the two opening interfaces on repeated, comparable tasks; test the limits of their observations and completion contracts.

**Purpose:** Complete the design argument using outcomes, erroneous actions, calls, context volume, elapsed time, and intervention.

**Why here:** Discovery optimizes the interface already defined. Its value cannot be judged using schema-token savings alone.

**Section result:** A justified interface choice with known limitations and the evidence supporting it.

**Useful calculation designs**

1. **Schema and observation budget.** Supply hypothetical token counts for schemas, fixed task context, output headroom, and bounded results. Compare a static catalog with selected tools and full output with pagination. Quantities: available context, tokens per step, truncation frequency, and whether required evidence remains accessible. Decision: which interface fits the workload without hiding necessary information. Count logical input tokens separately from newly computed prefill work or billed tokens; caching changes the latter.
2. **Granularity versus completion time.** Provide measured or explicitly hypothetical distributions for model-call duration, tool duration, call count, and repair attempts under two interfaces. Compare observed task completion and latency, with failed attempts included. Decision: whether a compact expressive tool actually saves time without causing unacceptable errors. Do not assume independent step errors or infer completion from syntax-validity rates.
3. **Discovery value and omission cost.** Give candidate-set sizes, schema footprints, tool-retrieval latency, and held-out task results including cases where the needed tool was omitted. Compare context savings with extra turns and missed tasks. Decision: static task-specific exposure versus just-in-time discovery. Avoid a universal threshold such as 30 tools.

Numbers entering the manuscript must follow the book's registry/calculation workflow. These are calculation specifications, not validated measurements.

**Culminating design problem**

Design tools for a bounded task that requires inspecting files, changing permitted content, running checks, and producing a reviewable result. Supply two possible action sets, a finite context budget, representative check output, and one service interaction whose response may be lost. Require:

1. An interface choice and justification (7-L1).
2. Request acceptance and explicit rejection behavior (7-L2).
3. A bounded result format retaining evidence and artifact access (7-L3).
4. A retry policy that preserves unknown completion and stable operation identity (7-L4).
5. A small evaluation protocol for catalog selection and schema changes (7-L5).

**Cumulative conclusion:** The agent has a usable action/observation interface, with explicit limits on what its checks establish. Chapter 8 now asks who is allowed to exercise each operation and how those limits are enforced.

**Coverage changes and source disposition**

| Existing source | Action and destination | Reason |
|---|---|---|
| sec-vol3-actuation-io-subsystem, lines 62–133 | Consolidate into 7.1–7.2. | Keep mediation/lifecycle; remove repeated architectural overview, protection-ring tour, and MMIO/device-driver equivalences. |
| Lines 76–85, Action-Boundary Principle and p^N discussion | Remove purported risk law and universal integrity decay. | No operational definitions justify multiplying expressive power by authority; observed outcomes should assess interfaces. |
| Lines 134–194, generations and GPU grammar masking | Condense into 7.2; move a bounded mechanism explanation to supporting material if needed. | Remove era ranking, unconditional zero-error guarantees, and unsupported performance arithmetic. |
| sec-vol3-actuation-protocols, lines 196–278 | Retain decoupling in 7.3; shorten protocol-specific examples. | Avoid a protocol manual and unsourced disaster story. |
| Lines 279–349, payloads, schema tax, deep modules | Rework into 7.1, 7.4, 7.6. | Retain information cost and granularity. Remove universal four-tool optimum and assumptions that every step recomputes or permanently retains all schema tokens. |
| Lines 350–393, capability security | Move main development to 8.2. | Chapter 7 needs the gate; Chapter 8 owns the authority model. |
| sec-vol3-actuation-idempotency, lines 394–494 | Keep operation semantics in 7.5; move WAL/deduplication/compensation to 9.4–9.5. | A local log or expiring lock does not by itself resolve a remote effect committed before an acknowledgment was recorded. |
| Lines 495–540, latency and error products | Keep failure categories; apply brief accounting in 7.6. | Detailed waiting/allocation belongs in Chapters 10–11; repeated independent-step products obscure recoverable failures. |
| sec-vol3-actuation-discovery, lines 541–631 | Retain in 7.6 with alternatives and omission cases. | Remove mandatory vector search, power-law or exponential-selection claims, fabricated protocol extensions, and OOM anecdotes unless separately established. |
| Lines 632–657, tool cache and prefix synergy | Keep conditional result caching in 7.4. | Identical output alone does not establish an identical preceding prompt or resident KV state. |
| sec-vol3-actuation-summary and learning objectives | Rebuild against the backward design. | Current takeaways memorialize the literal analogy and implementation absolutes. |

**Gaps to add:** tool granularity as an empirical choice; explicit preconditions and version checks; honest unknown completion; pagination/truncation semantics; outcome lookup; limits of result caching; discovery omissions; comparison across stable workloads. These additions replace repetition and overclaims rather than increase the chapter without limit.

### Chapter 8. Authority and Execution Isolation

**Current manuscript:** [08_virtualization](08_virtualization/08_virtualization.qmd)

**Backward design**

**Governing question:** What may an agent access and affect, and which mechanisms enforce that boundary when model behavior or generated code is untrusted?

**Teaching claim:** An agent's permitted effects and resource use must be enforced through mechanisms selected against a stated threat model.

**Assumed prerequisites:** The execution loop, context provenance, proposed versus executed actions, and Chapter 7's acceptance gate and request/result semantics. OS/process/virtualization foundations are background knowledge. No dependency on future human-approval protocols, recovery machinery, or distributed-agent coordination is required.

**Exit capability:** Design and test an authority and containment policy for a defined workload, justify the execution environment, and identify permitted but harmful behavior and other residual risks.

**Lasting takeaways**

1. **Authority belongs to enforced interfaces.** A model's statement of intent does not grant permission. Access decisions need a trusted enforcement point and resource-specific scope.
2. **Isolation follows the threat model.** The requirements for an arbitrary-code runner differ from those for a restricted data API. Compatibility, trusted components, and overhead determine the choice.
3. **Local containment has a boundary.** A sandbox can restrict local execution while allowed remote services still expose consequential actions or information flows.
4. **Lifetime is a design decision.** Recreating an environment, reusing it for a task, and restoring a snapshot impose different state-continuity, provenance, and startup costs.
5. **Test the promised limits.** Resource ceilings, access restrictions, and data-flow controls need observable enforcement behavior; successful tests establish evidence within the tested scope.

**Assessable learning objectives**

| ID | Objective | Sections | Assessment evidence |
|---|---|---|---|
| 8-L1 | Construct a threat model identifying principals, assets, untrusted inputs, and permitted effects. | 8.1 | Explicit boundary diagram covering local execution and external effects. |
| 8-L2 | Design scoped authority and credential handling that limit misuse at each action interface. | 8.2, 8.5 | Permission policy and attempted-bypass analysis. |
| 8-L3 | Select an execution boundary using compatibility requirements, trusted components, and measured overhead. | 8.3 | Justified comparison of restricted API, sandbox, and virtualized execution. |
| 8-L4 | Compare environment lifetimes using state continuity, startup overhead, and workspace capacity. | 8.4 | Capacity/accounting exercise with exported artifacts and cleanup behavior. |
| 8-L5 | Evaluate network and data-access policies against allowed-service and credential-exposure risks. | 8.5 | Information-flow paths including an allowed destination that can leak data. |
| 8-L6 | Specify resource limits and containment tests with explicit failure behavior and residual risks. | 8.6 | Test matrix and interpretation of results. |

**Section arc**

#### 8.1 Establishing the threat model

- Examine an instruction embedded in an untrusted document that the agent reads as task data.
- Identify user, runtime, model service, tool runner, external service, assets, and authority.
- Distinguish accidental mistakes, adversarial instructions, malicious generated code, and compromised dependencies.
- Map intended and forbidden effects, including permitted operations that could still be harmful.

**Purpose:** Define what the design protects and from whom. Explain prompt injection as an input/authority problem without claiming a proof from transformer attention or equating it to a memory-safety exploit.

**Why first:** A technology comparison cannot be meaningful until the required boundary is known.

**Section result:** A small threat table and a diagram of enforcement points to carry through the chapter.

#### 8.2 Bounding authority at the action interface

- Separate the user's task authorization from the permissions held by a runtime or external service.
- Explain least privilege, resource scopes, allowed operations, preconditions, credential placement, and complete mediation.
- Compare identity-based permissions and capability-style delegation as mechanisms; show a restricted grant and enforcement path.
- Prevent model-generated assertions from becoming permission grants; identify the limit of local checks when remote services own effects.

**Purpose:** Develop Chapter 7's authorization gate into an enforceable policy. Introduce scoped delegation here; Chapter 15 later applies it to multiple agents.

**Why here:** The threat table establishes the assets and principals that each permission must refer to.

**Section result:** An authority matrix for repository reads, local edits/tests, external record updates, and exporting evidence.

#### 8.3 Choosing an execution boundary

- Compare a restricted service API with a tool that executes arbitrary generated programs.
- Identify the trusted components in processes, containers, language sandboxes, userspace kernels, and virtual machines.
- Evaluate compatibility, exposed interfaces, cross-task isolation, startup, and steady-state costs.
- Distinguish local isolation from authorization at remote services and model-hosting boundaries.

**Purpose:** Teach how to select containment appropriate to a task. A microVM is one useful mechanism; it is not the definition of agent security.

**Why here:** Once permission scope is explicit, the reader can determine which execution environment can enforce it with acceptable compatibility.

**Section result:** A justified boundary for untrusted program execution and a contrasting constrained retrieval interface.

#### 8.4 Managing workspaces and environment lifetimes

- Separate immutable base dependencies, writable task state, secret material, and exported artifacts.
- Compare per-call environments, per-task environments, and prepared pools with explicit reset policies.
- Explain copy-on-write sharing, snapshot-derived branches, contamination, and preservation of intended task progress.
- State what disposal removes, what persists elsewhere, and what local restoration cannot undo.

**Purpose:** Extend isolation into state and time without teaching recovery twice. Snapshot storage is a mechanism here; Chapter 9 owns reconstructing and resuming execution.

**Why here:** An execution boundary needs a lifecycle, and useful agent work frequently spans multiple calls.

**Section result:** A workspace-lifetime policy with exported outputs and disposal responsibilities.

#### 8.5 Controlling network access and information flow

- Identify secrets and sensitive records available to the model, tool runner, or external service.
- Compare no network, mediated dependency access, restricted destinations, and narrow remote APIs.
- Explain why read permissions, permitted writes, URLs, logs, and allowed services can create information-flow paths.
- Distinguish prevention, detection, filtering, and human authorization; document gaps instead of declaring exfiltration impossible.

**Purpose:** Show why local containment does not settle external effects or information release. Reading and quoting private source material can be consequential even without filesystem writes.

**Why here:** Students now know what the local environment contains and which interfaces let information leave it.

**Section result:** A data-flow policy with allowed paths, blocked paths, and remaining trust assumptions.

#### 8.6 Bounding resources and testing containment

- Set CPU, memory, processes, storage, I/O, execution duration, and external-request budgets where relevant.
- Define rejection, throttling, termination, and recorded evidence when a limit is reached.
- Test forbidden access, hostile tool output, malicious dependencies, resource exhaustion, and leakage through allowed interfaces.
- Assess compatibility failures and residual risk; carry findings into the opening threat model.

**Purpose:** Finish with a defensible design backed by tests, rather than a catalog of security technologies.

**Why here:** Resource limits complete the execution envelope; testing can now exercise every promised boundary.

**Section result:** A containment test matrix. Chapter 10 owns cancellation and human waiting protocols; Chapter 11 allocates resources across work.

**Useful calculation designs**

1. **Startup amortization by lifetime.** Supply a workload's tool-call count, measured initialization cost, tool work, and model/wait durations. Compare fresh-per-call with one environment per task and a prepared-pool option. Quantities: total startup work, elapsed-time contribution, environment reuse, and reset work. Decision: the required lifetime within an overhead budget. State explicitly that a fast reset does not establish that retained state is trustworthy.
2. **Workspace capacity with sharing.** Supply immutable base size, task-specific changes, per-environment memory, concurrency, and host reserve. Compare complete copies with a shared base plus private changes. Quantities: storage, resident memory, provisioning traffic, and capacity under each separate resource constraint. Decision: feasible concurrency and when to export/dispose artifacts. Do not infer CPU or I/O feasibility from memory capacity alone.
3. **Bounded execution under a runaway workload.** Provide CPU/memory/storage limits and a script's demand profile plus a legitimate test-run profile. Determine which limit fires, expected retained evidence, and whether a legitimate task would be terminated. Decision: set an envelope that isolates failures while admitting the target workload. Use declared enforcement granularity and measured behavior; do not multiply invented breach probabilities.

**Culminating design problem**

Apply Chapter 7's interface concepts to a task on shared infrastructure. The agent may read untrusted files, edit a task workspace, run untrusted programs, and obtain prepared dependencies. It must produce a reviewable artifact. A separate external tool may update a versioned record under a scoped grant.

Require the student to submit:

1. A principal/asset/threat table (8-L1).
2. Scoped permissions, credentials, and the authority checks at local and remote boundaries (8-L2).
3. An execution boundary justified by compatibility and trust assumptions (8-L3).
4. A workspace-lifetime and export policy with capacity estimates (8-L4).
5. Allowed information-flow paths, including risks through an allowed destination (8-L5).
6. Resource ceilings, expected failure behavior, and tests with explicit limitations (8-L6).

**Cumulative conclusion:** The student can explain what a chosen design permits and contains, with evidence supporting its limits. Isolation does not establish whether a pending remote action completed or preserve progress after failure; Chapter 9 now takes responsibility for the durable record and reconciliation.

**Coverage changes and source disposition**

| Existing source | Action and destination | Reason |
|---|---|---|
| sec-vol3-virtualization-intro, lines 50–63 | Replace repeated overview with 8.1's concrete threat model. | Current introduction presupposes a hardware solution before defining requirements. |
| sec-vol3-virtualization-threat-models, lines 64–160 | Retain threat surfaces and confused deputy in 8.1–8.2; remove W⊕X equivalence, attention derivation, disjoint-domain guarantee. | Familiar analogy cannot establish impossibility, independence, or absolute containment. |
| Chapter 7 capability section, lines 350–393 | Consolidate into 8.2. | Keep scope/delegation; avoid mandatory signed-token tuples and lattice formalism. |
| sec-vol3-virtualization-microvms, lines 161–285 | Rework comparison for 8.3 and lifetime accounting for 8.4. | Remove universal product winner, mandatory per-call isolation, and unsourced vendor timing/density verdicts. |
| Hypervisor architecture subsection, lines 235–285 | Optional implementation case study after factual review. | KVM ioctls, VirtIO ring layouts, page walks, and hardware model detail do not belong in the main chapter argument. |
| sec-vol3-virtualization-ephemeral-fs, lines 286–370 | Retain workspace separation/sharing in 8.4; limit recovery claims. | Local discard does not erase remote changes or establish zero leakage; deliberate task state must survive selected boundaries. |
| sec-vol3-virtualization-network-isolation, lines 371–506 | Retain network/data-flow reasoning in 8.5. | Remove universal absorbing-taint policy, automatic safe-read category, entropy threshold guarantee, and unqualified exfiltration prevention. |
| sec-vol3-virtualization-cgroups, lines 507–614 | Retain envelopes and enforcement in 8.6; demote configuration listings. | Host controls and guest behavior need distinct scope; syscall counts and generic cosine thresholds do not certify security. |
| sec-vol3-virtualization-fallacies and summary | Rebuild against the lasting takeaways. | Current form teaches absolutes such as mandatory microVMs, safe platforms, and permanent mathematical containment. |
| Chapter connection near line 705 | Hand off to Chapter 9, not Chapter 10. | Canonical sequence next teaches checkpointing/recovery. |

**Gaps to add:** explicit principal and asset model; local versus remote authority; model-service/data boundaries; permission distinctions for reading and releasing information; workload-dependent environment lifetime; output export; credential lifecycle; compatibility costs; adversarial test evidence and residual risk. Keep detailed human approval mechanics in Chapter 10.

### Part IV — The Agent Operating System

Durable records support recovery; lifecycle protocols govern intervention and waiting; scheduling allocates resources against that state. The host OS, model service, and external services retain their own responsibilities.

### Chapter 9. Durable Execution and Recovery

**Current manuscript:** [09_checkpointing](09_checkpointing/09_checkpointing.qmd)

**Governing question:** After an interrupted trajectory, what can the runtime establish, and what must it reconcile before continuing?

**Teaching claim:** Recovery reconstructs recorded execution and reconciles it with the environment's actual state; recovery of the program and correctness of its decisions require different evidence.

**Lasting takeaways**

1. **A saved conversation is incomplete execution state.** Recovery also needs accepted action identities, results, artifacts, versions, and unresolved operations.
2. **An unknown outcome deserves its own state.** A lost acknowledgment does not establish whether the action happened; retry decisions depend on the external interface.
3. **Restoring local state cannot erase external effects.** Compensation and forward repair depend on application semantics and can themselves fail.
4. **Durability and cache retention serve different purposes.** Recoverable logical records may survive while expensive inference state is discarded and rebuilt.
5. **Replay is evidence about recorded execution.** It is distinct from reproducing a model decision, resuming against a changed world, or establishing task success.

**Assessable learning objectives**

| Objective | Sections | Takeaways |
|---|---|---|
| Classify interrupted actions by known outcome and identify evidence required before recovery. | 9.1, 9.4 | 1–3 |
| Design a durable trajectory record that preserves pending actions and versioned artifacts. | 9.2–9.3 | 1, 4 |
| Compare snapshot and event-history policies against storage and recovery objectives. | 9.3 | 4 |
| Select retry, reconciliation, compensation, or escalation for a specified tool contract. | 9.4–9.5 | 2–3 |
| Distinguish historical reconstruction, live resumption, and experimental re-execution using their evidence requirements. | 9.6 | 5 |
| Evaluate a recovery design through failures injected around external action boundaries. | 9.7 | 1–5 |

**Prerequisites:** Chapter 1's task/outcome contract; Chapter 4's working context; Chapter 5's reconstructible KV state; Chapter 6's versioned records; Chapter 7's tool contract and Chapter 8's authority/isolation boundary. Do not require cancellation machinery from Chapter 10.

**Exit capability and culminating artifact:** The student produces a recovery design for a local workspace operation and an external state-changing request: a durable record schema, an action-status transition table, a restart procedure, and a fault-injection result table. Supply the external service's versioning and completion semantics with the problem. The table includes failure before dispatch, after remote acceptance before acknowledgment, after acknowledgment before local persistence, and during compensation. It explains unresolved cases rather than counting every restart as success.

**Ordered body sections**

#### 9.1 What survives an interrupted task

- Follow the coding task through a worker crash, a rejected patch, and an unanswered remote request.
- Distinguish recorded local state, environment state, and observations of that environment.
- Separate infrastructure failure, unsuccessful decisions, and unknown action completion.
- Establish recovery objectives: tolerable lost internal work, acceptable recovery delay, and effects requiring reconciliation.

**Purpose and position:** Begin with the ambiguity students must resolve, rather than a generic fault taxonomy. The different outcomes motivate the contents of a durable record. This section applies the already known trajectory and tool contract; it first develops their recovery consequences.

#### 9.2 Recording trajectory state

- Record task and run identity, step/action IDs, accepted model outputs, observations, and references to artifact versions.
- Represent actions as proposed/authorized/dispatched/completed/failed/unknown as appropriate; distinguish an intended operation from a recorded result.
- Preserve runtime/model/tool/configuration versions and the authority context needed to interpret a past action; record permissions as historical evidence, not permanent credentials.
- Establish record ownership, ordered updates per trajectory, and handling of concurrent tool completions without requiring a universal vector-clock structure.

**Purpose and position:** Define the logical state once, before selecting persistence machinery. The record must be sufficient to determine the next legitimate operation. Chapter 10 extends this record with lifecycle/control state; Chapter 11 reads it for scheduling.

#### 9.3 Checkpoints and execution histories

- Compare full snapshots, append-only events, and snapshots plus a bounded history tail.
- Explain persistent artifacts versus cache handles: process-local addresses and expired leases cannot simply be copied into a new worker.
- Explain durable intent/result records, snapshot consistency, and reconstructing state after loss of a worker.
- Compare storage growth, persistence cost, record-application time, artifact retrieval, and inference-state rebuild time.

**Purpose and position:** A defined record now makes persistence alternatives meaningful. Periodic snapshot frequency and recording consequential action intent/results are separate choices; neither should be derived from semantic-error MTBF. End with a restored internal state whose pending external operation remains unresolved, motivating 9.4.

#### 9.4 Retries and ambiguous external outcomes

- Follow an external record-update request through the lost-acknowledgment window.
- Compare stable operation IDs, deduplication support and retention windows, outcome queries, and version-conditional updates.
- Handle unknown completion when the service cannot query or deduplicate; decide whether to wait, reconcile, or escalate.
- Address concurrent restart workers and stale owners: prevent duplicate local dispatch where the runtime can enforce ownership, and state what the external service must support.

**Purpose and position:** Students can now explain why a durable log alone does not create exactly-once external effects. This section deepens tool retry semantics from Chapter 7. It introduces the actual distributed failure boundary before compensation is offered as an option.

#### 9.5 Compensation and forward recovery

- Compare canceling uncommitted work, compensating completed effects, repairing toward a new acceptable state, and abandoning the task.
- Use a short ordered workflow with explicit dependencies and application-defined compensations.
- Preserve resource IDs and compensation progress; handle compensation failure, retries, and unresolved repair work.
- Explain why a model-proposed repair needs bounded authority and independent checks, and why no inverse may exist.

**Purpose and position:** Once completed effects are known, students can choose a recovery action. This is a comparison of designs, not a claim that every action has a deterministic inverse or that every agent must use Sagas. Dependency order constrains any proposed fail-fast reordering.

#### 9.6 Historical replay and live resumption

- Reconstruct accepted outputs and observations from the recorded history without reissuing completed mutations.
- Distinguish that reconstruction from rerunning inference with identical seeds and from generating a new continuation.
- Preserve relevant inputs, versions, and tool fixtures for an experiment; identify unrecorded context and replay limits.
- Recheck the current world and permissions before continuing live execution, including whether previously retrieved evidence remains current.

**Purpose and position:** The recovered state now supports two different uses: explanation of a past run and continuation of a task. The section explicitly separates them so later observability and evaluation chapters can reuse the distinction.

#### 9.7 Testing a complete recovery design

- Inject failure before/after record persistence, dispatch, result receipt, and compensation.
- Record duplicate effects, lost artifacts, unresolved outcomes, time to resumed useful work, and final task evidence.
- Compare a conversation-only restart with a durable record plus reconciliation under the same failure schedule.
- Defend the smallest design meeting the stated recovery objectives and name remaining unsupported failure cases.

**Purpose and position:** This final section demonstrates all six objectives through the cumulative artifact. It ends with an agent that can survive selected failures; intentional intervention and waiting become Chapter 10's problem.

**Useful calculation designs**

1. **Record and snapshot trade-off (9.3).** Assumptions: specified event count/rate, event sizes, snapshot size, retention, available read/write throughput, measured record-application cost, and artifact/model-cache rebuild times. Quantities: total bytes, persistence overhead, worst or percentile recovery time for each snapshot interval. Decision: select a snapshot/event policy meeting recovery objectives. Include the next useful model/tool step in recovery time; do not equate reading the WAL with full task recovery.
2. **Failed-workflow recovery cost (9.5–9.7).** Assumptions: an explicit dependency graph, measured forward/compensation durations, named failure locations, and separately stated compensation failures. Quantities: completed effects, outstanding repair work, elapsed time, extra calls and cost for each failure trace. Decision: reorder only independent checks or select a forward-repair plan that reduces unnecessary effects. An expected-value extension may use declared conditional probabilities; no universal ordering theorem is implied.

**Optional advanced material:** A classical periodic-checkpoint model for independent infrastructure failures and constant snapshot cost. Keep the derivation outside the main arc. Semantic decision errors do not become Poisson hardware failures merely to reuse Young–Daly.

**Concept coverage, additions, and removals**

- **Add:** Explicit unknown action outcome; completed-result persistence gap; deduplication expiry; current versus historical authority; record ownership during failover; recoverability of artifact references; version changes during long-lived runs; a recovery test matrix.
- **Retain/rebuild:** `sec-vol3-checkpoint-irreversibility` (line 61); idempotency discussion (208); event history (335); compensation debt (564); recorded replay (669).
- **Move:** Human oversight latency (244) → 10.4; accelerator copy-on-write details (385) → Chapter 5; full inference nondeterminism mechanics (624) → optional numerical foundations, with only the implication retained in 9.6.
- **Consolidate:** ACB definition (272) becomes one readable trajectory-record definition. Remove its independent redefinitions in current 10.182 and 11.694. Binary WAL offsets and serialization code around 351 are supporting implementation material, not central exposition.
- **Remove/rewrite:** The claim at 67 that “fail-plausible” is a new category invalidating Byzantine models; universal zero-liability internal rollback at 120; categorical collapse of transactions at 150; attention arithmetic used to assert all error feedback poisons context at 176; semantic MTBF combined with infrastructure MTBF at 410; universal precompiled inverse requirement at 489; invented mandatory Fork-on-Error protocol at 582.
- **Accuracy repairs required before reuse:** Logit example near 665 compares a 0.002 gap with noise of 10^-3–10^-4 and incorrectly treats the gap as smaller; RTO claims omit material restoration work; formulas must not manufacture guarantees of external consistency.

### Chapter 10. Supervision and Runtime Control

**Current manuscript:** [10_interrupts](10_interrupts/10_interrupts.qmd)

**Governing question:** How can a task receive new instructions or a stop request while its model calls and tools continue asynchronously?

**Teaching claim:** Control requires explicit protocols for lifecycle transitions, in-flight operations, approval, and resumption, with observable limits at service boundaries.

**Lasting takeaways**

1. **Control belongs outside model cooperation.** The runtime can stop issuing new work and enforce its own permissions and limits without waiting for a model to agree.
2. **Stopping has several observable stages.** Receipt of cancellation, cessation of new dispatch, local termination, and external completion are different events.
3. **Approval is tied to an action and context.** A permission decision must remain applicable to the arguments, versions, and authority used at execution.
4. **Waiting changes the task's assumptions.** The environment, permissions, and user intent may change before a task resumes; its old plan is not automatically current.
5. **Supervision consumes attention and resources.** Effective control policies bound unattended work and present enough evidence for meaningful human decisions.

**Assessable learning objectives**

| Objective | Sections | Takeaways |
|---|---|---|
| Construct a lifecycle model that distinguishes task state from outstanding operation state. | 10.1–10.2 | 1–2 |
| Diagnose duplicate, late, and out-of-order events in an asynchronous trajectory. | 10.2 | 1–2 |
| Specify cancellation behavior at model, local-tool, and remote-service boundaries. | 10.3 | 1–2 |
| Design an approval protocol binding a proposed action to relevant versions and authority. | 10.4–10.5 | 3–4 |
| Evaluate budget, timeout, and escalation policies using responsiveness and supervision evidence. | 10.6 | 1–5 |

**Prerequisites:** The record and action uncertainty from Chapter 9; authority and isolation from Chapter 8; context freshness from Chapters 4 and 6. OS/event-loop concepts are reader prerequisites, not subjects for a tutorial.

**Exit capability and culminating artifact:** A control protocol for a task with local work, an approval wait, and an outstanding external operation: task/operation state model, event table, cancel semantics, a review packet, an expiry/resumption rule, and a timed trace containing a simultaneous approval, user cancellation, and late tool completion. Supply the operation's effects and cancellation contract with the trace. Students demonstrate which event wins for each operation and how the losing/late event is recorded.

**Ordered body sections**

#### 10.1 The trajectory lifecycle

- Apply the durable task record to ready, running, waiting, paused, completed, and failed task states.
- Track model calls and tools separately: a paused task may still have an external operation awaiting completion.
- Assign lifecycle transitions and observable acknowledgments to the runtime rather than the model's textual status claims.
- Identify control ownership across runtime, model service, local worker, and external service.

**Purpose and position:** Establish a clear meaning for run/pause/stop before discussing implementation. Separate task and operation state prevents the draft's single ACB state from hiding concurrency. This is the first full lifecycle treatment; it extends, rather than replaces, 9.2.

#### 10.2 Events and concurrent work

- Handle model results, tool results, user steering, approvals, and timer events through a responsive control path.
- Correlate events with run/action IDs and versions; tolerate duplicate and late delivery according to explicit state rules.
- Use bounded concurrent operations and serialize conflicting decisions; distinguish arrival order from logical applicability.
- Preserve durable state transitions and restart behavior by applying Chapter 9's record contract.

**Purpose and position:** Events give the lifecycle operational meaning. A compact state-transition pseudocode example is sufficient; generic epoll/kqueue and POSIX tutorials are unnecessary. End with two operations in flight when a cancellation arrives.

#### 10.3 Cancellation and in-flight actions

- Distinguish acknowledging the request, stopping new dispatch, requesting service cancellation, terminating owned workers, and draining or reconciling external work.
- Compare local sandbox work, streamed inference, and a remote mutation already accepted by a service.
- Handle partial artifacts and late results; retain evidence even when their outputs are no longer used for continuation.
- Define measurable cancellation objectives and what the UI or caller may truthfully report.

**Purpose and position:** The runtime can now respond to an event, but the service contract determines its effects. Apply Chapter 9's unknown-outcome state; do not promise a socket close reverses a remote operation. Model-service or kernel-level details enter only if the runtime actually owns them.

#### 10.4 Human review and steering

- Assemble a review packet containing the proposed action, relevant arguments/targets, evidence, expected effects, and remaining uncertainty.
- Bind approval to the pending operation, applicable version, scope, expiry, and authenticated reviewer authority.
- Handle rejection, edited arguments, revised user goals, and delegated or unavailable reviewers.
- Distinguish approval of one consequential action from feedback that modifies the broader plan.

**Purpose and position:** Human intervention is now a well-defined control event with information and authority requirements. This applies Chapter 8's enforcement; it does not reteach cryptographic capability implementations or call a human a peripheral.

#### 10.5 Resuming in a changed environment

- Re-observe task-relevant state after long waits and validate action-specific preconditions or versions.
- Check that the approval and available permissions still cover the action now proposed.
- Resolve stale observations and conflicting updates by replanning, requesting revised approval, or abandoning the action.
- Reconcile late completions and avoid duplicated execution when resuming a task after restart or cancellation races.

**Purpose and position:** Approval is insufficient when its target changed during the wait. This joins Chapters 4, 6, 8, and 9 in one concrete resumption decision. Do not use a hash of “the environment” as universal evidence; identify precisely the state it covers.

#### 10.6 Budgets, progress, and an intervention policy

- Bound elapsed time, accepted work, retries, and outstanding expensive operations; account for work whose final charge arrives later.
- Monitor progress using task evidence and repeated failures, rather than assuming token activity is useful progress.
- Specify timeout, unattended-review, escalation, and retained-artifact behavior using the earlier recovery policy.
- Walk through the cumulative concurrent-event trace and evaluate responsiveness, remaining side effects, reviewer burden, and task outcome.

**Purpose and position:** The final design must sustain control across unattended operation and avoid accounting races. It closes on assessable behavior rather than a universal safety/liveness invariant. Resource occupancy during its waits motivates Chapter 11.

**Useful calculation designs**

1. **Cancellation timeline and outstanding work (10.3/10.6).** Assumptions: bounded local event-processing delay, declared provider cancellation behavior, a known set of accepted operations, and explicit limits on in-flight usage. Quantities: time to stop new dispatch, time to owned-worker termination, time to external outcome resolution, additional accepted work/charges. Decision: choose concurrency and reservation limits that meet the specified control objective. Unknown external completion stays unknown rather than receiving an invented finite bound.
2. **Review load and waiting (10.4/10.6).** Assumptions: a declared review-arrival workload, measured review effort, staffing/availability, and unchanged mandatory authority requirements. Quantities: pending-review growth, reviewer utilization, age of decisions, and interruptions per task. Decision: redesign the review packet, combine compatible reviews, change task scope, or add capacity. A simple utilization calculation is useful; cost minimization cannot remove required approvals or assume a reviewer catches every error.

**Concept coverage, additions, and removals**

- **Add:** Orthogonal task and operation state; cancellation acknowledgment semantics; stale/duplicate approvals; canceled run receiving a late successful mutation; user steering that changes the pending action; budget reservation for concurrently accepted work; explicit applicability of evidence to a versioned target; task-aware progress measures.
- **Retain/rebuild:** Event loop at `sec-vol3-interrupts-event-loop` (148), lifecycle (182), delivery handling (468), environment drift (559), watchdog/budget ideas (717/775).
- **Move:** KV capacity, bus-transfer and residency calculations (503–559) → 11.4; constitutional enclaves/capability internals (631–717) → Chapter 8; compensation rederivation (754) → Chapter 9.
- **Consolidate:** Keep one lifecycle illustration; use Chapter 9's record schema. Replace the long implementation (202–467) with a small transition routine whose correctness students can inspect.
- **Remove/rewrite:** Hardware interrupt mapping and invented authoritative POSIX-like signal specification (111/154); universal <5 ms cancellation bound (99/139); human-peripheral framing (483); escalation “if and only if” rule assuming perfect review (589); dated procurement incident without support (631); infinite-risk-threshold argument implying reversible work never needs human authorization (646); generic Coffman table and guaranteed rebase/rollback claims (732). Keep application-specific hold-and-wait examples briefly, then apply recovery/expiry rules.

### Chapter 11. Scheduling and Resource Management

**Current manuscript:** [11_scheduling](11_scheduling/11_scheduling.qmd)

**Governing question:** Which task should receive which resource next when agent computation is bursty, stateful, and interleaved with external work?

**Teaching claim:** Task lifetime, active inference time, cache lifetime, and tool execution lifetime differ; scheduling must coordinate them according to workload and service objectives.

**Lasting takeaways**

1. **Scheduling granularity changes the problem.** Admitting a trajectory, placing a model call, and building an inference iteration expose different information and controls.
2. **Idle computation can retain scarce capacity.** Cache residency during waits trades memory opportunity cost against transfer, rebuild, and resumption delay.
3. **Resource demand has multiple dimensions.** Tokens alone do not describe compute, memory, tool capacity, or the interference among concurrent tasks.
4. **Locality and specialization carry costs.** A warm cache can sit behind a long queue; split serving phases add state transfer and coordination.
5. **A scheduling policy earns its place through outcomes.** Useful completed tasks, deadline misses, fairness, and overload behavior must accompany throughput and occupancy.

**Assessable learning objectives**

| Objective | Sections | Takeaways |
|---|---|---|
| Decompose a trajectory trace into scheduling decisions and resource lifetimes. | 11.1 | 1–3 |
| Select admission and priority policies for heterogeneous tasks under explicit service objectives. | 11.2 | 1, 3, 5 |
| Compare inference batching choices using interference and response-time evidence. | 11.3 | 1, 3 |
| Choose retain, offload, or recompute policies under memory pressure and resumption requirements. | 11.4 | 2 |
| Evaluate request placement against queue delay, cache reuse, and state-transfer cost. | 11.5 | 4 |
| Defend a scheduling design through mixed-workload and overload experiments. | 11.6 | 1–5 |

**Prerequisites:** Invocation phases and costs (2), physical cache mechanisms (5), tool capacity (7), durable record (9), task lifecycle/control contract (10). Independent agent workloads suffice; cooperative swarms and their protocol design remain Chapter 15.

**Exit capability and culminating artifact:** A scheduling design for a supplied mix of inference demand, retrieval, tool waits, and human-review waits: admission limits, a priority policy, inference-batch budget, residency/placement decisions, and a report comparing useful completions and deadline/fairness outcomes under normal and overloaded conditions. Supply demand and timing assumptions locally so the comparison does not depend on an earlier application narrative. The design identifies which decisions belong to the application runtime and which require control of the inference service.

**Ordered body sections**

#### 11.1 Workload lifetimes and scheduling boundaries

- Follow a supplied execution trace through context preparation, inference, tool wait, review wait, and resumed work.
- Separate trajectory admission, model-request placement, inference-iteration batching, and external tool service queues.
- Identify resource lifetimes: accelerator work, KV residency, host memory, tool workers, network requests, and durable records.
- Characterize demand with trace distributions and dependency structure; distinguish active service time from task elapsed time.

**Purpose and position:** The old chapter starts inside a serving kernel before explaining what is scheduled. This section establishes scope and provides the trace reused by every later choice. It introduces the three scheduling granularities without claiming a trajectory must monopolize a server until completion.

#### 11.2 Admission, priority, and fair sharing

- Set task deadlines, responsiveness needs, useful-completion criteria, and tenant constraints before picking a queue.
- Compare simple FIFO/round-robin/priority policies against observed variable durations; apply aging or service accounting when justified.
- Bound admitted work and resource reservations; manage admission delay, backpressure, and rejection under overload.
- Compare single-resource accounting with multiple resource demands; present dominant-resource fairness as an optional policy under stated assumptions.

**Purpose and position:** Establish who may run before optimizing their accelerator batches. Generic queueing proofs are unnecessary; a short trace exposes the consequence of long jobs and finite memory. Cancellation semantics from Chapter 10 constrain scheduling preemption; they are not retaught.

#### 11.3 Sharing inference capacity

- Explain iteration-level admission/retirement and batching variable-length sequences.
- Use the established prefill/decode distinction to show long-prefill interference with active generation.
- Compare unchunked versus chunked prefill, iteration budgets, and batching efficiency under measured service objectives.
- Explain application-visible limits when inference is remote: request size/concurrency choices may be available while internal batching is not.

**Purpose and position:** A chosen admitted workload now shares an inference service. The mechanism is actual inference scheduling, not a CPU time-slice metaphor. Demonstrate a throughput/response trade-off, rather than asserting all prefill is compute-bound and all decode memory-bound.

#### 11.4 Memory residency during waits

- Apply Chapter 5's retain/offload/recompute mechanisms to a waiting trajectory.
- Account for expected wait duration, cache reuse likelihood, memory pressure, host capacity, and transfer contention.
- Separate release of accelerator state from release of task records, tool resources, and permissions.
- Compare resumption latency and completed work under alternate residency policies, including an unexpected early return.

**Purpose and position:** Chapter 10 established that tasks wait; this section decides what resources should remain attached. It owns the scheduling policy, while Chapter 5 owns the underlying mechanism. Transfer duration alone is not an economic threshold.

#### 11.5 Placement and cache locality

- Reuse Chapter 5's cache validity rules to identify which workers can avoid repeated input processing.
- Compare queue delay plus remaining service time for warm/busy and cold/idle workers.
- Handle changing load, stale cache directories, invalid cache entries, and migration costs.
- Choose a simple affinity/load policy and expose the workload assumptions under which it helps.

**Purpose and position:** After deciding whether to retain state, decide where a request should go. This is a system-level placement decision, not a second radix-tree tutorial. The optional worked example below applies this placement decision to separated serving phases.

##### Optional worked example: separating serving phases

- Compare colocated and separated prefill/decode workers under the same task trace and service objectives.
- Explain independent resource sizing versus KV transfer, network contention, and added coordination.
- Identify workload and scale conditions that can justify specialization; preserve a simpler colocated baseline.
- Evaluate whether transfer can overlap useful work rather than assuming it always disappears.

**Purpose and position:** This is an optional architectural extension of placement, taught after the mechanisms it depends on. A bounded example contributes by showing when to keep the simpler service or pay for separation. Detailed per-layer streaming proofs and vendor deployment configurations belong in supporting material.

#### 11.6 Evaluating the complete scheduler

- Run mixed traces through two policies using the same task/configuration/evaluation conditions introduced in Chapter 1.
- Measure useful completions, deadline misses, queue age, resource occupancy, fairness, and costs of preemption/rebuild.
- Stress overload, unexpectedly long tasks, synchronized tool returns, and large-context arrivals.
- Justify one design and show the changed workload that would invalidate the choice.

**Purpose and position:** The student now integrates decisions at all three granularities. This closes Part IV with an executable system design; later learning and distributed-agent chapters apply the established runtime rather than redefining it.

**Useful calculation designs**

1. **Capacity and interference trace (11.1–11.3).** Assumptions: a small workload with measured or explicitly hypothetical model/tool intervals, memory demands, dependencies, and service objectives. Quantities: admitted concurrency, queue delay, iteration timing, and task completion times under two policies. Decision: choose admission and batching limits. Do not infer infinite variance or queue collapse from a universal Pareto distribution.
2. **Residency decision under pressure (11.4).** Assumptions: valid KV size, actual remaining device capacity, measured one-way transfer/restore and recomputation times, host capacity, known queued work, and a range of wait durations. Quantities: memory-byte-seconds, transfer occupancy, resumption delay, and additional useful work admitted. Decision: retain/offload/discard for several states, including a lightly loaded server. Hardware inputs later come through MLSysIM; service times should be measured or declared scenario assumptions.
3. **Warm/busy versus cold/idle and split serving (11.5).** Assumptions: valid prefix lengths, queue estimates, remaining input work, effective state-transfer bandwidth, and competing traffic. Quantities: predicted completion time and bytes moved, with sensitivity to queue and bandwidth changes. Decision: select placement and determine whether phase separation is justified. No universal affinity coefficient or guaranteed transfer hiding is needed.

**Concept coverage, additions, and removals**

- **Add:** Scheduling-granularity map; distinction between service and elapsed time; hosted-inference control boundary; bounded admission/overload behavior; tool-service limits; synchronized returns; fair treatment of long jobs; an end-to-end task metric alongside TTFT/ITL; warm/busy versus cold/idle example.
- **Retain/rebuild:** `sec-vol3-scheduling-continuous-batching` (54); chunking (127); service objectives (398); placement decision (532); retain/offload/recompute comparison (755); multi-resource allocation (797).
- **Reorder:** Resource-lifetime framing and service objectives before kernel mechanisms; priorities/admission before batching; phase separation after locality and residency.
- **Consolidate/move:** ACB (694) and lifecycle (735) → first introduction in Chapters 9–10; radix-tree/index internals (507) → Chapter 5; reversibility/preemption doctrine (841) → apply Chapters 9–10; generic Roofline derivation (77) → reader foundations or concise reminder.
- **Demote:** Full chunked scheduler (193), full router (590), layer-streaming lemma (359), advanced fairness proof, and queue-distribution proofs. A compact implementation may appear only if it exposes the decision being taught.
- **Remove/rewrite:** Universal Pareto alpha 1.1–1.4 and infinite waiting claims (409–428); universal decreasing-hazard/LAS optimality (432); mandatory disaggregation (301); unconditional latency guarantees from token batch size (138); round-trip-transfer-as-economic-break-even (755); rule pinning memory during every irreversible external operation (850); incorrect universal Banker's Algorithm condition/forward-progress guarantee (864).

### Part V — Learning and Policy Adaptation

Diagnosis determines whether adaptation is warranted. The sequence then develops useful data, supervised targets, and learning from environment feedback. Each chapter returns to independently assessed task behavior.

### Chapter 12. Trajectory Data and Feedback

**Current manuscript:** [12_data_flywheel](12_data_flywheel/12_data_flywheel.qmd)

**Governing question:** When an agent repeatedly fails, what evidence should be collected, and how can it become useful material for improvement?

**Claim:** Trajectories become useful learning material through coverage, provenance, outcome evidence, and representation.

**Prerequisites:** A complete execution loop; distinction between actions and observations; model/runtime boundaries; context and persistent records; isolation; basic evaluation. Prior scheduling and storage concepts are applications, not new foundations here.

**Exit capability:** Given failures, candidate data sources, assessment limits, and a collection budget, the student can justify a collection and curation plan and specify how to test its usefulness.

**Lasting takeaways**

1. A policy limitation must be distinguished from a missing-information, tool-interface, or runtime defect before choosing training as the intervention.
2. The task distribution and recorded states determine what the data can teach; a large corpus does not establish useful coverage.
3. Acceptance means satisfying specified checks. It does not establish complete task correctness or erase uncertainty.
4. Successes, failures, and recovery traces serve different learning purposes; selection rules should follow the target capability.
5. Data value is established through independent downstream evaluation, with collection cost and provenance visible.

**Assessable learning objectives**

| Objective | Sections |
|---|---|
| LO12.1 Diagnose whether an observed failure supports policy adaptation or a change to another subsystem. | 12.1, 12.6 |
| LO12.2 Design a trajectory collection strategy that covers relevant tasks, states, and recovery behavior. | 12.2, 12.4 |
| LO12.3 Evaluate what an acceptance pipeline can establish and where incorrect trajectories may pass. | 12.3 |
| LO12.4 Calculate usable-data yield and locate the limiting stage in a collection pipeline. | 12.3, 12.5 |
| LO12.5 Construct data lineage and evaluation splits that limit leakage and measure downstream data value. | 12.5, 12.6 |

**Ordered sections and the argument they carry**

#### 12.1 Diagnosing a capability gap

- Examine repeated misinterpretation of a test failure as a concrete capability gap.
- Compare possible explanations: unavailable evidence, ambiguous tool output, defective retry logic, or weak action selection.
- Compare appropriate interventions: improve the interface or context, select another existing model, obtain demonstrations, or adapt the policy.
- Establish the specific behavior to improve and the baseline tasks on which the claim will be tested.

**Purpose and sequence:** This is the decision that earns the entire learning part. It prevents the current manuscript's unsupported premise that runtime engineering inevitably becomes inadequate. **First introduction:** selecting a policy-adaptation intervention. **Application:** failures, tool contracts, context, and baseline evaluation.

#### 12.2 Collecting tasks and trajectories

- Define a task fixture: initial state, permitted tools, relevant versions, completion criteria, and environment reset requirements.
- Compare operational traces, expert demonstrations, synthetic tasks, and sampled model rollouts; expose selection bias and provenance for each.
- Build a coverage matrix across task families, horizon, uncommon states, and authority requirements.
- Explain deduplication and task-family separation before collection volume becomes the optimization target.

**Purpose and sequence:** Once the missing capability is specified, collection must expose relevant decisions. Collection sources are alternatives with costs, not a human-versus-synthetic contest. **First introduction:** trajectory dataset design. **Application:** durable records and reproducible environments.

#### 12.3 Establishing outcome evidence

- Distinguish syntax checks, state assertions, executable tests, source-backed assessments, and expert review.
- Design cheap-to-expensive assessment stages and record what each stage checked, skipped, or could not determine.
- Treat false acceptance, flaky execution, underspecified tests, and grader disagreement explicitly.
- Distinguish mechanically checked properties from quality and completeness judgments, and state which outcomes remain uncertain.

**Purpose and sequence:** Collected traces need evidence before the system selects them as examples. Acceptance must have a stated scope before it becomes a filter. **First introduction:** assessment metadata attached to training examples. **Application:** early verifier limits and authority boundaries.

#### 12.4 Curating behavior and recovery

- Separate efficient successful behavior, corrective demonstrations, unrecovered failures, and ambiguous outcomes.
- Show how deleting all exploratory or diagnostic steps can remove useful learning targets.
- Collect recovery examples from observed failures and controlled perturbations; preserve the distinction between a failure and a demonstrated correction.
- Explain search-derived demonstrations and preference pairs as optional corpus products; no algorithm catalog or required mixture ratio.

**Purpose and sequence:** Evidence enables selection, but selection still needs a teaching purpose. A failed trace is not automatically a target to imitate. **First introduction:** curation by learning role. **Application:** recovery and bounded search. Detailed learning-distribution arguments belong in Chapter 13.

#### 12.5 Engineering the collection pipeline

- Connect task generation, policy rollout, environment execution, assessment, curation, and durable storage through versioned records.
- Apply isolated execution and reset mechanisms already taught; quantify environment preparation and assessment time.
- Apply queues and backpressure to unequal stage rates, variable durations, retries, and costly rejected work.
- Select retained content and access controls; preserve lineage, assessment versions, and failure evidence while minimizing unnecessary sensitive material.

**Purpose and sequence:** The student now knows what must be collected and assessed, so the pipeline can be sized for that workload. **First introduction:** end-to-end data yield as a pipeline objective. **Application:** scheduling, isolation, storage, and data-handling boundaries.

#### 12.6 Measuring whether the data helps

- Establish independent evaluation by task family, environment version, or time where appropriate; prevent near-duplicate leakage.
- Compare candidate corpora with the same downstream model, training budget, and runtime.
- Report both coverage and outcome changes, including recovery, unacceptable actions, assessment uncertainty, and cost per usable trace.
- Complete the collection plan and identify a stopping condition or next justified collection experiment.

**Purpose and sequence:** Pipeline throughput is valuable only if the produced data addresses the diagnosed gap. This section closes the causal argument begun in 12.1. **Application:** baseline and repeated evaluation; detailed experimental statistics remain Chapter 16 material.

**Useful calculation designs**

1. **Staged acceptance cost.** Variables: attempted traces `N`, stage costs `c_i`, conditional pass fractions `p_i`, and estimated accepted-but-incorrect fraction from an independently audited sample. Compute expected cost per attempt `c_1 + p_1 c_2 + p_1 p_2 c_3 + ...` and cost per accepted trace. Assumptions: pass fractions are conditional on the preceding gates; cost estimates use the same workload; unknown correctness remains unknown. Decision: order or strengthen gates without mistaking high yield for quality. Demonstrate why simply multiplying unrelated marginal pass rates is invalid.
2. **Pipeline provisioning.** Variables: generation rate `lambda_g`, number of assessment workers `m`, mean assessment service time `t_v`, mean stored bytes `b`, retention horizon `D`. Compare generation with service capacity `m/t_v`, calculate storage growth, and examine a specified slow-tail scenario. Assumptions: one assessment slot per worker in the simple estimate; measured service time includes reset; averages alone do not guarantee bounded queue latency. Decision: add capacity, change admission, or reduce expensive collection work.
3. **Targeted collection allocation.** Variables: failure-family prevalence `w_j`, collection cost `c_j`, number of independently assessed examples `n_j`, and measured held-out improvement from a small controlled pilot. Compare two allocations at fixed budget. Assumptions: uncertainty and pilot-to-full-run transfer are stated; no assumed linear gain from more data. Decision: prioritize a missing state family rather than maximizing raw trace count.

**Culminating design/problem and closure**

Provide a failure packet containing clean successes, recovery traces, failed runs, duplicate tasks, changed environment versions, and incomplete outcome checks. Students diagnose the gap, choose source mixtures and acceptance rules, size the pipeline, and specify an independent evaluation. Require them to identify where additional human judgment or evidence is needed. This demonstrates LO12.1–LO12.5. The final artifact is a defensible collection plan; Chapter 13 develops learning objectives for versioned action/observation examples.

**Coverage, additions, and removals**

| Current source | Disposition and target |
|---|---|
| `12_data_flywheel.qmd` Purpose and `sec-vol3-flywheel-intro` | Rebuild into 12.1. Remove assertions that all fixed-model runtimes are brittle and that agency requires own post-training. |
| `sec-vol3-flywheel-architecture`, demonstration scarcity | Keep economic motivation in 12.2; treat human demonstrations as a valid source. Remove unsourced universal prices, task lengths, and claims that humans only record flawless actions. |
| `eq-quadratic-error-compounding` and associated distribution derivation | Move the conceptual issue to 13.4 and the bounded theorem to optional foundations. Do not duplicate the derivation in both chapters. |
| Context compilation, current line 115 | Condense into 12.1/13.5 as an optional cost motivation. Retire “compilation restores attention” and fixed savings claims; tool schemas may still need live context. |
| Four subsystems and distributed disaggregation, lines 135–181 | Keep pipeline responsibilities in 12.5; apply prior batching and queueing mechanisms. Remove named products and fleet-size prescriptions. |
| `sec-vol3-flywheel-synthetic-gen`, task synthesis and sampling | Keep in 12.2 and 12.4. Sampling configuration is measured per workload, with no universal temperature interval. |
| Verification cascade, lines 226–266 | Keep in 12.3, explicitly bounded by specification and assessment quality. |
| MicroVM and CoW implementation, lines 267–307 | Condense to an application in 12.5; detailed isolation/reset mechanics stay with Chapter 8/9. |
| `sec-vol3-flywheel-rejection-sampling`, cost and filtering funnel | Keep curation mechanisms in 12.3–12.5. Majority agreement cannot repair an incomplete specification by itself. |
| Search distillation and entropy-regularized objective, lines 393–418 | Keep a short source-of-demonstrations example in 12.4; demote the objective, optional preference-optimization catalog, and fixed yield table. |
| `sec-vol3-flywheel-negative-mining` | Keep and expand learning-role distinctions in 12.4; remove required 70/30 or 25–35 percent recovery mixtures. |
| Storage and indexing, lines 540–636 | Keep lineage, retention, and sizing in 12.5. Condense database/codec/index catalogs and unsupported compression or storage-expansion numbers. |
| Fallacies and summary | Rewrite from lasting takeaways. Remove zero-bias/ground-truth guarantees and mandatory CPU-per-accelerator ratios. |
| Missing in existing body | Add a failure-to-intervention decision; evaluation splits before collection; data rights/provenance; downstream corpus comparison; an outcome whose completeness cannot be mechanically established. |

### Chapter 13. Learning from Agent Trajectories

**Current manuscript:** [13_sft](13_sft/13_sft.qmd)

**Governing question:** How should selected execution examples change a policy, and how can we tell whether that change helps the complete agent?

**Claim:** Supervised adaptation teaches selected behavior from demonstrations, subject to representation, objective, coverage, resource constraints, and execution-time evaluation.

**Prerequisites:** Chapter 12's curated examples and evaluation separation; model inputs/outputs; action and observation roles; context boundaries; basic probability. Locally explain next-token loss and parameter updates at the level needed for design. Apply existing resource and execution concepts.

**Exit capability:** The student can construct a trajectory training objective and adaptation plan, explain its resource costs and limitations, and judge the resulting policy through task execution.

**Lasting takeaways**

1. The training example must preserve who produced each token and what information was available at the decision.
2. Masking, weighting, and normalization define the behavior and examples emphasized by the objective; they do not enforce runtime correctness.
3. Demonstration coverage and deployed-state coverage can differ, making recovery evaluation and corrective data important.
4. Parameter-efficient adaptation trades update capacity and deployment complexity against training and storage resources.
5. Lower training loss is evidence about the chosen objective; task evaluation establishes whether the adapted system improved.

**Assessable learning objectives**

| Objective | Sections |
|---|---|
| LO13.1 Construct a trajectory training example that preserves roles, information availability, and action boundaries. | 13.1, 13.2 |
| LO13.2 Analyze how masking and normalization change the relative learning contribution of heterogeneous examples. | 13.2, 13.3 |
| LO13.3 Diagnose coverage gaps between demonstrations and states encountered by the deployed policy. | 13.4 |
| LO13.4 Compare full and parameter-efficient adaptation under explicit memory and deployment constraints. | 13.5 |
| LO13.5 Design an evaluation that distinguishes improved task execution from lower training loss. | 13.6 |

**Ordered sections and the argument they carry**

#### 13.1 From execution record to training example

- Serialize a supplied trajectory with task context, model-generated actions, externally returned observations, and termination.
- Preserve tool identity, versions, role delimiters, and the information available before each action; exclude future observations from the input for that decision.
- Explain next-token prediction as learning a conditional target on a recorded prefix, without duplicating model architecture.
- Describe rationale traces only when available, permitted, and useful; an explicit thought field is not a universal requirement.

**Purpose and sequence:** The previous chapter supplied records, not a fully defined training problem. This section creates that problem from one concrete example. **First introduction:** supervised trajectory examples and teacher forcing. **Application:** invocation and action/observation contracts.

#### 13.2 Choosing the learning target

- Distinguish tokens supplied as context from tokens selected as prediction targets.
- Explain an action-targeted loss mask; observations still influence later action predictions through the context.
- Compare target choices for tool calls, final answers, optional rationale, and successful corrections.
- Identify failure modes of incorrectly assigned role boundaries and distinguish them from runtime parser or authority failures.

**Purpose and sequence:** Once roles are preserved, the student can select what behavior the objective should teach. **First introduction:** masking as objective design. It is not a physical causality constraint or proof against invented observations.

#### 13.3 Weighting heterogeneous examples

- Compare equal-action trajectories with short and long tool output; calculate how the normalization choice changes their contribution.
- Distinguish equal-token, equal-example, and task-family weighting and the priorities they imply.
- Explain padding and packing, including attention isolation between packed examples and preservation of complete decision boundaries.
- Analyze truncation: losing the eventual correction or result can change the example's teaching content.

**Purpose and sequence:** A valid single example does not establish a valid corpus-level objective. The section links statistical priorities to batch engineering. **First introduction:** training normalization and packing choices. **Application:** context budgets and resource accounting.

#### 13.4 Learning beyond the demonstrated path

- Examine a run whose earlier model decision creates an unfamiliar state; distinguish that situation from an exogenous service fault.
- Explain distribution shift between expert-recorded prefixes and student-generated execution histories.
- Compare student rollout plus corrective labeling, dataset aggregation, and controlled fault-injection examples.
- Preserve teacher uncertainty and the limits of generalizing from injected faults; distinguish learned recovery behavior from runtime-enforced action limits.

**Purpose and sequence:** The objective is now explicit, allowing the student to see where its examples cease to represent deployed decisions. **First introduction:** student-induced distribution shift and dataset aggregation. **Application:** Chapter 12 curation and earlier recovery mechanisms. Optional theorem discussion must distinguish assumptions from measured behavior.

#### 13.5 Adapting within a resource budget

- Decompose training memory into frozen/trainable weights, gradients, optimizer state, activations, and temporary buffers.
- Explain low-rank parameter updates as one parameter-efficient option and compare their capacity trade-off with full adaptation.
- Account for long trajectories, activation memory, and batch size; fewer trainable parameters do not remove context-processing work.
- Introduce one bounded serving consequence: a base model and adapter must be version-compatible, and changing the active parameters affects reuse of derived inference state.

**Purpose and sequence:** Only now is it meaningful to choose how much of the policy to adapt. **First introduction:** parameter-efficient adaptation. **Application:** resource budgets, model versioning, inference-cache validity. Kernel and multi-tenant serving internals are supporting material.

#### 13.6 Evaluating the adapted policy

- Compare adapted and unchanged policies with the same tools, context policy, budgets, and independent task fixtures.
- Test unfamiliar tasks, changed schemas, recovery, unacceptable actions, and preserved general capabilities.
- Separate training loss from task completion and distinguish apparent gains caused by changed runtime settings or leaked tasks.
- Decide whether to deploy, collect targeted examples, reduce specialization, or retain the original model.

**Purpose and sequence:** This closes the chapter with evidence about the full system rather than treating optimization convergence as the endpoint. **Application:** early evaluation contract and Chapter 12 data independence.

**Useful calculation designs**

1. **Mask and normalization comparison.** Provide two synthetic trajectories with the same count of target action tokens but different numbers of observation tokens. Variables: target counts `A_i`, total lengths `T_i`, per-target losses `ell_ij`, and example weights `w_i`. Compare per-sequence normalization by `T_i`, normalization by `A_i`, and a global active-token mean. Assumptions: specify whether per-example losses are averaged afterward; do not call any denominator universally correct. Decision: align weighting with the intended task distribution rather than external verbosity. The calculation demonstrates relative objective contribution, not a predicted proportional parameter update.
2. **Adaptation memory budget.** Variables: base parameter count `P`, trainable fraction or adapter dimensions/ranks, bytes per weight/gradient/optimizer state, measured activation footprint `M_act(B,T)`, reserve `M_temp`, and device capacity. Compare full and low-rank adaptation with explicit optimizer assumptions. Decision: feasible batch/context/adaptation design; expose the residual activation bottleneck. Avoid the old text's inconsistent 14/16-byte sums and unsupported per-device activation totals.
3. **Training-versus-serving break-even.** Variables: one-time adaptation and evaluation cost `C_a`, measured per-task cost change `delta_c`, expected stable-version task count `N`, and task-quality threshold. Break-even requires `N*delta_c > C_a` only when measured quality meets the requirement and costs are comparable. Account for prompt caching, retraining after schema changes, and failed tasks. Decision: adapt, retain the existing model, or improve another subsystem. This is an optional worked application, not a claim that learning can remove all live specifications.

**Culminating design/problem and closure**

Students receive a curated trajectory corpus containing long observations, omitted corrections, an ambiguous role delimiter, and a small group of student-induced error states. They construct examples and masks, justify weighting and adaptation size, and design the independent comparison. Require an explanation of why a trained preference for approved actions does not replace a runtime authorization gate. This demonstrates LO13.1–LO13.5. The chapter closes when the student can defend an adaptation decision with system evidence. Chapter 14 then considers learning from assessed outcomes when good target sequences are unavailable or incomplete.

**Coverage, additions, and removals**

| Current source | Disposition and target |
|---|---|
| Purpose and `sec-vol3-sft-intro` | Rebuild around 13.1. Remove universal base-model collapse and specialized-compiler necessity. |
| `sec-vol3-sft-formatting`, trajectory formalism and serialization | Keep in 13.1 with a concrete trace first; distinguish recorded state from information actually visible to the policy. |
| Context compilation, lines 79–112 | Keep optional break-even reasoning in 13.5; remove fixed 85/99 percent savings and deterministic-schema promises. |
| Tokenizer extensions and collator code, lines 162–217 | Move low-level implementation to supporting material; keep role boundaries and version compatibility. |
| `sec-vol3-sft-masked-loss`, target and weighting equations | Keep one explained objective in 13.2–13.3; replace causality rhetoric with objective semantics. Observation prediction can be an auxiliary task; it is not forbidden by causal physics. |
| Normalization dynamics, lines 314–338 | Keep the contrast in 13.3, specifying batching and weighting assumptions; no universal library-default claims. |
| CUDA/PyTorch implementation, lines 339–413 | Move to a lab or supporting example; the body should expose the objective and data flow, not an API recipe. |
| `sec-vol3-sft-peft-lora`, LoRA/QLoRA | Keep resource/capacity comparison in 13.5; demote gradient and quantizer derivations and vendor-specific throughput assertions. |
| Adapter swapping, lines 469–490 | Keep a small serving consequence after introducing adapters; move implementation details to supporting material. Do not create an earlier-chapter dependency on an untaught adapter concept. |
| Schema overfitting, lines 491–504 | Keep changed-schema evaluation in 13.6. Treat regularization strategies as testable choices, not guarantees of schema generalization. |
| `sec-vol3-sft-exposure-bias` and DAgger | Keep concrete shift/recovery reasoning in 13.4. Distinguish DAgger's assumptions from offline fault injection; demote proofs and remove guaranteed reliability gains. |
| Fixed curriculum staging and anchor replay percentages, lines 596–639 | Keep curriculum and retained-capability considerations in 13.4/13.6; remove mandatory ratios and claims of eliminating forgetting. |
| Missing in existing body | Add independent end-to-end policy evaluation; accidental future-information leakage; packed-example attention isolation; ambiguous training targets; inference-cache validity under adapter changes. |

### Chapter 14. Reinforcement Learning from Environment Feedback

**Current manuscript:** [14_rlvr](14_rlvr/14_rlvr.qmd)

**Governing question:** When useful outcomes can be assessed, how can an agent learn through interaction without optimizing the wrong behavior or overwhelming its execution infrastructure?

**Claim:** Learning through interaction couples policy updates to reward design, environment behavior, sampling, and execution infrastructure.

**Prerequisites:** Supervised adaptation and policy versions; complete trajectories; bounded environments; assessment scope; scheduling and memory reuse. Explain policy, reward, return, exploration, and a sampled update locally. A complete POMDP course, Bellman derivation, PPO derivation, or survey of current RL algorithms is unnecessary.

**Exit capability:** The student can specify an interaction-learning loop, calculate its resource constraints, explain its feedback vulnerabilities, and distinguish higher training reward from stronger independent performance.

**Lasting takeaways**

1. Reinforcement learning optimizes the specified reward; agreement between that reward and the actual task objective must be examined.
2. A terminal outcome provides less local information than an action-level assessment, while denser feedback has its own cost and error modes.
3. The update method creates statistical and infrastructure demands that must be considered together.
4. Environments and assessors are part of the training system, with integrity, reproducibility, latency, and resource requirements.
5. A policy that achieves more training reward still needs independent evaluation under deployment conditions and constraints.

**Assessable learning objectives**

| Objective | Sections |
|---|---|
| LO14.1 Evaluate whether a task supports useful and feasible learning through environmental interaction. | 14.1, 14.2 |
| LO14.2 Design a reward specification that exposes proxy failures, constraints, and assessment uncertainty. | 14.2, 14.5 |
| LO14.3 Compare outcome and intermediate feedback for credit assignment, cost, and susceptibility to error. | 14.3 |
| LO14.4 Interpret a sampled policy update and identify its memory, exploration, and freshness requirements. | 14.4, 14.6 |
| LO14.5 Design a resource-balanced learning loop with independent evaluation and protected assessment. | 14.5, 14.6 |

**Ordered sections and the argument they carry**

#### 14.1 Learning when demonstrations are insufficient

- Compare candidate patches that can be assessed even when a complete expert action trace is unavailable.
- Explain policy → action → environment → observation → assessed outcome → update, distinguishing training from deployment.
- Establish feasibility: resettable or reproducible tasks, permissible exploration, informative feedback, and a starting policy capable of useful behavior.
- Compare interaction learning with further demonstrations or runtime changes; no universal imitation ceiling or requirement to proceed to RL.

**Purpose and sequence:** This makes the learning regime arise from a systems need rather than a claimed hierarchy of algorithms. **First introduction:** reinforcement learning from interaction and return. **Application:** isolation, permissions, task fixtures, and policy adaptation.

#### 14.2 Specifying rewards and their limits

- Map task outcomes, constraints, and resource use to explicit assessed quantities; distinguish hard runtime limits from reward penalties.
- Compare executable checks, formal checks of specified propositions, learned assessment, and human feedback by scope and cost.
- Explain that hidden tests used repeatedly for training are still part of the training signal; independent evaluation must remain separate.
- Examine a composite objective involving attributable claims, coverage, uncertainty, and assessment judgment.

**Purpose and sequence:** The system needs a credible learning signal before discussing how to maximize it. **First introduction:** reward specification and proxy mismatch. **Application:** earlier assessment limits and Chapter 12 provenance.

#### 14.3 Assigning credit across a trajectory

- Trace a late failing test back through several potentially useful diagnostic actions; terminal failure does not identify the faulty decision.
- Compare outcome feedback, intermediate checks, and learned process assessments, including error, delay, and extra execution cost.
- Explain exploration, rare successes, and groups with no reward variation; distinguish more samples from genuinely informative samples.
- Discuss bounded counterfactual exploration in a reproducible environment as an optional way to gather information, not an exact credit oracle.

**Purpose and sequence:** Once rewards are specified, their temporal information becomes the central learning constraint. **First introduction:** credit assignment and exploration. **Application:** bounded branching and environment restoration.

#### 14.4 Updating the policy under a compute budget

- Explain the intuition of making useful sampled behavior more likely relative to a baseline, with conservative updates and retained diversity.
- Work through one group-relative example: candidate rewards, a baseline, relative scores, and the lack of a within-group signal when all outcomes match.
- Contrast a learned value estimate with a sampled baseline in terms of estimation, additional model state, group size, and sample cost.
- Explain the roles of policy-version records and update limits without presenting clipping or regularization as correctness guarantees.

**Purpose and sequence:** Credit assignment gives the update mechanism a job. GRPO is a concrete worked instance, not the definition of RLVR or the part's final destination. **First introduction:** sampled advantage and policy-update mechanics. Detailed objectives can be optional.

#### 14.5 Protecting the learning signal

- Show a repair policy passing tests by modifying the assessment apparatus; distinguish reward exploitation from broader claims about deception.
- Apply authority boundaries to source edits, test assets, result channels, and grader access; assess extracted artifacts in an independently controlled environment.
- Handle flaky tests, contaminated tasks, grader changes, and behavioral collapse with measured diagnostics.
- Compare explicit runtime resource limits with reward costs for redundant actions; avoid penalizing necessary checking or recovery merely to shorten traces.

**Purpose and sequence:** Having seen how updates amplify rewarded behavior, the student can reason about amplified proxy errors. **Application:** security, authority, provenance, and versioning; the learning-specific responsibility is protection of the feedback channel.

#### 14.6 Operating and evaluating the learning loop

- Connect rollout inference, interactive environments, assessment workers, and policy training; record the policy and assessment versions used for every sample.
- Apply batching, prefix reuse, backpressure, and straggler handling; expose the trade-off between asynchronous throughput and stale experience.
- Budget model-state transfer, environment resets, observation length, and assessment retries; no universal topology or safe number of stale updates.
- Compare checkpoints on independent tasks and deployment budgets, including recovery, authority violations, and behavior changes; decide whether to release, revise the reward, or stop.

**Purpose and sequence:** This reunites the statistical learning loop with the execution system and closes the design problem from 14.1. **First introduction:** learning-specific policy freshness. **Application:** resource scheduling, cache validity, recovery, and evaluation.

**Useful calculation designs**

1. **Group-relative learning signal.** Use a small illustrative group containing two passing and two failing attempts at a task with stated checks; compute the mean and standardized relative scores. Repeat with all failures and discuss the absence of a useful within-group ranking. Variables: group size `G`, rewards `r_i`, mean, spread, and numerical stabilizer. Assumptions: scores describe this sampled group, not absolute correctness; candidate outcomes can be correlated. Decision: whether to improve task curriculum, feedback informativeness, or sampling rather than blindly increase group size. No full policy-gradient derivation is required.
2. **Reward/resource trade-off.** Variables: terminal task score `r_task`, action count `n`, cost coefficient `alpha`, hard deadline, and a separate unacceptable-action rule. Compare a short incomplete trace, a slightly longer successful trace with a needed check, and a much longer redundant trace. Assumptions: coefficients encode a stated preference and cannot replace authority enforcement. Decision: choose reward shaping and limits that discourage waste without teaching the agent to skip necessary checking.
3. **Coupled-loop throughput and freshness.** Variables: rollout arrival rate, mean/p95 environment time, assessor workers, optimizer batch consumption, sample age, policy-version transfer time, and allowable age criterion derived from measured learning behavior. Compare synchronous and bounded-asynchronous operation on a supplied duration trace. Assumptions: grouped outcomes retain their group identity; dropping late samples may bias task coverage; stability must be measured. Decision: capacity allocation, queue limits, admission, and stale-sample handling. Apply Chapter 5 prefix-memory accounting only if it changes the choice.

**Culminating design/problem and closure**

Give students a learning run with rising training reward, unchanged independent task success, intermittent test modifications, and slow assessment workers. Require a reward specification, a sample-update interpretation, a protected evaluation design, and a resource/freshness policy. Include an outcome that needs human or learned assessment and an operation whose authority limits restrict exploration. Students must account for both when choosing the learning procedure. This demonstrates LO14.1–LO14.5. The concluding decision concerns a versioned policy with measured capabilities, not an assumed superhuman policy or a mandatory move to multiple agents.

**Coverage, additions, and removals**

| Current source | Disposition and target |
|---|---|
| Purpose, `sec-vol3-rlvr-intro`, imitation ceiling | Rebuild 14.1; remove claims that supervised learning cannot produce robust agents or is strictly bounded by individual demonstrations. |
| `sec-vol3-rlvr-paradigm`, RLHF collapse | Replace with feedback comparison in 14.2. Human and learned assessments have limitations; deterministic checks also have limited coverage and exploitable specifications. |
| POMDP tuple and formal kernels, lines 99–116 | Keep a simple interaction model in 14.1; demote the full formal specification. |
| Deterministic oracles and taxonomy, lines 117–155 and 333–341 | Consolidate 14.2. Distinguish checking a formal proposition, passing a test suite, and accomplishing the intended task. |
| `sec-vol3-rlvr-ppo-grpo` | Keep one worked group-based update in 14.4 and explicit resource accounting. Demote full PPO/GRPO derivations and fixed hardware-topology claims. |
| `sec-vol3-rlvr-verifiers`, credit assignment | Move before update mechanics into 14.3; explain noisy intermediate evidence instead of promising pinpoint credit. |
| Emergent backtracking and test-time scaling, lines 342–386 | Retain only a bounded observed-behavior example in 14.6 if supported. Move externally controlled revision/search to Chapter 3. Remove inferred attention-head mechanisms and equivalence with external search. |
| `sec-vol3-rlvr-reward-hacking` | Keep concrete assessment-integrity problem in 14.5; distinguish proxy exploitation, deception, and scheming. Remove claimed universal incident statistics unless sourced. |
| Entropy and length-regularization treatment, lines 415–462 | Keep qualitative stability/resource trade-offs in 14.4–14.5; no universal remedy or guarantee that length normalization removes verbosity bias. |
| Verification enclave, lines 463–497 | Keep as application in 14.5. Protection of the training feedback channel does not make tests complete or eliminate all leakage. |
| `sec-vol3-rlvr-distributed-rollout` | Keep the coupled pipeline in 14.6. Apply existing scheduling concepts; retain policy-version freshness as a new learning-specific issue. |
| Prefix KV equations, lines 514–534 | Reference/use Chapter 5's accounting in one exercise; do not derive it again. |
| FIFO buffering and early termination, lines 536–594 | Keep conditional design choices in 14.6, including coverage bias from dropped long tasks and the lack of a universal safe stale-update count. |
| Missing in existing body | Add feasibility of exploration, independent evaluation distinct from secret training tests, feedback-version tracking, reward uncertainty, no-variation groups, and a final deploy/stop decision. |

### Part VI — System Integration and Operation

Coordination extends the execution system across workers. Evaluation, observability, and economics then assess and operate the complete design; these responsibilities also apply to a single agent.

### Chapter 15. Multi-Agent Coordination

**Current manuscript:** [15_multi_agent](15_multi_agent/15_multi_agent.qmd)

**Backward design**

**Driving question:** When does dividing a task among agents improve the result, and what makes their work fit together?

**Teaching claim:** Delegation helps when improved capability or useful parallel work justifies communication, duplicated context, coordination, and integration costs.

**Lasting takeaways**

1. **Partition work before adding workers.** Useful delegation requires separable responsibilities and explicit dependencies; concurrency alone does not make the task parallel.
2. **Handoffs carry state and evidence.** A usable result identifies its inputs, version, authority, output artifact, and checks so another participant can integrate it.
3. **Coordination does not establish correctness.** Infrastructure can agree on ownership or committed metadata while every agent still agrees on an incorrect answer.
4. **Evaluate the complete team.** Compare quality, elapsed time, resource use, and integration effort with a single-agent baseline under the same constraints.

**Prerequisites**

- Chapters 1 and 3: trajectory, explicit completion criteria, bounded single-agent decomposition and deliberation, basic evaluation contract.
- Chapters 4 and 6: private context, persistent records, provenance, and versions.
- Chapters 7–10: typed action interfaces, authority, recovery, interruption, human escalation.
- Chapter 11: shared-resource contention and admission; reuse only the scheduler's interface here.
- Post-training is not a prerequisite. This chapter may use an existing model without learning anything new.

**Assessable learning objectives**

| ID | Objective | Sections |
|---|---|---|
| 15-LO1 | Decompose a task into agent responsibilities with explicit dependencies and integration requirements. | 15.1–15.2 |
| 15-LO2 | Design a handoff contract that preserves artifact versions, delegated authority, and evidence of completion. | 15.3 |
| 15-LO3 | Diagnose conflicting updates and stalled dependencies in a coordinated execution. | 15.4 |
| 15-LO4 | Distinguish agreement about execution state from evidence that generated work satisfies the task. | 15.5 |
| 15-LO5 | Select a coordination design using measured outcomes, critical-path time, and total resource consumption. | 15.6 |

**Exit capability:** Design and defend a bounded cooperating team, including what each member owns, how work is combined, and when one agent is preferable.

**Section sequence and argument**

#### 15.1 When multiple agents help

- Use a code-change task to identify file inspection, implementation, tests, and review, including their dependencies under one agent.
- Distinguish capability specialization, independent search, and latency reduction as different motivations for delegation.
- Separate a team working on one task from many independent tasks served by one platform.
- Identify duplicated context and final integration as costs that begin before any network complexity.

**Purpose and order:** This section gives delegation a concrete problem to solve. It applies the Chapter 1 baseline rather than asserting that a single context creates an inevitable distributed-scaling barrier.

#### 15.2 Decomposition and coordination structures

- Construct a task graph with accountable roles, deliverables, and dependency edges.
- Compare supervisor–worker, pipeline, peer collaboration, and shared-artifact arrangements using the same task.
- Explain coordinator bottlenecks and partial results; allow a simple fixed workflow when its dependencies are already known.
- Compare independent evidence-gathering subtasks with duplicated retrieval to expose useful parallelism and wasted effort.

**Purpose and order:** Once delegation has a reason, the communication structure follows from the work. Graph topology is a design choice, not a universal prescription of actor trees.

#### 15.3 Messages, artifacts, and state ownership

- Specify a task envelope: task identity, input versions, expected artifact, authority, budget, and completion status.
- Distinguish private reasoning context, shared immutable source material, and shared mutable artifacts.
- Choose message payloads versus references to large versioned artifacts; preserve provenance and unresolved uncertainty.
- Define who may accept a result, reassign work, or authorize an external effect.

**Purpose and order:** A diagram of roles is insufficient until its edges have meaningful contracts. This section first develops multi-agent handoffs while applying memory and tool contracts already established.

#### 15.4 Concurrency, integration, and recovery

- Follow conflicting edits or record updates through isolation, version checks, integration, and rejection of stale work.
- Handle duplicate or late results, a failed worker, and an unknown external outcome; reuse Chapter 9 recovery semantics.
- Detect missing dependencies, waiting cycles, unproductive exchanges, and overloaded recipients; connect backpressure to bounded work.
- Propagate cancellation and restrict effects while a parent is suspended or its authority expires.

**Purpose and order:** Explicit contracts make failure and integration cases diagnosable. The section concludes with a completed integrated artifact rather than an inventory of distributed primitives.

#### 15.5 Disagreement and correlated errors

- Separate coordination of task ownership and committed metadata from assessment of answer or artifact correctness.
- Explain how shared models, prompts, evidence, and evaluators can create correlated errors; do not equate correlation with identity.
- Compare independent candidate assessment, discussion, executable checks, source inspection, and human review against specified properties.
- Keep one compact replicated-metadata example if the task needs reliable ownership across failures; leave Raft/Paxos mechanics outside the main argument.

**Purpose and order:** Successful integration does not establish a correct result. Evaluate the integrated artifact against its task criteria to show why acceptance depends on evidence and why agreement alone is insufficient.

#### 15.6 Evaluating a coordinated system

- Compare the complete team with a single-agent baseline using the same task distribution, allowed tools, and resource limits.
- Separate time spent on useful work, coordination, waiting, and final integration; count duplicated work and unsuccessful attempts.
- Examine scaling at several concurrency settings and error patterns, including correlated mistakes and coordinator overload.
- Resolve the opening task by choosing a topology, concurrency limit, handoff design, and escalation rule.

**Purpose and order:** The section closes the chapter's claim with a decision supported by evidence. Chapter 16 now explains how to obtain trustworthy comparisons and diagnose the observed failures.

**Culminating design problem and closure**

Provide a code-change task graph, a shared-interface change, timing observations, candidate patch results, and one stale test artifact. Ask students to choose single-agent or coordinated execution, define roles and handoffs, repair the integration failure, assess the final result, and justify concurrency under a fixed task budget. Require them to state which conclusions the available checks leave unresolved and what further evidence would be needed. This exercises LO1–LO5 and resolves the chapter's coordination decision.

**Calculation designs**

1. **Useful parallel speedup under measured coordination.** Inputs: baseline elapsed time `T1`, subtask durations `ti`, dependency graph, measured dispatch/communication/integration overhead `h(M)`, concurrency `M`, and acceptable-completion rate by design. Assume identical task scope and stated resource contention. Calculate the dependency-constrained makespan plus measured overhead, then compare elapsed time and total worker time. The decision is whether an additional worker improves acceptable completion within budget. Do not assume an `M²` term or a universal optimal fanout. If a simplified Amdahl comparison is retained, normalize its baseline explicitly.
2. **Communication and duplicated-context budget.** Inputs: number of workers `M`, exchanges `R`, task-envelope size, result size, and broadcast/targeted delivery pattern. Under a fixed-message-size assumption, count actual deliveries and ingested tokens for two topologies. Then vary result size or number of rounds. The decision is whether to send full text, bounded summaries, or artifact references. Message-edge counts do not by themselves establish model-quality or wall-time scaling.

**Coverage, additions, and removals**

| Existing source | Disposition and target |
|---|---|
| `sec-vol3-multiagent-intro`, `sec-vol3-multiagent-single-agent-ceiling` | Rebuild as 15.1. Remove assertions that large tasks inevitably require multi-agent systems. Refer to context and serving limits without rederiving them. |
| `sec-vol3-multiagent-coordination-tax`, `sec-vol3-multiagent-amdahl-derivation`, `sec-vol3-multiagent-operational-regimes` | Preserve the overhead question in 15.6; replace named law, fixed 4–6 optimum, and cubic proof with a scoped measured comparison. A decorative or incorrectly normalized law should be removed, not relocated to an appendix. |
| `sec-vol3-multiagent-topologies`, `sec-vol3-multiagent-hierarchical`, `sec-vol3-multiagent-p2p`, `sec-vol3-multiagent-market` | Consolidate into 15.2. Keep topological alternatives and costs; demote market protocols unless an actual task motivates them. Remove universal anti-swarm verdicts and zero-shared-context requirement. |
| `sec-vol3-multiagent-rpc`, `sec-vol3-multiagent-shared-context`, `sec-vol3-multiagent-wire-protocols` | Keep the handoff and ownership problem in 15.3. Replace schema/vendor inventory with one complete interface example. Shared immutable material remains permitted. |
| `sec-vol3-multiagent-backpressure`, `sec-vol3-multiagent-vector-clocks`, `sec-vol3-multiagent-concurrency-control`, `sec-vol3-multiagent-deadlocks` | Consolidate in 15.4. Keep dependency lineage, conflicts, bounded queues, and stalled-work decisions. Demote vector-clock algorithms; vector clocks neither make a graph unforgeable nor implement effect rollback. |
| `sec-vol3-multiagent-consensus`, `sec-vol3-multiagent-classical-consensus` | Retain the metadata/answer distinction in 15.5, with a compact infrastructure example only. No full consensus course inside this chapter. |
| `sec-vol3-multiagent-quorum-voting`, `sec-vol3-multiagent-classical-bft`, `sec-vol3-multiagent-bft-breakdown`, `sec-vol3-multiagent-law-of-verification` | Rebuild as 15.5. BFT does not assume statistically independent faults; an agreement guarantee does not prove semantic correctness. Remove deterministic-oracle dominance and zero-error claims. Keep correlated-error reasoning as a measured limitation of ensembles. |
| `sec-vol3-multiagent-quarantines`, `sec-vol3-multiagent-capability-attenuation` | Apply authority and recovery in 15.3–15.4; detailed mechanisms remain in Chapters 8–10. Retain delegation-specific authority checks without repeating syscall/firewall instructions. |
| `sec-vol3-multiagent-fallacies`, `sec-vol3-multiagent-summary` | Rewrite from the takeaways above after body completion; remove repeated universal mandates. |

**Material gaps to add:** explicit task decomposition and acceptance contracts; stale/partial/late-result handling; evidence accompanying handoffs; team-versus-serving-fleet distinction; fair single-agent baseline; coordinator bottlenecks; cancellation propagation. These additions supply missing responsibility boundaries rather than new mathematical machinery.

### Chapter 16. Evaluation and Observability

**Current manuscript:** [16_observability](16_observability/16_observability.qmd)

**Backward design**

**Driving question:** What evidence supports the claim that an agent works, and how can that evidence guide a change?

**Teaching claim:** Trustworthy operation combines task-level evaluation with execution records that support diagnosis and controlled release.

**Lasting takeaways**

1. **Measure the task and its constraints.** A successful model call or tool invocation does not establish acceptable completion; evaluation must name the outcome and the properties its checks cover.
2. **A comparison needs an experimental contract.** Task selection, environment state, system versions, repeated runs, and uncertainty affect what a measured difference means.
3. **A trace records observations.** Execution lineage can locate effects and dependencies, but a recorded explanation does not prove the causal reason for a model's decision.
4. **Release decisions combine imperfect evidence.** Offline tests, isolated shadow runs, and monitored deployment reveal different failure modes; none alone establishes general correctness.

**Prerequisites:** Chapter 1's evaluation contract; Chapters 3 and 12–14's uses of feedback; Chapter 6 provenance/versioning; Chapters 7–10 tools, isolation, durable execution, and intervention; Chapter 15 for distributed extensions. The main evaluation argument must remain fully intelligible for a single-agent system.

**Assessable learning objectives**

| ID | Objective | Sections |
|---|---|---|
| 16-LO1 | Define task outcomes and acceptance evidence while stating the limits of each evaluator. | 16.1 |
| 16-LO2 | Construct an evaluation with controlled initial conditions, held-out tasks, and explicit resource limits. | 16.2 |
| 16-LO3 | Assess a measured system difference using repeated runs, task variation, and uncertainty. | 16.3 |
| 16-LO4 | Design execution records and retention policies that support diagnosis within an evidence budget. | 16.4–16.5 |
| 16-LO5 | Diagnose a failed trajectory and specify a testable repair hypothesis. | 16.6 |
| 16-LO6 | Select release gates and monitoring criteria appropriate to an agent's authority and workload. | 16.7 |

**Exit capability:** Produce a defensible quality claim, reconstruct a relevant failure, and specify the evidence needed to release a change.

**Section sequence and argument**

#### 16.1 What the system must demonstrate

- Start with a release claim that omits unsuccessful tasks, unacceptable effects, or unresolved outcomes. Identify the missing evidence before choosing evaluation machinery.
- Separate task success, constraints, partial completion, abstention/escalation, resource use, and user intervention.
- Compare executable checks, learned assessment, source inspection, and human judgment; state false acceptance and false rejection possibilities.
- Distinguish correct citations and supported claims from fluent text or agreement among judges.

**Purpose and order:** Start with the claim to be measured. Success vocabulary is an application of Chapter 1; evaluator limitations receive their first full experimental treatment here.

#### 16.2 Constructing evaluation tasks and environments

- Define task population, coverage strata, development/held-out split, and target deployment conditions. Include relevant user groups and operating conditions when they change acceptable outcomes or expose uneven failure rates.
- Preserve task inputs, initial state, dependency versions, permissions, and budgets; explain reset and isolation requirements.
- Distinguish repeatable recorded environments, controlled service mocks, and live-environment evaluation with recorded provenance.
- Separate evaluation artifacts and permissions from what the agent may alter; distinguish present information leakage from prior training contamination.

**Purpose and order:** Acceptance criteria become an executable experiment. Air-gapping is a possible control, not a universal requirement or a cure for memorized training data.

#### 16.3 Comparing stochastic systems

- Compare versions on the same tasks; separate variation across tasks from variation across repeated runs on one task.
- Explain confidence intervals and practical effect sizes using a small paired example; define the independent sampling unit.
- Distinguish pass@1, pass@k, and a deployed selection policy; at least one successful candidate is not necessarily one successfully chosen candidate.
- Inspect outcome, latency, cost, and intervention distributions by workload stratum rather than relying on one average.

**Purpose and order:** A runnable evaluation now needs a justified interpretation. This section prevents a small noisy score increase from becoming a release verdict.

#### 16.4 Recording and linking execution

- Extend the durable trajectory record from Chapter 9 with observation links for configuration versions, inputs, artifacts, model responses, tool results, and observed effects.
- Link delegated work and asynchronous events using identifiers and dependency references; distinguish parent trees from cross-links.
- Attribute cost once to the operation that incurred it and aggregate without double-counting parent and child spans.
- Distinguish replay of recorded observations, rerunning the environment, and identical model generation.

**Purpose and order:** Evaluation identifies what failed; execution records supply the evidence needed to investigate it. Begin with one trajectory before adding delegated work.

#### 16.5 Collecting and retaining useful evidence

- Separate lightweight event metadata from large payloads and versioned artifacts.
- Choose retention, sampling, and priority preservation against observed volume and storage costs.
- Account for dropped events, late spans, interrupted collectors, and unavailable payloads.
- Set access, redaction, and retention boundaries for task content, credentials, and personal information.

**Purpose and order:** A desired record is not free or automatically complete. This section establishes the practical evidence budget before the reader relies on a trace for diagnosis.

#### 16.6 Diagnosing failures and testing repairs

- Reconstruct a failed task from its context, tool result, artifact versions, and external effects; use one bounded trace that exposes a cross-component cause.
- Separate plausible causes in the model proposal, selected information, interface, runtime, or environment; identify missing evidence.
- Design a controlled ablation or targeted replay that could falsify the leading explanation.
- Trace an ambiguous external acknowledgment to show why reconciliation requires service evidence, not merely the recorded last message.

**Purpose and order:** The chapter now uses its instrumentation for a systems decision. Logged reasoning is evidence of what was emitted, not proof of an internal causal explanation.

#### 16.7 Deploying and monitoring changes

- Version the entire configuration, define release thresholds, and run held-out and regression evaluations.
- Handle trajectories already running when runtime, tool-schema, or record-format versions change: pin and drain an existing version, migrate compatible state with validation, or reject unsafe resumption. Preserve recorded intent and recheck external preconditions; changing a version does not undo an external effect.
- Compare replay, state-isolated shadow execution, and staged live exposure; preserve action/approval boundaries from Chapters 8–10.
- Monitor task outcomes, resource budgets, intervention, and workload drift with explicit missing-label delays.
- Define pause, rollback/recovery, escalation, and further-evidence decisions; introduce sequential testing as one optional tool under stated assumptions.

**Purpose and order:** A diagnosed repair becomes a controlled system change. This closes the chapter's evidence argument and supplies measured inputs to Chapter 17.

**Culminating design problem and closure**

Give two candidate versions evaluated on the same supplied tasks, repeated outcomes, costs, a task-stratum shift, and one incomplete trace. Require a quality assessment with uncertainty, an explanation of what cannot yet be claimed, a targeted repair experiment, and a staged release protocol that addresses trajectories still running on the previous configuration. Under a supplied storage and privacy constraint, require a revised record and retention specification that addresses the incomplete trace and identifies evidence the policy will still lose. Include an outcome whose formatting checks pass while support for its substantive claims is incomplete. Students must revise the acceptance evidence rather than declare the task solved. LO1–LO6 are assessed without requiring new theory in the conclusion.

**Calculation designs**

1. **A paired release comparison.** Inputs: task IDs, strata, repeated binary/graded outcomes by system, intervention counts, costs, and minimum practically useful change. Treat tasks as the resampling unit, preserving repeated runs within each task; state what generalization to future tasks assumes. Calculate per-stratum differences and an interpretable interval, then compare the cost of more independent tasks with more repetitions. The decision is promote, reject, or gather targeted evidence. Do not infer statistical power from an observed p-value or treat correlated repetitions as independent tasks.
2. **The evidence-retention budget.** Inputs: trajectories/day, events/trajectory, metadata bytes/event, payload bytes/trajectory, observed exception fraction, retention days, replication/compression factors, and sampling policy. Calculate daily ingest and retained storage for two policies, then show which evidence a dropped trajectory loses. The decision is what to retain and how long. The calculation must not promise 100% incident retention: incidents can be unrecognized, uninstrumented, or lost before sampling.

**Coverage, additions, and removals**

| Existing source | Disposition and target |
|---|---|
| `sec-vol3-observability-intro`, `sec-vol3-observability-otel` | Reorder: outcomes first in 16.1, instrumentation in 16.4–16.5. Keep HTTP success versus task success distinction; remove claim that traditional distributed software only fails syntactically. |
| `sec-vol3-observability-otel`: conventions, user-space/eBPF, payload offloading | Keep lineage and payload budget. Standards illustrate implementation only. Remove guaranteed <1% eBPF overhead, tamper-proof/zero-bypass claims, and mandatory collector technology. |
| `sec-vol3-observability-tracing`: context propagation, span links, cost attribution | Consolidate in 16.4. Distinguish a causal-dependency graph from causal explanation of model choice. Aggregate costs without double counting. |
| `sec-vol3-observability-tracing`: tail/head sampling | Develop trade-offs in 16.5, including long-running incomplete traces. Remove guaranteed complete incident capture and flawed conditional-probability rhetoric. |
| `sec-vol3-observability-eval-gyms`: static benchmarks and five invariants | Rebuild as 16.2. Preserve interactive state, reset, evaluator isolation, environment/version control. Remove universal network air-gap, fixed reset deadlines, bit-exact replay, and claim that isolation eliminates prior contamination. Static component tests remain useful inside a complete evaluation. |
| `sec-vol3-observability-eval-gyms`: pass@k, power, drift | Develop 16.3. Keep estimator interpretation, uncertainty, environment drift. Demote factorial-overflow derivation and sample-size algebra; correct pass@k versus selected-answer success and unwarranted certainty about noisy differences. |
| `sec-vol3-observability-canary`: autonomy spectrum | Apply existing authority controls in 16.7. Remove fixed horizon tiers and inevitable ladder toward unattended production operation. |
| `sec-vol3-observability-canary`: shadow execution and online quality | Keep in 16.7 and evaluator limits in 16.1. State shadow-environment fidelity limitations; learned judges are calibrated assessments, not formal oracles. |
| `sec-vol3-observability-canary`: SPRT | Optional worked/supplementary method in 16.7. Explain monitoring decisions before equations; no universal replacement for fixed-sample evaluation or proof of production readiness. |
| `sec-vol3-observability-sre-metrics`: goodput/CPVG | 16.1 and 16.3 own definitions/measurement with explicit numerator, denominator, units, and workload. Optimization and economic comparisons move to 17. |
| `sec-vol3-observability-sre-metrics`: dense/MoE, Pareto, latency/Amdahl | Move economic comparison and critical-path application to 17. Do not reteach serving architecture here. |
| `sec-vol3-observability-sre-metrics`: timeouts and error budgets | Apply earlier control mechanisms in 16.7; report observed timeout and outcome rates. Remove universal alert thresholds and guaranteed suppression of transient variation. |
| `sec-vol3-observability-fallacies`, `sec-vol3-observability-summary` | Rebuild after the argument; remove incompatible statements that checks are simultaneously complete correctness gates and insufficient. |

**Material gaps to add:** single-agent evaluation explicitly before fleet extensions; paired comparisons; task versus run sampling units; grading uncertainty; missing/delayed production labels; evaluation integrity distinct from security containment; diagnostic hypothesis testing; replay versus re-execution; data-minimized observability. This chapter earns seven sections because evidence capture and diagnostic use are distinct competencies, and neither should be hidden in a telemetry catalog.

### Chapter 17. Performance and Cost Engineering

**Current manuscript:** [17_tokenomics](17_tokenomics/17_tokenomics.qmd)

**Backward design**

**Driving question:** Which change improves acceptable task completion under the actual latency and resource budget?

**Teaching claim:** System-level optimization follows measured task costs and critical paths, including unsuccessful work and the trade-offs introduced by each mechanism.

**Lasting takeaways**

1. **Count the complete task.** Inference, tools, retrieval, waiting, assessment, retries, and human effort determine useful completion cost; provider token prices are only part of one accounting model.
2. **Optimize what limits the objective.** Faster generation matters when generation limits the critical path or useful throughput; it can leave the task bottleneck unchanged.
3. **Cheaper computation can change behavior.** Routing, cascades, and reduced deliberation require outcome evaluation. Distribution-preserving generation acceleration is a different contract from changing the policy.
4. **Capacity and spending require aggregate accounting.** Live trajectories, active inference, retained state, and outstanding delegated reservations create different obligations; a small budget per child does not bound the sum of many children.

**Prerequisites:** Chapters 2–3 generation/cost and deliberation; Chapters 4–6 information/memory mechanisms; Chapter 11 scheduling and resource placement; Chapter 15 coordination overhead; Chapter 16 evaluation, trace attribution, and distributions. Chapter 1 already introduced whether autonomy is appropriate. No new GPU microarchitecture course is needed here.

**Assessable learning objectives**

| ID | Objective | Sections |
|---|---|---|
| 17-LO1 | Construct task-level cost accounts that include unsuccessful attempts, assessment, and human intervention. | 17.1 |
| 17-LO2 | Identify a trajectory's critical path and estimate the benefit of a proposed optimization. | 17.2 |
| 17-LO3 | Compare model-routing and deliberation policies under common quality, latency, and cost constraints. | 17.3 |
| 17-LO4 | Select capacity and budget controls using workload variation, retained state, and outstanding work. | 17.4–17.5 |
| 17-LO5 | Defend a complete execution design using measured trade-offs and sensitivity to uncertain inputs. | 17.6 |

**Exit capability:** Justify an end-to-end optimization and its economic value while identifying the measurements and assumptions that could change the decision.

**Section sequence and argument**

#### 17.1 Accounting for a complete task

- Build a complete task cost account covering inference, retrieval, tools, workspace lifetime, assessment, retries, and review. Contrast it with the narrower model-call bill.
- Distinguish customer billing, provisioned resource expense, opportunity cost, and organizational development/maintenance effort.
- Define cost per acceptable completion for a cohort and report failures/abstentions explicitly rather than hiding them in the denominator.
- Record quality, latency, intervention, and cost together, including approval waiting and reconciliation when the task requires them.

**Purpose and order:** Start from the objective and accounting boundary. The chapter applies Chapter 16's metrics instead of inventing a universal unit combining dollars and accelerator-hours.

#### 17.2 Finding the critical path and dominant demand

- Read the existing trajectory trace as sequential and overlapping work; separate elapsed time from summed resource time.
- Identify model generation, tool execution, retrieval, queueing, or human interaction on the limiting path.
- Estimate a local optimization's maximum system effect, including a possible bottleneck shift.
- Apply context/cache/scheduling choices from their primary chapters without rederiving attention arithmetic or the roofline.

**Purpose and order:** Complete accounting identifies what is expensive; the critical path identifies what controls time. Both are needed before choosing an optimization.

#### 17.3 Choosing computation and model routing

- Compare one model, task/step routing, and a bounded escalation cascade using the same acceptance criteria.
- Allocate deliberation, candidate generation, checking, and retry budgets; refer to Chapter 3 for mechanisms.
- Include router error, verifier false acceptance/rejection, escalation latency, and hard cases after an earlier failure.
- Choose among quality–latency–cost alternatives using workload-specific results, accounting for imperfect automated assessment.

**Purpose and order:** This is the first optimization decision at the policy level. It makes explicit that changing which model runs or how much reasoning occurs can change behavior.

##### Optional worked example: accelerating token generation

- Introduce speculative decoding as a bounded mechanism: draft several tokens, score with the target, accept a valid prefix, and continue from an appropriate correction.
- State the distribution-preservation contract under the chosen exact algorithm, sampling rules, and numerical assumptions; distinguish it from semantic correctness and from approximate variants.
- Account for draft time, target verification time, accepted-prefix length, cache updates, memory demand, and rejected work.
- Compare measured throughput and task latency at different serving loads; disable or change speculation when overhead outweighs advancement.

**Purpose and order:** Contrasting this mechanism with 17.3 prevents model cascades, task search, and token speculation from becoming one vague form of “speculation.” This is the primary home of detailed speculative decoding migrated from Chapter 2. Keep it a compact worked mechanism; rejection-sampling proof and optimized kernels are supplementary, not a new chapter-length detour.

#### 17.4 Matching capacity to demand

- Distinguish task arrival, active model calls, tool waits, paused tasks, and retained inference/task state.
- Characterize service-time and memory-demand distributions, burstiness, deadlines, and admission limits.
- Compare pooled provisioned capacity with usage-based service under measured duty cycle and stated pricing assumptions.
- Select headroom from observed workload/service objectives; apply Chapter 11 scheduling and placement rather than repeating their implementation.

**Purpose and order:** A faster or cheaper task policy changes demand on shared resources. Capacity must now be sized for the resulting workload rather than a universal utilization percentage.

#### 17.5 Governing resource expenditure

- Distinguish a rate limit, concurrent-work limit, per-task budget, and aggregate spending limit.
- Reserve budget for outstanding calls and delegated work; debit, settle, refund, and handle unknown final costs.
- Propagate cancellation or degraded service before budget exhaustion; preserve sufficient budget for safe completion or escalation.
- Use concurrent delegated operations to show why each child spending less than its parent does not bound total spending.

**Purpose and order:** Capacity planning anticipates demand; runtime accounting handles what is actually admitted and still outstanding. This section applies authority/control to resource obligations.

#### 17.6 Choosing a complete design

- Compare candidate architectures under a supplied workload contract using the preceding measurements. State the task inputs and operating assumptions locally; the conclusion concerns system cost and acceptable outcomes.
- Include maintenance and evaluation cost, integration effort, unsuccessful attempts, and human intervention.
- Vary task difficulty, usage volume, model prices/resource costs, and evaluation uncertainty to find where the choice changes.
- Revisit fixed workflow, bounded agent execution, and human-assisted alternatives without universal forbidden regimes.

**Purpose and order:** The final choice resolves the opening economic question. Chapter 18 can reuse the same comparison while integrating all component contracts.

**Culminating design problem and closure**

Provide a mixed workload with per-phase traces, completion outcomes, API/provisioned cost options, and a modest interactive latency target. Students choose routing, a generation policy, capacity, and spending controls; show an aggregate reservation ledger after subagents are dispatched; and examine sensitivity to assessment costs and approval delays. Assess LO1–LO5. Require one optimization they decline because its local gain has little task-level benefit.

**Calculation designs**

1. **Complete-task cost and critical-path sensitivity.** Inputs: timestamped dependency graph, direct operation costs/resource times, accepted outcomes, failures, retries, interventions, and a stated accounting window. Compute cohort cost per accepted outcome and critical-path elapsed time; accelerate one measured component and recompute. State fixed task mix, whether overhead changes, and whether quality remains constant. The decision is which subsystem to optimize first. Do not sum overlapping spans as elapsed time or infer retry-until-success cost from bounded attempts without specifying the policy.
2. **Two distinct computation comparisons.** Main routing example: small-model cost `Cs`, evaluator cost `Cv`, escalation probability `q`, fallback cost conditioned on escalation `Cf`, plus joint correctness/evaluator outcomes; compare `Cs + Cv + q*Cf` with a direct route under a quality threshold. Optional speculative-decoding subexample: measured draft-plus-target round time `Tr`, mean tokens advanced `a`, baseline per-token time `Tb`, and memory/load context; compare `Tr/a` with `Tb`, including correction tokens consistently. These are separate contracts and separate observations, not one universal acceptance-rate threshold. The decision is which mechanism to enable for the workload.
3. **Capacity and aggregate reservations.** Inputs: measured task arrivals, service and waiting distributions, retained-state bytes, service objectives, capacity price, and outstanding spend reservations. Use recorded workloads or a clearly specified simulation to compare pool sizes; use average population/arrival/time accounting only under stable conditions. Then present a concrete budget ledger in which parent allowance is reserved across children and final costs settle once. The decisions are capacity/headroom and the admission of another task or child. No mandatory 65–70% utilization and no unqualified geometric-tree bound.

**Coverage, additions, and removals**

| Existing source | Disposition and target |
|---|---|
| `sec-vol3-tokenomics-intro`, `sec-vol3-tokenomics-roofline` | Rebuild around 17.1–17.2. Existing roofline/performance mechanics are applied via prior chapters; demote hardware generation tables, “Rule of 600,” and categorical decode laws. |
| `sec-vol3-tokenomics-prefill-decode`: multidimensional attribution | Keep in 17.1, adding evaluation, intervention, failed attempts, and accounting boundaries. Remove claim that provider price spread is dictated solely by physics. |
| `sec-vol3-tokenomics-prefill-decode`: prefill arithmetic, TTFT, KV footprint, compounding turn tax | Retain only task-level consequences in 17.2. Mechanisms live in Chapters 2, 4, 5, and 11; avoid rederivation. |
| `sec-vol3-tokenomics-prefill-decode`: prefix caching, commercial discounts, compression | Apply measured costs in 17.1–17.3. Separate billed token categories from physical work and include compaction quality loss/overhead. |
| `sec-vol3-tokenomics-prefill-decode`: CPVG and `eq-vol3-tokenomics-cpvg-exponential` | Keep explicit cohort outcome accounting. Demote bounded illustrative retry model only with assumptions; remove `p^N` as a general agent reliability/cost law. |
| `sec-vol3-tokenomics-fleet-sizing`: workload, queueing, provisioned versus on-demand | Keep in 17.4. Replace fixed reserve law with measurements/simulation and workload assumptions. Mean-delay approximations do not guarantee p99 service levels. |
| `sec-vol3-tokenomics-fleet-sizing`: prefill/decode disaggregation | Refer to Chapter 11; include its measured total cost only when comparing capacity choices. |
| `sec-vol3-tokenomics-cascades`: hierarchy, economics, speculative routing, Pareto | Rebuild as 17.3; separate task/model routing from exact token speculation. Remove guaranteed savings and universal required tier count. |
| `sec-vol3-processor-speculative-decoding` (incoming from Chapter 2) | Primary home 17.3. Reuse draft/verify/rejection example after checking assumptions. Supporting proof only if needed; do not repeat roofline or claim semantic correctness. |
| `sec-vol3-tokenomics-boundaries`: modality utility and five forbidden regimes | Basic autonomy choice is in Chapter 1; measured economic revisit in 17.6. Remove fixed human “near-zero blast radius,” deterministic code reliability restricted to {0,1}, universal 200 ms bound, and absolute exclusions. |
| `sec-vol3-tokenomics-boundaries`: economic governance, bucket limits, monotonic budgets | Keep aggregate accounting in 17.5. Correct branching-budget error; vendor-specific lock-free/Redis timing belongs outside the main argument. |
| `sec-vol3-tokenomics-fallacies`, `sec-vol3-tokenomics-summary` | Rebuild from measured trade-offs; discard universal speculative acceptance threshold, fanout geometry, and utilization mandates. |

**Material gaps to add:** quality-constrained cost denominator; direct versus attributed costs; critical path with concurrency; model-routing mistakes and evaluator errors; serving load changing speculative benefit; aggregate reservation ledger; maintenance/evaluation expense; sensitivity to usage/workload. The chapter has six main sections. Keep policy-changing allocation and exact token-generation acceleration conceptually distinct within 17.3; the latter is an optional worked mechanism rather than a required chapter-level competency.

### Synthesis

The closing chapter selects and integrates mechanisms against a supplied workload contract. Its structure follows design decisions, with no requirement to finish a book-long application story.

### Chapter 18. Designing the Stochastic Computer

**Current manuscript:** [18_conclusion](18_conclusion/18_conclusion.qmd)

**Central point:** A defensible agentic system combines compatible component contracts and evidence that their interaction satisfies the workload's requirements.

**Prerequisites and placement:** All earlier components and evaluation methods are available. This chapter applies them to a complete design. It introduces no new mandatory architecture, universal principle catalog, learning algorithm, or resource model.

**Lasting takeaways**

- **A complete design connects its contracts.** Context, action, authority, state, recovery, and completion must refer to the same task and environment assumptions.
- **Requirements select the necessary mechanisms.** The appropriate architecture depends on the workload and may use an existing model, a single agent, or a tightly bounded workflow.
- **Evidence determines the next intervention.** Improving memory, tools, control, learning, or coordination should address a diagnosed limitation and survive task-level evaluation.
- **A justified system states its limits.** Unmeasured outcomes, incomplete checks, and unresolved external effects remain part of the design judgment.

**Learning objectives and assessment alignment**

| Objective | Supporting sections | Evidence of learning |
|---|---|---|
| Specify a workload contract with measurable outcomes and bounded authority. | 18.1 | Complete requirements and evidence specification. |
| Synthesize model, memory, tool, and runtime choices into a compatible architecture. | 18.2–18.3 | End-to-end architecture with explicit interfaces and state ownership. |
| Justify learning and coordination choices against a diagnosed limitation. | 18.4 | An intervention comparison including an unchanged-system baseline. |
| Evaluate complete designs using task outcomes, failure behavior, latency, and cost. | 18.5 | Calculations and evidence under common assumptions. |
| Defend a design's operating envelope and unresolved limitations. | 18.6 | A design review and proposed discriminating experiment. |

**Ordered body sections**

#### 18.1 Specify the task and its operating envelope

Specify a digital task with constraints on data access, tool use, completion, resource budgets, and recovery. Supply the domain facts needed for a compact worked design; the assessed problem should change the task or operating conditions enough to require independent judgment. Name the evidence needed before increasing authority or workload scale. The section establishes a requirements contract.

**Contribution and transition:** Establish the yardstick for the capstone. Computation and information choices can now be evaluated against explicit needs.

#### 18.2 Choose the computation and information design

Select a model invocation contract, deliberation policy, context policy, inference-state policy, and persistent-memory policy. Explain what each stores, what it exposes to a decision, and when it becomes invalid. Compare a simple baseline with a targeted alternative, such as additional retrieval or bounded candidate checking.

**Contribution and transition:** Define how decisions receive information and computational effort. The next section connects those proposals to actual effects and managed execution.

#### 18.3 Choose the action and execution design

Specify tools, permissions, isolation, durable records, recovery, supervision, and scheduling. Walk through a normal run and an interrupted action with an uncertain outcome. Ensure cancellation, approval, restoration, and resumption have compatible meanings. Account for the host OS and external services rather than assigning every responsibility to one agent runtime.

**Contribution and transition:** Assemble a complete execution system. Its remaining limitations determine whether learning or delegation is warranted.

#### 18.4 Decide whether to adapt or distribute

Use an observed failure category to compare changing context or tools, supervised adaptation, environment-feedback learning, and delegation. Carry forward evidence and costs from earlier chapters. Justify one intervention and explain the circumstances under which retaining the existing model and execution topology is the better engineering decision.

**Contribution and transition:** Prevent a capstone that mechanically includes every mechanism taught. The resulting designs can now be evaluated as competing complete systems.

#### 18.5 Evaluate, diagnose, and improve the whole system

Compare designs using the same tasks, repeated-run procedure, acceptable-outcome criteria, and accounting boundary. Apply earlier critical-path and cost-per-completion calculations. Reconstruct one cross-component failure and identify the smallest experiment capable of distinguishing its plausible causes. Demonstrate how improving a local metric can change another system requirement.

**Contribution and transition:** Supply evidence for architectural choices. The closing section turns that evidence into a defensible engineering position.

#### 18.6 Defend the design and its limits

Present the completed architecture and explain its envelope, tradeoffs, and remaining uncertainty. Test which choices remain justified when a task constraint or operating condition changes. End with durable open questions about evaluation, state validity, control, and adaptation. Bound the discussion of physical AI to the change in consequences when actions affect the physical world.

**Contribution and closure:** Students finish by defending a system, interpreting its evidence, and identifying the next useful experiment. The final synthesis returns to the stochastic computer as a functional architecture.

**Quantitative work:** Reuse the memory, recovery, scheduling, evaluation, and cost methods developed earlier. Compare two complete designs with common workload inputs and measurement boundaries. Do not introduce a new composite score that conceals an unacceptable authority or outcome failure. Show a tradeoff table when no design dominates all requirements.

**Culminating assessment:** A complete design dossier containing workload contract, architecture, interface/state ownership, failure and intervention walkthroughs, budget calculations, evaluation results or experiment design, and justified next intervention. Include an oral or written defense for an unfamiliar digital task or a materially changed operating envelope. Supply necessary domain knowledge so the assessment tests systems transfer, not prior expertise in the application. Students must select and adapt the methods rather than reproduce a practiced solution. Map each design claim back to a result established in the book.

**Coverage decisions:** Replace the repeated five-part synthesis and timeless-law catalog in `sec-vol3-conclusion-synthesis` and `sec-vol3-conclusion-invariants` with the worked design. Fold the useful improvement and autonomy material into applications of Chapters 12 and 8/10/16. Condense `sec-vol3-conclusion-frontiers` and `sec-vol3-conclusion-challenges`; omit an extended robotics syllabus. Treat `sec-vol3-conclusion-hourglass` as an optional comparison if it illuminates the design, rather than introducing a second compulsory organizing architecture.

**Visual plan:** Reuse the introductory functional architecture with the capstone's choices annotated. Rework the existing five-part composition to show the agreed six responsibilities. Retain the conclusion cover's base artwork if its generic labels are revised. Move the improvement/autonomy figures to their main teaching homes and reference them here only when they support a capstone decision.

## 6. Coverage audit and editorial decisions

### Coverage of lasting systems competencies

The outline has a defensible systems foundation. [MIT's Computer Systems Engineering curriculum](https://web.mit.edu/6.1800/2025/wwwdocs/lectures.shtml) develops modularity, naming, memory, concurrency, resource management, reliability, transactions, logging, isolation, and security. [Berkeley's LLM Agents course](https://github.com/rdi-berkeley/llm-agents-mooc/blob/main/f24.md) includes reasoning, planning, tools, retrieval, infrastructure, evaluation, privacy, human interaction, and coordination. The comparison supports the topic families below. It does not establish a canonical agent architecture, validate this particular chapter order, or constitute either institution's endorsement.

The following is an editorial coverage judgment. It distinguishes the enduring competency from the implementations that may be replaced. Each row should be inspectable in the chapter's objectives, explanation, and assessment; a keyword in a heading is insufficient.

| Durable competency | Teaching home | Evidence an expert should examine |
|---|---|---|
| Specify a task and decide what evidence would establish an acceptable outcome. | 1; applied in 10, 16, 18 | Requirements include ambiguity, affected users, unacceptable effects, and conditions for asking, abstaining, or escalating. Success is not reduced to an attractive final message. |
| Make uncertain learned computation usable through an explicit interface. | 2–3 | The student separates proposal probability, response validity, task evidence, additional computation, and stopping decisions. Planning and feedback are taught before elaborate search methods. |
| Manage information and state across different lifetimes. | 4–6 | Context selection, inference representations, persistent records, provenance, and invalidation have distinct roles. A student can size and compare alternatives without assuming literal cache-tier equivalence. |
| Design useful action and observation boundaries. | 7 | Tool granularity, schemas, failures, identifiers, observations, and completion semantics appear in a coherent interface design. A protocol does not substitute for its contract. |
| Enforce authority and account for people affected by execution. | 8, 10; applied in 12, 16 | Permissions, untrusted inputs, isolation, information release, review, steering, and intervention have operational meanings. Learned preferences do not replace access controls, and retained data has a stated purpose and scope. |
| Preserve and control execution despite concurrency and partial failure. | 9–10 | Durable records, pending effects, retries, reconciliation, compensation limits, cancellation, and resumption fit together under a stated failure model. |
| Allocate finite resources under variable demand. | 5, 11, 17 | Memory capacity, data movement, queueing, waiting, admission, fairness, and end-to-end costs lead to justified design decisions. Examples reveal when a local optimization changes the system bottleneck. |
| Improve a policy using appropriate data and feedback. | 12–14 | Data coverage and lineage, target construction, distribution shift, reward quality, delayed credit, and rollout freshness support an independently evaluated adaptation decision. Training is optional when another intervention better addresses the limitation. |
| Coordinate work while preserving ownership and integration. | 15 | A student can justify decomposition, communication, concurrency, and synthesis against a single-agent baseline, including shared failures and correlated errors. |
| Evaluate, diagnose, and operate a changing system. | 1 introduces; 16 develops | Task sampling, repeated runs, assessor limitations, records, uncertainty, regression testing, version changes, and release decisions connect measurement to action. Inspect relevant user/task strata so an aggregate improvement does not hide a materially worse subgroup. |
| Transfer architectural judgment to a new setting. | 18 | An unfamiliar task or materially changed envelope requires selecting mechanisms and defending tradeoffs, rather than reproducing the worked design. Necessary domain facts are supplied. |

The transfer requirement follows the distinction in [Wiggins and McTighe's assessment guidance](https://files.ascd.org/pdfs/publications/books/UBD_Guide_AdvancedConcepts_downloads.pdf): completing a rehearsed performance does not by itself demonstrate the ability to adapt knowledge to a new problem. A learner pilot should test whether students can explain their choices and recognize when a mechanism is unnecessary.

The structure is ready for expert scrutiny, with three questions still requiring judgment during implementation: whether the treatment has enough depth to support the assessments; whether boundaries such as context/persistence and recovery/control avoid duplication; and whether contemporary mechanisms are presented with appropriate evidence and scope. Systems, ML learning, and evaluation/human-interaction expertise should all be represented in that scrutiny. No actual external expert approval is claimed by this planning pass.

### Preserving the ML systems identity

Each chapter should develop a constraint, a component responsibility, the relevant mechanisms, and a justified design decision. Quantitative work accounts for memory, computation, movement, latency, capacity, learning signal, uncertainty, or cost when it helps that decision. Interface and recovery chapters can use precise contracts and failure traces instead of an unnecessary equation. The stochastic-computer map gives the components a memorable relationship; it does not establish their properties by analogy.

The manuscript would lose this identity if it became a sequence of application-building tutorials, prompt recipes, framework instructions, or descriptions of novel “agent” components without evidence. Concrete contracts, traces, local examples, and measurements should make the engineering argument inspectable. These are drafting and review criteria; an outline alone cannot establish how the completed prose will read.

### Gaps that the detailed plan closes

| Needed competency | Where it is taught | Why it needs an explicit place |
|---|---|---|
| Translate an ambiguous request into a bounded task and completion criteria. | 1.2, 1.5–1.7; supervision in 10 | A student cannot select an architecture without knowing what outcome and authority it must support. |
| Distinguish observations, model context, recorded state, and the actual environment. | 1.2; 4–6; 9 | Conflating these produces misleading claims about memory, recovery, and certainty. |
| Compose model calls with ordinary program control and revise a plan. | 2–3 | The book needs single-agent programming and planning before distributed delegation. |
| Reason about evidence and comparison before advanced evaluation. | 1.6; applied in 3 and 12–15; developed in 16 | Evaluation cannot first appear after chapters already depend on quality comparisons. |
| Preserve instruction meaning, provenance, and uncertainty during context transformation. | 4 | Token reduction alone does not establish a useful context policy. |
| Manage memory writes, stale records, contradictions, and retrieval failures. | 6 | A retrieval chapter must cover record lifecycle and task usefulness as well as index selection. |
| Define action/observation semantics, including partial and ambiguous completion. | 7; recovery in 9 | Calling a tool requires a contract for what the runtime can infer from its response. |
| Separate permission, containment, successful execution, and task correctness. | 7–8; applied in 10 and 16 | These are different responsibilities and need different evidence. |
| Handle human steering, approval validity, and resumption after the world changes. | 10 | Human supervision is part of the execution protocol and affects system behavior. |
| Allocate resources at trajectory, request, and inference-iteration levels. | 11 | An agent's lifetime spans computation, waiting, and external service use. |
| Diagnose whether adaptation is the appropriate intervention. | 12; applied in 13–14 and 18 | Improving data, context, tools, or runtime can address a failure without changing model weights. |
| Evaluate feedback quality, training-data coverage, and execution-time behavior. | 12–14; experimental depth in 16 | Training loss and reward alone cannot establish better complete-task behavior. |
| Define ownership, integration, and termination in delegated work. | 15 | Parallel generation leaves coordination and synthesis work that the architecture must assign. |
| Measure and operate releases under uncertainty. | 16 | Offline scores need a connection to versions, incidents, monitoring, and decisions about deployment. |
| Account for unsuccessful work, tool costs, and human attention. | 1.6, 10–11, 17 | Optimizing token cost alone omits resources that can govern acceptable task completion. |
| Integrate the book's mechanisms selectively. | 18 | A capstone must justify compatibility and necessity, rather than reproduce the table of contents. |

### Material to consolidate, demote, or remove

- **Repeated historical or architectural overviews:** keep the whole-machine orientation in Chapter 1 and enter the concrete problem directly in later chapters.
- **Literal CPU microarchitecture:** remove fetch/decode/execute/writeback equivalences, neural ALU claims, and invented clock-cycle parallels that do not explain a real interface.
- **Literal L1/L2/L3 mapping:** replace with the separate roles of context, derived inference state, and persistent records. Keep actual paging and cache mechanisms where they explain allocation or reuse.
- **Universal “laws” built from conditional models:** retain a calculation only with its assumptions and decision purpose. Do not elevate independent-step reliability, optimal agent counts, fixed wait distributions, or deterministic-checkpoint claims into universal results.
- **Repeated runtime definitions:** Chapter 9 owns the durable trajectory record; Chapter 10 extends its lifecycle; Chapter 11 allocates resources against it. Later chapters refer to that shared state model.
- **Premature distributed mechanisms:** move multi-agent topologies, consensus comparisons, and integration issues into Chapter 15 after single-agent planning and execution are established.
- **Implementation catalogs:** use protocol standards, sandbox types, search algorithms, vector indexes, and training methods as selected mechanisms. A catalog is justified only if comparing alternatives is itself the learning objective.
- **Mandatory learning or autonomy ladders:** teach improvement selection from a diagnosed constraint. Cost and risk need explicit assumptions and may differ across interventions.
- **Repeated frontier surveys and robotics material:** concentrate enduring open questions at the end; keep physical action as a boundary of this volume's scope.
- **Unsupported visual claims:** correct the source graphic as well as its surrounding prose. An old graph, cover label, or caption cannot be treated as evidence for the new argument.

### Topics considered without adding chapters

| Candidate topic | Decision |
|---|---|
| Agent programming | Explicit model-call contracts, bounded control flow, planning, tool composition, and runtime state belong in Chapters 2–3 and 7–10. Add a separate chapter only if a distinct unserved competency remains after those revisions. |
| Prompt engineering | Teach instructions, examples, output contracts, and context construction where they affect system behavior. Exclude a growing catalog of prompt recipes. |
| Multimodal digital inputs | Include a bounded treatment of text, image/document, and structured observations at the model and tool interfaces. Encoding, observation size, provenance, and uncertainty matter; a full multimodal model course would change the volume's scope. |
| Security and human interaction | Explicitly owned by Chapters 8 and 10 and applied throughout. These need substantive sections, not a late warning box or an additional miscellaneous chapter. |
| Evaluation earlier in the book | Move the minimum contract to Chapter 1; retain Chapter 16 for experimental design, observability, diagnosis, and release practice. Revisit the placement if earlier chapters still require unexplained evaluation methods. |
| Data systems and memory | Chapter 6 owns persistent records and retrieval; Chapter 12 owns learning datasets. Keep their different consumers and validity requirements explicit. |
| A separate model-routing chapter | Chapter 2 defines the component contract; Chapter 17 compares complete routing/cascade designs. Separate only if the engineering argument cannot fit without crowding whole-task optimization. |
| Lifelong learning and self-modification | Discuss as bounded extensions and open questions after the data/adaptation mechanisms. Avoid treating autonomous self-improvement as required functionality or an established guarantee. |
| Deployment orchestration | Apply existing systems knowledge in Chapters 11 and 16. Include version compatibility and state migration when they affect agent execution; omit infrastructure setup tutorials. |

The current evidence supports retaining eighteen chapters. This is an editorial judgment based on distinct teaching responsibilities, not a target chapter count. The closest boundaries to recheck during drafting are 4/6 (context versus retrieved records), 9/10 (recovery versus intervention), 11/17 (allocation mechanisms versus whole-system optimization), and 12/16 (learning evidence versus system evaluation). If either side merely repeats the other, move the repeated content and shorten the chapter.

### Supporting material and its teaching role

Supporting material should help readers apply or extend an argument already established in the body. It must not contain an undeclared prerequisite needed to complete a chapter's assessment.

| Existing material | Revision role |
|---|---|
| [Reference architecture](appendices/app_a_reference_architecture.qmd) | Align with Chapter 18's defended design. Show responsibilities, interfaces, and alternatives; keep replaceable product choices out of the architectural contract. |
| [Tool design](appendices/app_b_tool_design.qmd) | Supply detailed interface examples and review aids for Chapters 7–8. The core action, observation, and authority contracts remain in those chapters. |
| [Failure taxonomy](appendices/app_c_failure_taxonomy.qmd) | Support diagnosis in Chapters 9 and 16 with observed symptoms, evidence requirements, and recovery implications. Avoid presenting a list of labels as an explanation of causation. |
| [Mathematical foundations](backmatter/appendix_math.qmd) | Retain only derivations needed by a scoped body example or a clearly optional extension. Audit assumptions and conclusions before relocation; an unsupported claim does not become suitable by moving to an appendix. |
| [Glossary](backmatter/glossary/glossary.qmd) | Update definitions as chapters are accepted, using each concept's owning chapter. Distinguish established systems terms from the book's limited architectural analogy. |

## 7. Quantitative reasoning that earns its place

Every chapter should include a worked quantitative decision where quantities improve the judgment. The right amount varies by the problem. Interface or authority chapters may center on a contract, state transition, or failure trace with a supporting budget calculation. Memory, scheduling, learning, and cost chapters can carry more arithmetic.

A worked calculation specifies the workload, defines variables and units, states assumptions, computes the comparison, checks the result, and explains which decision changes. Follow a dense derivation with interpretation or a return to the workload. Keep extended derivations in supporting material when the chapter only needs their consequence.

| Family of reasoning | Typical use | Assumption that must remain visible |
|---|---|---|
| Resource accounting | Calls, tokens, context occupancy, data yield, memory, tool use, and human review | What is included, whether work overlaps, and the denominator of the outcome metric. |
| Capacity and data movement | KV allocation, shared prefixes, transfer/recompute choices, concurrent tasks | Model architecture, representation, effective bandwidth, valid sharing, and reserved memory. |
| Latency and queueing | Deliberation, tool waits, cancellation, scheduling, critical path | Workload distribution, concurrency, service capacity, contention, and steady-state assumptions where used. |
| Learning objectives and feedback | Masked loss, weighting, policy updates, rollout throughput | Which decisions receive signal, source of reward, data separation, and policy/environment versions. |
| Reliability and evaluation | Recovery traces, repeated runs, uncertainty, acceptable outcomes | Failure model, dependence among observations, completeness of checks, and experimental sampling unit. |
| Economics and tradeoffs | Cost per acceptable completion, routing, capacity, delegation | Failed work, tool charges, infrastructure accounting, human attention, and comparable outcome criteria. |

Symbolic calculation briefs in the chapter cards are plans for worked examples, not new empirical claims. Before publication, source factual inputs, reproduce numerical results through the book's calculation infrastructure, and check that a result supports the surrounding conclusion. This planning pass does not certify the existing manuscript's equations or citations.

## 8. Drafting and integration workflow

### Keep the fresh argument independent of the old wording

Maintain two inputs with different audiences:

- **Writer packet:** approved chapter takeaways, objectives, assessment, section sequence, prerequisite ledger, local example assumptions when needed, factual evidence, calculation brief, applicable editorial rules, and only already accepted new prose needed for continuity.
- **Editor ledger:** the old manuscript, source-anchor migration map, figure reuse audit, duplication inventory, and rejected claims.

The editor converts useful old material into a reviewed factual or visual brief. The writer does not receive the old chapter, a “rewrite this section” prompt, unsanitized figure labels, or old draft histories. Unverified legacy claims remain outside the writer packet. Existing images enter only after their teaching content has been reviewed; otherwise the writer receives a description of the intended visual relationship.

Fresh wording is insufficient by itself. The review must verify that the section builds the new argument and avoids importing the old sequence, unsupported guarantees, or named-law framing.

### One chapter, one section, one acceptance decision

1. Freeze the current chapter's takeaways, assessable objectives, prerequisite requirements, and culminating problem.
2. Prepare the first section's packet from those decisions. Give it the chapter-wide map so local prose serves the whole argument.
3. Generate one body section in a fresh writing context. Supply the accepted preceding section's ending and an updated concept ledger for continuity.
4. Review its claim, prerequisites, evidence, calculations, explanation, and handoff. Revise that section before treating its concepts as established for the next one.
5. Integrate the accepted prose and a reviewed figure through the editor. Preserve or deliberately migrate anchors, and update captions and alt text together with the figure.
6. Continue sequentially through the chapter. Use parallel reviewers for evidence, mathematics, figures, and coverage; keep the prose sequence under one editorial owner.
7. Revisit Purpose and learning objectives, then write the final takeaways, fallacies, assessment materials, and connection from the completed argument. Backward design sets the targets; the completed chapter verifies whether it met them.
8. Review the assembled chapter for continuity and completeness. Before publication, render and inspect it, and check references, calculations, and visual integration.

Begin with Chapter 1 to establish the introduction's terminology, component responsibilities, and voice. After that first review, independent chapter workers may run concurrently from the agreed prerequisite contracts and book-wide vocabulary. Each chapter has one owner and advances strictly one section at a time; concurrency across chapters never permits bulk chapter generation or concurrent writing of dependent sections within a chapter.

After the selected chapters finish, a separate integration reviewer checks each complete chapter, neighboring chapter handoffs, and the volume-wide progression. Findings identify the affected chapter and section. Any prose revision returns to that section's writer, followed by renewed review of affected downstream dependencies. Integration is a coherence pass over accepted work, not a request to regenerate whole chapters.

### A section packet must answer

- What chapter takeaway and learning objective does this section support?
- What does the reader already know at this point?
- What concrete problem opens the section?
- What concepts or mechanisms must be established, and in what internal order?
- Which claims have evidence, and which need research before drafting?
- Which calculation or example changes an engineering decision?
- Which local example, if any, makes the mechanism or decision clearer? Are its necessary assumptions supplied without depending on an earlier application story?
- What figure relationship helps the explanation, and which approved asset can provide it?
- What can the reader now conclude, and what unresolved question motivates the next section?

### Publication coherence

After a chapter's argument is accepted, update its associated objectives, takeaways, exercises/quizzes, glossary entries, concept maps, citations, notation, cross-references, figure captions, and alt text. Apply the same decisions to part openers, the margin stack, introduction/conclusion maps, front matter, and book-format configurations. The current configurations still contain old part counts and literal memory/verification descriptions; those must change when the manuscript architecture changes.

Keep the tool-specific execution protocol and private agent configuration outside the manuscript repository. The requested writing-model workflow must enforce the approved input boundary and verify its effective tool exposure before production drafting. A prompt asking a tool-enabled agent to ignore old files is not sufficient evidence that the boundary exists.

## 9. Completion criteria for this plan's implementation

The implementation is complete when each chapter has an identifiable teaching result; every objective is taught and assessed; every section contributes to the result; all prerequisites are available at their point of use; calculations support decisions under stated assumptions; and local examples make the reasoning concrete without requiring a selected book-wide archetype.

At the book level, the introduction must establish the complete system, the six parts must develop distinct responsibilities, and the capstone must demonstrate how those responsibilities work together. A reader should be able to explain the architecture and defend a design after implementation details and current product names have changed.
