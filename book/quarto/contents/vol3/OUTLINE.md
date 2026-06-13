# Volume III — Topic Outline (sketch)

*One tight block per chapter. Each names the **durable anchor** (the fundamental it teaches), what
is **out of scope** (the moving surface it must not chase), and flags the two **open-frontier**
chapters that are taught as problems, not answers.*

---

## Part I — The Inference-Time Frontier

**1. From Trained Models to Acting Systems** `introduction/`
The thesis chapter and the feasibility probe. Capability moved into training-time compute (Volume II);
the frontier now moves into inference-time compute, and an agent is what spending that compute at
runtime looks like as a system.
- *Anchor:* the bitter lesson and scaling laws, continued one axis further (train-time → test-time compute).
- *Out of scope:* what agents "will become"; any product's definition of "agent."

**2. The Control Loop** `control_loop/`
Strip the hype and an agent is an old structure: perceive, decide, act, observe, over state and an
environment. Robotics, RL, and control theory already named this; the model is a new policy in a
loop that is decades stable.
- *Anchor:* the sense-plan-act loop as the invariant beneath every agent architecture.
- *Out of scope:* named agent "patterns" presented as if they were primitives.

**3. Inference-Time Scaling** `inference_scaling/`
Search, sampling, and self-correction are all ways of trading runtime compute for capability. The
empirical regularity is real; the clean "laws" are still emerging, so they are framed as a frontier
with shape but not yet exponents.
- *Anchor:* compute-for-capability at inference, as the successor to the training-time scaling laws.
- *Out of scope:* leaderboard numbers; the specific reasoning models that demonstrated the effect.

## Part II — The Substrate

**4. State and Memory** `state_memory/` — **open frontier**
What an acting system knows, across a long horizon, is a memory-hierarchy problem: what to hold in
context, what to retrieve, what to externalize, what to evict. The *problem* is durable and maps onto
classic CS; the *resolved architecture* does not exist yet, and the chapter says so.
- *Anchor:* caching, locality, and eviction as the lens on agent state.
- *Open:* long-context vs. retrieval vs. external memory is unsettled; taught as the open question.

**5. Tool and Environment Interfaces** `tool_interfaces/`
The boundary between the model and the world, expressed as a typed action surface with sandboxing
and effects. The wire protocol of the month churns; the *concept* of a typed, permissioned action
interface is durable and is what gets taught.
- *Anchor:* the action interface as a contract between policy and environment.
- *Out of scope:* any specific tool-calling protocol or schema standard by name.

**6. Orchestration and Multi-Agent Systems** `orchestration/`
Many agents is distributed systems with stochastic nodes. Coordination, consistency, failure
detection, and recovery are fifty-year-old problems wearing new clothes; the novelty is that the
nodes are nondeterministic.
- *Anchor:* distributed-systems coordination under nondeterministic participants.
- *Out of scope:* specific multi-agent frameworks and their role taxonomies.

## Part III — Reliability and Economics

**7. Error Compounding and Reliability** `reliability/`
A single step's error rate compounds over a trajectory. This is a measurable systems property: how
reliability decays with horizon length, and what mechanisms (verification, checkpoints, recovery)
bound it.
- *Anchor:* trajectory-level reliability as a quantifiable, modelable property.
- *Out of scope:* anecdotes about specific failures of specific systems.

**8. The Economics of Autonomy** `economics/`
Agentic systems multiply inference calls; cost and latency become a budget engineered over a
trajectory. A direct extension of Volume II's serving-cost-dominance accounting, now with steps as
the multiplier.
- *Anchor:* multi-step inference cost/latency as an extension of the serving-cost laws.
- *Out of scope:* current per-token prices; vendor cost comparisons.

**9. Evaluating Acting Systems** `evaluation/` — **open frontier**
The hardest honest gap. Volume II could lean on loss curves as ground truth; here the output is a
trajectory and there is no settled measurement theory. The chapter teaches *why* it is hard and what
partial approaches exist, and refuses to pretend at a durable method.
- *Anchor:* the structure of the measurement problem for trajectory-shaped output.
- *Open:* no agreed method exists; taught as the field's central open problem.

## Part IV — The Responsible Agent

**10. Safety, Control, and Oversight** `safety_control/`
Containment, permissioning, and human-in-the-loop as engineering, grounded in control theory rather
than policy fashion: how to keep an autonomous loop observable, interruptible, and bounded.
- *Anchor:* controllability and oversight of a closed autonomous loop.
- *Out of scope:* current policy debates and governance frameworks by name.

**11. Security of Autonomous Systems** `security/`
The action boundary is a new attack surface. Injection through tools and inputs becomes a systems
vulnerability class, not a prompt-engineering footnote; taught as threat model and mitigation
structure.
- *Anchor:* the action surface as a securable boundary, with a durable threat-model framing.
- *Out of scope:* specific exploits-of-the-week against specific products.

**12. The Durable Core of Autonomy** `conclusion/`
Distill, do not repeat. What the practices of the book collapse into as durable principle, and an
honest ledger of what was deliberately left out because it is not yet settled (the open frontiers of
Chapters 4 and 9).
- *Anchor:* the principles that survive once this year's surface is gone.
- *Out of scope:* predictions; a victory lap over solved problems that are not solved.

---

### The go/no-go

Write Chapter 1 first, under the no-product-names rule. If it reads at the altitude of the Volume I
and Volume II introductions, the book is viable. If it can only describe what today's agents do, the
probe has answered: wait.
