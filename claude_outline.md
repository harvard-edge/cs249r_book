# Agentic Machine Learning Systems

Working outline, field survey, and resource-tracking design.

Drafted 2026-09-05. Untracked working document. Move it once the book has a repo home.

---

## 1. Thesis

**The model stops being the system.**

The unit of engineering moves from a trained artifact to a running process. A trained
artifact has weights, a fixed input and output contract, and one forward pass. A running
process consumes compute at inference time, takes actions with side effects, and behaves
according to its context, tools, memory, and control flow at least as much as its
parameters.

The consequence is that the discipline shifts from training-time optimization to runtime
systems engineering. Scheduling, state, isolation, observability, failure recovery, and
cost control become the core, not the operational afterthought.

**Second-order thesis, the one that produces a tradeoff table in every chapter.** Agentic
systems trade determinism for generality. Every engineering decision in the book is an act
of buying back some determinism at a price, and the book's job is to make that price
legible.

**Positioning against the existing volumes.** Volume I taught the bitter lesson. Volume II
taught scaling laws. This one teaches that scaling moved to inference time and dragged the
entire engineering surface along with it. That gives a three-book arc with a real
intellectual progression rather than three independent surveys.

## 2. Organizing principle

Chapters are named for invariants and abstractions, never for frameworks. MCP, LangGraph,
CrewAI, and whatever ships next quarter appear as instances of a concept, inside sidebars
and labs, never as the reason a chapter exists.

**The test for every chapter.** Would this still be true if the frontier model doubled in
capability tomorrow? If no, it is a sidebar, not a chapter.

**One spine artifact across the whole book.** Chapter 1 builds a working agent in roughly a
hundred lines. Every subsequent chapter breaks it and repairs it. Chapter 7 makes it serve
a thousand concurrent users. Chapter 6 injects tool failures. Chapter 10 attacks it. This
solves the usual textbook problem where labs feel bolted on, and it matches the way the
material is actually learned.

## 3. The outline

Twelve chapters plus a closer. One semester with room for project weeks.

The coverage column records what the September 2026 survey found. It is the reason to keep
this table rather than the earlier twenty-two chapter version.

### Part I. The Agent as a System

**1. From Models to Agents.**
Thesis, the autonomy spectrum, the control loop as an interpreter over a nondeterministic
instruction stream, the reference architecture, and the model properties that constrain
everything downstream (tokens as the resource unit, prefill versus decode, test-time
compute as the scaling axis). Introduces the spine artifact.
*Coverage: saturated. Every course opens here.*

**2. Context and Memory.**
The context window as a memory hierarchy. Packing, compaction, retrieval as paging,
grounding and citation, prompt caching as a cost primitive, attention budget and position
effects. Durable state, checkpointing, and what belongs in a file or a database rather than
in the context.
*Coverage: saturated. Context engineering is the field's own named canon and has explicitly
displaced prompt engineering in course descriptions.*

**3. Tools and Action Interfaces.**
Tool calling as an ABI. Schema and error design, idempotency, effect boundaries,
sandboxing, and code execution as the universal tool. Protocols appear here as instances.
*Coverage: saturated.*

### Part II. Control

**4. Planning and Orchestration.**
The workflow-to-agent continuum and the engineering rule for where to hardcode control flow
and where to delegate it. Decomposition, orchestrator and worker patterns, context
isolation, coordination cost, and an honest account of when fanning out loses.
*Coverage: saturated, usually taught as multi-agent design patterns.*

**5. Verification and Oversight.**
Verification is cheaper than generation, and that asymmetry defines the design space.
Critics, invariants, schema constraints, rollback and compensation for irreversible
actions, approval gates, interruption and steering, progressive autonomy, trust
calibration.
*Coverage: thin. Guardrails appear at Stanford and safety framing at Berkeley, but nobody
teaches verifier asymmetry as a first principle.*

**6. Reliability and Failure.**
A real failure taxonomy for agents, covering looping, context exhaustion, tool flakiness,
cascading hallucination, and partial action. Budgets, timeouts, retries and idempotency,
graceful degradation, chaos testing, and SLOs for probabilistic systems.
*Coverage: **white space**. Only CMU's ML in Production touches it, generically, from a
classical software engineering angle.*

### Part III. Running Them

**7. Agent Runtimes and Serving.**
Process model, isolation, durable execution, resumption. Then the serving half, which is
where agentic traffic differs from model traffic. It is bursty, stateful, long-tailed, and
cache-heavy. Prefix-cache-aware routing, admission control, multi-tenancy and fairness,
queueing for tail latency.
*Coverage: **near-white space**. Harvard CS2680 is the only peer, and it teaches model
serving beside agents rather than agent-shaped serving. The 2026 research literature on
agent-aware KV scheduling, session-centric scheduling, and cache lifetime across tool-call
pauses has not been turned into pedagogy anywhere.*

**8. Cost and Efficiency.**
Cost per resolved task replaces cost per token as the governing metric. Model routing and
cascades, caching layers, trajectory distillation, and energy accounting.
*Coverage: **white space**. Taught essentially nowhere.*

**9. Evaluation and Observability.**
Why static benchmarks break. Outcome versus trajectory evaluation, judge design and its
failure modes, task suites and environment reproducibility, variance and statistical power,
regression suites as CI. Then tracing a nondeterministic call graph, replay, cost and
latency attribution, and drift detection.
*Coverage: split. Evaluation is canon everywhere. Observability is absent everywhere.*

### Part IV. In the World

**10. Security.**
Prompt injection as the defining threat. The trifecta of private data, untrusted content,
and exfiltration capability. Least privilege, capability tokens, egress control, and the
tool supply chain.
*Coverage: well covered. A full course at UW, lectures at Berkeley and CMU.*

**11. Deployment and Operations.**
Shipping a nondeterministic system. Shadow and canary rollouts with statistical gates,
versioning model plus prompt plus tools as a single artifact, migration when the model
changes underneath you, on-call and incident response, audit trails and governance.
*Coverage: thin. CMU has the classical version. Nobody has the nondeterministic version.*

**12. Improving Agents.**
The improvement ladder, running from context to tools to workflow and only then to weights.
Supervised fine-tuning on trajectories, reinforcement learning from verifiable execution
feedback, environment and reward design as systems work, data flywheels, and when not to
train at all.
*Coverage: covered and contested. See the fault line in section 5.*

**Closer. Durable Principles.**
Distills what survives the next model generation. Not a recap.

### What was cut, and why

- **A standalone case-study chapter.** Better distributed as the spine artifact running
  through all twelve chapters. A case-study chapter is where vendor tours go to hide.
- **Embodied and edge agents.** Its two real ideas, irreversibility and hierarchical
  autonomy, live in chapters 5 and 7. As a single chapter it would have been exactly the
  thin technology-tour chapter the rest of the outline is designed to avoid. If embodied
  agents matter here, they are a separate book.

### Merges I would resist

- **Folding 8 into 7.** Cost per resolved task is the metric that reframes the field.
  Burying it inside a serving chapter demotes it to an optimization footnote.
- **Folding 6 into 5.** A ten-chapter version is achievable this way, but failure taxonomy
  stops being first-class, and it is the thing practitioners most need taught.

## 4. What universities are actually teaching

Surveyed 2026-09-05.

| Course | Angle | What it teaches |
|---|---|---|
| **Harvard CS2680**, Modern AI Systems: Agents and System Optimizations (Juncheng Yang) | Systems | Part I is agents from the user's then the designer's perspective. Part II is GPU kernels, serving with paging, batching and scheduling, KV cache, prefix cache, routing and load balancing, pruning and quantization, speculative decoding. Assignments build an agent, then serve it on open weights and optimize the full stack. |
| **CMU 11-768**, AI Agents (Graham Neubig, Fall 2026) | Research core | Capabilities (tool use, context management, skills, memory, planning), domains (coding, GUI, deep research), training (SFT, RL), safety (sandboxing, adversarial defense), interaction and frameworks. Assignments build a harness, build evals, train with RL, and a research project. |
| **Berkeley MOOCs**, f24 / sp25 / f25 (Dawn Song) | Seminar survey | Guest lecture format. The Fall 2025 edition explicitly moved from what agents are to how agentic systems are designed, evaluated, deployed, and governed. |
| **Stanford CS224G**, Building and Scaling LLM Applications | Studio | Four sprints to demo day. Context engineering, agentic patterns, orchestration, evals, data flywheels, safety assessment, pitching. |
| **UW CSE 599R**, Agentic Systems Security (Franziska Roesner) | Security | Classical systems security principles applied to agents, vulnerabilities in shipped systems, emerging defenses. |
| **CMU MLiP / AI Engineering** (Christian Kästner) | SE lifecycle | Requirements, planning for mistakes, testing, deployment, scaling, operations, security, safety, versioning and provenance, accountability. Agents enter as one component with an MCP lab. |
| **CMU 11-766**, LLM Applications | Application survey | Organized by application domain. Tool use, multi-agent, and deep research are individual weeks. |
| **NYU Stern**, Foundations of AI Agents; UCSD Extension; JHU; MIT Professional Education; Cornell | Professional | Six or so sessions. n8n or CrewAI or LangGraph, tool calling, RAG, evaluation, demo day. |

### Sources

- Harvard CS2680: https://course.agentic-system.org/
- CMU 11-768: https://www.cmu-agents.com/
- Berkeley Agentic AI MOOC F25: https://agenticai-learning.org/f25
- Berkeley Advanced LLM Agents SP25: https://agenticai-learning.org/sp25
- Stanford CS224G: https://web.stanford.edu/class/cs224g/schedule.html
- UW CSE 599R: https://courses.cs.washington.edu/courses/cse599r/26sp/
- CMU MLiP: https://mlip-cmu.github.io/s2026/
- CMU LLM Applications: https://cmu-llms.org/schedule/
- NYU Stern Foundations of AI Agents: https://pages.stern.nyu.edu/~ilobel/Foundations_of_AI_Agents.pdf
- Stanford CS329A, Self-Improving AI Agents: https://online.stanford.edu/courses/cs329a-self-improving-ai-agents
- UCSD Building Agentic AI Systems: https://extendedstudies.ucsd.edu/courses/building-agentic-ai-systems-cse-41415

## 5. What the survey shows

**There is no consensus core.** Four departments teach four different courses under the same
name. The field has not converged the way ML systems converged around training and serving.

**But a canon is emerging.** Five topics appear in essentially every serious course, which is
as close to settled as this gets:

1. The agent loop and tool calling
2. Context engineering, repeatedly described as having displaced prompt engineering
3. Memory
4. Planning and multi-agent architecture
5. Evaluation, and specifically why static benchmarks fail

Security is the sixth and converging fast, with prompt injection named as the threat rather
than a general gesture at safety.

**The unresolved fault line is whether students train the agent or only build around it.**
CMU makes RL training a required assignment. Berkeley's advanced track went deep on
reasoning and post-training. Harvard never trains anything and instead serves and optimizes.
Stanford CS329A is entirely about self-improvement. This is a genuine curricular division,
and no course frames it as a decision with tradeoffs. It maps directly onto chapter 12.

**The seminar format is a symptom, not a pedagogical choice.** Most of these courses are
guest lectures plus paper presentations plus a project. That is what a field looks like when
it has no textbook and no canon, so instructors teach from slides and arXiv links. This is
the opening.

**Framework-centered courses are dating fastest.** Anything built on a named orchestration
library is already showing its age. Neubig's framing survives best because scaffold,
evaluation, and training are skills rather than tools.

**Almost nobody bridges capability and systems.** Harvard CS2680 is the only course found
that does it structurally, by making students build an agent and then take responsibility
for the GPUs underneath it. It is worth noting this is in the same school.

### The white space

Consistently absent or thin across every course surveyed:

- Cost and energy as first-class engineering, with cost per resolved task as the metric
- Reliability engineering, failure taxonomy, budgets, degradation, SLOs
- Observability, tracing a nondeterministic call graph, replay, cost attribution
- The runtime as engineering rather than as a security concern, covering isolation, durable
  execution, checkpointing, resumption
- Serving agentic workloads specifically, as distinct from serving models
- Deployment of nondeterministic systems, canarying with statistical gates, versioning the
  composite artifact, model migration

## 6. Consequences for the book

**Adoption risk, and the response to it.** Teaching demand today is concentrated in the
canon, which is chapters 1 through 5 and the evaluation half of 9. A book that is mostly
runtime will be admired and not assigned. The winning shape teaches the canon better than
the seminars do in the first half, then delivers the systems half nobody else can teach.
The current outline already does this. The thing to protect is the first half, which must
not be compressed on the grounds that it is the obvious part.

**One structural addition worth considering.** Every course either trains the agent or does
not, and none of them treats that as a decision. A chapter that frames "should this agent be
trained at all, and where does that sit on the improvement ladder" as an engineering
question would be genuinely novel. That is chapter 12, and it should adjudicate the fault
line rather than survey RL methods.

**The strongest single differentiator** remains chapter 7. Every competing course stops at
chapter 5 or teaches serving as a separate subject. A systems textbook that treats agentic
serving as a scheduling and caching problem is the thing this author can write and others
cannot.

## 7. Resource tracking design

### The reframe

This is not a bibliography. It is the defense of a falsifiable claim. "Nobody teaches cost
per resolved task" is true today and decays every semester as syllabi are updated. What is
needed is a dated coverage matrix that reports when a differentiator begins to erode, not a
folder of links.

### Layout

A `curriculum/` directory in whichever repo the book lives in.

```
curriculum/
  courses.yml       # one entry per external course, chapter-tagged
  readings.bib      # papers, same chapter tags
  coverage.py       # renders the matrix and the white-space report
  CHANGELOG.md      # what each sweep found
```

### Entry shape

Each course carries a depth marker per chapter, not just a link.

```yaml
- id: harvard-cs2680
  institution: Harvard SEAS
  number: CS 2680
  title: Modern AI Systems - Agents and System Optimizations
  instructor: Juncheng Yang
  url: https://course.agentic-system.org/
  format: systems        # systems | research | studio | security | lifecycle | professional
  first_seen: 2026-09-05
  last_checked: 2026-09-05
  coverage:
    ch01: lecture
    ch03: assignment
    ch07: assignment
    ch08: lecture
  note: Only course found that makes students serve their own agent on GPUs.
```

Three depth levels, `mention` below `lecture` below `assignment`. Assignment depth is the
real signal, because what a field makes students build is what it considers core.

### What the script emits

Two artifacts from one source file, so nothing drifts.

1. **Coverage matrix**, chapters against courses. Becomes an appendix in the book and a
   positioning tool outside it.
2. **White-space report**, listing every chapter whose maximum depth across all courses is
   at or below `lecture`. When a chapter drops off that list, the differentiator is eroding
   and the chapter needs to go deeper.

### How it reaches the reader

Both generated from the same YAML.

- A short closing note per chapter, "How this is taught elsewhere." No competing book has
  this, and it is what an instructor evaluating adoption wants to see.
- One appendix, a map of the field's courses.

### Cadence

Syllabi post on the academic calendar, so sweep twice a year, in early September and late
January. Automate it as a scheduled agent that re-fetches every `url`, diffs against stored
state, bumps `last_checked`, and opens a pull request with the deltas. The automation is
what makes this survive. A manual reminder to check the courses never gets done.

### Rejected alternatives

- **Zotero alone.** Loses the chapter mapping and the depth signal.
- **A Notion or Airtable board.** Drifts away from the repo and cannot be built into the
  book.
- **A plain bibliography.** Throws away the positioning information, which is the entire
  value.

## 8. Open decisions

1. **Repo home.** `MLSysBook-vol3` exists, but its branch `docs/vol3-outline` currently
   holds stale Volume II polish work, so it is not a clean home as it stands. Options are a
   fresh repo, a real Volume III, or a scratch outline repo promoted later.
2. **Twelve or ten.** Ten is reachable by merging 6 into 5 and 12 into 9, at the cost of
   demoting failure taxonomy and the training decision.
3. **Where security sits.** Chapter 10 is conventional. The argument for moving it into
   Part I, right after tools, is that injection is a property of the tool interface rather
   than an operational afterthought, and late placement teaches students to bolt it on.
4. **Whether the book trains.** Chapter 12 currently adjudicates the fault line. The
   alternative is to commit to a side, which would sharpen the book and narrow its audience.

## 9. Immediate next step

Name the repo home, then scaffold `curriculum/` with all eleven surveyed courses already
entered and chapter-tagged, plus `coverage.py` and the first white-space report.
