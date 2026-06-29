# Volume III: Agentic Machine Learning Systems

*The systems engineering of inference-time compute and autonomous control loops.*

---

> **Note:** This is a **sketch**, not a commitment. It exists to test one question: is there a
> durable fundamentals layer under "agentic" yet, the way the bitter lesson anchors Volume I and
> scaling laws anchor *Machine Learning Systems at Scale*? The discipline below is the test, not a
> promise to ship. If the introduction cannot be written at the same altitude as the first two
> introductions, this volume waits.

## Why a Third Volume

The common objection is correct about the thing it is looking at, and it is looking at the wrong
thing. The agentic *application surface*, the frameworks, the prompt patterns, the speculation about
what autonomy will become, is moving too fast to teach. But neither earlier volume was ever written
on that surface. *Introduction to Machine Learning Systems* is anchored by the bitter lesson;
*Machine Learning Systems at Scale* is anchored by scaling laws. Those are empirical regularities
about compute and generalization, not facts about any model or tool.

So the question for this volume is not "has the agent field settled?" (it has not). It is narrower
and answerable: **has the systems substrate beneath agents settled enough to teach?** The wager of
this sketch is that it mostly has, and that the durable spine is the shift the field has been living
through since the reasoning-model era:

> *Machine Learning Systems at Scale* already names the inference-time-compute shift: its
> introduction draws the three scaling regimes (pretraining, posttraining, test-time), and its
> inference chapter treats test-time compute as a serving resource. So "inference-time compute is
> the frontier" cannot be Volume III's thesis; the companion volume said it first. Volume III's
> thesis is what that compute, spent in a loop with state and tools, turns the system *into*: an
> **actor**. The unit of engineering stops being the served request and becomes the **trajectory**,
> the whole sense-decide-act-observe arc a system runs to accomplish a goal.

That framing is durable because it is a claim about *what the system becomes when it spends runtime
compute*, not about which framework wins, and it sits at a different altitude than the companion
volume's serving treatment (see "Boundaries" below). It also makes the series legible as one
progression. Each volume is defined by what the **unit of the system** is:

| Volume | Title | Unit of system | You learn to |
| --- | --- | --- | --- |
| I | *Introduction to Machine Learning Systems* | the model | build one system |
| II | *Machine Learning Systems at Scale* | the fleet | run many, where scaling laws bite |
| III | *Agentic Machine Learning Systems* | the agent | engineer a system that acts |

> **Note on the title:** keeping "Learning" in the name holds the series family
> (`Machine Learning Systems`). "Agentic Machine Systems" drops it; use that only if broadening past
> learning is deliberate, not by default.

## How I Intend to Write It

The volume earns durability only by holding altitude. Five rules, all falsifiable:

1. **No framework, product, or this-year benchmark may appear in the body.** If a chapter cannot be
   written without naming one, that chapter is too early and gets cut, not faked. This is the single
   discipline that separates a textbook from a survey that dates in two years.
2. **Each chapter must state a fundamental, not catalog current practice.** The control loop is
   robotics and control theory. Multi-agent is distributed systems with stochastic nodes. Inference
   economics extends Volume II's serving-cost accounting. Teach the substrate the trend sits on.
3. **The honest holes are written as open problems, not solved methods.** Evaluation of acting
   systems, and the memory/state architecture, are genuinely unsettled. They are taught as "here is
   the problem and why it is hard," never as "here is the answer." Naming the frontier honestly is
   itself durable; faking a method is what dates.
4. **The introduction is written first, as a feasibility probe.** Exactly the artifact trusted for
   the first two volumes. If the intro states the inference-compute thesis and frames the substrate
   with no product names, the volume is viable. If it leans on what today's agents do, the probe has
   told us to wait, at the cost of one draft.
5. **One chapter per working session; companions referenced by title.** This volume may name both
   *Introduction to Machine Learning Systems* and *Machine Learning Systems at Scale* directly.

## Boundaries with the earlier volumes

The asymmetric reference rule still holds: this volume may name *Introduction to Machine Learning
Systems* and *Machine Learning Systems at Scale* by title, but neither of them points forward to
this one, so adding Volume III forces no cross-reference edits on either. The real work is keeping
Volume III at a *different altitude* than ground the companion volumes already cover, so it never
merely re-announces them.

| Topic | The companion volume owns (keep) | Volume III takes (different altitude) |
| --- | --- | --- |
| Inference-time compute | test-time compute as a *serving resource*: scheduler thinking-time, the logic wall, reasoning latency budgets | the system that *spends* it is an **actor**; the engineering unit is the loop/trajectory, not the served request |
| The action boundary | injection, tool permissions, and side effects as part of *securing ML systems* | the autonomous loop as a *threat model* (Chapter 11 sits above the companion treatment, never repeats it) |
| Orchestration | *fleet orchestration* = scheduling accelerators | **multi-agent coordination** = many stochastic nodes (renamed so the word does not collide) |

## Underlying principles (candidate invariants)

The spine of each earlier volume is a small set of durable laws (the bitter lesson; the scaling,
roofline, and serving-cost laws). Volume III is viable only if it has its own. These are the
candidates, each a claim about compute or systems, not about any tool:

- **The Trajectory Reliability Law.** End-to-end success decays roughly geometrically with horizon
  length: for per-step reliability `p`, an `n`-step task succeeds at about `p^n`. Horizon length is
  therefore bounded by per-step reliability, and verification and recovery are mandatory, not
  optional. (The agentic analog of the scaling laws.)
- **Inference-Time Scaling.** Capability improves with runtime compute (samples, steps, search),
  with diminishing returns; "thinking" becomes a schedulable resource. (The successor axis to
  training-time scaling.)
- **The Autonomy Cost Law.** Cost and latency scale with trajectory length times per-step inference,
  so an agentic system's serving cost is steps times the companion volume's serving cost. Autonomy
  has a metered budget; depth is an economic decision.
- **The Control-Loop Invariant.** Every agent is sense, decide, act, observe over state and an
  environment; the loop, not the model call, is the unit of engineering.
- **The Action-Boundary Principle.** Capability and risk both scale with what the loop is permitted
  to do, so both power and vulnerability concentrate at the typed, permissioned action interface.
- **State as a Memory Hierarchy.** Context is a finite, costly resource with locality; what to
  retain, retrieve, and evict is a caching problem. (Open: the resolved architecture does not exist
  yet.)
- **The Coordination Tax.** `N` stochastic agents pay the distributed-systems coordination cost
  amplified by nondeterminism; more agents does not buy linear capability.

## Why a signature version is possible

A scan of the 2026 landscape shows the agent-book market is almost entirely two things: framework
tutorials (build agents with this month's library) that date in roughly eighteen months, and
research surveys or courses on agent *capabilities* (reasoning, code, robotics). The serious
engineering-first generalist (*AI Engineering*, O'Reilly 2025) is an application-stack book, not a
systems-from-first-principles treatment, and not agent-specific. Nobody is writing the
systems-engineering, invariant-anchored, vendor-neutral treatment of agents that the first two
volumes' method would produce. That gap is the signature opportunity, and it is wider than it was
for the earlier volumes precisely because the competing books cluster on the disposable end.

## Chapter Map (sketch)

A 12-chapter, four-part arc that parallels Volumes I and II and holds the substrate altitude.

<table width="100%">
  <thead>
    <tr>
      <th width="5%">#</th>
      <th width="28%">Chapter</th>
      <th width="22%">Directory</th>
      <th width="45%">Core Question</th>
    </tr>
  </thead>
  <tbody>
    <tr><td colspan="4"><b>Part I &mdash; The Inference-Time Frontier</b> <i>(why a third book exists)</i></td></tr>
    <tr><td>1</td><td><b>From Trained Models to Acting Systems</b></td><td><code>introduction/</code></td><td>What changes when capability moves from training-time to inference-time compute?</td></tr>
    <tr><td>2</td><td><b>The Control Loop</b></td><td><code>control_loop/</code></td><td>What is an agent, as a system, beneath the hype?</td></tr>
    <tr><td>3</td><td><b>Inference-Time Scaling</b></td><td><code>inference_scaling/</code></td><td>How does spending compute at runtime buy capability, and what are the (nascent) laws?</td></tr>
    <tr><td colspan="4"><b>Part II &mdash; The Substrate</b> <i>(the durable engineering layer)</i></td></tr>
    <tr><td>4</td><td><b>State and Memory</b></td><td><code>state_memory/</code></td><td>How does an acting system manage what it knows, as a memory hierarchy problem? <i>(open)</i></td></tr>
    <tr><td>5</td><td><b>Tool and Environment Interfaces</b></td><td><code>tool_interfaces/</code></td><td>How is the boundary between model and world engineered as a typed action surface?</td></tr>
    <tr><td>6</td><td><b>Multi-Agent Coordination</b></td><td><code>multi_agent/</code></td><td>How do many stochastic nodes coordinate, fail, and stay consistent? (renamed to avoid colliding with the companion volume's fleet orchestration)</td></tr>
    <tr><td colspan="4"><b>Part III &mdash; Reliability and Economics</b> <i>(quantifiable systems properties)</i></td></tr>
    <tr><td>7</td><td><b>Error Compounding and Reliability</b></td><td><code>reliability/</code></td><td>How does error accumulate over a trajectory, and how is it bounded?</td></tr>
    <tr><td>8</td><td><b>The Economics of Autonomy</b></td><td><code>economics/</code></td><td>What does multi-step inference cost, and how is an autonomy budget engineered?</td></tr>
    <tr><td>9</td><td><b>Evaluating Acting Systems</b></td><td><code>evaluation/</code></td><td>How do you measure a system whose output is a trajectory, not a label? <i>(open)</i></td></tr>
    <tr><td colspan="4"><b>Part IV &mdash; The Responsible Agent</b> <i>(governing systems that act)</i></td></tr>
    <tr><td>10</td><td><b>Safety, Control, and Oversight</b></td><td><code>safety_control/</code></td><td>How is an autonomous system contained, permissioned, and kept under human oversight?</td></tr>
    <tr><td>11</td><td><b>Security of Autonomous Systems</b></td><td><code>security/</code></td><td>What does the autonomous loop add to the action-boundary threat model the companion volume already covers? (sits above, does not repeat it)</td></tr>
    <tr><td>12</td><td><b>The Durable Core of Autonomy</b></td><td><code>conclusion/</code></td><td>What distills into principle, and what did we deliberately leave out because it is not settled?</td></tr>
  </tbody>
</table>

## Status

Sketch / feasibility probe. Nothing here is wired into the build. The next concrete step is to draft
`introduction/` under the no-product-names rule and judge it against the Volume I and II
introductions. That draft is the go/no-go.
