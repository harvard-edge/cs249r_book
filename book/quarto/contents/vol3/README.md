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

> Volume II's thesis is that capability moved into **training-time** compute. Volume III's thesis is
> that the frontier now moves capability into **inference-time** compute, and agentic systems are
> the engineering of that shift. A loop, a search, a self-correction, a tool call: each is inference
> compute spent at runtime instead of baked in at training.

That framing is durable because it is a claim about *where compute is spent*, not about which
framework wins. It also makes the series legible as one progression. Each volume is defined by what
the **unit of the system** is:

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
    <tr><td>6</td><td><b>Orchestration and Multi-Agent Systems</b></td><td><code>orchestration/</code></td><td>How do many stochastic nodes coordinate, fail, and stay consistent?</td></tr>
    <tr><td colspan="4"><b>Part III &mdash; Reliability and Economics</b> <i>(quantifiable systems properties)</i></td></tr>
    <tr><td>7</td><td><b>Error Compounding and Reliability</b></td><td><code>reliability/</code></td><td>How does error accumulate over a trajectory, and how is it bounded?</td></tr>
    <tr><td>8</td><td><b>The Economics of Autonomy</b></td><td><code>economics/</code></td><td>What does multi-step inference cost, and how is an autonomy budget engineered?</td></tr>
    <tr><td>9</td><td><b>Evaluating Acting Systems</b></td><td><code>evaluation/</code></td><td>How do you measure a system whose output is a trajectory, not a label? <i>(open)</i></td></tr>
    <tr><td colspan="4"><b>Part IV &mdash; The Responsible Agent</b> <i>(governing systems that act)</i></td></tr>
    <tr><td>10</td><td><b>Safety, Control, and Oversight</b></td><td><code>safety_control/</code></td><td>How is an autonomous system contained, permissioned, and kept under human oversight?</td></tr>
    <tr><td>11</td><td><b>Security of Autonomous Systems</b></td><td><code>security/</code></td><td>What new attack surface does the action boundary open?</td></tr>
    <tr><td>12</td><td><b>The Durable Core of Autonomy</b></td><td><code>conclusion/</code></td><td>What distills into principle, and what did we deliberately leave out because it is not settled?</td></tr>
  </tbody>
</table>

## Status

Sketch / feasibility probe. Nothing here is wired into the build. The next concrete step is to draft
`introduction/` under the no-product-names rule and judge it against the Volume I and II
introductions. That draft is the go/no-go.
