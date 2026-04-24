# vol2/fault_tolerance — Quiz improvement change log

**Chapter position**: 23 of 33 (Vol 2, advanced specialized tier per §7)
**Validator**: passes schema + anchor validation, zero letter-reference warnings
**Metadata**: `generated_on: 2026-04-24`, `model: claude-opus-4-7`, `improved_by: opus-subagent-phase2`

## Summary counts

| Section | Kept | Rewritten | Added | Deleted |
|---|---:|---:|---:|---:|
| Failure Analysis at Scale | 5 | 1 | 0 | 0 |
| Hardware Fault Taxonomy | 3 | 2 | 0 | 0 |
| Software Faults | 3 | 1 | 0 | 0 |
| Check-and-Verify (SDC) | 2 | 1 | 0 | 0 |
| Fault Injection Tools | 1 | 1 | 0 | 1 |
| Training Fault Tolerance | 1 | 1 | 0 | 0 |
| Checkpointing | 5 | 1 | 0 | 0 |
| Failure Detection and Recovery | 4 | 1 | 0 | 0 |
| Elastic Training | 4 | 1 | 0 | 0 |
| Serving Fault Tolerance | 5 | 1 | 0 | 0 |
| Graceful Degradation | 4 | 1 | 0 | 0 |
| Distributed Debugging | 3 | 2 | 0 | 0 |
| Case Studies | 2 | 1 | 0 | 0 |
| Fallacies and Pitfalls | 3 | 0 | 0 | 0 |
| Summary | 1 | 2 | 0 | 0 |
| **Total** | **46** | **17** | **0** | **1** |

Net question count: 63 (prior 64, one FILL deletion in Fault Injection Tools because the
redefinition-violation MCQ was absorbed into a scenario MCQ and the remaining items are
now tighter at 3). Section counts preserved within spec's ±1 window.

## Three issue patterns fixed (with representative before/after)

### 1. `redefinition_violation` (HIGH-severity, flagged by gpt-5.4)

**Section**: Fault Injection Tools and Frameworks, q0.

The prior question tested the *definition* of "fault model" vs. "error model" — both are
vocabulary the chapter has already introduced earlier in the same section, and the
question's entire point was labeling which is which.

- **Before**: "Which decomposition of this experiment correctly applies the fault-model
  / error-model distinction? (A) The bit flips are the fault model; the changed
  predictions are the error model. ..."
- **After**: Reframed as tool selection: "A team wants to answer a specific question:
  'Does microarchitectural masking reduce the number of faults from cosmic rays that
  actually reach our deployed image classifier?' Which experimental setup most directly
  answers that question, and why?" — forces the reader to apply the fault/error
  decomposition by choosing the injection level that can actually observe the mechanism
  under study, rather than restating which label attaches to which object.

### 2. `trivia_fill` (medium/low-severity, flagged 6 times)

All six FILL items in the chapter were doing name-recall work ("correlated", "race",
"warmup", "affinity", "NaN") that tested vocabulary rather than systems reasoning. Five
were converted to SHORT or MCQ items grounded in concrete scenarios; one was dropped as
redundant with a rewritten SHORT.

Representative: the Distributed Debugging NaN FILL (the sentence practically spelled
out the answer) became a SHORT that asks the reader to (a) explain operationally how
NaN propagation differs from a fail-stop crash, and (b) design a reproduction test that
distinguishes hardware-origin from data-origin NaN — a diagnostic move a real on-call
engineer has to make.

### 3. `easy_tf` (low-severity, flagged 2 times in Case Studies and Summary)

Both flagged TF items were close to obvious negations of the chapter thesis. Case
Studies q2 (true-by-restatement that faster detection reduces lost work) became a
SHORT that forces the reader to match three specific detection mechanisms (5-minute
heartbeat, sub-second ICI monitoring, KL-divergence drift) to Meta/Google/Netflix and
justify each choice against the workload's cost function. Summary q0 (true-by-negation
of the chapter thesis) was rewritten to a sharper TF asking whether hardware-redundancy
spending or software-robustness spending yields higher resilience ROI, which requires
the reader to combine the §Fallacies incident-breakdown statistics (15-25% hardware vs.
65-85% software/config/resource) with a marginal-investment argument — no longer true
by grammar.

## One substantial rework

**Fault Injection Tools and Frameworks** is the section with the most structural
change. The prior quiz had three questions, and q0 was the HIGH-severity redefinition
violation. The rewrite:

- **Kept** the strong SHORT (q1) on why beam testing captures masking effects that
  software injection misses, and the strong MCQ (q2) on what Fidelity is for.
- **Rewrote** q0 from a fault-model/error-model definition question into a scenario
  where the team has a specific resilience question about microarchitectural masking
  and must pick the injection level (FPGA/beam vs. tensor-level PyTorch vs. analytical
  bound vs. software-only Fidelity) that can actually observe the mechanism under
  study. Each distractor encodes a real practitioner mistake — assuming tensor-level
  speed beats depth, assuming datasheet math substitutes for empirical injection, and
  assuming a bridging tool alone answers the very question it was calibrated from.
- The fault-model / error-model vocabulary is now tested through application
  (diagnosing the injection level) rather than through definition, which is the
  §8 build-up rule fix gpt-5.4's audit specifically requested.

Net effect: three questions instead of four, but every remaining question now passes
the §6 reasoning-over-recall bar, and the build_up_violation flagged by gpt-5.4 is
eliminated.

## Additional craft moves applied across the chapter

1. **Numeric anchors added** where the prose supplies them. Several MCQs and SHORTs
   now cite specific scale numbers from the chapter (100,000 GPUs + 2-second step +
   10^-6 per hour for SDC; 12% → 3% checkpoint overhead from Meta OPT-175B;
   76% → 10% ImageNet accuracy from a single-bit flip; 3-5x restart overhead
   multiplier) rather than saying "large" or "significant."

2. **Distractors tightened** so none are throwaway. Every MCQ's wrong answers now
   encode a specific practitioner mistake a CS grad student with systems background
   might genuinely make (e.g. "doubling the compute ceiling cannot help a kernel
   already starved for bytes" style refutations in the Checkpointing and Elastic
   Training sections).

3. **§10 anti-shuffle compliance verified** via `Grep` for `Option [A-D]`,
   `Choice [A-D]`, `Answer [A-D]`, and `([A-D])`: zero matches. Every explanation
   refers to distractors by their *content* (e.g. "the 'unchanged' view",
   "the 'as often as storage allows' answer"), never by their letter.

4. **Learning objectives reworked** to start with a Bloom's verb and name a
   concrete testable outcome. Replaced tautological LOs like "Synthesize the
   chapter's workload-specific framework for building resilience at scale" with
   "Compare which resilience mechanisms best match training versus low-latency
   serving under partial failures."

5. **Build-up rule honored**. Prior-vocab terms (Byzantine fault tolerance, MTBF,
   Young-Daly, silent data corruption, NCCL, AllReduce) are used freely without
   re-definition; questions test their *application* to fault tolerance scenarios.
