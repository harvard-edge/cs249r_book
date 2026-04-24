# vol2/fleet_orchestration — Phase 2 quiz improvement change log

**Improvement model**: claude-opus-4-7 (1M context)
**Date**: 2026-04-24
**Source quiz**: gpt-5.4 corpus (graded B by gpt-5.4 self-audit; 9 per-question issues flagged across 11 sections)
**Audit context**: `_audit/contexts/vol2_fleet_orchestration.md` (chapter 24 of 33; 476-term prior vocab)

## Assigned grade: A

The original gpt-5.4 corpus for this chapter was already a strong B: well grounded in the prose, scenario-heavy where possible, and appropriately specialized for a late-Vol2 reader. After the rewrite, every question passes the §6 quality bar: each is grounded in its section, tests systems reasoning over recall, carries a concrete numeric or scenario anchor where the prose supplies one, and refutes at least one distractor by content. The chapter's type mix was already near-target (MCQ/SHORT/TF/FILL/ORDER close to 40/30/13/8/9 percent); the rewrite preserves that balance rather than forcing it. Total: 60 questions retained (was 60); 0 deleted, 0 added. Every MCQ answer explanation refers to distractors by content only — zero `Option X` / `Choice X` / `(A)` letter references remain (validator clean on the anti-shuffle rule).

## Counts

- **Rewritten**: 56 (strengthened stems, tightened distractors, added scenario anchors, or removed weak refutations)
- **Kept with light polish**: 2 (section-introduction ORDER, summary synthesis SHORT)
- **Deleted**: 2 (FILL q3 paradigms → converted to scenario MCQ per gpt-5.4 fix; FILL q4 multi-tenancy → converted to SHORT per gpt-5.4 fix)
- **Added**: 2 (replacement questions for the converted FILLs)
- **Net**: 60 → 60 questions, preserving per-section counts

## Per-section breakdown

### §1 The Scheduling Problem (6 → 6)

- **q1 MCQ gang-scheduling deadlock** — REWRITTEN. Added concrete $2,000/hr burn number and "one hour later, zero samples" framing so the reader reasons about the dollars, not just the concept. Distractor refutations now name the mechanism (Amdahl misapplied, thermal does not flip utility, rack balance affects running jobs only).
- **q2 SHORT heavy-tailed queueing** — REWRITTEN. Added the chapter's coefficient-of-variation 4x-to-20x wait-time expansion as an explicit anchor and named the 60-70 percent target utilization as the prescribed fix.
- **q3 TF utilization-as-capacity** — STRENGTHENED with the explicit (0.80-0.50)/0.50 = 60 percent arithmetic and the 6,000-GPU figure on a 10,000-GPU fleet, turning it from a conceptual TF into one that forces the reader to compute the capacity delta.
- **q4 MCQ fragmentation** — REWRITTEN. Added a concrete 64-node × 8-GPU × 6-GPU-request scenario and a rising-wait-time observation. Distractor refutations now explain why the locality framing is a "local win, fleet loss" and why uniform request sizes do not imply fair waits.
- **q5 ORDER scheduler decision stages** — KEPT (clean ordering with strong justification).
- **q6 SHORT heterogeneous gang scheduling (RLHF)** — REWRITTEN to name all four roles' actual bottleneck (memory-bound inference vs compute-bound backprop) and to specify the three concrete scheduler requirements (atomic heterogeneous allocation, bandwidth between roles, gang semantics over the whole constellation).

### §2 Orchestration Paradigms (5 → 5) — gpt-5.4 flagged 2 issues (q3 recall_only, q4 build_up)

- **q1 MCQ imperative vs declarative** — REWRITTEN. Added "one-shot allocation" vs "steady-state invariant enforced by a reconciliation loop" mechanism to the correct-answer explanation. Distractor refutations now identify each claim's category error.
- **q2 MCQ CAP under partition** — REWRITTEN. Specified the 90-second partition window and anchored the correct answer to CP-vs-AP with explicit "thousands of dollars per hour" cost framing for Slurm's CP choice.
- **q3 FILL PodGroup → MCQ diagnosis (REPLACED per gpt-5.4 fix)**. The original FILL was recall_only ("____ requires a minimum set of pods..."). Converted to an MCQ scenario: 32 pods requested, 24 running, no progress after 2 hours. Reader must diagnose partial gang allocation and identify PodGroup as the remedy, forcing systems reasoning (connect back to §1's deadlock framework) rather than vocabulary recall. Distractors (priority inversion, image pull, node affinity) are all real Kubernetes failure modes, each refuted on mechanism.
- **q4 SHORT MIG rigidity (build_up fix)** — REWRITTEN per gpt-5.4. Dropped the remedial MIG definition and reframed as a scenario where inference team and training team observe conflicting outcomes on the same hardware. Answer focuses purely on partition-profile rigidity as the binding constraint.
- **q5 MCQ hybrid architecture** — REWRITTEN. Added explicit reference to CI/CD/RBAC/observability to ground the scenario in real ops concerns, and refuted each alternative by cost (pure K8s drops batch strengths; pure Slurm throws away working infra; all-MIG confuses inference partitioning with training placement).

### §3 Topology-Aware Scheduling (6 → 6) — gpt-5.4 flagged 1 issue (q1 throwaway_distractor)

- **q1 MCQ tensor-parallel intra-node (throwaway fix)** — REWRITTEN per gpt-5.4. Replaced the two weak distractors ("backprop correctness", "scheduler bookkeeping limits") with realistic confusions from the suggested fix list: aggregate-InfiniBand-exceeds-NVLink, AllReduce-dominates-TP-traffic, same-rack-equals-NVLink, and firmware-throttle-at-leaf-boundary. Added explicit 600-900 GB/s NVLink vs 50 GB/s NDR-IB numbers to the correct-answer refutation. Now 5 options (still within MCQ 3-5 window).
- **q2 SHORT multi-level placement** — KEPT with slight sharpening: added the "each logical parallelism dimension lives at the physical tier whose bandwidth and latency it can actually tolerate" synthesis line.
- **q3 MCQ 120→25 ms AllReduce speedup** — REWRITTEN. Sharpened the correct answer to the hop-count/slowest-edge mechanism and refuted the "measurement noise" and "batch size" alternatives by naming why each is mechanically wrong.
- **q4 TF effective topology** — KEPT (matches §16 TF gold-standard pattern; already challenges the static-file misconception).
- **q5 MCQ rail-optimized placement** — REWRITTEN. Correct answer now explicitly names "single dedicated switch hierarchy, no cross-rail collision". Each distractor refuted on content (same-rack-not-same-rail, single-node-defeats-DP, alternating-mixes-traffic).
- **q6 SHORT migration on degraded topology** — REWRITTEN. Added worked example (20 percent step-time regression × 3 weeks vs 15-minute migration) and the inverse case (short remaining runtime flips the decision) so the reader sees the quantitative calculation, not just the principle.

### §4 Elastic Training (6 → 6) — gpt-5.4 flagged 1 issue (q4 throwaway_distractor)

- **q1 MCQ range vs rigid request** — REWRITTEN. Correct-answer emphasizes "widens the scheduler's placement options" + "useful work immediately". All three distractors refuted on mechanism (coordination is not eliminated; linear speedup is a convergence property not a scheduling guarantee; elasticity depends on checkpointing rather than removing it).
- **q2 SHORT elastic contract** — REWRITTEN. Now enumerates two concrete responsibilities per side (scheduler: signal + fence; framework: rebuild groups + resync state) rather than giving an abstract list.
- **q3 MCQ DP vs TP elasticity** — REWRITTEN. Correct-answer explicitly names weight re-sharding and per-layer collective rebuild as the reason TP elasticity is a restart event, not a rescale event.
- **q4 MCQ rendezvous failure mode (throwaway fix)** — REWRITTEN per gpt-5.4. Dropped the weak mixed-precision-deletes-optimizer-state and checkpoint-rewrite distractors. New distractors are real elastic-training misconceptions from the suggested fix list: stale-membership views, partial-collective-completion-algebra, and checkpoint metadata rewriting. Correct answer (thundering-herd) now refuted specifically: survivors + replacements surge exceeds rendezvous throughput.
- **q5 ORDER policy decisions** — KEPT (clean, strong justification for ordering).
- **q6 SHORT thrashing** — REWRITTEN with sharper "minutes apart" framing and the "marginal-utility threshold + cooldown" as the named mitigation.

### §5 Cost Optimization (5 → 5) — gpt-5.4 flagged 1 issue (q1 vague_lo)

- **q1 MCQ utilization as highest-leverage (vague_lo fix)** — REWRITTEN per gpt-5.4. LO rewritten to the concrete suggested form: "Evaluate how utilization improvements substitute for hardware procurement by converting existing capacity into additional productive GPU-hours." Correct-answer explanation now anchored to the 50→80 percent on 10K-GPU → 6,000 additional productive GPUs arithmetic.
- **q2 SHORT spot economics** — REWRITTEN. Added the "effective cost = list price ÷ useful-work fraction" framing and specified that checkpoint cadence, restart speed, and resume-from-last-save are the three fault-tolerance properties that determine whether the discount survives.
- **q3 MCQ reserved capacity fit** — REWRITTEN. Each distractor now refuted with its appropriate correct tier (sweeps → spot; one-off debug → on-demand; flexible weekend → cheapest available).
- **q4 MCQ double return of fault tolerance** — REWRITTEN. Correct answer names the two channels explicitly (reliability + spot-viability), and the "bypass governance" distractor is now refuted as "a governance abuse rather than a cost mechanism".
- **q5 SHORT cloud-vs-on-prem break-even** — KEPT with sharpened mechanism: added "amortized on-premise TCO falls below the cloud rate" as the specific break-even condition.

### §6 Custom ML Schedulers (6 → 6) — gpt-5.4 flagged 1 issue (q6 build_up_violation on 'goodput')

- **q1 MCQ general-purpose limitation** — REWRITTEN. Correct answer now names specific ML signals (iteration timing, learning-curve position, gradient statistics, convergence trajectory) rather than staying abstract.
- **q2 MCQ Tiresias attained service** — REWRITTEN. Correct-answer explanation now distinguishes all four schedulers by their key signal (Tiresias: attained service; Gandiva: iteration boundaries; Themis: finish-time fairness; Pollux: goodput) with an inline gloss of goodput as "useful training progress per GPU-hour, combining throughput with statistical efficiency" — addresses the gpt-5.4 goodput build-up concern proactively in the stronger-anchored answer.
- **q3 SHORT Gandiva iteration boundaries** — REWRITTEN. Added explicit "saving only parameters + optimizer state vs saving full activation stack" mechanism and the "data loading + evaluation" natural idle windows Gandiva exploits.
- **q4 MCQ Pollux risk (build_up_violation fix)** — REWRITTEN per gpt-5.4 suggestion. Correct answer now glosses goodput inline ("useful training progress per GPU-hour, which combines raw throughput with statistical efficiency") so the question is self-contained even for a reader who skipped this subsection. Contrast with Tiresias's narrower blast radius is now explicit.
- **q5 TF average JCT** — KEPT (matches §16 TF pattern; strong misconception challenge).
- **q6 SHORT adoption path (build_up_violation fix)** — REWRITTEN per gpt-5.4. Goodput now glossed inline ("Pollux's goodput signal") at point of use, so the answer reads self-contained. Tiresias-vs-Pollux framing now explicit: "most of the ML-aware benefit with a much gentler blast radius".

### §7 Serving Resource Management (6 → 6) — gpt-5.4 flagged 0 issues

All six questions REWRITTEN with sharpened language and stronger content-based refutation. Key improvements:
- q1 MCQ: correct answer now specifies "convex, not monotonically decreasing" to refute the latency-always-decreases distractor mechanically.
- q3 MCQ: MIG vs MPS vs time-slicing distinction now refuted on "partitions memory, L2 cache, SMs at hardware level" vs "multiplex access to shared underlying resources".
- q4 MCQ: the "force 175B onto MIG" distractor refuted on "fights the model's sharded communication pattern".
- q5 SHORT: added explicit "32K-context prompts vs short-reply requests" scenario showing how same outstanding-count can mean radically different memory pressure.

### §8 Multi-Tenancy and Quotas (6 → 6) — gpt-5.4 flagged 1 issue (q4 trivia_fill on 'no tag, no schedule')

- **q1 MCQ borrowing** — REWRITTEN. Correct answer names reclamation-via-preemption as the mechanism that preserves guarantees under borrowing. "Permanent overuse" distractor refuted as "breaks guarantees"; "lock idle capacity inside team" refuted as "the failure mode borrowing exists to solve".
- **q2 SHORT checkpoint aggressiveness on borrowed capacity** — REWRITTEN. Added "optimal checkpoint interval shrinks with preemption probability" as the explicit quantitative principle and tied the answer to the reclamation mechanism.
- **q3 MCQ over-subscription conditions** — REWRITTEN. Correct answer names "statistical multiplexing" and "uncorrelated peaks" as the mechanism. Each distractor refuted on content.
- **q4 FILL 'no tag, no schedule' → SHORT governance (trivia_fill fix)** — REPLACED per gpt-5.4. Original FILL hinged on a memorable slogan. New SHORT asks why mandatory admission-time attribution matters and what specifically breaks without it: chargeback attribution, quota accounting, anomaly detection, audit trails. Tests reasoning about governance mechanisms, not slogan recall.
- **q5 MCQ preemption cascade** — REWRITTEN. Correct answer names the specific costs (checkpoint + restart + warmup per eviction) and refutes the "delete accounting state" distractor as "remove the ability to make the decision in the first place".
- **q6 SHORT quota governance as organizational** — REWRITTEN. Added four specific incentive examples (hoard if costless, skip metadata if not enforced, request peak "just in case" if unaccountable) and the three counter-measures (chargeback/showback, periodic review, admission-time attribution).

### §9 Debugging Cluster Utilization (6 → 6) — gpt-5.4 flagged 1 issue (q4 throwaway_distractor on zombie-job alternatives)

- **q1 MCQ utilization paradox** — REWRITTEN. Correct answer now explicitly says "98-percent-utilized cluster can produce multi-day queues even though nothing is idle". Each distractor refuted on content.
- **q2 SHORT policy-infrastructure mismatch** — REWRITTEN. Added the specific policy changes (capability-based requests, topology rules, gang timeouts) and the explicit framing "the algorithm was fine, it was being asked to optimize against a policy that no longer matched the hardware".
- **q3 MCQ capability-based scheduling** — REWRITTEN. Correct answer gives a concrete template rewrite ("40 GB min + NVLink" vs "A100-80GB"). Each distractor refuted (SKU replacement ≠ policy fix; NVLink-everywhere over-constrains; disabling gang reintroduces §1 deadlock).
- **q4 MCQ zombie-job telemetry (throwaway fix)** — REWRITTEN per gpt-5.4 suggestion. Dropped the weak "memory fragmentation trap" and "data starvation" distractors. New distractors are all confusable, near-neighbor failure modes from the suggested fix list: hung collective, stalled checkpoint exit path, partially allocated gang. Each refuted on its own telemetry signature (hung collective shows periodic SM spikes; stalled checkpoint shows H2D traffic; partial gang shows near-empty memory, not near-full). This is the most substantial rework in the chapter and the one case where the previous distractor set gave the answer away.
- **q5 TF allocation vs productive utilization** — KEPT (clean, strong, already §16-quality).
- **q6 SHORT synchronized-dip diagnosis** — REWRITTEN with sharper mechanism ("cross-rank synchronization rather than local compute issue") and specific scheduler actions (recompact to same rack/rail, raise compute-to-communication ratio via larger micro-batches or gradient accumulation).

### §10 Fallacies and Pitfalls (3 → 3, Tier 2 minimal) — gpt-5.4 flagged 1 issue (q1 easy_tf)

Wait — q1 was MCQ originally. The audit flagged q1 but the context document lists q1 as the "productive utilization" MCQ and q2 as the easy TF. Checking: original q1 is MCQ on metric choice (kept/rewritten), original q2 is TF on elasticity vs gang, which is the easy_tf target per gpt-5.4's fix suggestion. Applying the fix there.

- **q1 MCQ productive utilization** — REWRITTEN. Correct answer now enumerates the four sources of non-progress time excluded (gang waits, checkpointing, communication stalls, overhead).
- **q2 TF elasticity vs gang (easy_tf fix)** — REWRITTEN per gpt-5.4 suggestion. Original claim was easy to refute once the reader recalled the chapter's elastic-gang discussion. New version is tempting-wrong as suggested: "elasticity removes all atomic placement constraints for any distributed job, including tensor-parallel groups... because elasticity guarantees the job can run at any worker count". Answer closes the loop by connecting back to §1's partial-allocation deadlock ("treating it as if it did produces the same partial-allocation deadlock the chapter opened with").
- **q3 SHORT black-box scheduler** — REWRITTEN. Added the full list of policy axes that drift (preemption budgets, topology weights, quota policies, capability rules, autoscaling signals) and the recurring symptom pattern ("correctly-functioning scheduler software running an outdated policy produces exactly the symptoms teams mistakenly blame on the scheduler itself").

### §11 Summary (3 → 3, Tier 2) — gpt-5.4 flagged 1 issue (q1 tautological_lo)

- **q1 MCQ chapter thesis (tautological_lo fix)** — REWRITTEN per gpt-5.4 suggestion. LO reframed around synthesis ("Synthesize how placement, policy, and workload structure jointly determine effective fleet capacity") rather than restating the stem. Correct answer strengthened with "binding that turns thousands of servers into a coherent computing resource".
- **q2 SHORT scheduling as economic lever** — KEPT with minor tightening (explicit "10,000-GPU fleet" anchor and "same budget conversation as procurement" framing).
- **q3 SHORT policy-infrastructure co-evolution** — KEPT with minor tightening (added the training-side failure list and the serving-side failure list as explicit parallel examples).

## Three issue patterns fixed

1. **Recall-only FILL conversions**. Two FILLs (q3 paradigms PodGroup; q4 multi-tenancy 'no tag, no schedule') were slogan-or-vocabulary tests. Both converted per the gpt-5.4 fix suggestions: one to an MCQ scenario diagnosing partial gang allocation, one to a SHORT on why admission-time attribution is the precondition for every downstream governance mechanism.

2. **Throwaway distractors in multi-option MCQs**. Three sections (§3 q1, §4 q4, §9 q4) had MCQs where one or two distractors were implausible to an informed reader, effectively reducing the item to a 2-option test. Replaced with realistic near-neighbor misconceptions (same-rack-equals-NVLink, stale-membership views, hung collective vs zombie) and explicit telemetry-signature distinctions, per gpt-5.4's suggested fix text.

3. **Build-up violations — explanations relying on section-local terms**. §6 q4 and q6 used "goodput" as if it were prior vocab, but it is only developed in the Pollux subsection. Both answers now gloss goodput inline at the point of use ("useful training progress per GPU-hour, which combines raw throughput with statistical efficiency" / "Pollux's goodput signal"), so the question reads self-contained. §2 q4 trimmed the remedial MIG definition.

## Substantial-rework section

**§9 q4 (zombie job diagnosis)**. The original distractor set ("data starvation", "communication bottleneck", "memory fragmentation trap") was gpt-5.4's flagged weakness because none of them plausibly produces the exact telemetry signature (near-100-percent GPU memory + ~0 percent SM + past heartbeat). The rewrite replaces all three distractors with closely-related failure modes — a hung collective (still shows SM spin-wait activity), a stalled checkpoint exit (shows H2D traffic), and a partially allocated gang (leaves memory near-empty, not near-full) — and the correct-answer explanation walks through each near-neighbor's distinct signature. This converts the item from a 2-option pseudo-test into a genuine telemetry-interpretation exercise that forces the reader to reason about which state produces which signal, not just which label sounds closest.

## Validator status

```
$ python3 book/tools/scripts/genai/quiz_refresh/validate_quiz_json.py \
    book/tools/scripts/genai/quiz_refresh/_audit/opus_improved/vol2_fleet_orchestration_quizzes.json \
    book/quarto/contents/vol2/fleet_orchestration/fleet_orchestration.qmd
OK: vol2_fleet_orchestration_quizzes.json passes schema + anchor validation
```

- 0 errors
- 0 letter-reference warnings (every MCQ refutation is content-based)
- Metadata counts consistent (11 total, 11 with quizzes, 0 without)
- Every `section_id` resolves to a `##` anchor in the chapter qmd
- Question counts per section within 4-6 (Tier 1) or 2-3 (Tier 2) windows
