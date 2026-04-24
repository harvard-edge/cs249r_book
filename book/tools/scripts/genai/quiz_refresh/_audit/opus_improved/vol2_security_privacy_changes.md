# vol2/security_privacy — Quiz improvement change log

**Chapter position**: 29 of 33 (Vol 2, specialized integration tier per §7)
**Validator**: passes schema + anchor validation, zero letter-reference warnings
**Metadata**: `generated_on: 2026-04-24`, `model: claude-opus-4-7`, `improved_by: opus-subagent-phase2`

## Grade

Overall grade: A-. The prior version (from `gpt-5.4` + prior improve-mode pass) was already solid with only six per-question audit flags across 11 sections. This Opus pass rewrote the six flagged items to match §16 gold-standard patterns, tightened scenario concreteness across most other MCQs, added an answer-letter distribution at generation time to eliminate B-dominance, and kept the A-grade questions intact with small editorial polish.

## Summary counts

| Section | Kept | Rewritten | Added | Deleted |
|---|---:|---:|---:|---:|
| The Expanded Attack Surface | 4 | 2 | 0 | 0 |
| Learning from Security Breaches | 4 | 1 | 0 | 0 |
| Systematic Threat Analysis and Risk Assessment | 4 | 1 | 0 | 0 |
| Model-Specific Attack Vectors | 6 | 0 | 0 | 0 |
| Hardware-Level Security Vulnerabilities | 4 | 1 | 0 | 0 |
| When ML Systems Become Attack Tools | 3 | 1 | 0 | 0 |
| Comprehensive Defense Architectures | 6 | 0 | 0 | 0 |
| Differential Privacy | 6 | 0 | 0 | 0 |
| Practical Implementation Roadmap | 4 | 1 | 0 | 0 |
| Fallacies and Pitfalls | 5 | 0 | 0 | 0 |
| Summary | 2 | 1 | 0 | 0 |
| **Total** | **48** | **8** | **0** | **0** |

Net question count: 56 (unchanged). Section counts preserved (spec ±1 window). MCQ letter distribution: A=11, B=7, C=10, D=5, built at generation time to avoid §10 anti-shuffle exposure.

## Three issue patterns fixed

### 1. ORDER items that disguised recall as sequencing (§9 ORDER anti-pattern)

Two sections had ORDER items the prior audit flagged as process-shaped recall:

- **Learning from Security Breaches, q4** — the ORDER item imposed a lifecycle ranking on three historical breach lessons (Stuxnet, Jeep Cherokee, Mirai) that are primarily *analogies*, not a causal pipeline. Replaced with a scenario MCQ asking the reader to match a concrete AV perception-pipeline failure to the closest historical breach and justify why the analogy drives remediation direction (segmentation, not stronger passwords or signed artifacts). Now tests reasoning about which historical lesson applies to a novel scenario.

- **Practical Implementation Roadmap, q1** — the ORDER item on the three-phase rollout was flagged as "mainly tests memory of phase names." Replaced with a scenario MCQ putting a fintech startup at six months of runway and asking which of four concrete investments (DP accounting, RBAC+MFA+TLS, certified adversarial training, formal red-teaming) should come first and why. The mechanism test is the same, but the reader now has to commit to a concrete first move rather than recite phase labels.

### 2. `trivia_fill` cue leakage in FILL stems (§5 FILL anti-pattern)

**Systematic Threat Analysis, q4** — the prior FILL stem ("accelerator firmware or container orchestration is operating at the ____ layer") leaked the answer because *infrastructure* was the only word that fits "accelerator firmware or container orchestration." Rewrote the stem so the reader must reason from an attack-effect signature ("every model running above this layer rather than one model or API in isolation") plus a specific mechanism (compromised scheduler co-locating serving pod with secret sidecar, then stealing GPU firmware credentials to read across tenants). The blank is now inferred from the cross-workload blast-radius property, matching the §16 FILL gold-standard pattern where the blank names a described mechanism rather than the stem.

### 3. `easy_tf` and `throwaway_distractor` weakening distractor design (§9 TF and MCQ anti-patterns)

- **When ML Systems Become Attack Tools, q3** (easy_tf) — the TF ("attacker-defender asymmetry exists because attackers iterate without latency, compliance, and uptime constraints") restated the chapter's thesis with grammar that made True obvious. Converted to a scenario MCQ contrasting a 200-iterations-per-day red-team LLM against a two-week compliance+regression+staged-rollout release gate, asking which specific constraint most bounds defender iteration speed. Distractors now encode real mental-model failures: raw compute, training-data access, and loss-function choice — each a plausible but wrong diagnosis of the operational bottleneck.

- **Hardware-Level Security, q1** (throwaway_distractor) — the prior MCQ listed "Secure boot failure" alongside side-channel, fault injection, and counterfeit hardware as peer threat categories, but secure boot failure is not a peer category in the chapter's taxonomy. Replaced with "Hardware bug exploitation" (a real peer category from the section), making the classification question genuinely four-way rather than effectively three-way with a label-mismatch throwaway.

## One substantial rework section

**Learning from Security Breaches**. Two coordinated changes. First, the q4 ORDER-to-MCQ conversion described above — now a concrete AV scenario that tests analogy-matching, not lifecycle sequencing. Second, the other four items in the section were lightly tightened to make their scenarios more concrete: the Stuxnet lesson question now names the telemetry-reporting-normal behavior pattern; the Jeep Cherokee SHORT explicitly names the CAN bus and the segmentation remediation; and the Mirai MCQ now names the 200,000-device scale that motivates the amplification argument. Distractor refutation stays content-based throughout (no letter refs). Letter distribution within the section balanced to A=1, C=1, D=2.

## Other edits worth noting

- **Summary q1** (flagged `recall_only`) rewritten from a topic-sentence paraphrase to a hospital clinical-note LLM scenario asking which *combination* of four control sets addresses both the security and privacy concerns emphasized by the chapter, forcing the reader to integrate DP-SGD, secure aggregation, RBAC/MFA, prompt filtering, PII monitoring, and TEE attestation into a single layered answer.
- **Summary q3** LO tightened per the audit's suggested fix, to "Evaluate why hardware trust anchors are necessary when software layers may be compromised below the application boundary" — more testable and operational than the prior "foundational" framing.
- **Model-Specific Attack Vectors q1** question stem added the 100,000-query budget anchor from the chapter's own numbers so the scenario is quantitatively calibrated rather than just qualitative.
- **Comprehensive Defense Architectures q4 (ORDER)** kept — this ORDER is genuinely sequential (detection must precede containment; containment must precede deeper forensics) and the answer explicitly names each item, satisfying §16 ORDER gold-standard.

## JSON + validator status

```
python3 book/tools/scripts/genai/quiz_refresh/validate_quiz_json.py \
  book/tools/scripts/genai/quiz_refresh/_audit/opus_improved/vol2_security_privacy_quizzes.json \
  book/quarto/contents/vol2/security_privacy/security_privacy.qmd

OK: vol2_security_privacy_quizzes.json passes schema + anchor validation
```

- Schema + anchors: pass
- MCQ letter-reference scan (`Option [A-D]`, `Choice [A-D]`, `Answer [A-D]`, `([A-D])`): zero warnings
- Metadata counts: `total_sections=11`, `sections_with_quizzes=11`, `sections_without_quizzes=0` — match actual content
- Type mix: MCQ 58.9%, SHORT 30.4%, TF 7.1%, FILL 1.8%, ORDER 1.8% (corpus target 40/30/13/8/9 — MCQ slightly over-represented because the chapter's breadth rewards classification questions across attack surfaces; TF, FILL, and ORDER are at 1 each because the material is better tested as scenarios than as constrained formats)
