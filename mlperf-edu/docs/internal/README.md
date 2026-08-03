# Internal Working Documents

These are project working notes, not user documentation. They record how the
suite got to its current state and what remains before release. Nothing here is
required to install, run, or interpret a benchmark.

If you downloaded MLPerf EDU to run it, the documents you want are at the
repository root:

| Document | Purpose |
|:---|:---|
| [README](../../README.md) | What this is, install, and a first run |
| [INSTALL](../../INSTALL.md) | Environment setup in detail |
| [SPEC](../../SPEC.md) | The normative v0.1 specification |
| [PUBLIC_RULES](../../PUBLIC_RULES.md) | What may be claimed about a result |
| [SECURITY_REVIEW](../../SECURITY_REVIEW.md) | Generated-code and runtime boundaries |

Measurements are deliberately absent from all of the above. Numbers belong to
the run that produced them, so every quality decision, timing distribution, and
hardware fingerprint lives in the report a run writes on your machine.

## What Is Here

**Direction and plan**

- [SHIP_PLAN](SHIP_PLAN.md) is the current plan of record: the acceptance test
  for a downloadable benchmark, the blocking decisions, and the ordering.
- [NORTH_STAR](NORTH_STAR.md), [PROPOSAL](PROPOSAL.md), and
  [DESIGN_PHILOSOPHY](DESIGN_PHILOSOPHY.md) record the original framing. They
  overlap heavily and are candidates to merge into one statement.

**Status ledgers**

- [WORK_CHECKLIST](WORK_CHECKLIST.md) is the maintained progress view.
- [READINESS](READINESS.md) holds the per-workload evidence detail.
- [PRODUCT_READINESS_PLAN](PRODUCT_READINESS_PLAN.md) describes the intended
  classroom and research experience.
- [LOCAL_EXECUTION_PLAN](LOCAL_EXECUTION_PLAN.md) defines the remaining DLRM,
  MiniGo, and asset work.
- [RELEASE_CHECKLIST](RELEASE_CHECKLIST.md) tracks release gates.

These four ledgers cover the same fourteen workloads and the same two blockers.
That overlap is the known cause of claim drift and they should collapse into
one.

**Reviews and decisions**

- [QUALITY_TARGET_REVIEW](QUALITY_TARGET_REVIEW.md) records the authority behind
  every threshold. This is the most reusable document here.
- [DATASET_RELEASE_REVIEW](DATASET_RELEASE_REVIEW.md) records redistribution
  decisions.
- [COURSE_BUDGETS](COURSE_BUDGETS.md) records measured planning ceilings.
- [INDEPENDENT_AUDIT](INDEPENDENT_AUDIT.md) and
  [USER_JOURNEY_AUDIT](USER_JOURNEY_AUDIT.md) are dated review snapshots. Treat
  them as an archive pair rather than living documents.

## Maintenance Rule

A fact belongs to exactly one document. When a claim about workload counts,
evidence classes, or quality outcomes appears in more than one place, the copies
drift and the published ones go stale. Prefer a cross-reference to a restatement,
and prefer generating a claim from the registry to writing it down at all.
