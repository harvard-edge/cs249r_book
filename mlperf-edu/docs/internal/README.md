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

Measurements are deliberately absent from all of the above, and from everything
here. Numbers belong to the run that produced them, so every quality decision,
timing distribution, and hardware fingerprint lives in the report a run writes
on your machine.

## The Three Live Documents

| Document | Answers |
|:---|:---|
| [DIRECTION](DIRECTION.md) | Why this exists, what it refuses to do, how a workload is admitted |
| [STATUS](STATUS.md) | Decisions and open work |
| [WORKLOAD_STATUS](WORKLOAD_STATUS.md) | Generated: per-workload quality, runtime, and gaps |
| [SHIP_PLAN](SHIP_PLAN.md) | The plan of record for making it downloadable |

`DIRECTION` replaced three documents that shared a verbatim mission paragraph
and restated the same design commitments. `STATUS` replaced four checkbox
ledgers that covered the same fourteen workloads and the same two blockers,
listing several open items in three slightly different wordings each. That
duplication was the direct cause of every stale published claim found in audit,
because correcting one copy left the others wrong.

## Reference and Archive

- [QUALITY_TARGET_REVIEW](QUALITY_TARGET_REVIEW.md) records the authority
  behind every threshold. The most reusable document here.
- [DATASET_RELEASE_REVIEW](DATASET_RELEASE_REVIEW.md) records redistribution
  decisions.
- [COURSE_BUDGETS](COURSE_BUDGETS.md) records measured planning ceilings and is
  the source the paper reads for reference-host facts.
- [RELEASE_CHECKLIST](RELEASE_CHECKLIST.md) tracks release gates.
- [INDEPENDENT_AUDIT](INDEPENDENT_AUDIT.md) and
  [USER_JOURNEY_AUDIT](USER_JOURNEY_AUDIT.md) are dated review snapshots. Treat
  them as an archive pair, not living documents.

## Maintenance Rule

A fact belongs to exactly one document. When a claim about workload counts,
evidence classes, or quality outcomes appears in more than one place, the copies
drift and the published ones go stale.

In order of preference: generate the claim from the registry, cross-reference
the one document that owns it, or, last, write it down. Anything countable
should come from `tools/audit_workload_readiness.py`,
`provisional_results/index.json`, or the registry itself rather than from prose.
