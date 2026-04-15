# StaffML Vault Migration — Session Kickoff

> **Purpose**: This file is the copy-paste prompt to start a fresh Claude Code session for the StaffML vault architecture migration.
>
> **How to use**: Launch Claude Code from `/Users/VJ/GitHub/MLSysBook-staffml`, then paste the block below verbatim.

---

## Copy-paste this into the new session

```
You are starting the StaffML vault architecture migration. This is a planned, staged project. Do not start implementing. Read the plan first, review it, harden it, and only execute after explicit user green-light.

# Context (read these in order, before anything else)

1. interviews/vault/ARCHITECTURE.md — the full 21-section design document. Read end-to-end.
2. Project memory: project_vault_architecture.md, project_staffml_ask_interviewer.md, project_staffml_session_state.md, feedback_staffml_workflow.md, feedback_staffml_quality_bars.md.
3. interviews/vault/SYSTEM.md — stale but useful context on how the current pipeline evolved.

# Current branch situation (verify before any action)

- You are in /Users/VJ/GitHub/MLSysBook-staffml, which is a git worktree on branch `dev`.
- The user's primary working branch in /Users/VJ/GitHub/MLSysBook is `feat/mitpress-vol1-copyedit-r1` — DO NOT TOUCH.
- 10 worktrees exist total, one per feature. Keep actions scoped to the staffml worktree.

# FIRST ACTION: create the feature branch

Before Stage 1 reviews, before any doc edits, create the dedicated feature branch
so ALL work (review integrations, testing spec, eventual implementation) lives
on one branch instead of polluting `dev`:

```bash
cd /Users/VJ/GitHub/MLSysBook-staffml
git pull --ff-only origin dev
git checkout -b feat/vault-architecture
```

All review commits (ARCHITECTURE.md v2, v3), testing plan commits (TESTING.md,
CUTOVER_QA.md), and implementation commits (Phases 0–6) land on this branch.
Push to origin after first real commit: `git push -u origin feat/vault-architecture`.

# Your task, in three stages

## STAGE 1 — Plan review (iterative, 2–3 rounds)

Run ARCHITECTURE.md by four expert reviewers IN PARALLEL. Send each the same brief: "Read this architecture document cold. Identify risks, gaps, bugs, and design weaknesses. Rank findings by severity (critical/high/medium/low)."

Reviewers:
- expert-chip-huyen  (production ML, DX, security)
- expert-jeff-dean   (large-scale systems, reliability)
- expert-soumith-chintala (framework & API design)
- student-david      (industry-user perspective on ergonomics)

After round 1:
- Aggregate all findings into a table at interviews/vault/REVIEWS.md: {issue, reviewer, severity, proposed-fix, status}.
- Integrate every critical and high item into ARCHITECTURE.md v2.
- For medium items: address OR explicitly defer with written rationale.
- Low items: log, defer, not blocking.
- Commit: `docs(vault): architecture v2 (round-1 review integration)`.

Round 2: send v2 plus the response table back to the same reviewers. "Does this address your concerns? What did we miss? Anything new?"
- If no new critical/high: plan is ready. Proceed to Stage 2.
- If new critical/high: integrate into v3, run round 3.

Round 3 (conditional): final adversarial pass. After this, any remaining critical/high must be resolved by explicit engineering decision documented inline, or the plan is not ready for execution.

## STAGE 2 — Testing plan

Write interviews/vault/TESTING.md. Use §19 of ARCHITECTURE.md as the skeleton. Flesh out:
- Test fixtures (the 20-question frozen test corpus, golden vault.db, drift fixtures)
- Full unit + integration + contract test inventory
- CI workflow spec for .github/workflows/vault-ci.yml
- Cutover manual QA checklist (expand from §19.4)
- Observability + rollback protocol for Phase 4

Write CUTOVER_QA.md in vault-cli/docs/.

Commit: `docs(vault): detailed testing plan and cutover QA`.

## STAGE 3 — Gate and wait

Post a summary to the user:
- Rounds of review completed
- Critical/high issues found and how they were resolved
- Testing plan overview
- Readiness assessment: green / yellow / red

WAIT for explicit user green-light ("proceed", "go ahead", "execute") before Stage 4.

## STAGE 4 — Autonomous execution (only after green-light)

Execute Phases 0 → 6 per ARCHITECTURE.md §14:

- Phase 0: scaffold vault-cli Python package
- Phase 1: question schema + per-file YAML migration + basic CLI commands
- Phase 2: release pipeline + fixes to paper/site disagreement bugs
- Phase 3: Cloudflare D1 database + staffml-vault worker
- Phase 4: website cutover (the risky one — follow §19.5 observability)
- Phase 5: chain discoverability UX (§8)
- Phase 6: About page paper prominence (§9)

Rules during autonomous execution:
- Stay on the feat/vault-architecture branch created at the start (do not branch again).
- Work only inside: interviews/vault/, interviews/vault-cli/, interviews/staffml/src/lib/corpus.ts, interviews/staffml/src/lib/vault-api.ts, interviews/staffml-vault-worker/, interviews/paper/scripts/, interviews/staffml/src/data/* (deletions).
- Atomic commits, no Co-Authored-By, descriptive messages, one logical change per commit.
- After every phase: run full test suite, commit, push, post a phase-complete summary, wait for brief user ack, then proceed.
- Stop conditions (ask user): failing test not covered by TESTING.md, ambiguous invariant, surprising design cost, anything outside the scoped directories, user asks.
- Never force-push. Never merge to dev without explicit user approval. Never delete data without a rollback path in the same commit.

# Success definition

ARCHITECTURE.md §20.5 is the checklist. All nine criteria green = done.

# Immediate first action

After reading ARCHITECTURE.md + memory, before anything else: spawn the four reviewers in parallel for Round 1. Do not begin implementation. Do not write any code or files other than REVIEWS.md until the plan has cleared review and the user has green-lit execution.
```

---

## Notes on using this kickoff

- **Paste the block above in one message.** Do not split it.
- **Before pasting**, make sure you're in the staffml worktree:
  ```bash
  cd /Users/VJ/GitHub/MLSysBook-staffml
  claude
  ```
- **Timing**: start this session AFTER the 2026-04-22 MIT Press copyedit deadline. Any modification to paper/macros.tex during copyedit conflicts with the typesetter.
- **Expected duration of Stage 1**: 1–2 days of calendar time (reviewers run in parallel, iteration happens across chat turns with you between rounds).
- **Expected duration of Stage 4**: ~11 working days per ARCHITECTURE.md §14.

## If something goes wrong mid-execution

The autonomous operator is instructed to stop and ask. But if YOU see something concerning:
- Interrupt the session, describe the issue plainly.
- The operator should capture the concern in REVIEWS.md and adjust ARCHITECTURE.md before resuming.
- If the concern is strategic (wrong direction), use the brake: "pause all autonomous execution, let's re-plan."

## Keeping this file accurate

If ARCHITECTURE.md materially changes — especially §14 (phases), §19 (testing), §20 (autonomous rules) — update this kickoff file in the same commit. The prompt must stay consistent with the plan it invokes.

---

**Last synchronized with ARCHITECTURE.md**: 2026-04-15 (initial version).
