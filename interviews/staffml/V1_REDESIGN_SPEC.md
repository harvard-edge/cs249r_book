# StaffML V1 Redesign Spec — Final Consensus

> Output of a 3-round feedback loop: 4 user personas (student, senior MLE, hiring manager, career switcher) → product strategist + growth expert → editor-in-chief synthesis.

---

## Core Insights

**#1: Onboarding is the bottleneck.** Every persona said "I don't know where to start." The app has 5,198 questions but no guided entry point. Fix this and the content sells itself.

**#2: Levels are opaque.** Users don't know what L1-L6+ means. The Bloom's Taxonomy mapping is the product's moat — but it needs to be visible, labeled, and exemplified everywhere.

**#3: Acquisition before retention.** With zero users, streaks and spaced repetition are invisible. V1 must create a viral loop that brings people in. Retention mechanics follow in V1.1.

---

## The Levels — The Backbone

Every surface in StaffML references these. Dual labels: academic + practical.

| Level | Name | What it tests | Example question | Target role |
|-------|------|---------------|------------------|-------------|
| L1 | **Recall** (Entry) | Can you name the thing? | "What does GPU HBM stand for?" | Intern / new grad studying |
| L2 | **Understand** (Junior) | Can you explain why it matters? | "Why is memory bandwidth often more important than FLOPS for inference?" | Junior MLE |
| L3 | **Apply** (Mid-Level) | Can you use it to solve a problem? | "Calculate the minimum batch size to saturate an H100's memory bandwidth for BERT inference." | Mid-level MLE |
| L4 | **Analyze** (Senior) | Can you compare trade-offs and diagnose? | "Your serving latency spiked 3x after upgrading from FP16 to FP8. What are the likely causes?" | Senior MLE |
| L5 | **Evaluate** (Staff) | Can you make architecture decisions under constraints? | "Design a serving stack for Llama-70B at 10K QPS with a $50K/month GPU budget." | Staff MLE |
| L6+ | **Architect** (Principal) | Can you design novel systems and anticipate failure modes? | "Design fault-tolerant training for a 1T param model across 3 data centers." | Staff+ / Principal |

### How levels appear in the UI
1. **Level badges on every question** — colored pill: "L3: Apply" not just "L3"
2. **Level explainer** — inline on landing page (not a modal), shown to first-time users
3. **Placement quiz** — 3 questions, inline on landing page, 90 seconds, auto-places you
4. **Placement result** — "You're solid at L3 (Apply). Staff-level starts at L5." ← This is the aha moment.

---

## V1 Must Ship (6 features)

| # | Feature | Rationale | Status |
|---|---------|-----------|--------|
| 1 | **Level definitions everywhere** | Unanimous. The moat. Dual labels, badges, explainer card. | Not started |
| 2 | **Inline placement quiz** | 3 questions, 90 seconds, on the landing page. No modal, no signup. The aha moment: seeing your level gap. | Not started |
| 3 | **Vault-first landing page** | Show the content immediately. Taxonomy browser as hero, slim intro above, placement quiz for first-timers. | Not started |
| 4 | **Simplified nav** | 4 primary (Browse, Practice, Gauntlet, Progress) + Tools dropdown (Plans, Roofline, Simulator). | Not started |
| 5 | **Light/dark theme** | User feedback. | ✅ Done |
| 6 | **Shareable challenge links** | `staffml.com/challenge/abc123` — complete a question set, send to a friend, they see your score after finishing. This is the viral loop. | Not started |

### What got CUT from V1 (and why)

| Cut feature | Why |
|-------------|-----|
| GitHub Star gate | Consensus: gating core content backfires. Stars earned through coercion → resentment. Instead: polite ask post-Gauntlet. |
| QuestionPanel unification | Important but invisible to users. Ship V1.1 when adding follow-up chains. |
| Guided onboarding modal (4 steps) | Too heavy. Replaced by inline placement quiz (zero steps, 90 seconds). |
| Shareable daily results card | Challenge links are the better viral mechanic. Daily results card is V1.1. |

---

## Growth Mechanics

### The Star Ask (not a gate)
- After completing a Gauntlet, show: "StaffML is free and open source. If it helped, star us on GitHub." with a direct link
- No gate, no limit, no lockout
- Polite, grateful, non-coercive
- Repeat at most once per week

### Challenge Links (the viral loop)
- User completes a Gauntlet or a curated 3-question set
- Gets a shareable URL: `staffml.com/challenge/abc123`
- Recipient lands on the same questions with a timer
- After finishing: sees challenger's score vs. their own
- Loop: complete → challenge → recipient signs up → completes → challenges someone else

### HN/Reddit Launch Framing
- "We built 5,198 ML systems interview questions using Bloom's Taxonomy — open source, no accounts, runs entirely in your browser"
- Lead with the placement quiz as the first click
- Lean hard into: local-first, no tracking, physics-grounded

---

## Onboarding — Streamlined

### First Visit (no localStorage)
```
Landing page loads with:
  ┌─────────────────────────────────────────────┐
  │ StaffML                                     │
  │ 5,198 physics-grounded ML systems questions │
  │ 100% client-side · No accounts · No tracking│
  │                                             │
  │ [Find Your Level — 90 seconds]              │
  │        or [Browse All Questions]            │
  └─────────────────────────────────────────────┘

  ↓ Below the fold: vault browser (full taxonomy)

"Find Your Level" triggers inline placement quiz:
  → 3 questions (L2, L3, L4) shown one at a time
  → Quick self-assessment per question
  → Result: "You're at L3 (Apply, Mid-Level). Staff starts at L5."
  → [Start Practicing at L4 →]
  → Vault auto-filters to recommended level
```

### Returning Visit (has localStorage)
- Collapsed hero: "Welcome back · 12-day streak · 5 due for review"
- [Continue Practicing →]
- Vault browser below

---

## V1.1 Fast Follow (ship within 2 weeks of V1)

| # | Feature | Rationale |
|---|---------|-----------|
| 1 | **QuestionPanel unification** | Single component for all question views. Prerequisite for follow-up chains. |
| 2 | **Streak counter + daily challenge** | Retention mechanic. Activate once there's a user base. |
| 3 | **Spaced repetition scheduling** | Surfaces weak-area questions based on past performance. |
| 4 | **Embeddable question widgets** | Viral content for blogs, Twitter, course sites. |
| 5 | **Level-aware heat map** | Area × level grid so staff-level preppers can verify L5+ mastery. |

---

## V1.2+ Backlog

| Feature | Persona |
|---------|---------|
| Follow-up question chains in Gauntlet | Hiring manager |
| Study plan day-by-day with reading links | Career switcher |
| "Why this matters" tooltips on simulator/roofline | Career switcher |
| Company-specific filtering | Student |
| LLM-as-judge for free-text answers | Senior engineer |
| Interviewer view (stem + rubric only) | Hiring manager |
| Community question flagging | Senior engineer |
| Progress-over-time sparklines | Career switcher |

---

## Implementation Phases

| Phase | What | Scope |
|-------|------|-------|
| A | **Ship V1** | Levels + placement quiz + vault-first landing + simplified nav + challenge links |
| B | **Observe + iterate** | Watch usage data, fix what's broken, ship V1.1 based on actual behavior |

No 8-phase waterfall. Two phases. Ship and learn.

---

## Persona Satisfaction Check

| Persona | V1 solves their #1? | Remaining gap |
|---------|---------------------|---------------|
| **Anika** (student) | ✅ Placement quiz + levels = knows where to start | Company-specific filtering (V1.2) |
| **Derek** (senior MLE) | ⚠️ Levels help, but QuestionPanel unification is V1.1 | Gauntlet still lacks rubric until V1.1 |
| **Riya** (hiring manager) | ⚠️ Challenge links give her shareability, but follow-up chains are V1.2 | Interviewer view (V1.2) |
| **Marcus** (career switcher) | ⚠️ Placement quiz gives direction, but study plans need work | Day-by-day breakdown (V1.2) |

**Honest assessment:** V1 fully solves 1 of 4 personas (Anika). The other 3 get meaningful improvement but full satisfaction requires V1.1-V1.2. This is acceptable for a first launch — the placement quiz + challenge links create the acquisition loop, and fast follows close the gaps.
