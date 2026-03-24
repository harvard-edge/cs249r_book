# StaffML Autonomous Build Plan

## Unique Value Proposition

StaffML sits at the intersection of three things nobody else combines:

1. **mlsysim** — A physics engine with 470+ hardware constants, 25+ formulas, 15 solver models. Real H100/B200/TPU specs, not made-up numbers. This is the computational backbone.

2. **The Textbook** — MLSysBook (Harvard, MIT Press) gives every question academic grounding. Deep-dive links connect practice to theory. Questions are Bloom's-calibrated.

3. **Interactive Napkin Math** — No other tool makes you compute. LeetCode tests algorithms. Exponent tests storytelling. StaffML tests whether you can estimate tokens/sec on an H100 with a 70B model from first principles.

**The moat:** To replicate StaffML, a competitor would need to build a hardware physics engine, write 1,800+ questions grounded in real specs, AND build the interactive tool. That's years of work.

## Current State (12 commits)

- 6 routes: Home, Daily, Gauntlet, Heat Map, Drill, (+ 404)
- 1,815 questions with competency areas + company archetypes
- SM-2 spaced repetition, rubric checkboxes, napkin math v2
- Streak system, toast notifications, welcome back card
- Hardware reference card (mlsysim constants)
- Roofline logo, favicon, OG meta
- Static export ready for Vercel
- Dead deps removed, unused components cleaned

## Autonomous Execution Loop

For each cycle:
1. **Pick highest-impact item** from the backlog
2. **Build it** (write code, test with `npm run build`)
3. **Launch 2-3 feedback agents** in parallel (student personas, UX, frontend)
4. **Apply feedback** immediately
5. **Commit** with clean message
6. **Update DEVLOG** with what changed
7. **Repeat** until backlog is empty or feedback saturates

## Priority Backlog (ordered by impact)

### Tier 1: Ship-blocking (must have for launch)
- [x] Static export config (next.config.mjs)
- [ ] Study Plans page (/plans) — "MLE Interview in 2 Weeks" curated sequences
- [ ] Interactive Roofline chart — THE signature feature matching the logo
- [ ] README rewrite with hero screenshot + quick start

### Tier 2: Engagement multipliers
- [ ] Shareable gauntlet/daily results as canvas-to-PNG
- [ ] Activity calendar (GitHub-style contribution grid)
- [ ] Manifest.json for PWA installability

### Tier 3: Content depth (mlsysim integration)
- [ ] "What-If Calculator" — tweak batch size/precision/hardware, see throughput change
- [ ] Per-question hardware context — show which specs are relevant to THIS question
- [ ] Textbook deep-dive integration — inline chapter previews

### Tier 4: Growth
- [ ] SEO: per-track landing pages, structured data
- [ ] Social sharing cards for results
- [ ] GitHub star count badge on landing page
