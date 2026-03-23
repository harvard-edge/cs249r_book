# StaffML Development Log

## 2026-03-22 — Full Build Session (18 commits)

### Architecture
```
src/app/           — 7 routes (Home, Daily, Gauntlet, Heat Map, Drill, Plans, Roofline)
src/components/    — 6 components (Nav, HardwareRef, HardwareConfigurator, StreakBadge, Toast, Providers)
src/lib/           — 6 modules (corpus, hardware, plans, progress, rubric, hooks)
public/            — favicon.svg, 5 logo concepts (10 SVGs)
```

### Features Built
1. Route architecture — rebuilt from 872-line monolith to 7 clean routes
2. The Gauntlet — timed mock with competency breadth, timer expiry, elapsed time
3. Drill Mode — filters (track/level/competency/interview style), napkin math v2 (5-tier gradient), rubric checkboxes, spaced repetition review queue, weakest area recommendation, hardware reference card
4. Heat Map — competency x track grid, clickable cells, readiness verdict, per-track bars, export/import
5. Daily Challenge — deterministic 3 questions/day, completion tracking, summary card
6. Study Plans — 5 curated paths (72hr Blitz, MLE Sprint, Staff Deep Dive, Edge, Mobile)
7. Interactive Roofline — SVG chart with real mlsysim specs, 6 workload dots, custom OI input
8. Streak system — daily tracking, flame badge, milestone toasts
9. Welcome Back card — personalized landing for returning users
10. Company archetypes renamed to interview styles per Jordan feedback
11. Progress export/import for data portability
12. Toast notification system for badge milestones
13. Roofline logo (5 concepts), favicon, OG/Twitter meta
14. Dead deps removed (-3,607 lines), unused components cleaned
15. Static export configured (next.config.mjs)

### Feedback Rounds (5 reviewers, 3 rounds)

**Round 1 — UX Designer + Frontend Dev + Jordan (ML Engineer):**
- 43 issues found, 24 fixed immediately
- Timer expiry, mobile nav, responsive grids, self-assessment consistency
- Napkin math gradient, answer markers, constrained scoring

**Round 2 — Jordan revisit:**
- "Round 1 prototype. Round 2 daily habit tool."
- Archetype rename applied, readiness verdict built

**Round 3 — Emma (beginner) + David (industry) + Sophia (distributed):**
- Emma: "Locked door for beginners" — too much jargon, no learning path. Valid but wrong target audience.
- David: "Strong 0.8" — export/import needed (done), rubric heuristic fragile, missing ML monitoring category
- Sophia: "Useful but thin on distributed" — wants cluster simulator, multi-GPU roofline, collective ops coverage

### What's Saturating
App features are getting positive feedback. Remaining asks are:
- **Content depth** (authored rubrics, distributed systems concepts, follow-up questions)
- **Simulation** (cluster-level training simulator using mlsysim formulas)
- **Social** (shareable results, user count, testimonials)

### Next Session Priorities
1. Distributed training simulator (Sophia's killer feature request)
2. Authored rubric items in corpus (all reviewers asked for this)
3. Deploy to Vercel
4. README rewrite with hero screenshots
5. Continue vault fills toward 85% 3D coverage
