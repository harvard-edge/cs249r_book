# StaffML Development Log

## 2026-03-22 — Build Cycle 1: Full App Rebuild (3 feedback rounds)

### What was built
Restructured the monolithic 872-line `page.tsx` into a proper Next.js route-based architecture with 4 pages, shared infrastructure, and 3 rounds of expert feedback applied.

**Architecture:**
```
src/app/
  layout.tsx          — Shared layout with Nav
  page.tsx            — Landing page (hero, features, stats, CTA)
  gauntlet/page.tsx   — The Gauntlet (timed mock interview)
  heatmap/page.tsx    — Readiness Heat Map (progress dashboard)
  drill/page.tsx      — Drill Mode (focused practice + napkin math)

src/components/
  Nav.tsx             — Responsive nav with mobile hamburger menu

src/lib/
  corpus.ts           — Question data + napkin math (gradient scoring, answer markers)
  progress.ts         — LocalStorage progress with caps (500 attempts, 50 gauntlets)
  hooks.ts            — useMounted() utility hook
```

### Features
1. **Landing page** — Hero with live question count, feature cards, stats bar, moat section, footer CTA
2. **The Gauntlet** — Track/level/duration → timed quiz → self-assessment → results with competency breakdown + elapsed time. Auto-finishes when timer expires.
3. **Heat Map** — Competency × Track grid, color-coded, clickable cells → drill, clear progress
4. **Drill Mode** — Filter sidebar with "weakest area" recommendation, napkin math verification with gradient scoring, constrained self-assessment, URL params from heat map
5. **Napkin Math v2** — 5-tier grades (exact/close/ballpark/off/way_off), answer markers (`=>`), caps self-assessment to prevent overconfidence
6. **Progress tracking** — LocalStorage with caps, feeds heat map and weakest area recommendations
7. **Gauntlet algorithm** — Round-robin across competency areas for breadth
8. **Keyboard shortcuts** — ⌘Enter to reveal, 1-4 for scoring, N to skip

---

### Round 1: UX Designer Feedback
**19 issues found. 13 applied, 4 deferred.**
- ✅ Timer expiry handling (auto-finish, score unanswered as 0)
- ✅ Mobile hamburger nav
- ✅ Responsive grids (2→4 cols on mobile)
- ✅ Panel width consistency (460px everywhere)
- ✅ Self-assessment label consistency
- ✅ `cleanScenario()` utility
- ✅ Dead CSS cleanup
- ✅ Clickable heat map cells → drill
- ✅ Elapsed time in results

### Round 2: Frontend Developer Feedback
**18 issues found. 6 applied, rest deferred or already fixed.**
- ✅ Timer zero-state (already fixed in Round 1)
- ✅ Dead imports removed (ReactMarkdown, AnimatePresence, ChevronRight)
- ✅ localStorage capped (500 attempts, 50 gauntlets)
- ✅ `useMounted()` hook created
- ✅ Aria progressbar on gauntlet
- Deferred: corpus lazy loading, server components, error boundaries

### Round 3: Jordan (ML Engineer) Product Review
**7 areas covered. 5 changes applied.**
- ✅ Napkin math gradient (5 grades instead of binary pass/fail)
- ✅ Answer markers (`=>` prefix for final answer)
- ✅ Constrained self-assessment (napkin math caps max score)
- ✅ "Weakest area" recommendation in drill sidebar
- ✅ Level labels clarified (Bloom's → difficulty mapping explained)
- Deferred: spaced repetition, shareable heat map, rubric-based scoring

**Jordan's verdict:** "Would use it for napkin math practice. The physics-grounded angle is the exact differentiation that matters for Staff-level. Core thesis is right."

### Build status
- ✅ `npm run build` passes — 7 routes, 0 errors
- ✅ TypeScript strict — no `any` types
- ✅ All 3 feedback rounds applied

### Deferred to next session
- [ ] Spaced repetition (SM-2 or Leitner scheduler)
- [ ] Shareable heat map export
- [ ] Rubric-based scoring checkboxes
- [ ] Error boundaries
- [ ] Server components for static content
- [ ] Favicon + og:image meta tags
- [ ] Continue vault fills to push 3D coverage
