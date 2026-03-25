# StaffML Iteration Plan

**Last updated:** 2026-03-22
**Current state:** Build Cycle 1 complete. 4 routes, 1,815 questions, dark theme, SM-2 spaced repetition, rubric scoring, napkin math with gradient grading. `npm run build` passes clean.

---

## Baseline Audit

### What exists (working)
- Landing page with live stats, feature cards, moat section, footer CTA
- The Gauntlet: track/level/duration selection, timed quiz, self-assessment, competency breakdown results
- Heat Map: competency x track grid, color-coded, clickable cells link to drill
- Drill Mode: filter sidebar, weakest-area recommendation, napkin math gradient scoring (5 tiers), rubric checkboxes, spaced repetition review queue, URL params from heat map
- Progress: LocalStorage with caps (500 attempts, 50 gauntlets), SM-2 spaced repetition cards
- Nav: responsive with mobile hamburger
- Keyboard shortcuts: Cmd+Enter reveal, 1-4 scoring, N skip

### What is missing
- No favicon configured in layout.tsx (logos exist in /public but not linked)
- No og:image or social meta tags
- No streak tracking
- No shareable results
- No daily challenge
- No leaderboard
- Unused dependencies: mermaid, reactflow, react-markdown, remark-gfm (bloating bundle)
- corpus.json is 3.7 MB loaded as static import (entire bundle issue)
- No error boundaries
- No next.config.js (no bundle analyzer, no image optimization config)
- Components dir has unused files: AnimatedFlow.tsx, HardwareConfigurator.tsx, MermaidRenderer.tsx, scenarios/ArchitectureDebugger.tsx, scenarios/MultipleChoiceEngine.tsx
- Empty ui/ directory (shadcn never installed)

---

## Sprint 1: Polish + Social Presence

**Goal:** Make StaffML screenshot-ready and shareable so every visitor becomes a distribution channel.

### Features to build

1. **Favicon + app icons**
   - In `layout.tsx`, add `<link rel="icon" href="/logo-concept-1-favicon.svg" type="image/svg+xml" />` and an apple-touch-icon variant (export a 180x180 PNG from the SVG)
   - Add `manifest.json` in `/public` with `name`, `short_name`, `icons`, `theme_color: "#000"`, `background_color: "#000"`

2. **Open Graph + Twitter meta tags**
   - In `layout.tsx` metadata, add:
     ```typescript
     openGraph: {
       title: "StaffML — ML Systems Interview Prep",
       description: "1,800+ physics-grounded system design questions. Napkin math, bottleneck analysis, architecture trade-offs.",
       url: "https://staffml.ai",
       siteName: "StaffML",
       images: [{ url: "/og-image.png", width: 1200, height: 630 }],
       type: "website",
     },
     twitter: {
       card: "summary_large_image",
       title: "StaffML — ML Systems Interview Prep",
       description: "Free, open-source ML systems interview prep with napkin math verification.",
       images: ["/og-image.png"],
     }
     ```
   - Create og-image.png: 1200x630, dark background, StaffML logo, tagline "Prep for your Staff ML Systems interview", stats ("1,800+ questions, 12 competency areas, 4 tracks"), subtle grid pattern background

3. **Streak counter**
   - Add to `progress.ts`:
     ```typescript
     export interface StreakData {
       currentStreak: number;   // consecutive days
       longestStreak: number;
       lastActiveDate: string;  // ISO date string YYYY-MM-DD
     }
     const STREAK_KEY = 'staffml_streak';
     export function updateStreak(): StreakData { ... }
     export function getStreak(): StreakData { ... }
     ```
   - Logic: on each `saveAttempt()`, call `updateStreak()`. Compare `lastActiveDate` to today. If same day, no-op. If yesterday, increment `currentStreak`. If older, reset to 1. Update `longestStreak` if exceeded.
   - Display streak as a fire icon + number in the Nav component, right side: `🔥 7` (use Flame icon from lucide-react)
   - Display streak card on landing page stats bar: "Current Streak: 7 days"

4. **Share heat map as image**
   - Add a "Share" button to the heat map page header
   - Use `html-to-image` library (add as dependency: `npm install html-to-image`)
   - On click: capture the heat map table as PNG, open browser share dialog via `navigator.share()` with fallback to download
   - Include overlay text on the image: "My StaffML Readiness — staffml.ai"

5. **Create next.config.mjs**
   - At `/interviews/staffml/next.config.mjs`:
     ```javascript
     /** @type {import('next').NextConfig} */
     const nextConfig = {
       output: 'export',  // static export for Vercel
       images: { unoptimized: true },
     };
     export default nextConfig;
     ```

### Expert feedback agents to launch
- **Visual Designer**: Review og-image composition, favicon rendering at 16x16/32x32/180x180, color contrast on streak display
- **Frontend Developer**: Verify `html-to-image` works with Tailwind classes, test `navigator.share()` fallback chain, validate manifest.json

### Success criteria
- [ ] Favicon visible in browser tab
- [ ] Pasting StaffML URL into Slack/Twitter shows rich preview with og-image
- [ ] Streak counter visible in nav after answering one question
- [ ] "Share" button on heat map produces a downloadable PNG
- [ ] `npm run build` still passes clean

### Dependencies
- None (sprint can start immediately)

---

## Sprint 2: Engagement Loop

**Goal:** Give users a reason to come back every day through daily challenges, readiness signals, and streak incentives.

### Features to build

1. **Daily challenge**
   - Add route: `src/app/daily/page.tsx`
   - Algorithm: hash today's date to deterministically pick 3 questions from the corpus (ensures all users get the same daily set)
     ```typescript
     function getDailyQuestions(): Question[] {
       const today = new Date().toISOString().slice(0, 10); // "2026-03-22"
       const seed = hashString(today); // simple string hash → number
       const pool = getQuestions();
       const indices = [seed % pool.length, (seed * 31) % pool.length, (seed * 97) % pool.length];
       return indices.map(i => pool[i]);
     }
     ```
   - UI: card-based, one question at a time, same reveal/score flow as drill but with "1 of 3" progress
   - After completing all 3: show daily summary card with share button
   - Add "Daily Challenge" as a nav item with a subtle badge dot if not completed today
   - Track completion in localStorage: `staffml_daily_completed: { "2026-03-22": true }`

2. **Readiness percentage per track**
   - On the heat map page, add a summary row above the grid:
     ```
     Cloud: 72% ready | Edge: 45% ready | Mobile: 31% ready | TinyML: 0%
     ```
   - Calculate as: (cells with >70% accuracy) / (total cells in track) * 100
   - On the landing page, if user has progress, replace the static stats bar with personalized: "You're 72% ready for Cloud infrastructure interviews"

3. **Streak badges / milestones**
   - Define milestones: 3-day, 7-day, 14-day, 30-day, 60-day, 100-day
   - Store earned badges in localStorage: `staffml_badges: ["streak_3", "streak_7", ...]`
   - Show badge popup (toast) when a milestone is hit using a simple toast component
   - Display earned badges on heat map page below the grid
   - Badge visuals: small SVG icons or styled divs with emoji (Flame, Lightning, Star, Trophy from lucide-react)

4. **"Continue where you left off" on landing page**
   - If user has progress data, show a card below the hero:
     ```
     Welcome back! You have 3 questions due for review and a 5-day streak.
     [Continue Drilling] [Start Daily Challenge]
     ```
   - Pull data from `getDueCount()` and `getStreak()`

5. **Toast notification system**
   - Build a minimal toast component (no library needed):
     ```
     src/components/Toast.tsx — fixed bottom-right, auto-dismiss after 3s, framer-motion slide-in
     src/lib/toast.ts — event emitter pattern: toast.success("Badge earned: 7-day streak!")
     ```
   - Use for: badge earned, daily challenge completed, gauntlet finished

### Expert feedback agents to launch
- **Product Manager**: Review daily challenge UX flow for friction points, validate that readiness percentage is meaningful with sparse data (what if user only attempted 2 cells?)
- **Behavioral Psychologist**: Review streak/badge system for healthy engagement (avoid dark patterns, cap notifications, don't punish streak loss)

### Success criteria
- [ ] Daily challenge shows same 3 questions for all users on a given day
- [ ] Completing daily challenge marks the day as done (no repeat)
- [ ] Streak badge popup appears on milestone days
- [ ] Landing page shows personalized "welcome back" card for returning users
- [ ] Readiness percentages render correctly even with sparse data (show "Not enough data" for <5 attempts)
- [ ] `npm run build` passes clean

### Dependencies
- Sprint 1 (streak counter must exist for badges to build on)

---

## Sprint 3: Community + Virality

**Goal:** Add social proof and sharing mechanisms that drive organic growth and GitHub stars.

### Features to build

1. **Shareable gauntlet results**
   - After gauntlet completion, add "Share Result" button
   - Generate a result card (canvas or html-to-image):
     ```
     ┌────────────────────────────┐
     │  StaffML Gauntlet Result   │
     │  Cloud × L5 — 10 questions │
     │                            │
     │        78%                 │
     │                            │
     │  Compute: ████░░ 80%      │
     │  Memory:  ███░░░ 60%      │
     │  Latency: █████░ 90%      │
     │                            │
     │  staffml.ai                │
     └────────────────────────────┘
     ```
   - Share via `navigator.share()` or copy-to-clipboard with text: "I scored 78% on the StaffML Cloud L5 Gauntlet. Try it: staffml.ai/gauntlet"

2. **Public leaderboard (localStorage-only for Phase 1)**
   - Route: `src/app/leaderboard/page.tsx`
   - For Phase 1 (no backend): show the user's own stats as a "personal leaderboard"
     - Total questions answered
     - Gauntlets completed
     - Highest gauntlet score per track
     - Current streak
     - Mastered competency areas (>70% across all tracks)
   - Design it as a profile card that looks like it belongs on a real leaderboard (rank #1 placeholder)
   - Include a CTA: "Leaderboard coming soon. Star the repo to get notified." with GitHub star button
   - This page is the template for Phase 2 when Supabase adds real multi-user leaderboards

3. **GitHub stars integration**
   - Add GitHub star count to the landing page footer using the GitHub API:
     ```typescript
     // Fetch at build time or client-side with cache
     const res = await fetch('https://api.github.com/repos/harvard-edge/cs249r_book');
     const { stargazers_count } = await res.json();
     ```
   - Display as: "⭐ 1,234 stars on GitHub"
   - Add a more prominent "Star on GitHub" CTA after completing a gauntlet: "If StaffML helped you prep, star the repo!"

4. **Social meta tags per page**
   - Each route gets its own og:title and og:description:
     - `/gauntlet` → "Take the StaffML Gauntlet — timed ML systems mock interview"
     - `/drill` → "Drill ML systems questions with napkin math verification"
     - `/heatmap` → "Track your ML systems interview readiness"
     - `/daily` → "Today's StaffML Daily Challenge"
   - Use Next.js `generateMetadata` in each page file

5. **Nav link to GitHub**
   - Add GitHub icon link in Nav component, right-aligned

### Expert feedback agents to launch
- **Growth Hacker**: Review share card design for social media impact, validate copy on share text, suggest optimal CTA placement for GitHub stars
- **Frontend Developer**: Verify GitHub API rate limits (unauthenticated = 60/hr, cache aggressively), test share functionality across iOS Safari / Chrome / Firefox

### Success criteria
- [ ] Gauntlet result card generates and is shareable
- [ ] Personal leaderboard page renders with real user stats
- [ ] GitHub star count displays on landing page (with fallback if API fails)
- [ ] Each page has unique og:title and og:description
- [ ] `npm run build` passes clean

### Dependencies
- Sprint 1 (og-image and share infrastructure)
- Sprint 2 (streak data for leaderboard)

---

## Sprint 4: Content Quality + Bundle Performance

**Goal:** Audit question quality, improve explanations, slash the bundle size from 3.7 MB corpus to under 500 KB initial load.

### Features to build

1. **Bundle size reduction — corpus splitting**
   - The corpus.json (3.7 MB) is the primary bundle problem. Split it:
     - `corpus-index.json` (~200 KB): `{ id, track, level, competency_area, title, topic }` for all 1,815 questions
     - `corpus-details-{track}.json` (~700 KB each): full question data per track
   - Load index on app init. Load details on-demand when a question is selected.
   - Implementation:
     - Create build script: `scripts/split-corpus.ts` that reads corpus.json and outputs split files to `/public/data/`
     - Update `corpus.ts` to:
       ```typescript
       // Load index synchronously (small)
       import indexData from '../data/corpus-index.json';
       // Load details lazily
       async function loadTrackDetails(track: string): Promise<QuestionDetails[]> {
         const res = await fetch(`/data/corpus-details-${track}.json`);
         return res.json();
       }
       ```
     - Add a loading state to drill and gauntlet while details fetch

2. **Remove unused dependencies**
   - Remove from package.json:
     - `mermaid` — not imported anywhere in active code
     - `reactflow` — not imported in active routes
     - `react-markdown` — not imported in active routes
     - `remark-gfm` — not imported in active routes
   - Run: `npm uninstall mermaid reactflow react-markdown remark-gfm`
   - Estimated savings: ~400 KB from mermaid alone, ~200 KB from reactflow
   - If any of these are used by files in `components/scenarios/` or `components/`, either delete those unused components or keep the dependency

3. **Remove unused components**
   - Audit and remove if not imported anywhere:
     - `src/components/AnimatedFlow.tsx` (uses reactflow)
     - `src/components/HardwareConfigurator.tsx`
     - `src/components/MermaidRenderer.tsx` (uses mermaid)
     - `src/components/scenarios/ArchitectureDebugger.tsx`
     - `src/components/scenarios/MultipleChoiceEngine.tsx`
   - Before deleting: `grep -r "AnimatedFlow\|HardwareConfigurator\|MermaidRenderer\|ArchitectureDebugger\|MultipleChoiceEngine" src/app/` to confirm no imports

4. **Question quality audit**
   - Create script: `scripts/audit-corpus.ts` that checks every question for:
     - `scenario` length > 50 chars (too short = vague)
     - `realistic_solution` length > 100 chars (too short = unhelpful)
     - `napkin_math` field exists for L4+ questions (should have quantitative component)
     - `common_mistake` field exists (improves learning)
     - `deep_dive_url` is a valid URL format
     - No duplicate `title` values
   - Output: `_audit/corpus-quality-report.md` with stats and flagged questions
   - Fix the worst offenders (short answers, missing napkin math on quantitative questions)

5. **Better answer explanations**
   - For questions where `realistic_solution` is under 200 chars, enrich with:
     - Step-by-step reasoning
     - Why the common mistake is wrong
     - A "key insight" callout
   - This is a corpus generation task — run through the generation pipeline with a "explain more deeply" prompt
   - Target: all L4+ questions should have solutions > 200 chars

6. **Deep-dive links audit**
   - Verify all `deep_dive_url` values point to working pages on mlsysbook.ai
   - For questions missing deep-dive links, add the most relevant chapter URL
   - Create a mapping: competency_area -> chapter URL for bulk assignment

7. **Add bundle analyzer**
   - Install `@next/bundle-analyzer`: `npm install -D @next/bundle-analyzer`
   - Create `next.config.mjs`:
     ```javascript
     import withBundleAnalyzer from '@next/bundle-analyzer';
     const analyzed = withBundleAnalyzer({ enabled: process.env.ANALYZE === 'true' });
     export default analyzed({ /* config */ });
     ```
   - Run `ANALYZE=true npm run build` to get baseline, target < 500 KB first-load JS

### Expert feedback agents to launch
- **Performance Engineer**: Run Lighthouse audit, measure Time to Interactive, validate corpus splitting strategy doesn't add perceptible latency
- **Content Reviewer (ML Engineer)**: Sample 50 random questions across tracks/levels, grade each on: clarity (1-5), accuracy (1-5), difficulty calibration (is L5 actually harder than L3?)
- **Technical Writer**: Review 20 worst-scoring solutions for explanation quality

### Success criteria
- [ ] First-load JS bundle < 500 KB (down from ~1.17 MB)
- [ ] `npm uninstall` removes 4 unused packages
- [ ] Corpus quality audit produces actionable report
- [ ] All L4+ questions have napkin_math field
- [ ] All deep_dive_url values resolve to real pages
- [ ] Lighthouse performance score > 90
- [ ] `npm run build` passes clean

### Dependencies
- None (can run in parallel with Sprint 2 or 3)

---

## Sprint 5: Advanced Features

**Goal:** Add differentiated features that no competitor has — company-tagged questions, personalized study plans, and the concept for an AI mock interviewer.

### Features to build

1. **Company-tagged questions**
   - Add `companies?: string[]` field to the Question interface
   - Create a mapping file: `src/data/company-tags.json`:
     ```json
     {
       "question-id-123": ["google", "meta"],
       "question-id-456": ["apple", "nvidia"]
     }
     ```
   - Populate based on: which companies ask about which competency areas (e.g., Google = serving systems, Meta = training infra, NVIDIA = GPU optimization)
   - Add company filter to drill sidebar: "Filter by company: Google, Meta, Apple, NVIDIA, Microsoft, Amazon"
   - Add company badges on question cards (small logo-style pills)
   - Create route: `src/app/company/[slug]/page.tsx` showing all questions tagged for that company
   - Landing page addition: "Practice questions from: Google, Meta, NVIDIA, Apple" with company logos

2. **Personalized study plans**
   - Route: `src/app/study-plan/page.tsx`
   - Algorithm:
     1. Look at heat map data to find weak areas (< 40% accuracy or 0 attempts)
     2. Look at target track and level (user selects or inferred from most-practiced)
     3. Generate a 2-week plan:
        - Week 1: focus on weakest 3 competency areas, 5 questions/day
        - Week 2: mixed review + gauntlet practice
     4. Each day's questions are pre-selected and stored in localStorage
   - UI: timeline view with daily cards, checkmark when completed
   - "Start Study Plan" CTA from heat map page when areas show red

3. **AI mock interviewer concept (design only, no implementation)**
   - Create a design doc at `src/app/mock/DESIGN.md` describing:
     - User flow: select question → voice/text conversation with AI → AI scores response using rubric
     - Technical approach: Gemini/Claude API call with system prompt containing the rubric + model answer
     - Scoring: AI checks rubric items automatically, gives 0-3 score + detailed feedback
     - Cost model: ~$0.01 per question at Gemini Flash pricing, viable for premium tier
   - Build a placeholder route: `src/app/mock/page.tsx` with "Coming Soon" UI
   - Include a waitlist email capture (just a form that logs to localStorage for now, Supabase later)

4. **Question difficulty recalibration**
   - Use actual user performance data to recalibrate difficulty:
     - If L3 questions have < 30% success rate across all users, they might be miscalibrated
     - Create `scripts/recalibrate.ts` that reads aggregate attempt data and flags:
       - L1-L2 questions with < 50% success (too hard for recall level)
       - L5-L6+ questions with > 90% success (too easy for design level)
   - For now, run locally. In Phase 2 with Supabase, this becomes an analytics query.

5. **Keyboard-first power user mode**
   - Add vim-style navigation for drill mode:
     - `j` / `k` to scroll question
     - `Enter` to reveal
     - `1-4` to score (already exists)
     - `n` to skip (already exists)
     - `f` to toggle filter sidebar
     - `r` to enter review mode
   - Show shortcut cheatsheet on `?` key press (modal overlay)
   - Add "Keyboard shortcuts" link in nav dropdown

### Expert feedback agents to launch
- **ML Hiring Manager**: Review company-tag assignments for accuracy (does Google actually ask about these topics?)
- **Product Strategist**: Validate study plan algorithm produces reasonable schedules, review AI mock interviewer pricing model
- **Accessibility Auditor**: Ensure keyboard navigation covers all interactive elements, test with screen reader

### Success criteria
- [ ] Company filter works in drill mode and shows relevant company badges
- [ ] At least 4 companies have tagged questions (Google, Meta, NVIDIA, Amazon)
- [ ] Study plan generates a 2-week calendar based on user weakness data
- [ ] AI mock interviewer design doc is complete and detailed enough to implement in Phase 2
- [ ] Keyboard shortcut overlay appears on `?` press
- [ ] `npm run build` passes clean

### Dependencies
- Sprint 4 (corpus splitting must be done before adding company tags to avoid further bloating the monolithic JSON)

---

## Cross-Sprint Concerns

### Technology Recommendations

**Add:**
- `html-to-image` (Sprint 1) — lightweight, no DOM dependency issues, ~15 KB
- `@next/bundle-analyzer` (Sprint 4) — dev dependency only, critical for tracking bundle size

**Do NOT add:**
- `shadcn/ui` — the existing Tailwind design system is consistent and complete. Adding shadcn would require migrating all existing components to its patterns with no user-visible benefit. The custom components are already polished.
- `chart.js` or `recharts` — the heat map is a CSS grid, not a chart library use case. Keep it simple.
- `zustand` or `jotai` — LocalStorage + React state is sufficient for Phase 1. Add state management only when Supabase adds real-time sync.

**Consider for later:**
- `framer-motion` is already installed and used well. No changes needed.
- `swr` or `tanstack-query` — add when corpus is split and fetched lazily (Sprint 4)

**Remove:**
- `mermaid` (~400 KB) — not used in any active route
- `reactflow` (~200 KB) — not used in any active route
- `react-markdown` (~50 KB) — not used in any active route
- `remark-gfm` (~30 KB) — not used in any active route
- `tailwind-merge` — check if actually used; `clsx` alone may be sufficient

### Performance Optimization Plan

| Problem | Size Impact | Fix | Sprint |
|---------|------------|-----|--------|
| corpus.json static import | ~3.7 MB | Split into index + per-track details, lazy load | 4 |
| mermaid dependency | ~400 KB | Remove (unused) | 4 |
| reactflow dependency | ~200 KB | Remove (unused) | 4 |
| react-markdown + remark-gfm | ~80 KB | Remove (unused) | 4 |
| No static export config | N/A | Add next.config.mjs with output: 'export' | 1 |
| No font subsetting | ~100 KB | Add `&subset=latin` to Google Fonts URL | 1 |
| Framer Motion full import | ~60 KB | Use `import { motion } from 'framer-motion'` (tree-shakes fine in v11) | N/A |

**Target:** First-load JS < 500 KB, Lighthouse Performance > 90.

### Mobile-First Improvements

**Current state:** Responsive grids exist (grid-cols-2 on mobile), hamburger nav works. But several issues remain:

1. **Gauntlet active phase** — the left/right split (question + answer panel) stacks vertically on mobile but the answer panel is too far below the fold. Fix: on mobile, make the answer panel a slide-up sheet (bottom drawer pattern) instead of stacking.

2. **Heat map table** — horizontal scroll on mobile works but is not discoverable. Add: scroll indicator arrow or "swipe to see all tracks" hint that disappears after first interaction.

3. **Drill sidebar** — on mobile, the sidebar stacks above the question, pushing content down. Fix: make it a collapsible drawer or slide-out panel on mobile (toggle with filter icon button).

4. **Touch targets** — scoring buttons (Skip/Wrong/Partial/Nailed It) are 40px tall, should be 48px minimum for mobile per WCAG guidelines.

5. **Text sizing** — `text-[10px]` labels are too small on mobile. Use `text-xs` (12px) as the minimum for any interactive or informational text.

**Sprint allocation:** Spread across all sprints. Each sprint should include a mobile QA pass on its new features.

### SEO Strategy for Organic Discovery

**Phase 1 (Sprint 1-2): Foundation**
- Unique title/description per page via `generateMetadata`
- og:image for social sharing previews
- Semantic HTML: proper heading hierarchy (h1 per page, h2 for sections)
- Add `robots.txt` and `sitemap.xml` to `/public`

**Phase 2 (Sprint 3-4): Content pages**
- Create static pages for each competency area: `/competency/[area]` showing question count, sample questions, study tips
- These pages are indexable and target long-tail searches: "ml systems memory optimization interview questions"
- Add structured data (JSON-LD) for FAQ schema on competency pages

**Phase 3 (Sprint 5): Company pages**
- `/company/google` → "Google ML Systems Interview Questions" — highly searchable
- Each page shows tagged questions, difficulty distribution, and common topics
- Target searches: "google staff ml engineer interview questions"

**Key SEO targets:**
- "ml systems interview questions" (primary)
- "staff ml engineer interview prep" (primary)
- "ml infrastructure interview" (secondary)
- "napkin math ml systems" (long tail, unique to us)
- "[company] ml systems interview" (per-company pages)

### README Rewrite Plan

**Current state:** README likely describes the textbook repo, not StaffML specifically. StaffML needs its own README at `interviews/staffml/README.md`.

**Structure:**

```markdown
# StaffML

> Physics-grounded ML systems interview prep. 1,800+ questions, napkin math verification, readiness tracking.

[hero screenshot — landing page with dark theme]

## Features

[3-column feature grid with screenshots]
- The Gauntlet — timed mock interviews
- Heat Map — track your readiness
- Drill Mode — focused practice with napkin math

## Quick Start

git clone ... && cd interviews/staffml && npm install && npm run dev

## Stats

![Questions](badge) ![Tracks](badge) ![Levels](badge)

## Built from the ML Systems Textbook

Link to mlsysbook.ai and harvard-edge/cs249r_book

## License
```

**Assets needed:**
- Hero screenshot (1280x800, dark theme, landing page)
- Feature screenshots (3x, 640x400 each)
- Badge images or shields.io URLs
- GIF of gauntlet flow (15s, start → question → reveal → score → results)

**Sprint allocation:** Create after Sprint 2 when daily challenge and streak are built (more features to screenshot).

---

## Sprint Sequencing

```
Week 1-2:  Sprint 1 (Polish + Social)
Week 3-4:  Sprint 2 (Engagement Loop)
Week 3-4:  Sprint 4 (Bundle + Content Quality) — runs in parallel with Sprint 2
Week 5-6:  Sprint 3 (Community + Virality)
Week 5-6:  README Rewrite
Week 7-8:  Sprint 5 (Advanced Features)
```

Sprints 2 and 4 can run in parallel because they touch different files:
- Sprint 2 adds new routes and components (daily, toast, badges)
- Sprint 4 modifies build pipeline and audits existing corpus

---

## Deferred Items (Post-Phase 1)

These require Supabase (Phase 2) and are explicitly not in this plan:

- [ ] User authentication
- [ ] Cross-device progress sync
- [ ] Real multi-user leaderboard
- [ ] AI mock interviewer (API calls)
- [ ] Community question submissions
- [ ] Premium tier / monetization
- [ ] Weekly contests
- [ ] Mobile app (React Native or PWA)
- [ ] API for third-party integrations
- [ ] Analytics dashboard (PostHog or Plausible)
