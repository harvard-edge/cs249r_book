# StaffML Improvement Plan: Pelin Balcı Feedback

**Branch:** `feat/staffml-pelin-feedback`
**Worktree:** `MLSysBook-staffml-pelin-feedback`
**Date:** 2026-05-27

## Background

Pelin Balcı is an ML Engineer with an industrial engineering/optimization background (no formal CS). She represents the self-taught ML practitioner audience transitioning to systems thinking. Her feedback covers UX gaps that affect interview prep workflows specifically.

Peter's feedback (addressed on `feat/staffml-feedback`) covers bugs (visual questions, LLM checking) and content gaps (RISC-V, DSP+NN). Pelin's feedback is orthogonal: it is about navigation, progress, and the structured learning experience.

---

## Current State

| Component | Status | Key files |
|:---|:---|:---|
| Welcome/onboarding | Exists: track selector, first-run explainer | `src/app/welcome/page.tsx`, `src/components/FirstRunExplainer.tsx` |
| Progress tracking | Exists: LocalStorage (attempts, gauntlets, SM-2 spaced repetition) | `src/lib/progress.ts` |
| Dashboard | Exists: heatmap, streaks, totals, recent activity | `src/app/dashboard/page.tsx` |
| Progress page | Exists: analytics, history, competency-area breakdown | `src/app/progress/page.tsx` |
| Explore (Vault) | Exists: radial/sunburst hierarchy, filters by level/topic/area/zone | `src/app/explore/page.tsx` |
| Practice | Exists: single-question drill, scenario, rubric, SM-2, star-gate | `src/app/practice/page.tsx` |
| Gauntlet | Exists: rapid-fire challenge mode | `src/app/gauntlet/` |
| Learning chains | Exist: 843 chains in vault, topic-scoped, Bloom-level progression | `vault/chains.json`, `src/data/tracks/*.json` |
| Question corpus | 9,525 questions: L1(543), L2(1053), L3(2363), L4(2591), L5(2157), L6+(818) | `src/data/corpus-summary.json` |
| Chapter cross-refs | Not present: questions have `topic`, `competency_area`, `zone` but no book chapter mapping | (new feature) |

---

## Pelin's Five Feedback Items

### F1. Vault and Practice sections confusing on first use

**Problem:** Pelin was unsure where to start. The Welcome page exists but doesn't clearly explain the relationship between Vault (browse), Practice (drill), Explore (radial viz), and Gauntlet (rapid-fire).

**Proposed solution:** Add a "Recommended Path" component to the Welcome page that shows a 3-step flow:

1. **Pick your track** (Cloud / Edge / Mobile / TinyML)
2. **Start at your level** (quick 5-question placement quiz or self-assessment)
3. **Follow the path** (sequential mode within your track+level)

Also add a persistent "Guide" tooltip or sidebar on first visit to Practice/Vault that explains what each section does (one sentence each). Dismiss after first interaction.

**Scope:** Small-medium. Touch `welcome/page.tsx`, possibly add a `GuidedOnboarding.tsx` component, update nav to show breadcrumbs or step indicators for new users.

**Dependencies:** None.

---

### F2. Link questions to MLSysBook chapters

**Problem:** After solving a question, there is no way to loop back to the relevant book chapter to fill knowledge gaps.

**Proposed solution:** Build a `chapter-map.json` mapping from `competency_area` and `topic` to MLSysBook chapter slugs and section anchors. Display a "Learn more" link after each question that points to the relevant chapter on the MLSysBook website.

**Data model addition:**
```json
{
  "competency_area_to_chapters": {
    "deployment": {
      "vol1": ["model_serving", "ml_ops"],
      "vol2": ["inference", "edge_intelligence"]
    },
    "memory_hierarchy": {
      "vol1": ["hw_acceleration"],
      "vol2": ["compute_infrastructure"]
    }
  },
  "topic_to_section": {
    "model-serving-infrastructure": "https://mlsysbook.ai/contents/vol1/model_serving/",
    "gpu-memory-hierarchy": "https://mlsysbook.ai/contents/vol1/hw_acceleration/#sec-hw-memory-hierarchy"
  }
}
```

**Scope:** Medium. Create the mapping file (87 topics to audit), add a `ChapterLink.tsx` component, integrate into the practice post-answer view.

**Dependencies:** Requires auditing the 87 topics against the book's chapter structure to build the mapping.

---

### F3. Sequential L-level progression

**Problem:** Questions may appear randomly. Pelin wants to start at L1 and work through systematically, with the ability to resume where she stopped.

**Current state:** Learning chains exist (843 chains, topic-scoped, Bloom-level progression L2-L5) but there is no UI mode that presents them sequentially with a bookmark.

**Proposed solution:** Add a "Structured Path" mode alongside the current random/daily modes:

1. **Path selection:** User picks a track + competency area (or "all areas").
2. **Level progression:** Within the path, questions are ordered L1 -> L2 -> L3 -> L4 -> L5 -> L6+.
3. **Within-level ordering:** Use existing chain ordering (Bloom taxonomy: remember -> understand -> apply -> analyze -> evaluate -> create).
4. **Bookmark state:** Save `{ trackId, areaId, currentQuestionIndex, completedIds[] }` to LocalStorage under a new key `staffml_paths`.
5. **Resume:** On return, show "Continue where you left off" with the next question in sequence.

**UI additions:**
- New "Paths" route (`/paths`) showing available structured paths with progress bars.
- "Continue Path" button on the welcome/landing page.
- Path progress indicator in the practice view header when in path mode.

**Scope:** Medium-high. New route, new state management in `progress.ts`, path-ordering logic, practice page integration.

**Dependencies:** F1 (onboarding should point to this mode).

---

### F4. Progress tracking (NeetCode-style)

**Problem:** Progress section exists but doesn't show per-topic completion percentage or clear L-level position. Pelin cites NeetCode as a motivating example.

**Current state:** The progress page shows a competency-area x track heatmap and total counts. It does not show:
- Per-topic completion bars (X of Y questions solved)
- Current L-level position per topic
- Visual progress percentage under each category

**Proposed solution:** Redesign the progress page to show:

1. **Topic-level progress bars** grouped by competency area:
   ```
   Memory Hierarchy                          68%  ████████████░░░░░  (34/50)
     ├── Cache optimization          L3  ✅  100%  ████████████████  (12/12)
     ├── HBM bandwidth               L2  🟡   45%  ███████░░░░░░░░░  (9/20)
     └── Memory-bound bottlenecks    L1  🟡   72%  ███████████░░░░░  (13/18)
   ```

2. **L-level indicator** per topic showing the highest level the user has answered correctly.

3. **Overall dashboard summary** at the top:
   - Questions attempted: X / 9,525
   - Topics touched: X / 87
   - Current strongest area, weakest area
   - Streak information

4. **Color coding:** Green (>80%), Yellow (40-80%), Red (<40%), Gray (not started).

**Scope:** Medium. Primarily UI work in the progress page. The data (attempts per question, per topic, per level) already exists in LocalStorage.

**Dependencies:** None (can be built independently).

---

### F5. Interview prep mode (combines F3 + F4)

**Problem:** During interview prep, people create a goal and timeline. Being able to track which question they stopped at and see completion progress is critical.

**Proposed solution:** A dedicated "Interview Prep" mode that wraps F3 (structured paths) and F4 (progress tracking) into a cohesive experience:

1. **Goal setting:** User picks target level (e.g., "Staff Engineer: L4-L5"), target track, and timeline (e.g., "2 weeks").
2. **Daily plan:** System calculates questions/day based on goal and shows a daily quota.
3. **Session tracking:** Each practice session in prep mode shows:
   - Today's quota: 5/12 questions
   - Current path position: L3, Topic 14/87
   - Time spent today: 45 min
4. **Weekly review:** Summary of the week's progress, areas of strength/weakness.

**Scope:** High. This is the capstone feature that ties F3 and F4 together. Build F3 and F4 first, then add the goal/timeline/daily-plan layer.

**Dependencies:** F3, F4.

---

## Implementation Order

The items have natural dependencies. Recommended build order:

```
Phase 1 (independent, can parallelize)
├── F4: Progress tracking redesign (NeetCode-style bars)
└── F2: Chapter-to-question mapping

Phase 2 (depends on F4)
├── F3: Sequential L-level paths + bookmark/resume
└── F1: Onboarding improvements (depends on F3 existing to point to)

Phase 3 (depends on F3 + F4)
└── F5: Interview prep mode (goal/timeline/daily wrapper)
```

**Estimated scope per phase:**

| Phase | Items | Est. files touched | Complexity |
|:---|:---|:---|:---|
| Phase 1 | F4, F2 | ~6 files + 1 new data file | Medium |
| Phase 2 | F3, F1 | ~8 files + 1 new route | Medium-high |
| Phase 3 | F5 | ~5 files + 1 new route | High |

---

## Design Constraints

- **All state is LocalStorage.** No backend, no auth. Progress must be exportable/importable (already exists in `progress.ts`).
- **Corpus is pre-bundled + lazy-loaded.** Summary data (9,525 questions) is in `corpus-summary.json`; full details hydrate from Cloudflare Worker. New features should work with summary data only (no heavy-field dependency for progress/path tracking).
- **Existing features must not regress.** Random mode, daily question, gauntlet, explore, and spaced repetition continue to work unchanged.
- **Mobile-first.** Many users will practice on phones. Progress bars and path views must be touch-friendly and readable at 375px width.

---

## Open Questions (for VJ)

1. **Chapter mapping granularity:** Should `topic_to_section` link to chapter-level pages or specific `#sec-` section anchors within chapters? Section-level is more useful but requires mapping 87 topics to section IDs (substantial one-time effort).

2. **Interview prep timeline:** How sophisticated should the daily plan be? Options:
   - A: Simple (total questions / days = daily quota)
   - B: Adaptive (weight harder levels more, adjust based on performance)
   - C: Calendar-integrated (mark prep days, skip weekends)

3. **Placement quiz:** Should F1's onboarding include a quick placement assessment to suggest starting level, or just let users self-select?

4. **Priority:** Want to start with one specific phase, or lay out all the data structures first and build UI incrementally?
