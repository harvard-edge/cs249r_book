# StaffML UX Overhaul Plan

## Context

StaffML is an interview prep app for Staff ML Systems engineers. It has 4,494 questions across 659 taxonomy concepts, 13 competency areas, 4 tracks (Cloud/Edge/Mobile/TinyML), and 6 difficulty levels (L1-L6+).

### Current Pages (8)
| Page | Purpose | Maturity |
|------|---------|----------|
| Daily | 3 daily challenge questions | Working |
| Gauntlet | Timed mock interview (5/10/15 Qs) | Working |
| Drill | Free-form question practice | Working (drill link bug just fixed) |
| Heat Map | User's readiness by track × competency | Working (needs data) |
| Plans | Study plans | Needs review |
| Roofline | Interactive roofline model calculator | Working (purpose unclear) |
| Simulator | Distributed training config simulator | Working (needs branding) |
| Vault (taxonomy) | Question bank browser | In progress |

### Core Problems
1. **Dark theme is unfriendly** — hard to read, not inviting
2. **Vault doesn't let you examine the question bank** — can't see stats, distribution, taxonomy graph
3. **Taxonomy search is hidden** — users can't explore concept connections
4. **No light/dark toggle** — dark is the only option
5. **Drill link from Vault loads wrong question** — just fixed
6. **Many topics have very few questions** (1-3) — not drillable

---

## Design Principles

1. **Light theme is default.** Dark is available via toggle. Light is easier to read, more inviting, works better on mobile.
2. **Plan before code.** Every UI change starts as a wireframe, gets user feedback, then gets built.
3. **The vault is the centerpiece.** It's where users explore what's available. It needs three views: Browse, Stats, and Map.
4. **Existing features are mature — don't break them.** Daily, Gauntlet, Drill, Heat Map all work. Theme change is global but layout stays.
5. **Mobile-first.** Every view must work on phones.

---

## Workstream 1: Light/Dark Theme Toggle

### Architecture
- CSS custom properties in `globals.css` define all colors
- Light theme = new set of `--` variables under a `.light` or `data-theme="light"` class on `<html>`
- Toggle stored in `localStorage`, respected on initial load (no flash)
- Toggle component in Nav bar (sun/moon icon)

### Light Theme Palette
```css
[data-theme="light"] {
  --background:       #ffffff;
  --surface:          #f8f9fa;
  --surface-hover:    #f0f1f3;
  --surface-elevated: #ffffff;

  --border:           #e2e4e8;
  --border-subtle:    #eef0f2;
  --border-highlight: #c8ccd0;

  --text-primary:     #1a1a2e;
  --text-secondary:   #4a4a5e;
  --text-tertiary:    #7a7a8e;
  --text-muted:       #a0a0b0;

  --accent-blue:      #2563eb;
  --accent-red:       #dc2626;
  --accent-amber:     #d97706;
  --accent-green:     #059669;
  --accent-purple:    #7c3aed;
}
```

### Implementation Steps
1. Add `data-theme` attribute system to `<html>` in layout.tsx
2. Create `ThemeProvider` component with localStorage persistence
3. Add `[data-theme="light"]` CSS block to globals.css
4. Add sun/moon toggle to Nav.tsx
5. Audit all components for hard-coded colors (e.g., `bg-white`, `text-black`)
6. Test every page in both themes
7. Set light as default

### Reference Apps
- **Linear**: excellent light/dark toggle, same layout both modes
- **Notion**: light default, clean and inviting
- **Vercel**: dark default but light is well-done
- **GitHub**: light default, dark is popular

---

## Workstream 2: Vault — Three Ways to Examine

The Vault page should have three tabs/views:

### View A: Browse & Drill (current, refined)
**"I want to practice questions on a specific topic"**

- Competency area pills as filters
- Search bar (searches topic names + descriptions)
- Area → Topic → Level → Questions progressive drill-down
- "Drill This" buttons at every level
- This is what we have now, just needs polish

### View B: Stats Dashboard
**"Show me the big picture of what's in the vault"**

#### Layout
```
┌─────────────────────────────────────────────────┐
│  4,494 questions  │  659 topics  │  13 areas    │
├─────────────────────────────────────────────────┤
│                                                   │
│  Track × Level Heatmap        │ Area Bar Chart   │
│  (4×6 grid, color=count)      │ (13 bars)        │
│                                                   │
├─────────────────────────────────────────────────┤
│                                                   │
│  Level Distribution Donut     │ Format Split     │
│  (L1-L6+ proportions)         │ (MCQ vs Open)    │
│                                                   │
├─────────────────────────────────────────────────┤
│  Topic Depth Histogram                           │
│  (X: Qs per topic, Y: how many topics)           │
├─────────────────────────────────────────────────┤
│  Top 20 Topics (by question count)               │
│  GPU Memory Hierarchy ████████████████ 159       │
│  Blue-Green Deployment ██████████████  120       │
│  ...                                             │
└─────────────────────────────────────────────────┘
```

All charts are interactive — click a heatmap cell to filter Browse view.

#### Data available
- Track × Level: 4 tracks × 6 levels = 24 cells (already in corpus)
- Area distribution: 13 areas with question counts
- Level distribution: L1=551, L2=743, L3=1152, L4=892, L5=822, L6+=334
- Format split: 2,658 open-ended, 1,836 MCQ
- Topic depth: 659 topics, distribution from 1 to 300+ Qs
- Mapped vs unmapped: 4,471/4,494 (99%)

#### Chart library
Options: recharts (already React-friendly), Chart.js, or hand-rolled SVG.
Recommendation: **recharts** — lightweight, composable, dark/light theme friendly.

### View C: Taxonomy Map
**"How do these concepts connect?"**

- Interactive force-directed graph (Sigma.js — we had this, deleted it)
- 659 nodes colored by competency area
- 746 prerequisite edges
- Click a node → see name, description, question count, prereqs, dependents
- Search highlights matching nodes
- This is the "curriculum designer" / "interviewer planning" view
- Shows the intellectual structure behind the questions

#### Why bring it back
You asked "where can I see the taxonomy graph?" — it was there, we deleted it during the Vault redesign. It should come back as a dedicated view, not the primary interface.

### Tab Navigation
```
Vault:  [ Browse ]  [ Stats ]  [ Map ]
```
- Browse is default (question-focused)
- Stats is for understanding the vault's coverage
- Map is for exploring concept relationships

---

## Workstream 3: Taxonomy Search

Users need to search the taxonomy — not just topic names, but concept connections.

### Search Behavior
1. **Topic search** (current): matches topic name/description → shows topic cards
2. **Concept search** (new): matches concept ID/name → shows concept with prereqs + dependents
3. **Question search** (new): matches question title/scenario text → shows matching questions

### Implementation
- Unified search bar at top of Vault
- Results grouped: "Topics", "Concepts", "Questions"
- Each result type has a distinct card style
- Concept results show the prerequisite chain (breadcrumb)

### Search should work on mobile
- Full-width input
- Results as a scrollable list
- Tap to drill

---

## Workstream 4: Feature Maturity Audit

### Pages to keep as-is (just apply theme)
- **Daily** — working well, just needs light/dark
- **Gauntlet** — working well
- **Heat Map** — working, needs user data to populate

### Pages to refine
- **Drill** — fix the `?q=` bug (just done), apply theme
- **Vault** — the three views above

### Pages to rebrand/rethink
- **Roofline** — "not sure what the point is" per user feedback. Keep but clarify the value prop: "Enter your model's operational intensity → see if you're compute-bound or memory-bound on real hardware"
- **Simulator** — rename to **MLSySim**. Same functionality, better identity.

### Pages to evaluate
- **Plans** — is this useful? Review content.

---

## Implementation Phases

### Phase 1: Light/Dark Toggle (foundation)
**Priority: Highest — fixes readability across all pages at once**

1. Create ThemeProvider + useTheme hook
2. Define light palette in globals.css
3. Add toggle to Nav
4. Audit all 8 pages + 7 components for theme compatibility
5. Set light as default
6. Test on mobile

**Deliverable:** Every page readable in light mode. Dark mode still works.

### Phase 2: Vault Stats Dashboard
**Priority: High — answers "what's in the vault?"**

1. Install recharts
2. Build Stats view with 5 charts:
   - Track × Level heatmap
   - Area bar chart
   - Level distribution donut
   - Topic depth histogram
   - Top 20 topics
3. Add tab navigation to Vault page (Browse | Stats | Map)
4. Make charts interactive (click to filter)

**Deliverable:** Stats tab shows the vault's shape at a glance.

### Phase 3: Taxonomy Map
**Priority: Medium — power user feature**

1. Re-add Sigma.js graph component (we have the code in git history)
2. Add as third tab in Vault
3. Click node → shows concept detail
4. Search highlights nodes

**Deliverable:** Map tab shows 659 concepts + 746 edges as interactive graph.

### Phase 4: Enhanced Search
**Priority: Medium — improves discoverability**

1. Add question-level search to Vault
2. Group results by type (topics, concepts, questions)
3. Concept results show prerequisite chain

**Deliverable:** One search bar finds anything in the vault.

### Phase 5: Polish + Branding
**Priority: Lower — after core features work**

1. Simulator → MLSySim rename
2. Roofline value prop clarification
3. Plans page review
4. Consolidate thin topics (<3 Qs) or generate more questions
5. Mobile testing pass on all pages

---

## Design Process (for each phase)

```
1. WIREFRAME  — sketch the layout (paper or Figma)
2. SCREENSHOT — build a static mockup, screenshot it
3. FEEDBACK   — show to 3-5 people, collect reactions
4. ITERATE    — apply feedback, re-screenshot
5. BUILD      — implement the winning design
6. TEST       — verify on desktop + mobile
7. SHIP       — commit + push
```

Repeat steps 2-4 until feedback saturates (no new issues for 2 rounds).

---

## Key Decisions Needed

1. **Default theme**: Light (recommended) or Dark?
2. **Chart library**: recharts, Chart.js, or hand-rolled SVG?
3. **Vault URL**: Keep `/taxonomy` or move to `/vault`?
4. **Search scope**: Topic-only or unified (topics + concepts + questions)?
5. **Thin topics**: Consolidate (<3 Qs merge into parent) or generate more?
6. **Simulator branding**: "MLSySim" or something else?
