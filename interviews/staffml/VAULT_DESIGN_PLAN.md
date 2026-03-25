# StaffML Vault — Design Plan

## The Three Views

The Vault isn't one page. It's three distinct modes that share the same data:

### View 1: Browse & Drill (Primary)
**User**: "I want to practice questions on a topic"
- Entry: search or browse by competency area
- Flow: area → topic → level → questions → drill
- This is what we've been building

### View 2: Dashboard & Stats (Analytics)
**User**: "Show me the big picture of what's in here"
- Vault stats at a glance: total Qs, distribution by track/level/area
- Heatmap: competency × level (how many Qs in each cell)
- Area breakdown: bar charts showing question depth per area
- Top topics: ranked list by question count
- Gaps: areas with few questions (opportunity to contribute)
- This is a read-only analytics view, NOT the Heat Map page (which tracks user progress)

### View 3: Taxonomy Map (Knowledge Graph)
**User**: "How do these concepts connect?"
- Interactive graph visualization (Sigma.js — we already built this)
- Prerequisites and dependents visible
- Click to navigate between connected concepts
- This is the "see the structure" view for curriculum designers / interviewers
- NOT the primary user flow — it's a power-user / instructor tool

## Navigation

```
/vault            → Browse & Drill (default)
/vault/stats      → Dashboard & Stats
/vault/map        → Taxonomy Graph
```

Or use tabs within the /taxonomy page: **Browse** | **Stats** | **Map**

## Design Process (Iterative, Screenshot-Based)

### Round 1: Wireframes (no code)
1. Create wireframes for all three views in Figma or hand-drawn
2. Show to 3-5 people (mix of interview candidates + interviewers)
3. Ask: "What would you click first?" and "What's missing?"
4. Collect feedback

### Round 2: High-Fidelity Mockups
1. Apply feedback from Round 1
2. Create dark-mode mockups with real data
3. Show individual screenshots to 5-8 people
4. A/B test two layout options for Browse view:
   - Option A: Current Tier 1 area overview → expand
   - Option B: Flat topic grid with sidebar filters (like an e-commerce catalog)
5. Test the detail panel: side panel vs. full page vs. modal

### Round 3: Prototype
1. Build the winning design from Round 2
2. Deploy to a staging URL
3. Have 5 people actually USE it (not just look at screenshots)
4. Track: time to find a topic, time to start drilling, abandonment points
5. Fix the top 3 friction points

### Round 4: Polish
1. Typography pass — every text element reviewed for readability
2. Color pass — check all area colors for distinctness + accessibility
3. Animation pass — are transitions helpful or distracting?
4. Mobile pass — test on actual iPhone + Android

### Saturation Criteria
Stop iterating when:
- 4 out of 5 testers can find and start drilling a topic in <15 seconds
- No tester mentions readability issues unprompted
- Mobile and desktop get similar task-completion scores

## Dashboard Stats View — What to Show

### Top Row (Hero Stats)
- Total questions: 4,494
- Topics covered: 735
- Competency areas: 13
- Difficulty range: L1 → L6+

### Distribution Charts
1. **Track × Level heatmap**: 4 tracks × 6 levels = 24 cells
   - Cell color = question count (darker = more)
   - Shows where the vault is deep vs. thin

2. **Area bar chart**: 13 horizontal bars, sorted by Q count
   - Each bar segmented by level (stacked)
   - Shows which areas are well-covered

3. **Level distribution**: pie/donut chart
   - L1-L6+ distribution across all questions
   - Shows if the vault is balanced

4. **Topic depth histogram**:
   - X axis: questions per topic (1, 2-5, 6-10, 11-50, 50+)
   - Y axis: how many topics
   - Shows the long tail (most topics have <5 Qs)

### Taxonomy Keywords
- Tag cloud or treemap of most common topic names
- Sized by question count
- Colored by competency area
- Clickable → goes to Browse view filtered to that topic

## Mobile Responsive Strategy

### Browse View
- **Phone (<640px)**: Single column, area cards stack vertically, detail opens as bottom sheet
- **Tablet (640-1024px)**: 2-column topic grid, detail as side panel (narrower)
- **Desktop (1024px+)**: 3-column grid, full 420px side panel

### Stats View
- **Phone**: Stack all charts vertically, heatmap scrolls horizontally
- **Tablet**: 2-column chart grid
- **Desktop**: Full dashboard layout

### Map View
- **Phone**: Full-screen graph, detail as bottom sheet overlay
- **Tablet/Desktop**: Graph + side panel

## Key Decisions Needed

1. **URL structure**: `/vault` vs keeping `/taxonomy`?
2. **Should Stats be a separate page or a tab?**
3. **Should the taxonomy graph come back?** (We deleted TaxonomyGraph.tsx)
4. **Do we need a "recently drilled" or "bookmarked" section?**
5. **Should interviewers see different things than candidates?**
