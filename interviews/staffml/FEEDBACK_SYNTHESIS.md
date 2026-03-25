# StaffML V1 — 20-Persona Feedback Synthesis

> 20 reviewers across students, industry, product, and edge-case personas.
> Consensus distilled into prioritized action items.

---

## Universal Findings (mentioned by 10+ personas)

1. **No guided onboarding.** Every persona said "I don't know where to start." The vault dumps 5,000+ questions with no path. Need: a "Start Here" flow or quiz.

2. **Self-assessment is unreliable.** Skip/Wrong/Partial/Nailed has no calibration. Rubric checkboxes exist in Practice but not Gauntlet. Need: rubric in all modes.

3. **No sharing/export for results.** Gauntlet results can't be shared. No tweet card, no challenge link, no mentor export. This kills virality and mentorship.

4. **Textbook integration is promising but half-built.** "Learn First" links in TopicDetail are great. But answer panels have no "read more" link. Mobile hides the Textbook nav link entirely.

5. **No contribution/flagging workflow.** Report Issue button was added to Practice but not to Gauntlet results or TopicDetail. No "suggest a question" form.

---

## Top 10 Actions — Prioritized

### Ship Immediately (this session)

| # | Action | Why | Effort |
|---|--------|-----|--------|
| 1 | **Add "Read in textbook" link to answer panels** (practice + gauntlet) | Highest-intent moment — student just got it wrong | S |
| 2 | **Add track filter pills to vault landing** (cloud/edge/mobile/tinyml) | Users can't find their domain | S |
| 3 | **Add Report Issue to Gauntlet per-question review** | Currently only in Practice | S |

### Ship This Week

| # | Action | Why | Effort |
|---|--------|-----|--------|
| 4 | **Gauntlet results: "Share Score" button** | Copy text summary to clipboard. Viral moment. | M |
| 5 | **Landing page "Start Here" for new users** | 3-question inline quiz → level recommendation | M |
| 6 | **About page** with methodology, paper link, Bloom's explanation | Credibility + SEO + explains the "why" | M |

### Ship Next Week

| # | Action | Why | Effort |
|---|--------|-----|--------|
| 7 | **Mobile sidebar → collapsible drawer** | Practice sidebar pushes question below fold on mobile | M |
| 8 | **Rubric checkboxes in Gauntlet mode** | Self-assessment is meaningless without them | M |

### Design Now, Build Later

| # | Action | Why | Effort |
|---|--------|-----|--------|
| 9 | **Guided study plans from heat map gaps** | "You're weak in reliability → here's a 5-day plan" | L |
| 10 | **Interviewer curation mode** | Select questions → generate shareable link/set | L |

---

## Persona Highlights

### Students (5 personas)
- Beginners bounce because there's no "start here"
- L-levels now show Bloom names which helps
- Simulator/Roofline are impressive but need explanatory context

### Industry (5 personas)
- Hiring managers want shareable Gauntlet configs
- Mentors love Study Plans but need custom curation
- Recruiters value zero-friction (no signup) as a differentiator

### Product/Growth (5 personas)
- Star gate bypass undermines the gate — commit or remove
- Zero sharing hooks = zero viral growth
- Homepage doesn't explain WHY this is different from LeetCode

### Edge Cases (5 personas)
- Screen reader users: missing ARIA labels on filter pills, heat map cells
- Mobile-only: sidebar layout needs work, Textbook link hidden
- Contributors: need Report Issue everywhere, not just Practice
- Textbook author: answer panels should link back to chapters
