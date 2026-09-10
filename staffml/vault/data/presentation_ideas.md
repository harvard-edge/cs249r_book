# StaffML: Beyond the Paper — Presentation & Demo Ideas

## The Question: What Makes This Stand Out?

Everyone has slides. Everyone has a paper. What would make someone say
"I've never seen that before" at a conference talk or workshop?

---

## Tier 1: High-Impact, Unique

### 1. Live "Zone Diagnosis" Demo
During the talk, show a real interview question on screen. Ask the
audience: "What zone is this?" Show 4 options. Then reveal the answer
and explain WHY — the skill decomposition that makes it diagnosis vs
optimization vs fluency. The audience experiences the zone model
viscerally, not abstractly.

Could do 3-4 of these throughout the talk. Each one teaches a different
zone distinction. By the end, the audience has internalized the model.

### 2. Interactive Competency Radar
A live web visualization where you type a candidate's responses and
see their "zone radar" fill in — strong in diagnosis, weak in
specification, etc. Show it filling in real-time during the talk.

This is the "aha moment" — the zone model isn't just classification,
it's a DIAGNOSTIC TOOL for understanding engineering profiles.

Could be a feature of the staffml web app.

### 3. "Same Topic, Four Tracks" Side-by-Side
Show the SAME concept (e.g., "memory pressure") asked across all 4
tracks. The physics changes completely:
- Cloud: "80 GB HBM3, model needs 140 GB, how do you parallelize?"
- Edge: "32 GB LPDDR5 shared with OS, thermal throttle at 60W"
- Mobile: "8 GB unified, ANR timeout in 5 seconds, app in background"
- TinyML: "256 KB SRAM, model must fit with activations, no swap"

Same concept, four different universes. This IS the thesis —
constraints drive architecture.

### 4. The "Napkin Math Challenge"
Put a question on screen. Give the audience 60 seconds. Then show
the solution step by step. Compare their answers. This is electrifying
in a room of engineers — they ALL want to try it.

Works perfectly for a workshop or tutorial session.

### 5. Applicability Matrix as a Physical Poster
Print the 79×4 matrix at poster size. Green/red cells. People can
walk up and challenge exclusions: "Why can't you ask about kernel
fusion on TinyML?" And you can point to the physics reason.
Interactive, tactile, memorable.

---

## Tier 2: Strong, Complementary

### 6. "Before/After" Corpus Evolution Animation
Animate the corpus growing: start with empty matrix, fill cells,
watch the distribution skew, then rebalance. Show the methodology
as a PROCESS, not a result. Could be a 30-second GIF or video.

### 7. The Backward Design Chain as a Poster/Infographic
One beautiful visual that shows:
competency → skills → zones → topics → physics filter → capacity → corpus
Each step with a number and an example. Frameable, shareable.

### 8. "Misconception Gallery"
The `common_mistake` field is gold. Curate the 20 best misconceptions
and present them as a "gallery of things smart engineers get wrong."
Each one is a mini-lesson. Could be a Twitter/X thread, a blog post,
or a section of the talk.

Examples:
- "Blaming GPU underclocking when decode is memory-bound at 1 FLOP/Byte"
- "Doubling batch size and expecting 2x throughput (ignoring memory wall)"
- "Using FP32 KV-cache when INT8 would save 42 GB at 128K context"

### 9. Comparison Table: StaffML vs Everything Else
| | LeetCode | Blind75 | System Design | StaffML |
|---|---|---|---|---|
| Napkin math | ✗ | ✗ | Sometimes | Every Q |
| Hardware specs | ✗ | ✗ | ✗ | Real specs |
| Zone model | ✗ | ✗ | ✗ | 11 zones |
| Physics grounded | ✗ | ✗ | ✗ | ✓ |
| ML-specific | ✗ | ✗ | Partial | ✓ |

### 10. "Design Your Interview Loop" Workshop
Give attendees the zone model and topic list. Have them design a
45-minute interview loop for a specific role (e.g., "Staff Inference
Engineer"). Then compare designs. Shows the practical value immediately.

---

## Tier 3: Creative Extensions

### 11. StaffML Trading Cards
Physical cards with one question per card. Front: scenario + specs.
Back: solution + napkin math. Zone color-coded. Collectible, fun,
memorable. Hand them out at the conference booth.

### 12. "Physics of AI Engineering" Mini-Lecture Series
5 short videos (3-5 min each), one per fundamental law:
1. The Roofline Bound
2. The Memory Capacity Wall
3. The Communication Tax
4. The Power Envelope
5. The Cost Multiplier
Each video uses a StaffML question as the narrative anchor.

### 13. Automated "Interview Simulator"
Web tool: select your track and level, get a timed question, type
your napkin math, get scored. Like LeetCode but for systems reasoning.
This is the staffml web app's ultimate form.

---

## Expert Feedback (Chip Huyen persona)

### Top 3 to prioritize:
1. **Same-topic-four-tracks** — "killer demo." Put in first 3 minutes.
   Shows physics changes but concept is invariant. THE thesis.
2. **Napkin math challenge merged with misconceptions** — expose a
   misconception, don't just quiz. The "aha" is WHY people are wrong.
3. **Trading cards** (physical) or **zone radar** (digital) as takeaway.

### What's missing:
- **"Grade your interview process" self-assessment** — 5-min checklist
  for hiring managers. Converts interest to adoption.
- **Before/after case study** — even synthetic. Shows what StaffML catches.
- **Explicit comparison** to Huyen's book, system design interviews, etc.

### For NeurIPS D&B specifically:
- Lead with INVARIANT STRUCTURE argument (79 concepts stable across 4 tracks)
- Show LinkML schema briefly (D&B loves formal structure)
- Frame as "competency model" not "recruiting tool"
- Show 19 invariant checks as validation

### Viral format:
- Twitter/X thread: "Same concept, 4 deployment targets. Thread."
  Engineers argue in replies = algorithmic engagement.
- Web tool: enter job title → get 5 sample questions. Shareable.

### Adoption mechanism:
1. Zone-to-level mapping table (lookup, not theory)
2. Time-calibrated interview plans (45-min screen, 4-hr onsite)
3. Signal quality evidence (pilot study, even preliminary)

## What to Build First

For a NeurIPS D&B presentation specifically:
1. Live napkin math challenge (audience engagement, 2 minutes)
2. Same-topic-four-tracks slide (thesis demonstration, 1 slide)
3. Applicability matrix figure (already built as SVG)
4. Misconception gallery (3-4 best examples, 1-2 slides)
5. Zone radar mockup (future work slide, shows product vision)

For a workshop/tutorial:
1. Design-your-interview-loop exercise (hands-on, 15 min)
2. Napkin math challenge × 5 (progressive difficulty)
3. Zone classification exercise (audience votes)
