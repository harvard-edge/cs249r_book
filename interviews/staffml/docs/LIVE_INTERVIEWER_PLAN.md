# Live Interviewer: Detailed Plan

## What This Is

An AI-driven mock interview mode for StaffML where a conversational
interviewer draws from the 10,711-question vault to conduct a realistic
staff-level ML systems interview. The interviewer adapts dynamically:
zooming into weak areas, accelerating through strengths, requesting
napkin math, and asking for system diagrams.

This is a separate mode from Practice (flashcard self-assessment) and
Gauntlet (timed quiz). It lives at `/interview`.

---

## Why It Matters

Peter's feedback crystallized the gap: his team's real interviews are
conversational and unstructured. They flow based on the candidate. They
zoom into blocks, zoom out to system level, and do napkin math together.
No existing mode replicates that.

The question vault has everything needed: 10,711 questions with
scenarios, solutions, napkin math, and rich metadata (zone, bloom level,
competency area, topic). What's missing is an orchestrator that weaves
these into a coherent interview arc.

---

## Existing Assets to Leverage

The platform already has substantial infrastructure the live
interviewer should build on, not rebuild.

### Chains (843 chains, 2,853 questions)

Pre-built difficulty progressions that are the interviewer's primary
navigation structure. A chain is a curated sequence of questions on
the same topic that escalates through levels and zones:

```
Chain: cloud-chain-auto-002-04 (collective-communication)
  [0] L3 fluency     → MoE AllToAll Communication Time
  [1] L4 diagnosis   → Diagnosing MoE AllToAll Network Bottlenecks
  [2] L5 evaluation  → Trade-offs of DCQCN Parameters in RoCEv2 AI Clusters
  [3] L6+ mastery    → MoE Interconnect Bottleneck on TPU Pods
```

How the interviewer uses chains:

- **Entry point selection**: Start the candidate at a chain position
  matching their target level (e.g., position [1] for L4 target)
- **Zoom in/out**: Move backward in the chain to simplify, forward
  to escalate. The chain already has the right questions.
- **Natural transitions**: When a chain is exhausted, the coverage
  map identifies the next uncovered area and picks a chain there
- **Primary vs secondary tiers**: 1,480 primary memberships (clean
  Bloom progressions, shown by default) and 1,374 secondary
  (alternate paths, used when the primary chain doesn't fit)

The most common chain shapes:

| Progression | Count | Interview use |
|---|---|---|
| L3 → L4 → L5 | 127 | Default for staff-level (start L3 warmup) |
| L3 → L4 → L5 → L6+ | 64 | Full escalation for strong candidates |
| L4 → L5 → L6+ | 33 | Skip warmup for senior candidates |
| L2 → L3 → L4 → L5 | 33 | Start easier for uncertain areas |

Zone progressions within chains:

| Zone flow | Count | What it tests |
|---|---|---|
| fluency → diagnosis → evaluation | 29 | Know it → find the bug → judge the trade-off |
| recall → fluency → evaluation | 28 | Define it → apply it → evaluate it |
| fluency → design → mastery | 28 | Understand it → build it → own it |
| fluency → diagnosis → specification | 23 | Apply it → debug it → spec the fix |

### Gauntlet Mode (structural template)

The gauntlet page (`src/app/gauntlet/page.tsx`) already implements the
multi-phase interview pattern:

- Phase machine: `setup | active | review | results`
- Track/level/duration selectors
- Warm-up question selection (easier question first)
- Round-robin across zones for breadth
- Realism modes: strict / standard / open (controls tool access)
- Timer, progress tracking, results persistence

The live interviewer reuses this phase structure but replaces the
quiz loop with a chat interface.

### Worker Infrastructure (LLM routing + personas)

The Cloudflare Worker (`worker/src/index.ts`) already has:

- **6 LLM provider adapters**: Groq (Llama 3.3 70B), OpenAI
  (GPT-4o mini), Anthropic, Gemini, OpenRouter, CF Workers AI
- **Priority-based failover**: if Groq is down, falls through to
  OpenAI, then Anthropic, etc.
- **Two persona system prompts**: Socratic (clarification only,
  never reveals answer) and Tutor (post-reveal explanation)
- **Server-side prompt enforcement**: system prompt injected
  server-side so the client cannot override it
- **Rate limiting**: per-IP bucketing with configurable windows

The live interviewer adds a third persona: Conductor. It reuses the
same adapter registry and failover logic.

### Spaced Repetition + Progress System

`src/lib/progress.ts` tracks:

- Per-question attempt history (score, timestamp)
- SM-2 spaced repetition cards (due dates, intervals)
- Gauntlet results (score, questions, timing)
- Activity log (streaks, session history)

The live interviewer can:

- Exclude already-mastered questions from the pool (SR interval
  > 30 days = candidate knows this)
- Feed interview results back into the SR system (weak areas get
  shorter intervals, come up sooner in practice mode)
- Save interview sessions alongside gauntlet results

### Existing Components (direct reuse)

| Component | Live interviewer use |
|---|---|
| `NapkinCalc` | Collapsible calculator during estimation questions |
| `HardwareRef` | Quick spec lookup (H100 TFLOPS, A100 memory, etc.) |
| `GlossaryText` | Acronym tooltips in interviewer messages |
| `MarkdownText` | Render interviewer messages with bold/code/numbers |
| `MetaTooltip` | Hover info on competency badges in coverage bar |
| `LevelBadge` | Show question difficulty in transcript |
| `ChainBadge` | Show chain context when the AI follows a chain |
| `QuestionVisual` | Display SVG diagrams when relevant to the scenario |
| `QuestionFeedback` | Post-interview per-question feedback UI |
| `Toast` | Notifications (session saved, time warning, etc.) |

### Analytics Events

`src/lib/analytics.ts` already defines event types for gauntlet
(started, completed, abandoned). The live interviewer adds parallel
events: `interview_started`, `interview_completed`,
`interview_abandoned`.

### Napkin Math Checking

`checkNapkinMath()` in `corpus.ts` does algorithmic tolerance
checking with track-specific thresholds (cloud: 25%, edge: 15%,
mobile: 10%). Every vault question has napkin math with assumptions,
calculations, and conclusions. The live interviewer uses these to
silently evaluate the candidate's estimates and calibrate follow-ups.

### Glossary (831 terms)

The glossary we just built (`src/data/glossary.json`, 831 terms with
definitions and acronym expansions) serves double duty:

- Inline tooltips in the interviewer's messages (via GlossaryText)
- The AI interviewer's acronym awareness: when it detects the
  candidate is unfamiliar with a term, it can pull the definition
  from the glossary data passed in the system prompt context

---

## Goals

### G1. Feel like a real interview, not a quiz

A real staff-level interview is a conversation. The interviewer presents
a scenario, listens, follows up, probes, and transitions. The candidate
should forget they're talking to software.

Concrete requirements:

- The AI paraphrases scenarios naturally (never reads vault text
  verbatim)
- Follow-up questions reference what the candidate just said
- The AI explains acronyms inline when the candidate seems unfamiliar
  (Peter: "we don't expect them to know all acronyms")
- The AI does napkin math *with* the candidate, not just checks their
  number
- Transitions between topics feel motivated ("That memory budget is
  interesting. Let's see what happens when we scale this to a
  multi-node setup...")

### G2. Cover breadth and depth in a single session

Staff interviews assess both. The interviewer should:

- Touch 4-6 competency areas in a 45-minute session
- Go deep on 2-3 (3+ follow-ups in one area)
- Know when to move on (candidate clearly strong or clearly stuck)
- Track coverage and bias toward uncovered areas

Coverage targets per session length:

| Duration | Areas touched | Deep dives | Total scenarios |
|----------|---------------|------------|-----------------|
| 30 min   | 3-4           | 1-2        | 4-6             |
| 45 min   | 4-6           | 2-3        | 6-9             |
| 60 min   | 5-7           | 3-4        | 8-12            |

### G3. Calibrate difficulty dynamically

The interviewer starts at the target level (e.g., L4) and adjusts:

- Strong answer at L4: follow up at L5, then L6+
- Weak answer at L4: zoom in, simplify, try L3 angle
- No answer: give a hint, ask a more focused sub-question

This matches Peter's description: "We focus on system-level impacts,
chain of thought, but of course also napkin math."

### G4. Exercise the full question taxonomy

The vault has 11 zones (diagnosis, fluency, evaluation, recall, analyze,
design, mastery, specification, optimization, implement, realization)
and 6 Bloom levels (remember through create). A good interview mixes
these:

- Open with a scenario that requires diagnosis or design
- Follow up with analysis ("What's the bottleneck?")
- Request napkin math ("Can you estimate the memory footprint?")
- Push to evaluation ("What breaks first at 10x scale?")
- Ask for synthesis ("How would you design a monitoring system for
  this?")

### G5. Produce actionable feedback

The end-of-interview report should tell the candidate:

- Which competency areas are strong vs weak
- Where their napkin math was off and by how much
- Which concepts they should study (linked to specific practice
  questions)
- How their chain-of-thought was (clear reasoning vs jumping to
  answers)

---

## What the Interviewer Needs to Do Well

### 1. Question selection (the conductor function)

The AI does not see all 10,711 questions. The client preselects a pool
of questions and chains matching the session config. The AI navigates
chains for depth and switches chains for breadth.

Pool selection strategy (client-first, chain-aware):

**Step 1: Select chains (primary navigation structure)**

1. Filter chains by track
2. For each competency area, find chains whose entry point matches
   the target level range (+/- 1 level)
3. Pick 1-2 chains per competency area (primary tier preferred)
4. For a 45-min session targeting 5 areas: ~8-10 chains selected

**Step 2: Hydrate chain members**

1. For each selected chain, hydrate the full question details (scenario,
   solution, napkin_math) via the vault worker
2. Send chain structure to the AI: the position order, levels, zones,
   and which question is the "entry point" for this candidate's level

**Step 3: Fill gaps with standalone questions**

1. For competency areas with no good chain match, add 2-3 standalone
   questions filtered by track + level + zone diversity
2. Prefer higher Bloom levels (analyze, evaluate, create)
3. Prefer questions with visuals (144 available)

**Step 4: Exclude mastered questions**

1. Check SR cards from progress.ts
2. Exclude questions with interval > 30 days (well-known)
3. Prefer questions the candidate has attempted but scored poorly on
   (targeted remediation)

**Step 5: Refresh during session**

1. When < 3 unused chains remain, fetch replacement chains biased
   toward uncovered areas
2. Track coverage map: which areas have been touched, at what depth

**How the AI uses chains during the interview:**

The AI receives chains as ordered sequences, not individual questions.
When it presents a scenario from chain position [1], it knows that:
- Position [0] is the simpler version (zoom in if candidate struggles)
- Position [2] is the harder follow-up (escalate if candidate is strong)
- The chain's zone progression tells it what cognitive move comes next
  (e.g., fluency → diagnosis → evaluation)

This means the AI's follow-up questions are real vault questions with
canonical solutions and napkin math, not improvised prompts. Every
turn is grounded in curated content.

### 2. Conversational flow (the dialogue function)

Each AI turn has an intent. The intents form a grammar:

```
Session = Greeting, Body, Closing
Body    = Topic+
Topic   = Present, Response+, (Transition | End)
Response = FollowUp | Probe | NapkinMath | Diagram | ZoomIn | ZoomOut
```

Intent definitions:

- **greeting**: Ask about background, set the stage
- **present_scenario**: Introduce a new scenario (paraphrased from vault
  question)
- **follow_up**: Build on candidate's answer ("And if we doubled the
  batch size?")
- **probe_weakness**: Target something the candidate was vague about
  ("You mentioned memory-bound. Can you quantify that?")
- **napkin_math**: Request an estimation ("What's the memory footprint
  of the KV cache at 128K context?")
- **diagram_request**: Ask for a system description ("Walk me through
  the data flow from ingestion to serving")
- **zoom_in**: Focus on a component ("Let's focus on just the prefill
  stage")
- **zoom_out**: Expand scope ("Now how does this change across a
  1024-GPU cluster?")
- **transition**: Move to a new competency area with motivation
- **hint**: When candidate is stuck, offer a specific constraint or
  starting point
- **closing**: Summarize, invite questions, end

### 3. Performance evaluation (the assessor function)

The AI silently evaluates against the vault's canonical solution and
napkin math. It never reveals the answer during the interview.

What it tracks per question:

- **approach_quality**: Did they identify the right bottleneck/
  constraint/trade-off?
- **napkin_accuracy**: Was their estimate within an order of magnitude?
  Within 2x?
- **depth**: Did they reason about *why*, not just *what*?
- **breadth**: Did they consider failure modes, operational concerns,
  cost?
- **communication**: Was their reasoning clear and structured?

Rating scale per question: strong / adequate / weak / not assessed

### 4. Acronym awareness

When a candidate says "I'm not sure what that means" or gives a
response that suggests they don't know a term, the interviewer:

- Briefly explains the acronym (sourced from glossary.json)
- Moves on without penalizing
- Notes it as a knowledge gap, not a failure

This directly addresses Peter's feedback about candidates from
different backgrounds.

### 5. Diagram prompts

When the AI asks "Can you describe the architecture?", the candidate
types a text description. The AI evaluates:

- Did they identify the key components?
- Is the data flow correct?
- Did they account for failure/redundancy?

Future: integrate a simple block-diagram canvas (Phase 3+).

---

## Data Model

### InterviewConfig

```typescript
interface InterviewConfig {
  track: string;
  targetLevel: string;        // L4, L5, L6+
  durationMinutes: number;    // 30, 45, 60
  focusAreas?: string[];      // optional competency area filter
  roleDescription?: string;   // optional: "Staff ML Engineer, Edge AI"
}
```

### InterviewSession

```typescript
interface InterviewSession {
  id: string;
  config: InterviewConfig;
  phase: "setup" | "active" | "feedback";
  startedAt: number | null;
  endedAt: number | null;
  transcript: TranscriptEntry[];
  questionsUsed: QuestionUsage[];
  coverageMap: Record<string, AreaCoverage>;
  feedback: InterviewFeedback | null;
}

interface TranscriptEntry {
  id: string;
  role: "interviewer" | "candidate";
  content: string;
  timestamp: number;
  questionRef?: string;       // vault question ID
  intent?: InterviewerIntent;
}

interface QuestionUsage {
  questionId: string;
  area: string;
  topic: string;
  zone: string;
  level: string;
  performance: "strong" | "adequate" | "weak" | "not_assessed";
  napkinAccuracy?: "exact" | "close" | "off" | "way_off";
  notes: string;              // AI's brief assessment
}

interface AreaCoverage {
  area: string;
  questionsAsked: number;
  deepestLevel: string;
  overallRating: "strong" | "adequate" | "weak" | "not_covered";
}
```

### InterviewFeedback

```typescript
interface InterviewFeedback {
  overallAssessment: string;  // 2-3 sentence summary
  rating: "strong" | "hire" | "borderline" | "below";
  areas: AreaScore[];
  strengths: string[];
  improvements: string[];
  recommendations: Recommendation[];
  sessionStats: {
    durationMinutes: number;
    questionsPresented: number;
    areasExplored: number;
    napkinMathAttempts: number;
  };
}

interface AreaScore {
  area: string;
  rating: "strong" | "adequate" | "weak" | "not_covered";
  evidence: string;           // specific example from the session
}

interface Recommendation {
  area: string;
  suggestion: string;
  practiceQuestionIds: string[];
}
```

---

## Architecture

### Client-worker interaction

```
Client (Next.js)                    Worker (Cloudflare)
  |                                   |
  |  1. Hydrate question pool         |
  |     (30 questions via vault API)  |
  |                                   |
  |  2. POST /interview               |
  |     { action: "start",            |
  |       config, questionPool }      |
  |                                   |
  |  <-- { message, intent }          |
  |                                   |
  |  3. POST /interview               |
  |     { action: "respond",          |
  |       candidateMessage,           |
  |       transcript,                 |
  |       questionPool,               |
  |       coverageMap }               |
  |                                   |
  |  <-- { message, intent,           |
  |        questionRef?,              |
  |        performanceNote? }         |
  |                                   |
  |  ... repeat ...                   |
  |                                   |
  |  4. POST /interview               |
  |     { action: "end",              |
  |       transcript,                 |
  |       questionsUsed }             |
  |                                   |
  |  <-- { feedback }                 |
```

The worker is stateless. Every request carries the full context.

### Question pool (client-side selection)

The client manages the pool. Algorithm:

```
1. Filter corpus by track + level range
2. Group by competency_area
3. From each area, pick 2-3 questions:
   - Prefer zones: design, diagnosis, evaluation (over recall, fluency)
   - Prefer higher Bloom: analyze, evaluate, create
   - Prefer questions with visuals
   - Exclude already-practiced question IDs
4. Hydrate selected questions via vault worker (get full scenario,
   solution, napkin_math)
5. Send hydrated pool to interview worker
6. When pool runs low (< 10 unused), fetch replacement batch biased
   toward uncovered areas
```

### Token budget management

A 45-minute interview might produce 40 turns. The transcript grows.

Strategy:
- Send full transcript for first 20 turns
- After 20 turns, summarize turns 1-10 into a 200-word recap, send
  recap + turns 11-current
- Question pool: send title + first 200 chars of scenario + metadata
  (not full details)
- Full question details sent only for the question the AI selected
- Max response tokens: 400 (interviewer turn), 1500 (feedback)

### Model requirements

The conductor prompt is complex (role + structure + evaluation +
conversation style). Minimum model quality:

- **Preferred**: Claude Sonnet, GPT-4o, Gemini Pro, Llama 3.3 70B
- **Acceptable**: GPT-4o-mini, Gemini Flash, Llama 3.1 70B
- **Too small**: Llama 3.1 8B (current Ask Interviewer model)

The existing worker supports multiple providers with fallback. Add a
model tier preference for the interview endpoint.

---

## UI Design

### Setup Phase

```
┌─────────────────────────────────────────────┐
│  Live Interview                             │
│                                             │
│  Track:    [Cloud] [Edge] [Mobile] [TinyML] │
│  Level:    [L4 Staff] [L5 Senior] [L6+ Pr]  │
│  Duration: [30 min] [45 min] [60 min]       │
│                                             │
│  Focus areas (optional):                    │
│  [x] memory  [x] compute  [ ] networking    │
│  [ ] latency [x] deployment [ ] power       │
│                                             │
│  Role description (optional):               │
│  [ Staff ML Engineer, Edge AI for KWS    ]  │
│                                             │
│  [Start Interview]                          │
└─────────────────────────────────────────────┘
```

### Active Phase

```
┌─────────────────────────────────────────────┐
│  Interview in Progress    ██████░░░ 24 min   │
│  Areas: memory ✓  compute ✓  latency ·       │
├─────────────────────────────────────────────┤
│                                             │
│  Interviewer:                               │
│  "Tell me about a system you've worked on   │
│  where memory was the binding constraint."   │
│                                             │
│  You:                                       │
│  "I worked on a serving system for a 70B    │
│  model where the KV cache dominated..."     │
│                                             │
│  Interviewer:                               │
│  "Interesting. Let's dig into that. If       │
│  your context window is 128K tokens, can     │
│  you estimate the KV cache size for that     │
│  70B model on an H100?"                      │
│                                             │
│  You:                                       │
│  [                                        ] │
│  [                    ] [Send] [End Early]   │
│                                             │
│  ┌─ Tools ───────────────────────────────┐  │
│  │ [NapkinCalc] [HardwareRef] [Glossary] │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

### Feedback Phase

```
┌─────────────────────────────────────────────┐
│  Interview Complete                         │
│                                             │
│  Rating: STRONG                             │
│  "Demonstrated deep systems thinking with    │
│  accurate napkin math and clear trade-off    │
│  analysis across memory and compute."        │
│                                             │
│  ┌─ Competency Coverage ───────────────────┐│
│  │ memory       ████████░░  Strong         ││
│  │ compute      ██████░░░░  Adequate       ││
│  │ latency      █████████░  Strong         ││
│  │ deployment   ░░░░░░░░░░  Not covered    ││
│  │ networking   ███░░░░░░░  Weak           ││
│  └─────────────────────────────────────────┘│
│                                             │
│  Strengths:                                 │
│  • KV cache sizing was exact (within 5%)    │
│  • Strong roofline reasoning                │
│                                             │
│  To improve:                                │
│  • Network topology: confused ring with     │
│    tree AllReduce [Practice: cloud-1234]     │
│  • Power budgeting not covered              │
│    [Practice: cloud-3336, cloud-3337]        │
│                                             │
│  [View Transcript] [New Interview] [Share]   │
└─────────────────────────────────────────────┘
```

---

## System Prompt (Conductor)

The system prompt has five sections. It is parameterized by the session
config and the question pool.

### Section 1: Role

You are a senior staff-level ML systems engineer conducting a technical
interview. You evaluate candidates for {TARGET_LEVEL} {TRACK} ML
engineer roles. Your style is conversational: you flow based on the
candidate's responses, zooming into weak areas and accelerating through
strengths.

{If roleDescription provided}: The candidate is interviewing for:
{roleDescription}.

### Section 2: Interview arc

Opening (2-3 min): Ask about their background. Use their answer to
pick a starting area.

Core (bulk of time): Present scenarios from the question pool. For
each:
- Paraphrase the scenario. Never read vault text verbatim.
- Let them ask clarifying questions. Answer with specific numbers.
- Follow up: "What if we doubled the traffic?" or "Walk me through
  the napkin math."
- Strong answer: zoom out or increase difficulty.
- Weak answer: zoom in, simplify, or offer a hint.

Cross-cutting probes: Weave in estimation, system impact, and
trade-off questions.

Closing (3-5 min): Summarize coverage, ask if they have questions,
end naturally.

### Section 3: Chain navigation instructions

You have chains of curated questions organized as difficulty
progressions. Each chain covers one topic and escalates through levels
and cognitive zones (e.g., fluency → diagnosis → evaluation).

When navigating a chain:
- Start at the entry point matching the candidate's target level
- If the candidate answers well: advance to the next position in the
  chain (harder level, different zone)
- If the candidate struggles: step back to the previous position
  (simpler framing of the same concept)
- When you reach the end of a chain (or the candidate clearly owns
  the topic): transition to a chain in an uncovered competency area

When presenting a scenario from the chain:
- Paraphrase naturally. Never read vault text verbatim.
- After the candidate responds, evaluate silently against the canonical
  solution and napkin math at that chain position
- Never reveal the canonical answer
- Use the chain's zone progression as your dialogue guide:
  fluency positions → test application
  diagnosis positions → ask them to find the bottleneck
  evaluation positions → ask them to judge trade-offs
  design positions → ask them to propose an architecture
  mastery positions → ask them to solve an open-ended extension

Coverage tracking:
- {COVERED_AREAS} have been explored. Prioritize {UNCOVERED}.
- Aim for {TARGET_AREAS} competency areas in this session.
- Go deep (3+ chain positions) in at least {DEEP_DIVE_COUNT} areas.

### Section 4: Evaluation criteria

Silently track:
- Chain of thought: can they articulate reasoning?
- Napkin math: right order of magnitude?
- System thinking: do they consider failure modes, scale, cost?
- Trade-off analysis: do they weigh alternatives?
- Depth vs breadth: where is expertise deep vs surface?

Signal weak areas by probing deeper. Signal strong areas by moving on.

### Section 5: Conversation style

- Be conversational. Real interviewers adapt.
- When candidates don't know an acronym, explain it briefly and move
  on. Do not penalize.
- Use "we" language: "So if we were deploying this to production..."
- Ask for system descriptions: "Walk me through the architecture."
- Do napkin math together: "Let's estimate that."
- Keep turns concise (under 150 words) except for new scenarios.
- Never reveal the canonical solution.

---

## Implementation Phases

### Phase 1: MVP (target: working end-to-end)

Files to create:
- `src/app/interview/page.tsx` (setup + active + feedback phases,
  modeled after gauntlet page.tsx phase machine)
- `src/lib/interview.ts` (session state, chain-aware pool selection,
  transcript management, coverage tracking)
- `src/lib/interview-prompt.ts` (system prompt builder with chain
  context injection)

Files to modify:
- `worker/src/index.ts` (add Conductor persona + POST /interview
  endpoint, alongside existing Socratic + Tutor personas)
- `src/lib/corpus.ts` (export chain selection helpers:
  getChainsForArea, getChainEntryPoint)

Components to reuse directly:
- `NapkinCalc`, `HardwareRef` (collapsible tool panels)
- `GlossaryText`, `MarkdownText` (message rendering)
- `LevelBadge`, `ChainBadge` (transcript annotations)
- `QuestionVisual` (inline diagrams)
- `Toast` (notifications)

What the MVP delivers:
- Setup page with track/level/duration/focus-area selectors
- Chat interface with scrolling transcript
- Chain-aware AI interviewer: navigates chains for depth (follow-up
  questions are real vault questions with solutions and napkin math),
  switches chains for breadth (coverage-driven transitions)
- Basic feedback page with per-area ratings and evidence
- Practice recommendations linked to specific questions
- Session persistence in localStorage
- Interview results fed back into SR system (weak areas get shorter
  review intervals)

What the MVP skips:
- Streaming responses (request/response for now)
- Diagram canvas (text descriptions only)
- Competency radar chart (simple bar chart instead)
- Transcript export
- Session comparison
- Voice input

### Phase 2: Polish

- Session timer with visual progress bar
- Inline NapkinCalc and HardwareRef panels (collapsed)
- Transcript context windowing (summarize early turns)
- Pool refresh as questions are used
- Keyboard shortcuts (Enter to send, Esc to end)

### Phase 3: Rich feedback

- Competency coverage visualization (bar chart or radar)
- Performance timeline (how quality evolved over session)
- "Practice these" deep links to practice mode with pre-set filters
- Transcript review with AI annotations per turn
- Transcript export (markdown)

### Phase 4: Future

- Streaming responses (SSE from worker)
- Voice input (Web Speech API)
- Simple diagram canvas (block + arrow drawing)
- Job-description-driven customization (paste a JD, AI tailors
  questions)
- Multi-candidate comparison (interviewer dashboard)
- Collaborative mode (human interviewer + AI observer/coach)

---

## Open Questions

1. **Model cost**: A 45-minute interview with 40 LLM calls at GPT-4o
   rates costs roughly $0.50-1.00. Is that acceptable per session?
   Alternatives: use a smaller model for follow-ups and a larger one
   for scenario selection and feedback.

2. **Rate limiting**: How many free sessions per day? The current Ask
   Interviewer allows 10 clarifications/hour. Interviews are heavier
   (30-40 calls/session). Proposal: 2 sessions/day free, or require
   API key for unlimited.

3. **Diagram interaction**: Text-based diagram description works for
   MVP. When should we add a visual canvas? What's the minimum viable
   diagram tool?

4. **Candidate answer persistence**: Should we store interview
   transcripts for later review? Currently all state is localStorage.
   For cross-device access, would need server-side storage.

5. **Multiplayer**: Peter described interviews where the interviewer
   uses draw.io together with the candidate. Could two users share a
   session (one as interviewer, one as candidate)? This is Phase 4+.
