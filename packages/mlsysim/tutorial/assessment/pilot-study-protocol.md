# Pilot Study Protocol: Evaluating mlsysim as a Teaching Tool for ML Systems Reasoning

## Study Overview

**Title:** First-Principles Performance Modeling Improves ML Systems Reasoning:
A Pre/Post Assessment of the mlsysim Tutorial at ISCA 2026

**Target venue:** SIGCSE Technical Symposium 2027 or L@S 2027

**Study type:** Single-group pre/post quasi-experiment (within-subjects design)

**Research question:** Does a 6-hour hands-on tutorial using first-principles
analytical modeling (mlsysim) improve attendees' ability to reason quantitatively
about ML system performance, as measured by transfer questions that do not require
the tool?

**Key distinction:** The quiz tests mental models, not tool proficiency. Questions
are answerable with mental arithmetic alone. This isolates the pedagogical contribution
of the framework from the utility of the software.

---

## Study Design

### Design: Within-Subjects Pre/Post

Each participant serves as their own control. The same 12-point quiz is administered
immediately before the tutorial (pre-test) and immediately after (post-test).

**Notation:** O1 X O2

- O1 = Pre-test (9:00 AM, before any instruction)
- X = Treatment (6-hour tutorial)
- O2 = Post-test (4:50 PM, after closing reflection)

### Why No Control Group

A control group is not feasible or necessary for the first publication:

1. **Logistical constraint:** ISCA tutorial attendees self-select into the room. There
   is no natural control population attending the same conference who would agree to take
   the quiz without attending the tutorial.

2. **Sufficient for initial evidence:** Pre/post designs are the standard for first
   reports on educational interventions in computing (see: Software Carpentry evaluations,
   CSEd workshop studies at SIGCSE). The within-subjects design controls for individual
   differences in background knowledge.

3. **Threats to validity are addressable:** The main threat is maturation/history (would
   scores improve just from thinking about the topics for 8 hours?). This is mitigated
   by the transfer nature of the questions -- they require specific frameworks taught in
   the tutorial, not general familiarity.

4. **Future work:** A controlled study (tutorial group vs. self-study group) is planned
   for the second offering, where classroom deployment makes random assignment feasible.

### Threats to Internal Validity

| Threat | Severity | Mitigation |
|--------|----------|------------|
| Testing effect (pre-test sensitizes attendees to post-test) | Moderate | Questions test transfer, not recall. The pre-test does not reveal correct answers or teach the framework. |
| History (external events during the day) | Low | The tutorial runs continuously; attendees are in the same room all day. |
| Maturation (natural improvement from 8 hours of thought) | Low | Questions require specific quantitative frameworks (Iron Law, Roofline) that are not common knowledge. |
| Attrition (attendees leave before post-test) | Moderate | Administer post-test before the final 5 minutes of closing. Offer it at 4:50 PM, not 5:00 PM. Report attrition rate. |
| Selection (self-selected ISCA attendees are not representative) | High | Acknowledged as a limitation. Results generalize to "motivated computer architects" not "all engineers." |

---

## Sample Size Justification

### Power Analysis

**Parameters:**
- Test: Paired (two-sided) t-test on total quiz score
- Expected effect size: Cohen's d = 1.0 (conservative estimate; we expect d ~ 1.2 based
  on similar workshop studies, but power for d = 1.0 provides a safety margin)
- Significance level: alpha = 0.05
- Desired power: 1 - beta = 0.80

**Required sample size:** N = 10 participants (paired t-test, d = 1.0, alpha = 0.05,
power = 0.80). Computed via G*Power or the formula:

```
N = ((z_alpha/2 + z_beta) / d)^2 + 1
N = ((1.96 + 0.84) / 1.0)^2 + 1
N = 7.84 + 1 = 8.84 -> 10
```

**Adjusted for attrition:** Assume 20% attrition (attendees who complete pre-test but
leave before post-test). Required enrollment: N = 10 / 0.80 = 13 participants minimum.

**Expected enrollment:** 40--80 attendees (ISCA tutorial capacity). Even with 50%
participation in the research component, we expect 20--40 paired responses. This
provides power > 0.99 for d = 1.0 and allows meaningful per-question analysis.

### Per-Question Analysis Power

For per-question McNemar tests (binary: correct/incorrect), the minimum detectable
effect requires:

- At least 10 discordant pairs (participants who change from wrong-to-right or
  right-to-wrong) per question
- With N = 30, and an expected pre-test accuracy of 40--50% rising to 70--80%,
  we expect 12--18 discordant pairs per question -- sufficient for McNemar's test.

---

## Analysis Plan

### Primary Analysis: Overall Learning Gain

**Test:** Paired two-sided t-test on total quiz scores (pre vs. post).

**Outcome variable:** Total score (0--12 scale).

**Hypothesis:** H1: mean(post) > mean(pre). H0: mean(post) = mean(pre).

**Reporting:** Mean pre-score, mean post-score, mean gain, 95% CI on the gain,
paired t-statistic, p-value, Cohen's d with 95% CI.

**Normality check:** Shapiro-Wilk test on the gain scores. If non-normal (p < 0.05),
supplement with the Wilcoxon signed-rank test. Report both if normality is violated.

### Secondary Analysis: Per-Understanding-Goal Gains

For each understanding goal (U1--U6), compute a 2-point sub-score (two questions per
goal). Report:

- Mean pre and post sub-scores per goal
- Paired t-test per goal (with Bonferroni correction: alpha = 0.05/6 = 0.0083)
- Rank goals by effect size to identify which understandings the tutorial teaches most
  effectively

### Tertiary Analysis: Per-Question McNemar Tests

For each multiple-choice question (Q1--Q5, Q8--Q10), construct a 2x2 table:

|  | Post-correct | Post-incorrect |
|--|-------------|----------------|
| **Pre-correct** | a | b |
| **Pre-incorrect** | c | d |

**Test:** McNemar's exact test on the discordant cells (b, c).

**Reporting:** For each question: pre-accuracy, post-accuracy, number of positive changes
(c), number of negative changes (b), McNemar p-value, odds ratio.

**Purpose:** Identifies which specific questions show the strongest learning signal and
which (if any) show negative transfer (post-test regression).

For short-answer questions (Q6, Q7), use the Wilcoxon signed-rank test on the 0/1/2
scores.

### Exploratory Analysis: Misconception Tracking

Using the distractor analysis from `quiz.md`, compute the prevalence of each named
misconception at pre-test and post-test:

- "More FLOPS = faster" (Q1a + Q2b selection rate)
- "Just add more GPUs" (Q3a + Q4c selection rate)
- "Quantization is just a latency trick" (Q5b selection rate)
- "Carbon = energy efficiency" (Q8a selection rate)
- "Benchmark first, decide later" (Q9d + Q10c selection rate)

Report the reduction in misconception prevalence with 95% CIs. This is the most
publishable aspect for a SIGCSE audience -- it connects learning gains to specific
conceptual changes.

### Exploratory Analysis: Demographic Subgroups

If the optional demographic survey yields sufficient responses (N >= 10 per subgroup),
compare learning gains by:

- Career stage (PhD student vs. industry engineer vs. faculty)
- Self-reported ML systems experience (none / some / extensive)
- Architecture background (strong / moderate / weak)

Use independent t-tests or Mann-Whitney U on gain scores. These are exploratory and
will be reported as such (no multiple comparison correction; findings used to generate
hypotheses for the controlled study).

---

## Data Collection Instruments

| Instrument | When | Required/Optional | Contains PII |
|------------|------|-------------------|--------------|
| Pre-test quiz | 9:00 AM | Required for research | No (participant ID only) |
| Post-test quiz | 4:50 PM | Required for research | No (participant ID only) |
| Demographic survey | Registration or lunch | Optional | Minimal (career stage, years experience, institution type) |
| Consent form | 9:00 AM (top of pre-test) | Required for research | No (opt-out model) |

All instruments are in the `assessment/` directory.

---

## Timeline

### Pre-Tutorial (8+ weeks before ISCA)

| Week | Task |
|------|------|
| T-10 | File IRB application (exempt category). Include all instruments. |
| T-8 | Receive IRB approval (or clarification requests). |
| T-6 | Finalize quiz wording. Pilot with 3--5 colleagues for timing and clarity. |
| T-5 | Revise quiz based on pilot feedback. Finalize digital form (Google Forms). |
| T-4 | Generate participant ID codes (100 six-digit random numbers). Print ID cards. |
| T-3 | Print paper quiz forms (100 copies, double-sided). |
| T-2 | Prepare data analysis scripts (R or Python). Pre-register analysis plan on OSF. |
| T-1 | Dry-run the quiz administration with a practice group. Time it. |

### Day of Tutorial (ISCA 2026)

| Time | Task |
|------|------|
| 8:30 AM | Set up: distribute ID cards at seats. Prepare quiz links/forms. |
| 8:55 AM | Read consent statement aloud. Display quiz link/form. |
| 9:00 AM | Start pre-test timer (5 minutes). |
| 9:05 AM | Collect pre-test forms. Begin tutorial. |
| 4:50 PM | Distribute post-test (same quiz). Start 5-minute timer. |
| 4:55 PM | Collect post-test forms. Proceed to closing. |
| 5:00 PM | Distribute optional demographic survey (paper or QR code). |

### Post-Tutorial (2--8 weeks after ISCA)

| Week | Task |
|------|------|
| T+1 | Enter paper responses into spreadsheet (if paper forms used). |
| T+1 | Destroy name-to-ID mapping. Data is now permanently de-identified. |
| T+2 | Run primary analysis (paired t-test). Check assumptions. |
| T+3 | Run secondary and tertiary analyses. Generate figures. |
| T+4 | Draft results section. Compute all CIs and effect sizes. |
| T+6 | Complete manuscript draft. |
| T+8 | Submit to SIGCSE 2027 (September deadline) or L@S 2027. |

---

## Expected Results

### Hypotheses

**H1 (Primary):** The mean post-test score will be significantly higher than the mean
pre-test score (paired t-test, p < 0.05), with a large effect size (Cohen's d > 0.8).

**Rationale:** The tutorial is 6 hours of intensive, hands-on instruction using the
predict-code-reflect cycle. Similar computing workshops (Software Carpentry, CS
Unplugged) report d = 0.8--1.5 for pre/post designs. The ISCA audience is highly
motivated and technically sophisticated, which should amplify learning gains.

**H2 (Secondary):** The largest per-goal gains will be on U4 (compression as architecture)
and U5 (carbon geography), because these represent the most novel content for an
architecture audience.

**Rationale:** ISCA attendees likely already understand compute vs. memory bottlenecks
(U1) and parallelism (U3) from their architecture training. The Roofline model is
widely taught. However, the fleet-level implications of quantization and the dominance
of grid carbon intensity over hardware efficiency are not standard architecture
curriculum. These are the "aha moments" most likely to produce score gains.

**H3 (Tertiary):** The misconception "more FLOPS = faster" will decrease by at least
50% from pre to post.

**Rationale:** This is directly addressed by Aha Moment #1 at 10:00 AM, with the
predict-then-reveal structure designed to create cognitive conflict. It is the
single most targeted misconception in the tutorial.

### Expected Quantitative Results

| Metric | Expected value | Basis for estimate |
|--------|---------------|-------------------|
| Pre-test mean | 5.5 / 12 (46%) | ISCA audience: strong architecture, partial ML systems |
| Post-test mean | 8.5 / 12 (71%) | 6 hours of targeted instruction on exactly these topics |
| Mean gain | 3.0 points | Difference |
| Cohen's d | 1.0--1.3 | Comparable workshop studies |
| Pre-test "more FLOPS = faster" prevalence | 40--50% | Common misconception even among architects |
| Post-test "more FLOPS = faster" prevalence | 10--15% | After Aha #1 and extensive Roofline practice |
| Attrition rate | 10--20% | Typical for full-day ISCA tutorials |
| Research participation rate | 60--80% | Opt-out consent model with no compensation |

### What Would Be Surprising

- **Pre-test mean > 8:** Would indicate the ISCA audience already reasons this way,
  reducing the tutorial's contribution. The quiz may need harder questions.
- **No gain on U1 (Roofline):** Would suggest the Roofline model is already well-known
  to this audience (possible, since it is taught in architecture courses).
- **Negative gain on any question:** Would indicate the tutorial introduced a new
  misconception. This requires immediate investigation and tutorial revision.
- **Attrition > 30%:** Would threaten statistical power and suggest engagement problems
  in the afternoon sessions.

---

## Manuscript Outline (SIGCSE 2027)

For planning purposes, the target paper structure:

1. **Introduction:** The need for quantitative ML systems reasoning; the gap between
   architecture education and ML systems practice.

2. **Related Work:** Roofline model pedagogy (Williams et al.), ML systems courses
   (Stanford CS229S, CMU 10-414), computing education assessment (ITiCSE working
   groups on concept inventories).

3. **The mlsysim Tutorial:** Design principles, the six understanding goals, the
   predict-code-reflect cycle. Reference the DESIGN.md document.

4. **Assessment Design:** The 10-question quiz, mapping to understanding goals,
   distractor rationale. Reference `quiz.md`.

5. **Methods:** Participants, procedure, analysis plan (this document).

6. **Results:** Pre/post scores, effect sizes, per-goal gains, misconception tracking.

7. **Discussion:** Which understandings transferred, which did not, implications for
   ML systems curriculum design.

8. **Limitations:** Single group, self-selected ISCA population, testing effect,
   no long-term retention data.

9. **Future Work:** Controlled study with self-study comparison group, deployment in
   semester-long courses, longitudinal retention assessment at 6 months.

---

## Pre-Registration

Before the tutorial, pre-register the study on OSF (Open Science Framework):

- **URL:** https://osf.io/registries
- **Template:** AsPredicted or OSF Standard Pre-Data Collection Registration
- **Include:** Research question, hypotheses H1--H3, analysis plan (primary, secondary,
  tertiary), sample size justification, quiz instrument, expected effect size.

Pre-registration strengthens the publication by demonstrating that the analysis plan
was not influenced by the observed data. This is increasingly expected at SIGCSE and L@S.

---

## Budget

| Item | Cost | Notes |
|------|------|-------|
| Printing (200 quiz forms) | $50 | Double-sided, B&W |
| Google Forms (digital backup) | $0 | Free with institutional account |
| Participant ID cards | $20 | Pre-printed labels |
| Statistical software | $0 | R (free) or scipy (free) |
| OSF pre-registration | $0 | Free |
| IRB filing | $0 | Typically free for exempt studies |
| **Total** | **$70** | |

---

## Contact and Responsibilities

| Role | Person | Responsibility |
|------|--------|---------------|
| PI | [TBD] | IRB filing, study design, manuscript lead |
| Tutorial lead | [TBD] | Quiz administration, data collection |
| Data analyst | [TBD] | Analysis scripts, figures, results section |
| Second author | [TBD] | Related work, discussion, editing |
