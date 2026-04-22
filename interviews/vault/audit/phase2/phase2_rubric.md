# StaffML vault audit rubric — schema v1.0

Background: `interviews/vault/questions/<track>/<id>.yaml`. Classification is four orthogonal axes: `track`, `level`, `zone`, `topic`. Paper at `interviews/paper/paper.tex` §3 defines the model.

## Levels (Bloom's revised)
L1 Remember · L2 Understand · L3 Apply · L4 Analyze · L5 Evaluate · L6+ Create/Mastery

## Zones (what skills the question exercises)
- recall           [remember]                 natural L1-L2
- fluency          [remember + quantify]      natural L2-L3
- analyze          [analyze]                  natural L3-L4
- diagnosis        [remember + analyze]       natural L3-L4
- design           [create]                   natural L4-L5
- specification    [remember + create]        natural L4-L5
- optimization     [analyze + quantify]       natural L4-L5
- evaluation       [analyze + create]         natural L5-L6+
- realization      [create + quantify]        natural L5-L6+
- mastery          [all four]                 natural L6+
- implement        [execute/apply]            natural L2-L4

## Decision procedure for each question

1. Read `scenario` + `details.scenario/realistic_solution/common_mistake/napkin_math`.
2. Identify which of the four skills (recall, analyze, design/create, quantify/napkin-math) the question actually exercises.
3. Pick the zone that best matches the skills exercised.
4. Pick the level based on cognitive depth:
   - L1/L2: "what is X" / short quantification from memory
   - L3: "given this setup, compute/apply"
   - L4: "diagnose this failure" / "compare these two"
   - L5: "design X" / "evaluate trade-offs with napkin math"
   - L6+: "diagnose + design + size" compound; all four skills
5. If the CURRENT (zone, level) pair is defensible based on content, output `decision: keep`.
   Otherwise output `decision: reclassify` with the proposed labels.

## Heuristics

- L6+ zone=design with concrete napkin math across multiple decisions → usually **mastery**.
- evaluation@L4 without a quantitative comparison → usually **diagnosis@L4** or **analyze@L4**.
- diagnosis@L5 where the question demands redesign, not just root-cause → usually **evaluation@L5**.
- recall@L3+ where the answer requires computation → usually **fluency** at the same level.
- If the question requires napkin math + a full architecture proposal + root-cause → **mastery**.

## Confidence

- `high`: clear mismatch, content obviously exercises different skills than labels say
- `medium`: labels are arguable but one is clearly better
- `low`: labels are defensible; only a mild preference
