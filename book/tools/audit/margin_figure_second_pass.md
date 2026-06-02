# Margin Figure Second Pass

Date: 2026-06-02

Scope: editorial and visual-language pass after the full 212-figure render audit.

## Decisions

1. Keep the margin device kit closed by default. The current kit already covers
   the durable textbook relationships: magnitude spans, thresholds, trends,
   roofline regimes, term dominance, framework locators, taxonomies, and
   one-to-many propagation.
2. Treat `other-new` as an audit flag, not a production device. A candidate with
   no honest kit fit should usually become a body figure, prose, or a cut.
3. Draft the caption before drawing. The caption is the editorial takeaway, not
   a title or legend; it must reinforce the same local paragraph beat as the
   figure.
4. Keep semantic color. Same-metric bars should usually share one hue. For
   energy/power ladders, orange is correct; recoloring bars for variety would
   imply false categories.

## Rule Updates Made

| File | Update |
|---|---|
| `.claude/rules/margin-figures.md` | Added caption discipline: repeatable tests, device-specific caption patterns, bad-caption rejection rules, and alt-text distinction. |
| `.claude/rules/margin-figures.md` | Tightened the device catalog to eight production devices plus an `other-new` audit exception and a new-device gate. |
| `.claude/rules/figure-visual-language.md` | Added master-grammar reminder that margin visuals must reduce to the tested kit unless approved as a durable grammar addition. |

## Strict Candidate List From Parallel Audits

These are not automatic additions. They are the small set worth considering
after applying the "does the reader lose a relationship?" gate.

| Priority | Chapter | Candidate | Device fit | Purpose |
|---|---|---|---|---|
| High | vol1/data_engineering | Data debt compounding curve | `sparkline-trend(style="inflection")` or body figure if too detailed | Show that data debt grows superlinearly rather than linearly. |
| Medium-high | vol1/hw_acceleration | Heterogeneous SoC workload split | likely body/mechanism figure; do not force as margin | Anchor CPU/GPU/NPU partitioning only if it can reduce to one clean relationship. |
| High | vol1/model_serving | Utilization-latency cliff | `scale-anchor(style="shaded")` | Show why high utilization creates latency divergence. |
| Medium | vol2/compute_infrastructure | CXL capacity-vs-bandwidth boundary | `hierarchy-ladder` | Prevent readers from treating CXL as HBM replacement. |
| High | vol2/network_fabrics | PFC pause cascade | `blast-radius(style="tree")` if simplified; body figure if mechanism detail is needed | Show one slow receiver stalling unrelated flows. |
| Medium | vol2/edge_intelligence | Federated heterogeneity envelope | likely body/table; do not add radar as margin | Show multi-resource heterogeneity only if reducible to a simple ladder. |
| High | vol2/ops_scale | Telemetry correlation hierarchy | `hierarchy-ladder(style="staircase")` or `taxonomy-mini` | Show GPU/node/rack/facility localization as hierarchy. |
| Medium | vol2/security_privacy | Output leakage shrinkage | `hierarchy-ladder` or `iron-law-bar` | Show that logits leak more information than rounded/top-k/class-only outputs. |

## Visual Fix From This Pass

`vol2_sustainable_ai_margin_004.svg` had a clipped long label in the rendered
sheet. The generator now splits the energy labels into two lines:

- `radio bit / 250K pJ`
- `FP32 mult / 4 pJ`
- `INT32 add / 0.1 pJ`

The ratio label remains in the gap and the energy color remains orange.

## Deferred Editorial TODO

- Add a short About-the-Book note explaining that margin visuals appear beside
  the prose as glanceable thinking aids, not decorative art. Consider a compact
  row of four or five representative device types after the figure work is
  fully settled.
