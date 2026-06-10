# Verified findings — inference.qmd (vol2)
Prior findings: 3 | Survived: 2 | Refuted: 1

---

## SURVIVING findings

### ⚠️ `fig-serving-hierarchy` — def L589
- Ref: "A related deployment stack appears in @fig-serving-hierarchy, showing how requests pass through edge, routing, and model-serving infrastructure in production."
- Why it survives: The ref sentence is a bare announcer. The prev paragraph (L585) establishes a four-level conceptual hierarchy (request, replica, service, platform). The figure shows a three-tier physical deployment topology (CDN/Edge Cache, Gateway/Router, Model Serving Cluster). No element in the neighborhood — not the ref sentence, not the caption, not the payoff (L593, which pivots immediately to the table) — ever states the relationship between the four conceptual levels and the three physical tiers, nor why the figure reinforces the hierarchy argument made in the preceding paragraph. The caption describes what each tier does (SLA numbers per tier) but does not connect the physical tiers to the conceptual levels. A reader cannot tell whether Tier 1 maps to the "service level," the "platform level," or neither. The figure reads as a digression from the hierarchy argument rather than its physical embodiment.
- Suggested rewrite (no em-dash/hyphen, ≤1 colon/para):
  ```diff
  - The hierarchy matters because each level changes a different metric and fails at a different boundary. A related deployment stack appears in @fig-serving-hierarchy, showing how requests pass through edge, routing, and model-serving infrastructure in production.
  + The hierarchy matters because each level changes a different metric and fails at a different boundary. @Fig-serving-hierarchy shows the physical deployment stack that hosts these four levels: the CDN tier absorbs request-level traffic before it reaches the serving infrastructure, the gateway tier routes and rate-limits at the service level, and the model-serving cluster is where replica-level and platform-level optimizations operate. Each tier boundary is a latency checkpoint with its own SLA budget, and a failure at any boundary manifests as a different metric violation.
  ```

---

### ⚠️ `lst-metric-based-scaling` — def L4709
- Ref: "@Lst-metric-based-scaling shows a typical metric-based scaling configuration."
- Why it survives: The ref is a bare announcer with no substance. The YAML listing contains three distinct design choices that embody tradeoffs: asymmetric thresholds (80 percent scale-up, 50 percent scale-down), a specific cooldown duration (300 seconds), and the choice of cpu_utilization as the primary metric. None of these choices is explained anywhere in the neighborhood. The prev paragraph (L4705) discusses cold-start latency, not threshold design. The caption (L4709) names the cooldown period and says it prevents oscillation, but does not explain why 300 seconds or why the scale-down threshold is set 30 points below the scale-up threshold. The payoff paragraph (L4722) moves immediately to a queue-depth alternative without explaining what the listing's asymmetric thresholds achieve. A student reading this listing has no way to understand why the thresholds differ or what would happen with a symmetric design.
- Suggested rewrite (no em-dash/hyphen, ≤1 colon/para):
  ```diff
  - @Lst-metric-based-scaling shows a typical metric-based scaling configuration.
  + @Lst-metric-based-scaling shows a typical threshold-based autoscaler. The asymmetric thresholds (80 percent to scale up, 50 percent to scale down) create a hysteresis band that prevents oscillation between scale-up and scale-down decisions when utilization hovers near a single threshold. The 300-second cooldown reinforces this stability by blocking further scaling actions until newly provisioned replicas have fully warmed up and begun absorbing traffic, so the system does not interpret transient underutilization during warm-up as a signal to scale back down.
  ```

---

## REFUTED findings

- `fig-embedding-sharding` — REFUTED: explanation is in caption (L3458): "Row-wise sharding places complete embedding vectors on specific servers based on entity ID, **requiring a network gather for lookup**. Column-wise sharding splits each vector across all servers, allowing **parallel local lookups followed by an AllGather**, which is efficient for popular 'hot' embeddings. Hybrid sharding combines these approaches, using column sharding for hot items and row sharding for the 'cold' long tail to balance load and memory." The caption explicitly describes the data movement mechanics that distinguish the three strategies (single-hop gather vs. fan-out AllGather vs. mixed), which is the visual information the figure adds over the table at L3448. The refutation bar is met: the caption tells the reader what the float shows and why the visual form adds value over the table.
