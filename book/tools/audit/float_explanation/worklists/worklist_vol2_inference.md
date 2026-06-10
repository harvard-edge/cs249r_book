# Float-explanation worklist — inference.qmd (vol2)

## Summary
| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 26 | 24 | 2 | 0 |
| table | 64 | 64 | 0 | 0 |
| listing | 8 | 7 | 1 | 0 |
| algorithm | 2 | 2 | 0 | 0 |
| equation | 15 | 15 | 0 | 0 |
| **total** | **115** | **112** | **3** | **0** |

## Findings (⚠️ only — ✅ floats tallied above, not expanded)

---

### ⚠️ `fig-serving-hierarchy` — def L589  (Thin)
- **Caption:** "**Serving Deployment Stack**: A three-tier serving stack with cumulative latency budget from top to bottom. Tier 1 (CDN/Edge Cache)… Tier 2 (Gateway/Router)… Tier 3 (Model Serving Cluster)…"
- **Ref(s):** L587 `@fig-serving-hierarchy`: "The hierarchy matters because each level changes a different metric and fails at a different boundary. A related deployment stack appears in @fig-serving-hierarchy, showing how requests pass through edge, routing, and model-serving infrastructure in production."
- **Context checked:** ref ⚠️ (announcer, no takeaway) · prev ¶ ✅ (explains 4-level conceptual hierarchy) · next ¶ ✅ (payoff leads to @Tbl-serving-hierarchy) · caption ✅ (SLA budgets per tier) · payoff ✅ (table maps levers)
- **Issue:** The reference sentence introduces the figure with a distancing qualifier ("A related deployment stack") and only says what the figure shows ("how requests pass through"), not what the reader should take away from it. The preceding paragraph describes a conceptual four-level hierarchy (request, replica, service, platform); the figure shows a three-tier deployment topology (CDN, gateway, model cluster). The relationship between the two framings is never stated, so the figure appears as a digression rather than reinforcement.
- **Suggested rewrite (flag-only):**
  ```diff
  - The hierarchy matters because each level changes a different metric and fails at a different boundary. A related deployment stack appears in @fig-serving-hierarchy, showing how requests pass through edge, routing, and model-serving infrastructure in production.
  + The hierarchy matters because each level changes a different metric and fails at a different boundary. @Fig-serving-hierarchy shows the physical deployment stack that hosts these four levels: requests arrive at the CDN tier (request level), are routed through the gateway tier (service level), and land on the model-serving cluster (replica level), with the platform level governing all three. Each tier boundary is a latency checkpoint where the cumulative SLO budget shrinks.
  ```

---

### ⚠️ `fig-embedding-sharding` — def L3458  (Thin)
- **Caption:** "**Embedding Sharding Strategies**: Row-wise sharding places complete embedding vectors on specific servers based on entity ID… Column-wise sharding splits each vector across all servers… Hybrid sharding combines these approaches…"
- **Ref(s):** L3456 `@Fig-embedding-sharding`: "@Fig-embedding-sharding visualizes how each strategy distributes embedding lookups across devices:"
- **Context checked:** ref ✗ (bare colon-announcer) · prev ¶ ✅ (prose + table explain the three strategies) · next ¶ (figure def itself) · caption ✅ (describes panels) · payoff ✅ (L3520: "All three sharding strategies operate at massive scale in production")
- **Issue:** The reference is a pure colon-announcer. The prose paragraph before the figure and `@tbl-embedding-sharding` (directly above) already explain all three strategies textually. The reference gives no reason why the visual form adds insight over the table: the figure's value is showing the physical data movement (gather arrows, column slices) that the table cannot convey, and that distinction is never stated.
- **Suggested rewrite (flag-only):**
  ```diff
  - @Fig-embedding-sharding visualizes how each strategy distributes embedding lookups across devices:
  + @Fig-embedding-sharding makes the data movement explicit: row-wise sharding requires a single-hop network gather per lookup, column-wise sharding fans out to every device but parallelizes the bandwidth, and hybrid sharding compresses the hot-shard bottleneck by spreading popular vectors while keeping cold ones local. The arrows distinguish strategies that look identical in the table above.
  ```

---

### ⚠️ `lst-metric-based-scaling` — def L4709  (Thin)
- **Caption:** "**Metric-Based Scaling**: Autoscaling policy driven by utilization thresholds with a cooldown period to prevent oscillation."
- **Ref(s):** L4707 `@Lst-metric-based-scaling`: "@Lst-metric-based-scaling shows a typical metric-based scaling configuration."
- **Context checked:** ref ✗ (bare announcer, no substance) · prev ¶ ✅ (explains why reactive scaling cannot handle sudden spikes) · next ¶ ✅ (L4722: explains queue-depth alternative) · caption ✅ (names cooldown) · payoff ✅ (explains oscillation limitation)
- **Issue:** The reference is a bare "shows a typical configuration" with no explanation of what the configuration elements mean or why they matter. The YAML listing contains two asymmetric thresholds (80 percent up, 50 percent down) and a 300-second cooldown — both embody design choices the student should understand. Neither the ref sentence nor the payoff paragraph explains the asymmetry or why the cooldown duration matters.
- **Suggested rewrite (flag-only):**
  ```diff
  - @Lst-metric-based-scaling shows a typical metric-based scaling configuration.
  + @Lst-metric-based-scaling shows a typical threshold-based autoscaler: the asymmetric thresholds (80 percent up, 50 percent down) create hysteresis that prevents thrashing, and the 300-second cooldown prevents rapid scale-down before new replicas finish serving their in-flight requests.
  ```
