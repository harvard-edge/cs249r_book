# Float Exposition Audit — vol2 / network_fabrics

**Chapter:** `network_fabrics.qmd` (vol2)
**Standard:** FLOAT_EXPOSITION_STANDARD.md
**Audited:** 2026-06-09

---

## Summary table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|------|-------|--------|----|----|-----|
| Figure | 🟠 high | 11 | 8 | 3 | 0 |
| Table | 🟠 high | 1 | 1 | 0 | 0 |
| Listing | 🟡 medium | 0 | — | — | — |
| Equation | 🔴 strict | 0 | — | — | — |
| **Total** | | **12** | **9** | **3** | **0** |

---

## Findings (⚠️ only)

---

### F1 — `fig-network-five-level-model` (Figure 🟠) — def L80

**Verbatim ref sentence (L78):**
> @Fig-network-five-level-model organizes the fabric design space into five co-dependent levels, from physical signaling to cluster-scale orchestration. The acronyms in the figure are a roadmap: each mechanism is defined when it becomes the binding constraint.

**Issue — missing lead-out / interpret.** The cite sentence frames the figure as a roadmap (orientation) but does not deliver the takeaway: why this five-level decomposition is the right analytical lens and what it reveals about network performance. The immediately following payoff paragraph (L84) pivots to the fleet law and communication cost, never returning to interpret the figure's insight. The actual interpret — that performance is the product of interactions across all five levels, not link speed alone — appears only in the Summary at L1858 and lives nowhere in the body prose near the float's definition.

**Where the takeaway currently lives:** Summary paragraph (L1858), caption level descriptions.

**Rule-compliant rewrite (replace the second sentence of the L78 cite):**

> @Fig-network-five-level-model organizes the fabric design space into five co-dependent levels, from physical signaling to cluster-scale orchestration. The framework's central insight is that network performance is the product of interactions across all five levels simultaneously: a fast link (Level 1) cannot compensate for a misconfigured congestion-control policy (Level 4), and a well-chosen topology (Level 3) delivers no benefit if transport-level losslessness (Level 2) is absent. Each mechanism in the figure is defined in this chapter when it becomes the binding constraint at its level.

---

### F2 — `fig-fat-tree-detail` (Figure 🟠) — def L914

**Verbatim ref sentence (L918):**
> @Fig-fat-tree-detail shows the resulting three-tier structure.

**Issue — cite arrives after the float; cite sentence is a bare pointer.** The float is defined at L914–916; the only citation is at L918 (after the float). The sentence "@Fig-fat-tree-detail shows the resulting three-tier structure" is a structural announcer that points at the figure without interpreting it. The BW_bisect argument appears in the sentence that precedes the cite, not as a lead-out conclusion drawn from the figure. The prose then narrates the Leaf/Spine/Core tier labels, which is structural description rather than the interpret move the standard requires (the key result: what the tier structure achieves and why multiple parallel paths matter for AllReduce).

**Where the takeaway currently lives:** Implicitly in the callout definition at L904–912 (BW_bisect significance) and in the caption.

**Rule-compliant rewrite (replace the cite sentence at L918):**

> The fat-tree[^fn-fat-tree-clos] is a common default for ML clusters because a non-blocking design can provide full $\text{BW}_{\text{bisect}}$, a nonnegotiable requirement for the AllReduce collective, which demands simultaneous, all-to-all communication. @Fig-fat-tree-detail shows why: the three-tier structure creates multiple equal-cost spine paths between any two leaf switches, so aggregate bandwidth at each tier matches the edge bandwidth feeding into it and no single switch becomes a bottleneck. The network is constructed in hierarchical tiers: **Leaf** switches (ToR) connect directly to servers, **Spine** switches interconnect all leaves within a locality domain known as a pod\index{Pod}, and **Core** switches bind multiple pods together.

---

### F3 — `fig-rail-optimized-cabling` (Figure 🟠) — def L1067

**Verbatim ref sentence (L1065):**
> Rail-optimized topology begins from the communication pattern rather than from a generic switch hierarchy. In dense accelerator nodes, GPUs can be grouped by local slot position into rails; @fig-rail-optimized-cabling makes that physical cabling pattern concrete.

**Issue — cite is a bare pointer; no interpret in the cite sentence or nearby lead-out.** The single citation sentence at L1065 points at the figure ("makes the cabling pattern concrete") without stating what the pattern achieves or why it matters. The interpret — that same-rank GPU pairs get dedicated rail switches, eliminating bandwidth competition between data-parallel AllReduce traffic and other flows — appears only in the payoff paragraph at L1073, which is separated from the cite by the figure itself. The standard requires the interpret to be in body prose at or near the cite, not delegated entirely to the post-float paragraph.

**Where the takeaway currently lives:** Payoff paragraph at L1073, caption.

**Rule-compliant rewrite (replace the second sentence of the L1065 cite):**

> Rail-optimized topology begins from the communication pattern rather than from a generic switch hierarchy. In dense accelerator nodes, GPUs can be grouped by local slot position into rails, so that same-rank GPUs across different nodes share a dedicated switch fabric; @fig-rail-optimized-cabling makes this wiring pattern concrete. The key result is that data-parallel AllReduce traffic, which synchronizes GPU 0 on one node with GPU 0 on every other node, never competes for bandwidth with other parallel groups, cutting the hop count and eliminating contention for the most latency-sensitive collective.

---

## Notes

- `fig-bandwidth-hierarchy` (L150): The first cite sentence at L150 is a bare announcer ("makes this hierarchy concrete by plotting"), but the full interpret is delivered in the payoff paragraph at L162 immediately after the figure. The overall contract is met; the thin first cite is compensated by the strong lead-out.
- `fig-network-topologies` (L1182): Both main cite sentences (L890 "turns these topology definitions into a bandwidth and cost comparison" and L1180 "quantifies the bandwidth and cost differences") are float-announcer sentences. However, the surrounding prose at L1178 is an exceptionally strong lead-in delivering the full contrast (fat-tree vs. torus, workload-regularity argument, cost vs. universality), and the payoff at L1188 delivers workload-dependent conclusions. The contract is marginally met; the cite sentences themselves are weak, but the surrounding context carries the interpret. Not elevated to a finding given the richness of adjacent prose, but noted as a candidate for refinement.
- `@fig-fleet-stack` (L78) and `@eq-fleet-law` (L84) are dangling references with no definitions in this chapter. Not a float-exposition finding (the floats live in another chapter), but worth flagging for cross-reference audit.
