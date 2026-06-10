# Float Exposition Audit — `fleet_orchestration.qmd` (vol2)

> Standard: FLOAT_EXPOSITION_STANDARD.md
> Method: scan_floats.py bundle + ±40-line body-prose review per float
> Caption, fig-alt, in-figure labels, code comments, callout interiors excluded from prose credit.

---

## Summary table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|------|-------|--------|----|----|-----|
| Algorithm | 🔴 strict | 1 | 1 | 0 | 0 |
| Equation | 🔴 strictest | 5 | 2 | 3 | 0 |
| Figure | 🟠 high | 6 | 5 | 1 | 0 |
| Listing | 🟡 medium | 2 | 1 | 1 | 0 |
| Table | 🟠 high | 7 | 4 | 3 | 0 |
| **Total** | | **21** | **13** | **8** | **0** |

---

## Findings (⚠️ only — no 🛑)

---

### 1. `eq-fleet-orchestration-spot-cost` ⚠️ (equation 🔴) — def L1393

**Verbatim ref sentence (L1391):**
> "@Eq-fleet-orchestration-spot-cost gives the **effective cost** of spot training:"

**Missing move:** The equation has three symbols — $C_{\text{spot}}$, $T_{\text{total}}$, $T_{\text{productive}}$ — none of which receive a prose "where" clause. The lead-in names only the abstract concepts ("cost of each interruption" and "frequency of interruption") without mapping them to the equation's variables. The payoff paragraph (L1397) discusses fault tolerance and elastic training generally but never states the quantitative consequence: what ratio $T_{\text{total}} / T_{\text{productive}}$ looks like under typical interruption rates, or what effective cost results when, say, 10 percent of run time is lost to checkpoint/restart overhead. The takeaway lives in the reader's arithmetic, not in the prose.

**Takeaway currently lives:** Implied only; no symbol definitions or worked example anywhere in body prose.

**Rule-compliant rewrite** — add after the equation display at L1393, before the existing L1397 payoff:

> where $C_{\text{spot}}$ is the hourly spot price, $T_{\text{total}}$ is total wall-clock hours billed, and $T_{\text{productive}}$ is hours spent on productive forward and backward passes (excluding checkpoint overhead and restart time). A job with 10 percent overhead from hourly checkpoints and two interruptions per day pays a $T_{\text{total}} / T_{\text{productive}}$ penalty of roughly 1.15, so a 70 percent spot discount translates to an effective cost of about $0.35 per GPU-hour instead of the naive $0.30. Interruption frequency and checkpoint interval jointly determine whether the discount survives the overhead.

---

### 2. `eq-fleet-orchestration-tco` ⚠️ (equation 🔴) — def L1541

**Verbatim ref sentence (L1539):**
> "@Eq-fleet-orchestration-tco expresses the TCO of an on-premise cluster over a typical three-year hardware lifecycle as the sum of Capital Expenditure $(C_{\text{cap}})$, Operational Expenditure $(C_{\text{ops}})$, and the often-overlooked Opportunity Cost $(C_{\text{opp}})$ of tying up capital in depreciating assets:"

**Missing move:** $C_{\text{cap}}$ and $C_{\text{ops}}$ are unpacked in the payoff (L1543) with concrete figures ($40M and $10M in hardware, high fixed baseline for ops). $C_{\text{opp}}$ appears only in the lead-in as a definition label ("the often-overlooked Opportunity Cost") and vanishes from the payoff entirely. The equation multiplies $C_{\text{ops}}$ by 3 (three-year lifecycle) but the prose never states the regime implication of that multiplier: when $3 \times C_{\text{ops}}$ exceeds $C_{\text{cap}}$, on-premise ownership becomes more expensive than cloud rental. That crossover is the equation's point and it is absent from the prose.

**Takeaway currently lives:** Absent from prose; $C_{\text{opp}}$ consequence and the crossover logic live only implicitly in the equation.

**Rule-compliant rewrite** — add to the payoff paragraph (L1543), appending after the existing $C_{\text{ops}}$ sentence:

> $C_{\text{opp}}$ represents the return the organization forgoes by deploying capital in depreciating hardware rather than alternatives. For a $50M cluster, a conservative opportunity cost rate of 10 percent adds $5M per year, or $15M over the three-year lifecycle, bringing total TCO to roughly $115M for a 1,024-H100 cluster. When $3 \times C_{\text{ops}}$ approaches or exceeds $C_{\text{cap}}$, the on-premise cost advantage over cloud disappears, and the decision shifts to utilization certainty: an owned cluster only wins financially if it sustains the high utilization needed to amortize both capital and operational costs.

---

### 3. `eq-fleet-orchestration-quota` ⚠️ (equation 🔴) — def L2004

**Verbatim ref sentence (L2002):**
> "@Eq-fleet-orchestration-quota bounds the effective quota for any team by both its own allocation and the remaining capacity within its parent organization:"

**Missing move:** The equation contains three variables — $Q_{\text{team}}$, $Q_{\text{department}}$, and $\sum_{\text{other teams}} U_{\text{allocated}}$ — none of which receive a prose "where" clause. The lead-in establishes why hierarchical quotas exist; the payoff (L2006) correctly states the proportional allocation and borrowing behavior. However, the symbols themselves are never defined in body prose, violating the equation-level standard that every symbol must be named in prose. A reader who does not expand the equation cannot know what $U_{\text{allocated}}$ means (it is active allocations by other teams within the same department, not total cluster usage).

**Takeaway currently lives:** The high-level borrowing behavior is in the payoff (L2006); the symbol definitions are absent from prose.

**Rule-compliant rewrite** — add a "where" clause immediately after the equation at L2004:

> where $Q_{\text{team}}$ is the team's own quota, $Q_{\text{department}}$ is the total quota assigned to its parent department, and $\sum_{\text{other teams}} U_{\text{allocated}}$ sums all resources currently allocated to sibling teams within that department. The $\min$ takes the binding constraint: a team with a 500-GPU quota whose department siblings are already using 450 of the department's 600-GPU limit can access at most 150 GPUs regardless of its own quota. This ceiling is the mechanism that makes departmental limits enforceable without requiring each team to manually track its siblings' consumption.

---

### 4. `fig-utilization-paradox` ⚠️ (figure 🟠) — def L2232

**Verbatim ref sentence (L2230):**
> "As @fig-utilization-paradox illustrates, the relationship between utilization and wait time is highly nonlinear: operating above 80 percent causes wait times to explode."

**Missing move:** The citation names the nonlinear relationship but does not explain the mechanism (queueing theory: at high utilization, large jobs cannot find contiguous blocks, and small-job wait times grow because they cannot preempt), the Danger Zone boundary (why 80 percent specifically), or the safe target (why 75 percent). The figure shows three distinct curves for small, mixed, and large jobs, but the prose never states why large jobs degrade first or what the practical implication is (target 75 percent utilization, not 100 percent, to keep queue times predictable). The payoff paragraph (L2422) discusses the debugging method and does not return to the figure's lesson. The takeaway that 100 percent utilization is an anti-pattern and the rationale for the 75 percent target live only in the caption.

**Takeaway currently lives:** In the caption ("illustrating why 100 percent utilization is an anti-pattern") and the alt-text; absent from body prose.

**Rule-compliant rewrite** — extend the citation sentence (L2230) to carry the payoff:

> As @fig-utilization-paradox illustrates, the relationship between utilization and wait time is highly nonlinear: operating above 80 percent causes wait times to explode, particularly for large jobs that cannot be placed until a contiguous block of GPUs opens. The mechanism mirrors M/M/1 queueing theory — as the server approaches saturation, mean queue depth grows without bound. For small jobs the effect is mild; for large gang-scheduled jobs it is severe, because the probability of a full contiguous allocation drops sharply near 100 percent utilization. The practical target is 75 percent cluster utilization: below that threshold, large jobs find placements within minutes; above it, the queue for any large job can stretch to hours or days regardless of priority. Running a cluster at 95 percent utilization in pursuit of hardware efficiency actually reduces total throughput by starving the large, high-value jobs.

---

### 5. `lst-fleet-orchestration-k8s-gpu` ⚠️ (listing 🟡) — def L660

**Verbatim ref sentence (L658):**
> "@Lst-fleet-orchestration-k8s-gpu shows the resulting pod fragment."

**Missing move:** The citation sets up the mechanism (device plugins expose GPUs as extended resources) but then reduces the listing to "the resulting pod fragment," which is a float-announcer pointer without naming what the reader should notice in the code. The standard requires prose to state the mechanism the listing embodies and the design choice that matters. In this listing, the design choice is the `nvidia.com/gpu` resource key under `limits` (not `requests`): GPU resources in Kubernetes use `limits`-only semantics, which means any pod that requests a GPU gets exclusive access to the whole device. The payoff (L674) pivots immediately to the binary allocation inefficiency problem without anchoring it to a specific element of the listing.

**Takeaway currently lives:** In the caption ("The `nvidia.com/gpu` resource name follows Kubernetes extended resource conventions"); absent from body prose.

**Rule-compliant rewrite** — replace the citation sentence at L658 with:

> @Lst-fleet-orchestration-k8s-gpu shows the declarative result: a pod's `resources.limits` block requests GPUs by the `nvidia.com/gpu` key registered by the NVIDIA device plugin. The critical design choice is that GPU resources appear only under `limits`, not `requests`: Kubernetes treats any resource that appears only in `limits` as requiring exclusive device-level allocation, so the pod receives the whole GPU or nothing. This all-or-nothing binding is what makes integer GPU counts the natural unit of scheduling and what forces the binary allocation problem described next.

---

### 6. `tbl-fleet-orchestration-slurm-partitions` ⚠️ (table 🟠) — def L614

**Verbatim ref sentence (L616):**
> "As @tbl-fleet-orchestration-slurm-partitions shows, GPU allocation strategies significantly impact utilization, and Slurm provides several mechanisms for controlling GPU placement."

**Missing move:** The citation pivots immediately to GPU allocation flags (`--gres`, `--gpus`, `--gpus-per-node`) without stating what the table's partition structure demonstrates. The table's load-bearing point is the interconnect-driven partition boundary: NVLink partitions exist for tensor parallelism because the all-reduce at every transformer layer requires NVLink bandwidth, while PCIe partitions serve data parallelism where less-frequent gradient syncs tolerate lower bandwidth. That decision logic is in the caption and nowhere in the body prose. A reader who skips the table and reads only the citation paragraph learns about Slurm flags but not about why partitions are organized this way.

**Takeaway currently lives:** In the caption ("NVLink-connected partitions support tensor parallelism, while PCIe partitions serve workloads that rely primarily on data parallelism"); absent from body prose.

**Rule-compliant rewrite** — replace the citation sentence at L616 with:

> @tbl-fleet-orchestration-slurm-partitions shows how interconnect determines partition structure. The dgx-a100 partition reserves NVLink-connected nodes exclusively for tensor-parallel training because tensor parallelism exchanges activations at every transformer layer, a pattern that saturates PCIe but fits within NVLink's 600 GB/s aggregate bandwidth. The a100-pcie and inference partitions serve workloads whose communication patterns tolerate PCIe, and the debug partition isolates single-GPU development from production queues. GPU allocation strategies within each partition significantly impact utilization, and Slurm provides several mechanisms for controlling GPU placement.

---

### 7. `tbl-fleet-orchestration-paradigm-comparison` ⚠️ (table 🟠) — def L755

**Verbatim ref sentence (L757):**
> "As @tbl-fleet-orchestration-paradigm-comparison illustrates, the choice also reflects organizational trajectory."

**Missing move:** The citation picks up one secondary point (organizational trajectory) while the table's primary load is a multi-dimensional contrast between Slurm and Kubernetes across scheduling model, gang scheduling, preemption, autoscaling, fair-share, container overhead, service management, and ecosystem. The standard requires that the prose deliver the load-bearing contrast or the specific rows that matter. The key discriminating finding — Slurm provides native gang scheduling (essential for distributed ML training) while Kubernetes requires extensions, and Kubernetes provides native autoscaling (essential for inference serving) while Slurm has none — is entirely absent from body prose. The reader must read all cells to understand the decision.

**Takeaway currently lives:** Distributed across table cells; absent from body prose.

**Rule-compliant rewrite** — replace the citation sentence at L757 with:

> @tbl-fleet-orchestration-paradigm-comparison surfaces the discriminating rows. Slurm provides native gang scheduling, mature fair-share with configurable decay, and bare-metal execution with no container overhead, which is why it dominates dedicated training clusters where distributed synchronous jobs need atomic allocation guarantees. Kubernetes provides native autoscaling, rolling deployment updates, and a cloud-native service model, which is why it dominates inference serving where replica counts must track live traffic. No single paradigm is strong on both axes. The choice therefore reflects organizational trajectory as much as technical preference: teams starting from HPC infrastructure add Kubernetes for serving, while teams starting from cloud-native infrastructure add HPC extensions (Volcano, Kueue) for training.

---

### 8. `tbl-fleet-orchestration-autoscaling-metrics` ⚠️ (table 🟠) — def L1678

**Verbatim ref sentence (L1680):**
> "Beyond the HPA-oriented metrics in @tbl-fleet-orchestration-autoscaling-metrics, **Vertical Pod Autoscaling (VPA)** operates on a different axis..."

**Missing move:** The citation uses the table only as a transition pivot to VPA and never states what the table's metrics demonstrate. The lead-in (L1669) explains why CPU utilization fails for GPU inference, but the body prose never identifies which metric in the table should be used as the primary leading indicator (request queue depth predicts latency degradation before it manifests in P99), or why P99 latency is reactive rather than predictive. The key decision the table drives (use queue depth or pending tokens as the trigger metric, not CPU or even GPU utilization) lives only in the caption's summary.

**Takeaway currently lives:** In the caption ("Queue depth provides a leading indicator of latency degradation, while pending tokens captures outstanding autoregressive decode work"); absent from body prose.

**Rule-compliant rewrite** — replace the citation sentence at L1680 with:

> @tbl-fleet-orchestration-autoscaling-metrics shows that not all metrics are equal as scaling triggers. Request queue depth is the preferred leading indicator: it rises before latency degrades, giving the autoscaler time to provision new replicas during the 60 to 120 second model cold-start window. P99 latency is a reactive metric that lags demand changes by seconds to minutes and is therefore too slow to drive proactive scale-out. Pending tokens serves a complementary role for autoregressive LLM serving: it captures the outstanding decode work across all active requests, which determines GPU occupancy independently of new arrival rate. Beyond these HPA-oriented metrics, **Vertical Pod Autoscaling (VPA)** operates on a different axis, adjusting resource requests and limits for individual pods.

---

*End of findings.*
