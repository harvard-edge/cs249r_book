# Float Exposition Worklist — `edge_intelligence.qmd` (vol2)

Graded against the Float Exposition Standard. Caption, fig-alt, in-figure labels, code comments,
and callout interiors do not count toward the prose's job. Only running body prose is judged.

---

## Summary Table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|:-----|:------|-------:|---:|---:|---:|
| Figure | 🟠 High | 17 | 10 | 6 | 1 |
| Listing | 🟡 Medium | 4 | 4 | 0 | 0 |
| Table | 🟠 High | 7 | 3 | 4 | 0 |
| **Total** | | **28** | **17** | **10** | **1** |

---

## Findings (⚠️ and 🛑 only)

---

### F01 — `fig-ondevice-gboard` (Figure 🟠) — def L460

**Ref sentence (L458):**
> @Fig-ondevice-gboard demonstrates how different prediction strategies enable local adaptation in real time. Next-word prediction suggests likely continuations based on prior text, while Smart Compose uses on-the-fly rescoring to offer dynamic completions. These techniques demonstrate the sophistication of local inference mechanisms.

**Missing move:** The Interpret move is thin. The prose names the two mechanisms but the final sentence restates sophistication without stating the design implication: why does on-the-fly rescoring matter for local adaptation? The payoff paragraph pivots immediately to wearables, so no lead-out arrives. The takeaway (that these mechanisms achieve personalization without server round-trips, making them models for any on-device adaptation pipeline) lives only in the caption.

**Where takeaway lives:** Caption ("enhancing personalization and preserving privacy ... without transmitting data to a central server").

**Rule-compliant rewrite of the citing paragraph:**

> @Fig-ondevice-gboard contrasts two on-device prediction strategies that achieve personalization without a server round-trip. Next-word prediction refines continuations from local typing history, while Smart Compose's on-the-fly rescoring adapts completions to the user's evolving linguistic patterns in real time. Both strategies expose the core design principle for on-device adaptation: the model state must be updatable on the device itself, so latency and privacy constraints eliminate the option of round-tripping raw keystrokes to a central server for scoring.

---

### F02 — `fig-centralized-vs-decentralized` (Figure 🟠) — def L504

**Ref sentence (L502):**
> @Fig-centralized-vs-decentralized traces this evolution from centralized training through local adaptation to federated coordination. Each phase increases coordination complexity while enabling capabilities impossible in purely centralized deployments.

**Missing move:** The second sentence is the intended takeaway but is too vague to constitute one. "Capabilities impossible in purely centralized deployments" does not name those capabilities. The prose does not state what the figure's comparison reveals about the trade-off axis (privacy vs. coordination cost). The payoff paragraph pivots to design constraints without returning to the figure. The specific trade-off named in the caption (privacy gain vs. coordination overhead) lives only there.

**Where takeaway lives:** Caption ("Each phase trades coordination cost for privacy and collaboration benefits").

**Rule-compliant rewrite of the citing paragraph:**

> @Fig-centralized-vs-decentralized traces this architectural progression across three phases. Moving from local-only to centralized cloud to federated coordination, each step adds coordination cost in exchange for a specific capability: centralized cloud enables fleet-wide model improvement at the cost of raw data exposure, and federated coordination preserves data locality while retaining that fleet-wide benefit. The figure makes the key trade-off concrete: the privacy gain of federated learning does not come free but is paid for with per-round communication overhead and aggregation latency.

---

### F03 — `fig-ondevice-pretraining` (Figure 🟠) — def L652

**Ref sentence (L650):**
> @Fig-ondevice-pretraining illustrates how the complete training pipeline combines offline pretraining with online adaptive learning on resource-constrained IoT devices. The system first undergoes meta-training with generic data. During deployment, device-specific constraints such as data availability, compute, and memory shape the adaptation strategy by ranking and selecting layers and channels to update. This selective fine-tuning allows efficient on-device learning within limited resource envelopes.

**Missing move:** The prose walks through the pipeline stages (good lead-in) but the final sentence is a circular restatement rather than a lead-out: "allows efficient on-device learning within limited resource envelopes" rephrases the headline without stating why this two-stage structure is the right design choice or what goes wrong without it. The payoff paragraph pivots to model sizing constraints, not back to this figure.

**Where takeaway lives:** Partially in the caption ("only the selected layers are fine-tuned on-device, keeping the rest frozen"), but the consequence of that choice (frozen backbone preserves cloud-learned representations while local adaptation stays within the device memory budget) is stated nowhere.

**Rule-compliant rewrite of the last sentence of the citing paragraph:**

Replace: "This selective fine-tuning allows efficient on-device learning within limited resource envelopes."

With: "Freezing the non-selected layers is not merely an efficiency shortcut: it preserves the generic representations that make the pretrained backbone useful, ensuring that the on-device update budget is spent only on the parameters most sensitive to local conditions."

---

### F04 — `fig-federated-averaging-cycle` (Figure 🟠) — def L2527

**Ref sentence (L2525):**
> @Fig-federated-averaging-cycle breaks down the Federated Averaging protocol into four phases: client selection (with over-selection to handle stragglers), local training on private data, parameter upload with optional compression, and weighted aggregation that produces the updated global model.

**Missing move:** The prose enumerates the four phases correctly but delivers no Interpret move. What does this cycle tell the reader? The payoff paragraph asks a follow-on design question (local epochs, client selection) but does not state the cycle's architectural implication. A reader who skips the figure gains nothing beyond a list of steps.

**Where takeaway lives:** Caption describes the cycle without a systems-level conclusion. The insight (that the protocol's round structure makes straggler latency, not average latency, the binding constraint on convergence speed) is absent from both prose and caption.

**Rule-compliant rewrite of the citing paragraph:**

> This cyclical coordination protocol is the unit of work in federated learning, and its structure determines which system parameters govern convergence speed. @Fig-federated-averaging-cycle breaks down the Federated Averaging protocol into four phases: client selection with over-selection to absorb stragglers, local training on private data, compressed parameter upload, and weighted aggregation. The architectural consequence is that a round's wall-clock duration is set by the slowest client the server is still waiting on, not by the median client, which is why over-selection (recruiting more clients than strictly needed) is a scheduling necessity rather than a redundancy measure.

---

### F05 — `fig-fl-communication-computation` (Figure 🟠) — def L2942 🛑

**Ref sentence (L2940):**
> While @fig-fl-communication-computation illustrates the fundamental tradeoff between local computation and network bandwidth, other communication-efficient updates introduce their own tradeoffs.

**Missing move:** Pivot-away pattern. The prose uses the figure as a pivot point to a different discussion (compression tradeoffs) rather than delivering the figure's own takeaway. The word "While" signals the pivot explicitly. The figure's specific finding, that the optimal number of local epochs shifts rightward as bandwidth decreases but that excessive local computation eventually increases total time due to drift, is stated only in the caption. No payoff paragraph returns to this figure's content.

**Where takeaway lives:** Caption only.

**Rule-compliant rewrite replacing the citing sentence:**

> @Fig-fl-communication-computation makes the bandwidth-epoch interaction concrete: as network bandwidth decreases from fast to slow, the optimal number of local epochs shifts rightward because amortizing a larger per-round communication cost requires more local work per round. However, the curves turn upward past the optimal point in every bandwidth regime because excessive local computation drives client drift, forcing more global rounds to recover convergence. The practical implication is that a system's bandwidth tier, not its compute tier, should set the default local epoch count.

---

### F06 — `fig-odl-design-flow` (Figure 🟠) — def L3319

**Ref sentence (L3317):**
> Consider @fig-odl-design-flow for a systematic decision framework: the flowchart guides practitioners through key decision points about adaptation complexity, compute availability, and data sharing requirements, mapping these choices to concrete implementation strategies from bias-only updates to full federated learning with privacy measures.

**Missing move:** The Interpret move is missing. The prose describes the flowchart's structure but does not state the conclusion a practitioner should draw from traversing it. What is the decisive branch? What does the flowchart's shape reveal about the design space? The payoff paragraph pivots to operational concerns.

**Where takeaway lives:** Caption (describes the three decision axes but not the key insight).

**Rule-compliant rewrite of the citing sentence:**

> @Fig-odl-design-flow maps these constraints to concrete technique choices: adaptation complexity is the first branch because bias-only updates are feasible on any device, while adapter-based and full fine-tuning require progressively larger memory and compute budgets. The flowchart reveals that the data-sharing decision is downstream of the hardware decision, not independent of it: only devices that can afford adapter-class or full fine-tuning have enough update signal to make federated coordination worth its communication overhead.

---

### F07 — `fig-shadow-validation` (Figure 🟠) — def L3376

**Ref sentence (L3374):**
> The shadow validation approach operates as a continuous comparison pipeline running on each device (@fig-shadow-validation), where incoming data flows through both a frozen baseline and the locally adapted model before an arbiter decides whether to accept or roll back adaptations.

**Missing move:** The prose describes the mechanism but not the consequence. What does shadow validation make possible that would otherwise be missing? The key implication (that it provides drift detection without labeled ground truth, which is the condition that makes unsupervised monitoring at the edge feasible) is not stated. The payoff paragraph pivots to general MLOps without returning to this figure.

**Where takeaway lives:** Caption ("To detect model drift without labels ... the system detects personalization drift and can trigger an automatic rollback"). The critical phrase "without labels" is the insight that belongs in the citing prose.

**Rule-compliant rewrite of the citing sentence:**

> Because edge devices rarely have labeled ground truth to evaluate a locally adapted model against, the shadow validation approach provides a label-free drift signal by running a frozen baseline model in parallel with the adapted model (@fig-shadow-validation). Incoming data flows through both paths, and the arbiter flags personalization drift when the adapted model's confidence consistently falls below the baseline, triggering rollback without requiring any external feedback loop. The architectural consequence is that validation becomes self-contained on the device, removing the dependency on a centralized label oracle.

---

### F08 — `fig-tiered-adaptation` (Figure 🟠) — def L3443

**Primary ref sentence (L3441):**
> A tiered adaptation strategy (@fig-tiered-adaptation) maps these techniques to device capabilities.

**Missing move:** The primary citation is a bare parenthetical; no prose derives a takeaway. The L3487 reference ("maps each technique to device-class capability") is slightly fuller but still a pointer. The payoff (L3487 paragraph) simply presents checkpoint questions. The key insight from the tiered structure, that the data-sharing decision is only available to the upper two tiers, making federated learning inaccessible to the most-constrained devices, is not stated anywhere in the prose.

**Where takeaway lives:** Caption (describes the three-tier split by device capability but not the policy implication).

**Rule-compliant rewrite (replacing L3441's bare parenthetical with a fuller sentence):**

Replace: "A tiered adaptation strategy (@fig-tiered-adaptation) maps these techniques to device capabilities."

With: "A tiered adaptation strategy, shown in @fig-tiered-adaptation, reveals the non-obvious constraint: the data-sharing decision is only reachable by devices that can afford adapter-class or full fine-tuning compute budgets, which means the most resource-constrained devices in a fleet are structurally excluded from federated coordination and must rely entirely on local bias-only updates."

---

### T01 — `tbl-adaptation-strategies` (Table 🟠) — def L2006

**Ref sentence (L1998):**
> @Tbl-adaptation-strategies contrasts adaptation strategy trade-offs across trainable parameters, memory overhead, expressivity, use-case suitability, and system requirements, revealing how the optimal choice depends on application domain, available hardware, latency constraints, and expected distribution shift.

**Missing move:** The prose names the table's dimensions but does not name the load-bearing contrast or the specific row that matters for the typical deployment scenario. "Optimal choice depends on application domain, available hardware..." restates the column headers rather than drawing the conclusion the table encodes. A reader who skips the table learns only that trade-offs exist.

**Where takeaway lives:** Cells (the contrast between bias-only at "extreme memory/compute limits" vs. sparse layer updates requiring "profiling or meta-training" is the decision boundary).

**Rule-compliant rewrite of the citing sentence:**

> @Tbl-adaptation-strategies contrasts the three main adaptation strategies across memory overhead, expressivity, and deployment requirements. The key boundary is between bias-only updates and residual adapters: bias-only requires no runtime infrastructure and almost no memory overhead but provides low expressivity, making it the only viable option when total on-device memory for training falls below a few megabytes. Residual adapters cross both the expressivity and infrastructure threshold simultaneously, making them the practical default for mobile-class SoCs where runtime support is available. Sparse layer updates offer the highest expressivity but require offline profiling or meta-training, a cost that only systems with controlled pretraining pipelines can absorb.

---

### T02 — `tbl-ondevice-techniques` (Table 🟠) — def L2165

**Ref sentence (L2157):**
> @Tbl-ondevice-techniques summarizes the on-device learning trade-offs across data requirements, memory overhead, and use case suitability for each technique.

**Missing move:** Classic "summarizes" pointer. No conclusion. The table's key contrast, that few-shot adaptation minimizes memory overhead but requires labeled shots while compressed representations are label-free but require a well-aligned encoder, is stated nowhere in the prose. The payoff paragraph leads with "not mutually exclusive" and then gives a combination example, but without first stating why any individual method is insufficient alone.

**Where takeaway lives:** Cells.

**Rule-compliant rewrite of the citing sentence:**

> @Tbl-ondevice-techniques reveals a structural gap in each individual technique: few-shot adaptation is the lightest path but requires labeled support examples that may be unavailable; experience replay achieves continual stability but demands ongoing memory and compute for buffer updates; and compressed representations minimize overhead and work without labels but degrade when the fixed encoder does not capture deployment-specific variability. No single technique addresses all three constraint axes simultaneously, which is the direct motivation for combining them.

---

### T03 — `tbl-personalization-strategies` (Table 🟠) — def L3112

**Ref sentence (L3103):**
> Examine @tbl-personalization-strategies to see how compute overhead, privacy guarantees, and adaptation latency vary across local finetuning, personalization layers, clustered federated learning, and meta-learning approaches.

**Missing move:** Imperative pointer without a stated conclusion. "Examine to see" shifts the interpretive work to the reader. The key result encoded in the table is that meta-learning dominates on adaptation speed and privacy but carries the highest compute overhead, making it viable only for devices that can absorb a meta-training phase. That decision implication lives only in the cells.

**Where takeaway lives:** Cells (meta-learning row: "Very Fast (few-shot)" speed with "High (meta-objective)" compute overhead).

**Rule-compliant rewrite of the citing sentence:**

> @Tbl-personalization-strategies maps these four strategies along compute overhead, privacy preservation, and adaptation speed. The dominant tension is between meta-learning and local finetuning: meta-learning achieves the fastest deployment-time adaptation (a few-shot gradient steps) and preserves the strongest privacy guarantees, but it requires an expensive meta-objective during pretraining that is available only to systems with controlled cloud infrastructure. Local finetuning and personalization layers are the practical defaults for systems that cannot afford that offline cost, with the choice between them turning on whether a shared feature extractor is stable enough to freeze.

---

### T04 — `tbl-ondevice-challenges` (Table 🟠) — def L3676

**Ref sentence (L3666):**
> @Tbl-ondevice-challenges synthesizes these interconnected issues, mapping each challenge category to its root cause and system-level implications for on-device learning deployments.

**Missing move:** "Synthesizes" and "mapping" describe what the table contains rather than what the reader should conclude. The payoff paragraph (L3680) introduces the consequences of these challenges but does not refer back to the table. The key row-level insight, that limited observability and feedback is the challenge with the longest tail because it is unaddressable by compression or scheduling alone and requires architectural choices made before deployment, is absent from the prose.

**Where takeaway lives:** Cells (limited observability row: "No centralized testing or logging" causing "Makes update validation and debugging difficult").

**Rule-compliant rewrite of the citing sentence:**

> @Tbl-ondevice-challenges maps five challenge categories to their root causes and system-level consequences. The most architecturally binding is limited observability: whereas system heterogeneity and resource contention can be addressed through capability-tiered deployment, the absence of centralized testing and logging cannot be patched at runtime. It must be designed for upfront, through shadow validation, on-device metrics, and rollback policies, because once a model is adapting in the field, no post-hoc instrumentation can substitute for the monitoring hooks that were never built in.
