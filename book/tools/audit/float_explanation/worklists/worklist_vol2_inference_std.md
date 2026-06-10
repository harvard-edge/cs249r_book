# Float Exposition Audit — `vol2/inference/inference.qmd`

Graded against FLOAT_EXPOSITION_STANDARD.md. Caption, alt-text, in-figure labels, code comments,
and callout interiors do NOT count toward the prose's job — only running body prose.

---

## Summary table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|------|-------|--------|----|----|-----|
| Algorithm | 🔴 strict | 2 | 2 | 0 | 0 |
| Equation | 🔴 strictest | 15 | 9 | 5 | 1 |
| Figure | 🟠 high | 26 | 20 | 5 | 1 |
| Listing | 🟡 medium | 8 | 4 | 4 | 0 |
| Table | 🟠 high | 64 | 51 | 13 | 0 |
| **Total** | | **115** | **86** | **27** | **2** |

**Dominant finding type:** Tables (13 ⚠️) — most are bare-pointer citations inside callout frames
that deliver the takeaway in a "Systems insight" label rather than running body prose.

---

## Findings (⚠️ and 🛑 only)

---

### EQUATIONS

---

**`eq-serving-tax`** (eq 🔴) — def L201

Ref sentence (L199): *"The total serving tax often consumes 10–30 percent of the latency budget in
distributed systems, as @eq-serving-tax shows:"*

Missing move: symbols never individually named in running prose; the five terms of the equation are
not given meaning in the body (only the inline "where" clause is absent — there is no where clause
at all). The payoff paragraph (L203) jumps to mitigation without stating what the equation says.
The takeaway (which term dominates and why it matters) lives nowhere in body prose.

Rule-compliant rewrite:

> The total serving tax often consumes 10–30 percent of the latency budget in distributed systems.
> The five components decompose as $T_{\text{total}} = T_{\text{compute}} + T_{\text{network}} +
> T_{\text{serialization}} + T_{\text{coordination}} + T_{\text{queuing}}$ (@eq-serving-tax), where
> $T_{\text{compute}}$ is the irreducible GPU execution cost and the remaining four terms are the
> infrastructure overhead the system adds around it. In practice, $T_{\text{queuing}}$ and
> $T_{\text{coordination}}$ are the two most controllable terms: queuing vanishes under light load
> and explodes under saturation, while coordination cost scales with the number of fan-out
> services rather than with model size.

---

**`eq-dynamic-batch-latency`** (eq 🔴) — def L918

Ref sentence (L916): *"...The expected latency under Poisson arrivals with rate $\lambda_{\text{arr}}$
follows @eq-dynamic-batch-latency:"*

Missing move: the "where" clause (L920) names the three terms, but the prose never states the
consequence or regime this equation implies — which term dominates, under what arrival rate the
window fires first vs. the batch-full condition, and what engineers should conclude. Payoff at L924
skips straight into a numerical example without closing the equation's argument.

Rule-compliant rewrite (add between L920 and L922):

> Under light load, $T_{\text{batch}}$ dominates because windows fill by timeout rather than by
> request count; increasing $T_{\text{window}}$ raises average latency without improving utilization.
> Under heavy load, $T_{\text{queue}}$ and $T_{\text{batch}}$ shrink as windows fill quickly, so
> the throughput-to-latency trade-off becomes favorable. The worked example below shows how to
> choose $T_{\text{window}}$ and $B_{\text{max}}$ for a given SLO.

---

**`eq-waste-ratio`** (eq 🔴) — def L1096

Ref sentence (L1259): *"Waste calculation using @eq-waste-ratio:"*

Rating: 🛑 FAILS. The citation is a bare pointer inside a worked-example callout, preceded and
followed entirely by bullet arithmetic. The equation's interpretation ("waste depends entirely on
the ratio of mean to maximum output length"; present at L1098 in body prose before the worked
example) is correct, but the actual cite at L1259 has zero body-prose interpretation around it.
The prose at L1098 explains the formula but this cite instance is a callout-interior pointer only.
Because the standard requires that the cite instance carry the interpret move, this specific
reference fails — the payoff is in a different location from the citation.

Rule-compliant rewrite (replace L1259):

> Traditional batching discards compute on all shorter sequences while waiting for the longest
> one to finish. Applying @eq-waste-ratio with the values above gives $W = 1 - 125/200 = 37.5$
> percent: more than a third of every iteration's GPU work produces no output token.

---

**`eq-tensor-parallel-time`** (eq 🔴) — def L3002

Ref sentence (L3000): *"The inference time with tensor parallelism follows @eq-tensor-parallel-time:"*

Missing move: symbols $t$ and $T_{\text{allreduce}}$ are not named in prose. No "where" clause
follows the equation. The payoff paragraph (L3073) jumps to a numerical example; the equation
itself is never interpreted. The key consequence — that inference time decreases with $t$ but the
AllReduce term grows, making there a sweet-spot degree of tensor parallelism — is absent.

Rule-compliant rewrite:

> The inference time with tensor parallelism follows @eq-tensor-parallel-time, where $t$ is the
> tensor-parallel degree and $T_{\text{allreduce}}(A/t)$ is the AllReduce cost over an activation
> shard of size $A/t$. Compute time scales as $1/t$, but the AllReduce term grows with $t$ because
> ring latency increases. The minimum of this sum identifies the optimal $t$ for a given
> interconnect bandwidth; beyond that point, adding more devices increases rather than decreases
> latency.

---

**`eq-allreduce-time`** (eq 🔴) — def L3651

Ref sentence (L3649): *"@Eq-allreduce-time quantifies AllReduce communication time for tensor
parallelism, where data is combined from all devices with the result available on all devices."*

Missing move: the "where" clause for $\alpha$, $\beta$, $M$, and $N$ is deferred to L3653 (payoff
paragraph after the equation), which is acceptable placement, but the prose never delivers the
consequence — which term dominates for intra-node vs. inter-node and what the practical upper bound
on $N$ is before $T_{\text{allreduce}}$ exceeds the layer compute budget. The takeaway lives only
in the combined payoff for both AllReduce and AllToAll at L3663.

Rule-compliant rewrite (add after L3653):

> For intra-node NVLink at 900 GB/s, the bandwidth term $2(N-1)/N \cdot M/\beta$ is negligible
> for typical activation payloads, so the startup latency $\alpha$ dominates; crossing to inter-node
> InfiniBand cuts $\beta$ by 36$\times$, making the bandwidth term the binding constraint and
> setting a practical ceiling on the tensor-parallel degree.

---

**`eq-p2p-time`** (eq 🔴) — def L3657

Ref sentence (L3655): *"@Eq-p2p-time expresses the simpler point-to-point communication for
pipeline parallelism, where data flows from one device to the next."*

Missing move: symbols $n$, $\alpha$, and $\beta$ are named in the payoff at L3659, but the
consequence of this equation for pipeline serving — that for large activations $n/\beta$ dominates
and pipeline depth is bounded by per-stage latency — is never stated. The payoff (L3663) addresses
both P2P and AllToAll together rather than this equation specifically.

Rule-compliant rewrite:

> @Eq-p2p-time expresses the point-to-point communication for pipeline parallelism, where $n$ is
> the activation payload size, $\alpha$ is the network startup latency, and $n/\beta$ is the
> transfer time. For transformer activations that are small relative to weights, $\alpha$ dominates,
> meaning pipeline parallelism is sensitive to latency rather than bandwidth — the opposite of
> tensor parallelism's AllReduce.

---

### FIGURES

---

**`fig-serving-hierarchy`** (fig 🟠) — def L589

Ref sentence (L587): *"A related deployment stack appears in @fig-serving-hierarchy, showing how
requests pass through edge, routing, and model-serving infrastructure in production."*

Missing move: the preceding paragraph explains the four-level abstraction in detail; the figure
citation is a pivot-away pointer ("A related deployment stack appears…"). The prose does not say
what the figure demonstrates — what the reader should notice about the cumulative latency budget,
the three-tier structure, or the SLA boundaries. The takeaway lives in the caption only.

Rule-compliant rewrite:

> @Fig-serving-hierarchy maps this three-tier deployment stack and shows how each tier imposes a
> cumulative latency budget: the CDN edge layer absorbs static responses in under 10 ms, the
> gateway layer adds routing and auth overhead within 50 ms, and the GPU cluster handles dynamic
> inference within 2 s. The figure makes the latency hierarchy physical — optimization at a lower
> tier cannot recover budget consumed by a higher tier.

---

**`fig-feature-parallel-pipeline`** (fig 🟠) — def L1547

Ref sentence (L1545): *"@Fig-feature-parallel-pipeline shows how the request is split into
feature-specific paths before the dense ranking head recombines the retrieved representations."*

Missing move: the cite names what the figure contains but does not deliver the relationship or
mechanism the figure demonstrates — why parallel feature paths are faster than sequential lookup,
and how decoupling embedding storage from dense compute enables trillion-parameter scaling. The
payoff paragraph (L1551) lists four computation stages but does not connect back to the figure's
story.

Rule-compliant rewrite:

> @Fig-feature-parallel-pipeline shows the feature-parallel dispatch pattern: embedding lookups
> for user, item, and context features fire simultaneously to separate shards, so the end-to-end
> embedding latency equals the slowest shard rather than the sum of all three. Decoupling these
> sparse paths from the dense ranking head means each can scale independently — a critical property
> when embedding tables reach terabytes and the ranking head stays relatively small.

---

**`fig-paged-attention`** (fig 🟠) — def L2167

Ref sentence (L2165): *"@Fig-paged-attention illustrates the key concepts including page tables
that map logical sequence positions to physical memory pages, block size that defines the number
of tokens per page (typically 16 tokens), and physical blocks that provide fixed-size memory
allocations assignable to any sequence."*

Missing move: the cite narrates the diagram's components (effectively captioning it in prose) but
does not deliver the mechanism — what the block table indirection achieves for fragmentation. The
payoff paragraph (L2171) lists benefits but in a bullet list inside the body that does not connect
the visual to the argument. The figure's demonstration (that scattered physical blocks serve
contiguous logical sequences) is not stated in body prose.

Rule-compliant rewrite:

> @Fig-paged-attention shows how the block table breaks the requirement for contiguous allocation:
> a sequence's logical pages map through an indirection table to any available physical block,
> so gaps left by completed sequences are immediately available to new requests regardless of
> size. The result is that GPU HBM can be packed to near 100 percent utilization — something
> impossible when every sequence must claim a contiguous region sized for its maximum length.

---

**`fig-prefix-caching`** (fig 🟠) — def L2234

Ref sentence (L2232): *"Prefix caching shares KV cache entries across requests with common prefixes.
@Fig-prefix-caching demonstrates how shared system prompts avoid redundant computation."*

Missing move: the citation describes what the figure demonstrates but not what the reader should
conclude — how many physical blocks are saved, or the mechanism (block table pointing to shared
physical blocks). The takeaway is entirely in the caption and in the payoff (L2300), which pivots
to copy-on-write semantics without connecting back to the figure.

Rule-compliant rewrite:

> Prefix caching shares KV cache entries across requests with common prefixes by letting multiple
> block tables point to the same physical pages. @Fig-prefix-caching makes the savings concrete:
> both Request A and Request B map their first six logical pages to the system-prompt blocks,
> so only the unique suffix tokens require new physical allocation. The shared pages are read-only
> until a request diverges, at which point copy-on-write semantics create a private copy.

---

**`fig-cold-start-breakdown`** (fig 🟠) — def L4666

Ref sentence (L4662): *"@Fig-cold-start-breakdown visualizes the cumulative timeline:"*

Rating: 🛑 FAILS. Pure float-announcer pointer with zero body-prose context. The preceding
"Systems insight" callout (L4660) states the conclusion but that is callout interior, not running
body prose. There is no lead-in paragraph before the citation that would satisfy the interpret
move; the payoff (L4705) begins a new section. The figure's demonstration — which phase consumes
the most time, and therefore where optimization yields the most — lives only in the caption.

Rule-compliant rewrite:

> @Fig-cold-start-breakdown traces the cumulative seven-minute timeline and reveals that model
> download from remote object storage (roughly five minutes) consumes nearly three quarters of the
> total cold-start budget — a phase that warm pools and local SSD caching can eliminate entirely
> for pre-staged replicas.

---

**`fig-predictive-reactive-scaling`** (fig 🟠) — def L4859

Ref sentence (L4857): *"As @fig-predictive-reactive-scaling shows, proactive provisioning avoids
the SLO violations that reactive-only systems suffer during ramp-up periods."*

Missing move: the figure's story (the gap between the reactive and predictive capacity curves during
the ramp-up period, and the SLO-violation zone) is named but not interpreted. The prose says
"proactive provisioning avoids SLO violations" — which the caption also says verbatim. The
"why" (reactive capacity chases a signal that arrives after the spike, so it always lags by the
cold-start time) is not stated in body prose near this citation.

Rule-compliant rewrite:

> As @fig-predictive-reactive-scaling shows, reactive capacity always lags the traffic spike by
> exactly the cold-start duration: because the autoscaler cannot fire before it observes pressure,
> the fleet is under-provisioned during the ramp precisely when user demand peaks. Predictive
> capacity eliminates that gap by initiating provisioning before the spike, ensuring replicas are
> warm when traffic arrives.

---

### LISTINGS

---

**`lst-resource-quotas`** (lst 🟡) — def L4433

Ref sentence (L4431): *"Hard quotas enforce strict limits, as shown in @lst-resource-quotas."*

Missing move: bare pointer. The preceding sentence (L4429) names what quotas do; the citation adds
nothing about the mechanism — what fields the listing configures, what the reader should look at.
The payoff (L4459) discusses soft quotas, a different topic.

Rule-compliant rewrite:

> Hard quotas enforce strict limits; @lst-resource-quotas shows the three axes they constrain:
> request concurrency per tenant, KV cache memory allocation, and token generation rate. The
> concurrency cap is the most directly load-bearing — once a tenant fills its concurrent-request
> slots, new arrivals are rejected rather than queued, which bounds the memory pressure the tenant
> can impose.

---

**`lst-metric-based-scaling`** (lst 🟡) — def L4709

Ref sentence (L4707): *"@Lst-metric-based-scaling shows a typical metric-based scaling
configuration."*

Missing move: bare pointer. No framing of what the reader should notice in the YAML — the
`cooldown_period` field and its role in preventing oscillation, or why `70` percent target differs
from the `80` scale-up threshold.

Rule-compliant rewrite:

> @Lst-metric-based-scaling shows a representative configuration: the `target_value` of 70 percent
> sits below the `scale_up_threshold` of 80 percent, creating a buffer that prevents the scaler
> from firing on transient load spikes. The `cooldown_period` of 300 seconds is the key guard
> against oscillation — without it, rapid scale-up followed by immediate scale-down wastes
> provisioning budget and introduces latency variance.

---

**`lst-event-driven-scaling`** (lst 🟡) — def L4802

Ref sentence (L4800): *"Event-driven scaling scales proactively for known events, as shown in
@lst-event-driven-scaling."*

Missing move: bare pointer with no framing of what to notice in the YAML — the `ramp_up: 30min`
lead time that compensates for cold-start latency, or the cron-based schedule for predictable
nightly events.

Rule-compliant rewrite:

> Event-driven scaling provisions capacity before demand arrives rather than after; @lst-event-driven-scaling
> shows the two canonical patterns. A `ramp_up: 30min` lead time compensates for cold-start
> latency so replicas are warm when the event fires, while a cron-based rule handles recurring
> events without manual intervention.

---

**`lst-active-failover`** (lst 🟡) — def L5071

Ref sentence (L5065): *"@Lst-active-failover shows active-active failover routing logic that falls
back to the next healthy region after timeout or unavailability."*

Missing move: bare pointer. The reader gets no framing of what the code's design choice is — that
the fallback is determined by proximity rather than round-robin, and that the timeout parameter is
the critical tuning knob.

Rule-compliant rewrite:

> @Lst-active-failover implements the two-path routing: it tries the nearest healthy region first,
> and on timeout or error routes to the second-nearest rather than a random region. Proximity-based
> fallback keeps RTT bounded even during failover; the timeout value is the key tuning knob
> because setting it too high delays the reroute and too low generates spurious failovers on
> transient GPU stalls.

---

### TABLES

---

**`tbl-distribution-triggers`** (tbl 🟠) — def L147

Ref sentence (L129): *"@Tbl-distribution-triggers categorizes these triggers by constraint type and
corresponding strategy."*

Missing move: the prose names three triggers and refers to the table, but the payoff paragraph
(L149) pivots to TTFT/TPOT without stating the table's load-bearing conclusion — which trigger type
is most often the binding constraint in practice and what it implies about the typical design path.

Rule-compliant rewrite (between L129 and L131):

> @Tbl-distribution-triggers categorizes the three triggers. Memory exhaustion is the most common
> driver in practice: a model that cannot fit on one GPU must shard regardless of throughput or
> latency requirements, so memory capacity is often the first constraint that forces distribution.

---

**`tbl-serving-hierarchy`** (tbl 🟠) — def L602

Ref sentence (L593): *"@Tbl-serving-hierarchy maps these levels to their primary targets and
techniques:"*

Missing move: float-announcer colon pointer. The payoff (L606) is a checkpoint callout, not body
prose. The table's key insight — that each level has a different binding metric and that optimizing
at one level does not move the needle at another — is not stated in body prose; it lives only in
the caption.

Rule-compliant rewrite:

> @Tbl-serving-hierarchy maps each level to its metric and technique set, and the structure itself
> carries the lesson: request-level and replica-level optimizations target different metrics
> (per-request latency vs. throughput) and do not substitute for each other. A system that
> maximizes batch efficiency at the replica level can still exhibit poor tail latency if the
> request-level caching layer is missing.

---

**`tbl-serving-architecture-dimensions`** (tbl 🟠) — def L704

Ref sentence (L694): *"@Tbl-serving-architecture-dimensions summarizes how the batching, memory,
scheduling, topology, and state dimensions interact across the major workload types."*

Missing move: summarizer pointer. The sentence names what the table contains but does not state
the decision the table drives — which dimension is the first to fix for each workload, or the
observation that no single architecture is optimal across all.

Rule-compliant rewrite:

> @Tbl-serving-architecture-dimensions shows that the batching and memory dimensions are tightly
> coupled: the stateful KV-cache requirement of autoregressive LLMs forces paged memory, which in
> turn requires preemptive scheduling, while stateless vision workloads allow preallocated memory
> and FCFS scheduling. Reading the table column by column, the workload's memory model determines
> two of the five architectural dimensions.

---

**`tbl-batching-by-model`** (tbl 🟠) — def L852

Ref sentence (L840): *"@Tbl-batching-by-model summarizes how these batching characteristics vary
across model architectures."*

Missing move: summarizer pointer. The payoff (L908) discusses queuing theory and does not extract
the table's key observation. The sentence that introduces the table (L840) is about Little's Law,
which is actually a footnote topic; the connection between Little's Law and the batching summary
is not explained.

Rule-compliant rewrite:

> @Tbl-batching-by-model shows the pattern: compute-bound workloads (vision) prefer large static
> batches because the bottleneck is GPU arithmetic throughput, while memory-bound workloads (LLMs)
> require iteration-level scheduling because the bottleneck is KV-cache capacity per sequence.
> The table confirms that choosing a batching strategy without identifying the binding resource
> first will misalign the optimization.

---

**`tbl-continuous-batching-benefit`** (tbl 🟠) — def L1397

Ref sentence (L1296): *"@Tbl-continuous-batching-benefit quantifies this relationship across
different workload types..."*

Missing move: cite references the table correctly and provides context, but the payoff paragraph
(L1399) lists four conditions in body prose without extracting the key observation from the cells —
that workloads with CV greater than 1 see super-linear speedup while code completion (CV 0.3) gains
little, and that the decision threshold is approximately CV 0.7.

Rule-compliant rewrite (add after L1296 citation):

> The key threshold is output length variance: workloads with $\text{CV} > 0.7$ (RAG,
> conversational chat) see speedups above 1.5$\times$, while code completion at $\text{CV} \approx
> 0.3$ gains under 1.1$\times$, making continuous batching's implementation cost hard to justify
> for that workload class.

---

**`tbl-batching-tradeoffs`** (tbl 🟠) — def L1438

Ref sentence (L1427): *"@Tbl-batching-tradeoffs summarizes these trade-offs:"*

Missing move: float-announcer colon pointer. No preceding or following prose states the
load-bearing conclusion from the table — that the scheduler latency of 0.5–1 ms per iteration
is the implementation cost that matters most for high-throughput LLM serving, and that for
uniform-length workloads continuous batching delivers essentially no throughput gain.

Rule-compliant rewrite:

> @Tbl-batching-tradeoffs captures the key asymmetry: the throughput column shows 1.5–3.5$\times$
> improvement for variable workloads but essentially no gain for uniform ones, while the
> implementation effort rises from low to high in every row. The scheduler latency of 0.5–1 ms
> per iteration compounds across thousands of iterations per request, so deployment must verify
> that the added scheduler overhead stays small relative to the decode kernel time before adopting
> continuous batching.

---

**`tbl-inference-streaming-speech-pipeline`** (tbl 🟠) — def L1730

Ref sentence (L1717): *"@Tbl-inference-streaming-speech-pipeline traces the per-stage latency,
including feature extraction, that the pipeline must hit to stay within the 100 ms budget:"*

Missing move: float-announcer colon pointer. The payoff (L1732) explains what "Streaming
Conformer" and "CTC" are rather than stating what the table shows — which stage consumes the most
budget and what that implies for optimization. The answer (encoder inference at 30 ms consumes the
largest single slice) lives only in the cells.

Rule-compliant rewrite:

> @Tbl-inference-streaming-speech-pipeline traces the per-stage latency and shows that the encoder
> inference step at 30 ms is the dominant consumer of the 100 ms budget, with network RTT claiming
> 35 ms total across both hops. The encoder is the natural optimization target: model compression
> or a streaming-capable architecture reduces the largest single stage without touching network
> infrastructure.

---

**`tbl-inference-triton-adaptive-batching`** (tbl 🟠) — def L1786

Ref sentence (L1777): *"@Tbl-inference-triton-adaptive-batching shows the result for ResNet-50 on
V100: the scheduler automatically increases batch size to maintain throughput as traffic grows,
with measured throughput tracking offered load until saturation near 2,000 QPS."*

Partial — the cite names the saturation point but not the latency cost of that throughput. The
key observation (latency grows from 8 ms to 28 ms as batch size scales from 2 to 24, a 3.5$\times$
penalty that may violate a latency SLO before throughput saturates) lives only in the cells.

Rule-compliant rewrite:

> @Tbl-inference-triton-adaptive-batching shows the result for ResNet-50 on V100: throughput tracks
> offered load up to saturation at about 2,000 QPS, but average latency climbs from 8 ms at low
> traffic to 28 ms at saturation — a 3.5$\times$ increase that may violate a latency SLO before
> the system exhausts GPU capacity. The table's lesson is that the adaptive batcher optimizes
> throughput, not latency; operators must cap the maximum batch size if latency SLOs are tighter
> than GPU utilization targets.

---

**`tbl-inference-kv-cache-memory-hierarchy`** (tbl 🟠) — def L2741

Ref sentence (L2733): *"Production systems manage KV-cache memory pressure across the three-tier
hierarchy in @tbl-inference-kv-cache-memory-hierarchy, paging from GPU HBM down to CPU DRAM and
NVMe SSD as the active working set exceeds each tier's capacity:"*

Missing move: float-announcer colon pointer. The payoff (L2766) mentions the order-of-magnitude
capacity increase but not the latency cost — that a tier-2 swap adds 1–5 ms per evicted sequence,
which may be acceptable for long-queued but not for active-decode sequences. The decision rule
lives in the caption, not in body prose.

Rule-compliant rewrite:

> Production systems page KV-cache across the three tiers in @tbl-inference-kv-cache-memory-hierarchy.
> The 1–5 ms swap-in latency from CPU DRAM is tolerable for a sequence sitting at the back of the
> queue, but interrupting an active decode stream to page from SSD adds 10–50 ms per page fault —
> a penalty large enough to violate interactive SLOs. Eviction policy must therefore distinguish
> active from queued sequences before evicting.

---

**`tbl-sharding-triggers`** (tbl 🟠) — def L2920

Ref sentence (L2895): *"@Tbl-sharding-triggers identifies the memory and latency constraints that
necessitate sharding:"*

Missing move: float-announcer colon pointer. The payoff (L2924) begins discussing tensor parallelism
without extracting the table's decision logic — which trigger is the harder constraint and why
memory typically forces sharding before latency does.

Rule-compliant rewrite:

> @Tbl-sharding-triggers shows that the two drivers operate on different timescales: memory
> constraints are absolute (a model either fits or it does not) while latency constraints are
> contextual (a model may fit on one GPU but still benefit from sharding when the decode latency
> exceeds the SLO). Memory sharding is therefore the more common entry point, and latency sharding
> is the follow-on optimization once memory headroom is established.

---

**`tbl-pipeline-tensor-comparison`** (tbl 🟠) — def L3158

Ref sentence (L3148): *"@Tbl-pipeline-tensor-comparison captures the tradeoffs between these two
sharding approaches:"*

Missing move: float-announcer colon pointer. The payoff (L3162) pivots to MoE without extracting
the table's key conclusion — that the communication pattern difference (AllReduce vs. P2P) makes
tensor parallelism strictly require high-bandwidth intra-node interconnects while pipeline
parallelism can work over lower-bandwidth inter-node links.

Rule-compliant rewrite:

> @Tbl-pipeline-tensor-comparison shows the decision criterion: tensor parallelism reduces
> single-request latency at the cost of bandwidth-intensive AllReduce synchronization that requires
> NVLink-class interconnects, while pipeline parallelism scales throughput over point-to-point
> communication that works across nodes on InfiniBand. The communication requirement is the primary
> selection axis — tensor parallelism is viable within an NVLink domain; beyond that boundary,
> pipeline parallelism becomes the practical choice.

---

**`tbl-inference-moe-load-balancing`** (tbl 🟠) — def L3428

Ref sentence (L3418): *"...@tbl-inference-moe-load-balancing quantifies the degradation."*

This is inside a "Systems insight" callout, so the sentence is callout interior, not running body
prose. There is no body-prose citation; the table's takeaway (that 30 percent routing imbalance
reduces utilization by roughly 25 percent) lives in the callout only.

Rule-compliant rewrite (add body prose before callout):

> Routing imbalance directly erodes throughput: when the gating function concentrates tokens on
> a subset of experts, the overloaded devices become bottlenecks while idle devices waste capacity.
> @Tbl-inference-moe-load-balancing quantifies this degradation for a Mixtral-8x7B layer, showing
> that 30 percent imbalance reduces effective GPU utilization from 90 percent to approximately
> 65 percent and cuts throughput proportionally.

---

**`tbl-inference-allreduce-interconnect`** (tbl 🟠) — def L3751

Ref sentence (L3743): *"@Tbl-inference-allreduce-interconnect translates those bandwidths into
AllReduce time for an 8-way tensor-parallel layer (activation ... per all-reduce; batch=1,
hidden=...) as a fraction of a 30 ms transformer-layer budget:"*

Missing move: float-announcer colon pointer with no lead-out. The payoff (L3757) pivots to
NVLink generation history without stating the table's conclusion — that NVLink consumes under
1 percent of the layer budget while Ethernet consumes over 30 percent, making inter-node tensor
parallelism impractical on commodity interconnects.

Rule-compliant rewrite:

> @Tbl-inference-allreduce-interconnect translates those bandwidths into AllReduce cost as a
> fraction of a 30 ms layer budget, and the contrast is stark: NVLink's AllReduce consumes under
> 1 percent of the budget, InfiniBand consumes roughly 3 percent, and 100G Ethernet consumes over
> 30 percent. The Ethernet row makes the constraint concrete — tensor parallelism across commodity
> interconnects would spend more time on AllReduce synchronization than on computation, making it
> unsuitable for latency-critical serving.

---

**`tbl-inference-health-check-cadences`** (tbl 🟠) — def L4198

Ref sentence (L4186): *"@tbl-inference-health-check-cadences fixes interval, timeout, and
failure-threshold values for liveness, readiness, and deep-health probes."*

This citation is inside a "Systems lesson" callout (body of callout counts as callout interior).
There is no body-prose citation for this table. The payoff (L4202) opens a new section without
referring back.

Rule-compliant rewrite (add body prose after L4190, after the callout closes):

> For GPU inference, the three-tier probe schedule in @tbl-inference-health-check-cadences balances
> fast failure detection against probe cost: the 10-second liveness interval catches process crashes
> quickly, while the 30-second deep-health interval protects against unnecessary GPU warmup probes
> — each of which consumes memory bandwidth and can interfere with live inference.

---

**`tbl-inference-circuit-breaker-gpu`** (tbl 🟠) — def L4297

Ref sentence (L4286): *"...the canonical thresholds and recovery settings appear in
@tbl-inference-circuit-breaker-gpu."*

This citation is inside a "Systems lesson" callout — callout interior, not body prose. There is no
running-prose citation for this table. The payoff (L4299) opens a new section.

Rule-compliant rewrite (add body prose before callout):

> Production GPU inference requires tight circuit-breaker tuning because a single slow replica can
> silently absorb requests without returning errors. @Tbl-inference-circuit-breaker-gpu shows
> thresholds calibrated for GPU behavior: the error threshold of 5 percent is lower than
> typical web-service defaults because GPU OOM conditions often manifest as individual request
> failures rather than process crashes, and the 30-second open duration gives HBM pressure time
> to subside before probe requests resume.

---

**`tbl-gpu-region-pricing`** (tbl 🟠) — def L5119

Ref sentence (L5111): *"@Tbl-gpu-region-pricing shows representative H100 prices so placement can
balance cost against latency requirements:"*

Missing move: float-announcer colon pointer. The payoff (L5121) routes to the listing without
stating the table's conclusion — that the 20 percent price spread is small compared to the 15$\times$
RTT spread, and therefore cost-aware routing should only apply to latency-tolerant workloads.
That insight lives only in the caption.

Rule-compliant rewrite:

> GPU pricing varies by region, but the variation is modest compared to the latency spread.
> @Tbl-gpu-region-pricing shows that EU-West is only 5 percent more expensive per hour than US-West,
> yet imposes 75–100 ms RTT on US users — seven to ten times the local-region floor. The price
> differential cannot compensate for that latency penalty on interactive workloads, so cost-aware
> routing is worth pursuing only for batch or background inference.

---

**`tbl-tiktok-video-priorities`** (tbl 🟠) — def L5718

Ref sentence (L5710): *"New video content is processed with different priorities, as
@tbl-tiktok-video-priorities shows:"*

Missing move: float-announcer colon pointer. The payoff (L5720) states the priority scheme ensures
popular creators' content reaches recommendations quickly, but the table's engineering insight —
the 5-minute SLA for high-follower creators vs. 2-hour for bulk implies a 24$\times$ differentiation
in queue priority — is not stated.

Rule-compliant rewrite:

> New video content is processed with differentiated priority; @tbl-tiktok-video-priorities shows
> a 24$\times$ SLA spread from 5 minutes for high-follower creators to 2 hours for bulk imports.
> The spread allows the ingestion pipeline to absorb large upload bursts during off-peak hours
> without delaying content freshness for the creators whose videos drive the highest engagement.

---

## Dangling refs (no matching float definition)

Two cross-references in body prose at L2790 point to equations defined in an appendix section
(`@eq-fleet-ridge-point`, `@eq-fleet-arithmetic-intensity`). These are not findings against the
exposition standard (the citing prose is substantive), but they are orphan references that the
scanner cannot resolve in this chapter's scope. No action needed here; track via the CI orphan
check.
