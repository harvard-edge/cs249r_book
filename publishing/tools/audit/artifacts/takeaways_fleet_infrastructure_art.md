# Takeaways ART: Fleet Infrastructure

## book/quarto/contents/vol2/introduction/introduction.qmd

- Recommendation: Modify
- Evidence reviewed: Purpose, real heading outline, Scale Moment framing, C$^3$ taxonomy/fleet law sections, Summary, existing callout, chapter connection, opening scale/MTBF calculation cells; no summary-adjacent calculation cell found.
- Issues:
  - Existing bullets are conceptually aligned but mostly too compressed for the 25-55 word target.
  - Several bullets read closer to topic labels than durable engineering results.
  - The callout should better preserve the summary arc from single-node intuition to C$^3$, routine failure, and governance.
- Proposed callout:

::: {.callout-takeaways title="Scale makes a new machine"}

* **More GPUs change the problem**: Techniques that work on one node stop being merely slower at fleet scale; they can become wrong. Bisection bandwidth, power delivery, reliability, and governance create new constraints that do not appear in single-accelerator experiments.
* **The network spends every speedup**: Adding accelerators divides local compute but adds synchronization, gradient traffic, and tail-latency exposure. A 175B-parameter model can move hundreds of gigabytes of gradients per step, so useful FLOPs depend on communication, not peak arithmetic alone.
* **Failure becomes steady state**: A 25,000-GPU training run with per-GPU MTBF measured in tens of thousands of hours still sees cluster-level failures every few hours. Checkpointing, recovery, and observability are therefore baseline design requirements, not cleanup work.
* **C$^3$ replaces local intuition**: The single-machine Data, Algorithm, Machine lens projects to Compute, Communication, and Coordination at fleet scale. The fleet law makes this shift explicit: scaling depends on synchronization, overlap, and consistency as much as hardware count.
* **Impact needs a control plane**: Foundation-scale systems amplify security, privacy, fairness, and policy failures across large user populations. Governance is not an appendix to infrastructure; it is the mechanism that decides what the fleet is permitted to optimize and deploy.

:::

## book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd

- Recommendation: Keep
- Evidence reviewed: Purpose, heading outline from accelerator spectrum through pod and emerging technologies, Summary, SummaryScaleScenario and closing recap calculation cells, existing takeaways, chapter connection.
- Issues: None
- Proposed callout: Existing title `The data center is the computer`; no replacement needed.

## book/quarto/contents/vol2/network_fabrics/network_fabrics.qmd

- Recommendation: Modify
- Evidence reviewed: Purpose, Five-Level Network Model, heading outline across wire/link, transport, topology, fabric behavior, cluster design, monitoring, Summary, existing callout, opening bandwidth hierarchy calculation cell; no summary-adjacent calculation cell found.
- Issues:
  - Existing bullets are mostly short and topic-label-like, leaving the Five-Level diagnostic arc underdeveloped.
  - The callout should make bisection bandwidth, tail latency, lossless transport risk, and telemetry more explicit as engineering results.
  - Quantitative modeling appears in the chapter, but the current callout does not foreground the decision value of $n^* = \alpha \cdot \beta$.
- Proposed callout:

::: {.callout-takeaways title="The fabric decides useful compute"}

* **Link speed is not fabric speed**: A global AllReduce is limited by the narrowest bisection cut, not the fastest advertised port. Topology decides how much purchased accelerator throughput becomes useful training throughput and how much becomes idle silicon.
* **Latency and bandwidth bind differently**: The $\alpha$-$\beta$ model separates startup cost from per-byte transfer cost, with $n^* = \alpha \cdot \beta$ marking the regime change. Message-size distributions, not vendor peak numbers, determine whether to optimize software latency or hardware bandwidth.
* **Losslessness moves the risk**: RDMA needs a lossless fabric to avoid expensive retransmission stalls. InfiniBand supplies this natively, while RoCE depends on PFC, ECN, DCQCN, and HPCC, trading hardware flexibility for tail-latency and operational complexity.
* **Topology must match traffic**: Fat-trees buy flexible bisection bandwidth, rail-optimized designs accelerate same-rank AllReduce, and dragonfly or torus designs trade cabling and locality differently. The right fabric depends on whether the workload stresses AllReduce, AllToAll, or multi-tenant sharing.
* **Telemetry saves GPU-hours**: PFC counters, link error rates, bandwidth baselines, and application throughput must be correlated across layers. Without that observability, a degraded transceiver or congestion storm silently converts a high-end fleet into a queue of waiting accelerators.

:::

## book/quarto/contents/vol2/data_storage/data_storage.qmd

- Recommendation: Keep
- Evidence reviewed: Purpose, heading outline from fuel line through hierarchy, pipeline equation, GPUDirect Storage, economics, checkpoint storage, retrieval, synthetic fuel line, Summary, GDSLatencyRecap and EconRatiosRecap cells, existing takeaways, chapter connection.
- Issues: None
- Proposed callout: Existing title `Feed the accelerators or waste them`; no replacement needed.

## book/quarto/contents/vol2/distributed_training/distributed_training.qmd

- Recommendation: Modify
- Evidence reviewed: Purpose, heading outline across necessity, data/model/tensor/pipeline/hybrid parallelism, RLHF/alignment, strategy comparison, Summary, DistTrainSummaryRecap cell, archetype spectrum, existing callout, chapter connection.
- Issues:
  - Existing bullets mostly enumerate parallelism techniques rather than stating the chapter's central result: parallelism relocates overhead among Compute, Communication, and Coordination.
  - Several bullets are shorter than the target range and do not fully integrate the 3D cube, sharding, and hardware-hierarchy mapping.
- Proposed callout:

::: {.callout-takeaways title="Parallelism relocates the tax"}

* **Splitting work moves the bottleneck**: Distributed training does not make the underlying computation smaller; it converts memory pressure into communication volume, synchronization delay, or idle pipeline time. The winning strategy is the split that sends overhead to the least binding part of the fleet.
* **Data parallelism ends at convergence**: Replicas scale throughput cleanly only while larger global batches still improve optimization. Past the critical batch size, AllReduce cost and reduced gradient noise turn "more workers" into slower or less stable learning unless schedules, warmup, and accumulation change.
* **Sharding buys memory with messages**: ZeRO and FSDP make 100B+ parameter models feasible by partitioning optimizer state, gradients, and parameters. The price is a stricter communication schedule of ReduceScatter and AllGather operations that must be overlapped or hidden.
* **Tensor parallelism belongs near NVLink**: Tensor parallelism splits matrix operations inside layers, so it needs the high-bandwidth intra-node fabric that A100 and H100 NVLink provide. Stretching that traffic across racks turns a memory-capacity solution into a communication bottleneck.
* **Pipeline parallelism trades bytes for bubbles**: Layer staging reduces communication frequency, but fill and drain slots leave accelerators idle. Microbatching with $m \gg p$ is the mechanism that turns model depth into throughput rather than pipeline slack.
* **The 3D cube is a hardware map**: Real frontier training combines data, tensor, pipeline, expert, and sharded parallelism because no single axis fits the model and the fleet. Logical groups must map to HBM, NVLink, InfiniBand, and failure domains together.

:::

## book/quarto/contents/vol2/collective_communication/collective_communication.qmd

- Recommendation: Keep
- Evidence reviewed: Purpose, heading outline across alpha-beta/LogP, primitive vocabulary, AllReduce algorithms, hierarchical communication, compression, libraries, overlap, Summary, communication archetype mapping, existing takeaways, early gradient synchronization and cost-model calculation cells.
- Issues: None
- Proposed callout: Existing title `Every byte has a travel cost`; no replacement needed.

## book/quarto/contents/vol2/fault_tolerance/fault_tolerance.qmd

- Recommendation: Keep
- Evidence reviewed: Purpose, heading outline across failure analysis, hardware/software faults, silent data corruption, checkpointing, recovery, elasticity, serving fault tolerance, degradation, observability, case studies, Summary, existing takeaways, reliability-rate and Young-Daly/checkpoint calculation cells.
- Issues: None
- Proposed callout: Existing title `Failure is normal operation`; no replacement needed.

## book/quarto/contents/vol2/fleet_orchestration/fleet_orchestration.qmd

- Recommendation: Keep
- Evidence reviewed: Purpose, heading outline across scheduling, orchestration paradigms, topology-aware scheduling, elasticity, cost optimization, custom schedulers, serving resource management, multi-tenancy, utilization debugging, Summary, existing takeaways, cluster economics and utilization calculation cells.
- Issues: None
- Proposed callout: Existing title `Scheduling is systems engineering`; no replacement needed.
