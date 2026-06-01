# MLSysBook LEGO Single-Source Registry Plan

**Created:** 2026-06-01
**Scope:** Volume 1 and Volume 2 LEGO cells in `book/quarto/contents`
**Goal:** Move book-defining quantitative inputs into MLSysIM registries or
typed scenario profiles so LEGO cells load from one source of truth and perform
only composition, calculation, guard checks, and output formatting.

## 1. Principle

The stronger rule is:

> If a numeric value defines the scenario being calculated, it belongs in
> MLSysIM, even when it appears only once in the book.

This extends the existing rule that hardware specs, model specs, grids, pricing,
datasets, and physical constants must come from MLSysIM. A one-off scenario such
as "64 A100 GPUs for 14 days on the US average grid" is still part of the
book's quantitative contract. It should be a named MLSysIM scenario profile,
not a cluster of local literals in a QMD cell.

The exception is narrow. The following may stay local:

- pure algebra operands whose point is the variable itself (`n = 64` in a GEMM
  scaling derivation, small probability vectors in a KL example);
- plot geometry and annotation coordinates;
- loop counters and table-construction scratch variables;
- values derived inside `EXECUTE` from registry inputs.

Everything else should load from MLSysIM.

## 2. Current Audit Snapshot

A first rough LOAD-stage inventory over both volumes found **2,243
literal-like assignment sites** in Python cells. That was intentionally broad
and included many pure algebra examples.

A refined advisory audit now exists:

```bash
python3 book/tools/audit/book_check_lego_scenario_inputs.py \
  book/quarto/contents --summary
```

Current output:

- **1,067 advisory MLSysIM source-of-truth candidates**
- **53 high-confidence candidates**
- Full JSON work queue:
  `book/tools/audit/artifacts/lego_scenario_inputs_audit.json`
- Full Markdown work queue:
  `book/tools/audit/artifacts/lego_scenario_inputs_audit.md`

The refined audit is still advisory. It is designed to produce the migration
queue, not to block builds. It deliberately skips existing registry-sourced
assignments and focuses on LOAD-stage numeric/constructor inputs.

High-confidence buckets:

| Target | Count | Meaning |
|---|---:|---|
| `Hardware.*` / `Hardware.Tech.*` | 18 | Hardware capacity, bandwidth, FLOP/s, TDP, memory/interconnect facts |
| `Systems.Clusters` / `Systems.Nodes` | 13 | Fleet, node, GPU-count, cluster-topology facts |
| `Systems.Fabrics` / `Systems.SwitchFabric` | 9 | Network/fabric/switch-sizing facts |
| `Infrastructure.Pricing.Cloud` / `Infrastructure.Pricing.Fleet` | 6 | GPU-hour, cloud-instance, and fleet price points |
| `Infrastructure.*` / `Scenarios.Sustainability` | 4 | PUE, cooling, carbon, sustainability scenario facts |
| `Systems.Storage` | 2 | Local NVMe, HDD, PFS, S3/object-store, checkpoint-path storage facts |
| `Infrastructure.Pricing.Storage` | 1 | Storage/monitoring price points |

Broader medium-confidence buckets:

| Target | Count | Meaning |
|---|---:|---|
| `Scenarios.*` / `Ops.*` | 287 | Workload, SLA, monitoring, drift, threshold, and operational policies |
| `Models.*` / `Scenarios.TrainingRuns` | 213 | Model dimensions and training-run workload inputs |
| `Scenarios.*` | 164 | Unit-bearing scenario inputs |
| `Datasets.*` / `Scenarios.DataWorkloads` | 109 | Dataset/sample/corpus sizes |
| `Infrastructure.Pricing.*` / `Scenarios.*` | 111 | Economic assumptions not yet in a named price registry |
| `Hardware.*` | 49 | Hardware-related values that need review |

This is the first real full-cell audit. Migration is **underway**, not done.

The earlier rough classification remains useful for planning the larger waves:

| Category | Approx. sites | Registry home |
|---|---:|---|
| Data/model/storage workload assumptions | 337 | `Systems.Storage`, `Scenarios.DataWorkloads`, `Datasets`, `Models` |
| Time/duration assumptions | 369 | scenario profiles, `Systems.Orchestration`, workload profiles |
| Cost/pricing assumptions | 139 | `Infrastructure.Pricing`, scenario profiles |
| Energy/carbon/power assumptions | 87 | `Infrastructure`, `Hardware`, `Scenarios.Sustainability` |
| Fleet/topology counts | 59 | `Systems.Clusters`, `Systems.Nodes`, `Systems.Fabrics` |
| Probability/statistical/algorithm parameters | 147 | `Literature`, `Ops`, scenario profiles |
| Other numeric assumptions | 1,105 | classify; many will be algebra/local scratch |

Known concrete findings from the first pass:

- Stage 1 moved the Sustainable AI lifecycle-carbon `DummyFleet` to
  `Systems.Clusters.Production_2K`.
- Stage 1 moved the 25,000-H100 reference-cluster examples in
  `sustainable_ai.qmd` and `conclusion.qmd` to
  `Systems.Clusters.Reference_25K_H100`.
- Stage 3 moved clear 1,024-GPU / 128-node H100 examples in
  `compute_infrastructure.qmd` and `ops_scale.qmd` to
  `Systems.Clusters.Training_1K`.
- Stage 3 moved the 128-node / 1,024-GPU A100 hierarchical-AllReduce debugging
  example in `distributed_training.qmd` to `Systems.Clusters.Training_1K_A100`.
  Do not force A100 examples through the H100 reference fleet.
- Stage 3 moved the 256-node HBM memory-budget example in `data_storage.qmd`
  to `Systems.Clusters.Production_2K`.
- Stage 4 moved the 128-node reliability example in
  `compute_infrastructure.qmd` to `Systems.Clusters.Training_1K`, and moved
  the composite DGX-node MTBF plus low/high recovery window assumptions to
  `Systems.Reliability`.
- Stage 5 added random-access storage profiles (`Hdd7200Rpm`, `SataSsd`,
  `LocalNvmeGen3`) with IOPS metadata under `Systems.Storage`, then migrated
  the `data_selection.qmd` random-access penalty table to load those tiers.
- Stage 6 added explicit round-number S3 and Glacier storage-price anchors
  (`S3StandardLowPerTbMonth`, `GlacierStandardPerTbMonth`) so examples that
  intentionally use $0.02/GB-month and $0.004/GB-month no longer hide those
  assumptions in QMD cells.
- Stage 7 added `Infrastructure.Pricing.Monitoring` and migrated the
  `ml_ops.qmd` single-model monitoring budget to load ingestion, storage, and
  query rates from MLSysIM.
- Stage 8 added effective training-storage throughput profiles for cloud block
  volumes, object-store single-stream reads, SATA SSDs, and local NVMe, then
  migrated the repeated `data_engineering.qmd` storage-throughput examples.
- Stage 9 migrated the `fault_tolerance.qmd` checkpoint-debug NVMe bandwidth
  to `Systems.Storage.LocalNvmeGen3`.
- Stage 10 added small H100 fleet tiers (`Lab_64_H100` and
  `Training_512_H100`) and migrated the matching
  `fleet_orchestration.qmd` spot-training, chargeback, and preemption examples.
  The same pass changed chargeback/preemption duration outputs from plain
  `fmt(...)` numbers plus prose units to Pint-backed `fmt_time(...)` output.
- Stage 10 also added a sourced `Systems.Clusters.Kempner_H100_384` profile
  for the published Kempner Institute H100 partition. It intentionally models
  only the homogeneous H100 partition; the full 2026 Kempner expansion is
  heterogeneous and should wait for a composite cluster type.
- Stage 11 migrated four obvious GPU-hour price anchors in
  `ops_scale.qmd` and `distributed_training.qmd` to existing
  `Infrastructure.Pricing.Fleet.GpuHourRef` and
  `Infrastructure.Pricing.Cloud.GpuTrainingUtilityScenarioPerHour` entries.
  Ambiguous or scenario-specific price assumptions remain for later review.
- Stage 12 migrated Sustainable AI PUE anchors to `Infrastructure.FacilityCooling`.
  `PueEfficiency` now loads legacy and state-of-art PUE values from the
  registry, and `PueSavings` loads the simple air-cooled baseline from the new
  `SimpleAir` cooling profile rather than redefining `1.5` locally.
- Stage 13 migrated obvious GPU-hour price anchors in `data_engineering.qmd`,
  `model_serving.qmd`, `inference.qmd`, and `ml_ops.qmd` to existing
  `Infrastructure.Pricing.Cloud` and `Infrastructure.Pricing.Fleet` entries.
  These were exact matches to the registry price points; remaining price
  findings include labor rates, per-query business assumptions, or scenario
  prices that need a separate classification pass.
- Stage 14 tightened the advisory audit's high-confidence classifier so it no
  longer sends agents after pure scenario values such as Amdahl processor
  counts, preprocessing/postprocessing latency budgets, AllReduce bucket sizes,
  or workload FLOP formulas. Those remain in the advisory queue as medium
  scenario/profile work; high-confidence findings now mean "likely registry
  source-of-truth migration" more reliably.
- 100,000-GPU examples should load `Systems.Clusters.Mega_100K`.
- Storage/checkpoint examples mix two different kinds of facts: storage-system
  facts such as local NVMe drive count, local/PFS bandwidth, capacity, and
  per-node staging bandwidth belong in named `Systems.Storage` profiles;
  workload policy facts such as checkpoint interval and corpus size belong in
  scenario or workload profiles. Neither kind should live as arbitrary QMD
  literals.

Additional concrete findings from the refined audit:

- `sustainable_ai.qmd:873` previously defined
  `LifecycleCarbonEstimate.DummyFleet`; Stage 1 replaced it with
  `Systems.Clusters.Production_2K`.
- `sustainable_ai.qmd:211` previously used `cluster_gpus = 25000`; Stage 1
  replaced it with `Systems.Clusters.Reference_25K_H100`.
- Stage 2 added `Systems.Storage.Production2KCheckpointPath` and migrated the
  `data_storage.qmd:2330-2335` checkpoint-storm storage path to it. Checkpoint
  cadence remains local until training-run scenario profiles exist.
- `compute_infrastructure.qmd:3910-3912` previously used local 1,024-GPU and
  128-node literals for fabric sizing; Stage 3 now loads the fleet from
  `Systems.Clusters.Training_1K`. Switch-port topology constants remain local
  until a richer fabric-profile object exists.
- Repeated NVMe/HDD/S3 bandwidth examples in `data_engineering.qmd`,
  `data_selection.qmd`, `model_serving.qmd`, and `data_storage.qmd` should be
  normalized against `Hardware.Tech.Storage` plus `Systems.Storage` profiles.
- GPU-hour, storage, monitoring-ingest, and human-labeling rates should be
  reviewed against `Infrastructure.Pricing.*` rather than left as free QMD
  floats.

## 3. Registry Taxonomy

Use existing registries first. Add new ones only where no existing home is
semantically correct.

| Value kind | Registry home | Notes |
|---|---|---|
| Hardware specs | `Hardware.*` | Existing. QMD should not redefine TDP, HBM, FLOP/s, memory, storage specs. If the example is about one accelerator, load the hardware object directly instead of routing through a cluster. |
| Model specs | `Models.*` | Existing. Parameters, training tokens, layers, hidden dims belong here. |
| Dataset specs | `Datasets.*` | Existing for named datasets. Add missing datasets before using local literals. |
| Fleet topology | `Systems.Clusters`, `Systems.Nodes`, `Systems.Fabrics` | Existing; add stock fleets for book scenarios. Use this when the example is about aggregate fleet behavior, topology, node count, reliability, power, storage paths, or cluster throughput. |
| Grids/datacenters/cooling/racks | `Infrastructure.*` | Existing. Use grids/datacenters for carbon and PUE. |
| Prices/rates | `Infrastructure.Pricing.*` | Existing. Scenario-specific prices can be `PricePoint`s or scenario fields with provenance. |
| Literature constants | `Literature.*` | Existing for published or conventional model/system constants. |
| Operational policies | `Ops.*`, `Systems.Orchestration` | Existing for monitoring/orchestration assumptions. |
| Storage technologies | `Hardware.Tech.Storage` | Existing for technology classes such as NVMe Gen3/4/5, host DRAM, and system memory. |
| Device-attached storage | `Hardware.*.storage` | Existing for storage built into a specific box or accelerator system, such as workstation-local NVMe. |
| System storage subsystems | `Systems.Storage` | Add this for local NVMe arrays, parallel file systems, object stores, durable checkpoint paths, and staging tiers. |
| Book scenario inputs | `Scenarios.*` | Add typed use-case profiles for workloads such as a smart doorbell, a 70B training run, or a lifecycle-carbon estimate. Scenarios compose Hardware, Systems, Infrastructure, Models, Datasets, and Pricing. |

## 4. New MLSysIM Objects To Add

### 4.1 Stock Fleets

Use two classes of fleet entries.

1. **Book reference tiers** are anonymous, round-number systems used for
   derivations where no real operator is intended. These should use names such
   as `Reference_2K_H100`, `Reference_10K_H100`, or `Reference_25K_H100`, not
   names that imply a specific public deployment.
2. **Named public clusters** are sourced real-world infrastructure profiles.
   These are better when the prose is discussing frontier-scale practice rather
   than a synthetic example. They should carry explicit provenance and a
   verification date because public cluster specs change.

Add these under `Systems.Clusters` with `Metadata(provenance=...)`:

- `Reference_25K_H100`: 25,000 H100 GPUs, 3,125 DGX H100-style nodes, NDR
  fabric. Used only when the book needs a clean round-number fleet, not a
  named operator.
- `Lab_64_H100`: 64 H100 GPUs, 8 DGX H100-style nodes, HDR fabric. Use this
  for small physical-fleet examples such as chargeback or preemption when the
  prose really means a 64-H100 cluster.
- `Training_512_H100`: 512 H100 GPUs, 64 DGX H100-style nodes, HDR fabric.
  Use this for mid-small H100 fleet economics examples.
- `Training_1K_A100`: 1,024 A100 GPUs, 128 DGX A100-style nodes, HDR fabric.
  Use this for A100 debugging and communication examples that intentionally
  use A100 NVLink/HDR characteristics.
- `Kempner_H100_384`: public Kempner Institute H100 partition profile:
  384 H100 80GB GPUs in 96 four-GPU servers with NDR fabric. Do not use this
  for the full 1,144-GPU 2026 mixed H200/H100/A100/RTX Pro 6000 expansion
  until MLSysIM has a composite heterogeneous fleet representation.
- `XAI_Colossus_H100`: public xAI Colossus profile. Official xAI page states
  200,000 H100 GPUs in a single interconnected cluster and also reports
  "180K GPUs" in its "By the numbers" section, plus 170 PB/s aggregate memory
  bandwidth, 2.8 Tb/s per-server training network bandwidth, and >0.5 EB total
  storage. Preserve the exact sourced fields and do not silently reconcile the
  180K/200K discrepancy; model the value used by the prose explicitly.
- `Meta_GenAI_24K_H100_RoCE` and `Meta_GenAI_24K_H100_IB`: public Meta
  profiles for the two 24,576-H100 clusters described by Meta, one using RoCE
  and one using NVIDIA Quantum-2 InfiniBand.
- `Meta_Llama31_16K_H100_Run`: training-run/system profile for Llama 3.1 405B,
  which Meta says pushed model training to more than 16,000 H100 GPUs. This is
  better as a scenario/training-run profile referencing a system scale than as
  a generic cluster tier.
- `Meta_129K_H100_Cluster`: candidate public Meta profile from Meta's later
  infrastructure retrospective. Verify and record provenance before using this
  in calculations.

Consider after audit classification:

- `Lab_64_A100`: 64 A100 GPUs, 8 DGX A100 nodes. Use only where examples
  actually mean a 64-A100 training cluster.
- `SingleNode_H100` or direct `Systems.Nodes.DGX_H100` for 8-GPU examples.
  Prefer the node object when the example is about one node, not a fleet.

Do not collapse all "64 GPU" examples into one fleet blindly. Some are A100,
some H100, and some are pure sharding degree or algorithmic parallelism rather
than a physical fleet.

Rule of thumb:

- If the chapter says "suppose we have N GPUs," use a `Reference_*` tier.
- If the chapter says "frontier labs / Grok / Meta / Llama / public
  supercluster," use a named public cluster profile.
- If the number is just a parallelism degree, batch size, tensor axis, or
  algorithmic shard count, keep it local or move it to a training-run scenario,
  not `Systems.Clusters`.

### 4.2 System Storage Types

Add storage-system types under `mlsysim/systems/types.py` or a small
`mlsysim/systems/storage.py` module. Do not put cluster storage boxes under
`Scenarios`; a scenario is the workload/use case that chooses a storage
system.

```python
class StorageSubsystem(BaseModel):
    name: str
    storage_tech: Any | None = None
    capacity: Quantity | None = None
    bandwidth: Quantity
    latency: Quantity | None = None
    media: str | None = None
    interface: str | None = None
    durability: str | None = None
    metadata: Metadata = Field(default_factory=Metadata)

class NodeStorageConfig(BaseModel):
    name: str
    device: StorageSubsystem
    devices_per_node: int = 1
    metadata: Metadata = Field(default_factory=Metadata)

    @property
    def aggregate_bandwidth(self) -> Quantity:
        return self.devices_per_node * self.device.bandwidth

class CheckpointStoragePath(BaseModel):
    name: str
    local_stage: NodeStorageConfig | None = None
    durable_store: StorageSubsystem | None = None
    write_bandwidth: Quantity | None = None
    metadata: Metadata = Field(default_factory=Metadata)
```

Example registry homes:

- `Systems.Storage.LocalNvmeGen4x4`
- `Systems.Storage.LocalNvmeGen5x4`
- `Systems.Storage.PfsOneTbPerSecond`
- `Systems.Storage.Production2KCheckpointPath`

The scenario can then say "this GPT-3 checkpoint storm uses
`Systems.Storage.Production2KCheckpointPath` and checkpoints every 10 minutes."

### 4.3 Scenario Profile Types

Add lightweight Pydantic models in `mlsysim/scenarios/types.py`:

```python
class TrainingRunProfile(BaseModel):
    name: str
    model: Any | None = None
    fleet: Any | None = None
    duration: Quantity | None = None
    mfu: float | None = None
    grid: Any | None = None
    metadata: Metadata = Field(default_factory=Metadata)

class SustainabilityScenarioProfile(BaseModel):
    name: str
    fleet: Any | None = None
    grid: Any | None = None
    duration: Quantity | None = None
    amortization_months: int | None = None
    amortization_window: Quantity | None = None
    metadata: Metadata = Field(default_factory=Metadata)
```

Keep these profiles deliberately small. They are input containers, not solvers.
The LEGO cell still performs the calculation with MLSysIM physics helpers.

### 4.4 Scenario Registries

Add nested classes under `mlsysim/scenarios/registry.py`:

- `Scenarios.TrainingRuns`
- `Scenarios.Sustainability`
- `Scenarios.Serving`
- `Scenarios.ResponsibleAI`

Start with the high-value examples that currently cause drift:

- `Scenarios.Sustainability.LifecycleCarbon70BProduction2K`
  - fleet: `Systems.Clusters.Production_2K`
  - grid: `Infrastructure.Grids.US_Avg`
  - duration: `30 * day`
  - amortization: 36 months, 1-month window
  - embodied carbon source: `Hardware.Cloud.H100.embodied_carbon_kg`

- `Scenarios.Sustainability.Reference25KEnergyWall`
  - fleet: `Systems.Clusters.Reference_25K_H100`
  - hardware: H100 through the fleet

- `Scenarios.Sustainability.Training7B64A100UsAvg`
  - fleet: `Systems.Clusters.Lab_64_A100` if added
  - grid: `Infrastructure.Grids.US_Avg`
  - comparison grid: `Infrastructure.Grids.Quebec`
  - duration: `14 * day`

- `Scenarios.TrainingRuns.GPT3Production2KCheckpointStorm`
  - model: `Models.Language.GPT3`
  - fleet: `Systems.Clusters.Production_2K`
  - storage path: `Systems.Storage.Production2KCheckpointPath`
  - checkpoint interval: `10 * minute`

- `Scenarios.Ops.TrainingCapacityProduction2K`
  - fleet: `Systems.Clusters.Production_2K`
  - duration: `30 * day`
  - GPU-hour rate: `Infrastructure.Pricing.Fleet.GpuHourRef.rate` or a named
    book-specific price point if the prose needs a different rate.

### 4.5 Full System Boxes

Some entries are complete boxes rather than bare chips. `DGX_Spark` already
exists locally as `Hardware.Workstation.DGX_Spark`, and the current schema
captures compute, memory, attached storage, TDP, and cost. If a LEGO cell or
prose block wants a "what is in the box" view, do not duplicate those facts in
the QMD. Enrich MLSysIM first.

Likely schema additions:

- `HardwareNode.chip` or `package`
- `HardwareNode.cpu`
- `HardwareNode.gpu_architecture`
- `MemoryHierarchy.unified`
- `StorageHierarchy.media`, `interface`, and `self_encrypting`
- `HardwareNode.networking`
- `HardwareNode.dimensions` and `weight`
- `unit_cost_launch` separate from current or range pricing

Before changing modern product specs, verify against the official vendor source
and record provenance in the YAML. The taxonomy rule is:

- technology class facts go in `Hardware.Tech.Storage`;
- attached storage inside the box goes in `Hardware.Workstation.DGX_Spark.storage`;
- reusable cluster/storage infrastructure goes in `Systems.Storage`;
- a narrative use case such as a smart doorbell or training run goes in
  `Scenarios.*`.

### 4.6 Other System Registries

Apply the same reference-vs-public split beyond clusters.

| System kind | Registry home | Reference profiles | Named public profiles |
|---|---|---|---|
| GPU fleets | `Systems.Clusters` | `Reference_2K_H100`, `Reference_25K_H100` | xAI Colossus, Meta GenAI clusters, Meta RSC |
| Node/box designs | `Systems.Nodes`, `Hardware.Workstation`, `Hardware.Cloud` | `Reference_DGX_H100_Node`, `Reference_GB200_Rack` | DGX Spark, DGX H100/B200, Meta Grand Teton where sourced |
| Fabric designs | `Systems.Fabrics`, `Systems.SwitchFabric` | 2-tier IB, 3-tier leaf-spine, rail-optimized reference fabrics | Meta RoCE cluster, Meta InfiniBand cluster, xAI Colossus network profile |
| Storage systems | `Systems.Storage`, `Hardware.Tech.Storage` | Local NVMe x4, PFS 1 TB/s, object-store cold tier, checkpoint path | Meta Tectonic/Hammerspace profiles where sourced, xAI Colossus storage profile |
| Facility envelopes | `Infrastructure.Datacenters`, `Infrastructure.FacilityCooling` | 1 MW / 10 MW / 100 MW facility, air vs liquid cooling | Named public datacenter profiles only when sourced enough to be useful |
| Reliability stacks | `Systems.Reliability` | GPU/NIC/PSU/cable MTBF, checkpoint write path, hot-spare profile | Public incident/system papers when they include usable numbers |
| Scheduling/orchestration | `Systems.Orchestration`, `Ops.*`, `Scenarios.*` | gang scheduling, spot interruption, quota reservation, utilization targets | Public cluster-scheduler case studies when sourced |
| Monitoring/observability | `Ops.Monitoring`, `Infrastructure.Pricing.*` | scrape intervals, metric cardinality, ingest/storage/query price points | Vendor/public case studies where they support the narrative |
| Edge/product systems | `Hardware.Mobile`, `Hardware.Edge`, `Systems.Nodes`, `Scenarios.*` | smart doorbell, voice assistant, phone NPU, edge camera | Named devices only when the exact product matters |

Migration rule:

- **Component fact** -> `Hardware.*`. Use this for one accelerator's memory,
  TDP, FLOP/s, interconnect, storage, cost, or embodied carbon.
- **Physical box/spec** -> `Hardware.*` or `Systems.Nodes`.
- **Composed infrastructure design** -> `Systems.*`.
- **Facility/grid/cooling context** -> `Infrastructure.*`.
- **Workload/use case** -> `Scenarios.*`.
- **Operational policy** -> `Ops.*` or `Systems.Orchestration`.
- **Price/rate** -> `Infrastructure.Pricing.*`.

Do not derive every value from `Systems.Clusters`. The source object should
match the semantic focus of the calculation: GPU facts from `Hardware`, node
composition from `Systems.Nodes`, fleet/topology facts from `Systems.Clusters`,
and cross-domain workload assumptions from `Scenarios`.

Narrative rule:

- Use named public systems when the name teaches something: scale, topology,
  storage design, energy envelope, or real production trade-off.
- Use anonymous reference systems when the point is the algebra and a real name
  would distract or overclaim.
- Never put an unsourced named-system fact in a QMD. Add or extend MLSysIM with
  provenance first, then use the registry in LEGO.

## 5. QMD Migration Algorithm

Run this per registry domain, not per whole book.

1. **Inventory.** Use the scenario-input audit to produce a JSON queue of LOAD
   literals and local dummy objects.
2. **Classify.** Mark each item as existing registry, new stock fleet, new
   scenario profile, pure algebra/local, or plot scratch.
3. **Add MLSysIM object.** Add typed profile or registry entry with provenance.
4. **Add tests.** Validate counts, dimensions, fleet totals, and key scenario
   values.
5. **Migrate QMD LOAD.** Replace local literals with `case = Scenarios...` or
   `fleet = Systems.Clusters...`. Keep EXECUTE/GUARD/OUTPUT mostly unchanged.
6. **Run focused execution.** Execute the touched QMDs with the LEGO harness.
7. **Run gates.** `book_check_registry_sources.py`, `book_check_lego_quantity_flow.py`,
   `book_check_lego_prose_units.py`, `lint_lego_units.py`, focused pytest.
8. **Commit one domain.** One logical commit per registry domain.

The one-cell target form is:

```python
class LifecycleCarbonEstimate:
    # LOAD
    case = Scenarios.Sustainability.LifecycleCarbon70BProduction2K
    fleet = case.fleet
    baseline_grid = case.grid
    training_duration = case.duration
    h_h100 = fleet.node.accelerator

    # EXECUTE
    res = SustainabilityModel().solve(
        fleet=fleet,
        datacenter=baseline_grid,
        duration_days=training_duration.to(day).magnitude,
        mfu=1.0,
    )
```

## 6. Audit Tool Upgrade

Create or upgrade an advisory checker:

`book/tools/audit/book_check_lego_scenario_inputs.py`

Required behavior:

- Parse Python cells and stage markers.
- Report LOAD-stage numeric literals even when comments say "scenario
  assumption"; that comment should become a migration hint, not an exemption.
- Detect local `Dummy*`, `type('obj', ...)`, and local `Fleet(...)` /
  `Node(...)` creation in QMD.
- Detect storage assumptions and classify them explicitly:
  - drive count, per-drive bandwidth, aggregate storage bandwidth, storage
    capacity, storage latency, media, and interface -> `Systems.Storage`;
  - checkpoint interval, corpus size, request mix, and SLA/workload policy ->
    `Scenarios.*`, `Datasets.*`, or workload profiles.
- Suggest existing fleet replacements when values match:
  - 1,024 GPUs or 128 DGX nodes → `Systems.Clusters.Training_1K`
  - 1,024 A100 GPUs or 128 DGX A100 nodes →
    `Systems.Clusters.Training_1K_A100`
  - 2,048 GPUs or 256 DGX nodes → `Systems.Clusters.Production_2K`
  - 8,192 GPUs → `Systems.Clusters.Frontier_8K`
  - 10,000 GPUs → `Systems.Clusters.Training_10K`
  - 100,000 GPUs → `Systems.Clusters.Mega_100K`
  - 25,000 GPUs → `Systems.Clusters.Reference_25K_H100`
  - 512 H100 GPUs → `Systems.Clusters.Training_512_H100`
  - 64 H100 GPUs → `Systems.Clusters.Lab_64_H100`
  - Kempner H100 partition → `Systems.Clusters.Kempner_H100_384`
- Emit JSON for agent work queues.
- Allowlist pure algebra and figure-only cells explicitly.

Do not promote this to a blocking gate until the first migration wave is done
and the allowlist is honest.

## 7. First Safe Implementation Slice

The first slice should be small and high-confidence:

1. [x] Add `Systems.Clusters.Reference_25K_H100`.
2. [x] Replace the local 25,000-GPU literals in:
   - `vol2/sustainable_ai/sustainable_ai.qmd`
   - `vol2/conclusion/conclusion.qmd`
3. [x] Replace `LifecycleCarbonEstimate.DummyFleet` with
   `Systems.Clusters.Production_2K`.
4. [x] Replace obvious 1,024-GPU / 128-node and 2,048-GPU / 256-node cases that
   are already described as stock H100 or A100 training clusters with
   `Systems.Clusters.Training_1K`, `Systems.Clusters.Training_1K_A100`, or
   `Systems.Clusters.Production_2K`.
5. [x] Add the `Systems.Storage` skeleton and migrate one checkpoint/storage
   example end to end, preferably the 2,048-GPU checkpoint path, to prove the
   taxonomy before broad migration.
6. [x] Run focused LEGO execution for touched chapters and focused MLSysIM tests.
7. [x] Add `Systems.Clusters.Lab_64_H100` and
   `Systems.Clusters.Training_512_H100`, then migrate the matching
   `fleet_orchestration.qmd` economics examples.
8. [x] Add `Systems.Clusters.Kempner_H100_384` as a sourced public-cluster
   profile for the homogeneous H100 partition only.

This gives immediate value without forcing the whole scenario-profile taxonomy
to exist on day one.

## 8. Definition Of Done

This effort is complete when:

- No QMD LEGO cell constructs a local dummy fleet or local system object when an
  MLSysIM registry object can represent it.
- Stock fleet counts load from `Systems.Clusters`.
- Storage-system inputs load from `Systems.Storage`.
- Scenario-defining workload/use-case inputs load from `Scenarios.*` profiles
  or another typed MLSysIM registry.
- Physical quantities in profiles are Pint quantities, not untyped scalars.
- Scenario profiles carry provenance metadata, even when the provenance is an
  MLSysBook illustrative convention.
- The scenario-input audit is clean except for explicit pure-algebra and
  figure-layout allowlist entries.
- Existing unit/format gates remain green.

## 9. Adjacent Narrative And Layout Backlog

These are not the same as source-of-truth migration, but they should run after
the material is technically correct and before final layout judgment.

### 9.1 Table And Metric Text

- Audit table labels and units for consistency (`Samples/s` vs `Samples/second`,
  `Tokens/s`, `tokens/s`, `Token throughput`, etc.).
- Prefer semantic row labels plus explicit unit columns: metric name, rendered
  unit, interpretation.
- Avoid HTML/PDF split text unless layout truly requires it. If split text is
  needed, the two variants must remain semantically equivalent.
- Review tables that repeat nearby figure/table references in consecutive
  sentences; remove redundant references unless the second reference adds a new
  action for the reader.

### 9.2 Multipliers, Ratios, And Inline Math

- Finish the `fmt_multiple` decision: either keep number-only output and require
  prose `$\times$`, or add `style="symbol"` so the formatter owns the glyph.
- Audit for doubled multiplier prose such as `7x $\times$ speedup`.
- Add or standardize ratio helpers where needed. Use `fmt_ratio` for ratios
  like `41:1` and `fmt_multiple` for multiplicative speedups like `41x`.
- Keep spaces around prose operators such as `$\approx$` when the operator sits
  between inline Python substitutions.

### 9.3 Displayed Calculation Layout

- Audit isolated one-line equations after a prose colon, such as "loading takes
  at least:" followed by a single displayed expression. Decide case by case:
  fold into the sentence, use an aligned calculation block, or convert to a
  small LEGO callout if it is pedagogically important.
- Avoid standalone calculation fragments that look like accidental line breaks
  rather than deliberate textbook math.

### 9.4 Margin Notes, Footnotes, And PDF Layout

- Defer margin-note and footnote layout until after the numerical/prose content
  is stable.
- Use LaTeX logs as a first pass for `Overfull \hbox`, `Underfull \vbox`, and
  badness warnings, but do not rely on logs alone. Margin-note crowding and
  overlap require visual PDF review.
- When a margin note crowds a page, prefer moving or shortening the note before
  weakening the main explanation.
- Add a margin-visual audit after the source-of-truth work is complete. Keep
  strong explanatory drawings, identify margin figures that are weak,
  redundant, cramped, or visually inconsistent, and then improve or add small
  diagrams only where they clarify the adjacent concept. Treat this as an art
  and layout pass, not as part of the LEGO unit migration.

### 9.5 Glossary And Cross-Reference Prose

- Audit generic lines like "For a complete glossary, see @sec-glossary." Use
  that only if the surrounding paragraph genuinely needs a glossary pointer.
  Otherwise connect to the local concept or remove the sentence.
- Audit repeated close references to the same table/figure. Keep the first
  reference, or rephrase the second only if it points to a different reading
  task.

### 9.6 Floating Single-Sentence Paragraphs

- Run a separate single-sentence paragraph audit after the content migrations.
- Some single-sentence paragraphs are good transitions; the cleanup target is
  orphaned explanatory fragments that should be merged, expanded, or removed.
