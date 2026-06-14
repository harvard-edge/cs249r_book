# V2-02 Track Plan: The Compute Infrastructure Wall

## Concept-Module Packet

This packet applies the lab design methodology to Volume II, Chapter 2,
Compute Infrastructure. The lab is not four different track labs. It is one
shared A/B/C/D concept sequence whose selected track changes the stakeholder,
constraints, thresholds, evidence emphasis, failure mode, and final memo
framing.

## Chapter Invariant

Datacenter compute is constrained infrastructure: power, cooling, accelerator
mix, placement, utilization, cost, and carbon are coupled budgets. Peak FLOPs
matter only after the physical and economic envelopes can sustain them.

## Reading Map

| Lab module | Chapter anchor | Claim used in the lab |
|---|---|---|
| Opening | `#sec-compute-infrastructure-infrastructure-walls-4f2c` | Infrastructure walls appear as memory, power, communication, and reliability constraints as systems scale. |
| Part A | `#sec-compute-rack`, `#sec-compute-rack-design`, `#sec-compute-power-wall` | Rack power and cooling limits can reject accelerator counts before peak FLOPs are considered. |
| Part B | `#sec-compute-infrastructure-peak-vs-sustained-throughput-625a`, `#sec-compute-pod`, `#sec-compute-summary` | Sustained throughput and utilization convert nominal capacity into useful work, waste, cost, and carbon. |
| Part C | `#sec-compute-accelerator-selection`, `#sec-compute-infrastructure-accelerator-decision-matrix-8bba`, `#sec-compute-bandwidth-hierarchy` | Accelerator mix and placement change throughput, memory fit, cost, carbon, and communication exposure. |
| Part D | `#sec-compute-summary`, `#sec-compute-fallacies-pitfalls`, `#sec-compute-emerging` | A defensible infrastructure recommendation satisfies simultaneous power, utilization, cost, and carbon guardrails and names the next network/storage implication. |

## Accepted Concepts

- Rack/power/cooling budgets constrain accelerator plans before peak FLOPs do.
- Utilization is the conversion factor between purchased capacity and useful
  work; poor utilization creates economic waste and carbon waste.
- Accelerator mix and placement are coupled decisions: training, inference,
  memory bandwidth, interconnect locality, cost, and carbon do not optimize to
  one universal accelerator.
- Infrastructure recommendations are multi-guardrail decisions. A plan that
  satisfies only one metric is not a plan.
- Fleet amount-system reasoning is required: every decision converts chips into
  racks, racks into kW and cooling load, utilization into useful throughput,
  and placement into cost/carbon/network/storage consequences.

## Rejected Or Deferred Concepts

| Concept | Decision | Reason |
|---|---|---|
| Detailed GPU/TPU microarchitecture history | Rejected for this lab | Important chapter context, but it does not create the strongest A-D consequence chain. |
| Full roofline derivation | Deferred | The lab uses memory/throughput constraints but does not make roofline the primary student task; V1-11 and other labs cover roofline more directly. |
| Tensor/pipeline/data parallel mapping | Deferred to V2-03/V2-05 | This lab carries forward the network implication without making topology design the main activity. |
| Reliability/checkpoint math | Deferred | It is chapter-relevant, but power/utilization/mix/carbon form the tighter compute-infrastructure sequence for this lab. |
| Emerging CXL/optical/wafer-scale technologies | Synthesis note only | Useful as future wall-moving mechanisms, but they distract from the operational planning exercise. |

## Shared Concept Sequence

| UI part | Concept module | Student experience | Evidence produced |
|---|---|---|---|
| Part A | Rack/power/cooling budgets constrain accelerators before peak FLOPs do. | Student predicts the binding rack resource, changes accelerator count and rack density, and sees power/cooling reject plans that have enough nominal FLOPs. | Rack budget table, threshold chart, binding infrastructure budget, failure/recovery banner. |
| Part B | Utilization converts capacity into economics and waste. | Student predicts whether high utilization is always better, sweeps utilization, and sees useful throughput, idle spend, queue risk, and carbon change together. | Utilization frontier, waste/economics table, chosen utilization target. |
| Part C | Accelerator mix/placement changes throughput, memory, cost, and carbon. | Student predicts the best fleet mix, compares all-H100, mixed, efficiency, and offload placements, and identifies the rejected alternative. | Candidate comparison table, throughput-cost-carbon scatter, placement implication. |
| Part D | Infrastructure recommendation must satisfy power, utilization, cost, and carbon guardrails. | Student assembles a capacity plan under simultaneous guardrails and decides whether to revise, reject, or approve it. | Guardrail scorecard, pass/fail recommendation, final memo fields. |
| Synthesis | Compute infrastructure is a coupled budget system. | Student writes a compute infrastructure memo with chosen capacity plan, binding budget, rejected alternative, and network/storage carry-forward implication. | Ledger-ready memo and report export snapshot. |

## Track Narratives

Tracks realize the same concepts with different operating envelopes. The track
does not change the concept sequence.

| Track | Persona | Constraint emphasis | Failure mode | Evidence/report framing |
|---|---|---|---|---|
| iPhone | Mobile product engineer | Thermal envelope, device-tier support, offload cost, privacy, user responsiveness | Minimum supported device tier overheats or pushes too much inference to cloud assist | Memo frames a device-tier/offload infrastructure boundary and names privacy/network dependency. |
| Oura Ring | Wearable firmware engineer | SRAM/flash, tiny battery, duty cycle, phone/cloud assist, OTA payload | Local always-on inference exhausts duty-cycle or storage envelope | Memo frames MCU-plus-assist infrastructure and names buffering/storage implication. |
| RoboTaxi | Autonomous vehicle platform engineer | Vehicle-local power, p99/p999 latency, safety redundancy, sensor bandwidth | More accelerators exceed power/thermal envelope or erode safety headroom | Memo frames vehicle-local capacity and names deterministic network/storage implication. |
| Cloud Fleet | Fleet service owner | Rack power, cooling, utilization, cost/request, carbon, placement | All-H100 dense plan breaches rack/cooling/carbon or runs below utilization threshold | Memo frames accelerator/rack/region capacity and names network fabric/storage staging implication. |

## Concept Modules

### Part A: Concept Module - Rack Power And Cooling Bind First

```yaml
concept_module:
  part_label: "Part A"
  concept_name: "Rack/power/cooling budgets constrain accelerators before peak FLOPs do"
  chapter_claim: "Selecting the fastest accelerator is counterproductive if the cooling infrastructure cannot remove its heat."
  reading_connection:
    chapter_section: "#sec-compute-rack and #sec-compute-power-wall"
    claim_or_formula: "Rack kW = accelerator count x accelerator TDP plus node/network overhead; cooling fails when rack kW exceeds the cooling envelope."
  track_lens:
    stakeholder: "selected track stakeholder"
    system_decision: "how many accelerators/racks are feasible before power/cooling binds"
  student_prior:
    expected_belief: "More accelerators are feasible if they provide enough FLOPs."
    productive_failure: "A plan with enough peak FLOPs violates rack or cooling limits."
  storyline:
    beat_1_scenario: "Stakeholder asks for a capacity plan under a named physical envelope."
    beat_2_prediction: "Student predicts compute, power, cooling, memory, or cost as the first binding budget."
    beat_3_controls: "Student changes accelerators per rack and rack count."
    beat_4_evidence: "Budget chart and table show rack kW, cooling kW, total accelerators, and sustained PFLOP/s."
    beat_5_consequence: "Failure banner names the violated budget and mitigation."
    beat_6_math_peek: "Power and cooling formulas are shown with chapter anchors."
    beat_7_checkpoint: "Student chooses the infrastructure budget to carry forward."
  mechanics:
    controls: ["prediction radio", "accelerators per rack slider", "rack count slider", "cooling tier dropdown"]
    graphs: ["rack kW threshold bar", "fleet amount table"]
    failure_state: "rack_power_kw > cooling_kw_per_rack or fleet_power_kw > site_power_kw"
  depth_gate:
    activity_count: 7
    has_prediction: true
    has_manipulation: true
    has_failure_or_boundary: true
    has_math_peek: true
    has_track_specific_consequence: true
  ledger_output:
    fields: ["partA_prediction", "accelerators_per_rack", "rack_count", "rack_power_kw", "binding_infrastructure_budget"]
    downstream_use: "Part D capacity plan and V2-03 network/storage implication."
```

### Part B: Concept Module - Utilization Converts Capacity Into Economics And Waste

```yaml
concept_module:
  part_label: "Part B"
  concept_name: "Utilization converts capacity into economics and waste"
  chapter_claim: "Capacity planning must use sustained throughput and utilization, not peak specifications."
  reading_connection:
    chapter_section: "#sec-compute-infrastructure-peak-vs-sustained-throughput-625a and #sec-compute-summary"
    claim_or_formula: "Useful throughput = peak capacity x MFU x utilization; idle spend and carbon scale with unused provisioned capacity."
  student_prior:
    expected_belief: "The best target is maximum utilization."
    productive_failure: "High utilization can reduce queue/capacity headroom while low utilization wastes cost and carbon."
  storyline:
    beat_1_scenario: "Operations lead must set a utilization target for the same physical plan from Part A."
    beat_2_prediction: "Student predicts whether low, balanced, or maximum utilization is defensible."
    beat_3_controls: "Student changes utilization target and demand multiplier."
    beat_4_evidence: "Frontier chart shows useful throughput, idle cost, carbon waste, and queue risk."
    beat_5_consequence: "Notebook flags either wasteful underuse or unsafe saturation."
    beat_6_math_peek: "MFU/utilization and cost/carbon formulas are shown."
    beat_7_checkpoint: "Student chooses the utilization target to feed Part D."
  mechanics:
    controls: ["prediction radio", "utilization slider", "demand multiplier slider"]
    graphs: ["utilization frontier", "waste/economics table"]
    failure_state: "utilization below minimum economics threshold or above queue-risk threshold"
  depth_gate:
    activity_count: 7
    has_prediction: true
    has_manipulation: true
    has_failure_or_boundary: true
    has_math_peek: true
    has_track_specific_consequence: true
  ledger_output:
    fields: ["partB_prediction", "utilization_target", "useful_throughput", "idle_cost", "carbon_waste", "utilization_verdict"]
    downstream_use: "Part D utilization and economics guardrails."
```

### Part C: Concept Module - Accelerator Mix And Placement Change The Plan

```yaml
concept_module:
  part_label: "Part C"
  concept_name: "Accelerator mix/placement changes throughput, memory, cost, and carbon"
  chapter_claim: "The right accelerator is the one whose bandwidth, capacity, placement, and workload role jointly satisfy the workload."
  reading_connection:
    chapter_section: "#sec-compute-accelerator-selection, #sec-compute-bandwidth-hierarchy, and #sec-compute-fallacies-pitfalls"
    claim_or_formula: "Plan score depends on sustained throughput, memory fit, cost, carbon, and placement/network penalty."
  student_prior:
    expected_belief: "One best accelerator should run every workload."
    productive_failure: "All-premium or all-efficient fleets fail a guardrail that mixed placement can satisfy."
  storyline:
    beat_1_scenario: "Procurement asks which accelerator mix and placement should be bought or reserved."
    beat_2_prediction: "Student predicts all-premium, mixed, efficient, or offload placement as best."
    beat_3_controls: "Student chooses placement and workload role emphasis."
    beat_4_evidence: "Candidate table/scatter compares throughput, memory margin, cost, carbon, and placement penalty."
    beat_5_consequence: "Notebook names the rejected alternative and why it fails."
    beat_6_math_peek: "Weighted amount-system score and memory/placement formulas are shown."
    beat_7_checkpoint: "Student chooses selected mix and rejected alternative."
  mechanics:
    controls: ["prediction radio", "placement dropdown", "role emphasis dropdown"]
    graphs: ["candidate comparison table", "throughput-cost-carbon scatter"]
    failure_state: "candidate violates memory, cost, carbon, or placement guardrail"
  depth_gate:
    activity_count: 7
    has_prediction: true
    has_manipulation: true
    has_failure_or_boundary: true
    has_math_peek: true
    has_track_specific_consequence: true
  ledger_output:
    fields: ["partC_prediction", "selected_mix", "selected_placement", "rejected_alternative", "mix_binding_reason"]
    downstream_use: "Part D final recommendation and synthesis memo."
```

### Part D: Concept Module - Recommendation Under Simultaneous Guardrails

```yaml
concept_module:
  part_label: "Part D"
  concept_name: "Infrastructure recommendation must satisfy power, utilization, cost, and carbon guardrails"
  chapter_claim: "No layer can be tuned in isolation, because each inherits limits from the one beneath it."
  reading_connection:
    chapter_section: "#sec-compute-summary and #sec-compute-fallacies-pitfalls"
    claim_or_formula: "Feasible = power_ok and utilization_ok and cost_ok and carbon_ok."
  student_prior:
    expected_belief: "The winning Part C mix can be accepted directly."
    productive_failure: "A locally strong mix fails when all guardrails are checked together."
  storyline:
    beat_1_scenario: "A review board asks for a launch-ready infrastructure recommendation."
    beat_2_prediction: "Student predicts which guardrail will still reject the plan."
    beat_3_controls: "Student adjusts capacity margin, carbon region, and procurement stance."
    beat_4_evidence: "Scorecard evaluates power, utilization, cost, and carbon simultaneously."
    beat_5_consequence: "Notebook says approve, revise, or reject and names the binding guardrail."
    beat_6_math_peek: "Boolean guardrail model and cost/carbon formulas are shown."
    beat_7_checkpoint: "Student chooses final memo decision."
  mechanics:
    controls: ["prediction radio", "capacity margin slider", "carbon region dropdown", "procurement stance dropdown", "final decision radio"]
    graphs: ["guardrail scorecard", "memo evidence table"]
    failure_state: "any guardrail boolean is false"
  depth_gate:
    activity_count: 7
    has_prediction: true
    has_manipulation: true
    has_failure_or_boundary: true
    has_math_peek: true
    has_track_specific_consequence: true
  ledger_output:
    fields: ["partD_prediction", "capacity_margin", "carbon_region", "procurement_stance", "final_decision", "final_binding_guardrail"]
    downstream_use: "Synthesis report and V2-03/V2-04 carry-forward implication."
```

## Synthesis

The synthesis is a compute infrastructure memo, not a generic reflection. It
must include:

1. Chosen capacity plan.
2. Binding infrastructure budget.
3. Rejected alternative.
4. Evidence numbers from Parts A-D.
5. Carry-forward network/storage implication.

The memo's durable claim is that infrastructure planning is an amount-system:
accelerator counts become rack kW, cooling load, facility power, sustained
throughput, utilization, cost, carbon, and network/storage obligations.

## Mechanics And Evidence Plan

| Belt | Notebook implementation |
|---|---|
| Opening | Track selector, track mission, chapter invariant, reading map, shared A-D concept sequence. |
| Prediction | Structured `mo.ui.radio` predictions for every part. No free-text prediction gates. |
| Control | Sliders/dropdowns/radio controls for rack density, racks, cooling tier, utilization, demand, placement, role emphasis, margin, region, and procurement stance. |
| Evidence | Plotly charts plus markdown tables with exact values for accessibility. |
| Failure | Reversible failures: rack/cooling breach, utilization waste/saturation, mix guardrail failure, final guardrail rejection. |
| Source | Math Peek accordions name formulas and chapter anchors. |
| Decision | Checkpoint controls for each part and a final infrastructure memo decision. |
| Ledger | `DesignLedger.save` records predictions, controls, evidence numbers, binding guardrails, final decision, and carry-forward implication. |

## Notebook-Local Helper Plan

All new support remains notebook-local and uses the `v2_02_` prefix.

- `v2_02_track_packet(track_id, profile, variant)`: track-specific persona,
  budgets, thresholds, labels, and report framing.
- `v2_02_part_a_state(packet, accelerators_per_rack, rack_count, cooling_tier)`:
  rack/power/cooling amount system.
- `v2_02_part_b_state(packet, part_a, utilization_pct, demand_multiplier)`:
  utilization/economics/waste amount system.
- `v2_02_candidate_rows(packet, part_a, part_b, placement, role_emphasis)`:
  accelerator mix/placement table.
- `v2_02_part_d_state(packet, part_a, part_b, selected, margin_pct, region,
  procurement)`: simultaneous guardrail scorecard.
- `v2_02_metric_cards_html`, `v2_02_fields_html`, and `v2_02_status_badge`:
  local rendering helpers only.

## Ledger Fields

The lab saves:

- selected track and scenario
- Part A prediction, rack density, rack count, cooling tier, rack power, site
  power, binding infrastructure budget
- Part B prediction, utilization target, demand multiplier, useful throughput,
  idle cost, carbon waste, utilization verdict
- Part C prediction, selected mix, placement, role emphasis, rejected
  alternative, mix binding reason
- Part D prediction, margin, carbon region, procurement stance, final guardrail
  status, final decision
- synthesis memo fields: chosen capacity plan, binding infrastructure budget,
  rejected alternative, carry-forward network/storage implication

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part B | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part C | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part D | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Synthesis | 3 | 3 | 3 | 2 | 3 | 2 | Pass |

Traceability score is 2 rather than 3 because this Wave 5 implementation keeps
track thresholds notebook-local where MLSysIM lacks typed infrastructure
scenario objects. Hardware identities still flow through the existing track and
variant registries. The Math Peek/source model explicitly marks formulas and
scenario-owned thresholds so they can migrate to MLSysIM later.

## Implementation Risks

- The current MLSysIM/lab registry has a generic V2-02 system-design variant,
  not a typed compute-infrastructure solver. The notebook therefore uses
  notebook-local scenario constants with explicit source-model text.
- Numeric thresholds are teaching envelopes, not measured production hardware
  traces. The report labels them as planning assumptions.
- Shared helpers are not edited in this wave. The notebook duplicates only the
  small rendering and amount-system helpers required for this lab.
- Other workers may edit other lab files in parallel; this work owns only
  `labs/vol2/lab_02_compute_infra.py` and this track-plan file.
