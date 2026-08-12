# Margin Figure Caption Alignment Audit

Date: 2026-06-02

Scope: all 224 SVG margin figures referenced from QMD `.column-margin` blocks
after the expanded margin-device pass.

## Method

The audit used `inventory_margin_figures.py` for the placement inventory and
`audit_margin_caption_alignment.py` for a caption/prose packet. The alignment
script captures the nearest paragraph before and after each margin block and
uses a word-overlap score only to triage likely review cases. Final decisions
were made by reading each flagged caption in local chapter context.

Commands:

```bash
python3 book/tools/scripts/margin_figures/inventory_margin_figures.py \
  --format csv \
  --output /tmp/mlsysbook-margin-inventory.csv
python3 book/tools/scripts/margin_figures/audit_margin_caption_alignment.py \
  --markdown /tmp/mlsysbook-caption-alignment-full.md \
  --csv /tmp/mlsysbook-caption-alignment.csv
python3 book/tools/scripts/margin_figures/audit_margin_caption_alignment.py \
  --review-only \
  --markdown /tmp/mlsysbook-caption-alignment-review.md
python3 book/tools/scripts/margin_figures/audit_margin_caption_alignment.py \
  --review-threshold 0.30 \
  --review-only \
  --markdown /tmp/mlsysbook-caption-alignment-strict-review.md
python3 book/tools/scripts/margin_figures/render_margin_reader_link_audit.py
```

## Results

| Check | Result |
|---|---:|
| Referenced SVG margin figures inventoried | 224 |
| Missing captions | 0 |
| Title-like captions | 0 |
| Automated pass-triage captions | 200 |
| Captions manually reviewed for low lexical overlap | 24 |
| Strict narrative-click captions manually reviewed (`--review-threshold 0.30`) | 58 |
| Reader-link markdown packet entries generated | 224 |
| Reader-alignment verdict rows recorded | 224 |
| Caption/prose bridge edits made | 1 |
| Caption wording changes required | 0 |
| Remaining known caption/prose alignment issues | 0 |

The full caption list was also scanned for capitalization and sentence-case
consistency. The placed captions are short declarative takeaways rather than
titles, legends, or alt-text duplicates.

The stricter narrative-click pass checked each low-overlap row against three
conditions: the adjacent prose introduces the same engineering claim, the visual
encodes that claim directly, and the caption states the reader takeaway. Most
strict-review rows were false positives caused by formulas, code-derived
variables, tables, or captions that intentionally translate nearby math into a
plain-language takeaway.

The inspectable reader-link packet is
`book/tools/audit/margin_figure_reader_link_audit.md`. It is the artifact to open
when asking how a particular margin figure maps to the source markdown: every
entry shows the exact QMD line, the raw `.column-margin` block, the embedded SVG,
the caption, the `fig-alt`, and the nearest prose anchors.

The companion verdict record is
`book/tools/audit/margin_figure_reader_alignment_verdicts.md`. It records the
author-facing pass/fix judgment for every placed margin figure: 224 pass, 0
remaining changes needed. The single weak connection found during this audit
cycle was repaired with the Vol. 1 conclusion reliability bridge sentence before
`vol1_conclusion_fleet_mtbf_ladder.svg`.

## Change Made

| Chapter | Asset | Decision | Reason |
|---|---|---|---|
| `vol1/conclusion` | `vol1_conclusion_fleet_mtbf_ladder.svg` | Added one prose bridge sentence before the figure. | The caption and figure were correct, but the surrounding paragraph moved from "fleet-scale infrastructure" to the MTBF ladder without explicitly naming reliability. The new sentence makes the reader connection direct: independent components accumulate, so rare failures become routine fleet events. |

## Manual Review Notes

These rows were flagged by lexical overlap but passed manual editorial review.
The common causes were equation-heavy prose, code-derived variables, tables, or
captions that state the visual takeaway using clearer wording than the paragraph.

| Chapter | Asset | Caption | Editorial decision |
|---|---|---|---|
| `vol1/conclusion` | `vol1_conclusion_margin_002.svg` | Decode stays memory-bound, left of the roofline ridge. | Pass. The preceding systems-insight paragraph explicitly states that memory time dominates compute time and names the memory-bound regime. |
| `vol1/data_engineering` | `vol1_data_engineering_margin_001.svg` | Small per-window false-positive rates compound into operational failure. | Pass. The paragraph explains continuously listening KWS windows and compounding false positives. |
| `vol1/ml_workflow` | `vol1_ml_workflow_margin_004.svg` | Cloud and edge lifetime costs cross as scale grows. | Pass. The adjacent systems insight and cost example frame edge payback and the cloud/edge deployment trade-off. |
| `vol1/model_compression` | `vol1_model_compression_margin_004.svg` | Compression hits an end-to-end ceiling once non-model work dominates the pipeline. | Pass. The preceding warning explains why FLOP reduction does not translate directly to end-to-end latency reduction. |
| `vol1/model_serving` | `model_serving_blast_radius.svg` | One noisy neighbor perturbs every workload sharing the node. | Pass. The figure bridges the operating-system variability paragraph into the noisy-neighbor explanation. |
| `vol1/nn_computation` | `vol1_nn_computation_margin_001.svg` | Learned features buy accuracy by spending far more arithmetic. | Pass. The nearby paradigm recap gives the operation-count escalation from rule-based comparisons through HOG to neural MACs. |
| `vol1/responsible_engr` | `responsible_engr_blast_radius_sepsis.svg` | One upstream change; many silently affected. | Pass. The sepsis example immediately after the figure describes an EHR workflow change affecting recommendations without alarms. |
| `vol2/collective_communication` | `collective_communication_ring_tree_divergence.svg` | Ring latency grows with N; tree stays logarithmic. | Pass. The surrounding Ring AllReduce performance formula and following Tree AllReduce discussion explain the divergence. |
| `vol2/conclusion` | `vol2_conclusion_margin_001.svg` | Decode sits memory-bound, left of the roofline ridge. | Pass. The paragraph names memory bandwidth as the gating constraint for decode. |
| `vol2/conclusion` | `vol2_conclusion_margin_004.svg` | A datacenter draws megawatts; the brain runs on ~20 watts. | Pass. The paragraph introduces the brain/datacenter Fermi comparison and the notebook works the order-of-magnitude contrast. |
| `vol2/data_storage` | `data_storage_checkpoint_storm_write_time.svg` | Sharding collapses checkpoint-storm write time by two orders of magnitude. | Pass. The preceding paragraph introduces simultaneous GPU checkpoint writes; the following definition names checkpoint storms. |
| `vol2/distributed_training` | `vol2_distributed_training_margin_001.svg` | Past a communication-to-compute threshold, scaling stops being ideal. | Pass. The paragraph defines the communication-computation ratio and the bullets distinguish compute-bound from communication-bound regimes. |
| `vol2/fault_tolerance` | `fault_tolerance_detection_ladder.svg` | Detection latency climbs from seconds (crash) to hours (silent corruption). | Pass. The figure opens the failure-detection section; the following prose explains timeout trade-offs and silent data corruption. |
| `vol2/inference` | `inference_kv_cache_ladder.svg` | The KV cache fills HBM, capping concurrent requests. | Pass. The surrounding prose explains why long-context requests require sharding and turns the formula into a batch-size limit. |
| `vol2/inference` | `inference_decode_roofline.svg` | Decode is memory-bound; parallel verification moves work toward the ridge. | Pass. The paragraph introduces speculative decoding; the following phases explain parallel verification as the arithmetic-intensity shift. |
| `vol2/inference` | `vol2_inference_margin_003.svg` | Two-choice routing flattens the tail imbalance of random placement. | Pass. The surrounding text states the exponential improvement and gives the random-versus-two-choice queue-depth contrast. |
| `vol2/inference` | `inference_quantization_capacity_ladder.svg` | INT4 turns a two-GPU model into a one-GPU candidate. | Pass. The section lead explicitly states that a 140 GB FP16 model requiring two A100s becomes a 35 GB INT4 representation that fits on one GPU. |
| `vol2/introduction` | `vol2_introduction_reliability_knee.svg` | Fleet reliability collapses as node count climbs. | Pass. The adjacent GPT-4-class training scenario derives cluster MTBF from many GPUs and states that the system is always in partial failure. |
| `vol2/network_fabrics` | `network_fabrics_physical_reach_ladder.svg` | Each extra meter pushes the fabric from copper toward optics. | Pass. The optical-interconnect section immediately explains copper reach limits and the transition to fiber/photons. |
| `vol2/ops_scale` | `vol2_ops_scale_margin_001.svg` | Streaming closes the freshness lag that batch leaves open. | Pass. The feature-freshness example contrasts daily batch freshness with second-scale streaming updates. |
| `vol2/performance_engineering` | `vol2_performance_engineering_margin_002.svg` | Larger batches raise arithmetic intensity toward the ridge. | Pass. The formula and following paragraph state that batch size moves decode from memory-bound toward compute-bound utilization. |
| `vol2/robust_ai` | `robust_ai_psi_drift_knee.svg` | Drift stays benign until the index crosses a threshold; past the knee, reliability degrades fast. | Pass. The decision matrix and framework use PSI/KL thresholds to decide monitor versus retrain. |
| `vol2/security_privacy` | `security_privacy_output_leakage_ladder.svg` | Full distributions leak far more than top-k outputs. | Pass. The top-k truncation paragraph quantifies the information removed when returning only top-k classes. |
| `vol2/sustainable_ai` | `sustainable_ai_grid_interconnection_ladder.svg` | Substation lead time can run 4x longer than GPU procurement. | Pass. The interconnection-queue notebook explicitly compares GPU lead time with substation lead time and computes the 4x lag. |

## Bottom Line

The margin captions and local prose now line up. The only actual editorial gap
was a missing bridge sentence before the Vol. 1 conclusion fleet-MTBF ladder; no
caption text needed to change.
