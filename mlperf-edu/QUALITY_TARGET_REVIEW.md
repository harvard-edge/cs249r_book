# MLPerf EDU Quality Target Review

This document is the evidence ledger for the five score-bearing and three
performance-bearing candidates in the current 30-row registry. The labels are
MLPerf EDU project classifications. They are not approved MLCommons result
categories.

The eight retained summaries were produced from clean source commit
`b4366b7614f0bb8ba0a1d6224832d4caea64e68a`. They are current local-handoff
evidence for the present contracts.

## Evidence Vocabulary

| **State** | **Meaning** |
|:---|:---|
| Implemented | The runner, report fields, gate, and registry protocol exist. |
| Calibrated | One or more local measurements informed the current threshold. Calibration values can expose an implausible target, but they are not release evidence. |
| Validated | A fresh artifact passed its runner gate, provenance verification, grading, and report-level review contract. |
| Committed-summary | A clean source revision produced a complete create-once reference attempt, and the repository retains its exact summary and SHA-256 digest. The eight retained packets from `b4366b7614f0bb8ba0a1d6224832d4caea64e68a` are current local-handoff evidence. |
| Release-evidenced | A reviewer has received and verified the committed summary and complete raw packet at an agreed handoff or publication location. The current raw packets are available for local handoff but have not yet been transferred or assigned public URLs. |
| Externally approved | Dataset policy, target rationale, naming, and result wording have the required reviewer decisions. This state cannot be granted by repository tests. |

## Five-Seed Score Policy

Every score-bearing candidate declares the same release protocol.

- Run the `max` profile in five fresh processes with seeds `0,1,2,3,4`.
- Use eligible real data with no synthetic fallback.
- Require every run to report the requested seed, pass its individual target,
  verify its manifest, and pass grading.
- Aggregate the five quality values with the median and require that median to
  meet the registry target.
- Publish the median, mean, minimum, maximum, and standard deviation. Keep every
  JSON, HTML, CSV, manifest, checkpoint, and runner-declared artifact.
- Index artifacts with SHA-256 and byte size. Store the evidence summary with
  its adjacent unauthenticated SHA-256 digest.
- Create a new full five-seed attempt if any run fails or times out. Never
  replace one seed inside an existing attempt.
- Rerun all five seeds after model, data preprocessing, optimizer schedule,
  relevant framework version, or target machine-class changes.

`tools/run_reference_sweep.py` enforces this product path. Public-candidate
evidence must start from a clean source snapshot. Its default evidence root is
outside the checkout under `~/.mlperf-edu/reference_runs` so generated output
does not contaminate that source snapshot.

## Current Reference Snapshot

`reference_results/index.json` records eight summaries from clean source commit
`b4366b7614f0bb8ba0a1d6224832d4caea64e68a`. The public-candidate
repeatability limit is `5%` coefficient of variation for timed performance
references.

| **Workload** | **Evidence ID** | **Summary SHA-256** | **Primary Metric Median** | **Minimum** | **Maximum** | **CV** |
|:---|:---|:---|---:|---:|---:|---:|
| `anomaly-ae-train` | `anomaly-ae-train_max_20260711T185950.479514Z` | `036fb7f1f6cbef38f5c229cb08f16555a8086cd9af61b944925032b9aa6f22c7` | `3.1185` | `3.0778` | `3.2159` | n/a |
| `micro-dlrm-train` | `micro-dlrm-train_max_20260711T185902.723780Z` | `b5419752fd507afbef073063323c253d9afd2508cca35228e2353f4e78cd4dc5` | `1.8956` | `1.8684` | `1.9025` | n/a |
| `mobilenetv2-train` | `mobilenetv2-train_max_20260711T190411.719846Z` | `89ab833b133570f89144bb88644ed235e6f33e8d24875739789575f09fee6fb8` | `57.7264` | `57.5393` | `58.1860` | n/a |
| `nanogpt-decode` | `nanogpt-decode_max_20260711T191026.069877Z` | `0179a12d1e45afc11a9584f31cffc80e2104cbd638533fd82ab12a2a30b5b391` | `124.6054` | `123.4936` | `125.0866` | `0.47%` |
| `nanogpt-prefill` | `nanogpt-prefill_max_20260711T190945.837856Z` | `38118741360d53028e1e6977eba95eff33c7830aa382e45ee4744618e7938d83` | `122609.21` | `120209.46` | `122699.81` | `0.92%` |
| `nanogpt-train` | `nanogpt-train_max_20260711T185153.818986Z` | `9dee70ef74c70bae260679d1920395cd504c32ed2bfe33cdbe392d5bf772c13d` | `73.0170` | `70.8778` | `73.5426` | n/a |
| `resnet18-train` | `resnet18-train_max_20260711T190054.049258Z` | `b4318ae7f41e645a7260c0948e6861629d8e5e56b2f6346e5a89ddc75c7db30c` | `31.1051` | `30.3761` | `31.9132` | n/a |
| `slm-decode` | `slm-decode_max_20260711T191209.721134Z` | `74bd019ccf8f8551174ec95df3ba60af8849b523ec20d9c34263c72467496dc8` | `102.7963` | `97.3773` | `104.7385` | `2.95%` |

## Score-Bearing Candidates

| **Workload** | **Metric and Target** | **Current Calibration Boundary** | **Why the Gate Is Meaningful** | **Release State** |
|:---|:---|:---|:---|:---|
| `nanogpt-train` | cross-entropy loss `<= 2.30` | Existing bounded calibration supports the unchanged target. | Rejects a broken language-model training path on the deterministic Project Gutenberg corpus recipe. | Current five-run packet committed. |
| `micro-dlrm-train` | fixed-final-epoch ROC AUC `>= 0.76` | Five bounded seeds reached `0.7671`-`0.7696` on the official split without label-derived aggregate features. | Tests recommendation ranking on an untouched fixed evaluation split without checkpoint selection on test labels. | Current five-run packet committed; raw MovieLens-derived artifacts remain local-only pending policy review. |
| `anomaly-ae-train` | macro AUROC `>= 0.93`, worst-class AUROC `>= 0.90`, learned-control margin `>= 0.20` | Five bounded seeds reached macro AUROC `0.9370`-`0.9422`, worst-class AUROC `0.9132`-`0.9212`, and minimum control margin `0.2491`-`0.2570`. | Requires classwise reconstruction-error discrimination and measurable improvement over zero, centroid, and untrained controls. | Current five-run packet committed. |
| `resnet18-train` | Fashion-MNIST top-1 accuracy `>= 0.85` | Existing bounded calibration supports the unchanged target. | Rejects plumbing-only or materially degraded training while retaining a modest laptop and backend margin. | Current five-run packet committed. |
| `mobilenetv2-train` | Fashion-MNIST top-1 accuracy `>= 0.78` | Existing bounded calibration supports the unchanged target. | Provides a second architecture-level vision quality check at notebook scale. | Current five-run packet committed. |

Calibration values above are target rationale only. Current result claims must
cite the committed summaries listed in the current reference snapshot.

## Performance-Bearing Candidates

Performance-bearing rows need nonempty work, task or checkpoint quality, and a
repeatable timing protocol. The report-level contract requires an integer seed,
eligible data mode, at least one warmup, at least three measured runs, declared
latency statistics, complete artifacts, and a passing functional gate.
Each current registry row also declares five reference executions. That is
separate from the warmups and repeated timings inside one execution. Promotion
also requires the sample coefficient of variation across those five primary
performance values to be no greater than `0.05`.

| **Workload** | **Functional and Quality Gate** | **Default Measurement** | **Provenance Requirement** | **Release State** |
|:---|:---|:---|:---|:---|
| `nanogpt-inference --variant prefill` | Positive prefill throughput from a quality-approved NanoGPT checkpoint | Fixed content-addressed prompt, fresh KV cache, three discarded warmups, and 20 synchronized forward passes; median, p90, p99 | Checkpoint file, SHA-256 digest, source `nanogpt-train` quality dependency | Current five-run packet committed; CV `0.92%`. |
| `nanogpt-inference --variant decode` | Completes 64 decode steps with positive throughput from the same checkpoint lineage | Three discarded warmups and 20 single-stream requests; causal TTFT, first-decode latency, and subsequent ITL statistics | Checkpoint file, SHA-256 digest, source `nanogpt-train` quality dependency | Current five-run packet committed; CV `0.47%`. |
| `smollm2-chat-inference --variant baseline` | At least eight generated tokens, token-weighted continuation perplexity `<= 7`, and worst-category perplexity `<= 24` on 28 attributed cases | Three warmups and 20 measured requests; separate prefill and generation median, p90, p99 | Pinned model revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, model metadata, v2 fixture version and digest, case count, categories, and aggregation | Current five-run packet committed; CV `2.95%`. |

The NanoGPT timing rows inherit task quality from the training checkpoint. A
random or unidentified checkpoint cannot carry public-candidate performance.
Both rows use the same verified training checkpoint lineage selected from the
current NanoGPT training sweep.
The SLM row evaluates continuation-only negative log likelihood, weights every
continuation token equally, records the attributed 28-case v2 suite digest, and
keeps network access outside the measurement. A separate weakest-category gate
prevents easy categories from masking material degradation.

## Systems-Only Rows

The remaining 22 registry rows are useful for coursework and systems research,
but their metrics are not public score claims. Each row now declares a
machine-readable `max_execution` boundary. The generated site states the
reported data mode, asset use, and quality enforcement, including when the
declared research dataset is not consumed by the current systems-only runner.

| **Suite** | **Row** | **Current Boundary** |
|:---|:---|:---|
| agent | `nano-codegen-agent` | Local synthetic task and agent-systems scaffold; capability evaluation deferred. |
| agent | `nano-rag-agent` | Local retrieval and generation scaffold; corpus and quality policy deferred. |
| agent | `nano-react-agent` | Trace-execution scaffold; trace provenance and capability methodology deferred. |
| agent | `nano-toolcall-agent` | Structured-call systems check; tool-schema evaluation policy deferred. |
| distributed | `micro-dlrm-distributed` | Local Gloo and gradient-equivalence systems study, not a public quality score. |
| graph | `micro-gnn-train` | Public dataset, split, and target evidence need review. |
| language | `micro-bert-train` | Current path has not established a public SST-2 target and source policy. |
| language | `nano-lora-finetune` | Checks frozen-base and active-adapter gradients, not downstream task quality. |
| language | `nano-moe-train` | Deterministic micro-scale optimization row; the low loss threshold is not approved public evidence. |
| language | `nanogpt-inference --variant fp16-b16` | Precision systems comparison without a public quality-parity contract. |
| language | `nanogpt-inference --variant fp32-b16` | Batch decode systems comparison without the checkpoint review contract used by the candidate decode row. |
| language | `nanogpt-inference --variant speculative` | Speculative-decode systems row; acceptance and task-quality comparability need a stronger policy. |
| recommender | `micro-dlrm-dram-train` | Memory-pressure experiment with virtual embeddings, not a public recommender score. |
| rl | `micro-rl-train` | Stochastic-control methodology and repeated-run policy are not defined for public scoring. |
| slm | `smollm2-chat-inference --variant batched-b4` | Batched-serving research variant without a promoted comparison contract. |
| slm | `smollm2-chat-inference --variant long-context` | Long-context research variant without a promoted comparison contract. |
| slm | `smollm2-chat-inference --variant quantized-int8` | The historical v1 run completed generation but recorded perplexity `19.8108` and NLL delta `0.9580`. Those values and their parity decision are protocol-superseded; v2 calibration is required. |
| timeseries | `micro-lstm-train` | Synthetic micro-shard systems path; real-data target and source review deferred. |
| tiny | `dscnn-kws-train` | Fast paths use synthetic micro-shards; Speech Commands scoring protocol is not promoted. |
| tiny | `wake-vision-vww` | Fast paths use synthetic micro-shards; Wake Vision scoring and release policy are not promoted. |
| vision | `micro-diffusion-train` | Teaching-scale denoising scaffold without an approved generative-quality metric. |
| vision | `mobilenet-cifar100-composed-fp16` | Compression-composition correctness row without a public task-quality parity gate. |

## Reviewer Decisions

| **Area** | **Decision Required** | **Evidence to Review** |
|:---|:---|:---|
| Target robustness | Accept or revise each metric, threshold, five-seed rule, and rerun trigger. | Five clean score packets plus target margins and raw values. |
| NanoGPT inference | Approve checkpoint lineage, prompt shape, prefill and decode scenarios, and latency statistics. | Training packet, checkpoint digest, prefill packet, decode packet. |
| SLM inference | Approve the model revision, attributed 28-case continuation fixture, token-weighted aggregation, dual perplexity limits, token floor, and scenario. | Model dossier, fixture digest, per-case and per-category quality results, timing samples. |
| Vision | Accept Fashion-MNIST as the first course-scale public training dataset. | Dataset dossier and both five-seed vision packets. |
| Anomaly detection | Accept the zero-versus-nonzero split, AUROC target, and attribution. | MNIST dossier and five-seed anomaly packet. |
| Recommender | Approve fetch-only MovieLens use or move the row to systems-only and select a replacement. | Dataset terms, audit warning, and DLRM packet. |
| Result language | Define wording that cannot be confused with official competitive MLPerf submissions. | Public rules, sample reports, site, and release notes. |

The eight summaries are committed candidates for review, not released or
MLCommons-approved baselines. Complete raw packets remain available by local
handoff, public artifact URLs remain unassigned, and the reviewer decisions
above remain open.
