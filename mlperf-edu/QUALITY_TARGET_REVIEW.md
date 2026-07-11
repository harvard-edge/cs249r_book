# MLPerf EDU Quality Target Review

This document is the evidence ledger for the five score-bearing and three
performance-bearing candidates in the current 30-row registry. The labels are
MLPerf EDU project classifications. They are not approved MLCommons result
categories.

## Evidence Vocabulary

| **State** | **Meaning** |
|:---|:---|
| Implemented | The runner, report fields, gate, and registry protocol exist. |
| Calibrated | One or more local measurements informed the current threshold. Calibration values can expose an implausible target, but they are not release evidence. |
| Validated | A fresh artifact passed its runner gate, provenance verification, grading, and report-level review contract. |
| Release-evidenced | The final clean source revision produced the complete declared reference packet and its immutable digest. None of the candidates has this state until those packets are retained and reviewed. |
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

## Score-Bearing Candidates

| **Workload** | **Metric and Target** | **Recorded Calibration** | **Why the Gate Is Meaningful** | **Release State** |
|:---|:---|:---|:---|:---|
| `nanogpt-train` | cross-entropy loss `<= 2.30` | Seeds 0-4 at `1.9816`-`2.1345`, median `2.0878`, after 25 epochs | Rejects a broken language-model training path on the deterministic Project Gutenberg corpus recipe. | Implemented and five-seed development calibrated. Clean final-revision packet pending. |
| `micro-dlrm-train` | best validation accuracy `>= 0.70` | Seeds 0–4 recorded in the registry at `0.702`–`0.709`, median `0.704` | Checks learned recommendation quality on a fixed MovieLens split. The narrow margin makes raw per-seed review important. | Implemented and five-seed calibrated. Clean packet and MovieLens decision pending. |
| `anomaly-ae-train` | anomaly AUROC `>= 0.95` | Seeds 0–4 at `0.9645`–`0.9701`, median `0.9666`, on 10,000 test examples | Measures anomaly discrimination from per-sample reconstruction errors. Training MSE alone cannot establish detection quality. | Implemented and five-seed development calibrated. Clean packet pending. |
| `resnet18-train` | Fashion-MNIST top-1 accuracy `>= 0.85` | Recorded seeds 0–4 at `0.8630`–`0.8781`, median `0.8750` | Rejects plumbing-only or materially degraded training while retaining a modest laptop and backend margin. | Implemented and five-seed full-test development calibrated. Clean packet pending. |
| `mobilenetv2-train` | Fashion-MNIST top-1 accuracy `>= 0.78` | Seeds 0–4 at `0.7970`–`0.8238`, median `0.8089`; median wall time `96.705` seconds on the recorded MPS system | Replaces the former 70% plumbing threshold with a materially stronger quality floor for the mobile architecture. | Implemented and five-seed full-test development calibrated. Clean packet pending. |

The numbers above reproduce values currently recorded in the registry. They
must be updated from retained evidence rather than copied forward after the
final sweeps. A target is release-ready only when all five individual runs pass
and the median passes.

## Performance-Bearing Candidates

Performance-bearing rows need nonempty work, task or checkpoint quality, and a
repeatable timing protocol. The report-level contract requires an integer seed,
eligible data mode, at least one warmup, at least three measured runs, declared
latency statistics, complete artifacts, and a passing functional gate.
Each current registry row also declares five reference executions. That is
separate from the warmups and repeated timings inside one execution.

| **Workload** | **Functional and Quality Gate** | **Default Measurement** | **Provenance Requirement** | **Release State** |
|:---|:---|:---|:---|:---|
| `nanogpt-inference --variant prefill` | Positive prefill throughput from a quality-approved NanoGPT checkpoint | Three discarded warmups and ten synchronized forward passes; median, p90, p99 | Checkpoint file, SHA-256 digest, source `nanogpt-train` quality dependency | Implemented and locally exercised. Five-execution checkpoint-linked packet pending. |
| `nanogpt-inference --variant decode` | Completes 64 decode steps with positive throughput from the same checkpoint lineage | One discarded warmup and five measured requests; TTFT and inter-token median, p90, p99 | Checkpoint file, SHA-256 digest, source `nanogpt-train` quality dependency | Implemented and locally exercised. Five-execution checkpoint-linked packet pending. |
| `smollm2-chat-inference --variant baseline` | At least eight generated tokens and continuation perplexity `<= 10` on four bundled cases | One warmup and five measured requests; separate prefill and generation median, p90, p99 | Pinned model revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, model metadata, prompt-suite digest | Five development executions all passed at perplexity `7.6005`; output throughput median `74.04` tokens/s, range `73.67`-`87.12`. Clean retained packet pending. |

The NanoGPT timing rows inherit task quality from the training checkpoint. A
random or unidentified checkpoint cannot carry public-candidate performance.
The SLM row evaluates continuation-only negative log likelihood, records the
four-case suite digest, and keeps network access outside the measurement.

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
| slm | `smollm2-chat-inference --variant quantized-int8` | Generation completes, but recorded perplexity `19.8108` and NLL delta `0.9580` fail the current parity limits. |
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
| SLM inference | Approve the model revision, four-case continuation fixture, perplexity limit, token floor, and scenario. | Model dossier, fixture digest, quality results, timing samples. |
| Vision | Accept Fashion-MNIST as the first course-scale public training dataset. | Dataset dossier and both five-seed vision packets. |
| Anomaly detection | Accept the zero-versus-nonzero split, AUROC target, and attribution. | MNIST dossier and five-seed anomaly packet. |
| Recommender | Approve fetch-only MovieLens use or move the row to systems-only and select a replacement. | Dataset terms, audit warning, and DLRM packet. |
| Result language | Define wording that cannot be confused with official competitive MLPerf submissions. | Public rules, sample reports, site, and release notes. |

Until these decisions and the retained packets are complete, the table records
candidates for review rather than released baselines.
