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
| Committed-summary | A clean source revision produced a complete create-once reference attempt, and the repository retains its exact summary and SHA-256 digest. All eight current candidates have this state at source commit `0ec4d3e1c415944227d0754d170edb0addc1d925`. |
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

## Committed Reference Snapshot

`reference_results/index.json` records eight valid, review-eligible summaries
from clean source commit `0ec4d3e1c415944227d0754d170edb0addc1d925`.
The repository stores the compact summaries and their content digests. Complete
reports, manifests, checkpoints, model metadata, and raw timing samples remain
available by local handoff. Thirty-five policy-permitted run packages have also
been created and verified. Packaging correctly rejected the five DLRM runs
because the MovieLens-derived artifacts are not redistributable under the
current policy. No row has a public raw-artifact URL yet.

| **Workload** | **Evidence ID** | **Summary SHA-256** |
|:---|:---|:---|
| `nanogpt-train` | `nanogpt-train_max_20260711T083347.154092Z` | `6f58270368d1e75445a7c7bcc8c20ca710bb9994090aa4705440525ef8cc0638` |
| `micro-dlrm-train` | `micro-dlrm-train_max_20260711T085501.915367Z` | `2893278fccc3715c6237b50aeb889d05a3f988cecdc7a8e9660dba121edf8a28` |
| `anomaly-ae-train` | `anomaly-ae-train_max_20260711T085421.359195Z` | `a3393e127285bbb9dcba5af692a7cbd0105df1f6a25577c1b51fd9d491c27803` |
| `resnet18-train` | `resnet18-train_max_20260711T084315.135227Z` | `fa08884c733c616157cac879c46c9833bb1f5c9ade1dfb45648658d05c491aa8` |
| `mobilenetv2-train` | `mobilenetv2-train_max_20260711T084704.168976Z` | `936916009701875a9df311cc486230946609bd91357e7ff6e8686505aa3315e0` |
| `nanogpt-prefill` | `nanogpt-prefill_max_20260711T084140.159367Z` | `bc3e8f01c279d1d2bbf0f8a24b15e85270584a5d35e16883a9751b4b5a04b68b` |
| `nanogpt-decode` | `nanogpt-decode_max_20260711T084155.577249Z` | `bae464def8db558afcd377d506f9a25098b58d3ad934e03b474fa8b534beddf7` |
| `slm-decode` | `slm-decode_max_20260711T085533.624561Z` | `c13f7b7afb626cd4f3cdcb9620693a95ce8d46881d1e8c6f18ba0234442f1185` |

## Score-Bearing Candidates

| **Workload** | **Metric and Target** | **Committed Reference Result** | **Why the Gate Is Meaningful** | **Release State** |
|:---|:---|:---|:---|:---|
| `nanogpt-train` | cross-entropy loss `<= 2.30` | Seeds 0-4 at `1.9744`-`2.1648`, median `2.0789`, after 25 epochs | Rejects a broken language-model training path on the deterministic Project Gutenberg corpus recipe. | Committed-summary; all five runs and the median passed. |
| `micro-dlrm-train` | best validation accuracy `>= 0.70` | Seeds 0-4 at `0.7019`-`0.7094`, median `0.7041` | Checks learned recommendation quality on a fixed MovieLens split. The narrow margin makes raw per-seed review important. | Committed-summary; all five runs and the median passed. MovieLens policy remains external. |
| `anomaly-ae-train` | anomaly AUROC `>= 0.95` | Seeds 0-4 at `0.9645`-`0.9701`, median `0.9666`, on 10,000 test examples | Measures anomaly discrimination from per-sample reconstruction errors. Training MSE alone cannot establish detection quality. | Committed-summary; all five runs and the median passed. |
| `resnet18-train` | Fashion-MNIST top-1 accuracy `>= 0.85` | Seeds 0-4 at `0.8630`-`0.8781`, median `0.8750` | Rejects plumbing-only or materially degraded training while retaining a modest laptop and backend margin. | Committed-summary; all five runs and the median passed. |
| `mobilenetv2-train` | Fashion-MNIST top-1 accuracy `>= 0.78` | Seeds 0-4 at `0.7970`-`0.8238`, median `0.8089`; median wall time `85.335` seconds on the recorded MPS system | Replaces the former 70% plumbing threshold with a materially stronger quality floor for the mobile architecture. | Committed-summary; all five runs and the median passed. |

The values above come from the committed summaries rather than copied
calibration notes. Every individual run and each five-run median passed. Any
change to executable source, preprocessing, training schedule, framework
contract, or target machine class requires a new create-once sweep.

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
| `nanogpt-inference --variant prefill` | Positive prefill throughput from a quality-approved NanoGPT checkpoint | Three discarded warmups and ten synchronized forward passes; median, p90, p99 | Checkpoint file, SHA-256 digest, source `nanogpt-train` quality dependency | Committed-summary; five executions passed, median `124578.18` tokens/s, range `114052.80`-`125183.01`, CV `4.60%`. |
| `nanogpt-inference --variant decode` | Completes 64 decode steps with positive throughput from the same checkpoint lineage | Three discarded warmups and 20 measured requests; TTFT and inter-token median, p90, p99 | Checkpoint file, SHA-256 digest, source `nanogpt-train` quality dependency | Committed-summary; five executions completed 64 steps, median `131.8344` tokens/s, range `127.1281`-`133.7293`, CV `2.09%`. |
| `smollm2-chat-inference --variant baseline` | At least eight generated tokens and continuation perplexity `<= 10` on four bundled cases | Three warmups and 20 measured requests; separate prefill and generation median, p90, p99 | Pinned model revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, model metadata, prompt-suite digest | Committed-summary; all five executions passed the token and perplexity gates, with median `101.3853` tokens/s, range `100.4041`-`102.4841`, CV `0.86%`. |

The NanoGPT timing rows inherit task quality from the training checkpoint. A
random or unidentified checkpoint cannot carry public-candidate performance.
Both rows use the median-quality seed-4 training checkpoint, SHA-256
`a0d2f31a747355d47d11c6aa77eb09faf2232f84cb519accb286a78159fb2d8a`,
from the verified lineage package with SHA-256
`0b0173d78e2c3315c4687b6319beb8a2826c98bce7f52710542f4b496edadd20`.
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

The eight summaries are committed candidates for review, not released or
MLCommons-approved baselines. Complete raw packets remain available by local
handoff, public artifact URLs remain unassigned, and the reviewer decisions
above remain open.
