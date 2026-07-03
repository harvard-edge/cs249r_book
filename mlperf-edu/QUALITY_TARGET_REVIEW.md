# MLPerf EDU Quality Target Review

This matrix is the expert-review queue for deciding which workloads can carry
public MLPerf EDU scores and which should remain systems-only teaching or
research workloads.

## Review Rule

- `score-bearing`: target must be backed by real data, a reference protocol,
  reproducible reports, and reviewer approval.
- `performance-bearing`: functional check must prevent empty work while keeping
  the row comparable for systems studies.
- `systems-only`: workload may be excellent for architecture, kernel, backend,
  quantization, pruning, distributed, or agent studies without claiming task
  score comparability.

## Current Matrix

| Suite | Workload | Public status | Metric/check | Target or condition | Review state |
|---|---|---|---|---|---|
| language | `micro-bert-train` | systems-only | `val_accuracy` | 0.78 | Keep systems-only until baseline reaches target on a public dataset path |
| language | `nano-lora-finetune` | systems-only | `base_grad_norm` | frozen base gradients remain zero while LoRA gradients are nonzero | Systems check is acceptable; task-quality promotion deferred |
| language | `nano-moe-train` | systems-only | `cross_entropy_loss` | 0.05 | Target is not currently a public quality claim; needs review before promotion |
| language | `nanogpt-inference --variant fp32-b16` | systems-only | `output_tokens_per_sec` | positive throughput | Optimization row; no public quality claim |
| language | `nanogpt-inference --variant fp16-b16` | systems-only | `output_tokens_per_sec` | positive throughput | Optimization row; no public quality claim |
| language | `nanogpt-inference --variant speculative` | systems-only | `acceptance_rate` | emits configured tokens and records acceptance | TTC row; needs stronger comparability policy before public result use |
| language | `nanogpt-inference --variant prefill` | performance-bearing | `prefill_tokens_per_sec` | checkpoint-backed positive throughput | Review checkpoint lineage and serving scenario |
| language | `nanogpt-inference --variant decode` | performance-bearing | `decode_steps` | configured decode steps complete | Review checkpoint lineage and serving scenario |
| language | `nanogpt-train` | score-bearing | `cross_entropy_loss` | 2.3 | Candidate public row; review Project Gutenberg recipe and reference protocol |
| slm | `smollm2-chat-inference --variant baseline` | performance-bearing | `generated_tokens` | 8 | Candidate public row; review model choice and prompt fixture |
| slm | `smollm2-chat-inference --variant quantized-int8` | performance-bearing | `generated_tokens` | 8 | Candidate public row; review quantization comparability |
| slm | `smollm2-chat-inference --variant batched-b4` | systems-only | `generated_tokens` | 8 | Serving optimization row; public promotion deferred |
| slm | `smollm2-chat-inference --variant long-context` | systems-only | `generated_tokens` | 8 | Long-context systems row; public promotion deferred |
| vision | `micro-diffusion-train` | systems-only | `mse_loss` | 0.002 | Teaching scaffold; public quality target deferred |
| vision | `mobilenet-cifar100-composed-fp16` | systems-only | `effective_compression_ratio` | compressed inference emits logits and ratio > 1 | Optimization row; no public quality claim |
| vision | `mobilenetv2-train` | score-bearing | `top1_accuracy` | 0.7 | Candidate public row; review Fashion-MNIST target and baseline evidence |
| vision | `resnet18-train` | score-bearing | `top1_accuracy` | 0.75 | Candidate public row; review Fashion-MNIST target and baseline evidence |
| recommender | `micro-dlrm-dram-train` | systems-only | `accuracy` | 0.65 | Memory-pressure row; public promotion deferred |
| recommender | `micro-dlrm-train` | score-bearing | `accuracy` | 0.7 | Blocked for public endorsement until MovieLens policy is resolved or dataset is replaced |
| tiny | `anomaly-ae-train` | score-bearing | `reconstruction_mse` | 0.04 | Candidate public row; review MNIST attribution and threshold protocol |
| tiny | `dscnn-kws-train` | systems-only | `top1_accuracy` | 0.9 | Keep systems-only until Speech Commands path and target are reviewed |
| tiny | `wake-vision-vww` | systems-only | `binary_accuracy` | 0.85 | Keep systems-only until Wake Vision/proxy policy is reviewed |
| agent | `nano-codegen-agent` | systems-only | `pass_at_1` | 0.15 | Agent systems row; public quality methodology deferred |
| agent | `nano-rag-agent` | systems-only | `retrieval_accuracy` | 0.8 | Agent systems row; corpus and quality policy deferred |
| agent | `nano-react-agent` | systems-only | `trace_accuracy` | 0.6 | Agent systems row; trace provenance and quality policy deferred |
| agent | `nano-toolcall-agent` | systems-only | `valid_call_rate` | valid calls and positive throughput | Agent systems row; tool schema policy deferred |
| distributed | `micro-dlrm-distributed` | systems-only | `relative_loss_delta` | 0.05 | Distributed systems row; no public quality claim |
| graph | `micro-gnn-train` | systems-only | `test_accuracy` | 0.78 | Keep systems-only until dataset/source/target review |
| timeseries | `micro-lstm-train` | systems-only | `val_mse` | 0.13 | Keep systems-only until dataset/source/target review |
| rl | `micro-rl-train` | systems-only | `avg_episode_reward` | 195 | Keep systems-only until stochastic-run policy is defined |

## Reviewer Sign-Off Needed

| Area | Reviewers | Output |
|---|---|---|
| SLM serving | ML systems + MLCommons inference reviewers | approve model, prompts, decode token target, and quantized variant comparability |
| Vision training | vision systems + education reviewers | approve Fashion-MNIST as the first public teaching dataset |
| Recommender | recommender + policy reviewers | choose MovieLens approval path or replacement dataset |
| Tiny/anomaly | embedded/tiny reviewers | approve MNIST anomaly threshold and public attribution |
| Language training | language modeling reviewers | approve Project Gutenberg text recipe and NanoGPT target protocol |
