# Published request timings

`argonne-llama2-70b-h100.csv` preserves the selected rows from Argonne's
LLM-Inference-Bench `Plots/All_results.csv`. The adjacent provenance file records
the immutable source revision, source hash, and selection criteria. Column names
and source values are unchanged.

These rows report whole-request latency for four H100 GPUs, batch size one,
and Llama-2-70B. Dividing latency by output length yields a request-average time
per output token, including prefill. The source does not provide time to first
token for these rows, so this subset cannot establish decode inter-token latency,
a decode prediction error, or sustained decode bandwidth utilization.

`../scripts/paper_scenarios.py` loads the subset. The paper generator computes
the displayed request averages; the validation script labels the metric mismatch
and does not impose a decode-error gate. These published measurements are
separate from local Python execution timings in `../generated/timing_samples.json`.

# Published decode latencies

`nim-llama-70b-concurrency1.csv` preserves every concurrency-1 row for the 70B
Llama models on NVIDIA's NIM benchmarking page (documentation version 1.0.0).
The adjacent provenance file records the page URL, retrieval date, and the
SHA-256 of the retrieved page and of the metrics page that defines ITL. Model,
NIM version, GPU count and type, precision, tensor parallelism, and sequence
lengths come from the section headings above each table; the four measured
columns are the table cells, unchanged. `../scripts/check_nim_source.py` re-reads
the page (or a saved copy) and checks every row against it.

NIM reports inter-token latency as `(e2e_latency - TTFT) / (output_tokens - 1)`,
so these rows do isolate decode. They are Llama-3.3-70B (and, at FP8 only,
Llama-3.1-70B), not Llama-2-70B. `../scripts/paper_scenarios.py` compares the
16-bit rows with the weight-read floor for `Models.Language.Llama3_70B`; the
validation script checks that every floor lies below its measured ITL. Like the
Argonne rows, these are measurements of particular serving releases: the same
four-H100 configuration reports different ITL under NIM 1.5.0 and 1.8.0.

# Measured GPU board energy

`gpu-energy/` has a declared measurement plan and GPU board-energy
measurements for single-GPU Llama-3.1-8B inference (vLLM) and ResNet-50
training on H100 and A100 boards. Unlike the rows above, these measurements
are made with committed code. `../scripts/measure_gpu_energy.py` records every
trial and hashes it into a manifest, and `../scripts/analyze_gpu_energy.py`
verifies the runs and compares their energy, through `RunFootprintModel`, with
the package's two device-power assumptions: rated TDP and `Engine.solve`'s
energy-proportional rule. `gpu-energy/README.md` documents the protocol and
the measurement boundary.

`serverless-gpu-energy/` has per-run GPU energy measurements from Modal
serverless GPUs (T4 to B200): idle power with a model loaded, cold starts,
keep-warm windows, pausing a training job, and 148 power runs of seven
workloads. `../scripts/build_measured_gpu_energy.py` turns them into
`mlsysim/engine/data/measured_gpu_energy.yaml`, the table behind the opt-in
`mlsysim.engine.measured_energy` helpers. They return the measured standby
power and cold-start energy, and a power model calibrated on these runs.
