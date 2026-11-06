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
