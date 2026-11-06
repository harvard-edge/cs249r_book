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
