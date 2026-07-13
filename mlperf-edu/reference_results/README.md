# Reference Evidence Summaries

This directory contains the lightweight JSON summaries promoted from guarded
five-run MLPerf EDU reference sweeps. The v0.1 closure consists of ten cases,
one canonical `max` case for each of the seven admitted workloads plus full,
prefill, and decode inference for `causal-language-modeling`.

Each summary records the source commit, runner-tool digest, five raw metric
values, aggregate statistics, quality or functional decisions, timing
repeatability, hardware and software fingerprints, and a complete SHA-256 and
byte-size index of the retained run artifacts. `index.json` binds each summary
to its case ID and digest. `source_lock.json` binds the measurement-bearing
source and normalized canonical contracts.

The v0.1 evidence campaign is bound to clean source revision
`5bba8def62e3944901a9ce0ab0725ed32bf4d3ad`. The evidence source necessarily
precedes the commit that imports the summaries. Publication-only edits may
follow promotion, but a change to a runner, model or data preparation,
measurement boundary, evaluator, quality target, grading rule, or report
contract requires a fresh campaign.

Large raw attempts remain outside Git. They can contain checkpoints,
dataset-derived bytes, and absolute paths from the source machine. Portable
handoff packages retain only policy-permitted bytes, use relative paths, and
verify every file after clean extraction. The committed summary is sufficient
to audit the promoted claim but is not a substitute for the retained raw
packet or independent execution.

To reproduce an import from a retained evidence root, run:

```bash
uv run python tools/import_reference_evidence.py \
  --evidence-root /path/to/promotion-campaign \
  --source-git-sha 5bba8def62e3944901a9ce0ab0725ed32bf4d3ad
```

The importer requires exactly one accepted summary for every expected case. It
recomputes claims from the raw reports and manifests against the historical
source checkout, rejects duplicate or interrupted attempts, verifies causal
training lineage, and writes matching mirrors under
`src/mlperf_edu/reference_results/` for the installed wheel.
