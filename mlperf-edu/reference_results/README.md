# Reference Evidence Summaries

This directory contains the exact lightweight JSON summaries promoted from
clean, five-seed MLPerf EDU reference sweeps. Each summary records the source
commit, runner-tool digest, acceptance result, raw metric values, aggregate
statistics, hardware fingerprint, and a complete SHA-256 and byte-size index
of every retained run artifact.

The large raw sweep attempts are retained separately on the source machine for
local review. They are not embedded here because checkpoints and
dataset-derived artifacts would make the source repository unsuitable for
ordinary classroom use. Some raw manifests contain absolute source-machine
paths, so an attempt directory is not a portable publication archive. A
registry row marked `reference_package_availability: local-handoff` records
that bounded availability and the lack of a public URL. It does not mean that
the committed summary is a self-contained or portable result package.

`index.json` binds all promoted summaries to the exact source revision.
`source_lock.json` additionally binds the measurement-bearing code, quality
asset, and normalized native contracts used by the eight public candidates.
The lock deliberately excludes promoted baseline rows and publication-only
audit functions, while retaining every execution, target, protocol, dataset,
model, and lineage field. The
importer also creates an exact generated mirror under
`src/mlperf_edu/reference_results/`, which makes the evidence summaries
available in the installed wheel. The workload registry independently records
each summary path and SHA-256 digest, and `tools/check_taxonomy.py` rejects any
mismatch between a displayed baseline and its cited summary.

Evidence schema 0.2 retains the field name `aggregate.quality` for every
primary metric. On performance rows, that object contains throughput samples,
not task-quality scores. The schema also carries a legacy `quality_target`
field whose value can be the functional token floor. It is not a speed target.
The normalized index identifies `reference_metric_role: performance` and the
separate `functional_gate`; validators bind that gate to the raw report before
promotion.

The evidence source commit necessarily precedes the promotion commit that adds
these summaries to version control. Promotion may add evidence pointers,
fail-closed validation, documentation, and package mirrors, but it must not
change runners, model or data preparation, measurement, grading, or report
contracts. Any such measurement-bearing change requires a new clean sweep.

To reproduce the import from a retained sweep root:

```bash
uv run python tools/import_reference_evidence.py \
  --evidence-root /path/to/reference_runs/review-SHA \
  --source-git-sha SHA
```

Add `--check` to verify an existing import without writing files. The tool
requires exactly one clean, valid, public-candidate summary for every
score-bearing and performance-bearing workload, copies its bytes unchanged,
recomputes its claims from the raw reports and manifests against the historical
source checkout, and rebuilds the deterministic index and source lock.
