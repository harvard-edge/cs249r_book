# Promoted Reference Results

This directory is reserved for strict promoted reference summaries. A promoted
case requires five quality-passing fresh-process executions, timing CV no
greater than 5%, a clean source lock, complete artifact verification, and any
required training lineage. The importer fails unless all twelve cases in the
current nine-workload promotion scope satisfy that contract together. The five
functional-stage additions in the fourteen-workload registry remain excluded
until they pass quality conformance.

The current review draft has not reached that closure, so this directory does
not contain an `index.json`. The measured draft snapshot is under
`provisional_results/`. It contains six five-run verified project records and
six explicitly provisional records. Those provisional records do not qualify
as promoted baselines or MLCommons-verified results.

After a complete promotion campaign, import the retained external evidence
with:

```bash
uv run python tools/import_reference_evidence.py \
  --evidence-root /path/to/promotion-campaign \
  --source-git-sha SOURCE_GIT_SHA
```

The strict importer recomputes claims from raw reports and manifests against
the historical source checkout, rejects duplicate or interrupted attempts,
verifies causal training lineage, and writes matching wheel-resource mirrors
under `src/mlperf_edu/reference_results/`.

Large raw attempts remain outside Git because they can contain checkpoints,
dataset-derived bytes, and local paths. Portable handoff packages retain only
policy-permitted bytes, use relative paths, and verify every file after clean
extraction.
