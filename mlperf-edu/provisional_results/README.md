# Provisional Reference Results

This directory contains the v0.1 draft reference snapshot. It covers all nine
workloads and twelve canonical cases from clean source revision
`163d42ee3df54ab122543469ccf2b6b3bd119455`.

Six cases have complete five-run evidence that passes the declared quality and
timing-repeatability gates. Six cases remain provisional. Five of those have
one complete development execution, and causal-language-model training has two
complete executions whose 5.19% timing coefficient of variation narrowly
misses the 5% promotion gate.

The evidence classes are deliberately separate from the strict promotion
records under `reference_results/`. A provisional record is not a verified
baseline, is not review eligible, and is not an MLCommons-verified result. The
one-run records exist so the v0.1 classroom workflow, reports, website, paper,
and package can be exercised without weakening the five-run promotion rule.

Regenerate the snapshot from retained external evidence with:

```bash
uv run python tools/import_provisional_reference_results.py \
  --promotion-evidence-root /path/to/promotion-evidence \
  --provisional-evidence-root /path/to/provisional-evidence \
  --causal-training-attempt-root /path/to/causal-training-attempt \
  --causal-training-package /path/to/causal-training-package.zip \
  --source-git-sha 163d42ee3df54ab122543469ccf2b6b3bd119455
```

The importer verifies the retained reports, manifests, package, source lock,
case closure, digests, quality decisions, and evidence-class boundary before
writing matching source and wheel-resource mirrors.
