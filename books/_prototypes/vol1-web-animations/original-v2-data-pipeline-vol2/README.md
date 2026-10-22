# How do batches overlap without violating dependencies?

Source-art prototype for `vol2:fig-data-pipeline-vol2`. Static source and caption stay unchanged.

The original Read → Prep → Train dependency is preserved for every batch. States are logical snapshots, not measured durations; the full original timing comparison remains below.

Invariant: No invented speedup; logical snapshots supplement the original shared-axis schedules.

Run `python3 books/_prototypes/vol1-web-animations/original-v2-data-pipeline-vol2/check.py` for actual-chapter fixtures and browser checks. Review screenshots are written only under `/tmp`.

## Frozen review assets

`original-v2-data-pipeline-vol2.js`: `5dbd7a2c671d66ad157ebec04d19f8045697fcd3eba8b4c45eab9ebc69a44263`
`original-v2-data-pipeline-vol2.css`: `12357daa1de7ca79bba144b191ee34a3104738336722e32b58d6e800a5cea794`
`entry.json`: `b9286b401cab881935c2f3f3cf5aa34cd4f153dac813f924ab5033c416159c9a`
