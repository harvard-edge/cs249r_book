# MLPerf EDU Native Registry

This directory is the authoring source for the native workload registry.
Edit the workload YAML files here first, then refresh the flat compatibility
mirrors with `tools/export_flat_registry.py`.

Taxonomy claims fail closed. An axis remains `unmeasured` until its committed
measurement sidecar and SHA-256 digest pass `tools/check_taxonomy.py`.

| Suite | Workload | Internal ID | Public status |
|---|---|---|---|
| `graph` | `graph-node-classification` | `graph-node-classification` | `experimental` |
| `language` | `causal-language-modeling` | `causal-language-modeling` | `experimental` |
| `language` | `information-retrieval` | `information-retrieval` | `experimental` |
| `language` | `text-classification` | `text-classification` | `experimental` |
| `timeseries` | `time-series-forecasting` | `time-series-forecasting` | `experimental` |
| `tiny` | `anomaly-detection` | `anomaly-detection` | `experimental` |
| `tiny` | `keyword-spotting` | `keyword-spotting` | `experimental` |
| `tiny` | `visual-wake-words` | `visual-wake-words` | `experimental` |
| `vision` | `image-classification` | `image-classification` | `experimental` |

Promotion changes a workload's public status only after its canonical case is
bound to accepted timing evidence. The generated review packets and site
then inherit that status from this registry.
