# Tier-2 Data Provenance Standard v1 (`books/vol*/**/data/*.csv`)

Every empirical figure and table dataset stored under `books/vol{1..4}/<chapter>/data/*.csv` follows a two-part provenance contract:

1. **Human-edited source of truth**: Leading `# Key: Value` provenance and typed column-schema comments at the top of each `.csv`, parsed at runtime by `mlsysim.core.series_loader.LocalSeries`.
2. **Machine-verified Frictionless manifest**: A cryptographically sealed [Frictionless Tabular Data Package v1](https://specs.frictionlessdata.io/tabular-data-package/) manifest at `books/datapackage.json` recording full-file and data-only SHA-256 digests, row counts, licenses, sources, and Table Schema v1 column definitions.

Both layers are verified offline in under 200 ms on every commit by `./binder/binder check figures --scope data-provenance`.

---

## 1. Canonical Inline CSV Header

```csv
# Figures: vol2:fig-frontier-training-energy, vol2:tbl-training-emissions
# Source: Patterson et al. (2021), Carbon Emissions and Large Neural Network Training (@patterson2021carbon)
# Locator: Table 4 (final-run training energy and carbon emissions of five large NLP models)
# URL: https://arxiv.org/abs/2104.10350
# License: Factual-Extract (transcribed Table 4 training power, PUE, energy, and emissions from Patterson et al. 2021)
# Verified: 2026-10-20
# Epistemic: measured (metered average system power per chip integrated over wall-clock run duration and PUE)
# Method: transcribed (row-by-row transcription of the five NLP model columns from Table 4)
# Generator: python3 binder/tools/scripts/<script>.py ...     [optional]
# Upstream-SHA256: <64 lowercase hex chars>                   [optional]
# Computation: <filtering rules, exclusions, and formulas>    [optional when Notes is present]
# Columns:
#   model: string [identifier] — Trained large language model name
#   params_b: number [10^9 parameters] — Total model parameter count in billions
#   chips: integer [chips] — Number of accelerators used in final training run
#   energy_mwh: number [MWh] — Total facility electricity consumed including PUE overhead
# Notes: <accounting boundaries, exclusions, and caveats>     [optional when Computation is present]
```

### Field Reference

| Header Key | Required | Validation Rule |
| :--- | :--- | :--- |
| `# Figures:` | Yes | Comma-separated `vol<1-4>:(fig\|tbl\|sec)-<slug>` anchors (verified against `books/volN/**/*.qmd`) or `auxiliary (<repo_path>)` for non-figure companion series. Every `.qmd` consumer loading the CSV must be represented. |
| `# Source:` | Yes | Primary citation or dataset authority. Every `@citekey` token in `# Source:`, `# Locator:`, `# Computation:`, or `# Notes:` must exist in the volume bibliography (`bibs/references.bib`). |
| `# Locator:` | Yes | Exact table, figure, section, API endpoint, git tag/path, or snapshot filename inside `# Source:`. |
| `# URL:` | Yes | Single `https://` or `http://` landing page, DOI URL, or repository archive URL. |
| `# License:` | Yes | Explicit license identifier (`CC-BY-4.0`, `Apache-2.0`, `MIT`, `BSD-3-Clause`, `Public-Domain (...)`) or `Factual-Extract (...)` basis. |
| `# Verified:` | Yes | ISO-8601 date (`YYYY-MM-DD`) when values were extracted or verified against `# Source:`. |
| `# Epistemic:` | Yes | Must begin with `measured`, `modeled`, `assumed`, or `constructed`, followed by the accounting boundary. |
| `# Method:` | Yes | Must begin with `transcribed`, `digitized`, `scripted`, or `derived`, optionally followed by mechanism details. |
| `# Generator:` | Optional | Repository command used to materialize the CSV; any referenced repository script path must exist on disk. |
| `# Upstream-SHA256:` | Optional | 64-character lowercase SHA-256 hex digest of the raw upstream snapshot before filtering. |
| `# Computation:` / `# Notes:` | Yes (at least one) | Filtering rules, derivations, unit conversions, and scope boundaries. |
| `# Columns:` | Yes | Indented `#   <name>: <type> [<unit>] — <description>` entries matching the CSV header columns 1-to-1 in exact order. |

### Column Schema Types (`Table Schema v1`)

Each `# Columns:` entry declares one of the six Frictionless Table Schema v1 scalar types, and every non-empty cell in the CSV is validated against its declared type:

- `integer` — base-10 signed integer literal (`^-?\d+$`)
- `number` — finite floating-point or integer numeric literal
- `string` — UTF-8 string (identifiers, labels, categories, `YYYY-MM` month strings, URLs)
- `boolean` — `True` / `False` (`true` / `false`)
- `date` — ISO-8601 calendar date (`YYYY-MM-DD`)
- `datetime` — ISO-8601 timestamp with date and time components

---

## 2. Sealing and Verifying `books/datapackage.json`

Whenever you add, edit, or delete a CSV under `books/vol{1..4}/**/data/*.csv` (whether changing a `#` comment or a data row), reseal the Frictionless Data Package manifest:

```bash
python3 binder/cli/checks/figure_data_provenance.py --seal
# or equivalently via Binder:
./binder/binder check figures --scope data-provenance --fix
```

To run the read-only gate directly:

```bash
./binder/binder check figures --scope data-provenance
```

`books/datapackage.json` conforms to the Frictionless `tabular-data-package` profile (`dialect.commentChar = "#"`), so external tools such as DuckDB, Polars, R, and the Frictionless Python CLI can inspect schemas, units, sources, and cryptographic hashes directly from a single manifest.
