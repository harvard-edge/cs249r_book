# Tier-2 Data Provenance Standard v1 (`books/vol*/**/data/*.csv`)

Every empirical figure and table dataset stored under `books/vol{1..4}/<chapter>/data/*.csv` follows a three-layer provenance contract:

1. **Human-edited source of truth**: Leading `# Key: Value` provenance and typed column-schema comments at the top of each `.csv`, parsed at runtime by `mlsysim.core.series_loader.LocalSeries`.
2. **Cryptographic citation binding**: Every book CSV must cite at least one `@bibkey` authenticated and SHA-256-sealed as `"verified"` in `bibs/manifest.json` via `./binder/binder bib add`.
3. **Machine-verified Frictionless manifest**: A cryptographically sealed [Frictionless Tabular Data Package v1](https://specs.frictionlessdata.io/tabular-data-package/) manifest at `books/datapackage.json` recording full-file and data-only SHA-256 digests, row counts, licenses, sources, and Table Schema v1 column definitions.

All layers are verified offline in ~0.4 s on every commit by `./binder/binder check figures` (`data-provenance` and `cell-provenance` scopes).

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
| `# Figures:` | Yes | Comma-separated `vol<1-4>:(fig\|tbl\|sec)-<slug>` anchors (verified against `books/volN/**/*.qmd`) or `auxiliary (<repo_path>)` pointing to a `.py`/`.qmd`/`.sh` file that explicitly references the CSV filename. Bidirectional check ensures every `.qmd` consumer loading the CSV is listed. |
| `# Source:` | Yes | Primary citation or dataset authority containing at least one `@bibkey`. Every `@bibkey` in `# Source:`, `# Locator:`, `# Computation:`, `# Notes:`, or per-row `source_key` columns must exist in `bibs/references.bib` and match its `"verified"` SHA-256 entry in `bibs/manifest.json`. |
| `# Locator:` | Yes | Substantive (`>= 2` words, `>= 8` chars) table, figure, section, API endpoint, git tag/path, or snapshot filename inside `# Source:`. |
| `# URL:` | Yes | Single `https://` or `http://` landing page, DOI URL, or repository archive URL (dummy hosts such as `example.com` or `localhost` are rejected). |
| `# License:` | Yes | Explicit license identifier (`CC-BY-4.0`, `Apache-2.0`, `MIT`, `BSD-3-Clause`, `Public-Domain (...)`) or `Factual-Extract (...)` basis. |
| `# Verified:` | Yes | ISO-8601 date (`YYYY-MM-DD`, between `2020-01-01` and today) when values were extracted or verified against `# Source:`. |
| `# Epistemic:` | Yes | Must begin with `measured`, `modeled`, `assumed`, or `constructed`, followed by the accounting boundary (`>= 3` words total). |
| `# Method:` | Yes | Must begin with `transcribed`, `digitized`, `scripted`, or `derived`, optionally followed by mechanism details. |
| `# Generator:` | Optional | Repository command used to materialize the CSV; any referenced repository script path must exist on disk and reference the CSV filename. |
| `# Upstream-SHA256:` | Optional | 64-character lowercase SHA-256 hex digest of the raw upstream snapshot before filtering. |
| `# Computation:` / `# Notes:` | Yes (at least one) | Filtering rules, derivations, unit conversions, and scope boundaries (`>= 6` words combined). |
| `# Columns:` | Yes | Indented `#   <name>: <type> [<unit>] — <description>` entries (`>= 3`-word description, concrete `[unit]`) matching non-empty rectangular CSV header columns 1-to-1 in exact order. |

### Column Schema Types (`Table Schema v1`)

Each `# Columns:` entry declares one of the six Frictionless Table Schema v1 scalar types, and every non-empty cell in the CSV is validated against its declared type:

- `integer` — base-10 signed integer literal (`^-?\d+$`)
- `number` — finite floating-point or integer numeric literal
- `string` — UTF-8 string (identifiers, labels, categories, `YYYY-MM` month strings, URLs)
- `boolean` — `True` / `False` (`true` / `false`)
- `date` — ISO-8601 calendar date (`YYYY-MM-DD`)
- `datetime` — ISO-8601 timestamp with date and time components

---

## 2. Anti-Reward-Hacking Controls

To prevent automated edits from rubber-stamping fabricated or unverified data:

1. **No `--fix` auto-sealing**: `./binder/binder check figures --fix` never modifies `books/datapackage.json`.
2. **Immutable `data_sha256` ratchet on `--seal`**: Plain `--seal` only updates `books/datapackage.json` when CSV comment headers change while `custom.data_sha256` (the LF-normalized header + data rows) and the file inventory are unchanged. If any row changes or a CSV is added or removed, `--seal` aborts.
3. **Authenticated `--seal-data --reason "..."`**: Admitting a new CSV or changing data rows requires `--seal-data --reason "<>=5-word provenance explanation>"`, passes full schema/citation/consumer checks first, and performs a live HTTP(S) reachability check on `# URL:`.
4. **Bibliography ledger cross-seal**: Every CSV `@bibkey` (including per-row `source_key` columns) is checked against `bibs/manifest.json` (`status == "verified"` and normalized BibTeX SHA-256 `entry_hash`), which itself requires online DOI/arXiv/URL authentication via `./binder/binder bib add`.
5. **Chapter `.qmd` side-door blocks**: Direct file readers (`pd.read_csv`, `csv.reader`, `np.loadtxt`, `np.load`, `pickle.load`, raw `.csv` `open(...)`), dynamic `load_series(var)` calls, tabular/binary files placed outside `books/volN/<chapter>/data/`, stub `#fig-*` `Provenance:` headers, and inline empirical float literal arrays inside `measured` `#fig-*` cells are all blocked by `book-check-figures`.

---

## 3. Sealing and Verifying `books/datapackage.json`

After editing `#` metadata comments on an already-sealed CSV (with zero data-row changes):

```bash
python3 binder/cli/checks/figure_data_provenance.py --seal
```

When adding a new CSV (after authenticating its `@bibkey` via `./binder/binder bib add ...`) or updating data rows from an upstream source:

```bash
python3 binder/cli/checks/figure_data_provenance.py --seal-data \
  --reason "Added Table 3 latency measurements from authenticated primary paper"
```

To run both read-only gates (`data-provenance` and `cell-provenance`):

```bash
./binder/binder check figures
```
