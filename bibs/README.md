# Unified Bibliography Subsystem (`bibs/`)

This directory houses the unified bibliography subsystem for the Machine Learning Systems book, consolidating the master references, the cryptographic verification ledger, provenance records, and the staging inbox.

## Directory Layout

```
bibs/
├── references.bib             # Master unified bibliography (all 2,454 verified entries)
├── manifest.json              # Cryptographic SHA-256 verification ledger
├── canonical_provenance.json  # Curated external provenance database
├── inbox.bib                  # Candidate staging inbox for new entries
└── README.md                  # This documentation
```

### Backwards Compatibility
For backwards compatibility with Quarto and book compilation configurations under `books/config/`, a relative symlink is maintained:
```
books/references.bib -> ../bibs/references.bib
```

---

## Files

- **`references.bib`**: The single source of truth for citations across the entire book. Every entry is normalized and cryptographically verified.
- **`manifest.json`**: An immutable SHA-256 audit ledger tracking every citation key against primary academic databases (Crossref, DataCite, arXiv, OpenLibrary, or official standards registries).
- **`canonical_provenance.json`**: Curated ground-truth provenance for official hardware datasheets, software specifications, and industry standards.
- **`inbox.bib`**: Staging area for raw or unvetted BibTeX entries awaiting ingestion into the master bibliography.

---

## Reference Gateway CLI

The Reference Gateway prevents fabricated or malformed citations from entering the repository.

### 1. Add a single reference by DOI, arXiv, ISBN, or URL

```bash
# Add a paper via DOI (Crossref / DataCite)
./binder/binder bib add --doi 10.1145/3318464.3389700

# Add an arXiv preprint
./binder/binder bib add --arxiv 2305.10403

# Add a book via ISBN (OpenLibrary)
./binder/binder bib add --isbn 978-0262046824

# Add official documentation or standard
./binder/binder bib add --url https://docs.nvidia.com/cuda/ --title "CUDA C++ Programming Guide" --author "NVIDIA" --year 2024
```

### 2. Batch ingest from `bibs/inbox.bib`

Place candidate entries in `bibs/inbox.bib`, then run:

```bash
# Dry run to preview verification status
./binder/binder bib ingest --dry-run

# Ingest and seal verified entries into references.bib
./binder/binder bib ingest
```

Entries with verifiable external sources are appended to `bibs/references.bib` and sealed into `bibs/manifest.json`. Unverified entries remain in `bibs/inbox.bib` for manual review.

### 3. Re-seal an entry after a house-style edit

The gateway's output is rarely house style: titles need sentence case and braced exact-case tokens, and a `--url` source arrives as `@misc`. Edit the entry in `references.bib`, then re-seal it instead of hand-editing `manifest.json`:

```bash
# Re-resolve the source the entry was sealed from; check title, authors, year, DOI, pages
./binder/binder bib reseal --key villalobos2024will --note "converted to @inproceedings" --dry-run
./binder/binder bib reseal --key villalobos2024will --note "converted to @inproceedings"

# Point at a different source, e.g. the published version of a preprint
./binder/binder bib reseal --key KEY --url https://proceedings.mlr.press/v235/... --note "..."
```

`reseal` never edits `references.bib`. It rewrites the manifest record only if every field the source reports (title, author surnames in order, year, DOI, and pages) agrees with the edited entry, and it refuses otherwise. Publisher pages are checked through their `citation_*` meta tags.

### 4. Verify ledger integrity (Pre-commit gate)

```bash
# Fast offline SHA-256 verification (<350ms)
./binder/binder check bib --scope verify --offline
```
