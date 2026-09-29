#!/usr/bin/env python3
"""
generate_staged_patches.py — Generates clean staging .bib files from manifest fixes
following Section 8 of bib-check.md.
"""

import json
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[4]


FILE_MAPPING = {
    "books/references.bib": "staging-references-books.bib",
    "books/references-vol4.bib": "staging-references-vol4.bib",
    "books/references-vol4-staged.bib": "staging-references-vol4-staged.bib",
    "mlperf-edu/paper/refs.bib": "staging-references-mlperf-edu.bib",
    "mlsysim/paper/references.bib": "staging-references-mlsysim-paper.bib",
    "mlsysim/docs/references.bib": "staging-references-mlsysim-docs.bib",
    "staffml/paper/references.bib": "staging-references-staffml.bib",
    "tinytorch/paper/references.bib": "staging-references-tinytorch.bib",
}


PUBLISHER_NORMALIZATION = {
    "Association for Computing Machinery": "ACM",
    "Association for Computing Machinery (ACM)": "ACM",
    "Curran Associates": "Curran Associates Inc.",
    "Curran Associates, Inc.": "Curran Associates Inc.",
    "Institute of Electrical and Electronics Engineers": "IEEE",
    "Institute of Electrical and Electronics Engineers (IEEE)": "IEEE",
    "IEEE Computer Society Press": "IEEE Computer Society",
}

FORBIDDEN_FIELDS = {"address", "organization", "location", "x-no-doi", "x-verified", "x-verified-by", "x-verified-source", "x-verified-status"}


def format_bib_entry(key: str, entry_type: str, fields: Dict[str, Any]) -> str:
    """Render a clean BibTeX entry adhering strictly to bib-check.md."""
    field_order = [
        "title",
        "author",
        "editor",
        "booktitle",
        "journal",
        "year",
        "volume",
        "number",
        "pages",
        "publisher",
        "institution",
        "school",
        "edition",
        "series",
        "chapter",
        "howpublished",
        "isbn",
        "issn",
        "doi",
        "url",
        "note",
    ]

    # Convert organization to publisher
    if "organization" in fields and "publisher" not in fields:
        fields["publisher"] = fields.pop("organization")

    # Remove forbidden fields
    for forbidden in FORBIDDEN_FIELDS:
        fields.pop(forbidden, None)

    # For @article, publisher is not canonical / causes ratchet warning
    if entry_type.lower() == "article":
        fields.pop("publisher", None)

    # Normalize publisher if present
    pub = fields.get("publisher")
    if pub in PUBLISHER_NORMALIZATION:
        fields["publisher"] = PUBLISHER_NORMALIZATION[pub]

    lines = [f"@{entry_type}{{{key},"]
    emitted = set()

    for fname in field_order:
        val = fields.get(fname)
        if val is not None and str(val).strip():
            val_str = str(val).strip()
            # If value is raw number for year/volume, don't brace unless string
            if fname in ("year", "volume") and val_str.isdigit():
                lines.append(f"  {fname:<12} = {{{val_str}}},")
            else:
                lines.append(f"  {fname:<12} = {{{val_str}}},")
            emitted.add(fname)

    # Any extra fields (excluding forbidden)
    for fname, val in sorted(fields.items()):
        if fname not in emitted and fname.lower() not in FORBIDDEN_FIELDS and val is not None and str(val).strip():
            val_str = str(val).strip()
            lines.append(f"  {fname:<12} = {{{val_str}}},")

    lines.append("}\n")
    return "\n".join(lines)


def main():
    manifest_path = REPO_ROOT / "review/bib-verification/manifest.json"
    staging_dir = REPO_ROOT / "review/bib-verification/staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Collect entries by target file
    staged_by_file: Dict[str, list] = {bf: [] for bf in FILE_MAPPING}

    for key, rec in manifest.items():
        if rec.get("status") == "metadata_fix":
            bf = rec.get("file")
            if bf in staged_by_file:
                staged_by_file[bf].append(rec)

    for bf, staged_entries in staged_by_file.items():
        out_name = FILE_MAPPING[bf]
        out_path = staging_dir / out_name

        lines = [
            "% ==============================================================================",
            f"% STAGING PATCH FOR: {bf}",
            f"% Total Corrections: {len(staged_entries)}",
            "% Verified against primary canonical databases (Crossref, DataCite, publishers)",
            "% ==============================================================================\n",
        ]

        for rec in sorted(staged_entries, key=lambda x: x.get("key", "")):
            key = rec.get("key")
            entry_type = rec.get("recommended_fix", {}).get("entry_type") or rec.get("entry_type", "misc")

            # Merge original fields with recommended fixes
            fields = {}
            for k in ("title", "author", "year", "venue", "publisher", "doi", "url"):
                if rec.get(k):
                    if k == "venue":
                        if entry_type == "article":
                            fields["journal"] = rec.get("venue")
                        else:
                            fields["booktitle"] = rec.get("venue")
                    else:
                        fields[k] = rec.get(k)

            # Apply fixes
            fixes = rec.get("recommended_fix", {})
            for fk, fv in fixes.items():
                if fk != "entry_type":
                    fields[fk] = fv

            lines.append(format_bib_entry(key, entry_type, fields))

        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {len(staged_entries)} staged entries to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
