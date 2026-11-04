#!/usr/bin/env python3
"""Verify enums.py stays in sync with question_schema.yaml.

Runs in CI as a drift check. Exits non-zero if the hand-maintained Python
enum constants disagree with the authoritative LinkML schema.

Usage:
    python3 staffml/vault-cli/scripts/check_schema_sync.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
LINKML = REPO / "vault" / "schema" / "question_schema.yaml"
ENUMS_PY = REPO / "vault" / "schema" / "enums.py"
TAXONOMY_DATA = REPO / "vault" / "schema" / "taxonomy_data.yaml"
MODELS_PY = REPO / "vault-cli" / "src" / "vault_cli" / "models.py"
ML_TAXONOMY_PY = REPO / "vault-cli" / "src" / "vault_cli" / "ml_systems_taxonomy.py"
ROOT_LINKML = REPO.parent / "taxonomy" / "schema" / "mlsys-taxonomy.linkml.yml"

# Map LinkML enum name -> Python constant name in enums.py.
ENUM_MAPPING = {
    "Track":             "VALID_TRACKS",
    "Level":             "VALID_LEVELS",
    "Zone":              "VALID_ZONES",
    "BloomLevel":        "VALID_BLOOM_LEVELS",
    "Phase":             "VALID_PHASES",
    "Status":            "VALID_STATUSES",
    "Provenance":        "VALID_PROVENANCES",
    "HumanReviewStatus": "VALID_HUMAN_REVIEW_STATUSES",
    "CompetencyArea":    "VALID_COMPETENCY_AREAS",
}


def load_linkml_data() -> dict:
    with LINKML.open() as fh:
        return yaml.safe_load(fh)


def load_linkml_enums() -> dict[str, set[str]]:
    data = load_linkml_data()
    out = {}
    for name, spec in (data.get("enums") or {}).items():
        values = set((spec or {}).get("permissible_values") or {})
        out[name] = values
    return out


def load_python_enums():
    # Import the module directly; avoids running the validator machinery.
    spec_dir = ENUMS_PY.parent
    if str(spec_dir) not in sys.path:
        sys.path.insert(0, str(spec_dir))
    import enums  # type: ignore[import-not-found]

    out: dict[str, set[str]] = {}
    for linkml_name, py_name in ENUM_MAPPING.items():
        out[linkml_name] = set(getattr(enums, py_name))
    return out, enums


def main() -> int:
    linkml_doc = load_linkml_data()
    linkml = load_linkml_enums()
    py, enums_mod = load_python_enums()

    drift = False
    for enum_name in ENUM_MAPPING:
        linkml_vals = linkml.get(enum_name, set())
        py_vals = py.get(enum_name, set())
        if linkml_vals != py_vals:
            drift = True
            print(f"[drift] {enum_name}:")
            only_linkml = linkml_vals - py_vals
            only_py = py_vals - linkml_vals
            if only_linkml:
                print(f"  in LinkML only: {sorted(only_linkml)}")
            if only_py:
                print(f"  in enums.py only: {sorted(only_py)}")

    if TAXONOMY_DATA.exists():
        tax_data = yaml.safe_load(TAXONOMY_DATA.read_text()) or {}
        yaml_topics = {
            t["id"]
            for area in (tax_data.get("competency_areas") or [])
            for t in (area.get("topics") or [])
            if isinstance(t, dict) and "id" in t
        }
        py_topics = set(getattr(enums_mod, "VALID_TOPICS", set()))
        if yaml_topics and yaml_topics != py_topics:
            drift = True
            print(f"[drift] VALID_TOPICS vs taxonomy_data.yaml: {sorted(yaml_topics ^ py_topics)}")

    q_attrs = ((linkml_doc.get("classes") or {}).get("Question") or {}).get("attributes") or {}
    q_subj_pat = (q_attrs.get("taxonomy_subject") or {}).get("pattern")
    q_conc_pat = (q_attrs.get("taxonomy_concepts") or {}).get("pattern")
    if ROOT_LINKML.exists():
        root_doc = yaml.safe_load(ROOT_LINKML.read_text()) or {}
        u_attrs = ((root_doc.get("classes") or {}).get("UnifiedConcept") or {}).get("attributes") or {}
        root_subj_pat = (u_attrs.get("home") or {}).get("pattern")
        root_conc_pat = (u_attrs.get("id") or {}).get("pattern")
        if q_subj_pat != root_subj_pat or q_conc_pat != root_conc_pat:
            drift = True
            print(
                f"[drift] taxonomy ID patterns: question_schema=({q_subj_pat}, {q_conc_pat}) vs mlsys-taxonomy.linkml=({root_subj_pat}, {root_conc_pat})"
            )

    if MODELS_PY.exists():
        import re as _re
        models_text = MODELS_PY.read_text()
        subj_match = _re.search(r'_TAXONOMY_SUBJECT_RE\s*=\s*re\.compile\(r"([^"]+)"\)', models_text)
        conc_match = _re.search(r'_TAXONOMY_CONCEPT_RE\s*=\s*re\.compile\(r"([^"]+)"\)', models_text)
        m_subj_pat = subj_match.group(1) if subj_match else None
        m_conc_pat = conc_match.group(1) if conc_match else None
        if m_subj_pat != q_subj_pat or m_conc_pat != q_conc_pat:
            drift = True
            print(
                f"[drift] models.py taxonomy regexes ({m_subj_pat}, {m_conc_pat}) disagree with question_schema.yaml ({q_subj_pat}, {q_conc_pat})"
            )

    if ML_TAXONOMY_PY.exists():
        import re as _re
        tax_text = ML_TAXONOMY_PY.read_text()
        slug_match = _re.search(r'_SLUG_RE\s*=\s*re\.compile\(r"([^"]+)"\)', tax_text)
        conc_match = _re.search(r'_CONCEPT_RE\s*=\s*re\.compile\(r"([^"]+)"\)', tax_text)
        t_subj_pat = slug_match.group(1) if slug_match else None
        t_conc_pat = conc_match.group(1) if conc_match else None
        if t_subj_pat != q_subj_pat or t_conc_pat != q_conc_pat:
            drift = True
            print(
                f"[drift] ml_systems_taxonomy.py regexes ({t_subj_pat}, {t_conc_pat}) disagree with question_schema.yaml ({q_subj_pat}, {q_conc_pat})"
            )

    if drift:
        print()
        print("FAIL: schema/enums.py or question_schema.yaml disagrees with the canonical LinkML schema.")
        print("Update the drifted definitions, then re-run.")
        return 1
    print(f"OK: {len(ENUM_MAPPING)} enums and taxonomy patterns in sync.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
