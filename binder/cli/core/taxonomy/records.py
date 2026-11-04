"""Deterministic record serialization, identifiers, hashes, and version syntax."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from ..concept_graph import ConceptGraphError

CONCEPT_ID_RE = re.compile(r"^mlsys\.[a-z0-9]+(-[a-z0-9]+)*$")


_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


_SEMVER_RE = re.compile(r"^([0-9]+)\.([0-9]+)(?:\.([0-9]+))?(?:-(draft|rc([0-9]+)))?$")


def parse_taxonomy_semver(ver: str) -> tuple[int, int, int, int, int]:
    """Parse `<major>.<minor>[.<patch>][-(draft|rc<N>)]` into a monotonically comparable integer tuple."""
    m = _SEMVER_RE.match((ver or "").strip())
    if not m:
        raise ConceptGraphError(f"invalid taxonomy version string {ver!r}")
    major = int(m.group(1))
    minor = int(m.group(2))
    patch_v = int(m.group(3) or 0)
    pre = m.group(4) or ""
    if not pre:
        return (major, minor, patch_v, 2, 0)
    if pre == "draft":
        return (major, minor, patch_v, 0, 0)
    return (major, minor, patch_v, 1, int(m.group(5)))


def expected_status_for_version(ver: str) -> str:
    """Return the mandatory status enum coupled to `ver` (`editorial-draft`, `release-candidate`, or `normative-release`)."""
    _major, _minor, _patch_v, pre_type, _rc_num = parse_taxonomy_semver(ver)
    if pre_type == 0:
        return "editorial-draft"
    if pre_type == 1:
        return "release-candidate"
    return "normative-release"


def validate_ordinal_coverage_judgment(
    score: object,
    assigned_concepts: object,
    valid_concepts: set[str] | None,
    ctx: str,
    *,
    check_active: bool = True,
) -> tuple[float, list[str]]:
    """Enforce the ordinal coverage rubric (`0.0 -> 0`, `1.0 -> 1`, `0.5 -> 2..3` distinct active concepts)."""
    if not _is_finite_num(score) or float(score) not in (0.0, 0.5, 1.0):
        raise ConceptGraphError(
            f"schema sync: {ctx} score must be finite in {{0.0, 0.5, 1.0}}"
        )
    sc = float(score)
    if not isinstance(assigned_concepts, list) or not all(
        isinstance(x, str) and CONCEPT_ID_RE.match(x.strip()) for x in assigned_concepts
    ):
        raise ConceptGraphError(
            f"schema sync: {ctx} concepts must be a list of valid mlsys.* concept IDs"
        )
    clean = [x.strip() for x in assigned_concepts]
    if len(set(clean)) != len(clean):
        raise ConceptGraphError(
            f"schema sync: {ctx} concepts cannot contain duplicate concept IDs"
        )
    if (
        check_active
        and valid_concepts is not None
        and not set(clean).issubset(valid_concepts)
    ):
        raise ConceptGraphError(
            f"schema sync: {ctx} concepts must all be active in candidate taxonomy"
        )
    if sc == 0.0 and len(clean) != 0:
        raise ConceptGraphError(
            f"schema sync: {ctx} score 0.0 requires empty concept list (got {len(clean)})"
        )
    if sc == 1.0 and len(clean) != 1:
        raise ConceptGraphError(
            f"schema sync: {ctx} score 1.0 requires 1 concept ID (got {len(clean)})"
        )
    if sc == 0.5 and not (2 <= len(clean) <= 3):
        raise ConceptGraphError(
            f"schema sync: {ctx} score 0.5 requires 2..3 distinct concept IDs (got {len(clean)})"
        )
    return (sc, clean)


def _norm_prov_tokens(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _reject_json_constant(val: str) -> None:
    raise ConceptGraphError(
        f"schema sync: non-finite JSON constant {val!r} is forbidden"
    )


def _strict_json_parse_float(val_str: str) -> float:
    fval = float(val_str)
    if not math.isfinite(fval):
        raise ConceptGraphError(
            f"schema sync: non-finite JSON float overflow {val_str!r} is forbidden"
        )
    return fval


def _strict_json_no_dup_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in pairs:
        if k in out:
            raise ConceptGraphError(
                f"schema sync: duplicate JSON key {k!r} is forbidden"
            )
        out[k] = v
    return out


def json_loads_strict(
    raw_str: str, label: str = "JSON", *, context: str | None = None
) -> object:
    """Parse JSON rejecting duplicate keys, literal NaN/Infinity, and exponent float overflow (e.g. 1e999)."""
    eff_label = context if context is not None else label
    try:
        return json.loads(
            raw_str,
            object_pairs_hook=_strict_json_no_dup_pairs,
            parse_constant=_reject_json_constant,
            parse_float=_strict_json_parse_float,
        )
    except ConceptGraphError:
        raise
    except Exception as exc:
        raise ConceptGraphError(f"schema sync: malformed {eff_label}: {exc}") from exc


def _is_finite_num(val: object) -> bool:
    return (
        isinstance(val, (int, float))
        and not isinstance(val, bool)
        and math.isfinite(float(val))
    )


def canonical_taxonomy_json(doc: object) -> str:
    """Serialize parsed taxonomy YAML/JSON into deterministic key-sorted JSON for semantic equality checks."""
    try:
        return json.dumps(
            doc,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except ValueError as exc:
        raise ConceptGraphError(
            f"schema sync: non-finite float in canonical JSON serialization: {exc}"
        ) from exc


def compute_non_guides_taxonomy_sha256(raw_tax: dict) -> str:
    """Compute canonical SHA-256 over all top-level taxonomy.yml sections except guides."""
    non_guides = {k: v for k, v in raw_tax.items() if k != "guides"}
    return hashlib.sha256(
        canonical_taxonomy_json(non_guides).encode("utf-8")
    ).hexdigest()


def compute_campaign_entry_sha256(entry: dict) -> str:
    """Compute canonical SHA-256 over an applied_campaigns entry excluding campaign_entry_sha256."""
    body = {k: v for k, v in entry.items() if k != "campaign_entry_sha256"}
    return hashlib.sha256(canonical_taxonomy_json(body).encode("utf-8")).hexdigest()


def canonical_patch_json_sha256(patch_obj: dict) -> str:
    """Compute canonical JSON SHA-256 of a parsed patch/final YAML mapping."""
    return hashlib.sha256(
        canonical_taxonomy_json(patch_obj).encode("utf-8")
    ).hexdigest()


def compute_sidecar_content_sha256(prov_doc: dict) -> str:
    """Compute canonical SHA-256 over provenance.yml content excluding sidecar_content_sha256."""
    body = {k: v for k, v in prov_doc.items() if k != "sidecar_content_sha256"}
    return hashlib.sha256(canonical_taxonomy_json(body).encode("utf-8")).hexdigest()


def compute_implementation_sha256(root: Path) -> str:
    """Fingerprint the validator package as well as its compatibility entry point."""
    core = root / "binder/cli/core"
    paths = [core / "taxonomy_schema.py", *sorted((core / "taxonomy").glob("*.py"))]
    if not paths[0].is_file() or len(paths) == 1:
        raise ConceptGraphError("taxonomy implementation package is missing")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
