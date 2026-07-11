#!/usr/bin/env python3
"""Fail closed when hand-written reference claims drift from verified evidence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = Path("reference_results/index.json")
DOCUMENT_POLICIES = {
    "README.md": frozenset(
        {"medians", "score-ranges", "repeatability-limit", "score-row-bindings"}
    ),
    "PUBLIC_RULES.md": frozenset({"medians", "repeatability-limit"}),
    "PROPOSAL.md": frozenset(
        {"medians", "score-ranges", "repeatability", "row-bindings"}
    ),
    "QUALITY_TARGET_REVIEW.md": frozenset(
        {
            "evidence-ids",
            "evidence-digests",
            "medians",
            "ranges",
            "repeatability",
            "repeatability-limit",
            "row-bindings",
        }
    ),
    "RELEASE_CHECKLIST.md": frozenset({"repeatability"}),
}

CLAIM_ROW_ALIASES = {
    "nanogpt-train": ("nanogpt-train", "nanogpt training"),
    "micro-dlrm-train": ("micro-dlrm-train", "micro-dlrm training", "dlrm"),
    "anomaly-ae-train": (
        "anomaly-ae-train",
        "anomaly autoencoder",
        "anomaly auroc",
    ),
    "resnet18-train": ("resnet18-train", "resnet-18"),
    "mobilenetv2-train": ("mobilenetv2-train", "mobilenetv2"),
    "nanogpt-prefill": ("nanogpt-prefill", "nanogpt prefill", "variant prefill"),
    "nanogpt-decode": ("nanogpt-decode", "nanogpt decode", "variant decode"),
    "slm-decode": ("slm-decode", "smollm2"),
}

EVIDENCE_ID_RE = re.compile(r"\b[a-z0-9][a-z0-9-]*_max_\d{8}T\d{6}\.\d{6}Z\b")
MARKDOWN_REVISION_RE = re.compile(r"`([0-9a-f]{8,40})`")
SHA40_RE = re.compile(r"[0-9a-f]{40}")


class ClaimDataError(ValueError):
    """Raised when the evidence or registry cannot define a safe claim set."""


@dataclass(frozen=True)
class ReferenceClaim:
    workload: str
    evidence_id: str
    evidence_sha256: str
    source_git_sha: str
    public_status: str
    metric: str
    median: float
    minimum: float
    maximum: float
    coefficient_of_variation: float | None
    repeatability_limit: float | None

    @property
    def display_decimals(self) -> int:
        # Preserve the established two-decimal presentation for six-figure
        # throughput while retaining four decimals for the other references.
        return 2 if abs(self.median) >= 10_000 else 4

    def display(self, value: float) -> str:
        return f"{value:.{self.display_decimals}f}"

    @property
    def display_median(self) -> str:
        return self.display(self.median)

    @property
    def display_minimum(self) -> str:
        return self.display(self.minimum)

    @property
    def display_maximum(self) -> str:
        return self.display(self.maximum)

    @property
    def display_cv_percent(self) -> str | None:
        if self.coefficient_of_variation is None:
            return None
        return f"{100 * self.coefficient_of_variation:.2f}%"


@dataclass(frozen=True)
class ReferenceClaims:
    source_git_sha: str
    records: Mapping[str, ReferenceClaim]
    allowed_non_source_revisions: frozenset[str]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClaimDataError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ClaimDataError(f"{path} must contain a JSON object")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ClaimDataError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ClaimDataError(f"{path} must contain a YAML mapping")
    return value


def _number(mapping: Mapping[str, Any], key: str, label: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ClaimDataError(f"{label}.{key} must be numeric")
    if not math.isfinite(float(value)):
        raise ClaimDataError(f"{label}.{key} must be finite")
    return float(value)


def _relative_payload_path(root: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise ClaimDataError("reference index path must be a string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ClaimDataError(f"unsafe reference index path: {value!r}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ClaimDataError(
            f"reference index path escapes the repository: {value!r}"
        ) from exc
    return path


def _equal_number(actual: object, expected: float) -> bool:
    return (
        not isinstance(actual, bool)
        and isinstance(actual, (int, float))
        and float(actual) == expected
    )


def validate_registry_record(
    claim: ReferenceClaim,
    contract: Mapping[str, Any],
    *,
    path: str,
) -> list[str]:
    """Return exact registry-to-evidence binding errors for one reference."""

    errors: list[str] = []
    baseline = contract.get("verified_baseline")
    if not isinstance(baseline, dict):
        return [f"{path}: missing verified_baseline mapping"]

    expected_scalars = {
        "evidence_id": claim.evidence_id,
        "evidence_sha256": claim.evidence_sha256,
        "source_git_sha": claim.source_git_sha,
        "primary_metric": claim.metric,
    }
    for key, expected in expected_scalars.items():
        if baseline.get(key) != expected:
            errors.append(
                f"{path}: verified_baseline.{key} is {baseline.get(key)!r}; "
                f"expected {expected!r}"
            )

    for key, expected in {
        "median": claim.median,
        "min": claim.minimum,
        "max": claim.maximum,
    }.items():
        if not _equal_number(baseline.get(key), expected):
            errors.append(
                f"{path}: verified_baseline.{key} is {baseline.get(key)!r}; "
                f"expected {expected!r}"
            )

    if claim.coefficient_of_variation is None:
        if "coefficient_of_variation" in baseline:
            errors.append(
                f"{path}: score reference must not claim a performance repeatability value"
            )
        return errors

    if not _equal_number(
        baseline.get("coefficient_of_variation"), claim.coefficient_of_variation
    ):
        errors.append(
            f"{path}: verified_baseline.coefficient_of_variation is "
            f"{baseline.get('coefficient_of_variation')!r}; expected "
            f"{claim.coefficient_of_variation!r}"
        )
    protocol = contract.get("performance_reference_protocol")
    if not isinstance(protocol, dict):
        errors.append(f"{path}: performance reference lacks its protocol")
    elif not _equal_number(
        protocol.get("repeatability_limit"), claim.repeatability_limit
    ):
        errors.append(
            f"{path}: performance_reference_protocol.repeatability_limit is "
            f"{protocol.get('repeatability_limit')!r}; expected "
            f"{claim.repeatability_limit!r}"
        )
    if (
        claim.repeatability_limit is not None
        and claim.coefficient_of_variation > claim.repeatability_limit
    ):
        errors.append(
            f"{path}: measured CV {claim.coefficient_of_variation:.6f} exceeds "
            f"the promotion limit {claim.repeatability_limit:.6f}"
        )
    return errors


def _claim_from_entry(
    root: Path, entry: Mapping[str, Any], source_git_sha: str
) -> ReferenceClaim:
    workload = entry.get("workload")
    if not isinstance(workload, str) or not workload:
        raise ClaimDataError("every reference index entry needs a workload")
    evidence_id = entry.get("evidence_id")
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ClaimDataError(f"{workload}: evidence_id is missing")
    if EVIDENCE_ID_RE.fullmatch(evidence_id) is None:
        raise ClaimDataError(f"{workload}: malformed evidence_id {evidence_id!r}")
    if not evidence_id.startswith(f"{workload}_max_"):
        raise ClaimDataError(f"{workload}: evidence_id belongs to another workload")

    payload_path = _relative_payload_path(root, entry.get("path"))
    payload_bytes = payload_path.read_bytes()
    payload = _load_json(payload_path)
    for key, expected in {
        "workload": workload,
        "evidence_id": evidence_id,
        "public_status": entry.get("public_status"),
    }.items():
        if payload.get(key) != expected:
            raise ClaimDataError(
                f"{workload}: summary {key} is {payload.get(key)!r}; expected {expected!r}"
            )
    source = payload.get("source")
    if not isinstance(source, dict) or source.get("git_sha") != source_git_sha:
        raise ClaimDataError(
            f"{workload}: summary is not bound to index source_git_sha"
        )

    entry_aggregate = entry.get("aggregate")
    payload_aggregate = payload.get("aggregate")
    if not isinstance(entry_aggregate, dict) or not isinstance(payload_aggregate, dict):
        raise ClaimDataError(f"{workload}: aggregate is missing")
    entry_primary = entry_aggregate.get("primary_metric")
    payload_primary = payload_aggregate.get("primary_metric")
    if not isinstance(entry_primary, dict) or entry_primary != payload_primary:
        raise ClaimDataError(f"{workload}: index and summary primary aggregates differ")

    public_status = entry.get("public_status")
    if public_status not in {"score-bearing", "performance-bearing"}:
        raise ClaimDataError(
            f"{workload}: unsupported reference status {public_status!r}"
        )
    metric = entry.get("metric")
    if not isinstance(metric, str) or payload.get("quality_metric") != metric:
        raise ClaimDataError(f"{workload}: index and summary metric differ")

    cv: float | None = None
    repeatability_limit: float | None = None
    if public_status == "performance-bearing":
        repeatability = payload.get("repeatability")
        if not isinstance(repeatability, dict):
            raise ClaimDataError(f"{workload}: performance summary lacks repeatability")
        cv = _number(repeatability, "coefficient_of_variation", workload)
        repeatability_limit = _number(repeatability, "limit", workload)
        if repeatability.get("passed") is not True or cv > repeatability_limit:
            raise ClaimDataError(
                f"{workload}: repeatability promotion gate did not pass"
            )

    digest = entry.get("evidence_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ClaimDataError(f"{workload}: malformed evidence_sha256")
    if hashlib.sha256(payload_bytes).hexdigest() != digest:
        raise ClaimDataError(
            f"{workload}: evidence_sha256 does not match summary bytes"
        )
    return ReferenceClaim(
        workload=workload,
        evidence_id=evidence_id,
        evidence_sha256=digest,
        source_git_sha=source_git_sha,
        public_status=public_status,
        metric=metric,
        median=_number(entry_primary, "median", workload),
        minimum=_number(entry_primary, "min", workload),
        maximum=_number(entry_primary, "max", workload),
        coefficient_of_variation=cv,
        repeatability_limit=repeatability_limit,
    )


def load_reference_claims(root: Path = ROOT) -> ReferenceClaims:
    """Load and cross-check the index, raw summaries, and native registry."""

    index = _load_json(root / INDEX_PATH)
    source_git_sha = index.get("source_git_sha")
    if (
        not isinstance(source_git_sha, str)
        or SHA40_RE.fullmatch(source_git_sha) is None
    ):
        raise ClaimDataError(
            "reference index source_git_sha must be 40 lowercase hex digits"
        )
    entries = index.get("summaries")
    if not isinstance(entries, list) or not entries:
        raise ClaimDataError("reference index summaries must be a non-empty list")
    if index.get("summary_count") != len(entries):
        raise ClaimDataError("reference index summary_count does not match summaries")

    records: dict[str, ReferenceClaim] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ClaimDataError("reference index entries must be objects")
        claim = _claim_from_entry(root, entry, source_git_sha)
        if claim.workload in records:
            raise ClaimDataError(f"duplicate reference workload {claim.workload!r}")
        records[claim.workload] = claim

    registry_by_workload: dict[str, tuple[Path, dict[str, Any]]] = {}
    allowed_revisions: set[str] = set()
    for path in sorted((root / "registry" / "suites").rglob("*.yaml")):
        contract = _load_yaml(path)
        model_source = contract.get("model_source")
        if isinstance(model_source, dict):
            revision = model_source.get("revision")
            if isinstance(revision, str) and SHA40_RE.fullmatch(revision):
                allowed_revisions.add(revision)
        if "verified_baseline" not in contract:
            continue
        workload = contract.get("id")
        relative = path.relative_to(root)
        if not isinstance(workload, str):
            raise ClaimDataError(f"{relative}: verified baseline has no workload id")
        if workload in registry_by_workload:
            raise ClaimDataError(f"duplicate verified registry baseline for {workload}")
        registry_by_workload[workload] = (relative, contract)

    missing = set(records) - set(registry_by_workload)
    extra = set(registry_by_workload) - set(records)
    if missing or extra:
        raise ClaimDataError(
            "verified registry/index closure mismatch; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    registry_errors: list[str] = []
    final_records: dict[str, ReferenceClaim] = {}
    for workload, claim in records.items():
        path, contract = registry_by_workload[workload]
        final_records[workload] = claim
        registry_errors.extend(
            validate_registry_record(claim, contract, path=str(path))
        )
    if registry_errors:
        raise ClaimDataError("registry reference drift:\n" + "\n".join(registry_errors))

    return ReferenceClaims(
        source_git_sha=source_git_sha,
        records=final_records,
        allowed_non_source_revisions=frozenset(allowed_revisions),
    )


def load_document_texts(root: Path = ROOT) -> dict[str, str]:
    documents: dict[str, str] = {}
    for name in DOCUMENT_POLICIES:
        path = root / name
        try:
            documents[name] = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ClaimDataError(f"cannot load {path}: {exc}") from exc
    return documents


def _missing_token(name: str, label: str, token: str) -> str:
    return f"{name}: missing current {label} `{token}`"


def _performance_records(claims: ReferenceClaims) -> Iterable[ReferenceClaim]:
    return (
        claim
        for claim in claims.records.values()
        if claim.public_status == "performance-bearing"
    )


def _markdown_row_binds_claim(
    rows: Iterable[str], claim: ReferenceClaim, token: str
) -> bool:
    aliases = CLAIM_ROW_ALIASES.get(claim.workload)
    if not aliases:
        raise ClaimDataError(f"no Markdown-row aliases for {claim.workload}")
    return any(
        token in row and any(alias in row.lower() for alias in aliases) for row in rows
    )


def check_documents(
    claims: ReferenceClaims,
    documents: Mapping[str, str],
) -> list[str]:
    """Return drift errors for the hand-written public review documents."""

    errors: list[str] = []
    current_ids = {claim.evidence_id for claim in claims.records.values()}
    for name, policy in DOCUMENT_POLICIES.items():
        text = documents.get(name)
        if text is None:
            errors.append(f"{name}: document is missing")
            continue
        if claims.source_git_sha not in text:
            errors.append(
                _missing_token(name, "full source revision", claims.source_git_sha)
            )
        markdown_rows = [
            line for line in text.splitlines() if line.lstrip().startswith("|")
        ]

        for revision in MARKDOWN_REVISION_RE.findall(text):
            if claims.source_git_sha.startswith(revision):
                continue
            if revision in claims.allowed_non_source_revisions:
                continue
            errors.append(
                f"{name}: suspicious stale or unbound revision `{revision}`; "
                "only the current source revision and registry-pinned model revisions are allowed"
            )

        for evidence_id in EVIDENCE_ID_RE.findall(text):
            if evidence_id not in current_ids:
                errors.append(f"{name}: stale or unbound evidence ID `{evidence_id}`")

        if "evidence-ids" in policy:
            for claim in claims.records.values():
                if claim.evidence_id not in text:
                    errors.append(
                        _missing_token(
                            name, claim.workload + " evidence ID", claim.evidence_id
                        )
                    )
                elif "evidence-digests" in policy and not any(
                    claim.evidence_id in row and claim.evidence_sha256 in row
                    for row in markdown_rows
                ):
                    errors.append(
                        f"{name}: {claim.workload} evidence ID is not bound to its "
                        "SHA-256 digest in one Markdown row"
                    )

        if "medians" in policy:
            for claim in claims.records.values():
                token = f"`{claim.display_median}`"
                if token not in text:
                    errors.append(
                        _missing_token(
                            name, claim.workload + " median", claim.display_median
                        )
                    )

        range_claims: Iterable[ReferenceClaim] = ()
        if "ranges" in policy:
            range_claims = claims.records.values()
        elif "score-ranges" in policy:
            range_claims = (
                claim
                for claim in claims.records.values()
                if claim.public_status == "score-bearing"
            )
        for claim in range_claims:
            for label, value in {
                "minimum": claim.display_minimum,
                "maximum": claim.display_maximum,
            }.items():
                if f"`{value}`" not in text:
                    errors.append(
                        _missing_token(name, f"{claim.workload} {label}", value)
                    )

        if "repeatability" in policy:
            for claim in _performance_records(claims):
                value = claim.display_cv_percent
                if value is not None and value not in text:
                    errors.append(_missing_token(name, claim.workload + " CV", value))

        if "repeatability-limit" in policy:
            limits = {
                claim.repeatability_limit for claim in _performance_records(claims)
            }
            if None in limits or len(limits) != 1:
                errors.append(
                    f"{name}: performance references do not define one common repeatability limit"
                )
            else:
                limit = next(iter(limits))
                assert limit is not None
                percent = f"{100 * limit:g}%"
                decimal = f"{limit:g}"
                if percent not in text and f"`{decimal}`" not in text:
                    errors.append(_missing_token(name, "repeatability limit", percent))

        row_claims: Iterable[ReferenceClaim] = ()
        if "row-bindings" in policy:
            row_claims = claims.records.values()
        elif "score-row-bindings" in policy:
            row_claims = (
                claim
                for claim in claims.records.values()
                if claim.public_status == "score-bearing"
            )
        for claim in row_claims:
            required_tokens = {"median": claim.display_median}
            if "ranges" in policy or (
                "score-ranges" in policy and claim.public_status == "score-bearing"
            ):
                required_tokens.update(
                    {
                        "minimum": claim.display_minimum,
                        "maximum": claim.display_maximum,
                    }
                )
            if "repeatability" in policy and claim.display_cv_percent is not None:
                required_tokens["CV"] = claim.display_cv_percent
            for label, token in required_tokens.items():
                if not _markdown_row_binds_claim(markdown_rows, claim, token):
                    errors.append(
                        f"{name}: {claim.workload} {label} `{token}` is not bound "
                        "to that workload in one Markdown row"
                    )
    return errors


def check_repository(root: Path = ROOT) -> tuple[ReferenceClaims, list[str]]:
    claims = load_reference_claims(root)
    return claims, check_documents(claims, load_document_texts(root))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed claims without modifying files (the default behavior)",
    )
    parser.parse_args()
    try:
        claims, errors = check_repository()
    except ClaimDataError as exc:
        print(f"reference claim inputs are invalid: {exc}")
        return 1
    if errors:
        print("reference claims are out of date:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        f"reference claims are current ({len(DOCUMENT_POLICIES)} documents, "
        f"{len(claims.records)} references)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
