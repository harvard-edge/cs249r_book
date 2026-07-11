from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from tools import check_reference_claims


ROOT = Path(__file__).resolve().parents[1]


def as_current_claims(claims: check_reference_claims.ReferenceClaims):
    return replace(
        claims,
        records={
            workload: replace(claim, evidence_role="current-review-evidence")
            for workload, claim in claims.records.items()
        },
    )


def _errors_after_replacement(
    claims: check_reference_claims.ReferenceClaims,
    documents: dict[str, str],
    document: str,
    current: str,
    stale: str,
) -> list[str]:
    mutated = dict(documents)
    assert current in mutated[document]
    mutated[document] = mutated[document].replace(current, stale, 1)
    return check_reference_claims.check_documents(claims, mutated)


def test_repository_reference_claims_are_current():
    claims = check_reference_claims.load_reference_claims(ROOT)
    documents = check_reference_claims.load_document_texts(ROOT)
    errors = check_reference_claims.check_documents(claims, documents)

    assert len(claims.records) == 8
    assert len(claims.current_records) + len(claims.historical_records) == 8
    expected_disclosures = {
        f"{name}: missing fail-closed disclosure "
        f"`{check_reference_claims.HISTORICAL_DISCLOSURE}`; superseded packets "
        "must not be presented as current evidence"
        for name, text in documents.items()
        if claims.historical_records
        and check_reference_claims.HISTORICAL_DISCLOSURE not in text.lower()
    }
    assert set(errors) == expected_disclosures


def test_historical_references_require_explicit_disclosure():
    claims = check_reference_claims.load_reference_claims(ROOT)
    if not claims.historical_records:
        return
    documents = check_reference_claims.load_document_texts(ROOT)
    disclosed = {
        name: f"{check_reference_claims.HISTORICAL_DISCLOSURE}.\n{text}"
        for name, text in documents.items()
    }

    assert check_reference_claims.check_documents(claims, disclosed) == []

    missing = dict(disclosed)
    missing["README.md"] = documents["README.md"].replace(
        check_reference_claims.HISTORICAL_DISCLOSURE, "historical reference"
    )
    errors = check_reference_claims.check_documents(claims, missing)
    assert any(
        error.startswith("README.md: missing fail-closed disclosure")
        for error in errors
    )


def test_claim_check_rejects_stale_source_revision():
    claims = check_reference_claims.load_reference_claims(ROOT)
    documents = check_reference_claims.load_document_texts(ROOT)
    errors = _errors_after_replacement(
        claims,
        documents,
        "README.md",
        claims.source_git_sha,
        "318cd842efe3b90cbf56a109797d2bed4ad3dc09",
    )

    assert any("missing current full source revision" in error for error in errors)
    assert any("suspicious stale or unbound revision" in error for error in errors)


def test_claim_check_rejects_stale_evidence_id():
    claims = check_reference_claims.load_reference_claims(ROOT)
    documents = check_reference_claims.load_document_texts(ROOT)
    claim = claims.records["nanogpt-train"]
    stale_id = "nanogpt-train_max_20260711T000000.000000Z"
    errors = _errors_after_replacement(
        claims,
        documents,
        "QUALITY_TARGET_REVIEW.md",
        claim.evidence_id,
        stale_id,
    )

    assert any(
        f"stale or unbound evidence ID `{stale_id}`" in error for error in errors
    )
    assert any("nanogpt-train evidence ID" in error for error in errors)


def test_claim_check_binds_evidence_id_and_digest_in_one_row():
    claims = check_reference_claims.load_reference_claims(ROOT)
    documents = check_reference_claims.load_document_texts(ROOT)
    claim = claims.records["nanogpt-train"]
    errors = _errors_after_replacement(
        claims,
        documents,
        "QUALITY_TARGET_REVIEW.md",
        claim.evidence_sha256,
        "0" * 64,
    )

    assert any(
        "nanogpt-train evidence ID is not bound to its SHA-256 digest" in error
        for error in errors
    )


def test_claim_check_rejects_medians_swapped_between_rows():
    claims = as_current_claims(check_reference_claims.load_reference_claims(ROOT))
    documents = check_reference_claims.load_document_texts(ROOT)
    nanogpt = claims.records["nanogpt-train"].display_median
    dlrm = claims.records["micro-dlrm-train"].display_median
    proposal = documents["PROPOSAL.md"] + (
        f"\n| nanogpt-train | `{nanogpt}` |\n| micro-dlrm-train | `{dlrm}` |\n"
    )
    marker = "`__median_swap__`"
    assert f"`{nanogpt}`" in proposal
    assert f"`{dlrm}`" in proposal
    proposal = proposal.replace(f"`{nanogpt}`", marker, 1)
    proposal = proposal.replace(f"`{dlrm}`", f"`{nanogpt}`", 1)
    proposal = proposal.replace(marker, f"`{dlrm}`", 1)
    mutated = dict(documents)
    mutated["PROPOSAL.md"] = proposal

    errors = check_reference_claims.check_documents(claims, mutated)

    assert any(
        "nanogpt-train median" in error and "not bound" in error for error in errors
    )
    assert any(
        "micro-dlrm-train median" in error and "not bound" in error for error in errors
    )


def test_claim_check_rejects_public_median_and_range_drift():
    claims = as_current_claims(check_reference_claims.load_reference_claims(ROOT))
    documents = check_reference_claims.load_document_texts(ROOT)
    claim = claims.records["nanogpt-train"]
    documents["README.md"] += (
        f"\n| nanogpt-train | `{claim.display_median}` | "
        f"`{claim.display_minimum}` | `{claim.display_maximum}` |\n"
    )

    median_errors = _errors_after_replacement(
        claims,
        documents,
        "README.md",
        f"`{claim.display_median}`",
        "`2.0568`",
    )
    range_errors = _errors_after_replacement(
        claims,
        documents,
        "README.md",
        f"`{claim.display_minimum}`",
        "`1.9997`",
    )

    assert any("nanogpt-train median" in error for error in median_errors)
    assert any("nanogpt-train minimum" in error for error in range_errors)


def test_claim_check_rejects_repeatability_drift():
    claims = as_current_claims(check_reference_claims.load_reference_claims(ROOT))
    documents = check_reference_claims.load_document_texts(ROOT)
    claim = claims.records["nanogpt-prefill"]
    assert claim.display_cv_percent is not None
    documents["PROPOSAL.md"] += (
        f"\n| nanogpt-prefill | `{claim.display_median}` | "
        f"`{claim.display_minimum}` | `{claim.display_maximum}` | "
        f"{claim.display_cv_percent} |\n"
    )
    errors = _errors_after_replacement(
        claims,
        documents,
        "PROPOSAL.md",
        claim.display_cv_percent,
        "4.61%",
    )

    assert any("nanogpt-prefill CV" in error for error in errors)


def test_registry_binding_rejects_a_different_baseline_median():
    claims = check_reference_claims.load_reference_claims(ROOT)
    claim = claims.records["slm-decode"]
    path = ROOT / "registry/suites/slm/smollm2-chat-inference/variants/baseline.yaml"
    contract = check_reference_claims._load_yaml(path)
    mutated = deepcopy(contract)
    mutated["verified_baseline"]["median"] = claim.median + 1

    errors = check_reference_claims.validate_registry_record(
        claim,
        mutated,
        path=str(path.relative_to(ROOT)),
    )

    assert any("verified_baseline.median" in error for error in errors)


def test_schema_04_claim_binds_primary_and_quality_independently(tmp_path):
    payload = {
        "schema": "mlperf-edu-reference-evidence/0.4",
        "workload": "example-train",
        "evidence_id": "example-train_max_20260711T120000.000000Z",
        "public_status": "score-bearing",
        "source": {"git_sha": "a" * 40},
        "primary_metric": {
            "name": "train_and_eval_seconds",
            "role": "performance",
        },
        "quality_metric": "accuracy",
        "quality_gate": {
            "metric": "accuracy",
            "target": 0.7,
            "direction": "higher",
        },
        "aggregate": {
            "primary_metric": {"median": 12.0, "min": 10.0, "max": 14.0},
            "quality": {"median": 0.8, "min": 0.75, "max": 0.85},
        },
    }
    relative = Path("reference_results/example.json")
    path = tmp_path / relative
    path.parent.mkdir()
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    entry = {
        "workload": "example-train",
        "evidence_id": payload["evidence_id"],
        "public_status": "score-bearing",
        "path": relative.as_posix(),
        "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "metric": "train_and_eval_seconds",
        "quality_metric": "accuracy",
        "quality_gate": payload["quality_gate"],
        "aggregate": payload["aggregate"],
    }

    claim = check_reference_claims._claim_from_entry(tmp_path, entry, "a" * 40)

    assert claim.metric == "train_and_eval_seconds"
    assert claim.median == 12.0
    assert claim.quality_metric == "accuracy"
    assert claim.quality_median == 0.8
    contract = {
        "verified_baseline": {
            "review_eligible": True,
            "evidence_id": claim.evidence_id,
            "evidence_sha256": claim.evidence_sha256,
            "source_git_sha": claim.source_git_sha,
            "primary_metric": "train_and_eval_seconds",
            "median": 12.0,
            "min": 10.0,
            "max": 14.0,
            "quality_metric": "accuracy",
            "accuracy": 0.8,
            "quality_median": 0.8,
            "quality_min": 0.75,
            "quality_max": 0.85,
        }
    }
    assert (
        check_reference_claims.validate_registry_record(
            claim, contract, path="registry/example.yaml"
        )
        == []
    )
    contract["verified_baseline"]["quality_median"] = 0.81
    errors = check_reference_claims.validate_registry_record(
        claim, contract, path="registry/example.yaml"
    )
    assert any("verified_baseline.quality_median" in error for error in errors)

    entry["quality_metric"] = "loss"
    with pytest.raises(check_reference_claims.ClaimDataError, match="quality metric"):
        check_reference_claims._claim_from_entry(tmp_path, entry, "a" * 40)
