from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from tools import check_reference_claims


ROOT = Path(__file__).resolve().parents[1]


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
    claims, errors = check_reference_claims.check_repository(ROOT)

    assert len(claims.records) == 8
    assert errors == []


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
    claims = check_reference_claims.load_reference_claims(ROOT)
    documents = check_reference_claims.load_document_texts(ROOT)
    nanogpt = claims.records["nanogpt-train"].display_median
    dlrm = claims.records["micro-dlrm-train"].display_median
    proposal = documents["PROPOSAL.md"]
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
    claims = check_reference_claims.load_reference_claims(ROOT)
    documents = check_reference_claims.load_document_texts(ROOT)
    claim = claims.records["nanogpt-train"]

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
    claims = check_reference_claims.load_reference_claims(ROOT)
    documents = check_reference_claims.load_document_texts(ROOT)
    claim = claims.records["nanogpt-prefill"]
    assert claim.display_cv_percent is not None
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
