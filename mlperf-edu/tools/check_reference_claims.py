#!/usr/bin/env python3
"""Fail closed when public review claims drift from indexed case evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools import import_reference_evidence as evidence  # noqa: E402
from tools import reference_source_lock  # noqa: E402
from mlperf.registry import load_registry  # noqa: E402


INDEX_PATH = ROOT / "reference_results" / "index.json"
PROVISIONAL_INDEX_PATH = ROOT / "provisional_results" / "index.json"
# Documents that state counts or per-case values, and must therefore stay bound
# to the committed evidence index.
#
# docs/internal/DIRECTION.md and docs/internal/STATUS.md are deliberately absent.
# They carry no counts and no measurements, deferring every countable fact to the
# registry, the evidence index, and tools/audit_workload_readiness.py. Requiring
# them to restate an inventory would reintroduce exactly the duplication that
# made published claims drift in the first place. A document that makes no
# numeric claim has nothing for this gate to verify.
PUBLIC_DOCUMENTS = (
    "README.md",
    "SPEC.md",
    "PUBLIC_RULES.md",
    "docs/internal/QUALITY_TARGET_REVIEW.md",
    "docs/internal/RELEASE_CHECKLIST.md",
)
WORKLOAD_DOCUMENTS = frozenset(
    {
        "README.md",
        "SPEC.md",
        "docs/internal/QUALITY_TARGET_REVIEW.md",
        "docs/internal/RELEASE_CHECKLIST.md",
    }
)
RETIRED_PUBLIC_IDS = frozenset(
    {
        "anomaly-ae-train",
        "dscnn-kws-train",
        "micro-bert-train",
        "micro-diffusion-train",
        "micro-dlrm-distributed",
        "micro-dlrm-dram-train",
        "micro-dlrm-train",
        "micro-gnn-train",
        "micro-lstm-train",
        "micro-rl-train",
        "mobilenet-cifar100-composed-fp16",
        "mobilenetv2-train",
        "nano-codegen-agent",
        "nano-lora-finetune",
        "nano-moe-train",
        "nano-rag-agent",
        "nano-react-agent",
        "nano-toolcall-agent",
        "nanogpt-decode",
        "nanogpt-decode-fp16-b16",
        "nanogpt-decode-fp32-b16",
        "nanogpt-decode-spec",
        "nanogpt-prefill",
        "nanogpt-train",
        "resnet18-train",
        "slm-batched-decode",
        "slm-decode",
        "slm-long-context-decode",
        "slm-quantized-decode",
        "smollm2-chat-inference",
        "wake-vision-vww",
    }
)
EVIDENCE_ID_RE = re.compile(r"\b[a-z0-9][a-z0-9-]*_max_\d{8}T\d{6}\.\d{6}Z\b")
SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
NUMBER_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
}


class ClaimError(ValueError):
    """Raised when committed evidence cannot support the public claims."""


def count_claim_pattern(count: int, noun: str) -> re.Pattern[str]:
    """Match a numeric or conventional English portfolio-count claim."""
    alternatives = [str(count)]
    if count in NUMBER_WORDS:
        alternatives.append(NUMBER_WORDS[count])
    number = "|".join(re.escape(value) for value in alternatives)
    return re.compile(rf"\b(?:{number}) {re.escape(noun)}s?\b")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClaimError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ClaimError(f"{path} must contain a JSON object")
    return value


def safe_repository_path(value: object) -> Path:
    if not isinstance(value, str):
        raise ClaimError("reference index paths must be strings")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ClaimError(f"unsafe reference path {value!r}")
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ClaimError(f"reference path escapes the repository: {value!r}") from exc
    return path


def load_cases() -> tuple[str, dict[str, tuple[dict[str, Any], dict[str, Any]]]]:
    index = load_json(INDEX_PATH)
    if index.get("schema") != evidence.INDEX_SCHEMA:
        raise ClaimError(
            f"reference index schema is {index.get('schema')!r}, expected "
            f"{evidence.INDEX_SCHEMA!r}"
        )
    source_sha = index.get("source_git_sha")
    if not isinstance(source_sha, str) or not SHA40_RE.fullmatch(source_sha):
        raise ClaimError("reference index source_git_sha is invalid")
    entries = index.get("cases")
    if not isinstance(entries, list) or index.get("case_count") != len(entries):
        raise ClaimError("reference index case_count does not match cases")
    expected = evidence.expected_cases()
    if index.get("workload_count") != len(
        {case.workload.id for case in expected.values()}
    ):
        raise ClaimError("reference index workload_count is invalid")

    records: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ClaimError("reference index cases must be objects")
        case_id = entry.get("case_id")
        if not isinstance(case_id, str) or case_id in records:
            raise ClaimError(f"missing or duplicate case ID {case_id!r}")
        path = safe_repository_path(entry.get("path"))
        payload = load_json(path)
        if evidence.sha256_file(path) != entry.get("evidence_sha256"):
            raise ClaimError(f"{case_id}: evidence digest does not match index")
        expected_case = expected.get(case_id)
        if expected_case is None:
            raise ClaimError(f"unexpected evidence case {case_id}")
        expected_fields = {
            "workload": expected_case.workload.id,
            "profile": expected_case.profile,
            "mode": expected_case.mode,
            "phase": expected_case.phase,
            "result_role": expected_case.result_role,
            "evidence_id": entry.get("evidence_id"),
        }
        for field, expected_value in expected_fields.items():
            if payload.get(field) != expected_value:
                raise ClaimError(
                    f"{case_id}: summary {field} is {payload.get(field)!r}, "
                    f"expected {expected_value!r}"
                )
        source = payload.get("source") or {}
        if source.get("git_sha") != source_sha or source.get("git_dirty") is not False:
            raise ClaimError(f"{case_id}: summary is not from the clean source SHA")
        runs = payload.get("runs")
        if not isinstance(runs, list) or len(runs) != 5:
            raise ClaimError(f"{case_id}: summary does not contain five runs")
        if (
            payload.get("status") != "valid"
            or payload.get("eligible_for_promotion") is not True
        ):
            raise ClaimError(f"{case_id}: evidence is not promotion eligible")
        if (payload.get("acceptance") or {}).get("passed") is not True:
            raise ClaimError(f"{case_id}: acceptance gate did not pass")
        repeatability = payload.get("primary_metric_repeatability") or {}
        if repeatability.get("passed") is not True:
            raise ClaimError(f"{case_id}: repeatability gate did not pass")
        records[case_id] = (entry, payload)

    if set(records) != set(expected):
        raise ClaimError(
            "reference case closure mismatch; "
            f"missing={sorted(set(expected) - set(records))}, "
            f"extra={sorted(set(records) - set(expected))}"
        )
    return source_sha, records


def load_provisional_cases() -> tuple[
    str, dict[str, tuple[dict[str, Any], dict[str, Any]]]
]:
    index = load_json(PROVISIONAL_INDEX_PATH)
    expected_schema = "mlperf-edu-provisional-reference-index/0.1"
    if index.get("schema") != expected_schema:
        raise ClaimError(
            f"draft index schema is {index.get('schema')!r}, expected "
            f"{expected_schema!r}"
        )
    source_sha = index.get("source_git_sha")
    if not isinstance(source_sha, str) or not SHA40_RE.fullmatch(source_sha):
        raise ClaimError("draft index source_git_sha is invalid")
    entries = index.get("cases")
    if not isinstance(entries, list) or index.get("case_count") != len(entries):
        raise ClaimError("draft index case_count does not match cases")

    expected = evidence.expected_cases()
    expected_workloads = {case.workload.id for case in expected.values()}
    if index.get("workload_count") != len(expected_workloads):
        raise ClaimError("draft index workload_count is invalid")

    lock_entry = index.get("source_lock") or {}
    lock_path = safe_repository_path(lock_entry.get("path"))
    if evidence.sha256_file(lock_path) != lock_entry.get("sha256"):
        raise ClaimError("draft source-lock digest does not match index")
    reference_source_lock.load_source_lock(
        lock_path,
        project_root=ROOT,
        expected_source_git_sha=source_sha,
        # Draft evidence is an immutable historical snapshot. Validate its
        # closure and source-lock structure without requiring the active
        # development checkout to remain byte-identical to that old source.
        verify_current=False,
    )

    records: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    class_counts: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ClaimError("draft index cases must be objects")
        case_id = entry.get("case_id")
        if not isinstance(case_id, str) or case_id in records:
            raise ClaimError(f"missing or duplicate draft case ID {case_id!r}")
        expected_case = expected.get(case_id)
        if expected_case is None:
            raise ClaimError(f"unexpected draft case {case_id}")
        path = safe_repository_path(entry.get("path"))
        payload = load_json(path)
        if evidence.sha256_file(path) != entry.get("sha256"):
            raise ClaimError(f"{case_id}: draft result digest does not match index")
        if payload.get("schema") != "mlperf-edu-provisional-reference-result/0.1":
            raise ClaimError(f"{case_id}: draft result schema is invalid")
        expected_fields = {
            "case_id": case_id,
            "workload": expected_case.workload.id,
            "profile": expected_case.profile,
            "mode": expected_case.mode,
            "phase": expected_case.phase,
            "result_role": expected_case.result_role,
            "source_git_sha": source_sha,
        }
        for field, expected_value in expected_fields.items():
            if payload.get(field) != expected_value:
                raise ClaimError(
                    f"{case_id}: draft {field} is {payload.get(field)!r}, "
                    f"expected {expected_value!r}"
                )
        evidence_class = payload.get("evidence_class")
        run_count = (payload.get("measurement") or {}).get("run_count")
        expected_runs = {
            "five-run-verified": 5,
            "single-run-provisional": 1,
            "two-run-provisional": 2,
        }.get(evidence_class)
        if run_count != expected_runs:
            raise ClaimError(f"{case_id}: draft evidence class/run count mismatch")
        quality = payload.get("quality")
        if isinstance(quality, dict) and quality.get("all_runs_pass") is not True:
            raise ClaimError(f"{case_id}: draft quality gate did not pass")
        repeatability = payload.get("repeatability") or {}
        if evidence_class == "five-run-verified":
            if (
                payload.get("eligible_for_promotion") is not True
                or repeatability.get("passed") is not True
            ):
                raise ClaimError(f"{case_id}: five-run draft evidence is invalid")
        elif (
            payload.get("eligible_for_promotion") is not False
            or payload.get("review_eligible") is not False
        ):
            raise ClaimError(f"{case_id}: provisional evidence eligibility is invalid")
        class_counts[str(evidence_class)] = class_counts.get(str(evidence_class), 0) + 1
        records[case_id] = (entry, payload)

    if set(records) != set(expected):
        raise ClaimError(
            "draft case closure mismatch; "
            f"missing={sorted(set(expected) - set(records))}, "
            f"extra={sorted(set(records) - set(expected))}"
        )
    if class_counts.get("five-run-verified", 0) != index.get(
        "five_run_verified_case_count"
    ) or len(records) - class_counts.get("five-run-verified", 0) != index.get(
        "provisional_case_count"
    ):
        raise ClaimError("draft evidence-class counts do not match index")
    return source_sha, records


def check_registry_baselines(
    records: Mapping[str, tuple[dict[str, Any], dict[str, Any]]],
) -> list[str]:
    errors: list[str] = []
    by_workload: dict[str, dict[str, tuple[dict[str, Any], dict[str, Any]]]] = {}
    for case_id, pair in records.items():
        by_workload.setdefault(str(pair[1]["workload"]), {})[case_id] = pair
    for workload, relative in sorted(
        reference_source_lock.PROMOTED_CONTRACT_PATHS.items()
    ):
        path = ROOT / relative
        contract = yaml.safe_load(path.read_text(encoding="utf-8"))
        baselines = contract.get("verified_baselines") or {}
        expected_cases = by_workload.get(workload, {})
        if set(baselines) != set(expected_cases):
            errors.append(
                f"{relative}: verified_baselines closure differs from evidence; "
                f"expected={sorted(expected_cases)}, actual={sorted(baselines)}"
            )
            continue
        for case_id, (entry, _payload) in expected_cases.items():
            baseline = baselines[case_id]
            for field in ("evidence_id", "evidence_sha256"):
                if baseline.get(field) != entry.get(field):
                    errors.append(f"{relative}: {case_id} has stale {field}")
        canonical_mode = (contract.get("canonical_max_contract") or {}).get("mode")
        canonical_candidates = [
            case_id
            for case_id, (_entry, payload) in expected_cases.items()
            if payload.get("mode") == canonical_mode and payload.get("phase") is None
        ]
        canonical = canonical_candidates[0] if len(canonical_candidates) == 1 else None
        if canonical not in baselines or contract.get(
            "verified_baseline"
        ) != baselines.get(canonical):
            errors.append(
                f"{relative}: canonical verified_baseline is not bound to {canonical!r}"
            )
    return errors


def check_documents(
    source_sha: str,
    records: Mapping[str, tuple[dict[str, Any], dict[str, Any]]],
) -> list[str]:
    errors: list[str] = []
    evidence_workload_ids = sorted(
        {str(payload["workload"]) for _entry, payload in records.values()}
    )
    registry_workload_ids = sorted(load_registry(ROOT / "registry"))
    portfolio_claim = count_claim_pattern(len(registry_workload_ids), "workload")
    evidence_workload_claim = count_claim_pattern(
        len(evidence_workload_ids), "workload"
    )
    case_claim = count_claim_pattern(len(records), "evidence case")
    known_evidence_ids = {
        str(evidence_id)
        for entry, payload in records.values()
        for evidence_id in (
            entry.get("evidence_id"),
            (payload.get("source_summary") or {}).get("evidence_id"),
        )
        if evidence_id
    }
    for name in PUBLIC_DOCUMENTS:
        path = ROOT / name
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"{name}: cannot read document: {exc}")
            continue
        lowered = text.lower()
        for retired in sorted(RETIRED_PUBLIC_IDS):
            if retired in text:
                errors.append(f"{name}: retired public identifier remains: {retired}")
        for evidence_id in EVIDENCE_ID_RE.findall(text):
            if evidence_id not in known_evidence_ids:
                errors.append(f"{name}: unbound evidence ID remains: {evidence_id}")
        if name in WORKLOAD_DOCUMENTS:
            for workload in registry_workload_ids:
                if workload not in text:
                    errors.append(f"{name}: registered workload is missing: {workload}")
            if portfolio_claim.search(lowered) is None:
                errors.append(
                    f"{name}: missing the {len(registry_workload_ids)}-workload portfolio claim"
                )
            if evidence_workload_claim.search(lowered) is None:
                errors.append(
                    f"{name}: missing the {len(evidence_workload_ids)}-workload evidence-scope claim"
                )
            if case_claim.search(lowered) is None:
                errors.append(
                    f"{name}: missing the {len(records)}-evidence-case closure claim"
                )
        if "30 workload" in lowered or "all thirty" in lowered:
            errors.append(f"{name}: stale 30-workload claim remains")
        if "eight retained" in lowered or "eight summaries" in lowered:
            errors.append(f"{name}: stale eight-summary claim remains")
        if (
            name in {"docs/internal/QUALITY_TARGET_REVIEW.md", "docs/internal/RELEASE_CHECKLIST.md"}
            and source_sha not in text
        ):
            errors.append(f"{name}: full evidence source SHA is missing")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify claims without writing files"
    )
    parser.parse_args()
    try:
        if INDEX_PATH.is_file():
            source_sha, records = load_cases()
            errors = check_registry_baselines(records)
            evidence_label = "promoted"
        else:
            source_sha, records = load_provisional_cases()
            errors = []
            evidence_label = "draft"
        errors.extend(check_documents(source_sha, records))
    except (ClaimError, OSError, yaml.YAMLError) as exc:
        print(f"FAIL: reference claim inputs are invalid: {exc}")
        return 1
    if errors:
        print("FAIL: reference claims are out of date")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(
        f"PASS: {len(records)} {evidence_label} case claims across "
        f"{len({payload['workload'] for _entry, payload in records.values()})} workloads"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
