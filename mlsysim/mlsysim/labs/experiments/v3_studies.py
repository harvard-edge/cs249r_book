"""Deterministic agent policy studies over versioned, hypothetical trace fixtures.

These studies do not call an LLM, execute submitted code, or estimate production
attack rates. A recorded branch is a counterfactual teaching record, not a new
sample. Guarded branches actually replay the same operation records through the
selected checks; the result separates denied work, admitted work, and violations.
All project identities come from ``Agents``. Quantitative physical claims belong
in the existing physics layer, never in these policy records.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from importlib.resources import files
import json
from decimal import Decimal
import hashlib
from typing import Any

from mlsysim.agents.registry import Agents
from mlsysim.physics.agents import calc_kv_cache_bytes_per_token

MODEL_VERSION = "agent-policy-traces.v1"


def projects() -> dict[str, dict[str, str]]:
    """The four Volume III lighthouses, not deployment-size tracks."""
    profiles = {
        "repair": (Agents.Coding.SWE_Bench_Runner, "Repair a failing repository task and submit a tested patch.", "Private workspace and admitted repository operations; a patch is not a production deployment.", "repository repair", "candidate patch", "repository maintainer", "publish the candidate patch"),
        "research": (Agents.Research.OrchestratorWorker, "Answer a research question using cited, inspected evidence.", "Untrusted reading and evidence provenance; a retrieved assertion is not an authorized instruction.", "evidence-backed research", "research report", "research requester", "release the report"),
        "remediation": (Agents.Remediation.OnCall, "Diagnose an incident and propose an approved, verified change.", "Shared-state changes and residual duties; approval does not remove action-time checks.", "incident remediation", "change proposal", "on-call operator", "apply the approved change"),
        "support": (Agents.Support.CustomerOps, "Resolve a customer request within the task's permissions and handoff rules.", "Customer identity, private records, response deadline, and truthful handoff; a helpful reply is not proof of settlement.", "customer support", "customer response", "support requester", "send the authorized response"),
    }
    return {key: {"label": p.name, "mission": mission, "boundary": boundary,
                  "task": task, "artifact": artifact, "actor": actor, "effect": effect,
                  "registry": {"repair": "Agents.Coding.SWE_Bench_Runner", "research": "Agents.Research.OrchestratorWorker", "remediation": "Agents.Remediation.OnCall", "support": "Agents.Support.CustomerOps"}[key]}
            for key, (p, mission, boundary, task, artifact, actor, effect) in profiles.items()}


@lru_cache(maxsize=1)
def _catalog() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("early.json", "middle.json", "late.json"):
        path = files("mlsysim").joinpath("engine", "data", "agentic_studies", name)
        if not path.is_file():
            continue
        for chapter, parts in json.loads(path.read_text(encoding="utf-8"))["chapters"].items():
            if chapter in result:
                raise ValueError(f"Duplicate agentic study chapter {chapter}")
            result[chapter] = parts
    return result


def _track(track: str) -> dict[str, str]:
    try:
        return projects()[track]
    except KeyError as exc:
        raise ValueError(f"Unknown lighthouse {track!r}; choose {', '.join(projects())}") from exc


def _render(value: Any, track: str) -> Any:
    substitutions = _track(track)
    if isinstance(value, str):
        for key, replacement in substitutions.items():
            value = value.replace("{" + key + "}", replacement)
        return value
    if isinstance(value, list):
        return [_render(x, track) for x in value]
    if isinstance(value, dict):
        return {key: _render(x, track) for key, x in value.items()}
    return value


def definition(chapter: int, part: str, track: str = "repair") -> dict[str, Any]:
    """Return an independent copy of the canonical study specification."""
    _track(track)
    try:
        spec = _catalog()[str(int(chapter))][part.upper()]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"No study for chapter {chapter!r}, part {part!r}") from exc
    result = _render(deepcopy(spec), track)
    result.setdefault("coverage", {"status": "bounded policy fixture"})
    if "records" not in spec and spec.get("kind") not in ("uncertain_mutation", "grader_calibration", "kv_paging", "kv_pause"):
        result["coverage"]["instrument"] = "fixed counterfactual trace comparison"
        result["choice"] = "Choose a supplied policy trace for comparison. This view does not vary numerical parameters."
    return result


def choices(chapter: int, part: str, track: str = "repair") -> dict[str, str]:
    """Choice ID to human-readable label, in canonical comparison order."""
    return {key: value["label"] for key, value in definition(chapter, part, track)["choices"].items()}


def replay_guards(records: list[dict[str, Any]], checks: list[str]) -> dict[str, Any]:
    """Replay independent proposed operations through selected deterministic checks.

    Every requirement is tri-state: true, false, or unknown (null). A selected
    check admits only true; absence never becomes a pass. The fixture records
    what an unchecked dispatch would do. Counts describe only these records.
    """
    if len(checks) != len(set(checks)):
        raise ValueError("checks must be unique")
    known = {name for record in records for name in record["requirements"]}
    if not set(checks) <= known:
        raise ValueError("Unknown policy check")
    events = []
    admitted = violations = blocked_valid = held_unknown = 0
    for record in records:
        requirements = record["requirements"]
        if any(value is not None and type(value) is not bool for value in requirements.values()):
            raise ValueError("Requirement evidence must be true, false, or null")
        failures = [key for key, value in requirements.items() if value is False]
        unknowns = [key for key in set(requirements) | set(checks)
                    if requirements.get(key) is None]
        blockers = [key for key in checks if requirements.get(key) is not True]
        admitted_here = not blockers
        admitted += admitted_here
        violations += admitted_here and bool(failures)
        blocked_valid += bool(blockers) and not failures and not unknowns
        held_unknown += bool(blockers) and bool(unknowns)
        state = record["effect"] if admitted_here else record.get("hold", "Held without dispatch; obligation remains visible")
        events.append({"step": len(events) + 1, "event": record["event"],
                       "decision": "admit" if admitted_here else "hold",
                       "checked": ", ".join(checks) or "no guard",
                       "blocking_evidence": ", ".join(blockers) or "none",
                       "unmet_requirements": ", ".join(failures) or "none",
                       "unknown_requirements": ", ".join(unknowns) or "none",
                       "state": state,
                       **({"observation_eligible": record["observation_eligible"]}
                          if "observation_eligible" in record else {})})
    return {"events": events, "admitted": admitted, "violations": violations,
            "blocked_valid": blocked_valid, "held_unknown": held_unknown}


def _branch(spec: dict[str, Any], choice: str) -> dict[str, Any]:
    policy = spec["choices"][choice]
    if spec.get("kind") == "kv_paging":
        block_size = policy["block_tokens"]
        lengths = spec["sequence_tokens"]
        blocks = [(length + block_size - 1) // block_size for length in lengths]
        useful = sum(lengths)
        slots = sum(blocks) * block_size
        bytes_per_token = calc_kv_cache_bytes_per_token(Agents.Coding.SWE_Bench_Runner).to("byte").magnitude
        metrics = {"useful_tokens": useful, "allocated_blocks": sum(blocks), "allocated_token_slots": slots,
                   "unused_token_slots": slots - useful, "resident_kv_bytes": int(slots * bytes_per_token),
                   "declared_pool_fits": slots <= spec["kv_pool_token_slots"]}
        return {"events": [{"step": index + 1, "event": f"Reference sequence {index + 1}",
                            "useful_tokens": length, "allocated_blocks": count,
                            "allocated_slots": count * block_size}
                           for index, (length, count) in enumerate(zip(lengths, blocks))],
                "outcome": f"KV pool fit: {metrics['declared_pool_fits']}; unused slots: {metrics['unused_token_slots']}; block-table entries: {metrics['allocated_blocks']}.",
                "feedback": policy["feedback"], "computed_metrics": metrics}
    if spec.get("kind") == "kv_pause":
        tokens = spec["paused_prefix_tokens"]
        resident = policy["resident"]
        bytes_per_token = calc_kv_cache_bytes_per_token(Agents.Coding.SWE_Bench_Runner).to("byte").magnitude
        held = int(tokens * bytes_per_token) if resident else 0
        metrics = {"kv_bytes_while_paused": held, "kv_byte_seconds": held * spec["wait_seconds"],
                   "resume_prefill_tokens": 0 if resident else tokens}
        return {"events": [{"step": 1, "event": "Reference prefix paused during an external test",
                            "state": "KV remains resident" if resident else "Derived KV evicted; canonical token prefix retained"},
                           {"step": 2, "event": "External test returns; prefix is unchanged",
                            "state": "Reuse exact KV" if resident else "Recompute the exact prefix before continuation"}],
                "outcome": f"Paused KV bytes: {held}; resume prefill tokens: {metrics['resume_prefill_tokens']}.",
                "feedback": policy["feedback"], "computed_metrics": metrics}
    if spec.get("kind") == "grader_calibration":
        threshold = policy["minimum_checks"]
        tp = fp = tn = fn = 0
        events = []
        for candidate in spec["candidates"]:
            accepts = candidate["checks_passed"] >= threshold
            reference = candidate["reference_accepts"]
            tp += accepts and reference
            fp += accepts and not reference
            tn += not accepts and not reference
            fn += not accepts and reference
            events.append({"step": len(events) + 1, "event": candidate["candidate_id"],
                           "checks_passed": candidate["checks_passed"], "threshold": threshold,
                           "grader": "accept" if accepts else "reject",
                           "independent_reference": "accept" if reference else "reject"})
        return {"events": events, "feedback": policy["feedback"],
                "outcome": f"Reference-passing accepted: {tp}; false acceptances: {fp}; reference-passing rejected: {fn}; reference-failing rejected: {tn}.",
                "calibration": {"reference_failures": fp + tn, "grader_acceptances": tp + fp,
                                "false_acceptances": fp, "false_rejections": fn,
                                "false_acceptance_fraction": f"{fp}/{fp + tn}" if fp + tn else "undefined: no reference failures",
                                "failure_share_among_accepted": f"{fp}/{tp + fp}" if tp + fp else "undefined: no grader acceptances"}}
    if spec.get("kind") == "uncertain_mutation":
        endpoint = spec["endpoint"]
        ledger = [endpoint["operation_id"]] if endpoint["committed"] else []
        events = [{"step": 1, "event": "Original request lost its reply",
                   "state": "Caller outcome remains unknown; reservation stays open"}]
        probes = 0
        obligation = "in doubt"
        if policy["mode"] == "retry":
            ledger.append(endpoint["operation_id"] + "/another-event")
            events.append({"step": 2, "event": "Another event-producing request dispatched",
                           "state": "Additional endpoint effect committed; original obligation is not settled by the new reply"})
        elif policy["mode"] == "reconcile":
            probes += 1
            if endpoint["probe_available"]:
                obligation = "established" if endpoint["committed"] else "proven absent"
                events.append({"step": 2, "event": "Probe checks the captured baseline and bound operation identity",
                               "state": obligation})
            else:
                events.append({"step": 2, "event": "Probe unavailable", "state": "Still in doubt; no retry admitted"})
        else:
            events.append({"step": 2, "event": "Hold and retain ownership", "state": "No additional request; review still required"})
        return {"events": events, "outcome": f"Endpoint effects in supplied fixture: {len(ledger)}; authorized probes: {probes}; caller obligation: {obligation}.",
                "feedback": policy["feedback"], "endpoint_effects": len(ledger), "probes": probes,
                "obligation": obligation}
    if "records" in spec:
        result = replay_guards(spec["records"], policy["checks"])
        result["outcome"] = (f"Admitted {result['admitted']} fixture operations; "
                             f"{result['violations']} admitted despite false requirements; "
                             f"{result['held_unknown']} held with unknown evidence.")
        result["feedback"] = policy["feedback"]
        return result
    events = [dict(event, step=index + 1) if isinstance(event, dict)
              else {"step": index + 1, "event": event}
              for index, event in enumerate(policy["events"])]
    return {"events": events, "outcome": policy["outcome"], "feedback": policy["feedback"]}


def cohort_accounting(records: list[dict[str, Any]], accounting: str = "all_initiated") -> dict[str, Any]:
    """Compute a supplied cohort's exact-dollar accounting, including failures.

    ``accepted_only`` deliberately reproduces the exclusion error for contrast;
    true total spending always remains visible. No population expectation is
    estimated from the fixture and a zero accepted denominator is undefined.
    """
    if accounting not in ("all_initiated", "accepted_only"):
        raise ValueError("Unknown accounting policy")
    if any(type(r["accepted"]) is not bool for r in records):
        raise ValueError("Cohort acceptance must be a boolean reference verdict")
    costs = [Decimal(str(r["cost_usd"])) for r in records]
    if any(not value.is_finite() or value < 0 for value in costs):
        raise ValueError("Cohort costs must be finite and nonnegative")
    accepted = sum(r["accepted"] for r in records)
    total = sum(costs, Decimal(0))
    reported = sum((cost for cost, r in zip(costs, records)
                    if accounting == "all_initiated" or r["accepted"]), Decimal(0))
    return {"initiated_tasks": len(records), "accepted_tasks": accepted,
            "total_spend_usd": str(total), "reported_spend_usd": str(reported),
            "reported_cost_per_accepted_usd": str(reported / accepted) if accepted else "undefined: no accepted task",
            "actual_cost_per_accepted_usd": str(total / accepted) if accepted else "undefined: no accepted task"}


def study(chapter: int, part: str, choice: str, track: str = "repair") -> dict[str, Any]:
    """Return baseline and selected-policy evidence with explicit provenance."""
    spec = definition(chapter, part, track)
    if choice not in spec["choices"]:
        raise ValueError(f"Unknown choice {choice!r}; choose {', '.join(spec['choices'])}")
    baseline = next(iter(spec["choices"]))
    rows = []
    branches = {}
    for role, key in (("Starting policy", baseline), ("Selected policy", choice)):
        branch = _branch(spec, key)
        branches[role] = branch
        row = {"case": role, "policy": spec["choices"][key]["label"], "outcome": branch["outcome"],
               "events_recorded": len(branch["events"])}
        row.update(spec["choices"][key].get("evidence", {}))
        if "records" in spec:
            row.update({"admitted_operations": branch["admitted"],
                        "false_requirement_dispatches": branch["violations"],
                        "held_unknown": branch["held_unknown"]})
        if spec.get("kind") == "uncertain_mutation":
            row.update({key: branch[key] for key in ("endpoint_effects", "probes", "obligation")})
        if "calibration" in branch:
            row.update(branch["calibration"])
        if "computed_metrics" in branch:
            row.update(branch["computed_metrics"])
        if "cohort" in spec["choices"][key]:
            row.update(cohort_accounting(spec["choices"][key]["cohort"],
                                         spec["choices"][key].get("accounting", "all_initiated")))
        rows.append(row)
    selected = branches["Selected policy"]
    return {"title": spec["title"], "source_concepts": spec["source_concepts"],
            "rows": rows, "primary_fields": spec.get("primary_fields", {}), "events": selected["events"], "trace": selected["events"],
            "baseline_events": branches["Starting policy"]["events"],
            "feedback": selected["feedback"], "limits": [
                "Versioned hypothetical fixture, not a live model sample, benchmark rate, or production certification.",
                "Changing the lighthouse changes the mission and artifact framing; this controlled policy fixture does not infer workload-specific performance.",
                *spec["limits"]],
            "reversal_probe": spec["reversal_probe"], "fixed": spec["fixed"],
            "coverage": spec["coverage"], "transfer_scope": spec.get("transfer_scope", "Repair-primary reference"),
            "source": {"catalog": "engine/data/agentic_studies", "registry": _track(track)["registry"],
                       "fixture_id": f"agent-policy/{int(chapter):02}/{part.upper()}",
                       "physical_reference": ({"registry": "Agents.Coding.SWE_Bench_Runner.reference_model",
                                               "model": Agents.Coding.SWE_Bench_Runner.reference_model.name,
                                               "formula": "physics.agents.calc_kv_cache_bytes_per_token",
                                               "precision": "canonical formula default: two bytes per element"}
                                              if spec.get("kind") in ("kv_paging", "kv_pause") else None),
                       "kind": spec.get("kind", "guarded operation replay" if "records" in spec else "authored counterfactual trace replay")},
            "model_version": MODEL_VERSION,
            "inputs": {"chapter": int(chapter), "part": part.upper(), "track": track,
                       "choice": choice, "baseline": baseline,
                       "fixture_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
                       "fixture": {key: deepcopy(spec[key]) for key in
                                   ("records", "endpoint", "candidates", "sequence_tokens", "kv_pool_token_slots", "paused_prefix_tokens", "wait_seconds")
                                   if key in spec},
                       "selected_policy": deepcopy(spec["choices"][choice])}}
