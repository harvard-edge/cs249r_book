"""Responsible-engineering experiments for the Volume I Chapter 15 lab.

Every profile and fixture in this module is an illustrative teaching scenario,
not an empirical measurement or a claim about a named product.  Fixed labels
and scores make threshold experiments internally coherent.  Physical work is
represented with Pint quantities, and lifecycle accounting states its boundary.
"""

from __future__ import annotations

import math
import random
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from mlsysim.core.units import Q_


_GROUPS = ("reference", "affected")
_LINEAGE_FIELDS = (
    "model_version",
    "decision_threshold",
    "feature_hash",
    "training_data_version",
    "deployment_id",
)
_LINEAGE_BYTES = {
    "model_version": 32,
    "decision_threshold": 8,
    "feature_hash": 32,
    "training_data_version": 32,
    "deployment_id": 16,
}
LINEAGE_POLICIES = {
    "full": _LINEAGE_FIELDS,
    "decision_only": ("model_version", "decision_threshold", "feature_hash", "deployment_id"),
    "minimal": ("model_version", "deployment_id"),
}

# Scores are confidence values for the positive decision; labels are observed
# binary outcomes for the same task within a track.  The fixtures stay fixed
# when population weights or decision thresholds change.
_SCORES = {
    "tinyml": {
        "reference": ((0.94, 1), (0.86, 1), (0.78, 1), (0.67, 1), (0.58, 1), (0.42, 1), (0.61, 0), (0.48, 0), (0.34, 0), (0.22, 0), (0.12, 0), (0.05, 0)),
        "affected": ((0.87, 1), (0.73, 1), (0.62, 1), (0.51, 1), (0.39, 1), (0.28, 1), (0.71, 0), (0.57, 0), (0.44, 0), (0.31, 0), (0.18, 0), (0.08, 0)),
    },
    "mobile": {
        "reference": ((0.96, 1), (0.88, 1), (0.81, 1), (0.72, 1), (0.64, 1), (0.46, 1), (0.59, 0), (0.47, 0), (0.36, 0), (0.25, 0), (0.14, 0), (0.06, 0)),
        "affected": ((0.89, 1), (0.76, 1), (0.65, 1), (0.53, 1), (0.41, 1), (0.29, 1), (0.74, 0), (0.60, 0), (0.45, 0), (0.33, 0), (0.20, 0), (0.09, 0)),
    },
    "edge": {
        "reference": ((0.97, 1), (0.91, 1), (0.84, 1), (0.75, 1), (0.63, 1), (0.49, 1), (0.57, 0), (0.43, 0), (0.32, 0), (0.21, 0), (0.11, 0), (0.03, 0)),
        "affected": ((0.90, 1), (0.79, 1), (0.68, 1), (0.56, 1), (0.43, 1), (0.27, 1), (0.76, 0), (0.62, 0), (0.46, 0), (0.35, 0), (0.19, 0), (0.07, 0)),
    },
    "cloud": {
        "reference": ((0.95, 1), (0.89, 1), (0.82, 1), (0.70, 1), (0.60, 1), (0.45, 1), (0.63, 0), (0.49, 0), (0.37, 0), (0.24, 0), (0.13, 0), (0.04, 0)),
        "affected": ((0.88, 1), (0.74, 1), (0.64, 1), (0.52, 1), (0.38, 1), (0.25, 1), (0.72, 0), (0.58, 0), (0.43, 0), (0.30, 0), (0.17, 0), (0.06, 0)),
    },
}

TRACKS: dict[str, dict[str, Any]] = {
    "tinyml": {
        "display": "TinyML",
        "scenario": "An on-device wake-word detector",
        "positive_outcome": "intended wake words detected",
        "false_positive_consequence": "unwanted device activations",
        "false_negative_consequence": "missed intended commands",
        "default_affected_share_pct": 25.0,
        "privacy_values": (1, 3, 2, 5, 0, 4, 2, 1, 6, 3, 2, 4),
        "privacy_bounds": (0.0, 12.0),
        "privacy_value_label": "wake-word events per participant-day",
        "record_bytes": 96,
        "training_energy": Q_(1.8, "kilowatt_hour"),
        "retraining_energy": Q_(1.2, "kilowatt_hour"),
        "inference_energy": Q_(0.35, "millijoule"),
        "explanation_energy": Q_(0.08, "millijoule"),
        "daily_inferences": 80_000,
        "years": 3.0,
        "explanation_share_pct": 2.0,
        "carbon_intensity": Q_(180, "gram / kilowatt_hour"),
        "incident_decisions": 2_400,
        "decision_rate_per_hour": 1_000.0,
        "default_containment_delay_hours": 6.0,
    },
    "mobile": {
        "display": "Mobile",
        "scenario": "An on-device accessibility classifier",
        "positive_outcome": "requested content recognized",
        "false_positive_consequence": "incorrect spoken identifications",
        "false_negative_consequence": "requested content left unidentified",
        "default_affected_share_pct": 35.0,
        "privacy_values": (4, 8, 12, 7, 3, 16, 9, 5, 11, 6, 14, 10),
        "privacy_bounds": (0.0, 40.0),
        "privacy_value_label": "feature uses per participant-day",
        "record_bytes": 160,
        "training_energy": Q_(18, "kilowatt_hour"),
        "retraining_energy": Q_(12, "kilowatt_hour"),
        "inference_energy": Q_(18, "millijoule"),
        "explanation_energy": Q_(5, "millijoule"),
        "daily_inferences": 2_000_000,
        "years": 3.0,
        "explanation_share_pct": 5.0,
        "carbon_intensity": Q_(260, "gram / kilowatt_hour"),
        "incident_decisions": 8_000,
        "decision_rate_per_hour": 12_000.0,
        "default_containment_delay_hours": 4.0,
    },
    "edge": {
        "display": "Edge",
        "scenario": "A local visual-defect inspection service",
        "positive_outcome": "defective items intercepted",
        "false_positive_consequence": "sound items diverted for review",
        "false_negative_consequence": "defective items released",
        "default_affected_share_pct": 20.0,
        "privacy_values": (2, 6, 5, 9, 3, 7, 4, 11, 8, 5, 6, 10),
        "privacy_bounds": (0.0, 20.0),
        "privacy_value_label": "flagged events per operator-day",
        "record_bytes": 224,
        "training_energy": Q_(72, "kilowatt_hour"),
        "retraining_energy": Q_(48, "kilowatt_hour"),
        "inference_energy": Q_(110, "millijoule"),
        "explanation_energy": Q_(45, "millijoule"),
        "daily_inferences": 5_000_000,
        "years": 4.0,
        "explanation_share_pct": 10.0,
        "carbon_intensity": Q_(330, "gram / kilowatt_hour"),
        "incident_decisions": 25_000,
        "decision_rate_per_hour": 60_000.0,
        "default_containment_delay_hours": 2.0,
    },
    "cloud": {
        "display": "Cloud",
        "scenario": "A shared eligibility decision service",
        "positive_outcome": "eligible requests accepted",
        "false_positive_consequence": "ineligible requests accepted",
        "false_negative_consequence": "eligible requests denied",
        "default_affected_share_pct": 40.0,
        "privacy_values": (8, 22, 35, 14, 41, 18, 29, 53, 11, 25, 37, 20),
        "privacy_bounds": (0.0, 100.0),
        "privacy_value_label": "requests per account-day",
        "record_bytes": 320,
        "training_energy": Q_(640, "kilowatt_hour"),
        "retraining_energy": Q_(420, "kilowatt_hour"),
        "inference_energy": Q_(0.9, "joule"),
        "explanation_energy": Q_(0.35, "joule"),
        "daily_inferences": 25_000_000,
        "years": 3.0,
        "explanation_share_pct": 15.0,
        "carbon_intensity": Q_(410, "gram / kilowatt_hour"),
        "incident_decisions": 100_000,
        "decision_rate_per_hour": 250_000.0,
        "default_containment_delay_hours": 1.0,
    },
}

_ASSUMPTION = "Illustrative fixed teaching scenario; not an empirical deployment measurement."
_LIFECYCLE_BOUNDARY = (
    "One initial training run, selected retraining runs, and on-system inference plus "
    "requested decision explanations over the selected operating period. Embodied "
    "energy, networking, storage, and end-user equipment are outside this boundary."
)


def _track(track_id: str) -> Mapping[str, Any]:
    if track_id not in TRACKS:
        raise ValueError(f"track_id must be one of {', '.join(TRACKS)}")
    return TRACKS[track_id]


def _finite(value: Any, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _confusion(rows: Sequence[tuple[float, int]], threshold: float) -> dict[str, float | int]:
    tp = sum(label == 1 and score >= threshold for score, label in rows)
    fn = sum(label == 1 and score < threshold for score, label in rows)
    fp = sum(label == 0 and score >= threshold for score, label in rows)
    tn = sum(label == 0 and score < threshold for score, label in rows)
    total = len(rows)
    positives = tp + fn
    negatives = fp + tn
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "count": total,
        "accuracy": (tp + tn) / total,
        "selection_rate": (tp + fp) / total,
        "tpr": tp / positives,
        "fpr": fp / negatives,
        "fnr": fn / positives,
    }


def threshold_consequences(
    track_id: str,
    *,
    threshold: float = 0.5,
    affected_share_pct: float | None = None,
    population_size: int = 10_000,
) -> dict[str, Any]:
    """Evaluate one positive-score threshold on fixed labels and scores.

    A decision is positive exactly when ``score >= threshold``.  Consequently,
    raising the threshold cannot increase either subgroup's TPR or FPR.
    Consequences are reported as expected decision counts, without assigning an
    invented moral or monetary weight to either error type.
    """
    track = _track(track_id)
    threshold = _finite(threshold, "threshold", minimum=0.0, maximum=1.0)
    population_size = _positive_int(population_size, "population_size")
    if affected_share_pct is None:
        affected_share_pct = track["default_affected_share_pct"]
    affected_share_pct = _finite(
        affected_share_pct, "affected_share_pct", minimum=0.0, maximum=100.0
    )
    weights = {"reference": 1.0 - affected_share_pct / 100.0, "affected": affected_share_pct / 100.0}
    groups = {group: _confusion(_SCORES[track_id][group], threshold) for group in _GROUPS}
    aggregate = {
        metric: sum(weights[group] * float(groups[group][metric]) for group in _GROUPS)
        for metric in ("accuracy", "selection_rate", "tpr", "fpr", "fnr")
    }
    expected_fp = population_size * sum(
        weights[group] * float(groups[group]["fp"]) / float(groups[group]["count"])
        for group in _GROUPS
    )
    expected_fn = population_size * sum(
        weights[group] * float(groups[group]["fn"]) / float(groups[group]["count"])
        for group in _GROUPS
    )
    return {
        "track_id": track_id,
        "scenario": track["scenario"],
        "threshold": threshold,
        "affected_share_pct": affected_share_pct,
        "population_size": population_size,
        "groups": groups,
        "aggregate": aggregate,
        "tpr_gap_pp": 100.0 * (float(groups["reference"]["tpr"]) - float(groups["affected"]["tpr"])),
        "fpr_gap_pp": 100.0 * (float(groups["reference"]["fpr"]) - float(groups["affected"]["fpr"])),
        "expected_false_positives": expected_fp,
        "expected_false_negatives": expected_fn,
        "false_positive_consequence": track["false_positive_consequence"],
        "false_negative_consequence": track["false_negative_consequence"],
        "assumption": _ASSUMPTION,
        "inputs": {
            "track_id": track_id,
            "threshold": threshold,
            "affected_share_pct": affected_share_pct,
            "population_size": population_size,
        },
    }


def population_mix(
    track_id: str,
    *,
    affected_share_pct: float | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Change population weights while preserving subgroup prediction rules."""
    result = threshold_consequences(
        track_id, threshold=threshold, affected_share_pct=affected_share_pct
    )
    return {
        "track_id": track_id,
        "scenario": result["scenario"],
        "threshold": result["threshold"],
        "affected_share_pct": result["affected_share_pct"],
        "groups": result["groups"],
        "aggregate": result["aggregate"],
        "assumption": _ASSUMPTION,
        "inputs": {
            "track_id": track_id,
            "threshold": result["threshold"],
            "affected_share_pct": result["affected_share_pct"],
        },
    }


# Deliberately unequal base rates with equal TPR/FPR at threshold 0.5.
# Fixed fictional affected populations; no claim about real demographic groups.
_FAIRNESS_REFERENCE = {
    "reference": (
        ((0.9, 1),) * 6
        + ((0.6, 1),) * 6
        + ((0.4, 1),) * 2
        + ((0.1, 1),) * 2
        + ((0.6, 0),)
        + ((0.4, 0),) * 2
        + ((0.1, 0),)
    ),
    "affected": (
        ((0.9, 1),) * 3
        + ((0.6, 1),) * 3
        + ((0.4, 1),)
        + ((0.1, 1),)
        + ((0.9, 0),)
        + ((0.6, 0),) * 2
        + ((0.4, 0),) * 4
        + ((0.1, 0),) * 5
    ),
}
_FAIRNESS_CRITERIA = ("demographic_parity", "equal_opportunity", "equalized_odds")


def fairness_criteria(
    track_id: str,
    *,
    criterion: str = "equal_opportunity",
    threshold: float = 0.5,
    tolerance_pp: float = 5.0,
    minimum_group_count: int = 8,
) -> dict[str, Any]:
    """Contrast declared criteria on identical audited predictions and outcomes.

    The fixture demonstrates differing gates, not a proof that every criterion
    conflicts in every setting. Tolerance is a declared teaching policy, not a
    legal standard. Edge uses a recommendation-policy reference involving
    affected people; road objects are not demographic groups.
    """
    _track(track_id)
    if criterion not in _FAIRNESS_CRITERIA:
        raise ValueError("criterion must be one of " + ", ".join(_FAIRNESS_CRITERIA))
    threshold = _finite(threshold, "threshold", minimum=0, maximum=1)
    tolerance_pp = _finite(tolerance_pp, "tolerance_pp", minimum=0, maximum=100)
    minimum_group_count = _positive_int(minimum_group_count, "minimum_group_count")
    groups = {name: _confusion(rows, threshold) for name, rows in _FAIRNESS_REFERENCE.items()}
    for group in groups.values():
        group["base_rate"] = (group["tp"] + group["fn"]) / group["count"]
    gaps = {
        metric: 100 * abs(float(groups["reference"][metric]) - float(groups["affected"][metric]))
        for metric in ("selection_rate", "tpr", "fpr")
    }
    criterion_gaps = {
        "demographic_parity": gaps["selection_rate"],
        "equal_opportunity": gaps["tpr"],
        "equalized_odds": max(gaps["tpr"], gaps["fpr"]),
    }
    sufficient = all(group["count"] >= minimum_group_count for group in groups.values())
    gates = {
        name: (
            "insufficient_evidence" if not sufficient else "pass" if gap <= tolerance_pp else "fail"
        )
        for name, gap in criterion_gaps.items()
    }
    return {
        "track_id": track_id,
        "reference_name": "Affected-population recommendation-policy reference",
        "scenario": "Fixed audited recommendation opportunities for two fictional affected populations",
        "groups": groups,
        "gap_pp": gaps,
        "criterion_gap_pp": criterion_gaps,
        "criterion_gates": gates,
        "selected_criterion": criterion,
        "selected_gate": gates[criterion],
        "tolerance_pp": tolerance_pp,
        "minimum_group_count": minimum_group_count,
        "threshold": threshold,
        "evidence_sufficient": sufficient,
        "assumption": _ASSUMPTION,
        "limitation": "A chosen metric is a justified obligation, not a universal fairness verdict; this reference does not certify the native project, especially perception fairness.",
        "inputs": {
            "track_id": track_id,
            "criterion": criterion,
            "threshold": threshold,
            "tolerance_pp": tolerance_pp,
            "minimum_group_count": minimum_group_count,
        },
    }


def selective_feedback(
    track_id: str,
    *,
    referral_threshold: float = 0.5,
    audit_policy: str = "none",
) -> dict[str, Any]:
    """Reveal labels under referral selection and an independent audit policy.

    Routine labels exist only for referred opportunities. The sample audit uses
    a fixed request-ID panel chosen independently of scores and outcomes; it is
    not population ground truth. A complete audit reveals the finite supplied
    population, recovering its denominator without projecting future feedback.
    """
    _track(track_id)
    referral_threshold = _finite(referral_threshold, "referral_threshold", minimum=0, maximum=1)
    if audit_policy not in ("none", "independent_sample", "complete"):
        raise ValueError("audit_policy must be none, independent_sample, or complete")
    # Stable shuffled identities break any association between panel membership
    # and the fixture's storage order. The same panel is reused at every threshold.
    rows = [
        (group, index, score, label)
        for group, values in _FAIRNESS_REFERENCE.items()
        for index, (score, label) in enumerate(values)
    ]
    panel = set(random.Random(15).sample(range(len(rows)), len(rows) // 2))
    groups = {}
    for group in _GROUPS:
        all_rows = [(position, row) for position, row in enumerate(rows) if row[0] == group]
        routine = [(position, row) for position, row in all_rows if row[2] >= referral_threshold]
        audited = [
            (position, row)
            for position, row in all_rows
            if audit_policy == "complete"
            or (audit_policy == "independent_sample" and position in panel)
        ]
        observed = {position: row for position, row in routine + audited}
        positives = sum(row[3] for row in observed.values())
        groups[group] = {
            "eligible_count": len(all_rows),
            "referred_count": len(routine),
            "routine_positive_count": sum(row[3] for _, row in routine),
            "routine_positive_rate": (
                None if not routine else sum(row[3] for _, row in routine) / len(routine)
            ),
            "audit_count": len(audited),
            "additional_audit_labels": len(observed) - len(routine),
            "observed_count": len(observed),
            "observed_positive_count": positives,
            "observed_positive_rate": None if not observed else positives / len(observed),
            "unobserved_count": len(all_rows) - len(observed),
            "population_positive_rate": (
                positives / len(all_rows) if len(observed) == len(all_rows) else None
            ),
        }
    return {
        "track_id": track_id,
        "reference_name": "Selective recommendation-feedback audit reference",
        "groups": groups,
        "audit_policy": audit_policy,
        "population_outcomes_identified": all(
            group["unobserved_count"] == 0 for group in groups.values()
        ),
        "additional_audit_labels": sum(
            group["additional_audit_labels"] for group in groups.values()
        ),
        "assumption": _ASSUMPTION,
        "limitation": "Routine feedback is selected, and a partial panel does not identify all population outcomes. Audit counts represent observation work, not a trained mitigation or a future-loop forecast.",
        "inputs": {
            "track_id": track_id,
            "referral_threshold": referral_threshold,
            "audit_policy": audit_policy,
        },
    }


# Fixed fictional opportunities: clicks are the optimized proxy; an independent
# follow-up records whether the recommendation was helpful. Policies use the
# same six-slot budget. Records are supplied outcomes, not response curves.
FEEDBACK_MAX_ROUNDS = 3
_FEEDBACK_SEED = {"reference": (3, 4), "affected": (1, 4)}
_FEEDBACK_OUTCOMES = {"reference": (1, 1, 0, 0), "affected": (1, 1, 1, 0)}

_PROXY_RECORDS = tuple(
    zip(
        (1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0),
        (0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0),
    )
)
_PROXY_POLICIES = {"proxy_first": tuple(range(6)), "benefit_first": tuple(range(4, 10))}


def proxy_alignment(track_id: str, *, policy: str = "proxy_first", outcome_audit: str = "complete") -> dict[str, Any]:
    """Compare two supplied six-slot policies without equating clicks to benefit."""
    _track(track_id)
    if policy not in _PROXY_POLICIES:
        raise ValueError("unsupported policy")
    if outcome_audit not in ("none", "complete"):
        raise ValueError("outcome_audit must be none or complete")
    rows = [
        {
            "opportunity": index,
            "click": _PROXY_RECORDS[index][0],
            "independent_helpful_outcome": _PROXY_RECORDS[index][1] if outcome_audit == "complete" else None,
        }
        for index in _PROXY_POLICIES[policy]
    ]
    benefit = sum(row["independent_helpful_outcome"] for row in rows) if outcome_audit == "complete" else None
    return {
        "track_id": track_id,
        "reference_name": "Recommendation proxy-versus-benefit reference",
        "policy": policy,
        "slots": len(rows),
        "proxy_clicks": sum(row["click"] for row in rows),
        "independent_benefits": benefit,
        "minimum_benefits": 4,
        "benefit_gate": "insufficient_evidence" if benefit is None else "pass" if benefit >= 4 else "fail",
        "independent_followups": len(rows) if outcome_audit == "complete" else 0,
        "records": rows,
        "assumption": _ASSUMPTION,
        "limitation": "Fixed supplied outcomes compare two policies on one finite fixture. Helpful outcomes are independent follow-up labels, not clicks. No causal effect, trained improvement, or deployment certification is inferred.",
        "inputs": {"track_id": track_id, "policy": policy, "outcome_audit": outcome_audit},
    }


def feedback_replay(track_id: str, *, observation_policy: str = "routine", rounds: int = 3) -> dict[str, Any]:
    """Replay selected labels into the next policy on a finite supplied stream.

    Each round has four new opportunities per population with fixed outcomes.
    A segment is exposed when its cumulative observed success rate reaches 0.5.
    The historical seed is explicitly imperfect, not a fitted population model.
    Independent follow-up reveals outcomes without serving those slots; serving
    exploration instead exchanges one ranked slot for a predeclared hidden slot.
    No missing label becomes a negative; no trajectory beyond three rounds exists.
    """
    _track(track_id)
    if observation_policy not in ("routine", "independent_sample", "complete", "exploration"):
        raise ValueError("unsupported observation_policy")
    rounds = _positive_int(rounds, "rounds")
    if rounds > FEEDBACK_MAX_ROUNDS:
        raise ValueError("only three supplied rounds exist")
    outcomes = _FEEDBACK_OUTCOMES
    counts = {group: list(seed) for group, seed in _FEEDBACK_SEED.items()}
    history = []
    for interval in range(1, rounds + 1):
        estimates = {group: value[0] / value[1] for group, value in counts.items()}
        ranked = {(group, index) for group in _GROUPS if estimates[group] >= 0.5 for index in range(4)}
        exposed = set(ranked)
        displaced = 0
        if observation_policy == "exploration" and ("affected", 0) not in exposed:
            # IDs are declared before outcome inspection, not chosen by labels.
            if ("reference", 3) in exposed:
                exposed.remove(("reference", 3))
                displaced = 1
            exposed.add(("affected", 0))
        panel = (
            {(group, 0) for group in _GROUPS}
            if observation_policy == "independent_sample"
            else {(group, index) for group in _GROUPS for index in range(4)}
            if observation_policy == "complete"
            else set()
        )
        observed = exposed | panel
        rows = []
        for group in _GROUPS:
            for index, label in enumerate(outcomes[group]):
                seen = (group, index) in observed
                rows.append(
                    {
                        "request": f"round-{interval}-{group}-{index}",
                        "population": group,
                        "exposed": (group, index) in exposed,
                        "observed": seen,
                        "label": label if seen else None,
                    }
                )
                if seen:
                    counts[group][0] += label
                    counts[group][1] += 1
        groups = {
            group: {
                "estimate_before": estimates[group],
                "estimate_after": counts[group][0] / counts[group][1],
                "exposed": sum((group, i) in exposed for i in range(4)),
                "observed": sum((group, i) in observed for i in range(4)),
                "unobserved": sum((group, i) not in observed for i in range(4)),
                "realized_benefits": sum(outcomes[group][i] for i in range(4) if (group, i) in exposed),
            }
            for group in _GROUPS
        }
        history.append(
            {
                "round": interval,
                "groups": groups,
                "records": rows,
                "additional_followups": len(panel - exposed),
                "displaced_ranked_slots": displaced,
                "exposure_gap": abs(groups["reference"]["exposed"] - groups["affected"]["exposed"]),
            }
        )
    return {
        "track_id": track_id,
        "reference_name": "Finite recommendation-feedback replay",
        "seed_labels": {group: {"successes": seed[0], "count": seed[1]} for group, seed in _FEEDBACK_SEED.items()},
        "opportunities_per_population_per_round": len(outcomes["reference"]),
        "update_rule": "Expose when cumulative observed success rate reaches one half; append observed labels only.",
        "history": history,
        "total_followups": sum(row["additional_followups"] for row in history),
        "total_displaced_slots": sum(row["displaced_ranked_slots"] for row in history),
        "total_benefits": sum(g["realized_benefits"] for row in history for g in row["groups"].values()),
        "final_exposure_gap": history[-1]["exposure_gap"],
        "assumption": _ASSUMPTION,
        "limitation": "Three supplied rounds and an explicit empirical-rate update, not a fitted geometric growth law or forecast. Independent follow-up assumes these outcomes are observable without exposure; that assumption fails for clicks on unseen items or repayment on denied loans. Exploration consumes ranked slots and does not guarantee fairness. Native-project certification and causal intervention effects are outside this reference.",
        "inputs": {"track_id": track_id, "observation_policy": observation_policy, "rounds": rounds},
    }


def _laplace(rng: random.Random, scale: float) -> float:
    uniform = rng.random() - 0.5
    return -scale * math.copysign(1.0, uniform) * math.log1p(-2.0 * abs(uniform))


def bounded_mean_release(
    track_id: str,
    *,
    epsilon_per_release: float = 1.0,
    releases: int = 1,
    seed: int = 15,
) -> dict[str, Any]:
    """Simulate a Laplace release of a mean over fixed public bounds.

    The mechanism clips each value to public ``[lower, upper]`` bounds, computes
    the mean over a fixed public denominator ``n``, and adds Laplace noise with
    scale ``(upper - lower) / (n * epsilon_per_release)``.  Basic sequential
    composition reports ``releases * epsilon_per_release`` for independent new
    releases over the same private data.  Clipping the released value back to
    the public bounds is postprocessing and adds no privacy spend.

    ``seed`` exists only for repeatable simulation and debugging.  A fixed public
    seed must never be used to implement a deployed privacy mechanism.
    """
    track = _track(track_id)
    epsilon_per_release = _finite(epsilon_per_release, "epsilon_per_release", minimum=0.0)
    if epsilon_per_release == 0:
        raise ValueError("epsilon_per_release must be greater than 0")
    releases = _positive_int(releases, "releases")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    lower, upper = track["privacy_bounds"]
    values = tuple(min(upper, max(lower, float(value))) for value in track["privacy_values"])
    sample_size = len(values)
    true_mean = sum(values) / sample_size
    sensitivity = (upper - lower) / sample_size
    noise_scale = sensitivity / epsilon_per_release
    rng = random.Random(seed)
    raw_releases = tuple(true_mean + _laplace(rng, noise_scale) for _ in range(releases))
    released_values = tuple(min(upper, max(lower, value)) for value in raw_releases)
    errors = tuple(abs(value - true_mean) for value in released_values)
    return {
        "track_id": track_id,
        "scenario": track["scenario"],
        "mechanism": "bounded-mean Laplace mechanism",
        "value_label": track["privacy_value_label"],
        "public_bounds": (lower, upper),
        "sample_size": sample_size,
        "true_bounded_mean": true_mean,
        "l1_sensitivity": sensitivity,
        "epsilon_per_release": epsilon_per_release,
        "releases": releases,
        "basic_composed_epsilon": releases * epsilon_per_release,
        "laplace_scale": noise_scale,
        "released_values": released_values,
        "absolute_errors": errors,
        "mean_absolute_error": sum(errors) / releases,
        "raw_storage_bytes": sample_size * track["record_bytes"],
        "released_summary_bytes": releases * 8,
        "assumption": _ASSUMPTION,
        "randomness_caveat": (
            "The seed makes this teaching simulation repeatable; deployed privacy mechanisms "
            "require appropriately generated secret randomness, not a fixed public seed."
        ),
        "accounting_caveat": (
            "Basic sequential composition applies to independent new releases over the same private data; "
            "postprocessing does not add privacy spend. Retention alone does not change formal epsilon."
        ),
        "inputs": {
            "track_id": track_id,
            "epsilon_per_release": epsilon_per_release,
            "releases": releases,
            "seed": seed,
        },
    }


def lifecycle_footprint(
    track_id: str,
    *,
    daily_inferences: int | None = None,
    demand_scale: float = 1.0,
    years: float | None = None,
    retraining_runs: int = 2,
    explanation_share_pct: float | None = None,
) -> dict[str, Any]:
    """Sum training, retraining, inference, and explanation energy."""
    track = _track(track_id)
    demand_scale = _finite(demand_scale, "demand_scale", minimum=0.0)
    if demand_scale == 0:
        raise ValueError("demand_scale must be greater than 0")
    requested_daily_inferences = daily_inferences
    if daily_inferences is None:
        daily_inferences = round(track["daily_inferences"] * demand_scale)
    elif demand_scale != 1.0:
        raise ValueError("set daily_inferences or demand_scale, not both")
    daily_inferences = _positive_int(daily_inferences, "daily_inferences")
    if years is None:
        years = track["years"]
    years = _finite(years, "years", minimum=0.0)
    if years == 0:
        raise ValueError("years must be greater than 0")
    if isinstance(retraining_runs, bool) or not isinstance(retraining_runs, int) or retraining_runs < 0:
        raise ValueError("retraining_runs must be a non-negative integer")
    if explanation_share_pct is None:
        explanation_share_pct = track["explanation_share_pct"]
    explanation_share_pct = _finite(
        explanation_share_pct, "explanation_share_pct", minimum=0.0, maximum=100.0
    )
    query_count = daily_inferences * years * 365.0
    explanation_count = query_count * explanation_share_pct / 100.0
    energy_terms = {
        "initial_training": track["training_energy"].to("kilowatt_hour"),
        "retraining": (track["retraining_energy"] * retraining_runs).to("kilowatt_hour"),
        "inference": (track["inference_energy"] * query_count).to("kilowatt_hour"),
        "explanation": (track["explanation_energy"] * explanation_count).to("kilowatt_hour"),
    }
    total_energy = sum(energy_terms.values(), Q_(0, "kilowatt_hour"))
    carbon = (total_energy * track["carbon_intensity"]).to("kilogram")
    dominant_term = max(energy_terms, key=lambda name: energy_terms[name].magnitude)
    return {
        "track_id": track_id,
        "scenario": track["scenario"],
        "boundary": _LIFECYCLE_BOUNDARY,
        "terms_kwh": {name: value.magnitude for name, value in energy_terms.items()},
        "total_energy_kwh": total_energy.magnitude,
        "carbon_intensity_g_per_kwh": track["carbon_intensity"].to("gram / kilowatt_hour").magnitude,
        "operational_carbon_kg": carbon.magnitude,
        "query_count": query_count,
        "explanation_count": explanation_count,
        "dominant_term": dominant_term,
        "assumption": _ASSUMPTION,
        "inputs": {
            "track_id": track_id,
            "daily_inferences": requested_daily_inferences,
            "demand_scale": demand_scale,
            "years": years,
            "retraining_runs": retraining_runs,
            "explanation_share_pct": explanation_share_pct,
        },
    }



# Fictional execution alternatives, not measured hardware efficiency profiles.
# A lower per-decision energy assumption also incurs explicit preparation work.
_LIFECYCLE_EXECUTIONS = {
    "reference": {"inference_factor": 1.0, "preparation_factor": 0.0, "replay": "matched"},
    "replayed_candidate": {"inference_factor": 0.5, "preparation_factor": 0.5, "replay": "matched"},
    "unreplayed_candidate": {"inference_factor": 0.5, "preparation_factor": 0.5, "replay": "missing"},
    "changed_decision_candidate": {"inference_factor": 0.5, "preparation_factor": 0.5, "replay": "changed"},
}


def lifecycle_policy(
    track_id: str,
    *,
    execution: str = "reference",
    demand_scale: float = 1.0,
    retraining_runs: int = 2,
    explanation_share_pct: float | None = None,
) -> dict[str, Any]:
    """Account for a candidate, then check eligibility for a savings comparison.

    The reference service obligation is fixed: the track's daily request count,
    operating period and explanation coverage, plus two retraining runs. More
    requests are a workload expansion, not a matched-service efficiency result.
    The supplied finite replay checks decision equality on the existing fixed
    score fixture; it does not establish real deployment quality or safety.
    Candidate energy factors are explicit fictional accounting assumptions,
    independent of replay outcomes. They are not empirical performance curves.
    """
    track = _track(track_id)
    if execution not in _LIFECYCLE_EXECUTIONS:
        raise ValueError("execution must be one of " + ", ".join(_LIFECYCLE_EXECUTIONS))
    policy = _LIFECYCLE_EXECUTIONS[execution]
    result = lifecycle_footprint(
        track_id, demand_scale=demand_scale, retraining_runs=retraining_runs,
        explanation_share_pct=explanation_share_pct,
    )
    reference = lifecycle_footprint(track_id)
    result["terms_kwh"]["inference"] *= policy["inference_factor"]
    result["terms_kwh"]["preparation"] = (
        track["training_energy"].to("kilowatt_hour").magnitude * policy["preparation_factor"]
    )
    result["total_energy_kwh"] = sum(result["terms_kwh"].values())
    result["operational_carbon_kg"] = (
        Q_(result["total_energy_kwh"], "kilowatt_hour") * track["carbon_intensity"]
    ).to("kilogram").magnitude
    result["dominant_term"] = max(result["terms_kwh"], key=result["terms_kwh"].get)
    replay_rows = []
    if policy["replay"] != "missing":
        for group, rows in _SCORES[track_id].items():
            for index, (score, label) in enumerate(rows):
                decision = score >= 0.5
                replay_rows.append({
                    "request": f"{group}-{index}", "label": label,
                    "reference_decision": decision, "candidate_decision": decision,
                })
        if policy["replay"] == "changed":
            replay_rows[0]["candidate_decision"] = not replay_rows[0]["reference_decision"]
    mismatch_count = (
        sum(row["reference_decision"] != row["candidate_decision"] for row in replay_rows)
        if replay_rows else None
    )
    checks = {
        "same_request_volume": result["query_count"] == reference["query_count"],
        "same_retraining_obligation": retraining_runs == reference["inputs"]["retraining_runs"],
        "same_explanation_obligation": result["explanation_count"] == reference["explanation_count"],
    }
    failures = [name for name, passes in checks.items() if not passes]
    if mismatch_count is not None and mismatch_count > 0:
        failures.append("finite_replay_decisions_changed")
    if failures:
        gate = "ineligible"
    elif mismatch_count is None:
        gate = "insufficient_quality_evidence"
    else:
        gate = "comparable_on_supplied_evidence"
    delta = result["total_energy_kwh"] - reference["total_energy_kwh"]
    result.update({
        "execution": execution,
        "service_checks": checks,
        "comparison_gate": gate,
        "comparison_failures": failures,
        "replay_count": len(replay_rows),
        "replay_mismatch_count": mismatch_count,
        "replay_rows": replay_rows,
        "inference_energy_factor": policy["inference_factor"],
        "preparation_energy_factor": policy["preparation_factor"],
        "energy_delta_kwh": delta,
        "supported_energy_saving": gate == "comparable_on_supplied_evidence" and delta < 0,
        "limitation": (
            "Energy factors and preparation work are fictional supplied accounting assumptions. "
            "Decision equality covers only the finite replay; it does not establish deployment quality, "
            "explanation fidelity, retraining efficacy, latency, or safety. Currency, embodied emissions, "
            "facility overhead, networking, storage, and end-user equipment are not modeled. "
            "Operational carbon uses the same fixed illustrative grid for both executions."
        ),
        "boundary": result["boundary"] + " Candidate preparation work is included separately.",
        "inputs": {
            "track_id": track_id, "execution": execution, "demand_scale": demand_scale,
            "retraining_runs": retraining_runs,
            "explanation_share_pct": result["inputs"]["explanation_share_pct"],
        },
    })
    return result


def investigate_incident(
    track_id: str,
    *,
    retained_fields: Collection[str] = _LINEAGE_FIELDS,
    containment_delay_hours: float | None = None,
    threshold: float = 0.5,
    affected_share_pct: float | None = None,
) -> dict[str, Any]:
    """Evaluate reconstructability and exposure for one fixed incident.

    Evidence retention controls which concrete audit questions can be answered.
    Delay controls how many further decisions occur before containment.  Neither
    input changes the already-observed fixed-fixture fairness outcomes.
    """
    track = _track(track_id)
    retained = set(retained_fields)
    unknown = retained.difference(_LINEAGE_FIELDS)
    if unknown:
        raise ValueError(f"unknown retained_fields: {', '.join(sorted(unknown))}")
    if containment_delay_hours is None:
        containment_delay_hours = track["default_containment_delay_hours"]
    containment_delay_hours = _finite(
        containment_delay_hours, "containment_delay_hours", minimum=0.0
    )
    questions = {
        "reproduce_decision": {"model_version", "decision_threshold", "feature_hash"},
        "trace_training_data": {"model_version", "training_data_version"},
        "identify_release": {"model_version", "deployment_id"},
    }
    answerable = {name: fields.issubset(retained) for name, fields in questions.items()}
    complete = all(answerable.values())
    incident_decisions = track["incident_decisions"]
    fairness = threshold_consequences(
        track_id, threshold=threshold, affected_share_pct=affected_share_pct
    )
    return {
        "track_id": track_id,
        "scenario": track["scenario"],
        "retained_fields": tuple(field for field in _LINEAGE_FIELDS if field in retained),
        "missing_fields": tuple(field for field in _LINEAGE_FIELDS if field not in retained),
        "answerable_questions": answerable,
        "answerable_question_count": sum(answerable.values()),
        "fully_reconstructable_decisions": incident_decisions if complete else 0,
        "incident_decisions": incident_decisions,
        "evidence_storage_bytes": incident_decisions * sum(_LINEAGE_BYTES[field] for field in retained),
        "containment_delay_hours": containment_delay_hours,
        "decisions_before_containment": containment_delay_hours * track["decision_rate_per_hour"],
        "fairness_snapshot": {
            "groups": fairness["groups"],
            "aggregate": fairness["aggregate"],
            "tpr_gap_pp": fairness["tpr_gap_pp"],
            "fpr_gap_pp": fairness["fpr_gap_pp"],
        },
        "assumption": _ASSUMPTION,
        "inputs": {
            "track_id": track_id,
            "retained_fields": tuple(field for field in _LINEAGE_FIELDS if field in retained),
            "containment_delay_hours": containment_delay_hours,
            "threshold": threshold,
            "affected_share_pct": fairness["affected_share_pct"],
        },
    }


__all__ = [
    "FEEDBACK_MAX_ROUNDS",
    "proxy_alignment",
    "feedback_replay",
    "fairness_criteria",
    "selective_feedback",
    "LINEAGE_POLICIES",
    "TRACKS",
    "bounded_mean_release",
    "investigate_incident",
    "lifecycle_footprint",
    "lifecycle_policy",
    "population_mix",
    "threshold_consequences",
]
