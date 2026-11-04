"""Authored controlled task and deployment studies."""

from ....core.units import Q_
from ... import scenario_types as _st
from ...deployment_journeys import load_deployment_journeys

_DEPLOYMENT_JOURNEYS = load_deployment_journeys()

def _placement_study():
    import json
    from pathlib import Path
    from ....core.provenance import Provenance
    data = json.loads((Path(__file__).parents[2] / "data" / "placement_study.json").read_text())
    for track_id, fixture in data["tracks"].items():
        journey = _DEPLOYMENT_JOURNEYS.get(track_id)
        if fixture["scenario"] != journey.scenario_id:
            raise ValueError("Placement fixture must reference its canonical deployment journey")
        fixture["scenario"] = journey.title
    return _st.PlacementStudyScenario(
        name="Controlled deployment placement study",
        description="Screen locality, resources, mission operation, and supplied task evidence before cost.",
        version=data["version"], fixtures=data,
        provenance=Provenance(kind="illustrative", ref=data["provenance"]["note"],
                              verified=data["provenance"]["verified"]),
    )

DeploymentJourneys = _DEPLOYMENT_JOURNEYS

PlacementStudy = _placement_study()

ControlledTaskFactors = _st.TaskFactorStudyScenario(
    name="Controlled task-factor study",
    description="Illustrative single-host task factors with independent supplied quality observations; no training or selection costs are simulated.",
    version="constraint-study.v2",
    quality_floor_pct=82.0,  # fixed acceptance requirement, independent of outcomes
    shift_options=(0, 5, 10, 15, 20, 25, 30),
    capability_options=(1.0, 2.0, 4.0),
    provenance={"kind": "illustrative", "ref": "Controlled task-factor teaching fixture", "verified": "2026-10-20", "notes": "Authored comparison inputs and held-out observations, not measurements or an accuracy law."},
    cards={
        "baseline": _st.TaskFactorCard(label="Unchanged host study", samples=1000, ops_per_sample=Q_("2 megaflop"), seconds_per_op=Q_("10 nanosecond / flop"), quality_pct=84.0, factor="none"),
        "data": _st.TaskFactorCard(label="Fewer training examples", samples=500, ops_per_sample=Q_("2 megaflop"), seconds_per_op=Q_("10 nanosecond / flop"), quality_pct=80.0, factor="data"),
        "algorithm": _st.TaskFactorCard(label="Less work per example", samples=1000, ops_per_sample=Q_("1 megaflop"), seconds_per_op=Q_("10 nanosecond / flop"), quality_pct=84.0, factor="algorithm"),
        "execution": _st.TaskFactorCard(label="Faster realized execution", samples=1000, ops_per_sample=Q_("2 megaflop"), seconds_per_op=Q_("5 nanosecond / flop"), quality_pct=84.0, factor="machine"),
        "unverified": _st.TaskFactorCard(label="Fewer examples, missing evaluation", samples=500, ops_per_sample=Q_("2 megaflop"), seconds_per_op=Q_("10 nanosecond / flop"), quality_pct=None, factor="data"),
    },
)

ShallowPowerLawExample = _st.PowerLawScenario(
name="A shallow power law", description="Illustrative exponent of 0.1.", exponent=0.1)
