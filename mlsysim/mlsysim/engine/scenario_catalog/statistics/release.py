"""Named statistics scenarios for release.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....embodied.registry import Embodied as _Embodied
from ... import scenario_types as _st
from ..robotics.perception import WarehouseNavCameraAge


# --- Release of the warehouse machine: lighting, argument confidence (added 2026-10-20) ---
WarehouseNavCameraAgeAtAisleSpeed = WarehouseNavCameraAge.variant(
    name="The warehouse machine's navigation-camera observation age at the candidate aisle speed",
    description="Exposure midpoint to dispatch, priced as travel at the aisle speed; one frame.",
    speed=_Embodied.Scenario.WarehouseAisle.v_aisle,
    batch_frames=1,
)


WarehouseNavCameraAgeDimAisle = WarehouseNavCameraAgeAtAisleSpeed.variant(
    name="The same observation age in a dim aisle that doubles the camera's exposure",
    description="Auto-exposure lengthens the exposure; the capture epoch is its midpoint, so the age "
    "grows by half the added exposure.",
    exposure=(
        2.0  # scenario assumption: a dim aisle doubles the exposure (illustrative)
        * _Embodied.MobileManipulator.WarehouseMobileManipulator.nav_camera.exposure_time
    ).to("ms"),
)


SafetyArgumentConfidence = _st.ArgumentConfidenceScenario(
    name="A stipulated safety argument: ten required premises and two independent legs",
    description="Confidence in a conjunction of required premises, and in a claim either of two "
    "independent legs suffices for (a teaching model, not a machine value).",
    premise_confidence=0.99,  # scenario assumption: stipulated confidence in each required premise
    premises=10,  # scenario assumption: stipulated number of required premises
    leg_confidence=0.90,  # scenario assumption: stipulated confidence in each leg
    legs=2,
)
