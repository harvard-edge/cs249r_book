"""Named optimization scenarios for lifecycle.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....core.units import DAYS_PER_YEAR as _DAYS_PER_YEAR
from ....hardware.registry import Hardware as _Hardware
from ....reference_stats.registry import ReferenceStats as _ReferenceStats
from ... import scenario_types as _st


RetinopathyScreenerRequirements = _st.ScreeningRequirementsScenario(
    name="A diabetic retinopathy screener's release floors and clinic-device budget",
    description="Sensitivity and specificity floors that hold in every subgroup, a memory reservation on "
    "an Orin Nano-class clinic device, and two measured sensitivity slices at the release gate.",
    sensitivity_floor=0.90,  # scenario assumption: few sick patients missed
    specificity_floor=0.80,  # scenario assumption: specialists not flooded with false referrals
    memory_budget=Q_(500, "MB"),  # scenario assumption: the clinic IT plan's reservation for the model
    device=_Hardware.Edge.JetsonOrinNano,
    slice_sensitivity={
        "all clinic photos": 0.94,  # scenario assumption: overall sensitivity on clinic photos
        "cataract patients": 0.76,  # scenario assumption: the subgroup slice that fails the floor
    },
    field_operating_point=(0.88, 0.80),  # scenario assumption: illustrative (specificity, sensitivity)
    # of the lab threshold on clinic photos
)


RetinopathyScreenerLateMemoryLimit = _st.LateConstraintScenario(
    name="A screener whose 500 MB memory limit surfaces on day 151",
    description="A team runs the stages as a line; the model reaches its accuracy target, then needs 4 GB "
    "on a device that reserves 500 MB for it. Four dependent artifacts are redone in sequence.",
    model_memory=Q_(4, "GB"),  # scenario assumption: memory the trained model needs
    memory_budget=RetinopathyScreenerRequirements.memory_budget,
    device_memory_low=Q_(4, "GB"),  # scenario assumption: shared memory of the clinic device, low
    device_memory_high=Q_(8, "GB"),  # scenario assumption: shared memory of the clinic device, high
    request_day=1,
    validation_sensitivity={90: 0.92, 120: 0.94},  # scenario assumption: day -> validation sensitivity
    handoff_day=150,  # scenario assumption: model handed to deployment engineers
    discovery_day=151,  # scenario assumption: memory need reported
    check_day=152,  # scenario assumption: device budget finally checked
    discard_day=153,  # scenario assumption: work discarded
    repair_tasks={  # scenario assumption: a supplied schedule, not a multiplier
        "Revise preparation": Q_(1, "day"),
        "Train new weights": Q_(2, "day"),
        "Repeat validation": Q_(2, "day"),
        "Package the release": Q_(1, "day"),
    },
)


SlowVersusFastIteration = _st.IterationBudgetScenario(
    name="A slow ensemble against a fast edge model over 26 weeks of development",
    description="One-week runs from 95 percent at 0.15 points each, against one-hour runs from 90 percent at "
    "0.1 points each of which a team can learn from about 100; both capped at 99 percent.",
    window=Q_(26, "week"),  # scenario assumption: six months of development
    slow_cycle=Q_(1, "week"),  # scenario assumption: one large-ensemble run
    slow_start=95.0,  # scenario assumption: percent
    slow_gain=0.15,  # scenario assumption: points per run
    fast_cycle=Q_(1, "hour"),  # scenario assumption: one small-model run
    fast_start=90.0,  # scenario assumption: percent
    fast_gain=0.1,  # scenario assumption: points per run
    fast_effective_runs=100,  # scenario assumption: experiments a team can design and learn from
    ceiling=99.0,  # scenario assumption: accuracy ceiling, percent
)


RuralClinicRetinalUpload = _st.DailyUploadScenario(
    name="A rural screening clinic uploading a day of retinal photos over a 2 Mb/s uplink",
    description="100 patients, four photos each, against an eight-hour shift, and the 10 KB summaries "
    "per patient that local inference sends instead.",
    cases_per_day=100,  # scenario assumption: a busy rural screening clinic
    files_per_case=4,  # screening protocol: two fields per eye
    file_size=_ReferenceStats.ClinicalImaging.RetinalPhotoSize,
    uplink=Q_(2.0, "megabit/second"),  # scenario assumption: clinic uplink
    window=Q_(8.0, "hour"),  # scenario assumption: one clinic shift
    summary_size=Q_(10.0, "KB"),  # scenario assumption: one result summary per patient
)


RetinopathyScreeningCloudVersusEdge = _st.CloudEdgeCostScenario(
    name="500 screening clinics served from the cloud or from a device in each clinic",
    description="50 patients a day per clinic and one billed image each, every day of the year; cloud per-image "
    "fees plus connectivity against a $500 device per clinic plus maintenance.",
    sites=500,  # scenario assumption: clinics in the network
    cases_per_site_day=50,  # scenario assumption: patients per clinic per day
    images_per_case=1,  # scenario assumption: one graded image billed per patient
    days_per_year=_DAYS_PER_YEAR,  # scenario assumption: clinics operate every day
    cloud_cost_per_image=Q_(0.01, "dollar"),  # scenario assumption
    cloud_fixed_annual=Q_(45000, "dollar/year"),  # scenario assumption: connectivity and network operations
    edge_device_cost=Q_(500, "dollar"),  # scenario assumption: one device per clinic
    edge_maintenance_annual=Q_(25000, "dollar/year"),  # scenario assumption
    edge_cost_per_image=Q_(0.001, "dollar"),  # scenario assumption
    image_size=_ReferenceStats.ClinicalImaging.RetinalPhotoSize,
    uplink=RuralClinicRetinalUpload.uplink,
    cloud_latency=Q_(200, "ms"),  # scenario assumption: remote processing after upload
    edge_latency=Q_(50, "ms"),  # scenario assumption: on-device inference
)
