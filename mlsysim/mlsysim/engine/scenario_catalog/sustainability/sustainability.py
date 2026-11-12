"""Named sustainability scenarios for sustainability.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....core.units import DAYS_PER_YEAR as _DAYS_PER_YEAR
from ....core.units import HOURS_PER_YEAR as _HOURS_PER_YEAR
from ....hardware.registry import Hardware as _Hardware
from ....systems.registry import Clusters as _Clusters
from ....reference_stats.registry import ReferenceStats as _ReferenceStats
from ....core.units import BYTES_FP16 as _BYTES_FP16
from ....core.units import BYTES_FP32 as _BYTES_FP32
from ....literature.registry import Literature as _Literature
from ....infrastructure.registry import Infrastructure as _Infrastructure
from ... import scenario_types as _st
from ....systems.registry import Systems as _Systems


# --- Sustainability (added 2026-10-20) -----------------------------------
# Rated-power (TDP) planning accounts: accelerator power times the DGX H100
# rack's IT-to-accelerator ratio, times PUE once at the facility boundary.
Mega100KFacilityPower = _st.FacilityPowerScenario(
    name="100,000-H100 cluster, facility power",
    description="A 100,000-GPU H100 cluster through the DGX H100 rack profile at the cluster's PUE.",
    devices=_Clusters.Mega_100K.total_accelerators,
    device_power=_Hardware.Cloud.H100.tdp,
    rack=_Systems.Racks.DGX_H100_4Node,
    pue=_Clusters.Mega_100K.effective_pue,
)


Reference25KFacilityPower = Mega100KFacilityPower.variant(
    name="25,000-H100 reference fleet, facility power",
    description="The 25,000-GPU H100 reference fleet through the same rack profile and PUE.",
    devices=_Clusters.Reference_25K_H100.total_accelerators,
    pue=_Clusters.Reference_25K_H100.effective_pue,
)


Training10KCoolingRollback = _st.RepeatedWorkEnergyScenario(
    name="10,000-GPU training fleet rolled back after a cooling failure",
    description="Accelerator energy repeated when the fleet loses 30 to 60 minutes of work to the last checkpoint.",
    fleet=_Clusters.Training_10K,
    rollbacks={"low": Q_(0.5, "hour"), "high": Q_(1.0, "hour")},  # scenario assumption: work lost
)


PueRetrofit2MW = _st.PueRetrofitScenario(
    name="2 MW cluster moved from legacy air cooling to state-of-the-art PUE",
    description="One year at a fixed 2 MW IT load, PUE 1.58 to 1.10, at the on-premises electricity price.",
    it_power=Q_(2.0, "MW"),  # scenario assumption: about 3,000 H100s of IT load
    pue_before=_Infrastructure.FacilityCooling.Legacy.pue,
    pue_after=_Infrastructure.FacilityCooling.StateOfArt.pue,
    period=Q_(_HOURS_PER_YEAR, "hour"),
    electricity_price=_Infrastructure.Pricing.OnPremises.ElectricityPerKwh.rate,
)


QuebecVersusPolandPlacement = _st.GridPlacementScenario(
    name="Reference training run on Quebec's grid or Poland's",
    description="The carbon-placement reference run's facility energy on the two grids, and Poland "
    "after a 15 percent algorithmic energy cut.",
    energy=_ReferenceStats.CarbonPlacementProfile.TrainingRunEnergy,
    grids={"Quebec": _Infrastructure.Grids.Quebec, "Poland": _Infrastructure.Grids.Poland},
    energy_cut=0.15,  # scenario assumption: algorithmic energy reduction
)


RegionalGridPlacement = QuebecVersusPolandPlacement.variant(
    name="Reference training run on nine regional grids",
    description="The carbon-placement reference run's facility energy on each rounded annual-average grid.",
    grids={
        "Norway": _Infrastructure.Grids.Norway,
        "Quebec": _Infrastructure.Grids.Quebec,
        "France": _Infrastructure.Grids.France,
        "EU average": _Infrastructure.Grids.EU_Avg,
        "Germany": _Infrastructure.Grids.Germany,
        "US average": _Infrastructure.Grids.US_Avg,
        "China": _Infrastructure.Grids.China_Avg,
        "India": _Infrastructure.Grids.India_Avg,
        "Poland": _Infrastructure.Grids.Poland,
    },
    energy_cut=0.0,
)


H100ReplacementPayback = _st.EmbodiedPaybackScenario(
    name="Replacing a continuously running H100 with one that needs half the energy",
    description="Days for the replacement's avoided emissions to repay its manufacturing carbon.",
    device=_Hardware.Cloud.H100,
    saved_fraction=0.5,  # scenario assumption: same work at half the energy
    grids={"US average": _Infrastructure.Grids.US_Avg, "Quebec": _Infrastructure.Grids.Quebec},
)


AcceleratorLifeExtension = _st.ServiceLifeExtensionScenario(
    name="Accelerator service life extended from three to five years",
    description="Cut in the embodied carbon charged per year of service.",
    short_life=Q_(_Infrastructure.Pricing.Capital.AcceleratorServiceLifeYears.rate, "year"),
    long_life=Q_(5, "year"),  # scenario assumption: extended five-year life
)


EdgeFirstImageCascade = _st.CascadeEnergyScenario(
    name="Edge-first image cascade escalating 10 percent to the cloud",
    description="A small edge classifier on every image; one in ten goes over cellular to a large cloud model.",
    edge_energy=Q_(0.5, "mJ"),  # scenario assumption: small edge classifier per image
    transmit_energy=Q_(10.0, "mJ"),  # scenario assumption: cellular uplink of one image
    cloud_energy=Q_(50.0, "mJ"),  # scenario assumption: large cloud model per inference
    escalation=0.10,  # scenario assumption: share of inputs escalated
)


WakeWordHierarchyPower = _st.DutyCycledPowerScenario(
    name="Three-stage wake-word hierarchy",
    description="An analog voice-activity detector always on, a small network during speech, and a full "
    "model for 50 ms per wake.",
    always_on_power=Q_(0.01, "mW"),  # scenario assumption: analog voice-activity detector
    stage_power=Q_(0.1, "mW"),  # scenario assumption: small wake-word network
    stage_activity=0.05,  # scenario assumption: share of time with speech
    full_power=Q_(10, "mW"),  # scenario assumption: full model
    event_rate=Q_(0.001, "1/s"),  # scenario assumption: wakes per second
    full_duration=Q_(0.05, "s"),  # scenario assumption: full-model run per wake
)


BatchOneDecodeWeightEnergy = _st.DataMovementEnergyScenario(
    name="One FP16 weight used once in a batch-1 decode step",
    description="The modeled GPU FLOP energy against DRAM movement energy per byte.",
    energy_per_flop=_Hardware.Tech.EffectiveCompute.Gpu.energy_per_flop,
    energy_per_byte=_Hardware.Tech.Memory.DRAM.energy_per_byte,
    flops_per_param=2,  # one multiply-accumulate per weight per token
    bytes_per_param=_BYTES_FP16,
    operand_bytes=_BYTES_FP32,
)


TokenPriceRebound = _st.DemandReboundScenario(
    name="Token price cut thirtyfold while query volume grows a hundredfold",
    description="Energy per query assumed proportional to price.",
    unit_price_before=0.06,  # scenario assumption: $ per 1,000 tokens before optimization
    unit_price_after=0.002,  # scenario assumption: $ per 1,000 tokens after optimization
    volume_factor=100.0,  # scenario assumption: growth in query volume
)


CandidateSweep50 = _st.DevelopmentSweepScenario(
    name="Fifty-candidate sweep at a tenth of full scale",
    description="Final-run equivalents consumed before the reported run.",
    candidates=50,  # scenario assumption
    relative_cost=0.1,  # scenario assumption: each candidate's cost relative to the final run
)


ReferenceItLoadAnnualWater = _st.WaterUseScenario(
    name="Reference IT load cooled evaporatively for a year",
    description="The reference IT load for one year at the evaporative WUE.",
    it_power=_ReferenceStats.FacilityScaleAnchors.ReferenceItLoad,
    period=Q_(_HOURS_PER_YEAR, "hour"),
    wue={"evaporative": Q_(_Infrastructure.FacilityCooling.Typical.wue, "liter/kWh")},
)


CarbonPricedRegionPlacement = _st.CarbonPricedPlacementScenario(
    name="Reference training run placed among three regions with a carbon price",
    description="The carbon-placement reference run's IT energy on an 8,192-GPU fleet, ranked by total "
    "cost in the US average, Quebec, and Iowa with and without a $100/t carbon price.",
    fleet=_Clusters.Frontier_8K,
    it_energy=_ReferenceStats.CarbonPlacementProfile.TrainingRunEnergy,
    regions=["US_Avg", "Quebec", "Iowa"],
    carbon_price_per_ton=100.0,  # scenario policy assumption: $ per tonne CO2
    mfu=_Literature.Training.MfuHigh,
    reference_grid=_Infrastructure.Grids.US_Avg,
)


# Illustrative hourly carbon intensity from a 16:00 release: evening peak,
# overnight trough, midday solar dip. Made up for teaching; not measured data.
_EVENING_RAMP_G_PER_KWH = (520, 560, 600, 620, 580, 520)  # 16:00-21:00
_OVERNIGHT_G_PER_KWH = (420, 380, 350, 330, 320, 330, 360, 400)  # 22:00-05:00
_SOLAR_DIP_G_PER_KWH = (440, 400, 320, 240, 190, 170, 180, 220, 300, 400)  # 06:00-15:00
_EVENING_RELEASE_SIGNAL = [
    Q_(g, "g/kWh") for g in _EVENING_RAMP_G_PER_KWH + _OVERNIGHT_G_PER_KWH + _SOLAR_DIP_G_PER_KWH
]


EveningReleaseTimeShift = _st.TemporalShiftScenario(
    name="Flexible 6-hour run released into an evening carbon peak",
    description="A 1 MW, 6-hour run released at 16:00 that may finish up to 18 hours late, "
    "with 100 kW of standby power while it waits.",
    intensities=_EVENING_RELEASE_SIGNAL,
    power=Q_(1.0, "MW"),  # scenario assumption: facility power, PUE included
    duration=Q_(6, "hour"),
    slack=Q_(18, "hour"),  # scenario assumption: must finish within 24 h of release
    standby_power=Q_(100, "kW"),  # scenario assumption: reserved capacity kept powered
)


EveningReleaseNoSlack = EveningReleaseTimeShift.variant(
    name="The same evening-release run with no deadline slack",
    description="Zero slack forces an immediate start, so no time shift is possible.",
    slack=Q_(0, "hour"),
)


EveningReleaseShortSlack = EveningReleaseTimeShift.variant(
    name="The same evening-release run with 3 hours of slack",
    description="Three hours of slack only reaches the end of the evening ramp.",
    slack=Q_(3, "hour"),
)


EveningReleaseCapacityLimited = EveningReleaseTimeShift.variant(
    name="The same evening-release run when capacity is booked through the morning",
    description="Capacity is booked from 01:00 to 11:59, so no 6-hour window reaches the clean morning hours.",
    # scenario assumption: steps 9-19 (01:00-11:59) are booked
    capacity_available=[not 9 <= hour <= 19 for hour in range(24)],
)


EveningReleaseCostlyStandby = EveningReleaseTimeShift.variant(
    name="The same evening-release run with 400 kW of standby power",
    description="Waiting costs more carbon than any later window saves, so starting now is best.",
    standby_power=Q_(400, "kW"),  # scenario assumption: large reserved block kept powered
)

KaplanTenPercentLowerLoss = _st.PowerLawScenario(
    name="Ten percent lower loss under Kaplan's compute law",
    description="Compute multiple a 10 percent loss reduction costs at Kaplan et al.'s compute exponent.",
    exponent=next(e["alpha"] for e in _Literature.KaplanScalingLaws.entries if e["axis"] == "compute"),
    loss_reduction=0.10,  # pedagogical target within the studied range
)


Lab64A100TrainingFootprint = _st.RunFootprintScenario(
    name="7B model trained for two weeks on a 64-A100 lab cluster",
    description="The mid-scale training carbon profile through the DGX rack ratio at the US-average "
    "grid's PUE, on the US-average and Quebec grids, embodied carbon over the extended life.",
    device_hours=Q_(
        _Clusters.Lab_64_A100.total_accelerators
        * _ReferenceStats.MidScaleTrainingCarbonProfile.TrainingDays.to("hour").magnitude,
        "hour",
    ),
    devices=_Clusters.Lab_64_A100.total_accelerators,
    device=_Hardware.Cloud.A100,
    rack=_Systems.Racks.DGX_H100_4Node,
    pue=_Infrastructure.Grids.US_Avg.pue,
    grids={"US average": _Infrastructure.Grids.US_Avg, "Quebec": _Infrastructure.Grids.Quebec},
    service_life=Q_(
        _Infrastructure.Pricing.Capital.AcceleratorExtendedServiceLifeYears.rate * _DAYS_PER_YEAR, "day"
    ).to("hour"),
)


TenMillionQueriesAgainstLab64Training = _st.ServingCrossoverScenario(
    name="Ten million queries a day against the 64-A100 training run",
    description="Days of serving at 1 Wh per query that equal the lab run's facility energy.",
    one_time_energy=Lab64A100TrainingFootprint.result.facility_energy,
    requests_per_day=10_000_000,  # scenario assumption: high-volume deployed model
    energy_per_request=Q_(0.001, "kWh"),  # scenario assumption: pedagogical per-query energy
)
