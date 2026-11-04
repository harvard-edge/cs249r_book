"""Named training scenarios for federated.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....hardware.registry import Hardware as _Hardware
from ....reference_stats.registry import ReferenceStats as _ReferenceStats
from ....core.units import BYTES_FP16 as _BYTES_FP16, param as _param
from ....core.units import BYTES_FP32 as _BYTES_FP32
from ... import scenario_types as _st


# --- On-device learning and federated rounds (added 2026-10-20) ---------
# A phone's full update, parameter-efficient updates, personalization
# storage, the uplink, the battery, and the mechanics of one federated round.
PhoneFullUpdateWorkingSet = _st.UpdateWorkingSetScenario(
    name="Full fine-tune of a 10M-parameter FP32 model with Adam on a flagship phone",
    description="Weights, gradients, two Adam moments, and saved activations at 5x the weight bytes; "
    "DRAM traffic of one step with one crossing per tensor use, at the A17 Pro's bandwidth.",
    parameters=10 * 10**6,  # scenario assumption: compact on-device model
    bytes_per_param=_BYTES_FP32,
    optimizer_moments=2,  # Adam keeps first and second moments
    activation_factor=5,  # scenario assumption: saved activations = 5x weight bytes
    activation_factor_high=8,  # scenario assumption: upper end of the quoted range
    bandwidth=_Hardware.Mobile.iPhone15Pro.memory.bandwidth,
)


PhoneBackgroundFineTuneBattery = _st.BatteryBudgetScenario(
    name="A 30-minute unthrottled fine-tune at 4.5 W against a flagship phone battery",
    description="A background voice-assistant adaptation job drawing above the sustained envelope.",
    battery_energy=_ReferenceStats.PhoneBattery.EnergyWh,
    power_draw=Q_(4.5, "watt"),  # scenario assumption: unthrottled SoC draw during training
    required_runtime=Q_(30, "minute"),  # scenario assumption: one adaptation job
)


TransformerLayerRank4Update = _st.LowRankUpdateScenario(
    name="Rank-4 low-rank update to a 768 by 768 transformer weight",
    description="Trainable parameters of the update against the full layer.",
    in_dim=768,  # scenario assumption: square weight of a generic transformer layer
    out_dim=768,
    rank=4,  # scenario assumption
)


CompactEncoderResidualAdapters = _st.ResidualAdapterScenario(
    name="Residual adapters of width 768 and bottleneck 16 after each of 12 blocks, FP16",
    description="A compact transformer encoder personalized with one bottleneck adapter per block.",
    width=768,  # scenario assumption: hidden width
    bottleneck=16,  # scenario assumption: adapter bottleneck
    sites=12,  # scenario assumption: one adapter per block
    bytes_per_param=_BYTES_FP16,
)


CameraModelTenContextPersonalization = _st.PersonalizationStorageScenario(
    name="A 10M-parameter camera model personalized for 10 contexts: copies or adapters",
    description="FP32 storage of ten fine-tuned copies against one backbone plus ten 50,000-parameter adapters.",
    model_params=10 * 10**6 * _param,  # scenario assumption
    adapter_params=50_000 * _param,  # scenario assumption: one adapter per context
    bytes_per_param=_BYTES_FP32 / _param,
    contexts=10,  # scenario assumption: home, office, car, ...
)


PhoneImagesVersusQuantizedUpdates = _st.FederatedUplinkScenario(
    name="A week of phone photos against 4-bit updates of a 5M-parameter model",
    description="1,000 compressed images of 200 KB a week against one 2.5 MB update per round, "
    "at a daily and an hourly round cadence.",
    items_per_week=1_000,  # scenario assumption
    item_size=Q_(200, "kB"),  # scenario assumption: compressed JPEG
    update_params=5 * 10**6,  # scenario assumption
    update_bits_per_param=4,  # scenario assumption: 4-bit quantized update
    rounds_per_week={"daily": 7, "hourly": 24 * 7},
)


FederatedAdapterUploadMobileUplink = _st.TransferScenario(
    name="The reference federated adapter update over a 1 to 10 Mb/s mobile uplink",
    description="Communication term of one federated round for one phone.",
    payload=_ReferenceStats.EdgeAdaptationTierProfile.FederatedLoraUpdate,
    links={
        "low": Q_(1, "megabit/second"),  # scenario assumption: constrained uplink
        "high": Q_(10, "megabit/second"),
    },  # scenario assumption: upper end
)
