"""Cloud, storage, labeling, and fleet economics (2024 illustrative baselines)."""

from ..core.units import USD, ureg, GB, TB, hour
from ..core.registry import Registry
from ..core.types import Metadata
from ..core import provenance_catalog as pc
from .types import PricePoint

_CLOUD = Metadata(provenance=pc.BOOK_CLOUD_PRICING_2024)
_STORAGE = Metadata(provenance=pc.BOOK_STORAGE_PRICING_2024)
_LABELING = Metadata(provenance=pc.BOOK_LABELING_PRICING_2024)
_FLEET = Metadata(provenance=pc.BOOK_FLEET_ECONOMICS_2024)
_CAPITAL = Metadata(provenance=pc.BARROSO_DATACENTER_ECONOMICS)
_ONPREM = Metadata(provenance=pc.BARROSO_DATACENTER_ECONOMICS)


class Cloud(Registry):
    """Public-cloud list-price anchors."""
    ElectricityPerKwh = PricePoint(
        name="Cloud electricity (US baseline)",
        rate=0.12 * USD / ureg.kilowatt_hour,
        metadata=_CLOUD,
    )
    EgressPerGB = PricePoint(
        name="Cloud egress",
        rate=0.09 * USD / GB,
        metadata=_CLOUD,
    )
    GpuTrainingPerHour = PricePoint(
        name="Cloud GPU training",
        rate=4.0 * USD / hour,
        metadata=_CLOUD,
    )
    GpuTrainingUtilityScenarioPerHour = PricePoint(
        name="Cloud GPU training (utility-bill scenario)",
        rate=3.0 * USD / hour,
        metadata=_CLOUD,
    )
    GpuInferencePerHour = PricePoint(
        name="Cloud GPU inference",
        rate=2.5 * USD / hour,
        metadata=_CLOUD,
    )
    TpuV4PerHour = PricePoint(
        name="Cloud TPU v4",
        rate=4.0 * USD / hour,
        metadata=_CLOUD,
    )


class Storage(Registry):
    """Object/block storage tiers."""
    S3StandardPerTbMonth = PricePoint(
        name="S3 standard",
        rate=23 * USD / TB / ureg.month,
        metadata=_STORAGE,
    )
    GlacierPerTbMonth = PricePoint(
        name="Glacier",
        rate=1 * USD / TB / ureg.month,
        metadata=_STORAGE,
    )
    NvmeLowPerTbMonth = PricePoint(
        name="NVMe (low)",
        rate=100 * USD / TB / ureg.month,
        metadata=_STORAGE,
    )
    NvmeHighPerTbMonth = PricePoint(
        name="NVMe (high)",
        rate=300 * USD / TB / ureg.month,
        metadata=_STORAGE,
    )
    GlacierRetrievalPerGB = PricePoint(
        name="Glacier retrieval",
        rate=0.02 * USD / GB,
        metadata=_STORAGE,
    )


class Labeling(Registry):
    """Data-labeling cost tiers."""
    CrowdLow = PricePoint(name="Crowd labeling (low)", rate=0.01 * USD, metadata=_LABELING)
    CrowdHigh = PricePoint(name="Crowd labeling (high)", rate=0.05 * USD, metadata=_LABELING)
    BoxLow = PricePoint(name="BBox labeling (low)", rate=0.05 * USD, metadata=_LABELING)
    BoxHigh = PricePoint(name="BBox labeling (high)", rate=0.20 * USD, metadata=_LABELING)
    MedicalLow = PricePoint(name="Medical labeling (low)", rate=50 * USD, metadata=_LABELING)
    MedicalHigh = PricePoint(name="Medical labeling (high)", rate=200 * USD, metadata=_LABELING)


class Fleet(Registry):
    """Internal fleet chargeback and spot references."""
    GpuHourRef = PricePoint(name="Fleet GPU-hour (reference)", rate=2.0 * USD / hour, metadata=_FLEET)
    SpotGpuHourRef = PricePoint(name="Fleet spot GPU-hour", rate=0.70 * USD / hour, metadata=_FLEET)
    InternalChargebackPerHour = PricePoint(
        name="Internal chargeback per GPU-hour",
        rate=2.50 * USD / hour,
        metadata=_FLEET,
    )
    CarbonPerGpuHr = PricePoint(
        name="Carbon per GPU-hour (illustrative)",
        rate=0.16 * ureg.kilogram,
        metadata=Metadata(provenance=pc.BOOK_CARBON_PER_GPU_HR),
    )


class Capital(Registry):
    """CapEx / OpEx conventions."""
    AnnualMaintenanceRatio = PricePoint(
        name="Annual maintenance (fraction of CapEx)",
        rate=0.05,
        metadata=_CAPITAL,
    )
    DgxH100NodeUsd = PricePoint(
        name="DGX H100 node (CapEx reference)",
        rate=350_000 * USD,
        metadata=_CAPITAL,
    )
    H100GpuPurchaseUsd = PricePoint(
        name="H100 GPU purchase (utility-bill scenario)",
        rate=30_000 * USD,
        metadata=_CAPITAL,
    )


class OnPremises(Registry):
    """Owned-facility utility rates for TCO build-vs-buy examples."""
    ElectricityPerKwh = PricePoint(
        name="On-premises electricity (US industrial)",
        rate=0.07 * USD / ureg.kilowatt_hour,
        metadata=_ONPREM,
    )


class Pricing(Registry):
    Cloud = Cloud
    Storage = Storage
    Labeling = Labeling
    Fleet = Fleet
    Capital = Capital
    OnPremises = OnPremises
