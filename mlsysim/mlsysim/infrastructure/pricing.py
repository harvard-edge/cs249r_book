"""Cloud, storage, labeling, and fleet economics (2024 illustrative baselines)."""

from ..core.units import USD, ureg, GB, TB, hour
from ..core.registry import Registry
from ..core.types import Metadata
from ..core import provenance_catalog as pc
from .types import PricePoint

_CLOUD = Metadata(provenance=pc.CLOUD_PRICING_2024)
_STORAGE = Metadata(provenance=pc.STORAGE_PRICING_2024)
_LABELING = Metadata(provenance=pc.LABELING_PRICING_2024)
_FLEET = Metadata(provenance=pc.FLEET_ECONOMICS_2024)
_CAPITAL = Metadata(provenance=pc.BARROSO_DATACENTER_ECONOMICS)
_ONPREM = Metadata(provenance=pc.BARROSO_DATACENTER_ECONOMICS)
_DISCOUNTS = Metadata(provenance=pc.CLOUD_COMMITMENT_DISCOUNT_RANGES)
_NETWORK = Metadata(provenance=pc.NETWORK_CABLING_PRICES_2024)
_LABOR = Metadata(provenance=pc.ML_ENGINEER_LOADED_RATE)


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
    LowCostGpuPerHour = PricePoint(
        name="Cloud GPU (low-cost reference)",
        rate=0.50 * USD / hour,
        metadata=_CLOUD,
    )
    # Model serving on one machine (added 2026-10-20): the CPU and T4 instances
    # of a cost-per-image comparison against an H100 at GpuTrainingPerHour.
    CpuInstanceC5XlargePerHour = PricePoint(
        name="c5.xlarge CPU instance",
        rate=0.17 * USD / hour,
        metadata=_CLOUD,
    )
    T4InstanceG4dnXlargePerHour = PricePoint(
        name="g4dn.xlarge instance (one NVIDIA T4)",
        rate=0.53 * USD / hour,
        metadata=_CLOUD,
    )
    A10GInferencePerHour = PricePoint(
        name="A10G inference GPU",
        rate=0.75 * USD / hour,
        metadata=_CLOUD,
    )
    TpuV4PerHour = PricePoint(
        name="Cloud TPU v4",
        rate=4.0 * USD / hour,
        metadata=_CLOUD,
    )
    # --- Added 2026-09-28: discount fractions off on-demand (dimensionless) ---
    SpotDiscountLow = PricePoint(name="Spot/preemptible discount (low)", rate=0.60, metadata=_DISCOUNTS)
    SpotDiscountHigh = PricePoint(name="Spot/preemptible discount (high)", rate=0.90, metadata=_DISCOUNTS)
    ReservedDiscountLow = PricePoint(name="Reserved/committed-use discount (low)", rate=0.40, metadata=_DISCOUNTS)
    ReservedDiscountHigh = PricePoint(name="Reserved/committed-use discount (high)", rate=0.60, metadata=_DISCOUNTS)


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
    S3StandardLowPerTbMonth = PricePoint(
        name="S3 standard (low reference)",
        rate=0.02 * USD / GB / ureg.month,
        metadata=_STORAGE,
    )
    GlacierStandardPerTbMonth = PricePoint(
        name="Glacier standard",
        rate=0.004 * USD / GB / ureg.month,
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
    ParallelFileSystemPerTbMonth = PricePoint(
        name="Parallel file system (Lustre/GPFS-class)",
        rate=30 * USD / TB / ureg.month,
        metadata=_STORAGE,
    )
    GlacierRetrievalPerGB = PricePoint(
        name="Glacier retrieval",
        rate=0.02 * USD / GB,
        metadata=_STORAGE,
    )
    HbmEquivalentPerGb = PricePoint(
        name="HBM capital cost per GB (storage-ladder comparison)",
        rate=15 * USD / GB,
        metadata=_STORAGE,
    )
    # --- Added 2026-09-28 ---
    HbmLowPerGb = PricePoint(
        name="HBM capital cost per GB (low reference)",
        rate=10 * USD / GB,
        metadata=_STORAGE,
    )
    HostDramLowPerGb = PricePoint(
        name="Host DRAM (DDR5) cost per GB (low)",
        rate=3 * USD / GB,
        metadata=_STORAGE,
    )
    HostDramHighPerGb = PricePoint(
        name="Host DRAM (DDR5) cost per GB (high)",
        rate=5 * USD / GB,
        metadata=_STORAGE,
    )
    S3GetPer1kRequests = PricePoint(
        name="S3 standard GET requests (per 1,000)",
        rate=0.0004 * USD,
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
    # Rate is per in-service (offered) accelerator-hour: the fleet pays it whether or not a
    # complete allocation holds the hour, so cost per useful hour = rate / ML productivity goodput.
    GpuHourRef = PricePoint(
        name="Fleet GPU-hour (reference)",
        rate=2.0 * USD / hour,
        metadata=Metadata(
            provenance=pc.FLEET_ECONOMICS_2024,
            description=(
                "Price of one in-service (offered) accelerator-hour, paid whether or not a complete "
                "allocation holds it, a retained step uses it, or that step runs at the ideal rate; "
                "dividing it by ML productivity goodput gives the cost of one useful accelerator-hour."
            ),
        ),
    )
    SpotGpuHourRef = PricePoint(name="Fleet spot GPU-hour", rate=0.70 * USD / hour, metadata=_FLEET)
    InternalChargebackPerHour = PricePoint(
        name="Internal chargeback per GPU-hour",
        rate=2.50 * USD / hour,
        metadata=_FLEET,
    )
    CarbonPerGpuHr = PricePoint(
        name="Carbon per GPU-hour (illustrative)",
        rate=0.16 * ureg.kilogram,
        metadata=Metadata(provenance=pc.CARBON_PER_GPU_HR),
    )


class Network(Registry):
    """Network link-media unit prices (added 2026-09-28)."""
    DacCableUsd = PricePoint(name="Direct-attach copper cable (per cable)", rate=50 * USD, metadata=_NETWORK)
    AocCableUsd = PricePoint(name="Active optical cable (per cable)", rate=500 * USD, metadata=_NETWORK)
    OpticsPerLinkUsd = PricePoint(name="Optical link (per link)", rate=500 * USD, metadata=_NETWORK)


class Labor(Registry):
    """Engineering labor rates (added 2026-09-28)."""
    MlEngineerLoadedHourly = PricePoint(name="ML engineer (fully loaded, per hour)", rate=150 * USD / hour, metadata=_LABOR)


class Monitoring(Registry):
    """Monitoring and observability price anchors."""

    IngestionPerMillionDatapoints = PricePoint(
        name="Monitoring ingestion per million datapoints",
        rate=0.30 * USD,
        metadata=_CLOUD,
    )
    StoragePerGbMonth = PricePoint(
        name="Monitoring metric storage",
        rate=1.00 * USD / GB / ureg.month,
        metadata=_CLOUD,
    )
    QueryPerRequest = PricePoint(
        name="Monitoring query request",
        rate=0.02 * USD,
        metadata=_CLOUD,
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
    # --- Added 2026-09-28: amortization lifetimes (years, dimensionless) ---
    AcceleratorServiceLifeYears = PricePoint(
        name="Accelerator service life for amortization (years)",
        rate=3,
        metadata=_CAPITAL,
    )
    AcceleratorExtendedServiceLifeYears = PricePoint(
        name="Accelerator extended service life (years, longer-life contrast)",
        rate=4,
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
    Monitoring = Monitoring
    Capital = Capital
    OnPremises = OnPremises
    Network = Network
    Labor = Labor
