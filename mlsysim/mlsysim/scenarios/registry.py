"""Reference statistics for real-world scenarios and case studies.

Executable scenarios live in :mod:`mlsysim.engine.scenarios`. This module holds
non-executable, provenance-carrying anchors such as Gmail volume, Waymo sensor
rate, mobile power envelopes, and TinyML case-study measurements.

The boundary is intentional:

* ``Models.*`` owns model/workload facts.
* ``Hardware.*`` and ``Systems.*`` own machine and fleet facts.
* ``Scenarios.*`` owns runnable workload + system + constraint bundles.
* ``ReferenceStats.*`` owns sourced statistics that describe the world but are
  not themselves runnable simulator inputs.
"""
from ..core.provenance import sourced, sourced_qty
from ..core.registry import Registry
from ..core import provenance_catalog as pc
from ..core.units import ureg, TB

_hour = ureg.hour
_joule = ureg.joule


class Workloads(Registry):
    """Illustrative real-world workload scale anchors (order-of-magnitude intuition)."""

    GmailEmailsPerDay = sourced(
        121e9, pc.BOOK_WORKLOAD_SCALE,
        name="Gmail emails per day", description="Approximate daily Gmail volume.")
    GoogleSearchesPerDay = sourced(
        8.5e9, pc.BOOK_WORKLOAD_SCALE,
        name="Google searches per day", description="Approximate daily Google search volume.")
    WaymoDataPerHourLow = sourced_qty(
        1 * TB / _hour, pc.BOOK_WORKLOAD_SCALE,
        name="Waymo sensor data rate (low)", description="Lower-bound AV sensor data generation rate.")
    WaymoDataPerHourHigh = sourced_qty(
        19 * TB / _hour, pc.BOOK_WORKLOAD_SCALE,
        name="Waymo sensor data rate (high)", description="Upper-bound AV sensor data generation rate.")


class AnomalyModel(Registry):
    """TinyML anomaly-detection case study (benchmarking example)."""

    Latency = sourced_qty(
        10.4 * ureg.ms, pc.BOOK_ANOMALY_CASE,
        name="Anomaly model latency", description="Inference latency of the TinyML anomaly detector.")
    Auc = sourced(
        0.86, pc.BOOK_ANOMALY_CASE,
        name="Anomaly model AUC", description="Area under the ROC curve for the TinyML anomaly detector.")
    Energy = sourced_qty(
        516 * ureg.microjoule, pc.BOOK_ANOMALY_CASE,
        name="Anomaly model energy", description="Per-inference energy of the TinyML anomaly detector.")


class EnergyAnchors(Registry):
    """Everyday energy-scale comparison anchors (order-of-magnitude intuition)."""

    SmartphoneCharge = sourced_qty(
        40_000 * _joule, pc.BOOK_ENERGY_ANCHORS,
        name="Smartphone full charge", description="Approximate energy to fully charge a smartphone battery.")
    BoilingWater = sourced_qty(
        100_000 * _joule, pc.BOOK_ENERGY_ANCHORS,
        name="Boiling 1 L of water", description="Approximate energy to bring one liter of water to a boil.")


class MobilePower(Registry):
    """Mobile/edge device + workload power-envelope reference figures."""

    MobileNpuTypical = sourced_qty(3 * ureg.watt, pc.BOOK_DEVICE_ANCHORS,
        name="Mobile NPU typical power", description="Typical sustained power for on-device mobile inference.")
    MobileNpuPeak = sourced_qty(4 * ureg.watt, pc.BOOK_DEVICE_ANCHORS,
        name="Mobile NPU peak power", description="Higher-bound mobile inference power envelope.")
    ObjectDetector = sourced_qty(2 * ureg.watt, pc.BOOK_DEVICE_ANCHORS,
        name="Object-detector power", description="Reference power draw of an always-on object-detection workload.")
    MobileMlSustainedLow = sourced_qty(2 * ureg.watt, pc.BOOK_DEVICE_ANCHORS,
        name="Mobile ML sustained power (low)", description="Lower bound for sustained smartphone ML processing.")
    MobileMlSustainedHigh = sourced_qty(3 * ureg.watt, pc.BOOK_DEVICE_ANCHORS,
        name="Mobile ML sustained power (high)", description="Upper bound for sustained smartphone ML processing.")
    MobileMlBurstLow = sourced_qty(5 * ureg.watt, pc.BOOK_DEVICE_ANCHORS,
        name="Mobile ML burst power (low)", description="Lower bound for brief smartphone ML burst processing.")
    MobileMlBurstHigh = sourced_qty(10 * ureg.watt, pc.BOOK_DEVICE_ANCHORS,
        name="Mobile ML burst power (high)", description="Upper bound for brief smartphone ML burst processing.")
    BackgroundAdaptationLow = sourced_qty(500 * ureg.milliwatt, pc.BOOK_DEVICE_ANCHORS,
        name="Background adaptation power budget (low)", description="Lower bound for background on-device adaptation power.")
    BackgroundAdaptationHigh = sourced_qty(1000 * ureg.milliwatt, pc.BOOK_DEVICE_ANCHORS,
        name="Background adaptation power budget (high)", description="Upper bound for background on-device adaptation power.")


class PhoneBattery(Registry):
    """Flagship smartphone battery reference figures.

    Note: EnergyWh (15 Wh, a flagship pack rating) and EnergyJ (capacity x voltage =
    3000 mAh x 3.7 V = 11.1 Wh) are two DISTINCT battery models the book uses in
    different examples; both are preserved as-is rather than reconciled."""

    CapacityMah = sourced_qty(3000 * ureg.milliampere_hour, pc.BOOK_DEVICE_ANCHORS,
        name="Phone battery capacity", description="Typical flagship smartphone battery charge capacity.")
    VoltageV = sourced_qty(3.7 * ureg.volt, pc.BOOK_DEVICE_ANCHORS,
        name="Phone battery voltage", description="Nominal Li-ion cell voltage.")
    EnergyJ = sourced_qty((3000 * ureg.milliampere_hour * 3.7 * ureg.volt).to(ureg.joule), pc.BOOK_DEVICE_ANCHORS,
        name="Phone battery energy (capacity x voltage)", description="Battery energy derived from capacity x voltage.")
    EnergyWh = sourced_qty(15 * ureg.watt * ureg.hour, pc.BOOK_DEVICE_ANCHORS,
        name="Flagship phone battery energy", description="Modern flagship smartphone battery pack energy rating.")


class ReferenceStats(Registry):
    """Registry namespace for non-executable real-world scenario statistics."""

    Workloads = Workloads
    AnomalyModel = AnomalyModel
    EnergyAnchors = EnergyAnchors
    MobilePower = MobilePower
    PhoneBattery = PhoneBattery
