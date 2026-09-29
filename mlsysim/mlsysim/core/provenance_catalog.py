"""Shared provenance records (stable ids, single definition)."""

from __future__ import annotations

from .provenance import Provenance, ProvenanceKind


def _ds(
    id: str,
    ref: str,
    url: str,
    *,
    verified: str = "2026-03-06",
    notes: str | None = None,
) -> Provenance:
    """Creates a Provenance object for a hardware datasheet or technical specification."""
    return Provenance(
        id=id,
        kind=ProvenanceKind.DATASHEET,
        ref=ref,
        url=url,
        verified=verified,
        notes=notes,
    )


def _lit(
    id: str,
    ref: str,
    *,
    url: str | None = None,
    verified: str = "2026-03-06",
    notes: str | None = None,
) -> Provenance:
    """Creates a Provenance object for peer-reviewed literature or academic whitepapers."""
    return Provenance(
        id=id,
        kind=ProvenanceKind.LITERATURE,
        ref=ref,
        url=url,
        verified=verified,
        notes=notes,
    )


def _est(
    id: str,
    ref: str,
    notes: str,
    *,
    url: str | None = None,
    verified: str = "2026-03-06",
) -> Provenance:
    """Creates a Provenance object for expert estimates or rules of thumb."""
    return Provenance(
        id=id,
        kind=ProvenanceKind.ESTIMATE,
        ref=ref,
        url=url,
        verified=verified,
        notes=notes,
    )


def _conv(id: str, ref: str, *, notes: str | None = None) -> Provenance:
    """Creates a Provenance object for widely accepted industry conventions."""
    return Provenance(
        id=id,
        kind=ProvenanceKind.CONVENTION,
        ref=ref,
        verified="2026-03-06",
        notes=notes,
    )


# --- Grid and reference anchors ---
IEA_WEO_2023 = Provenance(
    id="prov:iea-weo-2023-carbon",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="IEA World Energy Outlook 2023 (rounded gCO2/kWh)",
    url="https://www.iea.org/reports/world-energy-outlook-2023",
    verified="2026-03-06",
)

UPTIME_PUE_2022 = Provenance(
    id="prov:uptime-pue-survey-2022",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="Uptime Institute Global Data Center Survey 2022",
    url="https://uptimeinstitute.com/resources/research-and-reports/uptime-institute-global-data-center-survey-2022",
    verified="2026-03-06",
)

CLUSTER_TIER_CONVENTIONS = _conv(
    "prov:cluster-tier-convention",
    "MLSysIM reference cluster tiers (256 / 2k / 8k / 100k GPUs)",
)

KEMPNER_AI_CLUSTER_H100 = _ds(
    "prov:kempner-ai-cluster-h100",
    "Kempner Institute Computing Handbook, Overview of Cluster - H100 partition specs",
    "https://handbook.eng.kempnerinstitute.harvard.edu/s1_high_performance_computing/kempner_cluster/overview_of_kempner_cluster.html",
    verified="2026-06-01",
    notes="H100 partition: 384 H100 80GB GPUs, 24 servers per rack, 4 GPUs per server, four H100 racks.",
)

# --- Real-world case-study / workload scale anchors (Scenarios registry) ---
REFERENCE_WORKLOAD_SCALE = Provenance(
    id="prov:reference-workload-scale",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Illustrative real-world scale anchors (Gmail volume, Google searches, Waymo sensor rate) for order-of-magnitude intuition",
    verified="2026-03-06",
)
TINYML_ANOMALY_CASE = Provenance(
    id="prov:tinyml-anomaly-case",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="TinyML anomaly-detection case study (latency / AUC / energy) used as a benchmarking example",
    verified="2026-03-06",
)
CLINICAL_IMAGING_WORKFLOW_ANCHORS = Provenance(
    id="prov:clinical-imaging-workflow-anchors",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Clinical imaging workflow anchors for rural-clinic bandwidth and edge-deployment examples",
    verified="2026-06-01",
)
OURA_SLEEP_STAGE_STUDY = _lit(
    "prov:oura-sleep-stage-study",
    "Altini and Kinnunen (2021), \"The Promise of Sleep: A Multi-Sensor Approach for Accurate Sleep Stage Detection Using the Oura Ring\"",
    url="https://doi.org/10.3390/s21134302",
    verified="2026-06-03",
)
PHYSICAL_AI_SENSORS = _ds(
    "prov:physical-ai-sensors",
    "Automotive and industrial physical AI sensor specifications and datasheets",
    "https://www.sony-semicon.com/en/products/is/automotive/",
    notes="4K 30fps automotive camera sensors, MIPI CSI-2 interfaces, 128-beam mechanical LiDARs, and industrial process monitoring transducers.",
)
ENERGY_SCALE_ANCHORS = Provenance(
    id="prov:energy-scale-anchors",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Everyday energy-scale comparison anchors (smartphone charge ~40 kJ, boiling a cup of water ~100 kJ, US household electricity ~10.7 MWh/year) for order-of-magnitude intuition about ML energy",
    verified="2026-03-06",
)
MOBILE_DEVICE_ANCHORS = Provenance(
    id="prov:mobile-device-anchors",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Mobile/edge device reference figures (flagship phone battery ~15 Wh / 3000 mAh @ 3.7 V, mobile NPU power 3-4 W, object-detector ~2 W) for on-device ML intuition",
    verified="2026-03-06",
)
EDGE_DEVICE_SPECTRUM_ANCHORS = _est(
    "prov:edge-device-spectrum-anchors",
    "Representative edge-device spectrum anchors for on-device learning heterogeneity examples",
    notes="Pedagogical class endpoints spanning microcontroller/tiny-sensor and flagship-mobile envelopes; not a single SKU specification.",
    verified="2026-06-04",
)
EDGE_ADAPTATION_TIER_PROFILE = _est(
    "prov:edge-adaptation-tier-profile",
    "Reference edge adaptation tier profile",
    notes=(
        "Reusable device-tier, wearable, and fleet-mix assumptions for on-device "
        "learning examples. These are pedagogical deployment-profile anchors, "
        "not a claim about one production fleet."
    ),
    verified="2026-06-04",
)

# --- Hardware technology-class facts (Hardware.Tech) ---
HOROWITZ_ENERGY = _lit(
    "prov:horowitz-2014",
    "Horowitz (2014), \"Computing's Energy Problem (and what we can do about it)\", ISSCC — 45 nm per-operation/per-byte energies",
    url="https://ieeexplore.ieee.org/document/6757323",
)
ON_CHIP_MEMORY_ACCESS_ENERGY_ANCHORS = Provenance(
    id="prov:on-chip-memory-access-energy-anchors",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="MLSysIM illustrative register, L1, and L2 access-energy anchors",
    verified="2026-08-11",
    notes="Pedagogical hierarchy anchors, not values reported by Horowitz (2014).",
)
MEMORY_LATENCY_HIERARCHY = _conv(
    "prov:memory-latency-hierarchy",
    "MLSysIM memory/interconnect access-latency hierarchy (order-of-magnitude class figures)",
)
STORAGE_TIER_CONVENTIONS = _conv(
    "prov:storage-tier-conventions",
    "Generic storage/memory bandwidth tiers (NVMe Gen3/4/5, DDR, host DRAM) from vendor datasheet ranges",
)
CXL_PCIE_GEN5_BW = Provenance(
    id="prov:cxl-pcie-gen5-x16-bw-derived",
    kind=ProvenanceKind.DERIVED,
    ref="Reference CXL memory tier over PCIe Gen5 x16 -> 64 GB/s",
    notes="Teaching anchor for a CXL 3.x memory-expansion link using the same byte-rate convention as the H100 PCIe Gen5 x16 profile.",
    verified="2026-06-03",
)
STORAGE_ACCESS_PATH_REFERENCE = _conv(
    "prov:storage-access-path-reference",
    "Reference GPU-storage access-path latencies for traditional CPU-mediated I/O and GPU Direct Storage bypass paths",
)
FRAMEWORK_RUNTIME_OVERHEAD_REFERENCE = _conv(
    "prov:framework-runtime-overhead-reference",
    "Reference framework/runtime overhead latencies for dispatch, kernel launch, and tiny memory-access operations",
)

RELIABILITY_MTTF_LITERATURE = _lit(
    "prov:reliability-mttf-literature",
    "Kokolis et al. (2025, HPCA); Zu et al. (2024, NSDI); Barroso et al. (2018) — order-of-magnitude steady-state MTTF",
    url="https://doi.org/10.1109/hpca61900.2025.00096",
)

ILLUSTRATIVE_IOWA_CARBON = Provenance(
    id="prov:illustrative-iowa-carbon",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Illustrative high-carbon US grid contrast (not IEA country average)",
    verified="2026-03-06",
)

HYDRO_QUEBEC_GRID = _ds(
    "prov:hydro-quebec-grid",
    "Hydro-Québec electricity mix and carbon intensity",
    "https://www.hydroquebec.com/about/our-energy.html",
)

# --- Cloud accelerators ---
NVIDIA_H200 = _ds(
    "prov:nvidia-h200-datasheet",
    "NVIDIA H200 Tensor Core GPU product documentation",
    "https://www.nvidia.com/en-us/data-center/h200/",
)

NVIDIA_GB200_NVL72 = _ds(
    "prov:nvidia-gb200-nvl72-datasheet",
    "NVIDIA GB200 NVL72 rack-scale system documentation",
    "https://www.nvidia.com/en-us/data-center/gb200-nvl72/",
)

AMD_MI250X = _ds(
    "prov:amd-mi250x-datasheet",
    "AMD Instinct MI250X product documentation",
    "https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html",
)

INTEL_GAUDI2 = _ds(
    "prov:intel-gaudi2-datasheet",
    "Intel Gaudi 2 AI accelerator product documentation",
    "https://www.intel.com/content/www/us/en/products/details/processors/ai-accelerators/gaudi2.html",
)

INTEL_GAUDI3 = _ds(
    "prov:intel-gaudi3-datasheet",
    "Intel Gaudi 3 AI accelerator product documentation",
    "https://www.intel.com/content/www/us/en/products/details/processors/ai-accelerators/gaudi3.html",
)

AWS_TRAINIUM2 = _ds(
    "prov:aws-trainium2-datasheet",
    "AWS Trainium2 accelerator (EC2 Trn2) product documentation",
    "https://aws.amazon.com/ai/machine-learning/trainium/",
)

GOOGLE_TPU_V1 = _lit(
    "prov:google-tpu-v1",
    "Jouppi et al. (2017), In-Datacenter Performance Analysis of a Tensor Processing Unit",
    url="https://arxiv.org/abs/1704.04760",
)

GOOGLE_TPU_V2_V3 = _lit(
    "prov:google-tpu-v2-v3",
    "Jouppi et al. (2020), A Domain-Specific Supercomputer for Training Deep Neural Networks",
    url="https://doi.org/10.1145/3360307",
)

GOOGLE_TPU_V4 = _lit(
    "prov:google-tpu-v4",
    "Jouppi et al. (2023), TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings, ISCA",
    url="https://arxiv.org/abs/2304.01433",
)

GOOGLE_TPU_V5P = _ds(
    "prov:google-tpu-v5p",
    "Google Cloud TPU v5p documentation",
    "https://cloud.google.com/tpu/docs/v5p",
)
GOOGLE_TPU_V5P_WITH_VMEM = Provenance(
    id="prov:google-tpu-v5p-with-vmem",
    kind=ProvenanceKind.DERIVED,
    ref="Google Cloud TPU v5p specifications and JAX TPU hardware reference",
    url="https://docs.jax.dev/en/latest/pallas/tpu/hardware.html",
    verified="2026-08-11",
    notes="The 128 MiB VMEM capacity is derived from two TensorCores per chip and 64 MiB of VMEM per TensorCore; other fields follow the Google Cloud v5p specification.",
)

GOOGLE_TPU_V6 = _ds(
    "prov:google-tpu-v6-trillium",
    "Google Cloud TPU v6e (Trillium) documentation",
    "https://cloud.google.com/tpu/docs/v6e",
)

CEREBRAS_CS3 = _ds(
    "prov:cerebras-cs3-datasheet",
    "Cerebras CS-3 system product documentation",
    "https://www.cerebras.net/product-system/",
)

INTEL_SGX = _ds(
    "prov:intel-sgx-datasheet",
    "Intel Software Guard Extensions (SGX) developer guide",
    "https://www.intel.com/content/www/us/en/developer/tools/software-guard-extensions/overview.html",
)

REFERENCE_DESKTOP_CPU = _conv(
    "prov:reference-desktop-cpu",
    "Reference 1 TFLOP/s FP32 desktop CPU for pedagogy (order-of-magnitude)",
)

# --- Workstation / mobile / edge / tiny ---
NVIDIA_DGX_SPARK = _ds(
    "prov:nvidia-dgx-spark-gb10",
    "NVIDIA DGX Spark (GB10 Grace Blackwell) product page",
    "https://www.nvidia.com/en-us/products/workstations/dgx-spark/",
)

APPLE_M3_MAX = _est(
    "prov:apple-m3-max-estimate",
    "Apple M3 Max technical specifications (GPU core count × peak FLOP/core, rounded)",
    notes="Peak TFLOP/s is an MLSysIM rounded estimate of Apple-published core counts, not a sustained ML benchmark.",
    url="https://www.apple.com/macbook-pro/specs/",
)

MOBILE_SOC_ESTIMATE = _est(
    "prov:mobile-npu-peak-estimate",
    "Smartphone SoC NPU peak TOPS from vendor product briefs (marketing peak)",
    notes="Used for order-of-magnitude edge inference examples; not MLPerf-submitted sustained throughput.",
)

NVIDIA_JETSON_AGX_ORIN = _ds(
    "prov:nvidia-jetson-agx-orin",
    "NVIDIA Jetson AGX Orin technical brief",
    "https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/",
)

NVIDIA_JETSON_ORIN_NX = _ds(
    "prov:nvidia-jetson-orin-nx",
    "NVIDIA Jetson Orin NX series documentation",
    "https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/",
)

NVIDIA_DRIVE_THOR = _ds(
    "prov:nvidia-drive-thor",
    "NVIDIA DRIVE Thor architecture whitepaper and technical brief",
    "https://www.nvidia.com/en-us/autonomous-machines/drive-thor/",
    notes="Thor autonomous vehicle and physical AI SoC integrating Blackwell GPU architecture and Arm Neoverse V3AE CPU cores.",
)

# Embodied platforms (Embodied.*). A record is DATASHEET only when every
# numeric field on its registry entry was checked against the linked source;
# otherwise it is ESTIMATE, keeps the real spec URL, and its notes name the
# fields that are verified and the fields that are MLSysIM estimates.
BOSTON_DYNAMICS_SPOT = _est(
    "prov:boston-dynamics-spot",
    "Boston Dynamics Spot SDK documentation, About Spot (robot specifications table)",
    notes=(
        "Verified: 12 degrees of freedom, 1.6 m/s maximum speed, 14 kg maximum "
        "payload. The SDK table and the Spot + Spot Arm Information for Use v1.0 "
        "list net weight 32.5 kg; the current product page "
        "(bostondynamics.com/products/spot) lists 33.8 kg with battery. The "
        "stored 32.7 kg matches neither. Unsourced MLSysIM estimates: 1000 Hz "
        "control_frequency, 2.0 m/s^2 max_acceleration, and 400 W nominal_power "
        "(the SDK lists 400 W charger power and a 605 Wh battery with 90 minute "
        "typical runtime, about 400 W average draw, but no power rating). The "
        "transmission description and Jetson AGX Orin compute_soc are not from "
        "Boston Dynamics documentation."
    ),
    url="https://dev.bostondynamics.com/docs/concepts/about_spot.html",
    verified="2026-09-15",
)

BOSTON_DYNAMICS_ATLAS = _est(
    "prov:boston-dynamics-atlas",
    "Boston Dynamics Atlas (hydraulic) product page, Internet Archive capture of 2022-05-30",
    notes=(
        "The stored values describe the retired hydraulic Atlas. Verified "
        "against the archived page: 89 kg weight, 28 hydraulic joints, 2.5 m/s "
        "speed (1.5 m height). Unsourced MLSysIM estimates: 11 kg "
        "payload_capacity, 5.0 m/s^2 max_acceleration, 1000 Hz "
        "control_frequency, 1500 W nominal_power, the electric QDD half of the "
        "transmission description, and the Jetson AGX Orin compute_soc. The "
        "live bostondynamics.com/atlas page now describes the electric Atlas "
        "(90 kg, 56 degrees of freedom, 50 kg instant and 30 kg sustained lift, "
        "4 hour battery), which these values do not represent."
    ),
    url="https://web.archive.org/web/20220530005139/https://www.bostondynamics.com/atlas",
    verified="2026-09-15",
)

FRANKA_FER_CONTROL_LIMITS = _ds(
    "prov:franka-fer-control-limits",
    "Franka Robotics documentation, Robot and interface specifications: 'Limits for "
    "Franka Emika Robot (FER)' and the Panda Denavit-Hartenberg parameters",
    "https://github.com/frankaemika/docs/blob/master/source/robot_specifications.rst",
    verified="2026-09-29",
    notes=(
        "Joint limits from source/control-parameters-joint-fer.csv (master, "
        "checked 2026-09-29): velocity 2.1750 rad/s on joints 1-4 and 2.6100 on "
        "5-7; acceleration 15, 7.5, 10, 12.5, 15, 20, 20 rad/s^2; jerk 7500, "
        "3750, 5000, 6250, 7500, 10000, 10000 rad/s^3; torque rate 1000 Nm/s on "
        "every joint. FER Cartesian translation velocity 1.7 m/s. DH d3 = 0.316 m "
        "and d5 = 0.384 m: the current page labels the table for FR3; the Panda "
        "label with d3 = 0.316 m is in earlier revisions of the docs (commits "
        "6346f562cc and 7a46004875, control_parameters.rst). Rendered site: "
        "https://frankarobotics.github.io/docs."
    ),
)

FRANKA_EMIKA_PANDA = _ds(
    "prov:franka-emika-panda",
    "Franka Emika Panda datasheet (May 2018) and Robot Instruction Handbook (October 2021)",
    "https://www.generationrobots.com/media/panda-franka-emika-datasheet.pdf",
    verified="2026-09-15",
    notes=(
        "Datasheet (reseller-hosted copy of the Franka Emika PDF): 7 DOF, 3 kg "
        "payload, 855 mm reach, torque sensors in all 7 axes, up to 2 m/s "
        "end-effector speed, arm weight about 18 kg, controller power "
        "consumption about 300 W average and 600 W maximum. The May 2019 "
        "datasheet revision lists 1 kHz control. The handbook "
        "(generationrobots.com/media/franka-emika-robot-handbook.pdf) lists "
        "87 Nm repeatable peak torque on axes 1 to 4 and 12 Nm on axes 5 to 7; "
        "max_torque stores the 87 Nm limit. The harmonic-drive transmission "
        "description and Jetson AGX Orin compute_soc are not from Franka "
        "documentation."
    ),
)

DJI_MATRICE_350_RTK = _est(
    "prov:dji-matrice-350-rtk",
    "DJI Matrice 350 RTK specifications page",
    notes=(
        "Verified: weight with two TB65 batteries about 6.47 kg, max takeoff "
        "weight 9.2 kg, max horizontal speed 23 m/s, max flight time 55 minutes. "
        "payload_capacity 2.7 kg is derived as takeoff weight minus aircraft "
        "weight (9.2 - 6.47 = 2.73 kg); DJI lists a single-gimbal max payload of "
        "960 g. Unsourced MLSysIM estimates: 6.0 m/s^2 max_acceleration, 400 Hz "
        "control_frequency, 1000 W nominal_power. The Jetson Orin Nano "
        "compute_soc is not a DJI component."
    ),
    url="https://enterprise.dji.com/matrice-350-rtk/specs",
    verified="2026-09-15",
)

INDUSTRIAL_AMR = _est(
    "prov:industrial-amr-ansi-itsdf",
    "Representative industrial logistics AMR class profile (not a single product)",
    notes=(
        "No accessible source was found for these values: 150 kg mass, 500 kg "
        "payload_capacity, 1.8 m/s max_velocity, 2.5 m/s^2 max_acceleration, "
        "100 Hz control_frequency, 500 W nominal_power, Jetson AGX Orin "
        "compute_soc. ANSI/ITSDF B56.5 and ISO 3691-4 are safety standards for "
        "driverless industrial trucks, not product specifications; their texts "
        "are paywalled and were not checked."
    ),
    verified="2026-09-15",
)

WAREHOUSE_AMR = _est(
    "prov:warehouse-amr-300kg",
    "Representative warehouse fulfillment AMR class profile (300 kg gross mass)",
    notes=(
        "300 kg mass and 1.5 m/s max velocity are class-profile values. The 2.0 m/s^2 "
        "max_acceleration is the illustrative credible loaded deceleration that the "
        "stopping budget adopts; it is not measured. It is checked against the traction, "
        "drive-torque, and tip-over ceilings by calc_braking_limits with the machine's "
        "sourced hub-motor drive (ZLTECH_ZLLG10ASM800); the tip-over ceiling needs a "
        "measured center of gravity. (2026-09-28, M6: previously attributed to a "
        "'Chapter 4 watchdog model', which is not its origin.)"
    ),
    verified="2026-09-28",
)

HEAVY_AMR = _est(
    "prov:heavy-amr-250kg",
    "Representative heavy industrial logistics AMR class profile (250 kg mass)",
    notes="250 kg mass, 1.8 m/s cruise velocity, 2.0 m/s^2 braking deceleration (Chapter 1 model).",
    verified="2026-09-18",
)

# OpenVLA on the Jetson AGX Orin (added 2026-09-28, vol4 M2 intent-model latency).
JETSON_AI_LAB_OPENVLA_ORIN = _lit(
    "prov:jetson-ai-lab-openvla-orin",
    "Franklin, D., NVIDIA Jetson AI Lab, 'OpenVLA' tutorial (2024): OpenVLA-7B "
    "quantization results on Jetson AGX Orin 64GB, JetPack 6",
    url="https://www.jetson-ai-lab.com/archive/openvla.html",
    verified="2026-09-28",
    notes=(
        "Measured mean per-frame latency over 10 BridgeData V2 episodes: FP16 "
        "(HuggingFace backend) 840 ms / 1.19 FPS / 95.3% NRMSE accuracy; FP8 "
        "(MLC q8f16_ft) 471 ms / 2.12 FPS / 95.2%; INT4 (MLC q4f16_ft) 336 ms / "
        "2.97 FPS / 90.1%. Single 224x224 image, ~275 image tokens plus the text "
        "prompt, 7 autoregressive action tokens; vision encoders in FP16 TensorRT. "
        "Archived page; first published August 2024 (repository history)."
    ),
)

# Volume IV running machine (added 2026-09-21, vol4 flow canon section 3.3).
WAREHOUSE_MOBILE_MANIPULATOR = _est(
    "prov:warehouse-mobile-manipulator",
    "Representative warehouse mobile manipulator: WarehouseAMR base, Panda arm, "
    "Jetson AGX Orin application processor, lockstep safety MCU",
    notes=(
        "Composite of existing registry entries; component values carry their own "
        "records. Illustrative or chosen here: 50 kg tote-rack capacity, 1.00 m "
        "footprint width, 1 kHz permission loop, 20 kHz current loop, 1 ms EtherCAT "
        "cycle, 9 servo axes, 2 Hz intent model (latency computed, see "
        "JETSON_AI_LAB_OPENVLA_ORIN), 20 Hz / 40 ms ACT chunk policy, "
        "25 ms setpoint period (two per 50 ms renewal), chunk horizon 16, 0.70 sustained DRAM efficiency, "
        "1.0 m/s free-space TCP limit. The WarehouseAMR record calls 300 kg gross, so "
        "the loaded mass is an upper bound."
        " Control rail: 24 V nominal is a design choice; the 21.5 V depleted-battery "
        "voltage, 60 mOhm shared harness resistance, and 18.0 V point-of-load "
        "regulator dropout are illustrative."
        " Permission-path rail isolated and held up (round-3 D15): isolated from the "
        "application processor's supply and held up by a supercapacitor store behind "
        "an ideal-diode controller, feeding the safety MCU, drive-logic and gate-driver "
        "supplies, encoder interfaces, and all spring-brake coils. The 2.0 s hold-up is "
        "a design choice; the 70 W permission-path load and the 30-80 ms spring-brake "
        "engage window are illustrative. The permission loads are assumed to share the "
        "18.0 V regulator dropout."
        " Drive bus and drivetrain (2026-09-28, M6): the wheel drives, arm drives, and a "
        "48-to-24 V converter share a 48 V-class drive bus whose pack envelope "
        "(54 / 47.7 / 42 V) comes from MIR250_BATTERY; the converter's 33 V lockout, "
        "67.2 V input maximum, and 91 percent efficiency come from MEANWELL_DDR240; the "
        "60 mOhm source resistance is illustrative (needs a measured value); the 60 V "
        "chopper clamp and 100 V switch rating are design choices. The driven wheels are "
        "ZLTECH_ZLLG10ASM800 hub motors with the current limit chosen at the motor's "
        "25 A peak. Loaded center-of-gravity height, its distance to the leading support, "
        "and the driven wheels' load share need source (corner-scale or tilt measurement)."
        " The 24 V control-rail fields remain for chapters that still model a 24 V "
        "shared rail; in the 48 V architecture they describe the permission-rail store "
        "(charged to 21.5 V, discharging to the 18 V regulator dropout)."
    ),
    verified="2026-09-28",
)

WAREHOUSE_AISLE_SCENARIO = _est(
    "prov:warehouse-aisle-scenario",
    "Illustrative warehouse aisle site scenario: delay, clearance, friction, contact, "
    "and conveyor budget terms for the warehouse mobile manipulator",
    notes=(
        "No row is measured. Clearances, delays, friction coefficients, latch, "
        "conveyor, coworker-contact, and remote-takeover times are illustrative; the "
        "lease, margin, aisle speed, heartbeat, enforcer deadline, handover speed, and "
        "the slow-down speeds before a takeover request are design choices. The human "
        "approach speed carries ISO_13855_APPROACH_SPEED."
        " The 70 A coincident control-rail transient (inference burst plus drive "
        "motors on the door threshold) is illustrative."
    ),
    verified="2026-09-21",
)

LOCKSTEP_SAFETY_MCU_REFERENCE = _est(
    "prov:lockstep-safety-mcu-reference",
    "Reference-class dual-core lockstep real-time MCU (200–400 MHz class)",
    notes=(
        "Not a single product. 400 MHz x 2 FLOP/cycle peak, 4 MiB on-chip flash, "
        "1 MiB tightly coupled SRAM, about 1 W; values are representative of the "
        "class, not a datasheet."
        " The registry clock_rate (400 MHz) is the upper end of the class."
    ),
    verified="2026-09-21",
)

ISO_13855_APPROACH_SPEED = _ds(
    "prov:iso-13855-approach-speed",
    "ISO 13855:2010 walking approach speed K = 1600 mm/s, applied over the whole "
    "stopping time T in S = K*T + C",
    "https://www.iso.org/standard/42845.html",
    verified="2026-09-21",
    notes=(
        "ISO 13855:2010 (Safety of machinery: positioning of safeguards with respect "
        "to the approach speeds of parts of the human body); the URL is that edition. "
        "The standard's text is paywalled, so the value was checked against secondary "
        "sources that quote it, not the standard itself, and not against the 2024 "
        "edition."
    ),
)

ISO_TS_15066_HAND_BODY_MODEL = _ds(
    "prov:iso-ts-15066-hand-body-model",
    "ISO/TS 15066:2016 Annex A body model, hand and fingers: effective mass "
    "m_H = 0.6 kg, effective spring constant k_H = 75 N/mm",
    "https://arxiv.org/abs/2203.02706",
    verified="2026-09-29",
    notes=(
        "ISO/TS 15066:2016 (Robots and robotic devices: collaborative robots), "
        "Annex A, whose collaborative-application content ISO 10218-2:2025 "
        "absorbed. The standard is paywalled; the values were checked against "
        "Kirschner, Mansfeld, Abdolshah, and Haddadin, 'ISO/TS 15066: How "
        "Different Interpretations Affect Risk Assessment' (arXiv:2203.02706, "
        "2022), which reproduces TS Appendix A (hand m_h = 0.6 kg, "
        "k = 75 N/mm; back m_h = 40 kg, k = 35 N/mm) and its transient-contact "
        "model F = v*sqrt(mu*k), mu = (1/m_r + 1/m_h)^-1 (eqs. 1-2)."
    ),
)

# --- Human takeover response times (2026-09-28, M4) ---
CARD_MORAN_NEWELL_MHP = _lit(
    "prov:card-moran-newell-1983-mhp",
    "Card, S. K., Moran, T. P., and Newell, A., The Psychology of Human-Computer "
    "Interaction, Lawrence Erlbaum Associates (1983); CRC Press reprint 2018, "
    "DOI 10.1201/9780203736166. Model Human Processor cycle times.",
    url="https://doi.org/10.1201/9780203736166",
    verified="2026-09-28",
    notes=(
        "Processor cycle times, nominal [range]: perceptual 100 [50-200] ms, cognitive "
        "70 [25-170] ms, motor 70 [30-100] ms; a simple reaction takes one cycle of each, "
        "240 ms nominal and 470 ms with every processor at its slow bound. Values checked "
        "against Jastrzembski and Charness (2007), PMC4591021, which quotes the 1983 "
        "parameters. The ranges bracket the model, not a percentile of a measured "
        "population. Applies to an attending person responding to an expected stimulus."
    ),
)

GREEN_2000_PERCEPTION_BRAKE = _lit(
    "prov:green-2000-perception-brake",
    "Green, M., '\"How long does it take to stop?\" Methodological analysis of driver "
    "perception-brake times', Transportation Human Factors 2(3):195-216 (2000), "
    "DOI 10.1207/STHF0203_1",
    url="https://doi.org/10.1207/STHF0203_1",
    verified="2026-09-28",
    notes=(
        "Abstract (Crossref and TRID record): a driver fully aware of the time and place "
        "of the brake signal detects it and moves the foot to the brake in about "
        "0.70-0.75 s; an unexpected but common signal (lead-car brake lights) takes about "
        "1.25 s; a surprise event about 1.5 s. Expectation changes the time by about a "
        "factor of two. The registry uses 0.75 s, the upper end of the expected range. "
        "These are typical values synthesized from many data sets, not percentiles."
    ),
)

ZHANG_2019_TAKEOVER_META = _lit(
    "prov:zhang-2019-takeover-meta-analysis",
    "Zhang, B., de Winter, J., Varotto, S., Happee, R., and Martens, M., 'Determinants "
    "of take-over time from automated driving: A meta-analysis of 129 studies', "
    "Transportation Research Part F 64:285-307 (2019), DOI 10.1016/j.trf.2019.04.020; "
    "supplementary data DOI 10.4121/uuid:75c28abe-6559-4273-85f4-927e969c1c59 (CC0)",
    url="https://doi.org/10.1016/j.trf.2019.04.020",
    verified="2026-09-28",
    notes=(
        "Paper, section 3.1: mean take-over time (TOT) across 520 conditions of 129 "
        "studies ranged 0.69-19.79 s; the average mean TOT was 2.72 s (SD 1.45), and the "
        "distribution of mean TOTs is right-skewed. Section 3.3: mean and SD of TOT "
        "correlate at r = 0.82 (n = 397). Percentiles computed here from the 520 MTOT "
        "values in the supplementary mastersheet.xlsx (observed-value, inverted-CDF "
        "percentiles): p50 2.37 s, p90 4.33 s, p99 7.0 s; the same file reproduces "
        "2.72 s mean and 1.45 s SD. These are percentiles of condition means, not of "
        "individual responses: the individual tail within a condition is longer (the "
        "file's SD rises about 0.39 s per second of mean). Take-over is measured from "
        "the take-over request to the driver's first input."
    ),
)

SCHIMPE_2022_TELEOP_G2G = _lit(
    "prov:schimpe-2022-teleop-glass-to-glass",
    "Schimpe, A., Feiler, J., Hoffmann, S., Majstorovic, D., and Diermeyer, F., 'Open "
    "Source Software for Teleoperated Driving', 2022 International Conference on "
    "Connected Vehicle and Expo (ICCVE), pp. 1-6, DOI 10.1109/ICCVE52871.2022.9742859",
    url="https://arxiv.org/abs/2109.11252",
    verified="2026-09-28",
    notes=(
        "Measured glass-to-glass latency of a 40 Hz, 520p video feed sent over a wired "
        "connection and shown on a 144 Hz monitor: about 104 ms. The same stack on a "
        "Jetson TX2 RC car at 15 Hz measured 241 ms. The wired figure is a best case; a "
        "site's wireless console link needs its own measurement."
    ),
)

AUTONOMOUS_VEHICLE_ROBOTAXI = _est(
    "prov:autonomous-vehicle-robotaxi",
    "Representative Level 4 robotaxi class profile (not a single vehicle)",
    notes=(
        "No accessible source was found for these values: 2200 kg mass, "
        "33.3 m/s (120 km/h) max_velocity, 4.0 m/s^2 max_acceleration, 100 Hz "
        "control_frequency, 2500 W nominal_power. The DRIVE Thor compute_soc "
        "carries its own NVIDIA_DRIVE_THOR record."
    ),
    verified="2026-09-15",
)

UNITREE_H1_HUMANOID = _est(
    "prov:unitree-h1-humanoid",
    "Unitree H1 product page and parameter table",
    notes=(
        "Verified: about 47 kg, 3.3 m/s moving speed, 360 N*m maximum joint "
        "torque (knee), M107 joint motors. The page lists 5 degrees of freedom "
        "per leg and 4 per arm, 18 in total, not the stored 19. Standard compute "
        "is an Intel Core i5 plus Core i7 with Orin NX optional, not the stored "
        "Jetson AGX Orin. Unsourced MLSysIM estimates: 30 kg payload_capacity "
        "(no H1 payload is listed), 1000 Hz control_frequency, 800 W "
        "nominal_power (the page lists an 864 Wh battery but no power rating)."
    ),
    url="https://www.unitree.com/h1/",
    verified="2026-09-15",
)

UBER_ATG_VOLVO_XC90 = _lit(
    "prov:uber-atg-volvo-xc90",
    "NTSB Highway Accident Report: Collision Between a Self-Driving Car and a Pedestrian, Tempe, Arizona, March 18, 2018 (NTSB/HAR-19/03)",
    url="https://www.ntsb.gov/investigations/AccidentReports/Reports/HAR1903.pdf",
    verified="2026-09-18",
    notes="Modified 2017 Volvo XC90 test vehicle operated by Uber ATG. NTSB reported one-second action suppression after hazard recognition at T-1.2 s; 8.0 m/s^2 braking deceleration is an illustrative assumption.",
)

ALOHA_BIMANUAL_MANIPULATOR = _lit(
    "prov:aloha-bimanual-manipulator",
    "Zhao et al. (2023), Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware",
    url="https://arxiv.org/abs/2304.13705",
    verified="2026-09-18",
    notes="ALOHA dual ViperX 300 6-DOF arms with leader-follower teleoperation, 50 Hz control loop, 14 total DOFs.",
)

DED_MELT_POOL_PROCESS = _est(
    "prov:ded-melt-pool-process",
    "Representative laser directed energy deposition (DED) additive manufacturing testbed",
    verified="2026-09-18",
    notes="High-speed coaxially monitored melt pool process with 1000 Hz closed-loop control.",
)

SONY_IMX477 = _ds(
    "prov:sony-imx477",
    "Sony IMX477 Diagonal 7.857 mm (Type 1/2.3) 12.3MP CMOS Image Sensor Datasheet",
    "https://www.sony-semicon.com/files/62/pdf/p-13_IMX477-AACK_Flyer.pdf",
    verified="2026-09-18",
    notes="Full-resolution 4056x3040 at 60 fps uses the flyer-supported four-lane, 10-bit CSI-2 mode (up to 2.1 Gbps/lane); chapter pipeline stage delays beyond exposure/readout are illustrative assumptions.",
)

SONY_IMX296 = _ds(
    "prov:sony-imx296",
    "Sony IMX296LLR/LQR Diagonal 6.3 mm (Type 1/2.9) 1.58MP Global Shutter CMOS Sensor",
    "https://www.sony-semicon.com",
    verified="2026-09-18",
    notes="Global shutter CMOS sensor with 1440x1080 resolution, sub-10ms latency profile for robotics.",
)

INTEL_REALSENSE_D435I = _ds(
    "prov:intel-realsense-d435i",
    "RealSense D400 Series Product Family Datasheet, revision 023 (March 2026), D435/D435i",
    "https://www.realsenseai.com/wp-content/uploads/2026/03/RealSense-D400-Series-Datasheet-Mar-2026.pdf",
    verified="2026-09-29",
    notes=(
        "Active IR stereo depth camera with integrated IMU, 90 fps depth stream. "
        "Table 3-49: the D435/D435i uses the D430 depth module (wide imagers) and "
        "an OV2740 color sensor. Table 3-18: OV2740 is rolling shutter, 1920x1080. "
        "The wide left/right imagers (OV9282) are global shutter. shutter_type "
        "describes the color stream; depth_shutter_type the depth imagers. "
        "nominal_latency and nominal_power are not from this datasheet."
    ),
)

OUSTER_OS1_64 = _ds(
    "prov:ouster-os1-64",
    "Ouster OS1 Mid-Range High-Resolution Imaging Lidar Datasheet, hardware Rev7.0, firmware 3.1.x",
    "https://data.ouster.io/downloads/datasheets/datasheet-rev7-v3p1-os1.pdf",
    verified="2026-09-29",
    notes=(
        "64-channel digital LiDAR, 10 or 20 Hz rotation. Range precision 0.5-3 cm "
        "(1 sigma, beyond 1 m, 1024 @ 10 Hz); range accuracy +/-2.5 cm Lambertian; "
        "data latency < 10 ms (stored as nominal_latency); PTP / gPTP / PPS "
        "accuracy < 1 ms; internal clock 10 ppm drift; power 14-20 W. The Rev7 "
        "sheet gives range as 170 m (80 percent target) and 90 m (10 percent), so "
        "the stored 120 m max_range needs source; nominal_power 18 W sits in the "
        "14-20 W band."
    ),
)

BOSCH_BMI088 = _ds(
    "prov:bosch-bmi088",
    "Bosch Sensortec BMI088 High-Performance 6-Axis Inertial Measurement Unit Datasheet",
    "https://www.bosch-sensortec.com/products/motion-sensors/imus/bmi088/",
    verified="2026-09-18",
    notes="Automotive and robotics grade 6-DoF IMU with 1000 Hz gyroscope and 1600 Hz accelerometer update rates.",
)

HARMONIC_DRIVE_CSG = _ds(
    "prov:harmonic-drive-csg",
    "Harmonic Drive CSG-25-50-2UH product performance data",
    "https://www.harmonicdrive.net/products/gear-units/gear-units/csg-2uh/csg-25-50-2uh",
    verified="2026-09-20",
    notes="Specific 50:1 gear unit: 51 N m L10 rated torque, 72 N m average limit, 127 N m repeated peak, and 242 N m momentary peak. Registry rotor inertia and electrical fields are teaching assumptions, not product specifications.",
)

UNITREE_M107_MOTOR = _ds(
    "prov:unitree-m107-motor",
    "Unitree M107 High Torque Joint Motor Technical Specifications",
    "https://www.unitree.com",
    verified="2026-09-18",
    notes="Planetary/QDD joint motor providing 360 N*m peak torque used in humanoid hip and knee joints.",
)

TMOTOR_AK80_9 = _ds(
    "prov:tmotor-ak80-9",
    "T-Motor AK80-9 Dynamic Actuator Specifications",
    "https://store.tmotor.com",
    verified="2026-09-18",
    notes="9:1 planetary quasi-direct-drive actuator for dynamic legged robotics.",
)

DYNAMIXEL_XM430 = _ds(
    "prov:dynamixel-xm430-w350",
    "ROBOTIS DYNAMIXEL XM430-W350-T/R E-Manual",
    "https://emanual.robotis.com/docs/en/dxl/x/xm430-w350/",
    verified="2026-09-18",
    notes="Integrated robot actuator with contactless absolute encoder, TTL/RS-485 multidrop bus.",
)

GOOGLE_CORAL = _ds(
    "prov:google-coral-edge-tpu",
    "Google Coral Edge TPU product documentation",
    "https://coral.ai/products/",
)

INTEL_NUC_MOVIDIUS = _est(
    "prov:intel-nuc-movidius-estimate",
    "Intel NUC + Movidius VPU reference kit (peak TOPS from Intel Neural Compute Stick spec)",
    notes="Composite edge reference node for tutorials; not a single-SKU datasheet.",
    url="https://www.intel.com/content/www/us/en/products/details/processors/neural-processing-unit.html",
)

REFERENCE_EDGE_SERVER = _conv(
    "prov:reference-edge-server",
    "Reference edge server (1 TFLOP/s, 128 GB) for pedagogy",
)

ESP32_S3 = _ds(
    "prov:esp32-s3-datasheet",
    "Espressif ESP32-S3 technical reference manual",
    "https://www.espressif.com/en/products/socs/esp32-s3",
)

NORDIC_NRF52840 = _ds(
    "prov:nordic-nrf52840-datasheet",
    "Nordic nRF52840 product specification (MLPerf Tiny reference MCU)",
    "https://www.nordicsemi.com/Products/Development-hardware/Other/nRF52840",
)

HIMAX_WE1 = _ds(
    "prov:himax-we1-plus",
    "Himax WE-I Plus Edge AI platform documentation",
    "https://www.himax.com.tw/products/edge-ai-platform/we-plus/",
)

# --- Models ---
RADFOR_GPT2 = _lit(
    "prov:radford-gpt2-2019",
    "Radford et al. (2019), Language Models are Unsupervised Multitask Learners",
    url="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
)

DEVLIN_BERT = _lit(
    "prov:devlin-bert-2019",
    "Devlin et al. (2019), BERT: Pre-training of Deep Bidirectional Transformers",
    url="https://arxiv.org/abs/1810.04805",
)

META_LLAMA = _ds(
    "prov:meta-llama-model-card",
    "Meta Llama model card / Hugging Face model documentation",
    "https://github.com/meta-llama/llama-models",
)

HE_RESNET = _lit(
    "prov:he-resnet-2016",
    "He et al. (2016), Deep Residual Learning for Image Recognition",
    url="https://arxiv.org/abs/1512.03385",
)

SANDLER_MOBILENETV2 = _lit(
    "prov:sandler-mobilenetv2-2018",
    "Sandler et al. (2018), MobileNetV2: Inverted Residuals and Linear Bottlenecks",
    url="https://arxiv.org/abs/1801.04381",
)
MOBILENETV2_WITH_ENERGY_ANCHOR = Provenance(
    id="prov:mobilenetv2-with-energy-anchor",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Sandler et al. (2018) MobileNetV2 architecture with an illustrative book energy anchor",
    url="https://arxiv.org/abs/1801.04381",
    verified="2026-08-11",
    notes="Alpha=1.0, 1000-class ImageNet classifier. The paper reports about 300M multiply-adds, represented here as 600 MFLOP under the book's convention that one multiply-accumulate is two FLOPs. The 0.1 mJ inference-energy value is an illustrative book anchor, not a measurement reported by Sandler et al. (2018).",
)

YOLOV8 = _ds(
    "prov:ultralytics-yolov8",
    "Ultralytics YOLOv8 documentation (nano variant parameter count)",
    "https://docs.ultralytics.com/models/yolov8/",
)

Krizhevsky_ALEXNET = _lit(
    "prov:krizhevsky-alexnet-2012",
    "Krizhevsky et al. (2012), ImageNet Classification with Deep Convolutional Neural Networks",
    url="https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks",
)

MLPERF_TINY_KWS = _lit(
    "prov:mlperf-tiny-kws",
    "MLPerf Tiny keyword spotting reference (DS-CNN profile)",
    url="https://github.com/mlcommons/tiny",
    notes="Parameter and FLOP counts aligned to TinyMLPerf KWS reference model scale.",
)

MLPERF_TRAINING_V30_RESNET50_A100 = Provenance(
    id="prov:mlperf-training-v30-resnet50-a100",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="MLPerf Training v3.0 ResNet-50 results, A100 class systems",
    url="https://mlcommons.org/benchmarks/training/",
    verified="2026-05-31",
    notes="Used as a broad single-accelerator throughput sanity anchor for ResNet-50/A100 teaching examples.",
)

MLPERF_TRAINING_V31_RESNET50_H100 = Provenance(
    id="prov:mlperf-training-v31-resnet50-h100",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="MLPerf Training v3.1 ResNet-50 results, H100 class systems",
    url="https://mlcommons.org/benchmarks/training/",
    verified="2026-05-31",
    notes="Used as a broad single-accelerator throughput sanity anchor for ResNet-50/H100 teaching examples.",
)

NVIDIA_NIM_LLAMA3_8B_H100 = Provenance(
    id="prov:nvidia-nim-llama3-8b-h100",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="NVIDIA NIM LLM benchmarking results for Llama 3.x 8B on H100",
    url="https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/performance.html",
    verified="2026-05-31",
    notes="Used as a broad H100 Llama-family ITL sanity range; exact serving latency depends on engine, prompt shape, batching, and precision.",
)

BROWN_GPT3_2020 = _lit(
    "prov:gpt3-brown-2020",
    "Brown et al. (2020), Language Models are Few-Shot Learners",
    url="https://arxiv.org/abs/2005.14165",
)

IMAGENET_DATASET = _lit(
    "prov:imagenet-ilsvrc-2015",
    "Russakovsky et al. (2015), ImageNet Large Scale Visual Recognition Challenge",
    url="https://arxiv.org/abs/1409.0575",
)

MSWC_DATASET = _lit(
    "prov:mswc-2021",
    "Mazumder et al. (2021), Multilingual Spoken Words Corpus",
    url="https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/fe131d7f5a6b38b23cc967316c13dae2-Abstract-round2.html",
)

CIFAR10_DATASET = _lit(
    "prov:cifar10-2009",
    "Krizhevsky (2009), Learning Multiple Layers of Features from Tiny Images",
    url="https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf",
)

MNIST_DATASET = _lit(
    "prov:mnist-1998",
    "LeCun et al. (1998), Gradient-Based Learning Applied to Document Recognition",
    url="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf",
)

OPEN_X_EMBODIMENT_DATASET = _lit(
    "prov:open-x-embodiment-2023",
    "Open X-Embodiment Collaboration et al. (2023), Open X-Embodiment: Robotic Learning Datasets and RT-X Models",
    url="https://arxiv.org/abs/2310.08864",
    verified="2026-09-18",
    notes="1M+ robot trajectories across 22 embodiments and 527 tasks.",
)

DROID_DATASET = _lit(
    "prov:droid-dataset-2024",
    "Khazatsky et al. (2024), DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset",
    url="https://arxiv.org/abs/2403.12945",
    verified="2026-09-18",
    notes="76k demonstration trajectories, 350 hours of Franka interaction data across diverse household and industrial environments.",
)

BRIDGE_DATA_V2_DATASET = _lit(
    "prov:bridge-data-v2-2023",
    "Walke et al. (2023), BridgeData V2: A Dataset for Robot Manipulation at Scale",
    url="https://arxiv.org/abs/2308.08451",
    verified="2026-09-18",
    notes="60k trajectories across 24 environments for visual-motor skill learning with WidowX 250.",
)

ALOHA_BIMANUAL_DATASET = _lit(
    "prov:aloha-bimanual-dataset-2023",
    "Zhao et al. (2023), Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware",
    url="https://arxiv.org/abs/2304.13705",
    verified="2026-09-18",
    notes="Fine-grained bimanual teleoperated demonstration dataset for insertion, threading, and slotting.",
)

DEPLOYMENT_ENVELOPES = _est(
    "prov:mlsysim-deployment-envelopes",
    "MLSysIM deployment envelope defaults for cloud, edge, mobile, and TinyML systems",
    notes="Pedagogical order-of-magnitude envelopes used for first-pass deployment reasoning; not vendor SLA targets.",
)

SWE_BENCH_HARNESS = Provenance(
    id="prov:swe-bench-harness",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref=(
        "Illustrative SWE-bench-style coding agent profile; benchmark from "
        "Jimenez et al. (2024), SWE-bench: Can Language Models Resolve "
        "Real-World GitHub Issues? (ICLR 2024)"
    ),
    url="https://arxiv.org/abs/2310.06770",
    verified="2026-09-15",
    notes=(
        "The paper defines the benchmark only. Its full text does not contain "
        "the 128,000-token context_window, 32,000-token working_memory, 5 ms "
        "sandbox_startup_latency, Llama 3 70B reference model, or DGX H100 "
        "serving node; it evaluates models such as gpt-4-32k and Claude 2 at "
        "their own context limits. These values are MLSysIM teaching "
        "assumptions."
    ),
)

FIRECRACKER_MICROVM = _lit(
    "prov:firecracker-microvm",
    "Agache et al. (2020), Firecracker: Lightweight Virtualization for Serverless Applications (NSDI '20)",
    url="https://www.usenix.org/conference/nsdi20/presentation/agache",
    notes="Sub-5 ms CoW fork latency, <5 MB memory overhead per jail, deterministic CPU/memory boundary.",
)

TEST_TIME_DELIBERATION = Provenance(
    id="prov:test-time-deliberation",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref=(
        "Illustrative tree-search deliberation agent profile; test-time compute "
        "scaling from Snell et al. (2024), Scaling LLM Test-Time Compute "
        "Optimally can be More Effective than Scaling Model Parameters"
    ),
    url="https://arxiv.org/abs/2408.03314",
    verified="2026-09-15",
    notes=(
        "The paper studies best-of-N, beam search, and lookahead search guided "
        "by a process reward model, using PaLM 2-S* on the MATH benchmark. Its "
        "full text does not contain the 64,000-token context_window, "
        "16,000-token working_memory, 8 deliberation_branches, 0.15 "
        "verifier_cost_ratio, or an MCTS agent. These values are MLSysIM "
        "teaching assumptions."
    ),
)

MULTI_AGENT_ORCHESTRATION = Provenance(
    id="prov:multi-agent-orchestration",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref=(
        "Illustrative supervisor-worker multi-agent fleet profile; orchestration "
        "frameworks from Hong et al. (2023), MetaGPT, and Wu et al. (2023), "
        "AutoGen (arXiv:2308.08155)"
    ),
    url="https://arxiv.org/abs/2308.00352",
    verified="2026-09-15",
    notes=(
        "Neither paper's full text contains the 0.03 "
        "coordination_overhead_beta, the 32,000-token context_window, or the "
        "8,000-token working_memory, and neither defines a linear coordination "
        "coefficient. These values are MLSysIM teaching assumptions."
    ),
)

STREAMING_VOICE_AGENT_PROFILE = Provenance(
    id="prov:streaming-voice-agent-profile",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Illustrative real-time streaming voice agent profile",
    verified="2026-09-15",
    notes=(
        "The 8,000-token context_window, 2,000-token working_memory, and 0 ms "
        "sandbox_startup_latency are MLSysIM teaching assumptions with no "
        "published source."
    ),
)

MLOPS_DRIFT_THRESHOLDS = _conv(
    "prov:mlops-drift-threshold-conventions",
    "Common PSI and two-sample Kolmogorov-Smirnov drift-threshold conventions",
    notes="PSI bands and KS coefficient are common monitoring defaults; production thresholds should be calibrated per application.",
)

MEMORY_SOFT_ERROR_RATE = _est(
    "prov:memory-soft-error-rate-estimate",
    "Order-of-magnitude memory soft-error bit rate for reliability and monitoring examples",
    notes="Used as a teaching-scale sanity anchor rather than a device-specific FIT rate.",
)

HBM_SOFT_ERROR_FIT_PER_MBIT = _est(
    "prov:hbm-soft-error-fit-per-mbit",
    "Unprotected HBM soft-error rate, low end of the published 200-5000 FIT/Mbit DRAM range",
    notes=(
        "Teaching-scale figure (250 FIT/Mbit) at the low end of the 200-5000 FIT/Mbit "
        "DRAM soft-error range reported in the soft-error literature (Tezzaron, 'Soft "
        "Errors in Electronic Memory'; en.wikipedia.org/wiki/Soft_error). Motivates why "
        "unprotected HBM at fleet scale mandates ECC; not a device-specific datasheet value."
    ),
    url="https://tezzaron.com/media/soft_errors_1_1_secure.pdf",
)

FHE_OVERHEAD = _lit(
    "prov:fhe-overhead-slowdown",
    "Fully homomorphic encryption compute overhead, 2-6 orders of magnitude vs plaintext",
    url="https://www.math-lock.com/benchmarks.html",
    notes=(
        "General-purpose FHE libraries (HElib, PALISADE) run 1e4-1e6x slower than "
        "plaintext; the book uses 1e4x (10,000x) as a conservative low-end teaching figure."
    ),
)

TEE_HARDWARE_SPECS = _est(
    "prov:tee-hardware-specs",
    "Trusted-execution-environment overheads (Intel SGX, ARM TrustZone) from vendor documentation",
    notes=(
        "SGX enclave page cache ~128 MB with ~100x paging penalty on overflow and 15-30 us "
        "enclave transitions; TrustZone world switch ~300-1000 cycles and 15-30% secure-mode "
        "power; mTLS handshake 15-30 ms. Order-of-magnitude figures from Intel SGX / ARM "
        "TrustZone documentation and security-engineering literature."
    ),
)

HSM_GPU_CRYPTO = _est(
    "prov:hsm-gpu-crypto-throughput",
    "HSM vs GPU RSA-2048 throughput and unit cost from security-engineering practice",
    notes=(
        "Enterprise HSMs ~10,000 RSA-2048 ops/s at $20k-$100k/unit; general-purpose GPUs "
        "~100,000 ops/s at ~$1k; the ~10x throughput gap is the tamper-resistance tax."
    ),
)

RESPONSIBLE_AI_OVERHEAD = _est(
    "prov:responsible-ai-overhead-benchmarks",
    "Responsible-AI technique overheads (accuracy, training, inference, memory) across published benchmarks",
    notes=(
        "Synthesis of reported overheads for DP-SGD, fairness-aware training, SHAP/LIME "
        "explainability, adversarial training, and federated learning. Ranges are "
        "order-of-magnitude empirical findings across the responsible-AI efficiency "
        "literature, not vendor specifications."
    ),
)

ORCHESTRATION_ASSUMPTIONS = _est(
    "prov:mlsysim-orchestration-assumptions",
    "MLSysIM cluster orchestration defaults for utilization, queue discipline, and job duration",
    notes="First-pass scheduler assumptions for analytical examples; production clusters should calibrate from trace data.",
)

FLEET_EVOLUTION_HEURISTIC = Provenance(
    id="prov:fleet-evolution-heuristic",
    kind=ProvenanceKind.HEURISTIC,
    ref="MLSysIM fleet-evolution heuristic for conclusion synthesis",
    notes=(
        "Pedagogical multiplicative gain profile used to reason about how "
        "hardware, algorithmic, and orchestration improvements combine. The "
        "numbers are scenario anchors, not forecasts."
    ),
    verified="2026-06-06",
)

CRITICAL_BATCH_SIZE_ESTIMATES = _lit(
    "prov:mccandlish-critical-batch-2018",
    "McCandlish et al. (2018), An Empirical Model of Large-Batch Training",
    url="https://arxiv.org/abs/1812.06162",
    notes="Order-of-magnitude critical batch size anchors; model-specific entries are rounded for analytical examples.",
)

WAKE_VISION = _ds(
    "prov:wake-vision-dataset",
    "Wake Vision / doorbell-classifier TinyML reference",
    "https://github.com/harvard-edge/Wake_Vision",
)

REFERENCE_ANOMALY_MLP = _conv(
    "prov:reference-anomaly-mlp",
    "TinyML anomaly-detector reference MLP (~270k parameters)",
)

NAUMOV_DLRM = _lit(
    "prov:naumov-dlrm-2019",
    "Naumov et al. (2019), Deep Learning Recommendation Model (DLRM)",
    url="https://arxiv.org/abs/1906.00091",
)

GU_GUARDRAILS_MAMBA = _lit(
    "prov:gu-mamba-2023",
    "Gu & Dao (2023), Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
    url="https://arxiv.org/abs/2312.00752",
)

ROMBACH_STABLE_DIFFUSION = _lit(
    "prov:rombach-stable-diffusion-2022",
    "Rombach et al. (2022), High-Resolution Image Synthesis with Latent Diffusion Models",
    url="https://arxiv.org/abs/2112.10752",
)

CHINCHILLA = _lit(
    "prov:hoffmann-chinchilla-2022",
    "Hoffmann et al. (2022), Training Compute-Optimal Large Language Models",
    url="https://arxiv.org/abs/2203.15556",
)

PALM_MFU = _lit(
    "prov:chowdhery-palm-2022",
    "Chowdhery et al. (2022), PaLM: Scaling Language Modeling with Pathways",
    url="https://arxiv.org/abs/2204.02311",
)

POPE_INFERENCE = _lit(
    "prov:pope-inference-2023",
    "Pope et al. (2023), Efficiently Scaling Transformer Inference",
    url="https://proceedings.mlsys.org/paper_files/paper/2023/hash/c4be71ab8d24cdfb45e3d06dbfca2780-Abstract-mlsys2023.html",
)

MEGASCALE = _lit(
    "prov:jiang-megascale-2024",
    "Jiang et al. (2024), MegaScale: Scaling Large Language Model Training",
    url="https://arxiv.org/abs/2402.15627",
)

GENDER_SHADES = _lit(
    "prov:buolamwini-gendershades-2018",
    "Buolamwini & Gebru (2018), Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification",
    url="https://proceedings.mlr.press/v81/buolamwini18a.html",
)

CROWDFLOWER_2016 = _lit(
    "prov:crowdflower-data-science-report-2016",
    "CrowdFlower (2016), Data Science Report — 'What data scientists spend the most time doing'",
    url="https://visit.figure-eight.com/data-science-report.html",
)

GPT2_TRAINING_COST_EST = _est(
    "prov:gpt2-training-cost-estimate",
    "Community estimate of GPT-2 (1.5B) cloud training cost in 2019 (~$50K)",
    notes="Order-of-magnitude anchor for the training-vs-inference cost asymmetry; OpenAI did not disclose a figure. (Added 2026-06-07 when the cost moved out of a chapter-local literal.)",
)

GPT4_TRAINING_COST_EST = _est(
    "prov:gpt4-training-cost-estimate",
    "Industry-reported nine-figure estimate for GPT-4-class training (Altman: 'more than $100 million', Wired, 2023)",
    url="https://www.wired.com/story/openai-ceo-sam-altman-the-age-of-giant-ai-models-is-already-over/",
    notes="Pedagogical anchor; OpenAI has not disclosed the exact training cost. (Added 2026-06-07 when the cost moved out of a chapter-local literal.)",
)

MEGATRON_OVERLAP = _lit(
    "prov:shoeybi-megatron-2019",
    "Shoeybi et al. (2019), Megatron-LM: Training Multi-Billion Parameter Language Models",
    url="https://arxiv.org/abs/1909.08053",
)

SCALING_EFFICIENCY_RULE_OF_THUMB = _conv(
    "prov:scaling-efficiency-rule-of-thumb",
    "Common industry rule-of-thumb (~90% parallel efficiency on well-tuned clusters)",
)

DGX_GPUS_PER_HOST = _conv(
    "prov:dgx-gpus-per-host",
    "NVIDIA DGX H100/H200 node envelope (8 GPUs per host)",
    notes="Used for cluster tier node counts in fleet appendices.",
)

DGX_H100_SYSTEM_SPEC = _ds(
    "prov:nvidia-dgx-h100-system-spec",
    "NVIDIA DGX H100 System Datasheet and Architecture Whitepaper",
    url="https://resources.nvidia.com/en-us-dgx-systems/dgx-h100-datasheet",
    notes=(
        "Official NVIDIA DGX H100 chassis specification: 8x H100 80GB SXM5 GPUs, "
        "Dual Intel Xeon Platinum 8480C processors (112 cores / 224 threads total), "
        "2 TB DDR5-4800 RAM across 16 channels, 8x ConnectX-7 400 Gbps InfiniBand OSFP ports, "
        "30.72 TB internal NVMe U.2 storage in RAID-0."
    ),
)

HGX_H100_EPYC_SYSTEM_SPEC = _ds(
    "prov:supermicro-hgx-h100-epyc-spec",
    "Supermicro AS-8125GS-TNHR / Dell PowerEdge XE9680 HGX H100 8-GPU Datasheet",
    url="https://www.supermicro.com/en/products/system/gpu/8u/as-8125gs-tnhr",
    notes=(
        "High-density agent evaluation node: 8x H100 80GB SXM5 GPUs, Dual AMD EPYC 9654 "
        "processors (192 physical Zen 4 cores / 384 threads total), 1.5 TB DDR5-4800 RAM "
        "across 24 memory channels (460 GB/s sustained read bandwidth), PCIe Gen5 NVMe arrays."
    ),
)

APPLE_M3_MAX_WORKSTATION_SPEC = _ds(
    "prov:apple-m3-max-workstation-spec",
    "Apple MacBook Pro 16-inch M3 Max Technical Specifications",
    url="https://www.apple.com/macbook-pro/specs/",
    notes=(
        "Apple Silicon unified memory developer baseline: 16-core CPU (12 Performance + 4 Efficiency), "
        "40-core GPU, 128 GiB unified LPDDR5X at 400 GB/s shared between CPU and GPU."
    ),
)

GIBIANSKY_ALLREDUCE = _lit(
    "prov:gibiansky-allreduce-factor",
    "Gibiansky (2017), Ring AllReduce communication identity (2× factor)",
    url="https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/",
)

INFINIBAND_SPEC = _ds(
    "prov:infiniband-trade-association-spec",
    "InfiniBand Trade Association link-speed specifications",
    "https://www.infinibandta.org/",
)

INFINIBAND_NDR_GBS = Provenance(
    id="prov:infiniband-ndr-gbs-derived",
    kind=ProvenanceKind.DERIVED,
    ref="InfiniBand NDR 400 Gbps per port → 50 GB/s",
    url="https://www.infinibandta.org/",
    notes="Byte rate = line rate ÷ 8.",
    verified="2026-03-06",
)

INFINIBAND_HDR_GBS = Provenance(
    id="prov:infiniband-hdr-gbs-derived",
    kind=ProvenanceKind.DERIVED,
    ref="InfiniBand HDR 200 Gbps per port → 25 GB/s",
    url="https://www.infinibandta.org/",
    notes="Byte rate = line rate ÷ 8.",
    verified="2026-03-06",
)

INFINIBAND_XDR_GBS = Provenance(
    id="prov:infiniband-xdr-gbs-derived",
    kind=ProvenanceKind.DERIVED,
    ref="InfiniBand XDR 800 Gbps per port → 100 GB/s",
    url="https://www.infinibandta.org/",
    notes="Byte rate = line rate ÷ 8 (2025 generation).",
    verified="2026-03-06",
)

ETHERNET_400G_GBS = Provenance(
    id="prov:ethernet-400g-gbs-derived",
    kind=ProvenanceKind.DERIVED,
    ref="400 GbE → 50 GB/s",
    notes="Byte rate = 400 Gb/s ÷ 8.",
    verified="2026-03-06",
)

ETHERNET_800G_GBS = Provenance(
    id="prov:ethernet-800g-gbs-derived",
    kind=ProvenanceKind.DERIVED,
    ref="800 GbE → 100 GB/s",
    notes="Byte rate = 800 Gb/s ÷ 8.",
    verified="2026-03-06",
)

ROCE_100G_GBS = Provenance(
    id="prov:roce-100g-gbs-derived",
    kind=ProvenanceKind.DERIVED,
    ref="100 GbE RoCE → 12.5 GB/s",
    notes="Byte rate = 100 Gb/s ÷ 8.",
    verified="2026-03-06",
)

FABRIC_LATENCY_ASSUMPTIONS = _conv(
    "prov:fabric-latency-assumptions",
    "MLSysIM α-model one-way latency anchors (InfiniBand NDR/HDR, RoCE, TCP)",
    notes="Order-of-magnitude μs values for napkin math; not vendor QoS guarantees.",
)

SWITCH_OPTICS_REFERENCE = _conv(
    "prov:switch-optics-reference",
    "Datacenter switch-ASIC capacity (51.2T/102.4T) and 400G optics power (pluggable/CPO) reference figures",
    notes="2025-26 datacenter-switching reference points for network-fabric sizing analyses.",
)

NVIDIA_QUANTUM2_QM97XX_SWITCH = _ds(
    "prov:nvidia-quantum2-qm97xx-switch",
    "NVIDIA QM97XX 1U NDR 400Gbps InfiniBand Switch Systems User Manual",
    "https://docs.nvidia.com/networking/display/qm97x0um/introduction",
    verified="2026-06-01",
    notes="QM9700/QM9701/QM9790-class switches: 64 NDR 400 Gb/s ports and 51.2 Tb/s aggregate bidirectional throughput.",
)

NDR_LEAF_SPINE_64_PORT_SPLIT = _conv(
    "prov:ndr-leaf-spine-64-port-split",
    "Reference non-oversubscribed two-tier leaf-spine split for a 64-port NDR switch",
    notes="Uses 32 endpoint downlinks and 32 spine uplinks per leaf switch.",
)

NETWORK_ENERGY_ANCHORS = _conv(
    "prov:network-energy-anchors",
    "Network data-transfer energy anchors (5G per-MB, generic per-KB)",
    notes="Order-of-magnitude transfer-energy figures for intuition; not measured device values.",
)

RECOVERY_TIME_ASSUMPTIONS = _conv(
    "prov:recovery-time-assumptions",
    "Fleet recovery design assumptions (heartbeat, reschedule, checkpoint BW)",
    notes="Engineering targets for reliability analyses; calibrate from cluster traces when available.",
)

OVERHEAD_BUDGETS = _conv(
    "prov:overhead-budgets",
    "Combined overhead budgets (pipeline, checkpoint, failure, maintenance)",
    notes="Fractions of wall time for 10k+ GPU training scenarios.",
)

SCALING_EFFICIENCY_TIERS = _conv(
    "prov:scaling-efficiency-tiers",
    "Illustrative scaling efficiency vs GPU count (32→1024 GPUs)",
    notes="8192-GPU tier uses MEGASCALE literature anchor separately.",
)

ENERGY_HIERARCHY_CONVENTIONS = _conv(
    "prov:energy-hierarchy-conventions",
    "Simplified energy hierarchy: architecture-class effective pJ/FLOP (CPU→ASIC) and per-byte data-movement cost (register→network)",
    notes="Order-of-magnitude teaching figures; effective system-level energy, consistent with the Horowitz (2014) energy trend.",
)

LPDDR5_6400_TIMINGS = _ds(
    "prov:lpddr5-6400-timings",
    "JEDEC JESD209-5C LPDDR5-6400 timings as encoded in the Ramulator 2.0 "
    "LPDDR5_6400 preset (tCK = 1250 ps; nRCD = 15, nRP = 15, nRPab = 17 cycles)",
    "https://github.com/CMU-SAFARI/ramulator2/blob/main/python/ramulator/dram/lpddr5.py",
    verified="2026-09-29",
    notes=(
        "Checked at ramulator2 main 72427a1bba (2026-09-02). BG mode, CKR 4:1: "
        "t_RCD = t_RP(per bank) = 15 x 1.25 ns = 18.75 ns. The JEDEC standard is "
        "not read directly (jedec.org access), so the values rest on the "
        "simulator's encoding of JESD209-5C Table 339/340."
    ),
)

MEMORY_INTERFACE_BANDWIDTH_TIERS = _conv(
    "prov:memory-interface-bandwidth-tiers",
    "Representative memory-interface bandwidth tiers",
    notes=(
        "Technology-class bandwidth anchors for DDR4-3200, HBM2, HBM3, and "
        "GDDR6X used in memory-protection and hierarchy examples."
    ),
)

MEMORY_PROTECTION_OVERHEADS = _conv(
    "prov:memory-protection-overheads",
    "Memory protection overhead conventions",
    notes=(
        "SECDED-style ECC reserve is represented as 12.5 percent parity overhead; "
        "no-ECC profile is included for comparative GDDR-style examples."
    ),
)

WUE_ANCHORS = _conv(
    "prov:wue-anchors",
    "Water-usage effectiveness (WUE) tiers for sustainability examples",
)

RACK_POWER_TIERS = _conv(
    "prov:rack-power-tiers",
    "Rack power tiers (traditional vs AI cluster, air-cooling limit)",
)

DGX_H100_RACK_REFERENCE = _conv(
    "prov:dgx-h100-rack-reference",
    "Reference DGX H100 rack profile",
    notes=(
        "Four DGX H100 nodes per rack, yielding 32 H100 GPUs per rack for rack-level power "
        "and cooling models; representative non-accelerator rack support load is 11.1 kW "
        "(host CPUs/DRAM, NVSwitch, InfiniBand, power conversion, and cooling overhead)."
    ),
)

STORAGE_TRAINING_CORPUS_REFERENCE = _conv(
    "prov:storage-training-corpus-reference",
    "Reference 175B-model storage running example",
    notes=(
        "Chapter-level storage scenario anchor: 1.5T training tokens, 3 TB compressed "
        "source corpus, 4-byte token IDs, and 14 bytes/parameter resumable mixed-precision "
        "Adam checkpoint storage (FP16 weights, FP32 master weights, and two FP32 moments)."
    ),
)

MODEL_LOADING_SCENARIO_ASSUMPTIONS = _est(
    "prov:model-loading-scenario-assumptions",
    "Reference model-loading scenario assumptions",
    notes=(
        "Representative cold-start loading anchors for serialized model checkpoints; "
        "values are teaching-scale assumptions and should be calibrated from fleet "
        "measurements for production sizing."
    ),
)

LLM_SERVING_PRECISION_DIVIDEND_PROFILE = _est(
    "prov:llm-serving-precision-dividend-profile",
    "Reference LLM serving precision-dividend profile",
    notes=(
        "Reusable deployment-shape assumptions for the performance-engineering "
        "70B LLM KV-cache and batch-size example."
    ),
)

HETEROGENEOUS_ROUTING_SCENARIO = _est(
    "prov:heterogeneous-routing-scenario",
    "Reference heterogeneous GPU routing scenario",
    notes=(
        "Reusable H100/A100 server counts, per-server service rates, and traffic "
        "target for the inference weighted-routing example."
    ),
)

CIRCUIT_BREAKER_SERVING_PROFILE = _est(
    "prov:circuit-breaker-serving-profile",
    "Reference GPU inference circuit-breaker profile",
    notes=(
        "Reusable threshold and recovery-window assumptions for teaching "
        "GPU inference circuit-breaker behavior; production thresholds should "
        "be calibrated from service SLOs and fleet telemetry."
    ),
)

CHECKPOINT_ARCHETYPE_SCENARIO_ASSUMPTIONS = _est(
    "prov:checkpoint-archetype-scenario-assumptions",
    "Reference checkpoint-size archetype assumptions",
    notes=(
        "Representative checkpoint footprints and bytes-per-parameter policy used "
        "to compare fault-tolerance overhead across model families."
    ),
)

CLOUD_PRICING_2024 = Provenance(
    id="prov:cloud-pricing-2024",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Illustrative US cloud list prices (2024–2025 order of magnitude)",
    notes="GPU-hour, egress, and electricity rate anchors; not a specific vendor quote.",
    verified="2026-03-06",
)

STORAGE_PRICING_2024 = Provenance(
    id="prov:storage-pricing-2024",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Illustrative cloud/object-storage list prices (2024 order of magnitude)",
    notes="S3, Glacier, and NVMe tier rate anchors for data-engineering scenarios.",
    verified="2026-03-06",
)

LABELING_PRICING_2024 = Provenance(
    id="prov:labeling-pricing-2024",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Illustrative data-labeling cost ranges (2024 estimates)",
    notes="Crowd, bounding-box, and medical labeling tiers for workflow examples.",
    verified="2026-03-06",
)

FLEET_ECONOMICS_2024 = Provenance(
    id="prov:fleet-economics-2024",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Illustrative internal GPU-hour and chargeback rates (2024)",
    notes="On-demand, spot, and internal chargeback references for fleet orchestration examples.",
    verified="2026-03-06",
)

BARROSO_DATACENTER_ECONOMICS = _lit(
    "prov:barroso-datacenter-economics",
    "Barroso et al. (2018), The Datacenter as a Computer (3rd ed.)",
    url="https://doi.org/10.1007/978-3-031-01761-2",
)

CAPACITY_LEAD_TIMES = _est(
    "prov:capacity-lead-times",
    "Illustrative datacenter build-out lead times",
    notes="Order-of-magnitude planning anchors for compute-infrastructure analyses.",
)

CARBON_PER_GPU_HR = _est(
    "prov:carbon-per-gpu-hr",
    "Illustrative per-GPU-hour carbon proxy for responsible-AI examples",
    notes="0.16 kg/GPU-hr order-of-magnitude; not a grid-specific intensity calculation.",
)

LIT_TRANSATLANTIC_ROUND_TRIP_CO2 = _est(
    "prov:lit-transatlantic-round-trip-co2",
    "Aviation CO2e factors for long-haul economy passenger travel (DEFRA-class)",
    notes=(
        "Rounded to 1000 kg CO2e for the NY-London round-trip reference anchor; "
        "agency calculators typically report about 900-1100 kg CO2e per economy passenger."
    ),
    verified="2026-05-31",
)

MFU_INFERENCE_BATCHED_LIT = _lit(
    "prov:mfu-inference-batched",
    "Pope et al. (2023); batched inference MFU upper illustrative bound",
    url="https://proceedings.mlsys.org/paper_files/paper/2023/hash/c4be71ab8d24cdfb45e3d06dbfca2780-Abstract-mlsys2023.html",
    notes="0.40 is an upper illustrative bound for large-batch inference, not batch-1.",
)

PHYSICAL_AI_NUMBERS = _est(
    "prov:physical-ai-numbers",
    "Physical AI systems rate hierarchy and engineering orders of magnitude (Volume IV)",
    notes=(
        "Standard multi-rate execution bands across deliberative intent (1-5 Hz), "
        "reactive chunk policy (10-50 Hz), safety permission path (1 kHz), and "
        "motor field-oriented current control (10-25 kHz); kinetic blind travel "
        "conversion (1 mm/ms at 1 m/s; 1 cm/ms at 10 m/s; 3 cm/ms at 30 m/s); "
        "edge VLA memory streaming and action chunk amortization."
    ),
    verified="2026-09-22",
)

ISO_26262_ASIL_D = _ds(
    "prov:iso-26262-asil-d",
    "ISO 26262-5:2018 Road vehicles - Functional safety - Part 5: Product development at the hardware level",
    "https://www.iso.org/standard/68387.html",
    verified="2026-09-22",
    notes="ASIL D hardware target probabilistic metric for random hardware failures (PMHF) < 10 FIT (10^-8 / hour).",
)

IEC_61508_SIL_3 = _ds(
    "prov:iec-61508-sil-3",
    "IEC 61508-1:2010 Functional safety of electrical/electronic/programmable electronic safety-related systems",
    "https://webstore.iec.ch/publication/5515",
    verified="2026-09-22",
    notes="SIL 3 high demand / continuous mode probability of dangerous failure per hour (PFH) target 10^-8 to 10^-7 / hour; matches ISO 13849 PL e.",
)

RULE_OF_THREE_SAFETY = _lit(
    "prov:rule-of-three-safety",
    "Hanley, J. A., & Lippman-Hand, A. (1983). If nothing goes wrong, is everything all right? Interpreting zero numerators. JAMA, 249(13), 1743-1745.",
    url="https://doi.org/10.1001/jama.1983.03330370053031",
    verified="2026-09-22",
    notes="Rule of Three: 95% upper confidence bound for a Poisson event rate with zero observed occurrences in n trials is approximately 3/n.",
)

ETHERCAT_STANDARD = _ds(
    "prov:ethercat-standard",
    "ETG.1000 EtherCAT Specification, EtherCAT Technology Group",
    "https://www.ethercat.org",
    verified="2026-09-22",
    notes="Deterministic industrial Ethernet fieldbus with hardware cut-through forwarding, cycle jitter < 1 us, typical cycle times 100 us to 1 ms.",
)

CAN_FD_SPECIFICATION = _ds(
    "prov:can-fd-specification",
    "ISO 11898-1:2015 Road vehicles - Controller area network (CAN) - Part 1: Data link layer and physical signalling",
    "https://www.iso.org/standard/63648.html",
    verified="2026-09-22",
    notes="CAN with Flexible Data-Rate (CAN-FD) up to 5-8 Mbps payload bit rate; non-preemptive arbitration introduces frame queueing and blocking jitter.",
)

THERMAL_CONSTANTS_ELECTROMECHANICAL = _est(
    "prov:thermal-constants-electromechanical",
    "Representative thermal time constants for robotic electromechanical drive trains and compute silicon",
    notes=(
        "Adiabatic silicon junction heating tau ~ 1-10 ms; heat sink convective dissipation "
        "tau ~ 10-60 s; motor stator copper winding Joule heating tau ~ 30-180 s; "
        "motor casing/frame bulk thermal mass tau ~ 10-30 min; lithium-ion battery pack "
        "tau ~ 5-20 min. Values reflect typical orders of magnitude across industrial and mobile robotics."
    ),
    verified="2026-09-22",
)


# ---------------------------------------------------------------------------
# Drivetrain, drive bus, and arm-joint sources (added 2026-09-28, vol4 M6).
# ---------------------------------------------------------------------------

ZLTECH_ZLLG10ASM800 = _ds(
    "prov:zltech-zllg10asm800",
    "Shenzhen Zhongling Technology (ZLTECH), ZLLG10ASM800-48 V1.0 robot hub motor "
    "drawing and motor parameters (V1.00, 2022-09-26)",
    "https://www.zlingkj.com/Data/zlingkj/upload/file/20240102/ZLLG10ASM800%20V1.0%E5%8F%8C%E5%87%BA%E8%BD%B4.pdf",
    verified="2026-09-28",
    notes=(
        "10-inch direct-drive BLDC hub motor for AGV/AMR drive wheels, no gearbox. "
        "Motor-parameter table: outer (wheel) diameter 266 +/- 2 mm; 48 VDC rated; "
        "20 N m rated torque at 8.5 A; 60 N m peak torque at 25 A peak current; "
        "150 rpm rated, 180 rpm peak speed; 20 pole pairs; line resistance "
        "0.7 ohm +/- 10 percent; line inductance 1.885 mH +/- 10 percent; back-EMF "
        "constant 0.262 V/rpm +/- 10 percent; 9.5 kg; maximum load 300 kg per two "
        "motors; ambient -20 to +40 C. The table does not publish rotor inertia "
        "(needs source). Torque constant is derived as 60 N m / 25 A = 2.4 N m/A. "
        "Cross-check: 0.262 V/rpm x 180 rpm = 47.2 V, the rated bus, so the "
        "back-EMF constant is a line-to-line value compared directly with the DC bus."
    ),
)

ZLTECH_ZLAC8030L = _ds(
    "prov:zltech-zlac8030l",
    "ZLTECH ZLAC8030L hub-servo driver product page (Shenzhen Zhongling Technology)",
    "https://www.robotmotor.com/zlac8030l-zltech-20v-60v-60a-canopen-rs485-dc-servo-motor-controller-driver-for-robot-agv-product/",
    verified="2026-09-28",
    notes=(
        "Driver matched to the ZLLG10ASM800 (its wiring table names the 8030L): "
        "24-48 VDC input, rated 30 A, maximum 60 A output current, CANopen/RS485. "
        "The driver's 60 A exceeds the motor's 25 A peak, so the motor rating sets "
        "the drive current limit. Overvoltage and undervoltage thresholds are not "
        "published (needs source)."
    ),
)

MIR250_BATTERY = _ds(
    "prov:mir250-specifications",
    "Mobile Industrial Robots, MiR250 Specifications (2025-04-14)",
    "https://ai.motion.com/wp-content/uploads/2025/10/MiR250-Specifications-2.99.pdf",
    verified="2026-09-28",
    notes=(
        "Reference 48 V-class AMR (94 kg robot, 250 kg payload, 2.0 m/s). Battery: "
        "lithium-ion, 47.7 V nominal, minimum 42 V, maximum 54 V, 1.63 kWh (34.2 Ah). "
        "Floor-to-wheel friction coefficient 0.60-0.80; clean and dry floors; design "
        "based on ISO 3691-4 (with listed exceptions) and ISO 13849-1. Used as the "
        "source of the running machine's drive-bus voltage envelope."
    ),
)

MEANWELL_DDR240 = _ds(
    "prov:meanwell-ddr-240",
    "MEAN WELL DDR-240 series 240 W DIN-rail DC-DC converter specification (DDR-240C-24, 48 V input)",
    "https://www.meanwell.com/Upload/PDF/DDR-240/DDR-240-spec.pdf",
    verified="2026-09-28",
    notes=(
        "DDR-240C-24 (48 V in, 24 V out): continuous input 33.6-67.2 VDC; 100 ms "
        "ride-through 28.8-33.6 VDC; input under-voltage lockout ON >= 33.6 V, "
        "OFF <= 33 V; typical efficiency 91 percent. Used as the 48-to-24 V converter "
        "feeding the control electronics; its 33 V turn-off is the drive bus's "
        "undervoltage threshold and its 67.2 V continuous maximum bounds the clamp."
    ),
)

TQ_ROBODRIVE_ILM = _ds(
    "prov:tq-robodrive-ilm",
    "TQ-RoboDrive ILM series frameless servo kits data sheet (Rev. 0414)",
    "https://www.tq-group.com/filedownloads/files/products/robodrive/data-sheets/en/DRVA_DB_Servo-Kits_ILM_EN_Rev0414_Web.pdf",
    verified="2026-09-28",
    notes=(
        "ILM 70x18, star-serial winding: 48 V rated, 1.25 N m rated torque, "
        "4.05 N m peak torque (20 percent deviation from linearity), 6.7 A rated "
        "current, torque constant 187 mNm/A, motor constant 0.266 N m/sqrt(W), "
        "terminal resistance 660 mOhm and inductance 1,460 uH at 20 C, rotor inertia "
        "0.321 kg cm^2, 10 pole pairs, 330 g. The rated copper loss (22.2 W) equals "
        "(3/2) I_r^2 (R_TT / 2), so the rated current is the amplitude-invariant "
        "(peak phase) current. The data sheet states that thermal behavior depends "
        "strongly on the installation, so no thermal resistance or capacitance is "
        "published (needs source). The ILM line derives from DLR drive engineering."
    ),
)

HARMONIC_DRIVE_CSG_2UH_ENGINEERING = _ds(
    "prov:harmonic-drive-csg-2uh-engineering-data",
    "Harmonic Drive AG, Engineering Data CSG-2UH Units (1019642, 12/2014), Table 10.2",
    "https://innodrive.ru/upload/iblock/387/pv21y3bafuu4i675xna8v70s8naalpa2.pdf",
    verified="2026-09-28",
    notes=(
        "Distributor-hosted copy of the Harmonic Drive AG document; the torque "
        "ratings match harmonicdrive.net's CSG-25-50-2UH product page. "
        "CSG-25-2UH, ratio 50: repeatable peak torque 127 N m, average torque 72 N m, "
        "rated torque 51 N m, momentary peak 242 N m; moment of inertia at the input "
        "0.413e-4 kg m^2; 1.5 kg. Efficiency varies with load, speed, and temperature "
        "(Section 4.4)."
    ),
)

FRANKA_CLASS_STRAIN_WAVE_JOINT = _est(
    "prov:franka-class-strain-wave-joint",
    "Panda-class strain-wave joint stand-in: Harmonic Drive CSG-25-50-2UH with a "
    "TQ-RoboDrive ILM 70x18 frameless motor",
    notes=(
        "Franka does not publish the Panda's joint motor or gear parameters; its "
        "datasheet gives 87 N m peak torque on axes 1-4 and a harmonic-drive "
        "transmission with joint torque sensing. The stand-in pairs sourced parts of "
        "that class: the CSG-25-50 (127 N m repeatable peak covers 87 N m) and the "
        "smallest ILM whose peak torque through 50:1 covers the gear's repeatable "
        "peak (ILM 70x18: 4.05 N m x 50 = 202 N m; the ILM 70x10 gives 106 N m). "
        "Rotor inertia at the motor shaft = ILM rotor 0.321e-4 + CSG-25 input "
        "0.413e-4 = 0.734e-4 kg m^2. The pairing is a choice; each value is sourced."
    ),
    verified="2026-09-28",
)


# ---------------------------------------------------------------------------
# 2026-09-28 additions (vol2 registry integration). Add-only: every record
# below is new; no existing record above was changed.
# ---------------------------------------------------------------------------
_V = "2026-09-28"


def _report(id: str, ref: str, url: str, *, notes: str | None = None) -> Provenance:
    """Creates a Provenance object for an industry / government report."""
    return Provenance(
        id=id,
        kind=ProvenanceKind.INDUSTRY_REPORT,
        ref=ref,
        url=url,
        verified=_V,
        notes=notes,
    )


# --- Training-state accounting ---
ZERO_MIXED_PRECISION_ADAM_ACCOUNTING = _lit(
    "prov:rajbhandari-zero-2020",
    "Rajbhandari et al. (2020), ZeRO: Memory Optimizations Toward Training Trillion Parameter Models",
    url="https://arxiv.org/abs/1910.02054",
    verified=_V,
    notes=(
        "Mixed-precision Adam model state = 2 (FP16 weights) + 2 (FP16 gradients) + K=12 "
        "(FP32 master weights + Adam m and v) = 16 bytes/parameter. A resumable checkpoint "
        "omits the gradients: 16 - 2 = 14 bytes/parameter."
    ),
)
PATTERSON_CARBON_2021 = _lit(
    "prov:patterson-carbon-2021",
    "Patterson et al. (2021), Carbon Emissions and Large Neural Network Training",
    url="https://arxiv.org/abs/2104.10350",
    verified=_V,
    notes=(
        "GPT-3 training (Table 4 / App. A): 10,000 V100 GPUs (NVIDIA-suggested count), "
        "24.6 TFLOP/s per GPU measured by OpenAI, ~14.8 days derived, 330 W system power "
        "per GPU, PUE 1.10, 1,287 MWh, 552.1 tCO2e (net, metric tonnes)."
    ),
)

# --- Model papers ---
DUBEY_LLAMA3_2024 = _lit(
    "prov:dubey-llama3-2024",
    "Dubey et al. (2024), The Llama 3 Herd of Models",
    url="https://arxiv.org/abs/2407.21783",
    verified=_V,
    notes=(
        "Sec. 3.3: 405B trained on up to 16K H100 GPUs; Table 4: 43% BF16 MFU on 8,192 GPUs, "
        "41% on 16,384 GPUs; Sec. 3.3.4: 54-day snapshot with 466 job interruptions "
        "(47 planned, 419 unexpected), >90% effective training time, manual intervention "
        "required only three times."
    ),
)
DOSOVITSKIY_VIT_2021 = _lit(
    "prov:dosovitskiy-vit-2021",
    "Dosovitskiy et al. (2021), An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
    url="https://arxiv.org/abs/2010.11929",
    verified=_V,
    notes="ViT-L/16: 307M parameters; 87.76% ImageNet top-1 after JFT-300M pre-training (Table 2).",
)
TAN_EFFICIENTNET_2019 = _lit(
    "prov:tan-efficientnet-2019",
    "Tan & Le (2019), EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
    url="https://arxiv.org/abs/1905.11946",
    verified=_V,
    notes="EfficientNet-B4: 19M parameters, 4.2B FLOPS (paper counts multiply-adds), 82.9% ImageNet top-1 (Table 2).",
)
HOWARD_MOBILENETV3_2019 = _lit(
    "prov:howard-mobilenetv3-2019",
    "Howard et al. (2019), Searching for MobileNetV3",
    url="https://arxiv.org/abs/1905.02244",
    verified=_V,
    notes="MobileNetV3-Large 1.0: 5.4M parameters, 219M multiply-adds, 75.2% ImageNet top-1 (Table 3).",
)

# --- Cited training runs (Literature.TrainingRuns) ---
KRIZHEVSKY_ALEXNET_2012 = _lit(
    "prov:krizhevsky-alexnet-2012",
    "Krizhevsky, Sutskever & Hinton (2012), ImageNet Classification with Deep Convolutional Neural Networks",
    url="https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html",
    verified=_V,
    notes="Trained for five to six days on two NVIDIA GTX 580 3GB GPUs.",
)
NARAYANAN_MEGATRON_2021 = _lit(
    "prov:narayanan-megatron-2021",
    "Narayanan et al. (2021), Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM",
    url="https://arxiv.org/abs/2104.04473",
    verified=_V,
    notes="1T-parameter model on 3,072 A100 GPUs at 502 PFLOP/s aggregate, 52% of theoretical peak per GPU.",
)
GOYAL_IMAGENET_1HR_2017 = _lit(
    "prov:goyal-imagenet-1hr-2017",
    "Goyal et al. (2017), Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour",
    url="https://arxiv.org/abs/1706.02677",
    verified=_V,
    notes="ResNet-50 on ImageNet in one hour on 256 GPUs with a minibatch of 8,192 images.",
)
ZHANG_OPT_2022 = _lit(
    "prov:zhang-opt-2022",
    "Zhang et al. (2022), OPT: Open Pre-trained Transformer Language Models",
    url="https://arxiv.org/abs/2205.01068",
    verified=_V,
    notes=(
        "OPT-175B trained on 992 80GB A100 GPUs at up to 147 TFLOP/s per GPU. Hardware "
        "failures caused at least 35 manual restarts and the cycling of over 100 hosts over "
        "2 months; the authors estimate 70+ additional automatic restarts (Sec. 2.5)."
    ),
)
MICROSOFT_OPENAI_SUPERCOMPUTER_2020 = _report(
    "prov:microsoft-openai-supercomputer-2020",
    "Microsoft (2020), Microsoft announces new supercomputer, lays out vision for future AI work",
    "https://news.microsoft.com/source/features/ai/openai-azure-supercomputer/",
    notes="Azure supercomputer built for OpenAI: more than 285,000 CPU cores and 10,000 GPUs.",
)
TOUVRON_LLAMA_2023 = _lit(
    "prov:touvron-llama-2023",
    "Touvron et al. (2023), LLaMA: Open and Efficient Foundation Language Models",
    url="https://arxiv.org/abs/2302.13971",
    verified=_V,
    notes="The 65B model was trained on 2,048 A100 80GB GPUs.",
)
DEEPSEEK_V3_TRAINING_CLUSTER = _lit(
    "prov:deepseek-v3-training-cluster-2024",
    "DeepSeek-AI (2024), DeepSeek-V3 Technical Report",
    url="https://arxiv.org/abs/2412.19437",
    verified=_V,
    notes="DeepSeek-V3 was trained on a cluster of 2,048 NVIDIA H800 GPUs.",
)

# --- Measured fleet operations (Literature.FleetOperations) ---
KOKOLIS_RSC_RELIABILITY_2025 = _lit(
    "prov:kokolis-rsc-reliability-2025",
    "Kokolis et al. (2025), Revisiting Reliability in Large-Scale Machine Learning Research Clusters",
    url="https://arxiv.org/abs/2410.21680",
    verified=_V,
    notes=(
        "Measured MTTF: 1,024-GPU jobs 7.9 hours; 8-GPU jobs 47.7 days (1,144.8 hours). "
        "Projected MTTF: 16,384-GPU jobs 1.8 hours; 131,072-GPU jobs 0.23 hours."
    ),
)
JEON_PHILLY_2019 = _lit(
    "prov:jeon-philly-2019",
    "Jeon et al. (2019), Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads (USENIX ATC)",
    url="https://arxiv.org/abs/1901.05758",
    verified=_V,
    notes="Hardware utilization of GPUs in use is only around 52% on average (Microsoft Philly trace).",
)
LI_PERLMUTTER_2023 = _lit(
    "prov:li-perlmutter-2023",
    "Li et al. (2023), Analyzing Resource Utilization in an HPC System: A Case Study of NERSC's Perlmutter",
    url="https://arxiv.org/abs/2301.05145",
    verified=_V,
    notes="About 50% of GPU-enabled jobs used up to 25% of the GPU memory (one-month usage analysis).",
)
SIGLER_OPENAI_K8S_2021 = _report(
    "prov:sigler-openai-k8s-2021",
    "OpenAI (2021), Scaling Kubernetes to 7,500 nodes",
    "https://openai.com/index/scaling-kubernetes-to-7500-nodes/",
)

# --- Datacenter electricity projections (Literature.DatacenterElectricity) ---
LBNL_US_DATACENTER_ENERGY_2024 = _report(
    "prov:lbnl-us-datacenter-energy-2024",
    "Shehabi et al. (2024), 2024 United States Data Center Energy Usage Report (LBNL)",
    "https://eta.lbl.gov/publications/2024-lbnl-data-center-energy-usage-report",
    notes="US data centers: ~58 TWh (2014), ~176 TWh (2023), 325-580 TWh projected for 2028.",
)
IEA_ENERGY_AND_AI_2025 = _report(
    "prov:iea-energy-and-ai-2025",
    "IEA (2025), Energy and AI",
    "https://www.iea.org/reports/energy-and-ai",
    notes="Global data-center electricity projected to ~945 TWh by 2030, just under 3% of global electricity.",
)

# --- Hardware additions ---
HOPPER_MICROBENCHMARK_LATENCY = _lit(
    "prov:luo-hopper-microbenchmark-2024",
    "Luo et al. (2024), Benchmarking and Dissecting the Nvidia Hopper GPU Architecture",
    url="https://arxiv.org/abs/2402.13499",
    verified=_V,
    notes=(
        "Table IV (H800 PCIe, Hopper): shared memory 29.0 cycles, L1 40.7, L2 263.0, "
        "global 478.8 cycles; max clock 1,755 MHz (Table III). Nanosecond values are "
        "derived as cycles / 1.755 GHz."
    ),
)
LITHOGRAPHY_RETICLE_FIELD = _conv(
    "prov:lithography-reticle-field",
    "Standard 26 mm x 33 mm full-field exposure of 4x-reduction steppers/scanners (DUV and 0.33-NA EUV)",
    notes="Area 858 mm^2 is derived (26 x 33). Sets the maximum single-die size without stitching.",
)
ST_STM32F407_DATASHEET = _ds(
    "prov:st-stm32f407-datasheet",
    "STMicroelectronics STM32F405xx/STM32F407xx datasheet (DS8626)",
    "https://www.st.com/resource/en/datasheet/stm32f407vg.pdf",
    verified=_V,
    notes=(
        "Arm Cortex-M4 with FPU at up to 168 MHz; up to 1 MB Flash; 192 KB SRAM including "
        "64 KB core-coupled memory. peak_flops (1 single-precision FLOP per cycle) and "
        "memory.bandwidth (32-bit AHB at 168 MHz) are derived, not datasheet figures."
    ),
)
GOOGLE_TPU_V6E_ICI = _ds(
    "prov:google-tpu-v6e-ici",
    "Google Cloud TPU v6e system architecture",
    "https://docs.cloud.google.com/tpu/docs/v6e",
    verified=_V,
    notes="Interchip interconnect bandwidth per chip: 800 GBps (Google quotes ICI as a bidirectional total).",
)
NVIDIA_HOPPER_HBM_STACKS = _ds(
    "prov:nvidia-hopper-blackwell-hbm-stacks",
    "NVIDIA Hopper Architecture In-Depth; NVIDIA Blackwell architecture technical brief",
    "https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/",
    verified=_V,
    notes="H100 SXM5: five active HBM3 stacks (six sites) for 80 GB; B200: eight HBM3e stacks for 192 GB.",
)
DGX_H100_USER_GUIDE_PHYSICAL = _ds(
    "prov:nvidia-dgx-h100-user-guide-physical",
    "NVIDIA DGX H100/H200 User Guide: hardware overview and physical specifications",
    "https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html",
    verified=_V,
    notes="Four third-generation NVSwitch chips; 18 fourth-generation NVLink links per GPU; system weight 287.6 lb (130.45 kg).",
)

DGX_H100_USER_GUIDE_POWER = _ds(
    "prov:nvidia-dgx-h100-user-guide-power",
    "NVIDIA DGX H100/H200 User Guide: hardware overview, power supplies and networking",
    "https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html",
    verified="2026-09-29",
    notes=("Six 3.3 kW PSUs (3300 W @ 200-240 V, 16 A) in 4+2 redundancy; 10.2 kW max system power; "
           "runs at reduced performance on three PSUs, boots only with at least three. Networking: eight "
           "single-port ConnectX-7 InfiniBand cards (compute) plus two dual-port ConnectX-7 Ethernet cards "
           "(storage/in-band); nics_per_node counts only the eight compute cards."),
)

# --- Composed systems ---
NVIDIA_DGX_SUPERPOD_H100_RA = _ds(
    "prov:nvidia-dgx-superpod-h100-ra",
    "NVIDIA DGX SuperPOD Reference Architecture (DGX H100)",
    "https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/",
    verified=_V,
    notes="One scalable unit (SU) = 32 DGX H100 nodes (256 GPUs) on an NDR 400 Gb/s InfiniBand compute fabric.",
)
META_GENAI_INFRA_2024 = _report(
    "prov:meta-genai-infrastructure-2024",
    "Meta Engineering (2024), Building Meta's GenAI Infrastructure",
    "https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/",
    notes=(
        "Two 24,576-GPU H100 clusters on Grand Teton (8 GPUs per server): one on a 400 Gb/s "
        "RoCE fabric, one on NVIDIA Quantum-2 400 Gb/s InfiniBand."
    ),
)
META_RSC_2022 = _report(
    "prov:meta-rsc-2022",
    "Meta AI (2022), Introducing the AI Research SuperCluster",
    "https://ai.meta.com/blog/ai-rsc/",
    notes="Full build-out: 16,000 A100 GPUs in DGX A100 systems (2,000 nodes) on a 200 Gb/s InfiniBand fabric.",
)
FABRIC_OVERSUBSCRIPTION_PROFILES = _est(
    "prov:fabric-oversubscription-profiles",
    "Reference 2:1 and 4:1 oversubscribed fat-tree profiles",
    notes="Documented teaching profiles: an NDR fabric with 2:1 or 4:1 leaf downlink-to-uplink oversubscription; no single field source.",
    verified=_V,
)

# --- Economics ---
NETWORK_CABLING_PRICES_2024 = _est(
    "prov:network-cabling-prices-2024",
    "Illustrative 400G-class link-media prices (DAC, AOC, optical link)",
    notes="Documented teaching assumption: ~$50 per DAC, ~$500 per AOC or per optical link; order-of-magnitude list prices, not a vendor quote.",
    verified=_V,
)
ML_ENGINEER_LOADED_RATE = _est(
    "prov:ml-engineer-loaded-rate",
    "Illustrative fully loaded ML engineer cost per hour",
    notes="Documented teaching assumption: $150/hour fully loaded (salary, benefits, overhead); no field source.",
    verified=_V,
)
CLOUD_COMMITMENT_DISCOUNT_RANGES = _est(
    "prov:cloud-commitment-discount-ranges",
    "Illustrative public-cloud spot and reserved discount ranges versus on-demand",
    notes="Documented teaching ranges: spot/preemptible 60-90% off on-demand; 1-3 year reserved/committed 40-60% off. Providers advertise 'up to' figures; realized discounts vary by region and instance.",
    verified=_V,
)
CLOUD_PREEMPTION_NOTICE = _ds(
    "prov:cloud-preemption-notice",
    "AWS EC2 Spot Instance interruption notices; Google Cloud preemptible VM instances",
    "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-instance-termination-notices.html",
    verified=_V,
    notes="AWS Spot gives a two-minute interruption notice; Google Cloud preemptible/Spot VMs give 30 seconds (https://cloud.google.com/compute/docs/instances/preemptible).",
)
SCHEDULER_PREEMPTION_CASE_STUDY = _est(
    "prov:scheduler-preemption-case-study",
    "Reference scheduler/preemption goodput loss for a shared 100K-GPU fleet case study",
    notes="Documented teaching assumption: 17% of fleet time lost to scheduling and preemption; with the four OVERHEAD_BUDGETS fractions it sums to a 0.60 goodput ratio.",
    verified=_V,
)
GPT3_TRAINING_COST_EST = _est(
    "prov:gpt3-training-cost-estimate",
    "Lambda Labs (2020), OpenAI's GPT-3 Language Model: A Technical Overview (~$4.6M on V100 cloud instances)",
    url="https://lambdalabs.com/blog/demystifying-gpt-3",
    notes="Third-party estimate at 2020 cloud prices; OpenAI did not disclose the training cost.",
    verified=_V,
)
EPA_WATERSENSE_HOUSEHOLD = _est(
    "prov:epa-watersense-household",
    "US EPA WaterSense: average American family uses more than 300 gallons of water per day at home",
    url="https://www.epa.gov/watersense/how-we-use-water",
    notes="300 gal/day x 365 = ~414,000 L/year; the 380,000-450,000 L band brackets that household figure.",
    verified=_V,
)

# --- Reusable scenario anchors (ReferenceStats) ---
RECOMMENDER_AT_SCALE_PROFILE = _est(
    "prov:recommender-at-scale-profile",
    "Reference industrial recommender serving profile (DLRM-class)",
    notes=(
        "Documented teaching assumption: 100 ms P99 end-to-end SLO, embedding tables of at "
        "least 10 TB, 10^7 QPS platform traffic, a 100 TB platform-wide resident embedding "
        "footprint across many models, and a 10-trillion-identifier entity catalog. "
        "Architecture per Naumov et al. (2019)."
    ),
    verified=_V,
)
LOCAL_COMPUTE_EFFICIENCY_ASSUMPTION = _est(
    "prov:local-compute-efficiency-assumption",
    "Reference per-accelerator compute efficiency for large-model training",
    notes="Documented teaching assumption: 45% of peak achieved per accelerator before communication losses; no field source.",
    verified=_V,
)
DATA_PARALLEL_SCALING_SCENARIO = _est(
    "prov:data-parallel-scaling-scenario",
    "Reference single-node data-parallel scaling scenario (GPT-2 class)",
    notes="Documented teaching assumption: 1.8 s single-GPU step, 16 samples per GPU, 25 h single-GPU training time; no field source.",
    verified=_V,
)
COLLECTIVE_COMMUNICATION_SCENARIO = _est(
    "prov:collective-communication-scenario",
    "Reference collective-communication scenario",
    notes="Documented teaching assumption: 1 GB gradient buffer per AllReduce and a 100,000-step training run; no field source.",
    verified=_V,
)
NETWORK_ECONOMICS_SCENARIO = _est(
    "prov:network-economics-scenario",
    "Reference cluster network-economics scenario",
    notes="Documented teaching assumption: $300M cluster capex, 30% of step time in communication, 5 GB per-GPU gradient payload; no field source.",
    verified=_V,
)
CLUSTER_TCO_SCENARIO = _est(
    "prov:cluster-tco-scenario",
    "Reference build-vs-buy cluster TCO scenario (1K and 10K GPU)",
    notes="Documented teaching assumption: network, facility, opex, and staffing line items for 1K- and 10K-GPU clusters; no field source.",
    verified=_V,
)
FACILITY_SCALE_ANCHORS = _est(
    "prov:facility-scale-anchors",
    "Reference datacenter facility scale anchors",
    notes="Documented teaching anchors: ~100 MW hyperscale AI campus; 10 MW reference IT load; order-of-magnitude only.",
    verified=_V,
)
CARBON_PLACEMENT_SCENARIO = _est(
    "prov:carbon-placement-scenario",
    "Reference carbon-aware placement scenario",
    notes="Documented teaching assumption: a 10,000 MWh training run used to compare grid siting; no field source.",
    verified=_V,
)
MIDSCALE_TRAINING_CARBON_SCENARIO = _est(
    "prov:midscale-training-carbon-scenario",
    "Reference mid-scale training carbon scenario (7B model, 64 A100s)",
    notes="Documented teaching assumption: a 14-day run on a 64-GPU A100 lab cluster; no field source.",
    verified=_V,
)
