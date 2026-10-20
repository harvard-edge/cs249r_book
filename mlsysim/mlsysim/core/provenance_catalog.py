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
OCONNOR_FGDRAM_2017 = _lit(
    "prov:oconnor-fgdram-2017",
    "O'Connor et al. (2017), Fine-Grained DRAM: Energy-Efficient DRAM for Extreme Bandwidth Systems, MICRO-50 (Section 1, Figure 1b)",
    url="https://research.nvidia.com/publication/2017-10_fine-grained-dram-energy-efficient-dram-extreme-bandwidth-systems",
    verified="2026-10-15",
    notes="'The energy to access a bit in HBM2 is approximately 3.97 pJ/bit' (activation + on-die data movement + I/O).",
)
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
        "every joint. FER Cartesian translation velocity 1.7 m/s and acceleration 13.0 m/s^2 (FER Cartesian-space limits table, re-read 2026-10-19). DH d3 = 0.316 m "
        "and d5 = 0.384 m: the current page labels the table for FR3; the Panda "
        "label with d3 = 0.316 m is in earlier revisions of the docs (commits "
        "6346f562cc and 7a46004875, control_parameters.rst). Rendered site: "
        "https://frankarobotics.github.io/docs."
    ),
)

FRANKA_FER_STOPPING_TIMES = _ds(
    "prov:franka-fer-handbook-stopping",
    "Franka Emika Robot Instruction Handbook (October 2021), 'Stopping time and "
    "distance', p. 85, measured per EN ISO 10218-1 Annex B",
    "https://www.generationrobots.com/media/franka-emika-robot-handbook.pdf",
    verified="2026-10-19",
    notes=(
        "Quoted: 'Under absolute worst case conditions (e.g. maximum payload of "
        "3 kg and maximum velocity), the stopping distance for the Arm's main "
        "axes (i.e. A1, A3, A5, A7) is 141.71 deg (by cutting 110/230 V) and "
        "116.5 deg (by cutting 48 V), and the stopping time is 652 ms "
        "(110/230 V) and 552 ms (48 V).' The registry stores the 48 V figure, "
        "the cut a permission path makes on the arm supply. The handbook gives "
        "stopping in joint angle and time only; a Cartesian deceleration must "
        "be derived by the consumer and stated as such."
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
    "ISO 13855:2024 walking approach speed K = 1600 mm/s, applied over the whole "
    "stopping time T in S = K*T + C",
    "https://www.iso.org/standard/80590.html",
    verified="2026-09-29",
    notes=(
        "ISO 13855:2024 (Safety of machinery: positioning of safeguards with respect "
        "to the approach of the human body), third edition, published 2024-11; it "
        "replaced ISO 13855:2010 (iso.org/standard/42845), withdrawn 2024-11-22 "
        "(both iso.org pages read via 2025 Wayback captures, 2026-09-29). The "
        "1600 mm/s value was checked against secondary sources quoting the 2010 "
        "edition. Carry-forward note: the 2024 abstract says its approach-speed "
        "values are 'time tested and proven in practical experience', which indicates "
        "K = 1600 mm/s carries forward, but the value has not been read from the "
        "2024 text (paywalled); confirm before print."
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
        "Annex A. ISO 10218-2:2025 incorporates the collaborative-application "
        "requirements formerly published as ISO/TS 15066; ISO/TS 15066:2016 itself "
        "remains published and is under revision. The standard is paywalled; the values were checked against "
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
        "per leg and 4 per arm, 18 in total (stored value corrected from 19 on 2026-09-29). Standard compute "
        "is an Intel Core i5 plus Core i7 with Orin NX optional, not the stored "
        "Jetson AGX Orin. Unsourced MLSysIM estimates: 30 kg payload_capacity "
        "(no H1 payload is listed), 1000 Hz control_frequency, 800 W "
        "nominal_power (the page lists an 864 Wh battery but no power rating)."
    ),
    url="https://www.unitree.com/h1/",
    verified="2026-09-15",
)

UNITREE_H1_COM_HEIGHT_ESTIMATE = _est(
    "prov:unitree-h1-com-height-estimate",
    "Unitree H1 standing center-of-mass height (MLSysIM estimate, not a datasheet value)",
    notes=(
        "Illustrative estimate, 0.9 m. The Unitree H1 product page (checked "
        "2026-09-29) lists a height of about 180 cm and thigh and calf lengths "
        "of 400 mm each, but states no center-of-mass height. 0.9 m places the "
        "CoM near hip height, about half the stature, the usual inverted-pendulum "
        "height for an adult-scale biped. Replace with a value computed from the "
        "vendor URDF link masses or a measured CoM if one is published."
    ),
    url="https://www.unitree.com/h1/",
    verified="2026-09-29",
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
        "accuracy < 1 ms; internal clock 10 ppm drift; power 14-20 W; laser "
        "wavelength 865 nm, Class 1 eye-safe per IEC/EN 60825-1:2014 (re-read "
        "2026-09-29). The Rev7 "
        "sheet gives range as 170 m (80 percent target) and 90 m (10 percent), so "
        "the stored 120 m max_range needs source; nominal_power 18 W sits in the "
        "14-20 W band."
    ),
)

BOSCH_BMI088 = _ds(
    "prov:bosch-bmi088",
    "Bosch Sensortec BMI088 Data Sheet, document BST-BMI088-DS000-19, revision 1.9 (January 2024)",
    "https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi088-ds001.pdf",
    verified="2026-09-29",
    notes=(
        "Automotive and robotics grade 6-DoF IMU with 1000 Hz gyroscope and 1600 Hz "
        "accelerometer update rates. Typical values read from the PDF 2026-09-29: "
        "Table 4 (accelerometer) output noise density 160 ug/sqrt(Hz) on X and Y "
        "(190 on Z), zero-g offset 20 mg (nominal supplies, 25 C, +/-6 g range); "
        "Table 5 (gyroscope) zero-rate offset +/-1 deg/s (25 C, offset cancellation "
        "off), output noise 0.1 deg/s rms at 47 Hz bandwidth (0.014 deg/s/sqrt(Hz)). "
        "Bias instability and angle random walk are not given."
    ),
)

KEYENCE_SZ_V32 = _ds(
    "prov:keyence-sz-v32",
    "Keyence SZ-V Series Safety Laser Scanner, specifications (SZ-V32)",
    "https://www.keyence.com/products/safety/laser-scanner/sz-v/specs/",
    verified="2026-09-29",
    notes=(
        "Representative certified AMR safety scanner, not the only choice. Read "
        "2026-09-29: IEC 61496-3 Type 3 AOPDDR; EN ISO 13849-1:2015 PL d, "
        "Category 3; scan angle 190 deg. Protection zone for 70/150 mm objects: "
        "8.4 m (standard mode), 5.7 m (high-speed mode); for 30 mm objects 2.9 m / "
        "2.0 m. Response time (ON to OFF): 160-1280 ms standard mode, 80-640 ms "
        "high-speed mode (2 to 16 scans); network variants add 6 ms (PROFIsafe) or "
        "10 ms (CIP Safety). The registry stores the high-speed-mode values for "
        "70/150 mm objects (5.7 m, 80 ms), the mode a moving base would use."
    ),
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

NVIDIA_DLE_RESNET50_A100_TRAINING = Provenance(
    id="prov:nvidia-dle-resnet50-a100-training",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="NVIDIA DeepLearningExamples, ResNet-50 v1.5 training performance on DGX A100 (8x A100 80GB), mixed precision",
    url="https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5",
    verified="2026-10-13",
    notes=(
        "Measured per-GPU training throughput: MXNet (mxnet-22.10 container) 3,411 img/s on 1 GPU and "
        "26,674 img/s on 8 (3,334 per GPU); PyTorch 2,470 and 16,621 (2,078 per GPU); TensorFlow (A100 "
        "40GB) 2,375 and 17,400 (2,175 per GPU). 3,200 img/s is the well-tuned anchor, just below the "
        "fastest framework's 8-GPU rate. Replaces an unverified 'MLPerf Training v3.0' attribution."
    ),
)

MLPERF_TRAINING_V31_RESNET50_H100 = Provenance(
    id="prov:mlperf-training-v31-resnet50-h100",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref=(
        "MLPerf Training v3.1 (November 2023), closed division, available on-premises: "
        "NVIDIA Eos_n8 (8x DGX H100, 64 H100-SXM5-80GB, MXNet NVIDIA Release 23.09), "
        "ResNet-50; five runs, each 35 epochs in 2.49 min"
    ),
    url="https://github.com/mlcommons/training_results_v3.1/tree/main/NVIDIA/results/eos_n8_ngc23.09_mxnet/resnet",
    verified="2026-10-19",
    notes=(
        "Per-accelerator rate = median logged training throughput (tracked_stats, "
        "about 318,100 img/s across the five result logs) / 64 = about 4,970 img/s, "
        "rounded to 5,000. End-to-end including evaluation is about 4,680 img/s. "
        "The single-node Eos_n1 submission (8 H100, same release) logs about 7,050 "
        "img/s per accelerator, so the per-accelerator rate falls with scale."
    ),
)

MLPERF_TRAINING_V40_RESNET = Provenance(  # added 2026-10-20
    id="prov:mlperf-training-v40-resnet",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref=(
        "MLPerf Training v4.0 (June 2024), closed division, ResNet-50: NVIDIA eos_ngc24.04_mxnet "
        "(8x H100) and eos_n8_ngc24.04_mxnet (64x H100) result logs and scaling.json; MLPerf Training "
        "rules, ResNet-50 quality target"
    ),
    url="https://github.com/mlcommons/training_results_v4.0/tree/main/NVIDIA/results",
    verified="2026-10-17",
    notes=(
        "The published score is the mean of the middle three of five run times (a run that misses the "
        "quality target counts as the slowest) times the scaling.json factor; every ResNet folder carries "
        "the same factor. Global batch from the global_batch_size MLLOG line of both submissions."
    ),
)

MLPERF_INFERENCE_V50_RESNET50_H100 = Provenance(
    id="prov:mlperf-inference-v50-resnet50-h100",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="MLPerf Inference v5.0 Datacenter, closed division, ResNet-50: Cisco UCS C885A M8 (8x H100-SXM-80GB, TensorRT, INT8), result 5.0-0005; Offline 715,761 samples/s, Server 652,002 samples/s at p99 10.86 ms under a 15 ms bound",
    url="https://github.com/mlcommons/inference_results_v5.0/tree/main/closed/Cisco/results/C885A_H100_SXMx8_TRT/resnet50",
    verified="2026-10-12",
    notes="Per-accelerator figures divide the eight-GPU system result by eight, as MLCommons and NVIDIA do; MLPerf itself reports system throughput.",
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

RESEARCH_AGENT_WORKLOAD = Provenance(
    id="prov:research-agent-workload",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref=(
        "Illustrative orchestrator-worker research agent workload; token "
        "multiples from Hadfield et al. (2025), How We Built Our Multi-Agent "
        "Research System, Anthropic Engineering Blog"
    ),
    url="https://www.anthropic.com/engineering/built-multi-agent-research-system",
    verified="2026-09-29",
    notes=(
        "Sourced: agent_token_multiple = 4 and fanout_token_multiple = 15 "
        "quote the report's 'agents typically use about 4x more tokens than "
        "chat interactions, and multi-agent systems use about 15x more tokens "
        "than chats', measured on the builders' own data. Teaching "
        "assumptions with no published source: the 20-turn horizon, 100 "
        "untrusted documents read per task, the 16,000-input and 800-output "
        "token chat answer, the USD 0.10 search/fetch and USD 0.05 rubric "
        "charges per attempt, the 0.60 acceptance probability, and the USL "
        "coefficients sigma = 0.10 and kappa = 0.002."
    ),
)

REMEDIATION_AGENT_WORKLOAD = Provenance(
    id="prov:remediation-agent-workload",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Illustrative on-call remediation agent workload (approved production configuration change)",
    verified="2026-09-29",
    notes=(
        "Every value is an MLSysIM teaching assumption with no published "
        "source: 10-turn horizon, 10 untrusted items (alerts, log excerpts) "
        "per task, USD 0.15 model, USD 0.10 tool, and USD 0.20 health-check "
        "charges per attempt, 0.70 acceptance probability, 2 approval requests "
        "per hour, 6 min diagnosis, 15 min mean approval decision, 9 min "
        "apply-and-verify, 10 min reviewer effort at USD 120 per hour, 0.04 "
        "probability a proposed fleet change is wrong, USD 12,000 impact of a "
        "committed wrong change, 0.80 canary catch rate, and 0.90 of a caught "
        "change's impact recovered by canary revert."
    ),
)

SUPPORT_AGENT_WORKLOAD = Provenance(
    id="prov:support-agent-workload",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Illustrative customer-support agent workload (short interactive sessions)",
    verified="2026-09-29",
    notes=(
        "Every value is an MLSysIM teaching assumption with no published "
        "source: 6-turn session, 6 untrusted items (messages, attachments) per "
        "session, 8,192-token session context with a 6,144-token shared "
        "prefix, 150-token replies, 2 s generation residence and 20 s customer "
        "reply time per turn, USD 0.03 model, USD 0.01 record-lookup, and "
        "USD 0.01 grounding-check charges per session, 0.85 acceptance "
        "probability, and a USD 4 human handoff per rejected session."
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

EFFECTIVE_COLLECTIVE_ALPHA_RANGES = _est(
    "prov:effective-collective-alpha-ranges",
    "Effective per-message collective-launch latency (alpha) ranges by interconnect: NVLink 4.0, "
    "PCIe Gen5, InfiniBand NDR and HDR, 100G RoCE",
    notes=(
        "Teaching assumption, not a vendor figure: alpha here is the EFFECTIVE launch latency of one "
        "collective message, communication-library software stack included, so it sits above the wire "
        "latencies in Tech.Interconnect and Fabrics. Ranges: NVLink 1-2 us, PCIe Gen5 2-5 us, IB NDR "
        "1-3 us, IB HDR 2-5 us, RoCE 100G 5-10 us. They bracket the single-value anchors AlphaNdr "
        "(1.5 us) and AlphaRoce (5 us) from prov:fabric-latency-assumptions. Consolidated 2026-10-15 "
        "from three identical hand-typed copies in book cells into this one source."
    ),
    verified="2026-10-15",
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
    "Book convention: small-job scaling efficiency (32 GPUs)",
    notes="Only the 32-GPU tier; larger tiers are modeled (prov:scaling-efficiency-reference-job-modeled).",
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
        "simulator's encoding of JESD209-5C Table 339/340. The analog minima "
        "t_RCD,min = t_RPpb,min = 18 ns and t_RPab,min = 21 ns, and the x16 "
        "(16-bit) channel width, agree with a second independent encoding: "
        "coreboot util/spd_tools/src/spd_gen/lp5.go (TRCDMinNs = 18, "
        "TRPPBMinNs = 18, TRPABMinNs = 21, 'Table 372 from JESD209-5B'; channel "
        "widths 8 or 16 bits), read 2026-09-29."
    ),
)

JEDEC_MEMORY_INTERFACE_WIDTHS = _conv(
    "prov:jedec-memory-interface-widths",
    "JEDEC JESD238 (HBM3) and JESD79-5 (DDR5) interface widths",
    notes=(
        "Added 2026-10-20. HBM3: 1,024 data bits per stack in 16 channels of 64 bits. "
        "DDR5: 64 data bits per DIMM channel as two independent 32-bit subchannels. "
        "Added 2026-10-20: the DDR5-4800 speed grade transfers 4,800 MT/s per pin."
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

REPLICA_COLD_START_SCENARIO = _est(
    "prov:replica-cold-start-scenario",
    "Reference cold start of a newly placed LLM serving replica",
    notes=(
        "Documented teaching assumption: 60 seconds to acquire and boot an instance, the weight "
        "artifact fetched from object storage at an effective 0.5 GB/s, and 60 seconds of device "
        "loading and warmup. Calibrate all three from fleet measurements for production sizing; "
        "cached artifacts or overlapped phases shorten the path, shared storage lengthens it."
    ),
    verified="2026-10-15",
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

# Pilz PNOZ s4 safety relay (added 2026-10-20).
PILZ_PNOZ_S4 = _ds(
    "prov:pilz-pnoz-s4",
    "Pilz GmbH & Co. KG, PNOZ s4 safety relay operating manual 1001742-EN-05, Technical details, Times",
    "https://www.farnell.com/datasheets/1679412.pdf",
    verified="2026-10-20",
    notes=(
        "Delay-on de-energisation with E-STOP: typ. 10 ms, max. 20 ms. Transcribed from the "
        "Times table; the PDF fetch on entry (2026-10-20) timed out, so re-check against the "
        "manual before citing elsewhere."
    ),
)

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
# Added 2026-10-20: model-training lecture sources (LoRA memory, the Llama 3.1
# model card's compute table, the OPT-175B logbook).
HU_LORA_2021 = _lit(
    "prov:hu-lora-2021",
    "Hu et al. (2021), LoRA: Low-Rank Adaptation of Large Language Models",
    url="https://arxiv.org/abs/2106.09685",
    verified="2026-10-20",
    notes="Sec. 4.2 (v2): on GPT-3 175B, training VRAM falls from 1.2 TB to 350 GB.",
)
META_LLAMA31_MODEL_CARD = _ds(
    "prov:meta-llama31-model-card",
    "Meta, Llama 3.1 model card, models/llama3_1/MODEL_CARD.md (commit 0e0b8c5), Training Energy Use",
    "https://github.com/meta-llama/llama-models/blob/0e0b8c5/models/llama3_1/MODEL_CARD.md",
    verified="2026-10-20",
    notes=(
        "Training time (GPU hours): 8B 1.46M, 70B 7.0M, 405B 30.84M, total 39.3M; training "
        "power consumption 700 W (H100-80GB, TDP 700 W, peak capacity adjusted for PUE); "
        "location-based emissions 405B 8,930 tons CO2eq."
    ),
)
META_OPT_CHRONICLES_2021 = _ds(
    "prov:meta-opt175b-logbook-2021",
    "Zhang et al. (Meta AI), OPT-175B chronicles: logbook screenshot and 27_percent_update.md",
    "https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles",
    verified="2026-10-20",
    notes="Logbook entries run 12.17 (2021-11-24) to 12.38 (2021-12-01); 12.27 to 12.33 are dated 2021-11-28.",
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
    notes="US data centers: ~58 TWh (2014), ~176 TWh (2023, 4.4% of US electricity), 325-580 TWh projected for 2028 (6.7-12%). LBNL-2001637, doi:10.71468/P1WC7Q.",
)
LBNL_US_DATACENTER_ENERGY_2025 = _report(
    "prov:lbnl-us-datacenter-energy-2025",
    "Smith et al. (2026), United States Data Center Energy Usage Report: 2025 Update (LBNL)",
    "https://doi.org/10.71468/P1RP4F",
    notes="US data centers: 192 TWh in 2024 (4.7% of US electricity); 2030 Reference Case 649 TWh (11.8%), Compounded Uncertainty range 521-843 TWh (9.5-15.3%). Revises 2023-2024 slightly below the 2024 Report; prints no pre-2024 values. LBNL-2001758, OSTI 3374245.",
)
IEA_ENERGY_AND_AI_2025 = _report(
    "prov:iea-energy-and-ai-2025",
    "IEA (2025), Energy and AI",
    "https://www.iea.org/reports/energy-and-ai",
    notes="Global data-center electricity ~415 TWh in 2024 (~1.5%), projected ~945 TWh by 2030 (just under 3%); emissions from data-center electricity 180 Mt today, 300 Mt by 2035 (Base Case), below 1.5% of energy-sector emissions.",
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
        "derived as cycles / 1.755 GHz. Re-verified 2026-10-20 from the arXiv v1 PDF: "
        "Table IV row 'Global 541.5 466.3 478.8' (RTX4090, A100, H800); Sec. III-A-4 "
        "allocates 'a global memory that exceeds the L2 size' for the latency test. "
        "The H800 PCIe uses HBM2e (Table III)."
    ),
)
DEAN_NUMBERS_EVERYONE_SHOULD_KNOW = _lit(
    "prov:dean-ladis-2009-numbers",
    "Dean (2009), \"Designs, Lessons and Advice from Building Large Distributed Systems\", "
    "LADIS 2009 keynote, slide \"Numbers Everyone Should Know\"",
    url="https://www.cs.cornell.edu/projects/ladis2009/talks/dean-keynote-ladis2009.pdf",
    verified="2026-10-20",
    notes=(
        "Quoted from the slide: 'Main memory reference 100 ns' (same slide: 'L1 cache "
        "reference 0.5 ns', 'L2 cache reference 7 ns'). A CPU main-memory (host DRAM) "
        "reference; an order-of-magnitude figure, not a per-part measurement."
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
NVIDIA_BLACKWELL_NV_HBI = _ds(
    "prov:nvidia-blackwell-nv-hbi",
    "NVIDIA Blackwell Architecture (technology overview page)",
    "https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/",
    verified="2026-10-15",
    notes=("Quoted: two reticle-limited dies 'connected by a 10 terabytes per second (TB/s) "
           "chip-to-chip interconnect in a unified single GPU' (NV-HBI). NVIDIA does not say "
           "whether 10 TB/s is per direction or a bidirectional total."),
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

DGX_A100_USER_GUIDE_POWER = _ds(
    "prov:nvidia-dgx-a100-user-guide-power",
    "NVIDIA DGX A100 User Guide: hardware overview, power specifications and N+N redundancy",
    "https://docs.nvidia.com/dgx/dgxa100-user-guide/introduction-to-dgxa100.html",
    verified="2026-09-29",
    notes=("Six 3 kW PSUs (3000 W @ 200-240 V, 16 A) in 3+3 redundancy; full power on any three; "
           "6.5 kW max system power (also DGX A100 datasheet 'System Power Usage 6.5kW max'); boots on "
           "two PSUs with GPUs unavailable; NVIDIA directs a shutdown to replace faulty PSUs. "
           "nics_per_node counts the eight single-port compute ConnectX cards; the dual-port storage "
           "cards are excluded."),
)

DGX_B200_USER_GUIDE_POWER = _ds(
    "prov:nvidia-dgx-b200-user-guide-power",
    "NVIDIA DGX B200 User Guide: hardware overview, power specifications and PSU redundancy",
    "https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html",
    verified="2026-09-29",
    notes=("Six 3.3 kW PSUs (3300 W @ 200-240 V, 16 A) in 5+1 redundancy; 14.3 kW max system power; "
           "with 5 PSUs no effect (1,000 W per GPU), with 4 or 3 PSUs reduced performance (800 W per "
           "GPU), boots only with at least three. nics_per_node counts the eight single-port ConnectX-7 "
           "compute cards; the two dual-port BlueField-3 DPUs are excluded."),
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


# ---------------------------------------------------------------------------
# 2026-09-29 additions (scaling-efficiency tiers, registry-items-2). Add-only.
# ---------------------------------------------------------------------------
SCALING_EFFICIENCY_REFERENCE_JOB_MODELED = Provenance(
    id="prov:scaling-efficiency-reference-job-modeled",
    kind=ProvenanceKind.DERIVED,
    ref="MLSysIM DistributedModel sweep of Scenarios.DenseLLMPretrainingReference",
    notes=(
        "GPT-3 175B on DGX H100 + HDR InfiniBand; TP8 x PP8 (64-GPU replica); fixed global batch "
        "1,536 sequences (Brown et al. 2020 Table 2.1), one sequence per micro-batch, overlap_comm=True, "
        "efficiency=LocalComputeEfficiency 0.45, fp16. Raw 0.8738 (256), 0.8293 (1,024), 0.5619 (8,192), "
        "rounded to 2 dp. Same sweep as vol2 fig-fleet-orchestration-scaling-efficiency-curve. Published "
        "strong-scaling retention is higher with larger batch and interleaved pipelines: MegaScale Table 2 "
        "is 0.90 from 256 to 1,024 and 0.93 from 3,072 to 12,288 GPUs."
    ),
    verified="2026-09-29",
)

# --- Accelerator life cycle (2026-10-13) ---
SCHNEIDER_TPU_LIFECYCLE_2025 = _lit(
    "prov:schneider-tpu-lifecycle-2025",
    "Schneider et al. (2025), Life-Cycle Emissions of AI Hardware: A Cradle-to-Grave Approach and Generational Trends",
    url="https://arxiv.org/abs/2502.01671",
    verified="2026-10-13",
    notes=(
        "First published AI-accelerator LCA (five TPU generations, extraction through disposal). "
        "Abstract: compute carbon intensity (CCI) improves 3x from TPU v4i to TPU v6e."
    ),
)
GOOGLE_SEARCH_VOLUME_2025 = _lit(
    "prov:google-search-volume-2025",
    "Srinivasan (2025), AI, personalization and the future of shopping (Google, The Keyword, 3 March 2025)",
    url="https://blog.google/products/ads-commerce/ai-personalization-and-the-future-of-shopping/",
    verified="2026-10-13",
    notes="'We already see more than 5 trillion searches on Google annually.' Daily figure = 5e12 / 365, a lower bound.",
)

# --- Fleet scheduling scenarios (2026-10-15) ---
# Shared scenario anchors for the orchestration and performance-engineering
# examples, registered so every reader computes from one source instead of
# copying the values by hand.
FORCED_PREEMPTION_SCENARIO = _est(
    "prov:forced-preemption-scenario",
    "Reference forced-preemption scenario for a low-priority training job on a shared fleet",
    notes=(
        "Documented teaching assumption: a forced (non-graceful) preemption discards 15 minutes of "
        "work since the last durable checkpoint (half of a 30-minute checkpoint interval), and a "
        "low-priority job with no preemption budget is struck 12 times a day. Relaunch, checkpoint "
        "reload, and warmup are not part of this record: they come from "
        "Systems.Reliability.Recovery.cold_restart_time(), the same cold-restart steps as a failure "
        "recovery, without failure detection because the scheduler initiates the stop."
    ),
    verified="2026-10-15",
)
SPOT_INTERRUPTION_SCENARIO = _est(
    "prov:spot-interruption-scenario",
    "Reference spot-capacity interruption scenario for a large training job on slow storage",
    notes=(
        "Documented teaching assumption: each spot reclamation costs a 15-minute graceful checkpoint "
        "to slow (object-class) storage plus 30 minutes to reacquire capacity, restart, and reload, "
        "after every 2 hours of retained progress. Slower than the cluster recovery budget "
        "(Systems.Reliability.Recovery) because the capacity is re-provisioned and the checkpoint "
        "path is not the cluster's parallel file system."
    ),
    verified="2026-10-15",
)
ELASTIC_RESIZE_SCENARIO = _est(
    "prov:elastic-resize-scenario",
    "Reference elastic-resize decision scenario (128/256/512 GPUs over a 24-hour window)",
    notes=(
        "Documented teaching assumption: relative throughput 1.0, 1.8, and 3.2 epochs/hour at 128, "
        "256, and 512 GPUs (90 and 80 percent scaling efficiency against 128); the full 512-GPU "
        "allocation frees after a 4-hour queue wait; one resize costs 10 minutes to checkpoint, "
        "reshard, and restart; 24-hour comparison window."
    ),
    verified="2026-10-15",
)
UTILIZATION_TIERS_SCENARIO = _est(
    "prov:utilization-tiers-scenario",
    "Reference allocated/active/productive capacity tiers for one team's allocation",
    notes=(
        "Documented teaching assumption: 500 GPUs allocated, 350 active-device equivalents, and 280 "
        "device-equivalents of useful work at the job's fault-free step rate (time averages)."
    ),
    verified="2026-10-15",
)
FLEET_UTILIZATION_SNAPSHOT_SCENARIO = _est(
    "prov:fleet-utilization-snapshot-scenario",
    "Reference shared-fleet utilization snapshot (assigned, incomplete-gang, input-wait shares)",
    notes=(
        "Documented teaching assumption: 80 percent of in-service capacity assigned to jobs, of which "
        "15 percent of the fleet sits in gangs still waiting for members and another 5 percent sits "
        "in complete allocations waiting for input."
    ),
    verified="2026-10-15",
)
DEEPSEEK_3FS_README = _lit(
    "prov:deepseek-3fs-readme",
    "DeepSeek, Fire-Flyer File System (3FS), README, peak read throughput stress test",
    url="https://github.com/deepseek-ai/3FS",
    verified="2026-10-20",
    notes=(
        "Read stress test: 180 storage nodes, each with 2x200 Gb/s InfiniBand NICs and sixteen "
        "14 TiB NVMe SSDs; about 500+ client nodes with one 200 Gb/s NIC each; aggregate read "
        "throughput about 6.6 TiB/s with background traffic from training jobs."
    ),
)

# --- Incident investigation reports (added 2026-10-20) ----------------------
LIONS_ARIANE_501_1996 = _lit(
    "prov:lions-ariane-501-1996",
    "J. L. Lions (chair), ARIANE 5 Flight 501 Failure: Report by the Inquiry Board, ESA/CNES, Paris, 19 July 1996",
    url="https://esamultimedia.esa.int/docs/esa-x-1819eng.pdf",
    verified="2026-10-21",
    notes=(
        "Read from the scanned report (OCR, sections 1 and 1.1-1.3): the flight was nominal until "
        "approximately H0 + 37 s; both Inertial Reference Systems ceased to function at around "
        "H0 + 36.7 s; the launcher broke up and exploded about 40 s after initiation of the flight "
        "sequence, at an altitude of about 3700 m (self-destruction near the pad at about 4000 m)."
    ),
)
NTSB_HAR_19_03 = _lit(
    "prov:ntsb-har-19-03",
    "NTSB, Collision Between Vehicle Controlled by Developmental Automated Driving System and "
    "Pedestrian, Tempe, Arizona, March 18, 2018, Highway Accident Report NTSB/HAR-19/03 (2019)",
    url="https://www.ntsb.gov/investigations/AccidentReports/Reports/HAR1903.pdf",
    verified="2026-10-20",
    notes=(
        "Read from the report PDF: emergency threshold and braking cap 7 m/s2 (0.71 g) and the "
        "1-second action suppression (p. 13); TPD average deceleration capability 0.92 g "
        "(footnote 33, p. 13; footnote 91, p. 43); table of selected ADS parameters "
        "(pp. 15-16): first radar detection at -5.6 s, emergency and suppression at -1.2 s and "
        "43.2 mph, suppression end at -0.2 s and 40.5 mph, operator takes the wheel at -0.02 s; "
        "impact at 39 mph (p. v and p. 2); about 350 ft from the pedestrian at first detection (2.1.2); "
        "footnote 91 states 2.1 s to stop from an initial 44 mph at 0.92 g."
    ),
)
QUINN_EMANUEL_CRUISE_2024 = _lit(
    "prov:quinn-emanuel-cruise-2024",
    "Quinn Emanuel Urquhart & Sullivan, Report to the Boards of Directors of Cruise LLC, GM "
    "Cruise Holdings LLC, and General Motors Holdings LLC regarding the October 2, 2023 "
    "accident in San Francisco (2024)",
    url="https://assets.ctfassets.net/95kuvdv8zn1v/1mb55pLYkkXVn0nXxEXz7w/9fb0e4938a89dc5cc09bf39e86ce5b9c/2024.01.24_Quinn_Emanuel_Report_re_Cruise.pdf",
    verified="2026-10-20",
    notes=(
        "Pullover after the collision: the vehicle moved about 20 ft at up to 7.7 mph with the "
        "pedestrian beneath it (p. 10). Transcribed from the Volume IV chapter's citation; "
        "not re-read against the report in this pass."
    ),
)

# --- Agent task horizon (added 2026-10-20) -----------------------------------
METR_TIME_HORIZON_V1_1 = _lit(
    "prov:metr-time-horizon-v1-1",
    "METR, Time Horizon v1.1 benchmark results; Kwa et al. (2025), Measuring AI Ability to "
    "Complete Long Software Tasks, arXiv:2503.14499",
    url="https://metr.org/assets/benchmark_results_1_1.yaml",
    verified="2026-10-12",
    notes=(
        "Stated 2023-on doubling time of the 50% time horizon: 128.744 days (CI 104.4-158.0), "
        "fit on frontier models with p50 at or below 16 h."
    ),
)

# --- Model calls and agent trajectories (added 2026-10-20) -------------------
ANTHROPIC_2025_SERVING_POSTMORTEM = _report(
    "prov:anthropic-2025-serving-postmortem",
    "Anthropic, A postmortem of three recent issues, September 2025",
    "https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues",
    notes=(
        "Routing bug: 0.8 percent of one model's requests misrouted at first, 16 percent at the worst "
        "hour (August 31); approximately 30 percent of the coding tool's active users had at "
        "least one message misrouted. Approximate top-k reproduction: an array of 12,000 zeros "
        "with a single one; the approximate result missed the maximum for every index from 10,240 on."
    ),
)
HUANG_2024_SELF_CORRECTION = _lit(
    "prov:huang-2024-self-correction",
    "Huang et al. (2024), Large Language Models Cannot Self-Correct Reasoning Yet, ICLR 2024, "
    "Table 2 (oracle labels) and Table 3 (intrinsic self-correction), GPT-4 rows",
    url="https://arxiv.org/abs/2310.01798",
    verified="2026-10-18",
    notes="Accuracy in percent; read from the paper's HTML (arxiv.org/html/2310.01798v2).",
)

# --- Collective communication (added 2026-10-20) ------------------------------
SERGEEV_HOROVOD_2018 = _lit(
    "prov:sergeev-2018-horovod",
    "Sergeev and Del Balso (2018), Horovod: fast and easy distributed deep learning in TensorFlow, arXiv:1802.05799",
    url="https://arxiv.org/abs/1802.05799",
    verified="2026-10-20",
    notes=(
        "Checked against arXiv v3 (HTML). Section 2: 128 NVIDIA Pascal GPUs, about half the resources lost to "
        "communication. Section 7: Tensor Fusion up to 65 percent on many-layer models over TCP; default fusion "
        "buffer 64 MB. Section 8: 88 percent efficiency; RDMA three to four percent (Inception V3, ResNet-101) "
        "and 30 percent (VGG-16). In v3 the scaling bars are Figure 7 and the Timeline is Figure 6. Bar heights "
        "and Timeline phase edges are readings of those figures (about +/-250 images/s and +/-2 ms); the "
        "ALLREDUCE start and span are printed in the Timeline's detail pane."
    ),
)
TANG_ONEBIT_ADAM_2021 = _lit(
    "prov:tang-2021-onebit-adam",
    "Tang et al. (2021), 1-bit Adam: communication efficient large-scale training with Adam's convergence speed, ICML",
    url="https://arxiv.org/abs/2102.02888",
    verified="2026-10-20",
    notes="Abstract: communication volume reduced by up to 5x; up to 3.3x throughput for BERT-Large pretraining "
          "and 2.9x for SQuAD fine-tuning, in experiments on up to 256 GPUs.",
)

# --- Silent degradation evidence (added 2026-10-20) --------------------------
LAZER_GOOGLE_FLU_2014 = _lit(
    "prov:lazer-google-flu-2014",
    "Lazer, Kennedy, King & Vespignani (2014), The Parable of Google Flu: Traps in Big Data Analysis, "
    "Science 343(6176):1203-1205",
    url="https://doi.org/10.1126/science.1248506",
    verified="2026-10-20",
    notes="GFT missed high for 100 out of 108 weeks starting with August 2011.",
)
CAMPBELL_DEEP_BLUE_2002 = _lit(
    "prov:campbell-deep-blue-2002",
    "Campbell, Hoane & Hsu (2002), Deep Blue, Artificial Intelligence 134(1-2):57-83",
    url="https://doi.org/10.1016/S0004-3702(01)00129-1",
    verified="2026-10-20",
    notes="30 nodes with 16 chess chips each (480 chips); search speed 100 million positions per second "
          "in tactical positions to about 200 million in quieter ones.",
)
SILVER_ALPHAGO_ZERO_2017 = _lit(
    "prov:silver-alphago-zero-2017",
    "Silver et al. (2017), Mastering the game of Go without human knowledge, Nature 550:354-359",
    url="https://doi.org/10.1038/nature24270",
    verified="2026-10-20",
    notes="After 3 days of self-play training AlphaGo Zero defeated AlphaGo Lee by 100 games to 0.",
)

# --- Deployment-tier field figures (added 2026-10-20) ------------------------
JOUPPI_TPU_V1_ISCA_2017 = _lit(
    "prov:jouppi-tpu-v1-isca-2017",
    "Jouppi et al. (2017), In-Datacenter Performance Analysis of a Tensor Processing Unit, ISCA 2017",
    url="https://arxiv.org/abs/1704.04760",
    verified="2026-10-18",
    notes=("Fig. 5 caption: the TPU roofline's slanted part meets the 92 TOPS ceiling at 1,350 "
           "operations per weight byte (Table 1 counts one op per multiply-add in the intensity)."),
)
GHOLAMI_MEMORY_WALL_2024 = _lit(
    "prov:gholami-memory-wall-2024",
    "Gholami et al. (2024), AI and Memory Wall, IEEE Micro 44(3), Fig. 1",
    url="https://arxiv.org/abs/2403.14123",
    verified="2026-10-18",
    notes=("Growth factors printed on Fig. 1 for server hardware over 20 years: peak FLOP/s 60,000x "
           "(3.0x every two years), DRAM bandwidth 100x (1.6x), interconnect bandwidth 30x (1.4x)."),
)
GSMARENA_IPHONE15PROMAX_3DMARK = _report(
    "prov:gsmarena-iphone15promax-3dmark",
    "GSMArena, Apple iPhone 15 Pro Max review (2023): 3DMark Wild Life Stress Test result screen (UL)",
    "https://m.gsmarena.com/apple_iphone_15_pro_max-review-2618p5.php",
    notes=("Twenty loops of the Wild Life scene, run 2023-09-22; best loop 10,012 and lowest loop "
           "7,045 as the app's result screen reports them."),
)
APPLE_HEY_SIRI_2017 = _report(
    "prov:apple-hey-siri-2017",
    "Apple Machine Learning Research (2017), Hey Siri: An On-device DNN-powered Voice Trigger for "
    "Apple's Personal Assistant",
    "https://machinelearning.apple.com/research/hey-siri",
    notes=("The always-on processor runs a DNN of 5 layers of 32 hidden units; the main-processor "
           "detector uses 5 layers of 192 hidden units (two-pass detection, fig. 5)."),
)
# --- Spaceflight anomaly inquiries (added 2026-10-20) -----------------------
ESA_SCHIAPARELLI_INQUIRY = _lit(
    "prov:esa-schiaparelli-inquiry-2017",
    "ESA Investigation Board, ExoMars 2016 - Schiaparelli Anomaly Inquiry (2017)",
    url="https://exploration.esa.int/web/mars/-/59176-exomars-2016-schiaparelli-anomaly-inquiry",
    verified="2026-10-20",
    notes=(
        "Descent of 19 October 2016: the IMU saturation flag persisted longer than the 15 ms "
        "assumed at acceptance; the attitude estimate drifted by about 165 degrees and the "
        "computed altitude went negative; the parachute and backshell were released and the "
        "thrusters fired for 3 s instead of the planned 30 s at about 3.7 km altitude; free fall "
        "ended in an impact at about 540 km/h."
    ),
)
# --- Workload-aware cluster schedulers (added 2026-10-20) --------------------
QIAO_POLLUX_2021 = _lit(
    "prov:qiao-pollux-2021",
    "Qiao et al. (2021), Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep "
    "Learning (OSDI 21)",
    url="https://arxiv.org/abs/2008.12260",
    verified="2026-10-20",
    notes=("Abstract: Pollux reduces average job completion times by 37-50% relative to "
           "state-of-the-art DL schedulers, on its testbed and trace."),
)

# --- Diabetic retinopathy screening and distribution shift (added 2026-10-20) ---
GULSHAN_DR_JAMA_2016 = _lit(
    "prov:gulshan-dr-jama-2016",
    "Gulshan et al. (2016), Development and Validation of a Deep Learning Algorithm for Detection of "
    "Diabetic Retinopathy in Retinal Fundus Photographs, JAMA 316(22)",
    url="https://jamanetwork.com/journals/jama/fullarticle/2588763",
    verified="2026-10-20",
    notes=("Development set of 128,175 graded images. EyePACS-1 validation (8,788 fully gradable "
           "images, Fig. 2A with eight ophthalmologists plotted): AUC 0.991; high-sensitivity "
           "operating point 97.5 percent sensitivity at 93.4 percent specificity; high-specificity "
           "point 90.3 and 98.1 percent."),
)
BEEDE_DR_CHI_2020 = _lit(
    "prov:beede-dr-chi-2020",
    "Beede et al. (2020), A Human-Centered Evaluation of a Deep Learning System Deployed in Clinics "
    "for the Detection of Diabetic Retinopathy, CHI 2020",
    url="https://doi.org/10.1145/3313831.3376718",
    verified="2026-10-20",
    notes=("Field study in 11 clinics in Thailand over eight months; 393 of 1,838 submitted images "
           "were rejected as ungradable."),
)
FDA_DEN180001_IDX_DR = _report(
    "prov:fda-den180001-idx-dr",
    "U.S. FDA (2018), De Novo classification request DEN180001 (IDx-DR), decision summary",
    "https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf",
    notes=("Pivotal trial of 900 patients at 10 primary care sites, pp. 6-7: 892 completed, 852 "
           "graded by the reading center, 819 analyzable; 173 of 198 mtmDR-positive patients "
           "detected and 556 of 621 mtmDR-negative patients cleared; prespecified endpoints of "
           "85.0 percent sensitivity and 82.5 percent specificity."),
)
KOH_WILDS_ICML_2021 = _lit(
    "prov:koh-wilds-icml-2021",
    "Koh et al. (2021), WILDS: A Benchmark of in-the-Wild Distribution Shifts, ICML 2021",
    url="https://arxiv.org/abs/2012.07421",
    verified="2026-10-20",
    notes=("Table 1, Camelyon17-WILDS, standard (ERM) training: average accuracy 93.2 percent "
           "in distribution (held-out patches from the training hospitals) and 70.3 percent out "
           "of distribution (hospital 5)."),
)
TORCHVISION_0_28_PARAMETER_COUNTS = Provenance(
    id="prov:torchvision-0-28-parameter-counts",
    kind=ProvenanceKind.DERIVED,
    ref="Parameter counts of torchvision 0.28.0 model constructors (torch 2.13.0)",
    url="https://github.com/pytorch/vision",
    verified="2026-10-20",
    notes=("Counted by building each architecture without weights and summing p.numel() over its "
           "parameters (slides/vol1/03_ml_lifecycle/images/data/count_params.py); default "
           "1000-class ImageNet heads."),
)
DASTIN_REUTERS_AMAZON_RECRUITING_2018 = _report(
    "prov:dastin-reuters-amazon-recruiting-2018",
    "Dastin (2018), Amazon scraps secret AI recruiting tool that showed bias against women, Reuters, "
    "October 10, 2018",
    "https://www.reuters.com/article/us-amazon-com-jobs-automation-insight-idUSKCN1MK08G",
    notes=("The models were trained on resumes submitted to the company over a 10-year period; the "
           "system downgraded graduates of two all-women's colleges."),
)

# --- Test-time compute results (added 2026-10-20) -----------------------------
SNELL_2024_TEST_TIME_COMPUTE = _lit(
    "prov:snell-2024-test-time-compute",
    "Snell et al. (2024), Scaling LLM Test-Time Compute Optimally Can Be More Effective Than "
    "Scaling Model Parameters, arXiv:2408.03314v1",
    url="https://arxiv.org/abs/2408.03314v1",
    verified="2026-10-18",
    notes=(
        "PaLM 2-S* on MATH. Section 3.2: questions binned into five difficulty levels by the base "
        "model's pass@1 estimated from 2,048 samples. Section 5.3 and Figure 3: on the middle bins "
        "beam search matches best-of-N with about 4x less compute (16 against 64 generations). "
        "Sections 1 and 7, Figure 9: FLOPs-matched, test-time compute beats a 14x larger model "
        "except on the hardest questions."
    ),
)

LIGHTMAN_2024_PRM800K = _lit(
    "prov:lightman-2024-prm800k",
    "Lightman et al. (2024), Let's Verify Step by Step, ICLR 2024; PRM800K dataset README "
    "(github.com/openai/prm800k @ 7ecc794)",
    url="https://github.com/openai/prm800k/blob/7ecc794703b2877f63226f2477a49b34f9b25163/README.md",
    verified="2026-10-18",
    notes=(
        "About 800,000 step-level human correctness labels. The README's annotated example line: "
        "16 pre-generated steps, the third rated -1, final answer 320,000 against a ground truth "
        "of 40,000, total_time 278,270 ms."
    ),
)

# --- Context engineering: long-context use, summaries, prefix caching (added 2026-10-20) ---
LIU_2024_LOST_IN_THE_MIDDLE = _lit(
    "prov:liu-2024-lost-in-the-middle",
    "Liu et al. (2024), Lost in the Middle: How Language Models Use Long Contexts, TACL 12, "
    "Table 1 (closed-book) and Table 6 (20 total retrieved documents), GPT-3.5-Turbo rows",
    url="https://arxiv.org/abs/2307.03172",
    verified="2026-10-20",
    notes=("Multi-document QA accuracy in percent with the answer document at positions 1, 5, 10, "
           "15, and 20 of 20 (about 4K tokens); closed-book 56.1. Read from arxiv.org/html/2307.03172v3."),
)
HSIEH_2024_RULER = _lit(
    "prov:hsieh-2024-ruler",
    "Hsieh et al. (2024), RULER: What's the Real Context Size of Your Long-Context Language "
    "Models?, COLM 2024, Table 3",
    url="https://arxiv.org/abs/2404.06654",
    verified="2026-10-20",
    notes=("Claimed and effective context length per model; effective length is the longest tested "
           "length scoring above the threshold, 85.6, Llama2 (7B) at 4K. 17 models evaluated. "
           "Read from arxiv.org/html/2404.06654v3."),
)
MAYNEZ_2020_FAITHFULNESS = _lit(
    "prov:maynez-2020-faithfulness",
    "Maynez et al. (2020), On Faithfulness and Factuality in Abstractive Summarization, ACL 2020, Table 2",
    url="https://arxiv.org/abs/2005.00661",
    verified="2026-10-20",
    notes=("Percent of XSum summaries with at least one hallucinated span, intrinsic or extrinsic "
           "(I union E column); 500 test articles per system, three annotators per summary."),
)
ZHENG_2024_SGLANG = _lit(
    "prov:zheng-2024-sglang",
    "Zheng et al. (2024), SGLang: Efficient Execution of Structured Language Model Programs, "
    "NeurIPS 2024, Section 6 (production deployment)",
    url="https://arxiv.org/abs/2312.07104",
    verified="2026-10-20",
    notes=("After one month of public serving: RadixAttention cache hit rate 52.4 percent for "
           "LLaVA-Next-34B and 74.1 percent for Vicuna-33B; first-token latency 1.7 times lower on "
           "average for Vicuna-33B. Read from arxiv.org/html/2312.07104v2."),
)
ABHYANKAR_2024_INFERCEPT = _lit(  # added 2026-10-20
    "prov:abhyankar-2024-infercept",
    "Abhyankar et al. (2024), InferCept: Efficient Intercept Support for Augmented Large Language "
    "Model Inference, ICML 2024",
    url="https://arxiv.org/abs/2402.01869",
    verified="2026-10-18",
    notes=("Abstract: discarding context at every interception and recomputing it took 37 to 40 "
           "percent of model forwarding time in the systems measured."),
)
SHI_2023_DISTRACTED = _lit(
    "prov:shi-2023-distracted",
    "Shi et al. (2023), Large Language Models Can Be Easily Distracted by Irrelevant Context, "
    "ICML 2023, Tables 2 and 3 (code-davinci-002)",
    url="https://arxiv.org/abs/2302.00093",
    verified="2026-10-20",
    notes=("Chain-of-thought accuracy in percent: 95.0 on the 100-problem base set (Table 2); "
           "GSM-IC-4K micro accuracy 72.4, and 77.8 with an instruction to ignore irrelevant "
           "information (Table 3, exemplars without irrelevant context)."),
)

# --- Published vision-language-action systems (added 2026-10-20) ------------
KIM_OPENVLA_2024 = _lit(
    "prov:kim-openvla-real-robot-2024",
    "Kim, M. J. et al. (2024), OpenVLA: An Open-Source Vision-Language-Action Model, arXiv:2406.09246",
    url="https://arxiv.org/abs/2406.09246",
    verified="2026-09-29",
    notes=("Table 2: real-robot task success with bfloat16 71.3%, int8 58.1%, int4 71.9% weights; "
           "Sec. 5.3: the int8 build ran at 1.2 Hz on the A5000 evaluation GPU against the 5 Hz control "
           "rate of the training data; about 6 Hz on one RTX 4090 in bfloat16."),
)
ZHAO_ACT_2023 = _lit(
    "prov:zhao-act-2023",
    "Zhao, T. Z. et al. (2023), Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT), "
    "RSS 2023, arXiv:2304.13705; code github.com/tonyzhaozh/act",
    url="https://arxiv.org/abs/2304.13705",
    verified="2026-10-20",
    notes=("Chunk size k = 100 at a 50 Hz control rate; temporal ensembling weights exp(-k i) with "
           "k = 0.01 (imitate_episodes.py, commit 742c753, lines 255-259)."),
)
CHI_DIFFUSION_POLICY_2024 = _lit(
    "prov:chi-diffusion-policy-2024",
    "Chi, C. et al. (2024), Diffusion Policy: Visuomotor Policy Learning via Action Diffusion, IJRR",
    url="https://arxiv.org/abs/2303.04137",
    verified="2026-09-29",
    notes=("Prediction horizon T_p = 16, action execution horizon T_a = 8 (simulation tasks); "
           "real-world Push-T executes T_a = 6 of T_p = 16 (App. A.4, Tables 7 and 8, checked 2026-10-20)."),
)
BLACK_PI0_2024 = _lit(
    "prov:black-pi0-2024",
    "Black, K. et al. (2024), pi0: A Vision-Language-Action Flow Model for General Robot Control, "
    "arXiv:2410.24164",
    url="https://arxiv.org/abs/2410.24164",
    verified="2026-09-29",
    notes=("Sec. IV and App. A-D: 3.3B parameters in total, a 300M action expert, H = 50, 10 flow "
           "integration steps, up to 50 Hz control, 73 ms on-board latency (RTX 4090). App. A-D: "
           "robots at 50 Hz run inference every 0.5 s, after executing 25 actions (checked 2026-10-20)."),
)
BERSCHEID_RUCKIG_2021 = _lit(
    "prov:berscheid-ruckig-2021",
    "Berscheid, L. and Kroger, T. (2021), Jerk-limited Real-time Trajectory Generation with Arbitrary "
    "Target States, Robotics: Science and Systems XVII",
    url="https://arxiv.org/abs/2105.04830",
    verified="2026-10-20",
    notes=("Sec. V-B, Table III (7 degrees of freedom, Intel i7-8700K, single thread, PREEMPT Linux): "
           "Ruckig mean 19.8 +/- 0.2 us, worst case 123 +/- 13 us."),
)
KIM_OPENVLA_OFT_2025 = _lit(
    "prov:kim-openvla-oft-2025",
    "Kim, M. J., Finn, C., Liang, P. (2025), Fine-Tuning Vision-Language-Action Models: Optimizing "
    "Speed and Success, RSS 2025",
    url="https://arxiv.org/abs/2502.19645",
    verified="2026-09-29",
    notes="LIBERO on an A100: OpenVLA-OFT 109.7 Hz with 8-step chunks against OpenVLA's 4.2 Hz.",
)
PERTSCH_FAST_2025 = _lit(
    "prov:pertsch-fast-2025",
    "Pertsch, K. et al. (2025), FAST: Efficient Action Tokenization for Vision-Language-Action Models, "
    "arXiv:2501.09747",
    url="https://arxiv.org/abs/2501.09747",
    verified="2026-09-29",
    notes=("30 to 60 tokens per chunk; Table I shirt fold at 50 Hz, 700 naive tokens against 53 FAST "
           "tokens; pi0-FAST about 750 ms against pi0's about 100 ms on an RTX 4090."),
)
FIGURE_AI_HELIX_2025 = _report(
    "prov:figure-ai-helix-2025",
    "Figure AI (2025), Helix: A Vision-Language-Action Model for Generalist Humanoid Control",
    "https://www.figure.ai/news/helix",
    notes=("System 2: a 7B vision-language model at 7-9 Hz; System 1: an 80M visuomotor policy at "
           "200 Hz; 35 degrees of freedom."),
)
BJORCK_GR00T_N1_2025 = _lit(
    "prov:bjorck-gr00t-n1-2025",
    "Bjorck, J. et al. (2025), GR00T N1: An Open Foundation Model for Generalist Humanoid Robots, "
    "arXiv:2503.14734",
    url="https://arxiv.org/abs/2503.14734",
    verified="2026-09-29",
    notes=("2.2B parameters with a 1.34B vision-language model at 10 Hz; a flow-matching action "
           "transformer at 120 Hz; 16-action chunks, 4 integration steps, 63.9 ms per chunk (L40, bf16)."),
)
GEMINI_ROBOTICS_2025 = _lit(
    "prov:gemini-robotics-2025",
    "Gemini Robotics Team (2025), Gemini Robotics: Bringing AI into the Physical World, arXiv:2503.20020",
    url="https://arxiv.org/abs/2503.20020",
    verified="2026-09-29",
    notes=("Cloud backbone under 160 ms, about 250 ms end to end, an effective 50 Hz control rate "
           "with chunks."),
)
LIN_AWQ_2024 = _lit(
    "prov:lin-awq-2024",
    "Lin, J. et al. (2024), AWQ: Activation-aware Weight Quantization for LLM Compression and "
    "Acceleration, MLSys 2024",
    url="https://arxiv.org/abs/2306.00978",
    verified="2026-10-20",
    notes="Sec. 5.1: group size 128 for weight-only INT4 quantization.",
)

# --- Data engineering: wake-word benchmarks, annotation times, Speech Commands (added 2026-10-20) ---
PICOVOICE_WAKE_WORD_BENCHMARK = Provenance(
    id="prov:picovoice-wake-word-benchmark",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="Picovoice, wake-word-benchmark (GitHub), README 'Accuracy' and doc/img/summary.png, commit 84919190fb18",
    url="https://github.com/Picovoice/wake-word-benchmark",
    verified="2026-10-17",
    notes=("Miss rates at 1 false alarm per 10 hours (plot.py target_false_alarm_per_hour=0.1), averaged "
           "over six wake words; more than 300 recordings per word from more than 50 speakers, mixed at "
           "10 dB SNR with DEMAND noise from 18 environments, with LibriSpeech as background speech. "
           "Run by the maker of Porcupine."),
)
PAPADOPOULOS_EXTREME_CLICKING_2017 = _lit(
    "prov:papadopoulos-extreme-clicking-2017",
    "Papadopoulos, Uijlings, Keller, and Ferrari (2017), Extreme Clicking for Efficient Object "
    "Annotation, ICCV",
    url="https://doi.org/10.1109/ICCV.2017.528",
    verified="2026-10-20",
    notes="About 35 s to draw one high-quality box in ImageNet's annotation campaign, as the paper reports it.",
)
CITYSCAPES_2016 = _lit(
    "prov:cordts-cityscapes-2016",
    "Cordts et al. (2016), The Cityscapes Dataset for Semantic Urban Scene Understanding, CVPR",
    url="https://www.cityscapes-dataset.com/",
    verified="2026-10-20",
    notes="Fine pixel-level annotation and quality control took more than 1.5 h on average for one image.",
)
SPEECH_COMMANDS_V2_CARD = _report(
    "prov:speech-commands-v002-hf-card",
    "Hugging Face dataset card google/speech_commands, config v0.02 (rev. 57ba463); Warden (2018), "
    "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition, arXiv:1804.03209",
    "https://huggingface.co/datasets/google/speech_commands",
    notes="Split sizes from the card's dataset_info: train 84,848, validation 9,982, test 4,890 clips.",
)
TENSORFLOW_SPEECH_COMMANDS_SPLIT = _conv(
    "prov:tensorflow-speech-commands-which-set",
    "TensorFlow v2.16.1, examples/speech_commands train.py flags --validation_percentage and "
    "--testing_percentage (default 10 each), applied by input_data.which_set()",
    notes="The reference loader assigns each hashed speaker key to a split by these default percentages.",
)
MICROSOFT_TAY_2016 = _report(
    "prov:microsoft-tay-2016",
    "Peter Lee (2016), Learning from Tay's Introduction, The Official Microsoft Blog, March 25, 2016",
    "https://blogs.microsoft.com/blog/2016/03/25/learning-tays-introduction/",
    notes="Tay was taken offline about 16 hours after launch, inside the 'first 24 hours' the post describes.",
)


# --- Data delivery and drift: measured pipelines, files, and releases (added 2026-10-20) ---
MOHAN_DATA_STALLS_2021 = _lit(
    "prov:mohan-data-stalls-2021",
    "Mohan, Phanishayee, Raniwala, and Chidambaram (2021), Analyzing and Mitigating Data Stalls in DNN "
    "Training, PVLDB 14(5)",
    url="https://www.vldb.org/pvldb/vol14/p771-mohan.pdf",
    verified="2026-10-18",
    notes="Nine vision and audio models profiled; CoorDL (coordinated caching and preprocessing) cuts training "
          "time by up to 5x on a single server with the same model and accelerators.",
)
LECLERC_FFCV_2023 = _lit(
    "prov:leclerc-ffcv-2023",
    "Leclerc, Ilyas, Engstrom, Park, Salman, and Madry (2023), FFCV: Accelerating Training by Removing Data "
    "Bottlenecks, CVPR",
    url="https://arxiv.org/abs/2306.12517",
    verified="2026-10-18",
    notes="ResNet-50 trained to 75 percent ImageNet top-1 in about 20 minutes on one machine (8 A100s).",
)
PYTORCH_PROFILER_TUTORIAL = Provenance(
    id="prov:pytorch-profiler-tensorboard-tutorial",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="PyTorch tutorials, PyTorch Profiler With TensorBoard (intermediate_source/tensorboard_profiler_tutorial.py, "
        "commit 45a5b17, the last version before its removal on 2026-06-01; _static/img/profiler_overview1.png "
        "and profiler_overview2.png)",
    url="https://github.com/pytorch/tutorials/blob/45a5b17/intermediate_source/tensorboard_profiler_tutorial.py",
    verified="2026-10-18",
    notes="ResNet-18 on CIFAR-10 resized to 224, batch 32, one Tesla V100; Execution Summary of the average step "
          "as the tool prints it, for the default loader and for num_workers=4.",
)
NYC_TLC_YELLOW_TAXI_2024_01_PARQUET = Provenance(
    id="prov:nyc-tlc-yellow-taxi-2024-01-parquet",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="NYC Taxi and Limousine Commission, Yellow Taxi Trip Records, January 2024 (yellow_tripdata_2024-01.parquet, "
        "49,961,641 bytes); metadata printed by pyarrow 25.0.0",
    url="https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet",
    verified="2026-10-18",
    notes="ParquetFile(path).metadata.row_group(0).column(i) sizes and pyarrow.compute.count_distinct per column; "
          "uncompressed sizes are measured after encoding.",
)
WEBDATASET_PUBLAYNET_SHARD = Provenance(
    id="prov:webdataset-publaynet-shard",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="WebDataset README (github.com/webdataset/webdataset, v0.2.107), test shard publaynet-train-000000.tar",
    url="https://storage.googleapis.com/webdataset/testdata/publaynet-train-000000.tar",
    verified="2026-10-18",
    notes="Content-Length 303,779,840 bytes; `tar tf | wc -l` prints 1,970 members, a .json and a .png per sample. "
          "The README suggests a shuffle buffer of about 1,000 samples.",
)
WARDEN_SPEECH_COMMANDS_2018 = _lit(
    "prov:warden-speech-commands-2018",
    "Warden (2018), Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition, arXiv:1804.03209",
    url="https://arxiv.org/abs/1804.03209",
    verified="2026-10-18",
    notes="Section 6 (v2: 105,829 utterances, 35 words, 2,618 speakers), Section 7.3 (v1, released 2017-08-03: "
          "64,727 utterances, 1,881 speakers), and the Top-One table: the default convolution model trained on "
          "each release and scored on each release's test set.",
)
EVIDENTLY_DEFAULT_STATTESTS = Provenance(
    id="prov:evidently-default-stattests-v0-7-23",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="Evidently v0.7.23 (Apache 2.0), src/evidently/legacy/calculations/stattests/: registry.py "
        "_get_default_stattest, ks_stattest.py, wasserstein_distance_norm.py",
    url="https://github.com/evidentlyai/evidently/tree/v0.7.23/src/evidently/legacy/calculations/stattests",
    verified="2026-10-18",
    notes="Numerical features: two-sample K-S at 1,000 or fewer reference rows (drift when p <= 0.05), normed "
          "Wasserstein distance above (drift when distance / reference sd >= 0.1).",
)

# Fleet efficiency (added 2026-10-20)
WONGPANICH_ML_GOODPUT_2025 = _lit(
    "prov:wongpanich-ml-goodput-2025",
    "Wongpanich et al. (2025), Machine Learning Fleet Efficiency: Analyzing and Optimizing "
    "Large-Scale Google TPU Systems with ML Productivity Goodput",
    url="https://arxiv.org/abs/2502.06982",
    verified="2026-10-20",
    notes=("Defines ML productivity goodput as scheduling x runtime x program goodput; scheduling "
           "goodput exceeds 95 percent for every job-size cohort of Google's TPU fleet."),
)
CLOUDFLARE_OUTAGE_2019 = _lit(
    "prov:cloudflare-outage-2019",
    "Cloudflare (2019), Details of the Cloudflare outage on July 2, 2019",
    url="https://blog.cloudflare.com/details-of-the-cloudflare-outage-on-july-2-2019/",
    verified="2026-10-20",
    notes=("A WAF rule's regular expression backtracked and drove edge CPUs to nearly 100 percent; "
           "a 27-minute outage in which traffic dropped by about 82 percent."),
)
NSIGHT_TRACE_VOLUME_SCENARIO = _est(
    "prov:nsight-trace-volume-scenario",
    "Reference Nsight Systems trace volume per GPU for a short profiling window",
    notes=("Documented teaching assumption: a 5-second Nsight Systems trace of one training GPU, "
           "with CUDA, NVTX, and NCCL ranges, is about 500 MB."),
    verified="2026-10-20",
)

# --- Retrieval from long-term memory: agents, poisoning, a tribunal record (added 2026-10-20) ---
JIMENEZ_2024_SWE_BENCH = _lit(
    "prov:jimenez-2024-swe-bench",
    "Jimenez et al. (2024), SWE-bench: Can Language Models Resolve Real-World GitHub Issues?, "
    "ICLR 2024, Table 2 (Claude 2, BM25 retrieval) and Table 3 (BM25 recall, Avg. row)",
    url="https://arxiv.org/abs/2310.06770",
    verified="2026-10-20",
    notes=("Claude 2 percent resolved and BM25 recall of the oracle (edited) files at maximum "
           "retrieved contexts of 13K, 27K, and 50K tokens; the issue text is the query. "
           "Read from arxiv.org/html/2310.06770v3."),
)
YANG_2024_SWE_AGENT = _lit(
    "prov:yang-2024-swe-agent",
    "Yang et al. (2024), SWE-agent: Agent-Computer Interfaces Enable Automated Software "
    "Engineering, NeurIPS 2024, Tables 1 and 3 (SWE-bench Lite, GPT-4 Turbo) and Section 5",
    url="https://arxiv.org/abs/2405.15793",
    verified="2026-10-20",
    notes=("Percent resolved on SWE-bench Lite with GPT-4 Turbo: RAG (BM25) 2.67, shell-only "
           "agent 11.00, SWE-agent 18.00. Section 5: 'compared to RAG on Lite, SWE-agent is "
           "8-13x more costly but yields a 6.7-fold improved % Resolved rate.' Table 3 "
           "ablations: whole-file viewer 12.7, no edit command 10.3. Read from "
           "arxiv.org/html/2405.15793v3."),
)
ZOU_2025_POISONEDRAG = _lit(
    "prov:zou-2025-poisonedrag",
    "Zou et al. (2025), PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented "
    "Generation of Large Language Models, USENIX Security 2025, Section 5.1 and Table 1",
    url="https://arxiv.org/abs/2402.07867",
    verified="2026-10-20",
    notes=("Knowledge bases of 2,681,468 (NQ), 5,233,329 (HotpotQA), and 8,841,823 (MS-MARCO) "
           "texts; N = 5 malicious texts per target question; black-box attack success rate "
           "with PaLM 2 as the generator. Read from arxiv.org/html/2402.07867v3."),
)
MOFFATT_V_AIR_CANADA_2024 = Provenance(
    id="prov:moffatt-v-air-canada-2024",
    kind=ProvenanceKind.INDUSTRY_REPORT,  # a published tribunal decision (public record)
    ref=("Moffatt v. Air Canada, 2024 BCCRT 149 (British Columbia Civil Resolution Tribunal, "
         "February 14, 2024), paras 15, 40, and 44"),
    url="https://www.canlii.org/en/bc/bccrt/doc/2024/2024bccrt149/2024bccrt149.html",
    verified="2026-10-20",
    notes=("Amounts in Canadian dollars. Para 15: the chatbot's answer quotes a 90-day window "
           "from ticket issue. Para 40: paid 1,630.36, should have paid 979.48, damages 650.88. "
           "Para 44: total 812.02 (650.88 damages, 36.14 pre-judgment interest, 125 CRT fees). "
           "Checked 2026-10-20 against a PDF copy of the CanLII decision (CanLII blocks scripted "
           "access)."),
)

# --- Robot demonstration data: dated systems, interventions, recorders (added 2026-10-20) ---
DROID_RECALIBRATION_2025 = _lit(
    "prov:droid-recalibration-2025",
    "DROID project website, April 2025 release of improved camera calibrations (droid-dataset.github.io)",
    url="https://droid-dataset.github.io/",
    verified="2026-10-19",
    notes="New camera calibrations published for about 36,000 episodes after an overlap check.",
)
BU_AGIBOT_WORLD_2025 = _lit(
    "prov:bu-agibot-world-2025",
    "Bu, Q. et al. (2025), AgiBot World Colosseo: A Large-Scale Manipulation Platform for Scalable and "
    "Intelligent Embodied Systems, IROS 2025",
    url="https://doi.org/10.1109/IROS60139.2025.11247088",
    verified="2026-10-18",
    notes=("Sec. 3: 1,001,552 trajectories, 217 tasks, more than 100 homogeneous robots; abstract: "
           "30 percent average gain over pretraining on Open X-Embodiment."),
)
LIN_DATA_SCALING_2025 = _lit(
    "prov:lin-data-scaling-2025",
    "Lin, F. et al. (2025), Data Scaling Laws in Imitation Learning for Robotic Manipulation, ICLR 2025",
    url="https://arxiv.org/abs/2410.18647",
    verified="2026-10-18",
    notes=("Sec. 1: over 40,000 demonstrations; recipe of 32 environments with 50 demonstrations each; "
           "about 90 percent success in a new environment with a new object."),
)
BLACK_PI05_2025 = _lit(
    "prov:black-pi05-2025",
    "Black, K. et al. (2025), pi0.5: A Vision-Language-Action Model with Open-World Generalization, CoRL 2025",
    url="https://proceedings.mlr.press/v305/black25a.html",
    verified="2026-10-18",
    notes="Sec. I: about 400 hours of mobile manipulation in homes; Sec. V-B: 3 to 104 training locations.",
)
MANDLEKAR_MIMICGEN_2023 = _lit(
    "prov:mandlekar-mimicgen-2023",
    "Mandlekar, A. et al. (2023), MimicGen: A Data Generation System for Scalable Robot Learning using "
    "Human Demonstrations, CoRL 2023",
    url="https://proceedings.mlr.press/v229/mandlekar23a.html",
    verified="2026-10-18",
    notes="Abstract: over 50K demonstrations generated from about 200 human demonstrations across 18 tasks.",
)
MADDUKURI_COTRAINING_2025 = _lit(
    "prov:maddukuri-cotraining-2025",
    "Maddukuri, A. et al. (2025), Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic "
    "Manipulation, RSS 2025",
    url="https://doi.org/10.15607/RSS.2025.XXI.109",
    verified="2026-10-18",
    notes="Abstract: 38 percent average gain in real-robot success.",
)
HOQUE_THRIFTYDAGGER_2022 = _lit(
    "prov:hoque-thriftydagger-2022",
    "Hoque, R. et al. (2022), ThriftyDAgger: Budget-Aware Novelty and Risk Gating for Interactive "
    "Imitation Learning, CoRL 2021 (PMLR 164)",
    url="https://proceedings.mlr.press/v164/hoque22a.html",
    verified="2026-10-19",
    notes=("Table 3, physical cable routing, 15 autonomous trials each: behavior cloning 0, HG-DAgger 10, "
           "ThriftyDAgger 12; training interventions per episode 1.55 +/- 1.16 and 1.42 +/- 1.14."),
)
LEROBOT_ALOHA_STATIC_COFFEE = _ds(
    "prov:lerobot-aloha-static-coffee",
    "Hugging Face, lerobot/aloha_static_coffee, meta/info.json",
    "https://huggingface.co/datasets/lerobot/aloha_static_coffee",
    verified="2026-10-19",
    notes="total_episodes 50, fps 50, observation.state, observation.effort, and action of 14 values.",
)
ECFR_49_CFR_563_EDR = _report(
    "prov:ecfr-49-cfr-563-7",
    "eCFR, 49 CFR Part 563 (Event Data Recorders), Sec. 563.7 Table I and footnote 4",
    "https://www.ecfr.gov/current/title-49/subtitle-B/chapter-V/part-563",
    notes=("Speed, accelerator pedal, and service brake from -20 to 0 s at 10 samples/s (phase-in from "
           "2028); before the phase-in, -5.0 to 0 s at 2 samples/s. Retrieved 2026-10-19."),
)
NASA_MCO_PHASE1_1999 = _report(
    "prov:nasa-mco-phase1-1999",
    "Stephenson, A. G. et al. (1999), Mars Climate Orbiter Mishap Investigation Board Phase I Report, NASA",
    "https://llis.nasa.gov/llis_lib/pdf/1009464main1_0641-mr.pdf",
    notes="Planned orbit-insertion altitude 226 km; final estimate about 57 km.",
)

# Agent sandboxes (added 2026-10-20)
ZHANG_2024_CYBENCH = _lit(
    "prov:zhang-2024-cybench",
    "Zhang et al. (2024), Cybench: A Framework for Evaluating Cybersecurity Capabilities and "
    "Risks of Language Models",
    url="https://cybench.github.io/",
    verified="2026-10-12",
    notes="40 professional-level capture-the-flag tasks from four competitions (cybench.github.io).",
)

YOUNG_2019_GVISOR = _lit(
    "prov:young-2019-gvisor",
    "Young et al. (2019), The True Cost of Containing: A gVisor Case Study, HotCloud '19, "
    "Figure 3 (System Call Overhead)",
    url="https://www.usenix.org/system/files/hotcloud19-paper-young.pdf",
    verified="2026-10-20",
    notes=("Average gettimeofday latency over 100M executions, microseconds, as labeled on "
           "Figure 3: native 0.22, runc 0.29; gVisor on KVM: Sentry only 0.63, invoke on host "
           "5.96, help from the Gofer 45.5 (ptrace: 9.23, 9.71, 51.1). Read from the PDF."),
)

# --- On-device and federated learning results (added 2026-10-20) ---
HARD_2018_FEDERATED_KEYBOARD = _lit(
    "prov:hard-2018-federated-keyboard",
    "Hard et al. (2018), Federated Learning for Mobile Keyboard Prediction, arXiv:1811.03604, "
    "Section 5.2 and Tables 3, 4, and 6 (Gboard next-word CIFG model)",
    url="https://arxiv.org/abs/1811.03604",
    verified="2026-10-20",
    notes="Read from the arXiv PDF. Recall and CTR in percent; quoted errors are 95% CIs (jackknife).",
)
BONAWITZ_2019_FL_AT_SCALE = _lit(
    "prov:bonawitz-2019-fl-at-scale",
    "Bonawitz et al. (2019), Towards Federated Learning at Scale: System Design, MLSys 2019, "
    "Sections 6, 8 (footnote 3), 9, and Appendix A Table 1",
    url="https://arxiv.org/abs/1902.01046",
    verified="2026-10-20",
    notes="Read from the arXiv PDF. Operational figures of Google's production federated learning system.",
)
CAI_2020_TINYTL = _lit(
    "prov:cai-2020-tinytl",
    "Cai et al. (2020), TinyTL: Reduce Memory, Not Parameters for Efficient On-Device Learning, NeurIPS 2020, abstract",
    url="https://arxiv.org/abs/2007.11622",
    verified="2026-09-29",
    notes="Up to 6.5x training-memory saving against full fine-tuning; lite residual modules add 3.8% memory.",
)
MCMAHAN_2017_FEDAVG = _lit(
    "prov:mcmahan-2017-fedavg",
    "McMahan et al. (2017), Communication-Efficient Learning of Deep Networks from Decentralized Data, "
    "AISTATS 2017, Table 2 (MNIST CNN, rounds to 99% test accuracy)",
    url="https://arxiv.org/abs/1602.05629",
    verified="2026-10-20",
    notes="Read from the arXiv PDF; 100 clients, C = 0.1; u = E n / (K B) expected local updates per round.",
)

# --- Robot policy training: sim-to-real, parallel simulation, on-robot RL (added 2026-10-20) ---
ANDRYCHOWICZ_DACTYL_2020 = _lit(
    "prov:andrychowicz-dactyl-2020",
    "OpenAI: Andrychowicz, M. et al. (2020), Learning Dexterous In-Hand Manipulation, "
    "International Journal of Robotics Research 39(1), arXiv:1808.00177v5",
    url="https://arxiv.org/abs/1808.00177",
    verified="2026-10-18",
    notes=("24-DoF Shadow Dexterous Hand. Table 3: block (state), simulation median 50 consecutive "
           "rotations over 100 trials (the cap), locked wrist on the physical hand median 28.5. "
           "Tables 4 and 5: 10 physical trials per condition, sorted values as published. Sec. 6.3: "
           "no randomization trained in about 1.5 h (3 simulated years), all randomizations in about "
           "50 h (100 simulated years)."),
)
RUDIN_LEGGED_GYM_2022 = _lit(
    "prov:rudin-legged-gym-2022",
    "Rudin, N., Hoeller, D., Reist, P., and Hutter, M. (2022), Learning to Walk in Minutes Using "
    "Massively Parallel Deep Reinforcement Learning, CoRL 2021 (PMLR 164), arXiv:2109.11978",
    url="https://arxiv.org/abs/2109.11978",
    verified="2026-10-18",
    notes=("4,096 simulated ANYmal robots, 24 steps per robot per PPO update, 1,500 updates, under "
           "20 minutes on one RTX A6000; policy at 50 Hz."),
)
LUO_HILSERL_2025 = _lit(
    "prov:luo-hil-serl-2025",
    "Luo, J., Xu, C., Wu, J., and Levine, S. (2025), Precise and Dexterous Robotic Manipulation via "
    "Human-in-the-Loop Reinforcement Learning, Science Robotics, arXiv:2410.21845",
    url="https://arxiv.org/abs/2410.21845",
    verified="2026-10-18",
    notes=("1 to 2.5 h of real-world training per task; 20 to 30 teleoperated demonstrations seed the "
           "replay buffer. Figure 5 (top row) plots success as a running average over 20 episodes; "
           "RAM insertion reaches 100% within about 90 minutes and USB grasp-insert within about 150 "
           "(read from the figure)."),
)
LEVINE_GRASPING_FARM_2018 = _lit(
    "prov:levine-grasping-farm-2018",
    "Levine, S., Pastor, P., Krizhevsky, A., Ibarz, J., and Quillen, D. (2018), Learning Hand-Eye "
    "Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection, IJRR 37(4-5)",
    url="https://arxiv.org/abs/1603.02199",
    verified="2026-10-18",
    notes="Between 6 and 14 manipulators at a time collected over 800,000 grasp attempts over two months.",
)
KALASHNIKOV_QTOPT_2018 = _lit(
    "prov:kalashnikov-qt-opt-2018",
    "Kalashnikov, D. et al. (2018), QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based "
    "Robotic Manipulation, CoRL 2018, arXiv:1806.10293",
    url="https://arxiv.org/abs/1806.10293",
    verified="2026-10-18",
    notes="Over 580,000 real-world grasp attempts.",
)

# --- Network architectures lecture and chapter (added 2026-10-20) ---
JOHNSTON_NETFLIX_PRIZE_2012 = _report(
    "prov:johnston-netflix-prize-2012",
    "Johnston, C. (2012), Netflix Never Used Its $1 Million Algorithm Due to Engineering Costs, Ars Technica",
    "https://arstechnica.com/information-technology/2012/04/netflix-never-used-its-1-million-algorithm-due-to-engineering-costs/",
    notes="The 2009 Grand Prize: $1 million for a 10 percent accuracy improvement; the code was not deployed.",
)
AMATRIAIN_NETFLIX_RECOMMENDATIONS_2012 = _report(
    "prov:amatriain-netflix-recommendations-2012",
    "Amatriain, X. and Basilico, J. (2012), Netflix Recommendations: Beyond the 5 stars (Part 1), Netflix Tech Blog",
    "https://netflixtechblog.com/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429",
    notes=("2007 Progress Prize: a linear blend of 107 algorithms, 8.43 percent improvement; Netflix kept two "
           "(matrix factorization and an RBM) and rebuilt them for more than 5 billion ratings against the "
           "contest's 100 million. Retrieved 2026-10-18."),
)
NETFLIX_PRIZE_LEADERBOARD = _report(
    "prov:netflix-prize-leaderboard-2009",
    "Netflix Prize leaderboard, June 2009 (screenshot by Ivongala, Wikimedia Commons, public domain)",
    "https://commons.wikimedia.org/wiki/File:Netflix_prize.jpg",
    notes="Rank 1, BellKor's Pragmatic Chaos: best score 0.8558, 10.05 percent improvement, submitted 2009-06-26.",
)
KERAS_APPLICATIONS_TABLE = _report(
    "prov:keras-applications-table",
    "Keras Applications, table of available models (keras.io/api/applications)",
    "https://keras.io/api/applications/",
    notes=("Top-1 ImageNet accuracy and time per inference step: batch 32, mean of 30 batches times 10 "
           "repeats; CPU AMD EPYC (92 cores), GPU Tesla A100. Retrieved 2026-10-18."),
)
VASWANI_TRANSFORMER_2017 = _lit(
    "prov:vaswani-transformer-2017",
    "Vaswani et al. (2017), Attention Is All You Need, NeurIPS 2017",
    url="https://arxiv.org/abs/1706.03762",
    verified="2026-10-18",
    notes="Table 2: WMT 2014 English-to-German BLEU and training cost (FLOPs).",
)
HE_RESNET_CIFAR10_TABLE6 = _lit(
    "prov:he-resnet-2016-cifar10",
    "He et al. (2016), Deep Residual Learning for Image Recognition, CVPR 2016, Table 6 (CIFAR-10)",
    url="https://arxiv.org/abs/1512.03385",
    verified="2026-10-18",
    notes="Classification error on the CIFAR-10 test set; ResNet-20 8.75 percent, ResNet-56 6.97 percent.",
)

LECUN_ZIPCODE_1989 = _lit(
    "prov:lecun-zipcode-1989",
    "LeCun et al. (1989), Backpropagation Applied to Handwritten Zip Code Recognition, Neural Computation 1(4)",
    url="https://doi.org/10.1162/neco.1989.1.4.541",
    verified="2026-10-20",
    notes="USPS test digits: 1 percent error on the accepted digits after rejecting 12.1 percent.",
)

# --- ML operations at scale: platforms, data validation, outages, SLO alerting (added 2026-10-20) ---
META_FBLEARNER_FLOW_2016 = Provenance(
    id="prov:meta-fblearner-flow-2016",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="Dunn, J. (2016), Introducing FBLearner Flow: Facebook's AI backbone, Meta Engineering blog, 9 May 2016",
    url="https://engineering.fb.com/2016/05/09/core-infra/introducing-fblearner-flow-facebook-s-ai-backbone/",
    verified="2026-10-20",
    notes=("More than 1 million models trained; more than 500,000 workflow runs in April; fewer than 150 "
           "workflow authors; used by more than 25 percent of Facebook's engineering team; more than "
           "6 million predictions per second."),
)
BRECK_DATA_VALIDATION_2019 = _lit(
    "prov:breck-data-validation-2019",
    "Breck, Polyzotis, Roy, Whang, Zinkevich (2019), Data Validation for Machine Learning (MLSys), Table 1",
    url="https://mlsys.org/Conferences/2019/doc/2019/167.pdf",
    verified="2026-10-20",
    notes=("Over more than 700 Google pipelines and 30 days, the share of pipelines on which each anomaly "
           "type fired: new feature column 10%, out-of-domain categorical value 6%, missing feature column 6%, "
           "feature present in too few examples 3%, data completely missing 3%."),
)
BAYLOR_TFX_2017 = _lit(
    "prov:baylor-tfx-2017",
    "Baylor et al. (2017), TFX: A TensorFlow-Based Production-Scale Machine Learning Platform (KDD), Sec. 6.1 and 7",
    url="https://dl.acm.org/doi/10.1145/3097983.3098021",
    verified="2026-10-20",
    notes=("Model loading raised 99.9th-percentile inference latency to 500 to 1,500 ms; a dedicated "
           "loading threadpool cut it to 75 to 150 ms. Removing training-serving skew in Google Play raised "
           "the app install rate on the main page by 2 percent in an online A/B test."),
)
NETFLIX_INTERLEAVING_2017 = Provenance(
    id="prov:netflix-interleaving-2017",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="Netflix Technology Blog (2017), Innovating Faster on Personalization Algorithms at Netflix Using Interleaving",
    url="https://netflixtechblog.com/interleaving-in-online-experiments-at-netflix-a04ee392ec55",
    verified="2026-10-20",
    notes="Interleaving detected ranking preferences with more than 100 times fewer subscribers than an A/B test.",
)
CLOUDFLARE_META_OUTAGE_2021 = Provenance(
    id="prov:cloudflare-facebook-outage-2021",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="Cloudflare (2021), Understanding How Facebook Disappeared from the Internet, 4 October 2021",
    url="https://blog.cloudflare.com/october-2021-facebook-outage/",
    verified="2026-10-20",
    notes=("Times UTC: BGP routing changes peaked about 15:40; facebook.com stopped resolving on 1.1.1.1 "
           "about 15:50 and resolved again about 21:20; renewed BGP activity peaked at 21:17; resolvers saw "
           "about 30 times the usual DNS query volume."),
)
GOOGLE_SRE_SLO_ALERTING = Provenance(
    id="prov:google-sre-workbook-slo-alerting",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="Google, The Site Reliability Workbook (2018), ch. 5, Alerting on SLOs, multiwindow multi-burn-rate table",
    url="https://sre.google/workbook/alerting-on-slos/",
    verified="2026-10-20",
    notes=("For a 99.9 percent SLO over 30 days: page on burn rate 14.4 over 1 hour (2 percent of the budget) "
           "or 6 over 6 hours (5 percent); open a ticket on burn rate 1 over 3 days (10 percent)."),
)

# --- Network architectures, part 2: attention, decode, recommenders (added 2026-10-20) ---
OPENAI_DEVDAY_2023 = _report(
    "prov:openai-devday-2023",
    "OpenAI (2023), New models and developer products announced at DevDay, 6 November 2023",
    "https://openai.com/index/new-models-and-developer-products-announced-at-devday/",
    notes="GPT-4 Turbo supports a 128K context window.",
)
ANTHROPIC_CLAUDE_21_2023 = _report(
    "prov:anthropic-claude-2-1-2023",
    "Anthropic (2023), Introducing Claude 2.1, 21 November 2023",
    "https://www.anthropic.com/news/claude-2-1",
    notes="A 200K-token context window.",
)
GOOGLE_GEMINI_15_2024 = _report(
    "prov:google-gemini-1-5-2024",
    "Google (2024), Our next-generation model: Gemini 1.5, 15 February 2024",
    "https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/",
    notes=("Gemini 1.5 Pro ships with a standard 128,000-token context window; a limited group of "
           "developers could try up to 1 million tokens in private preview."),
)
DAO_FLASHATTENTION_2022 = _lit(
    "prov:dao-flashattention-2022",
    "Dao et al. (2022), FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, "
    "NeurIPS 2022, Tables 20 and 21",
    url="https://arxiv.org/abs/2205.14135",
    verified="2026-10-20",
    notes=("arXiv 2205.14135v2, Appendix E benchmarks: forward plus backward runtime (ms, no dropout or "
           "masking, Table 20) and memory (MB, Table 21) on one A100 40 GB, batch 16, 8 heads of 64, FP16. "
           "Standard PyTorch attention runs out of memory beyond 4,096 tokens."),
)
KWON_PAGEDATTENTION_2023 = _lit(
    "prov:kwon-pagedattention-2023",
    "Kwon et al. (2023), Efficient Memory Management for Large Language Model Serving with "
    "PagedAttention, SOSP 2023, Figure 1 (left)",
    url="https://arxiv.org/abs/2309.06180",
    verified="2026-10-20",
    notes=("Memory layout when serving a 13B-parameter model on an NVIDIA A100 40 GB: parameters 26 GB "
           "(65 percent), KV cache over 30 percent, the rest activations (figure labels)."),
)
GUPTA_RECOMMENDATION_HPCA_2020 = _lit(
    "prov:gupta-recommendation-hpca-2020",
    "Gupta et al. (2020), The Architectural Implications of Facebook's DNN-based Personalized "
    "Recommendation, HPCA 2020",
    url="https://arxiv.org/abs/1906.03109",
    verified="2026-10-20",
    notes="Recommendation models take 79 percent of the AI inference cycles in Facebook's datacenters.",
)
MUDIGERE_ZIONEX_2022 = _lit(
    "prov:mudigere-zionex-2022",
    "Mudigere et al. (2022), Software-Hardware Co-design for Fast and Scalable Training of Deep "
    "Learning Recommendation Models, ISCA 2022",
    url="https://arxiv.org/abs/2104.05158",
    verified="2026-10-20",
    notes='The abstract: "train very large DLRMs with up to 12 Trillion parameters".',
)
CRITEO_KAGGLE_DATASET = _lit(
    "prov:criteo-kaggle-2014",
    "Criteo Display Advertising Challenge dataset (Kaggle, 2014), as used by Naumov et al. (2019), DLRM",
    url="https://arxiv.org/abs/1906.00091",
    verified="2026-10-20",
    notes="Each example has 13 integer (dense) features and 26 categorical features.",
)

# ML compilers and fused kernels (added 2026-10-20)
ANSEL_2024_PYTORCH2 = _lit(
    "prov:ansel-2024-pytorch2",
    "Ansel et al. (2024), PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode "
    "Transformation and Graph Compilation, ASPLOS '24",
    url="https://docs.pytorch.org/assets/pytorch2-2.pdf",
    verified="2026-10-20",
    notes=("Table 3: geometric-mean speedups over eager on an NVIDIA A100, float32, TorchInductor "
           "(inference/training): TorchBench 2.73/1.38, Hugging Face 1.47/1.24, TIMM 2.48/1.38. "
           "Table 4: Hugging Face float16 inference, all optimizations 1.91, without cudagraphs 1.81, "
           "without fusion and inlining 0.80."),
)
DAO_2022_FLASHATTENTION = _lit(
    "prov:dao-2022-flashattention",
    "Dao et al. (2022), FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, NeurIPS 2022",
    url="https://arxiv.org/abs/2205.14135",
    verified="2026-10-20",
    notes=("Figure 2 (left), GPT-2 medium on an A100: HBM reads and writes 40.3 GB for standard "
           "attention against 4.4 GB for FlashAttention; Sec. 1: up to 9x fewer HBM accesses. "
           "Figure 2 (left) also gives 66.6 against 75.2 GFLOPs and 41.7 against 7.3 ms (forward and "
           "backward). Figure 1, left panel labels (A100): SRAM 19 TB/s (20 MB), HBM 1.5 TB/s (40 GB); "
           "right panel caption: 7.6x speedup on the attention computation of GPT-2 (added 2026-10-20)."),
)

# Attacks on and defenses of learned systems (added 2026-10-20).
CARLINI_STEALING_LM_2024 = Provenance(
    id="prov:carlini-stealing-production-lm-2024",
    kind=ProvenanceKind.LITERATURE,
    ref="Carlini, N. et al. (2024), Stealing Part of a Production Language Model, ICML 2024",
    url="https://arxiv.org/abs/2403.06634",
    verified="2026-10-20",
    notes=("Abstract: under $20 extracts the entire projection matrix of OpenAI's ada and babbage; under "
           "$2,000 estimated for gpt-3.5-turbo. Table: ada width 1024, < 2e6 queries, RMS 5e-4; babbage "
           "2048, < 4e6, 7e-4; babbage-002 1536, < 4e6; gpt-3.5-turbo-instruct and -1106 < 4e7 queries."),
)
CARLINI_EXTRACTING_GPT2_2021 = Provenance(
    id="prov:carlini-extracting-training-data-2021",
    kind=ProvenanceKind.LITERATURE,
    ref="Carlini, N. et al. (2021), Extracting Training Data from Large Language Models, USENIX Security 2021",
    url="https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-extracting",
    verified="2026-10-20",
    notes="600,000 GPT-2 samples generated; 1,800 top-ranked inspected; 604 confirmed memorized training text.",
)
SHAN_NIGHTSHADE_2024 = Provenance(
    id="prov:shan-nightshade-2024",
    kind=ProvenanceKind.LITERATURE,
    ref="Shan, S. et al. (2024), Nightshade: Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models, IEEE S&P 2024",
    url="https://arxiv.org/abs/2310.13828",
    verified="2026-10-20",
    notes="Stable Diffusion XL fine-tuned with 50, 100, and 300 poison images per concept (dose-response figure).",
)
CARLINI_WEB_SCALE_POISONING_2024 = Provenance(
    id="prov:carlini-web-scale-poisoning-2024",
    kind=ProvenanceKind.LITERATURE,
    ref="Carlini, N. et al. (2024), Poisoning Web-Scale Training Datasets is Practical, IEEE S&P 2024",
    url="https://arxiv.org/abs/2302.10149",
    verified="2026-10-20",
    notes="Abstract: for $60 an attacker could have poisoned 0.01 percent of the LAION-400M or COYO-700M datasets.",
)
SCHUHMANN_LAION_400M_2021 = Provenance(
    id="prov:schuhmann-laion-400m-2021",
    kind=ProvenanceKind.LITERATURE,
    ref="Schuhmann, C. et al. (2021), LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-Text Pairs",
    url="https://arxiv.org/abs/2111.02114",
    verified="2026-10-20",
)
ABADI_DPSGD_2016 = Provenance(
    id="prov:abadi-deep-learning-dp-2016",
    kind=ProvenanceKind.LITERATURE,
    ref="Abadi, M. et al. (2016), Deep Learning with Differential Privacy, ACM CCS 2016",
    url="https://arxiv.org/abs/1607.00133",
    verified="2026-10-20",
    notes=("q = 0.01, sigma = 4, delta = 1e-5: moments accountant epsilon 1.26 after 100 epochs and 2.55 after "
           "400; strong composition 9.34 and 24.22. Section 5.2: 90, 95, and 97 percent MNIST test accuracy for "
           "(0.5, 1e-5), (2, 1e-5), and (8, 1e-5)-DP (large, medium, small noise); MNIST nonprivate baseline "
           "98.30 percent; CIFAR-10 73 percent at epsilon 8."),
)
DE_DP_IMAGE_CLASSIFICATION_2022 = Provenance(
    id="prov:de-dp-image-classification-2022",
    kind=ProvenanceKind.LITERATURE,
    ref="De, S. et al. (2022), Unlocking High-Accuracy Differentially Private Image Classification through Scale",
    url="https://arxiv.org/abs/2204.13650",
    verified="2026-10-20",
    notes=("CIFAR-10 from scratch, WRN-16-4: 81.4 percent at epsilon 8. The paper states that the nonprivate "
           "WRN-16-4 achieves >94 percent test accuracy, citing Zagoruyko and Komodakis (2016); the nonprivate "
           "reference value is taken from that source (ZAGORUYKO_WRN_2016)."),
)
ZAGORUYKO_WRN_2016 = Provenance(
    id="prov:zagoruyko-wide-residual-networks-2016",
    kind=ProvenanceKind.LITERATURE,
    ref="Zagoruyko, S. and Komodakis, N. (2016), Wide Residual Networks, BMVC 2016, Table 6",
    url="https://arxiv.org/abs/1605.07146",
    verified="2026-10-20",
    notes=("Table 6 (effect of dropout in residual block; mean/std preprocessing, CIFAR numbers are the median of "
           "5 runs): depth 16, k 4, no dropout, CIFAR-10 test error 5.02 percent (5.24 with dropout). Stored as "
           "accuracy 100 - 5.02 = 94.98 percent. WRN-16-4 is the architecture De et al. (2022) train privately."),
)
NARAYANAN_NETFLIX_DEANON_2008 = Provenance(
    id="prov:narayanan-shmatikov-netflix-2008",
    kind=ProvenanceKind.LITERATURE,
    ref="Narayanan, A. and Shmatikov, V. (2008), Robust De-anonymization of Large Sparse Datasets, IEEE S&P 2008",
    url="https://arxiv.org/abs/cs/0610105",
    verified="2026-10-20",
    notes="8 movie ratings (2 may be wrong) with dates within 14 days uniquely identify 99 percent of records.",
)
ANTONAKAKIS_MIRAI_2017 = Provenance(
    id="prov:antonakakis-mirai-2017",
    kind=ProvenanceKind.LITERATURE,
    ref="Antonakakis, M. et al. (2017), Understanding the Mirai Botnet, USENIX Security 2017",
    url="https://www.usenix.org/conference/usenixsecurity17/technical-sessions/presentation/antonakakis",
    verified="2026-10-20",
    notes="Mirai peaked at about 600,000 infected devices.",
)
FARWELL_STUXNET_2011 = Provenance(
    id="prov:farwell-rohozinski-stuxnet-2011",
    kind=ProvenanceKind.LITERATURE,
    ref="Farwell, J. P. and Rohozinski, R. (2011), Stuxnet and the Future of Cyber War, Survival 53(1)",
    url="https://doi.org/10.1080/00396338.2011.555586",
    verified="2026-10-20",
    notes="Stuxnet exploited four zero-day vulnerabilities.",
)

# --- Robustness studies: physical attacks, FGSM, calibration, jailbreaks, backdoors (added 2026-10-20) ---
# Values transcribed from the cited tables as the Robust AI lecture quotes them.
EYKHOLT_2018_PHYSICAL_ATTACKS = _lit(
    "prov:eykholt-2018-physical-attacks",
    "Eykholt et al. (2018), Robust Physical-World Attacks on Deep Learning Visual Classification, CVPR 2018",
    url="https://arxiv.org/abs/1707.08945",
    verified="2026-10-20",
    notes=("LISA-CNN accuracy 91 percent on the LISA test set (Sec. 1); targeted Speed Limit 45 "
           "misclassification in 100 percent of lab images (abstract); drive-by field test, camouflage "
           "abstract art, 84.8 percent of frames (Table 3)."),
)
GOODFELLOW_2015_FGSM = _lit(
    "prov:goodfellow-2015-fgsm",
    "Goodfellow, Shlens, and Szegedy (2015), Explaining and Harnessing Adversarial Examples, ICLR 2015",
    url="https://arxiv.org/abs/1412.6572",
    verified="2026-10-20",
    notes=("Figure 1, GoogLeNet on ImageNet: panda at 57.7 percent confidence; adding 0.007 times the sign "
           "of the input gradient gives gibbon at 99.3 percent confidence."),
)
GUO_2017_CALIBRATION = _lit(
    "prov:guo-2017-calibration",
    "Guo, Pleiss, Sun, and Weinberger (2017), On Calibration of Modern Neural Networks, ICML 2017",
    url="https://arxiv.org/abs/1706.04599",
    verified="2026-10-20",
    notes=("Table 1, expected calibration error (percent, 15 bins), uncalibrated and after temperature "
           "scaling: CIFAR-100 LeNet 5 4.85 and 2.02; CIFAR-100 ResNet 110 16.53 and 1.26; ImageNet "
           "ResNet 152 5.48 and 1.86."),
)
ZOU_2023_GCG = _lit(
    "prov:zou-2023-gcg",
    "Zou et al. (2023), Universal and Transferable Adversarial Attacks on Aligned Language Models",
    url="https://arxiv.org/abs/2307.15043",
    verified="2026-10-20",
    notes=("Table 2, attack success rate (percent) over 388 harmful behaviors, GPT-3.5 and GPT-4: "
           "behavior only 1.8 and 8.0; behavior plus 'Sure, here's' 5.7 and 13.1; plus a GCG suffix "
           "optimized on Vicuna and Guanacos 47.4 and 29.1; plus an ensemble of GCG suffixes 86.6 and 46.9."),
)
GU_2017_BADNETS = _lit(
    "prov:gu-2017-badnets",
    "Gu, Dolan-Gavitt, and Garg (2017), BadNets: Identifying Vulnerabilities in the Machine Learning "
    "Model Supply Chain",
    url="https://arxiv.org/abs/1708.06733",
    verified="2026-10-20",
    notes=("Table 4, U.S. traffic signs, single-target attack with the yellow-square trigger on Faster "
           "R-CNN: baseline average accuracy on clean images 90.0 percent; BadNet 89.3 percent; backdoored "
           "stop signs classified as speed limits 90.3 percent. A Post-it note on a real stop sign was read "
           "as a speed limit with 95 percent confidence."),
)
NTSB_HWY18MH010_PRELIMINARY = _lit(
    "prov:ntsb-hwy18mh010-preliminary",
    "NTSB (2018), Preliminary Report, Highway HWY18MH010, Tempe, Arizona, May 2018",
    url="https://www.ntsb.gov/investigations/AccidentReports/Reports/HWY18MH010-prelim.pdf",
    verified="2026-10-20",
    notes=("Put the system's determination that emergency braking was needed at 1.3 seconds before "
           "impact; the final report (HAR-19/03) gives 1.2 seconds."),
)

GU_2025_PROMPT_CACHE_AUDIT = _lit(
    "prov:gu-2025-prompt-cache-audit",
    "Gu et al. (2025), Auditing Prompt Caching in Language Model APIs, ICML 2025, "
    "Abstract, Section 4.2, and Table 1",
    url="https://arxiv.org/abs/2502.07776",
    verified="2026-10-20",
    notes=("Audits run September to October 2024. Section 4.2: \"We detected prompt caching in 8 out "
           "of 17 API providers, and we detected global cache sharing in 7 providers.\" Read from "
           "arxiv.org/html/2502.07776v2. Added 2026-10-20."),
)

# --- Closed-loop evaluation sources (added 2026-10-20) ---
NHTSA_PE16007_2017 = Provenance(
    id="prov:nhtsa-pe16-007-2017",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="NHTSA Office of Defects Investigation, ODI Resume, Preliminary Evaluation PE 16-007, closing report (January 2017)",
    url="https://static.nhtsa.gov/odi/inv/2016/INCLA-PE16007-7876.PDF",
    verified="2026-10-19",
    notes=("Figure 11 and p. 10: airbag-deployment crashes per million miles, Tesla Model S and Model X, "
           "1.3 before and 0.8 after Autosteer installation, 'almost 40 percent'; footnote 22: all miles "
           "before and after installation, not only miles with Autopilot in use."),
)
QCS_AUTOSTEER_2019 = Provenance(
    id="prov:qcs-autosteer-2019",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="Quality Control Systems Corp. (Whitfield, R. A.), NHTSA's Implausible Safety Claim for Tesla's "
        "Autosteer Driver Assistance System (February 2019)",
    url="http://www.safetyresearch.net/Library/NHTSA_Autosteer_Safety_Claim.pdf",
    verified="2026-10-19",
    notes=("Replication from per-vehicle mileage released after a FOIA lawsuit: exact exposure before and "
           "after installation computable for 13 percent of vehicles; in that cohort the airbag-deployment "
           "rate rose from 0.76 to 1.21 per million miles. Values as cited in the chapter's War Story; "
           "URL not re-fetched 2026-10-20."),
)
KALRA_PADDOCK_RAND_2016 = _lit(
    "prov:kalra-paddock-rand-rr1478-2016",
    "Kalra, N. and Paddock, S. M. (2016), Driving to Safety: How Many Miles of Driving Would It Take to "
    "Demonstrate Autonomous Vehicle Reliability?, RAND Corporation RR-1478",
    url="https://www.rand.org/pubs/research_reports/RR1478.html",
    verified="2026-10-19",
    notes=("Table 1: benchmark rates per 100 million miles (fatalities 1.09, reported injuries 77, reported "
           "crashes 190); miles required, millions: (1) zero failures, 95 percent upper limit at the "
           "benchmark 275 / 3.9 / 1.6; (2) rate within 20 percent at 95 percent 8,800 / 125 / 51; (3) 20 "
           "percent better than human at 95 percent and 80 percent power 11,000 / 161 / 65."),
)
KUSANO_WAYMO_2025 = _lit(
    "prov:kusano-waymo-56m-2025",
    "Kusano, K. D. et al. (2025), Comparison of Waymo Rider-Only Crash Rates by Crash Type to Human "
    "Benchmarks at 56.7 Million Miles, Traffic Injury Prevention 26(sup1)",
    url="https://www.tandfonline.com/toc/gcpi20/26/sup1",
    verified="2026-10-19",
    notes=("56.7 million rider-only miles through January 2025; any-injury-reported crashed-vehicle rate 79 "
           "percent and airbag-deployment rate 81 percent below an aligned human benchmark."),
)
MANDLEKAR_ROBOMIMIC_2021 = _lit(
    "prov:mandlekar-robomimic-2021",
    "Mandlekar, A. et al. (2021), What Matters in Learning from Offline Human Demonstrations for Robot "
    "Manipulation, CoRL 2021, arXiv:2108.03298",
    url="https://arxiv.org/abs/2108.03298",
    verified="2026-10-20",
    notes=("Appendix G: low-dim BC-RNN, 30 percent validation split, 3 seeds; best checkpoint against the "
           "lowest-validation-loss checkpoint: Square (PH) 80.7 against 2.7 percent, Transport (PH) 64.0 "
           "against 0.7 percent success."),
)
TRI_LBM_2025 = _lit(
    "prov:tri-lbm-2025",
    "TRI LBM Team (2025), A Careful Examination of Large Behavior Models for Multitask Dexterous "
    "Manipulation, arXiv:2507.05331",
    url="https://arxiv.org/abs/2507.05331",
    verified="2026-10-20",
    notes="Section 1: '1,800 rigorously controlled real-world trials' with blind A/B testing.",
)
LI_SIMPLER_2024 = _lit(
    "prov:li-simpler-2024",
    "Li, X. et al. (2024), Evaluating Real-World Robot Manipulation Policies in Simulation (SIMPLER), "
    "CoRL 2024, arXiv:2405.05941",
    url="https://arxiv.org/abs/2405.05941",
    verified="2026-10-20",
    notes="Figure 1 caption: real and simulated success of the same policies 'over ~1500 evaluation episodes'.",
)

# --- Sensor perception sources (added 2026-10-20) ---------------------------
KITTI_GEIGER_2013 = _lit(
    "prov:kitti-geiger-2013",
    "Geiger, Lenz, Stiller, and Urtasun, Vision meets Robotics: The KITTI Dataset, "
    "International Journal of Robotics Research 32(11), 2013",
    url="https://www.cvlibs.net/publications/Geiger2013IJRR.pdf",
    verified="2026-10-20",
    notes=("Sensor setup (Sec. II): Velodyne HDL-64E rotating 3D laser scanner, 10 Hz, 64 beams, "
           "field of view 360 deg horizontal and 26.8 deg vertical, range 120 m."),
)

NTSB_HAR_17_02 = _lit(
    "prov:ntsb-har-17-02",
    "NTSB, Collision Between a Car Operating With Automated Vehicle Control Systems and a "
    "Tractor-Semitrailer Truck Near Williston, Florida, May 7, 2016, Highway Accident Report "
    "NTSB/HAR-17/02 (2017)",
    url="https://www.ntsb.gov/investigations/AccidentReports/Reports/HAR1702.pdf",
    verified="2026-10-19",
    notes=("The car's speed just before impact was 74 mph, with Traffic-Aware Cruise Control and "
           "Autosteer active (p. 2). Read for the Sensor Perception lecture plan, 2026-10-19."),
)

# --- Responsible AI sources (added 2026-10-20) -------------------------------
OFQUAL_GRADING_GAPS_2021 = _report(
    "prov:ofqual-grading-gaps-2021",
    "Ofqual (2021), Grading gaps in summer 2020: who was affected by differences between centre "
    "assessment grades and calculated grades? (July 2021), Crown Copyright, Open Government Licence v3.0",
    "https://www.gov.uk/government/publications/grading-gaps-in-summer-2020",
    notes=("A level, England. Table 1: row percentages of calculated grade against centre assessment "
           "grade (CAG), with N per CAG grade; cells shown as '~' (under 0.5 percent) are entered as 0. "
           "Table 2: candidates by the number of A levels whose CAG exceeded the calculated grade, "
           "rounded to the nearest 10. Text: calculated grade lower in 39 percent of entries, equal in "
           "59 percent, higher in just over 2 percent. Results were issued on 13 August 2020; on "
           "17 August 2020 every student was awarded the higher of the CAG and the calculated grade."),
)

OFQUAL_INTERIM_2020 = _report(
    "prov:ofqual-interim-2020",
    "Ofqual (2020), Awarding GCSE, AS, A level, advanced extension awards and extended project "
    "qualifications in summer 2020: interim report (August 2020)",
    "https://www.gov.uk/government/publications/awarding-gcse-as-a-levels-in-summer-2020-interim-report",
    notes="A level: the calculated grade was lower than the CAG in 39.1 percent of cases.",
)

PROPUBLICA_COMPAS_2016 = _lit(
    "prov:propublica-compas-2016",
    "Angwin, Larson, Mattu, and Kirchner (2016), Machine Bias, ProPublica; data and analysis "
    "propublica/compas-analysis, compas-scores-two-years.csv",
    url="https://github.com/propublica/compas-analysis",
    verified="2026-10-18",
    notes=("Confusion counts recomputed from compas-scores-two-years.csv (master) with ProPublica's "
           "own filters: |days_b_screening_arrest| <= 30, is_recid != -1, c_charge_degree != 'O', "
           "score_text != 'N/A'. Flagged = score_text Medium or High; outcome = two_year_recid. "
           "ProPublica's article rounds the false positive rates to 45 and 23 percent."),
)

OBERMEYER_2019 = _lit(
    "prov:obermeyer-2019",
    "Obermeyer, Powers, Vogeli, and Mullainathan (2019), Dissecting racial bias in an algorithm "
    "used to manage the health of populations, Science 366(6464), 447-453",
    url="https://doi.org/10.1126/science.aax2342",
    verified="2026-10-20",
    notes=("Simulated correction at the automatic-enrollment threshold: the fraction of Black "
           "patients among those auto-identified rises from 17.7 to 46.5 percent."),
)

NYDFS_APPLE_CARD_2021 = _report(
    "prov:nydfs-apple-card-2021",
    "New York State Department of Financial Services (2021), Report on Apple Card Investigation",
    "https://www.dfs.ny.gov/system/files/documents/2021/03/rpt_202103_apple_card_investigation.pdf",
    notes=("The complaint that opened the investigation (November 2019) reported a credit limit 20 "
           "times that offered to the complainant's spouse; DFS found no violation of fair-lending law."),
)

EEOC_UNIFORM_GUIDELINES_1978 = _report(
    "prov:eeoc-uniform-guidelines-1978",
    "Uniform Guidelines on Employee Selection Procedures (1978), 29 CFR 1607.4(D)",
    "https://www.ecfr.gov/current/title-29/subtitle-B/chapter-XIV/part-1607/section-1607.4",
    notes=("A selection rate for any group below four-fifths (80 percent) of the rate for the group "
           "with the highest rate is generally regarded as evidence of adverse impact (a U.S. "
           "employment screen, not a lending standard)."),
)

THINKING_MACHINES_2025_NONDETERMINISM = _report(
    "prov:thinking-machines-2025-nondeterminism",
    "Horace He and Thinking Machines Lab, Defeating Nondeterminism in LLM Inference, "
    "Thinking Machines Lab: Connectionism, 10 September 2025",
    "https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/",
    notes=(
        "Qwen/Qwen3-235B-A22B-Instruct-2507, 1,000 completions at temperature 0 of 'Tell me about "
        "Richard Feynman', 1,000 tokens each: 80 unique completions, the most common 78 times; "
        "identical for the first 102 tokens; 992 continue 'Queens, New York' and 8 'New York City'. "
        "With batch-invariant kernels all 1,000 are identical. Read 2026-10-20."
    ),
)

# --- Failure-recovery sources (added 2026-10-20) ----------------------------
SEC_2013_KNIGHT_CAPITAL = _lit(
    "prov:sec-2013-knight-capital",
    "U.S. Securities and Exchange Commission, In the Matter of Knight Capital Americas LLC, "
    "Release No. 34-70694, Administrative Proceeding File No. 3-15570 (16 October 2013)",
    url="https://www.sec.gov/files/litigation/admin/2013/34-70694.pdf",
    verified="2026-10-20",
    notes=("Read from the order's text: eight SMARS servers, the deployment missed one, the seven "
           "that received the new code (paras. 15, 16, 27); 212 incoming parent orders, 4 million "
           "executions in 154 stocks for more than 397 million shares in approximately 45 minutes, "
           "net long approximately $3.5 billion and net short approximately $3.15 billion, a $460 "
           "million realized loss (para. 17); 97 'Power Peg disabled' e-mails (para. 19)."),
)

SIGELMAN_2010_DAPPER = _lit(
    "prov:sigelman-2010-dapper",
    "Sigelman et al. (2010), Dapper, a Large-Scale Distributed Systems Tracing Infrastructure, "
    "Google Technical Report dapper-2010-1, Table 2 and Section 4.4",
    url="https://research.google/pubs/dapper-a-large-scale-distributed-systems-tracing-infrastructure/",
    verified="2026-10-20",
    notes=("Table 2: effect of non-adaptive sampling frequencies on the average latency and throughput "
           "of a Web search cluster (1/1: +16.3 percent latency, -1.48 percent throughput; 1/16: "
           "+2.12, -0.08; 1/1024: -0.20, -0.06); experimental errors 2.5 and 0.15 percent. Section "
           "4.4: the first production version averaged one sampled trace for every 1024 candidates. "
           "Read from the report PDF with pdftotext."),
)

OPENAI_2025_EXPANDING_SYCOPHANCY = _lit(
    "prov:openai-2025-expanding-sycophancy",
    "OpenAI, Expanding on what we missed with sycophancy (2 May 2025), section 'What we did to "
    "address the issue'",
    url="https://openai.com/index/expanding-on-sycophancy/",
    verified="2026-10-20",
    notes=("Rollout started Thursday 24 April and completed Friday 25 April 2025; two days of "
           "monitoring; system-prompt update late Sunday night; full rollback initiated Monday "
           "(28 April) and took around 24 hours. Read from the Wayback Machine copy of 2026-10-10."),
)

CROWDSTRIKE_2024_PIR = _lit(
    "prov:crowdstrike-2024-pir",
    "CrowdStrike, Preliminary Post Incident Review (PIR): Content Configuration Update Impacting "
    "the Falcon Sensor and the Windows Operating System (BSOD), executive summary (24 July 2024)",
    url="https://www.crowdstrike.com/wp-content/uploads/2024/07/CrowdStrike-PIR-Executive-Summary.pdf",
    verified="2026-10-20",
    notes="Update released 19 July 2024 at 04:09 UTC; hosts online between 04:09 and 05:27 UTC were affected.",
)

MICROSOFT_2024_CROWDSTRIKE_DEVICES = _lit(
    "prov:microsoft-2024-crowdstrike-devices",
    "Microsoft, Helping our customers through the CrowdStrike outage (blog, 20 July 2024)",
    url="https://blogs.microsoft.com/blog/2024/07/20/helping-our-customers-through-the-crowdstrike-outage/",
    verified="2026-10-20",
    notes="'We currently estimate that CrowdStrike's update affected 8.5 million Windows devices, or "
          "less than one percent of all Windows machines.'",
)

CEMRI_2025_MAST = _lit(
    "prov:cemri-2025-mast",
    "Cemri et al. (2025), Why Do Multi-Agent LLM Systems Fail?, arXiv:2503.13657v3",
    url="https://arxiv.org/abs/2503.13657",
    verified="2026-10-20",
    notes=("Read from the HTML version: 1642 annotated execution traces (Figure 1 caption, Section 1) "
           "from 7 MAS frameworks (Section 1); step repetition FM-1.3 at 15.7 percent (Section 4)."),
)

# --- Robot odometry, mapping, and tracking results (added 2026-10-20) ---
BORENSTEIN_UMBMARK_1996 = _lit(
    "prov:borenstein-umbmark-1996",
    "Borenstein, J. and Feng, L. (1996), Measurement and Correction of Systematic Odometry Errors in "
    "Mobile Robots, IEEE Transactions on Robotics and Automation 12(6), 869-880",
    url="https://doi.org/10.1109/70.544770",
    verified="2026-10-20",
    notes=("UMBmark: bidirectional 4 m x 4 m square path (16 m per run) at 0.2 m/s with on-the-spot "
           "turns, TRC LabMate differential-drive base. Table I, experiment 1 (Fig. 11): E_max,syst "
           "317 mm before and 21 mm after calibrating wheelbase and wheel-diameter ratio; run-to-run "
           "standard deviation of the return error about 25 mm (Secs. 3.3 and 5)."),
)

MILLANE_NVBLOX_2024 = _lit(
    "prov:millane-nvblox-2024",
    "Millane, A. et al. (2024), nvblox: GPU-Accelerated Incremental Signed Distance Field Mapping, "
    "IEEE ICRA 2024, arXiv:2311.00626",
    url="https://arxiv.org/abs/2311.00626",
    verified="2026-10-20",
    notes=("Table II, Average row: incremental ESDF update time 1.9 ms for nvblox (GPU), 13 ms for "
           "Fiesta and 58 ms for voxblox (CPU), over the Redwood and Cow and Lady sequences. Table III "
           "caption: about 3 ns per uncorrelated distance query on Jetson."),
)

BEWLEY_SORT_2016 = _lit(
    "prov:bewley-sort-2016",
    "Bewley, A. et al. (2016), Simple Online and Realtime Tracking, IEEE ICIP 2016; reference code "
    "abewley/sort, sort.py at commit 2236dff5019565958b84df7d871d41cc1db58ac7",
    url="https://github.com/abewley/sort/blob/2236dff5019565958b84df7d871d41cc1db58ac7/sort.py",
    verified="2026-10-20",
    notes=("Sort.__init__ defaults: max_age=1, min_hits=3, iou_threshold=0.3. Sort.update outputs a "
           "track only if it was updated this frame and its hit streak reaches min_hits (or the "
           "first min_hits frames of the sequence have not passed)."),
)

# --- Grounded intent sources (added 2026-10-20) ------------------------------
BROHAN_RT1_2023 = _lit(
    "prov:brohan-rt1-2023",
    "Brohan, A. et al. (2023), RT-1: Robotics Transformer for Real-World Control at Scale, RSS 2023, "
    "arXiv:2212.06817",
    url="https://arxiv.org/abs/2212.06817",
    verified="2026-10-20",
    notes=("Fig. 1(a) caption: the model outputs tokenized actions 'at 3 Hz'; Sec. 5.1: 'each action "
           "dimension in RT-1 is discretized into 256 bins'."),
)
BROHAN_RT2_2023 = _lit(
    "prov:brohan-rt2-2023",
    "Brohan, A. et al. (2023), RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic "
    "Control, arXiv:2307.15818",
    url="https://arxiv.org/abs/2307.15818",
    verified="2026-10-20",
    notes=("Sec. 3.3: RT-2-PaLI-X-55B 'can run at a frequency of 1-3 Hz', the 5B version 'at a "
           "frequency of around 5 Hz', in a multi-TPU cloud service."),
)
OPENVLA_ACTION_BINS_2024 = _lit(
    "prov:kim-openvla-action-bins-2024",
    "Kim, M. J. et al. (2024), OpenVLA: An Open-Source Vision-Language-Action Model, arXiv:2406.09246",
    url="https://arxiv.org/abs/2406.09246",
    verified="2026-10-20",
    notes="Sec. 3.2: each action dimension is discretized separately 'into one of 256 bins'.",
)
AHN_AUTORT_2024 = _lit(
    "prov:ahn-autort-2024",
    "Ahn, M. et al. (2024), AutoRT: Embodied Foundation Models for Large Scale Orchestration of "
    "Robotic Agents, arXiv:2401.12963",
    url="https://arxiv.org/abs/2401.12963",
    verified="2026-10-20",
    notes=("Sec. 5.3: 259 tasks from 64 scenes, a base rate of 228/259 acceptable, 200/214 after the "
           "LLM filter, which rejected 17 of the 31 unsuitable tasks; all 14 errors arose in teleop "
           "task sampling. The 228, 214, 200, and 17 were read from a fetched summary of the HTML, "
           "not verbatim."),
)
LIU_OKROBOT_2024 = _lit(
    "prov:liu-okrobot-2024",
    "Liu, P. et al. (2024), OK-Robot: What Really Matters in Integrating Open-Knowledge Models for "
    "Robotics, arXiv:2401.12202",
    url="https://arxiv.org/abs/2401.12202",
    verified="2026-10-20",
    notes=("Fig. 1 caption: 58.5% success across 10 unseen cluttered homes and 82.4% in decluttered "
           "ones (verified). The failure-analysis counts are read from Fig. 4 (an image, three "
           "cleanup levels pooled to 375 trials) as transcribed by the lecture authors on "
           "2026-10-19; the text states 171 trials for the as-found homes, so the counts were not "
           "re-verified against the text."),
)
REN_KNOWNO_2023 = _lit(
    "prov:ren-knowno-2023",
    "Ren, A. Z. et al. (2023), Robots That Ask for Help: Uncertainty Alignment for Large Language "
    "Model Planners (KnowNo), CoRL 2023, arXiv:2307.01928",
    url="https://arxiv.org/abs/2307.01928",
    verified="2026-10-20",
    notes=("Table 2, hardware mobile manipulation: KnowNo success 0.76 and help 0.67, Simple Set 0.75 "
           "and 0.81, No Help 0.51 (read from a fetched summary). The 0.85 target success and the "
           "Fig. 1 threshold 0.21 are as transcribed by the lecture authors; Fig. 1 is an image."),
)
BOHANNON_WALKING_1997 = _lit(
    "prov:bohannon-walking-1997",
    "Bohannon, R. W. (1997), Comfortable and maximum walking speed of adults aged 20-79 years: "
    "reference values and determinants, Age and Ageing 26(1):15-19",
    url="https://academic.oup.com/ageing/article/26/1/15/39390",
    verified="2026-10-20",
    notes=("Reference table (cm/s) as reproduced by Shirley Ryan AbilityLab's 10 Meter Walk Test page; "
           "the abstract confirms 253.3 cm/s maximum for men in their twenties."),
)

# --- Data selection: pruning, deduplication, echoing, and labels (added 2026-10-20) ---
SHUMAILOV_COLLAPSE_2024 = _lit(
    "prov:shumailov-model-collapse-2024",
    "Shumailov, Shumaylov, Zhao, Papernot, Anderson, and Gal (2024), AI models collapse when trained on "
    "recursively generated data, Nature 631, 755-759",
    url="https://www.nature.com/articles/s41586-024-07566-y",
    verified="2026-10-18",
    notes="OPT-125m fine-tuned on wikitext2, five runs per setting: the real-data model reaches a mean perplexity "
          "of 34; generations trained only on generated text lose 20 to 28 points; keeping 10 percent of the "
          "original data gives only minor degradation (figure 1b and 1c).",
)
PAUL_DATA_DIET_2021 = _lit(
    "prov:paul-data-diet-2021",
    "Paul, Ganguli, and Dziugaite (2021), Deep Learning on a Data Diet: Finding Important Examples Early in "
    "Training, NeurIPS 2021",
    url="https://arxiv.org/abs/2107.07075",
    verified="2026-10-18",
    notes="Abstract: half of CIFAR-10's training set can be pruned while slightly improving test accuracy; "
          "figure 1 scores EL2N at epoch 20.",
)
SORSCHER_PRUNING_2022 = _lit(
    "prov:sorscher-beyond-scaling-2022",
    "Sorscher, Geirhos, Shekhar, Ganguli, and Morcos (2022), Beyond neural scaling laws: beating power law "
    "scaling via data pruning, NeurIPS 2022",
    url="https://arxiv.org/abs/2206.14486",
    verified="2026-10-18",
    notes="Figure 5C caption: training on 80 percent of ImageNet approximates training on 100 percent; Sec. 6: "
          "20 percent of ImageNet can be discarded without sacrificing performance (ResNet-50).",
)
COLEMAN_SVP_2020 = _lit(
    "prov:coleman-selection-via-proxy-2020",
    "Coleman, Yeh, Mussmann, Mirzasoleiman, Bailis, Liang, Leskovec, and Zaharia (2020), Selection via Proxy: "
    "Efficient Data Selection for Deep Learning, ICLR 2020",
    url="https://arxiv.org/abs/1906.11829",
    verified="2026-10-18",
    notes="Abstract: proxies an order of magnitude faster to compute; active-learning selection runtime cut by "
          "about that factor with final error often within 0.1 percent; on CIFAR10, proxies over 10x faster to "
          "train remove up to 50 percent of the data without harming final accuracy, a 1.6x end-to-end gain. "
          "Five datasets.",
)
CHOI_DATA_ECHOING_2019 = _lit(
    "prov:choi-data-echoing-2019",
    "Choi, Passos, Shallue, and Dahl (2019), Faster Neural Network Training with Data Echoing, arXiv:1907.05550",
    url="https://arxiv.org/abs/1907.05550",
    verified="2026-10-18",
    notes="Sec. 3.2 and figure 4: ResNet-50 on ImageNet with upstream stages about 6 times slower than the "
          "accelerator; echo factors 2 to 5; fresh examples and wall time to 75 percent accuracy, five runs; up "
          "to 3.25x less wall time at echo factor 5.",
)
LEE_DEDUP_2022 = _lit(
    "prov:lee-deduplicating-training-data-2022",
    "Lee, Ippolito, Nystrom, Zhang, Eck, Callison-Burch, and Carlini (2022), Deduplicating Training Data Makes "
    "Language Models Better, ACL 2022",
    url="https://arxiv.org/abs/2107.06499",
    verified="2026-10-18",
    notes="Near-duplicate removal affected 3.04 percent of C4 and 13.63 percent of RealNews; footnote 1: a "
          "61-word sentence repeated 61,036 times in C4; over 4 percent of validation examples overlap training "
          "data; deduplicated models emit memorized text ten times less often.",
)
FINEWEB_2024 = Provenance(
    id="prov:fineweb-dataset-card-2024",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="Hugging Face (Penedo et al.), HuggingFaceFW/fineweb dataset card, revision 9bb295d; "
        "huggingface/datatrove examples/fineweb.py at commit acd431b",
    url="https://huggingface.co/datasets/HuggingFaceFW/fineweb/blob/9bb295ddab0e05d785b879661af7260fed5140fc/README.md",
    verified="2026-10-18",
    notes="About 15 trillion tokens from 96 Common Crawl snapshots; MinhashConfig(num_buckets=14, "
          "hashes_per_bucket=8, n_grams=5) in the datatrove pipeline that built it.",
)
SOHN_FIXMATCH_2020 = _lit(
    "prov:sohn-fixmatch-2020",
    "Sohn, Berthelot, Li, Zhang, Carlini, Cubuk, Kurakin, Zhang, and Raffel (2020), FixMatch: Simplifying "
    "Semi-Supervised Learning with Consistency and Confidence, NeurIPS 2020",
    url="https://arxiv.org/abs/2001.07685",
    verified="2026-10-20",
    notes="Table 2, CIFAR-10 error rates over 5 folds: FixMatch (RA) 13.81, 5.07, 4.26 and FixMatch (CTA) 11.39, "
          "5.07, 4.31 percent at 40, 250, and 4000 labels.",
)
CUBUK_AUTOAUGMENT_2019 = _lit(
    "prov:cubuk-autoaugment-2019",
    "Cubuk, Zoph, Mane, Vasudevan, and Le (2019), AutoAugment: Learning Augmentation Strategies from Data, "
    "CVPR 2019",
    url="https://arxiv.org/abs/1805.09501",
    verified="2026-10-18",
    notes="The ImageNet policy search used about 15,000 GPU-hours.",
)
YUN_CUTMIX_2019 = _lit(
    "prov:yun-cutmix-2019",
    "Yun, Han, Oh, Chun, Choe, and Yoo (2019), CutMix: Regularization Strategy to Train Strong Classifiers "
    "with Localizable Features, ICCV 2019",
    url="https://arxiv.org/abs/1905.04899",
    verified="2026-10-18",
    notes="ImageNet top-1 improvement of 2.28 percentage points for ResNet-50.",
)
TF_PROFILER_GUIDE = Provenance(
    id="prov:tensorflow-profiler-guide",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="TensorFlow guide, Optimize TensorFlow performance using the Profiler, input-pipeline analyzer screenshot "
        "(guide/images/tf_profiler/input_pipeline_analyzer.png)",
    url="https://www.tensorflow.org/guide/profiler",
    verified="2026-10-18",
    notes="Summary line: 81.4 percent of the total step time sampled is waiting for input. The step-time chart "
          "shows steps of about 1.1 s with about 0.2 s of device compute (read from the chart).",
)

# --- Training footprints and lifecycle carbon (added 2026-10-20) -------------
LLAMA31_MODEL_CARD_EMISSIONS = _ds(
    "prov:llama31-model-card-emissions",
    "Meta, Llama 3.1 model card (meta-llama/llama-models, models/llama3_1/MODEL_CARD.md), "
    "Training Greenhouse Gas Emissions table",
    "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md",
    verified="2026-10-20",
    notes=("Llama 3.1 405B row: 30.84M training GPU-hours at 700 W (H100-80GB TDP), 8,930 t CO2eq "
           "location-based, 0 t market-based (Meta matches its electricity with renewable purchases)."),
)

LUCCIONI_BLOOM_2023 = _lit(
    "prov:luccioni-bloom-2023",
    "Luccioni, Viguier, and Ligozat (2023), Estimating the Carbon Footprint of BLOOM, a 176B Parameter "
    "Language Model, JMLR 24(253) (arXiv:2211.02001)",
    url="https://arxiv.org/abs/2211.02001",
    verified="2026-10-20",
    notes=("Table 3: final BLOOM training on the French grid (57 gCO2eq/kWh); embodied 11.2 t of 50.5 t "
           "total, 22.2 percent. Table 5: project dynamic emissions by process."),
)

# --- Safety enforcement sources (added 2026-10-20) ---
LEVESON_TURNER_THERAC_1993 = _lit(
    "prov:leveson-turner-therac-1993",
    "Leveson and Turner, 'An Investigation of the Therac-25 Accidents,' IEEE Computer 26(7), 1993",
    url="https://doi.org/10.1109/MC.1993.274940",
    verified="2026-10-20",
    notes=(
        "Six overdose accidents, 1985 to 1987; doses estimated at 4,000 to 20,000 rads against a "
        "typical single therapeutic dose of about 200 rads (up to 100 times); three patients died "
        "of the overdoses; the 25 MeV accelerator. Transcribed from the Volume IV chapter's War "
        "Story; not re-read against the paper in this pass."
    ),
)
KIRSCHNER_ISO_TS_15066_2022 = _lit(
    "prov:kirschner-iso-ts-15066-2022",
    "Kirschner, Mansfeld, Abdolshah, and Haddadin, 'ISO/TS 15066: How Different Interpretations "
    "Affect Risk Assessment,' arXiv 2203.02706, 2022",
    url="https://arxiv.org/abs/2203.02706",
    verified="2026-10-20",
    notes=(
        "Table II, hand, interpretation B1: TS hand values, constrained hand, speeds set for a "
        "280 N threshold; peak forces measured with a Pilz PRMS2 device at a central (C) and an "
        "outer (N) position. Transcribed from the lecture deck's table; not re-read against the "
        "paper in this pass."
    ),
)
NHTSA_CRUISE_RECALL_23E086 = _lit(
    "prov:nhtsa-cruise-recall-23e086",
    "NHTSA recall 23E-086, Cruise LLC collision-detection subsystem (November 2023)",
    url="https://www.nhtsa.gov/?nhtsaId=23E086",
    verified="2026-10-20",
    notes="Software recall covering 950 vehicles. Transcribed from the lecture deck; not re-read.",
)
SICK_NANOSCAN3_CORE = _ds(
    "prov:sick-nanoscan3-nans3-caaz30aa1",
    "SICK nanoScan3 Core safety laser scanner, NANS3-CAAZ30AA1, product data sheet (2024-12-17)",
    "https://www.sick.com/us/en/catalog/products/safety/safety-laser-scanners/nanoscan3/c/g464152",
    verified="2026-10-20",
    notes=(
        "Protective field range 3 m, warning field range 10 m, response time 85 ms, protective "
        "field supplement 65 mm, SIL 2 (IEC 61508), PL d (EN ISO 13849), Type 3, scan angle "
        "275 deg. Transcribed from the lecture deck's datasheet excerpt; scan angle from the "
        "product family page; not re-read against the PDF in this pass."
    ),
)

# --- Agent evaluation results (added 2026-10-20) --------------------------------
YAO_2024_TAUBENCH = _lit(
    "prov:yao-2024-taubench",
    "Yao, Shinn, Razavi, and Narasimhan (2024), tau-bench: A Benchmark for Tool-Agent-User "
    "Interaction in Real-World Domains, arXiv:2406.12045, abstract",
    url="https://arxiv.org/abs/2406.12045",
    verified="2026-10-20",
    notes=("Abstract: state-of-the-art function-calling agents (like gpt-4o) succeed on <50% of the "
           "tasks and are inconsistent (pass^8 <25% in retail). Upper bounds, not point values."),
)

METR_2025_REWARD_HACKING = _lit(
    "prov:metr-2025-reward-hacking",
    "von Arx, Chan, and Barnes (METR, 5 June 2025), Recent Frontier Models Are Reward Hacking, "
    "o3 task table",
    url="https://metr.org/blog/2025-06-05-recent-reward-hacking/",
    verified="2026-10-20",
    notes=("o3 runs judged reward hacking over runs: RE-Bench total 39/128 (30.4%), HCAST total "
           "8/1087 (0.7%), Optimize LLM Foundry 21/21; the post states hacking was more than 43x "
           "more common on RE-Bench than on HCAST."),
)

ZHENG_2023_LLM_JUDGE = _lit(
    "prov:zheng-2023-llm-judge",
    "Zheng et al. (2023), Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena, NeurIPS 2023 "
    "Datasets and Benchmarks (arXiv:2306.05685), Table 2, default prompt",
    url="https://arxiv.org/abs/2306.05685",
    verified="2026-10-20",
    notes="Position bias: each answer pair judged twice with the order swapped; percentages of pairs.",
)

KAPOOR_2024_AGENTS_THAT_MATTER = _lit(
    "prov:kapoor-2024-agents-that-matter",
    "Kapoor, Stroebl, Siegel, Nadgir, and Narayanan (2024), AI Agents That Matter, arXiv:2407.01502, "
    "Table A1 (HumanEval)",
    url="https://arxiv.org/abs/2407.01502",
    verified="2026-10-20",
    notes=("Mean accuracy and mean total API cost over five runs on the 164 HumanEval problems "
           "(2024 prices). Retry, Warming, and Escalation are the authors' simple baselines; the LDB "
           "(GPT-4) row carries the paper's asterisk for a specification its original authors did not evaluate."),
)

# Fitting a training step on one accelerator (added 2026-10-20)
MICIKEVICIUS_MIXED_PRECISION_2018 = _lit(
    "prov:micikevicius-mixed-precision-2018",
    "Micikevicius et al. (2018), Mixed Precision Training, ICLR 2018",
    url="https://arxiv.org/abs/1710.03740",
    verified="2026-10-18",
    notes=("arXiv 1710.03740v3, page 4, Figure 3: histogram of activation-gradient magnitudes of the "
           "SSD detector during FP32 training, log2 bins; the caption gives 67 percent of the values as "
           "zero. Sec. 3.2: SSD diverges in FP16 without scaling; scaling the loss by 8 matches FP32. "
           "The bar heights in the series micikevicius_ssd_gradient_histogram are pixel rows read from "
           "the figure image (898 x 575 px)."),
)
CHEN_SUBLINEAR_MEMORY_2016 = _lit(
    "prov:chen-sublinear-memory-2016",
    "Chen et al. (2016), Training Deep Nets with Sublinear Memory Cost",
    url="https://arxiv.org/abs/1604.06174",
    verified="2026-10-18",
    notes=("Abstract: the memory of a 1,000-layer deep residual network falls from 48 GB to 7 GB at "
           "about 30 percent more running time."),
)
KORTHIKANTI_RECOMPUTATION_2023 = _lit(
    "prov:korthikanti-recomputation-2023",
    "Korthikanti et al. (2023), Reducing Activation Recomputation in Large Transformer Models, MLSys 2023",
    url="https://arxiv.org/abs/2205.05198",
    verified="2026-10-18",
    notes=("arXiv 2205.05198v1. Table 2: activation bytes per layer sbh(34 + 5as/h) storing "
           "everything, 34sbh with selective recomputation, 2sbh with full recomputation. Table 4: one "
           "layer of the 22B GPT on A100s, forward plus backward, 19.6 ms without recomputation, 27.2 ms "
           "with full recomputation, 20.9 ms with selective recomputation. Abstract: the 530B model on "
           "2,240 A100s reaches 54.2 percent MFU, against 42.1 percent with full recomputation."),
)
HUGGINGFACE_PERF_TRAIN_GPU_ONE_V5 = _lit(
    "prov:huggingface-perf-train-gpu-one-v5",
    "Hugging Face Transformers v5.0.0 documentation, Efficient training on a single GPU "
    "(docs/source/en/perf_train_gpu_one.md)",
    url="https://huggingface.co/docs/transformers/perf_train_gpu_one",
    verified="2026-10-18",
    notes=("Gradient checkpointing: \"slower training speed (~20%)\". The accumulation example sets "
           "per_device_train_batch_size=4 and gradient_accumulation_steps=16 (Apache-2.0)."),
)
ACCELERATOR_ONCHIP_SRAM_RANGE = _est(
    "prov:accelerator-onchip-sram-range",
    "MLSysBook, Model Training: on-chip SRAM of a modern accelerator",
    ("The book's rounded range for one accelerator chip's on-chip SRAM (registers, shared memory, L1 and "
     "L2 together): 30 to 50 MB at roughly 15 to 20 TB/s. The H100 has 50 MiB of L2 and 228 KiB of shared "
     "memory per SM across 132 SMs; Dao et al. (2022) label the A100's SRAM 20 MB at 19 TB/s."),
    verified="2026-10-20",
)

# --- Model compression results (Literature.CompressionResults, added 2026-10-20) ---
GALE_STATE_OF_SPARSITY_2019 = _lit(
    "prov:gale-state-of-sparsity-2019",
    "Gale, Elsen, and Hooker (2019), The State of Sparsity in Deep Neural Networks",
    url="https://arxiv.org/abs/1902.09574",
    verified="2026-10-20",
    notes="Magnitude pruning with gradual retraining removes about 80 percent of ResNet-50's weights for "
          "roughly one point of ImageNet top-1 accuracy (section 5 and figure 3).",
)
SANH_DISTILBERT_2019 = _lit(
    "prov:sanh-distilbert-2019",
    "Sanh, Debut, Chaumond, and Wolf (2019), DistilBERT, a distilled version of BERT",
    url="https://arxiv.org/abs/1910.01108",
    verified="2026-10-20",
    notes="Abstract: 40 percent smaller, 60 percent faster, 97 percent of BERT's language-understanding "
          "capability; section 3: the student keeps one of every two teacher layers (6 of 12).",
)
GALE_SPARSE_GPU_KERNELS_2020 = _lit(
    "prov:gale-sparse-gpu-kernels-2020",
    "Gale, Zaharia, Young, and Elsen (2020), Sparse GPU Kernels for Deep Learning, SC20",
    url="https://arxiv.org/abs/2006.10901",
    verified="2026-10-20",
    notes="Abstract: 1.2 to 2.1 times speedups over dense on sparse Transformer and MobileNet models on a "
          "V100, reaching 27 percent of single-precision peak.",
)
MISHRA_SPARSE_TENSOR_CORES_2021 = _lit(
    "prov:mishra-sparse-tensor-cores-2021",
    "Mishra et al. (2021), Accelerating Sparse Deep Neural Networks",
    url="https://arxiv.org/abs/2104.08378",
    verified="2026-10-20",
    notes="Table 1: 2x design-peak math throughput for 2:4 sparse tensor cores (A100); figure 3: INT8 "
          "sparse-over-dense GEMM speedup at M = N = 10240 as K grows, values read from the plot "
          "(approximate); Table 2: ResNet-50 76.1 percent dense, 76.2 percent 2:4 sparse.",
)
KRISHNAMOORTHI_QUANTIZATION_2018 = _lit(
    "prov:krishnamoorthi-quantization-2018",
    "Krishnamoorthi (2018), Quantizing deep convolutional networks for efficient inference: A whitepaper",
    url="https://arxiv.org/abs/1806.08342",
    verified="2026-10-20",
    notes="Table 3: TensorFlow-Slim ResNet-v1-50 top-1 75.2 percent in float, 75.0 percent with 8-bit "
          "per-channel weights and 8-bit activations after quantization-aware training.",
)
HAN_DEEP_COMPRESSION_2016 = _lit(
    "prov:han-deep-compression-2016",
    "Han, Mao, and Dally (2016), Deep Compression, ICLR 2016",
    url="https://arxiv.org/abs/1510.00149",
    verified="2026-10-20",
    notes="Figure 6 (AlexNet on ImageNet): accuracy loss against compression rate for pruning, "
          "quantization, both, and SVD; the points are the chapter's transcription of the figure, and "
          "the cliffs (about 15, 6, and 3 percent of the original size) are read from it.",
)
HAN_LEARNING_WEIGHTS_2015 = _lit(
    "prov:han-learning-weights-2015",
    "Han, Pool, Tran, and Dally (2015), Learning both Weights and Connections for Efficient Neural "
    "Networks, NeurIPS 2015",
    url="https://arxiv.org/abs/1506.02626",
    verified="2026-10-20",
    notes="Figure 5 (AlexNet, ImageNet top-5, L2-regularized runs): accuracy change against the share "
          "of parameters pruned, without retraining, with one retraining, and with iterative pruning "
          "and retraining; points taken from the vector paths of the published figure (arXiv v3, p. 7).",
)

# --- Hardware acceleration: TPU energy update and H100 latency measurements (added 2026-10-20) ---
JOUPPI_TPUV4I_ISCA_2021 = _lit(
    "prov:jouppi-tpuv4i-isca-2021",
    "Jouppi et al. (2021), Ten Lessons From Three Generations Shaped Google's TPUv4i, ISCA 2021",
    url="https://doi.org/10.1109/ISCA52012.2021.00010",
    verified="2026-10-20",
    notes=("Table 2 updates Horowitz's 45 nm energy table to 7 nm: FP32 multiply 3.7 -> 1.31 pJ; "
           "memory per 64-bit access, DDR3/4 DRAM (I/O only) 1,300 pJ at both nodes."),
)
CHIPS_AND_CHEESE_H100_2023 = _lit(
    "prov:chips-and-cheese-h100-2023",
    "Chips and Cheese (2023), Nvidia's H100: Funny L2, and Tons of Bandwidth",
    url="https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth",
    verified="2026-10-19",
    notes=("Pointer-chasing latency by test size on an H100 PCIe card; the plateaus are the chart's "
           "data labels: L1 22.4 ns, L2 159.7 ns, far L2 294.0 ns, HBM 377.9 ns."),
)
BOEHM_SGEMM_WORKLOG_2022 = _lit(
    "prov:boehm-sgemm-worklog-2022",
    "Boehm (2022), How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog",
    url="https://siboehm.com/articles/22/CUDA-MMM",
    verified="2026-10-19",
    notes="Results table, FP32 SGEMM on an A6000: naive 309.0 GFLOP/s through cuBLAS 23,249.6 GFLOP/s.",
)
NVIDIA_NSIGHT_COMPUTE_ROOFLINE_GUIDE = _ds(
    "prov:nvidia-nsight-compute-roofline-guide",
    "NVIDIA, Nsight Compute Profiling Guide, Roofline Charts (documentation v2026.3.1)",
    "https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts",
    verified="2026-10-19",
    notes="The example report lists DRAM throughput at 0.34 percent of peak.",
)

# --- Shared-hardware interference and placement evidence (added 2026-10-20) --
# Values as read from each source at the location named in the notes; the
# section and figure locations follow the citations already carried by the
# course material that quotes them.
VALSAN_TAMING_NONBLOCKING_CACHES_2016 = _lit(
    "prov:valsan-taming-nonblocking-caches-2016",
    "Valsan, P. K., Yun, H., and Farshchi, F. (2016), Taming Non-blocking Caches to Improve "
    "Isolation in Multicore Real-Time Systems, IEEE RTAS 2016",
    url="https://doi.org/10.1109/RTAS.2016.7461361",
    verified="2026-10-20",
    notes=("Abstract: up to 21X execution-time increase with cache partitioning. Exp. 2, three "
           "BwRead(DRAM) co-runners against a BwRead(LLC) victim: 2.1x Cortex-A9, 7.9x Nehalem "
           "(text), 10.6x Cortex-A15 (Fig. 2(b) label); write co-runners 21.4x on Cortex-A15 "
           "(Fig. 2(e) label)."),
)
BECHTEL_YUN_CACHE_DOS_2019 = _lit(
    "prov:bechtel-yun-cache-dos-2019",
    "Bechtel, M. and Yun, H. (2019), Denial-of-Service Attacks on Shared Cache in Multicore: "
    "Analysis and Prevention, IEEE RTAS 2019",
    url="https://doi.org/10.1109/RTAS.2019.00037",
    verified="2026-10-20",
    notes=("Sec. IV-C: up to 346X slowdown of a victim on a Raspberry Pi 3 (Cortex-A53). Sec. VI, "
           "MemGuard extended to writebacks, three BwWrite attackers: unregulated 345X, writes "
           "capped at 100 MB/s 1.34X, at 50 MB/s 1.2X."),
)
ICHNOWSKI_FOGROS2_2023 = _lit(
    "prov:ichnowski-fogros2-2023",
    "Ichnowski, J. et al. (2023), FogROS2: An Adaptive Platform for Cloud and Fog Robotics Using "
    "ROS 2, IEEE ICRA 2023",
    url="https://arxiv.org/abs/2205.09778",
    verified="2026-10-20",
    notes="Abstract: grasp planning takes 14 s on the robot and 1.2 s offloaded with FogROS2.",
)
BAO_DEMOS_LINUX_FREERTOS_ZCU104 = _ds(
    "prov:bao-demos-linux-freertos-zcu104",
    "Bao hypervisor demos, demos/linux+freertos/configs/zcu104.c (bao-project/bao-demos)",
    "https://github.com/bao-project/bao-demos",
    verified="2026-10-20",
    notes=("Linux VM: cpu_num = 3, region size 0x40000000; FreeRTOS VM: cpu_num = 1, region size "
           "0x8000000; shared IPC region size 0x00010000."),
)
LINUX_TEGRA234_P3701_DTS = _ds(
    "prov:linux-tegra234-p3701-dts",
    "Linux v6.12, arch/arm64/boot/dts/nvidia/tegra234-p3701-0000.dtsi (Jetson AGX Orin module)",
    "https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/arch/arm64/boot/dts/nvidia/tegra234-p3701-0000.dtsi?h=v6.12",
    verified="2026-10-20",
    notes="tj-thermal zone: polling-delay = <1000> (ms); active trips at 75000 and 95000 m-degC, hysteresis 4000.",
)

# --- Mapping a computation onto an accelerator (added 2026-10-20) ---
NVIDIA_MATMUL_BACKGROUND_GUIDE = _ds(
    "prov:nvidia-matmul-background-guide",
    "NVIDIA, Matrix Multiplication Background User's Guide (Deep Learning Performance documentation)",
    "https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html",
    verified="2026-10-17",
    notes=("Sec. 2.3 lists cuBLAS tiles from 256x128 (most efficient) to 64x64; Sec. 3.1 and figure 7 "
           "measure tile quantization on an A100-SXM4-80GB (cuBLAS 11.4, M = 27,648, K = 4,096, "
           "256x128 tiles forced; values read from the published SVG plot); Sec. 3.2 measures wave "
           "quantization (M = 2,304, N = 1,536 and 1,544, one 256x128 tile per SM); Table 1 gives the "
           "dimension multiples (FP16 multiples of 8, on A100 multiples of 64)."),
)
NVIDIA_CUDA_PROGRAMMING_GUIDE_CC90 = _ds(
    "prov:nvidia-cuda-programming-guide-cc90",
    "NVIDIA, CUDA C++ Programming Guide, compute capabilities, Tables 30 and 31 (compute capability 9.0)",
    "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities",
    verified="2026-10-18",
    notes=("64 K 32-bit registers per SM, at most 2,048 resident threads per SM, at most 255 "
           "registers per thread, warps of 32 threads."),
)
CHEN_EYERISS_ISCA_2016 = _lit(
    "prov:chen-eyeriss-isca-2016",
    "Chen, Emer, Sze (2016), Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for "
    "Convolutional Neural Networks, ISCA 2016",
    url="https://dspace.mit.edu/handle/1721.1/102369",
    verified="2026-10-17",
    notes=("Abstract: on AlexNet, row stationary is 1.4x to 2.5x more energy efficient than other "
           "dataflows in convolutional layers and at least 1.3x in fully connected layers for batch "
           "sizes larger than 16. The ISSCC 2016 test chip has 168 PEs in a 12 x 14 array."),
)
CHEN_TVM_OSDI_2018 = _lit(
    "prov:chen-tvm-osdi-2018",
    "Chen et al. (2018), TVM: An Automated End-to-End Optimizing Compiler for Deep Learning, OSDI 2018",
    url="https://arxiv.org/abs/1802.04799",
    verified="2026-10-17",
    notes="Figure 4: fused operators run 1.2x to 2x faster than unfused on an NVIDIA Titan X.",
)
ZHENG_ANSOR_OSDI_2020 = _lit(
    "prov:zheng-ansor-osdi-2020",
    "Zheng et al. (2020), Ansor: Generating High-Performance Tensor Programs for Deep Learning, OSDI 2020",
    url="https://arxiv.org/abs/2006.06762",
    verified="2026-10-17",
    notes="Abstract: up to 3.8x on Intel CPU, 2.6x on ARM CPU, and 1.7x on NVIDIA GPU over the prior best.",
)
PYTORCH_TF32_NOTE_V290 = _ds(
    "prov:pytorch-tf32-note-v290",
    "PyTorch documentation, CUDA semantics, TensorFloat-32 (TF32) on Ampere (and later) devices (v2.9.0)",
    "https://github.com/pytorch/pytorch/blob/v2.9.0/docs/source/notes/cuda.rst",
    verified="2026-10-19",
    notes=("Example: one 10,240-square matrix multiply on GA100 takes 0.016 s with TF32 allowed and "
           "0.11 s without."),
)

# --- Quantization (added 2026-10-20) ---
NAGEL_DATA_FREE_QUANTIZATION_2019 = _lit(
    "prov:nagel-data-free-quantization-2019",
    "Nagel, van Baalen, Blankevoort, and Welling (2019), Data-Free Quantization Through Weight "
    "Equalization and Bias Correction, ICCV 2019",
    url="https://arxiv.org/abs/1906.04721",
    verified="2026-10-20",
    notes="Table 1: MobileNetV2 71.72 percent in FP32 and 0.12 percent with per-layer INT8; section 3.1: "
          "the [-128, 128] and (-0.5, 0.5) channel example; figure 2: 32 output channels of the first "
          "depthwise layer.",
)
DETTMERS_LLM_INT8_2022 = _lit(
    "prov:dettmers-llm-int8-2022",
    "Dettmers, Lewis, Belkada, and Zettlemoyer (2022), LLM.int8(): 8-bit Matrix Multiplication for "
    "Transformers at Scale, NeurIPS 2022",
    url="https://arxiv.org/abs/2208.07339",
    verified="2026-10-20",
    notes="Section 1: outlier features up to 20 times larger, in about 6 feature dimensions, affecting "
          "75 percent of sequence positions from 6.7B parameters; section 4.1: magnitude threshold 6.",
)
FRANTAR_GPTQ_2023 = _lit(
    "prov:frantar-gptq-2023",
    "Frantar, Ashkboos, Hoefler, and Alistarh (2023), GPTQ: Accurate Post-Training Quantization for "
    "Generative Pre-trained Transformers, ICLR 2023",
    url="https://arxiv.org/abs/2210.17323",
    verified="2026-10-20",
    notes="Table 5: OPT-175B WikiText2 perplexity (FP16 8.34; 4-bit RTN 10.54, GPTQ 8.37; 3-bit RTN "
          "7.3e3, GPTQ 8.68); Table 6: 230 ms per token in FP16 on five A100s, 71 ms with 3-bit "
          "weights on one.",
)
LLAMA_CPP_GGML_BLOCKS = Provenance(
    id="prov:llama-cpp-ggml-blocks",
    kind=ProvenanceKind.DATASHEET,
    ref="ggml-org/llama.cpp v0.7.0, ggml/src/ggml-common.h: block_q4_0 and block_q8_0 (MIT)",
    url="https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h",
    verified="2026-10-18",
    notes="QK4_0 = QK8_0 = 32 quants per block and one ggml_half (16-bit) scale d.",
)
THEBLOKE_LLAMA2_GGUF = Provenance(
    id="prov:thebloke-llama2-7b-gguf",
    kind=ProvenanceKind.DATASHEET,
    ref="TheBloke/Llama-2-7B-GGUF on Hugging Face, README (commit b4e04e1)",
    url="https://huggingface.co/TheBloke/Llama-2-7B-GGUF",
    verified="2026-10-18",
    notes="Explanation of quantization methods: GGML_TYPE_Q2_K ends up using 2.5625 bits per weight.",
)
LLAMA_BENCH_APPLE_SILICON = Provenance(
    id="prov:llama-bench-apple-silicon",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="ggml-org/llama.cpp, discussion 4167, Performance of llama.cpp on Apple Silicon M-series",
    url="https://github.com/ggml-org/llama.cpp/discussions/4167",
    verified="2026-10-18",
    notes="llama-bench rows for Llama 2 7B report params 6.74 B.",
)

# --- Supervisory intervention: takeover studies, regulations, embedded crypto (added 2026-10-20) ---
OWENS_2024_ED25519_M4 = _lit(
    "prov:owens-2024-ed25519-cortex-m4",
    "Owens, D., et al. (2024), Efficient and Side-Channel Resistant Ed25519 on ARM Cortex-M4",
    url="https://par.nsf.gov/servlets/purl/10507285",
    verified="2026-10-20",
    notes=("Abstract: 'Verify in 720,000 cycles on the ARM Cortex-M4' (optimized assembly). Table II "
           "(kilo-cycles) prints 722 at 24 MHz and 760 at 168 MHz; the abstract's 720,000 is stored."),
)
SCHWABE_STOFFELEN_2016_AES_M4 = _lit(
    "prov:schwabe-stoffelen-2016-aes-cortex-m",
    "Schwabe, P., and Stoffelen, K. (2016), All the AES You Need on Cortex-M3 and M4, SAC 2016",
    url="https://cryptojedi.org/papers/aesarm-20161019.pdf",
    verified="2026-10-20",
    notes=("Table 1 (Performance of unprotected AES), p. 6: AES-128 single block encryption, 661.7 "
           "cycles on Cortex-M4 (659.4 on M3); unprotected table-based implementation."),
)
LIU_2023_SIRIUS = _lit(
    "prov:liu-2023-sirius",
    "Liu, H., Nasiriany, S., Zhang, L., Bao, Z., and Zhu, Y. (2023), Robot Learning on the Job: "
    "Human-in-the-Loop Autonomy and Learning During Deployment (Sirius), RSS 2023, arXiv:2211.08416",
    url="https://arxiv.org/abs/2211.08416",
    verified="2026-10-20",
    notes=("Sec. IV-D: 'average reaction time of 2 seconds, roughly corresponding to the time of 15 "
           "robot actions. We thus set l = 15.'"),
)
MERAT_2014_TRANSITION = _lit(
    "prov:merat-2014-transition-to-manual",
    "Merat, N., Jamson, A. H., Lai, F. C. H., Daly, M., and Carsten, O. M. J. (2014), Transition to "
    "manual: Driver behaviour when resuming control from a highly automated vehicle, "
    "Transportation Research Part F 27:274-282",
    url="https://eprints.whiterose.ac.uk/84460",
    verified="2026-10-20",
    notes="Abstract: 'it took drivers around 35-40 s to stabilise their lateral control of the vehicle.'",
)
UNECE_R157_ALKS = _report(
    "prov:unece-r157-alks-2021",
    "UN Regulation No. 157, Automated Lane Keeping Systems (ALKS), OJ L 82, 9.3.2021",
    "https://op.europa.eu/o/opportal-service/download-handler?identifier=36fd3041-807a-11eb-9ac9-01aa75ed71a1&format=pdf&language=en&productionSystem=cellar",
    notes=("Para 5.4.4.1: a minimum risk manoeuvre 'shall be started, earliest 10 s after the start of "
           "the transition demand'. Para 8.3.1(d)(ii): DSSAD timestamp accuracy +/- 1,0 s. Read 2026-10-20."),
)
ISO_10218_1_MANUAL_REDUCED_SPEED = _report(
    "prov:iso-10218-1-reduced-speed",
    "ISO 10218-1, Robotics - Safety requirements - Part 1: Industrial robots, reduced speed in manual mode",
    "https://www.iso.org/standard/73933.html",
    notes=("Reduced speed in manual mode: tool centre point speed not exceeding 250 mm/s (2011 edition, "
           "clause 5.6.2, as quoted by vendor training material). The standard text is paywalled and was "
           "not re-read for the 2025 edition; needs check against ISO 10218-1:2025."),
)
CA_DMV_2024_DISENGAGEMENT_CSV = _report(
    "prov:ca-dmv-2024-disengagement-csv",
    "California DMV, 2024 Autonomous Vehicle Disengagement Reports (CSV), report year Dec 2023 to Nov 2024",
    "https://web.archive.org/web/20250213101223/https://www.dmv.ca.gov/portal/file/2024-autonomous-vehicle-disengagement-reports-csv/",
    notes=("Waymo LLC rows: 244, of which 230 initiated by Test Driver and 14 by AV System (counted from "
           "the file, retrieved 2026-10-19)."),
)
WARSAW_DLH2904_REPORT = _report(
    "prov:warsaw-dlh2904-report-1994",
    "Main Commission for Aircraft Accident Investigation (Poland), Report on the Accident to Airbus A320-211 "
    "D-AIPN at Warsaw Okecie Airport on 14 September 1993 (English translation)",
    "http://www.rvs.uni-bielefeld.de/publications/Incidents/DOCS/ComAndRep/Warsaw/warsaw-report.html",
    notes=("Sec. 1.1: the left gear touched the runway 9 seconds after the right; the wheel brakes 'began to "
           "operate after about 4 seconds'; the aircraft rolled over the end of the runway at GS = 72 kts. "
           "Sec. 1.2: fatal injuries 1 crew, 1 passenger. Read 2026-10-20."),
)

# --- Fault injection and attacks on robot proposers (added 2026-10-20) ---
JHA_2019_DRIVEFI = _lit(
    "prov:jha-2019-drivefi",
    "Jha, S. et al. (2019), ML-based Fault Injection for Autonomous Vehicles: A Case for Bayesian Fault "
    "Injection (DriveFI), DSN 2019",
    url="https://arxiv.org/abs/1907.01051",
    verified="2026-10-20",
    notes=("Read in the arXiv HTML of v1. Sec. I, Results: 98,400 faults; exhaustive evaluation 615 days; "
           "561 safety-critical faults in less than 4 hours; 3690x acceleration; 460 of the 561 manifested "
           "as hazards, in 68 of 7,200 scenes; several weeks of 5,000 random FI experiments found no hazard; "
           "1.93 percent of single-bit GPU injections led to SDC and 7.35 percent to kernel panics and hangs. "
           "Sec. VI-A: 2,400 GPU FI experiments into DriveAV."),
)
BROWN_2017_ADVERSARIAL_PATCH = _lit(
    "prov:brown-2017-adversarial-patch",
    "Brown, T. B., Mane, D., Roy, A., Abadi, M., Gilmer, J. (2017), Adversarial Patch, arXiv:1712.09665",
    url="https://arxiv.org/abs/1712.09665",
    verified="2026-10-20",
    notes="Figure 1 caption: VGG16 reports banana at 97 percent confidence; with the patch, toaster at 99 percent.",
)
ROBEY_2025_ROBOPAIR = _lit(
    "prov:robey-2025-robopair",
    "Robey, A. et al. (2025), Jailbreaking LLM-Controlled Robots, ICRA 2025, arXiv:2410.13691",
    url="https://arxiv.org/abs/2410.13691",
    verified="2026-10-20",
    notes=("Aggregate rows of 35 harmful tasks. Table 2 (Clearpath Jackal UGV, GPT-4o planner): direct prompt "
           "1/35, RoboPAIR 35/35. Table 3 (Unitree Go2): direct prompt 3/35, RoboPAIR 35/35."),
)
SRS_2013_TOYOTA_BOOKOUT = _lit(
    "prov:srs-2013-toyota-bookout",
    "Safety Research & Strategies (2013), Toyota Unintended Acceleration and the Big Bowl of 'Spaghetti' Code",
    url="https://www.safetyresearch.net/toyota-unintended-acceleration-and-the-big-bowl-of-spaghetti-code/",
    verified="2026-10-20",
    notes=("Opening paragraph: the Oklahoma jury found reckless disregard and delivered a $3 million verdict "
           "before it could determine punitive damages (Bookout v. Toyota, 2013)."),
)
LEIDECKER_2011_TIN_WHISKER = _lit(
    "prov:leidecker-2011-tin-whisker",
    "Leidecker, H., Panashchenko, L., Brusse, J. (2011), Electrical Failure of an Accelerator Pedal Position "
    "Sensor Caused by a Tin Whisker, 5th International Tin Whisker Symposium",
    url="https://nepp.nasa.gov/whisker/reference/tech_papers/2011-NASA-GSFC-whisker-failure-app-sensor.pdf",
    verified="2026-10-20",
    notes="Extract from NASA findings: a tin whisker formed a 248 ohm resistive short in a 2003 Camry APP sensor.",
)
OKELLY_2018_RARE_EVENT = _lit(
    "prov:okelly-2018-rare-event",
    "O'Kelly, M., Sinha, A., Namkoong, H., Duchi, J., Tedrake, R. (2018), Scalable End-to-End Autonomous "
    "Vehicle Testing via Rare-event Simulation, NeurIPS 2018",
    url="https://arxiv.org/abs/1811.00145",
    verified="2026-10-20",
    notes=("Sec. 4: the cross-entropy method 'achieves 2-16 times variance reduction depending on the threshold "
           "level' (Figure 2). Table 1 is Literature.OKelly2018RareEventTable1."),
)
