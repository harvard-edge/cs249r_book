from pydantic import BaseModel, ConfigDict
from typing import Optional, Union
from ..core.units import Q_
from ..core.types import Quantity
from ..models.types import Workload
from ..hardware.types import HardwareNode
from ..systems.types import Fleet
from ..core.exceptions import OOMError, SLAViolation
from ..core.registry import Registry
from .evaluation import SystemEvaluation, EvaluationLevel
from ..models.registry import Models as _Models
from ..hardware.registry import Hardware as _Hardware
from ..systems.registry import Clusters as _Clusters
from ..reference_stats.registry import ReferenceStats as _ReferenceStats
from ..systems.registry import Fabrics as _Fabrics, Nodes as _Nodes, NetworkEnergy as _NetworkEnergy
from ..systems.registry import SwitchFabric as _SwitchFabric
from ..systems.registry import Storage as _Storage
from ..datasets.registry import Datasets as _Datasets
from ..physics.quantities import memory_from_params as _memory_from_params
from ..core.units import BYTES_FP16 as _BYTES_FP16, param as _param
from ..core.units import BYTES_FP32 as _BYTES_FP32, BYTES_INT8 as _BYTES_INT8
from ..core.units import BYTES_TRAINING_STATE_MIXED_PRECISION as _BYTES_TRAINING_STATE
from ..literature.registry import Literature as _Literature
from ..systems.reliability import Reliability as _Reliability
from ..infrastructure.registry import Infrastructure as _Infrastructure
from ..ops import Ops as _Ops
from ..embodied.registry import Embodied as _Embodied  # embodied stopping scenarios (2026-10-20)
from ..agents.registry import Agents as _Agents  # agent harness profiles (added 2026-10-20)
from . import scenario_types as _st
from ..core.units import COLOR_DEPTH_8BIT as _COLOR_DEPTH_8BIT  # added 2026-10-20
from ..core.units import DAYS_PER_MONTH as _DAYS_PER_MONTH  # added 2026-10-20
# Deployment paradigms block (added 2026-10-20)
from ..systems.registry import Pods as _Pods
from ..core.units import (VIDEO_1080P_WIDTH as _VIDEO_1080P_WIDTH, VIDEO_1080P_HEIGHT as _VIDEO_1080P_HEIGHT,
                          VIDEO_4K_WIDTH as _VIDEO_4K_WIDTH, VIDEO_4K_HEIGHT as _VIDEO_4K_HEIGHT,
                          VIDEO_BYTES_PER_PIXEL_RGB as _VIDEO_BYTES_PER_PIXEL_RGB, VIDEO_FPS_STANDARD as _VIDEO_FPS_STANDARD)
# Physical-plant budgets (added 2026-10-20)
from ..embodied.registry import Embodied as _Embodied
from ..actuators.registry import Actuators as _Actuators
from ..physics.constants import INSULATION_CLASS_F_LIMIT as _CLASS_F
from ..physics.drivetrain import calc_traction_deceleration_limit as _traction_limit

class Scenario(BaseModel):
    """
    Concrete executable case: a workload on a system with constraints.

    ``application`` names the product/domain context, while the scenario itself
    is what the simulator can evaluate. Model facts stay in ``Models.*``;
    hardware and fleet facts stay in ``Hardware.*`` / ``Systems.*``.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    application: Optional[str] = None
    workload: Workload
    system: Union[Fleet, HardwareNode]
    
    # Constraints (SLAs)
    sla_latency: Optional[Quantity] = None
    target_accuracy: Optional[float] = None
    power_budget: Optional[Quantity] = None

    @property
    def is_distributed(self) -> bool:
        return isinstance(self.system, Fleet)

    def evaluate(self, batch_size: int = 1, precision: str = "fp16") -> SystemEvaluation:
        """
        Runs a full multi-level evaluation of the scenario.
        """
        from .solvers import DistributedModel, EconomicsModel
        
        # 1. Resolve Hardware
        hardware = self.system.node.accelerator if self.is_distributed else self.system
        
        # --- LEVEL 1: FEASIBILITY ---
        from .solvers import DataModel
        weights = self.workload.size_in_bytes(precision=precision)
        mem_feasible = weights <= hardware.memory.capacity
        
        # Data Pipeline Check
        data_status = "PASS"
        data_summary = ""
        if self.workload.data_rate:
            ds = DataModel().solve(self.workload.data_rate, hardware)
            if ds.is_stalled:
                data_status = "FAIL"
                data_summary = f" | Data Wall: Pipeline Stalled ({ds.utilization:.1f}x capacity)"
            else:
                data_summary = f" | Data Pipeline: OK ({ds.utilization*100:.1f}%)"

        feasible = mem_feasible and (data_status == "PASS")
        f_status = "PASS" if feasible else "FAIL"
        
        # Dynamic unit scaling for summary
        unit = "MB" if weights < Q_("1 GB") else "GB"
        mem_summary = f"Model fits in memory ({weights.to(unit):.1f~P} / {hardware.memory.capacity.to(unit):.1f~P})" if mem_feasible else f"OOM: Requires {weights.to(unit):.1f~P} but only has {hardware.memory.capacity.to(unit):.1f~P}"
        
        l1 = EvaluationLevel(
            level_name="Feasibility", 
            status=f_status, 
            summary=mem_summary + data_summary,
            metrics={
                "weight_size": weights,
                "capacity": hardware.memory.capacity,
                "data_stalled": data_status == "FAIL",
            }
        )

        # --- LEVEL 2: PERFORMANCE ---
        if self.is_distributed:
            solver = DistributedModel()
            perf = solver.solve(self.workload, self.system, batch_size=batch_size, precision=precision)
            actual_latency = perf.step_latency_total
            throughput = perf.effective_throughput
            perf_metrics = {
                "latency": actual_latency,
                "throughput": throughput,
                "scaling_eff": perf.scaling_efficiency,
                "mfu": perf.node_profile.mfu * perf.scaling_efficiency,
                "node_mfu": perf.node_profile.mfu,
                "sla_latency": self.sla_latency
            }
        else:
            from .solvers import SingleNodeModel
            perf = SingleNodeModel().solve(self.workload, self.system, batch_size=batch_size, precision=precision)
            actual_latency = perf.latency
            throughput = perf.throughput
            perf_metrics = {
                "latency": actual_latency, 
                "throughput": throughput, 
                "bottleneck": perf.bottleneck,
                "sla_latency": self.sla_latency
            }

        p_status = "PASS"
        if self.sla_latency and actual_latency > self.sla_latency:
            p_status = "FAIL"
        
        p_summary = f"Latency: {actual_latency:.2f~P} (Target: {self.sla_latency or 'N/A'})"
        l2 = EvaluationLevel(level_name="Performance", status=p_status, summary=p_summary, metrics=perf_metrics)

        # --- LEVEL 3: MACRO ---
        # Scale to 1 year operation for macro view
        if self.is_distributed:
             sim_fleet = self.system
        else:
             from ..systems.types import Node, Fleet
             from ..systems.registry import Fabrics
             dummy_node = Node(name="Standard", accelerator=hardware, accelerators_per_node=1, intra_node_bw="50 GB/s")
             sim_fleet = Fleet(name="SimFleet", node=dummy_node, count=1, fabric=Fabrics.Ethernet_10G)

        # EconomicsModel internally delegates to SustainabilityModel and merges
        # its results, so we only need one call (avoids computing energy twice).
        econ = EconomicsModel().solve(sim_fleet, duration_days=365)

        m_summary = f"Annual Carbon: {econ.carbon_footprint_kg:.1f} kg | TCO: ${econ.tco_usd:,.0f}"
        l3 = EvaluationLevel(
            level_name="Macro",
            status="PASS",
            summary=m_summary,
            metrics={"carbon_kg": econ.carbon_footprint_kg, "tco_usd": econ.tco_usd}
        )

        return SystemEvaluation(
            scenario_name=self.name,
            feasibility=l1,
            performance=l2,
            macro=l3
        )

    def validate(self, batch_size: int = 1, precision: str = "fp16") -> bool:
        """
        Authoritative validation of the scenario's physical and performance feasibility.
        Returns True if all levels PASS, raises Exception if FAIL.
        """
        eval_res = self.evaluate(batch_size=batch_size, precision=precision)
        
        # Level 1: Feasibility
        if eval_res.feasibility.status == "FAIL":
            raise OOMError(f"Feasibility Failure: {eval_res.feasibility.summary}")
            
        # Level 2: Performance
        if eval_res.performance.status == "FAIL":
            raise SLAViolation(f"Performance Failure: {eval_res.performance.summary}")
            
        return True

class Scenarios(Registry):
    """
    Executable scenario registry.

    Each entry composes one ``Models.*`` workload, one ``Hardware.*`` or
    ``Systems.*`` target, and scenario-local constraints such as latency or
    power budget. Non-executable sourced anchors live separately under
    ``ReferenceStats``.
    """

    # --- TINYML WORLD ---
    SmartDoorbell = Scenario(
        name="Smart Doorbell",
        description="Identifying humans at the door using a sub-watt microcontroller.",
        application="Smart Doorbell",
        workload=_Models.Tiny.WakeVision,
        system=_Hardware.Tiny.ESP32_S3,
        sla_latency=Q_("200 ms"),
        power_budget=Q_("100 mW"),
    )

    TinySensor = Scenario(
        name="Anomaly Sensor",
        description="Low-power vibration monitoring for industrial predictive maintenance.",
        application="Industrial Anomaly Detection",
        workload=_Models.Tiny.AnomalyDetector,
        system=_Hardware.Tiny.ESP32_S3,
        sla_latency=Q_("10 ms"),
        power_budget=Q_("50 mW"),
    )

    # --- EDGE WORLD ---
    AutonomousVehicle = Scenario(
        name="Autonomous Vehicle",
        description="Real-time object detection for safe urban navigation.",
        application="Autonomous Vehicle",
        workload=_Models.Vision.YOLOv8_Nano,
        system=_Hardware.Edge.JetsonOrinNX,
        sla_latency=Q_("10 ms")
    )

    AutonomousVehicle_Waymo = Scenario(
        name="Waymo AV Data Pipeline",
        description="High-throughput data ingestion for autonomous fleet training.",
        application="Autonomous Vehicle",
        workload=_Models.Vision.YOLOv8_Nano.model_copy(
            update={
                "name": "Waymo (High)",
                "data_rate": _ReferenceStats.Workloads.WaymoDataPerHourHigh,
            }
        ),
        system=_Hardware.Edge.JetsonOrinNX,
        sla_latency=Q_("10 ms")
    )

    # --- MOBILE WORLD ---
    MobileHealth = Scenario(
        name="Mobile Health",
        description="On-device medical image analysis for remote diagnostics.",
        application="Mobile Health",
        workload=_Models.Vision.MobileNetV2,
        system=_Hardware.Mobile.iPhone15Pro,
        sla_latency=Q_("30 ms")
    )

    # Added 2026-09-28. Federated on-device vision (the book's "Archetype C"):
    # MobileNet trained/served locally on smartphone-class devices. The
    # sustained power budget is the registry's 3 W upper sustained anchor; the
    # burst ceiling stays at ReferenceStats.MobilePower.MobileMlBurstHigh.
    FederatedMobileVision = Scenario(
        name="Federated Mobile Vision",
        description="On-device federated training and inference of a MobileNet on smartphone-class hardware.",
        application="Federated mobile learning",
        workload=_Models.Vision.MobileNetV2,
        system=_Hardware.Mobile.iPhone15Pro,
        power_budget=_ReferenceStats.MobilePower.MobileMlSustainedHigh,
    )

    MobileAssistant = Scenario(
        name="Mobile Assistant",
        description="On-device assistant with a quantized small LLM on a smartphone.",
        application="Mobile Assistant",
        workload=_Models.Language.Llama3_8B,
        system=_Hardware.Mobile.iPhone15Pro,
        sla_latency=Q_("100 ms"),
        power_budget=Q_("5 W"),
    )

    # --- WORKSTATION WORLD ---
    LocalTraining = Scenario(
        name="Local LLM Fine-tuning",
        description="Fine-tuning a Llama-3 model on a high-end student workstation.",
        application="Local Fine-tuning",
        workload=_Models.Language.Llama3_8B,
        system=_Hardware.Workstation.MacBookM3Max,
        sla_latency=Q_("100 ms")
    )

    # --- CLOUD WORLD ---
    # Teaching mission, H100 era. Frontier_8K (8,192 H100, ~26K A100-equivalents
    # of dense BF16 peak) is the H100 restatement of the GPT-4-class run;
    # SemiAnalysis (2023) estimates the same pre-training at ~8,192 H100 for
    # ~55 days. The historical A100 fleet lives on
    # Models.Language.GPT4.training_*_ref.
    FrontierTraining = Scenario(
        name="Frontier LLM Training",
        description="Pre-training a GPT-4-class frontier model on an 8,192-GPU H100 fleet.",
        application="Frontier Training",
        workload=_Models.Language.GPT4,
        system=_Clusters.Frontier_8K,
        sla_latency=Q_("500 ms") # Per-step target
    )

    # Added 2026-09-28. The dense-LLM pretraining reference (the book's
    # "Archetype A"): GPT-3-class 175B pretrained with mixed-precision Adam on
    # the 1,024-GPU reference cluster (128 DGX H100 nodes). No SLA: it is a
    # composition anchor. Byte policy: ReferenceStats.MixedPrecisionAdam.
    # NOTE: Training_1K.fabric is InfiniBand_HDR (vol1 reads it; unchanged),
    # while the book describes the reference cluster on NDR.
    DenseLLMPretrainingReference = Scenario(
        name="Dense LLM Pretraining (175B, 1,024 H100)",
        description="GPT-3-class dense transformer pretrained with mixed-precision Adam on the 1,024-GPU reference cluster.",
        application="Dense LLM pretraining",
        workload=_Models.Language.GPT3,
        system=_Clusters.Training_1K,
    )

    # --- SERVING WORLD ---
    ChatbotServing = Scenario(
        name="Chatbot Serving",
        description="Serving a Llama-3 8B chatbot on a single H100 with latency SLA.",
        application="Chatbot Serving",
        workload=_Models.Language.Llama3_8B,
        system=_Hardware.Cloud.H100,
        sla_latency=Q_("500 ms"),  # TTFT target
    )

    # --- TINYML: KEYWORD SPOTTING ---
    KeywordSpotting = Scenario(
        name="Keyword Spotting",
        description="Always-on wake-word detection on a microcontroller (MLPerf Tiny benchmark).",
        application="Keyword Spotting",
        workload=_Models.Tiny.DS_CNN,
        system=_Hardware.Tiny.ESP32_S3,
        sla_latency=Q_("30 ms"),
        power_budget=Q_("1 mW"),
    )

    # --- CALCULATION SCENARIOS (added 2026-10-20) ---------------------------
    # Named inputs for recurring calculations that chapters and lecture slides
    # both quote. Each entry states its assumptions once; .result runs the
    # solver that owns the arithmetic (engine/scenario_types.py).

    Llama3_70B_DecodeH100 = _st.DecodeScenario(
        name="Llama-3 70B batch-1 decode on one H100",
        description="FP16 weights streamed once per token on one H100-equivalent path "
                    "(capacity assumed, to isolate arithmetic from weight transfer).",
        workload=_Models.Language.Llama3_70B,
        hardware=_Hardware.Cloud.H100,
    )

    Llama3_70B_DecodeH100Batch64 = _st.DecodeScenario(
        name="Llama-3 70B decode at batch 64 on one H100",
        description="The batch-1 decode scenario with 64 requests sharing each weight read, "
                    "which raises the weight-only arithmetic intensity 64-fold.",
        workload=_Models.Language.Llama3_70B,
        hardware=_Hardware.Cloud.H100,
        batch_size=64,  # scenario assumption: concurrent decode requests
    )

    Llama3_70B_ServingH100 = _st.ServingScenario(
        name="Llama-3 70B single-request serving on one H100",
        description="ServingModel latency (weights, KV-cache reads, launch overhead) for one "
                    "request at a 2,048-token context; .variant(hardware=...) compares parts.",
        workload=_Models.Language.Llama3_70B,
        hardware=_Hardware.Cloud.H100,
        seq_len=2048,  # scenario assumption: serving context length
        batch_size=1,  # scenario assumption: single-request decode
    )

    BufferTransfer10GB = _st.TransferScenario(
        name="10 GB buffer across the H100 bandwidth hierarchy",
        description="One synchronized 10 GB buffer moved one hop on each tier: local HBM, "
                    "NVLink inside a node (one direction), InfiniBand NDR between nodes, "
                    "PCIe Gen5 to the host. Bandwidth-only bounds.",
        payload=Q_(10, "GB"),  # scenario assumption: synchronized buffer size
        links={
            "HBM": _Hardware.Cloud.H100.memory.bandwidth,
            "NVLink": _Hardware.Cloud.H100.nvlink.bandwidth_per_direction,
            "InfiniBand": _Fabrics.InfiniBand_NDR.bandwidth,
            "PCIe": _Hardware.Cloud.H100.interconnect.bandwidth,
        },
    )

    Gpt3TensorParallelStepDgxH100 = _st.TensorParallelScenario(
        name="175B tensor-parallel step inside one DGX H100",
        description="GPT-3 175B split eight ways inside one DGX H100 node; one activation "
                    "AllReduce per layer per pass over NVLink; 200 ms of compute is a stated input.",
        workload=_Models.Language.GPT3,
        tp_size=_Nodes.DGX_H100.accelerators_per_node,
        microbatch=4,  # scenario assumption: sequences per microbatch
        link_bandwidth=_Hardware.Cloud.H100.nvlink.bandwidth_per_direction,
        compute_time=Q_(200, "ms"),  # scenario assumption: compute per step, not derived
    )

    Gpt3PretrainingFrontier8K = _st.TrainingRunScenario(
        name="175B pretraining on 8,192 H100s",
        description="GPT-3 175B on its training tokens at the H100 FP8 peak, the 8,192-GPU "
                    "pod's modeled scaling efficiency, and the standard run overheads.",
        workload=_Models.Language.GPT3,
        fleet=_Clusters.Frontier_8K,
        precision="fp8",
        local_efficiency=_ReferenceStats.TrainingScaleProfiles.LocalComputeEfficiency,
        scaling_efficiency=_ReferenceStats.TrainingScaleProfiles.Eff8192Gpu,
        overheads={
            "checkpoint": _Ops.TrainingRunOverheads.Checkpoint,
            "failures": _Ops.TrainingRunOverheads.FailureRecovery,
            "maintenance": _Ops.TrainingRunOverheads.Maintenance,
        },
    )

    Gpt3TrainingStateTp8 = _st.TrainingStateScenario(
        name="175B training state under 8-way tensor parallelism",
        description="Mixed-precision Adam state of GPT-3 175B split over one DGX H100 node "
                    "(TP-8), with 128 data-parallel replicas for ZeRO sharding of the moments.",
        workload=_Models.Language.GPT3,
        tp_size=_Nodes.DGX_H100.accelerators_per_node,
        dp_size=128,  # scenario assumption: data-parallel replicas (1,024 GPUs)
        activations=Q_(200, "GB"),  # scenario assumption: inside the 100 to 400 GB activation range
    )

    H100BarrierLoadStep = _st.PowerTransientScenario(
        name="Barrier-aligned H100 load step",
        description="Each H100 steps from communication-phase power to its TDP when a "
                    "collective barrier releases it; one board loop, then 1,024 GPUs together.",
        hardware=_Hardware.Cloud.H100,
        low_power=Q_(400, "W"),  # scenario assumption: communication-phase GPU power
        rail_voltage=Q_(0.8, "V"),  # scenario assumption: accelerator core-rail voltage
        loop_inductance=Q_(50, "pH"),  # scenario assumption: VRM-to-package loop on one board
        rise_times={
            "slow": Q_(1, "microsecond"),  # edge once package capacitors act
            "fast": Q_(100, "nanosecond"),  # the same step with no package decoupling
        },
        n_devices=_Clusters.Training_1K.total_accelerators,
        ramp_window=Q_(100, "microsecond"),  # scenario assumption: cluster transition window
    )

    AiRackAirCooling = _st.CoolingScenario(
        name="High-density AI rack cooled by air",
        description="The high-density AI rack's heat load carried by air at a 15 K rise.",
        heat_load=Q_(_Infrastructure.Racks.AI_High.power_kw, "kW"),
        temperature_rise=Q_(15, "K"),  # scenario assumption: inlet 20 C, outlet 35 C
        coolant="air",
    )

    Nvl72WaterCooling = _st.CoolingScenario(
        name="GB200 NVL72 rack cooled by water",
        description="The GB200 NVL72 rack's heat carried by water at a 10 K coolant rise.",
        heat_load=_Hardware.Cloud.GB200_NVL72.tdp,
        temperature_rise=Q_(10, "K"),  # scenario assumption: coolant rise across the rack
        coolant="water",
    )

    H100AirCooling = _st.CoolingScenario(
        name="One H100 cooled by air",
        description="One H100's TDP carried by air at a 15 K rise.",
        heat_load=_Hardware.Cloud.H100.tdp,
        temperature_rise=Q_(15, "K"),
        coolant="air",
    )

    CduOutageTraining10K = _st.OutageScenario(
        name="Coolant distribution unit outage under a 10,000-GPU job",
        description="One rack's CDU pump fails for a four-hour repair; a synchronous job "
                    "on the 10,000-GPU cluster idles with it.",
        failed_devices=256,  # scenario assumption: GPUs behind the failed CDU
        fleet=_Clusters.Training_10K,
        repair_time=Q_(4, "hour"),  # scenario assumption: pump repair window
        device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
        rollback_times={"low": Q_(30, "minute"), "high": Q_(60, "minute")},
    )

    GpuFailuresTraining10K = _st.FailureIntervalScenario(
        name="GPU failures across 10,000 H100s",
        description="Independent GPU failures at the per-GPU MTTF over a two-week run.",
        component_mtbf=Q_(_Reliability.Gpu.mttf_hours, "hour"),
        n_components=_Clusters.Training_10K.total_accelerators,
        run_duration=Q_(14, "day"),  # scenario assumption: two-week training run
    )

    NodeFailuresTraining1K = _st.FailureIntervalScenario(
        name="Node failures across the 128-node reference cluster",
        description="Composite DGX node failures (GPUs, hosts, power, links) over a two-week run.",
        component_mtbf=Q_(_Reliability.DgxNodeComposite.mttf_hours, "hour"),
        n_components=_Clusters.Training_1K.count,
        run_duration=Q_(14, "day"),
    )

    V100ToB200 = _st.AcceleratorComparison(
        name="V100 to B200", description="Dense FP16 peak, TDP, and peak per watt.",
        baseline=_Hardware.Cloud.V100, candidate=_Hardware.Cloud.B200)

    V100ToH100 = _st.AcceleratorComparison(
        name="V100 to H100", description="Dense FP16 peak, TDP, and peak per watt; a "
                                         "10,000 V100 GPU-hour job restated in H100 GPU-hours.",
        baseline=_Hardware.Cloud.V100, candidate=_Hardware.Cloud.H100,
        baseline_device_hours=10_000)  # scenario assumption: job size measured on V100s

    PeakPlannedTrainingBudget = _st.SustainedThroughputScenario(
        name="Training budget planned from peak throughput",
        description="1,000 GPU-hours estimated from datasheet TFLOP/s, for a workload "
                    "that sustains 40 percent of peak.",
        planned_device_hours=1000,  # scenario assumption: budget from the datasheet rate
        sustained_fraction=0.4,  # scenario assumption: sustained share of peak
    )

    # --- Data-parallel training at fleet scale (added 2026-10-20) -----------
    Gpt3GradientSync = _st.GradientSyncScenario(
        name="175B data-parallel gradient sync over InfiniBand HDR and 100G Ethernet",
        description="One worker computes for a stated 1.2 s, then ring-AllReduces 16-bit "
                    "gradients with no overlap; energy charges each bit once on InfiniBand.",
        workload=_Models.Language.GPT3,
        precision="fp16",
        links={"InfiniBand": _Fabrics.InfiniBand_HDR.bandwidth, "Ethernet": _Fabrics.Ethernet_100G.bandwidth},
        compute_time=Q_(1.2, "second"),  # scenario assumption: illustrative compute per step, not a measured GPT-3 step
        energy_per_bit=_NetworkEnergy.InfiniBandEnergyPerBit,
        steps=1_000_000,  # scenario assumption: steps in the run
    )

    Gpt3GradientSyncFasterAccelerator = Gpt3GradientSync.variant(
        name="The same step on accelerators with twice the compute rate",
        description="Only the compute term halves; the exchange is unchanged.",
        compute_speedup=2.0,  # scenario assumption: one accelerator generation
    )

    Gpt3DataParallelIntensity = _st.DataParallelIntensityScenario(
        name="Communication intensity of a 175B data-parallel worker",
        description="Four full-length GPT-3 sequences per worker per step; balance of one "
                    "A100 on one 200 Gb/s HDR link.",
        workload=_Models.Language.GPT3,
        sequences_per_worker=4,  # scenario assumption: local batch of four sequences
        link_bandwidth=_Fabrics.InfiniBand_HDR.bandwidth,
        accelerator=_Hardware.Cloud.A100,
    )

    WorkloadAlphaIntensity = _st.IntensityScenario(
        name="Workload Alpha (compute-heavy)", description="10 MB sent per 10^14 FLOPs per GPU per step.",
        bytes_per_step=Q_(10, "MB"), flops_per_step=Q_(1e14, "flop"))

    WorkloadBetaIntensity = _st.IntensityScenario(
        name="Workload Beta (communication-heavy)", description="1 GB sent per 10^13 FLOPs per GPU per step.",
        bytes_per_step=Q_(1, "GB"), flops_per_step=Q_(1e13, "flop"))

    NodeAvailability1K = _st.AvailabilityScenario(
        name="1,000 nodes at 99.9 percent", description="Illustrative per-node availability, independent failures.",
        availability=0.999, n_components=1000)

    NodeAvailability10K = NodeAvailability1K.variant(
        name="10,000 nodes at 99.9 percent", n_components=10_000)

    FourNinesAvailability10K = NodeAvailability1K.variant(
        name="10,000 nodes at 99.99 percent", availability=0.9999, n_components=10_000)

    Gpt3ComputeOptimal = _st.ComputeOptimalScenario(
        name="GPT-3's budget at a compute-optimal split",
        description="GPT-3's 6ND budget re-split at Chinchilla's tokens per parameter.",
        workload=_Models.Language.GPT3,
        tokens_per_param=float(_Literature.Chinchilla.TokensPerParam),
        flops_per_param_token=float(_Literature.Chinchilla.ComputeConstant),
    )

    FleetLawProfileExample = _st.StepBreakdownScenario(
        name="A step profile dominated by communication",
        description="Illustrative Fleet Law terms: 0.8 s compute, 1.0 s communication, 0.2 s sync, no overlap.",
        compute_time=Q_(0.8, "second"), communication_time=Q_(1.0, "second"),
        synchronization_time=Q_(0.2, "second"), overlap_time=Q_(0, "second"))

    FleetLawDoublingExample = _st.ScalingStepScenario(
        name="Doubling N when communication does not shrink",
        description="Illustrative units: the step falls from 10 to 8 when N doubles.",
        step_at_n=10, step_at_scaled=8, factor=2)

    CoordinationOverheadExample = _st.ScalingOverheadScenario(
        name="A planned 3x speedup with 40 percent overhead",
        description="Communication and coordination take 40 percent of the scaled step.",
        planned_speedup=3.0, overhead_fraction=0.40)

    ChinchillaParameterPowerLaw = _st.PowerLawScenario(
        name="Chinchilla parameter term", description="Hoffmann et al. 2022 parameter exponent.",
        exponent=float(_Literature.Chinchilla.ParametricLossAlpha))

    ShallowPowerLawExample = _st.PowerLawScenario(
        name="A shallow power law", description="Illustrative exponent of 0.1.", exponent=0.1)

    NotableRunGrowth2012to2024 = _st.TrainingGrowthScenario(
        name="Training-compute growth, 2012 to 2024",
        description="Epoch AI notable models that report accelerator count and duration.",
        series_name="NotableModelTrainingRuns", first_year="2012", last_year="2024")

    GpuFailuresLlama3Run = _st.FailureIntervalScenario(
        name="GPU failures across the Llama 3 405B run",
        description="16,384 GPUs at the per-GPU MTTF over the 54-day snapshot.",
        component_mtbf=Q_(_Reliability.Gpu.mttf_hours, "hour"),
        n_components=int(_Models.Language.Llama3_405B.training_accelerators_ref.to("count").magnitude),
        run_duration=Q_(float(_Literature.TrainingRuns.Llama3_405B_SnapshotDays), "day"),
    )

    Gpt3PublishedRun = _st.PublishedRunScenario(
        name="GPT-3 175B as trained",
        description="GPT-3's training compute over its reference V100 count and days (Patterson et al. 2021).",
        workload=_Models.Language.GPT3,
        accelerator=_Hardware.Cloud.V100,
    )

    # --- Network fabrics (added 2026-10-20) ---------------------------------
    ExposedAllReduceIntraNode = _st.StepBreakdownScenario(
        name="A step on eight GPUs with its AllReduce over NVLink",
        description="Illustrative schedule: 100 ms forward, 100 ms backward, then a 1 ms AllReduce "
                    "inside one node, nothing overlapped. The durations are chosen, not derived.",
        compute_time=Q_(200, "ms"),
        compute_phases={"forward": Q_(100, "ms"), "backward": Q_(100, "ms")},  # scenario assumption
        communication_time=Q_(1, "ms"),  # scenario assumption: AllReduce over NVLink
        synchronization_time=Q_(0, "ms"), overlap_time=Q_(0, "ms"))

    ExposedAllReduceInterNode = ExposedAllReduceIntraNode.variant(
        name="The same step on 64 GPUs with its AllReduce across nodes",
        description="The same compute with an assumed 30 ms AllReduce over InfiniBand; every GPU waits through it.",
        communication_time=Q_(30, "ms"))  # scenario assumption: AllReduce across nodes

    EcmpHashCollision = _st.PathLoadScenario(
        name="Eight elephant flows hashed onto four equal-cost paths",
        description="One static-hash outcome: 4, 2, 2, and 0 flows per path, against an even split of two.",
        path_loads=[4, 2, 2, 0])  # scenario assumption: one hash outcome

    SpineOpticsTraining10K = _st.LinkBuildoutScenario(
        name="Spine-layer optics for a 10,000-GPU cluster",
        description="Two optical spine links per GPU, each priced as one optical link and powered by one "
                    "selected 400G pluggable module (the opposite end excluded).",
        fleet=_Clusters.Training_10K,
        links_per_device=2,  # scenario assumption: spine-layer optical links per accelerator
        cost_per_link=_Infrastructure.Pricing.Network.OpticsPerLinkUsd.rate,
        power_per_link=_SwitchFabric.OpticsPluggable400G)

    GoBackNResendNdr = _st.BandwidthDelayScenario(
        name="Go-Back-N resend window on an NDR link",
        description="Data resent after one drop: line rate times a 5 microsecond loss-detection round trip.",
        link_bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
        delay=Q_(5, "microsecond"))  # scenario assumption: feedback round trip

    AlphaBetaNdrRoce = _st.AlphaBetaScenario(
        name="A control message and a gradient shard on NDR and RoCE paths",
        description="Modeled alpha-beta paths: InfiniBand NDR, RoCE at 100 Gb/s, and RoCE's startup on a "
                    "400 Gb/s link; a 4,000-byte control message and a 350 MB gradient shard.",
        alphas={"NDR": _SwitchFabric.AlphaNdr, "RoCE 100G": _SwitchFabric.AlphaRoce,
                "RoCE 400G": _SwitchFabric.AlphaRoce},
        bandwidths={"NDR": _Fabrics.InfiniBand_NDR.bandwidth, "RoCE 100G": _Fabrics.Ethernet_100G.bandwidth,
                    "RoCE 400G": _Fabrics.RoCE_400G.bandwidth},
        messages={"control": Q_(4000, "byte"),  # scenario assumption: pipeline control message
                  "shard": Q_(350, "MB")})  # scenario assumption: one rank's gradient shard, rounded

    AlphaBetaHdrNdrSmallMessage = _st.AlphaBetaScenario(
        name="A 10 KB message on HDR and NDR InfiniBand",
        description="The NDR startup latency on both links; the link rate doubles from HDR to NDR.",
        alphas={"HDR": _SwitchFabric.AlphaNdr, "NDR": _SwitchFabric.AlphaNdr},
        bandwidths={"HDR": _Fabrics.InfiniBand_HDR.bandwidth, "NDR": _Fabrics.InfiniBand_NDR.bandwidth},
        messages={"small": Q_(10, "KB")})  # scenario assumption: small control packet

    NcclAllReduceH100NodeFit = _st.CollectiveFitScenario(
        name="Two-parameter fit to an eight-H100 NCCL AllReduce sweep",
        description="Floor = median time up to 64 KiB; bandwidth = algorithm bandwidth at the largest "
                    "message; worst measured/model ratio from 8 to 128 MiB.",
        series_name="NCCLAllReduceH100Node",
        floor_cutoff=Q_(64 * 1024, "byte"),
        mid_low=Q_(8 * 2**20, "byte"), mid_high=Q_(128 * 2**20, "byte"))

    RingAllReduceAcross4to1Spine = _st.OversubscriptionScenario(
        name="Ring AllReduce across a 4:1 spine on 1,024 GPUs",
        description="Ranks alternate between the halves, so every ring edge crosses the cut; 5 GB per GPU, "
                    "30 percent of the step communicating, a $300M cluster. The worst-case placement.",
        fleet=_Clusters.Training_1K,
        link_bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
        oversubscription_ratio=_Fabrics.InfiniBand_NDR_Oversub4to1.oversubscription_ratio,
        payload_per_device=_ReferenceStats.NetworkEconomicsProfile.PerGpuGradientPayload,
        communication_fraction=_ReferenceStats.NetworkEconomicsProfile.CommunicationFraction,
        cluster_cost=_ReferenceStats.NetworkEconomicsProfile.ClusterCapex)

    RingAllReduceAcross2to1Spine = RingAllReduceAcross4to1Spine.variant(
        name="Ring AllReduce across a 2:1 spine on 1,024 GPUs",
        oversubscription_ratio=_Fabrics.InfiniBand_NDR_Oversub2to1.oversubscription_ratio)

    FatTreeNdrSwitch = _st.FatTreeScenario(
        name="Fat-tree of 64-port NDR switches",
        description="Quantum-2 class radix: two tiers reach k^2/2 hosts, three reach k^3/4.",
        radix=int(_SwitchFabric.NdrSwitchPorts))

    FatTree102T4Switch = _st.FatTreeScenario(
        name="Fat-tree of 102.4 Tb/s switches split into 400 Gb/s ports",
        description="A 102.4 Tb/s switch ASIC used as 256 ports of 400 Gb/s (derived radix).",
        radix=round((_SwitchFabric.SwitchAsic102T4 / _Fabrics.Ethernet_400G.bandwidth).to("").magnitude))

    TorusVsFatTree4K = _st.TorusScenario(
        name="4,096 endpoints as a 3D torus and as a fat-tree",
        description="Four 1,024-host pods of the 64-port fat-tree (twice one standalone two-tier fabric), "
                    "against a 16 x 16 x 16 torus at the same per-link NDR rate.",
        n_endpoints=2 * FatTreeNdrSwitch.two_tier_hosts,  # doubling past one standalone two-tier fabric
        link_bandwidth=_Fabrics.InfiniBand_NDR.bandwidth)  # scenario assumption: equal per-link capacity

    RailVsLeafSpineHops = _st.HopPathScenario(
        name="Same-rank traffic through a rail switch and through leaf-spine-leaf",
        description="One rail switch against source leaf, spine, and destination leaf, at the per-hop latency.",
        hop_latency=_SwitchFabric.HopLatency,
        hops={"leaf-spine-leaf": 3, "rail": 1})

    IncastSuperPodUnit = _st.IncastScenario(
        name="One SuperPOD scalable unit into one 400 Gb/s egress",
        description="Deliberate worst case: 256 NDR senders converge on one egress port with a 32 MB buffer.",
        n_senders=_Clusters.DgxH100SuperPodSU.total_accelerators,
        sender_bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
        egress_bandwidth=_Fabrics.Ethernet_400G.bandwidth,
        buffer=Q_(32, "MB"),  # scenario assumption: switch egress buffer
        feedback_rtt=Q_(5, "microsecond"))  # scenario assumption: congestion-feedback round trip

    TransceiverDegradationRoce4K = _st.ComponentExposureScenario(
        name="Daily transceiver degradation in a 4,096-GPU RoCE cluster",
        description="Leaf, spine, and core links per GPU, an optical module at each end, each starting "
                    "to degrade with probability 0.001 percent per day.",
        counts={"devices": 4096,  # scenario assumption: large RoCE cluster
                "links_per_device": 3,  # scenario assumption: leaf, spine, and core tiers
                "modules_per_link": 2},  # one module at each end of an optical link
        probability=1e-5)  # scenario assumption: per module per day

    DegradedLinksReferenceFabric = _st.ComponentExposureScenario(
        name="Degraded links at one instant in a 3,000-link fabric",
        description="About 3,000 high-speed links in the 1,024-GPU fabric, each degraded with probability 0.1 percent.",
        counts={"links": 3000},  # scenario assumption: links that sustain full bisection
        probability=0.001)  # scenario assumption: point-in-time probability per link

    DegradedTransceiverRun1K = _st.SilentSlowdownScenario(
        name="A degraded transceiver during a 30-day run on 1,024 GPUs",
        description="A 10 percent bandwidth loss that slows each step 2 to 3 percent, priced at the "
                    "fleet GPU-hour rate; no alarm fires.",
        fleet=_Clusters.Training_1K,
        run_duration=Q_(30, "day"),  # scenario assumption: training-run duration
        slowdowns={"low": 0.02, "high": 0.03},  # scenario assumption: per-step slowdown after overlap
        device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
        bandwidth_loss=0.10,  # scenario assumption: bandwidth lost by the degraded link
        communication_fraction=_ReferenceStats.NetworkEconomicsProfile.CommunicationFraction)

    SriovNdrNic = _st.NicSharingScenario(
        name="An NDR NIC split into eight virtual functions",
        description="SR-IOV slices with a rate cap each, and a neighbor's 100 GB checkpoint at line rate.",
        nic_bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
        n_virtual_functions=8,  # scenario assumption
        burst_size=Q_(100, "GB"))  # scenario assumption: neighboring job's checkpoint

    HalfRateLinkDiagnosis = _st.SlowdownDiagnosisScenario(
        name="A training step that slowed on day 3, traced to one link at half rate",
        description="Step time and utilization before and after; a pairwise RDMA test reports one pair at "
                    "half the healthy rate.",
        step_before=Q_(4.2, "second"), step_after=Q_(4.8, "second"),  # scenario assumption
        utilization_before=0.92, utilization_after=0.85,  # scenario assumption: nvidia-smi
        days_since_launch=3,  # scenario assumption
        port_bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
        delivered_efficiency=0.96,  # scenario assumption: ib_write_bw payload share on a healthy link
        measured_fraction=0.5)  # scenario assumption: lanes or rate negotiated down by half
    # --- Feeding and saving a training fleet (added 2026-10-20) -------------
    Gpt3StorageFootprint = _st.StorageFootprintScenario(
        name="Storage footprint of a 30-day 175B pretraining run",
        description="1.5 trillion 4-byte token IDs read once; a resumable 14-byte-per-parameter "
                    "checkpoint every 10 minutes; every 100th checkpoint and the final one archived.",
        workload=_Models.Language.GPT3,
        bytes_per_param=_ReferenceStats.StorageTrainingCorpus.CheckpointBytesPerParameter,
        corpus_bytes=_ReferenceStats.StorageTrainingCorpus.TokenizedText,
        source_bytes=_ReferenceStats.StorageTrainingCorpus.CompressedSource,
        run_duration=_ReferenceStats.StorageTrainingCorpus.TrainingWindow,
        save_interval=_ReferenceStats.StorageTrainingCorpus.CheckpointInterval,
        shard_bytes=_ReferenceStats.StorageTrainingCorpus.TokenizedShardSize,
        archive_every=100,  # scenario assumption: retention keeps every 100th checkpoint
    )

    Gpt3TieredCheckpoint2K = _st.TieredCheckpointScenario(
        name="175B checkpoint staged on local NVMe across 256 DGX H100 nodes",
        description="Each node writes its shard to four striped Gen4 NVMe drives, then the "
                    "shards drain to the shared 1 TB/s parallel file system.",
        workload=_Models.Language.GPT3,
        bytes_per_param=_ReferenceStats.StorageTrainingCorpus.CheckpointBytesPerParameter,
        fleet=_Clusters.Production_2K,
        local_bandwidth=_Storage.Production2KCheckpointPath.local_stage.aggregate_bandwidth,
        durable_bandwidth=_Storage.Production2KCheckpointPath.durable_store.bandwidth,
        save_interval=_ReferenceStats.StorageTrainingCorpus.CheckpointInterval,
    )

    Llama2_70BCheckpointStorm1K = _st.CheckpointStormScenario(
        name="70B checkpoint written by every one of 1,024 replicas",
        description="Each data-parallel replica writes the full resumable state through "
                    "100 GB/s of storage ingest; a bound on write volume, not a deployable layout.",
        workload=_Models.Language.Llama2_70B,
        bytes_per_param=_ReferenceStats.StorageTrainingCorpus.CheckpointBytesPerParameter,
        replicas=_Clusters.Training_1K.total_accelerators,
        ingest_bandwidth=_Fabrics.InfiniBand_XDR.bandwidth,
    )

    TextInputDemand2K = _st.InputDemandScenario(
        name="Text input demand of 2,048 GPUs",
        description="Every GPU reads one packed 4,096-token sequence of token IDs per 200 ms step.",
        consumers=_Clusters.Production_2K.total_accelerators,
        samples_per_step=4096,  # scenario assumption: packed tokens per GPU per step
        bytes_per_sample=_ReferenceStats.StorageTrainingCorpus.TokenIdBytes,
        step_time=_ReferenceStats.StorageTrainingCorpus.StepTime,
    )

    ImageInputDemand2K = _st.InputDemandScenario(
        name="Image input demand of 2,048 GPUs",
        description="Every GPU reads 256 ImageNet-sized JPEGs per 200 ms step.",
        consumers=_Clusters.Production_2K.total_accelerators,
        samples_per_step=256,  # scenario assumption: images per GPU per step
        bytes_per_sample=_Datasets.ImageNet.mean_compressed_image_size,
        step_time=_ReferenceStats.StorageTrainingCorpus.StepTime,
    )

    VideoInputDemand2K = ImageInputDemand2K.variant(
        name="Video-clip input demand of 2,048 GPUs",
        description="The image job with 1 MB clips in place of JPEGs.",
        bytes_per_sample=Q_(1, "MB"),  # scenario assumption: a clip of about a megabyte
    )

    OverlappedInputStall = _st.DataStallScenario(
        name="250 ms delivery against 200 ms compute",
        description="Storage delivers each batch in 250 ms; compute takes one 200 ms step.",
        compute_time=_ReferenceStats.StorageTrainingCorpus.StepTime,
        io_time=Q_(250, "ms"),  # scenario assumption: delivery time per batch
    )

    QuarterDeliveryIoWall = _st.DataStallScenario(
        name="Delivery at one quarter of the original compute time",
        description="Normalized: compute takes one unit, delivery a quarter unit, overlapped.",
        compute_time=Q_(1.0, ""),  # normalized compute time
        io_time=Q_(0.25, ""),  # scenario assumption: delivery is a quarter of compute
    )

    PrefetchDepthByTier = _st.PrefetchDepthScenario(
        name="Prefetch depth against each tier's p99 read",
        description="200 ms compute windows; 40 MB decoded batches, five deep, eight GPUs per node.",
        compute_time=_ReferenceStats.StorageTrainingCorpus.StepTime,
        tail_latencies={
            "nvme": Q_(50, "ms"),  # scenario assumption: local NVMe p99 read
            "pfs": Q_(200, "ms"),  # scenario assumption: parallel file system p99 read
            "object": Q_(500, "ms"),  # scenario assumption: object storage p99 read
        },
        batch_bytes=Q_(40, "MB"),  # scenario assumption: collated batch per GPU
        safety_depth=5,  # scenario assumption: object-storage safety depth
        consumers_per_node=_Nodes.DGX_H100.accelerators_per_node,
    )

    HotTierCacheExample = _st.TieredCacheScenario(
        name="95 percent hits, misses 100 times slower",
        description="Illustrative two-tier cache in units of one hit's service time.",
        hit_rate=0.95,  # scenario assumption
        miss_penalty=100.0,  # scenario assumption
    )

    MultimodalCorpusNodeCache = _st.TieredCacheScenario(
        name="A 35 TB multimodal corpus over one node's local NVMe",
        description="Four 7.68 TB Gen4 drives hold part of a uniformly sampled corpus; misses "
                    "read the node's share of the parallel file system.",
        hot_capacity=(_Storage.LocalNvmeGen4_7T68x4.devices_per_node
                      * _Storage.LocalNvmeGen4_7T68x4.device.capacity),
        corpus_parts={
            "images": Q_(20, "TB"),  # scenario assumption
            "text": Q_(10, "TB"),  # scenario assumption
            "audio": Q_(5, "TB"),  # scenario assumption
        },
        hot_bandwidth=_Storage.LocalNvmeGen4_7T68x4.aggregate_bandwidth,
        miss_bandwidth=_Storage.PfsPerNodeShare.bandwidth,
    )

    GpuDirectStorageNodeCpu = _st.HostIoCpuScenario(
        name="Host CPU spent on small-image I/O in one DGX H100 node",
        description="8,000 images per second per GPU, through host buffers or GPUDirect Storage; "
                    "illustrative CPU service times per request.",
        requests_per_device=8000,  # scenario assumption: images per second per GPU
        devices=_Nodes.DGX_H100.accelerators_per_node,
        service_times={"host": _Storage.TraditionalGpuIoLatency, "gds": _Storage.GpuDirectStorageLatency},
    )

    StorageTailFanout100 = _st.AvailabilityScenario(
        name="A step reading from 100 storage servers",
        description="Each server is slow 1 percent of the time, independently; all_up is the "
                    "chance that none is slow.",
        availability=0.99,  # scenario assumption: share of fast responses per server
        n_components=100,  # scenario assumption: servers one step reads from
    )

    ImageFileMetadataMeltdown = _st.ShardPackagingScenario(
        name="200 million image files on a 500 GB/s file system",
        description="One file per image, then the same images packaged into 50,000 tar shards.",
        files=200_000_000,  # scenario assumption
        shards=50_000,  # scenario assumption
        rated_bandwidth=Q_(500, "GB/s"),  # scenario assumption
    )

    ParallelFileSystem1TBps = _st.ParallelFileSystemScenario(
        name="Object storage servers behind a 1 TB/s parallel file system",
        description="Servers of 10 GB/s each, enough for the reference 1 TB/s aggregate.",
        target_bandwidth=_Storage.PfsOneTbPerSecond.bandwidth,
        server_bandwidth=Q_(10, "GB/s"),  # scenario assumption: one OSS
    )

    ObjectStorage100TB = _st.ObjectStorageBillScenario(
        name="A 100 TB dataset in object storage, read ten times across regions",
        description="Representative S3 Standard, cross-boundary egress, GET, and Glacier "
                    "retrieval prices; one GET per file for 100 million files.",
        dataset_bytes=Q_(100, "TB"),  # scenario assumption
        storage_rate=_Infrastructure.Pricing.Storage.S3StandardLowPerTbMonth.rate,
        billing_period=Q_(12, "month"),
        egress_rate=_Infrastructure.Pricing.Cloud.EgressPerGB.rate,
        reads=10,  # scenario assumption: epochs read across the boundary
        files=100_000_000,  # scenario assumption
        request_fee=_Infrastructure.Pricing.Storage.S3GetPer1kRequests.rate,
        requests_per_fee=1000,  # the GET price is quoted per 1,000 requests
        retrieval_rate=_Infrastructure.Pricing.Storage.GlacierRetrievalPerGB.rate,
    )

    VisionCorpusStreamVsStage = _st.StreamVsStageScenario(
        name="Stream or stage a 50 TB vision corpus",
        description="20 epochs per run, four runs a year; object storage, local NVMe, and egress "
                    "at representative prices.",
        dataset_bytes=Q_(50, "TB"),  # scenario assumption
        epochs=20,  # scenario assumption
        runs_per_year=4,  # scenario assumption
        object_rate=_Infrastructure.Pricing.Storage.S3StandardLowPerTbMonth.rate,
        local_rate=_Infrastructure.Pricing.Storage.NvmeLowPerTbMonth.rate,
        egress_rate=_Infrastructure.Pricing.Cloud.EgressPerGB.rate,
        year=Q_(12, "month"),
    )

    InputStallIdleCost1K = _st.UtilizationLossScenario(
        name="1,024 GPUs idled by input stalls",
        description="The reference cluster at the fleet GPU-hour price while its busy fraction "
                    "falls from 90 to 70 percent, over a 30-day run.",
        fleet=_Clusters.Training_1K,
        device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
        healthy_utilization=0.90,  # scenario assumption
        degraded_utilization=0.70,  # scenario assumption
        duration=Q_(30, "day"),  # scenario assumption
    )

    SyntheticDataProvenance = _st.SyntheticFootprintScenario(
        name="1 TB of generated samples with provenance and verification",
        description="40 percent provenance metadata at rest; three verifier passes read the payload.",
        payload_bytes=Q_(1, "TB"),  # scenario assumption
        provenance_overhead=0.4,  # scenario assumption
        verification_passes=3,  # scenario assumption
    )

    MLPerfStorageV1Demand = _st.StorageBenchmarkScenario(
        name="MLPerf Storage v1.0 training demand per emulated accelerator",
        description="Closed-division passing results; medians for 3D U-Net, CosmoFlow, and "
                    "ResNet-50 on emulated A100s and H100s.",
        series_name="MLPerfStorageV1Training",
        division="Closed",
        workloads=("unet3d", "cosmoflow", "resnet50"),
        accelerators=("A100", "H100"),
    )

    Gpt3WeightsColdStart = _st.TransferScenario(
        name="Loading 175B FP16 weights into one serving node",
        description="One replica's weights from each source: a single object-store client, the "
                    "node's file-system share, one Gen5 drive, eight Gen4 drives in parallel, "
                    "eight PCIe links, or host DRAM shared by eight transfers.",
        payload=_memory_from_params(_Models.Language.GPT3.parameters, _BYTES_FP16 / _param),
        links={
            "object store": Q_(1, "GB/s"),  # scenario assumption: single-client object-store read
            "file system share": _Storage.PfsPerNodeShare.bandwidth,
            "one Gen5 NVMe": _Hardware.Tech.Storage.NvmeGen5.bandwidth,
            "eight Gen4 NVMe": _Nodes.DGX_H100.accelerators_per_node * _Hardware.Tech.Storage.NvmeGen4.bandwidth,
            "eight PCIe links": _Nodes.DGX_H100.accelerators_per_node * _Hardware.Cloud.H100.interconnect.bandwidth,
            "host DRAM": _Hardware.Tech.Storage.HostDram.bandwidth,
        },
    )

    Gpt3RolloutFanout100 = _st.TransferScenario(
        name="A rollout of 175B FP16 weights to 100 replicas at once",
        description="Every replica reads its weights from a shared tier in one burst.",
        payload=_memory_from_params(_Models.Language.GPT3.parameters, _BYTES_FP16 / _param),
        copies=100,  # scenario assumption: serving replicas updated in one rollout
        links={
            "file system": _Storage.PfsOneTbPerSecond.bandwidth,
            "object store": Q_(100, "GB/s"),  # scenario assumption: aggregate object-store rate
        },
    )

    NodeStripeCodecs = _st.CompressedReadScenario(
        name="Two codecs on one node's four-drive NVMe stripe",
        description="A light codec at 3:1 decoding 1.5 GB/s per core against a denser one at 4:1 "
                    "decoding 0.5 GB/s per core; illustrative rates.",
        read_bandwidth=_Storage.LocalNvmeGen4_7T68x4.aggregate_bandwidth,
        compression_ratios={"light": 3.0, "dense": 4.0},  # scenario assumption
        decode_rates={"light": Q_(1.5, "GB/s"), "dense": Q_(0.5, "GB/s")},  # scenario assumption, per core
    )


    # --- Embodied stopping, floor traction, and domain compute (added 2026-10-20)
    # The running machine's elementary stop (Embodied.MobileManipulator and
    # Embodied.Scenario hold every input), the Tempe record read through the
    # same budget (Literature.IncidentReports), a floor-friction drop, and
    # robot-policy training compute against the language frontier.
    WarehouseAisleStopAtDriveLimit = _st.StoppingBudgetScenario(
        name="Warehouse mobile manipulator stopping at its drive limit",
        description="The loaded base at its drive limit, its pre-brake delay, and its credible "
                    "deceleration, against the rack-end clear distance (illustrative inputs).",
        speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
        delay=_Embodied.Scenario.WarehouseAisle.tau_delay,
        deceleration=_Embodied.Scenario.WarehouseAisle.a_brake,
        clear_distance=_Embodied.Scenario.WarehouseAisle.d_clear,
    )

    WarehouseAisleStopHalfSpeed = WarehouseAisleStopAtDriveLimit.variant(
        name="Warehouse mobile manipulator stopping at half its drive limit",
        description="The drive-limit stop with the speed halved; the delay and deceleration are unchanged.",
        speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity / 2,  # scenario assumption: half speed
    )

    WarehouseAisleStopDoubledLease = WarehouseAisleStopAtDriveLimit.variant(
        name="Warehouse mobile manipulator stopping with its chunk lease doubled",
        description="The drive-limit stop with a second chunk lease added to the pre-brake delay.",
        delay=(_Embodied.Scenario.WarehouseAisle.tau_delay
               + _Embodied.Scenario.WarehouseAisle.t_lease).to("ms"),  # scenario assumption: lease doubled
    )

    WarehouseScannerSkippedDelay = _st.StoppingBudgetScenario(
        name="Delay terms a certified scanner stop skips on the warehouse base",
        description="A stop triggered by a safety scanner wired to the drives bypasses the "
                    "perception backbone and the chunk lease; their travel at the drive limit.",
        speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
        delay=(_Embodied.Scenario.WarehouseAisle.t_backbone
               + _Embodied.Scenario.WarehouseAisle.t_lease).to("ms"),
        deceleration=_Embodied.Scenario.WarehouseAisle.a_brake,
    )

    WarehouseInferenceHiccup = _st.StoppingBudgetScenario(
        name="A 100 ms inference stall on the warehouse base at its drive limit",
        description="The base repeats its last forward-velocity command while the policy stalls; "
                    "delay_travel is the distance driven with nothing checking it.",
        speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
        delay=Q_(100, "ms"),  # scenario assumption: inference stall
        deceleration=_Embodied.Scenario.WarehouseAisle.a_brake,
    )

    TempeActionSuppression = _st.StoppingBudgetScenario(
        name="Tempe: the one-second action suppression at the recorded speeds",
        description="NTSB HAR-19/03: emergency recognized at 43.2 mph, braking withheld for 1 s "
                    "until 40.5 mph; braking from the recognition speed at the road's 0.92 g.",
        speed=Q_(float(_Literature.IncidentReports.TempeEmergencyRecognitionSpeedMph), "mph"),
        speed_at_delay_end=Q_(float(_Literature.IncidentReports.TempeSuppressionEndSpeedMph), "mph"),
        delay=Q_(float(_Literature.IncidentReports.TempeActionSuppressionS), "s"),
        deceleration=(_Literature.IncidentReports.TempeRoadDecelerationG * Q_(1, "standard_gravity")).to("m/s**2"),
    )

    TempeActionSuppressionAdsCap = TempeActionSuppression.variant(
        name="Tempe: the action suppression with braking at the ADS's own cap",
        description="The suppression scenario with braking at the 7 m/s^2 the ADS allowed itself.",
        deceleration=Q_(float(_Literature.IncidentReports.TempeAdsBrakingCapMps2), "m/s**2"),
    )

    TempeFirstDetectionStop = _st.StoppingBudgetScenario(
        name="Tempe: a full stop at first detection",
        description="NTSB HAR-19/03: about 350 ft from the pedestrian at the initial 44 mph; "
                    "braking at the road's 0.92 g with no delay.",
        speed=Q_(float(_Literature.IncidentReports.TempeInitialSpeedMph), "mph"),
        delay=Q_(0, "s"),
        deceleration=(_Literature.IncidentReports.TempeRoadDecelerationG * Q_(1, "standard_gravity")).to("m/s**2"),
        clear_distance=Q_(float(_Literature.IncidentReports.TempeFirstDetectionDistanceFt), "ft"),
    )

    HydraulicFluidSlickTraction = _st.TractionLossScenario(
        name="Hydraulic fluid on a dry concrete floor under the warehouse base",
        description="Illustrative tire-floor pair, about 0.8 dry and 0.2 with hydraulic fluid, "
                    "against the running machine's credible deceleration with all weight on the "
                    "driven wheels.",
        mu_before=0.8,  # scenario assumption: tire on dry concrete (illustrative)
        mu_after=0.2,  # scenario assumption: tire on spilled hydraulic fluid (illustrative)
        deceleration=_Embodied.Scenario.WarehouseAisle.a_brake,
    )

    OpenVlaComputeGap = _st.DomainComputeGapScenario(
        name="OpenVLA against the largest language model of 2024",
        description="Epoch AI training-compute estimates (Literature.ModelsByDomain).",
        series_name="ModelsByDomain",
        model="OpenVLA",
    )

    Gr00tN1ComputeGap = _st.DomainComputeGapScenario(
        name="GR00T N1 2B against the largest language model of 2025",
        description="Epoch AI training-compute estimates (Literature.ModelsByDomain).",
        series_name="ModelsByDomain",
        model="GR00T N1 2B",
    )

    # --- Agent trajectories (added 2026-10-20) ------------------------------
    UncheckedRepairHorizon = _st.HorizonCompoundingScenario(
        name="Unchecked repair trajectory at 99 percent per step",
        description="Independent steps, no checks, no recovery: success at 10, 100, and 200 turns, "
                    "and the step error that holds 90 percent at 100 turns.",
        step_success=0.99,  # scenario assumption: hypothetical per-step accuracy
        short_horizon=10,  # scenario assumption: turns
        long_horizon=100,  # scenario assumption: turns
        target_success=0.90,  # scenario assumption: success to hold at the long horizon
        extended_horizon=200,  # twice the long horizon: success squares
    )

    HorizonCompounding30Steps = UncheckedRepairHorizon.variant(
        name="Unchecked repair trajectory at the coding-agent step budget",
        description="The same 99 percent per step, priced at the harness's trajectory step budget.",
        short_horizon=_Agents.Coding.SWE_Bench_Runner.max_trajectory_steps,
    )

    RepairAttemptCostLevers = _st.AcceptedTaskCostScenario(
        name="Repair attempt: cheaper tokens or more acceptances",
        description="$1.30 of model tokens and $0.80 of sandbox, tests, and review per attempt, "
                    "accepted 40 percent of the time; halve the token price, or raise acceptance "
                    "to 60 percent; halve the $0.80, or add 10 points of acceptance; five "
                    "independent attempts.",
        model_cost=Q_(1.30, "USD"),  # scenario assumption: tokens per attempt
        other_cost=Q_(0.80, "USD"),  # scenario assumption: sandbox, tests, and review per attempt
        acceptance=0.40,  # scenario assumption
        token_price_cut=0.5,  # scenario assumption: a tariff at half the token price
        improved_acceptance=0.60,  # scenario assumption: after a context or tool improvement
        other_cost_cut=0.5,  # scenario assumption: halve the sandbox, test, and review cost
        acceptance_gain=0.10,  # scenario assumption: a tool change adds 10 points
        independent_attempts=5,  # scenario assumption: retry budget
    )

    AppendOnlyRepairContext = _st.AppendOnlyContextScenario(
        name="Append-only repair transcript over the reference harness's turn budget",
        description="An 8,000-token first prompt plus 1,500 tokens per turn, resent every turn, "
                    "over the SWE-bench harness's step budget and window.",
        initial_tokens=Q_(8000, "token"),  # scenario assumption: first serialized prompt
        tokens_per_turn=Q_(1500, "token"),  # scenario assumption: model output plus observation
        turns=_Agents.Coding.SWE_Bench_Runner.max_trajectory_steps,
        context_window=_Agents.Coding.SWE_Bench_Runner.context_window,
    )

    FrontierTaskHorizonTrend = _st.TaskHorizonTrendScenario(
        name="Frontier 50 percent task horizon, METR Time Horizon v1.1",
        description="Doubling time of the frontier horizon at release, over the whole series and "
                    "since 2023, fit as the source fits it (frontier points at or below 16 h).",
        series_name="AgentTimeHorizon",
        fit_cap=Q_(16, "hour"),  # the source's own fit rule
        recent_from="2023-01-01",  # the source's recent window
        published_recent_doubling=Q_(_Literature.Benchmarks.AgentHorizonDoublingDays2023On, "day"),
    )

    MiniSweAgentSonnet4Run = _st.AgentRunScenario(
        name="mini-swe-agent 1.0.0 with Claude 4 Sonnet on SWE-bench Verified",
        description="500 tasks, one attempt each, bash only; hidden tests decide acceptance.",
        series_name="SweBenchMiniSonnet4Run",
        call_bin_edges=[20, 30, 40, 60],  # scenario assumption: model-call bins for acceptance by length
    )

    AstropyBitmaskRepair = _st.BenchmarkTaskScenario(
        name="astropy__astropy-14995 in the mini-swe-agent Claude 4 Sonnet run",
        description="Mask propagation fails when one NDData operand has no mask (astropy issue "
                    "14978); judged by one new test and the existing tests of its file.",
        run=MiniSweAgentSonnet4Run,
        instance_id="astropy__astropy-14995",
        fail_to_pass_tests=1,  # SWE-bench Verified record: test_nddata_bitmask_arithmetic
        pass_to_pass_tests=179,  # SWE-bench Verified record (princeton-nlp/SWE-bench_Verified)
    )

    # --- Model calls and agent trajectories (added 2026-10-20) ---------------
    # Per-token floors for one model call, then the calls a coding agent and a
    # support agent make. Token counts are teaching assumptions, not
    # measurements; the floors exclude attention, batching, and queueing.
    Llama3_70B_Int8DecodeH100 = _st.DecodeScenario(
        name="Llama-3 70B, 8-bit weights, at batch 1 on one H100",
        description="Per-token floors for one unbatched call: compute_time = 2P / FP8 peak is the "
                    "prefill floor per prompt token; memory_time = P x 1 byte / HBM bandwidth is the "
                    "decode floor per output token.",
        workload=_Models.Language.Llama3_70B,
        hardware=_Hardware.Cloud.H100,
        precision="int8",  # scenario assumption: 8-bit weights
        peak_precision="fp8",  # scenario assumption: prefill at the FP8 tensor-core peak
    )

    CodingAgentResend30Turns = _st.TrajectoryResendScenario(
        name="Coding-agent trajectory that resends its history for 30 turns",
        description="An 8,000-token first prompt; each turn appends 1,500 tokens of model output "
                    "and tool observation and generates 200 output tokens; the harness step budget "
                    "sets the horizon.",
        floors=Llama3_70B_Int8DecodeH100,
        first_prompt=8000,  # scenario assumption: serialized first prompt, tokens
        appended=1500,  # scenario assumption: output plus observation appended per turn, tokens
        output_per_turn=200,  # scenario assumption: output tokens per call
        turns=_Agents.Coding.SWE_Bench_Runner.max_trajectory_steps,
    )
    CodingAgentResend60Turns = CodingAgentResend30Turns.variant(
        name="Coding-agent trajectory that resends its history for 60 turns",
        turns=2 * CodingAgentResend30Turns.turns)
    CodingAgentResend90Turns = CodingAgentResend30Turns.variant(
        name="Coding-agent trajectory that resends its history for 90 turns",
        turns=3 * CodingAgentResend30Turns.turns)

    CodingAgentFirstTurnCall = _st.ModelCallScenario(
        name="First call of the coding-agent trajectory",
        description="The 8,000-token first prompt and a 200-token reply at the 8-bit floors.",
        floors=Llama3_70B_Int8DecodeH100,
        prompt_tokens=CodingAgentResend30Turns.first_prompt,
        output_tokens=CodingAgentResend30Turns.output_per_turn,
    )

    CodingAgentWholeFileEdit = _st.ModelCallScenario(
        name="Coding-agent edit returned as a whole replacement file",
        description="The first prompt plus 350 tokens of format instructions; a 1,850-token "
                    "replacement file; a 15 s call deadline.",
        floors=Llama3_70B_Int8DecodeH100,
        prompt_tokens=CodingAgentResend30Turns.first_prompt,
        instruction_tokens=350,  # scenario assumption: whole-file format instructions
        output_tokens=1850,  # scenario assumption: whole-file response tokens
        deadline=Q_(15, "second"),  # scenario assumption: call deadline
    )
    CodingAgentLocalizedEdit = CodingAgentWholeFileEdit.variant(
        name="Coding-agent edit returned as a localized edit",
        description="The first prompt plus 120 tokens of edit-format instructions; an 80-token "
                    "edit; 95 percent apply and 60 percent of those pass the tests.",
        instruction_tokens=120,  # scenario assumption: localized-edit format instructions
        output_tokens=80,  # scenario assumption: localized-edit response tokens
        stage_success={"applies": 0.95,  # scenario assumption: edits whose anchor text matches
                       "passes tests": 0.60},  # scenario assumption: test pass rate given it applies
    )

    SupportAgentTurnCall = _st.ModelCallScenario(
        name="One support-agent turn at the 8-bit floors",
        description="The support workload's session context and reply length; the shared policy "
                    "prefix can be served from cache.",
        floors=Llama3_70B_Int8DecodeH100,
        prompt_tokens=int(_Agents.Support.CustomerOps.session_context.to("token").magnitude),
        output_tokens=int(_Agents.Support.CustomerOps.reply_tokens.to("token").magnitude),
        shared_prefix_tokens=int(_Agents.Support.CustomerOps.shared_prefix.to("token").magnitude),
    )

    CanceledStreamDecodeSavings = _st.StreamCancelScenario(
        name="Streamed reply rejected at step 96 of a 4,096-token allowance",
        description="A stream monitor rejects the reply early; an honored cancel skips the "
                    "remaining decode steps at the 8-bit decode floor.",
        floors=Llama3_70B_Int8DecodeH100,
        output_allowance=4096,  # scenario assumption: output allowance of the call
        reject_step=96,  # scenario assumption: decode step at which the monitor rejects
    )

    MultimodalContextAdmission = _st.ContextAdmissionScenario(
        name="Text request plus a screenshot against an 8,192-token window",
        description="A 6,000-token serialized text request with 1,024 tokens reserved for the "
                    "reply; the vision front end charges 2,048 positions for the image.",
        window=8192,  # scenario assumption: combined context window, tokens
        output_reserve=1024,  # scenario assumption: reserved generation allowance
        text_tokens=6000,  # scenario assumption: serialized text request
        image_tokens=2048,  # scenario assumption: service-reported image charge
    )
    MultimodalContextAdmissionHalfReserve = MultimodalContextAdmission.variant(
        name="The same request with the reply reserve halved",
        output_reserve=MultimodalContextAdmission.output_reserve // 2)

    TemperatureNucleusExample = _st.SamplingScenario(
        name="Five hypothetical logits at temperature 1 and 0.5, then top-p 0.90",
        description="Logits [12, 10.5, 9, 6, 3]; halving the temperature sharpens the top token; "
                    "top-p 0.90 at temperature 1 keeps two tokens.",
        logits=[12.0, 10.5, 9.0, 6.0, 3.0],  # scenario assumption: hypothetical logits
        base_temperature=1.0,
        sharp_temperature=0.5,  # scenario assumption
        top_p=0.90,  # scenario assumption
    )

    PatchCheckerPosterior = _st.CheckerPosteriorScenario(
        name="A fallible patch check over generated patches",
        description="18 percent of patches violate the concurrency property; the check passes "
                    "65 percent of defective and 85 percent of valid patches.",
        defect_rate=0.18,  # scenario assumption
        false_accept_rate=0.65,  # scenario assumption: P(PASS | defective)
        valid_accept_rate=0.85,  # scenario assumption: P(PASS | valid)
    )
    PatchCheckerHalvedFalseAccept = PatchCheckerPosterior.variant(
        name="The same patch check with its false-accept rate halved",
        false_accept_rate=PatchCheckerPosterior.false_accept_rate / 2)

    LlmCodeTraceTokenMix = _st.TokenMixScenario(
        name="Token mix of a production code-completion trace, priced at the 8-bit floors",
        description="Azure LLM inference trace 2023 (code); the time share prices each request at "
                    "the Llama-3 70B 8-bit floors, an illustration rather than the trace's latency.",
        series_name="LlmInferenceTraceCode2023",
        prompt_column="ContextTokens",
        output_column="GeneratedTokens",
        floors=Llama3_70B_Int8DecodeH100,
    )
    # --- Collective communication (added 2026-10-20) ------------------------
    CriticalSizeByInterconnect = _st.AlphaBetaScenario(
        name="Critical message size n* = alpha beta on five interconnects",
        description="Effective collective-launch alpha at the midpoint of each registered range (NDR at its "
                    "canonical anchor) and each link's per-direction bandwidth.",
        alphas={
            "NVLink": (_SwitchFabric.EffectiveAlphaNvlinkLow + _SwitchFabric.EffectiveAlphaNvlinkHigh) / 2,
            "PCIe Gen5": (_SwitchFabric.EffectiveAlphaPcieGen5Low + _SwitchFabric.EffectiveAlphaPcieGen5High) / 2,
            "RoCE": (_SwitchFabric.EffectiveAlphaRoceLow + _SwitchFabric.EffectiveAlphaRoceHigh) / 2,
            "HDR": (_SwitchFabric.EffectiveAlphaHdrLow + _SwitchFabric.EffectiveAlphaHdrHigh) / 2,
            "NDR": _SwitchFabric.AlphaNdr,
        },
        bandwidths={
            "NVLink": _Hardware.Cloud.H100.nvlink.bandwidth_per_direction,
            "PCIe Gen5": _Hardware.Cloud.H100.interconnect.bandwidth,
            "RoCE": _Fabrics.Ethernet_100G.bandwidth,
            "HDR": _Fabrics.InfiniBand_HDR.bandwidth,
            "NDR": _Fabrics.InfiniBand_NDR.bandwidth,
        },
        messages={})

    RoutingMessageFusionNdr = _st.MessageFusionScenario(
        name="One hundred 4 KB routing messages on NDR, separately and fused",
        description="Each 4 KB message sits far below n*, so alone it costs its startup; fused into one "
                    "400 KB message the batch pays one startup and its bytes.",
        alpha=_SwitchFabric.AlphaNdr, bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
        message=Q_(4, "KB"),  # scenario assumption: MoE routing message
        count=100)  # scenario assumption: messages fused into one

    DdpBucketRingNdr = _st.AllReduceScheduleScenario(
        name="One 25 MB gradient bucket reduced on 64 and on 1,024 NDR ranks",
        description="DDP's common bucket size; the ring's per-round chunk M/N decides the regime.",
        message=Q_(25, "MB"),  # scenario assumption: DDP's common bucket reference size
        group_sizes=[64,  # scenario assumption: eight DGX H100 nodes
                     DenseLLMPretrainingReference.system.total_accelerators],
        alpha=_SwitchFabric.AlphaNdr, bandwidth=_Fabrics.InfiniBand_NDR.bandwidth)

    GradientBufferAllReduceNdr = _st.AllReduceScheduleScenario(
        name="The 1 GB reference gradient buffer reduced on NDR at 8, 64, and 1,024 ranks",
        description="Ring, simple tree, and parameter server for one 1 GB buffer over 50 GB/s ports.",
        message=_ReferenceStats.CollectiveCommunicationProfile.GradientBuffer,
        group_sizes=[8,  # scenario assumption: one node's group
                     64,  # scenario assumption: eight DGX H100 nodes
                     DenseLLMPretrainingReference.system.total_accelerators],
        alpha=_SwitchFabric.AlphaNdr, bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
        run_steps=_ReferenceStats.CollectiveCommunicationProfile.TrainingRunSteps)

    Llama2_70BGradientRing64Ndr = _st.AllReduceScheduleScenario(
        name="A 70B model's BF16 gradient ring-AllReduced on 64 NDR GPUs",
        description="Every rank reduces the full 140 GB gradient; the what-if halves the exposed AllReduce "
                    "on every step of a 100,000-step run.",
        message=_Models.Language.Llama2_70B.size_in_bytes(_BYTES_FP16),
        group_sizes=[64],  # scenario assumption: eight DGX H100 nodes
        alpha=_SwitchFabric.AlphaNdr, bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
        run_steps=_ReferenceStats.CollectiveCommunicationProfile.TrainingRunSteps,
        time_saved_fraction=0.5)  # scenario assumption: an optimization that halves the AllReduce

    RingVsTreeSmallBuffer = _st.AllReduceScheduleScenario(
        name="Ring against tree for a 1 MB buffer on 64 ranks",
        description="Illustrative link: 10 microseconds of startup and 10 GB/s; the buffer sits near the crossover.",
        message=Q_(1, "MB"),  # scenario assumption
        group_sizes=[64],  # scenario assumption
        alpha=Q_(10, "microsecond"),  # scenario assumption
        bandwidth=Q_(10, "GB/s"))  # scenario assumption

    HierarchicalAllReduce8x8DgxH100 = _st.HierarchicalAllReduceScenario(
        name="Hierarchical AllReduce of the 1 GB buffer on eight DGX H100 nodes",
        description="NVLink ReduceScatter, one NDR ring per GPU position on the 125 MB shard, NVLink AllGather.",
        message=_ReferenceStats.CollectiveCommunicationProfile.GradientBuffer,
        n_nodes=8,  # scenario assumption: a 64-GPU cluster
        gpus_per_node=_Nodes.DGX_H100.accelerators_per_node,
        intra_bandwidth=_Hardware.Cloud.H100.nvlink.bandwidth_per_direction,
        inter_bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
        inter_alpha=_SwitchFabric.AlphaNdr)

    HierarchicalAllReduceBucket100MB = HierarchicalAllReduce8x8DgxH100.variant(
        name="Hierarchical AllReduce of one 100 MB bucket on eight DGX H100 nodes",
        description="The same three phases at the ideal rate, for one overlap bucket.",
        message=Q_(100, "MB"))  # scenario assumption: AllReduce bucket size

    HierarchicalAllReduceReferenceCluster = HierarchicalAllReduce8x8DgxH100.variant(
        name="Hierarchical AllReduce of the 175B BF16 gradient on the 1,024-GPU reference cluster",
        description="128 DGX H100 nodes of eight; flat ring across all 1,024 GPUs for comparison.",
        message=DenseLLMPretrainingReference.workload.size_in_bytes(_BYTES_FP16),
        n_nodes=DenseLLMPretrainingReference.system.count,
        gpus_per_node=DenseLLMPretrainingReference.system.node.accelerators_per_node)

    MoeDispatch8Gpu64Experts = _st.AllToAllDispatchScenario(
        name="MoE dispatch of 4,096 tokens to 64 experts on eight GPUs over NDR",
        description="Uniform routing: each GPU keeps 64 of its 512 tokens and sends 64 to each of seven peers; "
                    "one GPU's experts draw an assumed 1.5 times the uniform remote load.",
        total_tokens=4096,  # scenario assumption
        n_devices=8,  # scenario assumption
        n_experts=64,  # scenario assumption
        hidden_dim=2048,  # scenario assumption
        bytes_per_element=_BYTES_FP16,
        alpha=_SwitchFabric.AlphaNdr, bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
        receive_skew=1.5)  # scenario assumption: realized remote-receive skew on one rank

    RingTraceFourGpu = _st.RingTraceScenario(
        name="Ring AllReduce traced on four GPUs with four-element vectors",
        description="Small integers so every partial sum can be followed by hand.",
        vectors=[[1, 5, 3, 7], [2, 6, 4, 8], [3, 7, 5, 9], [4, 8, 6, 10]])  # scenario assumption

    ErrorFeedbackFiveSteps = _st.ErrorFeedbackScenario(
        name="Error feedback on five gradients below half a unit",
        description="A compressor that rounds to whole units sends nothing without a residual.",
        gradients=[0.4, 0.3, 0.2, 0.4, 0.3])  # scenario assumption

    ErrorFeedbackConstantGradient = _st.ErrorFeedbackScenario(
        name="Error feedback on a constant 0.3 gradient for ten steps",
        description="The residual sends a whole unit about three times in ten steps.",
        gradients=[0.3] * 10)  # scenario assumption

    CompressionPayback8x = _st.CompressionPaybackScenario(
        name="Compressing one 40 ms synchronization by an effective 8x",
        description="2 ms of encoding and decoding; on a fast NVLink domain the same buffer takes 1 ms.",
        communication_time=Q_(40, "ms"),  # scenario assumption: uncompressed synchronization
        compression_ratio=8,  # scenario assumption: effective gain after metadata and protocol overhead
        codec_time=Q_(2, "ms"),  # scenario assumption: quantize and dequantize
        fast_communication_time=Q_(1, "ms"))  # scenario assumption: the same buffer over NVLink

    Int8CompressionRun = _st.CompressionRunScenario(
        name="INT8 compression with error feedback, time to target on a slow and a fast network",
        description="100,000 steps of 300 ms compute; INT8 against FP32 cuts communication fourfold but "
                    "needs 10 percent more steps.",
        steps=100_000,  # scenario assumption: steps to target quality, uncompressed
        compute_time=Q_(300, "ms"),  # scenario assumption
        communication_times={"slow": Q_(200, "ms"),  # scenario assumption
                             "fast": Q_(30, "ms")},  # scenario assumption
        compression_ratio=(_BYTES_FP32 / _BYTES_INT8).to("").magnitude,
        step_penalty=0.10)  # scenario assumption: extra steps from compression noise

    GradientEncodingRatios = _st.GradientEncodingScenario(
        name="Gradient encodings against FP32 and BF16 payloads",
        description="FP16, INT8, top-1 percent with 32-bit indices, and 1-bit signs; upper bounds before metadata.",
        baseline_bits={"FP32": _BYTES_FP32.to("bit"), "BF16": _BYTES_FP16.to("bit")},
        value_bits={"FP16": _BYTES_FP16.to("bit"), "INT8": _BYTES_INT8.to("bit"),
                    "Top-k": _BYTES_FP32.to("bit"), "1-bit": Q_(1, "bit")},
        kept_fractions={"Top-k": 0.01},  # scenario assumption: keep the largest 1 percent
        index_bits={"Top-k": _BYTES_FP32.to("bit")})  # 32-bit index per kept value

    ShardedDataParallelCalls7B = _st.CollectiveCallsScenario(
        name="Collective calls per step for a 7B model under FSDP",
        description="AllGather before and ReduceScatter after each layer; with resharding after forward, a "
                    "second AllGather in backward.",
        workload=_Models.Language.Llama2_7B,
        calls_per_layer={"FSDP": 2, "FSDP, reshard after forward": 3})

    OverlapBudget7B64Gpu = _st.OverlapBudgetScenario(
        name="Overlap budget for a 7B transformer on 64 GPUs",
        description="32 equal layers of FP32 gradients in 100 MB buckets at an assumed 3 ms each, against "
                    "15 ms of backward per layer.",
        workload=_Models.Language.Llama2_7B,
        bytes_per_value=_BYTES_FP32,
        backward_per_layer=Q_(15, "ms"),  # scenario assumption
        bucket_size=HierarchicalAllReduceBucket100MB.message,
        time_per_bucket=Q_(3, "ms"),  # scenario assumption: achieved time per bucket, final partial included
        n_devices=HierarchicalAllReduceBucket100MB.n_devices)

    OverlapBudget7B64GpuIdealBuckets = OverlapBudget7B64Gpu.variant(
        name="The same overlap budget at the hierarchical schedule's ideal bucket time",
        description="Each 100 MB bucket at the three-phase model's time on eight DGX H100 nodes.",
        time_per_bucket=HierarchicalAllReduceBucket100MB.result.total_time)

    # --- Distributed training (added 2026-10-20) ----------------------------
    # The 175B reference model on the 1,024-GPU reference cluster, laid out
    # four ways. All share one step: 2M tokens per update at 6 FLOPs per
    # parameter per token, H100 FP8 peak at 50 percent, FP16 gradients over one
    # NDR port per GPU, and up to 75 percent of each AllReduce hidden under
    # compute. The compute window is carried at 0.1 s, the precision the napkin
    # math prints, so every downstream ratio recomputes from the printed value.
    Gpt3DataParallel1K = _st.ParallelLayoutScenario(
        name="175B pure data parallelism on 1,024 H100s",
        description="Every GPU holds the full model and AllReduces the full FP16 gradient over NDR.",
        workload=_Models.Language.GPT3,
        dp_size=_Clusters.Training_1K.total_accelerators,
        batch_tokens=2e6,  # scenario assumption: tokens per optimizer update
        flops_per_param_token=float(_Literature.Chinchilla.ComputeConstant),
        accelerator=_Hardware.Cloud.H100,
        precision="fp8",
        local_efficiency=0.5,  # scenario assumption: local share of peak before communication
        link_bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
        overlap_fraction=0.75,  # scenario assumption: optimistic overlap schedule
        window_resolution=Q_(0.1, "second"),
        # Budget against the published "80 GB" label (80 GiB, labeled GB by the vendor).
        device_memory=Q_(_Hardware.Cloud.H100.memory.capacity.to("GiB").magnitude, "GB"),
    )

    Gpt3TensorDataParallel1K = Gpt3DataParallel1K.variant(
        name="175B with TP 8 inside each node and DP 128 across nodes",
        description="Each rank AllReduces one eighth of the gradient; state is split eight ways.",
        tp_size=_Nodes.DGX_H100.accelerators_per_node,
        dp_size=_Clusters.Training_1K.total_accelerators // _Nodes.DGX_H100.accelerators_per_node,
    )

    Gpt3ThreeDParallel1K = Gpt3DataParallel1K.variant(
        name="175B with TP 8, PP 4, and DP 32 on 1,024 H100s",
        description="Adds four pipeline stages at 32 microbatches; the DP shard falls to 1/32.",
        tp_size=_Nodes.DGX_H100.accelerators_per_node,
        pp_size=4,  # scenario assumption: pipeline stages across nodes
        dp_size=_Clusters.Training_1K.total_accelerators // (4 * _Nodes.DGX_H100.accelerators_per_node),
        microbatches=32,  # scenario assumption: microbatches per step
    )

    Gpt3ThreeDParallel16K = Gpt3ThreeDParallel1K.variant(
        name="175B with TP 8, PP 16, and DP 128 on 16,384 H100s",
        description="A sixteen-stage layout at the same 32 microbatches; a deeper pipeline, a larger bubble.",
        pp_size=16,  # scenario assumption
        dp_size=128,  # scenario assumption
    )

    Gpt3ActivationsMicrobatch4 = _st.ActivationMemoryScenario(
        name="175B activations at four sequences per microbatch",
        description="Full-context GPT-3 activations per replica, with and without recomputation.",
        workload=_Models.Language.GPT3,
        micro_batch=4,  # scenario assumption: per-GPU microbatch
    )

    Gpt3ActivationsMicrobatch1 = Gpt3ActivationsMicrobatch4.variant(
        name="175B activations at one sequence per microbatch",
        micro_batch=1)  # scenario assumption: single-sequence microbatch

    Llama2_7B_ZeroStages64 = _st.ZeroShardingScenario(
        name="7B ZeRO stages on 64 GPUs",
        description="Mixed-precision Adam state of a 7B model replicated, then sharded stage by stage.",
        workload=_Models.Language.Llama2_7B,
        n_devices=64,  # scenario assumption: data-parallel group
    )

    Gpt3Zero3On64 = _st.ZeroShardingScenario(
        name="175B fully sharded over 64 GPUs",
        description="ZeRO-3 state plus fully checkpointed activations of one sequence; every layer "
                    "regathered over one NDR port per GPU.",
        workload=_Models.Language.GPT3,
        n_devices=64,  # scenario assumption: one sixteenth of the reference cluster
        activations=Gpt3ActivationsMicrobatch1,
        activation_strategy="full",
        link_bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
    )

    Gpt2DataParallelOneNode = _st.DataParallelScaleOutScenario(
        name="GPT-2 data parallelism on one DGX H100 node",
        description="Eight GPUs at the one-GPU profile's local batch, gradients AllReduced over NVLink.",
        workload=_Models.Language.GPT2,
        step_time=_ReferenceStats.DataParallelScalingProfile.SingleGpuStepTime,
        training_time=_ReferenceStats.DataParallelScalingProfile.SingleGpuTrainingTime,
        per_device_batch=int(_ReferenceStats.DataParallelScalingProfile.PerGpuBatch),
        devices_per_node=_Nodes.DGX_H100.accelerators_per_node,
        intra_bandwidth=_Hardware.Cloud.H100.nvlink.bandwidth_per_direction,
        node_hour_rate=_Nodes.DGX_H100.accelerators_per_node * _Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
    )

    Gpt2DataParallelFourNodes10GbE = Gpt2DataParallelOneNode.variant(
        name="GPT-2 data parallelism on four nodes over 10 Gb Ethernet",
        description="The same local batch on 32 GPUs; the cross-node ring runs at 10 GbE.",
        nodes=4,  # scenario assumption: four-node scale-out
        inter_bandwidth=_Fabrics.Ethernet_10G.bandwidth,
    )

    Gpt2GradientAccumulationOneNode = Gpt2DataParallelOneNode.variant(
        name="GPT-2 on one node with four-step gradient accumulation",
        description="Reaches the four-node global batch without leaving NVLink.",
        accumulation_steps=4,  # scenario assumption
    )

    GPipeFourStageWalkthrough = _st.PipelineScheduleScenario(
        name="GPipe, four stages and four microbatches",
        description="The schedule drawn slot by slot: 14 slots per step, 6 idle per stage.",
        stages=4, microbatches=4)  # scenario assumption: the drawn schedule

    Pipeline8Stage32Micro = _st.PipelineScheduleScenario(
        name="Eight pipeline stages at 32 microbatches",
        description="A cross-node pipeline of eight stages.", stages=8, microbatches=32)

    Pipeline16Stage32Micro = _st.PipelineScheduleScenario(
        name="Sixteen pipeline stages at 32 microbatches",
        description="The stage count of the 16,384-GPU layout.", stages=16, microbatches=32)

    Pipeline8Stage8Micro = _st.PipelineScheduleScenario(
        name="Eight pipeline stages at eight microbatches",
        description="Too few microbatches for the depth: about half of each step idles.",
        stages=8, microbatches=8)

    Gpt3TensorParallelLayerTp8 = _st.TensorParallelLayerScenario(
        name="One 175B layer under 8-way tensor parallelism",
        description="One 2,048-token sequence at half the H100 dense FP16 peak; two activation "
                    "AllReduces per layer over NVLink or one NDR port.",
        workload=_Models.Language.GPT3,
        tp_size=_Nodes.DGX_H100.accelerators_per_node,
        tokens=_Models.Language.GPT3.context_length,  # scenario assumption: one sequence per microbatch
        accelerator=_Hardware.Cloud.H100,
        local_efficiency=0.5,  # scenario assumption: same local efficiency as the data-parallel step
        links={"NVLink": _Hardware.Cloud.H100.nvlink.bandwidth_per_direction,
               "InfiniBand NDR": _Fabrics.InfiniBand_NDR.bandwidth},
    )

    H100RingAttentionChunk = _st.RingAttentionScenario(
        name="Ring attention chunk size on H100",
        description="FP16 keys and values at half the dense FP16 peak, over NVLink or one NDR port.",
        accelerator=_Hardware.Cloud.H100,
        local_efficiency=0.5,  # scenario assumption: 50 percent of dense peak
        element_bytes=_BYTES_FP16,
        links={"NVLink": _Hardware.Cloud.H100.nvlink.bandwidth_per_direction,
               "InfiniBand NDR": _Fabrics.InfiniBand_NDR.bandwidth},
    )

    # Scaling a transformer from 8 to 64 workers at 32 examples per worker.
    # Iteration counts and communication shares are hypothetical inputs, not
    # measured results; one 0.4 s local iteration is shared by every run.
    TransformerBsp8Workers = _st.SyncStrategyScenario(
        name="Eight workers, bulk synchronous",
        description="12,800 iterations to the target at 15 percent communication overhead.",
        workers=8, per_worker_batch=32,
        iterations=12_800, communication_overhead=0.15,  # scenario assumption
        base_iterations=100_000,  # scenario assumption: one worker to the same target
        base_iteration_time=Q_(0.4, "second"),  # scenario assumption: local compute per iteration
        device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
    )

    TransformerBsp64Workers = TransformerBsp8Workers.variant(
        name="64 workers, bulk synchronous",
        description="1,720 iterations at 45 percent communication overhead.",
        workers=64, iterations=1_720, communication_overhead=0.45)  # scenario assumption

    TransformerSsp64Workers = TransformerBsp8Workers.variant(
        name="64 workers, stale synchronous (s = 4)",
        description="Less waiting, 25 percent overhead, but stale updates need 2,100 iterations.",
        workers=64, iterations=2_100, communication_overhead=0.25, staleness=4)  # scenario assumption

    CriticalBatchPilotTrace7B = _st.CriticalBatchScenario(
        name="Critical batch of a 7B pilot trace on 64 GPUs",
        description="Gradient-noise estimate from a pilot run against a proposed accumulated batch.",
        gradient_noise_trace=1.6e6,  # scenario assumption: tr(Sigma) from a pilot trace
        gradient_norm_sq=100,  # scenario assumption: squared true-gradient norm
        local_batch=4, accumulation_steps=32,  # scenario assumption: proposed configuration
        devices=64,
    )

    Llama2_70B_PpoAlignment = _st.AlignmentMemoryScenario(
        name="PPO and DPO alignment of a 70B policy",
        description="70B policy and reference with 13B reward and value models; policy under "
                    "TP 8 x PP 4 with 8 replicas; rollouts of 256 sequences of 1,024 tokens.",
        policy=_Models.Language.Llama2_70B,
        critic=_Models.Language.Llama2_13B,
        tp_size=8, pp_size=4, dp_size=8,  # scenario assumption: policy layout on 256 H100s
        rollout_sequences=256, rollout_tokens=1024,  # scenario assumption
        device_memory=Q_(_Hardware.Cloud.H100.memory.capacity.to("GiB").magnitude, "GB"),
    )

    MLPerfLlmScalingSeries = _st.ScalingSeriesScenario(
        name="MLPerf Training LLM series at several sizes",
        description="Five same-round series of the GPT-3 and Llama 3.1 405B benchmarks.",
        series_name="MLPerfTrainingLLMScaling",
        specs=[
            ("GPT-3, NVIDIA H100 (v3.1)", ("v3.1", "NVIDIA", "gpt3", "H100-SXM5")),
            ("GPT-3, Google TPU v5p (v4.1)", ("v4.1", "Google", "gpt3", "TPU-v5p")),
            ("GPT-3, Google Trillium (v4.1)", ("v4.1", "Google", "gpt3", "TPU-trillium")),
            ("Llama 3.1 405B, NVIDIA H100 (v5.0)", ("v5.0", "NVIDIA", "llama31_405b", "H100-SXM5")),
            ("Llama 3.1 405B, GB200 (v5.0)", ("v5.0", "IBM+CoreWeave", "llama31_405b", "Blackwell")),
        ],
    )

    GpuFailuresTraining1K = _st.FailureIntervalScenario(
        name="GPU failures across the 1,024-GPU reference cluster",
        description="Independent GPU failures at the per-GPU MTTF over a two-week run.",
        component_mtbf=Q_(_Reliability.Gpu.mttf_hours, "hour"),
        n_components=_Clusters.Training_1K.total_accelerators,
        run_duration=Q_(14, "day"),  # scenario assumption: two-week training run
    )

    StragglerFreeStep1K = _st.AvailabilityScenario(
        name="A synchronous step on 1,000 workers, each late 1 percent of the time",
        description="A step is on time only when every worker is; independent stragglers.",
        availability=0.99,  # scenario assumption: per-worker on-time probability per step
        n_components=1000,  # scenario assumption
    )

    NaiveFourStagePipeline = GPipeFourStageWalkthrough.variant(
        name="One batch through four stages, no microbatching",
        description="Each stage works two of eight slots and idles six.",
        microbatches=1)

    ResidualOnePercent512 = _st.ResidualShareScenario(
        name="A 1 percent residual on 512 devices",
        description="A residual equal to 1 percent of the one-device step fills most of the distributed step.",
        residual=0.01,  # scenario assumption: residual fraction of the one-device step
        devices=512)  # scenario assumption

    NotableModelMemoryWall = _st.MemoryWallScenario(
        name="Training state of notable models against accelerator memory",
        description="Epoch AI notable-model parameters at 16 B per parameter against K80 to B200 memory.",
        models=[("AlexNet", "AlexNet"), ("VGG-16", "VGG16"), ("ResNet-152", "ResNet-152 (ImageNet)"),
                ("BERT-Large", "BERT-Large"), ("GPT-2", "GPT-2 (1.5B)"), ("GPT-3", "GPT-3 175B (davinci)"),
                ("PaLM", "PaLM (540B)"), ("Llama 3 (405B)", "Llama 3.1-405B")],
        accelerators=["K80", "V100", "A100", "H100", "B200"],
        bytes_per_param=_BYTES_TRAINING_STATE,
    )

    # --- Single-machine iron law, Amdahl, and silent degradation (added 2026-10-20) ---
    ImageNetInputSpace = _st.InputSpaceScenario(
        name="Every possible ImageNet-sized input",
        description="224 x 224 RGB images at 8 bits per channel, against the 50,000 validation images.",
        dataset=_Datasets.ImageNet,
        levels=_COLOR_DEPTH_8BIT,
    )

    Gpt3TrainingTime1KA100 = _st.ClusterTrainingTimeScenario(
        name="GPT-3's training work on 1,024 A100s",
        description="GPT-3's 3.14e23 FLOPs at the A100's 16-bit peak, 45 percent end-to-end utilization.",
        workload=_Models.Language.GPT3,
        hardware=_Hardware.Cloud.A100,
        n_devices=int(_Clusters.Training_1K.total_accelerators),
        efficiency=0.45,  # scenario assumption: end-to-end utilization
    )

    Gpt3TrainingTime1KA100HigherUtilization = Gpt3TrainingTime1KA100.variant(
        name="The same run at higher utilization",
        description="The same work and hardware at 60 percent utilization.",
        efficiency=0.60,  # scenario assumption: utilization after scheduling improvements
    )

    IronLawMemoryBoundExample = _st.IronLawScenario(
        name="A memory-bound task on one accelerator",
        description="Moves 2 GB at 100 GB/s and runs 1 GFLOP at an effective 100 GFLOP/s, plus 5 ms "
                    "overhead; the upgrade doubles bandwidth.",
        data_volume=Q_(2, "GB"),                # scenario assumption
        bandwidth=Q_(100, "GB/s"),              # scenario assumption: effective rate
        operations=Q_(1, "GFLOP"),              # scenario assumption
        effective_rate=Q_(100, "GFLOP/s"),      # scenario assumption: peak times utilization
        overhead=Q_(5, "ms"),                   # scenario assumption
        memory_speedup=2.0,                     # scenario assumption: twice the bandwidth
    )

    IronLawComputeHeavyExample = _st.IronLawScenario(
        name="A compute-heavy task with a faster arithmetic rate",
        description="2 s of data movement, 8 s of computation, 1 s of overhead; the upgrade doubles "
                    "the arithmetic rate.",
        memory_time_stated=Q_(2, "second"),     # scenario assumption
        compute_time_stated=Q_(8, "second"),    # scenario assumption
        overhead=Q_(1, "second"),               # scenario assumption
        compute_speedup=2.0,                    # scenario assumption
    )

    ResNet50Batch64StepH100 = _st.SingleNodeScenario(
        name="ResNet-50 inference at batch 64 on one H100",
        description="FP16, 50 percent of peak; the comparison doubles memory bandwidth.",
        workload=_Models.Vision.ResNet50,
        hardware=_Hardware.Cloud.H100,
        batch_size=64,          # scenario assumption
        precision="fp16",
        efficiency=0.5,         # scenario assumption
        memory_speedup=2.0,     # scenario assumption: twice the bandwidth
    )

    Gpt2XlDecodeStepH100 = ResNet50Batch64StepH100.variant(
        name="GPT-2 XL decoding one token for one user on one H100",
        description="The same H100, precision, and utilization; batch 1.",
        workload=_Models.Language.GPT2,
        batch_size=1,           # scenario assumption: one user
    )

    ModelSpeedupRequestPath = _st.AmdahlScenario(
        name="A request path whose model gets 3x faster",
        description="60 ms preprocessing, 45 ms model, 25 ms postprocessing; only the model speeds up.",
        stages={"preprocess": Q_(60, "ms"), "model": Q_(45, "ms"), "postprocess": Q_(25, "ms")},  # scenario assumption
        stage="model",
        speedup=3.0,            # scenario assumption: better kernels and quantization
    )

    PedestrianRecallWinterShift = _st.RecallShiftScenario(
        name="A pedestrian detector in winter conditions",
        description="Recall falls from 95 to 85 percent under winter light and clothing rare in training.",
        recall_before=0.95,     # scenario assumption
        recall_after=0.85,      # scenario assumption
    )

    AccountAgeUnitSkew = _st.FeatureUnitSkewScenario(
        name="Account age in days for training, seconds for serving",
        description="A 412-day-old account as the serving pipeline computes it.",
        value=Q_(412, "day"),   # scenario assumption
        serving_unit="second",
    )

    ImageNetV2Shift = _st.ShiftedTestSetScenario(
        name="ImageNet models on ImageNetV2",
        description="67 published models on the original test set and on one re-collected by the "
                    "same procedure (Recht et al. 2019); fv_* are Fisher-vector baselines.",
        series_name="ImageNetV2MatchedFrequency",
        original_key="original_top1", new_key="imagenet_v2_top1", gap_key="gap",
        baseline_prefix="fv_",
    )

    GoogleFluTrendsDrift = _st.ProxyDriftScenario(
        name="Google Flu Trends against the CDC count",
        description="Seasons 2003-04 to 2012-13, weeks 40 to 20; drift from 2011-12; peak season 2012-13.",
        series_name="GoogleFluTrendsVsCdc",
        proxy_key="gft_ili_pct", reference_key="cdc_wili_pct",
        first_year=2003, last_year=2012,        # seasons with both series throughout
        start_week=40, end_week=20,             # the CDC influenza season
        first_drift_year=2011,                  # Lazer et al.: misses from August 2011
        peak_year=2012,
    )

    # --- Deployment paradigms: where a model can run (added 2026-10-20) -----
    # The light barrier, data off the device, the roofline on one node, battery
    # and heat, and splitting a pipeline across tiers. Inputs not read from a
    # registry are marked "scenario assumption".

    RoboticArmRegionalCloud = _st.PropagationScenario(
        name="A robot arm's safety stop served from a regional cloud",
        description="The model runs 1,500 km away; the stop must happen within 10 ms of seeing a person.",
        distance=Q_(1500, "km"),          # scenario assumption: distance to the cloud region
        budget=Q_(10, "ms"),              # scenario assumption: safety response budget
    )
    CrossCountryFiberPath = _st.PropagationScenario(
        name="A sensor packet from California to a data center in Virginia",
        description="Straight-line coast-to-coast distance against a sub-10 ms control budget.",
        distance=Q_(3600, "km"),          # scenario assumption: California to Virginia, straight line
        budget=Q_(10, "ms"),              # scenario assumption: real-time control budget
    )
    AwsUsEast1Routes = _st.MeasuredRoutesScenario(
        name="AWS round trips from N. Virginia to four regions",
        description="One-week median round trips measured between AWS regions, against each fiber floor.",
        series_name="AwsInterRegionRtt",
    )
    FactoryCameras1080p = _st.SensorUplinkScenario(
        name="100 factory cameras at 1080p on a 10 Gb/s link",
        description="Raw RGB video from a quality-control floor against the site link, and defect records "
                    "an edge server sends instead.",
        frame_width=_VIDEO_1080P_WIDTH,
        frame_height=_VIDEO_1080P_HEIGHT,
        bytes_per_pixel=_VIDEO_BYTES_PER_PIXEL_RGB,
        frame_rate=_VIDEO_FPS_STANDARD,
        streams=100,                      # scenario assumption: cameras on the floor
        link_bandwidth=_Fabrics.Ethernet_10G.bandwidth,
        event_size=Q_(1, "KB"),           # scenario assumption: one defect record
        event_rate=Q_(20, "1/second"),    # scenario assumption: detections per second across the floor
    )
    Drone4KObstacleOffload = _st.OffloadDeadlineScenario(
        name="A drone offloading 4K obstacle avoidance over cellular",
        description="One uncompressed 4K RGB frame per decision over a 100 Mb/s uplink, against the "
                    "60 FPS frame interval.",
        payload=_VIDEO_4K_WIDTH * _VIDEO_4K_HEIGHT * _VIDEO_BYTES_PER_PIXEL_RGB,
        uplink_bandwidth=Q_(100, "Mbit/s"),   # scenario assumption: cellular uplink
        network_latency=Q_(100, "ms"),        # scenario assumption: cloud round trip
        remote_compute_time=Q_(10, "ms"),     # scenario assumption: remote inference
        frame_rate=Q_(60, "Hz"),              # scenario assumption: camera frame rate
    )
    Drone4KObstacleOffloadCompressed = Drone4KObstacleOffload.variant(
        name="The same drone frame after 100-fold compression",
        compression=100.0,                    # scenario assumption: video compression ratio
    )
    ResNet50A100Batch1 = _st.SingleNodeInferenceScenario(
        name="ResNet-50, one image, on an A100 at FP16 and peak rates",
        description="Batch-1 inference with weights and activations in 16 bits; efficiency 1, so both "
                    "times are lower bounds.",
        workload=_Models.Vision.ResNet50,
        hardware=_Hardware.Cloud.A100,
        precision="fp16",
        efficiency=1.0,                       # scenario assumption: peak rates (lower bounds)
    )
    ResNet50PhoneNpuBatch1 = _st.SingleNodeInferenceScenario(
        name="ResNet-50, one image, on a flagship phone NPU at INT8 and peak rates",
        description="Batch-1 inference on the iPhone 15 Pro's neural engine; efficiency 1.",
        workload=_Models.Vision.ResNet50,
        hardware=_Hardware.Mobile.iPhone15Pro,
        precision="int8",
        efficiency=1.0,                       # scenario assumption: peak rates (lower bounds)
    )
    A100AtTenFlopPerByte = _st.RooflinePointScenario(
        name="A workload of 10 FLOP per byte on the A100 roofline",
        description="Self-check: which wall stops a kernel that does 10 operations per byte moved?",
        hardware=_Hardware.Cloud.A100,
        intensity=Q_(10, "flop/byte"),        # scenario assumption: self-check intensity
    )
    TpuV1ProductionRoofline = _st.PublishedRooflineScenario(
        name="Six production applications on the first TPU's roofline",
        description="Measured 2016 rates against the 92 TOPS ceiling and the 1,350 ops-per-weight-byte ridge.",
        peak=_Hardware.Cloud.TPUv1.compute.precision_flops["int8"],
        ridge=float(_Literature.DeviceMeasurements.TpuV1RooflineRidge),
        series_name="TpuV1ProductionApps",
    )
    TPUv4Pod4096 = _st.AcceleratorPodScenario(
        name="A TPU v4 pod of 4,096 chips",
        description="Aggregate BF16 peak and HBM of the pod envelope's chips.",
        pod=_Pods.TPUv4_4096,
        accelerator=_Hardware.Cloud.TPUv4,
    )
    DoubledClockPower = _st.ClockScalingScenario(
        name="Doubling a processor's clock",
        description="Dynamic power when voltage must rise with frequency.",
        frequency_ratio=2.0,                  # scenario assumption: clock doubled
    )
    IllustrativeMemoryWall = _st.GrowthGapScenario(
        name="Compute against memory bandwidth, illustrative rates",
        description="Compute doubling every 18 months against bandwidth growing 20 percent a year, "
                    "over a decade.",
        compute_doubling_period=Q_(18, "month"),  # scenario assumption: illustrative rate
        bandwidth_growth=0.20,                    # scenario assumption: illustrative rate
        horizon=Q_(10, "year"),
    )
    MemoryWallFlopsTrend = _st.TrendFitScenario(
        name="Server peak FLOP/s, fitted growth every two years",
        description="Least-squares fit on the log of the digitized markers, as the source fits them.",
        series_name="MemoryWallHardwareScaling", group_key="series", group="flops",
        x_key="year", y_key="normalized", period=2.0,
    )
    MemoryWallDramTrend = MemoryWallFlopsTrend.variant(
        name="DRAM bandwidth, fitted growth every two years", group="dram")
    MemoryWallInterconnectTrend = MemoryWallFlopsTrend.variant(
        name="Interconnect bandwidth, fitted growth every two years", group="interconnect")
    MicroprocessorTrendsAroundDennard = _st.TrendWindowScenario(
        name="Microprocessor trends around the end of Dennard scaling",
        description="Series maxima over the mid-1990s, the mid-2000s, and the early 2020s.",
        series_name="MicroprocessorTrends",
        windows={"mid_1990s": (1994, 1996), "mid_2000s": (2004, 2006),   # scenario assumption: windows
                 "after_2005": (2005, 2023), "early_2020s": (2019, 2023)},
    )
    AlwaysOnObjectDetectorPhone = _st.BatteryBudgetScenario(
        name="An always-on object detector on a phone",
        description="A background detector drawing 2 W continuously against a 15 Wh battery and a day.",
        battery_energy=_Hardware.Mobile.iPhone15Pro.battery_capacity,
        power_draw=_ReferenceStats.MobilePower.ObjectDetector,
        required_runtime=Q_(24, "hour"),      # scenario assumption: a full day of use
    )
    UnoptimizedVisionModelPhone = _st.ThermalCeilingScenario(
        name="An unoptimized vision model on a passively cooled phone",
        description="A 12 W model heats the SoC until it throttles; a 4x lighter redesign lands at the "
                    "sustained ceiling.",
        power=Q_(12, "watt"),                 # scenario assumption: unoptimized model power
        reduction_factor=4.0,                 # scenario assumption: lighter INT8 redesign
        sustained_limit=_ReferenceStats.MobilePower.MobileNpuTypical,
        heating_rate=Q_(1, "delta_degC/second"),  # scenario assumption: representative trace
        trip_time=Q_(60, "second"),           # scenario assumption: time to the trip point
        trip_temperature=Q_(80, "degC"),      # scenario assumption: thermal trip point
        burst_rate=100.0,                     # scenario assumption: frames per second before
        throttled_rate=30.0,                  # scenario assumption: frames per second after
    )
    MobileNetV2PhoneBatch1 = _st.SingleNodeInferenceScenario(
        name="MobileNetV2, one image, on a flagship phone",
        description="Whole-phone energy over the batch-1 window (solver defaults).",
        workload=_Models.Vision.MobileNetV2,
        hardware=_Hardware.Mobile.iPhone15Pro,
    )
    EnergyPerInferenceTiers = _st.EnergyPerInferenceScenario(
        name="Energy per inference across deployment tiers",
        description="Device-level estimates for three vision runs, between a TinyML keyword-spotting "
                    "anchor and a cloud LLM-query anchor; the tasks and boundaries differ.",
        anchors={
            "kws": Q_(10, "uJ"),          # scenario assumption: KWS on dedicated low-power audio silicon
            "llm_query": Q_(1, "kJ"),     # scenario assumption: cloud LLM query incl. data-center overhead
        },
        runs={
            "mobilenet_phone": MobileNetV2PhoneBatch1,
            "resnet_edge": _st.SingleNodeInferenceScenario(
                name="ResNet-50, one image, on a Jetson Orin NX", description="Solver defaults.",
                workload=_Models.Vision.ResNet50, hardware=_Hardware.Edge.JetsonOrinNX),
            "resnet_cloud": _st.SingleNodeInferenceScenario(
                name="ResNet-50, one image, on an A100", description="Solver defaults.",
                workload=_Models.Vision.ResNet50, hardware=_Hardware.Cloud.A100),
        },
    )
    RadioUploadVsMobileNetV2 = _st.OffloadEnergyScenario(
        name="Sending 1 MB by radio against one MobileNetV2 inference on the phone",
        description="Wi-Fi/LTE transmission energy against the NPU alone and against the whole phone.",
        payload=Q_(1, "MB"),                  # scenario assumption: one raw sensor payload
        transmit_energy_per_byte=_NetworkEnergy.Per5gMb,
        local_run=MobileNetV2PhoneBatch1,
    )
    SmartDoorbellFrame = _st.SingleNodeInferenceScenario(
        name="One smart-doorbell frame on the ESP32-S3",
        description="The doorbell's wake-vision model at 16 bits, decomposed into compute and overhead.",
        workload=SmartDoorbell.workload,
        hardware=SmartDoorbell.system,
        precision="fp16",
    )
    SmartDoorbellResponsiveChime = SmartDoorbell.model_copy(update={
        "name": "Smart Doorbell, responsive chime",
        "sla_latency": Q_(50, "ms"),      # scenario assumption: tightened chime budget
    })
    EmergencyBraking100kmh = _st.TravelDuringLatencyScenario(
        name="Emergency braking at 100 km/h under a 100 ms budget",
        description="Distance covered before braking begins, against a cloud round trip of 50 to 150 ms.",
        speed=Q_(100, "km/hour"),             # scenario assumption
        latency=Q_(100, "ms"),                # scenario assumption: end-to-end budget
        remote_delay_low=Q_(50, "ms"),        # scenario assumption: cloud round trip, best case
        remote_delay_high=Q_(150, "ms"),      # scenario assumption: cloud round trip, worst case
    )
    VoiceAssistantBillionDevices = _st.WakeWordFleetScenario(
        name="Wake-word detection for one billion voice assistants",
        description="Streaming every microphone to cloud detection against detecting on device and "
                    "serving only post-wake queries.",
        devices=1e9,                          # scenario assumption
        audio_sample_rate=Q_(16, "kHz"),      # scenario assumption: speech audio
        audio_sample_bits=Q_(16, "bit"),      # scenario assumption
        queries_per_day=20.0,                 # scenario assumption: queries per device
        query_audio=Q_(5, "second"),          # scenario assumption: audio uploaded per query
        accelerator_time_per_query=Q_(200, "ms"),   # scenario assumption: GPU time per query
        accelerators_per_datacenter=10_000,   # scenario assumption
        waking_hours=Q_(16, "hour"),          # scenario assumption
        peak_multiplier=3.0,                  # scenario assumption: peak over waking average
        cloud_detection_cost=Q_(0.50, "USD/year"),  # scenario assumption: per device
        local_power_low=Q_(0.1, "mW"),        # scenario assumption: on-device detector
        local_power_high=Q_(1, "mW"),         # scenario assumption
        local_detection_cost=Q_(0.01, "USD/year"),  # scenario assumption: per device
    )
    PhoneCameraPipeline = _st.PipelineSpeedupScenario(
        name="A phone camera pipeline with a 10x faster ML stage",
        description="Image processing, ML scene classification, and tone mapping in series.",
        stages={
            "isp": Q_(100, "ms"),             # scenario assumption: ISP and auto-exposure
            "ml": Q_(60, "ms"),               # scenario assumption: ML scene classification
            "post": Q_(40, "ms"),             # scenario assumption: tone mapping and HDR merge
        },
        accelerated="ml",
        speedup=10.0,                         # scenario assumption: 10x faster model
    )

    # --- Physical-plant budgets of the warehouse mobile manipulator (added 2026-10-20) ---
    # The machine and its site record are Embodied.MobileManipulator.WarehouseMobileManipulator
    # and Embodied.Scenario.WarehouseAisle; the arm joint is the Panda-class stand-in
    # Actuators.PandaClassJoint.CSG_25_50_ILM70x18. Inputs neither record carries are
    # stated here.
    WarehouseBaseStopBodyTerms = _st.StoppingBudgetScenario(
        name="Loaded warehouse base stopping from its drive limit, body terms only",
        description="Braking, the two spatial allowances, and travel during the brake onset; "
                    "the delays the body does not own are left out.",
        site=_Embodied.Scenario.WarehouseAisle,
        speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
        delay=_Embodied.Scenario.WarehouseAisle.t_brake_onset,
        added_delay=Q_(100, "millisecond"),  # scenario assumption: a delay step priced at the drive limit
    )
    WarehouseBaseStopFullDelay = _st.StoppingBudgetScenario(
        name="Loaded warehouse base stopping from its drive limit after the full pre-brake delay",
        description="The site's full pre-brake delay (brake onset, lease, tick, bus cycle, observation age).",
        site=_Embodied.Scenario.WarehouseAisle,
        speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
        delay=_Embodied.Scenario.WarehouseAisle.tau_delay,
    )
    WarehouseBaseStopFastSetpoint = _st.StoppingBudgetScenario(
        name="Loaded warehouse base stopping from a setpoint above its drive limit",
        description="The full pre-brake delay at 2 m/s, a speed-scaling case above the 1.5 m/s drive limit.",
        site=_Embodied.Scenario.WarehouseAisle,
        speed=Q_(2.0, "m/s"),  # scenario assumption: a proposed speed setpoint above the drive limit
        delay=_Embodied.Scenario.WarehouseAisle.tau_delay,
    )
    WarehouseBaseStopInspectedFloor = _st.StoppingBudgetScenario(
        name="Loaded warehouse base stopping on the inspected low-traction floor",
        description="The full pre-brake delay with the deceleration capped by the inspected floor's "
                    "friction, all weight on the driven wheels.",
        site=_Embodied.Scenario.WarehouseAisle,
        speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
        delay=_Embodied.Scenario.WarehouseAisle.tau_delay,
        deceleration=_traction_limit(_Embodied.Scenario.WarehouseAisle.mu_inspected_floor),
    )
    WarehouseBaseSpeedStep = _st.SpeedStepScenario(
        name="Raising the warehouse base's speed setpoint from 1.5 to 2 m/s",
        description="Braking travel added by the faster setpoint, and the delay that costs the same distance.",
        base=WarehouseBaseStopBodyTerms,
        raised=WarehouseBaseStopFastSetpoint,
    )
    WarehouseBaseBrakingCeilings = _st.BrakingCeilingsScenario(
        name="Ceilings on the loaded warehouse base's credible deceleration",
        description="Traction (dry and inspected floors), tip-over (center of gravity unmeasured), and the "
                    "hub drives' torque at their current limit, with back-EMF headroom at the drive limit.",
        site=_Embodied.Scenario.WarehouseAisle,
    )
    WarehouseBaseHubDriveElectrical = _st.HubDriveElectricalScenario(
        name="Current rise and phase-short braking of the warehouse base's hub drives",
        description="Electrical time constant, the stall current ramp from a depleted pack, and the wheel "
                    "speed at which shorted windings brake hardest.",
        site=_Embodied.Scenario.WarehouseAisle,
    )
    WristCameraGraspAgeProfile = _st.FreshnessProfileScenario(
        name="Wrist-camera grasp of a mug on the pick-station conveyor",
        description="An illustrative capture-to-onset age profile (P50, P99, max per stage) against the "
                    "conveyor mug's freshness deadline; a measured profile of the integrated machine replaces it.",
        site=_Embodied.Scenario.WarehouseAisle,
        stages=[  # scenario assumption: illustrative stage profile, (P50, P99, max)
            ("Exposure and readout", Q_(11.2, "ms"), Q_(16.6, "ms"), Q_(16.7, "ms")),
            ("Transport and DMA", Q_(1.4, "ms"), Q_(3.2, "ms"), Q_(6.8, "ms")),
            ("Backbone and policy", Q_(18.5, "ms"), Q_(24.1, "ms"), Q_(38.4, "ms")),
            ("Scheduling and IPC", Q_(0.6, "ms"), Q_(2.1, "ms"), Q_(12.3, "ms")),
            ("Fieldbus", Q_(1.0, "ms"), Q_(1.0, "ms"), Q_(2.0, "ms")),
            ("Current rise", Q_(2.3, "ms"), Q_(3.0, "ms"), Q_(4.5, "ms")),
        ],
        focus_stage="Backbone and policy",
        system_p99=Q_(46.2, "ms"),   # scenario assumption: system-level profile
        system_p999=Q_(68.4, "ms"),  # scenario assumption: system-level tail
    )
    GearedRotorFlywheel = _st.GearedRotorScenario(
        name="A 50 g rotor through a 100:1 joint reduction",
        description="Reflected rotor inertia sized as a 10 cm solid flywheel, beside an illustrative link.",
        gear_ratio=100,  # scenario assumption: joint reduction
        rotor_mass=Q_(50.0, "gram"),  # scenario assumption
        rotor_inertia=Q_(5.6e-6, "kg * m**2"),  # scenario assumption: inertia of that rotor
        flywheel_radius=Q_(0.10, "meter"),  # scenario assumption: comparison flywheel radius
        load_inertia=Q_(8.5e-3, "kg * m**2"),  # scenario assumption: link inertia at the joint
    )
    PandaClassJointGearRatios = _st.GearRatioCostScenario(
        name="The Panda-class joint holding 87 N m at three gear ratios",
        description="TQ ILM 70x18 motor holding the Panda's maximum joint torque at the CSG-25-50 ratio, "
                    "a catalog 100:1, and a quasi-direct-drive-class 10:1.",
        joint=_Actuators.PandaClassJoint.CSG_25_50_ILM70x18,
        motor=_Actuators.FramelessMotor.TQ_ILM70x18,
        output_torque=_Embodied.Manipulator.Panda.max_torque,
        gear_efficiency=0.85,  # scenario assumption: the catalog's varies with load, speed, and temperature
        ratios=[int(_Actuators.PandaClassJoint.CSG_25_50_ILM70x18.gear_ratio),
                100,  # scenario assumption: a catalog CSG-25 ratio
                10],  # scenario assumption: quasi-direct-drive class ratio
    )
    PandaClassJointBurst = _st.ThermalBurstScenario(
        name="A threefold rated-current burst in the Panda-class joint",
        description="First-order lumped winding at fixed resistance, from ambient and from rated-current "
                    "equilibrium, with the average-power duty screen and its periodic peak.",
        joint=_Actuators.PandaClassJoint.CSG_25_50_ILM70x18,
        motor=_Actuators.FramelessMotor.TQ_ILM70x18,
        thermal_capacitance=Q_(120.0, "J/K"),  # scenario assumption: lumped; the data sheet publishes none
        thermal_resistance=Q_(2.5, "K/W"),  # scenario assumption: installed
        ambient=Q_(40.0, "degC"),  # scenario assumption: enclosed ambient
        limit=_CLASS_F,
        ceiling=Q_(140.0, "degC"),  # scenario assumption: working ceiling chosen below the Class F rating
        burst_factor=3,  # scenario assumption: below the motor's peak torque
        burst_on=Q_(3.0, "second"),  # scenario assumption
    )
    PandaClassElbowStall = _st.StallDeratingScenario(
        name="The Panda-class elbow holding 87 N m at stall",
        description="Steady winding temperature with copper's resistance feedback, and the hold the "
                    "installation can sustain at the Class F limit.",
        joint=_Actuators.PandaClassJoint.CSG_25_50_ILM70x18,
        motor=_Actuators.FramelessMotor.TQ_ILM70x18,
        hold_torque=_Embodied.Manipulator.Panda.max_torque,
        gear_efficiency=PandaClassJointGearRatios.gear_efficiency,
        thermal_resistance=Q_(1.80, "K/W"),  # scenario assumption: unventilated limb
        ambient=Q_(40.0, "degC"),  # scenario assumption: limb ambient
        limit=_CLASS_F,
    )
    SealedJointCatalogDerating = _st.ContinuousDeratingScenario(
        name="A catalog continuous rating carried into a sealed 40 degC limb",
        description="Continuous torque at the working ceiling for a 25 degC catalog test against the "
                    "enclosed ambient, and again with the installed thermal resistance doubled.",
        ceiling=PandaClassJointBurst.ceiling,
        catalog_ambient=Q_(25.0, "degC"),  # scenario assumption: catalog test ambient
        installed_ambient=PandaClassJointBurst.ambient,
        thermal_resistance_factor=2.0,  # scenario assumption: installed theta relative to the catalog heat sink
    )
    WarehouseDriveBusStop = _st.DriveBusScenario(
        name="The warehouse base's 48 V drive bus: coincident step and a stop from the drive limit",
        description="Sustained sag under the coincident inference and drive step, and the regenerative "
                    "surge of a stop into a full pack, absorbed by the bus capacitance or a chopper.",
        site=_Embodied.Scenario.WarehouseAisle,
        link_capacitance=Q_(22000.0, "microfarad"),  # scenario assumption: bulk link capacitance
        regen_fraction=0.75,  # scenario assumption: share of kinetic energy returned to the bus
        candidate_resistor=Q_(3.9, "ohm"),  # scenario assumption: chosen chopper resistor
        candidate_power=Q_(1.0, "kilowatt"),  # scenario assumption: its pulse rating
    )
    CleanLoadedStopCampaign = _st.ZeroFailureScenario(
        name="3,000 failure-free loaded stops",
        description="The per-stop failure rate a clean campaign supports at 95 percent one-sided confidence.",
        trials=3000,  # scenario assumption
        confidence=0.95,
        target_rate=1e-4,  # scenario assumption: per-stop rate the record would need
    )
    ThrottledNavigationBackbone = _st.ThrottledStageScenario(
        name="The navigation backbone at two-thirds clock under thermal throttling",
        description="A compute-bound vision backbone stretched by DVFS, paid as blind travel at the drive limit.",
        site=_Embodied.Scenario.WarehouseAisle,
        clock_ratio=2 / 3,  # scenario assumption: DVFS clock as a fraction of full clock
        speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
    )

    # --- Data engineering: an always-on keyword spotter's data, and the labels behind it (added 2026-10-20) ---
    # One smart-speaker wake-word detector carries the product targets, the
    # false-wake budget and the listening it takes to verify, and the data budget.
    # The labeling entries price labels against compute, agreement, and active
    # learning. Cited figures come from Literature, Datasets, and Infrastructure;
    # everything else is a stated scenario assumption.

    KeywordSpotterTargets = _st.AlwaysOnDetectorTargetsScenario(
        name="A smart-speaker keyword spotter's product targets",
        description="Catch 98 percent of spoken wake words, wake falsely less than once a month, answer within "
                    "200 ms, fit a 256 KB always-on model and 16 KB of preprocessing state, on a $150K data "
                    "budget in six months.",
        detection_target=0.98,            # scenario assumption: share of spoken wake words caught
        false_wake_budget=1,              # scenario assumption: expected false wakes per month
        false_wake_horizon=Q_(_DAYS_PER_MONTH, "day"),
        response_limit=Q_(200, "ms"),     # scenario assumption: end of keyword to response
        model_memory=Q_(256, "KB"),       # scenario assumption: always-on island sized for the INT8 DS-CNN
        preprocessing_memory=Q_(16, "KB"),  # scenario assumption: preprocessing-state budget
        data_budget=Q_(150_000, "USD"),   # scenario assumption: total data-engineering budget
        timeline=Q_(6, "month"),          # scenario assumption: time to production
        base_detection_rate=0.90,         # scenario assumption: illustrative base-model detection rate
    )
    KeywordSpotterSampleRateChoice = _st.DesignTradeoffScenario(
        name="Keyword spotter: 8 kHz to 16 kHz sampling",
        description="Illustrative detection-rate gain; raw-audio storage and processing scale with the rate (kHz).",
        baseline=8, alternative=16,       # scenario assumption: sampling rates compared, kHz
        gain_low=0.02, gain_high=0.04,    # scenario assumption: illustrative detection-rate gain
    )
    KeywordSpotterMfccChoice = _st.DesignTradeoffScenario(
        name="Keyword spotter: 13 to 40 MFCC coefficients per frame",
        description="Illustrative detection-rate gain; model input and feature memory scale with the count.",
        baseline=13, alternative=40,      # scenario assumption: coefficients per frame compared
        gain_low=0.03, gain_high=0.05,    # scenario assumption: illustrative detection-rate gain
    )
    KeywordSpotterExampleCountChoice = _st.DesignTradeoffScenario(
        name="Keyword spotter: 1M to 10M training examples",
        description="Illustrative detection-rate gain; labeling cost, storage, and training time scale with the count.",
        baseline=1_000_000, alternative=10_000_000,  # scenario assumption: training examples compared
        gain_low=0.05, gain_high=0.08,    # scenario assumption: illustrative detection-rate gain
    )
    KeywordSpotterFalseWakes = _st.FalseAlarmBudgetScenario(
        name="One-second windows checked every second, all month, against one false wake a month",
        description="An always-on detector scores one non-keyword window a second around the clock; the "
                    "false-wake budget sets the per-window false-positive rate it must hold.",
        window=Q_(1, "second"),           # scenario assumption: one window scored each second
        listening_per_day=Q_(24, "hour"),  # always on
        horizon=KeywordSpotterTargets.false_wake_horizon,
        budget=KeywordSpotterTargets.false_wake_budget,
        rejection_levels=[0.999, 0.9999, 0.99999],  # scenario assumption: per-window rejection rates compared
    )
    KeywordSpotterFalseWakeEvidence = _st.ZeroEventExposureScenario(
        name="Listening needed to support one false wake a month with none observed",
        description="Representative non-keyword audio, at 95 percent one-sided confidence (the zero-event rule "
                    "of three), recorded at 16 kHz and 16 bits.",
        budget=KeywordSpotterFalseWakes.budget,
        horizon=KeywordSpotterFalseWakes.listening_time,
        confidence=0.95,                  # scenario assumption: one-sided confidence of the claim
        sample_rate=Q_(16_000, "Hz"),     # scenario assumption: the 16 kHz candidate configuration
        sample_width=Q_(2, "byte"),       # 16-bit mono audio
    )
    KeywordSpotterDataBudget = _st.DataBudgetScenario(
        name="A $150K keyword-spotting data budget split across labeling, storage, and governance",
        description="Sixty percent buys crowdsourced recordings and labels at $0.10 each plus 20 percent review.",
        total=KeywordSpotterTargets.data_budget,
        shares={"labeling": 0.60,         # scenario assumption
                "storage and processing": 0.25,  # scenario assumption
                "governance": 0.15},      # scenario assumption
        label_price=Q_(0.10, "USD"),      # scenario assumption: per recorded and labeled example
        review_overhead=0.20,             # scenario assumption: quality-review surcharge
    )
    ImageNetLabelingVersusResNetRun = _st.LabelingVersusTrainingScenario(
        name="Labeling one million images against one ResNet-50 run on eight A100s",
        description="A $100K labeling budget against 90 epochs of ImageNet at the A100 ResNet-50 sustained rate, "
                    "priced at the reference and cloud GPU-hour rates.",
        labeling_budget=Q_(100_000, "USD"),  # scenario assumption: crowd labels plus review and relabeling
        dataset_examples=_Datasets.ImageNet.training_examples,
        epochs=90,                        # scenario assumption: the standard ResNet-50 schedule
        throughput_per_accelerator=Q_(float(_Literature.Benchmarks.ResNet50A100TrainThroughput), "count/second"),
        accelerators=8,                   # scenario assumption: one 8-GPU server
        price_low=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
        price_high=_Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
        labeling_time_low=Q_(2, "week"),  # scenario assumption: crowdsourced labeling of 1M images, low
        labeling_time_high=Q_(4, "week"),  # scenario assumption: and high
    )
    FullHdMaskVersusBoxes = _st.LabelGranularityScenario(
        name="A 1920x1080 segmentation mask against ten bounding boxes",
        description="One class per pixel against four coordinates per box.",
        image_width=_VIDEO_1080P_WIDTH,
        image_height=_VIDEO_1080P_HEIGHT,
        boxes=10,                         # scenario assumption: objects in one street scene
        coords_per_box=4,
    )
    CityscapesMasksForMillionFrames = _st.LaborEffortScenario(
        name="Pixel masks for one million street frames at the Cityscapes annotation time",
        description="More than 1.5 hours per fine annotation (Cordts et al. 2016), in full-time person-years.",
        items=1_000_000,                  # scenario assumption: frames a driving team wants masked
        time_per_item=Q_(float(_Literature.AnnotationTimes.CityscapesFineMaskHours), "hour"),
        work_year=Q_(2000, "hour"),       # convention: a full-time person-year of 2,000 working hours
    )
    MswcByHand = _st.LaborEffortScenario(
        name="The Multilingual Spoken Words Corpus labeled by hand at 10 s per clip",
        description="What forced alignment saved: every clip of the corpus labeled manually.",
        items=_Datasets.MSWC.training_examples.to("count").magnitude,
        time_per_item=Q_(10, "second"),   # scenario assumption: a fast manual label
        work_year=CityscapesMasksForMillionFrames.work_year,
    )
    BinaryLabelChanceAgreement = _st.AnnotatorAgreementScenario(
        name="Two annotators who each label 90 percent of items 'not spam'",
        description="Chance agreement from the label mix alone, and kappa at 85 percent raw agreement.",
        class_shares=[0.90, 0.10],        # scenario assumption: binary label mix
        observed_agreement=0.85,          # scenario assumption: raw agreement that looks high
    )
    WakeWordLabelQualityDrift = _st.AgreementDriftScenario(
        name="A wake-word labeling campaign whose agreement falls after new annotators join",
        description="Kappa falls while every input-feature monitor stays flat.",
        baseline_kappa=0.85,              # scenario assumption: initial collection
        later_kappa=0.72,                 # scenario assumption: six months later
        elapsed=Q_(6, "month"),           # scenario assumption
    )
    ActiveLearningTenMillionPool = _st.ActiveLearningBudgetScenario(
        name="Active learning over a 10M-image pool on a $50K budget",
        description="Hosted-model scoring at $0.01 an image and labels at $0.50, from 85 to 95 percent accuracy; "
                    "a tenth of the budget scores a narrowed candidate pool.",
        pool=10_000_000,                  # scenario assumption: unlabeled images
        label_budget=Q_(50_000, "USD"),   # scenario assumption
        label_price=Q_(0.50, "USD"),      # scenario assumption: a label that needs careful review
        scoring_price=Q_(0.01, "USD"),    # scenario assumption: hosted multimodal model per image
        random_labels_needed=1_000_000,   # scenario assumption: random sampling to the target
        active_labels_low=100_000,        # scenario assumption: hard examples needed, low
        active_labels_high=200_000,       # scenario assumption: hard examples needed, high
        compute_share=0.10,               # scenario assumption: budget share spent on scoring
        baseline_accuracy=0.85,           # scenario assumption
        target_accuracy=0.95,             # scenario assumption
    )
    SpeechCommandsHashSplit = _st.HashSplitScenario(
        name="Speech Commands v0.02 split by hashed speaker key",
        description="TensorFlow's which_set() default percentages against the split sizes the dataset card records.",
        dataset=_Datasets.SpeechCommandsV2,
        validation_percent=10,            # TensorFlow train.py --validation_percentage default
        test_percent=10,                  # TensorFlow train.py --testing_percentage default
    )
    MillionEventBatchVersusStreaming = _st.BatchVersusStreamingIngestScenario(
        name="One million 1 KB events a second, streamed or batched hourly",
        description="Always-on streaming cores with a standby each, against a batch job that clears each hour's "
                    "data in 10 minutes, at one core-hour price.",
        event_rate=Q_(1_000_000, "1/second"),  # scenario assumption
        event_size=Q_(1, "KB"),           # scenario assumption
        stream_cores=100,                 # scenario assumption: 50 primary cores plus a standby each
        stream_hours_per_day=Q_(24, "hour"),
        batch_cores=200,                  # scenario assumption
        batch_window=Q_(10, "minute"),    # scenario assumption: per clock hour
        core_hour_price=Q_(0.05, "USD/hour"),  # scenario assumption: per core-hour
        redundancy_factor=2,              # scenario assumption: one standby core per primary
    )
    PretrainingCorpusGrowth = _st.TrendFitScenario(
        name="Reported pre-training tokens of landmark language models, fitted growth per year",
        description="Least-squares fit of log tokens on release date (Literature.Series.PretrainingTokens).",
        series_name="PretrainingTokens", x_key="date", y_key="tokens", x_is_date=True,
    )
