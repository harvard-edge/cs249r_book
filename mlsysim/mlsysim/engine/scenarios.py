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
from ..literature.registry import Literature as _Literature
from ..systems.reliability import Reliability as _Reliability
from ..infrastructure.registry import Infrastructure as _Infrastructure
from ..ops import Ops as _Ops
from . import scenario_types as _st
# Deployment paradigms block (added 2026-10-20)
from ..systems.registry import Pods as _Pods
from ..core.units import (VIDEO_1080P_WIDTH as _VIDEO_1080P_WIDTH, VIDEO_1080P_HEIGHT as _VIDEO_1080P_HEIGHT,
                          VIDEO_4K_WIDTH as _VIDEO_4K_WIDTH, VIDEO_4K_HEIGHT as _VIDEO_4K_HEIGHT,
                          VIDEO_BYTES_PER_PIXEL_RGB as _VIDEO_BYTES_PER_PIXEL_RGB, VIDEO_FPS_STANDARD as _VIDEO_FPS_STANDARD)

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
