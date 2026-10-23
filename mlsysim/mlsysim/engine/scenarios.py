from pydantic import BaseModel, ConfigDict
from typing import Optional, Union
from statistics import NormalDist as _NormalDist  # ML operations A/B sizing (added 2026-10-20)
from ..core.units import Q_
from ..core.units import DAYS_PER_YEAR as _DAYS_PER_YEAR  # added 2026-10-20
from ..core.units import HOURS_PER_YEAR as _HOURS_PER_YEAR  # sustainability (added 2026-10-20)
from ..core.types import Quantity
from ..models.types import Workload
from ..models.types import TransformerWorkload as _TransformerWorkload  # fine-tuning (added 2026-10-20)
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
from ..core.units import BYTES_INT4 as _BYTES_INT4  # serving at scale (added 2026-10-20)
from ..core.units import BYTES_INT16 as _BYTES_INT16, BYTES_INT32 as _BYTES_INT32  # model compression (added 2026-10-20)
from ..core.units import BYTES_INT64 as _BYTES_INT64  # framework training step (added 2026-10-20)
from ..core.units import BYTES_TRAINING_STATE_MIXED_PRECISION as _BYTES_TRAINING_STATE
from ..literature.registry import Literature as _Literature
from ..systems.reliability import Reliability as _Reliability
from ..infrastructure.registry import Infrastructure as _Infrastructure
from ..ops import Ops as _Ops
from ..embodied.registry import Embodied as _Embodied  # embodied stopping scenarios (2026-10-20)
from ..agents.registry import Agents as _Agents  # agent harness profiles (added 2026-10-20)
from . import scenario_types as _st
from .serving_profile import resnet50_h100_profile as _resnet50_h100_profile  # quantization (added 2026-10-20)
from ..core.units import BYTES_ADAM_MOMENTS as _BYTES_ADAM_MOMENTS  # neural computation (added 2026-10-20)
from ..core.units import FLOAT_FORMAT_BITS as _FLOAT_FORMAT_BITS, INT16_BITS as _INT16_BITS  # added 2026-10-20
from ..core.units import COLOR_DEPTH_8BIT as _COLOR_DEPTH_8BIT  # added 2026-10-20
from ..core.units import DAYS_PER_MONTH as _DAYS_PER_MONTH  # added 2026-10-20
# iPhone XS GPU drift on a warming phone (Lee et al. 2019, Figure 8; added 2026-10-20)
from ..core.units import BITS_PER_BYTE as _BITS_PER_BYTE  # security and privacy (added 2026-10-20)
from ..physics.statistics import calc_ks_coefficient as _calc_ks_coefficient  # robust AI (added 2026-10-20)
# Deployment paradigms block (added 2026-10-20)
from ..systems.registry import Pods as _Pods
from ..systems.registry import Systems as _Systems  # own against rent (added 2026-10-20)
from ..core.units import (VIDEO_1080P_WIDTH as _VIDEO_1080P_WIDTH, VIDEO_1080P_HEIGHT as _VIDEO_1080P_HEIGHT,
                          VIDEO_4K_WIDTH as _VIDEO_4K_WIDTH, VIDEO_4K_HEIGHT as _VIDEO_4K_HEIGHT,
                          VIDEO_BYTES_PER_PIXEL_RGB as _VIDEO_BYTES_PER_PIXEL_RGB, VIDEO_FPS_STANDARD as _VIDEO_FPS_STANDARD)
# Physical-plant budgets (added 2026-10-20)
from ..embodied.registry import Embodied as _Embodied
from ..actuators.registry import Actuators as _Actuators
from ..physics.constants import INSULATION_CLASS_F_LIMIT as _CLASS_F
from ..physics.drivetrain import calc_traction_deceleration_limit as _traction_limit
from ..physics.memory import calc_checkpoint_size as _calc_checkpoint_size  # fleet orchestration (2026-10-20)
from ..physics.memory import calc_low_rank_update_params as _calc_low_rank_update_params  # model training (2026-10-20)
# Physical demonstration data (added 2026-10-20)
from ..sensors.registry import Sensors as _Sensors
from ..physics.constants import AGE_OF_UNIVERSE as _AGE_OF_UNIVERSE
from ..physics.constants import ALUMINUM_THERMAL_EXPANSION_COEFFICIENT as _ALUMINUM_ALPHA  # sensor perception (2026-10-20)
from ..physics.constants import MOR_CONTRAST_THRESHOLD as _MOR_CONTRAST  # sensor perception (2026-10-20)

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

from ..physics.layers import calc_resnet_bottleneck_macs as _resnet_macs  # added 2026-10-21
_RESNET50_MACS = _resnet_macs()


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

    Llama2_70B_DecodeH100 = Llama3_70B_DecodeH100.variant(
        name="Llama-2 70B batch-1 decode on one H100",
        description="The batch-1 decode scenario for the Llama-2 70B serving replica (FP16 weights).",
        workload=_Models.Language.Llama2_70B,
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



    NodeAvailability1K = _st.AvailabilityScenario(
        name="1,000 nodes at 99.9 percent", description="Illustrative per-node availability, independent failures.",
        availability=0.999, n_components=1000)

    NodeAvailability10K = NodeAvailability1K.variant(
        name="10,000 nodes at 99.9 percent", n_components=10_000)

    FourNinesAvailability10K = NodeAvailability1K.variant(
        name="10,000 nodes at 99.99 percent", availability=0.9999, n_components=10_000)




    CoordinationOverheadExample = _st.ScalingOverheadScenario(
        name="A planned 3x speedup with 40 percent overhead",
        description="Communication and coordination take 40 percent of the scaled step.",
        planned_speedup=3.0, overhead_fraction=0.40)





    Gpt3PublishedRun = _st.PublishedRunScenario(
        name="GPT-3 175B as trained",
        description="GPT-3's training compute over its reference V100 count and days (Patterson et al. 2021).",
        workload=_Models.Language.GPT3,
        accelerator=_Hardware.Cloud.V100,
    )

    # --- Network fabrics (added 2026-10-20) ---------------------------------



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


    OverlappedInputStall = _st.DataStallScenario(
        name="250 ms delivery against 200 ms compute",
        description="Storage delivers each batch in 250 ms; compute takes one 200 ms step.",
        compute_time=_ReferenceStats.StorageTrainingCorpus.StepTime,
        io_time=Q_(250, "ms"),  # scenario assumption: delivery time per batch
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
    # same budget (published record), a floor-friction drop, and
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
        speed=Q_(43.2, "mph"),  # NTSB, Collision Between Vehicle Controlled by Developmental Automat...
        speed_at_delay_end=Q_(40.5, "mph"),  # NTSB, Collision Between Vehicle Controlled by Developmental Automat...
        delay=Q_(1.0, "s"),  # NTSB, Collision Between Vehicle Controlled by Developmental Automat...
        deceleration=(0.92 * Q_(1, "standard_gravity")).to("m/s**2"),  # NTSB, Collision Between Vehicle Controlled by Developmental Automat...
    )



    HydraulicFluidSlickTraction = _st.TractionLossScenario(
        name="Hydraulic fluid on a dry concrete floor under the warehouse base",
        description="The warehouse aisle's illustrative floor friction, dry (0.60) and under a "
                    "clear oil film (0.05), against the running machine's credible deceleration "
                    "with all weight on the driven wheels.",
        mu_before=_Embodied.Scenario.WarehouseAisle.mu_dry,  # the aisle's dry floor (illustrative)
        mu_after=_Embodied.Scenario.WarehouseAisle.mu_oil_film,  # the aisle's oil film (illustrative)
        deceleration=_Embodied.Scenario.WarehouseAisle.a_brake,
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


    ResidualOnePercent512 = _st.ResidualShareScenario(
        name="A 1 percent residual on 512 devices",
        description="A residual equal to 1 percent of the one-device step fills most of the distributed step.",
        residual=0.01,  # scenario assumption: residual fraction of the one-device step
        devices=512)  # scenario assumption


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



    ModelSpeedupRequestPath = _st.AmdahlScenario(
        name="A request path whose model gets 3x faster",
        description="60 ms preprocessing, 45 ms model, 25 ms postprocessing; only the model speeds up.",
        stages={"preprocess": Q_(60, "ms"), "model": Q_(45, "ms"), "postprocess": Q_(25, "ms")},  # scenario assumption
        stage="model",
        speedup=3.0,            # scenario assumption: better kernels and quantization
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
    TPUv4Pod4096 = _st.AcceleratorPodScenario(
        name="A TPU v4 pod of 4,096 chips",
        description="Aggregate BF16 peak and HBM of the pod envelope's chips.",
        pod=_Pods.TPUv4_4096,
        accelerator=_Hardware.Cloud.TPUv4,
    )
    IllustrativeMemoryWall = _st.GrowthGapScenario(
        name="Compute against memory bandwidth, illustrative rates",
        description="Compute doubling every 18 months against bandwidth growing 20 percent a year, "
                    "over a decade.",
        compute_doubling_period=Q_(18, "month"),  # scenario assumption: illustrative rate
        bandwidth_growth=0.20,                    # scenario assumption: illustrative rate
        horizon=Q_(10, "year"),
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

    # --- Fault tolerance: failure clocks, checkpoints, recovery, serving (added 2026-10-20) ---
    GpuFailureLadder = _st.FailureIntervalSweepScenario(
        name="GPU-only failure interval from 8 to 25,000 GPUs",
        description="Independent GPU failures at the per-GPU MTTF; the job's interval is MTTF / N.",
        component_mtbf=Q_(_Reliability.Gpu.mttf_hours, "hour"),
        sizes=[8, 64, 512, 1000, 4000, 10_000, 25_000],  # scenario assumption: illustrative job sizes
    )

    FullSystemFailures10K = _st.SeriesFailureScenario(
        name="Full-system failure interval of the 10,000-GPU cluster",
        description="GPU, PCIe link, and each node's network path (NIC and cables) in series; the "
                    "DGX H100's 4+2 redundant power supplies add no single-failure term.",
        n_devices=_Clusters.Training_10K.total_accelerators,
        devices_per_node=_Nodes.DGX_H100.accelerators_per_node,
        component_mttf={
            "gpu": Q_(_Reliability.Gpu.mttf_hours, "hour"),
            "pcie": Q_(_Reliability.PcieSwitch.mttf_hours, "hour"),
            "network": Q_(_Reliability.Nic.mttf_hours, "hour"),
            "power": Q_(_Reliability.Psu.mttf_hours, "hour"),
        },
        per_node={
            "gpu": _Nodes.DGX_H100.accelerators_per_node,
            "pcie": _Nodes.DGX_H100.accelerators_per_node,  # one PCIe link to the host per GPU
            "network": 1,  # scenario assumption: the node's network path (NIC and cables) as one unit
            "power": _Nodes.DGX_H100.series_psus,  # 0: no single supply failure stops the node
        },
    )


    Gpt3CheckpointInterval10K = _st.CheckpointIntervalScenario(
        name="Young-Daly interval of the 175B job on the 10,000-GPU cluster",
        description="A 14-byte-per-parameter resumable checkpoint written at the per-job checkpoint "
                    "bandwidth, against the cluster's full-system MTBF.",
        workload=_Models.Language.GPT3,
        bytes_per_param=_ReferenceStats.MixedPrecisionAdam.CheckpointBytesPerParameter,
        write_bandwidth=Q_(float(_Reliability.Recovery.checkpoint_write_bw_gbs), "GB/s"),
        # scenario assumption: the worked example carries the MTBF at the two decimals it displays (3.87 h)
        mtbf=Q_(round(FullSystemFailures10K.result.mtbf.m_as("hour"), 2), "hour"),
    )


    FiveMinuteCheckpoint1K = _st.CheckpointIntervalScenario(
        name="A five-minute checkpoint on 1,000 GPUs",
        description="A stated write time against the GPU-only MTBF of 1,000 GPUs.",
        write_time=Q_(5, "minute"),  # scenario assumption
        mtbf=GpuFailureLadder.intervals[1000],
    )

    FifteenMinuteCheckpoint1K = FiveMinuteCheckpoint1K.variant(
        name="A fifteen-minute checkpoint on 1,000 GPUs",
        write_time=Q_(15, "minute"),  # scenario assumption: slower storage or a larger model
    )

    Gpt3RecoveryBudget1K = _st.RecoveryBudgetScenario(
        name="Cold-restart budget of the 175B job on 1,024 GPUs",
        description="Detection, restart, checkpoint load, and warmup per failure, paid at the "
                    "GPU-only failure rate.",
        workload=_Models.Language.GPT3,
        bytes_per_param=_ReferenceStats.MixedPrecisionAdam.CheckpointBytesPerParameter,
        recovery=_Reliability.Recovery,
        component_mtbf=Q_(_Reliability.Gpu.mttf_hours, "hour"),
        n_devices=_Clusters.Training_1K.total_accelerators,
    )

    Gpt3RecoveryBudget10K = Gpt3RecoveryBudget1K.variant(
        name="Cold-restart budget of the 175B job on 10,000 GPUs",
        n_devices=_Clusters.Training_10K.total_accelerators,
    )

    HotSpareNode10K = _st.HotSpareScenario(
        name="One idle DGX H100 node beside the 10,000-GPU job",
        description="A hot spare priced against one minute removed from every full-system recovery.",
        failures=FullSystemFailures10K,
        spare_devices=_Nodes.DGX_H100.accelerators_per_node,
        device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
    )

    HalfSpeedStraggler1K = _st.StragglerCostScenario(
        name="One of 1,024 GPUs thermally throttled to half speed",
        description="A compute-bound synchronous job with a 1 s step; the barrier waits for the slow rank.",
        n_devices=_Clusters.Training_1K.total_accelerators,
        step_time=Q_(1, "second"),  # scenario assumption
        slow_speed=0.5,  # scenario assumption: thermal throttling to half speed
        device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
    )

    ServingReplica99 = _st.AvailabilityScenario(
        name="One serving replica at 99 percent",
        description="Independent replicas; the service is up while any one is up.",
        availability=0.99,  # scenario assumption: per-replica availability
        n_components=1,
        replicas=1,
    )
    ServingReplicas99x2 = ServingReplica99.variant(name="Two serving replicas at 99 percent", replicas=2)
    ServingReplicas99x3 = ServingReplica99.variant(name="Three serving replicas at 99 percent", replicas=3)

    GpuAvailability10K = FourNinesAvailability10K.variant(
        name="10,000 GPUs at 99.99 percent",
        description="Illustrative per-GPU availability; a synchronous job needs every GPU up.",
        n_components=_Clusters.Training_10K.total_accelerators,
    )

    InferenceReadinessProbe = _st.HealthProbeScenario(
        name="A readiness probe on an inference replica",
        description="Probe every 5 s with a 2 s timeout; three misses remove the replica, two "
                    "clean probes readmit it.",
        interval=Q_(5, "second"),  # scenario assumption: periodSeconds
        timeout=Q_(2, "second"),  # scenario assumption: timeoutSeconds
        failure_threshold=3,  # scenario assumption
        success_threshold=2,  # scenario assumption
    )

    SdcExposure10K = _st.SdcExposureScenario(
        name="Silent-corruption exposure of the 10,000-GPU cluster",
        description="The generic per-GPU-hour SDC rate across every GPU of a 2 s synchronous step, "
                    "over a three-month run.",
        n_devices=_Clusters.Training_10K.total_accelerators,
        event_rate=_Reliability.SdcRatePerGpuHr,
        step_time=Q_(2, "second"),  # scenario assumption: illustrative step time
        run_duration=Q_(90, "day"),  # scenario assumption: a three-month run
    )

    SdcExposure100K = SdcExposure10K.variant(
        name="Silent-corruption exposure of a 100,000-GPU fleet",
        n_devices=10 * _Clusters.Training_10K.total_accelerators,  # scenario assumption: ten times the cluster
        run_duration=None,
    )

    # --- Fleet orchestration: queueing, packing, interruption, and sharing (added 2026-10-20) ---
    Mg1MixedJobQueueAt80 = _st.QueueWaitScenario(
        name="A fleet at 80 percent load with widely mixed job durations (M/G/1)",
        description="One-server queue model; job durations spread three standard deviations per mean, "
                    "as when minute-long debugging jobs share a fleet with week-long training runs.",
        utilization=0.8,  # scenario assumption: offered load
        service_cv=3,  # scenario assumption: representative mixed ML queue
    )
    Mg1ExponentialJobQueueAt80 = Mg1MixedJobQueueAt80.variant(
        name="A fleet at 80 percent load with exponential job durations (M/G/1)",
        description="The reference case of the one-server queue model: exponential service, C_s = 1.",
        service_cv=1,
    )
    DrfTwoTenantExample = _st.DominantShareScenario(
        name="Dominant-resource fairness for two tenants on a small cluster",
        description="Tenant A's one-GPU replica and tenant B's two-GPU worker, filled progressively "
                    "on 8 GPUs, 32 cores, 256 GB, and 80 Gb/s.",
        # scenario assumption: a deliberately small teaching cluster and two bundles
        capacity={"accelerators": 8, "cpu_cores": 32, "memory_gb": 256, "network_gbps": 80},
        bundles={
            "A": {"accelerators": 1, "cpu_cores": 4, "memory_gb": 32, "network_gbps": 10},
            "B": {"accelerators": 2, "cpu_cores": 2, "memory_gb": 16, "network_gbps": 20},
        },
    )
    SixGpuJobsOnDgxH100Nodes = _st.NodePackingScenario(
        name="Six-GPU jobs on 64 eight-GPU nodes",
        description="Each node fits one six-GPU job and strands two GPUs.",
        nodes=64,  # scenario assumption
        gpus_per_node=_Nodes.DGX_H100.accelerators_per_node,
        job_gpus=6,  # scenario assumption
    )
    FragmentedEightNodeSnapshot = _st.NodePackingScenario(
        name="A fragmented eight-node snapshot with 19 free GPUs",
        description="Free GPUs scattered so that every node keeps a busy slot; an eight-GPU job cannot start.",
        nodes=8,  # scenario assumption: illustrative layout
        gpus_per_node=_Nodes.DGX_H100.accelerators_per_node,
        job_gpus=_Nodes.DGX_H100.accelerators_per_node,
        free_slots=19,  # scenario assumption: illustrative layout
    )
    ReservedGpusIdleThreeWeeks = _st.IdleCapacityCostScenario(
        name="500 reserved GPUs held idle for three weeks",
        description="A team's reservation waits on data; the guarantee is not lent.",
        gpus=500,  # scenario assumption
        duration=Q_(3, "week"),  # scenario assumption
        device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
    )
    PartialGangHold = _st.IdleCapacityCostScenario(
        name="A pretraining gang holding 900 of its GPUs for an hour",
        description="The reference pretraining gang waits for its last members while the GPUs it holds idle.",
        gpus=900,  # scenario assumption: GPUs already held while the gang assembles
        duration=Q_(1, "hour"),
        device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
        gang_gpus=DenseLLMPretrainingReference.system.total_accelerators,
    )
    Training10KIdleHour = _st.IdleCapacityCostScenario(
        name="The 10,000-GPU training cluster idle for an hour",
        description="What an hour of a cluster-wide hold-and-wait deadlock costs.",
        gpus=_Clusters.Training_10K.total_accelerators,
        duration=Q_(1, "hour"),
        device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
    )
    MixedFleetPolicyFix1K = _st.UtilizationGainScenario(
        name="A three-pool 1,024-GPU cluster raised from 60 to 84 percent utilization",
        description="Capability matching, memory profiling, and backfill on a fleet of A100-80 GB, "
                    "A100-40 GB, and V100-32 GB pools bought in three rounds.",
        fleet_gpus=_Clusters.Training_1K.total_accelerators,
        utilization_before=0.60,  # scenario assumption: observed pre-fix utilization
        utilization_after=0.84,  # scenario assumption: observed post-fix utilization
        device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
        # scenario assumption: pool inventory and per-pool measurements before and after
        pools={"A100-80GB": 400, "A100-40GB": 424, "V100-32GB": 200},
        pool_metrics={
            "A100-80GB": {"allocated_before": 0.95, "active_before": 0.72,
                          "allocated_after": 0.89, "active_after": 0.94},
            "A100-40GB": {"allocated_before": 0.64, "allocated_after": 0.88},
            "V100-32GB": {"allocated_before": 0.0, "allocated_after": 0.71},
        },
        pending_jobs=50,  # scenario assumption: queue depth before the fix
        job_memory_need=Q_(35, "GB"),  # scenario assumption: peak memory of the template-bound jobs
    )
    ForcedPreemption175BFineTune64 = _st.PreemptionTaxScenario(
        name="Forced preemptions of a 64-GPU fine-tuning run of the 175B model",
        description="A low-priority run with no preemption budget, stopped without a clean save.",
        n_devices=_Clusters.Lab_64_H100.total_accelerators,
        checkpoint_bytes=_calc_checkpoint_size(_Models.Language.GPT3.parameters),  # 14 B/param resumable state
        lost_work=_ReferenceStats.ForcedPreemptionProfile.LostWork,
        preemption_rate=_ReferenceStats.ForcedPreemptionProfile.UnbudgetedRate,
        recovery=_Reliability.Recovery,
        device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
    )
    SpotInterruptionEffectiveCost = _st.SpotEffectiveCostScenario(
        name="Spot capacity priced per hour of retained progress",
        description="A 65 percent nominal discount, with a graceful save to slow storage and a restart "
                    "after every two hours of retained progress.",
        on_demand_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
        spot_rate=_Infrastructure.Pricing.Fleet.SpotGpuHourRef.rate,
        checkpoint_overhead=_ReferenceStats.SpotInterruptionProfile.CheckpointOverhead,
        restart_overhead=_ReferenceStats.SpotInterruptionProfile.RestartOverhead,
        retained_interval=_ReferenceStats.SpotInterruptionProfile.RetainedInterval,
    )
    Llama2_70BSpotRun512 = _st.SpotRunScenario(
        name="A 14-day 70B training run on 512 H100s, on demand against spot",
        description="Spot adds checkpoint overhead and a restart per interruption.",
        fleet=_Clusters.Training_512_H100,
        workload=_Models.Language.Llama2_70B,
        run_duration=Q_(14, "day"),  # scenario assumption: 14-day run
        on_demand_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
        spot_rate=_Infrastructure.Pricing.Fleet.SpotGpuHourRef.rate,
        checkpoint_overhead_fraction=0.05,  # scenario assumption
        interruptions_per_day=14,  # scenario assumption: fixed budget per day of useful training
        restart_per_interruption=Q_(0.5, "hour"),  # scenario assumption
    )
    ElasticResizeDecision = _st.ElasticResizeScenario(
        name="Start small, wait for 512 GPUs, or step up to 256 over 24 hours",
        description="One job's three schedules over a 24-hour window (ReferenceStats.ElasticResizeProfile).",
        window=_ReferenceStats.ElasticResizeProfile.Window,
        queue_wait=_ReferenceStats.ElasticResizeProfile.QueueWait,
        rescale_overhead=_ReferenceStats.ElasticResizeProfile.RescaleOverhead,
        small_gpus=int(_ReferenceStats.ElasticResizeProfile.SmallGpus),
        step_up_gpus=int(_ReferenceStats.ElasticResizeProfile.StepUpGpus),
        full_gpus=int(_ReferenceStats.ElasticResizeProfile.FullGpus),
        small_throughput=float(_ReferenceStats.ElasticResizeProfile.SmallThroughput),
        step_up_throughput=float(_ReferenceStats.ElasticResizeProfile.StepUpThroughput),
        full_throughput=float(_ReferenceStats.ElasticResizeProfile.FullThroughput),
    )
    ServingConsolidationA100 = _st.ServingConsolidationScenario(
        name="100 7B and 10 175B serving models on A100s, exclusive against shared pairs",
        description="Small models share a device in pairs when load tests allow; each large model "
                    "takes an 8-GPU tensor-parallel group either way.",
        # branded decimal budget: the model footprints in this scenario are decimal GB
        device_memory=_Hardware.Cloud.A100.memory.capacity.to("GiB").magnitude * Q_(1, "GB"),
        partition_memory=next(p for p in _Hardware.Cloud.A100.mig_profiles
                              if p.name == "3g.40gb").gpu_memory.to("GiB").magnitude * Q_(1, "GB"),
        small_model=_Models.Language.Llama2_7B,
        small_footprint=Q_(26, "GB"),  # scenario assumption: weights + KV cache + activations
        tenants=2,  # scenario assumption: models per shared GPU
        n_small=100,  # scenario assumption
        large_model=_Models.Language.GPT3,
        n_large=10,  # scenario assumption
        gpus_per_large=8,  # scenario assumption: one 8-GPU tensor-parallel group per replica
        bytes_per_param=_BYTES_FP16,
        device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
    )
    DiurnalServingLending1K = _st.DiurnalLendingScenario(
        name="A 1,024-GPU fleet lending its serving reservation to training by the day",
        description="Serving reserves its forecast peak plus a surge buffer; lending returns the rest to training.",
        fleet_gpus=_Clusters.Training_1K.total_accelerators,
        serving_reservation=400,  # scenario assumption: static serving reservation
        serving_average=150,  # scenario assumption: average serving demand
        surge_buffer=117,  # scenario assumption: ready headroom above the forecast
        training_base=618,  # scenario assumption: training pool after reservation and headroom
        device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
    )
    TensorParallelExchange1GB = _st.TransferScenario(
        name="A 1 GB tensor-parallel exchange inside a node and across nodes",
        description="One activation exchange over H100 NVLink (one direction) and one NDR InfiniBand port.",
        payload=Q_(1, "GB"),  # scenario assumption: activation size
        links={
            "NVLink": _Hardware.Cloud.H100.nvlink.bandwidth_per_direction,
            "InfiniBand NDR": _Fabrics.InfiniBand_NDR.bandwidth,
        },
    )
    TeamQuotaLending500 = _st.QuotaLendingScenario(
        name="A 500-GPU guarantee used at 60 percent, its idle 200 lent to a sibling team",
        description="Hierarchical quota: the owner keeps the guarantee; the borrower runs preemptible work on the rest.",
        guarantee=500,  # scenario assumption
        used_share=0.60,  # scenario assumption
        borrower_guarantee=300,  # scenario assumption
    )
    Gpt3SelfContainedReplicaDgxH100 = _st.SelfContainedReplicaScenario(
        name="The smallest self-contained 175B training replica on DGX H100 nodes",
        description="TP across one node, pipeline depth the smallest power of two that fits "
                    "mixed-precision Adam state in HBM without sharding across replicas.",
        workload=_Models.Language.GPT3,
        fleet=DenseLLMPretrainingReference.system,
        bytes_per_param=_BYTES_TRAINING_STATE,
    )
    # --- ML lifecycle: a diabetic retinopathy screener from request to monitoring (added 2026-10-20) ---
    # One rural-clinic screener carries every stage: its release floors and device
    # budget, a memory limit found after the work was done, two iteration speeds,
    # the clinic uplink, cloud against edge cost, and the monitoring clocks. The
    # cited studies (Gulshan 2016, Beede 2020, FDA DEN180001) are read from
    # Literature.ScreeningStudies; everything else is a stated scenario assumption.

    RetinopathyScreenerRequirements = _st.ScreeningRequirementsScenario(
        name="A diabetic retinopathy screener's release floors and clinic-device budget",
        description="Sensitivity and specificity floors that hold in every subgroup, a memory reservation on "
                    "an Orin Nano-class clinic device, and two measured sensitivity slices at the release gate.",
        sensitivity_floor=0.90,           # scenario assumption: few sick patients missed
        specificity_floor=0.80,           # scenario assumption: specialists not flooded with false referrals
        memory_budget=Q_(500, "MB"),      # scenario assumption: the clinic IT plan's reservation for the model
        device=_Hardware.Edge.JetsonOrinNano,
        slice_sensitivity={
            "all clinic photos": 0.94,    # scenario assumption: overall sensitivity on clinic photos
            "cataract patients": 0.76,    # scenario assumption: the subgroup slice that fails the floor
        },
        field_operating_point=(0.88, 0.80),  # scenario assumption: illustrative (specificity, sensitivity)
                                             # of the lab threshold on clinic photos
    )
    RetinopathyScreenerLateMemoryLimit = _st.LateConstraintScenario(
        name="A screener whose 500 MB memory limit surfaces on day 151",
        description="A team runs the stages as a line; the model reaches its accuracy target, then needs 4 GB "
                    "on a device that reserves 500 MB for it. Four dependent artifacts are redone in sequence.",
        model_memory=Q_(4, "GB"),         # scenario assumption: memory the trained model needs
        memory_budget=RetinopathyScreenerRequirements.memory_budget,
        device_memory_low=Q_(4, "GB"),    # scenario assumption: shared memory of the clinic device, low
        device_memory_high=Q_(8, "GB"),   # scenario assumption: shared memory of the clinic device, high
        request_day=1,
        validation_sensitivity={90: 0.92, 120: 0.94},   # scenario assumption: day -> validation sensitivity
        handoff_day=150,                  # scenario assumption: model handed to deployment engineers
        discovery_day=151,                # scenario assumption: memory need reported
        check_day=152,                    # scenario assumption: device budget finally checked
        discard_day=153,                  # scenario assumption: work discarded
        repair_tasks={                    # scenario assumption: a supplied schedule, not a multiplier
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
        window=Q_(26, "week"),            # scenario assumption: six months of development
        slow_cycle=Q_(1, "week"),         # scenario assumption: one large-ensemble run
        slow_start=95.0,                  # scenario assumption: percent
        slow_gain=0.15,                   # scenario assumption: points per run
        fast_cycle=Q_(1, "hour"),         # scenario assumption: one small-model run
        fast_start=90.0,                  # scenario assumption: percent
        fast_gain=0.1,                    # scenario assumption: points per run
        fast_effective_runs=100,          # scenario assumption: experiments a team can design and learn from
        ceiling=99.0,                     # scenario assumption: accuracy ceiling, percent
    )
    RuralClinicRetinalUpload = _st.DailyUploadScenario(
        name="A rural screening clinic uploading a day of retinal photos over a 2 Mb/s uplink",
        description="100 patients, four photos each, against an eight-hour shift, and the 10 KB summaries "
                    "per patient that local inference sends instead.",
        cases_per_day=100,                # scenario assumption: a busy rural screening clinic
        files_per_case=4,                 # screening protocol: two fields per eye
        file_size=_ReferenceStats.ClinicalImaging.RetinalPhotoSize,
        uplink=Q_(2.0, "megabit/second"), # scenario assumption: clinic uplink
        window=Q_(8.0, "hour"),           # scenario assumption: one clinic shift
        summary_size=Q_(10.0, "KB"),      # scenario assumption: one result summary per patient
    )
    RetinopathyScreeningCloudVersusEdge = _st.CloudEdgeCostScenario(
        name="500 screening clinics served from the cloud or from a device in each clinic",
        description="50 patients a day per clinic and one billed image each, every day of the year; cloud per-image "
                    "fees plus connectivity against a $500 device per clinic plus maintenance.",
        sites=500,                        # scenario assumption: clinics in the network
        cases_per_site_day=50,            # scenario assumption: patients per clinic per day
        images_per_case=1,                # scenario assumption: one graded image billed per patient
        days_per_year=_DAYS_PER_YEAR,     # scenario assumption: clinics operate every day
        cloud_cost_per_image=Q_(0.01, "dollar"),             # scenario assumption
        cloud_fixed_annual=Q_(45000, "dollar/year"),         # scenario assumption: connectivity and network operations
        edge_device_cost=Q_(500, "dollar"),                  # scenario assumption: one device per clinic
        edge_maintenance_annual=Q_(25000, "dollar/year"),    # scenario assumption
        edge_cost_per_image=Q_(0.001, "dollar"),             # scenario assumption
        image_size=_ReferenceStats.ClinicalImaging.RetinalPhotoSize,
        uplink=RuralClinicRetinalUpload.uplink,
        cloud_latency=Q_(200, "ms"),      # scenario assumption: remote processing after upload
        edge_latency=Q_(50, "ms"),        # scenario assumption: on-device inference
    )

    # --- Test-time deliberation (added 2026-10-20) ---------------------------
    # One repair decision spent several ways with the weights fixed. Token
    # counts, pass rates, check error rates, and prices are teaching
    # assumptions; times use the Llama-3 70B 8-bit decode floor on one H100.
    ProbeVsGeneratedExplanation = _st.ProbeVsAnswerScenario(
        name="A 1,024-token explanation against a prepared 40 ms test",
        description="Generation at the 8-bit decode floor and a hypothetical $15 per million "
                    "output tokens, against a test that observes the running program.",
        floors=Llama3_70B_Int8DecodeH100,
        answer_tokens=1024,  # scenario assumption: explanation length, output tokens
        probe_time=Q_(40, "ms"),  # scenario assumption: an already-prepared diagnostic test
        output_price=Q_(15.00, "USD / Mtoken"),  # scenario assumption: hypothetical output price
    )

    OutputCeilingThreeAllocations = _st.AllocationAxesScenario(
        name="One 4,096-token output ceiling as depth, breadth, or feedback",
        description="A 4,096-token prompt; depth spends the ceiling in one chain; breadth runs four "
                    "concurrent 1,024-token candidates, each billed its own prompt; feedback runs "
                    "three 1,024-token rounds, each followed by a 1.5 s check returning 340 "
                    "tokens; $3 and $15 per million input and output tokens, no cache discount.",
        floors=Llama3_70B_Int8DecodeH100,
        prompt=4096,  # scenario assumption: serialized initial input, tokens
        ceiling=4096,  # scenario assumption: common generated-token ceiling
        candidates=4,  # scenario assumption: breadth concurrency and candidate count
        candidate_tokens=1024,  # scenario assumption: generated tokens per short candidate
        rounds=3,  # scenario assumption: feedback rounds used
        report_tokens=340,  # scenario assumption: tokens appended per check report
        check_time=Q_(1.5, "second"),  # scenario assumption: per feedback check
        input_price=Q_(3.00, "USD / Mtoken"),  # scenario assumption: hypothetical input price
        output_price=ProbeVsGeneratedExplanation.output_price,
        context_growth=8,  # scenario assumption: repository context grows eightfold
    )

    IndependentCandidatePools = _st.BestOfNPoolScenario(
        name="Pools of 1, 2, 4, and 8 independent candidates at a 30 percent pass rate",
        description="IID candidates for one task, a selector that finds any passing member; and a "
                    "mixed workload, half its tasks at 60 percent and half never passing.",
        pass_rate=0.30,  # scenario assumption: per-candidate pass probability for a fixed task
        pool_sizes=[1, 2, 4, 8],  # scenario assumption: pool sizes compared
        easy_pass_rate=0.60,  # scenario assumption: the easy half of a mixed workload
        hard_pass_rate=0.0,  # scenario assumption: the hard half never passes
    )



    SparseValidPoolFirstPass = _st.VerifiedPoolScenario(
        name="64 candidates at 5 percent valid, first passing one returned",
        description="A check that rejects 10 percent of valid and passes 2 percent of invalid "
                    "candidates; 1,024 output tokens each at $15 per million; a defective deploy "
                    "costs $500.",
        pool_size=64,  # scenario assumption
        valid_rate=0.05,  # scenario assumption: valid-candidate rate
        false_reject_rate=0.10,  # scenario assumption: epsilon_FR
        false_accept_rate=0.02,  # scenario assumption: epsilon_FA
        tokens_per_candidate=Q_(1024, "token"),  # scenario assumption
        output_price=ProbeVsGeneratedExplanation.output_price,
        incident_loss=Q_(500, "USD"),  # scenario assumption: loss per defective deploy
    )
    SparseValidPoolHalvedFalseAccept = SparseValidPoolFirstPass.variant(
        name="The same pool with the false-acceptance rate halved",
        false_accept_rate=SparseValidPoolFirstPass.false_accept_rate / 2)
    SparseValidPoolDoubledValidRate = SparseValidPoolFirstPass.variant(
        name="The same pool with the valid-candidate rate doubled",
        valid_rate=2 * SparseValidPoolFirstPass.valid_rate)

    ArgmaxScoreNoise = _st.ArgmaxNoiseScenario(
        name="Argmax over equally good candidates scored with Gaussian noise",
        description="Expected winner's scoring error at 1 to 64 candidates, and a loophole in "
                    "0.5 percent of single samples in a pool of 64.",
        pool_sizes=[1, 2, 4, 8, 16, 32, 64],  # scenario assumption: pool sizes plotted
        small_pool=4,  # scenario assumption
        large_pool=SparseValidPoolFirstPass.pool_size,
        exploit_rate=0.005,  # scenario assumption: share of samples exploiting a check gap
    )

    ScopedPlanRevision = _st.ScopedRevisionScenario(
        name="A 12-node candidate plan, 3 nodes invalidated",
        description="Equal 150-token nodes; regenerate only the refuted premise's descendants.",
        total_nodes=12,  # scenario assumption
        replaced_nodes=3,  # scenario assumption
        tokens_per_node=150,  # scenario assumption: generated tokens per node
    )

    StepCheckedSearchTree = _st.PrunedSearchTreeScenario(
        name="A depth-3 search tree, 5 children per node, step check rejecting 68 percent",
        description="Independent rejection of each child; and an unpruned tree of 4 alternatives "
                    "over 10 steps for scale.",
        branching=5,  # scenario assumption
        rejection_rate=0.68,  # scenario assumption
        depth=3,  # scenario assumption
        wide_branching=4,  # scenario assumption: unpruned illustration
        wide_depth=10,  # scenario assumption: unpruned illustration
    )

    FlatVsBeamSearch = _st.SearchAccountsScenario(
        name="16 flat candidates against a width-4 beam with 3 extensions per prefix",
        description="Four segments of 128 tokens over a 4,096-token prompt; 8 concurrent streams "
                    "and 8 test slots; 15 ms per full-prefix score and 0.5 s per final test; "
                    "$3, $15, and $0.50 per million input, output, and scorer tokens.",
        floors=Llama3_70B_Int8DecodeH100,
        segments=4,  # scenario assumption
        segment_tokens=128,  # scenario assumption
        prompt_tokens=OutputCeilingThreeAllocations.prompt,
        flat_candidates=16,  # scenario assumption
        beam_width=4,  # scenario assumption
        extensions=3,  # scenario assumption
        generation_slots=8,  # scenario assumption: concurrent streams at an unchanged rate
        test_slots=8,  # scenario assumption: concurrent isolated test runs
        score_time=Q_(15, "ms"),  # scenario assumption
        test_time=Q_(0.5, "second"),  # scenario assumption
        input_price=OutputCeilingThreeAllocations.input_price,
        output_price=OutputCeilingThreeAllocations.output_price,
        scorer_price=Q_(0.50, "USD / Mtoken"),  # scenario assumption: scorer input price
    )

    DeadlineBoundRepairPolicies = _st.DeadlinePolicyScenario(
        name="One candidate, 8 in parallel, or feedback under a 75 s deadline",
        description="A 2,048-token prompt, 1,000-token candidates, 10 s checks; the first candidate "
                    "passes 40 percent of the time and a revision fed a 340-token failure report "
                    "passes 35 percent; at most three feedback rounds requested.",
        floors=Llama3_70B_Int8DecodeH100,
        prompt=2048,  # scenario assumption
        output=1000,  # scenario assumption
        diagnostic=OutputCeilingThreeAllocations.report_tokens,
        breadth=8,  # scenario assumption
        max_rounds=3,  # scenario assumption: requested feedback cap
        deadline=Q_(75, "second"),  # scenario assumption
        check_time=Q_(10, "second"),  # scenario assumption
        initial_pass=0.40,  # scenario assumption
        revision_pass=0.35,  # scenario assumption: conditional on a failed first attempt
        input_price=OutputCeilingThreeAllocations.input_price,
        output_price=OutputCeilingThreeAllocations.output_price,
        input_ceiling=32_768,  # scenario assumption
        output_ceiling=8_192,  # scenario assumption
        check_ceiling=8,  # scenario assumption
        concurrency=8,  # scenario assumption: generator and check capacity
    )


    # --- Context engineering (added 2026-10-20) ------------------------------
    # The reference repair harness's request, budgeted by role; the layout,
    # compaction, and invalidation scenarios reuse its sizes. Token counts are
    # serialized-request stipulations, not measurements of any product.
    RepairContextBudget = _st.ContextBudgetScenario(
        name="Repair-harness request budgeted by role, appending against selecting",
        description="The SWE-bench harness's window and step budget; 8,000 tokens reserved for the "
                    "reply, 3,000 of instructions, 12,000 of tool declarations, 24,000 of task state; "
                    "a 4,000-token tool result per turn appended, or 8,000 tokens of selected "
                    "observations on every call; a 2,000-token digest.",
        window=int(_Agents.Coding.SWE_Bench_Runner.context_window.to("token").magnitude),
        output_reserve=8000,  # scenario assumption: K_max, tokens
        system_tokens=3000,  # scenario assumption: application instructions
        tool_tokens=12000,  # scenario assumption: tool declarations
        state_tokens=24000,  # scenario assumption: constraints, progress record, code excerpts
        observation_per_turn=4000,  # scenario assumption: tool result appended per turn
        selected_observation=8000,  # scenario assumption: observations a selecting policy keeps
        turns=_Agents.Coding.SWE_Bench_Runner.max_trajectory_steps,
        digest_tokens=2000,  # scenario assumption: a compacted per-turn result
    )

    ToyHeadAttentionDilution = _st.AttentionDilutionScenario(
        name="One attention head with fixed logits at 100 and 100,000 keys",
        description="Target logit 6, every other key at -2; only the number of keys changes.",
        target_logit=6.0,  # scenario assumption: toy scaled logit of the target key
        background_logit=-2.0,  # scenario assumption: toy scaled logit of every other key
        short_keys=100,  # scenario assumption: total keys, short context
        long_keys=100_000,  # scenario assumption: total keys, long context
    )
    ToyHeadLookalikeExcerpt = ToyHeadAttentionDilution.variant(
        name="The same head with a superseded look-alike excerpt among unrelated keys",
        description="40,000 unrelated keys at -2 and an 850-token stale excerpt whose keys share the "
                    "evidence's identifiers, at logit 2.",
        unrelated_keys=40_000,  # scenario assumption
        lookalike_keys=850,  # scenario assumption: the superseded excerpt, one key per token
        lookalike_logit=2.0,  # scenario assumption
    )


    RepairPrefixLayoutBill = _st.PrefixLayoutBillScenario(
        name="Repair trajectory's input bill, volatile first against root, trunk, leaf",
        description="The selecting-policy request split into a root of instructions and tools, a "
                    "trunk of task state, and a leaf of selected observations; two trunk rewrites; "
                    "a hypothetical $3 per million input tokens, cached reads at 10 percent.",
        turns=RepairContextBudget.turns,
        root_tokens=RepairContextBudget.system_tokens + RepairContextBudget.tool_tokens,
        trunk_tokens=RepairContextBudget.state_tokens,
        leaf_tokens=RepairContextBudget.selected_observation,
        trunk_rewrites=2,  # scenario assumption: one per compaction pass (RepairCompactionWatermarks)
        input_price=Q_(3.00, "USD / Mtoken"),  # scenario assumption: hypothetical, not a provider schedule
        cached_fraction=0.10,  # scenario assumption: cached reads billed at this share
    )

    RepairCompactionWatermarks = _st.CompactionCadenceScenario(
        name="Repair request compacted between a 92,000-token trigger and the selecting request",
        description="Each 4,000-token result is appended; a pass resets the request to the "
                    "selecting policy's size; an alternate 81,000-token target narrows the gap.",
        window=RepairContextBudget.window,
        output_reserve=RepairContextBudget.output_reserve,
        target=(RepairContextBudget.system_tokens + RepairContextBudget.tool_tokens
                + RepairContextBudget.state_tokens + RepairContextBudget.selected_observation),
        trigger=92_000,  # scenario assumption: hypothetical high watermark
        tokens_per_turn=RepairContextBudget.observation_per_turn,
        turns=RepairContextBudget.turns,
        alternate_target=81_000,  # scenario assumption: a narrower gap, same trigger
    )

    SupersededExcerptInvalidation = _st.ContextInvalidationScenario(
        name="A superseded 850-token source excerpt opening the repair request's trunk",
        description="Remove it, replace it with a 24-token marker, or append a 128-token notice; "
                    "the reused prefix is the root.",
        request_tokens=RepairPrefixLayoutBill.root_tokens + RepairPrefixLayoutBill.trunk_tokens
        + RepairPrefixLayoutBill.leaf_tokens,
        prefix_tokens=RepairPrefixLayoutBill.root_tokens,
        superseded_tokens=ToyHeadLookalikeExcerpt.lookalike_keys,
        marker_tokens=24,  # scenario assumption: in-place tombstone
        notice_tokens=128,  # scenario assumption: appended notice with role and message framing
    )

    CompactionConstraintRetention = _st.HorizonCompoundingScenario(
        name="One constraint carried through repeated compactions at 5 percent loss per round",
        description="Independent rounds; survival after 10 and 40 rounds and the first round below "
                    "even odds.",
        step_success=0.95,  # scenario assumption: hypothetical per-round survival of one constraint
        short_horizon=10,  # scenario assumption: rounds
        long_horizon=40,  # scenario assumption: rounds
        target_success=0.5,  # even odds
    )

    ProtectedFieldCheckPayback = _st.ProtectedFieldCheckScenario(
        name="Twelve constraints over three compactions, with and without a protected-field check",
        description="3 percent loss per constraint per round; acceptance 0.6 when all survive; a $2 "
                    "attempt; a check that catches 90 percent of losses for $0.05 per round.",
        constraints=12,  # scenario assumption
        loss_per_round=0.03,  # scenario assumption
        rounds=3,  # scenario assumption
        acceptance_if_intact=0.6,  # scenario assumption
        attempt_cost=Q_(2.00, "USD"),  # scenario assumption
        detection=0.90,  # scenario assumption: share of losses the check detects and repairs
        check_cost_per_round=Q_(0.05, "USD"),  # scenario assumption: one correction call per round
        survival_floor=0.5,  # even odds
        target_survival=0.95,  # scenario assumption: survival to hold without a check
    )

    # --- Learned proposers on the warehouse mobile manipulator (added 2026-10-20) ---
    IntentModelEvaluationFp16 = _st.VlaEvaluationScenario(
        name="The warehouse manipulator's 7B intent model with 16-bit weights",
        description="One fresh evaluation: compute-bound prefill of one view plus prompt, then a "
                    "bandwidth-bound weight sweep per remaining action token, on the Jetson AGX Orin.",
        machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
        weight_bytes_per_param=Q_(2, "byte/param"),  # scenario assumption: FP16 weights
    )
    IntentModelEvaluation8Bit = IntentModelEvaluationFp16.variant(
        name="The warehouse manipulator's 7B intent model with 8-bit weights",
        weight_bytes_per_param=Q_(1, "byte/param"),  # scenario assumption: 8-bit weights, 16-bit arithmetic
    )
    IntentModelEvaluationInt4 = _st.VlaEvaluationScenario(
        name="The warehouse manipulator's 7B intent model as deployed (INT4 weights)",
        description="The deployed configuration: weight-only INT4, one view plus prompt, one token per "
                    "action dimension.",
        machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
    )
    IntentModelEvaluationParallel = IntentModelEvaluationInt4.variant(
        name="The deployed intent model with parallel action decoding",
        parallel_decode=True,  # scenario assumption: OpenVLA-OFT-style parallel decoding
    )
    ThreeViewVlaTokens = _st.VlaTokenBudgetScenario(
        name="Tokens of a three-camera VLA step on the intent model's backbone",
        description="Patch tokens of 224-pixel views at patch 14 (OpenVLA config.json), with prompt and "
                    "joint-state tokens, and the embedding and key-value bytes they occupy.",
        model=_Embodied.MobileManipulator.WarehouseMobileManipulator.intent_model,
        views=3,            # scenario assumption: left wrist, right wrist, overhead
        frame_px=224,       # OpenVLA config.json image_sizes
        patch_px=14,        # OpenVLA config.json vision encoders (patch14)
        prompt_tokens=32,   # scenario assumption: instruction tokens
        state_tokens=16,    # scenario assumption: joint-state tokens
        context_tokens=800,  # scenario assumption: rounded context length S
        element_bytes=Q_(2, "byte"),  # scenario assumption: FP16 activations, keys, and values
    )
    IntentModelEvaluationThreeViews = IntentModelEvaluationInt4.variant(
        name="The deployed intent model reading three camera views",
        input_tokens=ThreeViewVlaTokens.context_tokens,
    )
    IntentModelFiveHertz = _st.RateTargetScenario(
        name="The deployed intent model against a 5 Hz target",
        description="One fresh evaluation per 200 ms period, not pipelined.",
        evaluation=IntentModelEvaluationInt4,
        target_rate=Q_(5, "Hz"),  # scenario assumption: candidate intent rate
    )
    IntentModelParallelFiveHertz = IntentModelFiveHertz.variant(
        name="Parallel action decoding against a 5 Hz target", evaluation=IntentModelEvaluationParallel)
    ThreeViewKvHistory = _st.KvHistoryScenario(
        name="Five seconds of retained three-camera context on the intent model's backbone",
        description="A hypothetical history-keeping variant: key-value cache at 10 frames a second beside "
                    "the 16-bit weights, against the module's memory.",
        tokens=ThreeViewVlaTokens,
        machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
        history_window=Q_(5, "second"),  # scenario assumption
        history_rate=Q_(10, "Hz"),       # scenario assumption
        weight_bytes_per_param=Q_(2, "byte/param"),  # scenario assumption: FP16 weights
    )
    IntentBackboneDecoderFamilies = _st.DecoderFamiliesScenario(
        name="One fresh 16-step chunk from the INT4 intent backbone, per decoder family",
        description="Binned autoregressive, FAST, parallel, and flow-expert decoding after the shared prefill.",
        backbone=IntentModelEvaluationInt4,
        fast_tokens=30,  # Pertsch, K. et al. (2025), FAST: Efficient Action Tokenization for ...
        expert_params=Q_(300000000.0, "param"),  # Black, K. et al. (2024), pi0: A Vision-Language-Action Flow Model f...
        expert_steps=10,  # Black, K. et al. (2024), pi0: A Vision-Language-Action Flow Model f...
        expert_bytes_per_param=Q_(2, "byte/param"),  # scenario assumption: 16-bit expert weights
        kv_bytes_per_value=Q_(2, "byte"),            # scenario assumption: FP16 keys and values
    )
    WarehouseProposerTraffic = _st.ProposerTrafficScenario(
        name="The warehouse manipulator's two proposers on its memory bus",
        description="Intent-model sweeps, chunk supply, renewal gap, and the proposers' share of the "
                    "sustained bandwidth.",
        machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
        fp16_bytes_per_param=Q_(2, "byte/param"),  # scenario assumption: FP16 weights
    )
    ConveyorMugEvidenceHorizon = _st.EvidenceHorizonScenario(
        name="The tracked conveyor mug against the chunk policy's renewal",
        description="Untracked deadline, tracked evidence horizon, age at replacement, and the drift of a "
                    "chunk executed to its end.",
        site=_Embodied.Scenario.WarehouseAisle,
        machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
    )
    ConveyorMugEvidenceHorizonHalfRenewal = ConveyorMugEvidenceHorizon.variant(
        name="The tracked conveyor mug with the chunk policy renewing at half rate",
        chunk_rate=_Embodied.MobileManipulator.WarehouseMobileManipulator.chunk_rate / 2,  # scenario assumption
    )
    OpenVlaOrinQuantization = _st.WeightOnlyQuantizationScenario(
        name="OpenVLA-7B weight formats on the Jetson AGX Orin, computed against measured",
        description="The intent model priced at FP16, 8-bit, and INT4 weights against the Jetson AI Lab "
                    "measurements and action scores of the same formats.",
        formats={"FP16": IntentModelEvaluationFp16, "8-bit": IntentModelEvaluation8Bit,
                 "INT4": IntentModelEvaluationInt4},
        measured={
            "FP16": Q_(840.0, "ms"),  # Franklin, D., NVIDIA Jetson AI Lab, 'OpenVLA' tutorial (2024): Open...
            "8-bit": Q_(471.0, "ms"),  # Franklin, D., NVIDIA Jetson AI Lab, 'OpenVLA' tutorial (2024): Open...
            "INT4": _Embodied.MobileManipulator.WarehouseMobileManipulator.intent_reference_latency,
        },
        scores={
            "FP16": 0.953,  # Franklin, D., NVIDIA Jetson AI Lab, 'OpenVLA' tutorial (2024): Open...
            "8-bit": 0.952,  # Franklin, D., NVIDIA Jetson AI Lab, 'OpenVLA' tutorial (2024): Open...
            "INT4": 0.901,  # Franklin, D., NVIDIA Jetson AI Lab, 'OpenVLA' tutorial (2024): Open...
        },
        deployed="INT4",
        reference="FP16",
        group_size=128,  # Lin, J. et al. (2024), AWQ: Activation-aware Weight Quantization fo...
        scale_bits=16,   # scenario assumption: one FP16 scale per group
        weight_bits=4,   # scenario assumption: INT4 weights
    )
    ProposerConfidenceRegimes = _st.ConfidenceRegimesScenario(
        name="Stipulated proposer confidence against task success under distribution shift",
        description="Illustrative values for four operating regimes, not test results.",
        regimes=[  # scenario assumption: stipulated, illustrative (confidence, task success)
            ("In distribution", 0.91, 0.89),
            ("Near OOD", 0.89, 0.52),
            ("Sensor degraded", 0.84, 0.31),
            ("Far OOD", 0.78, 0.08),
        ],
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
        time_per_item=Q_(1.5, "hour"),  # Cordts et al. (2016), The Cityscapes Dataset for Semantic Urban Sce...
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

    # --- Data engineering: delivering a dataset to the accelerator, and watching it drift (added 2026-10-20) ---
    # ResNet-50 on one A100 carries delivery: the arithmetic ceiling, the bytes a
    # sustained rate demands, what each storage tier supplies, and the loader
    # workers that decode it. The keyword spotter's loudness feature carries drift:
    # a K-S test at two sample sizes, an illustrative location shift, and PSI on
    # quintile bins. Cited figures come from Hardware, Models, Datasets, Systems,
    # Infrastructure, Ops, and Literature; the rest are stated scenario assumptions.

    PetabyteCrossRegionMove = _st.TransferScenario(
        name="One petabyte moved between cloud regions",
        description="A training corpus in US East and accelerators in US West, over a dedicated 100 Gb/s or a "
                    "10 Gb/s link, priced at an inter-region egress rate and set against a 20-hour training run.",
        payload=Q_(1, "PB"),              # scenario assumption: the corpus
        links={"100 Gb/s": _Fabrics.Ethernet_100G.bandwidth, "10 Gb/s": _Fabrics.Ethernet_10G.bandwidth},
        egress_rate=Q_(0.02, "USD/GB"),   # scenario assumption: AWS inter-region transfer rate (2024)
        compute_price=_Infrastructure.Pricing.Cloud.TpuV4PerHour.rate,
        job_time=Q_(20, "hour"),          # scenario assumption: the training run the data would feed
    )
    ResNet50A100InputFeed = _st.AcceleratorInputFeedScenario(
        name="ResNet-50 training on one A100, fed JPEGs from a cloud block volume",
        description="The A100's arithmetic ceiling for ResNet-50 training, the well-tuned sustained rate, 150 KB "
                    "JPEGs that decode to 224x224x3 FP32 tensors, a 250 MB/s attached volume, and host loader "
                    "workers at 375 images a second each.",
        workload=_Models.Vision.ResNet50,
        hardware=_Hardware.Cloud.A100,
        sustained_rate=ImageNetLabelingVersusResNetRun.throughput_per_accelerator,
        encoded_sample=Q_(150, "KB"),     # scenario assumption: average ImageNet-class JPEG as stored
        decoded_sample=(_Datasets.ImageNet.image_width * _Datasets.ImageNet.image_height
                        * _Datasets.ImageNet.image_channels * _BYTES_FP32),
        supply=_Storage.CloudBlockStorageBaseline,
        training_flops_factor=3,          # forward pass plus a backward pass of about twice its cost
        worker_rate=Q_(375, "count/second"),  # scenario assumption: JPEG decode and augment per CPU worker
        host_decode_cap=Q_(3500, "count/second"),  # scenario assumption: illustrative host-wide decode ceiling
    )
    ResNet50EightA100InputFeed = ResNet50A100InputFeed.variant(
        name="ResNet-50 training on eight A100s in one server, fed from one storage path",
        description="The one-A100 feed scaled to the eight accelerators of one server.",
        accelerators=8,                   # one 8-GPU server
    )
    RecommendationFeatureLookups = _st.FeatureLookupScenario(
        name="A recommendation service reading ten stored features per request",
        description="10,000 requests a second, ten feature reads each, 5 KB feature vectors summarizing 150 KB images.",
        request_rate=Q_(10_000, "1/second"),  # scenario assumption
        reads_per_request=10,             # scenario assumption
        feature_bytes=Q_(5, "KB"),        # scenario assumption: one stored feature vector
        raw_sample_bytes=ResNet50A100InputFeed.encoded_sample,
    )
    MswcCorpusStorageTiers = _st.CorpusTierScenario(
        name="The Multilingual Spoken Words Corpus on object storage and on local NVMe",
        description="Every clip at 16 kHz and 16 bits, priced per month and read sequentially on each tier.",
        dataset=_Datasets.MSWC,
        storage_rates={"object store": _Infrastructure.Pricing.Storage.S3StandardPerTbMonth.rate,
                       "NVMe low": _Infrastructure.Pricing.Storage.NvmeLowPerTbMonth.rate,
                       "NVMe high": _Infrastructure.Pricing.Storage.NvmeHighPerTbMonth.rate},
        read_bandwidths={"object store": _Storage.ObjectStoreSingleStream.bandwidth,
                         "NVMe": _Storage.LocalNvmeTrainingTypical.bandwidth},
    )
    FraudModelColumnProjection = _st.ColumnProjectionScenario(
        name="A fraud model that uses 20 of a table's 100 equal-size columns",
        description="A row format reads every column; a columnar format reads only the 20.",
        columns_used=20,                  # scenario assumption
        columns_total=100,                # scenario assumption
    )
    GzipVersusSnappyEpochs = _st.CodecEpochScenario(
        name="A 100 GB epoch decompressed with gzip or Snappy for 50 epochs",
        description="Host CPU decompression time per epoch and over the run, at each codec's decompression rate.",
        epoch_bytes=Q_(100, "GB"),        # scenario assumption
        epochs=50,                        # scenario assumption
        decode_rates={"gzip": Q_(120, "MB/s"),    # scenario assumption: typical gzip decompression
                      "Snappy": Q_(500, "MB/s")},  # scenario assumption: typical Snappy decompression
    )
    LoudnessDriftKs1000 = _st.KsDriftScenario(
        name="K-S drift test on clip loudness, 1,000 training against 1,000 recent clips",
        description="A quieter microphone on part of the fleet; the observed largest CDF gap against the "
                    "critical value at the 0.05 level.",
        coefficient=_Ops.Monitoring.KsTestCoefficient,
        n_reference=1000,                 # scenario assumption
        n_current=1000,                   # scenario assumption
        observed_statistic=0.08,          # scenario assumption: observed largest CDF gap
    )

    # ── Fleet goodput: one 70B training job on a shared pool (added 2026-10-20) ──
    SharedPoolSchedulingSnapshot = _st.SchedulingSnapshotScenario(
        name="Scheduling goodput of a shared pool in one utilization snapshot",
        description="Assigned capacity less the gangs still waiting for members; input waits inside "
                    "complete allocations are left to runtime goodput.",
        assigned_share=float(_ReferenceStats.FleetUtilizationSnapshot.AssignedShare),
        incomplete_gang_share=float(_ReferenceStats.FleetUtilizationSnapshot.IncompleteGangShare),
        input_wait_share=float(_ReferenceStats.FleetUtilizationSnapshot.InputWaitShare),
    )
    Llama2_70BJobFailures128 = _st.SeriesFailureScenario(
        name="Failure interval of a 128-GPU Llama 2 70B training job",
        description="Sixteen DGX H100 nodes; GPU, PCIe link, and the node's NIC in series.",
        n_devices=16 * _Nodes.DGX_H100.accelerators_per_node,  # scenario assumption: 16 nodes of 8
        devices_per_node=_Nodes.DGX_H100.accelerators_per_node,
        component_mttf={
            "gpu": Q_(_Reliability.Gpu.mttf_hours, "hour"),
            "pcie": Q_(_Reliability.PcieSwitch.mttf_hours, "hour"),
            "network": Q_(_Reliability.Nic.mttf_hours, "hour"),
        },
        per_node={
            "gpu": _Nodes.DGX_H100.accelerators_per_node,
            "pcie": _Nodes.DGX_H100.accelerators_per_node,  # one PCIe link to the host per GPU
            "network": 1,  # scenario assumption: one NIC shared by the node's GPUs
        },
    )
    Llama2_70BRuntimeGoodput128 = _st.RuntimeGoodputScenario(
        name="Runtime goodput of a 128-GPU Llama 2 70B training job over one week",
        description="Startup, input waits, checkpoint waste at the Young-Daly interval, recovery, and one "
                    "budgeted forced preemption a day, on one held-time denominator.",
        failures=Llama2_70BJobFailures128,
        workload=_Models.Language.Llama2_70B,
        bytes_per_param=_ReferenceStats.MixedPrecisionAdam.CheckpointBytesPerParameter,
        recovery=_Reliability.Recovery,
        input_wait=SharedPoolSchedulingSnapshot.input_wait_of_held,  # the pool's input waits, of held time
        window=Q_(1, "week"),  # scenario assumption: one launch per week of held time
        lost_work=_ReferenceStats.ForcedPreemptionProfile.LostWork,
        preemption_rate=Q_(1, "1/day"),  # scenario assumption: a per-job budget of one forced preemption a day
    )
    Llama2_70BFleetMfu128 = _st.FleetMfuScenario(
        name="Llama 2 70B training MFU on one DGX H100 node and on 128 H100s",
        description="The same work per GPU: 180 ms node steps at the node's batch, 245 ms steps on the "
                    "shared fleet with exposed cross-node gradient reduction.",
        workload=_Models.Language.Llama2_70B,
        hardware=_Hardware.Cloud.H100,
        node_devices=_Nodes.DGX_H100.accelerators_per_node,
        fleet_devices=Llama2_70BJobFailures128.n_devices,
        node_tokens=2204,  # scenario assumption: node baseline tokens per step
        fleet_tokens=35446,  # scenario assumption: 128-GPU tokens per step
        node_step_time=Q_(180, "ms"),  # scenario assumption: node baseline, optimizer state offloaded to host
        fleet_step_time=Q_(245, "ms"),  # scenario assumption: 128-GPU step on the shared fleet
    )
    Llama2_70BSharedPoolGoodput128 = _st.GoodputScenario(
        name="ML productivity goodput of the 128-GPU Llama 2 70B job on the shared pool",
        description="The pool's scheduling goodput, the job's runtime goodput, and its fleet MFU, with "
                    "each owner's ceiling: partial holds refilled, input waits removed, the node baseline.",
        scheduling=SharedPoolSchedulingSnapshot.scheduling_goodput,
        runtime=Llama2_70BRuntimeGoodput128.result.runtime_goodput,
        program=Llama2_70BFleetMfu128.fleet_mfu,
        ceilings={
            "S": SharedPoolSchedulingSnapshot.partial_hold_ceiling,
            "R": (Llama2_70BRuntimeGoodput128.result.runtime_goodput
                  + Llama2_70BRuntimeGoodput128.result.input_wait_share),
            "P": Llama2_70BFleetMfu128.node_mfu,
        },
        device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
    )
    Llama3_405BGoodput16K = _st.GoodputScenario(
        name="Llama 3 405B on its dedicated 16,384-GPU cluster",
        description="Effective training time (a lower bound) as R and BF16 MFU at 16,384 GPUs as P; S = 1 "
                    "on a dedicated cluster. The two come from different windows, so the product is an estimate.",
        scheduling=1.0,
        runtime=float(_Literature.TrainingRuns.Llama3_405B_EffectiveTrainingTimeFraction),
        program=float(_Literature.TrainingRuns.Llama3_405B_Mfu16K),
    )
    Llama3_405BOnSharedPool = Llama3_405BGoodput16K.variant(
        name="Llama 3 405B's runtime and program goodput on the shared pool",
        scheduling=SharedPoolSchedulingSnapshot.scheduling_goodput,
    )
    Llama2_70BDecodeFloorTp8 = _st.DecodeScenario(
        name="Batch-one decode floor of Llama 2 70B on one 8-way tensor-parallel H100 node",
        description="Each GPU streams its FP16 weight shard once per token.",
        workload=_Models.Language.Llama2_70B,
        hardware=_Hardware.Cloud.H100,
        precision="fp16",
        batch_size=1,
        tp_size=_Nodes.DGX_H100.accelerators_per_node,
    )
    LayerNormMeasuredH100 = _st.RooflinePointScenario(
        name="A profiled LayerNorm kernel placed on the H100 roofline",
        description="Illustrative counter readings: achieved FLOP rate and achieved HBM bandwidth.",
        hardware=_Hardware.Cloud.H100,
        achieved_rate=Q_(5, "TFLOPs/s"),  # scenario assumption: counter reading
        achieved_bandwidth=Q_(2.8, "TB/s"),  # scenario assumption: counter reading
    )
    Llama2_70BNodeTensorParallelStep = _st.TensorParallelScenario(
        name="Tensor-parallel AllReduces of one Llama 2 70B node-baseline training step",
        description="Two exchanges per layer forward and two backward over NVLink, bandwidth only, "
                    "against the 180 ms node step.",
        workload=_Models.Language.Llama2_70B,
        tp_size=_Nodes.DGX_H100.accelerators_per_node,
        microbatch=1,
        seq_len=Llama2_70BFleetMfu128.node_tokens,  # the node step's tokens as one sequence
        collectives_per_layer=4,  # Megatron layout: 2 forward + 2 backward
        link_bandwidth=_Hardware.Cloud.H100.nvlink.bandwidth_per_direction,
        compute_time=Llama2_70BFleetMfu128.node_step_time,
    )
    Llama2_70BDecodeAllReduceTp8 = _st.MessageFusionScenario(
        name="Batch-one tensor-parallel AllReduces of the Llama 2 70B serving replica",
        description="Each per-layer exchange moves tens of kilobytes over NVLink; the collective launch "
                    "costs far more than the bytes.",
        alpha=_Ops.RuntimeOverheads.KernelLaunch,
        bandwidth=_Hardware.Cloud.H100.nvlink.bandwidth_per_direction,
        # 2 x d_model activations of 2 bytes per AllReduce at batch one
        message=(2 * _Models.Language.Llama2_70B.hidden_dim * 1 * _BYTES_FP16).to("byte"),
        count=2 * _Models.Language.Llama2_70B.layers,  # two exchanges per layer per token
    )
    Decode13BCompilationPayback = _st.CompilationPaybackScenario(
        name="Compilation payback for a 13B decode step on one H100",
        description="A constructed decode budget: compilation keeps the matrix multiplies, removes most "
                    "element-wise and launch time; default and autotuning builds; 128 one-GPU replicas.",
        token_rate=Q_(120, "1/s"),  # scenario assumption: eager tokens per second, 13B INT4 weights
        step_time=Q_(8.33, "ms"),  # scenario assumption: 1/120 s, carried at the two decimals shown
        shares={"matmul": 0.50, "elementwise": 0.35, "launch": 0.15},  # scenario assumption
        removed={"elementwise": 0.70, "launch": 0.80},  # scenario assumption
        builds={"default": Q_(1, "minute"), "autotune": Q_(15, "minute")},  # scenario assumption
        replicas=Llama2_70BJobFailures128.n_devices,  # replicas filling an allocation of the same size
        gpus_per_replica=1,
        replica_life=Q_(1, "hour"),  # scenario assumption: a replica added for one burst hour
    )
    NsightTraceVolume1K = _st.TraceVolumeScenario(
        name="A 5-second Nsight Systems trace of every GPU in a 1,024-GPU job",
        description="Trace volume grows with every traced device.",
        per_device=_ReferenceStats.ProfilerTraceProfile.NsightTracePerGpu,
        devices=DenseLLMPretrainingReference.system.total_accelerators,
        window=_ReferenceStats.ProfilerTraceProfile.TraceWindow,
    )

    # --- Serving at scale (added 2026-10-20) --------------------------------
    # What counts as a served request, how many one replica admits, and what a
    # fleet of replicas costs. The reference replica is the Llama-3 70B GQA
    # model on one DGX H100 node; the measured traffic is the Azure LLM
    # inference traces; the measured frontier is InferenceX.
    LlmServingFirstYear = _st.ServingCostScenario(
        name="A 70B model trained once, then served for a year",
        description="One million daily users send 50 requests each at an assumed tenth of a cent "
                    "per request; training cost $2M once.",
        training_cost=2_000_000,  # scenario assumption: one-time training spend, USD
        daily_users=1_000_000,  # scenario assumption
        requests_per_user_per_day=50,  # scenario assumption
        cost_per_request=0.001,  # scenario assumption: serving cost per request, USD
        efficiency_gain=0.10,  # scenario assumption: cut in cost per request
        days_per_year=_DAYS_PER_YEAR,
    )


    InteractiveRequestServingTax = _st.ServingTaxScenario(
        name="A 100 ms interactive request with 70 ms of model execution",
        description="Everything outside the model (routing, network, serialization, coordination, "
                    "queueing) shares what is left; four sequential exchanges of 50 to 500 us.",
        latency_target=Q_(100, "ms"),  # scenario assumption
        model_time=Q_(70, "ms"),  # scenario assumption: idle-replica model execution
        exchanges=4,  # scenario assumption: sequential network exchanges
        exchange_low=Q_(50, "microsecond"),  # scenario assumption
        exchange_high=Q_(500, "microsecond"),  # scenario assumption
    )

    VisionBatchingKnee = _st.BatchingKneeScenario(
        name="Vision batch: 2 ms fixed, 1 ms per image",
        description="A compute-bound vision model whose fixed cost per batch is small.",
        fixed_time=Q_(2, "ms"),  # scenario assumption
        per_request_time=Q_(1, "ms"),  # scenario assumption
    )
    LlmDecodeBatchingKnee = VisionBatchingKnee.variant(
        name="LLM decode batch: 40 ms fixed, 0.5 ms per sequence",
        description="About 40 ms streams a 70B model's FP16 weights through one H100's HBM "
                    "(Llama3_70B_DecodeH100); 0.5 ms reads one sequence's own KV state.",
        fixed_time=Q_(40, "ms"),  # scenario assumption: rounded weight-stream time
        per_request_time=Q_(0.5, "ms"),  # scenario assumption: per-sequence KV read
    )

    VisionServingLittlesLaw = _st.LittlesLawScenario(
        name="Vision service at 1,000 requests per second, 100 ms mean latency",
        description="Replicas run batches of 8 in 80 ms; the fleet targets 80 percent mean "
                    "service utilization.",
        arrival_rate=Q_(1000, "1/second"),  # scenario assumption
        time_in_system=Q_(100, "ms"),  # scenario assumption: mean end-to-end latency
        batch_size=8,  # scenario assumption
        batch_service_time=Q_(80, "ms"),  # scenario assumption
        utilization_target=0.8,  # scenario assumption
    )


    Llama3_70B_ReplicaDecodeBudget = _st.DecodeBudgetScenario(
        name="Constructed batch-one decode budget of a 70B replica on 8 H100s",
        description="An 85 ms baseline step split over GEMM, attention, element-wise, AllReduce, and "
                    "launch; INT4 weights cut GEMM, then fusion and graph replay cut the rest.",
        workload=_Models.Language.Llama2_70B,
        tp_size=int(_ReferenceStats.ServingProfiles.PrecisionDividendTensorParallelDegree),
        accelerator=_Hardware.Cloud.H100,
        weight_bytes_per_param=_BYTES_FP16,
        tpot_target=Q_(50, "ms"),  # scenario assumption: illustrative endpoint TPOT target
        baseline_step=Q_(85, "ms"),  # scenario assumption: constructed baseline step
        intervals=["GEMM", "attention", "element-wise", "AllReduce", "launch"],
        baseline_shares=[0.40, 0.24, 0.12, 0.12, 0.12],  # scenario assumption
        gemm_cut=0.45,  # scenario assumption: INT4 weights
        attention_cut=0.25,  # scenario assumption: fused decode-attention kernel
        fused_elementwise=Q_(3.4, "ms"),  # scenario assumption
        replay_overhead=Q_(2.0, "ms"),  # scenario assumption: graph replay
        collective_launch=Q_(5, "microsecond"),  # scenario assumption: collective launch latency
    )
    Llama3_70B_ReplicaSpeculation = _st.SpeculativeRoundScenario(
        name="Speculative decoding on the tuned 70B replica step",
        description="A draft proposes 5 tokens in 4 ms; the target verifies them in one tuned step; "
                    "each candidate is accepted with probability 0.78.",
        acceptance=0.78,  # scenario assumption: draft/target pair
        draft_length=5,  # scenario assumption
        draft_time=Q_(4, "ms"),  # scenario assumption
        verify_time=Q_(Llama3_70B_ReplicaDecodeBudget.totals[2], "ms"),
    )


    # --- KV cache management on one DGX H100 node (added 2026-10-20) ----------
    RepairRequestZones = _st.RequestZonesScenario(
        name="Repair-agent request at revision r18, laid out in zones",
        description="A 32,768-token request: system prompt, tool schemas, and workspace form the "
                    "invariant shared prefix; turn history and the latest observation follow.",
        zones={  # scenario assumption: tokens per zone, in serialization order
            "System prompt": 2_048,
            "Tool schemas": 4_096,
            "Workspace": 8_192,
            "Turn history": 16_384,
            "Latest observation": 2_048,
        },
        shared_zones=3,  # system prompt, tool schemas, workspace
    )
    # Copied verbatim from the v2-10 inference branch; on merge keep one copy.
    Llama3_70BLongContextAdmissionDgxH100 = _st.KvAdmissionScenario(
        name="128K-token requests on the 70B GQA replica (8 H100s)",
        description="FP16 weights and KV cache; HBM at the registered byte capacity, less a 20 GB "
                    "runtime reserve.",
        workload=_Models.Language.Llama3_70B,
        accelerators=_Nodes.DGX_H100.accelerators_per_node,
        hbm_per_accelerator=_ReferenceStats.ServingProfiles.H100VendorMemoryBudget,
        weight_bytes_per_param=_BYTES_FP16,
        reserve=Q_(20, "GB"),  # scenario assumption: system reserve
        context_tokens=_Models.Language.Llama3_70B.context_length,
        kv_bytes_per_elem=_BYTES_FP16,
    )

    ChatbotSharedSystemPrompt = _st.PrefixSharingScenario(
        name="1,000 chat users sharing a 2,000-token system prompt",
        description="Full-width multi-head KV (64 heads of 128, FP16) with Llama-2 70B's 80 layers; "
                    "500 unique response tokens per user.",
        workload=_Models.Language.Llama2_70B,
        prefix_tokens=2000,  # scenario assumption
        users=1000,  # scenario assumption
        unique_tokens_per_user=500,  # scenario assumption: mean response
        kv_width=_Models.Language.Llama2_70B.hidden_dim,  # full-width MHA: heads x head width
        bytes_per_elem=_BYTES_FP16,
    )

    Llama2_70BPrecisionDividendGqaTp8 = _st.PrecisionDividendScenario(
        name="Precision dividend of a 70B GQA model on 8 H100s",
        description="INT4 weights and an INT8 KV cache against FP16 for both, at a 4,096-token "
                    "context, with a decimal 80 GB budget per GPU.",
        workload=_Models.Language.Llama2_70B,
        tp_size=int(_ReferenceStats.ServingProfiles.PrecisionDividendTensorParallelDegree),
        hbm_per_accelerator=Q_(_ReferenceStats.ServingProfiles.H100VendorMemoryBudget.to("GiB").magnitude,
                               "GB"),  # nameplate label read as decimal GB
        context_tokens=int(_ReferenceStats.ServingProfiles.PrecisionDividendContextLengthTokens),
        kv_heads=_Models.Language.Llama2_70B.kv_heads,
        weight_bytes=_BYTES_FP16, weight_bytes_low=_BYTES_INT4,
        kv_bytes=_BYTES_FP16, kv_bytes_low=_BYTES_INT8,
    )
    Llama2_70BPrecisionDividendMhaTp8 = Llama2_70BPrecisionDividendGqaTp8.variant(
        name="Precision dividend of the same 70B model with 64-head attention",
        kv_heads=_Models.Language.Llama2_70B.heads,
    )

    LongPromptChunkedPrefill = _st.ChunkedPrefillScenario(
        name="A 10,000-token prompt in 512-token chunks",
        description="Each iteration runs the active streams' decode tokens plus one chunk.",
        prompt_tokens=10_000,  # scenario assumption
        chunk_tokens=512,  # scenario assumption
    )

    Llama2_70BPrefillTensorParallel8 = _st.TensorParallelSpeedupScenario(
        name="Long-prompt prefill layer of a 70B model, 8-way tensor parallel",
        description="Stated 30 ms of serial compute per layer (12 attention, 18 feed-forward) and "
                    "0.3 ms per AllReduce, two per layer.",
        serial_layer_times={"attention": Q_(12, "ms"), "feed-forward": Q_(18, "ms")},  # scenario assumption
        tp_size=_Nodes.DGX_H100.accelerators_per_node,
        allreduce_time=Q_(0.3, "ms"),  # scenario assumption
        layers=_Models.Language.Llama2_70B.layers,
    )

    HeterogeneousH100A100Routing = _st.WeightedRoutingScenario(
        name="15,000 QPS over 10 H100 and 20 A100 servers",
        description="Weights proportional to each server's capacity equalize utilization.",
        servers={"H100": int(_ReferenceStats.ServingProfiles.HeterogeneousRoutingH100Servers),
                 "A100": int(_ReferenceStats.ServingProfiles.HeterogeneousRoutingA100Servers)},
        capacity={"H100": int(_ReferenceStats.ServingProfiles.HeterogeneousRoutingH100CapacityQps),
                  "A100": int(_ReferenceStats.ServingProfiles.HeterogeneousRoutingA100CapacityQps)},
        offered=int(_ReferenceStats.ServingProfiles.HeterogeneousRoutingTargetQps),
    )
    ServingFanout100Leaves = _st.AvailabilityScenario(
        name="A request that waits for 100 independent leaves",
        description="Each leaf meets its own p99 (fast 99 percent of the time); 1 - all_up is the "
                    "share of slow requests.",
        availability=0.99,  # scenario assumption: each leaf fast at its p99
        n_components=100,  # scenario assumption: leaves per request
    )
    RecommenderTailEvents = _st.TailEventRateScenario(
        name="A one-in-10,000 slow request at recommender platform scale",
        description="Archetype B request rate times a rare per-request event.",
        request_rate=_ReferenceStats.RecommenderAtScaleProfile.PlatformQps,
        event_probability=1e-4,  # scenario assumption: rare slow-request event
    )
    Llama70BReplicaColdStart = _st.ColdStartScenario(
        name="Cold start of a 70B FP16 serving replica",
        description="Provision, fetch the FP16 weights from object storage, load and warm up; a "
                    "forecast that starts replicas early lowers the peak fleet.",
        workload=_Models.Language.Llama2_70B,
        weight_bytes_per_param=_BYTES_FP16,
        fetch_bandwidth=_ReferenceStats.ReplicaColdStartProfile.ArtifactFetchBandwidth,
        provisioning_time=_ReferenceStats.ReplicaColdStartProfile.ProvisioningTime,
        load_warmup_time=_ReferenceStats.ReplicaColdStartProfile.LoadWarmupTime,
        reactive_peak_replicas=45,  # scenario assumption: reactive-only peak fleet
        predictive_peak_replicas=30,  # scenario assumption: forecast peak fleet
        prewarm_lead=Q_(30, "minute"),  # scenario assumption: pre-warm lead before each ramp
    )
    SharedPoolNoisyNeighbor = _st.NoisyNeighborScenario(
        name="Ten tenants on shared H100s, one bursting fivefold",
        description="Three tenants share a 60 GB KV pool; one swells to 45 GB. Latencies are "
                    "scenario values, not measurements.",
        kv_pool=Q_(60, "GB"),  # scenario assumption
        pool_tenants=3,  # scenario assumption
        swollen_share=Q_(45, "GB"),  # scenario assumption
        base_sequences=200,  # scenario assumption
        base_rate=100,  # scenario assumption: QPS per tenant
        burst_rate=500,  # scenario assumption
        slo=Q_(50, "ms"),  # scenario assumption: p99 SLO
        unprotected_p99_low=Q_(22, "ms"),  # scenario assumption
        unprotected_p99_high=Q_(45, "ms"),  # scenario assumption
        quota_rate=120,  # scenario assumption: quota-enforced QPS
        protected_p99=Q_(15, "ms"),  # scenario assumption
    )
    ReasoningTokenLatency = _st.ReasoningLatencyScenario(
        name="128 serial reasoning tokens at 100 ms per token",
        description="A round figure per token; the ratio does not depend on it.",
        per_token_time=Q_(100, "ms"),  # scenario assumption
        reasoning_tokens=128,  # scenario assumption
    )
    FourRequestDecodeBatch = _st.FixedBatchScheduleScenario(
        name="Four concurrent requests with answers of 50 to 200 tokens",
        description="One token per 20 ms decode iteration; fixed membership keeps every slot until "
                    "the longest answer ends.",
        output_lengths=[50, 200, 100, 150],  # scenario assumption
        prompt_lengths=[100, 80, 120, 90],  # scenario assumption
        iteration_time=Q_(20, "ms"),  # scenario assumption
    )


    # --- Retrieval from long-term memory (added 2026-10-20) ------------------
    RetrievalRecordsInAllowance = _st.RetrievalPayloadScenario(
        name="Whole retrieval records staged in a 32,768-token call",
        description="2,048 tokens reserved for output and 24,576 for the system prompt, tool "
                    "schemas, and history; a 128-token envelope and records of 512 body plus 64 "
                    "metadata tokens; metadata doubled for comparison.",
        window=32_768,           # scenario assumption: hypothetical call limit
        output_reserve=2_048,    # scenario assumption
        other_input=24_576,      # scenario assumption: system prompt, tool schemas, history
        envelope=128,            # scenario assumption: response envelope
        body=512,                # scenario assumption: body tokens per record
        metadata=64,             # scenario assumption: provenance tokens per record
        metadata_scale=2,        # scenario assumption: per-record metadata doubled
    )
    Bm25ShortAgainstLongFile = _st.Bm25PairScenario(
        name="One rare term in a short file with one match and a long file with six",
        description="100,000 documents, 5 containing the term, mean length 400 analyzer terms; "
                    "80 terms with one occurrence against 4,000 with six; k1 = 1.2, b = 0.75.",
        documents=100_000,       # scenario assumption: stipulated corpus statistics
        matching_documents=5,    # scenario assumption
        average_length=400,      # scenario assumption: analyzer terms
        short_length=80,         # scenario assumption
        short_frequency=1,       # scenario assumption
        long_length=4_000,       # scenario assumption
        long_frequency=6,        # scenario assumption
        k1=1.2,                  # scenario assumption: a common default
        b=0.75,                  # scenario assumption: a common default
    )
    ExactScanTenMillionVectors = _st.ExactScanScenario(
        name="Exact scan of 10 million 768-dimension FP16 vectors on the reference desktop CPU",
        description="20 searches in one trajectory; a scope filter that leaves 10,000 vectors; "
                    "a 10 ms per-search budget.",
        host=_Hardware.Cloud.ReferenceCPU,
        vectors=10_000_000,      # scenario assumption: hypothetical corpus
        dimension=768,           # scenario assumption: a common embedding width
        bytes_per_value=_BYTES_FP16,
        searches=20,             # scenario assumption: searches in one trajectory
        filtered_vectors=10_000,  # scenario assumption: eligible set after a selective filter
        budget=Q_(10, "ms"),     # scenario assumption: per-search latency budget
    )
    AnnIndexesTenMillionVectors = _st.AnnFootprintScenario(
        name="HNSW links and IVF-PQ codes over the 10-million-vector exact-scan corpus",
        description="32 eight-byte links per node; 64 one-byte subvector codes with 256 "
                    "centroids each and an FP32 lookup table.",
        vectors=ExactScanTenMillionVectors.vectors,
        dimension=ExactScanTenMillionVectors.dimension,
        bytes_per_value=ExactScanTenMillionVectors.bytes_per_value,
        links_per_node=32,       # scenario assumption: graph degree M
        link_bytes=Q_(8, "byte"),  # scenario assumption: one 64-bit neighbor id
        subvectors=64,           # scenario assumption: PQ subvectors m
        centroids=256,           # scenario assumption: one-byte codes
        code_bytes=Q_(1, "byte"),
        table_entry_bytes=_BYTES_FP32,
    )
    TypedTraversalDepthThree = _st.GraphNeighborhoodScenario(
        name="A depth-3 exact-tree neighborhood against the same call's retrieval allowance",
        description="3 children per node, 180 serialized tokens per node, 100 tokens of edges "
                    "and envelope.",
        window=RetrievalRecordsInAllowance.window,
        output_reserve=RetrievalRecordsInAllowance.output_reserve,
        other_input=RetrievalRecordsInAllowance.other_input,
        children=3,              # scenario assumption: exact tree, no repeated nodes
        depth=3,                 # scenario assumption: the seed is depth zero
        node_tokens=180,         # scenario assumption: per node, provenance included
        envelope_tokens=100,     # scenario assumption: edges and envelope
    )
    RepairEditIndexMaintenance = _st.IndexMaintenanceScenario(
        name="An eight-file edit: synchronous re-indexing against exclusion",
        description="Two chunks per file; stipulated per-stage costs for parsing, postings, a "
                    "local embedding batch of 16, and a remote embedding request; in-memory "
                    "exclusion entries and checks over 50 candidates.",
        files=8,                                   # scenario assumption
        chunks_per_file=2,                         # scenario assumption
        parse_time=Q_(3.5, "ms"),                  # scenario assumption: per file
        postings_time=Q_(2.0, "ms"),               # scenario assumption: per chunk
        local_batch_time=Q_(28.0, "ms"),           # scenario assumption: one local batch
        batch=16,                                  # scenario assumption
        remote_request_time=Q_(150, "ms"),         # scenario assumption: one remote request
        exclusion_write_time=Q_(50, "ns"),         # scenario assumption: per entry, in memory
        candidates=50,                             # scenario assumption
        exclusion_check_time=Q_(30, "ns"),         # scenario assumption: per candidate, in memory
    )
    LexicalGraphAgainstDenseStore = _st.StorageAmplificationScenario(
        name="A lexical-plus-graph store against a lexical-plus-dense store over 50,000 files",
        description="200 tokens and 800 bytes per file; 0.8 bytes of postings per token; four "
                    "32-byte edges per file; 100-token chunks of 768 FP32 values with the ANN "
                    "graph's links; 15 edited files; stipulated throughputs.",
        files=50_000,                              # scenario assumption
        tokens_per_file=200,                       # scenario assumption
        bytes_per_file=Q_(800, "byte"),            # scenario assumption
        postings_per_token=Q_(0.8, "byte"),        # scenario assumption
        edges_per_file=4,                          # scenario assumption
        edge_bytes=Q_(32, "byte"),                 # scenario assumption
        passage_tokens=100,                        # scenario assumption
        dimension=768,                             # scenario assumption
        bytes_per_value=_BYTES_FP32,
        links_per_node=AnnIndexesTenMillionVectors.links_per_node,
        link_bytes=AnnIndexesTenMillionVectors.link_bytes,
        edited_files=15,                           # scenario assumption
        lexical_rate=1_250,                        # scenario assumption: files per second
        parse_rate=680,                            # scenario assumption: files per second
        embed_rate=50,                             # scenario assumption: chunks per second
    )

    # --- Physical demonstration data (added 2026-10-20) ---
    PermissionTickTelemetryHour = _st.CorrelatedLogScenario(
        name="One hour of telemetry logged at the warehouse manipulator's permission tick",
        description="A slowly varying statistic with lag-one correlation 0.999: records against "
                    "effective independent samples.",
        rate=_Embodied.MobileManipulator.WarehouseMobileManipulator.permission_rate,
        duration=Q_(1, "hour"),     # scenario assumption
        lag1_correlation=0.999,     # scenario assumption: slowly varying statistic
    )
    ImitationBoundsFiveSecondEpisode = _st.ImitationBoundScenario(
        name="Imitation cost bounds over a 5 s episode at 100 Hz",
        description="Open loop, behavioral cloning (S^2 eps), and aggregation (u S eps, u = 1) at a "
                    "per-step error of 0.001.",
        episode=Q_(5, "second"),        # scenario assumption
        control_rate=Q_(100, "Hz"),     # scenario assumption
        per_step_error=1.0e-3,          # scenario assumption
        recoverability=1.0,             # scenario assumption: a recoverable task
    )
    LatchPairingLag = _st.PairingLagScenario(
        name="An uncompensated 35 ms observation-action lag at the cage-door latch",
        description="Pairing error at a transit speed and the two latch approaches against a 2 mm "
                    "clearance; full-frame rolling-shutter skew beside it.",
        site=_Embodied.Scenario.WarehouseAisle,
        lag=Q_(35, "ms"),               # scenario assumption: exposure-to-arrival lag
        clearance=Q_(2, "mm"),          # scenario assumption: latch clearance
        transit_speed=Q_(0.50, "m/s"),  # scenario assumption: generic arm transit speed
        shutter_skew_low=Q_(15, "ms"),  # scenario assumption: fast full-frame readout
        shutter_skew_high=Q_(33, "ms"), # scenario assumption: one 30 Hz frame
    )
    LatchPairingLagTightPlate = LatchPairingLag.variant(
        name="The same lag against a tighter 1 mm strike-plate clearance",
        clearance=Q_(1, "mm"),          # scenario assumption
    )
    TeleopCorrectionLoop = _st.TeleopLoopScenario(
        name="One teleoperated correction in an illustrative reference setup",
        description="Display, operator reaction, command transport, and mechanical response.",
        display=Q_(50, "ms"),       # scenario assumption: exposure, decode, raster refresh
        human=Q_(200, "ms"),        # scenario assumption: perception, planning, neuromuscular
        command=Q_(10, "ms"),       # scenario assumption: quantization, packetization, bus
        mechanical=Q_(80, "ms"),    # scenario assumption: torque ramp and structural inertia
    )
    WarehouseTeleopIngestion = _st.TeleopIngestionScenario(
        name="Raw telemetry of one teleoperated session on the warehouse mobile manipulator",
        description="Navigation and wrist cameras, two external RGB-D cameras, and the kinematic record "
                    "at the permission tick, against SATA and Gen4 NVMe write bandwidth.",
        machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
        external_camera=_Sensors.Camera.Intel_RealSense_D435i,
        external_cameras=2,         # scenario assumption
        rgb_bytes_per_pixel=3,      # scenario assumption: 24-bit RGB after debayering
        depth_bytes_per_pixel=2,    # scenario assumption: uint16 depth aligned to color
        values_per_axis=6,          # scenario assumption: position, velocity, torque, and three action taps
        bytes_per_value=4,          # scenario assumption: float32
        timestamp_bytes=8,          # scenario assumption: one 64-bit timestamp per record
        shift=Q_(8, "hour"),        # scenario assumption
        compression_ratio=8.0,      # scenario assumption: intra-frame video codec
        slow_drive_bandwidth=_Storage.SataSsd.bandwidth,
        fast_drive_bandwidth=_Hardware.Tech.Storage.NvmeGen4.bandwidth,
    )
    LatchDemonstrationCampaign = _st.DemonstrationCampaignScenario(
        name="Door-opening demonstrations on the cage-door latch, an illustrative campaign",
        description="One scheduled hour of attempt-and-reset cycles, the campaign that keeps 5,000 "
                    "demonstrations, and the wear on the wrist gear and flex cable.",
        session=Q_(60, "minute"),       # scenario assumption: declared session length
        startup=Q_(8, "minute"),        # scenario assumption: cell startup and bus sync
        calibration=Q_(7, "minute"),    # scenario assumption: sensor calibration and nulling
        rest=Q_(10, "minute"),          # scenario assumption: operator rest
        attempt=Q_(25, "second"),       # scenario assumption
        reset=Q_(15, "second"),         # scenario assumption: re-latch the door
        failure_rate=0.20,              # scenario assumption
        qa_rate=0.15,                   # scenario assumption
        target_kept=5000,               # scenario assumption: campaign target
        gear_reversals_per_attempt=14,  # scenario assumption
        cable_flexes_opening=6,         # scenario assumption
        cable_flexes_return=2,          # scenario assumption: return stroke during the re-latch
        gear_rated_cycles=5_000_000,    # scenario assumption: rated stress cycles
        cable_rated_cycles=200_000,     # scenario assumption: rated bending cycles
        repeat_campaigns=3,             # scenario assumption: campaigns compared against the cable's life
    )
    WarehouseManipulatorStateGrid = _st.StateGridScenario(
        name="Brute-force coverage of a 28-dimensional manipulation state grid",
        description="Cells at 10 and 100 bins per axis against a 1,000-robot fleet logging at 1 kHz "
                    "for the age of the universe.",
        state_components={            # scenario assumption: the state a contact task conditions on
            "joint angles": 7, "joint velocities": 7, "wheel states": 2,
            "wrist wrench": 6, "workpiece pose": 6,
        },
        bins=10,                        # scenario assumption
        fine_bins=100,                  # scenario assumption: fine enough to guide contact
        fleet=1000,                     # scenario assumption
        log_rate=Q_(1000, "Hz"),        # scenario assumption
        duration=_AGE_OF_UNIVERSE,
    )
    ContactAngleCoverage = _st.ContactGridScenario(
        name="Two archives of one million records on a contact-angle and normal-force grid",
        description="A cautious teleoperator and an exploratory policy against 1 degree by 1 N bins.",
        angle_min=Q_(-15, "degree"),    # scenario assumption
        angle_max=Q_(15, "degree"),     # scenario assumption
        angle_bin=Q_(1, "degree"),      # scenario assumption
        force_max=Q_(20, "N"),          # scenario assumption
        force_bin=Q_(1, "N"),           # scenario assumption
        records=1_000_000,              # scenario assumption
        record_rate=Q_(100, "Hz"),      # scenario assumption
        archives={                      # scenario assumption: occupied ranges and stipulated errors
            "cautious": (Q_(-2, "degree"), Q_(2, "degree"), Q_(4, "N"), Q_(6, "N"), Q_(0.001, "rad")),
            "exploratory": (Q_(-15, "degree"), Q_(15, "degree"), Q_(1, "N"), Q_(18, "N"),
                            Q_(0.012, "rad")),
        },
    )
    LatchScenarioTable = _st.ScenarioTableCoverageScenario(
        name="Cage-door latch archive against plate offset and approach speed",
        description="Three plate offsets by the two latch approaches; the archive holds the nominal "
                    "plate at the guarded speed.",
        offsets=[Q_(0, "mm"), Q_(3, "mm"), Q_(-3, "mm")],   # scenario assumption: +/-3 mm plate offset
        speeds=[_Embodied.Scenario.WarehouseAisle.v_latch_approach,
                _Embodied.Scenario.WarehouseAisle.v_latch_high],
        logged_offset=Q_(0.5, "mm"),    # scenario assumption: offset band the archive logged
        demonstrated_speed=_Embodied.Scenario.WarehouseAisle.v_latch_approach,
    )
    LatchScenarioTableAfterRecovery = LatchScenarioTable.variant(
        name="The latch table after a recovery collection at +3 mm and the guarded speed",
        added_offsets=[Q_(3, "mm")],    # scenario assumption: the stipulated latch-recovery-02
    )
    GripperStrokeTransfer = _st.GripperTransferScenario(
        name="A normalized gripper command moved from a 100 mm stroke to a 60 mm stroke",
        description="The same 0.80 command on both grippers against a 75 mm part; the donor's hold "
                    "force against the target's continuous rating.",
        command=0.80,                   # scenario assumption
        donor_stroke=Q_(100, "mm"),     # scenario assumption
        target_stroke=Q_(60, "mm"),     # scenario assumption
        part_width=Q_(75, "mm"),        # scenario assumption
        donor_hold_force=Q_(30, "N"),   # scenario assumption
        target_force_rating=Q_(15, "N"),  # scenario assumption
    )

    Llama3_70BAgentContextAdmissionDgxH100 = Llama3_70BLongContextAdmissionDgxH100.variant(
        name="32,768-token agent requests on the 70B GQA replica (8 H100s)",
        description="The repair request's 32,768 retained tokens per request, same replica and reserve.",
        context_tokens=RepairRequestZones.total_tokens,
    )
    Llama3_70BShortContextAdmissionDgxH100 = Llama3_70BLongContextAdmissionDgxH100.variant(
        name="8,192-token requests on the 70B GQA replica (8 H100s)",
        description="A short session of 8,192 retained tokens per request, same replica and reserve.",
        context_tokens=8_192,  # scenario assumption: short-session context
    )
    KvBytesPerTokenByAttentionDesign = _st.KvAttentionDesignScenario(
        name="KV bytes per token across multi-head, grouped-query, and latent attention",
        description="Six registry geometries at 16 bits; GPT-3 (MHA), Llama 3 70B (GQA), and "
                    "DeepSeek-V3 (MLA) as references, with one 131,072-token session.",
        models=[_Models.Language.GPT3, _Models.Language.Llama3_405B, _Models.Language.Llama3_70B,
                _Models.Language.Llama2_70B, _Models.Language.Mixtral_8x7B, _Models.Language.DeepSeek_V3],
        mha=_Models.Language.GPT3,
        gqa=_Models.Language.Llama3_70B,
        mla=_Models.Language.DeepSeek_V3,
        session_tokens=Llama3_70BLongContextAdmissionDgxH100.context_tokens,
    )
    CappedRequestsPagedAllocation = _st.PagedAllocationScenario(
        name="Requests capped at 32,768 tokens with a mean final length of 2,048: reserved or paged",
        description="Full-cap contiguous reservation against 16- and 64-token blocks in the same pool; "
                    "final lengths capped exponential (a modeling assumption).",
        pool=Llama3_70BAgentContextAdmissionDgxH100,
        cap=32_768,  # scenario assumption: deployment cap, not a model specification
        mean_final=2_048,  # scenario assumption: hypothetical mean final length, not a trace mean
        small_block=16,  # scenario assumption: allocator block sizes
        large_block=64,
    )
    ForkedCandidatesPrefixCopy = _st.KvForkCopyScenario(
        name="Four candidate repairs forked from an 8,192-token prefix without block sharing",
        description="Each candidate copies the prefix's KV state under contiguous allocation.",
        pool=Llama3_70BAgentContextAdmissionDgxH100,
        prefix_tokens=8_192,  # scenario assumption: processed prefix shared by the candidates
        branches=4,  # scenario assumption: sampled candidate repairs
    )
    InvariantPrefixRebuildDgxH100 = _st.PrefixRebuildScenario(
        name="A 4,096-token invariant prefix rebuilt on a miss, against a 32-token suffix pass on a hit",
        description="Llama 3 70B geometry on one DGX H100 node; sustained prefill at 90 percent of the "
                    "high-MFU figure (45 percent of 16-bit peak).",
        workload=_Models.Language.Llama3_70B,
        node=_Nodes.DGX_H100,
        precision="fp16",
        prefix_tokens=4_096,  # scenario assumption: invariant instructions and tool schemas
        suffix_tokens=32,  # scenario assumption: short novel suffix after a hit
        utilization=_Literature.Training.MfuHigh * 0.9,  # scenario assumption: sustained prefill share
    )
    ChunkedPrefillAmongDecodes = _st.ChunkedPrefillScenario(
        name="A 4,096-token uncached prompt among 32 decodes, whole or in 512-token chunks",
        description="Weight-only step time on the DGX H100 node at the sustained prefill rate.",
        prefill=InvariantPrefixRebuildDgxH100,
        decodes=32,  # scenario assumption: ongoing decode requests
        prompt_tokens=4_096,  # scenario assumption: new uncached prefill
        chunk_tokens=512,  # scenario assumption: prefill chunk per iteration
    )
    RepairToolWaitPausedState = _st.PausedKvStateScenario(
        name="Repair request paused for a 45 s integration test: keep, move, or drop its KV state",
        description="The 32,768-token request in 16-token blocks, moved over the eight PCIe host links "
                    "in parallel; 32 such requests returning together.",
        pool=Llama3_70BAgentContextAdmissionDgxH100,
        prefill=InvariantPrefixRebuildDgxH100,
        context_tokens=RepairRequestZones.total_tokens,
        block_tokens=16,  # scenario assumption: allocation granularity
        wait=Q_("45 s"),  # scenario assumption: stipulated integration-test wait
        link_fraction=0.25,  # scenario assumption: links sustain a quarter of rated bandwidth
        returns=32,  # scenario assumption: paused repairs whose tests finish together
    )
    RepairBlockSharingResidency = _st.KvBlockSharingScenario(
        name="Repair requests in 16-token blocks: unshared, sharing a workspace prefix, kept or released",
        description="32,768 input and 2,048 output tokens per request, a 24,576-token shareable "
                    "prefix, 10 percent headroom, 5 s of generation and a 45 s test per turn.",
        pool=Llama3_70BAgentContextAdmissionDgxH100,
        headroom=0.10,  # scenario assumption: fraction of the pool withheld
        context=RepairRequestZones.total_tokens,
        output=2_048,  # scenario assumption: maximum private output tokens
        common=24_576,  # scenario assumption: identical, compatible, shareable prefix tokens
        block_tokens=RepairToolWaitPausedState.block_tokens,
        generation=Q_("5 s"),  # scenario assumption: residence while generating
        wait=RepairToolWaitPausedState.wait,
    )
    RepairAgentResidency = _st.KvResidencyArchetypeScenario(
        name="Repair agent: sharing only its system prompt and tool schemas, 5 s generation, 45 s test",
        description="The 32,768-token repair context with the support profile's 6,144-token shared "
                    "prefix (taken to be the same length).",
        paused=RepairToolWaitPausedState,
        context_tokens=RepairRequestZones.total_tokens,
        shared_prefix=int(_Agents.Support.CustomerOps.shared_prefix.to("token").magnitude),
        generation=RepairBlockSharingResidency.generation,
        wait=RepairBlockSharingResidency.wait,
    )
    SupportAgentResidency = _st.KvResidencyArchetypeScenario(
        name="Customer-support agent: shared policy prefix, short generation, customer reply wait",
        description="The CustomerOps support profile's session context, shared prefix, generation "
                    "residence, and customer reply time.",
        paused=RepairToolWaitPausedState,
        context_tokens=int(_Agents.Support.CustomerOps.session_context.to("token").magnitude),
        shared_prefix=int(_Agents.Support.CustomerOps.shared_prefix.to("token").magnitude),
        generation=_Agents.Support.CustomerOps.generation_residence,
        wait=_Agents.Support.CustomerOps.customer_reply_time,
    )

    # --- The real-time permission path of the warehouse mobile manipulator (added 2026-10-20) ---
    # The machine and its site record are Embodied.MobileManipulator.WarehouseMobileManipulator
    # and Embodied.Scenario.WarehouseAisle (tick, bus cycle, lease, brake onset, the
    # permission check's C, B, I, D, and the event intervals). Inputs neither record
    # carries are stated here.
    WarehouseJointLoopThroughTick = _st.PhaseLagBudgetScenario(
        name="A 20 Hz joint loop closed through the 1 kHz permission tick and the fieldbus",
        description="Sample-and-hold, the declared permission-check WCET, and one bus cycle, "
                    "priced in phase at crossover against a 5 degree erosion allowance.",
        crossover=Q_(20.0, "Hz"),  # scenario assumption: illustrative joint-loop crossover
        sample_period=_Embodied.Scenario.WarehouseAisle.t_tick,
        compute_delay=_Embodied.Scenario.WarehouseAisle.enforcer_wcet,
        transport_delay=_Embodied.Scenario.WarehouseAisle.t_bus,
        phase_allowance=Q_(5.0, "degree"),  # scenario assumption: chosen phase erosion allowance
    )
    WarehouseJointLoopNextTickOutput = WarehouseJointLoopThroughTick.variant(
        name="The same joint loop with its output released at the next tick",
        description="The compute delay grows from the declared WCET to one whole tick.",
        compute_delay=_Embodied.Scenario.WarehouseAisle.t_tick,
    )
    WarehouseJointLoopSlowTick = WarehouseJointLoopThroughTick.variant(
        name="The same joint loop sampled by a 500 Hz tick",
        description="A 2 ms sample period, for the sample-and-hold term alone.",
        sample_period=Q_(2.0, "ms"),  # scenario assumption: 500 Hz comparison tick
    )
    WarehousePermissionTickResponse = _st.ResponseTimeScenario(
        name="The warehouse machine's permission check: response-time bound against its deadline",
        description="Declared WCET, blocking, and interference of the 1 kHz permission tick "
                    "against its 400 microsecond deadline, all from the site record.",
        site=_Embodied.Scenario.WarehouseAisle,
    )
    WarehousePermissionTickLongBlocking = WarehousePermissionTickResponse.variant(
        name="The same permission check behind a longer tick-masking section",
        description="A new link driver masks the tick for 120 microseconds instead of 50.",
        blocking=Q_(120.0, "us"),  # scenario assumption: a longer tick-masking section
    )
    WarehouseServoCanFdOneBus = _st.CanFdBusScenario(
        name="All nine servo axes of the warehouse machine on one CAN FD bus",
        description="An 8-byte command and an 8-byte feedback frame per axis per 1 ms cycle, "
                    "1 Mbps nominal and 5 Mbps data phase, worst-case stuffing.",
        num_axes=_Embodied.MobileManipulator.WarehouseMobileManipulator.servo_axes,
        payload_bytes=8,  # scenario assumption: position-mode record, each direction
        frames_per_axis=2,  # scenario assumption: one command and one feedback frame per axis
        baud_arbitration=Q_(1.0, "megabit/second"),  # scenario assumption: nominal rate
        baud_data=Q_(5.0, "megabit/second"),  # scenario assumption: data-phase rate
        cycle=_Embodied.MobileManipulator.WarehouseMobileManipulator.fieldbus_cycle,
    )
    WarehouseServoCanFdArmBus = WarehouseServoCanFdOneBus.variant(
        name="The warehouse machine's seven arm axes on their own CAN FD bus",
        description="One bus per limb: the arm joints only.",
        num_axes=_Embodied.MobileManipulator.WarehouseMobileManipulator.arm.dofs,
    )
    WarehouseServoCanFdBaseBus = WarehouseServoCanFdOneBus.variant(
        name="The warehouse machine's two drive wheels on their own CAN FD bus",
        description="One bus per limb: the drive-wheel axes only.",
        num_axes=(_Embodied.MobileManipulator.WarehouseMobileManipulator.servo_axes
                  - _Embodied.MobileManipulator.WarehouseMobileManipulator.arm.dofs),
    )
    WarehouseServoEtherCat = _st.EtherCatCycleScenario(
        name="All nine servo axes of the warehouse machine on one EtherCAT chain",
        description="One summation frame carries every axis's 8-byte command and 8-byte feedback; "
                    "each drive forwards it with a fixed hop delay.",
        num_axes=_Embodied.MobileManipulator.WarehouseMobileManipulator.servo_axes,
        bytes_per_axis=2 * 8,  # scenario assumption: command and feedback records share the frame
        baud_rate=Q_(100.0, "megabit/second"),  # 100BASE-TX link rate
        header_bytes=64,  # scenario assumption: preamble, Ethernet and EtherCAT headers, FCS
        hop_delay=Q_(0.6, "us"),  # scenario assumption: per-node forwarding delay
        cycle=_Embodied.MobileManipulator.WarehouseMobileManipulator.fieldbus_cycle,
    )
    WarehouseChunkLease = _st.LeaseBudgetScenario(
        name="The warehouse machine's 60 ms chunk lease at its drive limit",
        description="The lease path priced in the stopping budget, and the lease rule's floor "
                    "(renewal period plus jitter) and ceilings (pre-brake ceiling less tick, bus, "
                    "onset, and the evidence's age).",
        site=_Embodied.Scenario.WarehouseAisle,
        speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
        renewal_jitter=Q_(5.0, "ms"),  # scenario assumption: allowance for renewal variation
    )
    WarehouseChunkLeaseLengthened = WarehouseChunkLease.variant(
        name="The same chunk lease lengthened to 100 ms",
        description="A longer lease to ride out proposer hiccups, priced in clear distance.",
        lease=Q_(100.0, "ms"),  # scenario assumption: a lengthened lease
    )

    # --- Tool calling (added 2026-10-20) -------------------------------------
    # One mediated call, its retry, its observation, and its catalog. Loss
    # rates, sizes, counts, and prices are teaching assumptions; the window is
    # the repair harness's, and the input price and cached share are the ones
    # the context-layout bill uses, so the volume prices tokens one way.
    BlindRetryTwoPercentLoss = _st.BlindRetryScenario(
        name="One blind retry per timeout at 2 percent request and reply loss, 200,000 calls a day",
        description="A non-idempotent call with no deduplication; lost messages never arrive later; "
                    "a fleet of 50 workers making 4,000 mutating calls a day each.",
        request_loss=0.02,  # scenario assumption: hypothetical, not a measured rate
        reply_loss=0.02,  # scenario assumption: hypothetical, not a measured rate
        workers=50,  # scenario assumption: fleet size
        calls_per_worker=4000,  # scenario assumption: mutating calls per worker per day
    )

    TwentyFourMegabyteLogObservation = _st.ObservationCarryScenario(
        name="A 24 MB test log against a 4,000-token shaped view carried by 30 later calls",
        description="An assumed 3 bytes per token; the repair harness's window; uncached input at a "
                    "hypothetical $3 per million tokens, cached input at 10 percent of that.",
        raw_bytes=Q_(24, "MB"),  # scenario assumption: raw log size
        bytes_per_token=Q_(3, "byte / token"),  # scenario assumption: sizing ratio, not a tokenizer law
        window=RepairContextBudget.window,
        shaped_tokens=4000,  # scenario assumption: staged observation
        carry_calls=30,  # scenario assumption: later calls that carry it
        input_price=RepairPrefixLayoutBill.input_price,
        cached_fraction=RepairPrefixLayoutBill.cached_fraction,
    )

    ModelPollingEveryFiveSeconds = _st.ModelPollingScenario(
        name="A model polling a 300 s test job every 5 s",
        description="32,768 input and 40 output tokens per status call, 95 percent of input cached; "
                    "hypothetical $3 uncached, cached at 10 percent, and $15 output per million tokens.",
        job_duration=Q_(300, "second"),  # scenario assumption: fixed completion time
        poll_interval=Q_(5, "second"),  # scenario assumption: checks start one interval after dispatch
        context_tokens=32_768,  # scenario assumption: selected input per status call
        output_tokens=40,  # scenario assumption: generated status request
        cache_hit=0.95,  # scenario assumption: input share billed as cached
        input_price=RepairPrefixLayoutBill.input_price,
        cached_fraction=RepairPrefixLayoutBill.cached_fraction,
        output_price=ProbeVsGeneratedExplanation.output_price,
    )

    FiftyToolCatalogSixSelected = _st.ToolCatalogSelectionScenario(
        name="50 registered tools of 300 tokens, 6 staged with 400 tokens of selection overhead",
        description="Stipulated counts in final serialization; selection needs no extra model call.",
        registered=50,  # scenario assumption
        selected=6,  # scenario assumption
        descriptor_tokens=300,  # scenario assumption: per descriptor
        selection_overhead=400,  # scenario assumption: selection interface and metadata
    )

    ShellAgainstTwelveStructuredTools = _st.ToolkitGranularityScenario(
        name="One 500-token shell tool against 12 structured tools of 350 tokens",
        description="30 calls per trajectory at the volume's uncached input price; structured attempt "
                    "$1.20 accepted at 0.80 with a 0.001 chance of an unconfined change; shell accepted "
                    "at 0.78 with 0.004; each unconfined change costs $2,000 to detect and repair.",
        tools=12,  # scenario assumption
        tool_tokens=350,  # scenario assumption: per structured descriptor
        shell_tokens=500,  # scenario assumption
        calls=30,  # scenario assumption: model calls per trajectory
        input_price=RepairPrefixLayoutBill.input_price,
        structured_attempt=Q_(1.20, "dollar"),  # scenario assumption
        structured_acceptance=0.80,  # scenario assumption
        structured_change=0.001,  # scenario assumption
        shell_acceptance=0.78,  # scenario assumption
        shell_change=0.004,  # scenario assumption
        change_cost=Q_(2000, "dollar"),  # scenario assumption: detect and repair one change
    )


    # --- Agent sandboxes (added 2026-10-20) ---
    SyscallTaxPerMillionCalls = _st.SyscallTaxScenario(
        name="Added time of a million ordinary system calls in a container, microVM, and user-space kernel",
        description="Illustrative per-call terms: a 0.1 us trap, 0.05 us in-kernel filter, 1.5 us "
                    "emulation, and a 2 us switch to the emulator; device exits not counted.",
        calls=1_000_000,  # scenario assumption: calls in a build or test run
        trap=Q_(0.1, "us"),  # scenario assumption: trap into a kernel
        filter=Q_(0.05, "us"),  # scenario assumption: in-kernel filter
        emulate=Q_(1.5, "us"),  # scenario assumption: user-space kernel emulation
        switch=Q_(2.0, "us"),  # scenario assumption: switch to the emulator
    )
    ParserCallsEnvelopeSetup = _st.EnvelopeAmortizationScenario(
        name="100 parser calls of 3 ms each, envelope setup per call or once per trajectory",
        description="Setup is the SWE-bench coding harness's snapshot restore; a 10 percent cap "
                    "on setup's share of a fresh call.",
        calls=100,  # scenario assumption: sequential parser calls in one trajectory
        setup=_Agents.Coding.SWE_Bench_Runner.sandbox_snapshot_restore_latency.to("ms"),
        work_per_call=Q_(3, "ms"),  # scenario assumption: useful work per call
        overhead_cap=0.10,  # scenario assumption: setup at most this share of a fresh call
    )
    CopyOnWriteRepairWorkspaces = _st.WorkspaceProvisioningScenario(
        name="32 repair workspaces on a 2.4 GB base, by full copy or copy-on-write",
        description="Each task writes 40 MB; one PCIe Gen4 NVMe drive; a 1.2 GB shared fixture "
                    "changed by one byte under file or 64 KB block copying.",
        workspaces=32,  # scenario assumption: concurrent workspaces for one task
        base_image=Q_(2.4, "GB"),  # scenario assumption: repository, toolchain, dependencies
        written=Q_(40, "MB"),  # scenario assumption: edits and build outputs per workspace
        bandwidth=_Hardware.Tech.Storage.NvmeGen4.bandwidth,
        fixture=Q_(1.2, "GB"),  # scenario assumption: shared binary test fixture
        block=Q_(64, "KB"),  # scenario assumption: block-granular copying unit
        modified=Q_(1, "byte"),  # scenario assumption: bytes each task overwrites
    )
    CodingSandboxPoolTwoPerSecond = _st.SandboxPoolReserveScenario(
        name="Coding sandbox pool at 2 trajectories per second, 300 s leases, snapshot restore",
        description="The SWE-bench coding harness's restore time and guest image; 64 MB written "
                    "per lease, 8 MB per standby, a 0.1 percent request-level miss target.",
        arrival_rate=Q_(2.0, "1/s"),  # scenario assumption: offered trajectories per second
        hold=Q_(300, "s"),  # scenario assumption: lease held for the whole trajectory
        restore=_Agents.Coding.SWE_Bench_Runner.sandbox_snapshot_restore_latency,
        guest_memory=_Agents.Coding.SWE_Bench_Runner.sandbox_memory_footprint,
        dirty=Q_(64, "MB"),  # scenario assumption: private pages written per active lease
        standby=Q_(8, "MB"),  # scenario assumption: private state per standby instance
        target_miss=1e-3,  # scenario assumption: design target for finding standby empty
        initial_reserve=1,  # scenario assumption: candidate evaluated first
        slow_factor=10,  # scenario assumption: a cold boot an order of magnitude slower
    )

    # --- On-device learning and federated rounds (added 2026-10-20) ---------
    # A phone's full update, parameter-efficient updates, personalization
    # storage, the uplink, the battery, and the mechanics of one federated round.
    PhoneFullUpdateWorkingSet = _st.UpdateWorkingSetScenario(
        name="Full fine-tune of a 10M-parameter FP32 model with Adam on a flagship phone",
        description="Weights, gradients, two Adam moments, and saved activations at 5x the weight bytes; "
                    "DRAM traffic of one step with one crossing per tensor use, at the A17 Pro's bandwidth.",
        parameters=10 * 10 ** 6,          # scenario assumption: compact on-device model
        bytes_per_param=_BYTES_FP32,
        optimizer_moments=2,              # Adam keeps first and second moments
        activation_factor=5,              # scenario assumption: saved activations = 5x weight bytes
        activation_factor_high=8,         # scenario assumption: upper end of the quoted range
        bandwidth=_Hardware.Mobile.iPhone15Pro.memory.bandwidth,
    )
    PhoneBackgroundFineTuneBattery = _st.BatteryBudgetScenario(
        name="A 30-minute unthrottled fine-tune at 4.5 W against a flagship phone battery",
        description="A background voice-assistant adaptation job drawing above the sustained envelope.",
        battery_energy=_ReferenceStats.PhoneBattery.EnergyWh,
        power_draw=Q_(4.5, "watt"),       # scenario assumption: unthrottled SoC draw during training
        required_runtime=Q_(30, "minute"),  # scenario assumption: one adaptation job
    )
    TransformerLayerRank4Update = _st.LowRankUpdateScenario(
        name="Rank-4 low-rank update to a 768 by 768 transformer weight",
        description="Trainable parameters of the update against the full layer.",
        in_dim=768,                       # scenario assumption: square weight of a generic transformer layer
        out_dim=768,
        rank=4,                           # scenario assumption
    )
    CompactEncoderResidualAdapters = _st.ResidualAdapterScenario(
        name="Residual adapters of width 768 and bottleneck 16 after each of 12 blocks, FP16",
        description="A compact transformer encoder personalized with one bottleneck adapter per block.",
        width=768,                        # scenario assumption: hidden width
        bottleneck=16,                    # scenario assumption: adapter bottleneck
        sites=12,                         # scenario assumption: one adapter per block
        bytes_per_param=_BYTES_FP16,
    )
    CameraModelTenContextPersonalization = _st.PersonalizationStorageScenario(
        name="A 10M-parameter camera model personalized for 10 contexts: copies or adapters",
        description="FP32 storage of ten fine-tuned copies against one backbone plus ten 50,000-parameter adapters.",
        model_params=10 * 10 ** 6 * _param,   # scenario assumption
        adapter_params=50_000 * _param,       # scenario assumption: one adapter per context
        bytes_per_param=_BYTES_FP32 / _param,
        contexts=10,                          # scenario assumption: home, office, car, ...
    )
    PhoneImagesVersusQuantizedUpdates = _st.FederatedUplinkScenario(
        name="A week of phone photos against 4-bit updates of a 5M-parameter model",
        description="1,000 compressed images of 200 KB a week against one 2.5 MB update per round, "
                    "at a daily and an hourly round cadence.",
        items_per_week=1_000,                 # scenario assumption
        item_size=Q_(200, "kB"),              # scenario assumption: compressed JPEG
        update_params=5 * 10 ** 6,            # scenario assumption
        update_bits_per_param=4,              # scenario assumption: 4-bit quantized update
        rounds_per_week={"daily": 7, "hourly": 24 * 7},
    )
    FederatedAdapterUploadMobileUplink = _st.TransferScenario(
        name="The reference federated adapter update over a 1 to 10 Mb/s mobile uplink",
        description="Communication term of one federated round for one phone.",
        payload=_ReferenceStats.EdgeAdaptationTierProfile.FederatedLoraUpdate,
        links={"low": Q_(1, "megabit/second"),    # scenario assumption: constrained uplink
               "high": Q_(10, "megabit/second")},  # scenario assumption: upper end
    )

    # --- Policy training: discounting, decoders, trial campaigns, contact exploits (added 2026-10-20) ---
    LatchOpeningDiscount = _st.DiscountHorizonScenario(
        name="The discounted return over one latch opening at the warehouse manipulator's setpoint step",
        description="Decisions per 8 s opening, the door-open reward's weight at the first decision, the "
                    "effective horizon of a 0.99 discount, and the discount that keeps half the weight.",
        step=_Embodied.MobileManipulator.WarehouseMobileManipulator.chunk_step,
        rollout=Q_(8.0, "second"),      # scenario assumption: autonomous contact phase of one opening
        discount=0.99,                  # scenario assumption: a common discount factor
        retained_weight=0.5,            # scenario assumption: keep half the terminal weight
    )
    LatchDecoderBudget = _st.IterativeDecoderBudgetScenario(
        name="A 16-step diffusion head and a 4-step flow head against the warehouse manipulator's ACT decoder",
        description="Per-evaluation cost, renewal slack, and the overrun under a jitter-sized stretch, "
                    "on the 25 ms setpoint step and the 50 ms renewal period.",
        machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
        denoising_steps=16,                 # scenario assumption: DDIM denoising steps
        flow_steps=4,                       # scenario assumption: flow-matching evaluations
        encode=Q_(8.0, "ms"),               # scenario assumption: visual encoding, once per proposal
        denoising_latency=Q_(48.0, "ms"),   # scenario assumption: stipulated total DDIM inference
        renewal_jitter=Q_(5.0, "ms"),       # scenario assumption: the lease rule's renewal-jitter allowance
    )
    LeggedGymQuadrupedRun = _st.ParallelSimRunScenario(
        name="4,096 simulated ANYmal quadrupeds trained with PPO on one GPU (Rudin et al. 2022)",
        description="Batch, total policy steps, lower-bound throughput, and simulated days of walking.",
        robots=4096,  # Rudin, N., Hoeller, D., Reist, P., and Hutter, M. (2022), Learning ...
        steps_per_update=24,  # Rudin, N., Hoeller, D., Reist, P., and Hutter, M. (2022), Learning ...
        updates=1500,  # Rudin, N., Hoeller, D., Reist, P., and Hutter, M. (2022), Learning ...
        training_time=Q_(20.0, "minute"),  # Rudin, N., Hoeller, D., Reist, P., and Hutter, M. (2022), Learning ...
        policy_rate=Q_(50.0, "Hz"),  # Rudin, N., Hoeller, D., Reist, P., and Hutter, M. (2022), Learning ...
    )
    LatchTrialCampaign = _st.TrialCampaignScenario(
        name="A from-scratch reinforcement learning campaign of one million latch openings",
        description="Execution, automated-reset, manual-reset, and 14-rig calendars; drivetrain load "
                    "reversals; the trials a 2.5 h seeded campaign allows; and the same openings in a "
                    "GPU-parallel simulator at a published throughput.",
        trials=1_000_000,                   # scenario assumption: order of magnitude for learning from scratch
        rollout=LatchOpeningDiscount.rollout,
        decision_step=LatchOpeningDiscount.step,
        automated_reset=Q_(4.0, "second"),  # scenario assumption: dedicated RL re-latch fixture
        manual_reset=Q_(30.0, "second"),    # scenario assumption: manual operator reset
        farm_rigs=14,  # Levine, S., Pastor, P., Krizhevsky, A., Ibarz, J., and Quillen, D. ...
        reversals_per_trial=4,              # scenario assumption: joint load reversals per opening
        seeded_budget=Q_(2.5, "hour"),  # Luo, J., Xu, C., Wu, J., and Levine, S. (2025), Precise and Dextero...
        simulator=LeggedGymQuadrupedRun,
    )
    LatchContactDeadlines = _st.ContactDeadlineScenario(
        name="Constant-speed drives into the cage-door latch plate at the guarded and fast approaches",
        description="Contact yield deadline, tripwire crossing, brake onset, and the time left before "
                    "the latch limit.",
        site=_Embodied.Scenario.WarehouseAisle,
    )
    LatchSoftSpringExploit = _st.SoftContactExploitScenario(
        name="A soft simulated strike plate against the real plate under an unbraked fast approach",
        description="Penalty-spring and rigid-arm peak forces and the arm stiffness at which the "
                    "series chain reaches the latch limit.",
        machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
        site=_Embodied.Scenario.WarehouseAisle,
        penalty_penetration=Q_(2.4, "mm"),  # scenario assumption: penetration at the fast approach
    )
    LinearDriveRealityGap = _st.ForceRiseGapScenario(
        name="An illustrative linear drive regulating 10 N against a surface, in simulation and on hardware",
        description="One stated discrepancy per reality-gap channel: dynamics, timing, sensing, appearance.",
        approach_speed=Q_(0.20, "m/s"),         # scenario assumption
        target_force=Q_(10.0, "N"),             # scenario assumption
        simulated_rise_rate=Q_(500.0, "N/s"),   # scenario assumption
        physical_rise_rate=Q_(250.0, "N/s"),    # scenario assumption
        p99_delay=Q_(49.5, "ms"),               # scenario assumption: p99 end-to-end actuation delay
        simulated_delay=Q_(0.0, "ms"),          # scenario assumption: the simulator applies commands at once
        simulator_step=Q_(20.0, "ms"),          # scenario assumption: integration period
        drive_delay=Q_(8.0, "ms"),              # scenario assumption: unmodeled drive and bus delay
        overtravel=Q_(3.0, "mm"),               # scenario assumption: overtravel into the current limit
        range_bias=Q_(3.2, "mm"),               # scenario assumption: range-sensor bias
        dropout_frames=4,                       # scenario assumption: specular dropout burst
        pixel_difference=0.055,                 # scenario assumption: fraction of the image
        patch_shift=Q_(6.0, "mm"),              # scenario assumption: contact-patch estimate shift
        chamfer_tolerance=Q_(2.0, "mm"),        # scenario assumption
    )
    LatchResidualFineTune = _st.OnRobotFineTuneScenario(
        name="On-robot fine-tuning of a gated residual at the cage-door latch",
        description="Active contact time and above-tripwire samples of 20,000 interaction steps.",
        step=_Embodied.MobileManipulator.WarehouseMobileManipulator.chunk_step,
        interaction_steps=20_000,           # scenario assumption
        exceed_fraction=0.015,              # scenario assumption: exploratory steps past the tripwire
        plate_offset=Q_(1.5, "mm"),         # scenario assumption: real plate against its simulated placement
    )

    # --- Network architectures: dense, convolutional, and recurrent layers (added 2026-10-20) ---
    MnistMlpFirstLayer = _st.DenseLayerScenario(
        name="The first dense layer of the 784-128-64-10 MNIST multilayer perceptron",
        description="Flattened 28x28 digits to 128 hidden units; FP32 weights at batch one.",
        in_features=_Datasets.MNIST.image_width * _Datasets.MNIST.image_height,
        out_features=128,                 # scenario assumption: the running network's first hidden width
        bytes_per_weight=_BYTES_FP32,
    )
    MnistMlpLayerStack = _st.LayerStackScenario(
        name="The 784-128-64-10 MNIST multilayer perceptron, weights",
        description="Hidden widths 128 and 64, ten classes; FP32 weights, biases omitted.",
        dense_layers=[MnistMlpFirstLayer,
                      MnistMlpFirstLayer.variant(name="128 to 64", in_features=MnistMlpFirstLayer.out_features,
                                                 out_features=64),  # scenario assumption: second hidden width
                      MnistMlpFirstLayer.variant(name="64 to 10", in_features=64,
                                                 out_features=_Datasets.MNIST.num_classes)],
    )
    Dense2048Layer = _st.DenseLayerScenario(
        name="A production-width dense layer, 2,048 to 2,048, FP32",
        description="Weights grow as the product of the widths.",
        in_features=2048,                 # scenario assumption: production-width layer
        out_features=2048,
        bytes_per_weight=_BYTES_FP32,
    )
    Dense1024LayerFp16 = _st.DenseLayerScenario(
        name="A 1,024-to-1,024 dense layer in FP16",
        description="Quadratic scaling of dense weights at a modest width.",
        in_features=1024,                 # scenario assumption: illustrative width
        out_features=1024,
        bytes_per_weight=_BYTES_FP16,
    )
    MnistFirstConvLayer = _st.ConvLayerScenario(
        name="32 feature detectors on a 28x28 MNIST digit: 3x3 filters against dense units",
        description="One grayscale channel, same padding, FP32 values.",
        height=_Datasets.MNIST.image_height,
        width=_Datasets.MNIST.image_width,
        kernel=3,                         # scenario assumption: 3x3 filters
        in_channels=1,                    # grayscale input
        out_channels=32,                  # scenario assumption: first-layer features
        bytes_per_value=_BYTES_FP32,
    )
    MnistSmallCnn = _st.LayerStackScenario(
        name="A small MNIST CNN: Conv(32), pool, Conv(64), pool, dense 128, 10 classes",
        description="Two 3x3 convolutions, each followed by a 2x2 pool (28 to 14 to 7), then a dense head.",
        conv_layers=[MnistFirstConvLayer,
                     MnistFirstConvLayer.variant(
                         name="Second convolution, 32 to 64 channels at 14x14",
                         height=_Datasets.MNIST.image_height // 2,   # after one 2x2 pool
                         width=_Datasets.MNIST.image_width // 2,
                         in_channels=MnistFirstConvLayer.out_channels,
                         out_channels=64)],                         # scenario assumption
        dense_layers=[MnistMlpFirstLayer.variant(
                          name="Head: 64 maps of 7x7 to 128",
                          in_features=64 * (_Datasets.MNIST.image_width // 4) * (_Datasets.MNIST.image_height // 4),
                          out_features=128),                        # scenario assumption: head width
                      MnistMlpFirstLayer.variant(name="Head: 128 to 10", in_features=128,
                                                 out_features=_Datasets.MNIST.num_classes)],
    )
    ImageNetRgbFilterVersusDense = _st.ConvLayerScenario(
        name="One feature detector on a 224x224 RGB image: a 3x3 filter against a dense unit",
        description="Weights for one output feature; same padding.",
        height=_Datasets.ImageNet.image_height,
        width=_Datasets.ImageNet.image_width,
        kernel=3,                         # scenario assumption: 3x3 filter
        in_channels=3,                    # RGB
        out_channels=1,                   # one feature detector
        bytes_per_value=_BYTES_FP32,
    )
    ImageNet64ChannelConv = _st.ConvLayerScenario(
        name="A 3x3 convolution on a 224x224x64 feature map to 64 features, FP32",
        description="Same padding; the reference layer for weight sharing, reuse, activations, and factoring.",
        height=_Datasets.ImageNet.image_height,
        width=_Datasets.ImageNet.image_width,
        kernel=3,                         # scenario assumption: 3x3 filter
        in_channels=64,                   # scenario assumption: early-layer channel count
        out_channels=64,
        bytes_per_value=_BYTES_FP32,
    )
    ResidualAddFp32 = _st.ElementwiseIntensityScenario(
        name="An unfused FP32 residual addition",
        description="One FLOP per element; two operands read and one result written.",
        flops_per_element=1,
        operands_moved=3,
        bytes_per_value=_BYTES_FP32,
    )
    ResNet50ConvKernelShare = _st.KernelShareScenario(
        name="ResNet-50's 3x3 convolutions as a share of its multiply-adds",
        description="Per 224x224 image, derived from the architecture of He et al. (2016), Table 1 "
                    "(original network: stride on the first 1x1 of a downsampling block).",
        workload=_Models.Vision.ResNet50,
        kernel_macs=Q_(_RESNET50_MACS["3x3"] / 1e9, "GFLOP"),   # multiply-adds, 1.850 G
        total_macs=Q_(_RESNET50_MACS["total"] / 1e9, "GFLOP"),  # multiply-adds, 3.858 G
    )
    ResNet50ThirtyFpsStream = _st.FrameRateHeadroomScenario(
        name="ResNet-50 on a 30 frame-per-second stream on a 10 TFLOP/s GPU",
        description="Planning utilization of 50 to 60 percent of peak; a 100 GFLOP detector as the heavier model.",
        workload=_Models.Vision.ResNet50,
        frame_rate=Q_(30, "Hz"),          # scenario assumption: video frame rate
        peak_rate=Q_(10, "TFLOP/s"),      # scenario assumption: mid-range GPU peak
        utilization_low=0.50,             # scenario assumption: planning utilization range
        utilization_high=0.60,
        alternative_flops=Q_(100, "GFLOP"),  # scenario assumption: object detector per frame
    )
    ElmanRnnStep = _st.RecurrentStepScenario(
        name="A plain recurrent layer: 100 inputs, a 128-wide state, a 1,000-step sequence",
        description="One affine map per step for the state and one for the input; FP32.",
        input_dim=100,                    # scenario assumption: input width
        hidden_dim=128,                   # scenario assumption: state width
        bytes_per_value=_BYTES_FP32,
        gates=1,
        steps=1000,                       # scenario assumption: sequence length
    )
    RecurrentState512Fp32 = ElmanRnnStep.variant(
        name="A 512-wide FP32 recurrent state",
        description="The fixed state inference keeps at any stream length.",
        hidden_dim=512,                   # scenario assumption: illustrative state width
    )

    # --- Neural computation: the running MNIST network and its costs (added 2026-10-20) ---
    # One dense digit classifier carries the calculations: 28 x 28 inputs,
    # hidden layers of 128 and 64 units (scenario assumption), one output per
    # MNIST class, mini-batches of 32, FP32 throughout, Adam's two FP32 values
    # per parameter. Variants change one input each.

    MnistDenseNetwork = _st.DenseNetworkScenario(
        name="MNIST digit classifier, 784-128-64-10, batch 32, FP32, Adam",
        description="A fully connected network from the 784 pixels of a 28 x 28 MNIST digit through hidden "
                    "layers of 128 and 64 units to the 10 class scores, trained in batches of 32.",
        layer_widths=[_Datasets.MNIST.image_width * _Datasets.MNIST.image_height,
                      128, 64,  # scenario assumption: hidden widths
                      _Datasets.MNIST.num_classes],
        batch_size=32,  # scenario assumption: mini-batch size
        bytes_per_value=_BYTES_FP32,
        optimizer_bytes_per_parameter=_BYTES_ADAM_MOMENTS,
        dataset=_Datasets.MNIST,
    )
    MnistDenseNetworkWiderHidden = MnistDenseNetwork.variant(
        name="MNIST digit classifier with both hidden widths doubled, 784-256-128-10",
        layer_widths=(MnistDenseNetwork.layer_widths[:1] + [2 * w for w in MnistDenseNetwork.layer_widths[1:-1]]
                      + MnistDenseNetwork.layer_widths[-1:]),
    )
    MnistDenseNetworkDoubledInputSide = MnistDenseNetwork.variant(
        name="MNIST digit classifier on 56 x 56 inputs (the image side doubled)",
        layer_widths=[(2 * _Datasets.MNIST.image_width) * (2 * _Datasets.MNIST.image_height)]
                     + MnistDenseNetwork.layer_widths[1:],
    )
    MnistDenseNetworkLargeBatch = MnistDenseNetwork.variant(
        name="MNIST digit classifier at batch 1,024",
        batch_size=1_024,  # scenario assumption: a large batch, 32 times the running one
    )
    DenseWidthDepthSweep = _st.DenseScalingSweepScenario(
        name="Dense networks on MNIST swept in hidden width and in depth",
        description="Two hidden layers of width 64 to 4,096; and 1 to 32 hidden layers of width 512; "
                    "784 inputs and 10 outputs throughout.",
        input_dim=MnistDenseNetwork.layer_widths[0],
        output_dim=MnistDenseNetwork.layer_widths[-1],
        hidden_layers=2,  # scenario assumption: matches the running network
        widths=[2 ** k for k in range(6, 13)],  # scenario assumption: 64 to 4,096
        fixed_width=512,  # scenario assumption
        max_depth=32,  # scenario assumption
    )
    MnistEpochCpuVersusH100 = _st.EpochOverheadScenario(
        name="One MNIST training epoch on a desktop CPU and on an H100, FP32",
        description="1,875 steps of the 784-128-64-10 network at batch 32; each launched operation pays "
                    "Python dispatch, and on the accelerator also a kernel launch.",
        network=MnistDenseNetwork,
        host=_Hardware.Cloud.ReferenceCPU,
        accelerator=_Hardware.Cloud.H100,
        precision="fp32",
        dispatch=_Ops.RuntimeOverheads.PythonDispatch,
        launch=_Ops.RuntimeOverheads.KernelLaunch,
        dram_energy_per_byte=_Hardware.Tech.Memory.DRAM.energy_per_byte,
        # One FLOP is half a multiply-accumulate: the mean of the multiply and add energies.
        flop_energy=(_Hardware.Tech.Op.FlopFp32.energy + _Hardware.Tech.Op.AddFp32.energy) / 2,
    )
    DigitRecognitionParadigms = _st.ParadigmCostScenario(
        name="One MNIST digit by rules, by HOG features, and by the dense network",
        description="Illustrative implementations: about 100 pixel comparisons; a 7 x 7 grid of 9-bin "
                    "HOG cells in about 8,000 operations and 2 KB; the 784-128-64-10 network.",
        image_pixels=_Datasets.MNIST.image_width * _Datasets.MNIST.image_height,
        bytes_per_pixel=_BYTES_INT8,  # 8-bit grayscale pixels
        rule_comparisons=100,  # scenario assumption: illustrative rule tree
        feature_grid=7,  # scenario assumption: HOG cells per side
        feature_bins=9,  # scenario assumption: orientation bins per cell
        feature_operations=8_000,  # scenario assumption: HOG plus a linear classifier
        feature_memory=Q_(2, "KB"),  # scenario assumption: HOG working memory
        network=MnistDenseNetwork,
    )
    ReluVersusSigmoidLogic = _st.ActivationLogicCostScenario(
        name="Logic cost of a ReLU and of a sigmoid datapath",
        description="A stylized teaching budget in transistor equivalents, not a measured constant: a "
                    "sign-bit comparator and multiplexer against an exponential approximation.",
        cheap_label="ReLU",
        cheap_transistors=50,  # scenario assumption: comparator plus selection
        costly_label="sigmoid",
        costly_transistors=2_500,  # scenario assumption: exponential approximation
    )
    TwentySigmoidLayers = _st.GradientChainScenario(
        name="Gradient through 20 sigmoid layers, at the sigmoid's steepest slope",
        description="The activation-slope factor alone, before the weights multiply in.",
        depth=20,  # scenario assumption
        activation="sigmoid",
    )
    DigitCrossEntropyExamples = _st.CrossEntropyScenario(
        name="Cross-entropy for one MNIST digit: confident, correct, and misread",
        description="Probabilities on the true class of 0.9 and 0.8, and a seven read as a three "
                    "(0.6 on three, 0.1 on seven).",
        classes=_Datasets.MNIST.num_classes,
        probabilities={"confident": 0.9, "correct": 0.8},  # scenario assumption
        misread={"true": 0.1, "wrong": 0.6},  # scenario assumption
    )
    Gpt2Fp32AdamOnH100 = _st.TrainingStateFitScenario(
        name="GPT-2 (1.5B) trained in FP32 with Adam, against one H100's memory",
        description="Weights, gradients, and two Adam values per parameter, all FP32, before activations.",
        workload=_Models.Language.GPT2,
        hardware=_Hardware.Cloud.H100,
        bytes_per_value=_BYTES_FP32,
        optimizer_bytes_per_parameter=_BYTES_ADAM_MOMENTS,
    )
    ResNet50Fp32SgdMomentum = _st.TrainingStateFitScenario(
        name="ResNet-50 trained in FP32 with SGD and momentum",
        description="Weights, gradients, and one momentum value per parameter, all FP32, before activations.",
        workload=_Models.Vision.ResNet50,
        bytes_per_value=_BYTES_FP32,
        optimizer_bytes_per_parameter=_BYTES_FP32,  # one momentum value per parameter
    )

    # ── ML operations at scale (added 2026-10-20) ─────────────────────────────
    OperationalLoadSketch = _st.OperationalLoadScenario(
        name="Operational load of a model portfolio: alerts N, deployments N ln N, conflicts N^2/2",
        description="Assumed growth shapes, not measurements, against one team's capacity; the total "
                    "crosses it near 50 models.",
        alerts_per_model=20,  # scenario assumption: alert units each model adds
        conflict_coefficient=0.5,  # scenario assumption: N models form about N^2/2 pairs
        team_capacity=2500,  # scenario assumption: load one team can absorb
        max_models=500,  # scenario assumption: portfolio range plotted
    )
    RecommendationFeatureHistory = _st.FeatureHistoryScenario(
        name="Hourly feature snapshots for 100 million users and 1,000 features, kept one year",
        description="Full snapshots, no compression; each row an 8 B value, 8 B entity key, and event and "
                    "availability timestamps of 8 B each, as a point-in-time join requires.",
        entities=100e6,  # scenario assumption: users
        features=1000,  # scenario assumption
        retention=Q_(_DAYS_PER_YEAR, "day"),  # scenario assumption: one year of history
        update_interval=Q_(1, "hour"),  # scenario assumption: hourly snapshots
        value_bytes=Q_(8, "byte"),  # scenario assumption
        key_bytes=Q_(8, "byte"),  # scenario assumption
        timestamp_bytes=Q_(8, "byte"),  # scenario assumption
    )
    DailyBillionRecommendationLookups = _st.FeatureLookupScenario(
        name="One billion recommendations a day, 100 features each",
        description="Average feature-value lookups before peak traffic, and the store's share of a 50 ms "
                    "recommendation budget.",
        request_rate=Q_(1e9, "1/day"),  # scenario assumption: recommendations per day
        reads_per_request=100,  # scenario assumption: features per recommendation
        request_budget=Q_(50, "ms"),  # scenario assumption
        lookup_budget_low=Q_(5, "ms"),  # scenario assumption
        lookup_budget_high=Q_(10, "ms"),  # scenario assumption
    )
    AggregationWindowRevenueIncident = _st.RevenueRegressionScenario(
        name="A pipeline version changes a feature's aggregation window; CTR falls for five days",
        description="Revenue tracks CTR and traffic is uniform; an alert with containment would bound the "
                    "exposure to four hours (constructed values).",
        weekly_revenue=Q_(15e6, "USD"),  # scenario assumption: revenue per week
        relative_drop=0.08,  # scenario assumption: CTR and revenue fall
        metric_before=0.042,  # scenario assumption: CTR
        metric_after=0.0387,  # scenario assumption: CTR
        undetected=Q_(5, "day"),  # scenario assumption
        alerted=Q_(4, "hour"),  # scenario assumption: alert and containment
        feature_mean_before=0.42,  # scenario assumption: user_engagement_score mean
        feature_mean_after=0.31,  # scenario assumption
        freshness_slo=Q_(1, "hour"),  # scenario assumption
        shift_threshold=0.10,  # scenario assumption: distribution-gate band
    )
    DelayedAlertClickLoss = _st.ClickLossScenario(
        name="A silent CTR drop from 5 to 4.5 percent at 5,000 requests/s, caught after one day",
        description="Lost clicks and their value before detection and rollback.",
        request_rate=Q_(5000, "1/s"),  # scenario assumption
        detection_time=Q_(24, "hour"),  # scenario assumption
        ctr_before=0.05,  # scenario assumption
        ctr_after=0.045,  # scenario assumption
        value_per_click=Q_(0.50, "USD"),  # scenario assumption
    )
    CanaryVersusCutoverExposure = _st.RolloutExposureScenario(
        name="All-at-once cutover against a 5 percent canary, same time to rollback",
        description="Exposure is proportional to the affected traffic share when request rate, harm, and "
                    "rollback time are the same.",
        full_share=1.0,
        canary_share=0.05,  # scenario assumption
        time_to_rollback=Q_(2, "hour"),  # scenario assumption: time to detect and roll back
    )
    CanaryGuardrailSchedule = _st.CanaryScheduleScenario(
        name="Canary ladder for a model at 1 million requests/h, 10,000 guardrail samples per stage",
        description="Minimum stage time n / (r p) and the organization's configured ladder: 1 percent for "
                    "2 hours, 5 and 25 percent for 30 minutes each, 50 percent for an hour, then all traffic.",
        samples_per_variant=10000,  # scenario assumption: guardrail sample per variant
        request_rate=Q_(1e6, "1/hour"),  # scenario assumption
        stages=[(0.01, Q_(2, "hour")),  # scenario assumption: configured ladder
                (0.05, Q_(30, "minute")),
                (0.25, Q_(30, "minute")),
                (0.50, Q_(1, "hour"))],
    )
    CtrLiftExperiment = _st.TwoProportionTestScenario(
        name="Two-arm test for a 0.5-point CTR lift on a 5 percent baseline",
        description="95 percent confidence and 80 percent power on 1 million requests a day split evenly.",
        baseline_rate=0.05,  # scenario assumption
        lift=0.005,  # scenario assumption: 0.5 percentage points
        z_alpha=1.96,  # two-sided 5 percent significance
        z_beta=0.84,  # 80 percent power
        daily_requests=Q_(1e6, "1/day"),  # scenario assumption
    )
    ThreeSigmaModelFleetAlerts = _st.MonitorFleetAlertScenario(
        name="100 models x 10 metrics checked every 5 minutes at 3-sigma thresholds",
        description="Raw false threshold crossings per day on a stable Gaussian process, before grouping.",
        models=100,  # scenario assumption
        metrics_per_model=10,  # scenario assumption
        check_interval=Q_(5, "minute"),  # scenario assumption
        false_positive_rate=0.003,  # 3-sigma two-sided tail, rounded
    )
    TwoSigmaModelFleetAlerts = ThreeSigmaModelFleetAlerts.variant(
        name="The same fleet at 2-sigma thresholds, 99 percent of crossings deduplicated",
        false_positive_rate=0.05,  # 2-sigma two-sided tail, rounded
        dedup_share=0.99,  # scenario assumption: crossings merged or auto-resolved
    )
    OwnVsRentH100Cluster1K = _st.OwnVsRentScenario(
        name="Own or rent a 1,024-H100 cluster for three years at 80 percent utilization",
        description="128 DGX H100 nodes plus the reference fabric and facility, a fixed $1.5M a year to run, "
                    "against on-demand GPU-hours.",
        gpus=_Clusters.Training_1K.total_accelerators,
        gpus_per_node=_Nodes.DGX_H100.accelerators_per_node,
        node_price=_Infrastructure.Pricing.Capital.DgxH100NodeUsd.rate,
        rent_rate=_Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
        utilization=_Systems.Orchestration.target_cluster_utilization,
        years=_Infrastructure.Pricing.Capital.AcceleratorServiceLifeYears.rate,
        network_capex=_ReferenceStats.ClusterTcoProfile.Reference1KNetworkCapex,
        facility_capex=_ReferenceStats.ClusterTcoProfile.Reference1KFacilityCapex,
        opex_per_year=_ReferenceStats.ClusterTcoProfile.Reference1KAnnualOpex,
    )
    OwnVsRentDgxH100Node = _st.OwnVsRentScenario(
        name="Own or rent one DGX H100 node: purchase and energy only",
        description="The node price amortized over its service life plus facility electricity drawn while "
                    "busy, against on-demand GPU-hours.",
        gpus=_Nodes.DGX_H100.accelerators_per_node,
        gpus_per_node=_Nodes.DGX_H100.accelerators_per_node,
        node_price=_Infrastructure.Pricing.Capital.DgxH100NodeUsd.rate,
        rent_rate=_Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
        utilization=_Systems.Orchestration.target_cluster_utilization,
        years=_Infrastructure.Pricing.Capital.AcceleratorServiceLifeYears.rate,
        power=_Systems.Racks.DGX_H100_4Node.total_power / _Systems.Racks.DGX_H100_4Node.nodes_per_rack,
        pue=_Infrastructure.FacilityCooling.StateOfArt.pue,
        electricity_price=_Infrastructure.Pricing.OnPremises.ElectricityPerKwh.rate,
        energy_scales_with_use=True,
    )
    OwnVsRentH100Cluster10K = _st.OwnVsRentScenario(
        name="Own or rent 10,000 H100s for three years at 80 percent utilization",
        description="1,250 DGX H100 nodes, the reference fabric and liquid-cooled hall, GPU-only electricity "
                    "at all hours, and staffing; no maintenance or financing.",
        gpus=_Clusters.Training_10K.total_accelerators,
        gpus_per_node=_Nodes.DGX_H100.accelerators_per_node,
        node_price=_Infrastructure.Pricing.Capital.DgxH100NodeUsd.rate,
        rent_rate=_Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
        utilization=_Systems.Orchestration.target_cluster_utilization,
        years=_Infrastructure.Pricing.Capital.AcceleratorServiceLifeYears.rate,
        network_capex=_ReferenceStats.ClusterTcoProfile.Reference10KNetworkCapex,
        facility_capex=_ReferenceStats.ClusterTcoProfile.Reference10KFacilityCapex,
        staffing_per_year=_ReferenceStats.ClusterTcoProfile.Reference10KAnnualStaffing,
        power=_Clusters.Training_10K.total_accelerators * _Hardware.Cloud.H100.tdp,
        pue=_Infrastructure.FacilityCooling.StateOfArt.pue,
        electricity_price=_Infrastructure.Pricing.OnPremises.ElectricityPerKwh.rate,
    )
    BurstyTrainingOn1KCluster = _st.TrainingCampaignCostScenario(
        name="Six two-to-four-week training runs a year on the owned 1,024-H100 cluster",
        description="Owned lifetime cost per run against renting the same GPUs only while a run lasts.",
        ownership=OwnVsRentH100Cluster1K,
        runs_per_year=6,  # scenario assumption: bursty training cadence
        run_low=Q_(2, "week"),  # scenario assumption: run length
        run_high=Q_(4, "week"),  # scenario assumption: run length
    )
    StartupOneModelTco = _st.LifecycleTcoScenario(
        name="Monthly lifecycle cost of a one-model startup",
        description="Constructed budget: engineers iterating dominate.",
        monthly={"training": Q_(5000, "USD/month"),  # scenario assumption
                 "inference": Q_(2000, "USD/month"),  # scenario assumption
                 "data": Q_(500, "USD/month"),  # scenario assumption
                 "iteration": Q_(40000, "USD/month")},  # scenario assumption
    )
    FiftyModelCompanyTco = _st.LifecycleTcoScenario(
        name="Monthly lifecycle cost of a 50-model production company",
        description="Constructed budget: inference is the largest term.",
        monthly={"training": Q_(150000, "USD/month"),  # scenario assumption
                 "inference": Q_(400000, "USD/month"),  # scenario assumption
                 "data": Q_(80000, "USD/month"),  # scenario assumption
                 "iteration": Q_(350000, "USD/month")},  # scenario assumption
    )
    Int8InferencePayback = _st.OptimizationPaybackScenario(
        name="INT8 quantization that cuts a $400K monthly inference bill by 40 percent",
        description="One-time implementation and validation cost against the monthly saving.",
        implementation_cost=Q_(80000, "USD"),  # scenario assumption
        monthly_bill=FiftyModelCompanyTco.monthly["inference"],
        reduction=0.40,  # scenario assumption
    )
    TrainingOptimizationPayback = Int8InferencePayback.variant(
        name="A training optimization of the same cost that cuts a $150K monthly training bill by 30 percent",
        monthly_bill=FiftyModelCompanyTco.monthly["training"],
        reduction=0.30,  # scenario assumption
    )
    SharedPlatformBenefit50 = _st.PlatformBenefitScenario(
        name="A $120K-a-month ML platform serving 50 model teams, 20 hours saved per team per month",
        description="Engineering time saved at the loaded ML-engineer rate against the platform's cost.",
        models=50,  # scenario assumption
        hours_saved=Q_(20, "hour/month"),  # scenario assumption: per model team
        engineer_rate=_Infrastructure.Pricing.Labor.MlEngineerLoadedHourly.rate,
        platform_cost=Q_(120000, "USD/month"),  # scenario assumption: platform team and infrastructure
    )
    SharedPlatformBenefit20 = SharedPlatformBenefit50.variant(
        name="The same platform serving 20 model teams",
        models=20,  # scenario assumption
    )

    # --- Network architectures: attention, transformer blocks, decode, and recommenders (added 2026-10-20) ---
    # GPT-2 XL is the running transformer; Llama 2 and Llama 3.1 shapes size
    # decode and the KV cache; DLRM-style tables carry the recommender.
    Gpt2SmallSixTokenQkvProjection = _st.PackedQkvProjectionScenario(
        name="Six tokens through GPT-2 Small's packed query-key-value projection",
        description="One 768 x 2,304 weight matrix produces all three projections in one kernel.",
        workload=_Models.Language.GPT2_Small,
        tokens=6,  # scenario assumption: a six-token sequence
    )
    Gpt2XlBlockFullWindow = _st.TransformerBlockScenario(
        name="GPT-2 XL decoder blocks at its 1,024-token window",
        description="The classic 12 d^2 block (feed-forward width 4d); dense against attention multiply-adds.",
        workload=_Models.Language.GPT2,
        context_tokens=_Models.Language.GPT2.context_length,
    )
    TwelveHeadScoresAt100kTokens = _st.MaterializedScoresScenario(
        name="Materialized attention scores of a 12-head, 32-layer model at 100,000 tokens",
        description="FP16 scores stored in full, against one 80 GB accelerator.",
        seq_len=100_000,  # scenario assumption: long context
        heads=12,  # scenario assumption: generic model
        layers=32,  # scenario assumption: generic model
        bytes_per_score=_BYTES_FP16,
        hardware=_Hardware.Cloud.H100,
    )
    BertWindowToFourK = _st.ContextGrowthScenario(
        name="BERT's 512 learned positions against a 4,096-token input",
        description="Score memory grows with the square of the length ratio.",
        base_tokens=_Models.Language.BERT_Base.context_length,
        tokens=4096,  # scenario assumption: a longer input
    )
    BertWindowToOneK = BertWindowToFourK.variant(name="BERT's window doubled", tokens=1024)  # scenario assumption
    BertWindowToTwoK = BertWindowToFourK.variant(name="BERT's window quadrupled", tokens=2048)  # scenario assumption
    Gpt2XlKvFullWindow = _st.KvFootprintScenario(
        name="GPT-2 XL's FP16 KV cache at its 1,024-token window",
        description="2 x 48 layers x 1,600 x 2 bytes per kept token.",
        workload=_Models.Language.GPT2,
        tokens=_Models.Language.GPT2.context_length,
        bytes_per_value=_BYTES_FP16,
    )
    Llama2_7BKvAt2048Tokens = Gpt2XlKvFullWindow.variant(
        name="Llama 2 7B's FP16 KV cache for one 2,048-token request",
        description="32 layers, 32 key-value heads of 128, FP16.",
        workload=_Models.Language.Llama2_7B,
        tokens=2048,  # scenario assumption: serving context length
    )
    Gpt2XlContextMemoryOnH100 = _st.ContextMemoryCrossingScenario(
        name="GPT-2 XL's weights, KV cache, and one layer's scores against one H100",
        description="FP16 throughout; where the linear cache and the quadratic scores cross fixed budgets.",
        workload=_Models.Language.GPT2,
        hardware=_Hardware.Cloud.H100,
        bytes_per_value=_BYTES_FP16,
    )
    Llama2_70B_DecodeTwoH100 = Llama2_70B_DecodeH100.variant(
        name="Llama 2 70B batch-1 decode split across two H100s",
        description="140 GB of FP16 weights do not fit one 80 GB H100; each device streams its half per token.",
        tp_size=2,  # scenario assumption: the smallest split that fits
    )
    Gpt2XlDecodeH100 = Llama2_70B_DecodeH100.variant(
        name="GPT-2 XL batch-1 decode on one H100",
        description="FP16 weights streamed once per token.",
        workload=_Models.Language.GPT2,
    )
    CriteoShapedRequestOnH100 = _st.EmbeddingGatherScenario(
        name="One Criteo-shaped recommender request on an H100",
        description="One 128-wide FP32 row per categorical feature; device HBM against the PCIe host link.",
        dataset=_Datasets.CriteoKaggle,
        width=128,  # scenario assumption: embedding width
        bytes_per_value=_BYTES_FP32,
        hardware=_Hardware.Cloud.H100,
    )
    HundredMillionItemTableOnA100 = _st.EmbeddingTableScenario(
        name="A 100-million-item embedding table on one 80 GB A100",
        description="128 FP32 values per row, one categorical feature.",
        rows=100_000_000,  # scenario assumption: catalog size
        width=CriteoShapedRequestOnH100.width,
        bytes_per_value=_BYTES_FP32,
        hardware=_Hardware.Cloud.A100,
    )
    BillionUserTable = HundredMillionItemTableOnA100.variant(
        name="A billion-user embedding table, 128 FP32 values per row",
        rows=1_000_000_000,  # scenario assumption: user count
    )

    # --- Framework execution: dispatch, fusion, compilation, saved state, copies (added 2026-10-20) ---
    Gpt2XlFp16DecodeOnH100 = _st.DecodeScenario(
        name="GPT-2 XL batch-one decode step on one H100, FP16",
        description="Every FP16 weight read once from HBM for each generated token.",
        workload=_Models.Language.GPT2,
        hardware=_Hardware.Cloud.H100,
        precision="fp16",
        batch_size=1,
    )
    Gpt2XlEagerDecodeDispatch = _st.HostDispatchScenario(
        name="Eager host work to issue one GPT-2 XL decode step on an H100",
        description="One eager kernel per operation in each transformer block, each paying Python dispatch "
                    "and a kernel launch, against the step's HBM weight read.",
        # scenario assumption: one eager kernel each for two layer norms, the fused QKV projection,
        # attention scores, softmax, weighted sum, output projection, two feed-forward matmuls,
        # the activation, and two residual adds
        ops_per_block=12,
        blocks=_Models.Language.GPT2.layers,
        dispatch=_Ops.RuntimeOverheads.PythonDispatch,
        launch=_Ops.RuntimeOverheads.KernelLaunch,
        device_time=Gpt2XlFp16DecodeOnH100.result.memory_time,
    )
    SmallKernelDispatch = _st.HostDispatchScenario(
        name="One small eager operation: a short element-wise kernel",
        description="Python dispatch and one kernel launch against a 1 us kernel (a ReLU on a small vector).",
        ops_per_block=1,
        dispatch=_Ops.RuntimeOverheads.PythonDispatch,
        launch=_Ops.RuntimeOverheads.KernelLaunch,
        device_time=Q_(1, "microsecond"),  # scenario assumption: ReLU on a small vector
    )
    LargeKernelDispatch = SmallKernelDispatch.variant(
        name="One large eager operation: a long matrix multiply",
        description="The same host cost against a 100 us kernel (a large matrix multiply).",
        device_time=Q_(100, "microsecond"),  # scenario assumption: large matrix multiply
    )
    SmallMlpDispatch = SmallKernelDispatch.variant(
        name="A small MLP's eager step: six operations, 2.6 us of device work",
        description="Six dispatched operations against the small network's device time.",
        ops_per_block=6,  # scenario assumption: operations in the small MLP's step
        device_time=Q_(2.6, "microsecond"),  # scenario assumption: small MLP device time
    )
    Gpt3BlockDispatch = SmallKernelDispatch.variant(
        name="One GPT-3-scale transformer block over a large batch, eager",
        description="One block's eager kernels against 100 ms of device time.",
        ops_per_block=Gpt2XlEagerDecodeDispatch.ops_per_block,
        device_time=Q_(100, "ms"),  # scenario assumption: one GPT-3-scale layer over a large batch
    )
    ElementwisePairFusion = _st.KernelFusionScenario(
        name="Two element-wise kernels against one fused kernel",
        description="Launch overhead and device-memory trips of a two-operation chain, unfused and fused.",
        chain_length=2,  # scenario assumption
        dispatch=_Ops.RuntimeOverheads.PythonDispatch,
        launch=_Ops.RuntimeOverheads.KernelLaunch,
    )
    ResNet50CompileBreakeven = _st.CompileBreakevenScenario(
        name="ResNet-50 inference, eager against compiled (hypothetical rates)",
        description="Images that repay a compile, and a prototyping run that recompiles on every code change.",
        eager_rate=Q_(1450, "1/second"),  # scenario assumption: hypothetical eager images per second
        compiled_rate=Q_(2150, "1/second"),  # scenario assumption: hypothetical compiled images per second
        compile_time=Q_(30, "second"),  # scenario assumption: compile per code change
        experiment_items=10_000,  # scenario assumption: images in one prototyping run
        recompiles=10,  # scenario assumption: code changes in that run
    )
    Dense100kLayerJacobianFp32 = _st.JacobianScenario(
        name="Full Jacobian of one 100,000-wide layer, FP32",
        description="The matrix reverse mode never builds: every output against every input.",
        input_width=100_000,  # scenario assumption
        output_width=100_000,  # scenario assumption
        bytes_per_value=_BYTES_FP32,
    )
    ResNet50Fp32AdamBatch64 = ResNet50Fp32SgdMomentum.variant(
        name="ResNet-50 trained in FP32 with Adam at batch 64, low activation estimate",
        description="Weights, gradients, two Adam values per parameter, and 8 GB of saved activations "
                    "for 224x224 images, against one H100's memory.",
        optimizer_bytes_per_parameter=_BYTES_ADAM_MOMENTS,
        hardware=_Hardware.Cloud.H100,
        batch_size=64,  # scenario assumption
        activations=Q_(8, "GB"),  # scenario assumption: saved-activation estimate, low end
    )
    ResNet50Fp32AdamBatch64HighActivations = ResNet50Fp32AdamBatch64.variant(
        name="ResNet-50 trained in FP32 with Adam at batch 64, high activation estimate",
        description="The same training state with 12 GB of saved activations.",
        activations=Q_(12, "GB"),  # scenario assumption: saved-activation estimate, high end
    )
    ResNet50Fp32AdamBatch512 = ResNet50Fp32AdamBatch64.variant(
        name="ResNet-50 trained in FP32 with Adam at batch 512, low activation estimate",
        description="The batch-64 state with its saved activations scaled linearly to batch 512.",
        batch_size=512,  # scenario assumption
        activations=ResNet50Fp32AdamBatch64.activations_at(512),  # the batch above
    )
    Fp32MillionElementTensorAcrossH100Links = _st.TransferScenario(
        name="A 1,000 by 1,000 FP32 tensor across an H100's memory and links",
        description="One 4 MB tensor over HBM, NVLink (one direction), and the PCIe host link. "
                    "Bandwidth-only bounds.",
        payload=1000 * 1000 * _BYTES_FP32,  # scenario assumption: activation tensor shape
        links={
            "HBM": _Hardware.Cloud.H100.memory.bandwidth,
            "NVLink": _Hardware.Cloud.H100.nvlink.bandwidth_per_direction,
            "PCIe": _Hardware.Cloud.H100.interconnect.bandwidth,
        },
    )
    ImageNetBatch64HostFeed = _st.HostInputPipelineScenario(
        name="A data loader feeding 224x224 RGB images to one H100 at batch 64",
        description="Four loader workers prefetching two batches each; uint8 pixels decoded on the host, "
                    "FP32 tensors copied over PCIe.",
        batch_size=64,  # scenario assumption
        workers=4,  # scenario assumption
        prefetch_factor=2,  # scenario assumption
        raw_sample=(_Datasets.ImageNet.image_width * _Datasets.ImageNet.image_height
                    * _Datasets.ImageNet.image_channels * _BYTES_INT8),
        model_sample=(_Datasets.ImageNet.image_width * _Datasets.ImageNet.image_height
                      * _Datasets.ImageNet.image_channels * _BYTES_FP32),
        consume_rate=Q_(1000, "1/second"),  # scenario assumption: accelerator consumption rate
        host_link=_Hardware.Cloud.H100.interconnect.bandwidth,
        device_memory=_Hardware.Cloud.H100.memory.bandwidth,
        step_times={
            "large model": Q_(5, "ms"),  # scenario assumption
            "small model": Q_(0.5, "ms"),  # scenario assumption
        },
    )
    MnistTwoLayerTrainingStepOnH100 = _st.MlpTrainingStepScenario(
        name="One eager training step of a 784-128-10 MNIST MLP at batch 32 on an H100, FP32",
        description="Per-operator FLOPs and bytes, ideal compute and memory times, and the host's "
                    "Python entries and kernel launches.",
        batch_size=32,  # scenario assumption
        input_width=(_Datasets.MNIST.image_width * _Datasets.MNIST.image_height
                     * _Datasets.MNIST.image_channels),
        hidden_width=128,  # scenario assumption: two-layer version of the 784-128-64-10 network
        output_width=_Datasets.MNIST.num_classes,
        bytes_per_value=_BYTES_FP32,
        label_bytes=_BYTES_INT64,
        loss_flops_per_logit=3,  # scenario assumption: one exp, accumulation, and normalization per logit
        optimizer_values_moved=7,  # FP32 Adam reads p, g, m, v and writes p, m, v
        hardware=_Hardware.Cloud.H100,
        precision="fp32",
        forward_ops=4,  # GEMM 1, ReLU, GEMM 2, combined loss
        backward_ops=5,  # three GEMMs, ReLU derivative, loss derivative
        dispatch=_Ops.RuntimeOverheads.PythonDispatch,
        launch=_Ops.RuntimeOverheads.KernelLaunch,
    )

    # --- Security and privacy of a fleet (added 2026-10-20) -----------------
    ResNet50ApiExtractionBudget = _st.QueryExtractionBudgetScenario(
        name="Copying a ResNet-50 image API to 90 percent fidelity under three billing regimes",
        description="5 million answers against a $5K training cost: $0.001 per query, a $50-a-month tier "
                    "capped at 10,000 queries a day, or a 100,000-query lifetime allowance per principal "
                    "with every later query billed per query (constructed values).",
        training_cost=Q_(5000, "USD"),  # scenario assumption
        extraction_queries=5e6,  # scenario assumption: answers for 90 percent fidelity
        target_fidelity=0.90,  # scenario assumption
        price_per_query=Q_(0.001, "USD"),  # scenario assumption
        tier_price=Q_(50, "USD"),  # scenario assumption: per billing period
        tier_daily_quota=1e4,  # scenario assumption
        billing_period=Q_(_DAYS_PER_MONTH, "day"),
        lifetime_allowance=1e5,  # scenario assumption
    )
    ImageNetTop5RoundedScores = _st.OutputShapingScenario(
        name="ImageNet classifier API returning the top 5 scores rounded to two decimals",
        description="Output contract: top-k truncation of 1,000 class scores and rounding from six "
                    "decimal places to two.",
        class_count=int(_Datasets.ImageNet.num_classes),
        top_k=5,  # scenario assumption
        from_decimals=6,  # scenario assumption
        to_decimals=2,  # scenario assumption
    )
    ReferenceJobAttackSurface = _st.DataParallelAttackSurfaceScenario(
        name="175B pretraining on the 1,024-GPU reference cluster with 64 data-parallel replicas",
        description="FP16 gradients exchanged each step, the 14-byte-per-parameter checkpoint, and a "
                    "gather of every replica's gradient for a robust aggregate.",
        workload=DenseLLMPretrainingReference.workload,
        fleet=DenseLLMPretrainingReference.system,
        dp_degree=64,  # scenario assumption: two 8-GPU nodes of tensor parallelism per replica
        gradient_bytes_per_param=(_ReferenceStats.MixedPrecisionAdam.TrainingStateBytesPerParameter
                                  - _ReferenceStats.StorageTrainingCorpus.CheckpointBytesPerParameter),
        checkpoint_bytes_per_param=_ReferenceStats.StorageTrainingCorpus.CheckpointBytesPerParameter,
        weight_bytes_per_param=_BYTES_FP16 / _param,
    )
    SecureMultiTenantServing = _st.ThroughputCostScenario(
        name="A matched serving configuration at 1,000 tokens/s, 850 with tenant isolation",
        description="Assumed throughputs for the same workload and resource budget; the loss splits "
                    "into stranded partitions, scrubbing time, and lost cache hits.",
        baseline_rate=Q_(1000, "1/s"),  # scenario assumption: tokens per second
        controlled_rate=Q_(850, "1/s"),  # scenario assumption: tokens per second
    )
    ResNet50Fp32InSgxEnclave = _st.ProtectedMemoryFitScenario(
        name="ResNet-50 FP32 weights in a 128 MiB SGX enclave page cache",
        description="Weights only, before activations, against an older SGX configuration's EPC.",
        workload=_Models.Vision.ResNet50,
        hardware=_Hardware.Cloud.IntelSGX,
        bytes_per_weight=_BYTES_FP32,
    )
    ResNet50Int8InSgxEnclave = ResNet50Fp32InSgxEnclave.variant(
        name="ResNet-50 INT8 weights in the same enclave page cache", bytes_per_weight=_BYTES_INT8)
    ResNet18Fp32InSgxEnclave = ResNet50Fp32InSgxEnclave.variant(
        name="ResNet-18 FP32 weights in the same enclave page cache", workload=_Models.Vision.ResNet18)
    ResNet18Int8InSgxEnclave = ResNet50Fp32InSgxEnclave.variant(
        name="ResNet-18 INT8 weights in the same enclave page cache", workload=_Models.Vision.ResNet18,
        bytes_per_weight=_BYTES_INT8)
    HealthMonitoringEncryptedInference = _st.EncryptedInferenceLatencyScenario(
        name="A 20 ms health-monitoring inference, with AES around it or under FHE",
        description="Assumed overheads: AES-NI adds 0.5 ms; FHE runs at the conservative 10,000x slowdown.",
        plaintext_latency=Q_(20, "ms"),  # scenario assumption
        encryption_overhead=Q_(0.5, "ms"),  # scenario assumption: AES-NI accelerated
        fhe_multiplier=10000.0,  # Fully homomorphic encryption compute overhead, 2-6 orders of magnit...
    )
    SalaryMeanLaplaceRelease = _st.LaplaceMeanReleaseScenario(
        name="Mean salary of 1,000 employees on a public roster at epsilon 1",
        description="Salaries from 0 to $200,000; one Laplace draw on the sum, divided by the public count.",
        value_range=Q_(200000, "USD"),  # scenario assumption
        epsilon=1.0,  # scenario assumption
        group_size=1000,  # scenario assumption
        small_group_size=100,  # scenario assumption
    )
    MembershipAuditAtEpsilonOne = _st.MembershipAuditScenario(
        name="A membership audit below the pure epsilon = 1 DP ceiling",
        description="Illustrative audit curve; the ceiling follows from the DP definition with delta = 0.",
        epsilon=1.0,  # scenario assumption
        audit_shape=0.6,  # scenario assumption: must stay under the ceiling (needs <= 1 - e^-eps)
    )
    Cifar10DpSgdRun = _st.DpSgdRunScenario(
        name="DP-SGD on CIFAR-10: batch 4,000, 60 epochs, clip 1, sigma 1.3, target epsilon 3",
        description="Inputs a sampled-Gaussian accountant needs; epsilon itself comes from the accountant.",
        dataset_size=int(_Datasets.CIFAR10.training_examples.to("count").magnitude),
        batch_size=4000,  # scenario assumption
        clip_norm=1.0,  # scenario assumption
        noise_multiplier=1.3,  # scenario assumption
        epochs=60,  # scenario assumption
        target_epsilon=3.0,  # scenario assumption
        delta=1e-5,  # scenario assumption
    )

    # --- Robust AI: silent faults, drift tests, per-request answering, attacks (added 2026-10-20) ---
    FeatureMeanShiftTest = _st.MeanShiftTestScenario(
        name="One feature's mean moves from 0.50 to 0.55 over the last 1,000 requests",
        description="Spread 0.3; is the move noise? Standard error, z, and the two-sided p-value.",
        reference_mean=0.5,               # scenario assumption
        observed_mean=0.55,               # scenario assumption
        feature_sd=0.3,                   # scenario assumption
        n_samples=1000,                   # scenario assumption
        significance_bound=0.001,         # scenario assumption: the bound the alert reports
    )
    FraudCountryPsi = _st.BinShareShiftScenario(
        name="User-country PSI of a fraud model after six months in production",
        description="Baseline and current traffic shares by country bucket; the Other bucket grows.",
        bins=["USA", "UK", "Germany", "France", "Other"],
        baseline_shares=[0.45, 0.20, 0.15, 0.10, 0.10],   # scenario assumption
        current_shares=[0.38, 0.18, 0.14, 0.12, 0.18],    # scenario assumption
    )
    FraudAmountDrift = _st.GaussianDriftScenario(
        name="Transaction-amount drift of the same fraud model, log-dollars fitted as normals",
        description="KL divergence of the reference fit from the current fit.",
        reference_mean=4.2,               # scenario assumption
        reference_sd=1.1,                 # scenario assumption
        current_mean=4.5,                 # scenario assumption
        current_sd=1.3,                   # scenario assumption
        kl_trigger=0.05,                  # scenario assumption: the service's calibrated investigate threshold
    )
    FraudAmountKs = _st.KsDriftScenario(
        name="K-S test on transaction amount, 10,000 against 10,000, Bonferroni level 0.01",
        description="The observed largest CDF gap against the critical value at the corrected level.",
        coefficient=_calc_ks_coefficient(0.01),
        significance=0.01,                # scenario assumption: Bonferroni-corrected level of the Level 1 rule
        n_reference=10_000,               # scenario assumption
        n_current=10_000,                 # scenario assumption
        observed_statistic=0.089,         # scenario assumption: observed largest CDF gap
    )
    PgdSevenTraining = _st.AdversarialTrainingPassesScenario(
        name="PGD-7 adversarial training: seven attack steps and one update per batch",
        description="Equal-cost work model with no reuse across steps (Madry et al. 2018).",
        attack_passes=7,                  # scenario assumption: PGD-7
        update_passes=1,
    )
    SmoothingRadiusCifar10 = _st.CertifiedRadiusScenario(
        name="Randomized smoothing at sigma 0.5 and p_A 0.999 against the 8/255 box on a CIFAR-10 image",
        description="Binary certified radius against the l2 length of the l-infinity box corner.",
        noise_sd=0.5,                     # scenario assumption
        top_class_probability=0.999,      # scenario assumption
        linf_levels=8,                    # the RobustBench CIFAR-10 l-infinity budget, 8/255
        pixel_levels=_COLOR_DEPTH_8BIT,
        coordinates=(_Datasets.CIFAR10.image_width * _Datasets.CIFAR10.image_height
                     * _Datasets.CIFAR10.image_channels),
    )
    Fp32WeightExponentFlip = _st.FloatBitFlipScenario(
        name="The top exponent bit of an FP32 weight of 0.042 flips",
        description="The value is multiplied by 2^128 and stays finite, so NaN and infinity checks pass.",
        value=0.042,                      # scenario assumption: an illustrative weight
        flipped_bit=sum(_FLOAT_FORMAT_BITS["fp32"]) - 1,   # bit 30, the top exponent bit
        exponent_bits=_FLOAT_FORMAT_BITS["fp32"][0],
        fraction_bits=_FLOAT_FORMAT_BITS["fp32"][1],
    )

    # --- Agent security: injection exposure and defense evaluation (added 2026-10-20) ---
    InjectionExposureByReadVolume = _st.InjectionExposureScenario(
        name="Injection exposure of four agent workloads at a 1 percent payload rate",
        description="Per-task chance of reading at least one injected payload for repair (20 items), "
                    "research, support, and remediation (their workload profiles), unscreened and "
                    "behind a screen that misses 5 percent of payloads; residual research exposure "
                    "per 1,000 tasks.",
        payload_fraction=0.01,  # scenario assumption: share of retrievable items carrying a payload
        reads={
            "repair": 20,  # scenario assumption: READMEs, issues, and search results per repair task
            "research": _Agents.Research.OrchestratorWorker.untrusted_reads_per_task,
            "support": _Agents.Support.CustomerOps.untrusted_reads_per_task,
            "remediation": _Agents.Remediation.OnCall.untrusted_reads_per_task,
        },
        miss_rate=0.05,  # scenario assumption: share of payloads a screening classifier misses
        cohort=1000,  # scenario assumption: research tasks in the per-cohort count
    )
    FixtureReleasePolicyComparison = _st.PolicyComparisonScenario(
        name="Name-and-destination checks against labels plus a release check, 100 attacked and "
             "100 benign repair episodes each",
        description="Hypothetical outcome counts for two policies on equal, separate cohorts; attempts "
                    "priced at the repair attempt cost.",
        attacked=100,  # scenario assumption
        benign=100,  # scenario assumption
        baseline={"confirmed": 20, "unresolved": 5, "accepted_attacked": 65,  # scenario assumption
                  "accepted_benign": 90, "false_block": 5},
        candidate={"confirmed": 5, "unresolved": 10, "accepted_attacked": 55,  # scenario assumption
                   "accepted_benign": 80, "false_block": 15},
        attempt_cost=RepairAttemptCostLevers.result.attempt_cost,
    )
    CleanAttackSuite50 = _st.ZeroFailureScenario(
        name="50 attacked episodes with no attack success",
        description="The attack-success rate a clean run supports at 95 percent one-sided confidence, "
                    "and the clean episodes a 1 percent tolerance needs.",
        trials=50,  # scenario assumption
        confidence=0.95,
        target_rate=0.01,  # scenario assumption: tolerable attack-success rate
    )
    CleanAttackSuite100 = CleanAttackSuite50.variant(
        name="100 attacked episodes with no attack success",
        trials=100,  # scenario assumption
    )
    PrefixCacheProbe2K = InvariantPrefixRebuildDgxH100.variant(
        name="A 2,048-token probed prefix: prefill on a miss against a weight-read pass on a hit",
        prefix_tokens=2_048,  # scenario assumption: probed system-prompt prefix
    )

    # --- The agent harness (added 2026-10-20) ---
    # The runtime around one coding trajectory: a rule kept per turn, a pause
    # behind one call's output allowance (CanceledStreamDecodeSavings
    # .allowance_decode_time), approval escrow, fair scheduling of model calls,
    # admission of the next call, the budget meter, and ceilings read from
    # public runs. Prices are hypothetical, not a provider schedule.
    FreezeRuleOverStepBudget = HorizonCompounding30Steps.variant(
        name="A rule the model keeps on 99 percent of turns, over the coding-agent step budget",
        description="A stated freeze honored independently on each turn; the target is that 99 "
                    "percent of runs keep it on every turn.",
        step_success=0.99,  # scenario assumption: hypothetical per-turn compliance
        target_success=0.99,  # scenario assumption: share of runs that must keep the rule throughout
    )

    RepairRemediationApprovalEscrow = _st.ApprovalEscrowScenario(
        name="Approval escrow for a repair agent and an on-call remediation agent",
        description="Repair review requests at 6 per hour with 2 h mean decisions, each holding the "
                    "SWE-bench coding harness's workspace lease; remediation at the on-call "
                    "profile's request rate, diagnosis, decision, and apply times.",
        repair_rate=Q_(6, "1/hour"),  # scenario assumption: review requests per hour
        repair_wait=Q_(2, "hour"),  # scenario assumption: mean time to a review decision
        lease=_Agents.Coding.SWE_Bench_Runner.sandbox_memory_footprint,
        remediation_rate=_Agents.Remediation.OnCall.approval_request_rate,
        diagnosis=_Agents.Remediation.OnCall.diagnosis_time,
        remediation_wait=_Agents.Remediation.OnCall.approval_decision_time,
        apply_verify=_Agents.Remediation.OnCall.apply_verify_time,
    )

    InteractiveMaintenanceDrr = _st.DeficitRoundRobinScenario(
        name="Interactive and maintenance agent calls under weighted deficit round-robin",
        description="Weights 3 and 1 and an 8,000-token quantum per unit weight per round; an "
                    "interactive call charges 2,000 input tokens plus a 500-token output "
                    "allowance, a maintenance call 40,000 plus 2,000.",
        quantum=8000,  # scenario assumption: charge tokens per unit weight per round
        classes={"interactive": {"weight": 3, "input": 2000, "output": 500},  # scenario assumption
                 "maintenance": {"weight": 1, "input": 40000, "output": 2000}},  # scenario assumption
        simulated_rounds=10_000,  # rounds of the direct simulation that checks the formula
    )
    InteractiveMaintenanceDrrCompacted = InteractiveMaintenanceDrr.variant(
        name="The same schedule with maintenance prompts compacted to half their input",
        classes={**InteractiveMaintenanceDrr.classes,
                 "maintenance": {**InteractiveMaintenanceDrr.classes["maintenance"],
                                 "input": InteractiveMaintenanceDrr.classes["maintenance"]["input"] // 2}},
    )

    RepairNextCallAdmission = _st.BoundedLiabilityAdmissionScenario(
        name="The next model call of a repair trajectory against a $3.50 allowance",
        description="$2.58 settled, a $0.50 reservation on an unsettled test job, and a $0.30 "
                    "finalization reserve; the call sends 50,000 input tokens with a 4,096-token "
                    "output allowance at a hypothetical $3 and $15 per million tokens.",
        allowance=Q_(3.50, "USD"),  # scenario assumption: dollar allowance of the trajectory
        settled=Q_(2.58, "USD"),  # scenario assumption: actual spend so far
        reserved=Q_(0.50, "USD"),  # scenario assumption: reservation on the unsettled test job
        finalization_reserve=Q_(0.30, "USD"),  # scenario assumption: protected for checks and report
        input_tokens=50_000,  # scenario assumption: next call's input
        output_allowance=4096,  # scenario assumption: next call's output allowance
        input_price=OutputCeilingThreeAllocations.input_price,
        output_price=ProbeVsGeneratedExplanation.output_price,
        probe=Q_(0.10, "USD"),  # an unsettled amount used to express sensitivity
    )

    RepairTrajectoryBudgetMeter = _st.TrajectoryBudgetMeterScenario(
        name="The 30-turn coding trajectory metered against turns, tokens, time, and dollars",
        description="Allowances of 40 turns, 2 million tokens, 900 s, and the repair trajectory's "
                    "$3.50, read at 460 s on the clock; hypothetical $3 input, $0.30 cached, and "
                    "$15 output per million tokens.",
        trajectory=CodingAgentResend30Turns,
        input_price=OutputCeilingThreeAllocations.input_price,
        cached_fraction=RepairPrefixLayoutBill.cached_fraction,
        output_price=ProbeVsGeneratedExplanation.output_price,
        turn_allowance=40,  # scenario assumption
        token_allowance=2_000_000,  # scenario assumption
        time_allowance=Q_(900, "second"),  # scenario assumption
        dollar_allowance=RepairNextCallAdmission.allowance,
        elapsed=Q_(460, "second"),  # scenario assumption: wall clock at the readout
    )


    # --- Closed-loop evaluation: the latch campaign, sequential tests, exposure (added 2026-10-20) ---
    LatchEvaluationCampaign = _st.ClosedLoopCampaignScenario(
        name="Twenty clean closed-loop runs of the latch policy at the cage door, an illustrative campaign",
        description="The nominal-plate cell at the guarded approach, base stopped; the zero-failure bound, "
                    "the runs and cell time a 99.9 percent target needs, and the contrasts it is taught with.",
        site=_Embodied.Scenario.WarehouseAisle,
        table=LatchScenarioTable,
        standoff=Q_(2.0, "m"),              # scenario assumption: base start distance from the door
        run=LatchDemonstrationCampaign.attempt,   # drive-in, stop, guarded unlatch and swing
        reset=LatchDemonstrationCampaign.reset,   # re-latch, return to the standoff, telemetry flush
        jam_reset=Q_(60.0, "s"),            # scenario assumption: clear a jammed pin, recalibrate, return
        operators=2,                        # scenario assumption: safety operators watching each run
        shift=Q_(8.0, "hour"),              # scenario assumption: one operator shift
        clean_runs=20,                      # scenario assumption: the campaign's clean runs
        confidence=0.95,                    # scenario assumption: one-sided confidence
        target_rate=1e-3,                   # scenario assumption: the 99.9 percent target
        estimate_target_rate=0.005,         # scenario assumption: the 99.5 percent budget estimate
        weak_policy_one_in=8,               # scenario assumption: a policy failing one run in eight
        recorded_failures=1,                # scenario assumption: one failure in the same campaign
        joint_cells=2,                      # scenario assumption: nominal and displaced-plate cells
        looks=5,                            # scenario assumption: independent 5 percent looks
        omissions=2,                        # scenario assumption: soft penalty contact, missing transport lag
        fallacy_runs=1000,                  # scenario assumption: the perfect-record fallacy
    )
    LatchSequentialTest = _st.SequentialTestScenario(
        name="Wald's sequential test for the latch campaign",
        description="Unacceptable 10 percent against acceptable 1 percent failure, alpha 0.05, beta 0.10; "
                    "the score after 25 clean runs and one failure.",
        p_unacceptable=0.10,                # scenario assumption
        p_acceptable=0.01,                  # scenario assumption
        alpha=0.05,                         # scenario assumption: false acceptance under H0
        beta=0.10,                          # scenario assumption: false rejection under H1
        trace_clean_runs=25,                # scenario assumption: clean runs before the first failure
    )
    LatchCheckpointComparison = _st.ArmComparisonScenario(
        name="Two latch checkpoints, 20 interleaved runs each: 20 of 20 against 16 of 20",
        description="Fisher's exact test, and the runs per arm that separate 0.90 from 0.80 at 80 percent power.",
        arm_runs=20,                        # scenario assumption
        successes_a=20,                     # scenario assumption
        successes_b=16,                     # scenario assumption
        alpha=0.05,                         # scenario assumption: two-sided
        power=0.80,                         # scenario assumption
        p_high=0.90,                        # scenario assumption
        p_low=0.80,                         # scenario assumption
    )
    HazardExposureWall = _st.HazardExposureScenario(
        name="Failure-free hours to bound hazards of 1e-4, 1e-6, and 1e-9 per hour",
        description="The exposure wall at 95 percent one-sided confidence, per single machine, against one "
                    "rig-year and a 100-rig fleet.",
        rates=[Q_(1e-4, "1/hour"),          # scenario assumption: an industrial site requirement
               Q_(1e-6, "1/hour"),          # scenario assumption: teaching target
               Q_(1e-9, "1/hour")],         # scenario assumption: aviation's catastrophic band
        confidence=0.95,                    # scenario assumption
        rigs=100,                           # scenario assumption: the fleet-testing fallacy
        rig_time=Q_(1.0, "year"),           # one rig running continuously for a year
    )
    WarehouseMissionExposure = _st.MissionExposureScenario(
        name="Failure-free retrieval missions that bound 1e-4 per hour on the warehouse manipulator",
        description="Missions of 360 s at the drive limit with 120 s resets; calendar time and wheel revolutions.",
        machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
        hazard_rate=HazardExposureWall.rates[0],
        mission=Q_(360.0, "s"),             # scenario assumption: one retrieval mission
        reset=Q_(120.0, "s"),               # scenario assumption: dock recharge and telemetry reset
        confidence=0.95,                    # scenario assumption
        relaxed_mtbf=Q_(1000.0, "hour"),    # scenario assumption: the relaxed target of the checkpoint
    )
    WarehouseSupervisedMissions = _st.InterventionCampaignScenario(
        name="38 interventions in 100 supervised retrieval missions",
        description="Active hours, distance, MTBI, intervention density, and the Garwood 95 percent rate interval.",
        mission=WarehouseMissionExposure,
        missions=100,                       # scenario assumption
        interventions=38,                   # scenario assumption: interventions of the declared class
        alpha=0.05,                         # scenario assumption: two-sided interval
    )
    AisleHeadingDrift = _st.HeadingDriftScenario(
        name="A persistent 1 degree heading bias at the drive limit in the warehouse aisle",
        description="Open-loop angular error against where the closed loop spends the side clearance.",
        machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
        site=_Embodied.Scenario.WarehouseAisle,
        heading_offset=Q_(1.0, "degree"),   # scenario assumption: navigation heading bias
    )
    StiffSurfaceResidual = _st.StiffContactScenario(
        name="A 2 mm residual against a hypothetical 5e5 N/m stiff surface",
        description="The force a millimeter-scale tracking residual commands, and the penetrations at the "
                    "latch tripwire and limit.",
        site=_Embodied.Scenario.WarehouseAisle,
        stiffness=Q_(5e5, "N/m"),           # scenario assumption: a stiff surface, not the running machine's latch
        residual=Q_(2.0, "mm"),             # scenario assumption: tracking residual
    )
    LatchForceTail = _st.ForceTailScenario(
        name="A benign mean latch force with a damaging 99th-percentile peak",
        description="100 door trials: mean under the tripwire, tail past the structural limit.",
        site=_Embodied.Scenario.WarehouseAisle,
        trials=100,                         # scenario assumption
        mean_force=Q_(12.0, "N"),           # scenario assumption
        p99_force=Q_(115.0, "N"),           # scenario assumption: at an angular offset
    )
    RoadMilesToDemonstrate = _st.RoadExposureScenario(
        name="RAND's miles to demonstrate against Waymo's rider-only miles through January 2025",
        description="Kalra and Paddock 2016, Table 1, and Kusano et al. 2025 (millions of miles).",
        rates={"fatalities": 1.09,  # Kalra, N. and Paddock, S. M. (2016), Driving to Safety: How Many Mi...
               "injuries": 77,  # Kalra, N. and Paddock, S. M. (2016), Driving to Safety: How Many Mi...
               "crashes": 190},  # Kalra, N. and Paddock, S. M. (2016), Driving to Safety: How Many Mi...
        zero_failure={"fatalities": 275,  # Kalra, N. and Paddock, S. M. (2016), Driving to Safety: How Many Mi...
                      "injuries": 3.9,  # Kalra, N. and Paddock, S. M. (2016), Driving to Safety: How Many Mi...
                      "crashes": 1.6},  # Kalra, N. and Paddock, S. M. (2016), Driving to Safety: How Many Mi...
        precision={"fatalities": 8800,  # Kalra, N. and Paddock, S. M. (2016), Driving to Safety: How Many Mi...
                   "injuries": 125,  # Kalra, N. and Paddock, S. M. (2016), Driving to Safety: How Many Mi...
                   "crashes": 51},  # Kalra, N. and Paddock, S. M. (2016), Driving to Safety: How Many Mi...
        improvement={"fatalities": 11000,  # Kalra, N. and Paddock, S. M. (2016), Driving to Safety: How Many Mi...
                     "injuries": 161,  # Kalra, N. and Paddock, S. M. (2016), Driving to Safety: How Many Mi...
                     "crashes": 65},  # Kalra, N. and Paddock, S. M. (2016), Driving to Safety: How Many Mi...
        fleet_miles=56.7,  # Kusano, K. D. et al. (2025), Comparison of Waymo Rider-Only Crash R...
        confidence=0.95,                    # RAND row (1): 95 percent upper limit
    )
    AutosteerCrashRateDenominator = _st.CrashRateDenominatorScenario(
        name="NHTSA's Autosteer crash-rate drop against the complete-exposure cohort",
        description="Airbag-deployment crashes per million miles before and after installation (NHTSA PE 16-007; "
                    "Quality Control Systems 2019).",
        reported_before=1.3,  # NHTSA Office of Defects Investigation, ODI Resume, Preliminary Eval...
        reported_after=0.8,  # NHTSA Office of Defects Investigation, ODI Resume, Preliminary Eval...
        cohort_before=0.76,  # Quality Control Systems Corp. (Whitfield, R. A.), NHTSA's Implausib...
        cohort_after=1.21,  # Quality Control Systems Corp. (Whitfield, R. A.), NHTSA's Implausib...
        exact_exposure_share=13 / 100,  # Quality Control Systems Corp. (Whitfield, R. A.), NHTSA's Implausib...
    )

    # --- Sensor perception on the warehouse mobile manipulator (added 2026-10-20) ---
    # The machine's navigation camera, lidar, stage allowances, clearances, yaw
    # rate, and clock-conversion bound are in Embodied.Scenario.WarehouseAisle;
    # inputs neither record carries are stated here.
    WarehouseNavCameraAge = _st.ObservationAgeScenario(
        name="The warehouse machine's navigation-camera observation age at its drive limit",
        description="Exposure midpoint to dispatch, stage by stage, priced as travel at 1.5 m/s; "
                    "a four-frame temporal batch for comparison.",
        site=_Embodied.Scenario.WarehouseAisle,
        speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
        batch_frames=4,  # scenario assumption: a four-frame batch across capture times
    )
    WarehouseNavCameraRollingShutter = _st.RollingShutterScenario(
        name="Rolling-shutter skew of the warehouse machine's navigation camera at its drive limit",
        description="Row interval, along-track skew, and the sideways skew of the rack-end upright "
                    "while the base turns into the aisle.",
        site=_Embodied.Scenario.WarehouseAisle,
        speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
    )
    WarehouseNavCameraRollingShutterAisleSpeed = WarehouseNavCameraRollingShutter.variant(
        name="The same rolling-shutter skew at the chosen aisle speed",
        description="The drive-limit case with the speed lowered to the site's 1.3 m/s aisle speed.",
        speed=_Embodied.Scenario.WarehouseAisle.v_aisle,
    )
    WarehouseRackEndTimingAttitude = _st.TimingAttitudeErrorScenario(
        name="One millisecond and one degree of error at the warehouse rack end",
        description="Travel and misorientation per millisecond of misdating while turning, and the "
                    "sideways error per degree of attitude error at the 1.10 m rack end.",
        speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
        yaw_rate=_Embodied.Scenario.WarehouseAisle.yaw_rate_turn,
        distance=_Embodied.Scenario.WarehouseAisle.d_clear,
        timing_error=Q_(1.0, "ms"),  # scenario assumption: one millisecond of misdating
        angle_error=Q_(1.0, "degree"),  # scenario assumption: one degree of mount error
    )
    WarehouseLongAisleMountError = WarehouseRackEndTimingAttitude.variant(
        name="A one-degree mount error read 15 m down a long aisle",
        description="The same uncalibrated degree at a 15 m detection range, against the side clearance.",
        distance=Q_(15.0, "m"),  # scenario assumption: detection range down a long aisle
        reference=_Embodied.Scenario.WarehouseAisle.side_clearance,
    )
    WarehouseLidarExtrinsicShift = WarehouseRackEndTimingAttitude.variant(
        name="A 0.005 rad extrinsic shift after a shock, at a 10 m target",
        description="A small lidar-to-base rotation after a bump, against the localization allowance.",
        distance=Q_(10.0, "m"),  # scenario assumption: target range
        angle_error=Q_(0.005, "radian"),  # scenario assumption: extrinsic shift after a shock
        reference=_Embodied.Scenario.WarehouseAisle.delta_loc,
    )
    WarehouseRackEndLidarAge = _st.LidarSweepAgeScenario(
        name="The warehouse machine's lidar at its drive limit: one sweep, and the rack-end sector's age",
        description="Travel and turn during one 20 Hz sweep; the aisle sector at the rack end streamed "
                    "as it arrives or held for the full sweep, against the camera's age.",
        site=_Embodied.Scenario.WarehouseAisle,
        speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
    )
    WarehouseStereoRigDepthError = _st.StereoDepthErrorScenario(
        name="An illustrative stereo rig against the warehouse aisle's clearances",
        description="One-sigma depth error at 2 m, 10 m, and the rack end; the ranges where it spends "
                    "the localization allowance and the side clearance; a baseline change's bias at 20 m.",
        site=_Embodied.Scenario.WarehouseAisle,
        baseline=Q_(0.20, "m"),  # scenario assumption: stereo baseline
        focal_length_px=800.0,  # scenario assumption: focal length in pixels
        disparity_sigma_px=0.5,  # scenario assumption: disparity standard deviation in pixels
        near_range=Q_(2.0, "m"),  # scenario assumption
        far_range=Q_(10.0, "m"),  # scenario assumption
        ray_clearance=Q_(150.0, "mm"),  # scenario assumption: clearance kept along the viewing ray
        baseline_change=Q_(0.1, "mm"),  # scenario assumption: unmodeled baseline change
        bias_range=Q_(20.0, "m"),  # scenario assumption
    )
    VisualServoDelayPhaseMargin = _st.DelayPhaseMarginScenario(
        name="A visual-servo loop with 45 degrees of margin at 10 rad/s, closed through 60 ms of perception",
        description="The pure delay that erases the margin, and the share a 60 ms exposure, transport, "
                    "and inference age spends.",
        phase_margin=Q_(45.0, "degree"),  # scenario assumption: nominal phase margin
        crossover=Q_(10.0, "rad/s"),  # scenario assumption: gain crossover
        delay=Q_(60.0, "ms"),  # scenario assumption: exposure, transport, and inference
    )
    ClearanceInsetConformalRank = _st.ConformalInsetScenario(
        name="A 99th-percentile clearance inset from 1,000 measured residuals",
        description="The split-conformal rank for a one percent miss rate, and the fewest residuals that state it.",
        alpha=0.01,  # scenario assumption: tail mass for the clearance inset
        calibration_count=1000,  # scenario assumption: calibration residuals
    )
    WarehouseLidarClockConversion = _st.ClockConversionScenario(
        name="The warehouse machine's lidar clock between one-second syncs",
        description="Drift of the lidar's free-running clock between syncs, the producer's dispatch limit, "
                    "and the conversion bound's travel at the drive limit.",
        site=_Embodied.Scenario.WarehouseAisle,
        drift_rate=_Embodied.MobileManipulator.WarehouseMobileManipulator.lidar.internal_clock_drift,
        sync_interval=Q_(1.0, "s"),  # scenario assumption: sync interval
        speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
    )
    DenseFogVisibility = _st.VisibilityRangeScenario(
        name="Dense fog: the range at which contrast falls to the meteorological threshold",
        description="Koschmieder's law at an extinction coefficient of 0.05 per meter.",
        extinction=Q_(0.05, "1/m"),  # scenario assumption: dense-fog extinction coefficient
        contrast_threshold=_MOR_CONTRAST,
    )
    AluminumLinkWarming = _st.ThermalExpansionScenario(
        name="A 1 m aluminum link warmed by 10 K",
        description="Length change that shifts a calibrated sensor mount.",
        coefficient=_ALUMINUM_ALPHA,
        length=Q_(1.0, "m"),  # scenario assumption: link length
        temperature_rise=Q_(10.0, "K"),  # scenario assumption: uniform warming
    )

    # ── Model training on one machine (added 2026-10-20) ─────────────────────
    # GPT-2 XL trained on one H100: the run's time, energy, and rent; one step's
    # arithmetic and memory; and the input, checkpoint, and fine-tuning cases
    # beside it.
    # --- Data selection: corpus repetition, deduplication, coresets, echoing, labels, and cost (added 2026-10-20) ---
    # GPT-2 XL on WebText carries the data wall and deduplication; ResNet-50 on
    # ImageNet on one H100 carries coresets, the selection inequality, the data
    # path, and echoing. Cited figures come from Models, Hardware, Datasets,
    # Systems, Infrastructure, and Literature; the rest are stated scenario
    # assumptions of the data-selection examples.

    # Gpt2XlRunOnOneH100 and Gpt2XlRunOnOneDgxH100 carry the same names and
    # inputs as the training lecture's entries (feat/numbers-v1-08), which add
    # optional step and price fields to ClusterTrainingTimeScenario.
    Gpt2XlRunOnOneH100 = _st.ClusterTrainingTimeScenario(
        name="GPT-2 XL's training run on one H100",
        description="1.5e21 FLOPs at the H100's dense 16-bit peak and 50 percent utilization, in steps of "
                    "32 sequences of 1,024 tokens, at the H100's TDP and a cloud rental price.",
        workload=_Models.Language.GPT2,
        hardware=_Hardware.Cloud.H100,
        n_devices=1,
        efficiency=float(_Literature.Training.MfuHigh),
        tokens_per_step=32 * _Models.Language.GPT2.context_length,  # scenario assumption: 32 sequences per step
        hourly_price=_Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
    )
    Gpt2XlBatch32Block = Gpt2XlBlockFullWindow.variant(
        name="GPT-2 XL decoder blocks, one training step of 32 sequences at its 1,024-token window",
        description="The step counted layer by layer, attention products included, against the 6P rule.",
        batch_size=Gpt2XlRunOnOneH100.tokens_per_step // _Models.Language.GPT2.context_length,
    )
    Gpt2XlBatch32Activations = _st.ActivationMemoryScenario(
        name="GPT-2 XL's saved activations at batch 32 and 1,024 tokens, 16-bit",
        description="Korthikanti et al. per-layer bytes with no recomputation, and the per-layer line items "
                    "(projections, feed-forward expansion, attention scores, normalization).",
        workload=_Models.Language.GPT2,
        micro_batch=Gpt2XlBatch32Block.batch_size,
        seq_len=_Models.Language.GPT2.context_length,
        normalization_per_layer=Q_(10, "MB"),  # scenario assumption: normalization buffers per layer
    )
    Gpt2XlBatch32ActivationsFp32 = Gpt2XlBatch32Activations.variant(
        name="GPT-2 XL's saved activations at batch 32 and 1,024 tokens, FP32",
        description="The same tensors stored in 32 bits.",
        precision_bytes=_BYTES_FP32.to("byte").magnitude,
    )
    Gpt2Fp32AdamBatch32OnH100 = Gpt2Fp32AdamOnH100.variant(
        name="GPT-2 XL's FP32 Adam training step at batch 32 on one H100",
        description="FP32 weights, gradients, and Adam moments, plus FP32 activations at batch 32.",
        batch_size=Gpt2XlBatch32Block.batch_size,
        activations=Gpt2XlBatch32ActivationsFp32.activations("none"),
    )
    Gpt2MixedPrecisionAdamOnH100 = Gpt2Fp32AdamOnH100.variant(
        name="GPT-2 XL trained in mixed precision with Adam on one H100, batch 32",
        description="16-bit weights and gradients, an FP32 master copy, two FP32 Adam moments (16 B per "
                    "parameter), and 16-bit activations at batch 32 and 1,024 tokens.",
        bytes_per_value=_BYTES_FP16,
        master_bytes_per_parameter=_BYTES_FP32,
        batch_size=Gpt2XlBatch32Block.batch_size,
        activations=Gpt2XlBatch32Activations.activations("none"),
    )
    Llama2_7BMixedPrecisionAdam = Gpt2MixedPrecisionAdamOnH100.variant(
        name="A 7B model trained in mixed precision with Adam, against one H100",
        description="16 B per parameter before activations, against the 2 B per parameter it serves from.",
        workload=_Models.Language.Llama2_7B,
        batch_size=None,
        activations=None,
    )
    Llama3_8BBatch4Activations = _st.ActivationMemoryScenario(
        name="Llama 3.1 8B's saved activations at batch 4 and 2,048 tokens, 16-bit",
        description="Per-token tensors only: fused attention keeps no score matrices.",
        workload=_Models.Language.Llama3_8B,
        micro_batch=4,  # scenario assumption: sequences per step
        seq_len=2048,  # scenario assumption: tokens per sequence
    )
    Llama3_8BFullFineTuneOnH100 = Llama2_7BMixedPrecisionAdam.variant(
        name="Full fine-tuning of Llama 3.1 8B on one H100",
        description="Mixed-precision Adam on every parameter plus activations at batch 4 and 2,048 tokens.",
        workload=_Models.Language.Llama3_8B,
        batch_size=Llama3_8BBatch4Activations.micro_batch,
        activations=Llama3_8BBatch4Activations.per_token_activations,
    )
    Llama3_8BLoraOnH100 = Llama3_8BFullFineTuneOnH100.variant(
        name="Rank-16 LoRA fine-tuning of Llama 3.1 8B on one H100",
        description="Frozen 16-bit base weights; rank-16 adapters on every query and value projection carry "
                    "the mixed-precision Adam state; the same activations.",
        adapter_parameters=_Models.Language.Llama3_8B.layers * (
            _calc_low_rank_update_params(_Models.Language.Llama3_8B.hidden_dim,
                                         _Models.Language.Llama3_8B.hidden_dim, 16)  # scenario assumption: rank 16
            + _calc_low_rank_update_params(_Models.Language.Llama3_8B.hidden_dim,
                                           _Models.Language.Llama3_8B.kv_heads * (
                                               _Models.Language.Llama3_8B.hidden_dim
                                               // _Models.Language.Llama3_8B.heads), 16)),
    )
    ResNet50H100InputFeed = ResNet50A100InputFeed.variant(
        name="ResNet-50 training on one H100 of a DGX H100, fed by the host's loader workers",
        description="The H100 sustains the same fraction of its arithmetic ceiling as the A100; the host gives "
                    "each accelerator its share of the CPU cores as loader workers.",
        hardware=_Hardware.Cloud.H100,
        sustained_rate=(ResNet50A100InputFeed.sustained_fraction
                        * ResNet50A100InputFeed.variant(hardware=_Hardware.Cloud.H100).ceiling_rate),
        host_workers=_Nodes.DGX_H100.host_cpu_cores // _Nodes.DGX_H100.accelerators_per_node,
    )
    GilThreadedLoaderStall = _st.DataStallScenario(
        name="Python threads that hold the GIL preparing each batch",
        description="Host preparation of 250 ms and 12 ms of accelerator compute, run in sequence.",
        compute_time=Q_(12, "ms"),  # scenario assumption: accelerator time per batch
        io_time=Q_(250, "ms"),  # scenario assumption: serialized host preparation per batch
    )
    Gpt2XlRunOnOneDgxH100 = Gpt2XlRunOnOneH100.variant(
        name="GPT-2 XL's training run on the eight H100s of one DGX H100, scaling perfectly",
        description="The one-H100 run divided across one node's accelerators with no communication cost.",
        n_devices=_Nodes.DGX_H100.accelerators_per_node,
    )

    # --- Responsible AI (added 2026-10-20) -------------------------------------
    LoanScreeningTwoGroups = _st.GroupConfusionScenario(
        name="A loan-screening model scored on 100 applicants in each of two groups",
        description="Group A: 70 approved (40 repaid, 30 defaulted), 30 rejected (5 would repay); "
                    "Group B: 40 approved (30 repaid, 10 defaulted), 60 rejected (20 would repay). "
                    "Rejected applicants' repayment labels are hypothetical evaluation labels.",
        groups={  # scenario assumption: per-group confusion counts
            "A": {"tp": 40, "fp": 30, "fn": 5, "tn": 25},
            "B": {"tp": 30, "fp": 10, "fn": 20, "tn": 40},
        },
    )
    FraudFlagTwoGroups = _st.GroupConfusionScenario(
        name="A fraud model's flags on 1,000 records in each of two groups",
        description="Confusion counts for an audit of demographic parity and equal opportunity.",
        groups={  # scenario assumption: per-group confusion counts
            "A": {"tp": 450, "fp": 50, "fn": 30, "tn": 470},
            "B": {"tp": 180, "fp": 70, "fn": 120, "tn": 630},
        },
    )
    LoanApprovalGapTest = _st.ProportionGapTestScenario(
        name="The loan model's approval-rate gap, 0.70 against 0.40, tested at 100 and 10 per group",
        description="Pooled two-proportion z test and the gap's 95 percent interval.",
        rate_a=LoanScreeningTwoGroups.positive_rate("A"),
        rate_b=LoanScreeningTwoGroups.positive_rate("B"),
        n_per_group=round(LoanScreeningTwoGroups.size("A")),
        significance=0.05,                # scenario assumption: two-sided test level
        small_n_per_group=10,             # scenario assumption: a small-sample subgroup
    )
    CreditParityAccuracyCost = _st.ConstraintAccuracyCostScenario(
        name="Demographic parity imposed on a credit model: accuracy 85 to 81 percent",
        description="Holding Group A at 80 percent approval and raising Group B from 60 to 80 percent.",
        baseline_accuracy=85.0,           # scenario assumption
        constrained_accuracy=81.0,        # scenario assumption
        context={  # scenario assumption: the policy's stated rates, percent
            "group_a_default": 20, "group_b_default": 40, "default_threshold": 30,
            "group_a_approval": 80, "group_b_approval_unconstrained": 60, "default_threshold_parity": 50,
        },
    )
    FairnessMonitor10K = _st.SampledDecisionMonitorScenario(
        name="Fairness monitoring of a decision service at 10,000 decisions per second",
        description="1 percent of decisions logged at 200 bytes; 50 groups x 3 metrics at a 5 percent "
                    "false-alarm rate per metric.",
        request_rate=Q_(10_000, "1/s"),   # scenario assumption
        sample_share=0.01,                # scenario assumption
        record_size=Q_(200, "byte"),      # scenario assumption
        groups=50,                        # scenario assumption
        metrics_per_group=3,              # scenario assumption: FPR, TPR, calibration error
        false_alarm_rate=0.05,            # scenario assumption
    )
    InlineShapRetrofit = _st.InlineOverheadScenario(
        name="Sampled SHAP retrofitted inline on a 100 ms recommender",
        description="The approximate-SHAP inference overhead range added to every request.",
        base_latency=_ReferenceStats.RecommenderAtScaleProfile.P99LatencySlo,
        overhead_low=50.0 / 100,  # Responsible-AI technique overheads (accuracy, training, inference, ...
        overhead_high=200.0 / 100,  # Responsible-AI technique overheads (accuracy, training, inference, ...
        request_rate=Q_(10_000, "1/s"),   # scenario assumption
    )
    ImageExplanationLog = _st.ExplanationLogScenario(
        name="Logging an explanation with each of a million image predictions a day",
        description="1 to 10 MB per attribution map, or 4 to 8 bytes per feature for tabular SHAP.",
        predictions_per_day=1_000_000,    # scenario assumption
        days_per_month=_DAYS_PER_MONTH,
        map_size_low=Q_(1, "MB"),         # scenario assumption
        map_size_high=Q_(10, "MB"),       # scenario assumption
        attribution_bytes_low=Q_(4, "byte"),   # scenario assumption
        attribution_bytes_high=Q_(8, "byte"),  # scenario assumption
    )
    Gpt3ShardedUnlearning = _st.ShardedUnlearningScenario(
        name="Forgetting one person from a GPT-3-scale model: full retrain against 100 SISA shards",
        description="Full retrain on 1,024 A100s at about 32 percent MFU; 1,000 deletion requests a day.",
        workload=_Models.Language.GPT3,
        accelerator=_Hardware.Cloud.A100,
        n_accelerators=_Clusters.Training_1K_A100.total_accelerators,
        mfu=0.32,                         # scenario assumption: MFU of the retrain
        full_cost=_ReferenceStats.TrainingCostAnchors.Gpt3CostEstimate,
        shards=100,                       # scenario assumption
        requests_per_day=1000,            # scenario assumption
        accuracy_drop_low=0.03,           # scenario assumption
        accuracy_drop_high=0.07,          # scenario assumption
    )
    RewardModelLabelBudget = _st.PreferenceLabelBudgetScenario(
        name="Preference labels for a reward model: 50,000 to 500,000 comparisons",
        description="At 0.50 to 5 dollars per comparison.",
        comparisons_low=50_000,           # scenario assumption
        comparisons_high=500_000,         # scenario assumption
        price_low=Q_(0.50, "USD"),        # scenario assumption
        price_high=Q_(5.00, "USD"),       # scenario assumption
    )
    RadiologyAutomationBias = _st.AutomationBiasScenario(
        name="A radiologist reviewing an AI that misses 5 percent of tumors",
        description="Radiologist alone 92 percent, AI alone 95 percent; the radiologist accepts 70 "
                    "percent (range 60 to 80) of the AI's misses.",
        human_sensitivity=0.92,           # scenario assumption
        ai_sensitivity=0.95,              # scenario assumption
        miss_acceptance=0.70,             # scenario assumption
        acceptance_low=0.60,              # scenario assumption
        acceptance_high=0.80,             # scenario assumption
    )
    Gpt3PrivacyPlan = _st.DpTrainingPlanScenario(
        name="Differential privacy planned for a GPT-3-scale training run at three budgets",
        description="Assumed accountant results epsilon 1, 8, and 10; the epsilon-1 run needs three "
                    "times the training.",
        epsilon_strong=1.0,               # scenario assumption
        epsilon_moderate=8.0,             # scenario assumption
        epsilon_weak=10.0,                # scenario assumption
        base_cost=_ReferenceStats.TrainingCostAnchors.Gpt3CostEstimate,
        cost_multiplier=3.0,              # scenario assumption: training multiple at epsilon 1
        accuracy_drop_strong=6.0,         # scenario assumption
        accuracy_drop_moderate=1.0,       # scenario assumption
        clipped_share=40,                 # scenario assumption
        overhead_moderate=30.0,           # scenario assumption
    )

    # --- Fairness gates on a credit service and a ranking service (added 2026-10-20) -----
    # A constructed credit service whose two confusion matrices make every
    # fairness criterion computable, the price and stability of closing its
    # TPR gap, the data a per-group gate needs, published COMPAS rates against
    # Chouldechova's identity, and a ranking service's exploration slice.
    CreditServiceTwoGroups = _st.GroupConfusionScenario(
        name="A credit service scored on 10,000 Group A and 2,000 Group B applicants",
        description="Constructed confusion counts; half of each group would repay. Group A: TP 4,500, "
                    "FN 500, FP 1,000, TN 4,000; Group B: TP 600, FN 400, FP 200, TN 800.",
        groups={  # scenario assumption: per-group confusion counts
            "A": {"tp": 4500, "fp": 1000, "fn": 500, "tn": 4000},
            "B": {"tp": 600, "fp": 200, "fn": 400, "tn": 800},
        },
    )
    CreditServiceEqualOpportunityPrice = _st.EqualOpportunityPriceScenario(
        name="Raising Group B's TPR to Group A's by moving its threshold",
        description="A repaid loan earns $1,500 and a default loses $3,000; the move raises Group B's "
                    "FPR by 20 points.",
        confusion=CreditServiceTwoGroups,
        group="B",
        reference_group="A",
        repay_margin=Q_(1_500, "dollar"),  # scenario assumption: lifetime margin on a repaid loan
        default_loss=Q_(3_000, "dollar"),  # scenario assumption: loss on a defaulted loan
        fpr_increase=0.20,                # scenario assumption: FPR rise that closes the TPR gap
    )
    CreditServiceBiasFeedback = _st.BiasFeedbackScenario(
        name="The credit service's mitigated TPR gap under quarterly retraining",
        description="A 3.0-point gap at launch and 3.3 points at the next audit against a 5-point gate; "
                    "sampling marginal denials refits the factor to 0.90.",
        launch_gap=3.0,                   # scenario assumption: residual gap after mitigation, points
        next_audit_gap=3.3,               # scenario assumption: gap at the next quarterly audit, points
        gate=5.0,                         # scenario assumption: release gate on the TPR gap, points
        sampled_factor=0.90,              # scenario assumption: refit factor with outcome sampling
        unmitigated_gap=100 * CreditServiceTwoGroups.tpr("A") - 100 * CreditServiceTwoGroups.tpr("B"),
        horizon_cycles=4,                 # one year of quarterly retraining
    )
    OnePercentGroupEvaluationBudget = _st.SubgroupEvaluationBudgetScenario(
        name="Test data for a group that is 1 percent of users, error rate within one point at 95 percent",
        description="Worst-case rate 0.5; the rounded target of 10,000 examples from the group; a gap "
                    "between two equal groups pinned to 3.5 points.",
        rate=0.5,                         # worst case for the binomial standard error
        half_width=0.01,                  # scenario assumption: one percentage point
        confidence=0.95,                  # scenario assumption
        group_share=0.01,                 # scenario assumption: the group's share of users
        target_per_group=10_000,          # scenario assumption: per-group target, rounded
        gap_half_width=0.035,             # scenario assumption: an illustrative slice's interval
    )
    FairnessAccuracyFrontierSketch = _st.AccuracyFrontierScenario(
        name="A hypothetical fairness-accuracy frontier with three candidate models",
        description="accuracy = 0.85 + 0.10 (1 - exp(-20 x)); A unconstrained at disparity 0.18, B at "
                    "0.05, C at strict equality.",
        floor=0.85,                       # scenario assumption: accuracy at zero disparity
        span=0.10,                        # scenario assumption
        rate=20.0,                        # scenario assumption
        candidates={"A": 0.18, "B": 0.05, "C": 0.0},  # scenario assumption
    )
    RecommenderExplorationExposure = _st.ExplorationExposureScenario(
        name="A ranking service: 10 million daily users, 20 items each, a 1 million item catalog",
        description="1 percent of slots served by randomized exploration.",
        daily_users=10_000_000,           # scenario assumption
        slots_per_user=20,                # scenario assumption
        catalog_items=1_000_000,          # scenario assumption
        explore_share=0.01,               # scenario assumption
    )

    # --- Durable execution (added 2026-10-20) ---------------------------------
    # One trajectory's crash-recovery costs: a group commit, full snapshots
    # against deltas, checkpoint strides on a fixed crash trace, Young's interval
    # for re-executed work, the cold prefill of a resumed context, a rerun's
    # reproduction over the step budget, and a year of recovery data.
    IntentQuorumGroupCommit = _st.GroupCommitScenario(
        name="Sixteen intents acknowledged by one quorum flush",
        description="A 2 ms batch timer gathers 16 intents; one 4 ms quorum acknowledgment covers "
                    "them all. Scenario timings, not measurements of a particular store.",
        batch_timer=Q_(2.0, "ms"),  # scenario assumption: longest wait before the batch starts
        flush_time=Q_(4.0, "ms"),  # scenario assumption: shared quorum acknowledgment
        batch_size=16,  # scenario assumption: intents per acknowledgment
    )
    GrowingTrajectorySnapshots = _st.SnapshotVolumeScenario(
        name="Full snapshots against deltas over a 500-turn trajectory",
        description="State grows 128 KB per turn and a checkpoint is taken every 5 turns.",
        turns=500,  # scenario assumption: trajectory length
        stride=5,  # scenario assumption: checkpoint every 5 turns
        state_growth=Q_(128, "kB"),  # scenario assumption: state growth per turn, decimal kilobytes
    )
    TwoCrashCheckpointTrace = _st.CheckpointTraceScenario(
        name="Checkpoint strides of 10, 25, and 50 turns on a 100-turn trace with two crashes",
        description="Crashes after turns 37 and 83; a 0.5 s pause per checkpoint, 1 s restore per "
                    "crash, 0.1 s to rebuild one logged turn, 20 s to re-execute an unlogged one.",
        turns=100,  # scenario assumption
        crashes=[37, 83],  # scenario assumption: after these completed turns
        strides=[10, 25, 50],  # scenario assumption: the three policies compared
        pause=Q_(0.5, "s"),  # scenario assumption: serial pause including durable publication
        restore=Q_(1.0, "s"),  # scenario assumption: per crash
        rebuild_per_turn=Q_(0.1, "s"),  # scenario assumption: read, validate, fold one logged turn
        reexecute_per_turn=Q_(20.0, "s"),  # scenario assumption: re-run one turn's model call and tools
    )
    ReexecutedWorkCheckpointInterval = _st.CheckpointIntervalScenario(
        name="Young interval for re-executed agent work: a 2 s pause, a failure every 10 minutes",
        description="Work lost to a crash must be re-executed; the pause includes durable publication.",
        write_time=Q_(2.0, "s"),  # scenario assumption: pause per checkpoint
        mtbf=Q_(10, "minute"),  # scenario assumption: mean time between work-losing failures
        miss_factors=[2, 10],  # scenario assumption: an interval off by 2 and by an order of magnitude
    )
    ResumedContextColdPrefill = _st.ModelCallScenario(
        name="Cold prefill of a 32,768-token resumed context",
        description="No prefix state survives the resume; the whole recovered context is "
                    "prefilled at the 8-bit compute floor before the next reply.",
        floors=Llama3_70B_Int8DecodeH100,
        prompt_tokens=32_768,  # scenario assumption: recovered context of a long trajectory
        output_tokens=0,  # the floor before the first output token
    )
    GreedyRerunReproduction = HorizonCompounding30Steps.variant(
        name="A greedy rerun that reproduces each recorded reply 99 percent of the time",
        description="Each call of a temperature-zero rerun independently matches its recording; the "
                    "rerun reproduces only if all calls over the coding-agent step budget match.",
        step_success=0.99,  # scenario assumption: per-call reproduction rate
    )
    RepairTrajectoryRetention = _st.RetentionCostScenario(
        name="A year of recovery and replay data for one repair trajectory",
        description="Thirty turns of 4 KB event envelopes and 128 KB of delivered inputs and "
                    "results, one 50 MB workspace snapshot, kept 12 months in standard object "
                    "storage, against the cost of one repair attempt.",
        turns=_Agents.Coding.SWE_Bench_Runner.max_trajectory_steps,
        envelope=Q_(4, "kB"),  # scenario assumption: event envelopes per turn
        payload=Q_(128, "kB"),  # scenario assumption: delivered inputs and result bodies per turn
        workspace=Q_(50, "MB"),  # scenario assumption: one immutable workspace snapshot
        rate=_Infrastructure.Pricing.Storage.S3StandardPerTbMonth.rate,
        period=Q_(12, "month"),  # scenario assumption: promised replay horizon
        attempt_cost=RepairAttemptCostLevers.model_cost + RepairAttemptCostLevers.other_cost,
    )

    # --- Failure recovery (added 2026-10-20) ---------------------------------
    # The repair agent's temporary test environment and the remediation agent's
    # fleet change. Durations, token counts, and prices are teaching
    # assumptions at the hypothetical $3 and $15 per million tokens used across
    # the agent scenarios; the remediation risk comes from its agent profile.
    TestConfigRepairVsRebuild = _st.RepairBreakEvenScenario(
        name="One bounded repair of a rejected test configuration against compensating and rebuilding",
        description="Compensation 40.6 s and rebuild 46 s with 38,500 input and 3,100 output tokens, "
                    "against one isolated repair call of 2,390 input and 180 output tokens: 0.09 s "
                    "before output, 50 tokens/s, 0.4 s of validation; repair success assumed 0.5.",
        unwind_time=Q_(40.6, "second"),  # scenario assumption: compensation
        rebuild_time=Q_(46.0, "second"),  # scenario assumption: includes obtaining and validating a configuration
        rebuild_input=38_500,  # scenario assumption: rebuild model input tokens
        rebuild_output=3_100,  # scenario assumption: rebuild model output tokens
        repair_input=2_390,  # scenario assumption: entire isolated repair request
        repair_output=180,  # scenario assumption: repair output tokens
        startup_time=Q_(0.09, "second"),  # scenario assumption: pre-output startup, not time to first token
        output_rate=Q_(50, "token / second"),  # scenario assumption: decode rate
        validation_time=Q_(0.4, "second"),  # scenario assumption: local validation on either outcome
        input_price=OutputCeilingThreeAllocations.input_price,
        output_price=ProbeVsGeneratedExplanation.output_price,
        assumed_success=0.5,  # scenario assumption: policy comparison, not a measurement
    )

    RemediationCanaryPivotLoss = _st.IrreversibleLossScenario(
        name="The remediation agent's fleet rollout with canary checks before the pivot",
        description="Wrong-change probability, impact, canary catch rate, and revert fidelity from the "
                    "on-call remediation profile; attempt cost is its model, environment, health-check, "
                    "and approver costs; sensitivity to ten more points of catch rate.",
        workload=_Agents.Remediation.OnCall,
        catch_step=0.10,  # scenario assumption: catch-rate increment for the sensitivity
    )

    RepairLoopToTurnCeiling = _st.UndetectedLoopScenario(
        name="An append-only repair loop run to a 50-turn ceiling or stopped after turn 4",
        description="A 1,000-token first input; each turn appends 300 output tokens and an 800-token "
                    "error observation; 10 s per turn; a 64,000-token combined capacity; no "
                    "compaction or prefix caching.",
        first_input=1_000,  # scenario assumption: first looping input, tokens
        output_per_turn=300,  # scenario assumption
        observation_per_turn=800,  # scenario assumption: appended error observation
        turn_time=Q_(10, "second"),  # scenario assumption: complete serial turn
        ceiling=50,  # scenario assumption: turn ceiling counted from the loop's start
        stop_after=4,  # scenario assumption: intervention after this completed turn
        context_window=64_000,  # scenario assumption: combined input and output capacity
        input_price=OutputCeilingThreeAllocations.input_price,
        output_price=ProbeVsGeneratedExplanation.output_price,
    )

    ThreeLayerRetryAmplification = _st.RetryAmplificationScenario(
        name="Three retry layers, each with three attempts, under one failing call",
        description="The trajectory's recovery loop, an API gateway, and a client library, each "
                    "allowing one try plus two retries on its own allowance.",
        layer_names=["agent loop", "gateway", "client library"],  # from the caller down
        attempts_per_layer=3,  # scenario assumption: one try plus two retries
    )

    BlockingCallBulkhead = _st.BulkheadAdmissionScenario(
        name="256 blocking calls to a stalled tool against 300 workers, capped at 64 and 32 queued",
        description="One burst, no completions during admission, no nested calls.",
        arrivals=256,  # scenario assumption: trajectories each submitting one blocking call
        workers=300,  # scenario assumption: shared pool
        worker_cap=64,  # scenario assumption: dependency cap on occupied workers
        queue_cap=32,  # scenario assumption: dependency queue
    )

    RetryCohortJitter = _st.RetryJitterScenario(
        name="128 retries released together: inside 2 ms, or jittered over 2 s",
        description="No later retries and no unrelated traffic.",
        retries=128,  # scenario assumption: eligible retries released at one time
        burst_window=Q_(2, "ms"),  # scenario assumption: stipulated synchronized window
        jitter_span=Q_(2, "second"),  # scenario assumption: uniform jitter span
    )




    # --- Agent evaluation (added 2026-10-20) ----------------------------------
    # What graded trials support about the repair agent's candidate_v2 against
    # baseline_v1, and the leaderboards and judges such claims rest on. Counts,
    # rates, and prices are teaching assumptions unless a Literature series
    # supplies them; the solvers live in engine/solvers/acceptance.py.
    RecurringTaskConsistency = _st.HorizonCompoundingScenario(
        name="A recurring task accepted on 90 percent of single runs, over eight recurrences",
        description="pass^k is horizon compounding over recurrences: all eight runs pass with 0.9^8, "
                    "and holding 90 percent over eight needs a single-run rate of 0.9^(1/8).",
        step_success=0.90,  # scenario assumption: hypothetical single-run acceptance
        short_horizon=1,
        long_horizon=8,  # scenario assumption: recurrences the claim covers
        target_success=0.90,  # scenario assumption: consistency the recurring work must hold
    )
    RecurringTaskConsistencyLowerRate = RecurringTaskConsistency.variant(
        name="The same recurring task at 80 percent per run",
        description="Eight recurrences at an 80 percent single-run rate.",
        step_success=0.80,  # scenario assumption
    )
    RecurringTaskConsistencyHalvedFailure = RecurringTaskConsistency.variant(
        name="The same recurring task with its single-run failure rate halved",
        description="Eight recurrences at a 95 percent single-run rate.",
        step_success=0.95,  # scenario assumption: failure rate halved from 10 to 5 percent
    )
    RefundAnswerFiveSessions = RecurringTaskConsistency.variant(
        name="One refund-policy answer asked in five sessions at 95 percent per run",
        description="Consistency over five sessions and the single-run rate that holds 90 percent.",
        step_success=0.95,  # scenario assumption: checkpoint single-run rate
        long_horizon=5,  # scenario assumption: sessions
    )
    IndependentJudgesNinetyPercent = _st.AnnotatorAgreementScenario(
        name="Two judges that each pass 90 percent of reports independently of the reports",
        description="Observed agreement equals chance agreement, so kappa is zero.",
        class_shares=[0.90, 0.10],  # scenario assumption: shared marginal pass rate
        independent_raters=True,
    )
    IndependentJudgesEightyPercent = IndependentJudgesNinetyPercent.variant(
        name="Two judges that each pass 80 percent of reports independently of the reports",
        description="The same independence at an 80 percent marginal pass rate.",
        class_shares=[0.80, 0.20],  # scenario assumption
    )
    FiftyTaskIntervals = _st.BinomialIntervalsScenario(
        name="Wald and Wilson intervals at 50 independent tasks",
        description="0, 2, and 25 accepted (or violating) tasks of 50 at 95 percent confidence.",
        trials=50,  # scenario assumption: the size of many agent safety suites
        successes=[0, 2, 25],
    )
    AcceptanceRatePlanning = _st.ProportionPlanningScenario(
        name="Tasks one acceptance rate and a two-arm comparison need",
        description="Half-width at p = 0.5 and 100 tasks; tasks for a 2-point half-width; tasks per "
                    "independent arm to detect 4 and 2 points at 5 percent significance and 80 percent power.",
        rate=0.5,  # worst case for a proportion's variance
        planning_tasks=100,  # scenario assumption
        target_half_width=0.02,  # scenario assumption
        differences=[0.04, 0.02],  # scenario assumption: detectable differences
        significance=0.05,
        power=0.80,
        z_alpha=1.96,  # rounded two-sided 5 percent quantile, as planning tables print it
        z_beta=0.84,  # rounded 80 percent power quantile
    )
    RepairCandidatePairedComparison = _st.PairedComparisonScenario(
        name="candidate_v2 against baseline_v1 on 150 paired repair tasks",
        description="One trial of each on every task: both accepted 70, candidate only 18, baseline "
                    "only 12, neither 50; a 2-point noninferiority tolerance.",
        both=70,  # scenario assumption: illustrative complete verdicts
        candidate_only=18,  # scenario assumption
        baseline_only=12,  # scenario assumption
        neither=50,  # scenario assumption
        tolerance=0.02,  # scenario assumption: acceptable loss chosen before outcomes
    )
    TwoTaskTypeConsistency = _st.TaskMixConsistencyScenario(
        name="Two equally likely task types passing on 50 and 90 percent of runs",
        description="One task repeated twice against two independently drawn tasks.",
        task_pass_rates=[0.5, 0.9],  # scenario assumption
        repetitions=2,
    )
    CappedCandidateSelection = _st.CappedSelectionCostScenario(
        name="Batch against sequential selection of up to three candidates",
        description="A fixed task passing on 40 percent of candidates, $0.03 to generate and $0.01 to "
                    "check each, under an exact checker.",
        pass_rate=0.4,  # scenario assumption
        cap=3,  # scenario assumption
        generation_cost=Q_(0.03, "USD"),  # scenario assumption: illustrative price
        check_cost=Q_(0.01, "USD"),  # scenario assumption: illustrative price
    )

    # --- Spatial memory on the warehouse mobile manipulator (added 2026-10-20) ---
    # The machine, clearances, walking speed, conveyor, grasp record, and
    # observation age are in Embodied.Scenario.WarehouseAisle; the cited
    # odometry, mapping, and tracking figures in published record and
    # Inputs neither record carries are stated here.
    WarehouseRackEndRangeFusion = _st.RangeFusionScenario(
        name="The range to the warehouse rack-end upright, fused from lidar and stereo",
        description="One sweep's prediction, a lidar update, a stereo update, and the stereo update "
                    "misdated by the navigation camera's arrival stamp at the drive limit.",
        site=_Embodied.Scenario.WarehouseAisle,
        stereo=WarehouseStereoRigDepthError,
        age=WarehouseNavCameraAge,
        prior_sigma=Q_(20.0, "mm"),  # scenario assumption: posterior from the previous sweep
        speed_sigma=Q_(0.1, "m/s"),  # scenario assumption: odometry speed disturbance (wheel slip)
        comparison_scale=2.0,  # scenario assumption: the stereo spread doubled, for comparison
    )
    RackEndMisdatedUpdateConsistency = _st.EstimatorConsistencyScenario(
        name="Consistency tests for the rack-end range estimator",
        description="The 95 percent band for mean NEES over 50 qualification runs, and the 95 and 99 percent "
                    "NIS gates against one misdated stereo update.",
        fusion=WarehouseRackEndRangeFusion,
        runs=50,  # scenario assumption: qualification runs with ground truth
        confidence=0.95,  # scenario assumption: two-sided acceptance level
        gate_confidences=[0.95, 0.99],  # scenario assumption: single-update gate levels
    )
    WarehouseObservationReplayBuffer = _st.ReplayBufferScenario(
        name="Out-of-sequence replay buffer for the warehouse navigation camera's observation age",
        description="Slots at the base IMU's rate that reach back one observation age.",
        site=_Embodied.Scenario.WarehouseAisle,
    )
    WarehouseImuOneSecondCoast = _st.InertialCoastScenario(
        name="The warehouse machine's IMU coasting one second without a range fix",
        description="Heading and position error from white noise and from the datasheet offsets, "
                    "with the heading offset seen at the rack-end upright.",
        site=_Embodied.Scenario.WarehouseAisle,
        duration=Q_(1.0, "s"),  # scenario assumption: interval without a range update
    )
    WarehouseUmbmarkOdometryDrift = _st.OdometryDriftScenario(
        name="UMBmark odometry drift against the warehouse localization bound",
        description="Borenstein and Feng's return errors before and after calibration as drift rates, "
                    "priced against the 50 mm bound at the aisle speed and over one aisle.",
        site=_Embodied.Scenario.WarehouseAisle,
        square_side=Q_(4, "m"),  # Borenstein, J. and Feng, L. (1996), Measurement and Correction of S...
        legs=4,  # Borenstein, J. and Feng, L. (1996), Measurement and Correction of S...
        return_error_uncalibrated=Q_(317, "mm"),  # Borenstein, J. and Feng, L. (1996), Measurement and Correction of S...
        return_error_calibrated=Q_(21, "mm"),  # Borenstein, J. and Feng, L. (1996), Measurement and Correction of S...
        run_sigma=Q_(25, "mm"),  # Borenstein, J. and Feng, L. (1996), Measurement and Correction of S...
        test_speed=Q_(0.2, "m/s"),  # Borenstein, J. and Feng, L. (1996), Measurement and Correction of S...
    )
    LatchSlotFingerBelief = _st.BeliefExpiryScenario(
        name="The finger's position in the cage latch-release slot while the finger hides it",
        description="Initial error, unobserved drift, and disturbance acceleration against the slot tolerance.",
        initial_error=Q_(0.1, "mm"),  # scenario assumption: finger-to-slot error at the wrist capture
        growth_rate=Q_(12.0, "mm/s"),  # scenario assumption: unobserved lateral velocity bound
        growth_acceleration=Q_(50.0, "mm/s**2"),  # scenario assumption: disturbance acceleration bound
        tolerance=Q_(1.0, "mm"),  # scenario assumption: latch-release slot tolerance
    )
    LatchSlotFingerBeliefTightSlot = LatchSlotFingerBelief.variant(
        name="The same finger belief in a slot machined to 0.30 mm",
        description="A tighter slot; the fixed initial error takes a larger share of it.",
        tolerance=Q_(0.30, "mm"),  # scenario assumption: tighter machined slot
    )
    LatchSlotStaleWristTransform = _st.StaleTransformScenario(
        name="A one-frame-stale camera-to-slot edge composed with the current wrist pose",
        description="Drift and lever-arm swing over one wrist-camera frame while the wrist aligns the finger.",
        site=_Embodied.Scenario.WarehouseAisle,
        belief=LatchSlotFingerBelief,
        lever=Q_(0.15, "m"),  # scenario assumption: wrist-to-fingertip lever arm
        angular_rate=Q_(0.30, "rad/s"),  # scenario assumption: wrist rate while aligning the finger
    )
    ConveyorMugHandleBelief = _st.BeliefExpiryScenario(
        name="The mug handle's pose on the warehouse takeaway conveyor",
        description="Grasp-record initial error growing at the belt speed against the grasp tolerance, "
                    "admitted 35 ms after capture under the clock-conversion bound.",
        initial_error=_Embodied.Scenario.WarehouseAisle.grasp_initial_error,
        growth_rate=_Embodied.Scenario.WarehouseAisle.v_conveyor,
        tolerance=_Embodied.Scenario.WarehouseAisle.grasp_tolerance,
        use_age=Q_(35.0, "ms"),  # scenario assumption: capture-to-use age on the consumer's clock
        age_bound=_Embodied.Scenario.WarehouseAisle.clock_conversion_bound,
    )
    CoworkerBesideArmBelief = _st.BeliefExpiryScenario(
        name="A coworker tracked beside the warehouse arm",
        description="Position error at the last observation growing at the walking speed against the "
                    "tolerance to the arm's swept volume.",
        initial_error=Q_(15.0, "mm"),  # scenario assumption: coworker position error at the last observation
        growth_rate=_Embodied.Scenario.WarehouseAisle.v_human_approach,
        tolerance=Q_(150.0, "mm"),  # scenario assumption: belief tolerance to the arm's swept volume
    )
    CoworkerBesideArmBeliefTightTolerance = CoworkerBesideArmBelief.variant(
        name="The same coworker belief with a 100 mm tolerance",
        description="A tighter swept-volume tolerance.",
        tolerance=Q_(100.0, "mm"),  # scenario assumption: tighter swept-volume tolerance
    )
    WarehouseRackEndFreeSpace = _st.FreeSpaceExpiryScenario(
        name="Free space at the warehouse rack end, ahead and abreast, eroding at walking speed",
        description="The camera frame's free-space claim ahead (clear distance) and beside the rack end "
                    "(side clearance); what the observation age leaves; a walker during a stop; and a "
                    "single 100 ms staleness timeout for the whole robot.",
        site=_Embodied.Scenario.WarehouseAisle,
        uniform_timeout=Q_(100.0, "ms"),  # scenario assumption: one staleness timeout for every belief
    )
    ArmSensorEpochGap = _st.EpochGapScenario(
        name="Three arm sensors latching at different epochs in one cycle",
        description="Tactile, camera, and encoder epochs; the camera-to-encoder gap at the arm's Cartesian limit.",
        epochs={"tactile": Q_(12.0, "ms"),  # scenario assumption: tactile array latches force
                "camera": Q_(18.5, "ms"),  # scenario assumption: camera exposure midpoint
                "encoders": Q_(21.0, "ms")},  # scenario assumption: encoders sample angles
        pair=["camera", "encoders"],
        speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.arm.cartesian_velocity_limit,
    )
    WarehouseSkewedRecordPair = _st.SkewedRecordsScenario(
        name="A person detection and current odometry 20 ms apart, compared at the warehouse drive limit",
        description="Closing speed of the base and a walker, the envelope across the skew, and the slack "
                    "left in a 200 mm separation.",
        site=_Embodied.Scenario.WarehouseAisle,
        skew=Q_(20.0, "ms"),  # scenario assumption: acquisition-time gap between the two records
        separation=Q_(200.0, "mm"),  # scenario assumption: available separation
    )
    WristForceBiasDrift = _st.SensorBiasDriftScenario(
        name="A wrist strain gauge whose bias drifts with nearby joint heat",
        description="A steady 5 N reading while the bias drifts 0.2 N per minute for 15 minutes.",
        reading=Q_(5.0, "N"),  # scenario assumption: steady strain-gauge reading
        reported_sigma=Q_(0.05, "N"),  # scenario assumption: the estimator's reported bound
        drift_rate=Q_(0.2, "N/min"),  # scenario assumption: thermoelastic bias drift
        span=Q_(15.0, "min"),  # scenario assumption: drift interval
        excitation=Q_(40.0, "ms"),  # scenario assumption: deliberate contact break
    )
    ConveyorMugFrozenTransform = _st.FrozenTransformScenario(
        name="The mug's last transform republished after the wrist camera is lost",
        description="Fresh stamps every millisecond pass a 5 ms freshness check while the belt carries "
                    "the mug for 100 ms.",
        site=_Embodied.Scenario.WarehouseAisle,
        republish_period=Q_(1.0, "ms"),  # scenario assumption: middleware republish period
        freshness_threshold=Q_(5.0, "ms"),  # scenario assumption: publish-time freshness threshold
        loss_time=Q_(10.0, "ms"),  # scenario assumption: the camera stops producing evidence
        window=Q_(100.0, "ms"),  # scenario assumption: elapsed time with the frozen transform
    )
    WarehousePermissionClearanceQuery = _st.ClearanceQueryScenario(
        name="Sphere-hull clearance queries on the warehouse permission controller",
        description="The 100 us collision allocation over 50 hulls, the SRAM read budget, reads per "
                    "representation at 2, 5, and 10 cm, and the stopping-corridor snapshot.",
        site=_Embodied.Scenario.WarehouseAisle,
        collision_allocation=Q_(100.0, "us"),  # scenario assumption: collision checks within the permission tick
        hulls=50,  # scenario assumption: sphere hulls covering the machine
        fewest_hulls=20,  # scenario assumption: fewest evaluation spheres considered
        hull_radius=Q_(0.25, "m"),  # scenario assumption: sphere-hull radius
        esdf_cell_bytes=Q_(8, "B"),  # scenario assumption: 4 B distance, 2 B weight, 2 B evidence epoch
        occupancy_cell_bytes=Q_(4, "B"),  # scenario assumption: 2 B log-odds, 2 B evidence epoch
        map_edge=Q_(10.0, "m"),  # scenario assumption: host-map edge for the octree depth
        slab_height=Q_(2.0, "m"),  # scenario assumption: snapshot height covering the arm's reach
        pitches=[Q_(2, "cm"), Q_(5, "cm"), Q_(10, "cm")],  # scenario assumption: voxel pitches compared
        trilinear_reads=8,  # corner cells of one trilinear interpolation
    )
    WorkcellHostMapSizing = _st.HostMapSizingScenario(
        name="A 10 m by 10 m by 3 m workcell mapped at 1 cm",
        description="A dense voxel grid against a hashed 8-cubed-block band around 120 square meters of surface.",
        extent=[Q_(10.0, "m"), Q_(10.0, "m"), Q_(3.0, "m")],  # scenario assumption: workcell extent
        pitch=Q_(0.01, "m"),  # scenario assumption: manipulation resolution
        voxel_bytes=Q_(8, "B"),  # scenario assumption: distance, weight, and epoch payload
        surface_area=Q_(120.0, "m**2"),  # scenario assumption: obstacle surface area
        band_half_width=Q_(0.04, "m"),  # scenario assumption: truncation band half-width
        block_edge=8,  # voxels per block edge, the nvblox layout
    )
    WarehouseDepthRaymarchTraffic = _st.RaymarchTrafficScenario(
        name="One VGA depth stream ray-marched into a 2 cm map on the warehouse host SoC",
        description="Naive per-voxel read-modify-write traffic against the SoC's peak DRAM bandwidth.",
        site=_Embodied.Scenario.WarehouseAisle,
        width_px=640,  # scenario assumption: VGA depth stream
        height_px=480,  # scenario assumption: VGA depth stream
        frame_rate=Q_(30, "Hz"),  # scenario assumption: depth frame rate
        ray_length=Q_(2.0, "m"),  # scenario assumption: mean marched distance
        pitch=Q_(0.02, "m"),  # scenario assumption: voxel pitch
        bytes_per_step=Q_(8, "B"),  # scenario assumption: read-modify-write traffic per voxel, low end
    )
    WarehouseFoundationEncoderAtCameraRate = _st.EncoderRateScenario(
        name="A 3 TFLOP foundation encoder on every warehouse navigation frame",
        description="Sustained rate the encoder would need against the host SoC's dense FP16 peak, "
                    "and the semantic keyframe rates of the multi-rate alternative.",
        site=_Embodied.Scenario.WarehouseAisle,
        flop_per_frame=Q_(3.0, "TFLOP"),  # scenario assumption: encoder cost per frame
        keyframe_rate_low=Q_(1.0, "Hz"),  # scenario assumption: semantic keyframe rate, low end
        keyframe_rate_high=Q_(5.0, "Hz"),  # scenario assumption: semantic keyframe rate, high end
    )
    # --- Grounded intent: the intent lease and its gates (added 2026-10-20) ---
    ConveyorMugIntentCadence = _st.IntentCadenceScenario(
        name="The conveyor mug proposed by the warehouse machine's intent model",
        description="Untracked and tracked evidence horizons against the intent model's inference and "
                    "period, the oldest evidence behind a model-only renewal, a 200 ms host stall, and "
                    "the stop onset after expiry.",
        site=_Embodied.Scenario.WarehouseAisle,
        machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
        stall=Q_(200.0, "ms"),  # scenario assumption: a stalled proposer holds the mug target
    )
    ConveyorMugSettlingDrift = ConveyorMugIntentCadence.variant(
        name="The same mug sliding to rest at 20 mm/s",
        description="The untracked evidence horizon of a slow settling drift.",
        target_speed=Q_(20.0, "mm/s"),  # scenario assumption: settling drift
    )
    ConveyorMugSlowBelt = ConveyorMugIntentCadence.variant(
        name="The same mug on a belt slowed to 50 mm/s",
        description="Whether a model-only renewal fits the untracked horizon of a slower belt.",
        target_speed=Q_(50.0, "mm/s"),  # scenario assumption: slowed belt
    )
    ConveyorMugDynamicReach = _st.DynamicReachScenario(
        name="The mug handle 300 mm from the gripper at the arm's TCP speed ceiling",
        description="The transit lower bound against the tracked evidence window, the reach inside each "
                    "window, and the fresh captures a staged approach needs.",
        cadence=ConveyorMugIntentCadence,
        target_distance=Q_(300.0, "mm"),  # scenario assumption: gripper-to-handle distance at proposal
    )
    AisleCoworkerForwardOccupancy = _st.ForwardOccupancyScenario(
        name="A coworker's forward occupancy set over the mug's windows and a 2 s forecast",
        description="Growth of the speed-bound set at the ISO 13855 design speed, and the horizon past "
                    "which one person's set spans the aisle.",
        cadence=ConveyorMugIntentCadence,
        forecast=Q_(2.0, "s"),  # scenario assumption: a multi-second planning horizon
    )
    ConveyorMugIntentLeaseExpiry = _st.IntentLeaseExpiryScenario(
        name="The red-mug intent lease: evidence, host silence, and the stopping deadline",
        description="The arm's credible deceleration, its stop from the TCP ceiling, the stopping "
                    "deadline, and which term binds with and without a tracker.",
        cadence=ConveyorMugIntentCadence,
        administrative_lease=Q_(100.0, "ms"),  # scenario assumption: administrative lease (six wrist-camera frames)
        clearance=Q_(300.0, "mm"),  # scenario assumption: verified clearance ahead of the gripper
    )
    MugJarReferentSet = _st.ReferentSetScenario(
        name="A red mug and a red glass jar in the pick window",
        description="Normalized detector scores, their entropy, and the conformal prediction set.",
        labels=["mug", "jar"],
        scores=[0.62, 0.48],  # scenario assumption: raw per-box scores
        calibration_count=200,  # scenario assumption: labeled calibration scenes
        alpha=0.10,  # scenario assumption: allowed miscoverage
        threshold_score=0.60,  # scenario assumption: stands in for a measured calibration quantile
    )
    MugJarReferentSetLowJar = MugJarReferentSet.variant(
        name="The same pick window with the jar scoring 0.10",
        description="A low jar score leaves a singleton set.",
        scores=[0.62, 0.10],  # scenario assumption: the jar's raw score in the contrast case
    )
    DistantHandleConfidenceRegion = _st.ConfidenceRegionScenario(
        name="The mug handle seen from across the station by the wrist camera",
        description="The 95 percent region of a single-view estimate against the 15 mm grasp tolerance.",
        site=_Embodied.Scenario.WarehouseAisle,
        sigmas=[Q_(4.0, "mm"), Q_(5.0, "mm"), Q_(12.0, "mm")],  # scenario assumption: lateral x, lateral y, depth
        coverage=0.95,  # scenario assumption: confidence level
    )
    LatchUnbrakedStall = LatchContactDeadlines.variant(
        name="The guarded latch approach with no local trip, stalled for 200 ms",
        description="Plate travel and spring-model force when only the proposer or the lease ends the press.",
        stall=Q_(200.0, "ms"),  # scenario assumption: unbraked contact interval
    )
    ConveyorMugGoalSetOffsets = _st.GoalSetMembershipScenario(
        name="Terminal TCP offsets of 10 and 20 mm against the mug's 15 mm goal set",
        description="Normalized ellipsoidal distance of a pure x offset, inside and outside the set.",
        site=_Embodied.Scenario.WarehouseAisle,
        offsets=[Q_(10.0, "mm"), Q_(20.0, "mm")],  # scenario assumption: terminal TCP offsets along x
    )

    Gpt2XlWebTextCorpus = _st.CorpusRepetitionScenario(
        name="GPT-2 XL's published training compute on one eight-H100 machine, against WebText",
        description="WebText at about 4 bytes per token and the run's 1.5e21 FLOPs at six FLOPs per parameter "
                    "per token.",
        run=Gpt2XlRunOnOneDgxH100,
        bytes_per_token=Q_(4, "byte/count"),  # scenario assumption: English text averages about 4 bytes per BPE token
        tokens_per_param=float(_Literature.Chinchilla.TokensPerParam),
        flops_per_param_token=float(_Literature.Chinchilla.ComputeConstant),
    )
    ChinchillaComputeDoubling = _st.AllocationGrowthScenario(
        name="Compute-optimal tokens when the compute budget doubles",
        description="Chinchilla approach 1: D_opt grows as C^0.50.",
        compute_multiple=2.0,             # scenario assumption: double the budget
        exponent=float(_Literature.Chinchilla.Approach1TokensExponent),
    )
    Gpt2WebTextNearDuplicates = _st.DuplicateRemovalScenario(
        name="Near-duplicate removal on GPT-2's WebText run, at the C4 and RealNews shares",
        description="Lee et al. (2022) near-duplicate shares applied to the eight-H100 run; one read of WebText "
                    "from the node's NVMe; 128 32-bit MinHash values per document.",
        run=Gpt2XlWebTextCorpus,
        duplicate_shares={"C4": 3.04 / 100,  # Lee, Ippolito, Nystrom, Zhang, Eck, Callison-Burch, and Carlini (20...
                          "RealNews": 13.63 / 100},  # Lee, Ippolito, Nystrom, Zhang, Eck, Callison-Burch, and Carlini (20...
        documents=8_000_000,              # scenario assumption: WebText holds slightly over 8 million documents (Radford et al. 2019)
        signature_hashes=128,             # scenario assumption: signature length per document
        hash_bytes=Q_(4, "byte"),         # 32-bit hash values
        read_bandwidth=_Hardware.Cloud.H100.storage.bandwidth,
    )
    ResNet50HalfImageNetIcr = _st.InformationComputeRatioScenario(
        name="Learning per FLOP of one ResNet-50 epoch on all of ImageNet and on a half-size coreset",
        description="An assumed 5-point gain on all the data against 4.5 points on the half an EL2N score keeps.",
        workload=_Models.Vision.ResNet50,
        dataset=_Datasets.ImageNet,
        gain_full=5.0,                    # scenario assumption: percentage-point gain in epoch 1, all data
        gain_subset=4.5,                  # scenario assumption: the same with the 50% coreset
        keep_fraction=0.5,                # scenario assumption: keep half
    )
    ResNet50TenPercentCoresetH100 = _st.CoresetRunScenario(
        name="ResNet-50 on ImageNet for 100 epochs on one H100, against a 10% coreset scored by ResNet-50 or ResNet-18",
        description="The A100 ResNet-50 sustained fraction of peak carried to the H100; one forward pass scores "
                    "the pool.",
        workload=_Models.Vision.ResNet50,
        proxy=_Models.Vision.ResNet18,
        dataset=_Datasets.ImageNet,
        hardware=_Hardware.Cloud.H100,
        utilization=ResNet50A100InputFeed.sustained_fraction,
        keep_fraction=0.10,               # scenario assumption: 10% coreset
        epochs=100,                       # scenario assumption: training epochs
    )
    ResNet50ProxyTrainedCoreset = ResNet50TenPercentCoresetH100.variant(
        name="A 10% ImageNet coreset for ResNet-50 chosen by a proxy trained from scratch for 5 epochs",
        description="No pretrained checkpoint: the proxy trains briefly, then EL2N scores rank the pool.",
        proxy_epochs=5,                   # scenario assumption: proxy training epochs
    )
    ResNet50HalfCoresetEpochH100 = ResNet50TenPercentCoresetH100.variant(
        name="One ResNet-50 ImageNet epoch on one H100, full and with a half-size coreset, through the data path",
        description="Archive read from the node's NVMe, 8-bit RGB tensors over PCIe, and a host of 14 cores per "
                    "accelerator at the loader-worker rate, over the standard 90-epoch schedule.",
        keep_fraction=ResNet50HalfImageNetIcr.keep_fraction,
        epochs=90,                        # scenario assumption: standard ImageNet schedule
        host_rate=(_Nodes.DGX_H100.host_cpu_cores // _Nodes.DGX_H100.accelerators_per_node)
                  * ResNet50A100InputFeed.worker_rate,
        decoded_sample=Q_(_Datasets.ImageNet.image_width * _Datasets.ImageNet.image_height
                          * _Datasets.ImageNet.image_channels, "byte"),  # 8-bit RGB tensor
    )
    ResNet50EchoingH100 = _st.DataEchoingScenario(
        name="Data echoing for ResNet-50 on one H100 fed by a 14-core host",
        description="The H100's demand, stated to the hundred images a second, against the host's preparation "
                    "rate, over 90 epochs, each prepared image used twice.",
        accelerator_rate=Q_(round(ResNet50HalfCoresetEpochH100.sample_rate, -2), "count/second"),  # stated as about 10,100 images/s
        host_rate=ResNet50HalfCoresetEpochH100.host_rate,
        examples=_Datasets.ImageNet.training_examples,
        epochs=ResNet50HalfCoresetEpochH100.epochs,
        echo_factor=2,                    # scenario assumption: repeat each prepared batch twice
    )
    ExpensiveSelectionPitfall = _st.SelectionBudgetScenario(
        name="Ten hours of selection for a subset that trains in two, against an eight-hour baseline",
        description="Selection overhead that erases the saving.",
        selection_time=Q_(10, "hour"),    # scenario assumption: expensive selection
        subset_time=Q_(2, "hour"),        # scenario assumption: subset training run
        full_time=Q_(8, "hour"),          # scenario assumption: full training run
    )
    ProxySelectionPitfall = ExpensiveSelectionPitfall.variant(
        name="Half an hour of proxy scoring against the same eight-hour baseline",
        description="A cheap proxy keeps selection a small share of full training.",
        selection_time=Q_(0.5, "hour"),   # scenario assumption: proxy scoring
    )
    RareClassTenPercentCoreset = _st.CoverageFloorScenario(
        name="A rare class of 0.1 percent in a 1M-example pool under an unstratified 10% coreset",
        description="Expected rare examples kept against a coverage floor of 150.",
        pool=1_000_000,                   # scenario assumption
        rare_share=0.001,                 # scenario assumption: 0.1 percent rare
        keep_fraction=0.10,               # scenario assumption: 10% coreset
        floor=150,                        # scenario assumption: illustrative coverage floor
    )
    Cifar10LinearCurriculum = _st.CurriculumScheduleScenario(
        name="A linear-warmup curriculum on CIFAR-10: 115 epochs to target against 150",
        description="Illustrative epoch counts, with an illustrative 10-epoch linear warmup.",
        baseline_epochs=150,              # scenario assumption
        curriculum_epochs=115,            # scenario assumption
        warmup_epochs=10,                 # scenario assumption: illustrative warmup W
    )
    Cifar100Curriculum = Cifar10LinearCurriculum.variant(
        name="An illustrative CIFAR-100 curriculum: 180 epochs against 220",
        description="Illustrative epoch counts.",
        baseline_epochs=220, curriculum_epochs=180, warmup_epochs=None)  # scenario assumption
    ImageNetCurriculum = Cifar10LinearCurriculum.variant(
        name="An illustrative ImageNet curriculum: 80 epochs against 90",
        description="Illustrative epoch counts.",
        baseline_epochs=90, curriculum_epochs=80, warmup_epochs=None)  # scenario assumption
    MentorNetCurriculum = Cifar10LinearCurriculum.variant(
        name="An illustrative MentorNet-style curriculum: 70 epochs against 90",
        description="Illustrative epoch counts.",
        baseline_epochs=90, curriculum_epochs=70, warmup_epochs=None)  # scenario assumption
    ActiveLearningTenRounds = _st.ActiveLearningRoundsScenario(
        name="Active learning from 1,000 labels in ten rounds of 100, against 5,000 random labels",
        description="Labels at $10, $50 of pool scoring per round.",
        label_price=Q_(10, "USD"),        # scenario assumption: per label
        initial_labels=1000,              # scenario assumption
        labels_per_round=100,             # scenario assumption
        rounds=10,                        # scenario assumption
        round_scoring_cost=Q_(50, "USD"),  # scenario assumption: scoring the pool once
        random_labels=5000,               # scenario assumption: random sampling to the same quality
        label_turnaround=Q_(14, "day"),   # scenario assumption: annotation turnaround per round
    )
    MedicalScanActiveLearning = _st.BudgetMatchedLabelingScenario(
        name="A million medical scans at $5 a label under a $500K budget",
        description="Random labeling spends the budget; active learning reaches the same quality with 50,000 labels.",
        pool=Q_(1_000_000, "count"),      # scenario assumption: scans in the pool
        label_price=Q_(5.00, "USD/count"),  # scenario assumption: specialist label
        budget=Q_(500_000, "USD"),        # scenario assumption
        active_labels=Q_(50_000, "count"),  # scenario assumption: labels needed with active learning
        deadline=Q_(1, "month"),          # scenario assumption
    )
    FixMatchCifar10Cost = _st.SemiSupervisedCostScenario(
        name="Supervised training on 4,000 CIFAR-10 labels against FixMatch on 250 with five times the compute",
        description="Illustrative label and compute dollars.",
        label_price=Q_(1, "USD"),         # scenario assumption: per label
        supervised_labels=4000,  # Sohn, Berthelot, Li, Zhang, Carlini, Cubuk, Kurakin, Zhang, and Raf...
        supervised_compute=Q_(50, "USD"),  # scenario assumption
        ssl_labels=250,  # Sohn, Berthelot, Li, Zhang, Carlini, Cubuk, Kurakin, Zhang, and Raf...
        ssl_compute=Q_(250, "USD"),       # scenario assumption: five times more training
    )
    FoundationModelTenTasks = _st.PretrainingAmortizationScenario(
        name="Ten tasks trained from scratch against one pretrained foundation model fine-tuned ten times",
        description="GPU-hours and labels per task under illustrative assumptions.",
        pretrain_compute=Q_(10_000, "hour"),   # scenario assumption: one-time GPU-hours
        scratch_compute=Q_(1000, "hour"),      # scenario assumption: GPU-hours per task
        finetune_compute=Q_(50, "hour"),       # scenario assumption: GPU-hours per task
        tasks=10,                              # scenario assumption
        scratch_labels=Q_(100_000, "count"),   # scenario assumption
        finetune_labels=Q_(1_000, "count"),    # scenario assumption
        label_price=Q_(1, "USD/count"),        # scenario assumption
    )
    ImageNetDataCostBreakdown = _st.DataCostBreakdownScenario(
        name="An ImageNet-scale supervised bill: license, crowd labels, a year of storage, and 30 ResNet-50 runs",
        description="Every training image labeled at the crowd rate, the archive kept a year in object storage, "
                    "and a 30-run campaign of 100-epoch runs on one H100 at the cloud GPU rate.",
        acquisition=Q_(50_000, "USD"),    # scenario assumption: licensed dataset
        labels=_Datasets.ImageNet.training_examples,
        label_price=_Infrastructure.Pricing.Labeling.CrowdHigh.rate / Q_(1, "count"),
        storage_bytes=_Datasets.ImageNet.training_archive_size.to("GB"),
        storage_rate=_Infrastructure.Pricing.Storage.S3StandardLowPerTbMonth.rate,
        storage_duration=Q_(12, "month"),  # scenario assumption: retention
        run_time=ResNet50TenPercentCoresetH100.run_time,
        runs=30,                          # scenario assumption: runs in the training campaign
        accelerators=1,
        accelerator_rate=_Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
    )
    MinHashPipelineAmortization = _st.RunAmortizationScenario(
        name="A MinHash deduplication pipeline: $50K to build, $5K to run once, $10K saved per training run",
        description="Illustrative engineering, compute, and per-run savings.",
        build_cost=Q_(50_000, "USD"),     # scenario assumption: engineering time
        operating_cost=Q_(5000, "USD"),   # scenario assumption: one-time MinHash compute
        saving_per_run=Q_(10_000, "USD"),  # scenario assumption
        run_counts=[1, 5, 10, 50],        # campaign sizes the amortization table reads
    )
    TenfoldDataFourPoints = _st.DataScalingReturnScenario(
        name="Ten times the data for about four points of accuracy",
        description="An illustrative power-law return on more data.",
        base_examples=Q_(1_000_000, "count"),   # scenario assumption
        scaled_examples=Q_(10_000_000, "count"),  # scenario assumption
        accuracy_gain=4.0,                      # scenario assumption: percentage points
    )

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
        grids={"Norway": _Infrastructure.Grids.Norway, "Quebec": _Infrastructure.Grids.Quebec,
               "France": _Infrastructure.Grids.France, "EU average": _Infrastructure.Grids.EU_Avg,
               "Germany": _Infrastructure.Grids.Germany, "US average": _Infrastructure.Grids.US_Avg,
               "China": _Infrastructure.Grids.China_Avg, "India": _Infrastructure.Grids.India_Avg,
               "Poland": _Infrastructure.Grids.Poland},
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
        device_hours=Q_(_Clusters.Lab_64_A100.total_accelerators
                        * _ReferenceStats.MidScaleTrainingCarbonProfile.TrainingDays.to("hour").magnitude, "hour"),
        devices=_Clusters.Lab_64_A100.total_accelerators,
        device=_Hardware.Cloud.A100,
        rack=_Systems.Racks.DGX_H100_4Node,
        pue=_Infrastructure.Grids.US_Avg.pue,
        grids={"US average": _Infrastructure.Grids.US_Avg, "Quebec": _Infrastructure.Grids.Quebec},
        service_life=Q_(_Infrastructure.Pricing.Capital.AcceleratorExtendedServiceLifeYears.rate
                        * _DAYS_PER_YEAR, "day").to("hour"),
    )
    TenMillionQueriesAgainstLab64Training = _st.ServingCrossoverScenario(
        name="Ten million queries a day against the 64-A100 training run",
        description="Days of serving at 1 Wh per query that equal the lab run's facility energy.",
        one_time_energy=Lab64A100TrainingFootprint.result.facility_energy,
        requests_per_day=10_000_000,  # scenario assumption: high-volume deployed model
        energy_per_request=Q_(0.001, "kWh"),  # scenario assumption: pedagogical per-query energy
    )

    # --- Fleet-scale recall: expansion, rare events, MoE bytes, link energy (added 2026-10-20) ---
    Gpt3ClusterDoubling1K = _st.ClusterExpansionScenario(
        name="Doubling the 175B job from 1,024 to 2,048 GPUs on the same checkpoint tier",
        description="The 14-byte-per-parameter checkpoint at the per-job write bandwidth, priced by "
                    "Young-Daly checkpoint-and-rework overhead against the GPU-only interval at each size.",
        checkpoint=Gpt3CheckpointInterval10K,  # write path only; the MTBF is replaced at each size
        component_mtbf=Q_(_Reliability.Gpu.mttf_hours, "hour"),
        n_devices=_Clusters.Training_1K.total_accelerators,
        factor=2.0,  # scenario assumption: double the reference cluster
    )
    SlowWorker8K = HalfSpeedStraggler1K.variant(
        name="One of 8,000 workers at 80 percent speed",
        description="An equal-work, barrier-dominated synchronous step; every worker waits for the slow one.",
        n_devices=8_000,  # scenario assumption
        slow_speed=0.80,  # scenario assumption: one worker at 80 percent of peer speed
    )
    KaplanComputeDoubling = KaplanTenPercentLowerLoss.variant(
        name="Loss change when training compute doubles under Kaplan's compute law",
        description="The fitted power law multiplies loss by 2^-alpha per doubling, valid only over its "
                    "studied range.",
        loss_reduction=None,
        resource_multiple=2.0,  # scenario assumption: double the training compute
    )
    DeepSeekV3WeightTraffic = _st.ExpertWeightTrafficScenario(
        name="DeepSeek-V3 weight bytes against a dense 400B model, FP16",
        description="All experts resident against the active parameters one token reads, beside a dense "
                    "model that reads every weight each step. Payload counts, not latencies.",
        workload=_Models.Language.DeepSeek_V3,
        dense_parameters=Q_(400, "Bparam"),  # scenario assumption: illustrative dense baseline
        bytes_per_param=_BYTES_FP16,
    )

    # --- The warehouse base's finished stopping budget and its enforcer (added 2026-10-20) ---
    # Site record Embodied.Scenario.WarehouseAisle and machine
    # Embodied.MobileManipulator.WarehouseMobileManipulator: the elementary stop
    # plus the resident S-curve (jerk bound j_brake), the tracking inset, and,
    # in the walking-person cases, a person closing the gap. Ceilings are read
    # from speed_ceiling; the speed field sets the budget rows.
    WarehouseFinishedStopAtDriveLimit = _st.StoppingBudgetScenario(
        name="Warehouse base's finished stopping budget at its drive limit",
        description="Braking, the two allowances, brake onset, lease path, observation age, the "
                    "S-curve's jerk ramps, and the tracking inset, against the rack-end clear distance.",
        site=_Embodied.Scenario.WarehouseAisle,
        speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
        delay=_Embodied.Scenario.WarehouseAisle.tau_delay,
        jerk=_Embodied.Scenario.WarehouseAisle.j_brake,
        tracking_bound=_Embodied.Scenario.WarehouseAisle.eps_track,
    )
    WarehouseFinishedStopAtAisleSpeed = WarehouseFinishedStopAtDriveLimit.variant(
        name="Warehouse base's finished stopping budget at the candidate aisle speed",
        description="The same terms at the chosen aisle speed; the spare over the speed is the "
                    "unallocated timing allowance.",
        speed=_Embodied.Scenario.WarehouseAisle.v_aisle,
    )
    WarehouseFinishedStopWithDispatch = WarehouseFinishedStopAtAisleSpeed.variant(
        name="Warehouse base's finished stopping budget with dispatch-to-admission charged",
        description="The pre-brake delay plus the dispatch-to-admission allocation: the complete "
                    "static-obstacle ceiling, and the reserve left at the aisle speed.",
        delay=(_Embodied.Scenario.WarehouseAisle.tau_delay
               + _Embodied.Scenario.WarehouseAisle.t_dispatch_to_admission).to("ms"),
    )
    WarehouseFinishedStopInspectedFloor = WarehouseFinishedStopAtAisleSpeed.variant(
        name="Warehouse base's finished stopping budget on the inspected low-traction floor",
        description="Deceleration capped by the inspected floor's friction, all weight on the driven "
                    "wheels; the jerk bound belongs to the load and is unchanged.",
        deceleration=_traction_limit(_Embodied.Scenario.WarehouseAisle.mu_inspected_floor),
    )
    WarehouseFinishedStopInspectedFloorWithDispatch = WarehouseFinishedStopInspectedFloor.variant(
        name="Inspected-floor finished budget with dispatch-to-admission charged",
        description="The inspected-floor ceiling after the dispatch-to-admission allocation.",
        delay=WarehouseFinishedStopWithDispatch.delay,
    )
    WarehouseFinishedStopWalkingPerson = WarehouseFinishedStopAtAisleSpeed.variant(
        name="Warehouse base's finished stopping budget against a person walking toward it",
        description="A person at the ISO 13855 approach speed closes the gap through the delay and "
                    "the whole S-curve stop.",
        approach_speed=_Embodied.Scenario.WarehouseAisle.v_human_approach,
    )
    WarehouseFinishedStopWalkingPersonWithDispatch = WarehouseFinishedStopWalkingPerson.variant(
        name="Walking-person finished budget with dispatch-to-admission charged",
        description="The walking-person ceiling after the dispatch-to-admission allocation; the person "
                    "keeps walking through that interval too.",
        delay=WarehouseFinishedStopWithDispatch.delay,
    )
    WarehouseFinishedStopTenMsSlower = WarehouseFinishedStopAtAisleSpeed.variant(
        name="Warehouse base's finished stopping budget with 10 ms more pre-brake delay",
        description="Latency sensitivity of the ceiling.",
        delay=(_Embodied.Scenario.WarehouseAisle.tau_delay + Q_(10.0, "ms")).to("ms"),  # scenario assumption
    )
    WarehouseHostLeaseStall = WarehouseFinishedStopAtAisleSpeed.variant(
        name="A 200 ms host stall in place of the 60 ms chunk lease at the aisle speed",
        description="A lease enforced by a non-real-time host keeps the last proposal for the stall; "
                    "added_delay is the stall beyond the lease.",
        added_delay=(Q_(200.0, "ms")  # scenario assumption: proposer stall before any brake
                     - _Embodied.Scenario.WarehouseAisle.t_lease).to("ms"),
    )
    WarehouseIntegratorUnwind = WarehouseFinishedStopAtAisleSpeed.variant(
        name="A wound-up integrator that delays braking 150 ms at the aisle speed",
        description="An unprotected integrator must unwind before braking starts; added_delay is the unwind.",
        added_delay=Q_(150.0, "ms"),  # scenario assumption: illustrative unwind time
    )
    LoadedArmJointGravityStop = _st.GravityBiasedJointStopScenario(
        name="A loaded arm joint braking with and against gravity",
        description="The running machine's arm at its peak joint torque, a payload-carrying joint "
                    "approaching at 1.2 rad/s with a 0.80 m lever, gravity holding 75 percent of the "
                    "peak; the kinetic barrier priced at 110 and 70 mm of tool clearance; the "
                    "protective separation against a person walking in.",
        machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
        inertia=Q_(3.0, "kg*m**2"),          # scenario assumption: joint with payload (illustrative)
        lever=Q_(0.80, "m"),                 # scenario assumption: illustrative
        joint_speed=Q_(1.2, "1/s"),          # scenario assumption: rad/s (illustrative)
        bias_fraction=0.75,                  # scenario assumption: share of peak torque holding the load
        barrier_gain=Q_(10.0, "1/s"),        # scenario assumption: chosen barrier gain
        barrier_clearances=[Q_(110.0, "mm"), Q_(70.0, "mm")],  # scenario assumption: illustrative
        site=_Embodied.Scenario.WarehouseAisle,
    )
    TwoJointTorqueProjection = _st.TorqueHalfSpaceProjectionScenario(
        name="A two-joint torque proposal against one barrier half-space",
        description="Proposal (3.5, 3) N m against 2 tau_1 + tau_2 <= 5 N m: the least-change "
                    "projection against a per-axis clamp of joint 1 to 2 N m (abstract illustration).",
        proposal=[3.5, 3.0],      # scenario assumption: N m
        normal=[2.0, 1.0],        # scenario assumption
        bound=5.0,                # scenario assumption: N m
        clamp=[2.0, 3.0],         # scenario assumption: per-axis clamp, N m
        actuator_limit=5.0,       # scenario assumption: symmetric per-joint limit, N m
    )
    WarehouseEnforcerCycleBudget = _st.CycleBudgetScenario(
        name="Stage budget of the warehouse machine's 1 kHz permission check",
        description="Declared sub-budgets that sum to the site's declared WCET inside its deadline.",
        site=_Embodied.Scenario.WarehouseAisle,
        stages=[  # scenario assumption: chosen sub-budgets
            ("state acquisition", Q_(30.0, "us")),
            ("proposal and lease check", Q_(15.0, "us")),
            ("tracking check", Q_(10.0, "us")),
            ("QP solve", Q_(150.0, "us")),
            ("output write", Q_(15.0, "us")),
            ("reserve", Q_(30.0, "us")),
        ],
    )
    WarehouseBaseHoldDrift = _st.HoldIntervalDriftScenario(
        name="Drift of the warehouse base over one held 1 ms command at its drive limit",
        description="Zero-order hold: travel at the drive limit plus an independent approach-"
                    "acceleration upper bound over one permission tick.",
        speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
        acceleration=Q_(2.0, "m/s**2"),  # scenario assumption: approach-acceleration upper bound
        period=_Embodied.Scenario.WarehouseAisle.t_tick,
    )

    # --- Workload placement: the permission tick against what it shares (added 2026-10-20) ---
    # The warehouse machine's 1 kHz permission tick (site WarehouseAisle), its
    # unloaded sub-costs, and the rejected shared-die placements that price
    # memory queueing, miss-handling stalls, rail droop, and thermal throttling.
    # Inputs are illustrative teaching assumptions unless a source is named.
    WarehouseEnforcerUnloadedTick = _st.CycleBudgetScenario(
        name="Unloaded run of the warehouse machine's 1 kHz permission check",
        description="Sense, check, and stage sub-costs of the 135 microsecond unloaded tick on the "
                    "discrete safety MCU.",
        site=_Embodied.Scenario.WarehouseAisle,
        stages=[  # scenario assumption: illustrative sub-costs
            ("sense", Q_(40.0, "us")),               # encoders, IMU over SPI, evidence epoch
            ("constraint check", Q_(80.0, "us")),    # kinematic and dynamic envelope
            ("drive-frame staging", Q_(15.0, "us")),
        ],
    )
    WarehouseSharedDieLoadedRead = _st.SharedDramReadScenario(
        name="The permission tick's snapshot read behind proposer backlogs on the application SoC",
        description="A FIFO LPDDR5 controller serves whole backlogs (a 256 KiB weight tile, a camera "
                    "transfer, a telemetry record) before the tick's 4 KiB snapshot; addresses "
                    "interleave across every channel.",
        site=_Embodied.Scenario.WarehouseAisle,
        memory_tech=_Hardware.Tech.Memory.LPDDR5_6400,
        sense=WarehouseEnforcerUnloadedTick.stage("sense"),
        compute_after_sense=(WarehouseEnforcerUnloadedTick.stage("constraint check")
                             + WarehouseEnforcerUnloadedTick.stage("drive-frame staging")),
        tile_bytes=256 * 1024,      # scenario assumption: one weight tile
        camera_bytes=64 * 1024,     # scenario assumption
        telemetry_bytes=32 * 1024,  # scenario assumption
        snapshot_bytes=4 * 1024,    # scenario assumption: encoder, IMU, evidence-epoch snapshot
        transaction_bytes=4 * 1024,  # AXI4 4 KiB burst boundary
        bank_switches=3,            # scenario assumption: same-bank row conflicts per backlog
    )
    WarehouseSharedDieLoadedReadOneChannel = WarehouseSharedDieLoadedRead.variant(
        name="The same snapshot read with backlog and read confined to one LPDDR5 channel",
        description="Contrast case: the address mapping puts both on the snapshot's channel.",
        one_channel=True,
    )
    WarehouseTickMshrStallShare = _st.StallShareScenario(
        name="Miss-stall share of the permission tick that a 21x MSHR slowdown allows",
        description="Measured shared-cache contention with partitioning (Valsan et al. 2016) as the "
                    "slowdown of the tick's stalled share, against the declared WCET.",
        site=_Embodied.Scenario.WarehouseAisle,
        slowdown=21.0,  # Valsan, P. K., Yun, H., and Farshchi, F. (2016), Taming Non-blocking...
    )
    WarehouseTickMshrStallShareAtDeadline = WarehouseTickMshrStallShare.variant(
        name="The same 21x slowdown against the deadline less blocking and interference",
        description="Bound = execution ceiling of the full response budget.",
        bound=(_Embodied.Scenario.WarehouseAisle.enforcer_deadline
               - _Embodied.Scenario.WarehouseAisle.enforcer_blocking
               - _Embodied.Scenario.WarehouseAisle.enforcer_interference).to("us"),
    )
    WarehouseTickWritebackStallShare = WarehouseTickMshrStallShare.variant(
        name="Miss-stall share of the permission tick that a 300x writeback-buffer slowdown allows",
        description="A round floor under the writeback-buffer attack slowdowns Bechtel and Yun (2019) "
                    "measured (up to 346x).",
        slowdown=300.0,  # scenario assumption: round floor below the measured maximum
    )
    WarehouseTickTenfoldStallShare = WarehouseTickMshrStallShare.variant(
        name="Miss-stall share of the permission tick that a 10x slowdown allows",
        description="A self-check estimate between the measured slowdowns.",
        slowdown=10.0,  # scenario assumption: illustrative slowdown
    )
    PackageRailDroop = _st.RailDroopScenario(
        name="A 6 A accelerator current step on a shared 0.85 V core rail",
        description="L dI/dt plus I R through an illustrative package, and the gate-delay stretch of "
                    "the alpha-power model (Sakurai and Newton 1990) at the drooped rail.",
        rail_voltage=Q_(0.850, "V"),        # scenario assumption
        inductance=Q_(30.0, "pH"),          # scenario assumption: package loop inductance
        resistance=Q_(8.0, "milliohm"),     # scenario assumption
        delta_i=Q_(6.0, "A"),               # scenario assumption: accelerator current step
        rise_time=Q_(2.0, "ns"),            # scenario assumption
        threshold_voltage=Q_(0.420, "V"),   # scenario assumption
        alpha=1.3,                          # scenario assumption: velocity-saturation index
    )
    WarehouseSharedDieThermalThrottle = _st.ThermalThrottleScenario(
        name="The permission tick on a shared-die real-time cluster whose governor halves its clock",
        description="Rejected placement: the tick sized to its WCET at 1.2 GHz; continuous proposal "
                    "inference heats the junction past a 95 degC trip, and the clock stays at 600 MHz "
                    "until the junction clears 85 degC.",
        site=_Embodied.Scenario.WarehouseAisle,
        nominal_clock=Q_(1.2, "GHz"),       # scenario assumption
        throttled_clock=Q_(600.0, "MHz"),   # scenario assumption: the governor halves the clock
        time_constant=Q_(8.0, "s"),         # scenario assumption
        ambient=Q_(45.0, "degC"),           # scenario assumption: electronics bay
        theta=Q_(2.0, "K/W"),               # scenario assumption: junction to ambient
        quiet_power=Q_(4.0, "W"),           # scenario assumption: enforcer, idle, memory
        loaded_power=Q_(26.0, "W"),         # scenario assumption: with continuous inference
        trip=Q_(95.0, "degC"),              # scenario assumption
        clear=Q_(85.0, "degC"),             # scenario assumption
    )
    WarehouseLoadedCycleCheckpoint = WarehousePermissionTickResponse.variant(
        name="A 300 microsecond loaded cycle of the permission check on a shared die",
        description="Self-check: past the declared WCET, still inside the deadline.",
        wcet=Q_(300.0, "us"),  # scenario assumption: one measured loaded cycle
    )
    WarehouseRejectedPlacementTrace = _st.TickTraceScenario(
        name="Illustrative ten-million-tick trace of the permission check on the shared die",
        description="Cold with an idle proposer, warm under contention; percentiles are illustrative, "
                    "not a measurement.",
        site=_Embodied.Scenario.WarehouseAisle,
        ticks=10_000_000,  # scenario assumption: trace length
        cold={"p50": Q_(95.0, "us"), "p99": Q_(128.0, "us"), "max": Q_(152.0, "us")},  # scenario assumption
        warm={"p50": Q_(210.0, "us"), "p99": Q_(330.0, "us"), "p99.9": Q_(395.0, "us"),
              "max": Q_(468.0, "us")},  # scenario assumption
    )

    # --- Agent operations: traces, retention, postmortems, staged release (added 2026-10-20) ---
    # Types in scenario_types.py, solvers in engine/solvers/operations.py. Every
    # input is a stated teaching assumption unless a source is named.
    RepairTrialSpanTimeline = _st.SpanCriticalPathScenario(
        name="A repair-agent timing trial on one host clock, with the tests cut to 1 s",
        description="Retrieve supporting context, a model call proposing two checks, targeted tests and a "
                    "static lint check in parallel, and a model call synthesizing the report after both; "
                    "a 0.5 s gap before synthesis that no span covers. The intervention shortens the "
                    "tests and keeps every other duration and wait.",
        intervals={"retrieve": (Q_(0.0, "s"), Q_(0.2, "s")),  # scenario assumption: observed interval
                   "propose": (Q_(0.2, "s"), Q_(3.2, "s")),  # scenario assumption
                   "tests": (Q_(3.2, "s"), Q_(9.2, "s")),  # scenario assumption
                   "lint": (Q_(3.2, "s"), Q_(5.2, "s")),  # scenario assumption
                   "synthesize": (Q_(9.7, "s"), Q_(11.7, "s"))},  # scenario assumption
        depends_on={"propose": ["retrieve"], "tests": ["propose"], "lint": ["propose"],
                    "synthesize": ["tests", "lint"]},
        intervention={"tests": Q_(1.0, "s")},  # scenario assumption: a cache shortens the tests
    )

    RareFailureUnderHeadSampling = _st.HeadSamplingMissScenario(
        name="A failure mode seen five times under a 10 percent head sample",
        description="Each trial kept independently before its outcome is known.",
        retention_probability=0.10,  # scenario assumption: head-sampling probability
        occurrences=5,  # scenario assumption: occurrences of the failure mode
    )
    RecurringFailureUnderHeadSampling = RareFailureUnderHeadSampling.variant(
        name="The same failure mode seen fifty times under the 10 percent head sample",
        occurrences=50,  # scenario assumption
    )
    FailureUnderFivePercentHeadSample = _st.HeadSamplingMissScenario(
        name="A failure mode seen twenty times under a 5 percent head sample",
        description="Each trial kept independently before its outcome is known.",
        retention_probability=0.05,  # scenario assumption
        occurrences=20,  # scenario assumption
    )

    VerdictAwareTraceRetention = _st.TraceRetentionScenario(
        name="10,000 trials an hour with 1.4 MB trace bodies, kept for 24 hours",
        description="400 failed, 100 pending or unknown, and 9,500 ordinary accepted trials each hour; "
                    "head sampling at 5 percent against verdict-aware selection with a 2 percent control "
                    "sample; decimal megabytes, no capture loss.",
        failed=400,  # scenario assumption: failed a criterion per hour
        unresolved=100,  # scenario assumption: pending or unknown per hour
        ordinary=9_500,  # scenario assumption: ordinary accepted per hour
        cohort_period=Q_(1, "hour"),
        trace_size=Q_(1.4, "MB"),  # scenario assumption: complete permitted body, decimal MB
        window=Q_(24, "hour"),  # scenario assumption: retention window
        head_probability=0.05,  # scenario assumption
        control_probability=0.02,  # scenario assumption
    )

    InterfaceRepairBranches = _st.ControlledBranchScenario(
        name="Twenty continuations per branch from one saved prefix of a truncated-search repair",
        description="Original interface, a warning that results may be incomplete, and a paginated "
                    "interface with typed partial-result metadata; same model, sampling, budget, checks, "
                    "and admission gate. Hypothetical outcomes.",
        outcomes={"original": {"accepted": 1, "forbidden": 19},  # scenario assumption
                  "warning": {"accepted": 8, "budget_stop": 12},  # scenario assumption
                  "paginated": {"accepted": 20}},  # scenario assumption
        target_bound=0.05,  # scenario assumption: failure-rate bound to demonstrate
    )


    TestLintToolDrift = _st.ToolDriftScenario(
        name="A candidate that splits its checks evenly against a baseline at 70 percent tests",
        description="Two-tool call distributions from a shadow run; hypothetical shares.",
        baseline={"tests": 0.70, "lint": 0.30},  # scenario assumption
        candidate={"tests": 0.50, "lint": 0.50},  # scenario assumption
    )

    AcceptanceCanaryMonitor = _st.SequentialCanaryScenario(
        name="An SPRT canary: acceptance 0.90 against 0.75 on 5 percent of 60 tasks an hour",
        description="False-hold target 0.01 and missed-degradation target 0.05; stationary independent "
                    "labels, regular admissions, each label 0.5 h after admission, no tripwire.",
        p_base=0.90,  # scenario assumption: baseline acceptance
        p_degraded=0.75,  # scenario assumption: specified degradation
        alpha=0.01,  # scenario assumption
        beta=0.05,  # scenario assumption
        arrival_rate=Q_(60, "1/hour"),  # scenario assumption: eligible tasks
        fraction=0.05,  # scenario assumption: canary allocation
        label_delay=Q_(0.5, "hour"),  # scenario assumption
    )
    AcceptanceCanaryMonitorDoubled = AcceptanceCanaryMonitor.variant(
        name="The same SPRT canary at twice the allocation",
        fraction=AcceptanceCanaryMonitor.fraction * 2,  # scenario assumption: doubled allocation
    )



    # ── Fitting GPT-2 XL's training step on one H100 (added 2026-10-20) ──────
    # The batch-32 step from FP32 through mixed precision, FlashAttention,
    # micro-batches, and segment checkpointing; the number formats and loss
    # scaling behind mixed precision; and the attention tiles and online
    # softmax behind FlashAttention.
    Gpt2XlRunOnOneH100Fp32 = Gpt2XlRunOnOneH100.variant(
        name="GPT-2 XL's training run on one H100's FP32 cores",
        description="The same run at the H100's FP32 (non-tensor) peak, utilization held at 50 percent.",
        peak_precision="fp32",
    )
    Gpt2XlRunOnOneH100Tf32 = Gpt2XlRunOnOneH100.variant(
        name="GPT-2 XL's training run on one H100's TF32 tensor path",
        description="The same run at the H100's dense TF32 peak, utilization held at 50 percent.",
        peak_precision="tf32",
    )
    Gpt2XlMicroBatch16Activations = Gpt2XlBatch32Activations.variant(
        name="GPT-2 XL's saved activations for a micro-batch of 16 sequences of 1,024 tokens, 16-bit",
        description="Half the batch resident at a time, accumulated twice per update.",
        micro_batch=Gpt2XlBatch32Block.batch_size // 2,  # scenario assumption: two micro-batches per update
    )
    Gpt2XlBatch32SegmentCheckpointed = Gpt2XlBatch32Activations.variant(
        name="GPT-2 XL's batch-32 activations, checkpointed in 8 segments of 6 layers, scores not stored",
        description="Each segment's 16-bit input is kept; the backward pass rebuilds one segment at a time.",
        checkpoint_segments=8,  # scenario assumption: 48 layers in 8 equal segments
    )
    Gpt2MixedPrecisionFlashAttentionOnH100 = Gpt2MixedPrecisionAdamOnH100.variant(
        name="GPT-2 XL in mixed precision at batch 32 on one H100, attention scores not stored",
        description="16 B per parameter of state and the per-token activations only (FlashAttention).",
        activations=Gpt2XlBatch32Activations.per_token_activations,
    )
    Gpt2MixedPrecisionTwoMicroBatchesOnH100 = Gpt2MixedPrecisionFlashAttentionOnH100.variant(
        name="GPT-2 XL's batch of 32 as two accumulated micro-batches of 16 on one H100",
        description="Mixed precision, FlashAttention, and 16 sequences' activations resident at a time.",
        batch_size=Gpt2XlMicroBatch16Activations.micro_batch,
        activations=Gpt2XlMicroBatch16Activations.per_token_activations,
    )
    Gpt2MixedPrecisionSegmentCheckpointedOnH100 = Gpt2MixedPrecisionFlashAttentionOnH100.variant(
        name="GPT-2 XL's batch of 32 with segment checkpointing on one H100",
        description="Mixed precision, FlashAttention, and 8 kept segment inputs plus one rebuilt segment.",
        activations=Gpt2XlBatch32SegmentCheckpointed.checkpointed_activations,
    )
    Gpt2XlSingleCoreTokenizerFeed = _st.TokenizedBatchFeedScenario(
        name="GPT-2 XL's batch of 32 tokenized on one host core and copied to an H100, in sequence",
        description="Host tokenization and the PCIe copy of the token ids before every step, no prefetching.",
        compute_time=Gpt2XlRunOnOneH100.step_time,
        tokens_per_batch=Gpt2XlRunOnOneH100.tokens_per_step,
        token_rate=Q_(50_000, "1/s"),  # scenario assumption: tokens per second on one host core
        bytes_per_token=_BYTES_INT64,  # token ids as int64
        link_bandwidth=_Hardware.Cloud.H100.interconnect.bandwidth,
        workers=8,  # scenario assumption: loader workers in the parallel case
    )
    AttentionScores4096TokensFp32 = _st.AttentionTilingScenario(
        name="One attention head at 4,096 tokens in FP32, in 128 x 128 tiles",
        description="The score matrix, a tile, 16 heads, and what standard and FlashAttention store per head.",
        seq_len=4096,  # scenario assumption: long-context example
        head_dim=_Models.Language.GPT2.hidden_dim // _Models.Language.GPT2.heads,
        tile=128,  # scenario assumption: tile side
        heads=16,  # scenario assumption: heads in the example layer
        bytes_per_value=_BYTES_FP32,
    )

    # --- Model compression (added 2026-10-20) ---------------------------------
    Llama2_7BTokenOnFlagshipPhone = _st.WeightReadScenario(
        name="Llama 2 7B, FP16 weights read once per token on a flagship phone",
        description="Batch-one decoding reads every weight over the phone's LPDDR bus; DRAM energy per "
                    "byte prices the reads.",
        workload=_Models.Language.Llama2_7B,
        hardware=_Hardware.Mobile.iPhone15Pro,
        bytes_per_weight=_BYTES_FP16,
        energy_per_byte=_Hardware.Tech.Memory.DRAM.energy_per_byte,
    )
    MobileNetV2Fp32WeightReadPhone = Llama2_7BTokenOnFlagshipPhone.variant(
        name="MobileNetV2, FP32 weights read once per frame on a flagship phone",
        description="The weight-reading floor of one batch-one frame.",
        workload=_Models.Vision.MobileNetV2, bytes_per_weight=_BYTES_FP32, energy_per_byte=None,
    )
    MobileNetV2Int8WeightReadPhone = MobileNetV2Fp32WeightReadPhone.variant(
        name="MobileNetV2, INT8 weights read once per frame on a flagship phone",
        bytes_per_weight=_BYTES_INT8,
    )
    ResNet50Fp32WeightReadPhone = MobileNetV2Fp32WeightReadPhone.variant(
        name="ResNet-50, FP32 weights read once per image on a flagship phone",
        workload=_Models.Vision.ResNet50,
    )
    Gpt2XlFp16WeightReadPhone = Llama2_7BTokenOnFlagshipPhone.variant(
        name="GPT-2 XL, FP16 weights read once per token on a flagship phone",
        description="Batch-one decoding reads every weight over the phone's LPDDR bus.",
        workload=_Models.Language.GPT2, energy_per_byte=None,
    )
    Gpt2XlInt4WeightReadPhone = Gpt2XlFp16WeightReadPhone.variant(
        name="GPT-2 XL, INT4 weights read once per token on a flagship phone",
        bytes_per_weight=_BYTES_INT4,
    )
    MobileNetV2BlurPipelinePhone = _st.FrameCompressionScenario(
        name="Real-time background blur with MobileNetV2 on a mid-tier phone",
        description="A labeled composite: FP32 on the CPU, INT8 on the integer accelerator, and the "
                    "width-0.75 network as the channel-pruned model, against a 30 FPS camera.",
        full=_Models.Vision.MobileNetV2,
        pruned=_Models.Vision.MobileNetV2_Alpha0_75,  # scenario assumption: pruning keeps 75 percent of every layer's channels
        frame_rate=Q_(30, "1/s"),          # scenario assumption: camera frame rate
        cpu_rate=Q_(8, "1/s"),             # scenario assumption: FP32 on the phone CPU (illustrative)
        accelerator_rate=Q_(35, "1/s"),    # scenario assumption: INT8 on the integer accelerator (illustrative)
        float_bytes=_BYTES_FP32,
        int_bytes=_BYTES_INT8,
    )
    Rank128Factorization4096 = _st.LowRankUpdateScenario(
        name="A 4,096 by 4,096 FP32 layer factored at rank 128",
        description="Two thin factors (4,096 by 128 and 128 by 4,096) replace the dense matrix.",
        in_dim=4096,                       # scenario assumption
        out_dim=4096,                      # scenario assumption
        rank=128,                          # scenario assumption: captures nearly all of the layer
        bytes_per_param=_BYTES_FP32,
    )
    ResNet50Conv256FusionH100 = _st.ConvBnReluFusionScenario(
        name="One ResNet-50 3x3 layer with batch norm and ReLU, fused or not, on an H100",
        description="256 channels at 28 by 28 in FP32; traffic over HBM bandwidth, compute ignored.",
        channels=256, height=28, width=28,  # a ResNet-50 stage-2 layer
        kernel_size=3,
        bytes_per_value=_BYTES_FP32,
        norm_params_per_channel=2,          # batch norm's folded scale and shift
        hardware=_Hardware.Cloud.H100,
    )
    WearableMobileNetV2SparseFlash = _st.SparseFlashFitScenario(
        name="MobileNetV2 pruned to 2 MB of FP32 values for a wearable's 2 MB of flash",
        description="Unstructured pruning; each surviving value carries a 16-bit position index.",
        workload=_Models.Vision.MobileNetV2,
        value_bytes=_BYTES_FP32,
        index_bytes=_BYTES_INT16,
        surviving_bytes=Q_(2, "MB"),       # scenario assumption: surviving FP32 values after pruning
        flash_budget=Q_(2, "MB"),          # scenario assumption: wearable microcontroller flash
    )
    PrunedInt8PaperVsMeasured = _st.SpeedupCompositionScenario(
        name="Half the weights pruned and the rest quantized to INT8: paper against measured",
        description="Structured pruning of half the weights and FP32-to-INT8 storage, multiplied on "
                    "paper, against a hypothetical realized speedup.",
        factors={
            "pruning": 1 / (1 - 0.5),      # scenario assumption: 50 percent structured pruning
            "quantization": (_BYTES_FP32 / _BYTES_INT8).to("").magnitude,
        },
        measured=1.5,                      # scenario assumption: hypothetical realized speedup
    )
    IrregularSparsityLatencyTrace = _st.WorkLatencyGapScenario(
        name="A pruned model whose irregular sparsity barely moves latency",
        description="A representative trace: parameters fall far more than latency.",
        work_reduction=0.40,               # scenario assumption: share of parameters removed
        latency_reduction=0.12,            # scenario assumption: share of latency removed
    )
    BertBaseMobileCompression = _st.StagedCompressionScenario(
        name="BERT-Base pruned, distilled, and quantized for a mobile CPU (modeled)",
        description="A stipulated compact student; accuracies after each stage and with quantization first.",
        teacher=_Models.Language.BERT_Base,
        student_params=Q_(27_500_000, "param"),  # scenario assumption: final pruned graph, the distillation student
        float_bytes=_BYTES_FP32,
        int_bytes=_BYTES_INT8,
        baseline_accuracy=0.762,           # scenario assumption: before compression
        pruned_accuracy=0.751,             # scenario assumption: after pruning
        distilled_accuracy=0.759,          # scenario assumption: after distillation
        final_accuracy=0.756,              # scenario assumption: after INT8 quantization
        mobile_speedup=12,                 # scenario assumption: mobile CPU speedup
        quantize_first_loss=0.021,         # scenario assumption: loss when quantization precedes pruning
    )
    ModelFifthOfRequestLatency = _st.AmdahlScenario(
        name="A request path whose model is 20 percent of latency, made infinitely fast",
        description="Decoding, resizing, and display hold the other 80 percent.",
        stages={"model": Q_(0.2, ""), "rest": Q_(0.8, "")},  # scenario assumption: shares of latency
        stage="model",
        speedup=float("inf"),
    )

    # --- Hardware acceleration: reuse, rooflines, and limits on one accelerator (added 2026-10-20) ---
    H100OverReferenceCpu = _st.PeakRatioRangeScenario(
        name="H100 dense FP16 peak against a 1 to 2 TFLOP/s reference CPU",
        description="The reference CPU's registry peak is the low end of the range.",
        rate=_Hardware.Cloud.H100.compute.peak_flops,
        baseline_low=_Hardware.Cloud.ReferenceCPU.compute.peak_flops,
        baseline_high=Q_(2, "TFLOPs / s"),  # scenario assumption: upper end of the reference-CPU range
    )
    SystolicArrayReuseEnergy = _st.SystolicReuseEnergyScenario(
        name="Energy per MAC: every operand from DRAM against a 128-wide systolic array",
        description="A deliberately naive streaming baseline makes four DRAM accesses per MAC; the array "
                    "loads two operands at its edges for every 128 MACs and keeps partial sums inside.",
        access_energy=_Hardware.Tech.Memory.DRAM.energy_per_access,
        mac_energy=Q_(1, "pJ"),   # scenario assumption: about 1 pJ per MAC
        naive_accesses=4,         # scenario assumption: load both operands and the sum, write the sum
        edge_loads=2,             # two operands enter at the array's edges
        array_dim=128,            # scenario assumption: a 128-wide array
    )
    ResNet50InferenceH100OverServerCpu = _st.AcceleratedFractionScenario(
        name="ResNet-50 inference moved to an H100's INT8 path",
        description="95 percent of baseline time runs on the matrix path; the H100's INT8 peak against "
                    "an assumed 8 TOPS server CPU without matrix extensions sets the path gain.",
        fraction=0.95,                                   # scenario assumption: accelerated share of time
        accelerated_rate=_Hardware.Cloud.H100.compute.precision_flops["int8"],
        baseline_rate=Q_(8, "TOPS"),                     # scenario assumption: server-CPU INT8 rate
    )
    Gpt2DecodeH100OverServerCpu = ResNet50InferenceH100OverServerCpu.variant(
        name="GPT-2 XL token generation moved to an H100's INT8 path",
        description="80 percent of each token's time on the matrix path; host overhead and sampling stay.",
        fraction=0.80,                                   # scenario assumption: accelerated share of time
    )
    Gpt2DecodeH100OverServerCpuBandwidth = Gpt2DecodeH100OverServerCpu.variant(
        name="GPT-2 XL token generation with a bandwidth-set path gain",
        description="Decode is bandwidth bound, so the path gain is the H100's memory bandwidth over the "
                    "reference CPU socket's.",
        accelerated_rate=_Hardware.Cloud.H100.memory.bandwidth,
        baseline_rate=_Hardware.Cloud.ReferenceCPU.memory.bandwidth,
    )
    ResNet50Batch1H100Int8PeakRate = _st.SingleNodeInferenceScenario(
        name="ResNet-50, one image, on an H100 at INT8 and peak rates",
        description="Operations over the INT8 tensor peak: an arithmetic floor, not a latency.",
        workload=_Models.Vision.ResNet50,
        hardware=_Hardware.Cloud.H100,
        precision="int8",
        efficiency=1.0,                       # scenario assumption: peak rates (lower bounds)
    )
    ResNet50Batch1H100Fp16PeakRate = ResNet50Batch1H100Int8PeakRate.variant(
        name="ResNet-50, one image, on an H100 at FP16 and peak rates",
        precision="fp16",
    )
    # ResNet-50 layer shapes (He et al. 2016): stage 2 is 56 x 56 with 64-channel
    # bottlenecks expanding to 256; stage 3 is 28 x 28 with 128 channels; the
    # classifier maps 2,048 pooled features to 1,000 classes.
    ResNet50Stage3ConvBatch32 = _st.ConvLayerScenario(
        name="ResNet-50 stage-3 3x3 convolution, batch 32, FP16",
        description="28 x 28 x 128 to 128 channels; input and weights read once, output written once.",
        height=28, width=28, kernel=3, in_channels=128, out_channels=128,
        bytes_per_value=_BYTES_FP16,
        batch_size=32,                        # scenario assumption: inference batch
    )
    ResNet50Stage3ConvBatch32Int8 = ResNet50Stage3ConvBatch32.variant(
        name="ResNet-50 stage-3 3x3 convolution, batch 32, INT8", bytes_per_value=_BYTES_INT8)
    ResNet50Stage3ConvBatch32Int8Weights = ResNet50Stage3ConvBatch32.variant(
        name="ResNet-50 stage-3 3x3 convolution, batch 32, INT8 weights and FP16 activations",
        weight_bytes_per_value=_BYTES_INT8)
    ResNet50Stage2ConvBatch32 = ResNet50Stage3ConvBatch32.variant(
        name="ResNet-50 stage-2 3x3 convolution, batch 32, FP16",
        description="56 x 56 x 64 to 64 channels.",
        height=56, width=56, in_channels=64, out_channels=64)
    ResNet50Stage2ConvBatch32Int8 = ResNet50Stage2ConvBatch32.variant(
        name="ResNet-50 stage-2 3x3 convolution, batch 32, INT8", bytes_per_value=_BYTES_INT8)
    ResNet50Stage2ExpansionBatch32 = ResNet50Stage2ConvBatch32.variant(
        name="ResNet-50 stage-2 1x1 expansion, batch 32, FP16",
        description="56 x 56 x 64 to 256 channels.",
        kernel=1, out_channels=256)
    ResNet50Stage2ExpansionBatch32Int8 = ResNet50Stage2ExpansionBatch32.variant(
        name="ResNet-50 stage-2 1x1 expansion, batch 32, INT8", bytes_per_value=_BYTES_INT8)
    ResNet50ClassifierHeadBatch32 = _st.DenseLayerScenario(
        name="ResNet-50 classifier head, batch 32, FP16",
        description="2,048 pooled features to 1,000 classes; each weight is reused only across the batch.",
        in_features=2048, out_features=1000,
        bytes_per_weight=_BYTES_FP16,
        batch_size=32,                        # scenario assumption: inference batch
    )
    Gpt2XlQkvProjectionPrefill = _st.DenseLayerScenario(
        name="GPT-2 XL QKV projection at prefill, batch 8 full-context sequences, FP16",
        description="Hidden width to three projections; every token of the batch reuses each weight.",
        in_features=_Models.Language.GPT2.hidden_dim,
        out_features=3 * _Models.Language.GPT2.hidden_dim,
        bytes_per_weight=_BYTES_FP16,
        batch_size=8 * _Models.Language.GPT2.context_length,  # scenario assumption: 8 sequences
    )
    Gpt2XlLayerNormPrefill = _st.ElementwiseOpScenario(
        name="GPT-2 XL LayerNorm at prefill, batch 8 full-context sequences, FP16",
        description="About eight operations per element read and written; scale and shift read once.",
        elements=8 * _Models.Language.GPT2.context_length * _Models.Language.GPT2.hidden_dim,
        flops_per_element=8,                  # scenario assumption: mean, variance, normalize, scale, shift
        bytes_per_value=_BYTES_FP16,
        parameter_values=2 * _Models.Language.GPT2.hidden_dim,
    )
    Gpt2XlSoftmaxPrefill = _st.ElementwiseOpScenario(
        name="GPT-2 XL attention softmax at prefill, batch 8 full-context sequences, FP16",
        description="One score per head per query-key pair; exponential, sum, and divide per score.",
        elements=8 * _Models.Language.GPT2.heads * _Models.Language.GPT2.context_length ** 2,
        flops_per_element=3,                  # scenario assumption: exp, sum, divide
        bytes_per_value=_BYTES_FP16,
    )
    Gpt2XlDecodeH100Roofline = _st.RooflinePointScenario(
        name="GPT-2 XL decoding one token at batch 1 on the H100 FP16 roofline",
        description="Two operations per parameter against every FP16 weight read once.",
        hardware=_Hardware.Cloud.H100,
        intensity=(_Models.Language.GPT2.inference_flops
                   / _Models.Language.GPT2.size_in_bytes(_BYTES_FP16)).to("flop/byte"),
    )
    Gpt2XlLayerNormH100Roofline = Gpt2XlDecodeH100Roofline.variant(
        name="GPT-2 XL LayerNorm on the H100 FP16 roofline",
        description="The unfused LayerNorm's intensity against the H100's ceilings.",
        intensity=Q_(Gpt2XlLayerNormPrefill.intensity, "flop/byte"))
    ResNet50ClassifierHeadH100Roofline = Gpt2XlDecodeH100Roofline.variant(
        name="ResNet-50 classifier head on the H100 FP16 roofline",
        description="The batch-32 head's intensity against the H100's ceilings.",
        intensity=Q_(ResNet50ClassifierHeadBatch32.traffic_intensity, "flop/byte"))
    KernelAt200FlopPerByteA100 = _st.RooflinePointScenario(
        name="A 200 FLOP/byte kernel on the A100 FP16 roofline",
        description="Above the A100 ridge: compute bound at its dense FP16 peak.",
        hardware=_Hardware.Cloud.A100,
        intensity=Q_(200, "flop/byte"),       # scenario assumption: a kernel tuned for the A100
    )
    KernelAt200FlopPerByteH100 = KernelAt200FlopPerByteA100.variant(
        name="The same 200 FLOP/byte kernel on the H100 FP16 roofline",
        description="Below the H100 ridge: bandwidth bound.",
        hardware=_Hardware.Cloud.H100)
    Dense2048Fp16Batch1 = _st.DenseLayerScenario(
        name="A 2,048-to-2,048 dense layer in FP16 at batch 1",
        description="Each weight read serves one example; intensity is about 1 FLOP/byte.",
        in_features=2048, out_features=2048,  # scenario assumption: production-width layer
        bytes_per_weight=_BYTES_FP16,
        batch_size=1,
    )
    Dense2048Fp16Batch32 = Dense2048Fp16Batch1.variant(
        name="A 2,048-to-2,048 dense layer in FP16 at batch 32", batch_size=32)  # scenario assumption
    Dense2048Fp16Batch512 = Dense2048Fp16Batch1.variant(
        name="A 2,048-to-2,048 dense layer in FP16 at batch 512", batch_size=512)  # scenario assumption
    DenseParametersVsBandwidthSince2014 = _st.ParameterBandwidthGrowthScenario(
        name="Dense model parameters against per-device memory bandwidth since 2014",
        description="VGG-16 and one GPU of a K80 board are the 2014 baselines; DeepSeek-V3 is a sparse "
                    "expert model, reported but kept out of the growth figure.",
        accelerators=[_Hardware.Cloud.K80, _Hardware.Cloud.TPUv2, _Hardware.Cloud.V100, _Hardware.Cloud.A100,
                      _Hardware.Cloud.TPUv4, _Hardware.Cloud.H100, _Hardware.Cloud.MI300X,
                      _Hardware.Cloud.TPUv6, _Hardware.Cloud.MI325X, _Hardware.Cloud.B200],
        dense_models=[_Models.Vision.AlexNet, _Models.Vision.VGG16, _Models.Vision.ResNet50,
                      _Models.Language.BERT_Large, _Models.Language.GPT3, _Models.Language.PaLM_540B,
                      _Models.Language.Llama3_405B],
        sparse_models=[_Models.Language.DeepSeek_V3],
        baseline_accelerator=_Hardware.Cloud.K80,
        baseline_model=_Models.Vision.VGG16,
    )
    NvidiaGenerationsV100ToB200 = _st.AcceleratorGenerationsScenario(
        name="Dense FP16 peak and memory bandwidth, V100 through B200",
        description="Five NVIDIA datacenter generations; the H200 keeps the H100's compute with more bandwidth.",
        accelerators=[_Hardware.Cloud.V100, _Hardware.Cloud.A100, _Hardware.Cloud.H100,
                      _Hardware.Cloud.H200, _Hardware.Cloud.B200],
    )

    # --- Mapping a computation onto an accelerator (added 2026-10-20) ---
    Matmul4096Fp16Tile128 = _st.MatmulTileScenario(
        name="A 4,096-square FP16 matrix multiply in 128 x 128 x 128 tiles, single buffered",
        description="FP16 operands, FP32 accumulator; 1,024 output tiles, each summing 32 tile products.",
        m=4096, n=4096, k=4096,               # scenario assumption: a transformer-width layer
        tile_m=_Hardware.Cloud.TPUv4.compute.systolic_array_dim,  # tiles the width of a 128-wide array
        tile_n=_Hardware.Cloud.TPUv4.compute.systolic_array_dim,
        tile_k=_Hardware.Cloud.TPUv4.compute.systolic_array_dim,
        operand_bytes=_BYTES_FP16,
        accumulator_bytes=_BYTES_FP32,
    )
    ResNet50Stage2MapInt8ReadOnce = _st.MemoryRoundTripScenario(
        name="One ResNet-50 stage-2 output map (INT8, one image) read once from DRAM at 160 pJ/byte",
        description="45 nm off-chip DRAM energy per byte (Horowitz).",
        tensor_bytes=ResNet50Stage2ConvBatch32Int8.activation_bytes,
        energy_per_byte=_Hardware.Tech.Memory.DRAM.energy_per_byte,
        trips=1,
    )

    # --- Trajectory planning on the warehouse mobile manipulator (added 2026-10-20) ---
    # Joint rates through the arm's Jacobian, a replanner's mission exposure, chunk
    # replacement, the jerk-limited stops of the arm and the base, a trajectory
    # record's deadlines, a seam bridge on the Panda-class joint, the loaded
    # shoulder's swing, the base's dynamic window, and a stale anchor.
    WarehouseArmRightAngleElbow = _st.TwoLinkJointRateScenario(
        name="Warehouse arm at a right-angle elbow, TCP speed limit in the weakest direction",
        description="Planar two-link abstraction of the arm (FER link lengths): joint rates the 1 m/s TCP "
                    "limit asks, exact and with 50 mm damped least squares.",
        machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
        elbow_angle=Q_(90.0, "degree"),  # scenario assumption: right-angle elbow
        damping=Q_(50.0, "mm"),  # scenario assumption: illustrative damping, not a vendor setting
    )
    WarehouseArmNearStraightElbow = WarehouseArmRightAngleElbow.variant(
        name="Warehouse arm at a nearly straight elbow, TCP speed limit in the weakest direction",
        description="The right-angle case with the elbow straightened to 10 degrees, near the singular pose.",
        elbow_angle=Q_(10.0, "degree"),  # scenario assumption: nearly straight elbow
    )
    GenericReplannerMissionExposure = _st.HandoffExposureScenario(
        name="A p99 replanner at 5 Hz over a ten-minute mission",
        description="Handoffs, expected late handoffs, the chance all are on time, and the per-handoff "
                    "budget a 1-in-1,000 mission target leaves (independent misses).",
        mission_time=Q_(600.0, "s"),  # scenario assumption: ten-minute mission
        replan_rate=Q_(5.0, "Hz"),  # scenario assumption: a generic replanner, slower than the chunk renewal
        late_probability=0.01,  # definitional: exceedance of a p99 budget
        mission_budget=1e-3,  # scenario assumption: one failed mission in 1,000
    )
    WarehouseArmChunkReplacement = _st.ChunkReplacementScenario(
        name="Replacing the warehouse arm's admitted chunk",
        description="Capture-to-admission latency from the observation age and the dispatch-to-admission "
                    "allocation, against the 50 ms renewal and the 60 ms chunk lease.",
        machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
        site=_Embodied.Scenario.WarehouseAisle,
        interference_tail=Q_(150.0, "ms"),  # scenario assumption: inference pass under memory contention or preemption
    )
    WarehouseArmStallStop = _st.JerkLimitedStopScenario(
        name="Warehouse arm stalled at its TCP limit: lease path, then a jerk-limited stop",
        description="The proposer stalls; the chunk lease lapses, one tick, one bus cycle, and the brake "
                    "onset pass, then an S-curve stop with an empty gripper, against the clearance ahead.",
        speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.tcp_speed_limit,
        deceleration=Q_(5.0, "m/s**2"),  # scenario assumption: arm stop deceleration bound, empty gripper
        delay=(_Embodied.Scenario.WarehouseAisle.t_lease + _Embodied.Scenario.WarehouseAisle.t_tick
               + _Embodied.Scenario.WarehouseAisle.t_bus
               + _Embodied.Scenario.WarehouseAisle.t_brake_onset).to("ms"),  # brake onset charged as the arm's
        clear_distance=ConveyorMugIntentLeaseExpiry.clearance,
    )
    WarehouseArmOpenMugStop = _st.SpillLimitedStopScenario(
        name="Warehouse arm stalled while carrying an open mug",
        description="The stall stop with the deceleration capped where the liquid reaches the rim, its "
                    "jerk rescaled with it, and the carry speed whose stop fits the clearance.",
        free_stop=WarehouseArmStallStop,
        inner_radius=Q_(40.0, "mm"),  # scenario assumption: mug inner radius
        freeboard=Q_(10.0, "mm"),  # scenario assumption: liquid surface below the rim
    )
    WarehouseBaseJerkLimitedStop = _st.JerkLimitedStopScenario(
        name="Warehouse base at its drive limit stopping along an S-curve at the rack end",
        description="The elementary budget's delay, localization bound, and margin with the constant-"
                    "deceleration stop replaced by an S-curve whose jerk bound is the reference C2 "
                    "polynomial's peak at the drive limit.",
        speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
        deceleration=_Embodied.Scenario.WarehouseAisle.a_brake,
        jerk=_Embodied.Scenario.WarehouseAisle.j_brake,  # the site's stop jerk bound, shared with WarehouseFinishedStopAtDriveLimit
        delay=_Embodied.Scenario.WarehouseAisle.tau_delay,
        site=_Embodied.Scenario.WarehouseAisle,
    )
    WarehouseArmCommitmentSchedule = _st.TrajectoryDeadlinesScenario(
        name="Deadlines of a longer-lease trajectory record for the warehouse arm",
        description="Preferred admission, full reserve, admission cutoff, commitment, and rest for a "
                    "replacement requested 60 ms into the schedule, with the stall stop as its suffix.",
        machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
        site=_Embodied.Scenario.WarehouseAisle,
        stop=WarehouseArmStallStop,
        request_delay=Q_(60.0, "ms"),  # scenario assumption: request delay
        latency_target=Q_(160.0, "ms"),  # scenario assumption: design-tail replacement-latency target
        schedule_slack=Q_(20.0, "ms"),  # scenario assumption: scheduling slack before preferred admission
        blend=Q_(240.0, "ms"),  # scenario assumption: blend deadline
        commit=Q_(300.0, "ms"),  # scenario assumption: physical suffix onset
        late_wait=Q_(200.0, "ms"),  # scenario assumption: tracking authority run past commitment
    )
    WarehouseArmShoulderSwing = _st.ShoulderSwingScenario(
        name="Warehouse arm shoulder swinging the rated payload at full reach",
        description="Payload-only holding torque and inertia with the Panda-class joint's reflected rotor "
                    "inertia, and a 0.5 rad rest-to-rest swing, bang-bang and jerk-limited.",
        machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
        joint=_Actuators.PandaClassJoint.CSG_25_50_ILM70x18,
        swing=Q_(0.5, "radian"),  # scenario assumption: swing amplitude
    )
    PandaClassJointSeamBridge = _st.SeamBridgeScenario(
        name="A 0.2 rad/s seam gap on the Panda-class joint: one tick against a 50 ms quintic bridge",
        description="Reflected rotor inertia alone (rigid), screened against the arm's torque-rate limit; "
                    "the loaded step adds the rated payload at full reach.",
        joint=_Actuators.PandaClassJoint.CSG_25_50_ILM70x18,
        machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
        velocity_gap=Q_(0.2, "rad/s"),  # scenario assumption: reference velocity gap at the seam
        tick=_Embodied.Scenario.WarehouseAisle.t_tick,
        bridge_duration=Q_(50.0, "ms"),  # scenario assumption: bridge duration
        loaded_inertia=WarehouseArmShoulderSwing.result.effective_inertia,
    )
    PandaClassJointSeamBridgeHalved = PandaClassJointSeamBridge.variant(
        name="The Panda-class seam bridge at half its duration",
        description="The same gap bridged in half the time: torque doubles, jerk and torque rate quadruple.",
        bridge_duration=PandaClassJointSeamBridge.bridge_duration / 2,  # scenario assumption: halved bridge
    )
    PandaClassJointSeamBridgeCompressed = PandaClassJointSeamBridge.variant(
        name="The Panda-class seam bridge squeezed to 10 ms",
        description="A late admission leaves 10 ms for the same gap.",
        bridge_duration=Q_(10.0, "ms"),  # scenario assumption: compressed bridge
    )
    WarehouseBaseDynamicWindow = _st.DynamicWindowScenario(
        name="Warehouse base's dynamic window over one chunk renewal",
        description="Speeds reachable in 50 ms and the classical constant-deceleration admissible speed "
                    "on the rack-end clear distance.",
        machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
        site=_Embodied.Scenario.WarehouseAisle,
    )
    WarehouseArmStaleAnchor = _st.StaleAnchorScenario(
        name="A plan anchored at the pose sampled when the replacement was requested",
        description="A joint at 0.50 rad/s during an 80 ms replacement, at the arm's planar reach.",
        machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
        joint_speed=Q_(0.50, "rad/s"),  # scenario assumption: constant joint speed
        delay=Q_(80.0, "ms"),  # scenario assumption: compute and transport of the replacement
    )

    # --- Trajectory curation: triage, splits, admission, denominators, staging (added 2026-10-20) ---
    RepairAgentFailureTriage = _st.FailureTriageScenario(
        name="A maintenance agent's month of failures, triaged by counterfactual probes",
        description="50,000 trajectories at 6 percent failing; probes attribute 70 percent to a tool schema that "
                    "omits a required field and 20 percent to sandbox startup timeouts. Fine-tuning on every "
                    "failure: Llama 3 70B, 4,096-token sequences, 3 epochs, one node of eight H100s.",
        runs=50_000,                      # scenario assumption: trajectories per month
        failure_rate=0.06,                # scenario assumption: failing share
        probe_shares={"interface": 0.70,  # scenario assumption: schema probe removes the failure
                      "runtime": 0.20},   # scenario assumption: runtime probe removes the failure
        workload=_Models.Language.Llama3_70B,
        hardware=_Hardware.Cloud.H100,
        devices=8,                        # scenario assumption: one node
        utilization=float(_Literature.Training.MfuLow),
        hourly_price=_Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
        sequence_tokens=4_096,            # scenario assumption: tokens per training sequence
        epochs=3,                         # scenario assumption
    )
    EightAttemptRunSplit = _st.RunSplitLeakageScenario(
        name="Eight attempts per task, each run sent to training with probability 0.8",
        description="A random 80/20 split of runs rather than of tasks.",
        attempts=8,                       # scenario assumption: attempts per task
        train_fraction=0.8,               # scenario assumption
    )
    TrajectoryMinHashScreen = _st.MinHashBandingScenario(
        name="A 128-hash MinHash screen in 32 bands of 4 rows over 100,000 trajectory records",
        description="Command n-grams and code-tree features per record; 32-bit hash values stored raw.",
        bands=32,                         # scenario assumption
        rows=4,                           # scenario assumption
        probe_similarities=[0.5, 0.8],    # scenario assumption: a loose and a near-duplicate pair
        hash_bytes=Q_(4, "byte"),         # 32-bit hash values
        records=100_000,                  # scenario assumption
    )
    DemonstrationAdmissionOrder = _st.AdmissionOrderScenario(
        name="A demonstration route's five admission checks, process audit before tests and after",
        description="Record, use, process, outcome, and target checks, independent, on 100,000 candidates; "
                    "the two orders swap the process audit and the test run.",
        checks={"record": (0.95, Q_(0.01, "second")),   # scenario assumption: pass rate, slot time
                "use": (0.90, Q_(0.04, "second")),      # scenario assumption
                "process": (0.80, Q_(0.20, "second")),  # scenario assumption
                "outcome": (0.40, Q_(8.0, "second")),   # scenario assumption
                "target": (0.75, Q_(2.0, "second"))},   # scenario assumption
        orders=[["record", "use", "process", "outcome", "target"],
                ["record", "use", "outcome", "process", "target"]],
        candidates=100_000,               # scenario assumption
    )
    DemonstrationAdmissionOrderSlowAudit = DemonstrationAdmissionOrder.variant(
        name="The same route with a slower, less selective process audit",
        description="The process audit takes 2 s and passes 95 percent, so the tests should run first.",
        checks={**DemonstrationAdmissionOrder.checks,
                "process": (0.95, Q_(2.0, "second"))},  # scenario assumption
    )
    RecoveryPoolDenominators = _st.CorpusDenominatorsScenario(
        name="A retained pool of 50,000 direct, recovery, and negative records",
        description="Recovery records are longer and only their corrections are targets; the demonstration "
                    "route admits direct and recovery records.",
        classes={"direct": (32_500, Q_(7_200, "token"), Q_(1_200, "token")),      # scenario assumption
                 "recovery": (12_500, Q_(18_500, "token"), Q_(2_000, "token")),   # scenario assumption
                 "negative": (5_000, Q_(9_800, "token"), Q_(0, "token"))},        # scenario assumption
        focus="recovery",
        admitted=["direct", "recovery"],
    )
    EightAttemptGroupContrast = _st.AttemptGroupContrastScenario(
        name="Eight attempts per fixture at pass rates of 5, 50, and 95 percent",
        description="How often a group of attempts holds a success, and a success beside a failure.",
        attempts=8,                       # scenario assumption
        pass_rates=[0.05, 0.50, 0.95],    # scenario assumption
    )
    VerifierTimeoutStagingBurst = _st.StagingBurstScenario(
        name="A collector's staging store during a verifier timeout burst",
        description="1,024 rollouts of 45 s feed 512 verifier slots; a defective fixture drives every test to "
                    "its 120 s timeout; 8 MB record ceiling, 32 GB store, 30 s controller delay.",
        rollouts=1_024,                   # scenario assumption
        rollout_time=Q_(45, "second"),    # scenario assumption
        slots=512,                        # scenario assumption
        mean_service=Q_(16.875, "second"),  # scenario assumption
        timeout=Q_(120, "second"),        # scenario assumption
        record_ceiling=Q_(8, "MB"),       # scenario assumption
        store=Q_(32, "GB"),               # scenario assumption
        reaction=Q_(30, "second"),        # scenario assumption
    )
    CurationCohortDispositions = _st.CohortDispositionScenario(
        name="A cohort of 500 collection attempts by disposition at the cutoff",
        description="Fifteen selected and emitted; the rest assessed but not selected, ineligible, pending, "
                    "or unknown. Environments, generation, tools, checks, storage, and review cost $300.",
        dispositions={"selected_emitted": 15,        # scenario assumption
                      "assessed_not_selected": 470,  # scenario assumption
                      "ineligible": 10,              # scenario assumption
                      "pending": 3,                  # scenario assumption
                      "unknown": 2},                 # scenario assumption
        emitted="selected_emitted",
        cohort_cost=Q_(300, "USD"),       # scenario assumption
        yield_multiple=2.0,               # scenario assumption: the same cohort at double the yield
    )

    # --- Fine-tuning on trajectories (added 2026-10-20) ------------------------
    # The repair agent's recovery record turned into training examples, a weight
    # update, and a release decision. Token counts, log-probabilities, prices,
    # and trial outcomes are hypothetical stipulations, not measurements.
    RepairTrajectoryPerCallExamples = _st.PerInvocationSerializationScenario(
        name="A repair trajectory stored as one training example per model call",
        description="The repair harness's step budget in turns, each turn adding about the same tokens; "
                    "every call's delivered prefix stored separately against one transcript.",
        turns=_Agents.Coding.SWE_Bench_Runner.max_trajectory_steps,
        tokens_per_turn=CodingAgentResend30Turns.appended,
    )
    RecoveryRecordTargetMask = _st.TargetMaskScenario(
        name="One serialized repair recovery record and its selected targets",
        description="16,384 stored tokens: 1,200 of prompt, 184 of task, 1,400 of reviewed rationale, 960 "
                    "of policy response, and 12,640 of observation; the refused proposal's 160 response "
                    "tokens stay context.",
        prompt=1_200,  # scenario assumption: includes the initial boundary position
        task=184,  # scenario assumption
        rationale=1_400,  # scenario assumption: available, authorized, reviewed text
        response=960,  # scenario assumption: response content and authentic stop tokens
        observation=12_640,  # scenario assumption
        unselected_response=160,  # scenario assumption: the refused proposal, kept as context
    )
    RecoveryRecordTargetMaskDoubledObservations = RecoveryRecordTargetMask.variant(
        name="The same recovery record with its observations doubled",
        description="Every other segment unchanged.",
        observation=2 * RecoveryRecordTargetMask.observation,  # scenario assumption: doubled observations
    )
    RecoveryRecordTargetMaskLongObservations = RecoveryRecordTargetMask.variant(
        name="The same recovery record with 30,000 observation tokens",
        description="Every other segment unchanged.",
        observation=30_000,  # scenario assumption: longer observations
    )
    RepairTraceWithLookupWeighting = _st.LossNormalizationScenario(
        name="A long repair trace and a one-turn file lookup in one batch",
        description="8,192 selected targets against 256: token averaging against example averaging.",
        focus_targets=8_192,  # scenario assumption
        other_examples=1,  # scenario assumption
        other_targets=256,  # scenario assumption
    )
    EightExamplePackingBatch = _st.SequencePackingScenario(
        name="Eight independent examples, 512 to 32,768 tokens, against 32,768-token buffers",
        description="Padding every row to the longest against first-fit decreasing packing.",
        lengths=[32_768, 8_192, 4_096, 2_048, 1_024, 1_024, 512, 512],  # scenario assumption
        capacity=32_768,  # scenario assumption: buffer positions
    )
    RepairPolicyImitationBounds = _st.ImitationBoundScenario(
        name="Imitation cost bounds over the repair harness's 30-turn horizon",
        description="A 2 percent disagreement rate on reference states against the same rate on learner "
                    "states with deviation disadvantage u = 1.",
        horizon=RepairTrajectoryPerCallExamples.turns,
        per_step_error=0.02,  # scenario assumption
        recoverability=1.0,  # scenario assumption: one slip costs at most one unit
    )
    RepairPreferencePair = _st.PreferencePairScenario(
        name="Two passing repair continuations scored against a frozen reference",
        description="Natural-log probability sums over the selected tokens: preferred -8 against a "
                    "reference -10, dispreferred -11 against -10, beta 0.2.",
        policy_preferred=-8.0,  # scenario assumption
        reference_preferred=-10.0,  # scenario assumption
        policy_dispreferred=-11.0,  # scenario assumption
        reference_dispreferred=-10.0,  # scenario assumption
        beta=0.2,  # scenario assumption
    )
    RepairPreferencePairDoubledBeta = RepairPreferencePair.variant(
        name="The same preference pair at twice the beta",
        description="Identical log-probabilities; only the preference scale changes.",
        beta=2 * RepairPreferencePair.beta,  # scenario assumption: doubled preference scale
    )
    Rank16AdaptersOnHypothetical8B = _st.TrainingStateFitScenario(
        name="Rank-16 low-rank adapters on a hypothetical 8B-parameter transformer, mixed-precision Adam",
        description="Two 4,096-by-4,096 matrices adapted in each of 32 layers; 2 B per frozen parameter, "
                    "16 B per trained parameter, before activations.",
        workload=_TransformerWorkload(
            name="Hypothetical 8B-parameter transformer",
            architecture="Transformer",
            parameters=Q_(8e9, "param"),  # scenario assumption: exact illustrative size
            layers=32,  # scenario assumption
            hidden_dim=4_096,  # scenario assumption
        ),
        bytes_per_value=_BYTES_FP16,
        optimizer_bytes_per_parameter=_BYTES_ADAM_MOMENTS,
        master_bytes_per_parameter=_BYTES_FP32,
        adapter_rank=16,  # scenario assumption
        adapted_matrices=2,  # scenario assumption: matrices adapted per layer
        adapter_in_dim=4_096,  # scenario assumption
        adapter_out_dim=4_096,  # scenario assumption
    )
    Rank32AdaptersOnHypothetical8B = Rank16AdaptersOnHypothetical8B.variant(
        name="The same adapters at rank 32",
        description="Adapter state is linear in the rank; the frozen base is unchanged.",
        adapter_rank=2 * Rank16AdaptersOnHypothetical8B.adapter_rank,  # scenario assumption: doubled rank
    )

    # --- Quantization (added 2026-10-20) --------------------------------------
    Uint8RangeMinusOneToThree = _st.QuantizationGridScenario(
        name="A 2 by 2 activation matrix with range [-1, 3] stored in UINT8",
        description="Affine scale and zero-point by hand; real zero must land on an integer.",
        values=[-1.0, 1.0, 0.0, 3.0],      # scenario assumption: the matrix, row-major; no entry sits on a rounding tie
        bits=8,
        alpha=-1.0,                        # scenario assumption: clipping range
        beta=3.0,
    )
    Gpt2XlFp16DecodeFullContextPhone = Gpt2XlFp16WeightReadPhone.variant(
        name="GPT-2 XL decoding at a full context, FP16 weights and KV cache, on a flagship phone",
        description="Each batch-one token reads every weight and the whole KV cache.",
        context_tokens=_Models.Language.GPT2.context_length,
        kv_bytes_per_value=_BYTES_FP16,
    )
    Gpt2XlInt4DecodeFullContextPhone = Gpt2XlInt4WeightReadPhone.variant(
        name="GPT-2 XL decoding at a full context, INT4 weights and an FP16 KV cache, on a flagship phone",
        description="Unpacked to FP16 inside the kernel, or in a separate pass that writes and rereads them.",
        context_tokens=_Models.Language.GPT2.context_length,
        kv_bytes_per_value=_BYTES_FP16,
        unpacked_bytes_per_weight=_BYTES_FP16,
    )
    Int8VsFp32OperationEnergy45nm = _st.OperationEnergyComparisonScenario(
        name="INT8 against FP32 energy per add, multiply, and multiply-accumulate at 45 nm",
        description="The INT8 MAC accumulates in INT32; a 32-bit DRAM read against the INT8 MAC.",
        wide_add=_Hardware.Tech.Op.AddFp32.energy,
        wide_mul=_Hardware.Tech.Op.FlopFp32.energy,
        narrow_add=_Hardware.Tech.Op.AddInt8.energy,
        narrow_mul=_Hardware.Tech.Op.OpInt8.energy,
        accumulate_add=_Hardware.Tech.Op.AddInt32.energy,
        memory_read=_Hardware.Tech.Memory.DRAM.energy_per_access,
    )
    ResNet50Int8ProfileH100 = _st.QuantizedDeploymentProfileScenario(
        name="ResNet-50 in FP32 and INT8 on an H100, against accuracy and calibration limits",
        description="Accuracy and confidence calibration are stipulated; latency and throughput come from the "
                    "shared H100 engine profile; energy per image assumes full board power in both.",
        workload=_Models.Vision.ResNet50,
        hardware=_Hardware.Cloud.H100,
        profile=_resnet50_h100_profile(),
        float_bytes=_BYTES_FP32,
        int_bytes=_BYTES_INT8,
        float_top1_pct=76.1,               # scenario assumption: FP32 top-1
        int_top1_pct=75.8,                 # scenario assumption: INT8 top-1
        float_calibration_error_pct=2.1,   # scenario assumption: FP32 confidence-calibration error
        int_calibration_error_pct=3.4,     # scenario assumption: INT8 confidence-calibration error
        top1_limit_pct=75.5,               # scenario assumption: acceptance limit
        calibration_error_limit_pct=3.0,   # scenario assumption: acceptance limit
    )

    # --- Supervisory intervention on the warehouse base (added 2026-10-20) ---
    # Site record Embodied.Scenario.WarehouseAisle: the finished stopping budget
    # (S-curve jerk bound, tracking inset), the sourced takeover terms, the
    # dispatch-to-admission allocation, the permission tick, and the chunk lease.
    WarehouseInLoopTakeoverAtAisleSpeed = _st.TakeoverBudgetScenario(
        name="An in-loop takeover requested at the warehouse aisle speed",
        description="Slow-bound Model Human Processor response plus video and the four-phase handshake, "
                    "waited out before braking at the candidate aisle speed; the handover timer is clipped "
                    "to the fallback deadline.",
        site=_Embodied.Scenario.WarehouseAisle,
        speed=_Embodied.Scenario.WarehouseAisle.v_aisle,
        takeover=_Embodied.Scenario.WarehouseAisle.t_takeover_in_loop,
        configured_timeout=Q_(500.0, "ms"),  # scenario assumption: configured protocol ceiling on the handover timer
        manual_lease=Q_(100.0, "ms"),        # scenario assumption: the operator's loss-of-engagement lease
    )
    WarehouseOutOfLoopTakeoverAtAisleSpeed = WarehouseInLoopTakeoverAtAisleSpeed.variant(
        name="An out-of-loop takeover requested at the warehouse aisle speed",
        description="The 90th-percentile condition-mean take-over time plus the same console transport.",
        takeover=_Embodied.Scenario.WarehouseAisle.t_takeover_out_of_loop,
    )
    WarehouseInLoopTakeoverAtRequestSpeed = WarehouseInLoopTakeoverAtAisleSpeed.variant(
        name="An in-loop takeover requested after slowing to the in-loop request speed",
        description="The dispatch-charged in-loop ceiling rounded down to the speed resolution.",
        speed=_Embodied.Scenario.WarehouseAisle.v_takeover_in_loop,
    )
    WarehouseInLoopTakeoverAtRetiredSlowdown = WarehouseInLoopTakeoverAtAisleSpeed.variant(
        name="An in-loop takeover requested at 0.8 m/s, the partial ceiling rounded down",
        description="A slow-down that fits the partial budget and misses once dispatch to admission is charged.",
        speed=Q_(0.8, "m/s"),  # scenario assumption: the partial in-loop ceiling rounded down to 0.05 m/s
    )
    WarehouseInLoopTakeoverAtFastSlowdown = WarehouseInLoopTakeoverAtAisleSpeed.variant(
        name="An in-loop takeover requested at 1.2 m/s",
        description="A slow-down that would fit only a much faster human response.",
        speed=Q_(1.2, "m/s"),  # scenario assumption: illustrative slow-down
    )
    WarehouseOutOfLoopTakeoverAtCrawl = WarehouseOutOfLoopTakeoverAtAisleSpeed.variant(
        name="An out-of-loop takeover requested at the crawl speed",
        description="The chosen crawl under the out-of-loop ceiling.",
        speed=_Embodied.Scenario.WarehouseAisle.v_takeover_out_of_loop,
    )
    WarehouseOutOfLoopTakeoverAtCrawlP99 = WarehouseOutOfLoopTakeoverAtCrawl.variant(
        name="A 99th-percentile out-of-loop takeover at the crawl speed",
        description="The 99th-percentile condition-mean take-over time plus the console transport.",
        takeover=(_Embodied.Scenario.WarehouseAisle.t_takeover_driving_p99
                  + _Embodied.Scenario.WarehouseAisle.t_takeover_transport).to("ms"),
    )
    WarehouseFixedWindowAtAisleSpeed = WarehouseInLoopTakeoverAtAisleSpeed.variant(
        name="The configured 500 ms handover window waited out at the aisle speed",
        description="One fixed acknowledgment window judged against the finished budget.",
        takeover=WarehouseInLoopTakeoverAtAisleSpeed.configured_timeout,
    )
    WarehouseFixedWindowAtCrawl = WarehouseFixedWindowAtAisleSpeed.variant(
        name="The configured 500 ms handover window waited out at the crawl speed",
        description="The same window at the out-of-loop crawl.",
        speed=_Embodied.Scenario.WarehouseAisle.v_takeover_out_of_loop,
    )
    WarehouseTakeoverCeilingLadder = _st.TakeoverCeilingLadderScenario(
        name="Speed ceilings of the warehouse base against sourced human responses",
        description="Model Human Processor simple reaction (nominal, slow bound), perception-brake times "
                    "(expected, unexpected, surprise), and percentiles of condition-mean take-over time, "
                    "each plus video and the four-phase handshake; partial budget.",
        base=WarehouseInLoopTakeoverAtAisleSpeed,
        transport=_Embodied.Scenario.WarehouseAisle.t_takeover_transport,
        responses={
            "mhp": _Embodied.Scenario.WarehouseAisle.t_human_simple_reaction,
            "mhp_slow": _Embodied.Scenario.WarehouseAisle.t_human_simple_reaction_slow,
            "expected": _Embodied.Scenario.WarehouseAisle.t_brake_response_expected,
            "unexpected": _Embodied.Scenario.WarehouseAisle.t_brake_response_unexpected,
            "surprise": _Embodied.Scenario.WarehouseAisle.t_brake_response_surprise,
            "p50": _Embodied.Scenario.WarehouseAisle.t_takeover_driving_p50,
            "mean": _Embodied.Scenario.WarehouseAisle.t_takeover_driving_mean,
            "p90": _Embodied.Scenario.WarehouseAisle.t_takeover_driving_p90,
            "p99": _Embodied.Scenario.WarehouseAisle.t_takeover_driving_p99,
        },
    )
    WarehouseTakeoverBrakingBlend = _st.BrakingBlendScenario(
        name="A bumpless transfer into braking at the in-loop request speed, and an arm-joint blend",
        description="Smoothstep from cruise to the credible deceleration under the resident stop's jerk "
                    "bound, against a step and the resident S-curve stop; an arm-joint torque change under "
                    "a torque-rate ceiling.",
        site=_Embodied.Scenario.WarehouseAisle,
        speed=_Embodied.Scenario.WarehouseAisle.v_takeover_in_loop,
        torque_change=Q_(15.0, "N*m"),         # scenario assumption: an arm-joint command change (illustrative)
        torque_rate_limit=Q_(750.0, "N*m/s"),  # scenario assumption: torque-command rate ceiling (chosen)
    )
    WarehouseRemoteCommandAuthentication = _st.CommandAuthenticationScenario(
        name="Remote takeover commands authenticated on the warehouse machine's safety MCU",
        description="Published Cortex-M4 cycle counts (Ed25519 verification, AES-128 block) at the permission "
                    "MCU's clock, against the aisle-speed spare and fallback deadline.",
        budget=WarehouseInLoopTakeoverAtAisleSpeed,
        clock=_Embodied.MobileManipulator.WarehouseMobileManipulator.permission_mcu.compute.clock_rate,
        verify_cycles=720000.0,  # Owens, D., et al. (2024), Efficient and Side-Channel Resistant Ed25...
        mac_block_cycles=661.7,  # Schwabe, P., and Stoffelen, K. (2016), All the AES You Need on Cort...
        mac_blocks=3,                     # scenario assumption: a 48-byte frame (counter, timestamp, role, mask, command)
        freshness_window=Q_(100.0, "ms"),  # scenario assumption: freshness window (illustrative)
    )

    # --- Fault injection at the warehouse rack end: oracles, supply, and campaign statistics (added 2026-10-20) ---
    # The renewal-referenced oracle scores a proposer stall from the last valid
    # lease renewal: onset after the lease path and brake onset (the chunk
    # lease's admission-to-brake-onset), the resident S-curve stop, and the
    # tracking inset; it carries no evidence age and no localization or
    # protective allowance, because a stalled proposer has no detected target.
    WarehouseRenewalReferencedStop = _st.StoppingBudgetScenario(
        name="Warehouse base's renewal-referenced stopping oracle at the candidate aisle speed",
        description="Onset bound after the last valid renewal, the resident S-curve stop, and the tracking "
                    "inset: the travel bound of a proposer-stall trial.",
        speed=_Embodied.Scenario.WarehouseAisle.v_aisle,
        delay=WarehouseChunkLease.result.admission_to_brake_onset,
        deceleration=_Embodied.Scenario.WarehouseAisle.a_brake,
        clear_distance=_Embodied.Scenario.WarehouseAisle.d_clear,
        jerk=_Embodied.Scenario.WarehouseAisle.j_brake,
        tracking_bound=_Embodied.Scenario.WarehouseAisle.eps_track,
    )
    WarehouseRenewalReferencedStopRestricted = WarehouseRenewalReferencedStop.variant(
        name="Renewal-referenced stopping oracle at the restricted point on the inspected floor",
        description="1.0 m/s, below the inspected floor's ceiling, with deceleration capped by its friction "
                    "(optimistic all-weight screen); the jerk bound belongs to the load and is unchanged.",
        speed=Q_(1.0, "m/s"),  # scenario assumption: candidate restricted speed on inspected floors
        deceleration=_traction_limit(_Embodied.Scenario.WarehouseAisle.mu_inspected_floor),
    )
    WarehouseFinishedStopRestricted = WarehouseFinishedStopInspectedFloor.variant(
        name="Capture-referenced finished budget at the restricted point on the inspected floor",
        description="The design subtotal from capture at the restricted speed and the inspected floor's "
                    "deceleration.",
        speed=WarehouseRenewalReferencedStopRestricted.speed,
    )
    WarehouseFlagReferencedStop = WarehouseRenewalReferencedStop.variant(
        name="Stop at the candidate aisle speed scored from an undervoltage or feed-loss flag",
        description="One permission tick, one bus cycle, and the brake onset from the flag, the resident "
                    "S-curve stop, and the tracking inset, when the drives keep their authority.",
        delay=(_Embodied.Scenario.WarehouseAisle.t_tick + _Embodied.Scenario.WarehouseAisle.t_bus
               + _Embodied.Scenario.WarehouseAisle.t_brake_onset).to("ms"),
    )
    WarehouseRenewalStopLateOnsetTrial = _st.OnsetTrialScenario(
        name="A proposer-stall trace whose onset is 30 ms late at the candidate aisle speed",
        description="Signed margins against the renewal-referenced oracle: the onset predicate fails while "
                    "the travel predicate passes.",
        oracle=WarehouseRenewalReferencedStop,
        onset=(WarehouseChunkLease.result.admission_to_brake_onset
               + Q_(30.0, "ms")).to("ms"),  # scenario assumption: illustrative late onset (a memory stall)
        lease=_Embodied.Scenario.WarehouseAisle.t_lease,
        probe_after_expiry=Q_(5.0, "ms"),  # scenario assumption: chosen replay of a refused renewal
    )
    WarehouseRenewalStopNinetyMsOnsetTrial = WarehouseRenewalStopLateOnsetTrial.variant(
        name="A proposer-stall trace with deceleration onset measured at 90 ms",
        description="A checkpoint given: the onset margin and the travel the late onset adds.",
        onset=Q_(90.0, "ms"),  # scenario assumption: checkpoint given, a measured onset
    )
    WarehousePermissionRailHoldUp = _st.RailHoldUpScenario(
        name="The warehouse machine's permission-rail hold-up during the longest inspected-floor stop",
        description="Feed lost while the base stops from the inspected floor's optimistic ceiling: flag to "
                    "onset, the S-curve stop, and spring-brake engagement against the rail's hold-up.",
        site=_Embodied.Scenario.WarehouseAisle,
        stop=WarehouseFinishedStopInspectedFloor,
    )
    WarehouseQueueDelaySpareExhaustion = _st.ExponentialRareEventScenario(
        name="An unbudgeted queue delay that spends the rack-end spare at the candidate aisle speed",
        description="Exponential delay per stop; failure past the spare's delay at the aisle speed; plain "
                    "Monte Carlo, zero-failure testing, and a cross-entropy importance-sampling proposal.",
        budget=WarehouseFinishedStopAtAisleSpeed,
        mean=Q_(5.0, "ms"),   # scenario assumption: illustrative mean queue delay per stop
        elite_share=0.1,      # scenario assumption: chosen cross-entropy elite fraction
        relative_error=0.1,   # scenario assumption: chosen target relative standard error
        confidence=0.95,      # scenario assumption: chosen
    )
    WarehouseQueueDelayChargedSpareExhaustion = WarehouseQueueDelaySpareExhaustion.variant(
        name="The same queue delay against the spare left after the dispatch-to-admission allocation",
        description="The failure threshold after the dispatch-to-admission allocation is charged.",
        budget=WarehouseFinishedStopWithDispatch,
    )
    WarehouseDiagnosticCoverageHundredFit = _st.DiagnosticCoverageScenario(
        name="Diagnostic coverage and zero-miss injections that leave 100 FIT unmitigated",
        description="Four component strata of the running machine's raw fault rate, a residual-rate target, "
                    "and 95 percent confidence.",
        strata={  # scenario assumption: illustrative component fault rates, FIT
            "sensors": 1500.0,
            "compute": 1000.0,
            "transceivers and power": 2500.0,
            "gate drivers": 4800.0,
        },
        target=100.0,     # scenario assumption: chosen unmitigated-rate target, FIT
        confidence=0.95,  # scenario assumption: chosen
    )
    WarehouseDiagnosticCoverageTenFit = WarehouseDiagnosticCoverageHundredFit.variant(
        name="Diagnostic coverage and zero-miss injections that leave 10 FIT unmitigated",
        description="A tenfold tighter residual-rate target on the same strata.",
        target=10.0,      # scenario assumption: chosen tighter target, FIT
    )

    # --- Benchmarking (added 2026-10-20) ------------------------------------
    DatasheetVersusSustainedEdgeChip = _st.EfficiencyClaimGapScenario(
        name="A \"10 TOPS at 0.5 W\" datasheet line against sustained load",
        description="An edge accelerator's datasheet operating point against the same chip under sustained "
                    "inference load, after thermal throttling (illustrative, no named device).",
        claimed_rate=Q_(10, "TOPS"),  # scenario assumption: datasheet peak
        claimed_power=Q_(0.5, "W"),  # scenario assumption: datasheet power
        sustained_rate=Q_(3, "TOPS"),  # scenario assumption: throughput under sustained load
        sustained_power=Q_(2, "W"),  # scenario assumption: power under sustained load
    )
    MobileNetV2CameraTrapBoard = _st.HostAcceleratorBoardScenario(
        name="MobileNetV2 on a camera trap's host CPU and attached INT8 accelerator",
        description="A labeled composite edge board: FP32 and INT8 on the low-power host CPU, INT8 on the "
                    "attached accelerator; end to end adds host preprocessing and the link copy.",
        host_fp32_latency=Q_(50, "ms"),  # scenario assumption: FP32 on the host CPU
        host_int8_latency=Q_(15, "ms"),  # scenario assumption: INT8 on the host CPU
        accelerator_latency=Q_(2, "ms"),  # scenario assumption: INT8 on the accelerator
        host_end_to_end=Q_(18, "ms"),  # scenario assumption: host CPU, full request path
        accelerator_end_to_end=Q_(6, "ms"),  # scenario assumption: accelerator, full request path
        host_power=Q_(120, "mW"),  # scenario assumption: host CPU active power
        accelerator_power=Q_(500, "mW"),  # scenario assumption: accelerator active power
        frame_rate=Q_(30, "1/second"),  # scenario assumption: 30-frame-per-second camera stream
    )
    PreprocessBoundVisionPipeline = _st.PipelineSpeedupScenario(
        name="A vision pipeline whose inference stage is made 5x faster",
        description="8 ms of JPEG decode, resize, and normalize on the host, then 10 ms of inference.",
        stages={"preprocess": Q_(8, "ms"),  # scenario assumption
                "inference": Q_(10, "ms")},  # scenario assumption
        accelerated="inference",
        speedup=5,  # scenario assumption: inference-only optimization
    )
    ResNet50H100TrainingCeiling = _st.AcceleratorInputFeedScenario(
        name="ResNet-50 training on one H100: arithmetic ceiling against the MLPerf rate",
        description="Peak dense FP16 over three forward passes of work per image, against the per-accelerator "
                    "rate of NVIDIA's 64-H100 MLPerf Training v3.1 submission.",
        workload=_Models.Vision.ResNet50,
        hardware=_Hardware.Cloud.H100,
        sustained_rate=Q_(float(_Literature.Benchmarks.ResNet50H100TrainThroughput), "count/second"),
        accelerators=int(_Literature.Benchmarks.ResNet50H100TrainThroughputAccelerators),
        training_flops_factor=3,  # convention: the backward pass costs about twice the forward pass
    )
    KeywordSpottingOnEsp32S3 = _st.PeriodicInferencePowerScenario(
        name="DS-CNN keyword spotting once a second on an ESP32-S3-class board",
        description="One inference per one-second audio window at the board's peak INT8 rate; active at TDP, "
                    "idle at its minimum power between inferences.",
        workload=_Models.Tiny.DS_CNN,
        hardware=_Hardware.Tiny.ESP32_S3,
        period=Q_(1, "second"),  # scenario assumption: one inference per one-second window
        precision="int8",
        energy_cut=2.0,  # scenario assumption: a model that halves inference energy
    )
    ModelOnlyMeanVersusProductionP99 = _st.MeanVersusTailScenario(
        name="A model-only mean latency against a production p99",
        description="Mean latency on evenly spaced synthetic inputs against the p99 under production traffic "
                    "(illustrative).",
        mean=Q_(15, "ms"),  # scenario assumption: model-only mean
        tail_low=Q_(150, "ms"),  # scenario assumption: production p99, low
        tail_high=Q_(200, "ms"),  # scenario assumption: production p99, high
    )
    ImageClassifierEightGpuScaling = _st.StrongScalingPairScenario(
        name="An image classifier trained on one GPU and on eight",
        description="A fixed dataset: 24 hours on one GPU, 4 hours on eight (illustrative).",
        base_devices=1,
        base_time=Q_(24, "hour"),  # scenario assumption: single-GPU training time
        scaled_devices=8,
        scaled_time=Q_(4, "hour"),  # scenario assumption: eight-GPU training time
    )
    # --- Benchmarking: deciding from measurements (added 2026-10-20) --------
    AccuracyWithinOnePointSampleSize = _st.ProportionSampleSizeScenario(
        name="Test images that pin a 95 percent accuracy to within one point",
        description="n = z^2 p (1 - p) / m^2 at 95 percent confidence.",
        rate=0.95,         # scenario assumption: accuracy being estimated
        half_width=0.01,   # scenario assumption: one point
        confidence=0.95,   # scenario assumption
    )
    NinetyFiveAgainstNinetyFourOnSameImages = _st.SameTestSetAccuracyComparisonScenario(
        name="A 95 percent baseline against a 94 percent compressed model on the same 1,000 images",
        description="Independent-set interval on the one-point drop, and McNemar's disagreement limit.",
        n_samples=1000,               # scenario assumption: test images
        baseline_accuracy=0.95,       # scenario assumption
        candidate_accuracy=0.94,      # scenario assumption
        confidence=0.95,              # scenario assumption
    )
    WiderBeamSearchTranslation = _st.ProxyMetricLatencyScenario(
        name="Beam search widened from 1 to 10 candidates for half a BLEU point",
        description="An offline metric gain that breaks a 100 ms serving deadline (illustrative).",
        original_metric=28.0,            # scenario assumption: BLEU
        optimized_metric=28.5,           # scenario assumption: BLEU
        original_width=1,                # scenario assumption: beam size
        optimized_width=10,              # scenario assumption: beam size
        original_latency=Q_(50, "ms"),   # scenario assumption
        optimized_latency=Q_(200, "ms"),  # scenario assumption
        latency_budget=Q_(100, "ms"),    # scenario assumption: serving deadline
    )
    MobileNetV2Int8CameraTrapGates = _st.QuantizedAcceptanceGatesScenario(
        name="MobileNetV2 FP32 against INT8 for a camera trap, against gates set before compression",
        description="Assumed top-1, top-5, ECE, and edge-case accuracy; edge cases are about 5 percent of inputs.",
        workload=_Models.Vision.MobileNetV2,
        float_top1=71.8,             # scenario assumption: FP32 top-1, percent
        int_top1=70.9,               # scenario assumption: INT8 top-1, percent
        float_top5=91.0,             # scenario assumption
        int_top5=90.4,               # scenario assumption
        float_ece=0.031,             # scenario assumption
        int_ece=0.089,               # scenario assumption
        float_edge=68.2,             # scenario assumption: edge-case accuracy, percent
        int_edge=61.4,               # scenario assumption
        accuracy_loss_limit=1.0,     # scenario assumption: gate, percentage points
        edge_share=0.05,             # scenario assumption: edge cases among real inputs
        edge_occlusion=0.50,         # scenario assumption: edge-case definition
        edge_illuminance=Q_(100, "lux"),  # scenario assumption: edge-case definition
        edge_rotation_deg=30,        # scenario assumption: edge-case definition
        confidence_threshold=0.85,   # scenario assumption: the trap acts above this confidence
        test_images=1000,            # scenario assumption: lab test set
        confidence=0.95,             # scenario assumption
    )

    # --- Model serving on one machine (added 2026-10-20) ----------------------
    # The reference service is ResNet-50 in eager FP16 on one H100, from the
    # shared profile (engine.serving_profile): a fixed cost per pass plus device
    # work per image. Host phases, traffic, budgets, and prices are stated.
    ResNet50EagerFp16BatchingH100 = _st.BatchingKneeScenario(
        name="ResNet-50 eager FP16 batches on one H100",
        description="Fixed cost per pass (kernel issue, per-kernel floor, one weight read) plus device "
                    "work per image, from the shared ResNet-50 H100 profile.",
        fixed_time=_resnet50_h100_profile().eager_fp16(0),
        per_request_time=_resnet50_h100_profile().per_image_fp16,
        batch_sizes=[1, 8, 32, 128],  # scenario assumption: batch sizes compared
    )
    ResNet50BatchOneVersusEightCostH100 = _st.BatchServingCostScenario(
        name="ResNet-50 on one rented H100 at batch 1 and batch 8",
        description="Cost per million requests when the accelerator serves one image per pass or eight.",
        service=ResNet50EagerFp16BatchingH100,
        price=_Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
        batch_sizes=[1, 8],  # scenario assumption: the two batch sizes compared
    )
    PhotoLibraryPrecomputeH100 = _st.PrecomputeVersusOnDemandScenario(
        name="A 10,000-photo library classified on demand or overnight on one H100",
        description="Batch-one eager FP16 per photo against the MLPerf INT8 Offline rate per H100; the "
                    "on-demand alternative is a moderation service with a 100 ms deadline.",
        items=10_000,  # scenario assumption: photos in one user's library
        on_demand_time=_resnet50_h100_profile().eager_fp16(1),
        offline_rate=Q_(_resnet50_h100_profile().int8_offline_img_s, "1/second"),
        latency_budget=Q_(100, "ms"),  # scenario assumption: moderation deadline
    )
    ResNet50RequestPathH100 = _st.PipelineSpeedupScenario(
        name="One ResNet-50 image request on an idle H100 server",
        description="Host decode, resize, and normalize; copy over PCIe; the eager FP16 forward pass; "
                    "top-5 on the host. The comparison makes the model 10x faster.",
        stages={
            "JPEG decode": Q_(3.0, "ms"),  # scenario assumption
            "resize": Q_(1.0, "ms"),  # scenario assumption: to 224 x 224
            "normalize": Q_(0.5, "ms"),  # scenario assumption
            "copy to GPU": Q_(0.5, "ms"),  # scenario assumption
            "forward pass": _resnet50_h100_profile().eager_fp16(1),
            "top-5": Q_(0.1, "ms"),  # scenario assumption: softmax and top five
        },
        accelerated="forward pass",
        speedup=10,  # scenario assumption: a 10x faster model
    )
    ResNet50RequestPathTensorRtH100 = ResNet50RequestPathH100.variant(
        name="The same request with a TensorRT FP16 engine",
        description="Only the forward pass changes, to the profile's TensorRT FP16 batch-one time.",
        stages={**ResNet50RequestPathH100.stages, "forward pass": _resnet50_h100_profile().tensorrt("fp16", 1)},
    )
    DlrmRequestPath = _st.PipelineSpeedupScenario(
        name="One DLRM recommendation request",
        description="Parse the request, fetch 100+ embedding vectors, run the MLP, rank and filter.",
        stages={
            "parse": Q_(0.5, "ms"),  # scenario assumption
            "embedding lookup": Q_(6.0, "ms"),  # scenario assumption
            "MLP forward": Q_(1.5, "ms"),  # scenario assumption
            "rank and filter": Q_(1.0, "ms"),  # scenario assumption
        },
        accelerated="MLP forward",
    )
    ImageServiceLittlesLaw = _st.LittlesLawScenario(
        name="An image service at 1,000 requests per second, 50 ms mean time inside",
        description="The accelerator runs batches of up to 32; requests beyond those wait in host memory.",
        arrival_rate=Q_(1000, "1/second"),  # scenario assumption
        time_in_system=Q_(50, "ms"),  # scenario assumption: mean time in system
        batch_size=32,  # scenario assumption: largest batch the accelerator runs
    )
    ResNet50FleetPlanH100 = _st.MM1CapacityPlanScenario(
        name="H100s for 5,000 ResNet-50 requests per second under a 50 ms p99",
        description="Batch-one eager FP16 service; each H100 an M/M/1 server whose p99 is 4.6 times "
                    "its mean time in system; 30 percent headroom; re-planned at the batch-32 rate.",
        service_time=_resnet50_h100_profile().eager_fp16(1),
        tail_factor=4.6,  # scenario assumption: ln(100) = 4.605 rounded, as the p99 rule of thumb prints it
        tail_budget=Q_(50, "ms"),  # scenario assumption: queue-plus-GPU p99 budget
        peak_rate=Q_(5000, "1/second"),  # scenario assumption: peak traffic
        headroom=1.3,  # scenario assumption: 30 percent for spikes
        utilization_points=[0.5, 0.7, 0.8, 0.9, 0.95],  # scenario assumption: loads compared
        example_utilization=0.7,  # scenario assumption: the load where mean and p99 are compared
        batched_rate=ResNet50EagerFp16BatchingH100.throughput(32),
        accelerator_price=_Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
    )
    ResNet50FasterModelAt80PctH100 = _st.MM1ServiceSpeedupScenario(
        name="The batch-one ResNet-50 service at 80 percent load, before and after graph execution",
        description="Eager FP16 against TorchScript (no per-operation Python dispatch) on the same traffic.",
        slow_service=_resnet50_h100_profile().eager_fp16(1),
        fast_service=_resnet50_h100_profile().torchscript_fp16(1),
        utilization=0.8,  # scenario assumption
    )
    BatchWindow10ms = _st.BatchWindowScenario(
        name="A 10 ms batching window at 50 to 1,000 requests per second",
        description="Poisson arrivals; the count per window, the share of empty windows, and the batch "
                    "a request finds itself in.",
        window=Q_(10, "ms"),  # scenario assumption
        arrival_rates=[Q_(r, "1/second") for r in (50, 200, 500, 1000)],  # scenario assumption
    )
    BatchWindow20msIn50msBudget = BatchWindow10ms.variant(
        name="A 20 ms batching window inside a 50 ms latency budget",
        description="The mean wait for the window to close, as a share of the budget.",
        window=Q_(20, "ms"),  # scenario assumption
        arrival_rates=[],
        latency_budget=Q_(50, "ms"),  # scenario assumption
    )
    DualTriggerBatchFourRequests = _st.BatchDispatchScenario(
        name="Four requests under a 12 ms batch timeout behind a busy worker",
        description="Arrivals at 0, 2, 5, and 9 ms never reach the cap; the worker is busy until 18 ms; "
                    "4 ms of response path follow; the deadline is 50 ms.",
        arrivals=[Q_(t, "ms") for t in (0, 2, 5, 9)],  # scenario assumption: hypothetical trace
        timeout=Q_(12, "ms"),  # scenario assumption: tuned batching window
        worker_free=Q_(18, "ms"),  # scenario assumption
        response_path=Q_(4, "ms"),  # scenario assumption
        deadline=Q_(50, "ms"),  # scenario assumption
        service=ResNet50EagerFp16BatchingH100,
    )
    BatchCap16Or32At500Qps = _st.BatchCapChoiceScenario(
        name="Batch cap 16 or 32 at 500 requests per second",
        description="Batches wait until full; saturated throughput from the eager FP16 profile.",
        service=ResNet50EagerFp16BatchingH100,
        arrival_rate=Q_(500, "1/second"),  # scenario assumption
        small_cap=16,  # scenario assumption
        large_cap=32,  # scenario assumption
    )
    ResNet50ReplicaStartup = _st.ReplicaStartupScenario(
        name="Startup of a ResNet-50 serving replica",
        description="Representative phases: weights from a local SSD or object storage, CUDA context, "
                    "TensorRT engine build on a first deploy, warmup, runtime start.",
        phases={
            "weights from local SSD": Q_(0.5, "second"),  # scenario assumption
            "weights from object storage": Q_(4.0, "second"),  # scenario assumption
            "CUDA context": Q_(0.4, "second"),  # scenario assumption
            "TensorRT engine build": Q_(30.0, "second"),  # scenario assumption
            "warmup": Q_(0.2, "second"),  # scenario assumption
            "runtime": Q_(0.4, "second"),  # scenario assumption
        },
        ranges={
            "weights from object storage": [Q_(3, "second"), Q_(5, "second")],  # scenario assumption
            "CUDA context": [Q_(0.3, "second"), Q_(0.5, "second")],  # scenario assumption
            "TensorRT engine build": [Q_(15, "second"), Q_(30, "second")],  # scenario assumption
        },
        warm_path=["weights from local SSD", "CUDA context", "warmup", "runtime"],
        cold_path=["weights from object storage", "CUDA context", "TensorRT engine build", "warmup", "runtime"],
    )
    ModelSwapPcieGen4 = _st.TransferScenario(
        name="A 10 GB model swapped into an accelerator over PCIe Gen4 x16",
        description="Host memory to device at the link's theoretical bandwidth; bandwidth term only.",
        payload=_ReferenceStats.ModelLoading.PcieSwapReferenceModelSize,
        links={"PCIe Gen4 x16": _Hardware.Cloud.A100.interconnect.bandwidth},
    )
    RecommenderTrafficSurge = _st.TrafficSurgeScenario(
        name="A recommender sized for 1,000 requests per second receives ten times more",
        description="Utilization pins at 100 percent and the queue grows for as long as the surge lasts.",
        normal_rate=Q_(1000, "1/second"),  # scenario assumption
        surge_rate=Q_(10000, "1/second"),  # scenario assumption
        normal_latency=Q_(50, "ms"),  # scenario assumption
        surge_latency=Q_(10, "second"),  # scenario assumption: latency once the queue has grown
    )
    ResNet50BatchEnergyPerImage = _st.BatchEnergyPerImageScenario(
        name="ResNet-50 bytes and energy per image at batch 1, 8, and 32",
        description="FP32 weights read once per batch; 12.5 MB of activations written and read per image; "
                    "DRAM energy per byte and the mean of an FP32 multiply and add per FLOP (45 nm).",
        workload=_Models.Vision.ResNet50,
        weight_bytes_per_param=_BYTES_FP32,
        activation_bytes=Q_(12.5, "MB"),  # scenario assumption: 224 x 224 activation footprint
        activation_passes=2,  # scenario assumption: written once, read once
        byte_energy=_Hardware.Tech.Memory.DRAM.energy_per_byte,
        flop_energy=(_Hardware.Tech.Op.FlopFp32.energy + _Hardware.Tech.Op.AddFp32.energy) / 2,
    )
    ResNet50CloudInstanceCost = _st.ServingInstanceCostScenario(
        name="ResNet-50 on a CPU instance, a T4 instance, and an H100",
        description="Hourly prices and saturated image rates; the H100 runs batch 32 of the eager FP16 profile.",
        prices={
            "CPU": _Infrastructure.Pricing.Cloud.CpuInstanceC5XlargePerHour.rate,
            "T4": _Infrastructure.Pricing.Cloud.T4InstanceG4dnXlargePerHour.rate,
            "H100": _Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
        },
        rates={
            "CPU": Q_(50, "1/second"),  # scenario assumption: c5.xlarge ResNet-50 rate
            "T4": Q_(400, "1/second"),  # scenario assumption: T4 ResNet-50 rate
            "H100": ResNet50EagerFp16BatchingH100.throughput(32),
        },
        small="T4",
        large="H100",
    )

    # --- Reinforcement learning from verifiable rewards (added 2026-10-20) ---
    VerifierFalseAcceptsAtTenPercentPass = _st.RewardCompositionScenario(
        name="One update of 8 rollouts on each of 1,024 tasks, scored by a verifier that accepts 2.5% of failures",
        description="Ten percent of rollouts truly pass; the verifier accepts every pass and wrongly accepts "
                    "2.5 percent of failures. Expected counts, no false rejections.",
        tasks=1024,                  # scenario assumption
        group=8,                     # scenario assumption
        pass_rate=0.10,              # scenario assumption
        false_accept_rate=0.025,     # scenario assumption
        valid_accept_rate=1.0,       # scenario assumption: no false rejections
    )
    VerifierFalseAcceptsAtFivePercentPass = VerifierFalseAcceptsAtTenPercentPass.variant(
        name="The same verifier on a policy that passes 5 percent",
        description="Only the pass rate changes.",
        pass_rate=0.05,              # scenario assumption: a harder task mix
    )
    BinaryRewardFixedBaseline = _st.CenteredRewardScenario(
        name="Binary terminal reward centered on a fixed baseline of 0.25",
        description="Reward 1 for a pass and 0 for a failure.",
        baseline=0.25,               # scenario assumption
        passing_reward=1.0,
        failing_reward=0.0,
    )
    DebuggerTurnForkValue = _st.ForkValueScenario(
        name="Twenty continuations forked before and after a repair's debugger turn",
        description="One of 20 continuations passes from the bug report; 12 of 20 pass after the debugger turn.",
        continuations=20,            # scenario assumption
        passes_before=1,             # scenario assumption
        passes_after=12,             # scenario assumption
    )
    ThirtyTurnForksAtEveryBoundary = _st.ContinuationForkCostScenario(
        name="A 30-turn trajectory of 512 tokens per turn with 8 continuations forked at every boundary",
        description="Every one of the 29 intermediate boundaries reruns the remaining turns 8 times.",
        turns=30,                    # scenario assumption
        tokens_per_turn=512,         # scenario assumption
        branches=8,                  # scenario assumption
    )
    ThirtyTurnForksAfterTurnsTenAndTwenty = ThirtyTurnForksAtEveryBoundary.variant(
        name="The same trajectory with forks only after turns 10 and 20",
        description="Two chosen boundaries instead of all 29.",
        boundaries=[10, 20],         # scenario assumption
    )
    RepairGroupOnePassOfThree = _st.GroupCoefficientsScenario(
        name="Three repair rollouts, the middle one passing, under population dispersion",
        description="Rewards [0, 1, 0]; the dispersion divides by G and a 1e-8 stabilizer is added.",
        rewards=[0.0, 1.0, 0.0],     # scenario assumption
        stabilizer=1e-8,             # scenario assumption
    )
    LonePassInGroupOfSixteen = _st.GroupCoefficientsScenario(
        name="One pass among 16 rollouts of a hard task",
        description="Population dispersion with no stabilizer; the lone pass gets sqrt(15).",
        rewards=[1.0] + [0.0] * 15,  # scenario assumption
        stabilizer=0.0,
    )
    LoneFailureInGroupOfSixteen = LonePassInGroupOfSixteen.variant(
        name="One failure among 16 rollouts of an easy task",
        description="The mirror image of the lone pass.",
        rewards=[0.0] + [1.0] * 15,  # scenario assumption
    )
    ShortAndLongFailureLengths = _st.LengthNormalizationScenario(
        name="A 128-token and a 4,096-token failure with the same group coefficient",
        description="Per-rollout averaging divides each by its own length.",
        short_tokens=128,            # scenario assumption
        long_tokens=4_096,           # scenario assumption
    )
    HardTaskRolloutStep = _st.RolloutStepBudgetScenario(
        name="One update of 512 tasks with groups of 8 on a mix the policy passes 5 percent",
        description="Each rollout is the 30-turn, 512-token trajectory of the fork inventory.",
        tasks=512,                   # scenario assumption
        group=8,                     # scenario assumption
        turns=ThirtyTurnForksAtEveryBoundary.turns,
        tokens_per_turn=ThirtyTurnForksAtEveryBoundary.tokens_per_turn,
        pass_rate=0.05,              # scenario assumption: untargeted hard-task mix
    )
    HardTaskRolloutStepDoubledGroup = HardTaskRolloutStep.variant(
        name="The same update with groups of 16",
        description="Only the group size changes.",
        group=16,                    # scenario assumption: double the group
    )
    TargetedTaskRolloutStep = HardTaskRolloutStep.variant(
        name="The same update on tasks the policy passes about half the time",
        description="Difficulty-targeted selection; only the pass rate changes.",
        pass_rate=0.50,              # scenario assumption
    )
    ModerateTaskRolloutStep = HardTaskRolloutStep.variant(
        name="The same update on a mix the policy passes 20 percent",
        description="Only the pass rate changes.",
        pass_rate=0.20,              # scenario assumption
    )
    PairGroupsAtTenPercentPass = HardTaskRolloutStep.variant(
        name="Groups of 2 rollouts on a task the policy passes 10 percent",
        description="The smallest group that can carry contrast.",
        group=2,                     # scenario assumption
        pass_rate=0.10,              # scenario assumption
    )
    FlakyRaceTestAllPassRule = _st.RepeatedVerificationScenario(
        name="A timing-sensitive race test repeated three times on a candidate patch",
        description="An incorrect patch passes one trial with probability 0.20 and a correct one with 0.98; "
                    "a group of 16 incorrect patches graded once each.",
        incorrect_pass=0.20,         # scenario assumption
        correct_pass=0.98,           # scenario assumption
        trials=3,                    # scenario assumption
        group=16,                    # scenario assumption
    )
    DominantTokenRatioClip = _st.RatioClipScenario(
        name="A token at probability 0.9 under a ratio clip of 0.2",
        description="The ceiling (1 + epsilon) p exceeds 1, so the clip does not bind.",
        epsilon=0.2,                 # scenario assumption: PPO's common clip width
        probability=0.9,             # scenario assumption
    )
    RareTokenRatioClip = DominantTokenRatioClip.variant(
        name="A token at probability 0.01 under a ratio clip of 0.2",
        description="The token may rise only epsilon p in one update.",
        probability=0.01,            # scenario assumption
    )
    RepairRolloutSlotResets = _st.RolloutSlotCycleScenario(
        name="256 sandbox slots running 20-decision repair episodes, cold container reset against snapshot",
        description="120 ms of model service and 80 ms of tool execution per decision, 0.5 s of assessment, "
                    "then a 3.5 s cold reset or a 15 ms snapshot restore. Constant times, no contention.",
        slots=256,                   # scenario assumption
        decisions=20,                # scenario assumption
        model_time=Q_(120, "ms"),    # scenario assumption
        tool_time=Q_(80, "ms"),      # scenario assumption
        assessment_time=Q_(0.5, "s"),   # scenario assumption
        cold_reset=Q_(3.5, "s"),     # scenario assumption
        snapshot_reset=Q_(15, "ms"),    # scenario assumption
    )
    RolloutGroupOfFourBarrier = _st.GroupBarrierScenario(
        name="A group of four rollouts finishing between 0.2 s and 16.8 s",
        description="The group-relative update waits for the slowest member.",
        latencies=[Q_(0.2, "s"), Q_(2.1, "s"), Q_(4.2, "s"), Q_(16.8, "s")],   # scenario assumption
    )
    RepairVerifierPool = _st.VerifierPoolScenario(
        name="512 completed trajectories every 16 s, three 8 s verification trials each, 80 percent target",
        description="Stationary arrivals; every trial executes in this sizing case.",
        trajectories=512,            # scenario assumption
        interval=Q_(16, "s"),        # scenario assumption
        trials=3,                    # scenario assumption
        trial_time=Q_(8, "s"),       # scenario assumption
        target_utilization=0.8,      # scenario assumption: illustrative planning target
    )

    # --- Release of the warehouse machine: lighting, argument confidence (added 2026-10-20) ---
    WarehouseNavCameraAgeAtAisleSpeed = WarehouseNavCameraAge.variant(
        name="The warehouse machine's navigation-camera observation age at the candidate aisle speed",
        description="Exposure midpoint to dispatch, priced as travel at the aisle speed; one frame.",
        speed=_Embodied.Scenario.WarehouseAisle.v_aisle,
        batch_frames=1,
    )
    WarehouseNavCameraAgeDimAisle = WarehouseNavCameraAgeAtAisleSpeed.variant(
        name="The same observation age in a dim aisle that doubles the camera's exposure",
        description="Auto-exposure lengthens the exposure; the capture epoch is its midpoint, so the age "
                    "grows by half the added exposure.",
        exposure=(2.0  # scenario assumption: a dim aisle doubles the exposure (illustrative)
                  * _Embodied.MobileManipulator.WarehouseMobileManipulator.nav_camera.exposure_time).to("ms"),
    )
    SafetyArgumentConfidence = _st.ArgumentConfidenceScenario(
        name="A stipulated safety argument: ten required premises and two independent legs",
        description="Confidence in a conjunction of required premises, and in a claim either of two "
                    "independent legs suffices for (a teaching model, not a machine value).",
        premise_confidence=0.99,  # scenario assumption: stipulated confidence in each required premise
        premises=10,              # scenario assumption: stipulated number of required premises
        leg_confidence=0.90,      # scenario assumption: stipulated confidence in each leg
        legs=2,
    )

    # --- Coordinating several agents (added 2026-10-20) -------------------------
    RepairPartitionScreen = _st.PartitionScreenScenario(
        name="A repair partition with 80 percent divisible work and 0.02 overhead per added worker",
        description="Elapsed time over the serial reference for one to sixteen workers.",
        parallel_fraction=0.80,  # scenario assumption
        overhead_per_extra_worker=0.02,  # scenario assumption: normalized by the serial time
        max_workers=16,  # scenario assumption: resource ceiling
    )
    RepairPartitionScreenHalvedOverhead = RepairPartitionScreen.variant(
        name="The same partition with the per-worker overhead halved",
        overhead_per_extra_worker=RepairPartitionScreen.overhead_per_extra_worker / 2,
    )
    RepairPartitionScalability = _st.ScalabilityLawScenario(
        name="The repair partition as a universal scalability law: sigma 0.20, kappa 0.02",
        description="sigma = 1 - p and kappa = c reproduce the screen's elapsed times as capacities.",
        contention=1 - RepairPartitionScreen.parallel_fraction,
        coherency=RepairPartitionScreen.overhead_per_extra_worker,
        search_limit=64,  # scenario assumption: counts searched
    )
    ResearchFanoutScalability = RepairPartitionScalability.variant(
        name="The orchestrator-worker research agent: sigma 0.10, kappa 0.002",
        description="Independent search paths reconcile rarely, so coherency is a tenth of the repair partition's.",
        contention=_Agents.Research.OrchestratorWorker.serial_fraction,
        coherency=_Agents.Research.OrchestratorWorker.coherency_coefficient,
    )
    ResearchSystemTokenBreakEven = _st.DelegationTokenBreakEvenScenario(
        name="Delegation at the research system's 15x against 4x token multiples over chat",
        description="Break-even acceptance at a 3.75x token multiple, and the reported 90.2 percent gain "
                    "read as an acceptance multiple; a single agent that already accepts 60 percent.",
        agent_tokens_vs_chat=_Agents.Research.OrchestratorWorker.agent_token_multiple,
        multi_agent_tokens_vs_chat=_Agents.Research.OrchestratorWorker.fanout_token_multiple,
        reported_gain=90.2 / 100,  # Hadfield et al. (2025), How we built our multi-agent research system...
        single_acceptance=0.60,  # scenario assumption: a task one agent usually completes
    )
    InterfaceMigrationSchedule = _st.DependencyScheduleScenario(
        name="A seven-node interface migration on four slots",
        description="Specification, three module changes, an integration step, a test step, and the "
                    "final check; v2 and v3 tie on the critical path.",
        durations={1: Q_(20.5, "s"), 2: Q_(42, "s"), 3: Q_(42, "s"), 4: Q_(42, "s"),
                   5: Q_(43, "s"), 6: Q_(16.5, "s"), 7: Q_(69, "s")},  # scenario assumption: node durations
        predecessors={1: [], 2: [1], 3: [1], 4: [1], 5: [2, 3], 6: [4], 7: [5, 6]},
        slots=4,  # scenario assumption
    )
    SixChildReferencePassing = _st.ReferencePassingScenario(
        name="Six children: a pasted 48,000-token slice or a 350-token reference and 3,600 tokens of reads",
        description="Each supplied input token is charged once at a hypothetical $3 per million.",
        workers=6,  # scenario assumption
        slice_tokens=48_000,  # scenario assumption
        reference_tokens=350,  # scenario assumption
        read_tokens=3_600,  # scenario assumption
        input_price=Q_(3.00, "USD / Mtoken"),  # scenario assumption: hypothetical input price
    )
    ThousandFileSharedWrites = _st.SharedWriteScenario(
        name="Workers each writing 10 of 1,000 files in one shared workspace",
        description="Uniform, independent write sets; the exact disjoint product and its birthday approximation.",
        files=1_000,  # scenario assumption
        writes_per_worker=10,  # scenario assumption
        max_fleet=16,  # scenario assumption: fleet sizes evaluated
        small_fleet=2,  # scenario assumption
        large_fleet=8,  # scenario assumption
    )
    FiveCorrelatedReviewers = _st.CorrelatedReviewersScenario(
        name="Five reviewers, each falsely approving 20 percent, with mixture correlation 0.40",
        description="Majority and unanimous false approval, independent against beta-binomial, and the "
                    "floor under the majority as the panel grows to fifteen.",
        reviewers=5,  # scenario assumption
        false_approval=0.20,  # scenario assumption: marginal false-approval rate
        correlation=0.40,  # scenario assumption: pairwise correlation of the propensity
        large_ensemble=15,  # scenario assumption
    )
    EightWorkerDeadlineFanout = _st.FanoutDeadlineScenario(
        name="Eight workers, each on time 90 percent of the time, against a 10 s deadline",
        description="All eight results against any six of eight interchangeable ones.",
        workers=8,  # scenario assumption
        required=6,  # scenario assumption: the contract's quorum
        on_time_probability=0.90,  # scenario assumption
        deadline=Q_(10, "s"),  # scenario assumption
        max_fleet=16,  # scenario assumption: fleet sizes tabulated
    )
    DiffReviewCreditWindow = _st.CreditWindowScenario(
        name="A diff producer at 5 items/s feeding a reviewer of 500-token critiques at 40 tokens/s",
        description="Ten minutes unbounded against a window of eight credits; the window for a one-minute residence.",
        producer_rate=Q_(5, "1/s"),  # scenario assumption
        decode_rate=Q_(40, "token/s"),  # scenario assumption
        tokens_per_item=Q_(500, "token"),  # scenario assumption
        horizon=Q_(600, "s"),  # scenario assumption
        credits=8,  # scenario assumption
        residence_target=Q_(60, "s"),  # scenario assumption
    )
    MatchedDelegationCohort = _st.DelegationCohortScenario(
        name="100 migrations, single controller against delegation, under a $20 ceiling",
        description="Both accept 80 by the 600 s cutoff; only delegation meets 75 by the 300 s deadline.",
        attempted=100,  # scenario assumption
        accepted_floor=80,  # scenario assumption
        on_time_floor=75,  # scenario assumption
        deadline=Q_(300, "s"),  # scenario assumption
        cutoff=Q_(600, "s"),  # scenario assumption
        budget=Q_(20, "USD"),  # scenario assumption
        input_price=SixChildReferencePassing.input_price,
        output_price=Q_(10.00, "USD / Mtoken"),  # scenario assumption: hypothetical output price
        service_price=Q_(0.01, "USD / s"),  # scenario assumption: metered tool and assessment time
        single=_st.CohortArmRecord(
            accepted=80, on_time=70, median_accepted_time=Q_(240, "s"),  # scenario assumption
            input_tokens=Q_(2_000_000, "token"), output_tokens=Q_(400_000, "token"),  # scenario assumption
            service_time=Q_(200, "s"),  # scenario assumption
            discard_input_tokens=Q_(500_000, "token"), discard_output_tokens=Q_(100_000, "token"),  # scenario assumption
            discard_service_time=Q_(50, "s"),  # scenario assumption
        ),
        delegated=_st.CohortArmRecord(
            accepted=80, on_time=78, median_accepted_time=Q_(180, "s"),  # scenario assumption
            input_tokens=Q_(3_000_000, "token"), output_tokens=Q_(600_000, "token"),  # scenario assumption
            service_time=Q_(300, "s"),  # scenario assumption
            discard_input_tokens=Q_(1_000_000, "token"), discard_output_tokens=Q_(200_000, "token"),  # scenario assumption
            discard_service_time=Q_(100, "s"),  # scenario assumption
        ),
    )

    # --- Responsible physical AI: decisions, oversight, stages, groups, records (added 2026-10-20) ---
    AdmittedCommandDecisionChain = _st.DecisionChainScenario(
        name="Six separately kept decisions behind one admitted aisle command",
        description="Chain completeness p^6 at three per-link retrieval probabilities.",
        links=6,                                  # scenario assumption: policy, enforcer, authority, envelope, crossing rule, staffing
        link_probabilities=(0.99, 0.95, 0.90),    # scenario assumption: illustrative retrieval probabilities
    )
    WarehouseRackEndOversightTime = _st.OversightTimeScenario(
        name="The out-of-loop takeover against the time a person stepping out at the rack end leaves",
        description="Clear distance at the aisle speed, the design out-of-loop takeover, and its travel.",
        budget=WarehouseOutOfLoopTakeoverAtAisleSpeed,
    )
    WarehouseRemoteAssistantQueue = _st.SupervisorQueueScenario(
        name="One remote assistant serving twenty warehouse machines",
        description="M/M/1 utilization, mean wait, and 99th-percentile wait; the fleet a one-half cap allows.",
        request_rate=Q_(0.5, "1/hour"),           # scenario assumption: illustrative help requests per machine-hour
        service_time=Q_(3.0, "minute"),           # scenario assumption: illustrative mean time per request
        machines=20,                              # scenario assumption: illustrative fleet
        utilization_cap=0.5,                      # scenario assumption: chosen staffing ceiling
    )
    WarehouseRemoteAssistantQueueTenMachines = WarehouseRemoteAssistantQueue.variant(
        name="One remote assistant serving ten warehouse machines",
        description="The same requests from half the fleet.",
        machines=10,                              # scenario assumption: illustrative fleet
    )
    WarehouseRemoteAssistantQueueThirtyTwoMachines = WarehouseRemoteAssistantQueue.variant(
        name="One remote assistant serving thirty-two warehouse machines",
        description="The same requests from a larger fleet.",
        machines=32,                              # scenario assumption: illustrative fleet
    )
    WarehouseRemoteAssistantLabor = _st.SupervisionLaborScenario(
        name="Person-hours a twenty-machine warehouse fleet moves into rescue, watching, and sweeps",
        description="An assistant staffed through two shifts at the queue's utilization, plus corridor sweeps.",
        queue=WarehouseRemoteAssistantQueue,
        operating_day=Q_(16.0, "hour"),           # scenario assumption: two shifts of operation
        sweeps=4,                                 # scenario assumption: inspections of the restricted corridors
        sweep_time=Q_(20.0, "minute"),            # scenario assumption: one corridor sweep
        shift=Q_(8.0, "hour"),                    # scenario assumption: one full-time shift
    )
    WarehouseSupervisedStageExit = _st.HazardExposureScenario(
        name="Failure-free hours ten supervised warehouse machines need to leave their stage",
        description="Zero-failure exposure bounding one safety-relevant intervention per 1,000 h at 95 percent.",
        rates=[Q_(1.0e-3, "1/hour")],             # scenario assumption: chosen target rate
        confidence=0.95,                          # scenario assumption: chosen, one-sided
        rigs=10,                                  # scenario assumption: machines in the supervised stage
        hours_per_day=Q_(16.0, "hour/day"),       # scenario assumption: two shifts
    )
    WarehouseGroupDetectionShortfall = _st.DetectionShortfallScenario(
        name="A stop sized for the best-detected group, met by three groups seen less often",
        description="Frames to a reliable detection, distance lost, impact speed and energy at the aisle speed.",
        site=_Embodied.Scenario.WarehouseAisle,
        speed=_Embodied.Scenario.WarehouseAisle.v_aisle,
        all_miss_probability=1.0e-3,              # scenario assumption: chosen all-frames miss probability
        reference=0.95,                           # scenario assumption: illustrative best-detected group
        groups=(0.85, 0.75, 0.60),                # scenario assumption: illustrative lower-detection groups
    )
    PedestrianGroupEvaluationSample = _st.SubgroupSampleScenario(
        name="A 10,000-person detection test set with a 2 percent group",
        description="Exact 95 percent intervals per group and the per-group sample that tells 0.95 from 0.92.",
        total=10_000,                             # scenario assumption: illustrative test set
        small_share=0.02,                         # scenario assumption: illustrative group share
        recall=0.95,                              # scenario assumption: illustrative detection rate in both groups
        rate_a=0.95,                              # scenario assumption: chosen gap to resolve
        rate_b=0.92,                              # scenario assumption: chosen gap to resolve
        significance=0.05,                        # scenario assumption: chosen two-sided level
        power=0.80,                               # scenario assumption: chosen
    )
    WarehouseCrossingExposure = _st.ExposureConcentrationScenario(
        name="Machine-person encounters concentrated at three warehouse crossings",
        description="Per-person exposure of the crossing users and of everyone else against the site average.",
        encounter_share=0.80,                     # scenario assumption: illustrative share of encounters
        user_share=0.10,                          # scenario assumption: illustrative share of workers
    )
    WarehouseEventTriggeredRetention = _st.EventRetentionScenario(
        name="Event-triggered camera retention on the warehouse machine",
        description="The pre-trigger window plus a post-event capture per event, against keeping the whole day.",
        pre_trigger=_Embodied.Scenario.WarehouseAisle.pre_trigger_window,
        post_trigger=Q_(25.0, "s"),               # scenario assumption: chosen capture after the event
        events_per_day=20,                        # scenario assumption: illustrative events per machine-day
        operating_day=Q_(16.0, "hour"),           # scenario assumption: two shifts
    )
    CruiseDisclosurePenalties = _st.DisclosurePenaltyScenario(
        name="Penalties Cruise paid for its account of the October 2, 2023 collision",
        description="CPUC days at the daily maximum, the NHTSA consent-order penalty, and the DOJ fine.",
        days_withheld=15,  # California Public Utilities Commission, Decision 24-07-004, Presidi...
        daily_maximum=7500,  # California Public Utilities Commission, Decision 24-07-004, Presidi...
        others={
            "NHTSA": 1500000,  # NHTSA, 'NHTSA Announces Consent Order with Cruise After Company Fai...
            "DOJ": 500000,  # U.S. Attorney's Office, Northern District of California, 'Cruise Ad...
        },
    )

    # --- ML operations, part 1 (added 2026-10-20) ------------------------------
    # The ranking service (DLRM tables on H100s) and the keyword spotter
    # (DS-CNN on an ESP32-S3) as release, rebuild, and parity scenarios.
    RankingServiceAccuracyDrift = _st.AccuracyDecayScenario(
        name="The ranking service's click-prediction accuracy, launch against four weeks later",
        description="94 percent at launch and 88 percent four weeks later after a competitor's promotion "
                    "shifted clicks; a product manager noticed in week eight. Illustrative.",
        initial_accuracy=0.94,  # scenario assumption: accuracy at launch
        later_accuracy=0.88,  # scenario assumption: accuracy four weeks later
        elapsed=Q_(28, "day"),  # scenario assumption: four weeks
        noticed=Q_(8, "week"),  # scenario assumption: when a product manager noticed
    )
    DlrmReleaseFootprintH100 = _st.ReleaseFootprintScenario(
        name="Release controls for the DLRM ranking service on an eight-H100 node",
        description="Each control holds a second copy of the 100 GB tables; a retraining run takes four "
                    "H100s for six hours; a 5 percent canary ramps through 25 and 50 percent to all traffic.",
        model=_Models.Recommendation.DLRM,
        device=_Hardware.Cloud.H100,
        node_accelerators=8,  # scenario assumption: one eight-accelerator node
        retrain_accelerators=4,  # scenario assumption: accelerators a retraining run occupies
        retrain_time=Q_(6, "hour"),  # scenario assumption: duration of a retraining run
        canary_share=0.05,  # scenario assumption: the canary's share of live queries
        ramp=[0.05, 0.25, 0.50, 1.0],  # scenario assumption: staged exposure after the canary
    )
    ResNet50Fp16ReleaseFootprintH100 = DlrmReleaseFootprintH100.variant(
        name="Release controls for an FP16 ResNet-50 service on an H100",
        description="The same controls; a copy is the FP16 weights.",
        model=_Models.Vision.ResNet50,
        bytes_per_param=_BYTES_FP16,
    )
    KeywordSpotterDualBankEsp32S3 = _st.DualBankUpdateScenario(
        name="Dual-bank over-the-air update of the INT8 keyword spotter on an ESP32-S3",
        description="The device runs from bank A while bank B receives the new model; two INT8 DS-CNN "
                    "copies in the microcontroller's flash.",
        model=_Models.Tiny.DS_CNN,
        device=_Hardware.Tiny.ESP32_S3,
        bytes_per_param=_BYTES_INT8,
    )
    RetrainingPipelineAutomation = _st.AutomationBreakevenScenario(
        name="An 80-hour retraining pipeline against 4 hours a week of manual retraining",
        description="Simplified break-even with no pipeline upkeep.",
        build_time=Q_(80, "hour"),  # scenario assumption: one-time pipeline build
        manual_rate=Q_(4, "hour/week"),  # scenario assumption: manual retraining effort
    )
    RankingSkewOnePercent = _st.SkewCostScenario(
        name="One percent skew-induced errors on 1 million queries a day at $0.10 each",
        description="The yearly cost of training-serving skew that persists until consistency is restored.",
        skew_rate=0.01,  # scenario assumption: fraction of queries skew makes wrong
        query_rate=Q_(1e6, "1/day"),  # scenario assumption: daily query volume
        cost_per_error=Q_(0.10, "USD"),  # scenario assumption: business cost per wrong prediction
    )
    ConversionRelativeLiftExperiment = _st.TwoProportionTestScenario(
        name="Two-arm test for a 2 percent relative lift on a 5 percent conversion rate",
        description="Two-sided 95 percent confidence and 80 percent power at exact normal quantiles; "
                    "25,000 users per variant are what the traffic allows.",
        baseline_rate=0.05,  # scenario assumption
        lift=0.05 * (1 + 0.02) - 0.05,  # scenario assumption: 2 percent relative, 5.0 to 5.1 percent
        z_alpha=_NormalDist().inv_cdf(1 - (1 - 0.95) / 2),
        z_beta=_NormalDist().inv_cdf(0.80),
        confidence=0.95,  # scenario assumption
        power=0.80,  # scenario assumption
        relative_lift=0.02,  # scenario assumption
        available_per_variant=25_000,  # scenario assumption: users per variant traffic allows
    )

    # --- Serving a stream of tokens on one machine (added 2026-10-20) ---------
    # GPT-2 XL is the bandwidth-bound chat decoder, Llama 3 8B the case-study
    # service, and a 70B-class model (Llama 3 70B geometry) the larger model.
    Gpt2XlChatDecodeBatchesH100 = _st.KvDecodeBatchScenario(
        name="GPT-2 XL chat decode batches on one H100 with full 1,024-token contexts",
        description="FP16 weights read once per step; every sequence reads its own FP16 cache (no grouped-query "
                    "sharing); a 512-token prompt prefilled at 40 percent of the dense FP16 peak.",
        workload=_Models.Language.GPT2,
        hardware=_Hardware.Cloud.H100,
        weight_bytes_per_param=_BYTES_FP16,
        kv_bytes_per_value=_BYTES_FP16,
        context_tokens=_Models.Language.GPT2.context_length,
        batch_sizes=[1, 8, 32, 64, 128],  # scenario assumption: batch sizes compared
        prompt_tokens=512,  # scenario assumption: chat prompt length
        prefill_efficiency=0.4,  # scenario assumption: share of the FP16 peak sustained in prefill
    )
    Gpt2XlSpeculativeGpt2SmallH100 = _st.SpeculativeRoundScenario(
        name="Speculative decoding of GPT-2 XL with a GPT-2 Small draft on one H100",
        description="Batch one, both models bandwidth bound: each draft step reads GPT-2 Small's FP16 weights, "
                    "the verification pass GPT-2 XL's once.",
        acceptance=0.7,  # scenario assumption: per-token acceptance probability
        draft_length=4,  # scenario assumption: tokens proposed per round
        draft_step=(_Models.Language.GPT2_Small.size_in_bytes(_BYTES_FP16)
                    / _Hardware.Cloud.H100.memory.bandwidth).to("ms"),
        verify_time=Gpt2XlFp16DecodeOnH100.result.memory_time,
    )
    Llama3_8BFp16ConversationsH100 = _st.KvAdmissionScenario(
        name="Llama 3 8B conversations of 8,192 tokens on one H100, 16-bit weights and cache",
        description="The H100's memory after the weights, over each conversation's cache; no runtime reserve.",
        workload=_Models.Language.Llama3_8B,
        accelerators=1,
        hbm_per_accelerator=_Hardware.Cloud.H100.memory.capacity,
        weight_bytes_per_param=_BYTES_FP16,
        reserve=Q_(0, "GB"),  # scenario assumption: before runtime workspace
        context_tokens=8192,  # scenario assumption: one long conversation
        kv_bytes_per_elem=_BYTES_FP16,
    )
    Llama3_70BInt4KvBudgetH100 = Llama3_8BFp16ConversationsH100.variant(
        name="A 70B-class model with 4-bit weights on one H100: the 16-bit cache budget",
        description="Llama 3 70B geometry (80 layers, 8 key-value heads of 128); 4-bit weights.",
        workload=_Models.Language.Llama3_70B,
        weight_bytes_per_param=_BYTES_INT4,
        requests=32,  # scenario assumption: concurrent conversations for the scale-up
    )
    Llama3_8BInt4ChatServiceH100 = _st.SingleAcceleratorLlmServiceScenario(
        name="An 8-billion-parameter Llama 3 chat service on one H100 with 4-bit weights",
        description="1,000-token prompts and 256-token answers; AWQ 4-bit weights and a 16-bit cache; a 72 GiB "
                    "cache reserve; prefill admits 10,000 prompt tokens per second; $3 per H100-hour.",
        workload=_Models.Language.Llama3_8B,
        hardware=_Hardware.Cloud.H100,
        precision="int4",
        kv_bytes_per_value=_BYTES_FP16,
        prompt_tokens=1000,  # scenario assumption
        decode_tokens=256,  # scenario assumption
        prefill_efficiency=0.4,  # scenario assumption: share of peak in prefill
        kv_reserve=Q_(72, "GiB"),  # scenario assumption: memory set aside for the cache
        prefill_rate=Q_(10_000, "1/second"),  # scenario assumption: isolated prefill rate, prompt tokens
        price=Q_(3.00, "dollar/hour"),  # scenario assumption: H100 SXM5 rental
        ttft_target=Q_(200, "ms"),  # scenario assumption
        tpot_target=Q_(20, "ms"),  # scenario assumption
    )
    H100ChatEnergyPerToken = _st.TokenEnergyScenario(
        name="Energy per generated token of an H100 chat service, well used and poorly used",
        description="The H100 at its TDP plus a 300 W host share, 114 requests at 8 tokens per second each; "
                    "at 10 percent of that rate the GPU still draws 300 W.",
        accelerator_power=_Hardware.Cloud.H100.tdp,
        host_power=Q_(300, "watt"),  # scenario assumption: host server share
        low_use_accelerator_power=Q_(300, "watt"),  # scenario assumption: GPU near idle
        concurrent_requests=114,  # scenario assumption: concurrent requests per H100
        per_request_rate=Q_(8, "1/second"),  # scenario assumption: decode tokens/s per request
        response_tokens=500,  # scenario assumption: typical response length
        low_use_fraction=0.1,  # scenario assumption: poorly batched
    )

    # --- Agent cost accounting (added 2026-10-20) ------------------------------
    # Types in scenario_types.py, solvers in engine/solvers/cost_accounting.py.
    # Prices are hypothetical tariffs, not a provider schedule, unless a source
    # is named; every other input is a stated teaching assumption.
    FrontierModelRepairCohort = _st.CohortAttemptCostScenario(
        name="100 repair tickets on a frontier-priced model, one attempt each",
        description="Four turns of 16,000 input and 800 output tokens at $3 and $15 per million; $0.03 of "
                    "environment per turn, $0.05 of assessment per attempt, 75 accepted; $10 of triage per "
                    "rejected attempt.",
        tasks=100,  # scenario assumption
        accepted=75,  # scenario assumption
        input_price=OutputCeilingThreeAllocations.input_price,
        output_price=ProbeVsGeneratedExplanation.output_price,
        turns=4,  # scenario assumption
        input_tokens_per_turn=16_000,  # scenario assumption
        output_tokens_per_turn=800,  # scenario assumption
        environment_per_turn=Q_(0.03, "USD"),  # scenario assumption: tools, sandbox holding, cleanup
        assessment_per_attempt=Q_(0.05, "USD"),  # scenario assumption
        triage_per_rejected=Q_(10.00, "USD"),  # scenario assumption: engineer triage only
    )
    LowTokenPriceRepairCohort = FrontierModelRepairCohort.variant(
        name="The same 100 tickets on a model whose tokens cost a tenth as much",
        description="Seven turns of 28,000 input and 1,200 output tokens at $0.30 and $1.20 per million; "
                    "25 accepted; the same environment, assessment, and triage charges.",
        accepted=25,  # scenario assumption
        input_price=Q_(0.30, "USD / Mtoken"),  # scenario assumption: hypothetical low tariff
        output_price=Q_(1.20, "USD / Mtoken"),  # scenario assumption: hypothetical low tariff
        turns=7,  # scenario assumption
        input_tokens_per_turn=28_000,  # scenario assumption
        output_tokens_per_turn=1_200,  # scenario assumption
    )
    RepairCohortTokenPriceComparison = _st.TokenPriceComparisonScenario(
        name="Frontier against low-token-price repair cohorts under one acceptance contract",
        description="Cost per accepted repair with and without triage, and the token, attempt, and "
                    "acceptance cuts.",
        baseline=FrontierModelRepairCohort,
        alternative=LowTokenPriceRepairCohort,
    )
    RepairResendCacheBill = _st.ResendCacheBillScenario(
        name="The 30-turn resend trajectory billed fresh and with ideal prefix reuse",
        description="Hypothetical $3 fresh input, cached reads at 10 percent, $15 output per million "
                    "tokens; a 10 percent miss on the eligible prefix for sensitivity.",
        trajectory=CodingAgentResend30Turns,
        input_price=RepairTrajectoryBudgetMeter.input_price,
        cached_fraction=RepairTrajectoryBudgetMeter.cached_fraction,
        output_price=RepairTrajectoryBudgetMeter.output_price,
        miss_fraction=0.10,  # scenario assumption: share of the eligible prefix that misses
    )
    RepairCompactionPayback = _st.CompactionPaybackScenario(
        name="Compacting the repair trajectory at call 16 to a 12,000-token summary",
        description="The summary costs 1,500 output tokens and is resent fresh; every later call reads "
                    "18,500 fewer cached tokens.",
        bill=RepairResendCacheBill,
        compact_turn=16,  # scenario assumption
        compact_to=12_000,  # scenario assumption: compacted context, tokens
        summary_tokens=1_500,  # scenario assumption: summary output, tokens
    )
    RepairCompactionPaybackHalfCachedRate = RepairCompactionPayback.variant(
        name="The same compaction with cached reads at half the rate",
        description="Cached reads at 5 percent of the input price, as a newer published tariff charges.",
        bill=RepairResendCacheBill.variant(cached_fraction=RepairResendCacheBill.cached_fraction / 2),
    )
    CheapRetryAgainstStrongCall = _st.RetryPolicyScenario(
        name="A cheap attempt retried up to five times against one strong attempt",
        description="Cheap: passes 30 percent at $0.10; strong: 70 percent at $0.40; attempts independent.",
        pass_probability=0.30,  # scenario assumption
        attempt_cost=Q_(0.10, "USD"),  # scenario assumption: all-in
        max_attempts=5,  # scenario assumption
        strong_pass_probability=0.70,  # scenario assumption
        strong_attempt_cost=Q_(0.40, "USD"),  # scenario assumption
    )
    SerialRepairEpisode = _st.SerialEpisodeScenario(
        name="A short serial repair episode as disjoint intervals, model calls 4x faster",
        description="Six model calls, 245.60 s of required tests, sandbox creation and snapshot export, "
                    "and runtime setup and status.",
        model_intervals=[Q_(x, "second") for x in (1.20, 8.40, 1.85, 11.20, 2.10, 4.50)],  # scenario assumption
        tests=Q_(245.60, "second"),  # scenario assumption
        lifecycle_intervals=[Q_(x, "second") for x in (14.80, 22.55)],  # scenario assumption
        runtime_intervals=[Q_(x, "second") for x in (0.12, 0.08)],  # scenario assumption
        model_speedup=4.0,  # scenario assumption
    )
    SerialRepairEpisodeSpeculation = _st.SpeculativeEpisodeScenario(
        name="Speculative decoding on the serial repair episode's model time",
        description="Five drafted tokens per round at 0.08 of a target step each; 3.6 or 0.3 kept per "
                    "round. Treats the model time as pure decode, an optimistic bound.",
        episode=SerialRepairEpisode,
        draft_length=5,  # scenario assumption
        draft_cost_ratio=0.08,  # scenario assumption
        accepted_high=3.6,  # scenario assumption: drafts align
        accepted_low=0.3,  # scenario assumption: a hard reasoning task
    )
    CodeEditThreeTierCascade = _st.ModelCascadeScenario(
        name="A three-tier cascade over 100,000 code-editing tasks, one gate at every exit",
        description="Tier calls of $0.0004, $0.0020, and $0.0600 and 0.25, 0.80, and 4.50 s; a $0.0010, "
                    "0.40 s gate; 60,000, 28,000, and 11,400 accepted; the frontier tier alone accepts "
                    "97,000. A relaxed first-tier gate that falsely accepts 0.9 percent of its exits, at "
                    "$10 of remediation each.",
        tasks=100_000,  # scenario assumption: completed cohort
        call_costs=[Q_(x, "USD") for x in (0.0004, 0.0020, 0.0600)],  # scenario assumption
        gate_cost=Q_(0.0010, "USD"),  # scenario assumption: same contract at every tier
        call_times=[Q_(x, "second") for x in (0.25, 0.80, 4.50)],  # scenario assumption
        gate_time=Q_(0.40, "second"),  # scenario assumption
        accepted_by_tier=[60_000, 28_000, 11_400],  # scenario assumption
        direct_accepted=97_000,  # scenario assumption: separate full-cohort baseline
        false_accept_rate=0.009,  # scenario assumption: a relaxed first-tier gate
        harm_per_false_accept=Q_(10.0, "USD"),  # scenario assumption: find and remediate one false accept
    )
    RepairTrajectoryQuotaAdmission = _st.TrajectoryAdmissionScenario(
        name="Repair trajectories admitted once a minute against a 2 million token-per-minute quota",
        description="20 minutes present, 60 model calls of 30,000 quota-counted tokens each; acceptance "
                    "falling by a tenth of itself at fixed goodput.",
        arrival=Q_(1, "1 / minute"),  # scenario assumption
        residence=Q_(20, "minute"),  # scenario assumption
        calls=60,  # scenario assumption
        tokens_per_call=30_000,  # scenario assumption: every token the quota counts
        quota=Q_(2_000_000, "1 / minute"),  # scenario assumption
        acceptance_drop=0.10,  # scenario assumption
    )
    BurstyModelServerQueuePrice = _st.KingmanQueueScenario(
        name="One bursty model server at 70 and 90 percent utilization",
        description="1 s mean service, c_a^2 = 2 and c_s^2 = 4, held resources at $0.002 per second.",
        service=Q_(1, "second"),  # scenario assumption
        arrival_scv=2.0,  # scenario assumption
        service_scv=4.0,  # scenario assumption
        low_utilization=0.70,  # scenario assumption
        high_utilization=0.90,  # scenario assumption
        holding_rate=Q_(0.002, "USD / second"),  # scenario assumption
    )
    ThirtyCallTailExposure = _st.SerialTailExposureScenario(
        name="Thirty serial model calls, each past its own 99th percentile with probability 0.01",
        description="Independent per-call tail events along one trajectory.",
        calls=30,  # scenario assumption: serial calls on the path
        tail_probability=0.01,  # the definition of a 99th percentile
    )
    DedicatedCapacityBreakEven = _st.DedicatedBreakEvenScenario(
        name="Renting a matched-capability API against owning capacity at $28 an hour",
        description="Five input tokens per output token, 80 percent of input cached at $3, $0.30, and $15 "
                    "per million; owned capacity at $28 an hour plus $2 per million output, 1,200 output "
                    "tokens per second at full load, run at 52 percent.",
        input_per_output=5.0,  # scenario assumption
        cached_fraction=0.80,  # scenario assumption
        input_price=OutputCeilingThreeAllocations.input_price,
        cached_price=OutputCeilingThreeAllocations.input_price * RepairPrefixLayoutBill.cached_fraction,
        output_price=ProbeVsGeneratedExplanation.output_price,
        fixed_per_hour=Q_(28.0, "USD / hour"),  # scenario assumption: dedicated commitment
        variable_price=Q_(2.0, "USD / Mtoken"),  # scenario assumption: marginal cost per output token
        reference_rate=Q_(1_200, "1 / second"),  # scenario assumption: output tokens per second at full load
        load=0.52,  # scenario assumption: operating load
    )
    RepairBudgetLateBill = _st.ConservedBudgetScenario(
        name="A $5 repair budget across two children, one stopped with a remote job billed late",
        description="Planning settles $0.042; a trace child gets $1.20 and closes at $0.38; a fuzzing "
                    "child gets $2.50, settles $0.85, and stops with $1 reserved for a remote job that "
                    "bills $0.70 late, delivered twice.",
        budget=Q_(5.0, "USD"),  # scenario assumption
        planning=Q_(0.042, "USD"),  # scenario assumption
        allocations=[Q_(1.20, "USD"), Q_(2.50, "USD")],  # scenario assumption
        settled_children=[Q_(0.38, "USD"), Q_(0.85, "USD")],  # scenario assumption
        pending=Q_(1.0, "USD"),  # scenario assumption: enforceable maximum remaining remote charge
        late_charge=Q_(0.70, "USD"),  # scenario assumption: authenticated final bill
        deliveries=2,  # scenario assumption: the final bill is redelivered once
    )
    DependencyUpgradeArchitectures = _st.ArchitectureChoiceScenario(
        name="Dependency upgrades: a rule-based rewriter, one frontier call, or a bounded three-turn loop",
        description="10,000 upgrades a month; a $0.002 check per candidate; rejects go to a $35 engineer "
                    "who resolves 90 percent; a 95 percent combined resolution floor.",
        tickets=10_000,  # scenario assumption: monthly arrivals
        single_costs={"Rule-based rewriter": Q_(0.001, "USD"),  # scenario assumption
                      "Single model call": Q_(0.045, "USD")},  # scenario assumption
        single_acceptance={"Rule-based rewriter": 0.32,  # scenario assumption
                           "Single model call": 0.68},  # scenario assumption
        loop_name="Bounded 3-turn loop",
        turn_cost=Q_(0.006, "USD"),  # scenario assumption
        turn_acceptance=[0.55, 0.42, 0.30],  # scenario assumption: conditional on reaching the turn
        check_cost=Q_(0.002, "USD"),  # scenario assumption
        human_cost=Q_(35, "USD"),  # scenario assumption: one fallback with its final check
        human_resolution=0.90,  # scenario assumption
        resolution_floor=0.95,  # scenario assumption
    )
    CodeReviewAdaptationPayback = _st.AdaptationPaybackScenario(
        name="Adapting a smaller model for code review, repaid by cheaper tasks",
        description="Five steps of 12,000 input and 800 output tokens at $3 and $15 per million, against "
                    "four steps of 1,500 and 800 at $1 and $4; $4,000 of data, $500 of training, $8,000 of "
                    "evaluation; 20,000 reviews a month, or a quarter of that.",
        base={"input_tokens": 12_000, "output_tokens": 800, "steps": 5,  # scenario assumption
              "input_price": OutputCeilingThreeAllocations.input_price,
              "output_price": ProbeVsGeneratedExplanation.output_price},
        adapted={"input_tokens": 1_500, "output_tokens": 800, "steps": 4,  # scenario assumption
                 "input_price": Q_(1, "USD / Mtoken"),  # scenario assumption
                 "output_price": Q_(4, "USD / Mtoken")},  # scenario assumption
        data_cost=Q_(4_000, "USD"),  # scenario assumption: collection and curation
        training_cost=Q_(500, "USD"),  # scenario assumption: training compute
        evaluation_cost=Q_(8_000, "USD"),  # scenario assumption: evaluation and review
        monthly_volume=20_000,  # scenario assumption
        low_volume=5_000,  # scenario assumption: a quarter of the planned volume
    )
    LighthouseWorkloadLevers = _st.WorkloadLeverScenario(
        name="Cost per accepted task for repair, research, remediation, and support, and three levers",
        description="Repair's tokens are the cached 30-turn bill with $0.03 of sandbox and tools per turn; "
                    "research is the orchestrator-worker fan-out of a chat; remediation and support read "
                    "their workload profiles. Levers: halve the token price, halve the largest term, or "
                    "add 10 points of acceptance.",
        workloads={
            "repair": {"bill": RepairResendCacheBill,
                       "turns": _Agents.Coding.SWE_Bench_Runner.max_trajectory_steps,
                       "environment_per_turn": 0.03,  # scenario assumption: sandbox holding and tools, USD
                       "assessment": 0.10,  # scenario assumption: sealed regression suite, USD
                       "human": 0.0,
                       "acceptance": 0.40},  # scenario assumption
            "research": {"chat_input": _Agents.Research.OrchestratorWorker.chat_input_tokens.to("token").magnitude,
                         "chat_output": _Agents.Research.OrchestratorWorker.chat_output_tokens.to("token").magnitude,
                         "token_multiple": _Agents.Research.OrchestratorWorker.fanout_token_multiple,
                         "environment": _Agents.Research.OrchestratorWorker.environment_cost_per_attempt.to("USD").magnitude,
                         "assessment": _Agents.Research.OrchestratorWorker.assessment_cost_per_attempt.to("USD").magnitude,
                         "human": 0.0,
                         "acceptance": _Agents.Research.OrchestratorWorker.acceptance_probability},
            "remediation": {"model": _Agents.Remediation.OnCall.model_cost_per_attempt.to("USD").magnitude,
                            "environment": _Agents.Remediation.OnCall.environment_cost_per_attempt.to("USD").magnitude,
                            "assessment": _Agents.Remediation.OnCall.assessment_cost_per_attempt.to("USD").magnitude,
                            "human": _Agents.Remediation.OnCall.approval_cost_per_attempt.to("USD").magnitude,
                            "acceptance": _Agents.Remediation.OnCall.acceptance_probability},
            "support": {"model": _Agents.Support.CustomerOps.model_cost_per_attempt.to("USD").magnitude,
                        "environment": _Agents.Support.CustomerOps.environment_cost_per_attempt.to("USD").magnitude,
                        "assessment": _Agents.Support.CustomerOps.assessment_cost_per_attempt.to("USD").magnitude,
                        "handoff": _Agents.Support.CustomerOps.handoff_cost.to("USD").magnitude,
                        "acceptance": _Agents.Support.CustomerOps.acceptance_probability},
        },
        input_price=RepairResendCacheBill.input_price,
        output_price=RepairResendCacheBill.output_price,
        acceptance_gain=0.10,  # scenario assumption: ten points of acceptance
    )

    # --- Premises no sensor reports (added 2026-10-20) ---
    # The aisle budget on a clear oil film, the film detector's range and closure test,
    # the restricted transit's cost, a hypothetical packaging cell, the coverage bill of
    # a shielded hazard claim, and the series spring at the door latch.
    WarehouseFinishedStopOilFilm = WarehouseFinishedStopAtAisleSpeed.variant(
        name="Warehouse base's finished stopping budget on a clear oil film",
        description="Deceleration capped by the film's friction, all weight on the driven wheels; the jerk "
                    "bound belongs to the load and is unchanged.",
        deceleration=_traction_limit(_Embodied.Scenario.WarehouseAisle.mu_oil_film),
    )
    WarehouseFilmDetectorRange = WarehouseFinishedStopAtAisleSpeed.variant(
        name="Range a film detector needs at the aisle speed",
        description="The dry-floor finished budget plus the travel during the detector's sensing and "
                    "inference (added_delay): the stop then ends before the film's leading edge.",
        added_delay=Q_(100.0, "ms"),  # scenario assumption: chosen film-detector latency
    )
    WarehouseFilmDetectorClosure = _st.ZeroFailureScenario(
        name="Seeded-film closure test for the friction register entry",
        description="Clean trials that bound the detector's miss rate at the target, and the bound a "
                    "smaller 1,000-trial campaign supports.",
        trials=1000,           # scenario assumption: a smaller zero-miss campaign, for contrast
        confidence=0.95,       # scenario assumption: chosen one-sided level
        target_rate=0.001,     # scenario assumption: chosen miss-rate bound for closure
    )
    WarehouseRestrictedTransit = _st.RestrictedTransitScenario(
        name="The aisle leg at the restricted speed on inspected floors",
        description="Speed cut and leg time when the register holds the base to the restricted mode.",
        site=_Embodied.Scenario.WarehouseAisle,
        restricted_speed=WarehouseFinishedStopRestricted.speed,
    )
    PackagingCellTransit = _st.PackagingCellScenario(
        name="A hypothetical packaging cell's carriage at its nominal speed",
        description="Camera-triggered stop before a fixed stop line: four delay allocations, a braking "
                    "lower bound, and a reserve held back from the clearance. Not measurements.",
        mass=Q_(12.0, "kg"),                      # scenario assumption: total moving mass, including load
        speed=Q_(1.5, "m/s"),                     # scenario assumption: nominal speed
        clearance=Q_(60.0, "mm"),                 # scenario assumption: clearance to the stop line
        reserve=Q_(15.0, "mm"),                   # scenario assumption: allocated reserve
        deceleration=Q_(15.0, "m/s**2"),          # scenario assumption: braking lower bound after delay
        exposure=Q_(16.0, "ms"),                  # scenario assumption: event to usable frame
        processing=Q_(36.0, "ms"),                # scenario assumption: transfer, inference, scheduling, admission
        bus=Q_(2.0, "ms"),                        # scenario assumption: command to brake-driver delivery
        onset=Q_(10.0, "ms"),                     # scenario assumption: delivery to full deceleration
        check_rate=Q_(1.0, "kHz"),                # scenario assumption: independent kinematic-check rate
        force_limit=Q_(220.0, "N"),               # scenario assumption: limiter setting, not material strength
        temperature_trip=Q_(105.0, "degC"),       # scenario assumption: selected sensor threshold
        stall_timeout=Q_(400.0, "ms"),            # scenario assumption: selected stall-detection timeout
        bus_trip=Q_(60.0, "V"),                   # scenario assumption: inverter overvoltage threshold
    )
    PackagingCellRestrictedMode = PackagingCellTransit.variant(
        name="The packaging cell's carriage in the candidate restricted mode",
        description="The same chain and brake at a speed chosen below the ceiling.",
        speed=Q_(0.5, "m/s"),                     # scenario assumption: restricted speed
    )
    PackagingCellHalfProcessing = PackagingCellTransit.variant(
        name="The packaging cell with its processing allocation halved",
        description="Sensitivity of the ceiling to the transfer-through-admission allocation.",
        processing=(PackagingCellTransit.processing / 2).to("ms"),
    )
    PackagingCellHalfBrake = PackagingCellTransit.variant(
        name="The packaging cell with its braking bound halved",
        description="Sensitivity of the ceiling to the qualified deceleration.",
        deceleration=(PackagingCellTransit.deceleration / 2).to("m/s**2"),
    )
    PackagingCellQualifiedBrake = PackagingCellRestrictedMode.variant(
        name="The restricted mode with the brake qualified at a lower bound",
        description="Load testing qualifies a weaker brake; the ceiling and the restricted stop against it.",
        deceleration=Q_(12.0, "m/s**2"),          # scenario assumption: lower qualified braking bound
    )
    ShieldCoverageBill = _st.ShieldCoverageBillScenario(
        name="A shielded hazard claim against a borrowed 1e-9 per hour target",
        description="Zero hazardous proposals in one release cycle of a 100-unit fleet, a shield of 0.9999 "
                    "class coverage, common cause set to zero; the evidence the claim still lacks.",
        fleet_units=100,                          # scenario assumption: fleet size
        cycle_days=Q_(90.0, "day"),               # scenario assumption: release cycle
        hours_per_day=Q_(24.0, "hour/day"),       # scenario assumption: continuous operation
        coverage=0.9999,                          # scenario assumption: class-specific interception coverage
        common_cause_rate=Q_(0.0, "1/hour"),      # scenario assumption: hardware and common cause set to zero
        target_rate=Q_(1.0e-9, "1/hour"),         # scenario assumption: borrowed civil-aviation band, not statutory
        confidence=0.95,                          # scenario assumption: chosen one-sided level
    )
    WarehouseLatchSeriesSpring = _st.LatchSeriesSpringScenario(
        name="A series spring at the door latch under the declared force floor",
        description="Spring sized for the fast approach; coasting peaks with torque cut at the tripwire.",
        site=_Embodied.Scenario.WarehouseAisle,
        machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
    )

    # --- Closing an agent course (added 2026-10-20) ---------------------------
    # One on-call remediation attempt priced on its workload profile, a lost
    # reply settled by status rather than retried, and the closing arithmetic:
    # residual violations, a passing set's invalid share, a shortened health
    # window, blame search over a horizon, the specification boundary, a
    # passing-patch audit, and METR's 50 against 80 percent horizons.
    RemediationIncidentAttempt = _st.AcceptedTaskCostScenario(
        name="One on-call remediation attempt, priced on its workload profile",
        description="Model calls against the environment, health-check, and approver charges of the "
                    "on-call remediation profile, at its acceptance; then twice the model calls.",
        model_cost=_Agents.Remediation.OnCall.model_cost_per_attempt,
        other_cost=(_Agents.Remediation.OnCall.environment_cost_per_attempt
                    + _Agents.Remediation.OnCall.assessment_cost_per_attempt
                    + _Agents.Remediation.OnCall.approval_cost_per_attempt),
        acceptance=_Agents.Remediation.OnCall.acceptance_probability,
        model_cost_growth=1.0,  # scenario assumption: double the model calls for more diagnosis
    )
    BlindRetryAgainstStatusLookup = _st.BlindRetryScenario(
        name="A lost reply on 25,000 mutating calls a day: blind retry against a status lookup",
        description="Requests lost with probability 0.001 and replies with 0.004; a duplicated change "
                    "costs $2,000 and a read-only status query $0.01.",
        request_loss=0.001,  # scenario assumption: hypothetical, not a measured rate
        reply_loss=0.004,  # scenario assumption: hypothetical, not a measured rate
        workers=1,  # one fleet counted as a whole
        calls_per_worker=25_000,  # scenario assumption: mutating calls per day
        duplicate_cost=Q_(2_000, "USD"),  # scenario assumption: cost of one duplicated change
        status_lookup_cost=Q_(0.01, "USD"),  # scenario assumption: cost of one status query
    )
    ResidualViolationsFiftyTurns = _st.ResidualViolationScenario(
        name="A 0.1 percent residual violation rate per step over 50-turn trajectories, 10,000 a day",
        description="Training lowers the rate of violating proposals without driving it to zero.",
        step_violation=0.001,  # scenario assumption: residual per-step rate after training
        horizon=50,  # scenario assumption: turns per trajectory
        trajectories_per_day=10_000,  # scenario assumption
    )
    ThirtyPercentValidPassingSet = _st.CheckerPosteriorScenario(
        name="A check over candidates of which 30 percent are valid",
        description="The check misses 10 percent of valid candidates and passes 5 percent of invalid ones.",
        defect_rate=0.70,  # scenario assumption: 30 percent of candidates valid
        false_accept_rate=0.05,  # scenario assumption: P(PASS | invalid)
        valid_accept_rate=0.90,  # scenario assumption: P(PASS | valid), 10 percent missed
    )
    HalvedHealthWindow = _st.ShortenedWindowScenario(
        name="A health window shortened to half its adopted length",
        description="Half the samples; the latency interval widens by the square root of two.",
        window_share=0.5,  # scenario assumption
    )
    CheckedWindowBlameSearch = _st.CounterfactualSearchScenario(
        name="Blame for a failure found only at the end of a 50-step task, or within one checked window",
        description="Four candidate actions per step; intermediate checks every five steps; 1.2 s per "
                    "replayed check.",
        branching=4,  # scenario assumption: candidate actions per step
        horizon=50,  # scenario assumption: steps in the task
        window=5,  # scenario assumption: steps between intermediate checks
        replay_time=Q_(1.2, "s"),  # scenario assumption: time per replayed check
    )

    # --- Volume I synthesis: recalled calculations (added 2026-10-20) -------
    # The conclusion of a course recalls earlier calculations; each entry
    # below varies an existing scenario or reuses an existing type.
    Conv3x3At256Channels = ImageNet64ChannelConv.variant(
        name="A 3x3 convolution from 256 to 256 channels, FP32",
        description="A representative later layer; the standard-over-separable ratio depends only on the "
                    "kernel and the output channels.",
        in_channels=256,                  # scenario assumption: representative channel count
        out_channels=256,                 # scenario assumption
    )
    ChatEndpointTtftMeanVersusP99 = _st.MeanVersusTailScenario(
        name="An LLM chat endpoint: mean time to first token against its p99",
        description="Illustrative; the p99 is driven by long prompts and queueing behind long generations.",
        mean=Q_(50, "ms"),                # scenario assumption: mean time to first token
        tail_low=Q_(2000, "ms"),          # scenario assumption: p99 time to first token
        tail_high=Q_(2000, "ms"),         # scenario assumption: one stated value
    )
    HostPreprocessingRequestPath = _st.AmdahlScenario(
        name="A request that is 90 percent host preprocessing, with inference made 10x faster",
        description="Shares of the original request time; only the accelerator inference speeds up.",
        stages={"preprocess": Q_(90, "percent"),  # scenario assumption: CPU-bound preprocessing share
                "inference": Q_(10, "percent")},  # scenario assumption
        stage="inference",
        speedup=10.0,                     # scenario assumption: rewritten inference kernel
    )

    # --- ML operations, part 2: watching and acting (added 2026-10-20) -------
    # Monitoring, alerting, and retraining on the same ranking service.
    LabeledOutcomeDriftConfirmation = _st.DriftConfirmationScenario(
        name="Confirming a 5-point drop from 95 percent accuracy with 1,000 labeled outcomes",
        description="The policy's evidence budget above the one-sample binomial floor, collected at one "
                    "labeled outcome a second or 100 a day. Illustrative.",
        baseline_accuracy=0.95,  # scenario assumption: original accuracy
        drop=0.05,  # scenario assumption: a 5-percentage-point drop to confirm
        confidence=0.95,  # scenario assumption: two-sided confidence
        power=0.80,  # scenario assumption: chance of catching a real drop of that size
        evidence_budget=1000,  # scenario assumption: the monitoring policy's budget
        label_rates={"high": Q_(1, "1/s"),  # scenario assumption: one labeled outcome a second
                     "low": Q_(100, "1/day")},  # scenario assumption: 100 labeled outcomes a day
        compared_drops=[0.02, 0.01, 0.005],  # scenario assumption: smaller drops to size
    )
    RecommenderDetectionDelayLoss = _st.DetectionDelayLossScenario(
        name="A 5 percent conversion drop on a $50M-a-year recommender, found monthly or daily",
        description="Revenue lost per incident at a 28-day review and at a daily check, four incidents a year.",
        annual_value=Q_(50e6, "USD"),  # scenario assumption: a year's recommendation revenue
        quality_drop=0.05,  # scenario assumption: conversion-rate degradation
        slow_detection=Q_(28, "day"),  # scenario assumption: monthly review cycle (about four weeks)
        fast_detection=Q_(1, "day"),  # scenario assumption: daily automated monitoring
        incidents_per_year=4,  # scenario assumption: a high-drift domain
    )
    RankingRetrainingInterval = _st.RetrainingIntervalScenario(
        name="Retraining interval of the ranking service: 1 million queries a day at $5,000 a run",
        description="Square-root optimum of run cost against staleness on the service's fitted decay; the "
                    "run's compute is four H100s for six hours at the cloud training rate. Illustrative.",
        drift=RankingServiceAccuracyDrift,
        query_rate=Q_(1e6, "1/day"),  # scenario assumption: ranking requests per day
        value_per_query=Q_(0.50, "USD"),  # scenario assumption: value per query of a unit accuracy change
        run_cost=Q_(5000, "USD"),  # scenario assumption: total cost of one retraining run
        footprint=DlrmReleaseFootprintH100,
        accelerator_price=_Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
        compared_intervals=[Q_(1, "day"), Q_(3, "day"), Q_(7, "day"), Q_(30, "day")],  # scenario assumption
        scheduled_interval=Q_(90, "day"),  # scenario assumption: a quarterly schedule
    )
    RankingRetrainingIntervalFourfoldTraffic = RankingRetrainingInterval.variant(
        name="The same retraining interval at four times the traffic",
        query_rate=Q_(1e6, "1/day") * 4,  # scenario assumption: a fourfold change in query volume
    )
    RankingUserAgePsi = _st.BinShareShiftScenario(
        name="User-age PSI of the ranking service as its audience ages",
        description="Six age bins; four move by 3 points and two by 2. Illustrative.",
        bins=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"],
        baseline_shares=[0.15, 0.25, 0.20, 0.18, 0.12, 0.10],  # scenario assumption: training
        current_shares=[0.12, 0.22, 0.18, 0.20, 0.15, 0.13],  # scenario assumption: serving
    )
    OneModelDailyMetricTests = _st.MonitorFleetAlertScenario(
        name="One model in 3 deployment variants x 50 metrics, each tested once a day at 0.05",
        description="False alerts a day from a monitor with nothing to find; the variants are production, "
                    "canary, and staging.",
        models=3,  # scenario assumption: deployment variants of one model
        metrics_per_model=50,  # scenario assumption: metrics per deployment
        check_interval=Q_(1, "day"),  # scenario assumption: one test per series a day
        false_positive_rate=0.05,  # scenario assumption: per-test false-positive rate
    )
    RecommendationAccuracyByDevice = _st.SegmentAccuracyScenario(
        name="Recommendation accuracy by device: tablets fail behind a healthy aggregate",
        description="Traffic share and accuracy of four device segments. Illustrative.",
        traffic_shares={"Desktop": 0.45, "iOS": 0.30, "Android": 0.20, "Tablet": 0.05},  # scenario assumption
        accuracies={"Desktop": 0.94, "iOS": 0.92, "Android": 0.88, "Tablet": 0.62},  # scenario assumption
    )

    # --- Responsible engineering: lifecycle cost, carbon accounting, and duties (added 2026-10-20) ---
    RankingServiceLifecycleCost = _st.ServiceLifecycleCostScenario(
        name="Three years of the ranking service: twelve training cycles, unbatched inference, operations",
        description="The exploration-exposure ranking service's traffic at 10 ms of dedicated GPU time per "
                    "request, reserved inference and on-demand training prices, and the fleet GPU-hour "
                    "carbon factor (GPU only: no host, PUE, or embodied term).",
        daily_users=RecommenderExplorationExposure.daily_users,
        items_per_user=RecommenderExplorationExposure.slots_per_user,
        service_time=Q_(10, "ms"),  # scenario assumption: unbatched, non-overlapped GPU time per request
        inference_rate=_Infrastructure.Pricing.Cloud.GpuInferencePerHour.rate,
        training_rate=_Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
        data_prep_time=Q_(100, "hour"),  # scenario assumption: data preparation GPU-hours per cycle
        search_experiments=50,  # scenario assumption: hyperparameter configurations per cycle
        search_cost_per_experiment=Q_(40, "USD"),  # scenario assumption
        final_run_time=Q_(200, "hour"),  # scenario assumption: final training GPU-hours per cycle
        cycles=12,  # scenario assumption: quarterly retraining, initial cycle included
        years=3,  # scenario assumption: operational period
        operations_annual={"monitoring": Q_(50_000, "USD"),  # scenario assumption
                           "on-call": Q_(100_000, "USD"),  # scenario assumption: 0.5 FTE
                           "incident response": Q_(20_000, "USD")},  # scenario assumption
        carbon_per_gpu_hour=_Infrastructure.Pricing.Fleet.CarbonPerGpuHr.rate,
        serving_cut=0.20,  # scenario assumption: per-query accelerator demand reduction
        training_cut=0.50,  # scenario assumption: halving all training
    )
    Gpt3TrainingCarbon = _st.OperationalCarbonScenario(
        name="GPT-3 training energy on the US average grid",
        description="1,287 MWh, already at the facility boundary (PUE 1.1 included), against passenger-car years.",
        it_energy=_Models.Language.GPT3.training_energy_mwh,
        pue=1.0,  # the training energy already includes the datacenter's PUE
        intensity=Q_(_Infrastructure.Grids.US_Avg.carbon_intensity_g_kwh, "g/kWh"),
        comparison=Q_(4.6, "metric_ton"),  # scenario assumption: one passenger car's annual emissions
    )
    CreditDenialExplanationLoad = _st.ExplanationLoadScenario(
        name="Reasons for every denial of a credit service making 1 million decisions a day",
        description="Denial share from the two-group credit service; a 1,000-sample attributor against "
                    "exact Shapley over 20 features.",
        decisions_per_day=1_000_000,  # scenario assumption
        denial_share=1 - CreditServiceTwoGroups.pooled_positive_rate,
        samples=1_000,  # scenario assumption: model evaluations per sampled explanation
        features=20,  # scenario assumption: input features of the credit service
    )
    CreditAppealReview = _st.ReviewStaffingScenario(
        name="Substantive review of appeals against the credit service, against Cigna's reported pace",
        description="0.1 percent of 1 million daily decisions appealed, 6 minutes each, 8-hour shifts; "
                    "ProPublica's Cigna denials for comparison.",
        decisions_per_day=CreditDenialExplanationLoad.decisions_per_day,
        appeal_rate=0.001,  # scenario assumption: share of decisions appealed
        time_per_case=Q_(6, "minute"),  # scenario assumption
        shift=Q_(8, "hour"),  # scenario assumption
        compare_cases=300000.0,  # Rucker, Miller, and Armstrong, How Cigna Saves Millions by Having I...
        compare_time_per_case=Q_(1.2, "s"),  # Rucker, Miller, and Armstrong, How Cigna Saves Millions by Having I...
    )
    DpSgdPerExampleGradients = _st.PerExampleGradientScenario(
        name="Per-example FP32 gradients for the KWS DS-CNN and GPT-2 at batch 256",
        description="Naive DP-SGD clipping keeps every example's gradient before the batch sum.",
        models={"KWS": _Models.Tiny.DS_CNN, "GPT-2": _Models.Language.GPT2},
        batch_size=256,  # scenario assumption: DP-SGD batch size
        bytes_per_param=_BYTES_FP32,
    )

    # --- Course-overview anchors (added 2026-10-20) ------------------------------


    # --- SWE-bench Verified leaderboard (added 2026-10-20) ------------------------

    # --- Production gateway telemetry and repair trajectory calls (added 2026-10-21) ---
