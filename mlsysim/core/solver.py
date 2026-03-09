import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, ConfigDict
from .engine import Engine, PerformanceProfile
from .results import (
    SolverResult,
    DistributedResult, ReliabilityResult, SustainabilityResult,
    ServingResult, EconomicsResult, DataResult, TopologyResult,
    EfficiencyResult, TransformationResult, ScalingResult,
    CompressionResult, SynthesisResult, OrchestrationResult,
    InferenceScalingResult, SensitivityResult, ResponsibleEngineeringResult,
)
from .formulas import (
    calc_ring_allreduce_time,
    calc_tree_allreduce_time,
    calc_hierarchical_allreduce_time,
    calc_all_to_all_time,
    calc_mtbf_cluster,
    calc_mtbf_node,
    calc_young_daly_interval,
    calc_failure_probability,
    calc_pipeline_bubble
)
from .constants import (
    ureg, Q_, PRECISION_MAP,
    BYTES_FP16, BYTES_FP32, BYTES_INT8, BYTES_INT4,
    LATENCY_INFINIBAND, LATENCY_NVLINK,
    GPU_MTTF_HOURS, H100_TDP,
    CHINCHILLA_TOKENS_PER_PARAM, CHINCHILLA_COMPUTE_CONSTANT,
    H100_FLOPS_FP16_TENSOR,
    CLOUD_ELECTRICITY_PER_KWH,
)
from .defaults import (
    GPU_UNIT_COST_H100, ANNUAL_MAINTENANCE_RATIO,
    MFU_TRAINING_LOW, MFU_TRAINING_HIGH, MFU_INFERENCE_BATCH1, MFU_INFERENCE_BATCHED,
    QUANT_ACCURACY_DELTA_INT8, QUANT_ACCURACY_DELTA_INT4,
    PRUNING_ACCURACY_THRESHOLD, PRUNING_MILD_DELTA, PRUNING_STEEP_COEFFICIENT, PRUNING_STEEP_EXPONENT,
    NIC_MTTF_HOURS, PSU_MTTF_HOURS,
    MFU_FLASH_ATTENTION, MFU_FLASH_ATTENTION_CAP, MFU_FFN_CAP, MFU_CONV_CAP, HFU_MFU_RATIO,
    TOKENS_PER_REASONING_STEP, REFERENCE_MFU_SUSTAINED, DP_SGD_SLOWDOWN_COEFFICIENT,
)
from .types import Quantity
from ..models.types import Workload, TransformerWorkload
from ..hardware.types import HardwareNode
from ..systems.types import Fleet, NetworkFabric
from ..infra.types import Datacenter, GridProfile

class BaseSolver(ABC):
    """Base class for all mlsysim solvers.

    Each solver declares its input requirements, output type, and the
    wall(s) it resolves from the canonical taxonomy (see core/walls.py).

    Attributes
    ----------
    requires : tuple[str, ...]
        Domain concepts this solver needs (e.g., "workload", "fleet").
    produces : type[SolverResult] | None
        The typed result model this solver returns.
    walls : tuple[int, ...]
        Canonical wall numbers this solver resolves (see walls.py).
    """
    requires: tuple = ()
    produces: Optional[Type[SolverResult]] = None
    walls: tuple = ()

    @abstractmethod
    def solve(self, **kwargs) -> Any:
        pass

    @classmethod
    def schema(cls) -> dict:
        """Return a summary of this solver's interface for composition checking."""
        from .walls import wall as lookup_wall
        wall_info = []
        for n in cls.walls:
            try:
                w = lookup_wall(n)
                wall_info.append({"number": w.number, "name": w.name, "domain": w.domain.value})
            except KeyError:
                wall_info.append({"number": n, "name": "?", "domain": "?"})
        return {
            "solver": cls.__name__,
            "walls": wall_info,
            "requires": cls.requires,
            "produces": cls.produces.__name__ if cls.produces else "Any",
        }

class SingleNodeSolver(BaseSolver):
    """
    Resolves single-node hardware Roofline bounds and feasibility.

    This solver handles the 'Iron Law' of machine learning systems,
    calculating whether a model fits in memory and predicting its
    throughput based on arithmetic intensity.

    Literature Source: Williams et al. (2009), "Roofline: An Insightful Visual
    Performance Model for Floating-Point Programs and Multicore Architectures."
    """
    requires = ("workload", "hardware")
    produces = PerformanceProfile
    walls = (1, 2)

    def solve(self, model: Workload, hardware: HardwareNode, batch_size: int = 1, precision: str = "fp16", efficiency: float = 0.5, raise_errors: bool = False) -> PerformanceProfile:
        """
        Solves the performance profile for a single hardware node.
        """
        return Engine.solve(model, hardware, batch_size=batch_size, precision=precision, efficiency=efficiency, raise_errors=raise_errors)

class DistributedSolver(BaseSolver):
    """
    Resolves fleet-wide communication, synchronization, and pipelining constraints.
    
    This solver models the constraints of distributed scale for distributed training. It
    decomposes a workload across a cluster using 3D Parallelism (DP, TP, PP) 
    and calculates the resulting communication overheads and idle times 
    (bubbles) that determine the Model FLOPs Utilization (MFU).

    Literature Source: 
    1. Shoeybi et al. (2019), "Megatron-LM: Training Multi-Billion Parameter 
       Language Models Using Model Parallelism." (3D Parallelism Framework)
    2. Narayanan et al. (2019), "PipePipe: Efficient Pipeline Parallelism for 
       Training Large Models." (1F1B Pipeline Bubble Model)
    3. Patarasuk & Mueller (2009), "Bandwidth-Optimal All-Reduce Algorithms 
       for Clusters of Workstations." (Ring All-Reduce)
    """
    requires = ("workload", "fleet")
    produces = DistributedResult
    walls = (10,)

    def solve(self,
              model: Workload,
              fleet: Fleet, 
              batch_size: int = 1, 
              precision: str = "fp16", 
              efficiency: float = 0.5,
              tp_size: int = 1,
              pp_size: int = 1,
              ep_size: int = 1,
              v_stages: int = 1,
              microbatch_count: int = 1,
              topology_override: Optional[str] = None) -> DistributedResult:
        """
        Calculates distributed training performance using the 3D/4D Parallelism model.

        Parameters
        ----------
        model : Workload
            The model architecture to simulate.
        fleet : Fleet
            The hardware cluster and network topology.
        batch_size : int
            Global batch size.
        precision : str
            Numerical precision (fp16, fp32, int8).
        efficiency : float
            Achieved compute efficiency (0.0 to 1.0).
        tp_size : int
            Tensor Parallelism degree. Splits individual layers across GPUs, 
            usually within a single node over high-speed NVLink.
        pp_size : int
            Pipeline Parallelism degree. Chains model layers across multiple 
            nodes, introducing 'pipeline bubbles' while saving memory.
        ep_size : int
            Expert Parallelism degree for MoE models. Introduces All-to-All
            communication overhead across nodes.
        v_stages : int
            Number of virtual stages for interleaved pipeline schedules.
        microbatch_count : int
            Number of microbatches (M). Increasing M reduces the pipeline 
            bubble but increases synchronization overhead.
        topology_override : str, optional
            Force a specific topology (ring, tree).

        Returns
        -------
        Dict[str, Any]
            Metrics including DP/TP/EP latency, the Pipeline Bubble penalty, 
            and the final Scaling Efficiency.
        """
        # 1. 3D/4D Parallelism Decomposition
        n_accelerators = fleet.total_accelerators
        dp_size = n_accelerators // (tp_size * pp_size * ep_size)
        
        if dp_size < 1:
            raise ValueError(f"Infeasible 4D Parallelism: TP({tp_size}) * PP({pp_size}) * EP({ep_size}) > Total({n_accelerators})")

        # 2. Single Node Performance (Computation)
        local_batch = max(1, batch_size // dp_size)
        if batch_size < dp_size:
            import warnings
            warnings.warn(
                f"batch_size ({batch_size}) < dp_size ({dp_size}): "
                f"some ranks will be idle. Using local_batch=1.",
                stacklevel=2,
            )
        node_perf = Engine.solve(model, fleet.node.accelerator, batch_size=local_batch, precision=precision, efficiency=efficiency)

        # 3. Communication Overhead (Network)
        # DP AllReduce exchanges gradients, which equal model size in the active precision.
        # With TP, each rank holds 1/tp_size of the model, so gradient buffer is smaller.
        gradient_size = model.size_in_bytes() / tp_size
        
        # DP AllReduce (Gradients — same size as model weights per DP rank)
        if dp_size > 1:
            if fleet.node.accelerators_per_node > 1 and dp_size > fleet.node.accelerators_per_node:
                # Hierarchical: Ring within node, then Ring across nodes
                t_comm_dp = calc_hierarchical_allreduce_time(
                    message_bytes=gradient_size,
                    n_nodes=dp_size // fleet.node.accelerators_per_node,
                    gpus_per_node=fleet.node.accelerators_per_node,
                    intra_node_bw=fleet.node.intra_node_bw,
                    inter_node_bw=fleet.fabric.bandwidth / fleet.fabric.oversubscription_ratio,
                    inter_node_lat=fleet.fabric.latency or LATENCY_INFINIBAND
                )
            else:
                # Single node or small DP: Intra-node only
                t_comm_dp = calc_ring_allreduce_time(
                    gradient_size,
                    dp_size,
                    fleet.node.intra_node_bw,
                    LATENCY_NVLINK
                )
        else:
            t_comm_dp = Q_("0 ms")

        # TP Communication (activation exchange, intra-node NVLink)
        full_model_size = model.size_in_bytes()
        t_comm_tp = (full_model_size / tp_size / fleet.node.intra_node_bw).to("ms") if tp_size > 1 else Q_("0 ms")

        # EP Communication (All-to-All token routing for MoE)
        if ep_size > 1:
            t_comm_ep = calc_all_to_all_time(
                message_bytes=full_model_size,
                n_gpus=ep_size,
                bandwidth_bytes_s=fleet.fabric.bandwidth / fleet.fabric.oversubscription_ratio,
                latency_s=fleet.fabric.latency or LATENCY_INFINIBAND
            )
        else:
            t_comm_ep = Q_("0 ms")

        # 4. Pipeline Parallelism (PP) Bubble
        # Source: Narayanan et al. (2019), "PipePipe: Efficient Pipeline Parallelism"
        # Supports interleaved 1F1B schedules via v_stages
        bubble_fraction = calc_pipeline_bubble(pp_size, microbatch_count, v_stages=v_stages)
        t_bubble = (node_perf.latency * bubble_fraction) if pp_size > 1 else Q_("0 ms")

        # 5. Total Latency and Scaling Efficiency
        total_comm_latency = t_comm_dp + t_comm_tp + t_comm_ep
        step_latency_total = node_perf.latency + total_comm_latency + t_bubble
        
        scaling_efficiency = (node_perf.latency / step_latency_total).magnitude
        
        return DistributedResult(
            node_performance=node_perf,
            dp_communication_latency=t_comm_dp,
            tp_communication_latency=t_comm_tp,
            ep_communication_latency=t_comm_ep,
            communication_latency=total_comm_latency,
            pipeline_bubble_latency=t_bubble,
            bubble_fraction=bubble_fraction,
            step_latency_total=step_latency_total,
            scaling_efficiency=scaling_efficiency,
            effective_throughput=(n_accelerators * node_perf.throughput * scaling_efficiency),
            parallelism={"dp": dp_size, "tp": tp_size, "pp": pp_size, "ep": ep_size},
        )

class ReliabilitySolver(BaseSolver):
    """
    Calculates Mean Time Between Failures (MTBF) and optimal checkpointing intervals.
    
    This solver handles the reliability modeling of massive clusters, helping
    determine the 'Goodput' of long-running training jobs. It identifies 
    the probability of a job failure before completion and calculates the 
    Young-Daly optimal interval to minimize wasted compute time.

    Literature Source:
    1. Young (1974), "A First-Order Approximation to the Optimum Checkpoint 
       Interval."
    2. Daly (2006), "A Higher Order Estimate of the Optimum Checkpoint 
       Interval for Restart-Dump Strategy."
    """
    requires = ("fleet",)
    produces = ReliabilityResult
    walls = (11,)

    def solve(self, fleet: Fleet, job_duration_hours: float, checkpoint_time_s: float = 60.0) -> ReliabilityResult:
        """
        Calculates reliability and checkpointing metrics for a fleet.
        """
        # Use compound node MTBF accounting for GPUs, NICs, and PSUs
        node_mtbf = calc_mtbf_node(
            gpu_mtbf_h=GPU_MTTF_HOURS, n_gpus=fleet.node.accelerators_per_node,
            nic_mtbf_h=NIC_MTTF_HOURS, n_nics=fleet.node.nics_per_node,
            psu_mtbf_h=PSU_MTTF_HOURS, n_psus=fleet.node.psus_per_node,
        )
        fleet_mtbf = calc_mtbf_cluster(node_mtbf, fleet.count)
        
        job_dur_q = Q_(job_duration_hours, "hour")
        prob_fail = calc_failure_probability(fleet_mtbf, job_dur_q)
        
        ckpt_time_q = Q_(checkpoint_time_s, "second")
        optimal_interval = calc_young_daly_interval(ckpt_time_q, fleet_mtbf.to("second"))
        
        return ReliabilityResult(
            fleet_mtbf=fleet_mtbf,
            failure_probability=prob_fail,
            optimal_checkpoint_interval=optimal_interval,
            expected_failures=(job_dur_q / fleet_mtbf).magnitude,
        )

class SustainabilitySolver(BaseSolver):
    """
    Calculates Datacenter-scale Sustainability metrics.
    
    Handles Power Usage Effectiveness (PUE), Carbon Intensity, 
    and Water Usage Effectiveness (WUE) across different regional grids.
    This solver models the 'Infrastructure Tax' — the energy spent on 
    cooling and power delivery rather than on neural computation.

    Literature Source:
    1. Patterson et al. (2021), "Carbon Emissions and Large Neural Network 
       Training."
    2. Belkhir & Elmeligi (2018), "Assessing ICT Global Emissions Footprint."
    3. Wu et al. (2022), "Sustainable AI: Environmental Implications, 
       Challenges and Opportunities."
    """
    requires = ("fleet",)
    produces = SustainabilityResult
    walls = (14,)

    def solve(self, fleet: Fleet, duration_days: float, datacenter: Optional[Datacenter] = None) -> SustainabilityResult:
        """
        Calculates energy, carbon, and water footprint for a fleet operation.
        """
        # 1. Resolve Environment
        dc = datacenter or fleet.datacenter
        
        # Flexibly handle if dc is already a GridProfile or a Datacenter
        if hasattr(dc, 'grid'):
            region = dc.grid
        else:
            region = dc or fleet.region
        
        if not region:
             from ..infra.registry import Grids
             region = Grids.US_Avg
        
        duration_hours = duration_days * 24
        
        # 2. Power
        it_power_w = fleet.node.accelerator.tdp * fleet.total_accelerators if fleet.node.accelerator.tdp else H100_TDP * fleet.total_accelerators
            
        # 3. Energy Consumption
        it_energy_kwh = (it_power_w * Q_(duration_hours, "hour")).to("kWh")
        
        # Apply PUE
        pue = getattr(dc, 'pue', fleet.effective_pue)
        total_energy_kwh = it_energy_kwh * pue
        
        # 4. Carbon Footprint (use total facility energy, PUE already applied)
        carbon_kg = region.carbon_kg(total_energy_kwh.magnitude) if hasattr(region, 'carbon_kg') else total_energy_kwh.magnitude * (region.carbon_intensity_g_kwh / 1000.0)
        
        # 5. Water Usage
        # Resolve WUE from dc.grid, dc, or region
        if hasattr(dc, 'grid') and dc.grid:
            wue = dc.grid.wue
        elif hasattr(dc, 'wue'):
            wue = dc.wue
        else:
            wue = region.wue
            
        water_liters = total_energy_kwh.magnitude * wue
        
        return SustainabilityResult(
            it_energy_kwh=it_energy_kwh,
            total_energy_kwh=total_energy_kwh,
            carbon_footprint_kg=carbon_kg,
            water_usage_liters=water_liters,
            pue=pue,
            region_name=region.name,
        )

class ServingSolver(BaseSolver):
    """
    Analyzes the two-phase LLM serving lifecycle: Pre-fill vs. Decoding.
    
    LLM inference is not a single mathematical operation; it is a stateful 
    process with two distinct physical regimes (Compute-bound Pre-fill and 
    Memory-bound Decoding).

    Literature Source:
    1. Pope et al. (2023), "LLM.int8(): 8-bit Matrix Multiplication for 
       Transformers at Scale" (Inference Bottlenecks)
    2. Aminabadi et al. (2022), "DeepSpeed-Inference: Enabling Efficient 
       Inference of Transformer Models at Unprecedented Scale."
    3. Yu et al. (2022), "ORCA: A Distributed Serving System for 
       Transformer-Based Generative Models."
    """
    requires = ("workload", "hardware")
    produces = ServingResult
    walls = (2,)

    def solve(self, model: TransformerWorkload, hardware: HardwareNode, seq_len: int, batch_size: int = 1, precision: str = "fp16", efficiency: float = 0.5) -> ServingResult:
        """
        Solves for LLM serving performance.
        """
        prec_map = PRECISION_MAP
        bpp = prec_map.get(precision, BYTES_FP16)
        peak_flops = hardware.compute.precision_flops.get(precision, hardware.compute.peak_flops)
        
        prefill_ops = 2 * model.parameters.to(ureg.count).magnitude * seq_len * batch_size * ureg.flop
        t_prefill = (prefill_ops / (peak_flops * efficiency)).to("ms") + hardware.dispatch_tax
        
        model_weights_bytes = model.size_in_bytes(bpp)
        kv_cache_bytes = model.get_kv_cache_size(seq_len=seq_len, batch_size=batch_size, precision=bpp)
        
        t_decode_per_token = ((model_weights_bytes + kv_cache_bytes) / hardware.memory.bandwidth).to("ms")
        
        total_memory_required = model_weights_bytes + kv_cache_bytes
        feasible = total_memory_required <= hardware.memory.capacity
        
        return ServingResult(
            feasible=feasible,
            ttft=t_prefill,
            itl=t_decode_per_token,
            kv_cache_size=kv_cache_bytes.to("GB"),
            model_weights_size=model_weights_bytes.to("GB"),
            total_memory_required=total_memory_required.to("GB"),
            memory_utilization=(total_memory_required / hardware.memory.capacity).to_base_units().magnitude,
        )

class EconomicsSolver(BaseSolver):
    """
    Calculates Total Cost of Ownership (TCO) including Capex and Opex.
    
    Combines hardware costs, energy consumption, and maintenance 
    into a single financial model for the fleet.

    Literature Source:
    1. Barroso et al. (2018), "The Datacenter as a Computer: An Introduction 
       to the Design of Warehouse-Scale Machines."
    2. Patterson (2004), "Latent Bugs in Common-Case Software." (TCO Foundations)
    3. Meta (2024), "Sustainable AI Infrastructure at Meta Scale."
    """
    requires = ("fleet",)
    produces = EconomicsResult
    walls = (13,)

    def solve(self, fleet: Fleet, duration_days: float, kwh_price: Optional[float] = None, datacenter: Optional[Any] = None, grid: Optional[Any] = None) -> EconomicsResult:
        """
        Calculates the TCO for a fleet over a specified duration.

        Parameters
        ----------
        fleet : Fleet
            The hardware cluster configuration.
        duration_days : float
            Operation duration in days.
        kwh_price : float, optional
            Price of electricity per kWh.
        datacenter : Datacenter, optional
            A specific datacenter profile.
        grid : GridProfile, optional
            A specific grid profile.

        Returns
        -------
        Dict[str, Any]
            Financial metrics including CapEx, OpEx, and total TCO.
        """
        sust_solver = SustainabilitySolver()
        energy_result = sust_solver.solve(fleet, duration_days, datacenter=datacenter or grid)
        
        price = kwh_price
        if price is None:
            # Try to resolve from grid/datacenter or default
            target = grid or datacenter or fleet.datacenter or fleet.region
            price = getattr(target, 'kwh_price', None)
            if price is None:
                price = CLOUD_ELECTRICITY_PER_KWH.magnitude  # 0.12 USD/kWh
            
        opex_energy = energy_result.total_energy_kwh.magnitude * price
        
        unit_cost = fleet.node.accelerator.unit_cost
        if unit_cost is None:
            unit_cost = GPU_UNIT_COST_H100
        total_capex = unit_cost.magnitude * fleet.total_accelerators

        annual_maintenance_ratio = ANNUAL_MAINTENANCE_RATIO
        opex_maintenance = total_capex * annual_maintenance_ratio * (duration_days / 365.0)
        
        # Compose economics + sustainability into single result
        return EconomicsResult(
            capex_usd=total_capex,
            opex_energy_usd=opex_energy,
            opex_maintenance_usd=opex_maintenance,
            total_opex_usd=opex_energy + opex_maintenance,
            tco_usd=total_capex + opex_energy + opex_maintenance,
            it_energy_kwh=energy_result.it_energy_kwh,
            total_energy_kwh=energy_result.total_energy_kwh,
            carbon_footprint_kg=energy_result.carbon_footprint_kg,
            water_usage_liters=energy_result.water_usage_liters,
            pue=energy_result.pue,
            region_name=energy_result.region_name,
        )

class DataSolver(BaseSolver):
    """
    Analyzes the 'Data Wall' — the throughput bottleneck between storage and compute.
    
    This solver models the data pipeline constraints, comparing the data demand 
    of a workload (e.g., training tokens or high-resolution video frames) 
    against the physical bandwidth of the storage hierarchy and IO interconnects.

    Literature Source:
    1. Janapa Reddi et al. (2025), "Machine Learning Systems," Chapter 4 (Data Engineering).
    2. Beitzel et al. (2024), "The Data Wall: Scaling Laws for Data Ingestion in AI."
    3. Mohan et al. (2022), "Analyzing and Mitigating Data Bottlenecks in Deep Learning Training."
    """
    requires = ("workload", "hardware")
    produces = DataResult
    walls = (4,)

    def solve(self, workload_data_rate: Quantity, hardware: HardwareNode) -> DataResult:
        """
        Solves for data pipeline feasibility.

        Parameters
        ----------
        workload_data_rate : Quantity
            The required data ingestion rate (e.g., TB/hour or GB/s).
        hardware : HardwareNode
            The hardware node with storage and interconnect specs.

        Returns
        -------
        Dict[str, Any]
            Pipeline metrics including utilization and stall probability.
        """
        # 1. Resolve Hardware Supply
        storage_bw = getattr(hardware.storage, 'bandwidth', Q_("0 GB/s")) if hardware.storage else Q_("0 GB/s")
        io_bw = getattr(hardware.interconnect, 'bandwidth', Q_("0 GB/s")) if hardware.interconnect else Q_("0 GB/s")
        
        # The pipeline is limited by the minimum of storage and interconnect BW
        supply_bw = min(storage_bw.to("GB/s"), io_bw.to("GB/s"))
        demand_bw = workload_data_rate.to("GB/s")
        
        utilization = (demand_bw / supply_bw).magnitude if supply_bw.magnitude > 0 else float('inf')
        is_stalled = utilization > 1.0
        
        return DataResult(
            is_stalled=is_stalled,
            utilization=utilization,
            demand_bw=demand_bw,
            supply_bw=supply_bw,
            bottleneck="Storage" if (storage_bw < io_bw and storage_bw.magnitude > 0) else "Interconnect",
            margin=(supply_bw - demand_bw).to("GB/s"),
        )

class ScalingSolver(BaseSolver):
    """
    Analyzes the 'Scaling Physics' of model training (Chinchilla Laws).
    
    This solver determines the optimal model size (P) and dataset size (D) 
    given a compute budget (C), following the compute-optimal training 
    regime where D ≈ 20P.

    Literature Source:
    1. Hoffmann et al. (2022), "Training Compute-Optimal Large Language Models."
    2. Kaplan et al. (2020), "Scaling Laws for Neural Language Models."
    3. McCandlish et al. (2018), "An Empirical Model of Large-Batch Training."
    """
    requires = ("compute_budget",)
    produces = ScalingResult
    walls = (7,)

    def solve(self, compute_budget: Quantity, target_model_size: Optional[Quantity] = None) -> ScalingResult:
        """
        Solves for compute-optimal model and dataset parameters.

        Parameters
        ----------
        compute_budget : Quantity
            Total training budget (e.g., in TFLOPs or H100-GPU-days).
        target_model_size : Quantity, optional
            If provided, calculates the required tokens for this specific model size.

        Returns
        -------
        Dict[str, Any]
            Optimal parameters, token count, and training duration estimates.
        """
        # C = 6 * P * D
        # Chinchilla: D = 20 * P
        # C = 120 * P^2  => P = sqrt(C / 120)
        
        # Convert H100-days to FLOPs if necessary (simplified approximation)
        c_flops = compute_budget
        if compute_budget.units == ureg.day:
            # Convert GPU-days to FLOPs using H100 SXM reference
            # Source: NVIDIA H100 datasheet (989 TFLOPs FP16 dense)
            c_flops = (compute_budget * H100_FLOPS_FP16_TENSOR * REFERENCE_MFU_SUSTAINED).to(ureg.flop)

        if target_model_size:
            p_opt = target_model_size.to(ureg.count).magnitude
            d_opt = (c_flops.magnitude / (CHINCHILLA_COMPUTE_CONSTANT * p_opt))
        else:
            p_opt = math.sqrt(c_flops.magnitude / (CHINCHILLA_COMPUTE_CONSTANT * CHINCHILLA_TOKENS_PER_PARAM))
            d_opt = CHINCHILLA_TOKENS_PER_PARAM * p_opt

        return ScalingResult(
            optimal_parameters=Q_(p_opt, ureg.count),
            optimal_tokens=Q_(d_opt, ureg.count),
            compute_budget_flops=c_flops,
            tokens_per_parameter=d_opt / p_opt if p_opt > 0 else 0,
        )

class OrchestrationSolver(BaseSolver):
    """
    Analyzes Cluster Orchestration and Queueing (Little's Law).
    
    This solver models the 'Wait Wall' in shared research clusters, 
    calculating job completion times and researcher wait times based on 
    cluster utilization and arrival rates.

    Literature Source:
    1. Little (1961), "A Proof for the Queuing Formula: L = λW."
    2. Barroso et al. (2018), "The Datacenter as a Computer" (Cluster Mgmt).
    3. Jeon et al. (2019), "Analysis of Large-Scale Multi-Tenant GPU Clusters."
    """
    requires = ("fleet",)
    produces = OrchestrationResult
    walls = (12,)

    def solve(self, fleet: Fleet, arrival_rate_jobs_per_day: float, avg_job_duration_days: float) -> OrchestrationResult:
        """
        Solves for cluster wait times and utilization.

        Parameters
        ----------
        fleet : Fleet
            The hardware cluster configuration.
        arrival_rate_jobs_per_day : float
            λ: Rate at which new training jobs are submitted.
        avg_job_duration_days : float
            The average time a job takes to run if it has the whole cluster.

        Returns
        -------
        Dict[str, Any]
            Wait time, system length, and utilization metrics.
        """
        # ρ = λ / μ  (Utilization)
        # μ = 1 / avg_duration
        
        lambda_rate = arrival_rate_jobs_per_day
        mu_rate = 1.0 / avg_job_duration_days
        
        utilization = lambda_rate / mu_rate
        
        # M/D/1 Queue approximation for wait time (Fixed duration jobs)
        # T_wait = ρ / (2μ(1-ρ))
        if utilization < 1.0:
            wait_time_days = utilization / (2 * mu_rate * (1 - utilization))
        else:
            wait_time_days = float('inf')
            
        return OrchestrationResult(
            cluster_utilization=utilization,
            avg_wait_time_days=Q_(wait_time_days, ureg.day),
            avg_queue_length=utilization**2 / (2 * (1 - utilization)) if utilization < 1.0 else float('inf'),
            is_stable=utilization < 1.0,
        )

class CompressionSolver(BaseSolver):
    """
    Analyzes model compression trade-offs (Accuracy vs. Efficiency).
    
    This solver models the 'Compression Tax' — the accuracy degradation 
    that occurs when reducing model size via quantization or pruning, 
    balanced against the gains in memory footprint and inference latency.

    Literature Source:
    1. Han et al. (2015), "Deep Compression: Compressing Deep Neural Networks 
       with Pruning, Trained Quantization and Huffman Coding."
    2. Gholami et al. (2021), "A Survey of Quantization Methods for 
       Efficient Neural Network Inference."
    3. Blalock et al. (2020), "What is the State of Neural Network Pruning?"
    """
    requires = ("workload", "hardware")
    produces = CompressionResult
    walls = (9,)

    def solve(self, model: Workload, hardware: HardwareNode, method: str = "quantization", target_bitwidth: int = 8, sparsity: float = 0.0) -> CompressionResult:
        """
        Solves for compression gains and estimated accuracy impact.

        Parameters
        ----------
        model : Workload
            The model to be compressed.
        hardware : HardwareNode
            The target execution hardware.
        method : str
            The compression method ('quantization', 'pruning', 'distillation').
        target_bitwidth : int
            Target numerical precision in bits (e.g., 8 for INT8, 4 for INT4).
        sparsity : float
            Target sparsity ratio (0.0 to 1.0) for pruning.

        Returns
        -------
        Dict[str, Any]
            Compression metrics including memory savings, latency speedup, 
            and estimated accuracy delta.
        """
        original_size = model.size_in_bytes(Q_("4 byte")) # FP32 baseline
        
        if method == "quantization":
            compression_ratio = 32 / target_bitwidth
            # Source: Gholami et al. (2021), "A Survey of Quantization Methods"
            # Conservative estimates: <1% drop for INT8, 2-5% for INT4
            if target_bitwidth >= 8:
                accuracy_delta = QUANT_ACCURACY_DELTA_INT8
            elif target_bitwidth >= 4:
                accuracy_delta = QUANT_ACCURACY_DELTA_INT4
            else:
                accuracy_delta = -0.05   # Sub-INT4: significant degradation
        elif method == "pruning":
            compression_ratio = 1.0 / (1.0 - sparsity) if sparsity < 1.0 else 100.0
            # Source: Blalock et al. (2020), "What is the State of Neural Network Pruning?"
            # Log-linear degradation accelerates after 50% sparsity
            if sparsity <= PRUNING_ACCURACY_THRESHOLD:
                accuracy_delta = PRUNING_MILD_DELTA
            else:
                accuracy_delta = -PRUNING_STEEP_COEFFICIENT * math.exp(sparsity * PRUNING_STEEP_EXPONENT)
        else:
            compression_ratio = 1.0
            accuracy_delta = 0.0

        compressed_size = original_size / compression_ratio
        
        return CompressionResult(
            original_size_gb=original_size.to("GB"),
            compressed_size_gb=compressed_size.to("GB"),
            compression_ratio=compression_ratio,
            estimated_accuracy_delta=accuracy_delta,
            memory_savings_pct=(1.0 - 1.0/compression_ratio) * 100,
        )

class EfficiencySolver(BaseSolver):
    """
    Models the gap between peak and achieved FLOPS (Wall 3: Software Efficiency).

    This solver quantifies the software efficiency of a workload — the fraction
    of peak hardware FLOPS that the software stack actually converts into useful
    computation. It decomposes Model FLOPs Utilization (MFU) by workload type,
    accounting for kernel fusion efficiency, SM occupancy, and memory access
    patterns.

    Literature Source:
    1. Chowdhery et al. (2022), "PaLM: Scaling Language Modeling with Pathways."
       (First systematic MFU reporting for large Transformers.)
    2. Dao et al. (2022), "FlashAttention: Fast and Memory-Efficient Exact
       Attention with IO-Awareness." (FlashAttention MFU improvement.)
    3. NVIDIA (2023), "Hopper Architecture Tuning Guide." (SM Occupancy model.)
    """
    requires = ("workload", "hardware")
    produces = EfficiencyResult
    walls = (3,)

    def solve(self, model: Workload, hardware: HardwareNode,
              workload_type: str = "ffn", use_flash_attention: bool = False,
              precision: str = "fp16", efficiency: float = 0.5) -> EfficiencyResult:
        """
        Estimates achievable MFU and FLOPS for a given workload type.

        Parameters
        ----------
        model : Workload
            The model architecture to simulate.
        hardware : HardwareNode
            The target hardware node.
        workload_type : str
            The dominant kernel type ('attention', 'ffn', 'conv').
        use_flash_attention : bool
            Whether FlashAttention is enabled (only applies to 'attention').
        precision : str
            Numerical precision ('fp16', 'fp32', 'int8', 'int4').
        efficiency : float
            Base compute efficiency factor (0.0 to 1.0).

        Returns
        -------
        Dict[str, Any]
            MFU estimate, achievable FLOPS, and overhead breakdown.
        """
        peak_flops = hardware.compute.precision_flops.get(precision, hardware.compute.peak_flops)

        # Base MFU range from defaults (training regime)
        mfu_low = MFU_TRAINING_LOW
        mfu_high = MFU_TRAINING_HIGH

        # Workload-type MFU adjustment (heuristic calibrations — see defaults.py)
        # All values scale linearly with the efficiency parameter relative to 0.5.
        scale = efficiency / 0.5
        if workload_type == "attention":
            if use_flash_attention:
                eta = min(MFU_FLASH_ATTENTION * scale, MFU_FLASH_ATTENTION_CAP)
            else:
                # Standard attention is memory-bound due to O(S²) reads
                eta = mfu_low * scale
        elif workload_type == "ffn":
            # FFN layers are compute-dense GEMM — best MFU
            eta = min(mfu_high * scale, MFU_FFN_CAP)
        elif workload_type == "conv":
            # Convolutions via im2col + GEMM — moderate MFU
            eta = min((mfu_low + mfu_high) / 2.0 * scale, MFU_CONV_CAP)
        else:
            eta = mfu_low * scale

        eta = max(0.0, min(eta, 1.0))  # Clamp to [0, 1]

        achievable_flops = peak_flops * eta

        # Overhead breakdown
        occupancy_loss = 1.0 - min(efficiency * HFU_MFU_RATIO, 1.0)  # SM occupancy overhead
        memory_stall = max(0.0, 1.0 - eta - occupancy_loss) if (1.0 - eta - occupancy_loss) > 0 else 0.0
        kernel_overhead = max(0.0, 1.0 - eta - occupancy_loss - memory_stall)

        return EfficiencyResult(
            mfu=eta,
            achievable_flops=achievable_flops,
            peak_flops=peak_flops,
            workload_type=workload_type,
            use_flash_attention=use_flash_attention,
            overhead_breakdown={
                "occupancy_loss": occupancy_loss,
                "memory_stall": memory_stall,
                "kernel_overhead": kernel_overhead,
            },
        )

class TransformationSolver(BaseSolver):
    """
    Quantifies the CPU preprocessing bottleneck (Wall 5: CPU Preprocessing).

    This solver models the 'Transformation Wall' — the gap between CPU-bound
    data preprocessing (JPEG decode, tokenization, augmentation) and
    accelerator step time. When preprocessing cannot keep up, the accelerator
    starves and utilization drops.

    Literature Source:
    1. Mohan et al. (2022), "Analyzing and Mitigating Data Bottlenecks in
       Deep Learning Training."
    2. Murray et al. (2021), "tf.data: A Machine Learning Data Processing
       Framework." (Pipeline stall analysis.)
    3. NVIDIA DALI Documentation (2024). (GPU-accelerated preprocessing.)
    """
    requires = ("hardware",)
    produces = TransformationResult
    walls = (5,)

    def solve(self, batch_size: int, sample_size_bytes: Quantity,
              cpu_throughput: Quantity, accelerator_step_time: Quantity) -> TransformationResult:
        """
        Solves for CPU preprocessing bottleneck.

        Parameters
        ----------
        batch_size : int
            Number of samples per batch.
        sample_size_bytes : Quantity
            Size of one sample in bytes (e.g., Q_("500 KB")).
        cpu_throughput : Quantity
            CPU preprocessing throughput (e.g., Q_("2 GB/s")).
        accelerator_step_time : Quantity
            Time for one accelerator training step (e.g., Q_("50 ms")).

        Returns
        -------
        Dict[str, Any]
            Transform time, bottleneck status, and accelerator utilization.
        """
        # T_transform = (B × S_sample) / C_throughput
        batch_data = batch_size * sample_size_bytes.to("byte")
        transform_time = (batch_data / cpu_throughput.to("byte/s")).to("ms")

        accel_time = accelerator_step_time.to("ms")
        is_bottleneck = transform_time.magnitude > accel_time.magnitude

        # Accelerator utilization: fraction of time the accelerator is active
        total_step_time = max(transform_time.magnitude, accel_time.magnitude)
        accelerator_utilization = accel_time.magnitude / total_step_time if total_step_time > 0 else 0.0

        return TransformationResult(
            transform_time=transform_time,
            accelerator_step_time=accel_time,
            is_bottleneck=is_bottleneck,
            accelerator_utilization=accelerator_utilization,
            slowdown_factor=total_step_time / accel_time.magnitude if accel_time.magnitude > 0 else float('inf'),
        )

class TopologySolver(BaseSolver):
    """
    Models bisection bandwidth for different network topologies (Wall 6).

    This solver calculates the effective bandwidth available to collective
    communication operations based on the physical network topology. Different
    topologies trade cost against bisection bandwidth — the minimum bandwidth
    across any cut that divides the network in half.

    Literature Source:
    1. Leiserson (1985), "Fat-Trees: Universal Networks for Hardware-Efficient
       Supercomputing." (Fat-tree bisection bandwidth.)
    2. Kim et al. (2008), "Technology-Driven, Highly-Scalable Dragonfly Topology."
       (Dragonfly topology model.)
    3. Dally & Towles (2003), "Principles and Practices of Interconnection
       Networks." (Torus and ring analysis.)
    """
    requires = ("fabric",)
    produces = TopologyResult
    walls = (6,)

    # Bisection bandwidth fractions (β) relative to full fat-tree.
    # Ring and torus_3d are computed dynamically from num_nodes in solve().
    TOPOLOGY_BETA = {
        "fat_tree": 1.0,
        "dragonfly": 0.85,
    }

    # Average hop counts (normalized to diameter)
    TOPOLOGY_HOPS = {
        "fat_tree": 4,       # 2 up + 2 down in 3-level fat-tree
        "dragonfly": 3,      # Local + global + local
        "torus_3d": None,    # Computed from num_nodes
        "ring": None,         # Computed from num_nodes
    }

    def solve(self, fabric: NetworkFabric, topology: str = "fat_tree",
              num_nodes: int = 64) -> TopologyResult:
        """
        Solves for effective network bandwidth under a given topology.

        Parameters
        ----------
        fabric : NetworkFabric
            The network fabric specification (link bandwidth, oversubscription).
        topology : str
            Network topology ('fat_tree', 'dragonfly', 'torus_3d', 'ring').
        num_nodes : int
            Number of nodes in the network.

        Returns
        -------
        Dict[str, Any]
            Effective bandwidth, bisection fraction, and average hops.
        """
        # Compute bisection bandwidth fraction β.
        # Ring: bisection cuts 2 links out of N, so β = 2/N.
        # 3D torus: bisection cuts 2*N^(2/3) links, so β ≈ 2*N^(-1/3).
        if topology == "ring":
            beta = 2.0 / max(num_nodes, 1)
        elif topology == "torus_3d":
            beta = 2.0 * (num_nodes ** (-1.0 / 3.0))
        else:
            beta = self.TOPOLOGY_BETA.get(topology, 1.0)

        link_bw = fabric.bandwidth
        oversubscription = fabric.oversubscription_ratio

        effective_bw = (link_bw * beta / oversubscription).to("GB/s")
        bisection_bw = (link_bw * beta * num_nodes / 2 / oversubscription).to("GB/s")

        # Average hops
        fixed_hops = self.TOPOLOGY_HOPS.get(topology)
        if fixed_hops is not None:
            hops_avg = fixed_hops
        elif topology == "torus_3d":
            # 3D torus: avg hops ≈ 3/4 × (N^(1/3))
            hops_avg = 0.75 * (num_nodes ** (1.0 / 3.0))
        elif topology == "ring":
            # Ring: avg hops ≈ N/4
            hops_avg = num_nodes / 4.0
        else:
            hops_avg = 4  # Default to fat-tree

        return TopologyResult(
            effective_bw=effective_bw,
            bisection_bw=bisection_bw,
            bisection_bw_fraction=beta,
            hops_avg=hops_avg,
            topology=topology,
            num_nodes=num_nodes,
        )

class InferenceScalingSolver(BaseSolver):
    """
    Models inference-time compute scaling (Wall 8: Reasoning/CoT Cost).

    This solver quantifies the cost of 'System-2 thinking' — inference-time
    compute scaling via chain-of-thought (CoT) reasoning, where the model
    generates K intermediate reasoning steps before producing the final answer.
    Each step incurs the full cost of autoregressive decoding.

    Literature Source:
    1. Wei et al. (2022), "Chain-of-Thought Prompting Elicits Reasoning in
       Large Language Models."
    2. Snell et al. (2024), "Scaling LLM Test-Time Compute Optimally Can Be
       More Effective Than Scaling Model Parameters."
    3. OpenAI (2024), "Learning to Reason with LLMs." (o1 reasoning model.)
    """
    requires = ("workload", "hardware")
    produces = InferenceScalingResult
    walls = (8,)

    def solve(self, model: TransformerWorkload, hardware: HardwareNode,
              reasoning_steps: int = 8, context_length: int = 2048,
              precision: str = "fp16", efficiency: float = 0.5) -> InferenceScalingResult:
        """
        Solves for inference-time reasoning cost.

        Parameters
        ----------
        model : TransformerWorkload
            The language model used for reasoning.
        hardware : HardwareNode
            The target hardware node.
        reasoning_steps : int
            Number of reasoning steps K (each generates tokens).
        context_length : int
            Input context length in tokens.
        precision : str
            Numerical precision.
        efficiency : float
            Compute efficiency factor (0.0 to 1.0).

        Returns
        -------
        Dict[str, Any]
            Total reasoning time, cost per query, and token counts.
        """
        # Use ServingSolver internally to get per-step latency
        serving = ServingSolver()
        serving_result = serving.solve(
            model=model, hardware=hardware, seq_len=context_length,
            batch_size=1, precision=precision, efficiency=efficiency
        )

        # Inter-token latency (ITL) is the cost per generated token
        itl = serving_result.itl

        # Average tokens per reasoning step (heuristic — see defaults.py)
        tokens_per_step = TOKENS_PER_REASONING_STEP
        total_tokens = reasoning_steps * tokens_per_step

        # T_reason = K × tokens_per_step × ITL + TTFT (initial prefill)
        ttft = serving_result.ttft
        t_step = tokens_per_step * itl
        total_reasoning_time = ttft + reasoning_steps * t_step

        # Cost estimate (based on hardware TDP and time)
        if hardware.tdp is not None:
            energy_j = (hardware.tdp * total_reasoning_time.to("s")).to("J")
        else:
            energy_j = Q_("0 J")

        return InferenceScalingResult(
            total_reasoning_time=total_reasoning_time.to("ms"),
            ttft=ttft,
            itl=itl,
            tokens_generated=total_tokens,
            reasoning_steps=reasoning_steps,
            time_per_step=t_step.to("ms"),
            energy_per_query=energy_j,
            feasible=serving_result.feasible,
            serving_detail=serving_result,
        )

class SensitivitySolver(BaseSolver):
    """
    Identifies the binding constraint via numerical sensitivity analysis (Wall 16).

    This solver computes numerical partial derivatives of inference latency
    with respect to each hardware parameter (peak FLOPS, memory bandwidth,
    memory capacity). The parameter with the largest absolute sensitivity
    is the 'binding constraint' — the one most worth investing in.

    Literature Source:
    1. Williams et al. (2009), "Roofline: An Insightful Visual Performance
       Model for Floating-Point Programs and Multicore Architectures."
    2. Ofenbeck et al. (2014), "Applying the Roofline Model." (Sensitivity
       analysis of Roofline parameters.)
    """
    requires = ("workload", "hardware")
    produces = SensitivityResult
    walls = (16,)

    def solve(self, model: Workload, hardware: HardwareNode,
              precision: str = "fp16", perturbation_pct: float = 10.0,
              efficiency: float = 0.5) -> SensitivityResult:
        """
        Computes sensitivities and identifies the binding constraint.

        Parameters
        ----------
        model : Workload
            The model architecture to simulate.
        hardware : HardwareNode
            The target hardware node.
        precision : str
            Numerical precision.
        perturbation_pct : float
            Percentage by which to perturb each parameter (default 10%).
        efficiency : float
            Compute efficiency factor (0.0 to 1.0).

        Returns
        -------
        Dict[str, Any]
            Sensitivity values for each parameter and the binding constraint.
        """
        from copy import deepcopy
        from ..hardware.types import HardwareNode, ComputeCore, MemoryHierarchy

        # Baseline latency
        baseline = Engine.solve(model, hardware, precision=precision, efficiency=efficiency)
        t_base = baseline.latency.to("ms").magnitude

        factor = 1.0 + perturbation_pct / 100.0
        sensitivities = {}

        # 1. Perturb peak_flops
        hw_flops = deepcopy(hardware)
        hw_flops.compute = ComputeCore(
            peak_flops=hardware.compute.peak_flops * factor,
            precision_flops={k: v * factor for k, v in hardware.compute.precision_flops.items()}
        )
        t_flops = Engine.solve(model, hw_flops, precision=precision, efficiency=efficiency).latency.to("ms").magnitude
        sensitivities["peak_flops"] = (t_flops - t_base) / t_base if t_base > 0 else 0.0

        # 2. Perturb memory_bandwidth
        hw_bw = deepcopy(hardware)
        hw_bw.memory = MemoryHierarchy(
            capacity=hardware.memory.capacity,
            bandwidth=hardware.memory.bandwidth * factor
        )
        t_bw = Engine.solve(model, hw_bw, precision=precision, efficiency=efficiency).latency.to("ms").magnitude
        sensitivities["memory_bandwidth"] = (t_bw - t_base) / t_base if t_base > 0 else 0.0

        # 3. Perturb memory_capacity (affects feasibility, not latency directly)
        hw_mem = deepcopy(hardware)
        hw_mem.memory = MemoryHierarchy(
            capacity=hardware.memory.capacity * factor,
            bandwidth=hardware.memory.bandwidth
        )
        t_mem = Engine.solve(model, hw_mem, precision=precision, efficiency=efficiency).latency.to("ms").magnitude
        sensitivities["memory_capacity"] = (t_mem - t_base) / t_base if t_base > 0 else 0.0

        # Binding constraint = parameter with largest |sensitivity|
        binding = max(sensitivities, key=lambda k: abs(sensitivities[k]))

        return SensitivityResult(
            sensitivities=sensitivities,
            binding_constraint=binding,
            baseline_latency=baseline.latency,
            perturbation_pct=perturbation_pct,
        )

class SynthesisSolver(BaseSolver):
    """
    Given an SLA, synthesizes the required hardware specs (Wall 17: Inverse Solve).

    This solver inverts the Roofline model: instead of predicting latency from
    hardware specs, it starts from a target latency SLA and works backward to
    determine the minimum hardware specifications (bandwidth, FLOPS, memory)
    required to meet the target.

    Literature Source:
    1. Williams et al. (2009), "Roofline: An Insightful Visual Performance
       Model for Floating-Point Programs and Multicore Architectures."
       (Inverse application of the Roofline model.)
    2. Jouppi et al. (2017), "In-Datacenter Performance Analysis of a Tensor
       Processing Unit." (Hardware specification from workload requirements.)
    """
    requires = ("workload", "target_latency")
    produces = SynthesisResult
    walls = (17,)

    def solve(self, model: Workload, target_latency: Quantity,
              precision: str = "fp16", efficiency: float = 0.5) -> SynthesisResult:
        """
        Synthesizes hardware requirements from an SLA target.

        Parameters
        ----------
        model : Workload
            The model to be served.
        target_latency : Quantity
            Target latency SLA (e.g., Q_("10 ms")).
        precision : str
            Numerical precision.
        efficiency : float
            Expected compute efficiency η (0.0 to 1.0).

        Returns
        -------
        Dict[str, Any]
            Required bandwidth, FLOPS, memory, and compute-memory ratio.
        """
        prec_map = PRECISION_MAP
        bpp = prec_map.get(precision, BYTES_FP16)

        # Model weight size
        weight_bytes = model.size_in_bytes(bpp)

        # Model operations (inference)
        graph = model.lower(bpp)
        total_ops = graph.total_ops

        t_target = target_latency.to("s")

        # BW_required = |W| / T_target (must stream all weights within latency)
        required_bw = (weight_bytes / t_target).to("GB/s")

        # FLOPS_required = OPs / (T_target × η)
        required_flops = (total_ops / (t_target * efficiency)).to("TFLOP/s")

        # Memory must hold at least the model weights
        required_memory = weight_bytes.to("GB")

        # Compute-to-memory ratio (operational intensity requirement)
        compute_memory_ratio = (required_flops / required_bw).to("flop/byte")

        return SynthesisResult(
            required_bw=required_bw,
            required_flops=required_flops,
            required_memory=required_memory,
            compute_memory_ratio=compute_memory_ratio,
            target_latency=target_latency,
            model_size=weight_bytes.to("GB"),
            total_ops=total_ops,
        )

class ResponsibleEngineeringSolver(BaseSolver):
    """
    Models the computational cost of responsible AI practices (Wall 15).

    This solver quantifies the 'Ethics Tax' — the additional compute, data,
    and time required when training with differential privacy (DP-SGD) or
    fairness constraints. These are not optional overheads but engineering
    requirements that must be budgeted into system design.

    Literature Source:
    1. Abadi et al. (2016), "Deep Learning with Differential Privacy."
       (DP-SGD slowdown model: per-sample gradient clipping + noise.)
    2. Anil et al. (2022), "Large-Scale Differentially Private BERT."
       (Practical DP training at scale.)
    3. Chen et al. (2018), "My Fair Bandit: Distributed Learning of
       Max-Min Fairness with Multi-player Bandits."
       (Fairness constraint data requirements.)
    """
    requires = ("training_time",)
    produces = ResponsibleEngineeringResult
    walls = (15,)

    def solve(self, base_training_time: Quantity,
              epsilon: float = 1.0, delta: float = 1e-5,
              min_subgroup_prevalence: float = 0.01) -> ResponsibleEngineeringResult:
        """
        Solves for the overhead of responsible engineering practices.

        Parameters
        ----------
        base_training_time : Quantity
            Baseline training time without privacy/fairness overhead.
        epsilon : float
            Privacy budget ε (lower = more private, more expensive).
            Default: 1.0 (strong privacy).
        delta : float
            Privacy failure probability δ. Default: 1e-5.
        min_subgroup_prevalence : float
            Prevalence of the rarest subgroup p_min (0.0 to 1.0).
            Default: 0.01 (1% of population).

        Returns
        -------
        Dict[str, Any]
            DP slowdown factor, effective training time, and data requirements.
        """
        # DP-SGD slowdown: heuristic calibration (NOT from Abadi et al. 2016).
        # Calibrated to match reported slowdowns: ~3x at ε=1.0, ~1.2x at ε=10.0.
        dp_slowdown = 1.0 + (DP_SGD_SLOWDOWN_COEFFICIENT / max(epsilon, 0.01))

        # Fairness constraint: need enough samples from rarest subgroup
        # Additional data requirement ∝ 1/p_min
        # At p_min=0.01, need ~100× more data to get statistical significance
        additional_data_factor = 1.0 / max(min_subgroup_prevalence, 1e-6)

        # Combined effective training time
        effective_time = base_training_time * dp_slowdown

        return ResponsibleEngineeringResult(
            dp_slowdown_factor=dp_slowdown,
            effective_training_time=effective_time.to(base_training_time.units),
            additional_data_requirement=additional_data_factor,
            epsilon=epsilon,
            delta=delta,
            min_subgroup_prevalence=min_subgroup_prevalence,
            privacy_cost_ratio=dp_slowdown,
            fairness_data_ratio=additional_data_factor,
        )
