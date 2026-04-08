from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict
from .engine import PerformanceProfile, Engine
from .formulas import (
    calc_ring_allreduce_time, 
    calc_tree_allreduce_time,
    calc_hierarchical_allreduce_time,
    calc_all_to_all_time,
    calc_mtbf_cluster, 
    calc_young_daly_interval, 
    calc_failure_probability,
    calc_pipeline_bubble
)
from .constants import ureg, Q_
from ..models.types import Workload, TransformerWorkload
from ..hardware.types import HardwareNode
from ..systems.types import Fleet, NetworkFabric
from ..infra.types import Datacenter, GridProfile

class BaseSolver(ABC):
    @abstractmethod
    def solve(self, **kwargs) -> Any:
        pass

class SingleNodeSolver(BaseSolver):
    """
    Resolves single-node hardware Roofline bounds and feasibility.
    
    This solver handles the 'Iron Law' of machine learning systems,
    calculating whether a model fits in memory and predicting its
    throughput based on arithmetic intensity.

    Literature Source: Williams et al. (2009), "Roofline: An Insightful Visual 
    Performance Model for Floating-Point Programs and Multicore Architectures."
    """
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
              topology_override: Optional[str] = None) -> Dict[str, Any]:
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
        node_perf = Engine.solve(model, fleet.node.accelerator, batch_size=batch_size // dp_size, precision=precision, efficiency=efficiency)
        
        # 3. Communication Overhead (Network)
        # Apply Hierarchical Model: Intra-node (NVLink) vs Inter-node (InfiniBand)
        message_size = model.size_in_bytes()
        
        # DP AllReduce (Weights/Gradients)
        if dp_size > 1:
            if fleet.node.accelerators_per_node > 1 and dp_size > fleet.node.accelerators_per_node:
                # Hierarchical: Ring within node, then Ring across nodes
                t_comm_dp = calc_hierarchical_allreduce_time(
                    message_bytes=message_size,
                    n_nodes=dp_size // fleet.node.accelerators_per_node,
                    gpus_per_node=fleet.node.accelerators_per_node,
                    intra_node_bw=fleet.node.intra_node_bw,
                    inter_node_bw=fleet.fabric.bandwidth / fleet.fabric.oversubscription_ratio,
                    inter_node_lat=fleet.fabric.latency or Q_("5 us")
                )
            else:
                # Single node or small DP: Intra-node only
                t_comm_dp = calc_ring_allreduce_time(
                    message_size, 
                    dp_size, 
                    fleet.node.intra_node_bw, 
                    Q_("500 ns")
                )
        else:
            t_comm_dp = Q_("0 ms")

        # TP Communication (Assuming intra-node NVLink)
        t_comm_tp = (message_size / tp_size / fleet.node.intra_node_bw).to("ms") if tp_size > 1 else Q_("0 ms")

        # EP Communication (All-to-All token routing for MoE)
        if ep_size > 1:
            t_comm_ep = calc_all_to_all_time(
                message_bytes=message_size, 
                n_gpus=ep_size, 
                bandwidth_bytes_s=fleet.fabric.bandwidth / fleet.fabric.oversubscription_ratio, 
                latency_s=fleet.fabric.latency or Q_("5 us")
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
        
        return {
            "node_performance": node_perf,
            "dp_communication_latency": t_comm_dp,
            "tp_communication_latency": t_comm_tp,
            "ep_communication_latency": t_comm_ep,
            "communication_latency": total_comm_latency, # Backwards compatibility for tests
            "pipeline_bubble_latency": t_bubble,
            "bubble_fraction": bubble_fraction,
            "step_latency_total": step_latency_total,
            "scaling_efficiency": scaling_efficiency,
            "effective_throughput": (n_accelerators * node_perf.throughput * scaling_efficiency),
            "parallelism": {"dp": dp_size, "tp": tp_size, "pp": pp_size, "ep": ep_size}
        }

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
    def solve(self, fleet: Fleet, job_duration_hours: float, checkpoint_time_s: float = 60.0) -> Dict[str, Any]:
        """
        Calculates reliability and checkpointing metrics for a fleet.
        """
        accel_mtbf = Q_(50000, "hour")
        node_mtbf = accel_mtbf / fleet.node.accelerators_per_node
        fleet_mtbf = calc_mtbf_cluster(node_mtbf, fleet.count)
        
        job_dur_q = Q_(job_duration_hours, "hour")
        prob_fail = calc_failure_probability(fleet_mtbf, job_dur_q)
        
        ckpt_time_q = Q_(checkpoint_time_s, "second")
        optimal_interval = calc_young_daly_interval(ckpt_time_q, fleet_mtbf.to("second"))
        
        return {
            "fleet_mtbf": fleet_mtbf,
            "failure_probability": prob_fail,
            "optimal_checkpoint_interval": optimal_interval,
            "expected_failures": (job_dur_q / fleet_mtbf).magnitude
        }

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
    def solve(self, fleet: Fleet, duration_days: float, datacenter: Optional[Datacenter] = None) -> Dict[str, Any]:
        """
        Calculates energy, carbon, and water footprint for a fleet operation.
        """
        if duration_days < 0:
            raise ValueError("duration_days cannot be negative")
            
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
        it_power_w = fleet.node.accelerator.tdp * fleet.total_accelerators if fleet.node.accelerator.tdp else Q_("700 W") * fleet.total_accelerators
            
        # 3. Energy Consumption
        it_energy_kwh = (it_power_w * Q_(duration_hours, "hour")).to("kWh")
        
        # Apply PUE
        pue = getattr(dc, 'pue', fleet.effective_pue)
        total_energy_kwh = it_energy_kwh * pue
        
        # 4. Carbon Footprint
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
        
        return {
            "it_energy_kwh": it_energy_kwh,
            "total_energy_kwh": total_energy_kwh,
            "carbon_footprint_kg": carbon_kg,
            "water_usage_liters": water_liters,
            "pue": pue,
            "region_name": region.name
        }

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
    def solve(self, model: TransformerWorkload, hardware: HardwareNode, seq_len: int, batch_size: int = 1, precision: str = "fp16", efficiency: float = 0.5) -> Dict[str, Any]:
        """
        Solves for LLM serving performance.
        """
        from .constants import BYTES_FP16, BYTES_FP32, BYTES_INT8, BYTES_INT4
        
        prec_map = {"fp16": BYTES_FP16, "fp32": BYTES_FP32, "int8": BYTES_INT8, "int4": BYTES_INT4}
        bpp = prec_map.get(precision, BYTES_FP16)
        peak_flops = hardware.compute.precision_flops.get(precision, hardware.compute.peak_flops)
        
        prefill_ops = 2 * model.parameters.to(ureg.count).magnitude * seq_len * batch_size * ureg.flop
        t_prefill = (prefill_ops / (peak_flops * efficiency)).to("ms") + hardware.dispatch_tax
        
        model_weights_bytes = model.size_in_bytes(bpp)
        kv_cache_bytes = model.get_kv_cache_size(seq_len=seq_len, batch_size=batch_size, precision=bpp)
        
        t_decode_per_token = ((model_weights_bytes + kv_cache_bytes) / hardware.memory.bandwidth).to("ms")
        
        total_memory_required = model_weights_bytes + kv_cache_bytes
        feasible = total_memory_required <= hardware.memory.capacity
        
        return {
            "feasible": feasible,
            "ttft": t_prefill,
            "itl": t_decode_per_token,
            "kv_cache_size": kv_cache_bytes.to("GB"),
            "model_weights_size": model_weights_bytes.to("GB"),
            "total_memory_required": total_memory_required.to("GB"),
            "memory_utilization": (total_memory_required / hardware.memory.capacity).to_base_units().magnitude
        }

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
    def solve(self, fleet: Fleet, duration_days: float, kwh_price: Optional[float] = None, datacenter: Optional[Any] = None, grid: Optional[Any] = None) -> Dict[str, Any]:
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
            price = getattr(target, 'kwh_price', 0.12)
            
        opex_energy = energy_result["total_energy_kwh"].magnitude * price
        
        accel_cost = fleet.node.accelerator.unit_cost or Q_("30000 USD")
        node_cost = (accel_cost.magnitude * fleet.node.accelerators_per_node) * 1.25  # ~25% overhead for CPU/RAM/Chassis
        total_capex = node_cost * fleet.count
        
        annual_maintenance_ratio = 0.05
        opex_maintenance = total_capex * annual_maintenance_ratio * (duration_days / 365.0)
        
        # Merge energy result into TCO result
        result = {
            "capex_usd": total_capex,
            "opex_energy_usd": opex_energy,
            "opex_maintenance_usd": opex_maintenance,
            "total_opex_usd": opex_energy + opex_maintenance,
            "tco_usd": total_capex + opex_energy + opex_maintenance
        }
        result.update(energy_result)
        return result
