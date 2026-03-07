from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict
from .engine import PerformanceProfile, Engine
from .formulas import (
    calc_ring_allreduce_time, 
    calc_tree_allreduce_time,
    calc_hierarchical_allreduce_time,
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
    """
    def solve(self, model: Workload, hardware: HardwareNode, batch_size: int = 1, precision: str = "fp16", efficiency: float = 0.5, raise_errors: bool = False) -> PerformanceProfile:
        """
        Solves the performance profile for a single hardware node.

        Parameters
        ----------
        model : Workload
            The model architecture (Transformer, CNN).
        hardware : HardwareNode
            The target hardware specification.
        batch_size : int, optional
            Number of samples per inference/step, by default 1.
        precision : str, optional
            Numerical precision format ('fp32', 'fp16', 'int8', 'int4'), by default "fp16".
        efficiency : float, optional
            Hardware utilization efficiency (0.0 to 1.0), by default 0.5.
        raise_errors : bool, optional
            Whether to raise OOMError for infeasible workloads, by default False.

        Returns
        -------
        PerformanceProfile
            The resulting latency, throughput, and bottleneck analysis.
        """
        return Engine.solve(model, hardware, batch_size=batch_size, precision=precision, efficiency=efficiency, raise_errors=raise_errors)

class DistributedSolver(BaseSolver):
    """
    Resolves fleet-wide communication, synchronization, and pipelining constraints.
    
    This solver models the constraints of distributed scale for distributed training. It
    decomposes a workload across a cluster using 3D Parallelism (DP, TP, PP) 
    and calculates the resulting communication overheads and idle times 
    (bubbles) that determine the Model FLOPs Utilization (MFU).
    """
    def solve(self, 
              model: Workload, 
              fleet: Fleet, 
              batch_size: int = 1, 
              precision: str = "fp16", 
              efficiency: float = 0.5,
              tp_size: int = 1,
              pp_size: int = 1,
              microbatch_count: int = 1,
              topology_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculates distributed training performance using the 3D Parallelism model.

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
        microbatch_count : int
            Number of microbatches (M). Increasing M reduces the pipeline 
            bubble but increases synchronization overhead.
        topology_override : str, optional
            Force a specific topology (ring, tree).

        Returns
        -------
        Dict[str, Any]
            Metrics including DP/TP latency, the Pipeline Bubble penalty, 
            and the final Scaling Efficiency.
        """
        # 1. 3D Parallelism Decomposition
        n_accelerators = fleet.total_accelerators
        dp_size = n_accelerators // (tp_size * pp_size)
        
        if dp_size < 1:
            raise ValueError(f"Infeasible 3D Parallelism: TP({tp_size}) * PP({pp_size}) > Total({n_accelerators})")

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

        # 4. Pipeline Parallelism (PP) Bubble
        # Source: Narayanan et al. (2019), "PipePipe: Efficient Pipeline Parallelism"
        bubble_fraction = calc_pipeline_bubble(pp_size, microbatch_count)
        t_bubble = (node_perf.latency * bubble_fraction) if pp_size > 1 else Q_("0 ms")

        # 5. Total Latency and Scaling Efficiency
        total_comm_latency = t_comm_dp + t_comm_tp
        step_latency_total = node_perf.latency + total_comm_latency + t_bubble
        
        scaling_efficiency = (node_perf.latency / step_latency_total).magnitude
        
        return {
            "node_performance": node_perf,
            "dp_communication_latency": t_comm_dp,
            "tp_communication_latency": t_comm_tp,
            "communication_latency": total_comm_latency, # Backwards compatibility for tests
            "pipeline_bubble_latency": t_bubble,
            "bubble_fraction": bubble_fraction,
            "step_latency_total": step_latency_total,
            "scaling_efficiency": scaling_efficiency,
            "effective_throughput": (n_accelerators * node_perf.throughput * scaling_efficiency),
            "parallelism": {"dp": dp_size, "tp": tp_size, "pp": pp_size}
        }

class ReliabilitySolver(BaseSolver):
    """
    Calculates Mean Time Between Failures (MTBF) and optimal checkpointing intervals.
    
    This solver handles the reliability modeling of massive clusters, helping
    determine the 'Goodput' of long-running training jobs. It identifies 
    the probability of a job failure before completion and calculates the 
    Young-Daly optimal interval to minimize wasted compute time.
    """
    def solve(self, fleet: Fleet, job_duration_hours: float, checkpoint_time_s: float = 60.0) -> Dict[str, Any]:
        """
        Calculates reliability and checkpointing metrics for a fleet.

        Parameters
        ----------
        fleet : Fleet
            The hardware cluster configuration.
        job_duration_hours : float
            Total wall-clock duration of the training job.
        checkpoint_time_s : float, optional
            Time taken to save a single checkpoint, by default 60.0.

        Returns
        -------
        Dict[str, Any]
            Reliability metrics including fleet MTBF and failure probability.
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
    """
    def solve(self, fleet: Fleet, duration_days: float, datacenter: Optional[Datacenter] = None) -> Dict[str, Any]:
        """
        Calculates energy, carbon, and water footprint for a fleet operation.

        Parameters
        ----------
        fleet : Fleet
            The hardware cluster configuration.
        duration_days : float
            Operating duration in days.
        datacenter : Datacenter, optional
            A specific datacenter profile, defaults to fleet's region.

        Returns
        -------
        Dict[str, Any]
            Sustainability metrics including total energy (kWh) and carbon (kgCO2e).
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
        it_power_w = fleet.node.accelerator.tdp * fleet.total_accelerators if fleet.node.accelerator.tdp else Q_("700 W") * fleet.total_accelerators
            
        # 3. Energy Consumption
        it_energy_kwh = (it_power_w * Q_(duration_hours, "hour")).to("kWh")
        
        # Apply PUE
        pue = getattr(dc, 'pue', fleet.effective_pue)
        total_energy_kwh = it_energy_kwh * pue
        
        # 4. Carbon Footprint
        carbon_kg = region.carbon_kg(it_energy_kwh.magnitude) if hasattr(region, 'carbon_kg') else it_energy_kwh.magnitude * (region.carbon_intensity_g_kwh / 1000.0)
        
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
    process with two distinct physical regimes:
    
    1. **Pre-fill Phase**: The initial processing of the input prompt. This 
       is a 'Compute Beast' phase where all prompt tokens are processed 
       in parallel, saturating the GPU's arithmetic units.
    2. **Decoding Phase**: The token-by-token generation. This is a 
       'Bandwidth Hog' phase. Because the model must read all parameters 
       from memory just to generate a single token, it is limited entirely 
       by HBM bandwidth.
    
    This solver also models the **KV-Cache**, the memory required to store 
    previous token states, which grows linearly with sequence length and 
    batch size, eventually hitting the 'Memory Wall'.
    """
    def solve(self, model: TransformerWorkload, hardware: HardwareNode, seq_len: int, batch_size: int = 1, precision: str = "fp16", efficiency: float = 0.5) -> Dict[str, Any]:
        """
        Solves for LLM serving performance.

        Parameters
        ----------
        model : TransformerWorkload
            The LLM model architecture.
        hardware : HardwareNode
            The target hardware for inference.
        seq_len : int
            The total context window (prompt + generated tokens).
        batch_size : int, optional
            Number of concurrent user requests.
        precision : str, optional
            Numerical format. Lower precision (INT8/INT4) reduces 
            memory pressure and speeds up the Decoding phase.
        efficiency : float, optional
            Compute utilization efficiency, primarily affecting the Pre-fill phase.

        Returns
        -------
        Dict[str, Any]
            Inference metrics including Time-To-First-Token (TTFT), 
            Inter-Token Latency (ITL), and total KV-cache footprint.
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
    into a single financial model for the fleet. This solver exposes 
    the ROI of architectural efficiency by showing how reducing power 
    draw or increasing throughput directly impacts the bottom line.
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
        
        unit_cost = fleet.node.accelerator.unit_cost or Q_("30000 USD")
        total_capex = unit_cost.magnitude * fleet.total_accelerators
        
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
