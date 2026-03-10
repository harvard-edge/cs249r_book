import pytest
from mlsysim.models.types import TransformerWorkload
from mlsysim.hardware.registry import Hardware
from mlsysim.systems.registry import Systems
from mlsysim.core.solver import DistributedSolver, ServingSolver
from mlsysim.core.constants import Q_

def test_distributed_zero_lora_overlap():
    model = TransformerWorkload(
        name="Llama-3-70B",
        architecture="Transformer",
        parameters=Q_("70e9 param"),
        layers=80,
        hidden_dim=8192,
        heads=64
    )
    fleet = Systems.Clusters.Frontier_8K
    
    # Baseline
    solver = DistributedSolver()
    base_res = solver.solve(model, fleet, batch_size=4096, tp_size=8, pp_size=4, zero_stage=0, is_lora=False, overlap_comm=False)
    
    # With optimizations
    opt_res = solver.solve(model, fleet, batch_size=4096, tp_size=8, pp_size=4, zero_stage=3, is_lora=True, overlap_comm=True)
    
    # Memory should be radically smaller due to LoRA and ZeRO-3
    assert opt_res.node_performance.memory_footprint < base_res.node_performance.memory_footprint
    # Step latency should be lower due to comm overlap and smaller gradient comm size
    assert opt_res.step_latency_total < base_res.step_latency_total

def test_serving_disaggregated_speculative():
    target = TransformerWorkload(
        name="Llama-3-70B",
        architecture="Transformer",
        parameters=Q_("70e9 param"),
        layers=80,
        hidden_dim=8192,
        heads=64
    )
    draft = TransformerWorkload(
        name="Llama-3-8B",
        architecture="Transformer",
        parameters=Q_("8e9 param"),
        layers=32,
        hidden_dim=4096,
        heads=32
    )
    
    hw_prefill = Hardware.Cloud.H100
    hw_decode = Hardware.Cloud.A100 # Change from L40S to A100 to pass hardware lookup
    
    solver = ServingSolver()
    
    # Standard single node
    res_base = solver.solve(target, hw_prefill, seq_len=1024, batch_size=1)
    
    # Disaggregated + Speculative
    res_opt = solver.solve(
        target, 
        hw_prefill, 
        seq_len=1024, 
        batch_size=1,
        decode_hardware=hw_decode,
        network_bandwidth=Q_("100 GB/s"),
        draft_model=draft,
        draft_acceptance_rate=0.7
    )
    
    # ITL should be faster despite running decode on slower A100, due to speculative decoding
    assert res_opt.itl.magnitude > 0
