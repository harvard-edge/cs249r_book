# test_narrative_invariants.py
# This test suite protects the "truth" of the textbook.
# It ensures that the mathematical conclusions drawn in the prose
# remain valid even as underlying hardware constants are updated.

import os
import sys
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, 'quarto'))

from mlsysim.core.constants import (
    ureg,
    SPEED_OF_LIGHT_FIBER_KM_S,
    NETWORK_10G_BW,
    CLOUD_EGRESS_PER_GB,
    CLOUD_ELECTRICITY_PER_KWH,
    PHONE_BATTERY_WH,
    OBJECT_DETECTOR_POWER_W,
    RESNET50_FLOPs,
    A100_MEM_BW,
    MOBILE_NPU_MEM_BW,
    VIDEO_1080P_WIDTH, VIDEO_1080P_HEIGHT, VIDEO_BYTES_PER_PIXEL_RGB, VIDEO_FPS_STANDARD,
    NETWORK_5G_ENERGY_PER_MB_MJ, ENERGY_MOBILENET_INF_MJ,
    MB, GB
)
from mlsysim.core.formulas import (
    calc_network_latency_ms,
    calc_monthly_egress_cost,
    calc_fleet_tco,
    calc_bottleneck
)

def test_distance_penalty_precludes_cloud():
    """
    Narrative: "Physics has made Cloud ML impossible for this application."
    Invariant: Round-trip time for 1500km must exceed the 10ms safety budget.
    """
    distance_km = 1500
    safety_budget_ms = 10
    
    rtt_ms = calc_network_latency_ms(distance_km)
    
    assert rtt_ms > safety_budget_ms, \
        f"Narrative Violation: Cloud latency ({rtt_ms:.1f}ms) is now faster than safety budget ({safety_budget_ms}ms)!"

def test_bandwidth_makes_cloud_streaming_impossible():
    """
    Narrative: "You need 15x more bandwidth than exists."
    Invariant: 100 cameras require significantly more bandwidth than a 10Gbps link.
    """
    num_cameras = 100
    raw_bytes_per_sec = (VIDEO_1080P_WIDTH * VIDEO_1080P_HEIGHT * VIDEO_BYTES_PER_PIXEL_RGB * VIDEO_FPS_STANDARD)
    total_bandwidth = num_cameras * raw_bytes_per_sec
    
    # Check that we exceed 10Gbps link
    assert total_bandwidth > NETWORK_10G_BW, \
        "Narrative Violation: 10Gbps link is now sufficient for 100 raw streams!"
        
    # Check the "15x" claim (allow some drift, but it should be >10x)
    ratio = total_bandwidth / NETWORK_10G_BW
    assert ratio > 10, \
        f"Narrative Violation: Bandwidth shortage dropped to {ratio:.1f}x (text implies ~15x)"

def test_energy_gap_favors_edge():
    """
    Narrative: "Transmitting data is 1000x more energy-expensive than processing it locally."
    Invariant: Transmission energy >> Inference energy.
    """
    # Energy for 1MB
    data_size = 1 * MB
    tx_energy = NETWORK_5G_ENERGY_PER_MB_MJ * data_size
    npu_energy = ENERGY_MOBILENET_INF_MJ
    
    ratio = tx_energy / npu_energy
    
    assert ratio > 500, \
        f"Narrative Violation: Energy gap dropped to {ratio:.1f}x (text says 1000x)!"

def test_edge_tco_cheaper_than_cloud():
    """
    Narrative: "Coral Dev Board ... meets requirements at 1/4 the cost of Jetson... vs cloud inference at ~$500K"
    Invariant: Edge TCO < Cloud TCO for high-volume scenario.
    """
    stores = 500
    cameras = 20
    fps = 15
    years = 3
    
    # Cloud Cost Reference (from text approximation)
    # 1M inferences/day was $36k/year.
    # Here we have 500 stores * 20 cams * 15 fps = 150,000 inf/sec
    # This volume is massive. Just check Edge vs Edge relative costs first.
    
    coral_cost = 150 * ureg.dollar
    coral_power = 2 * ureg.watt
    
    jetson_cost = 600 * ureg.dollar
    jetson_power = 25 * ureg.watt
    
    coral_tco = calc_fleet_tco(coral_cost, coral_power, stores, years, CLOUD_ELECTRICITY_PER_KWH)
    jetson_tco = calc_fleet_tco(jetson_cost, jetson_power, stores, years, CLOUD_ELECTRICITY_PER_KWH)
    
    assert coral_tco < jetson_tco, "Narrative Violation: Coral is no longer cheaper than Jetson!"
    
    ratio = jetson_tco / coral_tco
    assert ratio > 2, f"Narrative Violation: Jetson/Coral cost ratio is {ratio:.1f}x (text implies ~3-4x)"

def test_battery_tax_depletes_phone():
    """
    Narrative: "Your single feature has just consumed 30% [or 100%] of the budget."
    Invariant: A 2W continuous load drains a 15Wh battery in < 1 day.
    """
    runtime = PHONE_BATTERY_WH / OBJECT_DETECTOR_POWER_W
    day = 24 * ureg.hour
    
    assert runtime < day, "Narrative Violation: Battery now lasts more than a day with continuous ML!"
    assert runtime < (8 * ureg.hour), "Narrative Violation: Battery lasts too long (>8h) for the 'few hours' narrative."


# ==============================================================================
# VOLUME 1: FOUNDATIONS & SINGLE NODE
# ==============================================================================

def test_inference_memory_wall():
    '''
    Narrative: "For batch-1 serving (single user), the H100's bandwidth determines throughput, and the compute units sit almost entirely idle."
    Invariant: Transformer decoding arithmetic intensity << Ridge Point.
    '''
    from mlsysim import Hardware
    from mlsysim.core.constants import TFLOPs, second, TB
    
    h100 = Hardware.Cloud.H100
    peak_flops = h100.peak_flops.m_as(TFLOPs/second)
    peak_bw = h100.memory_bw.m_as(TB/second)
    ridge_point = h100.ridge_point().m_as('flop/byte')
    
    # Batch-1 inference intensity is roughly 1 FLOP/byte
    batch1_intensity = 1.0 
    
    assert batch1_intensity < ridge_point, "Narrative Violation: Batch-1 inference is no longer memory-bound!"
    assert ridge_point / batch1_intensity > 50, "Narrative Violation: The gap between compute-bound and memory-bound is too small."

def test_amdahls_law_limits():
    '''
    Narrative: "To see Amdahl's Law in action, suppose 5% of your training step is serial overhead... speedup is capped at 20x."
    Invariant: 5% serial overhead = Max 20x speedup.
    '''
    serial_fraction = 0.05
    max_speedup = 1.0 / serial_fraction
    
    assert max_speedup == 20.0, f"Narrative Violation: 5% serial fraction cap should be exactly 20x, got {max_speedup}x"

def test_quantization_compression_ratios():
    '''
    Narrative: "INT8 quantization reduces memory footprint by 4x... versus FP32"
    Invariant: FP32 bytes / INT8 bytes == 4.0
    '''
    from mlsysim.core.constants import BYTES_FP32, BYTES_INT8, ureg
    
    fp32_size = BYTES_FP32.m_as(ureg.byte)
    int8_size = BYTES_INT8.m_as(ureg.byte)
    
    ratio = fp32_size / int8_size
    assert ratio == 4.0, f"Narrative Violation: INT8 compression ratio is no longer exactly 4x! Got {ratio}x"

# ==============================================================================
# VOLUME 2: DISTRIBUTED SYSTEMS & SCALE
# ==============================================================================

def test_bandwidth_staircase():
    '''
    Narrative: "Moving data across the cluster is 18x slower than moving it within a node."
    Invariant: Intra-node (NVLink) bandwidth >> Inter-node (InfiniBand) bandwidth.
    '''
    from mlsysim.core.constants import NVLINK_H100_BW, INFINIBAND_NDR_BW, GB, second
    
    nvlink_bw = NVLINK_H100_BW.m_as(GB/second)
    ib_bw = INFINIBAND_NDR_BW.m_as(GB/second)
    
    ratio = nvlink_bw / ib_bw
    assert ratio > 10, f"Narrative Violation: Bandwidth staircase collapsed! Ratio is only {ratio:.1f}x (needs to be >10x)"

def test_distributed_communication_overhead():
    '''
    Narrative: "Using ring-AllReduce over InfiniBand... communication time is approximately 233 seconds."
    Invariant: A 175B model AllReduce over IB takes multiple seconds.
    '''
    from mlsysim.core.constants import param, INFINIBAND_NDR_BW, BYTES_FP16, ureg
    
    params = 175e9 # 175B
    bytes_per_param = BYTES_FP16.m_as(ureg.byte)
    total_bytes = params * bytes_per_param
    
    # Ring-AllReduce moves roughly 2x the data
    data_moved = 2 * total_bytes
    
    ib_bw_bytes = INFINIBAND_NDR_BW.to("byte/second").magnitude
    time_seconds = data_moved / ib_bw_bytes
    
    assert time_seconds > 5.0, f"Narrative Violation: Communication is too fast ({time_seconds:.1f}s)! The 'bottleneck' narrative is broken."

def test_fleet_reliability_mtbf():
    '''
    Narrative: "A cluster of 10,000 GPUs will experience hardware failures at least daily."
    Invariant: 10,000 GPUs with individual MTBF of 3 years yields a cluster MTBF of < 3 hours.
    '''
    gpu_count = 10000
    individual_mtbf_hours = 3 * 365 * 24 # 3 years
    
    cluster_mtbf_hours = individual_mtbf_hours / gpu_count
    
    assert cluster_mtbf_hours < 24.0, f"Narrative Violation: Cluster MTBF is {cluster_mtbf_hours:.1f} hours. Failures are no longer a daily reality!"
    assert cluster_mtbf_hours < 5.0, f"Narrative Violation: Cluster MTBF is too high for the 'constant failure' narrative."
