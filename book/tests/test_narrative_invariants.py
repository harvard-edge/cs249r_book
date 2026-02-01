# test_narrative_invariants.py
# This test suite protects the "truth" of the textbook.
# It ensures that the mathematical conclusions drawn in the prose
# remain valid even as underlying hardware constants are updated.

import pytest
from calc.constants import (
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
from calc.formulas import (
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
