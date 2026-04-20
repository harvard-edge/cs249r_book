import time
import pytest
from mlperf_edu.power import PowerMeter, measure_power

def test_power_meter_basic():
    # Use a high TDP to make sure we see some energy used even in short time
    meter = PowerMeter(tdp_watts=200.0, interval=0.01)
    meter.start()
    time.sleep(0.1)
    joules = meter.stop()
    
    # Energy should be greater than 0
    # Expected: 200W * 0.1s = 20J (approximately)
    assert joules > 0
    assert joules < 50 # Safety margin

def test_measure_power_context_manager():
    with measure_power(tdp_watts=100.0) as meter:
        time.sleep(0.2)
        current_joules = meter.energy_joules
        assert current_joules > 0
        
    final_joules = meter.energy_joules
    assert final_joules >= current_joules
    # Expected: P_idle * TDP * time = 0.15 * 100 * 0.2 = 3.0J
    # With some slack:
    assert 2.0 < final_joules < 10.0

def test_power_meter_integration():
    # Simulate some work
    with measure_power(tdp_watts=100.0) as meter:
        # Busy wait to increase CPU util
        start = time.time()
        while time.time() - start < 0.2:
            _ = [i**2 for i in range(1000)]
    
    joules = meter.energy_joules
    assert joules > 0
    print(f"Measured {joules} Joules for 0.2s of work at 100W TDP")
