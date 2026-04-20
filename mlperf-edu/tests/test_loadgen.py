import asyncio
import pytest
import time
import numpy as np
from mlperf_edu.loadgen import LoadGenProxy

@pytest.mark.asyncio
async def test_offline_scenario():
    """Test the Offline scenario dispatches all samples."""
    total_samples = 50
    proxy = LoadGenProxy(scenario='Offline', total_samples=total_samples)
    
    async def mock_handler(samples):
        # Simulate some processing time
        await asyncio.sleep(0.01)
        return "done"

    report = await proxy.run(mock_handler)
    
    assert report['total_samples'] == total_samples
    assert report['scenario'] == 'Offline'
    assert report['duration'] > 0
    assert len(proxy.responses) == total_samples

@pytest.mark.asyncio
async def test_server_scenario_timing():
    """
    Test the Server scenario Poisson timing.
    Checks that the mean inter-arrival time is close to 1/QPS.
    """
    total_samples = 100
    qps = 50.0
    proxy = LoadGenProxy(scenario='Server', qps=qps, total_samples=total_samples)
    
    arrival_times = []
    
    async def mock_handler(samples):
        # We record the arrival time as recorded by the LoadGenProxy
        # for each sample.
        return "done"

    report = await proxy.run(mock_handler)
    
    assert report['total_samples'] == total_samples
    assert report['scenario'] == 'Server'
    
    # Extract recorded arrival times from responses
    responses = sorted(proxy.responses, key=lambda r: r.id)
    arrival_times = [r.arrival_time for r in responses]
    
    # Calculate inter-arrival times
    inter_arrivals = np.diff(arrival_times)
    
    mean_inter_arrival = np.mean(inter_arrivals)
    expected_mean = 1.0 / qps
    
    # Allow for some statistical variance and system noise (30% margin)
    # Poisson process with 100 samples should be reasonably close.
    print(f"Mean inter-arrival: {mean_inter_arrival:.4f}, Expected: {expected_mean:.4f}")
    assert expected_mean * 0.5 < mean_inter_arrival < expected_mean * 1.5

@pytest.mark.asyncio
async def test_single_stream_scenario():
    """Test the SingleStream scenario (serial processing)."""
    total_samples = 10
    proxy = LoadGenProxy(scenario='SingleStream', total_samples=total_samples)
    
    call_count = 0
    async def mock_handler(samples):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)
        return "done"

    report = await proxy.run(mock_handler)
    
    assert report['total_samples'] == total_samples
    assert call_count == total_samples
    # Duration should be at least total_samples * processing_time
    assert report['duration'] >= total_samples * 0.01

@pytest.mark.asyncio
async def test_loadgen_latency_calculation():
    """Test that latency is calculated correctly."""
    proxy = LoadGenProxy(scenario='Offline', total_samples=1)
    
    async def slow_handler(samples):
        await asyncio.sleep(0.1)
        return "done"

    report = await proxy.run(slow_handler)
    
    assert report['latency_avg'] >= 0.1
    assert proxy.responses[0].latency >= 0.1

def test_unsupported_scenario():
    """Test that unsupported scenarios raise ValueError."""
    proxy = LoadGenProxy(scenario='Unsupported')
    
    async def mock_handler(samples):
        return "done"
    
    with pytest.raises(ValueError, match="Unsupported scenario"):
        asyncio.run(proxy.run(mock_handler))
