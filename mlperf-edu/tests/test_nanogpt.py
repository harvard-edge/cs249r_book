import asyncio
import pytest
from labs.inference.nanogpt import run_nanogpt_benchmark

@pytest.mark.asyncio
async def test_nanogpt_single_stream():
    """Test the NanoGPT baseline in SingleStream scenario."""
    # We need to make sure labs.inference is importable
    # We set PYTHONPATH to the root during test execution
    report = await run_nanogpt_benchmark(scenario='SingleStream', total_samples=2, use_kv_cache=True)
    
    assert report['total_samples'] == 2
    assert report['scenario'] == 'SingleStream'
    assert 'ttft_avg' in report
    assert 'tpot_avg' in report
    assert report['ttft_avg'] > 0
    assert report['tpot_avg'] > 0

@pytest.mark.asyncio
async def test_nanogpt_offline_kv_cache_impact():
    """Test that disabling KV-cache increases TPOT."""
    # Run with KV-cache
    report_kv = await run_nanogpt_benchmark(scenario='Offline', total_samples=1, use_kv_cache=True)
    # Run without KV-cache
    report_no_kv = await run_nanogpt_benchmark(scenario='Offline', total_samples=1, use_kv_cache=False)
    
    # Without KV-cache, the simulation should result in higher TPOT (on average)
    # since it reprocesses the entire sequence for every token.
    assert report_no_kv['tpot_avg'] > report_kv['tpot_avg']
