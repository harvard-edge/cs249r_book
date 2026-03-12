import pytest
from mlsysim.core.solver import SingleNodeModel
from mlsysim.hardware import Hardware
from mlsysim.models import Models
from mlsysim.core.exceptions import OOMError

def test_engine_single_inference():
    resnet = Models.ResNet50
    a100 = Hardware.A100
    
    perf = SingleNodeModel().solve(resnet, a100, batch_size=1)
    
    # Check that performance profile is well-formed
    assert perf.feasible is True
    assert perf.latency.magnitude > 0
    assert perf.throughput.magnitude > 0
    assert perf.bottleneck in ["Compute", "Memory"]

def test_engine_oom_exception():
    gpt4 = Models.GPT4
    esp32 = Hardware.Tiny.ESP32
    
    # This should be infeasible
    perf = SingleNodeModel().solve(gpt4, esp32, batch_size=1, raise_errors=False)
    assert perf.feasible is False
    
    # This should raise
    with pytest.raises(OOMError):
        SingleNodeModel().solve(gpt4, esp32, batch_size=1, raise_errors=True)

def test_engine_precision_switching():
    resnet = Models.ResNet50
    a100 = Hardware.A100
    
    perf_fp16 = SingleNodeModel().solve(resnet, a100, batch_size=1, precision="fp16")
    perf_fp32 = SingleNodeModel().solve(resnet, a100, batch_size=1, precision="fp32")
    
    # FP32 should have lower peak flops than FP16 tensor core
    assert perf_fp32.peak_flops_actual < perf_fp16.peak_flops_actual
    assert perf_fp32.latency > perf_fp16.latency
