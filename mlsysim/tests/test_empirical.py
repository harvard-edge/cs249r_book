import pytest
import mlsysim
from mlsysim.core.constants import ureg

def test_mlperf_resnet_a100():
    """
    Empirical Anchor: ResNet-50 on NVIDIA A100 (SXM4).
    Reference: MLPerf Inference v4.0, NVIDIA Submission.
    Target: ~37,000 samples/second (Offline scenario).
    """
    model = mlsysim.Models.Vision.ResNet50
    hardware = mlsysim.Hardware.A100
    
    # We use an efficiency factor (eta) to match real-world overheads
    # observed in MLPerf (kernel launch, data loading, etc.)
    # 0.49 is a typical MFU/HFU for ResNet on A100 at scale.
    perf = mlsysim.Engine.solve(model, hardware, batch_size=2048, efficiency=0.49)
    
    predicted_throughput = perf.throughput.m_as("1/second")
    
    # Target is ~37,000
    assert 35000 <= predicted_throughput <= 40000
    print(f"Predicted: {predicted_throughput:.1f} samples/s | MLPerf Target: ~37,000")

def test_llama_inference_h100():
    """
    Empirical Anchor: Llama-2-70B on NVIDIA H100.
    Reference: NVIDIA/vLLM benchmarks.
    Target ITL: ~40-50ms (Batch 1, FP16).
    """
    model = mlsysim.Models.Language.Llama2_70B
    hardware = mlsysim.Hardware.H100
    
    solver = mlsysim.ServingSolver()
    result = solver.solve(model, hardware, seq_len=2048, batch_size=1, efficiency=1.0)
    
    itl = result.itl.m_as("ms")
    
    # ITL = ModelSize / BW = 140GB / 3.35TB/s = ~41.8ms
    assert 40 <= itl <= 45
    print(f"Predicted ITL: {itl:.2f} ms | vLLM Target: ~42ms")
