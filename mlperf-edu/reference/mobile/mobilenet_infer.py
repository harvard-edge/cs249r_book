import asyncio
import torch
from mlperf.loadgen import LoadGenProxy
from examples.mobile.mobilenet_core import build_model

def get_inference_handler(model: torch.nn.Module, device: torch.device, disable_nms: bool):
    """
    Creates the asynchronous LoadGen handler for object detection.
    """
    model.eval()
    
    async def handler(samples):
        # We process each sample query
        batch_size = len(samples)
        # MobileNet typically uses 300x300 for SSD
        input_tensor = torch.randn((batch_size, 3, 300, 300)).to(device)
        
        with torch.no_grad():
            telemetry = model(input_tensor, disable_nms=disable_nms)
            
        return telemetry
        
    return handler

def run_benchmark(config: dict, scenario: str = "offline", disable_nms: bool = False, use_golden: bool = False):
    """
    Hooks into MLPerf EDU LoadGen to measure Backbone vs NMS latency.
    """
    print(f"[Mobile:Infer] 🚀 Initiating Detection Phase. NMS Disabled: {disable_nms}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = build_model(use_golden=use_golden)
    model = model.to(device)
    
    # Pre-allocate random weights if not using golden to avoid warmup hits
    if not use_golden:
      pass # PyTorch initializes random weights by default
    
    handler = get_inference_handler(model, device, disable_nms=disable_nms)
    
    qps = 20.0 if scenario == "server" else 0.0
    proxy = LoadGenProxy(scenario=scenario, qps=qps, total_samples=15)
    report = asyncio.run(proxy.run(handler))
    
    print("\n" + "="*50)
    print("Mobile Object Detection Benchmark Report")
    print("="*50)
    for key, value in report.items():
        if isinstance(value, float):
            print(f"{key:.<25} {value:.4f}")
        else:
            print(f"{key:.<25} {value}")
            
    print("="*50)
