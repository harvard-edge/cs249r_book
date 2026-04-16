import torch
import torch.nn as nn
from mlperf.sut import SUT_Interface
from reference.edge.resnet_train import ResNet18WhiteBox

class OpenResNetQuantized(SUT_Interface):
    """
    OPEN DIVISION: Extreme Architectural Transformations allowed!
    
    Student Optimization Notes: 
    In the Open division, you do not have to hit 99% Accuracy! You can aggressively
    quantize (INT8), prune entire layers, or inject sparse convolutions to dramatically 
    increase your Speed (QPS) and minimize Power drainage! 
    """
    def __init__(self, config: dict):
        super().__init__(config)
        # Note: PyTorch native dynamic quantization runs predominantly on CPU currently
        self.device = torch.device('cpu')
        
        print("[Submitter:Open] 🧠 Loading Native ResNet18WhiteBox Parameters...")
        self.model = ResNet18WhiteBox(num_classes=100)
        self.model.eval()
        
        # 1. STUDENT OPTIMIZATION: Extreme INT8 Quantization!
        print("[Submitter:Open] ⚡ Smashing Math precision dynamically to INT8!")
        # We target specific linear and convolutional bottlenecks mathematically.
        self.optimized_model = torch.quantization.quantize_dynamic(
            self.model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )

    async def process_queries(self, samples: list):
        batch_size = len(samples)
        input_tensor = torch.randn((batch_size, 3, 32, 32))
        
        # 2. Execute Quantized execution structurally
        with torch.inference_mode():
            logits = self.optimized_model(input_tensor)
            
        # Due to 8-bit dynamic rounding, precision will crash heavily missing 
        # the 99% threshold. But throughput will spike immensely!
        return {
            "predictions": logits,
            "targets": torch.randint(0, 100, (batch_size,))
        }
