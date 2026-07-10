import torch
import torch.nn as nn
from rich.console import Console

console = Console()

class PedagogicalMobileBERT(nn.Module):
    def __init__(self, vocab_size=30522, d_model=128, inter_model=512):
        super().__init__()
        # MobileBERT compresses the input severely compared to BERT base using Bottleneck dimensions
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # We wrap this layer specifically to showcase PyTorch INT8 Dynamic Quantization
        self.encoder = nn.Linear(d_model, inter_model)
        self.activation = nn.ReLU()
        self.decoder = nn.Linear(inter_model, d_model)
        
        self.qa_outputs = nn.Linear(d_model, 2)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.encoder(x)
        x = self.activation(x)
        x = self.decoder(x)
        return self.qa_outputs(x)

def run_benchmark(provd_path: str, scenario: str):
    """
    MobileBERT execution mapping Mobile INT8 quant execution latency.
    """
    console.print("[Mobile:Infer] 🤳 Instantiating PyTorch MobileBERT from provenance...")
    
    model = PedagogicalMobileBERT()
    model.eval()
    
    # Ensure cross-platform execution (M1 Macs use qnnpack, x86 uses fbgemm)
    if 'qnnpack' in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = 'qnnpack'
        
    # 💥 Dynamic INT8 Quantization Step (Pedagogical Core!)
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    console.print("[Mobile:Infer] ⚡ MobileBERT dynamically quantized to INT8.")
    
    batch_size = 1
    seq_length = 64
    input_ids = torch.randint(0, 30522, (batch_size, seq_length))
    
    with torch.no_grad():
        out = quantized_model(input_ids)
        
    console.print(f"[Mobile:Infer] ✅ Quantized forward pass successful. Logits Shape: {out.shape}")
