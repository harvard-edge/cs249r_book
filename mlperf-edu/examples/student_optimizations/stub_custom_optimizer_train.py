import torch
import sys
import os

from mlperf.loadgen import LoadGenProxy 

# Import the Pristine Base Architecture (Students should absolutely not re-write the forward loop unless instructed!)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from reference.cloud.nanogpt_train import NanoGPTWhiteBox

def execute_student_training_optimization():
    """
    MLPerf EDU: The Open Division Training Sandbox!
    Target: Time-To-Target-Loss (TTTL)
    
    This template structurally demonstrates exactly how to swap `torch.optim.AdamW`
    with a completely custom mathematical systems engine! Your exact Backpropagation 
    physics will explicitly be tracked securely by the local LoadGen API accurately!
    """
    print("🚀 Booting Custom System-Under-Test (SUT) Training Wrapper...")
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Hardware Trace Locked: Exploiting {device.upper()} natively!")
    
    model = NanoGPTWhiteBox().to(device)
    model.train()
    
    # ---------------------------------------------------------------------------------
    # ⚠️ YOUR CUSTOM SYSTEMS LOGIC GOES HERE 
    # Do not use `torch.optim`. Write your own Stochastic Gradient Descent natively!
    # Tip: Evaluate memory overheads of storing custom momentums mathematically!
    # ---------------------------------------------------------------------------------
    
    # Example Custom Wrapper (Conceptual Placeholder)
    def my_custom_backward_step(loss, parameters):
        loss.backward()
        with torch.no_grad():
            for p in parameters:
                if p.grad is not None:
                    # Implement your mathematical routing natively here!
                    p.sub_(p.grad * 0.001)
                    p.grad.zero_()
                    
    # ---------------------------------------------------------------------------------

    # DUMMY LOOP (In reality, replace with the WikiText dataloader organically)
    for epoch in range(10): 
        dummy_data = torch.randint(0, 50257, (1, 16)).to(device)
        dummy_targets = torch.randint(0, 50257, (1, 16)).to(device)
        
        logits, loss = model(dummy_data, targets=dummy_targets)
        
        # Execute your custom physics!
        my_custom_backward_step(loss, model.parameters())
        
        # The MLPerf EDU referee automatically halts your run if Loss < target bound!
        # (This is just an integration demo, loadgen handles this internally physically)
        print(f"Step {epoch} | Custom SUT Loss Computed: {loss.item():.4f}")

if __name__ == "__main__":
    execute_student_training_optimization()
