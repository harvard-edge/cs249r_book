from tinytorch.core.tensor import Tensor
from tinytorch.core.optimizers import Optimizer

class LossScaler:
    """
    Mixed Precision Loss Scaler.
    Multiplies the loss by a large constant before backward() to prevent float16 underflow,
    then unscales the gradients before the optimizer step.
    """
    def __init__(self, scale: float = 65536.0):
        self.scale = scale
        
    def backward(self, loss: Tensor):
        # Multiply the loss to protect gradients from underflow
        scaled_loss = loss * self.scale
        # TinyTorch's autograd engine pushes the scaled gradients back
        scaled_loss.backward()
        
    def unscale_(self, optimizer: Optimizer):
        # Traverse the registered parameters and divide the gradients back down
        for p in optimizer.params:
            if p.grad is not None:
                p.grad = p.grad / self.scale
