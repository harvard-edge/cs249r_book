# Hand-maintained extension, NOT generated. This file is not produced by
# `tito dev export` or `tito module complete`; it is tracked in git and
# edited directly here. It is the loss-scaling example of the optimizer
# extension point; no src/ module or student notebook
# exports to it.
"""Mixed-Precision Loss Scaling for TinyTorch: an example extension.

This is an optional, hand-written extension: it is tracked in git as-is and is
not generated from src/. Read it as the example of extending around the
optimizer: it changes how gradients are produced and consumed without touching
autograd or the optimizer itself.

Why it exists: float16 halves memory and can double throughput on hardware with
half-precision units, but its smallest normal value is about 6e-5 (and its
smallest subnormal about 6e-8). Late in training many gradients are smaller
than that, and in float16 they round to exactly zero, so those weights stop
learning. Multiplying the loss by a scale S before backward multiplies every
gradient by S (the chain rule is linear), lifting them into float16's range.
Dividing by S before the optimizer step restores their true size. A power of
two such as 65536 = 2^16 only shifts the exponent, so scaling adds no rounding
error of its own.

How it works:

1. backward(loss) computes loss * scale and calls backward on that, so every
   gradient arrives multiplied by scale.
2. unscale_(optimizer) divides each parameter's gradient by scale. Call it
   after backward and before optimizer.step(), and before any gradient
   clipping, which must see true gradient sizes.

What this example leaves out: TinyTorch computes in float32, so here the
scaler shows the mechanism rather than preventing a real underflow. And its
scale is fixed. Production scalers are dynamic: when a scaled gradient
overflows to inf or NaN they skip that step and halve the scale, and after a
run of clean steps they double it. Adding that is the natural next step.
"""

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
