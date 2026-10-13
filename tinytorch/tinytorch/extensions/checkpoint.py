# Hand-maintained extension, NOT generated. This file is not produced by
# `tito dev export` or `tito module complete`; it is tracked in git and
# edited directly here. It is the activation-checkpointing example of
# the autograd extension point; no src/ module or student notebook
# exports to it.
"""Activation Checkpointing for TinyTorch: an example extension.

What it is: a wrapper that runs part of a model without keeping its
intermediate activations, and recomputes them when backward needs them. It is
the example of extending autograd itself: a custom Function that decides what
the tape keeps. Hand-written and tracked in git as-is, not generated from
src/.

Why it exists: during training, the tape keeps every intermediate activation
of the forward pass alive until backward has used it, so activation memory
grows with depth and batch size and is often what runs out first.
Checkpointing a segment keeps only its inputs and throws its intermediates
away; during backward it runs the segment's forward again to rebuild them.
The trade is memory for compute: one extra forward pass of the segment.
Applied to about sqrt(n) segments of an n-layer network, it cuts activation
memory from O(n) to O(sqrt(n)) for roughly one extra forward pass overall.

How it is triggered: you wrap a call to part of your model, usually inside a
model's forward(). It then runs at two moments: once when you call it, and
again, automatically, when loss.backward() reaches it.

    from tinytorch.extensions import checkpoint
    def block(x):                    # any function of Tensors that
        return layer2(layer1(x))     # returns a Tensor
    h = checkpoint(block, x)         # instead of h = block(x)
    ...
    loss.backward()                  # autograd reaches the checkpoint node
                                     # and calls its backward(), which
                                     # runs block again with gradients on

    On the call:     block runs once under no_grad(); only its output is kept.
    On backward:     block runs a second time, this time recording, and the
                     gradients of its inputs flow on to earlier layers.

How it works:

1. checkpoint(function, *args) wraps the segment in a Function, so autograd
   records one node for the whole segment instead of one per operation.
2. forward() runs the segment under no_grad(), so none of its intermediates
   are recorded or kept.
3. backward() detaches the saved inputs into fresh graph leaves, runs the
   segment again with gradients on, and calls backward on that recomputed
   output. The leaves' gradients are the gradients of the segment's inputs.

What it leaves out: randomness. A segment that uses dropout would draw
different random numbers the second time; production versions save and
restore the random state. It also does not choose the segments for you.

Where to go next: measure it. Count the bytes the tape holds for a deep stack
of layers with and without checkpointing, and time the extra forward pass.
"""

from tinytorch.core.tensor import Tensor, Function
from tinytorch.core.autograd import no_grad

def checkpoint(function, *args):
    """
    Activation Checkpointing.
    Recomputes the forward pass during the backward pass to save memory.
    """
    class CheckpointFunction(Function):
        def forward(self, *numpy_arrays):
            # Run forward pass WITHOUT tracking gradients to save memory
            with no_grad():
                out_tensor = function(*self.inputs)
            # The returned value of forward() must be a numpy array
            return out_tensor.data if isinstance(out_tensor, Tensor) else out_tensor
            
        def backward(self, grad_output):
            # 1. Detach inputs so they act as graph leaves for the recomputation
            detached_inputs = []
            for inp in self.inputs:
                if isinstance(inp, Tensor):
                    x = Tensor(inp.data, requires_grad=inp.requires_grad)
                else:
                    x = inp
                detached_inputs.append(x)
                
            # 2. Recompute the forward pass WITH gradients enabled
            recomputed_out = function(*detached_inputs)
            
            # 3. Trigger the local backward pass from the recomputed output
            grad_tensor = Tensor(grad_output) if not isinstance(grad_output, Tensor) else grad_output
            recomputed_out.backward(grad_tensor)
            
            # 4. Return the gradients of the inputs (numpy arrays)
            grads = []
            for x in detached_inputs:
                if isinstance(x, Tensor) and x.grad is not None:
                    g = x.grad.data if isinstance(x.grad, Tensor) else x.grad
                    grads.append(g)
                else:
                    grads.append(None)
            return tuple(grads)

    return CheckpointFunction.apply(*args)
