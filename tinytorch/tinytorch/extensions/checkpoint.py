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
            return out_tensor.data
            
        def backward(self, grad_output):
            # 1. Detach inputs so they act as graph leaves for the recomputation
            detached_inputs = []
            for inp in self.inputs:
                x = Tensor(inp.data, requires_grad=inp.requires_grad)
                detached_inputs.append(x)
                
            # 2. Recompute the forward pass WITH gradients enabled
            recomputed_out = function(*detached_inputs)
            
            # 3. Trigger the local backward pass from the recomputed output
            # We wrap the incoming grad_output numpy array into a Tensor
            recomputed_out.backward(Tensor(grad_output))
            
            # 4. Return the gradients of the inputs (numpy arrays)
            return tuple(x.grad.data if x.grad is not None else None for x in detached_inputs)

    return CheckpointFunction.apply(*args)
