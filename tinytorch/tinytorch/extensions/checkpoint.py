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
