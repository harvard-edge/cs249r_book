from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Dict, Any, Annotated, Union
from ..core.constants import Q_, ureg, BYTES_FP16
from ..core.types import Quantity, Metadata
from pydantic import AfterValidator

class ComputationGraph(BaseModel):
    """
    Hardware-Agnostic representation of a Workload.
    The 'Intermediate Representation' (IR) of demand.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str
    total_ops: Quantity
    parameter_count: Quantity
    weight_bytes: Quantity
    arithmetic_intensity: Quantity # Ops/Byte
    
    # Optional metadata
    layers: Optional[int] = None
    
    def __repr__(self):
        return f"ComputationGraph({self.name}, {self.total_ops:~P})"

class Workload(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    architecture: str
    metadata: Metadata = Field(default_factory=Metadata)
    parameters: Optional[Quantity] = None
    model_size: Optional[Quantity] = None
    inference_flops: Optional[Quantity] = None

    def lower(self, precision: Quantity = BYTES_FP16) -> ComputationGraph:
        """Lowers the workload into a hardware-agnostic computation graph."""
        raise NotImplementedError

    def size_in_bytes(self, precision: Quantity = BYTES_FP16) -> Quantity:
        if self.model_size is not None:
            return self.model_size
        if self.parameters is not None:
            param_count = self.parameters.to(ureg.count).magnitude
            bpp = precision.to(ureg.byte).magnitude
            return (param_count * bpp * ureg.byte).to(ureg.byte)
        raise NotImplementedError("Workload must define either parameters or model_size to calculate size in bytes.")

class TransformerWorkload(Workload):
    parameters: Quantity
    layers: int
    hidden_dim: Optional[int] = None
    heads: Optional[int] = None
    kv_heads: Optional[int] = None
    training_ops: Optional[Quantity] = None
    inference_flops: Optional[Quantity] = None
    
    def size_in_bytes(self, precision: Quantity = BYTES_FP16) -> Quantity:
        param_count = self.parameters.to(ureg.count).magnitude
        bpp = precision.to(ureg.byte).magnitude
        return (param_count * bpp * ureg.byte).to(ureg.byte)

    def get_kv_cache_size(self, seq_len: int, batch_size: int, precision: Quantity = BYTES_FP16) -> Quantity:
        from ..core.formulas import calc_kv_cache_size
        h_dim = self.hidden_dim or 4096
        n_heads = self.heads or 32
        head_dim = h_dim // n_heads
        n_kv_heads = self.kv_heads or n_heads
        return calc_kv_cache_size(n_layers=self.layers, n_heads=n_kv_heads, head_dim=head_dim, seq_len=seq_len, batch_size=batch_size, bytes_per_elem=precision)

    def training_memory(self, batch_size: int, seq_len: int, precision: str = "fp16", optimizer: str = "adam", strategy: str = "selective") -> Quantity:
        """
        Estimate training memory for a Transformer model.
        
        Source: Shoeybi et al. (2019), "Megatron-LM: Training Multi-Billion Parameter 
                Language Models Using Model Parallelism"
        
        Args:
            batch_size: Mini-batch size (B)
            seq_len: Sequence length (S)
            precision: Precision format ('fp32', 'fp16', 'int8', 'int4')
            optimizer: Optimizer type ('adam', 'sgd')
            strategy: Recompute strategy ('none', 'selective', 'full')
            
        Returns:
            Quantity[byte]: Total training memory per GPU
        """
        from ..core.constants import BYTES_FP32, BYTES_FP16, BYTES_INT8, BYTES_INT4
        from ..core.formulas import calc_activation_memory
        
        prec_map = {"fp32": BYTES_FP32, "fp16": BYTES_FP16, "int8": BYTES_INT8, "int4": BYTES_INT4}
        bpp = prec_map.get(precision, BYTES_FP16).to(ureg.byte).magnitude
        
        n_params = self.parameters.to(ureg.count).magnitude
        
        # 1. Weights and Gradients
        w_grad_mem = n_params * (bpp + bpp) * ureg.byte
        
        # 2. Optimizer States (Adam = 12 bytes/param for FP32 states)
        if optimizer.lower() == "adam":
            # Adam: master weights (4), momentum (4), variance (4) = 12 bytes/param
            opt_mem = n_params * 12 * ureg.byte
        else:
            # SGD: master weights (4) = 4 bytes/param
            opt_mem = n_params * 4 * ureg.byte
            
        # 3. Activation Memory (proportional to B, S, H)
        act_mem = calc_activation_memory(
            n_layers=self.layers,
            seq_len=seq_len,
            batch_size=batch_size,
            hidden_dim=self.hidden_dim or 4096,
            precision_bytes=bpp,
            strategy=strategy
        )
        
        return (w_grad_mem + opt_mem + act_mem).to(ureg.GB)
    
    @property
    def training_gpu_days(self):
        """
        Backward-compatibility alias for older book chapters.
        """
        if self.training_ops is not None:
            return self.training_ops
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute 'training_gpu_days'"
        )

    def lower(self, precision: Quantity = BYTES_FP16) -> ComputationGraph:
        ops = self.inference_flops or (2 * self.parameters.to(ureg.count).magnitude * ureg.flop)
        weights = self.size_in_bytes(precision)
        return ComputationGraph(
            name=self.name,
            total_ops=ops,
            parameter_count=self.parameters,
            weight_bytes=weights,
            arithmetic_intensity=(ops / weights).to("flop/byte"),
            layers=self.layers
        )

class SparseTransformerWorkload(TransformerWorkload):
    active_parameters: Quantity
    experts: int
    active_experts_per_token: int = 1

    def lower(self, precision: Quantity = BYTES_FP16) -> ComputationGraph:
        # For MoE, total parameters define the memory footprint,
        # but active parameters define the computation flops.
        ops = self.inference_flops or (2 * self.active_parameters.to(ureg.count).magnitude * ureg.flop)
        weights = self.size_in_bytes(precision) # uses self.parameters (total params)
        return ComputationGraph(
            name=self.name,
            total_ops=ops,
            parameter_count=self.parameters,
            weight_bytes=weights,
            arithmetic_intensity=(ops / weights).to("flop/byte"),
            layers=self.layers
        )

class CNNWorkload(Workload):
    parameters: Quantity
    inference_flops: Quantity
    layers: Optional[int] = None

    def size_in_bytes(self, precision: Quantity = BYTES_FP16) -> Quantity:
        param_count = self.parameters.to(ureg.count).magnitude
        bpp = precision.to(ureg.byte).magnitude
        return (param_count * bpp * ureg.byte).to(ureg.byte)

    def lower(self, precision: Quantity = BYTES_FP16) -> ComputationGraph:
        weights = self.size_in_bytes(precision)
        return ComputationGraph(
            name=self.name,
            total_ops=self.inference_flops,
            parameter_count=self.parameters,
            weight_bytes=weights,
            arithmetic_intensity=(self.inference_flops / weights).to("flop/byte"),
            layers=self.layers
        )
