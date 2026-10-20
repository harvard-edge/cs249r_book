from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Optional
from ..core.units import ureg, BYTES_FP16
from ..core.types import (
    Quantity,
    Metadata,
    require_dimensionality,
    require_unit_family,
    require_unit_families,
)

class ComputationGraph(BaseModel):
    """
    Hardware-Agnostic representation of a Workload.
    
    This is the 'Intermediate Representation' (IR) of computational demand. 
    It strips away high-level architectural details (like "Transformer" or 
    "CNN") and reduces the workload to pure math: Operations and Bytes.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    
    name: str
    total_ops: Quantity
    parameter_count: Quantity
    weight_bytes: Quantity
    arithmetic_intensity: Quantity # Ops/Byte
    
    # Optional metadata
    layers: Optional[int] = None

    @field_validator("total_ops", mode="after")
    @classmethod
    def _validate_total_ops(cls, v):
        return require_unit_family(v, ureg.flop, "total_ops", "operation")

    @field_validator("parameter_count", mode="after")
    @classmethod
    def _validate_parameter_count(cls, v):
        return require_unit_family(v, ureg.count, "parameter_count", "count")

    @field_validator("weight_bytes", mode="after")
    @classmethod
    def _validate_weight_bytes(cls, v):
        return require_unit_family(v, ureg.byte, "weight_bytes", "data")

    @field_validator("arithmetic_intensity", mode="after")
    @classmethod
    def _validate_arithmetic_intensity(cls, v):
        return require_unit_families(v, ureg.flop / ureg.byte, "arithmetic_intensity", ("operation", "data"))
    
    def __repr__(self):
        return f"ComputationGraph({self.name}, {self.total_ops:~P})"

class Workload(BaseModel):
    """
    Layer A (Workload Demand): Base representation of an ML model or task.
    
    A Workload defines the computational requirements of a task without any
    knowledge of the hardware it will run on. It must implement `lower()`
    to project its architectural definition down into a `ComputationGraph`.

    ``inference_flops`` semantics — the unit of work DIFFERS BY MODEL FAMILY
    (documented 2026-06-06; the field is dimensioned only as ``flop``, so the
    family convention is the contract):

    - Autoregressive LMs (``TransformerWorkload`` decoder-only,
      ``SparseTransformerWorkload``): FLOPs **per generated token**, following
      the 2 x params convention — 2 x TOTAL params for dense models, 2 x
      ACTIVE params for MoE.
    - Encoder models (BERT-family): FLOPs **per sequence** at the reference
      sequence length recorded in the entry's provenance (BERT-Base: 22 GFLOP
      at seq=128).
    - Vision / CNN / recommendation models: FLOPs **per input** (one image /
      one query forward pass).

    NEVER compare ``inference_flops`` across families without normalizing —
    the magnitudes differ by orders of magnitude purely from the work unit.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    architecture: str
    metadata: Metadata = Field(default_factory=Metadata)
    parameters: Optional[Quantity] = None
    model_size: Optional[Quantity] = None
    embedding_entries: Optional[Quantity] = None
    # Per-token (LMs) / per-sequence (encoders) / per-input (vision) — see
    # class docstring before comparing across model families.
    inference_flops: Optional[Quantity] = None
    inference_energy: Optional[Quantity] = None  # per-inference energy (e.g. mobile on-device)
    data_rate: Optional[Quantity] = None # e.g., TB/hour for autonomous driving
    training_dataset_size: Optional[Quantity] = None
    parameter_range_min: Optional[Quantity] = None
    parameter_range_max: Optional[Quantity] = None
    # Calendar year of first publication or release (added 2026-10-12 for
    # growth-trend figures that plot a model against time).
    release_year: Optional[int] = None

    @field_validator("parameters", "embedding_entries", mode="after")
    @classmethod
    def _validate_count_fields(cls, v, info):
        return require_unit_family(v, ureg.count, info.field_name, "count")

    @field_validator("model_size", mode="after")
    @classmethod
    def _validate_model_size(cls, v):
        return require_unit_family(v, ureg.byte, "model_size", "data")

    @field_validator("inference_flops", mode="after")
    @classmethod
    def _validate_inference_flops(cls, v):
        return require_unit_family(v, ureg.flop, "inference_flops", "operation")

    @field_validator("inference_energy", mode="after")
    @classmethod
    def _validate_inference_energy(cls, v):
        return require_dimensionality(v, ureg.joule, "inference_energy")

    @field_validator("data_rate", mode="after")
    @classmethod
    def _validate_data_rate(cls, v):
        return require_unit_family(v, ureg.byte / ureg.second, "data_rate", "data")

    def lower(self, precision: Quantity | str = BYTES_FP16) -> ComputationGraph:
        """
        Lowers the workload into a hardware-agnostic computation graph.

        Default implementation uses ``parameters`` and ``inference_flops``
        directly.  Subclasses (Transformer, CNN, etc.) override this with
        architecture-specific lowering.
        """
        from ..core.units import resolve_precision
        prec_qty = resolve_precision(precision)[1] if isinstance(precision, str) else precision
        if self.parameters is None and self.model_size is not None:
            weight_bytes = self.model_size.to(ureg.byte)
            ops = self.inference_flops if self.inference_flops else (0 * ureg.flop)
            return ComputationGraph(
                name=self.name,
                total_ops=ops,
                parameter_count=0 * ureg.count,
                weight_bytes=weight_bytes,
                arithmetic_intensity=(ops / weight_bytes).to("flop/byte") if weight_bytes.magnitude > 0 else 0 * ureg.flop / ureg.byte,
            )
        if self.parameters is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no 'parameters'; override lower() or set parameters."
            )
        bpp = prec_qty.to(ureg.byte).magnitude
        param_count = self.parameters.to(ureg.count).magnitude
        weight_bytes = (param_count * bpp * ureg.byte).to(ureg.byte)
        ops = self.inference_flops if self.inference_flops else (2 * param_count * ureg.flop)
        return ComputationGraph(
            name=self.name,
            total_ops=ops,
            parameter_count=self.parameters,
            weight_bytes=weight_bytes,
            arithmetic_intensity=(ops / weight_bytes).to("flop/byte"),
        )

    def size_in_bytes(self, precision: Quantity | str = BYTES_FP16) -> Quantity:
        if self.model_size is not None and precision in (BYTES_FP16, "fp16", "bf16"):
            return self.model_size
        if self.parameters is not None:
            from ..core.units import resolve_precision
            prec_qty = resolve_precision(precision)[1] if isinstance(precision, str) else precision
            param_count = self.parameters.to(ureg.count).magnitude
            bpp = prec_qty.to(ureg.byte).magnitude
            return (param_count * bpp * ureg.byte).to(ureg.byte)
        if self.model_size is not None:
            return self.model_size
        raise NotImplementedError("Workload must define either parameters or model_size to calculate size in bytes.")

class TransformerWorkload(Workload):
    """Workload representation of an autoregressive Transformer (e.g., LLMs)."""
    parameters: Quantity
    layers: int
    hidden_dim: Optional[int] = None
    heads: Optional[int] = None
    kv_heads: Optional[int] = None
    # Attention head width when it is not hidden_dim // heads (added 2026-10-20):
    # PaLM uses heads of 256 with d_model 18,432 and 48 heads (Chowdhery et al.
    # 2022, Table 1). None keeps the hidden_dim // heads convention.
    head_dim: Optional[int] = None
    # Multi-Head Latent Attention (DeepSeek-V2/V3). When both are present the
    # model caches one compressed latent per token per layer instead of a K and
    # a V tensor per key-value head, so KV sizing takes the MLA path below.
    kv_lora_rank: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    training_ops: Optional[Quantity] = None
    training_tokens: Optional[Quantity] = None
    training_accelerators_ref: Optional[Quantity] = None
    training_days_ref: Optional[Quantity] = None
    training_energy_mwh: Optional[Quantity] = None
    training_gpu_days: Optional[float] = None
    training_hardware_label: Optional[str] = None
    inference_flops: Optional[Quantity] = None
    # Architecture facts from the model card / paper (added 2026-09-28).
    # ``context_length`` is the pre-training / supported context window in
    # tokens; ``vocab_size`` is the tokenizer vocabulary. Both are optional
    # because not every source states them.
    context_length: Optional[int] = None
    vocab_size: Optional[int] = None
    # Reported training emissions (mass of CO2e, e.g. metric tonnes).
    training_co2e: Optional[Quantity] = None

    @field_validator("training_co2e", mode="after")
    @classmethod
    def _validate_training_co2e(cls, v):
        return require_dimensionality(v, ureg.kilogram, "training_co2e")

    @field_validator("training_ops", mode="after")
    @classmethod
    def _validate_training_ops(cls, v):
        return require_unit_family(v, ureg.flop, "training_ops", "operation")

    @field_validator("training_tokens", "training_accelerators_ref", mode="after")
    @classmethod
    def _validate_training_count_fields(cls, v, info):
        return require_unit_family(v, ureg.count, info.field_name, "count")

    @field_validator("training_days_ref", mode="after")
    @classmethod
    def _validate_training_days_ref(cls, v):
        return require_dimensionality(v, ureg.day, "training_days_ref")

    @model_validator(mode="after")
    def _check_training_fleet_consistency(self):
        """Keep the reference fleet internally consistent.

        When an entry states a reference accelerator count, a reference
        duration, and an accelerator-day total, the product of the first two
        must match the third within 1%. Added 2026-09-29 after GPT-4 stored
        25,000 accelerators x 90 days beside a 2.5M accelerator-day anchor,
        a 10% self-contradiction.
        """
        gpu_days = self.training_gpu_days
        accel = self.training_accelerators_ref
        days = self.training_days_ref
        if gpu_days is None or accel is None or days is None:
            return self
        implied = accel.to(ureg.count).magnitude * days.to(ureg.day).magnitude
        if abs(gpu_days - implied) / gpu_days > 0.01:
            raise ValueError(
                f"{self.name}: training_accelerators_ref x training_days_ref = "
                f"{implied:,.0f} accelerator-days, but training_gpu_days = "
                f"{gpu_days:,.0f} (must agree within 1%)."
            )
        return self

    def get_kv_cache_size(self, seq_len: int, batch_size: int, precision: Quantity | str = BYTES_FP16) -> Quantity:
        from ..core.units import resolve_precision
        from ..physics import calc_kv_cache_size, calc_mla_cache_size
        prec_qty = resolve_precision(precision)[1] if isinstance(precision, str) else precision

        if self.kv_lora_rank is not None and self.qk_rope_head_dim is not None:
            return calc_mla_cache_size(
                n_layers=self.layers,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                seq_len=seq_len,
                batch_size=batch_size,
                bytes_per_elem=prec_qty,
            )

        h_dim = self.hidden_dim or 4096
        n_heads = self.heads or 32
        head_dim = self.head_dim or h_dim // n_heads
        n_kv_heads = self.kv_heads or n_heads
        return calc_kv_cache_size(n_layers=self.layers, n_heads=n_kv_heads, head_dim=head_dim, seq_len=seq_len, batch_size=batch_size, bytes_per_elem=prec_qty)

    @property
    def head_width(self) -> int:
        """Width of one attention head: ``head_dim`` when stated, else hidden_dim // heads (added 2026-10-20)."""
        return self.head_dim or self.hidden_dim // self.heads

    @property
    def kv_width(self) -> int:
        """Key (or value) width cached per layer per token: key-value heads x head width (added 2026-10-20)."""
        return (self.kv_heads or self.heads) * self.head_width

    @property
    def queries_per_kv_head(self) -> int:
        """Query heads that share one key-value head (one for multi-head attention; added 2026-10-20)."""
        return self.heads // (self.kv_heads or self.heads)

    def training_memory(self, batch_size: int, seq_len: int, precision: str = "fp16", optimizer: str = "adam", strategy: str = "selective", zero_stage: int = 0, dp_size: int = 1, is_lora: bool = False, lora_rank: int = 16) -> Quantity:
        """
        Estimate training memory for a Transformer model.
        
        Source: Shoeybi et al. (2019), "Megatron-LM: Training Multi-Billion Parameter 
                Language Models Using Model Parallelism"
        Source: Rajbhandari et al. (2020), "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models."
        
        Args:
            batch_size: Mini-batch size (B)
            seq_len: Sequence length (S)
            precision: Precision format ('fp32', 'fp16', 'int8', 'int4')
            optimizer: Optimizer type ('adam', 'sgd')
            strategy: Recompute strategy ('none', 'selective', 'full')
            zero_stage: ZeRO optimization stage (0, 1, 2, 3)
            dp_size: Data parallel size for ZeRO sharding
            is_lora: Whether Low-Rank Adaptation (PEFT) is used
            
        Returns:
            Quantity[byte]: Total training memory per GPU
        """
        from ..core.units import resolve_precision
        from ..physics import calc_activation_memory
        
        _, precision_bytes = resolve_precision(precision)
        bpp = precision_bytes.to(ureg.byte).magnitude
        
        n_params = self.parameters.to(ureg.count).magnitude
        
        # 1. Weights and Gradients
        weight_mem = n_params * bpp * ureg.byte
        grad_mem = n_params * bpp * ureg.byte
        
        # 2. Optimizer States (Adam = 12 bytes/param for FP32 states)
        if optimizer.lower() == "adam":
            # Adam: master weights (4), momentum (4), variance (4) = 12 bytes/param
            opt_mem = n_params * 12 * ureg.byte
        else:
            # SGD: master weights (4) = 4 bytes/param
            opt_mem = n_params * 4 * ureg.byte
            
        # LoRA: trainable fraction ≈ 2 * r * d_model * n_adapted_layers / total_params
        # For typical configs (all linear layers in attention + MLP):
        #   n_adapted = 4 * layers (Q, K, V, O projections)
        #   trainable = 2 * r * d_model * 4 * layers / total_params
        if is_lora:
            d_model = self.hidden_dim or 4096
            n_adapted = 4 * self.layers  # Q, K, V, O per layer
            lora_params = 2 * lora_rank * d_model * n_adapted
            lora_fraction = min(lora_params / n_params, 1.0)
            grad_mem = grad_mem * lora_fraction
            opt_mem = opt_mem * lora_fraction

        # ZeRO Sharding
        if zero_stage >= 1:
            opt_mem = opt_mem / dp_size
        if zero_stage >= 2:
            grad_mem = grad_mem / dp_size
        if zero_stage >= 3:
            weight_mem = weight_mem / dp_size
            
        # 3. Activation Memory (proportional to B, S, H; quadratic attention
        # term needs the head count when strategy='none')
        act_mem = calc_activation_memory(
            n_layers=self.layers,
            seq_len=seq_len,
            batch_size=batch_size,
            hidden_dim=self.hidden_dim or 4096,
            n_heads=self.heads,
            precision_bytes=bpp,
            strategy=strategy
        )
        
        return (weight_mem + grad_mem + opt_mem + act_mem).to(ureg.GiB)

    def lower(self, precision: Quantity | str = BYTES_FP16) -> ComputationGraph:
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
    """Workload representation of a Mixture-of-Experts (MoE) Transformer."""
    active_parameters: Quantity
    experts: int
    active_experts_per_token: int = 1

    @field_validator("active_parameters", mode="after")
    @classmethod
    def _validate_active_parameters(cls, v):
        return require_unit_family(v, ureg.count, "active_parameters", "count")

    def lower(self, precision: Quantity | str = BYTES_FP16) -> ComputationGraph:
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
    """Workload representation of a Convolutional Neural Network (e.g., Vision)."""
    parameters: Quantity
    inference_flops: Quantity
    layers: Optional[int] = None

    def lower(self, precision: Quantity | str = BYTES_FP16) -> ComputationGraph:
        weights = self.size_in_bytes(precision)
        return ComputationGraph(
            name=self.name,
            total_ops=self.inference_flops,
            parameter_count=self.parameters,
            weight_bytes=weights,
            arithmetic_intensity=(self.inference_flops / weights).to("flop/byte"),
            layers=self.layers
        )

class SSMWorkload(Workload):
    """Workload representation of a State Space Model (e.g., Mamba)."""
    parameters: Quantity
    layers: int
    state_size: int
    hidden_dim: int
    inference_flops: Optional[Quantity] = None

    def get_state_cache_size(self, batch_size: int, precision: Quantity | str = BYTES_FP16) -> Quantity:
        # State space model memory footprint is independent of sequence length (O(1) w.r.t seq_len)
        # Represents the recursive state vector maintained by the model (e.g., Mamba)
        # Typically: layers * batch * hidden_dim * state_size * precision
        from ..core.units import resolve_precision
        prec_qty = resolve_precision(precision)[1] if isinstance(precision, str) else precision
        bpp = prec_qty.to(ureg.byte).magnitude
        size_bytes = self.layers * batch_size * self.hidden_dim * self.state_size * bpp
        return (size_bytes * ureg.byte).to(ureg.byte)

    def lower(self, precision: Quantity | str = BYTES_FP16) -> ComputationGraph:
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


class DiffusionWorkload(Workload):
    """Workload representation of a Diffusion generative model."""
    parameters: Quantity
    denoising_steps: int
    resolution: int
    inference_flops: Optional[Quantity] = None  # FLOPs for ONE denoising step

    def lower(self, precision: Quantity | str = BYTES_FP16) -> ComputationGraph:
        # Total inference flops = Flops per step * denoising steps
        base_ops = self.inference_flops or (2 * self.parameters.to(ureg.count).magnitude * ureg.flop)
        total_ops = base_ops * self.denoising_steps
        weights = self.size_in_bytes(precision)
        return ComputationGraph(
            name=self.name,
            total_ops=total_ops,
            parameter_count=self.parameters,
            weight_bytes=weights,
            arithmetic_intensity=(total_ops / weights).to("flop/byte"),
            layers=None
        )


class EmbodiedWorkload(Workload):
    """Workload representation of an embodied policy or vision-language-action (VLA) model."""
    parameters: Quantity
    action_chunk_horizon: Optional[int] = None
    action_dim: Optional[int] = None
    observation_horizon: Optional[int] = 1
    inference_flops: Optional[Quantity] = None
    layers: Optional[int] = None
    hidden_dim: Optional[int] = None

    def lower(self, precision: Quantity | str = BYTES_FP16) -> ComputationGraph:
        base_ops = self.inference_flops or (2 * self.parameters.to(ureg.count).magnitude * ureg.flop)
        weights = self.size_in_bytes(precision)
        return ComputationGraph(
            name=self.name,
            total_ops=base_ops,
            parameter_count=self.parameters,
            weight_bytes=weights,
            arithmetic_intensity=(base_ops / weights).to("flop/byte"),
            layers=self.layers,
        )

