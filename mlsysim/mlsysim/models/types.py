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

    @field_validator("parameters", "embedding_entries", "parameter_range_min", "parameter_range_max", mode="after")
    @classmethod
    def _validate_count_fields(cls, v, info):
        return require_unit_family(v, ureg.count, info.field_name, "count")

    @field_validator("model_size", "training_dataset_size", mode="after")
    @classmethod
    def _validate_model_size(cls, v, info):
        return require_unit_family(v, ureg.byte, info.field_name, "data")

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
        """Project stored workload facts into a computation graph.

        Parameters
        ----------
        precision : Quantity or str, default: BYTES_FP16
            Bytes per stored element, or a registered format such as 'fp16'.
            A byte quantity specifies the width directly; it must be positive.

        Returns
        -------
        ComputationGraph
            Work in FLOPs, resident weight bytes, parameter count, arithmetic
            intensity in FLOP/byte, and the available layer count.

        Notes
        -----
        Equation: W = P * b and I = F / W. F uses inference_flops when supplied; otherwise F = 2 * P. If P is absent but model_size exists, W = model_size, P = 0, and missing FLOPs become zero.

        Assumptions: one graph represents the work unit described by the workload family. With parameter count present, this base implementation computes W from P and b rather than preferring model_size. A zero-byte size-only workload has zero arithmetic intensity by convention.

        Excludes: activation or cache storage, backward work, hardware efficiency, and precision-dependent operation counts. The 2P fallback is an approximation, not a graph trace.

        References
        ----------
        Derivation: multiply parameter count by storage width to obtain weight
        bytes, then divide the declared or approximated work by those bytes.
        This is a demand-accounting projection; it does not predict device time.

        Examples
        --------
        >>> from mlsysim.models.types import Workload
        >>> from mlsysim.core.units import Q_
        >>> model = Workload(name='Example', architecture='dense', parameters=Q_(100, 'count'))
        >>> graph = model.lower('fp16')
        >>> graph.weight_bytes == model.parameters.magnitude * Q_(2, 'byte')
        True
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
        """Calculate stored model weights at the requested precision.

        Parameters
        ----------
        precision : Quantity or str, default: BYTES_FP16
            Bytes per stored element, or a registered format such as 'fp16'.
            A byte quantity specifies the width directly; it must be positive.

        Returns
        -------
        Quantity
            Resident weight size with data units, normally bytes. A supplied
            model_size retains its original compatible units when returned.

        Notes
        -----
        Equation: W = P * b when parameter count P is available. A supplied
        model_size takes precedence at FP16/BF16; if P is missing it is used
        at every precision because there is no count from which to rescale.

        Assumptions: all parameters use one element width. An explicit model_size
        is an authoritative footprint at the default storage format. Without
        parameters or model_size, the method raises NotImplementedError.

        Excludes: gradients, optimizer states, activations, KV caches, quantization
        scales, packing alignment, and allocator reserve.

        References
        ----------
        Derivation: multiply the number of stored elements by bytes per element.
        The model_size precedence is an implementation convention for workloads
        whose stored footprint is known but whose parameter inventory is absent.

        Examples
        --------
        >>> from mlsysim.models.types import Workload
        >>> from mlsysim.core.units import Q_
        >>> model = Workload(name='Example', architecture='dense', parameters=Q_(100, 'count'))
        >>> model.size_in_bytes('fp16') == 2 * model.size_in_bytes('int8')
        True
        """
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

    @property
    def effective_compute_parameters(self) -> Quantity:
        """Return the parameter count active in one forward evaluation.

        Returns
        -------
        Quantity
            Active parameter count in ``count``-family units.

        Notes
        -----
        Equation: ``P_compute = P`` for dense workloads whose full parameter
        inventory participates in every forward pass.

        Assumptions: dense models execute all declared parameters per sample or
        token. Workloads without ``parameters`` raise ``NotImplementedError``.

        Excludes: activation-dependent dynamic sparsity, early exit, and MoE
        expert routing (overridden on ``SparseTransformerWorkload``).

        References
        ----------
        Derivation: a dense workload touches every stored parameter once per
        forward pass, so active compute parameters equal total parameters.

        Examples
        --------
        >>> from mlsysim import Models
        >>> model = Models.Language.Llama3_8B
        >>> model.effective_compute_parameters == model.parameters
        True
        """
        if self.parameters is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no 'parameters'; cannot determine effective_compute_parameters."
            )
        return self.parameters

    def sharded_parameter_count(
        self,
        tp_size: int = 1,
        pp_size: int = 1,
        ep_size: int = 1,
    ) -> Quantity:
        """Calculate resident parameters per rank under model parallelism.

        Parameters
        ----------
        tp_size : int, default: 1
            Positive tensor-parallel degree splitting layers within a stage.
        pp_size : int, default: 1
            Positive pipeline-parallel degree partitioning layers across stages.
        ep_size : int, default: 1
            Positive expert-parallel degree; must be 1 for dense workloads.

        Returns
        -------
        Quantity
            Parameters stored on one ``(tp_size, pp_size, ep_size)`` rank in
            ``count`` units.

        Notes
        -----
        Equation: ``P_rank = P / (tp_size * pp_size)`` when ``ep_size == 1``.

        Assumptions: tensor and pipeline parallelism partition dense parameters
        uniformly across ``tp_size * pp_size`` shards. Passing ``ep_size > 1``
        on a non-MoE workload raises ``ValueError``.

        Excludes: untied embedding imbalance across pipeline stages, padding to
        tensor-core alignment multiples, and ZeRO data-parallel sharding.

        References
        ----------
        Shoeybi et al. (2019), "Megatron-LM: Training Multi-Billion Parameter
        Language Models Using Model Parallelism." https://arxiv.org/abs/1909.08053
        (bibliography key: shoeybi2019megatron).

        Examples
        --------
        >>> from mlsysim import Models
        >>> model = Models.Language.Llama3_8B
        >>> model.sharded_parameter_count(tp_size=4, pp_size=2) == model.parameters / 8
        True
        """
        from ..core._validation import validate_at_least

        validate_at_least(tp_size, 1, "tp_size")
        validate_at_least(pp_size, 1, "pp_size")
        validate_at_least(ep_size, 1, "ep_size")
        if ep_size > 1:
            raise ValueError(
                f"{self.name}: ep_size > 1 requires a SparseTransformerWorkload (MoE model), "
                f"got {type(self).__name__} with ep_size={ep_size}."
            )
        if self.parameters is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no 'parameters'; cannot shard parameter count."
            )
        return (self.parameters / (tp_size * pp_size)).to(ureg.count)

    def activation_traffic_bytes(
        self,
        precision: Quantity | str = BYTES_FP16,
        tokens_per_sample: int = 1,
        ops_multiplier: int = 1,
    ) -> Quantity:
        """Calculate per-sample off-chip activation read/write traffic.

        Parameters
        ----------
        precision : Quantity or str, default: BYTES_FP16
            Bytes per activation element, or a registered format such as 'fp16'.
        tokens_per_sample : int, default: 1
            Positive token count per sample for sequence architectures.
        ops_multiplier : int, default: 1
            Positive pass multiplier (1 for inference, 3 for training, 4 with
            full activation recomputation).

        Returns
        -------
        Quantity
            Activation bytes transferred per sample in ``byte`` units.

        Notes
        -----
        Equation: ``A_sample = 0 byte`` on the generic base workload when no
        feature-map or hidden-state geometry is declared.

        Assumptions: architectural subclasses override this method using their
        declared tensor dimensions; generic workloads with only parameter counts
        report zero rather than inventing a fraction of weight bytes.

        Excludes: weight traffic, KV-cache reads/writes, and temporary workspace
        spills; callers with measured activation I/O can pass an explicit override
        to ``Engine.solve``.

        References
        ----------
        Derivation: without layer activation dimensions, off-chip feature-map
        traffic is undetermined and defaults to zero until a subclass or caller
        supplies tensor geometry.

        Examples
        --------
        >>> from mlsysim.models.types import Workload
        >>> from mlsysim.core.units import Q_
        >>> w = Workload(name='Generic', architecture='mlp', parameters=Q_(1000, 'count'))
        >>> w.activation_traffic_bytes('fp16') == Q_(0, 'byte')
        True
        """
        from ..core._validation import validate_at_least
        from ..core.units import resolve_precision

        validate_at_least(tokens_per_sample, 1, "tokens_per_sample")
        validate_at_least(ops_multiplier, 1, "ops_multiplier")
        if isinstance(precision, str):
            resolve_precision(precision)
        return 0.0 * ureg.byte

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

    @field_validator("training_energy_mwh", mode="after")
    @classmethod
    def _validate_training_energy_mwh(cls, v):
        return require_dimensionality(v, ureg.joule, "training_energy_mwh")

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
        """Count cached attention state for a batch of sequences.

        Parameters
        ----------
        seq_len : int
            Nonnegative cached tokens per sequence, including any shared prefix
            separately in every request; this method does not deduplicate pages.
        batch_size : int
            Nonnegative number of resident sequences.
        precision : Quantity or str, default: BYTES_FP16
            Bytes per stored element, or a registered format such as 'fp16'.
            A byte quantity specifies the width directly; it must be positive.

        Returns
        -------
        Quantity
            Cache storage in bytes for all layers and sequences.

        See Also
        --------
        mlsysim.physics.memory.calc_kv_cache_size : Standard KV tensor accounting.
        mlsysim.physics.memory.calc_mla_cache_size : Compressed latent accounting.

        Notes
        -----
        Equation: M_KV = 2 * L * H_KV * d_head * S * B * b for standard
        multi-head or grouped-query attention. When both MLA ranks are present,
        M_MLA = L * (kv_lora_rank + qk_rope_head_dim) * S * B * b.

        Assumptions: all layers share the same attention geometry. Missing
        hidden_dim and heads fall back to 4096 and 32 here; an explicit head_dim
        overrides hidden_dim // heads. Missing kv_heads means full multi-head
        attention. Both MLA dimensions must be present to select the latent path.

        Excludes: query activations, weight storage, page rounding, cache sharing,
        allocator metadata, and inference-engine workspaces.

        References
        ----------
        Derivation: standard attention stores one key and one value vector per
        KV head, layer, token, and sequence. MLA stores a shared latent vector
        and the rotary-position key components instead of expanded head tensors.
        The arithmetic is delegated to the two named cache-accounting functions.

        Examples
        --------
        >>> from mlsysim import Models
        >>> model = Models.Language.Llama3_8B
        >>> model.get_kv_cache_size(128, 2) == 2 * model.get_kv_cache_size(128, 1)
        True
        """
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
        """Return the number of components in one attention head.

        Returns
        -------
        int
            Feature count per query/key head; a count without physical units.

        Notes
        -----
        Equation: d_head = head_dim when explicitly set, else hidden_dim // heads.

        Assumptions: the fallback requires populated hidden_dim and heads with a
        positive head count. Integer division follows the registered geometry;
        this property does not check divisibility.

        Excludes: grouped-query sharing ratios, latent-cache rank, and validation
        of a missing or inconsistent architectural specification.

        References
        ----------
        Derivation: partition the hidden feature width evenly among query heads;
        an explicit per-head width overrides that partition for architectures
        whose projected head dimension differs from hidden_dim / heads.

        Examples
        --------
        >>> from mlsysim import Models
        >>> model = Models.Language.Llama3_8B
        >>> model.head_width == model.hidden_dim // model.heads
        True
        """
        if self.head_dim is not None:
            return self.head_dim
        if self.hidden_dim is None or self.heads is None:
            raise ValueError(f"{self.name}: head_width requires head_dim or both hidden_dim and heads")
        return self.hidden_dim // self.heads

    @property
    def kv_width(self) -> int:
        """Return the expanded key or value feature width per layer.

        Returns
        -------
        int
            Components in one expanded key tensor, or one value tensor, per
            token and layer. This is a feature count, not bytes.

        Notes
        -----
        Equation: d_KV = (kv_heads or heads) * head_width.

        Assumptions: absent kv_heads means one KV head for every query head.
        This property describes expanded head geometry even for an MLA workload.

        Excludes: the second tensor (key versus value), element byte width,
        batch size, sequence length, and compressed MLA cache representation.

        References
        ----------
        Derivation: concatenate the per-head feature vectors across the KV heads.
        Cache storage additionally needs both K and V, storage width, and the
        number of resident tokens; get_kv_cache_size performs that accounting.

        Examples
        --------
        >>> from mlsysim import Models
        >>> model = Models.Language.Llama3_8B
        >>> model.kv_width == model.kv_heads * model.head_width
        True
        """
        n_kv = self.kv_heads if self.kv_heads is not None else self.heads
        if n_kv is None:
            raise ValueError(f"{self.name}: kv_width requires kv_heads or heads")
        return n_kv * self.head_width

    @property
    def queries_per_kv_head(self) -> int:
        """Return the query-head sharing ratio for grouped attention.

        Returns
        -------
        int
            Number of query heads assigned to each KV head, dimensionless.

        Notes
        -----
        Equation: sharing = heads // (kv_heads or heads).

        Assumptions: query heads partition evenly across the registered KV
        heads. Missing kv_heads denotes ordinary multi-head attention.

        Excludes: unequal groups, runtime routing, and a divisibility check;
        integer division preserves the implementation's architectural convention.

        References
        ----------
        Derivation: divide the query-head count by the number of shared KV heads.
        With equal query and KV head counts this ratio is one.

        Examples
        --------
        >>> from mlsysim import Models
        >>> model = Models.Language.Llama3_8B
        >>> model.queries_per_kv_head * model.kv_heads == model.heads
        True
        """
        if self.heads is None:
            raise ValueError(f"{self.name}: queries_per_kv_head requires heads")
        n_kv = self.kv_heads if self.kv_heads is not None else self.heads
        return self.heads // n_kv

    def training_memory(self, batch_size: int, seq_len: int, precision: str = "fp16", optimizer: str = "adam", strategy: str = "selective", zero_stage: int = 0, dp_size: int = 1, is_lora: bool = False, lora_rank: int = 16) -> Quantity:
        """Estimate per-device weights, gradients, optimizer state, and activations.

        Parameters
        ----------
        batch_size : int
            Positive sequences resident on this device for activation accounting.
        seq_len : int
            Positive tokens per sequence.
        precision : str, default: 'fp16'
            Registered weight and gradient format; optimizer storage uses its
            own mixed-precision convention in optimizer_state_bytes.
        optimizer : str, default: 'adam'
            Optimizer convention: 'adam', 'adamw', 'sgd', or 'none'.
        strategy : str, default: 'selective'
            Activation recomputation mode: 'none', 'selective', or 'full',
            delegated to calc_activation_memory.
        zero_stage : int, default: 0
            ZeRO stage: 0 leaves state local; 1 shards optimizer state;
            2 also shards gradients; 3 also shards weights.
        dp_size : int, default: 1
            Positive number of data-parallel ranks sharing the sharded state.
        is_lora : bool, default: False
            Restrict trainable gradient and optimizer state to an approximate
            adapter parameter fraction. Full frozen weights remain resident.
        lora_rank : int, default: 16
            Positive adapter rank when is_lora is true. The model assumes
            adapters on Q, K, V, and O projections in every layer.

        Returns
        -------
        Quantity
            Modeled resident memory per device in GiB. This method does not
            compare the result with a device capacity or declare feasibility.

        See Also
        --------
        mlsysim.physics.memory.calc_activation_memory : Activation assumptions.
        mlsysim.physics.memory.optimizer_state_bytes : Optimizer storage convention.

        Notes
        -----
        Equation: M = M_weights + M_gradients + M_optimizer + M_activations.
        Weights and gradients initially cost P * b each; optimizer state costs
        P * optimizer_state_bytes(optimizer). LoRA multiplies gradient and
        optimizer storage by min(8 * rank * hidden_dim * layers / P, 1).
        ZeRO divides each selected state category by dp_size.

        Assumptions: homogeneous data-parallel shards and the selected activation
        formula. Missing hidden_dim uses 4096. The adapter estimate covers four
        square attention projections, not every linear layer or architecture.
        Optimizer state follows the shared mixed-precision byte convention.

        Excludes: communication buffers, allocator reserve, runtime workspaces,
        fragmentation, tensor-parallel activation partitioning, and peak transient
        all-gather state. ZeRO sharding is a storage estimate, not a latency model.

        References
        ----------
        Rajbhandari et al. (2020), "ZeRO: Memory Optimizations Toward Training
        Trillion Parameter Models."
        https://doi.org/10.1109/sc41405.2020.00024 (bibliography key: rajbhandari2020).
        The LoRA projection count and frozen-weight accounting are explicit
        implementation assumptions; the ZeRO citation supports state partitioning.

        Examples
        --------
        >>> from mlsysim import Models
        >>> model = Models.Language.Llama3_8B
        >>> local = model.training_memory(1, 128, zero_stage=0)
        >>> sharded = model.training_memory(1, 128, zero_stage=3, dp_size=2)
        >>> sharded < local
        True
        """
        from ..core.units import resolve_precision
        from ..physics import calc_activation_memory
        from ..physics.memory import optimizer_state_bytes
        from ..core._validation import validate_at_least

        validate_at_least(dp_size, 1, "dp_size")
        if zero_stage not in (0, 1, 2, 3):
            raise ValueError("zero_stage must be 0, 1, 2, or 3")
        if is_lora:
            validate_at_least(lora_rank, 1, "lora_rank")
        
        _, precision_bytes = resolve_precision(precision)
        bpp = precision_bytes.to(ureg.byte).magnitude
        
        n_params = self.parameters.to(ureg.count).magnitude
        
        # 1. Weights and Gradients
        weight_mem = n_params * bpp * ureg.byte
        grad_mem = n_params * bpp * ureg.byte
        
        # 2. Shared mixed-precision optimizer accounting.
        opt_mem = n_params * optimizer_state_bytes(optimizer)
            
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
        """Lower one Transformer forward work unit and its resident weights.

        Parameters
        ----------
        precision : Quantity or str, default: BYTES_FP16
            Bytes per stored element, or a registered format such as 'fp16'.
            A byte quantity specifies the width directly; it must be positive.

        Returns
        -------
        ComputationGraph
            Work in FLOPs, resident weight bytes, parameter count, arithmetic
            intensity in FLOP/byte, and the available layer count.

        Notes
        -----
        Equation: F = inference_flops if specified, otherwise 2 * P; W = size_in_bytes(precision); I = F / W.

        Assumptions: the registered FLOP count defines the work unit when supplied; the fallback represents parameter-dominated dense work for one token. Layer count is copied into the graph for later overhead modeling.

        Excludes: attention over cached context, KV storage, backward work, activation traffic, and device timing. Do not interpret this graph alone as a complete decode model.

        References
        ----------
        Derivation: multiply parameter count by storage width to obtain weight
        bytes, then divide the declared or approximated work by those bytes.
        This is a demand-accounting projection; it does not predict device time.

        Examples
        --------
        >>> from mlsysim import Models
        >>> model = Models.Language.Llama3_8B
        >>> graph = model.lower('fp16')
        >>> graph.weight_bytes == model.size_in_bytes('fp16')
        True
        """
        from ..physics.transformer import calc_transformer_decode_flops

        ops = self.inference_flops or calc_transformer_decode_flops(self.parameters)
        weights = self.size_in_bytes(precision)
        return ComputationGraph(
            name=self.name,
            total_ops=ops,
            parameter_count=self.parameters,
            weight_bytes=weights,
            arithmetic_intensity=(ops / weights).to("flop/byte"),
            layers=self.layers
        )

    def activation_traffic_bytes(
        self,
        precision: Quantity | str = BYTES_FP16,
        tokens_per_sample: int = 1,
        ops_multiplier: int = 1,
    ) -> Quantity:
        """Calculate per-sample residual-stream activation HBM traffic.

        Parameters
        ----------
        precision : Quantity or str, default: BYTES_FP16
            Bytes per activation element, or a registered format such as 'fp16'.
        tokens_per_sample : int, default: 1
            Positive sequence length per sample (1 for single-token step, ``S``
            for prefill or training sequences).
        ops_multiplier : int, default: 1
            Positive pass multiplier (1 for forward inference, 3 for standard
            training, 4 with full activation recomputation).

        Returns
        -------
        Quantity
            Off-chip activation read+write bytes per sample in ``byte`` units.

        Notes
        -----
        Equation: ``A_sample = 4 * L * S * H * b * ops_multiplier`` when
        ``hidden_dim`` is populated, else ``0 byte``. Each of the ``L`` layers
        contains two residual sublayers (attention and FFN), each reading a
        ``(S, H)`` input tensor and writing a ``(S, H)`` output tensor of width ``b``.

        Assumptions: normalization and residual add are fused with sublayer
        boundaries while attention softmax intermediates stay in SRAM (FlashAttention
        regime). Workloads without ``hidden_dim`` return zero.

        Excludes: KV-cache reads/writes (modeled in ``ServingModel``), unfused
        ``S^2`` attention score matrices, and FFN intermediate-projection spills.

        References
        ----------
        Derivation: count one ``(S, H)`` read and one ``(S, H)`` write for each
        of the two residual sublayers in every Transformer block, multiplied by
        bytes per element and pass count.

        Examples
        --------
        >>> from mlsysim import Models
        >>> m = Models.Language.Llama3_8B
        >>> m.activation_traffic_bytes('fp16', tokens_per_sample=128) == 128 * m.activation_traffic_bytes('fp16', tokens_per_sample=1)
        True
        """
        from ..core._validation import validate_at_least
        from ..core.units import resolve_precision

        validate_at_least(tokens_per_sample, 1, "tokens_per_sample")
        validate_at_least(ops_multiplier, 1, "ops_multiplier")
        prec_qty = resolve_precision(precision)[1] if isinstance(precision, str) else precision
        if self.hidden_dim is None:
            return 0.0 * ureg.byte
        bpp = prec_qty.to(ureg.byte).magnitude
        total_bytes = 4.0 * self.layers * tokens_per_sample * self.hidden_dim * bpp * ops_multiplier
        return (total_bytes * ureg.byte).to(ureg.byte)

class SparseTransformerWorkload(TransformerWorkload):
    """Workload representation of a Mixture-of-Experts (MoE) Transformer."""
    active_parameters: Quantity
    experts: int
    active_experts_per_token: int = 1

    @field_validator("active_parameters", mode="after")
    @classmethod
    def _validate_active_parameters(cls, v):
        return require_unit_family(v, ureg.count, "active_parameters", "count")

    @property
    def effective_compute_parameters(self) -> Quantity:
        """Return the active parameter count executed per token in an MoE model.

        Returns
        -------
        Quantity
            Active parameters per token in ``count``-family units.

        Notes
        -----
        Equation: ``P_compute = active_parameters``.

        Assumptions: each token activates shared layers plus ``active_experts_per_token``
        routed experts as captured by ``active_parameters``.

        Excludes: token-dropping or expert capacity-factor overflow (modeled
        separately in ``MoERoutingModel``).

        References
        ----------
        Fedus et al. (2022), "Switch Transformers: Scaling to Trillion Parameter
        Models with Simple and Efficient Sparsity." https://arxiv.org/abs/2101.03961
        (bibliography key: fedus2022switch).

        Examples
        --------
        >>> from mlsysim import Models
        >>> moe = Models.Language.Mixtral_8x7B
        >>> moe.effective_compute_parameters == moe.active_parameters
        True
        """
        return self.active_parameters

    def sharded_parameter_count(
        self,
        tp_size: int = 1,
        pp_size: int = 1,
        ep_size: int = 1,
    ) -> Quantity:
        """Calculate per-rank MoE parameters separating shared and routed weights.

        Parameters
        ----------
        tp_size : int, default: 1
            Positive tensor-parallel degree splitting shared and expert layers.
        pp_size : int, default: 1
            Positive pipeline-parallel degree partitioning layers across stages.
        ep_size : int, default: 1
            Positive expert-parallel degree partitioning routed experts only.

        Returns
        -------
        Quantity
            Resident parameter count per ``(tp_size, pp_size, ep_size)`` rank in
            ``count`` units.

        Notes
        -----
        Equation: with ``E = experts`` and ``k = active_experts_per_token``,
        ``P_shared = max(0, (E * P_active - k * P_total) / (E - k))`` when ``E > k``
        (else ``P_total``), ``P_routed = P_total - P_shared``, and
        ``P_rank = P_shared / (tp_size * pp_size) + P_routed / (tp_size * pp_size * ep_size)``.

        Assumptions: ``P_total = P_shared + P_routed`` and
        ``P_active = P_shared + (k / E) * P_routed``. Shared attention, norms,
        embeddings, and any always-active shared experts are replicated across
        ``ep_size`` and sharded only by ``tp_size * pp_size``; routed experts are
        partitioned across ``tp_size * pp_size * ep_size``. Synthetic workloads
        with ``P_active / P_total < k / E`` clamp ``P_shared`` at zero while
        conserving ``P_shared + P_routed = P_total``.

        Excludes: uneven expert counts when ``ep_size`` does not divide ``E``
        and untied embedding stage imbalance.

        References
        ----------
        Lepikhin et al. (2021), "GShard: Scaling Giant Models with Conditional
        Computation and Automatic Sharding." https://arxiv.org/abs/2006.16668
        (bibliography key: lepikhin2021gshard).

        Examples
        --------
        >>> from mlsysim import Models
        >>> moe = Models.Language.Mixtral_8x7B
        >>> moe.sharded_parameter_count(ep_size=1) == moe.parameters
        True
        >>> moe.sharded_parameter_count(ep_size=8) > moe.parameters / 8
        True
        """
        from ..core._validation import validate_at_least

        validate_at_least(tp_size, 1, "tp_size")
        validate_at_least(pp_size, 1, "pp_size")
        validate_at_least(ep_size, 1, "ep_size")

        p_total = self.parameters.to(ureg.count).magnitude
        p_active = self.active_parameters.to(ureg.count).magnitude
        n_experts = self.experts
        top_k = self.active_experts_per_token

        if n_experts > top_k:
            p_shared = max(0.0, (n_experts * p_active - top_k * p_total) / (n_experts - top_k))
            p_shared = min(p_shared, p_total)
        else:
            p_shared = p_total
        p_routed = p_total - p_shared

        dense_shards = tp_size * pp_size
        expert_shards = tp_size * pp_size * ep_size
        p_rank = (p_shared / dense_shards) + (p_routed / expert_shards)
        return p_rank * ureg.count

    def lower(self, precision: Quantity | str = BYTES_FP16) -> ComputationGraph:
        # For MoE, total parameters define the memory footprint,
        # but active parameters define the computation flops.
        """Separate active-expert compute from total resident MoE weights.

        Parameters
        ----------
        precision : Quantity or str, default: BYTES_FP16
            Bytes per stored element, or a registered format such as 'fp16'.
            A byte quantity specifies the width directly; it must be positive.

        Returns
        -------
        ComputationGraph
            Work in FLOPs, resident weight bytes, parameter count, arithmetic
            intensity in FLOP/byte, and the available layer count.

        Notes
        -----
        Equation: F = inference_flops if supplied, otherwise 2 * P_active; W = size_in_bytes(precision) uses P_total; I = F / W.

        Assumptions: all expert weights remain resident even when only a subset executes for a token. A declared inference_flops value overrides the active-parameter approximation.

        Excludes: expert routing, inter-device dispatch, load imbalance, sparse memory fetch efficiency, attention-context work, and KV state.

        References
        ----------
        Derivation: multiply parameter count by storage width to obtain weight
        bytes, then divide the declared or approximated work by those bytes.
        This is a demand-accounting projection; it does not predict device time.

        Examples
        --------
        >>> from mlsysim.models.types import SparseTransformerWorkload
        >>> from mlsysim.core.units import Q_
        >>> model = SparseTransformerWorkload(name='MoE', architecture='moe', parameters=Q_(100, 'count'), active_parameters=Q_(25, 'count'), layers=1, experts=4)
        >>> graph = model.lower('fp16')
        >>> graph.total_ops == 2 * model.active_parameters.magnitude * Q_(1, 'flop')
        True
        >>> graph.weight_bytes == model.size_in_bytes('fp16')
        True
        """
        from ..physics.transformer import calc_transformer_decode_flops

        ops = self.inference_flops or calc_transformer_decode_flops(self.active_parameters)
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
    # Optional (added 2026-10-20): the channel-width multiplier of a scalable
    # family (MobileNet's alpha), relative to the family's base network.
    width_multiplier: Optional[float] = None
    # Sum of input and per-layer output feature-map tensor elements at the
    # declared input resolution (sum_l C_l * H_l * W_l). Used for exact
    # unfused layer-by-layer activation HBM traffic.
    feature_map_elements: Optional[Quantity] = None

    @field_validator("feature_map_elements", mode="after")
    @classmethod
    def _validate_feature_map_elements(cls, v):
        return require_unit_family(v, ureg.count, "feature_map_elements", "count")

    def activation_traffic_bytes(
        self,
        precision: Quantity | str = BYTES_FP16,
        tokens_per_sample: int = 1,
        ops_multiplier: int = 1,
    ) -> Quantity:
        """Calculate per-image feature-map read/write traffic across CNN layers.

        Parameters
        ----------
        precision : Quantity or str, default: BYTES_FP16
            Bytes per feature-map element, or a registered format such as 'fp16'.
        tokens_per_sample : int, default: 1
            Positive sample token count (unused for fixed-resolution CNN images;
            validated for interface uniformity).
        ops_multiplier : int, default: 1
            Positive pass multiplier (1 for forward inference, 3 for standard
            training, 4 with full activation recomputation).

        Returns
        -------
        Quantity
            Off-chip feature-map bytes read and written per image in ``byte`` units.

        Notes
        -----
        Equation: ``A_sample = 2 * feature_map_elements * b * ops_multiplier``
        when ``feature_map_elements`` is populated, else ``0 byte``.

        Assumptions: unfused layer-by-layer execution writes each layer output
        feature map once and reads it once as the next layer's input (factor of 2
        over summed feature-map elements). Conv+BN+ReLU fusion is assumed within
        each counted layer boundary.

        Excludes: cross-layer operator fusion into SRAM tiles, im2col workspace
        materialization, and residual-branch extra reads beyond sequential flow.

        References
        ----------
        Derivation: in unfused layer-by-layer execution, each feature map in
        ``sum_l C_l * H_l * W_l`` is written once and read once at byte width ``b``.

        Examples
        --------
        >>> from mlsysim import Models
        >>> resnet = Models.Vision.ResNet50
        >>> resnet.activation_traffic_bytes('fp16') == 2 * resnet.activation_traffic_bytes('int8')
        True
        """
        from ..core._validation import validate_at_least
        from ..core.units import resolve_precision

        validate_at_least(tokens_per_sample, 1, "tokens_per_sample")
        validate_at_least(ops_multiplier, 1, "ops_multiplier")
        prec_qty = resolve_precision(precision)[1] if isinstance(precision, str) else precision
        if self.feature_map_elements is None:
            return 0.0 * ureg.byte
        elems = self.feature_map_elements.to(ureg.count).magnitude
        bpp = prec_qty.to(ureg.byte).magnitude
        return (2.0 * elems * bpp * ops_multiplier * ureg.byte).to(ureg.byte)

    def lower(self, precision: Quantity | str = BYTES_FP16) -> ComputationGraph:
        """Lower the declared convolutional forward-pass work.

        Parameters
        ----------
        precision : Quantity or str, default: BYTES_FP16
            Bytes per stored element, or a registered format such as 'fp16'.
            A byte quantity specifies the width directly; it must be positive.

        Returns
        -------
        ComputationGraph
            Work in FLOPs, resident weight bytes, parameter count, arithmetic
            intensity in FLOP/byte, and the available layer count.

        Notes
        -----
        Equation: F = inference_flops; W = size_in_bytes(precision); I = F / W.

        Assumptions: inference_flops describes one image at the registered input resolution. Precision changes stored weights but does not change the declared operation count.

        Excludes: input-resolution scaling, convolution shape tracing, activation storage, backward work, and device-specific kernels.

        References
        ----------
        Derivation: multiply parameter count by storage width to obtain weight
        bytes, then divide the declared or approximated work by those bytes.
        This is a demand-accounting projection; it does not predict device time.

        Examples
        --------
        >>> from mlsysim import Models
        >>> model = Models.Vision.ResNet50
        >>> graph = model.lower('fp16')
        >>> graph.total_ops == model.inference_flops
        True
        """
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
        """Count persistent recurrent state for a batch of state-space sequences.

        Parameters
        ----------
        batch_size : int
            Nonnegative number of concurrent recurrent states.
        precision : Quantity or str, default: BYTES_FP16
            Bytes per stored element, or a registered format such as 'fp16'.
            A byte quantity specifies the width directly; it must be positive.

        Returns
        -------
        Quantity
            Persistent recurrent-state storage in bytes.

        Notes
        -----
        Equation: M_state = layers * batch_size * hidden_dim * state_size * b,
        where b is bytes per element. Sequence length does not enter this product.

        Assumptions: each layer retains a dense hidden_dim-by-state_size array
        per sequence and updates it in place. All arrays use the requested width.

        Excludes: convolution history, gate buffers, temporary activations,
        weights, and architecture-specific state packing. This is the declared
        recurrent-state tensor, not the full serving footprint.

        References
        ----------
        Derivation: multiply the dimensions of the retained recurrent-state
        array by element width. The state is overwritten per token, so this
        storage accounting does not accumulate one array per history token.

        Examples
        --------
        >>> from mlsysim.models.types import SSMWorkload
        >>> from mlsysim.core.units import Q_
        >>> model = SSMWorkload(name='SSM', architecture='ssm', parameters=Q_(100, 'count'), layers=2, hidden_dim=8, state_size=4)
        >>> model.get_state_cache_size(2) == 2 * model.get_state_cache_size(1)
        True
        """
        from ..core.units import resolve_precision
        prec_qty = resolve_precision(precision)[1] if isinstance(precision, str) else precision
        bpp = prec_qty.to(ureg.byte).magnitude
        size_bytes = self.layers * batch_size * self.hidden_dim * self.state_size * bpp
        return (size_bytes * ureg.byte).to(ureg.byte)

    def lower(self, precision: Quantity | str = BYTES_FP16) -> ComputationGraph:
        """Lower the declared state-space forward work and weights.

        Parameters
        ----------
        precision : Quantity or str, default: BYTES_FP16
            Bytes per stored element, or a registered format such as 'fp16'.
            A byte quantity specifies the width directly; it must be positive.

        Returns
        -------
        ComputationGraph
            Work in FLOPs, resident weight bytes, parameter count, arithmetic
            intensity in FLOP/byte, and the available layer count.

        Notes
        -----
        Equation: F = inference_flops if supplied, otherwise 2 * P; W = size_in_bytes(precision); I = F / W.

        Assumptions: the fallback approximates parameter-dominated work per recurrent step. A measured or separately derived operation count can be supplied through inference_flops.

        Excludes: recurrent-state storage and traffic, scan kernel details, training sequence work, and temporary activations. Use get_state_cache_size separately for retained state.

        References
        ----------
        Derivation: multiply parameter count by storage width to obtain weight
        bytes, then divide the declared or approximated work by those bytes.
        This is a demand-accounting projection; it does not predict device time.

        Examples
        --------
        >>> from mlsysim.models.types import SSMWorkload
        >>> from mlsysim.core.units import Q_
        >>> model = SSMWorkload(name='SSM', architecture='ssm', parameters=Q_(100, 'count'), layers=2, hidden_dim=8, state_size=4)
        >>> model.lower().weight_bytes == model.size_in_bytes()
        True
        """
        from ..physics.transformer import calc_transformer_decode_flops

        ops = self.inference_flops or calc_transformer_decode_flops(self.parameters)
        weights = self.size_in_bytes(precision)
        return ComputationGraph(
            name=self.name,
            total_ops=ops,
            parameter_count=self.parameters,
            weight_bytes=weights,
            arithmetic_intensity=(ops / weights).to("flop/byte"),
            layers=self.layers
        )

    def activation_traffic_bytes(
        self,
        precision: Quantity | str = BYTES_FP16,
        tokens_per_sample: int = 1,
        ops_multiplier: int = 1,
    ) -> Quantity:
        """Calculate per-sample residual-stream activation traffic for an SSM.

        Parameters
        ----------
        precision : Quantity or str, default: BYTES_FP16
            Bytes per activation element, or a registered format such as 'fp16'.
        tokens_per_sample : int, default: 1
            Positive token count per sample.
        ops_multiplier : int, default: 1
            Positive pass multiplier (1 for inference, 3 for training, 4 with
            full activation recomputation).

        Returns
        -------
        Quantity
            Off-chip activation read+write bytes per sample in ``byte`` units.

        Notes
        -----
        Equation: ``A_sample = 2 * layers * tokens_per_sample * hidden_dim * b * ops_multiplier``.

        Assumptions: each Mamba-style block fuses selective scan in SRAM, reading
        one ``(tokens_per_sample, hidden_dim)`` residual input and writing one
        ``(tokens_per_sample, hidden_dim)`` output to HBM per layer.

        Excludes: recurrent-state cache I/O across separate decode calls and
        unfused expanded inner-projection spills.

        References
        ----------
        Gu and Dao (2023), "Mamba: Linear-Time Sequence Modeling with Selective
        State Spaces." https://arxiv.org/abs/2312.00752 (bibliography key: gu2023mamba).

        Examples
        --------
        >>> from mlsysim.models.types import SSMWorkload
        >>> from mlsysim.core.units import Q_
        >>> ssm = SSMWorkload(name='SSM', architecture='ssm', parameters=Q_(100, 'count'), layers=2, hidden_dim=8, state_size=4)
        >>> ssm.activation_traffic_bytes('fp16', tokens_per_sample=10) == Q_(640, 'byte')
        True
        """
        from ..core._validation import validate_at_least
        from ..core.units import resolve_precision

        validate_at_least(tokens_per_sample, 1, "tokens_per_sample")
        validate_at_least(ops_multiplier, 1, "ops_multiplier")
        prec_qty = resolve_precision(precision)[1] if isinstance(precision, str) else precision
        bpp = prec_qty.to(ureg.byte).magnitude
        total_bytes = 2.0 * self.layers * tokens_per_sample * self.hidden_dim * bpp * ops_multiplier
        return (total_bytes * ureg.byte).to(ureg.byte)


class DiffusionWorkload(Workload):
    """Workload representation of a Diffusion generative model."""
    parameters: Quantity
    denoising_steps: int
    resolution: int
    inference_flops: Optional[Quantity] = None  # FLOPs for ONE denoising step

    def lower(self, precision: Quantity | str = BYTES_FP16) -> ComputationGraph:
        # Total inference flops = Flops per step * denoising steps
        """Accumulate forward work across the configured denoising steps.

        Parameters
        ----------
        precision : Quantity or str, default: BYTES_FP16
            Bytes per stored element, or a registered format such as 'fp16'.
            A byte quantity specifies the width directly; it must be positive.

        Returns
        -------
        ComputationGraph
            Work in FLOPs, resident weight bytes, parameter count, arithmetic
            intensity in FLOP/byte, and the available layer count.

        Notes
        -----
        Equation: F = denoising_steps * F_step; F_step is inference_flops when supplied, otherwise 2 * P. W = size_in_bytes(precision) is retained once; I = F / W.

        Assumptions: every step executes the same declared work and reuses the same weights. inference_flops is per denoising step, unlike a total-generation FLOP count.

        Excludes: resolution-dependent FLOP inference, classifier-free-guidance duplication, text-encoder and decoder work, per-step activation traffic, and sampler-specific overhead.

        References
        ----------
        Derivation: multiply parameter count by storage width to obtain weight
        bytes, then divide the declared or approximated work by those bytes.
        This is a demand-accounting projection; it does not predict device time.

        Examples
        --------
        >>> from mlsysim.models.types import DiffusionWorkload
        >>> from mlsysim.core.units import Q_
        >>> model = DiffusionWorkload(name='Diffusion', architecture='diffusion', parameters=Q_(100, 'count'), denoising_steps=4, resolution=64, inference_flops=Q_(200, 'flop'))
        >>> model.lower().total_ops == model.denoising_steps * model.inference_flops
        True
        """
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
        """Lower one declared embodied-policy evaluation.

        Parameters
        ----------
        precision : Quantity or str, default: BYTES_FP16
            Bytes per stored element, or a registered format such as 'fp16'.
            A byte quantity specifies the width directly; it must be positive.

        Returns
        -------
        ComputationGraph
            Work in FLOPs, resident weight bytes, parameter count, arithmetic
            intensity in FLOP/byte, and the available layer count.

        Notes
        -----
        Equation: F = inference_flops if supplied, otherwise 2 * P; W = size_in_bytes(precision); I = F / W.

        Assumptions: inference_flops describes one policy evaluation. action_chunk_horizon and observation_horizon are metadata and do not multiply this graph's work.

        Excludes: sensor acquisition, observation encoding beyond the supplied FLOPs, iterative decoder passes, control-loop scheduling, and actuator dynamics.

        References
        ----------
        Derivation: multiply parameter count by storage width to obtain weight
        bytes, then divide the declared or approximated work by those bytes.
        This is a demand-accounting projection; it does not predict device time.

        Examples
        --------
        >>> from mlsysim.models.types import EmbodiedWorkload
        >>> from mlsysim.core.units import Q_
        >>> model = EmbodiedWorkload(name='Policy', architecture='vla', parameters=Q_(100, 'count'), action_chunk_horizon=8)
        >>> model.lower().weight_bytes == model.size_in_bytes()
        True
        """
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

