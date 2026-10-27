# MLSysBook Taxonomy of Machine Learning Systems

**Editorial draft.** Classification is provisional; collection membership is not a broader/narrower assertion.

Engineering systems that learn from data. This pilot distinguishes subject hierarchy from navigation collections and conditional technical relationships.

## Topic reference

### Parallel execution strategies

Identity: `mlsys.topic.parallel-execution`

Strategies for partitioning or replicating computation and state so work can execute concurrently.

- **Data parallelism** (`mlsys.data-parallel`): Replicates model computation over different examples while coordinating learning state.

- **Tensor parallelism** (`mlsys.tensor-parallel`): Partitions tensor operations and state for concurrent execution.

- **Pipeline parallelism** (`mlsys.pipeline-parallel`): Partitions computation into concurrently executing stages.

### Learning paradigms

Identity: `mlsys.topic.learning-paradigms`

Approaches to learning distinguished by their training signals, observations, and interaction with an environment.

- **Supervised learning** (`mlsys.supervised-learning`): A learning approach distinguished by the source and structure of training signals; sibling topics may overlap.

- **Self-supervised learning** (`mlsys.self-supervised-learning`): A learning approach distinguished by the source and structure of training signals; sibling topics may overlap.

- **Unsupervised learning** (`mlsys.unsupervised-learning`): A learning approach distinguished by the source and structure of training signals; sibling topics may overlap.

- **Reinforcement learning** (`mlsys.reinforcement-learning`): A learning approach distinguished by the source and structure of training signals; sibling topics may overlap.

- **Imitation learning** (`mlsys.imitation-learning`): A learning approach distinguished by the source and structure of training signals; sibling topics may overlap.

## Navigation collections

These are authored groupings, not inferred taxonomic parents.

### Objectives and decisions

What should the system achieve, and within which limits?

- System requirements (`mlsys.requirements`)
- Baseline (`mlsys.baseline`)
- Proxy objective (`mlsys.proxy-objective`)
- Service-level objective (`mlsys.slo`)
- Workload characterization (`mlsys.workload`)
- Execution placement (`mlsys.placement`)
- Decision policy (`mlsys.decision-policy`)
- Service-level indicator (`mlsys.sli`)
- Error budget (`mlsys.error-budget`)
- Latency budget (`mlsys.latency-budget`)
- Workload trace (`mlsys.workload-trace`)
- Cost-sensitive decision (`mlsys.cost-sensitive-decision`)
- Pareto frontier (`mlsys.pareto-frontier`)
- Accuracy-efficiency frontier (`mlsys.efficiency-frontier`)
- Recovery time and point objectives (`mlsys.recovery-objectives`)

### Data and representation

What information reaches learning and execution?

- Data quality (`mlsys.data-quality`)
- Label quality (`mlsys.labels`)
- Feature engineering (`mlsys.features`)
- Missing data (`mlsys.missing-data`)
- Data leakage (`mlsys.leakage`)
- Point-in-time correctness (`mlsys.point-in-time`)
- Data lineage (`mlsys.lineage`)
- Training-serving skew (`mlsys.train-serve-skew`)
- Dataset deduplication (`mlsys.deduplication`)
- Data selection (`mlsys.data-selection`)
- Tokenization (`mlsys.tokenization`)
- Distribution shift (`mlsys.distribution-shift`)
- Data acquisition and sampling (`mlsys.data-acquisition`)
- Data contract (`mlsys.data-contract`)
- Delayed outcomes (`mlsys.delayed-outcomes`)
- Dataset storage layout (`mlsys.dataset-storage-layout`)
- Data ingestion (`mlsys.data-ingestion`)
- Schema evolution (`mlsys.schema-evolution`)
- Statistical heterogeneity (`mlsys.non-iid-data`)
- Parallel file system (`mlsys.parallel-file-system`)
- Object storage (`mlsys.object-storage`)
- Training input pipeline (`mlsys.input-pipeline`)
- Storage tiering and caching (`mlsys.storage-tiering`)
- Weak supervision (`mlsys.weak-supervision`)
- Active learning (`mlsys.active-learning`)
- Inter-annotator agreement (`mlsys.annotator-agreement`)
- Quality filtering (`mlsys.quality-filtering`)
- Data mixture (`mlsys.data-mixture`)
- Synthetic data (`mlsys.synthetic-data`)
- Data augmentation (`mlsys.data-augmentation`)
- Data validation (`mlsys.data-validation`)
- Feature store (`mlsys.feature-store`)
- Stream processing (`mlsys.stream-processing`)
- Feature backfill (`mlsys.feature-backfill`)
- Dataset versioning (`mlsys.data-versioning`)
- Data dependency debt (`mlsys.data-dependency`)
- Feature freshness (`mlsys.feature-freshness`)
- Distributed batch data processing (`mlsys.batch-data-processing`)
- Data warehouse and lakehouse (`mlsys.lakehouse`)
- Data cleaning (`mlsys.data-cleaning`)
- Class imbalance (`mlsys.class-imbalance`)
- Class rebalancing (`mlsys.rebalancing`)
- Feature hashing (`mlsys.feature-hashing`)
- Feature cross (`mlsys.feature-cross`)
- Exploratory data analysis (`mlsys.exploratory-data-analysis`)
- Data cascade (`mlsys.data-cascade`)
- Data valuation (`mlsys.data-valuation`)
- Input prefetching (`mlsys.input-prefetching`)
- Event-time windowing and watermarks (`mlsys.event-time-windowing`)
- Host-side data loading (`mlsys.host-data-loader`)
- PII detection and de-identification (`mlsys.pii-deidentification`)
- Processing delivery semantics (`mlsys.delivery-semantics`)
- Online and offline feature consistency (`mlsys.online-feature-consistency`)
- Zero-copy sensor input via DMA (`mlsys.zero-copy-dma`)
- Sequence packing (`mlsys.sequence-packing`)
- Model collapse (`mlsys.model-collapse`)
- Data curriculum and annealing (`mlsys.data-curriculum`)
- Sharded streaming dataset format (`mlsys.streaming-dataset-format`)
- Model-assisted labeling (`mlsys.model-assisted-labeling`)
- Erasure coding (`mlsys.erasure-coding`)

### Model regimes

How does model structure change computation and behavior?

- Linear models (`mlsys.linear-model`)
- Tree ensembles (`mlsys.tree-ensemble`)
- Kernel methods (`mlsys.kernel-method`)
- Nearest-neighbor models (`mlsys.nearest-neighbor`)
- Neural networks (`mlsys.neural-network`)
- Inductive bias (`mlsys.inductive-bias`)
- Embeddings (`mlsys.embeddings`)
- Attention (`mlsys.attention`)
- Position encoding (`mlsys.position-encoding`)
- Model artifact (`mlsys.model-artifact`)
- Transformer (`mlsys.transformer`)
- Deep recommendation model (`mlsys.recommendation-model`)
- Diffusion model (`mlsys.diffusion-model`)
- Convolution (`mlsys.convolution`)
- Recurrent computation (`mlsys.recurrence`)
- Normalization layers (`mlsys.normalization-layer`)
- Mixture of experts (`mlsys.mixture-of-experts`)
- Expert routing (`mlsys.expert-routing`)
- Expert capacity (`mlsys.expert-capacity`)
- Active parameters (`mlsys.active-parameters`)
- Early exit (`mlsys.early-exit`)
- Sparse and sliding-window attention (`mlsys.sliding-window-attention`)
- Linear attention (`mlsys.linear-attention`)
- State-space sequence model (`mlsys.state-space-model`)
- Grouped-query attention (`mlsys.grouped-query-attention`)
- Context-length extension (`mlsys.context-extension`)
- Weight serialization format (`mlsys.weight-serialization`)
- Model exchange format (`mlsys.model-exchange-format`)
- Model export (`mlsys.model-export`)
- Model signature (`mlsys.model-signature`)
- Neural architecture search (`mlsys.neural-architecture-search`)
- Hardware-aware NAS (`mlsys.hardware-aware-nas`)
- Depthwise separable convolution (`mlsys.depthwise-separable-convolution`)
- Multimodal model (`mlsys.multimodal-model`)
- Modality encoder (`mlsys.modality-encoder`)
- Generative adversarial network (`mlsys.gan`)
- Dynamic sparse attention (`mlsys.dynamic-sparse-attention`)
- Graph neural network (`mlsys.graph-neural-network`)
- Large language model (`mlsys.large-language-model`)
- Encoder-only, decoder-only, and encoder-decoder Transformers (`mlsys.transformer-variants`)
- Multi-head latent attention (`mlsys.multi-head-latent-attention`)
- Residual connection (`mlsys.residual-connection`)

### Learning and experimentation

How should learning work and experimental effort be allocated?

- Training loss (`mlsys.loss`)
- Automatic differentiation (`mlsys.autodiff`)
- Optimization algorithm (`mlsys.optimizer`)
- Regularization (`mlsys.regularization`)
- Training batch size (`mlsys.batch-size`)
- Training state (`mlsys.training-state`)
- Activation recomputation (`mlsys.activation-recompute`)
- Empirical scaling laws (`mlsys.scaling-laws`)
- Hyperparameter search (`mlsys.hyperparameter-search`)
- Experiment-budget allocation (`mlsys.experiment-budget`)
- Transfer learning (`mlsys.transfer-learning`)
- Parameter-efficient adaptation (`mlsys.peft`)
- Supervised learning (`mlsys.supervised-learning`)
- Self-supervised learning (`mlsys.self-supervised-learning`)
- Unsupervised learning (`mlsys.unsupervised-learning`)
- Reinforcement learning (`mlsys.reinforcement-learning`)
- Imitation learning (`mlsys.imitation-learning`)
- Load-balancing loss (`mlsys.load-balancing-loss`)
- Compute-optimal allocation (`mlsys.compute-optimal-allocation`)
- Training compute estimate (`mlsys.training-compute-estimate`)
- Federated averaging (`mlsys.federated-averaging`)
- Learning-rate schedule (`mlsys.learning-rate-schedule`)
- Critical batch size (`mlsys.critical-batch-size`)
- Training instability (`mlsys.training-instability`)
- Gradient accumulation (`mlsys.gradient-accumulation`)
- Training-state offloading (`mlsys.memory-offloading`)
- Experiment tracking (`mlsys.experiment-tracking`)
- Hyperparameter transfer (`mlsys.hyperparameter-transfer`)
- Supervised fine-tuning (`mlsys.supervised-fine-tuning`)
- Low-rank adaptation (`mlsys.lora`)
- RL from human feedback (`mlsys.rlhf`)
- Reward model (`mlsys.reward-model`)
- Direct preference optimization (`mlsys.preference-optimization`)
- RL with verifiable rewards (`mlsys.rlvr`)
- RL rollout generation (`mlsys.rl-rollout`)
- Policy staleness (`mlsys.policy-staleness`)
- Online learning (`mlsys.online-learning`)
- Incremental training (`mlsys.incremental-training`)
- Catastrophic forgetting (`mlsys.catastrophic-forgetting`)
- Experience replay (`mlsys.experience-replay`)
- On-device training (`mlsys.on-device-training`)
- Model selection (`mlsys.model-selection`)
- Pretraining (`mlsys.pretraining`)
- Causal language modeling (`mlsys.causal-lm`)
- Mode collapse (`mlsys.mode-collapse`)
- Training diagnostics (`mlsys.training-diagnostics`)
- In-context learning (`mlsys.in-context-learning`)
- Neighbor sampling (`mlsys.neighbor-sampling`)
- Compounding error in sequential decisions (`mlsys.compounding-error`)
- Interactive imitation learning (`mlsys.interactive-imitation`)
- Loss masking and token normalization (`mlsys.loss-masking`)
- Credit assignment (`mlsys.credit-assignment`)
- Memory-efficient optimizer (`mlsys.memory-efficient-optimizer`)
- Offline reinforcement learning (`mlsys.offline-rl`)
- Gradient clipping (`mlsys.gradient-clipping`)
- Weight initialization (`mlsys.weight-initialization`)
- Model merging (`mlsys.model-merging`)
- Group relative policy optimization (`mlsys.group-relative-policy-optimization`)
- Agent training environment (`mlsys.agent-training-environment`)
- Multi-task learning (`mlsys.multi-task-learning`)

### Numerical execution

How does a mathematical model become machine work?

- Tensor layout (`mlsys.tensor-layout`)
- Matrix multiplication (`mlsys.matrix-multiply`)
- Numerical precision (`mlsys.precision`)
- Numerical stability (`mlsys.numerical-stability`)
- Mixed-precision training (`mlsys.mixed-precision`)
- Quantization (`mlsys.quantization`)
- Sparsity (`mlsys.sparsity`)
- Knowledge distillation (`mlsys.distillation`)
- Graph compilation (`mlsys.compilation`)
- Kernel fusion (`mlsys.fusion`)
- Host dispatch overhead (`mlsys.dispatch`)
- Structured output constraint (`mlsys.grammar`)
- Tensor (`mlsys.tensor`)
- Computation graph (`mlsys.computation-graph`)
- Eager vs. graph execution (`mlsys.eager-execution`)
- Tracing and JIT capture (`mlsys.tracing-jit`)
- IO-aware attention kernel (`mlsys.io-aware-attention`)
- Convolution lowering (`mlsys.convolution-lowering`)
- Vendor kernel library (`mlsys.kernel-library`)
- ML compiler IR (`mlsys.intermediate-representation`)
- Kernel autotuning (`mlsys.autotuning`)
- Kernel DSL (`mlsys.kernel-dsl`)
- Static memory planning (`mlsys.memory-planning`)
- Device streams (`mlsys.device-streams`)
- Caching memory allocator (`mlsys.caching-allocator`)
- Captured graph replay (`mlsys.graph-replay`)
- Loss scaling (`mlsys.loss-scaling`)
- Block-scaled number formats (`mlsys.block-scaled-format`)
- Accumulation precision (`mlsys.accumulation-precision`)
- Pruning (`mlsys.pruning`)
- Structured pruning (`mlsys.structured-pruning`)
- Low-rank factorization (`mlsys.low-rank-factorization`)
- Quantization-aware training (`mlsys.quantization-aware-training`)
- Algorithm-based fault tolerance (`mlsys.abft`)
- Thread hierarchy (`mlsys.thread-hierarchy`)
- Memory coalescing and bank conflicts (`mlsys.memory-coalescing`)
- Occupancy and latency hiding (`mlsys.occupancy`)
- Asynchronous data movement (`mlsys.async-data-movement`)
- Matrix-engine programming (`mlsys.matrix-engine-programming`)
- Composable function transformations (`mlsys.function-transformations`)
- Post-training quantization (`mlsys.post-training-quantization`)
- Weight-only quantization (`mlsys.weight-only-quantization`)
- Activation outliers (`mlsys.activation-outliers`)
- KV-cache quantization (`mlsys.kv-cache-quantization`)
- N:M structured sparsity (`mlsys.nm-sparsity`)
- Numerical nondeterminism (`mlsys.numerical-nondeterminism`)
- Batch invariance (`mlsys.batch-invariance`)
- Embedded ML DSL (`mlsys.embedded-dsl`)
- Differentiable programming (`mlsys.differentiable-programming`)
- Persistent kernel (`mlsys.persistent-kernel`)
- Tensor graph superoptimization (`mlsys.graph-superoptimization`)
- Contextual activation sparsity (`mlsys.activation-sparsity`)
- Sparse tensor kernels (`mlsys.sparse-tensor-kernels`)
- CPU threading and NUMA placement (`mlsys.cpu-threading-numa`)
- Dynamic shapes and shape specialization (`mlsys.dynamic-shapes`)
- Rule-based graph rewriting (`mlsys.rule-based-graph-rewriting`)
- Sparse and ragged tensor representation (`mlsys.sparse-ragged-tensors`)
- Execution providers and operator fallback (`mlsys.execution-providers`)
- Runtime memory fragmentation (`mlsys.memory-fragmentation`)
- Python GIL and host-language concurrency (`mlsys.python-gil`)
- Memory-mapped model weights (`mlsys.mmap-weights`)
- OS memory pressure and paging (`mlsys.os-memory-pressure`)
- Managed-runtime interop and garbage collection (`mlsys.managed-runtime-interop`)
- Asynchronous on-device inference (`mlsys.ui-thread-inference`)
- Register pressure (`mlsys.register-pressure`)
- Software pipelining (`mlsys.software-pipelining`)
- Integer-only inference (`mlsys.integer-only-inference`)
- Quantization granularity (`mlsys.quantization-granularity`)

### Hardware and performance

Which physical resource limits useful work?

- Memory capacity (`mlsys.memory-capacity`)
- Memory bandwidth (`mlsys.memory-bandwidth`)
- Compute throughput (`mlsys.compute-throughput`)
- Arithmetic intensity (`mlsys.arithmetic-intensity`)
- Roofline bound (`mlsys.roofline`)
- Data locality (`mlsys.locality`)
- Tiling (`mlsys.tiling`)
- Profiling (`mlsys.profiling`)
- Resource utilization (`mlsys.utilization`)
- Serial fraction (`mlsys.serial-fraction`)
- Execution time (`mlsys.execution-time`)
- Accelerator execution architecture (`mlsys.accelerator-execution-architecture`)
- Memory hierarchy (`mlsys.memory-hierarchy`)
- Host-device interconnect (`mlsys.host-device-interconnect`)
- Power delivery and integrity (`mlsys.power-delivery`)
- Thermal behavior and throttling (`mlsys.thermal-dynamics`)
- Operation and byte accounting (`mlsys.flop-accounting`)
- Memory footprint (`mlsys.memory-footprint`)
- Ridge point (`mlsys.ridge-point`)
- Compute-bound vs. memory-bound (`mlsys.bound-regime`)
- Data-movement energy (`mlsys.data-movement-energy`)
- Amdahl's law (`mlsys.amdahls-law`)
- Systolic array (`mlsys.systolic-array`)
- Matrix engine (`mlsys.matrix-unit`)
- SIMT execution (`mlsys.simt`)
- Accelerator dataflow (`mlsys.dataflow`)
- Domain-specific architecture (`mlsys.domain-specific-architecture`)
- High-bandwidth memory (`mlsys.hbm`)
- On-chip SRAM and scratchpads (`mlsys.on-chip-sram`)
- Memory wall (`mlsys.memory-wall`)
- Scale-up interconnect (`mlsys.scale-up-interconnect`)
- Cache-coherent interconnect (`mlsys.coherent-interconnect`)
- Peer-to-peer device transfer (`mlsys.peer-to-peer-transfer`)
- Thermal design power (`mlsys.tdp`)
- Dynamic voltage and frequency scaling (`mlsys.dvfs`)
- Dark silicon (`mlsys.dark-silicon`)
- Power capping (`mlsys.power-capping`)
- Microcontroller-class device (`mlsys.microcontroller`)
- Heterogeneous mobile SoC (`mlsys.heterogeneous-soc`)
- On-device neural accelerator (`mlsys.edge-npu`)
- On-device memory budget (`mlsys.on-device-memory-budget`)
- Liquid cooling (`mlsys.liquid-cooling`)
- Rack power density (`mlsys.rack-power-density`)
- Warehouse-scale computer (`mlsys.warehouse-scale-computer`)
- Accelerator sharing and partitioning (`mlsys.gpu-partitioning`)
- Model FLOPs utilization (`mlsys.model-flops-utilization`)
- Bottleneck analysis (`mlsys.bottleneck-analysis`)
- Training goodput (`mlsys.training-goodput`)
- Model-hardware co-design (`mlsys.model-hardware-co-design`)
- Silent data corruption (`mlsys.silent-data-corruption`)
- Hardware fault injection (`mlsys.hardware-fault-injection`)
- Embedding cache and tiering (`mlsys.embedding-cache`)
- Analytical performance model (`mlsys.analytical-performance-model`)
- Performance simulation (`mlsys.performance-simulation`)
- Simulator fidelity (`mlsys.model-fidelity`)
- Compute-in-memory (`mlsys.compute-in-memory`)
- Emerging non-volatile memory (`mlsys.emerging-nvm`)
- Synchronous training power swings (`mlsys.training-power-swings`)
- UPS and battery energy storage (`mlsys.facility-energy-storage`)
- Adaptive routing (`mlsys.adaptive-routing`)
- Incast congestion (`mlsys.incast`)
- Operator execution trace (`mlsys.execution-trace`)
- CPU execution architecture (`mlsys.cpu-execution-architecture`)
- Flash write endurance (`mlsys.flash-endurance`)
- Energy-harvesting intermittent computing (`mlsys.intermittent-computing`)
- Duty cycling and sleep states (`mlsys.duty-cycling`)
- Sensor interconnect bandwidth (`mlsys.sensor-interconnect`)
- Advanced accelerator packaging (`mlsys.advanced-accelerator-packaging`)
- FPGA-based ML acceleration (`mlsys.fpga-acceleration`)
- Wafer-scale acceleration (`mlsys.wafer-scale-acceleration`)
- Execute-in-place weights (`mlsys.execute-in-place`)
- Always-on sensing cascade (`mlsys.always-on-cascade`)
- DMA cache maintenance (`mlsys.dma-cache-maintenance`)
- Cross-stack profiling (`mlsys.cross-stack-profiling`)

### Distributed execution and state

How do communication and failures affect collective work?

- Collective communication (`mlsys.collectives`)
- Data parallelism (`mlsys.data-parallel`)
- Tensor parallelism (`mlsys.tensor-parallel`)
- Pipeline parallelism (`mlsys.pipeline-parallel`)
- Training-state sharding (`mlsys.state-sharding`)
- Communication overlap (`mlsys.communication-overlap`)
- Network topology (`mlsys.network-topology`)
- Stragglers (`mlsys.stragglers`)
- Recovery checkpoint (`mlsys.checkpoint`)
- Consistent checkpoint cut (`mlsys.consistent-cut`)
- Replicated-log consensus (`mlsys.consensus`)
- Fencing (`mlsys.fencing`)
- Federated learning (`mlsys.federated-learning`)
- Idempotency (`mlsys.idempotency`)
- Compensating action (`mlsys.compensation`)
- Failure model (`mlsys.failure-model`)
- Strong scaling (`mlsys.strong-scaling`)
- Weak scaling (`mlsys.weak-scaling`)
- Parallel scaling efficiency (`mlsys.parallel-efficiency`)
- Expert parallelism (`mlsys.expert-parallel`)
- Sequence and context parallelism (`mlsys.sequence-parallel`)
- Hybrid parallelism (`mlsys.hybrid-parallel`)
- Pipeline bubble (`mlsys.pipeline-bubble`)
- Ring all-reduce (`mlsys.ring-allreduce`)
- Tree and hierarchical all-reduce (`mlsys.tree-allreduce`)
- All-to-all exchange (`mlsys.all-to-all`)
- Alpha-beta communication model (`mlsys.alpha-beta-model`)
- Gradient compression (`mlsys.gradient-compression`)
- Lease (`mlsys.lease`)
- Leader election (`mlsys.leader-election`)
- Bounded staleness (`mlsys.bounded-staleness`)
- Client selection (`mlsys.client-sampling`)
- Cross-device vs. cross-silo federation (`mlsys.federation-setting`)
- Remote direct memory access (`mlsys.rdma`)
- Lossless network fabric (`mlsys.lossless-fabric`)
- Datacenter congestion control (`mlsys.congestion-control`)
- Fat-tree (Clos) topology (`mlsys.fat-tree`)
- Rail-optimized topology (`mlsys.rail-optimized`)
- Bisection bandwidth (`mlsys.bisection-bandwidth`)
- In-network reduction (`mlsys.in-network-reduction`)
- Checkpoint I/O (`mlsys.checkpoint-io`)
- Elastic training (`mlsys.elastic-training`)
- Topology-aware placement (`mlsys.topology-aware-placement`)
- Policy weight synchronization (`mlsys.weight-sync`)
- Optimal checkpoint interval (`mlsys.checkpoint-interval`)
- Asynchronous checkpointing (`mlsys.async-checkpointing`)
- Embedding-table sharding (`mlsys.embedding-table-sharding`)
- Parameter server (`mlsys.parameter-server`)
- Automatic parallelization (`mlsys.auto-parallelization`)
- Sharding annotation and propagation (`mlsys.sharding-propagation`)
- Device mesh (`mlsys.device-mesh`)
- Distributed task and actor runtime (`mlsys.task-actor-runtime`)
- Single- vs. multi-controller execution (`mlsys.single-vs-multi-controller`)
- Checkpoint resharding (`mlsys.checkpoint-resharding`)
- Large-model training framework (`mlsys.llm-training-framework`)
- Heterogeneous-accelerator training and serving (`mlsys.heterogeneous-accelerator-execution`)
- Dynamic re-parallelization (`mlsys.dynamic-resharding`)
- ML-aware cluster scheduling (`mlsys.ml-aware-scheduling`)
- Local SGD (`mlsys.local-sgd`)
- Expert replication and placement (`mlsys.expert-placement`)
- Cluster manager (`mlsys.cluster-manager`)
- Synchronous vs. asynchronous training (`mlsys.sync-async-training`)
- Graph partitioning for GNN training (`mlsys.gnn-graph-partitioning`)
- Distributed job launch and process groups (`mlsys.process-group-rendezvous`)
- Network propagation delay (`mlsys.network-propagation-delay`)
- Consistency-availability trade-off (`mlsys.cap-tradeoff`)
- Decentralized gossip averaging (`mlsys.gossip-averaging`)
- Pipeline schedule (`mlsys.pipeline-schedule`)
- Optical circuit switching (`mlsys.optical-circuit-switching`)
- Co-packaged optics (`mlsys.co-packaged-optics`)
- Device-initiated communication (`mlsys.device-initiated-communication`)
- Collective algorithm selection (`mlsys.collective-algorithm-selection`)
- Incremental checkpointing (`mlsys.incremental-checkpointing`)
- Forward error correction (`mlsys.forward-error-correction`)
- Backfill scheduling (`mlsys.backfill-scheduling`)
- Network traffic classes and QoS (`mlsys.traffic-classes`)
- Variable-length parallelism planning (`mlsys.variable-length-parallelism`)
- Graph pipeline parallelism (`mlsys.graph-pipeline-parallelism`)

### Serving and capacity

How are requests admitted, executed, and completed?

- Arrival rate (`mlsys.arrival-rate`)
- Service capacity (`mlsys.service-capacity`)
- Queueing delay (`mlsys.queueing`)
- Tail latency (`mlsys.tail-latency`)
- Request batching (`mlsys.batching`)
- Admission control (`mlsys.admission`)
- Autoscaling (`mlsys.autoscaling`)
- KV cache (`mlsys.kv-cache`)
- Prefill and decode (`mlsys.prefill-decode`)
- Paged KV allocation (`mlsys.paged-kv`)
- Prefix caching (`mlsys.prefix-cache`)
- Speculative decoding (`mlsys.speculative-decoding`)
- SLO goodput (`mlsys.slo-goodput`)
- Context window (`mlsys.context`)
- Backpressure (`mlsys.backpressure`)
- Little's law (`mlsys.littles-law`)
- Latency-throughput curve (`mlsys.latency-throughput-curve`)
- Arrival burstiness (`mlsys.arrival-burstiness`)
- Online inference (`mlsys.online-inference`)
- Batch inference (`mlsys.batch-inference`)
- Model server (`mlsys.model-server`)
- Multi-model serving (`mlsys.multi-model-serving`)
- Model cascade (`mlsys.model-cascade`)
- Model routing (`mlsys.model-routing`)
- Time to first token (`mlsys.ttft`)
- Time per output token (`mlsys.tpot`)
- Hedged requests (`mlsys.hedged-requests`)
- Continuous batching (`mlsys.continuous-batching`)
- Request load balancing (`mlsys.load-balancing`)
- Model cold start (`mlsys.cold-start`)
- Chunked prefill (`mlsys.chunked-prefill`)
- Prefill-decode disaggregation (`mlsys.pd-disaggregation`)
- Decoding and sampling controls (`mlsys.sampling-controls`)
- Constrained decoding (`mlsys.constrained-decoding`)
- Output validation (`mlsys.output-validation`)
- Reasoning budget (`mlsys.reasoning-budget`)
- Best-of-n sampling (`mlsys.best-of-n`)
- Self-consistency voting (`mlsys.self-consistency`)
- Inference-time search (`mlsys.inference-time-search`)
- Test-time scaling (`mlsys.test-time-scaling`)
- Model conversion (`mlsys.model-conversion`)
- On-device inference runtime (`mlsys.on-device-runtime`)
- Device-cloud partitioning (`mlsys.device-cloud-partitioning`)
- Device heterogeneity (`mlsys.device-heterogeneity`)
- Multi-stage retrieval and ranking (`mlsys.ranking-funnel`)
- Multimodal encoder disaggregation (`mlsys.encoder-disaggregation`)
- KV-cache offloading and tiering (`mlsys.kv-cache-offload`)
- KV-cache transfer (`mlsys.kv-cache-transfer`)
- Compound AI system (`mlsys.compound-ai-system`)
- LM program optimization (`mlsys.lm-program-optimization`)
- Multi-adapter serving (`mlsys.multi-adapter-serving`)
- Denoising-step reduction (`mlsys.denoising-step-reduction`)
- Cross-step feature caching (`mlsys.diffusion-feature-caching`)
- Request correlation keys (`mlsys.keyed-predictions`)
- Stateless model serving (`mlsys.stateless-serving-function`)
- Model gateway (`mlsys.model-gateway`)
- Semantic cache (`mlsys.semantic-cache`)
- Prompt engineering (`mlsys.prompt-engineering`)
- KV-cache eviction (`mlsys.kv-cache-eviction`)
- CPU-GPU hybrid inference (`mlsys.cpu-gpu-hybrid-inference`)
- Non-prefix KV reuse (`mlsys.non-prefix-kv-reuse`)
- Inference compute and memory estimate (`mlsys.inference-compute-estimate`)
- Token streaming (`mlsys.token-streaming`)
- Serverless inference (`mlsys.serverless-inference`)
- LLM request scheduling and preemption (`mlsys.llm-request-scheduling`)
- Streaming inference (`mlsys.streaming-inference`)
- Inference API and payload serialization (`mlsys.inference-api`)
- Request throughput (`mlsys.request-throughput`)
- Latency percentiles (`mlsys.latency-percentiles`)
- Cache-aware request routing (`mlsys.cache-aware-routing`)
- Finite buffers and overflow (`mlsys.finite-buffer`)
- Inter-token latency (`mlsys.inter-token-latency`)
- Fan-out tail amplification (`mlsys.fan-out-tail-amplification`)
- Masked diffusion decoding (`mlsys.masked-diffusion-decoding`)
- Bandwidth-delay product (`mlsys.bandwidth-delay-product`)
- Expert offloading (`mlsys.expert-offloading`)
- Intra-device overlap of bottlenecks (`mlsys.intra-device-overlap`)

### Evaluation and uncertainty

What does the evidence justify?

- Generalization (`mlsys.generalization`)
- Evaluation split (`mlsys.evaluation-split`)
- Model-selection optimism (`mlsys.selection-bias`)
- Evaluation metrics (`mlsys.metrics`)
- Probability calibration (`mlsys.calibration`)
- Statistical uncertainty (`mlsys.statistical-uncertainty`)
- Benchmark protocol (`mlsys.benchmark`)
- Online experimentation (`mlsys.online-experiment`)
- Sliced evaluation (`mlsys.sliced-evaluation`)
- Selective prediction (`mlsys.selective-prediction`)
- Hallucination (`mlsys.hallucination`)
- Robustness evaluation (`mlsys.robustness-evaluation`)
- Test oracle (`mlsys.test-oracle`)
- Integration testing (`mlsys.integration-testing`)
- Failure injection (`mlsys.failure-injection`)
- Cross-validation (`mlsys.cross-validation`)
- Sequential testing (`mlsys.sequential-testing`)
- Experiment interference (`mlsys.experiment-interference`)
- Conformal prediction (`mlsys.conformal-prediction`)
- Slice discovery (`mlsys.slice-discovery`)
- Model-based grading (`mlsys.llm-judge`)
- pass@k (`mlsys.pass-at-k`)
- Pairwise preference evaluation (`mlsys.pairwise-preference-eval`)
- Human evaluation (`mlsys.human-evaluation`)
- Benchmark contamination (`mlsys.benchmark-contamination`)
- Time to train (`mlsys.time-to-train`)
- Inference load scenario (`mlsys.inference-load-scenario`)
- Metamorphic testing (`mlsys.metamorphic-testing`)
- Behavioral testing (`mlsys.behavioral-testing`)
- Production-readiness assessment (`mlsys.production-readiness`)
- Verifier (`mlsys.verifier`)
- Process reward model (`mlsys.process-reward-model`)
- Scenario-based testing (`mlsys.scenario-based-testing`)
- Simulation-based falsification (`mlsys.falsification`)
- Formal verification (`mlsys.formal-verification`)
- Neural network verification (`mlsys.neural-network-verification`)
- Rare-event safety validation (`mlsys.rare-event-validation`)
- X-in-the-loop testing (`mlsys.xil-testing`)
- Evaluation harness (`mlsys.evaluation-harness`)
- Dangerous-capability evaluation (`mlsys.dangerous-capability-eval`)
- Model and prediction visualization (`mlsys.model-visualization`)
- Perplexity (`mlsys.perplexity`)
- Reference-match scoring (`mlsys.reference-match-scoring`)
- Error analysis (`mlsys.error-analysis`)
- Closed-loop policy evaluation (`mlsys.closed-loop-evaluation`)
- Paired comparison and noninferiority testing (`mlsys.paired-comparison`)
- Benchmark representativeness (`mlsys.benchmark-representativeness`)
- Performance regression testing (`mlsys.performance-regression-testing`)
- Performance measurement variability (`mlsys.measurement-variability`)
- Out-of-distribution detection (`mlsys.ood-detection`)
- Off-policy evaluation (`mlsys.off-policy-evaluation`)
- Model-grader bias (`mlsys.judge-bias`)
- Multiple comparisons correction (`mlsys.multiple-comparisons`)

### Operations and recovery

How is behavior maintained after deployment?

- Artifact versioning (`mlsys.artifact-version`)
- Production monitoring (`mlsys.monitoring`)
- Staged rollout (`mlsys.rollout`)
- Rollback (`mlsys.rollback`)
- Deployment feedback loop (`mlsys.feedback-loop`)
- Retraining (`mlsys.retraining`)
- Incident response (`mlsys.incident-response`)
- System observability (`mlsys.observability`)
- Irreversible effects (`mlsys.irreversible-effects`)
- Dependency environment (`mlsys.dependency-environment`)
- Workflow orchestration (`mlsys.workflow-orchestration`)
- Experiment reproducibility (`mlsys.experiment-reproducibility`)
- Gang scheduling (`mlsys.gang-scheduling`)
- Job preemption (`mlsys.job-preemption`)
- Bin packing (`mlsys.bin-packing`)
- Multi-tenancy (`mlsys.multi-tenancy`)
- Resource quota (`mlsys.resource-quota`)
- Performance interference (`mlsys.performance-interference`)
- Over-the-air model update (`mlsys.ota-model-update`)
- Model registry (`mlsys.model-registry`)
- Shadow deployment (`mlsys.shadow-deployment`)
- CI/CD for ML (`mlsys.ml-ci-cd`)
- Release gate (`mlsys.release-gate`)
- Drift detection (`mlsys.drift-detection`)
- Prediction logging (`mlsys.inference-logging`)
- Alerting (`mlsys.alerting`)
- Concept drift (`mlsys.concept-drift`)
- Continuous training (`mlsys.continuous-training`)
- Blameless postmortem (`mlsys.postmortem`)
- ML failure attribution (`mlsys.failure-attribution`)
- ML configuration management (`mlsys.configuration-management`)
- Infrastructure as code (`mlsys.infrastructure-as-code`)
- ML platform (`mlsys.ml-platform`)
- ML technical debt (`mlsys.technical-debt`)
- Entanglement (`mlsys.entanglement`)
- Undeclared consumers (`mlsys.undeclared-consumers`)
- MLOps maturity (`mlsys.mlops-maturity`)
- ML team roles (`mlsys.ml-roles`)
- Model deprecation (`mlsys.model-deprecation`)
- Artifact archival (`mlsys.artifact-archival`)
- Retry with backoff (`mlsys.retry-backoff`)
- Circuit breaker (`mlsys.circuit-breaker`)
- Replication and failover (`mlsys.replication-failover`)
- Graceful degradation (`mlsys.graceful-degradation`)
- Mean time between failures (`mlsys.mtbf`)
- Availability (`mlsys.availability`)
- Gray failure (`mlsys.gray-failure`)
- Failure domain (`mlsys.failure-domain`)
- Chaos engineering (`mlsys.chaos-engineering`)
- Game day (`mlsys.game-day`)
- Blast radius (`mlsys.blast-radius`)
- Distributed tracing (`mlsys.distributed-tracing`)
- Model and agent call tracing (`mlsys.llm-call-tracing`)
- Glue code (`mlsys.glue-code`)
- Pipeline jungle (`mlsys.pipeline-jungle`)
- Build-versus-buy decision (`mlsys.build-vs-buy`)
- User feedback collection (`mlsys.user-feedback`)
- Containerization (`mlsys.containerization`)
- Distributed job hang diagnosis (`mlsys.distributed-hang-diagnosis`)
- Fleet data engine (`mlsys.fleet-data-engine`)
- Model and feature compatibility (`mlsys.model-feature-compatibility`)
- Data deletion propagation (`mlsys.deletion-propagation`)
- Health checks and readiness probes (`mlsys.health-checks`)
- Error-correcting code memory (`mlsys.ecc-memory`)
- Metastable failure (`mlsys.metastable-failure`)
- Misuse monitoring (`mlsys.misuse-monitoring`)
- Telemetry sampling and cardinality control (`mlsys.telemetry-sampling`)

### Security, privacy, and responsibility

Whose data, authority, and outcomes need protection?

- Threat modeling (`mlsys.threat-model`)
- Data poisoning (`mlsys.poisoning`)
- Adversarial evasion (`mlsys.evasion`)
- Privacy requirements (`mlsys.privacy`)
- Differential privacy (`mlsys.differential-privacy`)
- Fairness criteria (`mlsys.fairness`)
- Authorization (`mlsys.authorization`)
- Execution isolation (`mlsys.isolation`)
- Human oversight (`mlsys.human-oversight`)
- Privacy leakage (`mlsys.privacy-leakage`)
- Data retention (`mlsys.data-retention`)
- Backdoor attack (`mlsys.backdoor`)
- Jailbreak (`mlsys.jailbreak`)
- Adversarial training (`mlsys.adversarial-training`)
- Model extraction (`mlsys.model-extraction`)
- Model watermarking (`mlsys.model-watermarking`)
- Model artifact signing (`mlsys.artifact-signing`)
- Unsafe model deserialization (`mlsys.unsafe-deserialization`)
- ML supply-chain compromise (`mlsys.supply-chain-compromise`)
- Membership inference (`mlsys.membership-inference`)
- Model inversion (`mlsys.model-inversion`)
- DP-SGD (`mlsys.dp-sgd`)
- Machine unlearning (`mlsys.machine-unlearning`)
- Secure aggregation (`mlsys.secure-aggregation`)
- Least privilege (`mlsys.least-privilege`)
- Confidential computing (`mlsys.confidential-computing`)
- Remote attestation (`mlsys.remote-attestation`)
- Side-channel attack (`mlsys.side-channel`)
- Hazard analysis (`mlsys.hazard-analysis`)
- System-theoretic process analysis (`mlsys.stpa`)
- Failure modes and effects analysis (`mlsys.fmea`)
- Guardrails (`mlsys.guardrails`)
- Goal Structuring Notation (`mlsys.goal-structuring-notation`)
- Functional safety (`mlsys.functional-safety`)
- Safety integrity level (`mlsys.safety-integrity-level`)
- Safety performance indicator (`mlsys.safety-performance-indicator`)
- Group fairness metrics (`mlsys.group-fairness-metric`)
- Individual fairness (`mlsys.individual-fairness`)
- Fairness criteria incompatibility (`mlsys.fairness-incompatibility`)
- Bias mitigation (`mlsys.bias-mitigation`)
- Feature attribution (`mlsys.feature-attribution`)
- Counterfactual explanation (`mlsys.counterfactual-explanation`)
- Inherently interpretable model (`mlsys.interpretable-model`)
- Model card (`mlsys.model-card`)
- Datasheet for datasets (`mlsys.datasheet`)
- Contestability and recourse (`mlsys.contestability`)
- Automation bias (`mlsys.automation-bias`)
- Authority handover (`mlsys.authority-handover`)
- AI risk management (`mlsys.ai-risk-management`)
- Algorithmic impact assessment (`mlsys.impact-assessment`)
- Algorithmic audit (`mlsys.algorithmic-audit`)
- Risk-tiered AI regulation (`mlsys.risk-tiered-regulation`)
- Frontier safety framework (`mlsys.frontier-safety-framework`)
- Homomorphic-encryption inference (`mlsys.he-inference`)
- Secure multi-party computation inference (`mlsys.secure-computation-inference`)
- Red teaming (`mlsys.red-teaming`)
- Data governance (`mlsys.data-governance`)
- Consent and purpose limitation (`mlsys.data-consent`)
- Human-AI interaction design (`mlsys.human-ai-interaction`)
- Capability-based authorization (`mlsys.capability-based-security`)
- Ephemeral sandbox (`mlsys.ephemeral-sandbox`)
- Network egress control (`mlsys.egress-control`)
- Information-flow control (`mlsys.information-flow-control`)
- Event data recorder (`mlsys.event-data-recorder`)
- Decision accountability record (`mlsys.decision-accountability`)
- Safety change-impact analysis (`mlsys.change-impact-analysis`)
- Byzantine-robust aggregation (`mlsys.byzantine-robust-aggregation`)
- Sensor spoofing (`mlsys.sensor-spoofing`)
- Data and model licensing (`mlsys.data-model-licensing`)
- Content provenance (`mlsys.content-provenance`)
- Workload identity and secrets management (`mlsys.secrets-management`)
- Certified robustness by randomized smoothing (`mlsys.randomized-smoothing`)
- Training data extraction (`mlsys.training-data-extraction`)
- Resource-exhaustion attack (`mlsys.resource-exhaustion-attack`)
- Generated-content watermarking (`mlsys.generated-content-watermarking`)
- Representational bias evaluation (`mlsys.representational-bias-evaluation`)
- Confused deputy (`mlsys.confused-deputy`)
- Fault tree analysis (`mlsys.fault-tree-analysis`)
- Proxy discrimination (`mlsys.proxy-discrimination`)

### Energy and economics

What does useful work cost over the system lifecycle?

- Operating energy (`mlsys.energy`)
- Lifecycle carbon (`mlsys.carbon`)
- Cost per useful outcome (`mlsys.unit-cost`)
- Capacity planning (`mlsys.capacity-planning`)
- Energy per inference (`mlsys.energy-per-inference`)
- Power usage effectiveness (`mlsys.pue`)
- Power oversubscription (`mlsys.power-oversubscription`)
- Dominant resource fairness (`mlsys.dominant-resource-fairness`)
- Spot and preemptible capacity (`mlsys.spot-capacity`)
- Energy efficiency (`mlsys.energy-efficiency`)
- Grid carbon intensity (`mlsys.carbon-intensity`)
- Operational carbon (`mlsys.operational-carbon`)
- Emission scopes (`mlsys.emission-scopes`)
- Carbon-aware scheduling (`mlsys.carbon-aware-scheduling`)
- Embodied carbon (`mlsys.embodied-carbon`)
- Life cycle assessment (`mlsys.lifecycle-assessment`)
- Water footprint (`mlsys.water-footprint`)
- Hardware service lifetime (`mlsys.hardware-lifetime`)
- Electronic waste (`mlsys.e-waste`)
- Total cost of ownership (`mlsys.tco`)
- Cost per token (`mlsys.cost-per-token`)
- Cost attribution (`mlsys.cost-attribution`)
- Energy measurement methodology (`mlsys.energy-measurement`)
- Idle vs. dynamic power (`mlsys.idle-dynamic-power`)
- Cloud service models (`mlsys.cloud-service-model`)
- Grid interconnection (`mlsys.grid-interconnection`)
- On-site power generation (`mlsys.on-site-generation`)
- Datacenter demand response (`mlsys.demand-response`)
- Rebound effect (`mlsys.rebound-effect`)
- Data transfer cost (`mlsys.data-transfer-cost`)
- Fine-grained accelerator sharing (`mlsys.fine-grained-accelerator-sharing`)

### Agent systems

How are model-directed trajectories bounded and evaluated?

- Agent trajectory (`mlsys.trajectory`)
- Agent planning (`mlsys.planning`)
- Tool contract (`mlsys.tool-contract`)
- Prompt injection (`mlsys.prompt-injection`)
- Trajectory evaluation (`mlsys.agent-evaluation`)
- Reward hacking (`mlsys.reward-hacking`)
- Context compaction (`mlsys.context-compaction`)
- Long-context degradation (`mlsys.context-degradation`)
- Long-term agent memory (`mlsys.agent-memory`)
- Function calling (`mlsys.function-calling`)
- Tool discovery protocol (`mlsys.model-context-protocol`)
- Computer use (`mlsys.computer-use`)
- Agent harness (`mlsys.agent-harness`)
- Agent execution budget (`mlsys.agent-budget`)
- Action approval gate (`mlsys.approval-gate`)
- Durable execution (`mlsys.durable-execution`)
- Record and replay (`mlsys.deterministic-replay`)
- Suspend and resume (`mlsys.suspend-resume`)
- Multi-agent orchestration (`mlsys.multi-agent-orchestration`)
- Shared agent state (`mlsys.shared-agent-state`)
- Multi-agent coordination failure (`mlsys.coordination-failure`)
- Correlated agent errors (`mlsys.correlated-agent-errors`)
- Context selection (`mlsys.context-selection`)
- Context staleness (`mlsys.context-staleness`)
- Agent interruption and cancellation (`mlsys.agent-interruption`)
- Agent loop detection (`mlsys.agent-loop-detection`)
- Agent credentials and delegated authority (`mlsys.agent-credentials`)
- Agentic workload (`mlsys.agentic-workload`)
- Model-directed control flow (`mlsys.model-directed-control-flow`)

### Physical systems

When is a computed action feasible and admissible?

- Physical dynamics (`mlsys.dynamics`)
- Sensor calibration (`mlsys.sensing`)
- State estimation (`mlsys.state-estimation`)
- State observability (`mlsys.state-observability`)
- Observation age (`mlsys.observation-age`)
- Feedback delay (`mlsys.feedback-delay`)
- Worst-case execution time (`mlsys.wcet`)
- Feedback control (`mlsys.feedback-control`)
- Trajectory feasibility (`mlsys.trajectory-feasibility`)
- Contact and compliance (`mlsys.contact-compliance`)
- Simulation-to-real transfer (`mlsys.sim-to-real`)
- Runtime assurance (`mlsys.runtime-assurance`)
- Stopping envelope (`mlsys.stopping-envelope`)
- Operational design domain (`mlsys.operating-domain`)
- Safety case (`mlsys.safety-case`)
- Functional insufficiency (SOTIF) (`mlsys.functional-insufficiency`)
- Safety filter (`mlsys.safety-filter`)
- Control barrier function (`mlsys.control-barrier-function`)
- Simplex architecture (`mlsys.simplex-architecture`)
- Minimal-risk condition (`mlsys.minimal-risk-condition`)
- Robot perception (`mlsys.robot-perception`)
- Sensor fusion (`mlsys.sensor-fusion`)
- Sensor time synchronization (`mlsys.time-synchronization`)
- Recursive state filtering (`mlsys.recursive-filtering`)
- Simultaneous localization and mapping (`mlsys.slam`)
- Learned world model (`mlsys.world-model`)
- Motion planning (`mlsys.motion-planning`)
- Model predictive control (`mlsys.model-predictive-control`)
- Real-time scheduling (`mlsys.real-time-scheduling`)
- Mixed-criticality partitioning (`mlsys.mixed-criticality`)
- Vision-language-action model (`mlsys.vla-model`)
- Teleoperation data (`mlsys.teleoperation-data`)
- Action chunking (`mlsys.action-chunking`)
- Domain randomization (`mlsys.domain-randomization`)
- Streaming speech recognition (`mlsys.streaming-asr`)
- Voice interaction pipeline (`mlsys.voice-pipeline`)
- Actuator limits (`mlsys.actuator-limits`)
- Watchdog timer (`mlsys.watchdog-timer`)
- Task and motion planning (`mlsys.task-and-motion-planning`)
- Multi-object tracking (`mlsys.multi-object-tracking`)
- Diffusion or flow-matching policy (`mlsys.diffusion-policy`)
- Robotics middleware and dataflow (`mlsys.robotics-middleware`)
- Parallel physics simulation (`mlsys.parallel-simulation`)
- Perception latency budget (`mlsys.perception-latency-budget`)
- Onboard compute partitioning (`mlsys.onboard-compute-partitioning`)
- Active sensor interference (`mlsys.sensor-interference`)
- Interrupt latency (`mlsys.interrupt-latency`)
- Priority inversion (`mlsys.priority-inversion`)
- System identification (`mlsys.system-identification`)
- Motion forecasting (`mlsys.motion-forecasting`)
- Action tokenization (`mlsys.action-tokenization`)
- Over-refusal (`mlsys.over-refusal`)
- Coordinate frames and transforms (`mlsys.coordinate-frames`)
- Sensor degradation (`mlsys.sensor-degradation`)
- Cross-embodiment training (`mlsys.cross-embodiment-training`)
- Dual-rate hierarchical policy (`mlsys.dual-rate-hierarchical-policy`)
- Robot kinematics (`mlsys.robot-kinematics`)

### Retrieval and search

How is relevant information located within a resource budget?

- Retrieval (`mlsys.retrieval`)
- Approximate nearest-neighbor search (`mlsys.ann`)
- Retrieval-augmented generation (`mlsys.rag`)
- Dense retrieval (`mlsys.dense-retrieval`)
- Lexical retrieval (`mlsys.lexical-retrieval`)
- Reranking (`mlsys.reranking`)
- Graph-based vector index (`mlsys.graph-vector-index`)
- Partitioned and quantized vector index (`mlsys.quantized-vector-index`)
- Filtered vector search (`mlsys.filtered-vector-search`)
- Vector index freshness (`mlsys.index-freshness`)
- Iterative multi-hop retrieval (`mlsys.multi-hop-retrieval`)
- Document chunking (`mlsys.document-chunking`)
- Hybrid retrieval (`mlsys.hybrid-retrieval`)
- Embedding model migration (`mlsys.embedding-migration`)
- RAG pipeline scheduling (`mlsys.rag-pipeline-scheduling`)

## Concept definitions

### System requirements

Identity: `mlsys.requirements`

Measurable desired behavior and constraints for a specified workload.

### Baseline

Identity: `mlsys.baseline`

A reference method or system used to justify the value of a proposed change.

### Proxy objective

Identity: `mlsys.proxy-objective`

A measurable target used in place of the outcome actually desired.

### Service-level objective

Identity: `mlsys.slo`

A target for a defined service measure over a specified population and window.

### Workload characterization

Identity: `mlsys.workload`

The input, request, model, and execution distributions that a system must support.

### Execution placement

Identity: `mlsys.placement`

The choice of where system components execute, such as device, edge, or datacenter.

### Data quality

Identity: `mlsys.data-quality`

Fitness of recorded examples for the intended task, including validity, coverage, and error.

### Label quality

Identity: `mlsys.labels`

The correctness, consistency, and coverage of recorded supervision relative to the intended task, which bound the training signal and the trustworthiness of evaluation.

### Feature engineering

Identity: `mlsys.features`

Constructing model inputs whose semantics and availability match the task.

### Missing data

Identity: `mlsys.missing-data`

Absent observations whose pattern and treatment can change learned behavior.

### Data leakage

Identity: `mlsys.leakage`

Information unavailable at the intended prediction boundary entering development or evaluation.

### Point-in-time correctness

Identity: `mlsys.point-in-time`

Constructing examples using only information available at their stated prediction time.

### Data lineage

Identity: `mlsys.lineage`

Recorded dependencies that identify the sources and transformations of a dataset or artifact.

### Training-serving skew

Identity: `mlsys.train-serve-skew`

A mismatch between the data transformations or feature semantics used in training and serving.

### Dataset deduplication

Identity: `mlsys.deduplication`

Identifying and managing repeated or near-duplicate examples.

### Data selection

Identity: `mlsys.data-selection`

Allocating learning effort to examples according to value, coverage, and acquisition cost.

### Tokenization

Identity: `mlsys.tokenization`

Mapping text into vocabulary-dependent token identifiers, which sets sequence lengths and so influences context usage, embedding size, and processing cost.

### Distribution shift

Identity: `mlsys.distribution-shift`

A difference between development and deployment data-generating distributions.

### Linear models

Identity: `mlsys.linear-model`

Models whose score is linear in their supplied features, with a possible output link function.

### Tree ensembles

Identity: `mlsys.tree-ensemble`

Collections of decision trees whose branch traversal and aggregation determine predictions.

### Kernel methods

Identity: `mlsys.kernel-method`

Models using similarity functions whose storage and evaluation costs depend on their representation.

### Nearest-neighbor models

Identity: `mlsys.nearest-neighbor`

Predictions or retrieval based on similarity to stored examples or vectors.

### Neural networks

Identity: `mlsys.neural-network`

Parameterized compositions of differentiable operations used to learn representations and predictions.

### Inductive bias

Identity: `mlsys.inductive-bias`

Structural assumptions that favor some learned functions over others.

### Embeddings

Identity: `mlsys.embeddings`

Vector representations whose dimension and lookup or construction method affect storage and computation.

### Attention

Identity: `mlsys.attention`

Input-dependent aggregation of value vectors using query-key compatibility scores.

### Position encoding

Identity: `mlsys.position-encoding`

Representations that make sequence position available to a model.

### Training loss

Identity: `mlsys.loss`

A computable criterion optimized on training examples.

### Automatic differentiation

Identity: `mlsys.autodiff`

Computing derivatives by composing local derivative rules through an execution graph.

### Optimization algorithm

Identity: `mlsys.optimizer`

A rule that uses training information to update model parameters and optimizer state.

### Regularization

Identity: `mlsys.regularization`

Constraints or training procedures used to control sensitivity and generalization behavior.

### Training batch size

Identity: `mlsys.batch-size`

The number of examples contributing to an optimization update or microstep.

### Training state

Identity: `mlsys.training-state`

The parameters, gradients, optimizer variables, and other state needed to continue training.

### Activation recomputation

Identity: `mlsys.activation-recompute`

Recreating forward intermediates during backward execution to reduce retained memory.

### Empirical scaling laws

Identity: `mlsys.scaling-laws`

Fitted relationships between resources, data, model size, and observed model performance.

### Hyperparameter search

Identity: `mlsys.hyperparameter-search`

Choosing model or training settings through a sequence of evaluations.

### Experiment-budget allocation

Identity: `mlsys.experiment-budget`

Dividing a finite evaluation budget across configurations, repetitions, and fidelities.

### Transfer learning

Identity: `mlsys.transfer-learning`

Adapting representations or parameters learned from one setting to another.

### Parameter-efficient adaptation

Identity: `mlsys.peft`

Adapting a model by changing a limited parameter subset or added parameterization.

### Tensor layout

Identity: `mlsys.tensor-layout`

The mapping from tensor indices to storage locations, including strides and materialization.

### Matrix multiplication

Identity: `mlsys.matrix-multiply`

A dense linear-algebra operation whose tiling, shape, and precision affect execution cost.

### Numerical precision

Identity: `mlsys.precision`

The representable values and arithmetic behavior of a numerical format.

### Numerical stability

Identity: `mlsys.numerical-stability`

Sensitivity of a computed result to finite-precision operations and intermediate representations.

### Mixed-precision training

Identity: `mlsys.mixed-precision`

Training with multiple numerical formats and explicit handling of their range and accumulation behavior.

### Quantization

Identity: `mlsys.quantization`

Representing values using a reduced discrete set with a specified encoding and approximation rule.

### Sparsity

Identity: `mlsys.sparsity`

A pattern of absent or zero values whose execution benefit depends on representation and support.

### Knowledge distillation

Identity: `mlsys.distillation`

Training a student model using signals from a teacher model or ensemble.

### Graph compilation

Identity: `mlsys.compilation`

Transforming a model or execution graph into an executable representation.

### Kernel fusion

Identity: `mlsys.fusion`

Combining compatible operations to reduce intermediate materialization or dispatch.

### Host dispatch overhead

Identity: `mlsys.dispatch`

Host-side work required to launch or orchestrate device operations.

### Memory capacity

Identity: `mlsys.memory-capacity`

Available storage for concurrently resident state and intermediates.

### Memory bandwidth

Identity: `mlsys.memory-bandwidth`

The sustainable rate at which bytes move across a specified memory interface.

### Compute throughput

Identity: `mlsys.compute-throughput`

The sustainable rate of a specified arithmetic workload on an execution resource.

### Arithmetic intensity

Identity: `mlsys.arithmetic-intensity`

Arithmetic work per byte transferred at a specified memory boundary.

### Roofline bound

Identity: `mlsys.roofline`

A performance bound combining a compute ceiling with an intensity-dependent bandwidth ceiling.

### Data locality

Identity: `mlsys.locality`

Reuse of data near its consumers across time or execution locations.

### Tiling

Identity: `mlsys.tiling`

Partitioning computation so working data can be reused within a chosen storage level.

### Profiling

Identity: `mlsys.profiling`

Measuring execution to attribute time, work, stalls, and resource use.

### Resource utilization

Identity: `mlsys.utilization`

The fraction of a resource's defined service capacity that is occupied over a stated interval.

### Serial fraction

Identity: `mlsys.serial-fraction`

The portion of execution that a proposed parallel or accelerated transformation leaves unchanged.

### Collective communication

Identity: `mlsys.collectives`

Coordinated data exchange and reduction across a process group.

### Data parallelism

Identity: `mlsys.data-parallel`

Executing model replicas on different examples while coordinating learning state.

### Tensor parallelism

Identity: `mlsys.tensor-parallel`

Partitioning a layer's tensors and operations across devices so each computes a slice, reducing per-device state while adding communication along the partitioned dimensions, usually within a fast scale-up domain.

### Pipeline parallelism

Identity: `mlsys.pipeline-parallel`

Partitioning a computation into stages that process different microbatches concurrently.

### Training-state sharding

Identity: `mlsys.state-sharding`

Partitioning parameters, gradients, or optimizer state across workers.

### Communication overlap

Identity: `mlsys.communication-overlap`

Scheduling communication concurrently with independent computation.

### Network topology

Identity: `mlsys.network-topology`

The connectivity and capacity structure that constrains communication paths.

### Stragglers

Identity: `mlsys.stragglers`

Workers or tasks whose delayed completion holds back coordinated work.

### Recovery checkpoint

Identity: `mlsys.checkpoint`

A persisted state from which a specified computation can resume.

### Consistent checkpoint cut

Identity: `mlsys.consistent-cut`

A recoverable collection of distributed states with compatible dependencies.

### Replicated-log consensus

Identity: `mlsys.consensus`

Agreement on an ordered state history under specified timing and failure assumptions.

### Fencing

Identity: `mlsys.fencing`

Rejecting effects from an actor whose epoch or authority is stale.

### Federated learning

Identity: `mlsys.federated-learning`

Coordinating model learning across separately controlled data holders.

### Arrival rate

Identity: `mlsys.arrival-rate`

The frequency and distribution of requests reaching a service boundary.

### Service capacity

Identity: `mlsys.service-capacity`

The completion rate supported by a service under a stated workload and operating point.

### Queueing delay

Identity: `mlsys.queueing`

Time a request waits for a resource before receiving service.

### Tail latency

Identity: `mlsys.tail-latency`

A high quantile of a defined request-latency distribution.

### Request batching

Identity: `mlsys.batching`

Executing requests together to share work or improve resource use.

### Admission control

Identity: `mlsys.admission`

Accepting, delaying, or rejecting work according to capacity, policy, and service constraints.

### Autoscaling

Identity: `mlsys.autoscaling`

Changing provisioned service resources in response to measured or predicted demand.

### KV cache

Identity: `mlsys.kv-cache`

Retained attention keys and values reused during autoregressive execution.

### Prefill and decode

Identity: `mlsys.prefill-decode`

The prompt-processing and incremental token-generation phases of autoregressive inference.

### Paged KV allocation

Identity: `mlsys.paged-kv`

Managing attention state in blocks to reduce allocation waste and support sharing.

### Prefix caching

Identity: `mlsys.prefix-cache`

Reusing compatible computed state for an identical request prefix.

### Speculative decoding

Identity: `mlsys.speculative-decoding`

Using proposed tokens and a validating procedure to accelerate generation while preserving a specified output distribution.

### SLO goodput

Identity: `mlsys.slo-goodput`

The rate of useful completed work that satisfies the specified quality and service criteria.

### Generalization

Identity: `mlsys.generalization`

Performance on relevant examples or conditions beyond those used to fit the system.

### Evaluation split

Identity: `mlsys.evaluation-split`

A partitioning protocol that separates learning, selection, and final assessment.

### Model-selection optimism

Identity: `mlsys.selection-bias`

Overstated performance caused by choosing a winner using noisy evaluation results.

### Evaluation metrics

Identity: `mlsys.metrics`

Specified measurements of predictive or task behavior, with explicit populations and aggregation.

### Probability calibration

Identity: `mlsys.calibration`

Agreement between predicted probabilities and observed event frequencies under an evaluation distribution.

### Statistical uncertainty

Identity: `mlsys.statistical-uncertainty`

Uncertainty in an estimated quantity due to finite and dependent observations.

### Benchmark protocol

Identity: `mlsys.benchmark`

A reproducible workload, measurement boundary, and comparison procedure.

### Online experimentation

Identity: `mlsys.online-experiment`

A controlled intervention used to estimate effects on deployed-system outcomes.

### Sliced evaluation

Identity: `mlsys.sliced-evaluation`

Measuring behavior separately on relevant populations, conditions, or workload groups.

### Selective prediction

Identity: `mlsys.selective-prediction`

Predicting on admitted cases and abstaining or deferring the rest.

### Artifact versioning

Identity: `mlsys.artifact-version`

Identifying the exact model, code, data, configuration, and dependencies in an execution or release.

### Production monitoring

Identity: `mlsys.monitoring`

Collecting and interpreting signals about deployed system behavior and health.

### Staged rollout

Identity: `mlsys.rollout`

Increasing exposure to a candidate system through bounded release stages.

### Rollback

Identity: `mlsys.rollback`

Returning to a previously accepted deployment configuration with compatible state.

### Deployment feedback loop

Identity: `mlsys.feedback-loop`

A deployed system's outputs changing the data or behavior encountered by future learning or evaluation.

### Retraining

Identity: `mlsys.retraining`

Training a replacement version of a deployed model on updated or revised data, with timing and data chosen to balance staleness against training and validation cost.

### Incident response

Identity: `mlsys.incident-response`

Detecting, containing, diagnosing, and recovering from unexpected system behavior.

### System observability

Identity: `mlsys.observability`

Instrumentation that permits investigation of internal execution from recorded outputs, traces, and metrics.

### Threat modeling

Identity: `mlsys.threat-model`

Specifying adversaries, assets, interfaces, capabilities, and failure consequences.

### Data poisoning

Identity: `mlsys.poisoning`

Adversarial manipulation of learning data or signals to alter learned behavior.

### Adversarial evasion

Identity: `mlsys.evasion`

Adversarially chosen inference inputs intended to cause harmful model behavior.

### Privacy requirements

Identity: `mlsys.privacy`

Requirements governing the collection, use, retention, and disclosure of information about people or records.

### Differential privacy

Identity: `mlsys.differential-privacy`

A parameterized bound on changes in output distributions under a specified neighboring-dataset relation.

### Fairness criteria

Identity: `mlsys.fairness`

Explicit requirements for how system outcomes or errors are distributed across affected groups.

### Authorization

Identity: `mlsys.authorization`

Determining whether an actor may perform a specified action on a resource.

### Execution isolation

Identity: `mlsys.isolation`

Restricting which resources and effects a computation can access.

### Human oversight

Identity: `mlsys.human-oversight`

An arrangement in which people can understand, intervene in, and contest consequential system behavior.

### Operating energy

Identity: `mlsys.energy`

Energy consumed by a specified system boundary over a defined workload or interval.

### Lifecycle carbon

Identity: `mlsys.carbon`

Emissions attributed to operation and embodied system resources under declared accounting boundaries.

### Cost per useful outcome

Identity: `mlsys.unit-cost`

Total relevant expenditure divided by outcomes meeting a specified acceptance criterion.

### Capacity planning

Identity: `mlsys.capacity-planning`

Provisioning resources for a workload distribution and service objectives.

### Agent trajectory

Identity: `mlsys.trajectory`

A sequence of observations, model proposals, tool actions, and external outcomes for a task.

### Agent planning

Identity: `mlsys.planning`

Constructing and revising prospective actions in response to goals, dependencies, and observations.

### Context window

Identity: `mlsys.context`

The maximum token span a model supports in a stated execution mode, bounding the history it can use and the longest sequences the system must process.

### Retrieval

Identity: `mlsys.retrieval`

Selecting relevant records from an external collection using an explicit query and ranking process.

### Approximate nearest-neighbor search

Identity: `mlsys.ann`

Trading exact neighbor recovery for reduced search cost using an index or approximation.

### Hallucination

Identity: `mlsys.hallucination`

Generated factual assertions or summaries not supported by the relevant source or evidence.

### Tool contract

Identity: `mlsys.tool-contract`

A specification of tool inputs, outputs, authority, effects, and failure behavior.

### Idempotency

Identity: `mlsys.idempotency`

A specified operation's repeated execution having the same intended effect as one execution.

### Compensating action

Identity: `mlsys.compensation`

An explicit action intended to remedy an earlier effect whose direct reversal is unavailable.

### Prompt injection

Identity: `mlsys.prompt-injection`

Untrusted content attempting to redirect model-driven behavior across an instruction or authority boundary.

### Trajectory evaluation

Identity: `mlsys.agent-evaluation`

Assessing task outcomes, intermediate effects, and constraint compliance across a complete agent run.

### Reward hacking

Identity: `mlsys.reward-hacking`

Improving a measured reward through behavior that fails the intended task objective.

### Physical dynamics

Identity: `mlsys.dynamics`

The evolution of a physical system under inputs, disturbances, and constraints.

### Sensor calibration

Identity: `mlsys.sensing`

Establishing how observations relate to physical quantities, coordinates, and time.

### State estimation

Identity: `mlsys.state-estimation`

Inferring latent physical state and uncertainty from observations and a model.

### State observability

Identity: `mlsys.state-observability`

Whether distinct internal states can be distinguished through the available observations and model.

### Observation age

Identity: `mlsys.observation-age`

Elapsed time between a physical observation and its use in a decision.

### Feedback delay

Identity: `mlsys.feedback-delay`

Delay around a sensing, computation, communication, and actuation loop.

### Worst-case execution time

Identity: `mlsys.wcet`

A justified execution-time upper bound under specified hardware and execution assumptions.

### Feedback control

Identity: `mlsys.feedback-control`

Selecting inputs using measured or estimated deviation from desired behavior.

### Trajectory feasibility

Identity: `mlsys.trajectory-feasibility`

Satisfaction of motion, state, input, and dynamic constraints along a proposed trajectory.

### Contact and compliance

Identity: `mlsys.contact-compliance`

Mechanical interaction whose forces depend on geometry, motion, stiffness, and damping.

### Simulation-to-real transfer

Identity: `mlsys.sim-to-real`

Deployment of a learned policy under differences between simulated and physical conditions.

### Runtime assurance

Identity: `mlsys.runtime-assurance`

Monitoring and constraining execution using explicitly stated safety premises and fallback mechanisms.

### Stopping envelope

Identity: `mlsys.stopping-envelope`

The state and clearance conditions within which a specified stopping response remains feasible.

### Operational design domain

Identity: `mlsys.operating-domain`

The conditions under which a physical system's behavior and evidence are intended to apply.

### Safety case

Identity: `mlsys.safety-case`

A structured argument connecting claims, evidence, assumptions, and defeaters for a defined system.

### Structured output constraint

Identity: `mlsys.grammar`

Restricting the syntax or schema of model outputs without establishing factual correctness or authority.

### Data acquisition and sampling

Identity: `mlsys.data-acquisition`

The observation and selection process that determines which examples and populations enter a dataset.

### Data contract

Identity: `mlsys.data-contract`

A specification of schema, semantics, validity rules, and availability for data exchanged between components.

### Delayed outcomes

Identity: `mlsys.delayed-outcomes`

Target outcomes that become observable only after a delay or for a selected subset of cases.

### Decision policy

Identity: `mlsys.decision-policy`

The rule that converts model outputs and system state into actions, thresholds, abstentions, or escalation.

### Robustness evaluation

Identity: `mlsys.robustness-evaluation`

Testing behavior under explicitly chosen shifts, perturbations, faults, or operating stresses.

### Supervised learning

Identity: `mlsys.supervised-learning`

Learning a prediction rule from examples with target outputs.

### Self-supervised learning

Identity: `mlsys.self-supervised-learning`

Learning from targets constructed from the observed data itself.

### Unsupervised learning

Identity: `mlsys.unsupervised-learning`

Learning structure in data without externally supplied example-level targets.

### Reinforcement learning

Identity: `mlsys.reinforcement-learning`

Learning a policy from action-dependent observations and reward signals.

### Imitation learning

Identity: `mlsys.imitation-learning`

Learning a policy from demonstrated behavior and possibly corrective interaction.

### Model artifact

Identity: `mlsys.model-artifact`

The learned parameters, structures, example stores, and metadata required to execute a model.

### Failure model

Identity: `mlsys.failure-model`

The declared classes of faults and timing assumptions under which a mechanism is analyzed.

### Backpressure

Identity: `mlsys.backpressure`

A mechanism that signals downstream saturation to limit or slow upstream production.

### Privacy leakage

Identity: `mlsys.privacy-leakage`

Information about people or records revealed through data, artifacts, or model interaction.

### Execution time

Identity: `mlsys.execution-time`

Elapsed time for a specified execution boundary, workload, and operating condition.

### Irreversible effects

Identity: `mlsys.irreversible-effects`

External consequences that cannot be fully undone by restoring software state, retrying, or performing a compensating action.

### Dataset storage layout

Identity: `mlsys.dataset-storage-layout`

Organization of stored dataset records, columns, partitions, and metadata that determines which data an access reads or decodes.

### Data ingestion

Identity: `mlsys.data-ingestion`

Reading externally produced records into a processing or learning system under declared delivery, ordering, and timing assumptions.

### Schema evolution

Identity: `mlsys.schema-evolution`

Changing a data schema while explicitly managing how existing data and producers or consumers remain interpretable and compatible.

### Test oracle

Identity: `mlsys.test-oracle`

A specification, procedure, reference, or informed judgment used to assess whether observed behavior satisfies stated expectations for a test.

### Integration testing

Identity: `mlsys.integration-testing`

Testing interactions among assembled components against stated interface and behavior expectations.

### Failure injection

Identity: `mlsys.failure-injection`

Deliberately introducing selected faults or disruptive conditions to observe system response under controlled experimental boundaries.

### Dependency environment

Identity: `mlsys.dependency-environment`

The software packages, versions, native libraries, and runtime configuration on which an execution depends.

### Workflow orchestration

Identity: `mlsys.workflow-orchestration`

Coordinating dependent computation steps, their data or control flow, execution conditions, and failure handling.

### Experiment reproducibility

Identity: `mlsys.experiment-reproducibility`

The ability to repeat an experimental procedure and obtain results consistent with declared criteria when its inputs, environment, randomness, and execution conditions are specified.

### Accelerator execution architecture

Identity: `mlsys.accelerator-execution-architecture`

Organization of arithmetic units, execution lanes or threads, scheduling resources, and local storage that determines how supported operations run. SIMD/SIMT, matrix units, and systolic arrays are distinct examples.

### Memory hierarchy

Identity: `mlsys.memory-hierarchy`

Arrangement of storage levels and transfer paths with differing capacity, latency, bandwidth, access control, and management mechanisms. The available levels depend on the platform.

### Host-device interconnect

Identity: `mlsys.host-device-interconnect`

Communication path through which a host and accelerator exchange data and control, with transport, topology, and memory-access semantics that affect transfers.

### Power delivery and integrity

Identity: `mlsys.power-delivery`

Supply paths, conversion stages, limits, and transient behavior that must maintain required voltages while workloads and actuators draw or return power.

### Thermal behavior and throttling

Identity: `mlsys.thermal-dynamics`

Evolution of temperature under heat generation and removal, and the performance controls or operating restrictions that follow under a platform thermal policy.

### Service-level indicator

Identity: `mlsys.sli`

A quantitative measure of one aspect of delivered service, such as latency, availability, or accepted-output rate, computed over a stated population and window.

### Error budget

Identity: `mlsys.error-budget`

The amount of service-level-objective violation permitted over a window, used to pace changes and risk against a reliability target.

### Latency budget

Identity: `mlsys.latency-budget`

An allocation of an end-to-end latency target across the stages of a request path, such as network, preprocessing, inference, and postprocessing.

### Operation and byte accounting

Identity: `mlsys.flop-accounting`

Estimating a workload's arithmetic operations and bytes moved per example, token, or step from model shapes, to predict cost before measurement.

### Memory footprint

Identity: `mlsys.memory-footprint`

The peak memory a workload occupies, summed over parameters, activations, optimizer or cache state, and runtime buffers.

### Workload trace

Identity: `mlsys.workload-trace`

A recorded sequence of requests or jobs with arrival times, sizes, and durations, replayed to evaluate capacity, scheduling, or serving policies.

### Cost-sensitive decision

Identity: `mlsys.cost-sensitive-decision`

Choosing actions or thresholds to minimize expected cost when different error types carry unequal consequences.

### Pareto frontier

Identity: `mlsys.pareto-frontier`

The set of configurations for which no objective, such as accuracy, latency, or cost, can improve without worsening another.

### Transformer

Identity: `mlsys.transformer`

A neural architecture of stacked attention and feed-forward blocks with residual connections and normalization, processing all sequence positions in parallel during training.

### Deep recommendation model

Identity: `mlsys.recommendation-model`

Models combining large sparse embedding tables for categorical features with dense layers, whose cost is often dominated by embedding lookups and memory capacity.

### Diffusion model

Identity: `mlsys.diffusion-model`

A generative model that learns to reverse a gradual noising process, producing samples through many iterative denoising network evaluations.

### Convolution

Identity: `mlsys.convolution`

A locally connected operation applying shared filters across spatial or temporal positions, giving high weight reuse and arithmetic intensity.

### Recurrent computation

Identity: `mlsys.recurrence`

Computation that updates a hidden state step by step over a sequence, creating sequential dependencies that limit parallelism across time.

### Normalization layers

Identity: `mlsys.normalization-layer`

Layers that rescale activations using batch, layer, or root-mean-square statistics to stabilize and speed training of deep networks.

### Mixture of experts

Identity: `mlsys.mixture-of-experts`

An architecture in which a learned router activates a small subset of expert subnetworks per token, decoupling total parameter count from per-token compute.

### Expert routing

Identity: `mlsys.expert-routing`

The gating function that assigns each token to one or more experts, such as top-k softmax gating or expert-choice routing.

### Expert capacity

Identity: `mlsys.expert-capacity`

The maximum number of tokens an expert processes per batch, set by a capacity factor; tokens beyond it are dropped or rerouted.

### Load-balancing loss

Identity: `mlsys.load-balancing-loss`

An auxiliary training objective that penalizes uneven token assignment across experts so routing stays balanced.

### Active parameters

Identity: `mlsys.active-parameters`

The parameters actually used to process one token or example, as distinct from total stored parameters in conditionally computed models.

### Early exit

Identity: `mlsys.early-exit`

Terminating inference at an intermediate layer once a confidence criterion is met, so per-input compute varies with input difficulty.

### Sparse and sliding-window attention

Identity: `mlsys.sliding-window-attention`

Attention restricted to local windows or fixed sparse patterns so cost grows linearly rather than quadratically with sequence length.

### Linear attention

Identity: `mlsys.linear-attention`

Attention reformulated with kernel feature maps so it is computed in linear time with a fixed-size recurrent state.

### State-space sequence model

Identity: `mlsys.state-space-model`

Sequence models built on discretized linear state-space recurrences, trainable in parallel and run with constant-size state per step at inference.

### Grouped-query attention

Identity: `mlsys.grouped-query-attention`

Attention in which groups of query heads share key and value heads, shrinking the key-value cache and decode memory traffic.

### Context-length extension

Identity: `mlsys.context-extension`

Techniques that extend a trained model's usable sequence length, such as position-encoding interpolation followed by long-sequence fine-tuning.

### Weight serialization format

Identity: `mlsys.weight-serialization`

The on-disk encoding of a model's parameter tensors and metadata, governing load time, memory mapping, and safety of deserialization.

### Model exchange format

Identity: `mlsys.model-exchange-format`

A framework-neutral representation of a model's operator graph and weights used to move models between training frameworks and inference runtimes.

### Model export

Identity: `mlsys.model-export`

Converting a framework program into a self-contained deployable artifact by tracing or capturing its graph and fixing operators, shapes, and weights.

### Model signature

Identity: `mlsys.model-signature`

The declared names, types, and shapes of a model artifact's inputs and outputs, forming its interface contract with callers.

### Ridge point

Identity: `mlsys.ridge-point`

The arithmetic intensity at which a roofline's bandwidth and compute ceilings meet, equal to peak compute throughput divided by peak memory bandwidth.

### Compute-bound vs. memory-bound

Identity: `mlsys.bound-regime`

Classifying a kernel or phase by which roofline ceiling limits it, indicating whether more arithmetic throughput or more bandwidth would help.

### Data-movement energy

Identity: `mlsys.data-movement-energy`

The energy cost of moving operands between storage levels, which grows with distance and typically exceeds the energy of the arithmetic performed on them.

### Amdahl's law

Identity: `mlsys.amdahls-law`

The bound that overall speedup from accelerating part of an execution is limited by the fraction left unaccelerated.

### Strong scaling

Identity: `mlsys.strong-scaling`

How time to solution changes as resources increase for a fixed total problem size.

### Weak scaling

Identity: `mlsys.weak-scaling`

How time to solution changes as resources and problem size grow proportionally, keeping work per resource fixed.

### Parallel scaling efficiency

Identity: `mlsys.parallel-efficiency`

Achieved speedup divided by the factor of added resources, measuring how much additional hardware turns into useful progress.

### Little's law

Identity: `mlsys.littles-law`

The relation that the mean number of items in a stable system equals mean arrival rate times mean time in the system.

### Latency-throughput curve

Identity: `mlsys.latency-throughput-curve`

The relationship between offered load and latency, in which queueing delay grows sharply as utilization approaches capacity.

### Arrival burstiness

Identity: `mlsys.arrival-burstiness`

Variability of request inter-arrival times or sizes beyond a Poisson baseline, which raises queueing delay at a given mean utilization.

### Compute-optimal allocation

Identity: `mlsys.compute-optimal-allocation`

Choosing model size and training data quantity to minimize loss under a fixed training compute budget, using fitted scaling laws.

### Training compute estimate

Identity: `mlsys.training-compute-estimate`

Approximating dense transformer training compute as roughly six floating-point operations per parameter per training token.

### Systolic array

Identity: `mlsys.systolic-array`

A grid of processing elements that pass operands rhythmically to neighbors, performing matrix multiplication with high operand reuse and little register-file traffic.

### Matrix engine

Identity: `mlsys.matrix-unit`

A functional unit that executes small dense matrix multiply-accumulate tiles per instruction, supplying most of an accelerator's peak low-precision throughput.

### SIMT execution

Identity: `mlsys.simt`

An execution model in which groups of threads issue common instructions to active lanes, with divergent paths executed under lane masks and architecture-specific reconvergence.

### Accelerator dataflow

Identity: `mlsys.dataflow`

The mapping that decides which operands stay resident in local storage while others stream, such as weight-, output-, or row-stationary schedules.

### Domain-specific architecture

Identity: `mlsys.domain-specific-architecture`

Hardware specialized to a class of workloads, trading programmability for higher performance per watt on its target operations.

### High-bandwidth memory

Identity: `mlsys.hbm`

Stacked DRAM placed beside the processor on an interposer, providing very high bandwidth with limited capacity per stack.

### On-chip SRAM and scratchpads

Identity: `mlsys.on-chip-sram`

Small, fast on-die storage, such as caches, shared memory, and register files, used to stage reused operands near compute units.

### Memory wall

Identity: `mlsys.memory-wall`

The widening gap between processor arithmetic throughput and memory bandwidth or latency, which makes data movement the limiting factor.

### Scale-up interconnect

Identity: `mlsys.scale-up-interconnect`

High-bandwidth links and switches, such as NVLink-class fabrics, joining accelerators within a node or rack into one tightly coupled domain.

### Cache-coherent interconnect

Identity: `mlsys.coherent-interconnect`

A link protocol that lets hosts and devices share or pool memory with hardware coherence, such as CXL over a PCIe physical layer.

### Peer-to-peer device transfer

Identity: `mlsys.peer-to-peer-transfer`

Direct memory access between accelerators, or between accelerators and NICs or storage, without staging data through host memory.

### Thermal design power

Identity: `mlsys.tdp`

The sustained heat dissipation a cooling solution is designed to remove, used as a chip's nominal power envelope.

### Dynamic voltage and frequency scaling

Identity: `mlsys.dvfs`

Adjusting supply voltage and clock frequency at runtime to trade performance for power, since dynamic power grows superlinearly with voltage.

### Dark silicon

Identity: `mlsys.dark-silicon`

The fraction of a chip that cannot be powered at full frequency simultaneously within its power and thermal budget.

### Power capping

Identity: `mlsys.power-capping`

Enforcing a maximum power draw on a device or group by limiting clocks or activity, trading peak throughput for bounded power.

### Microcontroller-class device

Identity: `mlsys.microcontroller`

A low-power processor with on-chip flash and kilobytes to a few megabytes of SRAM, typically without an operating system or virtual memory.

### Heterogeneous mobile SoC

Identity: `mlsys.heterogeneous-soc`

A system-on-chip combining CPU, GPU, DSP, and NPU blocks under shared memory and power limits, across which model operators are partitioned.

### On-device neural accelerator

Identity: `mlsys.edge-npu`

An accelerator integrated in mobile or embedded silicon for low-precision neural inference, usually supporting a restricted set of operators.

### Energy per inference

Identity: `mlsys.energy-per-inference`

Energy consumed to process one input end to end on a device, the binding metric for battery-powered and energy-harvesting deployments.

### On-device memory budget

Identity: `mlsys.on-device-memory-budget`

The flash and SRAM limits that bound a deployed model's stored weights and peak activation memory on embedded hardware.

### Tensor

Identity: `mlsys.tensor`

A multidimensional array carrying shape, element type, device, and layout metadata, the core data abstraction of ML frameworks.

### Computation graph

Identity: `mlsys.computation-graph`

A directed graph of tensor operations and their data dependencies representing a model program, used for differentiation, optimization, and scheduling.

### Eager vs. graph execution

Identity: `mlsys.eager-execution`

The choice between running each operation as the host program issues it and building a whole graph first for later optimization and execution.

### Tracing and JIT capture

Identity: `mlsys.tracing-jit`

Recording operations executed on example inputs to build a graph compiled just in time, with guards or retracing when shapes or control flow change.

### IO-aware attention kernel

Identity: `mlsys.io-aware-attention`

An exact attention implementation that tiles queries, keys, and values through on-chip memory to avoid materializing the full score matrix in off-chip memory.

### Convolution lowering

Identity: `mlsys.convolution-lowering`

Implementing convolution by rearranging input patches into matrices for GEMM, or by direct, FFT, or Winograd algorithms with different memory costs.

### Vendor kernel library

Identity: `mlsys.kernel-library`

Tuned implementations of common operators supplied for a hardware platform, which frameworks call rather than generating their own kernels.

### ML compiler IR

Identity: `mlsys.intermediate-representation`

Program representations at graph, tensor, and loop levels on which compilers apply lowering and optimization passes.

### Kernel autotuning

Identity: `mlsys.autotuning`

Searching kernel schedule parameters, such as tile sizes and loop orders, by measurement or learned cost models to find fast implementations for a target.

### Kernel DSL

Identity: `mlsys.kernel-dsl`

A domain-specific language for writing tiled tensor kernels above the level of vendor assembly, with the compiler handling memory and thread mapping.

### Static memory planning

Identity: `mlsys.memory-planning`

Compile-time assignment of tensor buffers to memory with lifetime-based reuse, reducing peak memory and runtime allocation.

### Device streams

Identity: `mlsys.device-streams`

Ordered queues of device work that the host enqueues asynchronously, with events expressing dependencies and allowing overlap across queues.

### Caching memory allocator

Identity: `mlsys.caching-allocator`

A runtime allocator that retains and reuses freed device memory blocks to avoid costly allocation calls, at the risk of fragmentation.

### Captured graph replay

Identity: `mlsys.graph-replay`

Recording a fixed sequence of kernel launches once and replaying it as a unit to remove per-launch host overhead.

### Loss scaling

Identity: `mlsys.loss-scaling`

Multiplying the loss before backpropagation so small gradients remain representable in reduced-precision formats, then unscaling before the optimizer step.

### Block-scaled number formats

Identity: `mlsys.block-scaled-format`

Low-precision formats in which small blocks of values share a scale factor, extending dynamic range for FP8, FP6, FP4, or INT8 elements.

### Accumulation precision

Identity: `mlsys.accumulation-precision`

The numerical format used to sum products in dot products and reductions, which may differ from input precision and governs rounding-error growth.

### Expert parallelism

Identity: `mlsys.expert-parallel`

Placing different experts of a mixture-of-experts layer on different devices and exchanging routed tokens among them.

### Sequence and context parallelism

Identity: `mlsys.sequence-parallel`

Partitioning activations along the sequence dimension across devices, exchanging keys, values, or partial results so long sequences fit.

### Hybrid parallelism

Identity: `mlsys.hybrid-parallel`

Combining data, tensor, pipeline, and other parallelism dimensions, mapped onto the hardware hierarchy so the most communication-intensive dimension uses the fastest links.

### Pipeline bubble

Identity: `mlsys.pipeline-bubble`

Idle device time during pipeline fill and drain, whose fraction shrinks as the number of microbatches grows relative to pipeline stages.

### Ring all-reduce

Identity: `mlsys.ring-allreduce`

An all-reduce performed as reduce-scatter then all-gather around a ring, bandwidth-optimal for large messages but with latency linear in participants.

### Tree and hierarchical all-reduce

Identity: `mlsys.tree-allreduce`

All-reduce schedules over trees or hierarchical groups, with latency logarithmic in participants, suited to small messages and multi-level networks.

### All-to-all exchange

Identity: `mlsys.all-to-all`

A collective in which every participant sends distinct data to every other, as in expert routing; its throughput is sensitive to bisection bandwidth.

### Alpha-beta communication model

Identity: `mlsys.alpha-beta-model`

A cost model estimating message time as a fixed latency term plus message size times a per-byte cost, used to compare collective algorithms.

### Gradient compression

Identity: `mlsys.gradient-compression`

Reducing gradient communication through quantization, sparsification, or low-rank approximation, often with error feedback to preserve convergence.

### Lease

Identity: `mlsys.lease`

A time-bounded grant of authority or cached state that expires unless renewed, letting a system recover from failed holders without explicit release.

### Leader election

Identity: `mlsys.leader-election`

Selecting a single coordinator among replicas, typically through consensus and leases, so control decisions have one authoritative source.

### Bounded staleness

Identity: `mlsys.bounded-staleness`

A consistency model in which workers may compute on parameters up to a fixed number of updates old, trading freshness for less waiting.

### Federated averaging

Identity: `mlsys.federated-averaging`

A federated algorithm in which sampled clients run several local optimization steps and a server averages their model updates, weighted by data size.

### Statistical heterogeneity

Identity: `mlsys.non-iid-data`

Differences in data distribution and quantity across federated clients, which slow convergence and can bias a shared model.

### Client selection

Identity: `mlsys.client-sampling`

Choosing which available devices participate in each federated round, subject to availability, connectivity, and power constraints.

### Cross-device vs. cross-silo federation

Identity: `mlsys.federation-setting`

The distinction between federations of many intermittently available devices and federations of a few reliable organizational data holders.

### Power usage effectiveness

Identity: `mlsys.pue`

Total facility energy divided by IT equipment energy, measuring overhead from cooling, power conversion, and other non-compute loads.

### Liquid cooling

Identity: `mlsys.liquid-cooling`

Removing heat with liquid through cold plates or immersion, enabling rack power densities beyond the practical limits of air cooling.

### Rack power density

Identity: `mlsys.rack-power-density`

Electrical power drawn per rack, which determines cooling technology, floor layout, and how many accelerators fit within a facility's power envelope.

### Power oversubscription

Identity: `mlsys.power-oversubscription`

Deploying more equipment than the sum of nameplate power a feed supports, relying on non-coincident peaks and power capping to stay within limits.

### Warehouse-scale computer

Identity: `mlsys.warehouse-scale-computer`

A datacenter treated as a single computer whose power, cooling, network, and software are co-designed for a common workload.

### Remote direct memory access

Identity: `mlsys.rdma`

Network transfers that read or write remote memory directly through the network interface without remote CPU involvement, giving low latency and high throughput.

### Lossless network fabric

Identity: `mlsys.lossless-fabric`

A network that avoids packet drops through credit-based or priority flow control, as RDMA transports expect, at the risk of head-of-line blocking.

### Datacenter congestion control

Identity: `mlsys.congestion-control`

End-host and switch mechanisms, such as ECN-driven rate control, that limit queue buildup from incast and synchronized collective traffic.

### Fat-tree (Clos) topology

Identity: `mlsys.fat-tree`

A multistage switch topology providing many equal-cost paths and, without oversubscription, full bisection bandwidth between endpoints.

### Rail-optimized topology

Identity: `mlsys.rail-optimized`

A fabric design connecting the same-index accelerator of every node to a common leaf switch, keeping most collective traffic within one switch hop.

### Bisection bandwidth

Identity: `mlsys.bisection-bandwidth`

The minimum total bandwidth across any cut dividing a network into two equal halves, bounding throughput of all-to-all and global exchanges.

### In-network reduction

Identity: `mlsys.in-network-reduction`

Performing collective reductions inside network switches so data crosses each link fewer times, lowering all-reduce traffic and latency.

### Parallel file system

Identity: `mlsys.parallel-file-system`

A distributed file system that stripes data across many storage servers to deliver aggregate bandwidth to many concurrent clients.

### Object storage

Identity: `mlsys.object-storage`

Storage of immutable named objects behind an HTTP-style interface, offering scalable capacity and throughput with higher per-request latency than file systems.

### Training input pipeline

Identity: `mlsys.input-pipeline`

The read, decode, transform, shuffle, and prefetch stages that deliver batches to accelerators; when slower than compute, they cause data stalls.

### Checkpoint I/O

Identity: `mlsys.checkpoint-io`

Writing and reading large training-state snapshots, where storage bandwidth, asynchronous persistence, and sharded layout determine checkpoint cost.

### Storage tiering and caching

Identity: `mlsys.storage-tiering`

Placing data across tiers, from memory and local NVMe caches to shared and archival storage, according to access frequency and cost.

### Gang scheduling

Identity: `mlsys.gang-scheduling`

Allocating all resources a parallel job needs at once, or none, so tightly coupled workers start together and partial allocations cannot deadlock.

### Job preemption

Identity: `mlsys.job-preemption`

Suspending or evicting a running job to reclaim its resources for higher-priority work, at a cost set by checkpointing and lost progress.

### Dominant resource fairness

Identity: `mlsys.dominant-resource-fairness`

A multi-resource fair-sharing policy that equalizes each user's largest share across resource types such as accelerators, CPUs, and memory.

### Elastic training

Identity: `mlsys.elastic-training`

Training that continues while the number of workers changes, resharding state and adjusting batch size or learning rate as resources join or leave.

### Topology-aware placement

Identity: `mlsys.topology-aware-placement`

Assigning a job's workers to devices and nodes so that heavily communicating groups share fast links or switches.

### Bin packing

Identity: `mlsys.bin-packing`

Placing jobs onto nodes to minimize fragmentation of resources such as accelerators, trading packing density against interference and locality.

### Multi-tenancy

Identity: `mlsys.multi-tenancy`

Sharing infrastructure among teams, jobs, or customers under policies for allocation, isolation, priority, and accounting.

### Resource quota

Identity: `mlsys.resource-quota`

A limit on the resources a tenant may hold or request, sometimes with guaranteed minimums and borrowing of idle capacity.

### Accelerator sharing and partitioning

Identity: `mlsys.gpu-partitioning`

Running several workloads on one accelerator by time-slicing, concurrent contexts, or hardware partitions with dedicated memory and compute slices.

### Spot and preemptible capacity

Identity: `mlsys.spot-capacity`

Discounted compute that the provider may reclaim at short notice, usable by workloads that tolerate interruption.

### Performance interference

Identity: `mlsys.performance-interference`

Slowdown of one workload caused by co-located workloads contending for shared caches, memory bandwidth, network, or power.

### Weak supervision

Identity: `mlsys.weak-supervision`

Producing training labels from noisy programmatic sources such as heuristics, rules, or distant supervision, then modeling and combining their errors.

### Active learning

Identity: `mlsys.active-learning`

Selecting which unlabeled examples to annotate next according to their expected value to the model.

### Inter-annotator agreement

Identity: `mlsys.annotator-agreement`

The degree to which independent annotators assign the same label to the same item, used to measure label reliability and task ambiguity.

### Quality filtering

Identity: `mlsys.quality-filtering`

Removing or down-weighting training examples judged low quality by heuristics, classifiers, or model scores.

### Data mixture

Identity: `mlsys.data-mixture`

The proportions in which source domains or datasets are sampled during training.

### Synthetic data

Identity: `mlsys.synthetic-data`

Training or evaluation examples generated by simulators, programs, or models rather than collected from the target environment.

### Data augmentation

Identity: `mlsys.data-augmentation`

Creating additional training examples by applying label-preserving transformations to existing ones.

### Data validation

Identity: `mlsys.data-validation`

Automated checks of incoming data against expected schema, value ranges, and distributions before it is used for training or serving.

### Feature store

Identity: `mlsys.feature-store`

A system that defines, computes, stores, and serves features consistently for offline training and online inference.

### Stream processing

Identity: `mlsys.stream-processing`

Continuously transforming unbounded event streams with low latency and incrementally maintained state.

### Feature backfill

Identity: `mlsys.feature-backfill`

Recomputing a feature's historical values under a new or corrected definition so training data matches what serving will produce.

### Dataset versioning

Identity: `mlsys.data-versioning`

Recording immutable, addressable versions of datasets so that a training run or evaluation can identify exactly which data it used.

### Data dependency debt

Identity: `mlsys.data-dependency`

Undeclared, unstable, or underused input signals on which a model depends, which can change or break its behavior silently.

### Feature freshness

Identity: `mlsys.feature-freshness`

The age of a feature value at the time it is used for a prediction, relative to the freshness assumed during training.

### Learning-rate schedule

Identity: `mlsys.learning-rate-schedule`

The planned variation of the optimizer step size over training, such as warmup followed by decay.

### Critical batch size

Identity: `mlsys.critical-batch-size`

The batch size beyond which increasing it further yields diminishing reductions in optimization steps to reach a target loss.

### Training instability

Identity: `mlsys.training-instability`

Loss spikes, divergence, or collapse during training, often mitigated by gradient clipping, schedule changes, normalization, or restarting from a checkpoint.

### Gradient accumulation

Identity: `mlsys.gradient-accumulation`

Summing gradients over several micro-batches before one optimizer update to emulate a larger batch within limited memory.

### Training-state offloading

Identity: `mlsys.memory-offloading`

Moving optimizer state, parameters, or activations from accelerator memory to host memory or storage between uses.

### Experiment tracking

Identity: `mlsys.experiment-tracking`

Recording each run's code, data, configuration, metrics, and artifacts in a queryable store for comparison and reproduction.

### Hyperparameter transfer

Identity: `mlsys.hyperparameter-transfer`

Tuning hyperparameters on smaller proxy models and reusing them at larger scale under a parameterization designed to keep them stable.

### Supervised fine-tuning

Identity: `mlsys.supervised-fine-tuning`

Continued training of a pretrained model on input-output demonstrations, such as instruction-response pairs, to shape task behavior.

### Low-rank adaptation

Identity: `mlsys.lora`

Fine-tuning by learning low-rank update matrices added to frozen pretrained weights.

### RL from human feedback

Identity: `mlsys.rlhf`

Optimizing a model's policy against a reward model trained on human preference comparisons, usually with a penalty for drifting from a reference model.

### Reward model

Identity: `mlsys.reward-model`

A learned model that scores outputs or trajectories to supply the training signal for policy optimization.

### Direct preference optimization

Identity: `mlsys.preference-optimization`

Training a policy directly on preferred-versus-rejected output pairs with a classification-style loss, without a separate reward model or RL loop.

### RL with verifiable rewards

Identity: `mlsys.rlvr`

Reinforcement learning whose rewards come from programmatic outcome checks, making verifier correctness and reward computation part of the training system.

### RL rollout generation

Identity: `mlsys.rl-rollout`

Sampling responses or trajectories from the current policy, often with an inference engine, to produce experience for policy updates.

### Policy weight synchronization

Identity: `mlsys.weight-sync`

Transferring updated policy weights from training workers to rollout workers, whose cost and frequency shape RL post-training throughput.

### Policy staleness

Identity: `mlsys.policy-staleness`

The lag between the policy version that generated rollouts and the version being updated, introduced by asynchronous rollout and training.

### Online learning

Identity: `mlsys.online-learning`

Updating a model incrementally as each example or small batch arrives, rather than retraining on a fixed dataset.

### Incremental training

Identity: `mlsys.incremental-training`

Continuing training of an existing model from its current weights on new data instead of training from scratch.

### Catastrophic forgetting

Identity: `mlsys.catastrophic-forgetting`

Loss of previously learned capability when a model is trained on new data or tasks.

### Experience replay

Identity: `mlsys.experience-replay`

Mixing stored past examples into updates on new data to reduce forgetting or reuse experience.

### On-device training

Identity: `mlsys.on-device-training`

Updating model parameters on the deployment device using local data under its memory, energy, and compute limits.

### Cross-validation

Identity: `mlsys.cross-validation`

Estimating performance by repeatedly training on some data partitions and evaluating on the held-out remainder.

### Sequential testing

Identity: `mlsys.sequential-testing`

Experimental analysis that permits repeated looks at accumulating data while controlling error rates, unlike fixed-horizon tests.

### Experiment interference

Identity: `mlsys.experiment-interference`

Treatment of one experimental unit affecting outcomes of another, violating the no-interference assumption (SUTVA) of standard A/B analysis.

### Conformal prediction

Identity: `mlsys.conformal-prediction`

Constructing prediction sets with finite-sample coverage guarantees under exchangeability, using held-out calibration scores.

### Slice discovery

Identity: `mlsys.slice-discovery`

Automatically finding coherent data subsets on which a model underperforms, without predefined slice labels.

### Model-based grading

Identity: `mlsys.llm-judge`

Using a language model to score or compare generated outputs against a rubric, with its own biases and error rates to be measured.

### pass@k

Identity: `mlsys.pass-at-k`

The probability that at least one of k sampled outputs satisfies a correctness check.

### Pairwise preference evaluation

Identity: `mlsys.pairwise-preference-eval`

Ranking models from judgments of which of two outputs is better, typically aggregated with a Bradley-Terry or Elo-style model.

### Human evaluation

Identity: `mlsys.human-evaluation`

Assessment of system outputs by people following a protocol, used where automatic metrics or model graders are not trusted.

### Benchmark contamination

Identity: `mlsys.benchmark-contamination`

Presence of evaluation items, or close variants, in a model's training data, inflating measured performance.

### Time to train

Identity: `mlsys.time-to-train`

Wall-clock time for a system to train a specified model to a specified quality target under fixed run rules.

### Inference load scenario

Identity: `mlsys.inference-load-scenario`

A specified query-arrival pattern and latency constraint, such as single-stream, server, or offline, under which inference performance is measured.

### Metamorphic testing

Identity: `mlsys.metamorphic-testing`

Testing by checking that known relations hold between outputs for related inputs, when correct outputs for single inputs are unknown.

### Behavioral testing

Identity: `mlsys.behavioral-testing`

Testing model capabilities with targeted input suites for invariance, directional change, and minimum functionality, independent of aggregate accuracy.

### Production-readiness assessment

Identity: `mlsys.production-readiness`

A structured rubric of data, model, infrastructure, and monitoring tests used to judge whether an ML system is ready to operate.

### Online inference

Identity: `mlsys.online-inference`

Producing predictions on demand for individual requests within a latency target.

### Batch inference

Identity: `mlsys.batch-inference`

Computing predictions for many inputs in scheduled jobs and storing them for later lookup.

### Model server

Identity: `mlsys.model-server`

A service that loads model artifacts and executes inference requests behind a network interface, managing versions, batching, and resources.

### Multi-model serving

Identity: `mlsys.multi-model-serving`

Hosting many models or adapters on shared serving resources, trading isolation and load time against utilization.

### Model cascade

Identity: `mlsys.model-cascade`

Sending a request to cheaper models first and escalating to more capable models only when a confidence or quality criterion is not met.

### Model routing

Identity: `mlsys.model-routing`

Choosing which model serves each request from request features, predicted quality, and cost.

### Time to first token

Identity: `mlsys.ttft`

Latency from request arrival until the first generated token is returned, dominated by queueing and prefill.

### Time per output token

Identity: `mlsys.tpot`

For a request producing multiple tokens, its elapsed decoding time after the first output divided by its remaining generated-token count, summarizing average decoding speed at a declared measurement boundary.

### Hedged requests

Identity: `mlsys.hedged-requests`

Issuing a duplicate request to another replica when the first has not responded within a delay, and using whichever returns first.

### Continuous batching

Identity: `mlsys.continuous-batching`

Scheduling at each generation iteration so finished sequences leave and new requests join a running batch.

### Request load balancing

Identity: `mlsys.load-balancing`

Distributing requests across replicas according to load, locality, or cached state.

### Model cold start

Identity: `mlsys.cold-start`

Delay before a newly started replica can serve, from provisioning, loading weights, and warming runtime state.

### Chunked prefill

Identity: `mlsys.chunked-prefill`

Splitting a long prompt's prefill into chunks interleaved with decode steps to bound per-iteration latency.

### Prefill-decode disaggregation

Identity: `mlsys.pd-disaggregation`

Running prefill and decode on separate resource pools, transferring KV state between them, so each phase is provisioned and batched independently.

### Decoding and sampling controls

Identity: `mlsys.sampling-controls`

Parameters and rules, such as greedy, temperature, top-k, and nucleus sampling, that determine how tokens are chosen from a model's distribution.

### Constrained decoding

Identity: `mlsys.constrained-decoding`

Masking disallowed tokens at each step, typically from a grammar or schema automaton, so output satisfies a structural constraint.

### Output validation

Identity: `mlsys.output-validation`

Checking generated output against a schema or rules after generation and repairing, retrying, or rejecting it on failure.

### Reasoning budget

Identity: `mlsys.reasoning-budget`

A limit on the intermediate tokens or steps a model may spend before answering, set to trade quality against latency and cost.

### Best-of-n sampling

Identity: `mlsys.best-of-n`

Generating n candidate outputs and returning the one ranked highest by a scorer or verifier.

### Self-consistency voting

Identity: `mlsys.self-consistency`

Sampling multiple reasoning paths and returning the most frequent final answer.

### Verifier

Identity: `mlsys.verifier`

A model or program that judges whether a candidate output or reasoning step is correct, used for selection, search, or reward.

### Process reward model

Identity: `mlsys.process-reward-model`

A verifier that scores intermediate reasoning steps rather than only final answers.

### Inference-time search

Identity: `mlsys.inference-time-search`

Exploring a tree of partial solutions, as in beam or tree search, guided by a scorer, to find a higher-quality output.

### Test-time scaling

Identity: `mlsys.test-time-scaling`

The empirical relation between inference compute spent per query and output quality, used to allocate compute between model size and inference.

### Retrieval-augmented generation

Identity: `mlsys.rag`

Conditioning generation on documents retrieved for the request so outputs can draw on information outside the model's parameters.

### Dense retrieval

Identity: `mlsys.dense-retrieval`

Retrieving records by similarity between learned embeddings of query and records.

### Lexical retrieval

Identity: `mlsys.lexical-retrieval`

Retrieving records by weighted term overlap with the query using an inverted index, as in BM25.

### Reranking

Identity: `mlsys.reranking`

Reordering a retrieved candidate set with a more expensive relevance model before use.

### Model conversion

Identity: `mlsys.model-conversion`

Translating a trained model into the format and operator set of a target runtime, possibly changing numerics and supported operations.

### On-device inference runtime

Identity: `mlsys.on-device-runtime`

An interpreter or compiled engine that executes models locally on personal or embedded devices, including inside browsers, under the target environment's memory, operator, and execution limits.

### Over-the-air model update

Identity: `mlsys.ota-model-update`

Delivering new model versions to deployed devices remotely, with staging, verification, and rollback across the device fleet.

### Device-cloud partitioning

Identity: `mlsys.device-cloud-partitioning`

Splitting inference between device and server, by layer or by model, to balance latency, energy, bandwidth, and privacy.

### Device heterogeneity

Identity: `mlsys.device-heterogeneity`

Variation in hardware, operating system, and runtime capability across a deployed device fleet that a single model release must accommodate.

### Model registry

Identity: `mlsys.model-registry`

A versioned store of model artifacts with metadata, lineage, and lifecycle stage used to govern promotion and deployment.

### Shadow deployment

Identity: `mlsys.shadow-deployment`

Running a candidate model on live traffic without serving its outputs, to compare it against the production model.

### CI/CD for ML

Identity: `mlsys.ml-ci-cd`

Automated building, testing, and deployment of ML code, data checks, and model artifacts on each change.

### Release gate

Identity: `mlsys.release-gate`

A predeclared set of checks a candidate model must pass before it advances to a wider deployment stage.

### Drift detection

Identity: `mlsys.drift-detection`

Statistical tests or distances comparing production and reference distributions of inputs, predictions, or outcomes to flag change.

### Prediction logging

Identity: `mlsys.inference-logging`

Recording requests, model versions, outputs, and later outcomes so production behavior can be analyzed, debugged, and reused as data.

### Alerting

Identity: `mlsys.alerting`

Rules that turn monitored signals into notifications for responders, tuned to balance missed incidents against alert fatigue.

### Concept drift

Identity: `mlsys.concept-drift`

Change over time in the relationship between inputs and targets, so the same input warrants a different prediction.

### Continuous training

Identity: `mlsys.continuous-training`

An automated pipeline that retrains, validates, and proposes models on a schedule or when triggered by data or performance signals.

### Blameless postmortem

Identity: `mlsys.postmortem`

A written analysis after an incident of its timeline, causes, and corrective actions, focused on system fixes rather than blame.

### ML failure attribution

Identity: `mlsys.failure-attribution`

Determining whether a production failure stems from data, model, code, configuration, or infrastructure, or from an adversary.

### ML configuration management

Identity: `mlsys.configuration-management`

Treating model, feature, and pipeline configuration as versioned, reviewed, and tested code.

### Infrastructure as code

Identity: `mlsys.infrastructure-as-code`

Declaring compute, storage, and service infrastructure in versioned machine-readable files from which environments are provisioned.

### ML platform

Identity: `mlsys.ml-platform`

Shared internal infrastructure and services that standardize how teams train, deploy, and operate models.

### ML technical debt

Identity: `mlsys.technical-debt`

Accumulated system costs from shortcuts in ML code, data dependencies, configuration, and pipelines that make future change slower and riskier.

### Entanglement

Identity: `mlsys.entanglement`

The property that changing any input, feature, or hyperparameter of a model can change its behavior on everything else.

### Undeclared consumers

Identity: `mlsys.undeclared-consumers`

Systems that use a model's outputs without the model owner's knowledge, creating hidden coupling.

### MLOps maturity

Identity: `mlsys.mlops-maturity`

Staged levels of automation, from manual training and deployment to automated continuous training and delivery.

### ML team roles

Identity: `mlsys.ml-roles`

The division of responsibilities among data, modeling, platform, and operations roles that owns an ML system across its lifecycle.

### Model deprecation

Identity: `mlsys.model-deprecation`

Planned withdrawal of a model from service, including notifying and migrating its consumers before removing it.

### Artifact archival

Identity: `mlsys.artifact-archival`

Preserving retired models, data references, and evaluation records needed for audit, reproduction, or rollback after service ends.

### Data retention

Identity: `mlsys.data-retention`

Policy and mechanisms that set how long data, logs, and derived artifacts are kept and when they are deleted.

### Model FLOPs utilization

Identity: `mlsys.model-flops-utilization`

Model-required floating-point operations per second achieved, divided by the aggregate peak FLOP rate of the hardware used, excluding recomputation.

### Bottleneck analysis

Identity: `mlsys.bottleneck-analysis`

Identifying the resource or stage, such as compute, memory bandwidth, I/O, or communication, whose capacity limits end-to-end throughput or latency for a workload.

### Training goodput

Identity: `mlsys.training-goodput`

The fraction of allocated accelerator time that produces retained training progress, after losses to scheduling, failures, restarts, and lost work.

### Pruning

Identity: `mlsys.pruning`

Removing weights, neurons, channels, or other structures judged unimportant from a network, usually followed by fine-tuning to recover quality.

### Structured pruning

Identity: `mlsys.structured-pruning`

Pruning whole channels, filters, heads, or layers so that the remaining model stays dense and runs faster without sparse-execution support.

### Low-rank factorization

Identity: `mlsys.low-rank-factorization`

Approximating a weight matrix or tensor by a product of smaller factors of reduced rank to cut parameters and operations.

### Quantization-aware training

Identity: `mlsys.quantization-aware-training`

Training or fine-tuning with simulated quantization in the forward pass so the model adapts to reduced precision, in contrast to post-training quantization.

### Neural architecture search

Identity: `mlsys.neural-architecture-search`

Automated search over a defined space of network architectures for designs that optimize a stated objective such as accuracy under a cost budget.

### Hardware-aware NAS

Identity: `mlsys.hardware-aware-nas`

Architecture search whose objective includes latency, energy, or memory measured or predicted on the target device rather than proxy counts such as FLOPs.

### Depthwise separable convolution

Identity: `mlsys.depthwise-separable-convolution`

Factoring a standard convolution into a per-channel spatial convolution followed by a pointwise convolution, reducing operations and parameters.

### Accuracy-efficiency frontier

Identity: `mlsys.efficiency-frontier`

The set of model configurations for which no alternative improves quality without worsening a stated cost measure, such as latency, memory, or energy, on a stated platform.

### Model-hardware co-design

Identity: `mlsys.model-hardware-co-design`

Jointly choosing model architecture and hardware or kernel design so that each exploits the other's structure.

### Retry with backoff

Identity: `mlsys.retry-backoff`

Reissuing a failed request after increasing, typically jittered, delays and within a retry budget, so that transient faults recover without overloading dependencies.

### Circuit breaker

Identity: `mlsys.circuit-breaker`

A client-side guard that stops sending requests to a failing dependency after an error threshold and probes it before resuming traffic.

### Replication and failover

Identity: `mlsys.replication-failover`

Maintaining redundant replicas of a service or state and redirecting work to a healthy replica when one fails.

### Graceful degradation

Identity: `mlsys.graceful-degradation`

Continuing to serve with reduced quality, features, or freshness under failure or overload instead of failing outright.

### Optimal checkpoint interval

Identity: `mlsys.checkpoint-interval`

The checkpoint period that minimizes expected lost time, balancing checkpoint write cost against expected rework, approximately the square root of twice the write cost times MTBF.

### Asynchronous checkpointing

Identity: `mlsys.async-checkpointing`

Snapshotting training state quickly to host memory and persisting it in the background so computation resumes before the durable write completes.

### Silent data corruption

Identity: `mlsys.silent-data-corruption`

Hardware producing wrong computational results without raising an error or crash, so the corruption propagates undetected.

### Mean time between failures

Identity: `mlsys.mtbf`

The expected operating time between failures of a component or system; for independent components, system MTBF shrinks roughly in inverse proportion to component count.

### Availability

Identity: `mlsys.availability`

The fraction of time or requests for which a system performs its specified function, approximately MTBF divided by MTBF plus mean time to repair.

### Gray failure

Identity: `mlsys.gray-failure`

A partial or degraded failure that some observers detect and others do not, such as a slow or lossy node that still passes health checks.

### Failure domain

Identity: `mlsys.failure-domain`

A set of components that can fail together from a shared cause, such as a rack, power feed, switch, or software release.

### Algorithm-based fault tolerance

Identity: `mlsys.abft`

Detecting or correcting computational errors by carrying checksums through an algorithm, such as matrix multiplication, and verifying invariants on its outputs.

### Chaos engineering

Identity: `mlsys.chaos-engineering`

Running controlled experiments that inject real-world failures into production-like systems to test a hypothesis about steady-state behavior.

### Game day

Identity: `mlsys.game-day`

A scheduled exercise in which teams deliberately trigger failure scenarios to rehearse detection, response, and recovery procedures.

### Hardware fault injection

Identity: `mlsys.hardware-fault-injection`

Emulating hardware faults such as bit flips in weights, activations, or registers to measure how errors propagate to model outputs.

### Blast radius

Identity: `mlsys.blast-radius`

The extent of users, components, or effects that a single fault, change, or experiment can impact.

### Backdoor attack

Identity: `mlsys.backdoor`

Implanting a hidden behavior in a model, via poisoned data or modified weights, that activates only on inputs containing an attacker-chosen trigger.

### Jailbreak

Identity: `mlsys.jailbreak`

A user-crafted input that induces a model to bypass its safety training or usage policy and produce disallowed output.

### Adversarial training

Identity: `mlsys.adversarial-training`

Training on adversarially perturbed inputs generated during training to increase robustness within a specified perturbation set.

### Model extraction

Identity: `mlsys.model-extraction`

Reconstructing a functionally equivalent copy or the parameters of a model by querying its prediction interface.

### Model watermarking

Identity: `mlsys.model-watermarking`

Embedding a verifiable signature in a model's parameters or behavior so that ownership of copies or derivatives can later be demonstrated.

### Model artifact signing

Identity: `mlsys.artifact-signing`

Cryptographically signing model files and their provenance metadata so that consumers can verify integrity and origin before loading.

### Unsafe model deserialization

Identity: `mlsys.unsafe-deserialization`

Code execution triggered by loading a model stored in an executable serialization format, such as pickle, from an untrusted source.

### ML supply-chain compromise

Identity: `mlsys.supply-chain-compromise`

Introducing malicious or vulnerable components into an ML system through third-party datasets, pretrained weights, libraries, build tools, or hardware.

### Membership inference

Identity: `mlsys.membership-inference`

Inferring whether a specific record was in a model's training data from the model's outputs or parameters.

### Model inversion

Identity: `mlsys.model-inversion`

Reconstructing sensitive attributes or representative training inputs from a model's outputs, confidence scores, or shared parameter gradients.

### DP-SGD

Identity: `mlsys.dp-sgd`

Stochastic gradient descent with per-example gradient clipping and calibrated noise, with privacy loss tracked across steps by an accountant.

### Machine unlearning

Identity: `mlsys.machine-unlearning`

Removing the influence of specified training records from a trained model, either exactly by retraining or approximately at lower cost.

### Secure aggregation

Identity: `mlsys.secure-aggregation`

A cryptographic protocol that lets a server compute the sum of client updates without seeing any individual client's update.

### Least privilege

Identity: `mlsys.least-privilege`

Granting each component or actor only the permissions its task requires, for only as long as it requires them.

### Confidential computing

Identity: `mlsys.confidential-computing`

Protecting data and models while in use by running computation inside hardware-isolated, attestable trusted execution environments.

### Remote attestation

Identity: `mlsys.remote-attestation`

A procedure by which a platform proves to a remote verifier the identity and measured state of its hardware and software before receiving secrets.

### Side-channel attack

Identity: `mlsys.side-channel`

Inferring secrets from observable resource behavior, such as timing, power, memory, or cache contention, across an isolation boundary.

### Energy efficiency

Identity: `mlsys.energy-efficiency`

Useful work completed per unit energy, such as inferences or tokens per joule, measured within a stated system boundary.

### Grid carbon intensity

Identity: `mlsys.carbon-intensity`

Greenhouse-gas emissions per unit of electricity supplied, varying by region, time, and generation mix, in grams CO2-equivalent per kilowatt-hour.

### Operational carbon

Identity: `mlsys.operational-carbon`

Emissions from electricity consumed while operating a system, computed from energy use and the carbon intensity of its supply.

### Emission scopes

Identity: `mlsys.emission-scopes`

The GHG Protocol classification of emissions into direct (Scope 1), purchased energy (Scope 2), and value-chain (Scope 3) categories.

### Carbon-aware scheduling

Identity: `mlsys.carbon-aware-scheduling`

Shifting flexible workloads in time or location toward periods and regions of lower grid carbon intensity, subject to deadlines and capacity.

### Embodied carbon

Identity: `mlsys.embodied-carbon`

Emissions from manufacturing, transporting, and disposing of hardware and facilities, attributed to a system independent of its operation.

### Life cycle assessment

Identity: `mlsys.lifecycle-assessment`

A standardized method for compiling and evaluating a product's environmental inputs, outputs, and impacts from raw materials through disposal.

### Water footprint

Identity: `mlsys.water-footprint`

Water withdrawn or consumed by a system, on site for cooling and off site for electricity generation, per unit of work or over its lifetime.

### Hardware service lifetime

Identity: `mlsys.hardware-lifetime`

The period a hardware asset stays in service before replacement, over which its embodied impacts and capital cost are amortized.

### Electronic waste

Identity: `mlsys.e-waste`

Discarded electronic equipment and components, including retired accelerators and servers, and the hazards and lost materials of their disposal.

### Total cost of ownership

Identity: `mlsys.tco`

The full cost of acquiring and operating a system over its life, including capital, energy, facilities, networking, staff, and amortization.

### Cost per token

Identity: `mlsys.cost-per-token`

Serving or API expenditure divided by tokens processed or generated, often priced separately for input and output tokens.

### Cost attribution

Identity: `mlsys.cost-attribution`

Allocating shared infrastructure spend to teams, models, or workloads so that owners see and act on their costs.

### Hazard analysis

Identity: `mlsys.hazard-analysis`

Systematically identifying conditions under which a system could contribute to harm, with their causes, severity, and required controls.

### System-theoretic process analysis

Identity: `mlsys.stpa`

Hazard analysis that models a system as a control structure and identifies unsafe control actions and the flawed feedback or process models that cause them.

### Failure modes and effects analysis

Identity: `mlsys.fmea`

Enumerating each component's possible failure modes and tracing their effects on system behavior to rank and mitigate them.

### Functional insufficiency (SOTIF)

Identity: `mlsys.functional-insufficiency`

Hazardous behavior arising without any fault, from limits of a specification or of a component's performance, such as perception errors under particular triggering conditions.

### Safety filter

Identity: `mlsys.safety-filter`

A runtime component that passes a proposed action when it is certified safe and otherwise replaces it with a minimally different safe action.

### Control barrier function

Identity: `mlsys.control-barrier-function`

A function whose sign defines a safe set and whose derivative condition yields input constraints that keep the system state inside that set.

### Simplex architecture

Identity: `mlsys.simplex-architecture`

Pairing an unverified high-performance controller with a verified safety controller and a decision module that switches to the safety controller before safety is lost.

### Minimal-risk condition

Identity: `mlsys.minimal-risk-condition`

A predefined reduced-capability state the system reaches when it can no longer operate safely, often via an ordered ladder of fallback behaviors.

### Guardrails

Identity: `mlsys.guardrails`

Runtime checks on model inputs and outputs that detect and block, rewrite, or escalate content or actions violating a stated safety policy.

### Scenario-based testing

Identity: `mlsys.scenario-based-testing`

Validating a system against a structured catalog of operating scenarios and their parameter ranges, with coverage measured over the operating domain.

### Simulation-based falsification

Identity: `mlsys.falsification`

Searching scenario parameters, often guided by temporal-logic robustness, to find counterexamples that violate a safety specification.

### Formal verification

Identity: `mlsys.formal-verification`

Mathematically proving that a model of a system satisfies a formal specification for all behaviors within stated assumptions.

### Neural network verification

Identity: `mlsys.neural-network-verification`

Proving or refuting input-output properties of a trained network over a bounded input region, such as local robustness, using exact or bound-propagation methods.

### Rare-event safety validation

Identity: `mlsys.rare-event-validation`

Estimating or bounding failure probabilities too small to observe in feasible exposure, using importance sampling, extreme-value models, or zero-failure statistical bounds.

### X-in-the-loop testing

Identity: `mlsys.xil-testing`

A qualification ladder that runs the real software, then real hardware, against simulated or recorded environments before physical trials.

### Goal Structuring Notation

Identity: `mlsys.goal-structuring-notation`

A graphical notation for safety cases that links goals, strategies, solutions (evidence), context, and assumptions into an explicit argument structure.

### Functional safety

Identity: `mlsys.functional-safety`

Absence of unreasonable risk from malfunctioning electrical or electronic systems, achieved through a standardized safety lifecycle such as IEC 61508 or ISO 26262.

### Safety integrity level

Identity: `mlsys.safety-integrity-level`

A graded risk class assigned to a safety function that sets the rigor of development, verification, and tolerable failure rate required.

### Safety performance indicator

Identity: `mlsys.safety-performance-indicator`

A field-measured metric tied to a safety-case claim whose threshold violation signals that the argument may no longer hold.

### Group fairness metrics

Identity: `mlsys.group-fairness-metric`

Statistics comparing selection rates, error rates, or calibration of a model's decisions across protected groups.

### Individual fairness

Identity: `mlsys.individual-fairness`

A requirement that individuals who are similar with respect to the task receive similar outcomes under a stated similarity metric.

### Fairness criteria incompatibility

Identity: `mlsys.fairness-incompatibility`

The result that calibration and error-rate parity across groups cannot generally hold together when base rates differ, forcing an explicit choice among criteria.

### Bias mitigation

Identity: `mlsys.bias-mitigation`

Interventions on data (pre-processing), training objective (in-processing), or decisions (post-processing) that reduce measured disparities, usually at some cost in other metrics.

### Feature attribution

Identity: `mlsys.feature-attribution`

Post-hoc explanation assigning each input feature a contribution to a specific prediction.

### Counterfactual explanation

Identity: `mlsys.counterfactual-explanation`

An explanation stating the smallest feasible change to an input that would change the model's decision.

### Inherently interpretable model

Identity: `mlsys.interpretable-model`

A model whose structure, such as a sparse linear model, short rule list, or shallow tree, can be directly inspected to understand its predictions.

### Model card

Identity: `mlsys.model-card`

A standardized document reporting a model's intended use, evaluation results across conditions and groups, limitations, and ethical considerations.

### Datasheet for datasets

Identity: `mlsys.datasheet`

A standardized document recording a dataset's motivation, composition, collection process, preprocessing, intended uses, and maintenance.

### Contestability and recourse

Identity: `mlsys.contestability`

The ability of an affected person to understand, challenge, and obtain review or reversal of a consequential automated decision.

### Automation bias

Identity: `mlsys.automation-bias`

The tendency of human supervisors to over-rely on automated outputs and to lose the vigilance and skill needed to catch their errors.

### Authority handover

Identity: `mlsys.authority-handover`

Transfer of control between an automated system and a human operator, bounded by human reaction time, situational awareness, and mode clarity.

### AI risk management

Identity: `mlsys.ai-risk-management`

An organizational process to govern, map, measure, and manage risks of AI systems throughout their lifecycle.

### Algorithmic impact assessment

Identity: `mlsys.impact-assessment`

A structured pre-deployment analysis of who an AI system could affect, how, and with what severity, used to decide mitigations or whether to deploy.

### Algorithmic audit

Identity: `mlsys.algorithmic-audit`

An internal or independent examination of an AI system's development records and behavior against stated requirements, standards, or law.

### Risk-tiered AI regulation

Identity: `mlsys.risk-tiered-regulation`

Law that classifies AI uses by risk level and imposes obligations, such as conformity assessment, documentation, and human oversight, scaled to that level.

### Frontier safety framework

Identity: `mlsys.frontier-safety-framework`

A developer policy that ties dangerous-capability evaluation thresholds to required safeguards before further training or deployment of frontier models.

### Context compaction

Identity: `mlsys.context-compaction`

Replacing older parts of an agent's context with summaries or structured records so a long task fits the model's input budget.

### Long-context degradation

Identity: `mlsys.context-degradation`

Decline in a model's use of relevant information as context grows or as that information sits away from the context's start or end.

### Long-term agent memory

Identity: `mlsys.agent-memory`

A persistent store an agent writes to and retrieves from across calls and sessions, with policies for consolidation, expiry, and invalidation.

### Function calling

Identity: `mlsys.function-calling`

A model emitting a structured invocation of a declared tool, which the harness validates, executes, and returns as an observation.

### Tool discovery protocol

Identity: `mlsys.model-context-protocol`

A shared client-server interface through which agent hosts discover the tools, resources, and prompts external services expose, with their schemas, reducing custom integration.

### Computer use

Identity: `mlsys.computer-use`

An agent operating graphical software through screenshots or accessibility trees and synthesized keyboard and pointer input.

### Agent harness

Identity: `mlsys.agent-harness`

The runtime that repeatedly calls a model, dispatches its tool actions, appends observations, and decides whether the trajectory continues, pauses, or ends.

### Agent execution budget

Identity: `mlsys.agent-budget`

Limits on turns, tokens, wall-clock time, or spend for a trajectory, with stopping rules applied when a limit or completion criterion is reached.

### Action approval gate

Identity: `mlsys.approval-gate`

A harness checkpoint that holds a specific proposed action until an authorized human or policy approves it, binding the approval to that exact action.

### Durable execution

Identity: `mlsys.durable-execution`

Execution that persists workflow progress and reuses recorded results on recovery, with external-effect guarantees determined by endpoint idempotency, transactional boundaries, and reconciliation.

### Record and replay

Identity: `mlsys.deterministic-replay`

Re-executing a trajectory from its log by substituting recorded model outputs and tool results, to reproduce, debug, or test a divergence.

### Suspend and resume

Identity: `mlsys.suspend-resume`

Persisting a trajectory while it waits on a human, timer, or external event, then resuming with a validated state and authority.

### Multi-agent orchestration

Identity: `mlsys.multi-agent-orchestration`

Structuring a task as delegated subtasks among agents under a topology, such as orchestrator-worker or peer, with typed handoffs and result merging.

### Shared agent state

Identity: `mlsys.shared-agent-state`

Task state read and written by several agents, requiring concurrency control such as leases or optimistic versioning to avoid lost or conflicting updates.

### Multi-agent coordination failure

Identity: `mlsys.coordination-failure`

Task failure arising from interaction among agents, such as mis-specified delegation, lost context at handoff, conflicting actions, or premature termination.

### Correlated agent errors

Identity: `mlsys.correlated-agent-errors`

Errors shared across agents built on the same model or prompts, which make voting, debate, or cross-checking less independent than assumed.

### Robot perception

Identity: `mlsys.robot-perception`

Models that convert raw sensor streams into task-relevant scene estimates, such as detections, segmentation, depth, and pose, under latency limits.

### Sensor fusion

Identity: `mlsys.sensor-fusion`

Combining measurements from multiple sensors or modalities, at raw, feature, or decision level, into one estimate more accurate or robust than any single source.

### Sensor time synchronization

Identity: `mlsys.time-synchronization`

Aligning clocks and timestamps across sensors, compute, and actuators so measurements can be associated with the physical instant they describe.

### Recursive state filtering

Identity: `mlsys.recursive-filtering`

Updating a state estimate and its uncertainty by alternating model-based prediction with measurement correction at each step.

### Simultaneous localization and mapping

Identity: `mlsys.slam`

Jointly estimating a platform's pose and a map of an unknown environment from its own sensor observations.

### Learned world model

Identity: `mlsys.world-model`

A learned model that predicts future observations or latent states conditioned on actions, used for planning, policy training, or simulation.

### Motion planning

Identity: `mlsys.motion-planning`

Computing a collision-free path or trajectory from a start to a goal configuration using search, sampling, or trajectory optimization.

### Model predictive control

Identity: `mlsys.model-predictive-control`

Repeatedly solving a finite-horizon constrained optimal control problem from the current state and applying only the first input.

### Real-time scheduling

Identity: `mlsys.real-time-scheduling`

Assigning processor time to periodic and sporadic tasks so that each meets its deadline, with schedulability established by analysis.

### Mixed-criticality partitioning

Identity: `mlsys.mixed-criticality`

Running functions of different assurance levels on shared hardware while preventing lower-criticality work from delaying or corrupting higher-criticality functions.

### Vision-language-action model

Identity: `mlsys.vla-model`

A policy built on a pretrained vision-language backbone that maps images and language instructions to robot actions.

### Teleoperation data

Identity: `mlsys.teleoperation-data`

Robot demonstrations collected by humans driving the robot remotely, whose cost, coverage, and action-label fidelity bound imitation-learned policies.

### Action chunking

Identity: `mlsys.action-chunking`

Predicting a short sequence of future actions per policy inference, decoupling model invocation rate from control rate and reducing compounding error.

### Domain randomization

Identity: `mlsys.domain-randomization`

Training in simulation over randomized visual and physical parameters so the policy treats the real world as one more variation.

### Thread hierarchy

Identity: `mlsys.thread-hierarchy`

The decomposition of a kernel into grids of thread blocks and warps that share on-chip memory and synchronize at defined scopes.

### Memory coalescing and bank conflicts

Identity: `mlsys.memory-coalescing`

Access patterns that merge a warp's loads into few memory transactions, or that serialize on shared-memory banks.

### Occupancy and latency hiding

Identity: `mlsys.occupancy`

The ratio of resident to maximum warps, and how oversubscribing threads hides memory latency at the cost of registers and shared memory per thread.

### Asynchronous data movement

Identity: `mlsys.async-data-movement`

Hardware-assisted bulk copies into on-chip memory and producer-consumer warp roles that overlap loads with matrix-engine compute.

### Matrix-engine programming

Identity: `mlsys.matrix-engine-programming`

Issuing tile-level matrix-multiply-accumulate instructions with the layouts, fragment shapes, and precisions a tensor-core-class unit requires.

### Embedding-table sharding

Identity: `mlsys.embedding-table-sharding`

Partitioning very large sparse embedding tables row-, column-, or table-wise across devices, usually model-parallel beside data-parallel dense layers.

### Embedding cache and tiering

Identity: `mlsys.embedding-cache`

Keeping frequently accessed embedding rows in accelerator memory and cold rows in host memory or SSD, exploiting the skew of lookups.

### Parameter server

Identity: `mlsys.parameter-server`

An architecture in which workers pull and push model parameters from sharded server processes, common for sparse recommendation models and asynchronous training.

### Multi-stage retrieval and ranking

Identity: `mlsys.ranking-funnel`

Candidate generation followed by progressively costlier ranking stages under one latency budget, the dominant architecture for recommendation and search.

### Multimodal model

Identity: `mlsys.multimodal-model`

A model that consumes or produces several modalities, such as text, images, audio, or video, through modality encoders feeding a shared backbone.

### Modality encoder

Identity: `mlsys.modality-encoder`

A network that turns images, audio, or video into token sequences for a shared backbone; visual tokens often dominate prefill cost and KV size.

### Multimodal encoder disaggregation

Identity: `mlsys.encoder-disaggregation`

Scheduling modality encoders separately from prefill and decode and caching encoder outputs so image or audio inputs do not stall token generation.

### Automatic parallelization

Identity: `mlsys.auto-parallelization`

Compiler-derived multi-device partitioning of a single-device program, found by propagating sharding annotations or by searching parallelism strategies under a cost model.

### Sharding annotation and propagation

Identity: `mlsys.sharding-propagation`

Annotating a few tensors with layouts over a device mesh and letting the compiler infer the remaining layouts and insert required collectives, yielding one SPMD program.

### Device mesh

Identity: `mlsys.device-mesh`

A logical multidimensional arrangement of devices whose axes are assigned to parallelism dimensions, usually aligned so the busiest axis maps to the fastest interconnect.

### Composable function transformations

Identity: `mlsys.function-transformations`

Higher-order operations that map pure array functions to new functions, such as differentiation, vectorization, compilation, and sharding, and compose in any order.

### Distributed task and actor runtime

Identity: `mlsys.task-actor-runtime`

A runtime exposing stateless remote tasks and stateful actors as future-returning calls, scheduled across a cluster with a shared object store and lineage-based fault tolerance.

### Single- vs. multi-controller execution

Identity: `mlsys.single-vs-multi-controller`

Whether one coordinating process issues work to all devices or every host runs the same program independently, trading orchestration flexibility against dispatch scalability.

### Analytical performance model

Identity: `mlsys.analytical-performance-model`

A closed-form model composing compute, memory, communication, and queueing terms to predict a workload's time, memory, and cost on a configuration without running it.

### Performance simulation

Identity: `mlsys.performance-simulation`

Predicting system behavior by executing a model of compute, memory, network, or scheduling events at a chosen granularity, trading simulation speed against detail.

### Simulator fidelity

Identity: `mlsys.model-fidelity`

Agreement between a performance model's or simulator's predictions and measured behavior on reference workloads, reported as validation error over stated configurations.

### Post-training quantization

Identity: `mlsys.post-training-quantization`

Quantizing a trained model without retraining, choosing scales and rounding from a small calibration set of representative inputs.

### Weight-only quantization

Identity: `mlsys.weight-only-quantization`

Storing weights at low bit width while computing in higher-precision activations, cutting memory traffic for bandwidth-bound decoding, with error-compensating or activation-aware rounding.

### Activation outliers

Identity: `mlsys.activation-outliers`

Rare, large-magnitude activation channels in large transformers that inflate quantization ranges, handled by mixed-precision decomposition or by migrating scale into weights.

### KV-cache quantization

Identity: `mlsys.kv-cache-quantization`

Storing attention keys and values at reduced precision, often with per-channel keys and per-token values, to raise batch size or context length under fixed memory.

### N:M structured sparsity

Identity: `mlsys.nm-sparsity`

A fine-grained pattern keeping at most N nonzero values in every group of M, such as 2:4, that matrix engines can skip for real speedup.

### KV-cache offloading and tiering

Identity: `mlsys.kv-cache-offload`

Moving KV-cache blocks from accelerator memory to host memory or SSD and reloading them on reuse, extending cache capacity at transfer bandwidth and latency cost.

### KV-cache transfer

Identity: `mlsys.kv-cache-transfer`

Shipping KV state between instances or nodes, as in disaggregated prefill and decode or cross-instance prefix reuse, overlapped with compute to hide network latency.

### Compound AI system

Identity: `mlsys.compound-ai-system`

An application composing multiple model calls, retrievers, and tools into one program whose components and control flow are tuned and served jointly for quality and cost.

### LM program optimization

Identity: `mlsys.lm-program-optimization`

Automatically tuning the prompts, demonstrations, or weights of each model call in a declared pipeline against an end-to-end metric rather than per call.

### Multi-adapter serving

Identity: `mlsys.multi-adapter-serving`

Serving many low-rank adapters over one shared base model by batching requests for different adapters with heterogeneous kernels and paging adapter weights on demand.

### Denoising-step reduction

Identity: `mlsys.denoising-step-reduction`

Lowering network evaluations per diffusion sample with faster samplers, step distillation, or consistency models, trading sample quality against latency and cost.

### Cross-step feature caching

Identity: `mlsys.diffusion-feature-caching`

Reusing intermediate activations that change little between adjacent denoising steps instead of recomputing them, cutting per-step cost at bounded quality loss.

### Graph-based vector index

Identity: `mlsys.graph-vector-index`

An approximate nearest-neighbor index linking vectors into a proximity graph, often hierarchical, searched by greedy traversal for high recall at high memory cost.

### Partitioned and quantized vector index

Identity: `mlsys.quantized-vector-index`

An index that clusters vectors into partitions probed selectively and compresses them with product quantization, trading recall for memory footprint and scan cost.

### Filtered vector search

Identity: `mlsys.filtered-vector-search`

Nearest-neighbor search restricted to records satisfying metadata predicates, where naive pre- or post-filtering loses recall or cost under selective filters.

### Vector index freshness

Identity: `mlsys.index-freshness`

The delay and recall cost of reflecting inserts, updates, and deletes in an ANN index, managed by incremental insertion, delta indexes, tombstones, and periodic rebuilds.

### Numerical nondeterminism

Identity: `mlsys.numerical-nondeterminism`

Run-to-run variation in floating-point results because non-associative reductions change order with atomics, kernel selection, parallel layout, batch composition, or hardware.

### Batch invariance

Identity: `mlsys.batch-invariance`

A kernel property that each request's output is bitwise independent of batch size and co-batched requests, required for deterministic serving and matching training to inference numerics.

### Checkpoint resharding

Identity: `mlsys.checkpoint-resharding`

Loading a checkpoint saved under one parallel layout or world size into another by remapping sharded tensors, needed for elastic restarts and training-to-serving handoff.

### Distributed batch data processing

Identity: `mlsys.batch-data-processing`

Transforming bounded datasets with dataflow engines that partition records across workers, shuffle between stages, and re-execute failed tasks, for large-scale preprocessing and feature generation.

### Distributed tracing

Identity: `mlsys.distributed-tracing`

Recording request-scoped, causally linked spans across services so that one request's latency and failures can be attributed to the components it traversed.

### Model and agent call tracing

Identity: `mlsys.llm-call-tracing`

Tracing model calls, retrievals, and tool invocations as spans annotated with prompts, token counts, latency, and cost, so quality and spend are attributable per step.

### Evaluation harness

Identity: `mlsys.evaluation-harness`

A system running versioned benchmark suites against model checkpoints with fixed prompts, decoding settings, and scorers, sharding and caching work so results stay reproducible and comparable.

### Energy measurement methodology

Identity: `mlsys.energy-measurement`

The declared sensor sources, sampling rate, measurement boundary, and attribution rules by which a workload's energy is measured, from on-chip counters to facility meters.

### Idle vs. dynamic power

Identity: `mlsys.idle-dynamic-power`

Separating power drawn by provisioned but idle hardware from the incremental power of active work, which decides how shared energy is attributed to workloads.

### Homomorphic-encryption inference

Identity: `mlsys.he-inference`

Evaluating a model on encrypted inputs with homomorphic encryption so the server never sees plaintext, at large compute overhead and with polynomial approximations of nonlinearities.

### Secure multi-party computation inference

Identity: `mlsys.secure-computation-inference`

Jointly evaluating a model over secret-shared inputs or weights so no party learns another's data, with cost dominated by communication rounds for nonlinear layers.

### Red teaming

Identity: `mlsys.red-teaming`

Structured adversarial probing of a model or system, by people or automated attacker models, to elicit harmful, insecure, or policy-violating behavior before and after release.

### Dangerous-capability evaluation

Identity: `mlsys.dangerous-capability-eval`

Evaluating whether a model has capabilities, such as cyberoffense, biological uplift, or autonomous replication, that cross thresholds requiring added safeguards.

### Compute-in-memory

Identity: `mlsys.compute-in-memory`

Performing matrix-vector operations inside or beside memory arrays, often in the analog domain, so weights need not move to separate compute units.

### Emerging non-volatile memory

Identity: `mlsys.emerging-nvm`

Resistive, phase-change, and magnetic memory devices that retain state without power and can store weights densely or act as analog compute elements.

### Embedded ML DSL

Identity: `mlsys.embedded-dsl`

A domain-specific language hosted inside a general language such as Python, whose operations build a program representation that a separate compiler or runtime executes.

### Differentiable programming

Identity: `mlsys.differentiable-programming`

Writing general programs, including control flow, recursion, and data structures, that the language or framework differentiates end to end by automatic differentiation.

### Persistent kernel

Identity: `mlsys.persistent-kernel`

A long-running kernel that stays resident on the device and executes many operators, or a whole forward pass, from an in-kernel work queue.

### Tensor graph superoptimization

Identity: `mlsys.graph-superoptimization`

Automatically generating and verifying semantics-preserving rewrites of tensor computation graphs or kernels, then searching the rewritten space under a cost model.

### Large-model training framework

Identity: `mlsys.llm-training-framework`

A library that composes data, tensor, pipeline, and sharded-state parallelism with memory optimizations behind one training configuration for models exceeding one device.

### Cloud service models

Identity: `mlsys.cloud-service-model`

Divisions of responsibility between cloud provider and customer, from rented virtual infrastructure to managed platforms to hosted model APIs.

### Data warehouse and lakehouse

Identity: `mlsys.lakehouse`

Analytical stores of governed tables queried with SQL; a lakehouse adds transactional table formats over open files in object storage.

### Synchronous training power swings

Identity: `mlsys.training-power-swings`

Rapid cluster-wide swings in power draw as synchronized accelerators alternate between compute and communication or stall together at checkpoints and failures.

### UPS and battery energy storage

Identity: `mlsys.facility-energy-storage`

Facility batteries that carry load through outages until generators start and can also buffer fast load swings or shave demand peaks.

### Grid interconnection

Identity: `mlsys.grid-interconnection`

The utility connection that fixes how much power a site can draw; its capacity and multi-year lead times increasingly decide where AI datacenters are built.

### On-site power generation

Identity: `mlsys.on-site-generation`

Generators, turbines, fuel cells, or renewables at the datacenter that provide backup power or supply load alongside or instead of the grid.

### Data cleaning

Identity: `mlsys.data-cleaning`

Detecting and correcting erroneous, inconsistent, duplicated, or outlying records with rules, statistics, or learned repair models before the data is used for training.

### Class imbalance

Identity: `mlsys.class-imbalance`

A skewed label distribution in which some classes are rare, so aggregate accuracy misleads and training favors the majority classes.

### Class rebalancing

Identity: `mlsys.rebalancing`

Changing per-class sampling rates, synthesizing minority examples, or weighting the loss so rare classes contribute adequately to training.

### Feature hashing

Identity: `mlsys.feature-hashing`

Mapping high-cardinality or open-vocabulary categorical values into a fixed number of buckets with a hash function, bounding table size at the cost of collisions.

### Feature cross

Identity: `mlsys.feature-cross`

A synthetic feature formed from the conjunction of two or more categorical features so a model, especially a linear one, can learn their interaction.

### Data governance

Identity: `mlsys.data-governance`

Policies, roles, and controls that decide who may collect, access, use, and share which data, for which purposes, and under what accountability.

### Consent and purpose limitation

Identity: `mlsys.data-consent`

Restricting collection and use of personal data to purposes the data subject agreed to or a legal basis allows, recorded so downstream uses can be checked.

### Exploratory data analysis

Identity: `mlsys.exploratory-data-analysis`

Summarizing and plotting distributions, correlations, and anomalies in a dataset before modeling to inform cleaning, feature design, and validation rules.

### Model and prediction visualization

Identity: `mlsys.model-visualization`

Plotting training curves, activations, embeddings, and prediction errors such as confusion matrices to inspect what a model learned and where it fails.

### Model selection

Identity: `mlsys.model-selection`

Choosing among candidate model families, architectures, and hyperparameters by validated performance under the system's accuracy, cost, and latency constraints.

### Pretraining

Identity: `mlsys.pretraining`

Training a model from scratch on a large, broad corpus with a self-supervised objective to produce a general base model for later adaptation.

### Causal language modeling

Identity: `mlsys.causal-lm`

Training a model to predict each token from the tokens before it, maximizing the likelihood of text under a left-to-right factorization.

### Generative adversarial network

Identity: `mlsys.gan`

A generator and a discriminator trained against each other, so the generator learns to produce samples the discriminator cannot tell apart from data.

### Mode collapse

Identity: `mlsys.mode-collapse`

A generative-training failure in which the model covers only a few modes of the data distribution and produces low-diversity samples.

### Perplexity

Identity: `mlsys.perplexity`

The exponentiated average negative log-likelihood per token of held-out text, measuring how well a language model predicts that text.

### Reference-match scoring

Identity: `mlsys.reference-match-scoring`

Scoring a generated output by exact or token-overlap agreement with reference answers; cheap and reproducible but blind to valid answers phrased differently.

### Error analysis

Identity: `mlsys.error-analysis`

Inspecting a model's mistakes, grouping them into categories, and ranking categories by frequency and fixability to decide the next data or model change.

### Training diagnostics

Identity: `mlsys.training-diagnostics`

Checks such as overfitting a tiny batch, reading learning curves, and tracking gradient and activation statistics that localize bugs in training code, data, and configuration.

### Request correlation keys

Identity: `mlsys.keyed-predictions`

Passing a client-supplied key through inference unchanged so outputs of batch or asynchronous requests can be matched to their inputs regardless of order.

### Stateless model serving

Identity: `mlsys.stateless-serving-function`

Exporting a model as a stateless function from request inputs to prediction, so any replica can serve any request and replicas scale horizontally.

### Model gateway

Identity: `mlsys.model-gateway`

A proxy in front of one or more model providers that centralizes authentication, routing, rate limits, fallbacks, and usage logging for model calls.

### Semantic cache

Identity: `mlsys.semantic-cache`

Returning a stored model response when a new request is close in embedding space to an earlier one, trading answer fidelity for latency and cost.

### Glue code

Identity: `mlsys.glue-code`

Supporting code written to move data into and out of general-purpose ML packages, often far larger than the model code and binding the system to package specifics.

### Pipeline jungle

Identity: `mlsys.pipeline-jungle`

Data preparation grown incrementally into a tangle of scrapes, joins, and intermediate outputs that is hard to test, trace, and change.

### Build-versus-buy decision

Identity: `mlsys.build-vs-buy`

Choosing whether to build an ML platform component in house, adopt open source, or buy a managed service, weighing fit, cost, lock-in, and team capacity.

### User feedback collection

Identity: `mlsys.user-feedback`

Capturing explicit signals such as ratings and corrections and implicit signals such as clicks, edits, and abandonment, for use as labels or quality signals.

### Human-AI interaction design

Identity: `mlsys.human-ai-interaction`

Designing how a system conveys its capabilities, uncertainty, and errors to users and lets them correct, override, or dismiss its outputs.

### Datacenter demand response

Identity: `mlsys.demand-response`

Shifting, shedding, or capping datacenter power in response to grid signals, made practical by deferrable training and batch work.

### Prompt engineering

Identity: `mlsys.prompt-engineering`

Designing instructions, examples, and input layout for a model call to elicit desired behavior without changing the model's weights.

### In-context learning

Identity: `mlsys.in-context-learning`

A pretrained model performing a task from instructions or examples supplied in its input at inference time, with no weight updates.

### Streaming speech recognition

Identity: `mlsys.streaming-asr`

Serving speech recognition incrementally: consuming audio in chunks and emitting partial transcripts within a fixed lag, with encoder state carried between chunks.

### Voice interaction pipeline

Identity: `mlsys.voice-pipeline`

Serving a spoken interaction as one latency-budgeted pipeline of streaming speech recognition, a language model, and speech synthesis, with turn detection and interruption handling between stages.

### KV-cache eviction

Identity: `mlsys.kv-cache-eviction`

Retaining only a selected subset of cached key-value entries, chosen by attention scores, recency, or learned importance, to bound KV memory during long generation.

### Dynamic sparse attention

Identity: `mlsys.dynamic-sparse-attention`

Attention that computes only the query-key blocks predicted important for each input at runtime, rather than a fixed pattern, to cut long-context cost.

### CPU-GPU hybrid inference

Identity: `mlsys.cpu-gpu-hybrid-inference`

Splitting inference so the host CPU computes some operations, such as attention or rarely active neurons, from host memory when model or KV state exceeds accelerator memory.

### Heterogeneous-accelerator training and serving

Identity: `mlsys.heterogeneous-accelerator-execution`

Partitioning one training or serving job across accelerators of different types or generations, sizing stages and batch shares to each device's compute, memory, and links.

### Graph neural network

Identity: `mlsys.graph-neural-network`

Neural network that computes node, edge, or graph representations by repeatedly aggregating features from each node's neighbors along graph edges.

### Neighbor sampling

Identity: `mlsys.neighbor-sampling`

Building GNN mini-batches from sampled subsets of each node's multi-hop neighborhood, bounding the exponential neighbor growth that makes full-neighborhood computation exceed memory.

### Dynamic re-parallelization

Identity: `mlsys.dynamic-resharding`

Redistributing model weights among devices at runtime so successive phases, such as generation and training or prefill and decode, each run under a better parallel layout.

### Contextual activation sparsity

Identity: `mlsys.activation-sparsity`

Skipping neurons, heads, or channels whose activations are predicted negligible for the current input, reducing per-token compute and weight reads at inference.

### Non-prefix KV reuse

Identity: `mlsys.non-prefix-kv-reuse`

Reusing precomputed KV entries for text chunks appearing at arbitrary positions across requests, selectively recomputing cross-chunk attention instead of the full prefill.

### ML-aware cluster scheduling

Identity: `mlsys.ml-aware-scheduling`

Allocating cluster accelerators using ML job properties, such as scaling curves, iteration time, and parallelism choice, to optimize objectives like job completion time or goodput.

### Adaptive routing

Identity: `mlsys.adaptive-routing`

Choosing network paths per packet or flowlet from current congestion, avoiding the hash collisions static ECMP suffers with few large, synchronized training flows.

### Incast congestion

Identity: `mlsys.incast`

Congestion that occurs when many senders transmit to one receiver at once, overflowing its switch port buffer and causing queueing, drops, or pause propagation.

### Context selection

Identity: `mlsys.context-selection`

Choosing which instructions, tool descriptions, retrieved items, state, and observations enter a model's limited context at each step, within a token budget.

### Context staleness

Identity: `mlsys.context-staleness`

Observations held in an agent's context or memory that no longer reflect the environment after later actions or external changes, yet still steer decisions unless invalidated.

### Iterative multi-hop retrieval

Identity: `mlsys.multi-hop-retrieval`

Retrieval that issues successive queries, each formulated from previously retrieved evidence or reasoning steps, to answer questions no single lookup resolves.

### Capability-based authorization

Identity: `mlsys.capability-based-security`

Authorization in which holding an unforgeable, scoped reference grants a specific right that can be delegated with attenuated scope and later revoked.

### Ephemeral sandbox

Identity: `mlsys.ephemeral-sandbox`

An isolated execution environment created per task from a snapshot, often copy-on-write, and destroyed or reset afterward so no state carries across tasks or tenants.

### Network egress control

Identity: `mlsys.egress-control`

Restricting which external network destinations a workload or sandbox may contact, typically through default-deny allowlists, to block exfiltration and unintended remote actions.

### Information-flow control

Identity: `mlsys.information-flow-control`

Labeling data with confidentiality and integrity tags and propagating them through computation so policy can block untrusted inputs from steering privileged actions or secrets from leaking.

### Compounding error in sequential decisions

Identity: `mlsys.compounding-error`

Per-step errors of a learned sequential policy or generator push later inputs away from the training distribution, so failure probability grows with horizon length.

### Interactive imitation learning

Identity: `mlsys.interactive-imitation`

Imitation learning that queries an expert for corrective labels on states the learner's own policy visits and aggregates them into training data over rounds.

### Loss masking and token normalization

Identity: `mlsys.loss-masking`

Excluding prompt, tool-output, and other non-target tokens from the training loss, and choosing how remaining token losses are averaged across turns and sequences.

### Credit assignment

Identity: `mlsys.credit-assignment`

Attributing a delayed or episode-level reward to the individual actions, turns, or tokens that produced it, so updates reinforce the responsible decisions.

### Actuator limits

Identity: `mlsys.actuator-limits`

Bounds on the force, torque, speed, and sustained duty an actuator can deliver, set by drive current limits, gearing, and motor heating.

### Watchdog timer

Identity: `mlsys.watchdog-timer`

A timer that a monitored component must refresh within a deadline or window; expiry triggers a defined safe response such as reset or revoking control authority.

### Closed-loop policy evaluation

Identity: `mlsys.closed-loop-evaluation`

Evaluating a policy by letting its own actions determine subsequent observations, in simulation or deployment, rather than scoring its predictions against logged data.

### Event data recorder

Identity: `mlsys.event-data-recorder`

Tamper-evident onboard logging of inputs, decisions, control authority, and safety interventions around events, retained for incident reconstruction and assurance evidence.

### Decision accountability record

Identity: `mlsys.decision-accountability`

A record of who authorized each consequential decision, policy, or automated action and on what evidence, so responsibility can be assigned and reviewed later.

### Agent interruption and cancellation

Identity: `mlsys.agent-interruption`

Operator controls that pause, redirect, cancel, or terminate a running agent at defined boundaries while settling in-flight tool calls into a known state.

### Agent loop detection

Identity: `mlsys.agent-loop-detection`

Monitoring an agent trajectory for repeated actions, cycling states, or absent progress, and halting or escalating when the agent appears stuck.

### Task and motion planning

Identity: `mlsys.task-and-motion-planning`

Planning that jointly searches discrete symbolic action sequences and continuous motions, accepting a task plan only when each step admits a feasible motion.

### Multi-object tracking

Identity: `mlsys.multi-object-tracking`

Maintaining identities and states of multiple moving objects over time by associating new detections with existing tracks and starting or ending tracks.

### Safety change-impact analysis

Identity: `mlsys.change-impact-analysis`

Determining which hazards, safety-case claims, and evidence a change to a model, software, data, or operating domain invalidates, and what must be re-verified.

### Sparse tensor kernels

Identity: `mlsys.sparse-tensor-kernels`

Kernels such as sparse-dense matrix multiply, sampled dense-dense multiply, and sparse convolution that compute over compressed nonzero structure, trading irregular access for skipped work.

### Local SGD

Identity: `mlsys.local-sgd`

Data-parallel training in which workers take several local optimizer steps between parameter synchronizations, cutting communication frequency for low-bandwidth or geo-distributed settings.

### Expert replication and placement

Identity: `mlsys.expert-placement`

Assigning mixture-of-experts experts to devices, and replicating frequently routed ones, to balance per-device token load under skewed routing.

### Memory-efficient optimizer

Identity: `mlsys.memory-efficient-optimizer`

Optimizers that shrink per-parameter state through factorization, low-rank projection, or low-precision storage, cutting the memory adaptive methods such as Adam otherwise require.

### Operator execution trace

Identity: `mlsys.execution-trace`

A recorded graph of a training or inference step's compute and communication operators with their dependencies and timings, replayable for simulation and benchmarking.

### Offline reinforcement learning

Identity: `mlsys.offline-rl`

Learning a policy from a fixed dataset of previously logged interactions without further environment access, limited by how well the data covers states the policy would visit.

### Byzantine-robust aggregation

Identity: `mlsys.byzantine-robust-aggregation`

Combining worker or client updates with robust statistics, such as medians, trimmed means, or Krum, so a bounded fraction of arbitrary malicious updates cannot steer training.

### Cluster manager

Identity: `mlsys.cluster-manager`

The control plane that admits jobs, places their tasks on machines, enforces quotas and priorities, and reconciles running state toward declared intent across a cluster.

### Rebound effect

Identity: `mlsys.rebound-effect`

Efficiency gains lower the cost of a service enough that total demand, and with it total energy or resource use, grows, partly or fully offsetting the savings.

### Data cascade

Identity: `mlsys.data-cascade`

Compounding downstream harm from upstream data problems, such as poor collection or labeling, that surface late, often after deployment, and are costly to trace.

### Data valuation

Identity: `mlsys.data-valuation`

Estimating each training example's or data source's contribution to model quality, for example through influence or Shapley-style attribution, to guide selection, pricing, or removal.

### Diffusion or flow-matching policy

Identity: `mlsys.diffusion-policy`

Robot policies that generate continuous actions through iterative denoising or learned-flow integration, representing multiple action modes while spending an inference-iteration budget.

### Sensor spoofing

Identity: `mlsys.sensor-spoofing`

Attacks that inject false signals into a sensor's physical channel, such as fabricated LiDAR returns or GPS and acoustic spoofing, so perception receives counterfeit measurements.

### Paired comparison and noninferiority testing

Identity: `mlsys.paired-comparison`

Comparing two systems on the same items or units and testing per-item differences, including whether a candidate is no worse than a baseline by a preset margin.

### CPU execution architecture

Identity: `mlsys.cpu-execution-architecture`

Organization of general-purpose processor cores, SIMD vector units, and multilevel caches that runs host code, preprocessing, classical models, and CPU inference.

### CPU threading and NUMA placement

Identity: `mlsys.cpu-threading-numa`

Configuring thread pools, core affinity, and memory placement across NUMA domains so CPU operators parallelize without oversubscription or remote-memory penalties.

### Dynamic shapes and shape specialization

Identity: `mlsys.dynamic-shapes`

Handling inputs whose tensor dimensions vary across calls by specializing compiled code per shape, padding into buckets, or compiling with symbolic dimensions.

### Rule-based graph rewriting

Identity: `mlsys.rule-based-graph-rewriting`

Applying hand-written semantics-preserving transformations to a computation graph, such as constant folding, dead-code elimination, operator substitution, and algebraic simplification.

### Sparse and ragged tensor representation

Identity: `mlsys.sparse-ragged-tensors`

Storing tensors with mostly zero entries or variable-length dimensions in compressed formats such as COO, CSR, or offset-indexed ragged layouts instead of padded dense arrays.

### Execution providers and operator fallback

Identity: `mlsys.execution-providers`

Partitioning a model graph across runtime backends by operator support, running supported subgraphs on accelerators and falling back to a CPU backend otherwise.

### Large language model

Identity: `mlsys.large-language-model`

A language model large enough that parameter memory, attention state that grows with context, and token-by-token decoding usually dominate its training and serving cost. Tag it only when an item is about that workload class itself, not merely because an LLM is involved.

### Inference compute and memory estimate

Identity: `mlsys.inference-compute-estimate`

Approximating dense transformer inference as about two FLOPs per parameter per token, plus KV-cache bytes from layers, KV heads, head dimension, context length, batch, and precision.

### Synchronous vs. asynchronous training

Identity: `mlsys.sync-async-training`

Choosing whether workers apply an update only after every gradient for a step is combined, or apply updates independently on possibly stale parameters.

### Input prefetching

Identity: `mlsys.input-prefetching`

Preparing upcoming batches in background workers and buffers while the accelerator processes the current batch, overlapping input delivery with computation.

### Event-time windowing and watermarks

Identity: `mlsys.event-time-windowing`

Grouping stream events into windows by when they occurred, using watermarks to estimate completeness and explicit policies for events arriving after a window closes.

### Token streaming

Identity: `mlsys.token-streaming`

Delivering generated output to the client incrementally as tokens are produced, holding per-stream state and stopping generation when the client cancels or disconnects.

### Serverless inference

Identity: `mlsys.serverless-inference`

Running models on a managed platform that provisions replicas on demand, bills per use, and can scale to zero, exposing requests to cold-start latency.

### LLM request scheduling and preemption

Identity: `mlsys.llm-request-scheduling`

Ordering, prioritizing, and preempting in-flight generation requests at iteration granularity, swapping or recomputing their KV state, to meet latency targets and fairness goals.

### Containerization

Identity: `mlsys.containerization`

Packaging a model, its code, and its dependencies as an immutable image that runs unchanged across machines. Scheduling those containers across a cluster belongs to the cluster manager.

### Distributed job hang diagnosis

Identity: `mlsys.distributed-hang-diagnosis`

Detecting and localizing stalled distributed jobs through collective timeouts, heartbeats, per-rank progress traces, and node health checks, so faulty ranks can be isolated.

### Benchmark representativeness

Identity: `mlsys.benchmark-representativeness`

The degree to which a benchmark's workload, scale, and measurement boundary predict production behavior, from isolated microbenchmarks to end-to-end application runs.

### Data and model licensing

Identity: `mlsys.data-model-licensing`

Legal terms governing how datasets and model weights may be used, modified, redistributed, or deployed, which constrain acquisition, fine-tuning, and release decisions.

### Content provenance

Identity: `mlsys.content-provenance`

Cryptographically signed metadata attached to media that records its origin and edit history, including model generation, so recipients can verify source claims.

### Fleet data engine

Identity: `mlsys.fleet-data-engine`

Collecting training data from deployed devices through on-device triggers that flag rare or uncertain events, uploading selected samples under bandwidth budgets for labeling and retraining.

### Robotics middleware and dataflow

Identity: `mlsys.robotics-middleware`

Software connecting sensing, perception, planning, and control processes as nodes that exchange typed messages over publish-subscribe topics with configurable delivery and timing policies.

### Parallel physics simulation

Identity: `mlsys.parallel-simulation`

Running many simulated robot environments concurrently, often batched on accelerators, so policy learning and data generation gather experience faster than physical collection.

### Agent credentials and delegated authority

Identity: `mlsys.agent-credentials`

Issuing tool-using agents scoped, short-lived credentials that act on a user's behalf, brokering secrets outside model context so authority stays bounded and revocable.

### Graph partitioning for GNN training

Identity: `mlsys.gnn-graph-partitioning`

Splitting a large graph across workers to balance computation while minimizing cut edges, whose cross-partition neighbor features must be fetched remotely.

### Host-side data loading

Identity: `mlsys.host-data-loader`

Reading, decoding, augmenting, and collating examples on host CPU worker processes and staging batches in pinned memory so that a single machine's accelerators are not left waiting for input.

### Streaming inference

Identity: `mlsys.streaming-inference`

Serving a model over an open stream of input, producing outputs incrementally from partial input while carrying state between chunks, rather than answering one complete request at a time.

### Inference API and payload serialization

Identity: `mlsys.inference-api`

The request and response protocol of a model server, including the transport (HTTP, gRPC, streaming connections), the tensor and payload encoding, and the cost of serializing inputs and outputs.

### Request throughput

Identity: `mlsys.request-throughput`

The number of requests, or tokens, a serving system completes per unit time at a stated load and latency target.

### Latency percentiles

Identity: `mlsys.latency-percentiles`

Summarizing a latency distribution by percentiles such as p50, p95, and p99, so that a service is judged by the experience of most requests and of the slow tail rather than by the mean.

### Cache-aware request routing

Identity: `mlsys.cache-aware-routing`

Sending each request to the replica that already holds a reusable prefix or KV cache for it, trading load balance against cache reuse.

### Distributed job launch and process groups

Identity: `mlsys.process-group-rendezvous`

Starting the processes of a distributed job, having them discover each other through a rendezvous, assigning ranks, and forming the process groups and communicators that collectives run over.

### Agentic workload

Identity: `mlsys.agentic-workload`

The load an agent places on the serving stack, including context that grows each turn, idle gaps while tools run, many short dependent model calls, and reuse of KV state across steps.

### Performance regression testing

Identity: `mlsys.performance-regression-testing`

Benchmarking every change to code, kernels, models, or configuration continuously and failing the change when throughput, latency, or memory degrades beyond a set tolerance.

### Performance measurement variability

Identity: `mlsys.measurement-variability`

Run-to-run variation in measured time and throughput from warm-up, asynchronous execution, clock and thermal state, and contention, handled by synchronizing, repeating runs, and reporting spread.

### PII detection and de-identification

Identity: `mlsys.pii-deidentification`

Finding personal identifiers in training corpora, logs, and prompts and removing, masking, or pseudonymizing them before the data is stored, trained on, or shared.

### Gradient clipping

Identity: `mlsys.gradient-clipping`

Rescaling or truncating gradients whose norm exceeds a threshold so that a single large update cannot destabilize training.

### Weight initialization

Identity: `mlsys.weight-initialization`

Choosing the scale and distribution of initial parameters so that activations and gradients keep stable variance through depth at the start of training.

### Processing delivery semantics

Identity: `mlsys.delivery-semantics`

The guarantee a data pipeline gives about how often each record takes effect (at most once, at least once, or effectively once) under retries, replays, and failures, and the idempotent or transactional sinks that achieve it.

### Online and offline feature consistency

Identity: `mlsys.online-feature-consistency`

Keeping the feature values materialized in the low-latency online store consistent with the offline values used for training, as features are recomputed, backfilled, and updated.

### Out-of-distribution detection

Identity: `mlsys.ood-detection`

Flagging individual inputs that differ from the training distribution so the system can abstain, defer, or fall back instead of trusting the prediction.

### Model and feature compatibility

Identity: `mlsys.model-feature-compatibility`

Ensuring that a model version, its tokenizer and feature transformations, the schemas it reads, and the consumers of its outputs remain compatible as each is versioned and released independently.

### Workload identity and secrets management

Identity: `mlsys.secrets-management`

Giving training jobs, model servers, and pipelines verifiable identities and provisioning, rotating, and revoking the keys and credentials they use, without embedding secrets in code or images.

### Data deletion propagation

Identity: `mlsys.deletion-propagation`

Carrying a deletion request through every copy and derivative of the data, including datasets, features, caches, replicas, backups, and models trained on it.

### Runtime memory fragmentation

Identity: `mlsys.memory-fragmentation`

Free device memory split into blocks too small to satisfy new allocations, so a job runs out of memory while reserved memory exceeds what live tensors use.

### Perception latency budget

Identity: `mlsys.perception-latency-budget`

The share of a robot's sense-to-act deadline allotted to sensing and perception, which bounds model size, input resolution, and frame rate on the onboard computer.

### Onboard compute partitioning

Identity: `mlsys.onboard-compute-partitioning`

Assigning perception, planning, and control workloads to the CPUs, accelerators, and microcontrollers of a robot, or offloading them, so each meets its deadline and criticality level.

### Health checks and readiness probes

Identity: `mlsys.health-checks`

Periodic liveness and readiness checks that let an orchestrator or load balancer stop routing traffic to, or restart, a replica that is unhealthy or still loading its model.

### Python GIL and host-language concurrency

Identity: `mlsys.python-gil`

The CPython global interpreter lock serializes Python bytecode across threads, so ML host code achieves concurrency through native kernels that release the lock, multiple processes, or asynchronous I/O.

### Flash write endurance

Identity: `mlsys.flash-endurance`

Flash memory cells wear out after a limited number of program and erase cycles, which bounds how often embedded and edge devices can write logs, checkpoints, or model updates; wear leveling spreads writes to extend life.

### Energy-harvesting intermittent computing

Identity: `mlsys.intermittent-computing`

Running inference on devices powered by harvested energy that lose power unpredictably, which requires checkpointing progress to nonvolatile memory so computation resumes rather than restarts.

### Memory-mapped model weights

Identity: `mlsys.mmap-weights`

Mapping a weight file into virtual memory so pages load on demand and read-only weights are shared across processes, cutting load time and duplicate memory at the cost of page faults.

### OS memory pressure and paging

Identity: `mlsys.os-memory-pressure`

How an operating system reclaims memory from a model process under pressure, by evicting file-backed pages, swapping or compressing memory, or killing the app, which shows up as page thrashing, latency spikes, or abrupt termination.

### Managed-runtime interop and garbage collection

Identity: `mlsys.managed-runtime-interop`

Crossing between a managed language runtime (Java, Kotlin, Swift, C\#) and native inference code, where buffer copies, object pinning, and garbage-collection pauses add latency and memory.

### Zero-copy sensor input via DMA

Identity: `mlsys.zero-copy-dma`

Having sensors or peripherals write input directly into model input buffers by direct memory access, avoiding CPU copies between capture and inference.

### Active sensor interference

Identity: `mlsys.sensor-interference`

Crosstalk between active sensors, such as lidars, radars, or structured-light cameras, that emit into the same environment and corrupt each other's measurements.

### Network propagation delay

Identity: `mlsys.network-propagation-delay`

The minimum latency set by the distance a signal travels at the speed of light in fiber, which bounds round trips between regions regardless of bandwidth.

### Consistency-availability trade-off

Identity: `mlsys.cap-tradeoff`

Under a network partition, a replicated system must choose between answering with possibly stale state and refusing requests until replicas agree.

### Decentralized gossip averaging

Identity: `mlsys.gossip-averaging`

Training in which each worker averages its model with a few neighbors instead of a central server or global all-reduce, so updates diffuse through the communication graph.

### Interrupt latency

Identity: `mlsys.interrupt-latency`

The delay between a hardware event and its interrupt handler running, plus the handler's own run time, which must fit inside a real-time sensing or control budget alongside inference.

### Encoder-only, decoder-only, and encoder-decoder Transformers

Identity: `mlsys.transformer-variants`

The three arrangements of Transformer stacks, which differ in whether inputs are processed bidirectionally, outputs are generated autoregressively, or both, and therefore in how they are trained, cached, and served.

### Data transfer cost

Identity: `mlsys.data-transfer-cost`

The money charged for moving data across network, zone, region, or provider boundaries, such as cloud egress fees, which can dominate the cost of distributing data, checkpoints, or responses.

### Duty cycling and sleep states

Identity: `mlsys.duty-cycling`

Keeping a device in a low-power sleep state most of the time and waking it briefly to sense and infer, so average power is set by the active fraction and the cost of each wake-up.

### Priority inversion

Identity: `mlsys.priority-inversion`

A high-priority task waiting on a resource held by a lower-priority task, which a medium-priority task then preempts, so the high-priority task misses its deadline unless priority inheritance or ceilings are used.

### Certified robustness by randomized smoothing

Identity: `mlsys.randomized-smoothing`

Classifying many noise-perturbed copies of an input and taking the majority vote, which yields a provable radius within which no adversarial perturbation can change the prediction, at the cost of many inferences per input.

### Sensor interconnect bandwidth

Identity: `mlsys.sensor-interconnect`

The data rate of the links that carry sensor data into an embedded processor (MIPI CSI-2 for cameras, I2S for audio, SPI and I2C for low-rate sensors), which bounds the resolution, frame rate, and number of sensors a device can ingest.

### Finite buffers and overflow

Identity: `mlsys.finite-buffer`

A queue with bounded capacity drops or blocks arrivals once full, so its size trades memory and added delay against the loss rate under bursty input.

### Asynchronous on-device inference

Identity: `mlsys.ui-thread-inference`

Running inference off the application's main thread, so a model call does not block rendering and input handling, with results delivered back by callback or message.

### Multi-head latent attention

Identity: `mlsys.multi-head-latent-attention`

Attention that caches one low-rank latent vector per token in place of per-head keys and values, reconstructing them or absorbing the up-projections into the query and output weights, which shrinks the KV cache and decode memory traffic.

### Training data extraction

Identity: `mlsys.training-data-extraction`

Recovering particular training examples or their sensitive contents through a trained model's outputs, exposing information that was not intended for release.

### System identification

Identity: `mlsys.system-identification`

Estimating the parameters of a dynamics model, such as masses, friction, delays, and actuator gains, from measured input-output data of the physical system.

### Off-policy evaluation

Identity: `mlsys.off-policy-evaluation`

Estimating a proposed policy's expected outcomes from interactions collected under another policy, by propensity reweighting, outcome modeling, or doubly robust estimators, subject to assumptions about action coverage and confounding.

### Pipeline schedule

Identity: `mlsys.pipeline-schedule`

The order in which pipeline stages run the forward and backward passes of microbatches (fill-drain, one-forward-one-backward, interleaved), which sets the bubble fraction and how many microbatches' activations each stage holds.

### Sequence packing

Identity: `mlsys.sequence-packing`

Combining variable-length training examples into packed token buffers with explicit example boundaries and attention semantics to reduce padding work without unintended cross-example attention.

### Document chunking

Identity: `mlsys.document-chunking`

Dividing source documents into retrievable units with chosen boundaries, overlap, and source identifiers so retrieval granularity fits embedding and generation budgets.

### Resource-exhaustion attack

Identity: `mlsys.resource-exhaustion-attack`

Crafting inputs that maximize a model service's compute, energy, output length, or tool calls, degrading service for other users or running up the operator's cost.

### Generated-content watermarking

Identity: `mlsys.generated-content-watermarking`

Embedding a statistically detectable signal into generated text, images, or audio at generation time, such as by biasing token sampling, so that content can later be identified as model-generated.

### Model merging

Identity: `mlsys.model-merging`

Combining compatible model parameters or parameter updates, by averaging or arithmetic on weight differences, into one model that consolidates capabilities without further training or serving the sources separately.

### Optical circuit switching

Identity: `mlsys.optical-circuit-switching`

Reconfiguring optical connections between endpoints to match communication demand, trading switching delay and topology coordination against delivered bandwidth and network power.

### Co-packaged optics

Identity: `mlsys.co-packaged-optics`

Placing optical transceivers inside the switch or accelerator package beside the silicon, so electrical paths shrink and energy per bit falls, at the cost of field serviceability and added package heat.

### Advanced accelerator packaging

Identity: `mlsys.advanced-accelerator-packaging`

Integrating compute and memory dies through chiplets, interposers, or three-dimensional bonding to trade local bandwidth and manufacturing flexibility against power-delivery and heat-removal constraints.

### Motion forecasting

Identity: `mlsys.motion-forecasting`

Predicting the future trajectories of other moving agents, usually as several weighted hypotheses, from their tracked history and the map, so a planner can avoid where they will be rather than where they are.

### Representational bias evaluation

Identity: `mlsys.representational-bias-evaluation`

Measuring stereotyped, demeaning, or unequal portrayals of groups in generated outputs, revealing representational harms that aggregate quality or decision-rate metrics can miss.

### Group relative policy optimization

Identity: `mlsys.group-relative-policy-optimization`

A policy-gradient method that scores each sampled response against the mean reward of a group of responses to the same prompt, removing the learned critic at the cost of generating several rollouts per prompt.

### Inter-token latency

Identity: `mlsys.inter-token-latency`

The time between consecutive output tokens or chunks delivered to a streaming client, whose distribution, not only its mean, exposes decode stalls that an average time per output token hides.

### Hybrid retrieval

Identity: `mlsys.hybrid-retrieval`

Combining lexical and dense retrieval results through score or rank fusion to recover complementary matches while managing additional search cost and ranking calibration.

### FPGA-based ML acceleration

Identity: `mlsys.fpga-acceleration`

Mapping ML operations to reconfigurable logic, arithmetic blocks, and local memories so datapaths can match a workload at the cost of hardware compilation and resource constraints.

### Device-initiated communication

Identity: `mlsys.device-initiated-communication`

Communication in which kernels on the accelerator issue and synchronize inter-device transfers themselves, without a host call per exchange, allowing fine-grained fusion of compute and communication at the cost of explicit ordering and completion handling.

### Model collapse

Identity: `mlsys.model-collapse`

Loss of coverage of the original data distribution across generations trained on model-generated data, making provenance and real-versus-synthetic mixture policies part of quality control.

### Action tokenization

Identity: `mlsys.action-tokenization`

Encoding continuous robot actions or action sequences as discrete model tokens, making reconstruction error and token count determinants of policy fidelity and execution cost.

### Wafer-scale acceleration

Identity: `mlsys.wafer-scale-acceleration`

Executing ML on a compute-and-memory array spanning most of a semiconductor wafer, making spatial placement and communication between tiles central to performance.

### Embedding model migration

Identity: `mlsys.embedding-migration`

Changing the encoder behind a vector index while keeping queries and stored vectors comparable, by re-encoding the corpus, validated mappings, or backward-compatible training, with a controlled serving cutover.

### Register pressure

Identity: `mlsys.register-pressure`

Demand for registers from simultaneously live kernel values, which can reduce resident threads or force values to spill into slower memory.

### Software pipelining

Identity: `mlsys.software-pipelining`

Scheduling independent operations from different loop iterations to overlap, such as loading the next tile while computing the current one, hiding latency at the cost of extra live state and buffering.

### Error-correcting code memory

Identity: `mlsys.ecc-memory`

Memory protected by redundant check bits that correct supported error patterns and report detectable uncorrectable errors, reducing the risk of silent corruption.

### Integer-only inference

Identity: `mlsys.integer-only-inference`

Executing a quantized model with integer arithmetic throughout its operator graph, including accumulation and rescaling, so no floating-point operations are needed at inference time.

### Execute-in-place weights

Identity: `mlsys.execute-in-place`

Reading immutable model weights from memory-mapped flash during inference without first copying them into SRAM, reducing working memory while exposing flash-access latency.

### Confused deputy

Identity: `mlsys.confused-deputy`

A security failure in which a component uses its legitimate authority for an untrusted party's purpose because it does not distinguish the requester's authority from its own.

### Model-grader bias

Identity: `mlsys.judge-bias`

Systematic scoring errors in model-based graders, such as preferences for response position, length, or style, that distort comparisons independently of task quality.

### Over-refusal

Identity: `mlsys.over-refusal`

Refusing benign requests that resemble harmful ones, the false-positive side of a safety filter or refusal policy, which is traded against missed harmful requests when setting the operating point.

### Coordinate frames and transforms

Identity: `mlsys.coordinate-frames`

Reference frames and the spatial transforms between them that express sensor observations, robot geometry, and actions in consistent coordinates.

### Sensor degradation

Identity: `mlsys.sensor-degradation`

Loss of sensor measurement fidelity under environmental or operating conditions, such as glare, rain, fog, dust, or a dirty lens, that degrades perception even when the sensing hardware has not failed.

### Data curriculum and annealing

Identity: `mlsys.data-curriculum`

Changing the training data distribution over a run, such as raising the share of high-quality or long-sequence data during the final learning-rate decay, to improve final capability per unit of compute.

### Sharded streaming dataset format

Identity: `mlsys.streaming-dataset-format`

Organizing training examples into sequentially readable shard files so workers can stream records from remote storage with few small-object reads and bounded local staging.

### Collective algorithm selection

Identity: `mlsys.collective-algorithm-selection`

Choosing, per call, among collective algorithms (ring, tree, switch-offloaded) and wire protocols by message size, participant count, and link hierarchy, to minimize latency for small messages and maximize bandwidth for large ones.

### Always-on sensing cascade

Identity: `mlsys.always-on-cascade`

Using a low-power sensing or inference stage to wake a more expensive stage only when needed, cutting continuous-monitoring energy at the risk of missed triggers and wake latency.

### Fault tree analysis

Identity: `mlsys.fault-tree-analysis`

Top-down hazard analysis that decomposes an undesired top event into Boolean combinations of contributing faults, so minimal cut sets and their probabilities can be computed.

### Metastable failure

Identity: `mlsys.metastable-failure`

A state in which a transient trigger, such as a load spike or slowdown, starts a self-sustaining loop of retries, cache misses, or timeouts that keeps a system overloaded after the trigger is gone.

### Fan-out tail amplification

Identity: `mlsys.fan-out-tail-amplification`

The growth in a request's latency tail when it waits on many parallel subrequests, because the chance that at least one subrequest is slow rises with the fan-out.

### Proxy discrimination

Identity: `mlsys.proxy-discrimination`

Decisions driven by features that encode protected-group membership, so group disparities persist even when the protected attribute itself is omitted.

### Cross-embodiment training

Identity: `mlsys.cross-embodiment-training`

Training one policy across datasets from several robot embodiments, aligning observations and action interfaces so learning is shared despite differences in sensors and control.

### Dual-rate hierarchical policy

Identity: `mlsys.dual-rate-hierarchical-policy`

Splitting a robot policy into a slow high-level model that sets goals or latent commands and a fast low-level controller that tracks them at control rate, so large-model latency does not set the control loop's rate.

### Robot kinematics

Identity: `mlsys.robot-kinematics`

The geometric mapping between joint positions and velocities and the pose and velocity of a robot's links and end effector, including inverse solutions and singular configurations where motion authority is lost.

### Quantization granularity

Identity: `mlsys.quantization-granularity`

The grouping of values that share quantization parameters, such as a scale and an optional zero point, from per tensor to per channel to small groups, trading approximation error against metadata and kernel cost.

### Agent training environment

Identity: `mlsys.agent-training-environment`

A task system that supplies observations, accepts actions, and manages episode state for training tool-using policies, whose throughput and reset fidelity limit rollout collection.

### Model-directed control flow

Identity: `mlsys.model-directed-control-flow`

A control structure in which model outputs select the next step, tool call, or stopping decision within runtime-enforced limits, increasing flexibility while making cost and behavior harder to predict and test.

### Misuse monitoring

Identity: `mlsys.misuse-monitoring`

Detecting policy-violating use of a deployed model across requests, sessions, and accounts, such as coordinated abuse or capability probing, and responding with throttling, review, or account action.

### Masked diffusion decoding

Identity: `mlsys.masked-diffusion-decoding`

Generating text by repeatedly predicting and committing masked positions in parallel, trading refinement passes for token parallelism while weakening the attention-state reuse that autoregressive decoding relies on.

### Residual connection

Identity: `mlsys.residual-connection`

A connection that adds a block's input, or a projection of it, to the block's output, so information and gradients can bypass the block and deep stacks stay trainable.

### Multi-task learning

Identity: `mlsys.multi-task-learning`

Jointly learning several tasks through shared parameters, often a shared backbone with task-specific heads, so their training signals can help or interfere with one another.

### Recovery time and point objectives

Identity: `mlsys.recovery-objectives`

Stated limits on how long restoring service may take (RTO) and how far recovered state may lag the state before the failure (RPO), which constrain checkpointing, replication, and failover design.

### Incremental checkpointing

Identity: `mlsys.incremental-checkpointing`

Persisting only the state that changed since the preceding checkpoint, with periodic full checkpoints as bases, which cuts write volume when updates are sparse at the cost of reconstructing state from a chain on restore.

### Multiple comparisons correction

Identity: `mlsys.multiple-comparisons`

Adjusting significance thresholds when many hypotheses, metrics, or slices are tested together, so the chance of a false declared effect, by family-wise error or false discovery rate, stays controlled.

### Model-assisted labeling

Identity: `mlsys.model-assisted-labeling`

Using a model's predictions as draft labels that annotators accept or correct, which raises labeling throughput but can anchor annotators to the model's errors.

### Forward error correction

Identity: `mlsys.forward-error-correction`

Adding redundant bits to transmitted data so the receiver corrects supported error patterns without retransmission, at the cost of coding overhead and decoding delay.

### Erasure coding

Identity: `mlsys.erasure-coding`

Encoding stored data into data and parity fragments so it can be reconstructed after a bounded number of fragment losses, trading storage overhead against encoding and recovery work.

### Bandwidth-delay product

Identity: `mlsys.bandwidth-delay-product`

A path's bottleneck bandwidth times its round-trip time, the amount of outstanding data a window-controlled flow must sustain to keep the path full.

### Backfill scheduling

Identity: `mlsys.backfill-scheduling`

Running lower-priority jobs in idle resource-time gaps without delaying the reserved start of a higher-priority job, which raises cluster utilization.

### Network traffic classes and QoS

Identity: `mlsys.traffic-classes`

Classifying traffic on a shared network into separately scheduled queues with configured priorities or bandwidth shares, limiting interference between collective, storage, and control traffic.

### Telemetry sampling and cardinality control

Identity: `mlsys.telemetry-sampling`

Choosing which traces, events, and metric label combinations to keep, by sampling and by bounding or aggregating label values, so fleet telemetry stays affordable at the risk of losing rare failures and fine-grained detail.

### DMA cache maintenance

Identity: `mlsys.dma-cache-maintenance`

Cleaning and invalidating processor caches around DMA transfers on hardware without coherent DMA, with buffers aligned to cache lines, so a device and the CPU do not read stale copies of a shared buffer; each maintenance step adds latency.

### Expert offloading

Identity: `mlsys.expert-offloading`

Keeping some mixture-of-experts weights in host memory or storage and fetching the experts a batch needs into accelerator memory, often prefetching predicted ones, which lets a model larger than accelerator memory run at the cost of transfers.

### Variable-length parallelism planning

Identity: `mlsys.variable-length-parallelism`

Choosing, for each training step, sequence or context parallel degrees and how packed documents are sharded across ranks from that step's sequence lengths and attention patterns, so ranks receive balanced attention work.

### Fine-grained accelerator sharing

Identity: `mlsys.fine-grained-accelerator-sharing`

Scheduling co-located workloads on one accelerator in small units of a kernel, so latency-sensitive work can preempt background work quickly and interference stays bounded.

### Intra-device overlap of bottlenecks

Identity: `mlsys.intra-device-overlap`

Scheduling generative-inference operations with complementary compute, memory, or communication demands concurrently on one accelerator, using small batches or fused kernels so several hardware resources stay busy together.

### Graph pipeline parallelism

Identity: `mlsys.graph-pipeline-parallelism`

Pipeline parallelism that partitions a model whose layers form a branching graph into stages that follow the graph, so independent branches, such as separate modality encoders, run concurrently in parallel stages instead of being serialized into one chain.

### RAG pipeline scheduling

Identity: `mlsys.rag-pipeline-scheduling`

Coordinating the retrieval and generation work of retrieval-augmented requests across requests and processors, overlapping and batching work across the stage boundary and optionally adapting retrieval settings per query, to meet latency and answer-quality targets.

### Cross-stack profiling

Identity: `mlsys.cross-stack-profiling`

Profiling that correlates timestamped events and calling contexts from model code, frameworks, native libraries, and accelerator kernels, so that device time and stalls can be attributed to the operations that launched them.
