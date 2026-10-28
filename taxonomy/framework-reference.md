# Machine learning systems: a candidate body of knowledge

Version: `0.3-draft`. **Editorial draft.**

A candidate classification of the engineering knowledge needed to construct, evaluate, deploy, and operate machine learning systems, including generative, agentic, and physical settings. The focus is systems for ML; using ML to optimize unrelated systems is adjacent. This is a scoped editorial proposal, not an exhaustive or community-validated definition of the field.

Each concept has one candidate home for editorial maintenance. All facet placements are candidate editorial synthesis; empty contexts mean relevance has not been assessed. This does not assert exclusive membership or a broader/narrower hierarchy. Passage links do not certify coverage, teaching depth, prerequisites, or competence.

## Dimensions

- **Activities:** Specify and design; Acquire and prepare data; Learn and adapt; Evaluate and assure; Deploy and integrate; Operate and change; Retire and decommission.
- **Layers:** Application and workflow; Model and algorithm; Numerical software and runtime; Compute and communication infrastructure; People and operating environment.
- **Concerns:** Task and predictive quality; Latency, throughput, and resources; Reliability and recovery; Security; Privacy; Fairness; Human control and accountability; Energy and environmental impacts; Monetary cost; Safety; Accessibility; Reproducibility.
- **Contexts:** Classical ML; Deep learning; Generative systems; Agentic systems; Physical and control systems; Batch and offline; Interactive services; Streaming systems; Edge and device; Distributed systems; Federated settings; Recommendation and ranking; Multimodal systems.

## Foundations

What does the system compute, what must it achieve, and how do we reason about its cost?

### Workloads &amp; requirements

What must the system achieve, for which workload, and how is success decided?

Targets, workload descriptions, and objectives that every later engineering choice is judged against.

#### Requirements, SLOs &amp; baselines

Identity: `requirements-slos`

Stating measurable targets for a system's behavior and service, and the baselines and budgets against which they are judged.

- **Activities:** Specify and design; Evaluate and assure; Operate and change.
- **Layers:** Application and workflow; Numerical software and runtime.
- **Concerns:** Task and predictive quality; Latency, throughput, and resources; Reliability and recovery.
- **Contexts:** Interactive services; Batch and offline.

Includes: Acceptance criteria and baselines; SLIs, SLOs, error budgets; Latency budgets across a request path.

Excludes: Measured latency behavior and goodput (Latency, throughput &amp; goodput); Statistical evaluation of model quality (Model evaluation &amp; testing).

Design warrants: `se4ml`, `sre-slo`.

Candidate home concepts:

- **System requirements** (`mlsys.requirements`): Measurable desired behavior and constraints for a specified workload.
- **Baseline** (`mlsys.baseline`): A reference method or system used to justify the value of a proposed change.
- **Service-level objective** (`mlsys.slo`): A target for a defined service measure over a specified population and window.
- **Service-level indicator** (`mlsys.sli`): A quantitative measure of one aspect of delivered service, such as latency, availability, or accepted-output rate, computed over a stated population and window.
- **Error budget** (`mlsys.error-budget`): The amount of service-level-objective violation permitted over a window, used to pace changes and risk against a reliability target.
- **Latency budget** (`mlsys.latency-budget`): An allocation of an end-to-end latency target across the stages of a request path, such as network, preprocessing, inference, and postprocessing.
- **Recovery time and point objectives** (`mlsys.recovery-objectives`): Stated limits on how long restoring service may take (RTO) and how far recovered state may lag the state before the failure (RPO), which constrain checkpointing, replication, and failover design.

Secondary concept connections (not additional homes):

- **Tail latency** (`mlsys.tail-latency`)
- **SLO goodput** (`mlsys.slo-goodput`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: ML Lifecycle — System requirements (`vol1.lifecycle.requirements`)
- vol1: Benchmarking — Latency SLO (`vol1.measurement.slo`)
- vol2: ML Operations at Scale — Evaluation contract (`vol2.operations.evaluation-contract`)
- vol1: Model Serving — Request latency (`vol1.serving.latency`)
- vol2: Serving at Scale — SLO goodput (`vol2.serving.slo-goodput`)
- vol1: Model Serving — Request latency (`vol1.serving.latency`)

#### Workload characterization

Identity: `workload-characterization`

Describing a workload's compute, memory, data, and traffic demands, and where it runs, so it can be engineered and provisioned for.

- **Activities:** Specify and design; Deploy and integrate.
- **Layers:** Model and algorithm; Numerical software and runtime; Compute and communication infrastructure.
- **Concerns:** Latency, throughput, and resources; Monetary cost.
- **Contexts:** Edge and device; Distributed systems; Deep learning.

Includes: FLOP, byte, and memory accounting; Deployment tiers from cloud to TinyML; Workload traces.

Excludes: Node-level job placement (Cluster scheduling &amp; orchestration); Measured profiles (Profiling &amp; utilization).

Design warrants: `cs336`, `dlsys`.

Candidate home concepts:

- **Workload characterization** (`mlsys.workload`): The input, request, model, and execution distributions that a system must support.
- **Execution placement** (`mlsys.placement`): The choice of where system components execute, such as device, edge, or datacenter.
- **Operation and byte accounting** (`mlsys.flop-accounting`): Estimating a workload's arithmetic operations and bytes moved per example, token, or step from model shapes, to predict cost before measurement.
- **Memory footprint** (`mlsys.memory-footprint`): The peak memory a workload occupies, summed over parameters, activations, optimizer or cache state, and runtime buffers.
- **Workload trace** (`mlsys.workload-trace`): A recorded sequence of requests or jobs with arrival times, sizes, and durations, replayed to evaluate capacity, scheduling, or serving policies.
- **Inference compute and memory estimate** (`mlsys.inference-compute-estimate`): Approximating dense transformer inference as about two FLOPs per parameter per token, plus KV-cache bytes from layers, KV heads, head dimension, context length, batch, and precision.

Secondary concept connections (not additional homes):

- **Arrival rate** (`mlsys.arrival-rate`)
- **Capacity planning** (`mlsys.capacity-planning`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Model Serving — Arrival rate (`vol1.serving.arrivals`)
- vol1: Model Serving — Serving capacity (`vol1.serving.capacity`)

#### Objectives, proxies &amp; decision policies

Identity: `objectives-decisions`

How learned outputs become decisions, how optimized proxies relate to intended outcomes, and how competing objectives are traded.

- **Activities:** Specify and design; Evaluate and assure.
- **Layers:** Application and workflow; Model and algorithm.
- **Concerns:** Task and predictive quality; Safety; Fairness.
- **Contexts:** Classical ML; Agentic systems.

Includes: Proxy objectives and reward hacking; Cost-sensitive thresholds; Multi-objective tradeoffs.

Excludes: Fairness criteria (Responsible AI); Optimizer and loss mechanics (Training &amp; adaptation).

Design warrants: `rules`, `ml-test-score`.

Candidate home concepts:

- **Proxy objective** (`mlsys.proxy-objective`): A measurable target used in place of the outcome actually desired.
- **Decision policy** (`mlsys.decision-policy`): The rule that converts model outputs and system state into actions, thresholds, abstentions, or escalation.
- **Cost-sensitive decision** (`mlsys.cost-sensitive-decision`): Choosing actions or thresholds to minimize expected cost when different error types carry unequal consequences.
- **Pareto frontier** (`mlsys.pareto-frontier`): The set of configurations for which no objective, such as accuracy, latency, or cost, can improve without worsening another.

Secondary concept connections (not additional homes):

- **Probability calibration** (`mlsys.calibration`)
- **Selective prediction** (`mlsys.selective-prediction`)
- **Reward hacking** (`mlsys.reward-hacking`)

Related MLSysBook passages (teaching depth not assessed):

- vol2: Robust AI — Selective prediction (`vol2.robustness.selective-prediction`)
- vol3: Reinforcement Learning from Verifiable Rewards — Reward hacking (`vol3.reward-hacking`)

### Models &amp; architectures

What does the model compute, and how does its structure set system cost?

Model families, building blocks, and architectural choices viewed as workloads, plus the artifacts that carry them.

#### Model families

Identity: `model-families`

The classical and neural model regimes, with their inductive assumptions, viewed as workloads with characteristic compute and memory profiles.

- **Activities:** Specify and design; Learn and adapt.
- **Layers:** Model and algorithm.
- **Concerns:** Task and predictive quality; Latency, throughput, and resources.
- **Contexts:** Classical ML; Deep learning; Generative systems; Recommendation and ranking; Multimodal systems.

Includes: Classical models and neural networks; Transformers, recommendation models, diffusion models.

Excludes: Learning paradigms (Training &amp; adaptation); Layer-level building blocks (Architecture building blocks).

Design warrants: `sklearn`, `dlsys`.

Candidate home concepts:

- **Linear models** (`mlsys.linear-model`): Models whose score is linear in their supplied features, with a possible output link function.
- **Tree ensembles** (`mlsys.tree-ensemble`): Collections of decision trees whose branch traversal and aggregation determine predictions.
- **Kernel methods** (`mlsys.kernel-method`): Models using similarity functions whose storage and evaluation costs depend on their representation.
- **Nearest-neighbor models** (`mlsys.nearest-neighbor`): Predictions or retrieval based on similarity to stored examples or vectors.
- **Neural networks** (`mlsys.neural-network`): Parameterized compositions of differentiable operations used to learn representations and predictions.
- **Inductive bias** (`mlsys.inductive-bias`): Structural assumptions that favor some learned functions over others.
- **Transformer** (`mlsys.transformer`): A neural architecture of stacked attention and feed-forward blocks with residual connections and normalization, processing all sequence positions in parallel during training.
- **Deep recommendation model** (`mlsys.recommendation-model`): Models combining large sparse embedding tables for categorical features with dense layers, whose cost is often dominated by embedding lookups and memory capacity.
- **Diffusion model** (`mlsys.diffusion-model`): A generative model that learns to reverse a gradual noising process, producing samples through many iterative denoising network evaluations.
- **Multimodal model** (`mlsys.multimodal-model`): A model that consumes or produces several modalities, such as text, images, audio, or video, through modality encoders feeding a shared backbone.
- **Generative adversarial network** (`mlsys.gan`): A generator and a discriminator trained against each other, so the generator learns to produce samples the discriminator cannot tell apart from data.
- **Graph neural network** (`mlsys.graph-neural-network`): Neural network that computes node, edge, or graph representations by repeatedly aggregating features from each node's neighbors along graph edges.
- **Large language model** (`mlsys.large-language-model`): A language model large enough that parameter memory, attention state that grows with context, and token-by-token decoding usually dominate its training and serving cost. Tag it only when an item is about that workload class itself, not merely because an LLM is involved.
- **Encoder-only, decoder-only, and encoder-decoder Transformers** (`mlsys.transformer-variants`): The three arrangements of Transformer stacks, which differ in whether inputs are processed bidirectionally, outputs are generated autoregressively, or both, and therefore in how they are trained, cached, and served.

Related MLSysBook passages (teaching depth not assessed):

- vol1: Network Architectures — Inductive bias (`vol1.architecture.bias`)

#### Architecture building blocks

Identity: `architecture-blocks`

The layer-level operations from which neural models are composed and whose shapes determine compute, memory, and data movement.

- **Activities:** Specify and design; Learn and adapt.
- **Layers:** Model and algorithm.
- **Concerns:** Latency, throughput, and resources; Task and predictive quality.
- **Contexts:** Deep learning; Generative systems; Recommendation and ranking; Multimodal systems.

Includes: Embeddings, attention, position encoding, tokenization; Convolution, recurrence, normalization.

Excludes: Kernel implementations of these operations (Tensor operators &amp; memory layout); Efficient attention variants (Long-context &amp; efficient sequence models).

Design warrants: `dlsys`, `cs336`.

Candidate home concepts:

- **Embeddings** (`mlsys.embeddings`): Vector representations whose dimension and lookup or construction method affect storage and computation.
- **Attention** (`mlsys.attention`): Input-dependent aggregation of value vectors using query-key compatibility scores.
- **Position encoding** (`mlsys.position-encoding`): Representations that make sequence position available to a model.
- **Tokenization** (`mlsys.tokenization`): Mapping text into vocabulary-dependent token identifiers, which sets sequence lengths and so influences context usage, embedding size, and processing cost.
- **Convolution** (`mlsys.convolution`): A locally connected operation applying shared filters across spatial or temporal positions, giving high weight reuse and arithmetic intensity.
- **Recurrent computation** (`mlsys.recurrence`): Computation that updates a hidden state step by step over a sequence, creating sequential dependencies that limit parallelism across time.
- **Normalization layers** (`mlsys.normalization-layer`): Layers that rescale activations using batch, layer, or root-mean-square statistics to stabilize and speed training of deep networks.
- **Depthwise separable convolution** (`mlsys.depthwise-separable-convolution`): Factoring a standard convolution into a per-channel spatial convolution followed by a pointwise convolution, reducing operations and parameters.
- **Modality encoder** (`mlsys.modality-encoder`): A network that turns images, audio, or video into token sequences for a shared backbone; visual tokens often dominate prefill cost and KV size.
- **Residual connection** (`mlsys.residual-connection`): A connection that adds a block's input, or a projection of it, to the block's output, so information and gradients can bypass the block and deep stacks stay trainable.

Secondary concept connections (not additional homes):

- **KV cache** (`mlsys.kv-cache`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Network Architectures — Self-attention (`vol1.architecture.attention`)
- vol1: Model Serving — KV cache per conversation (`vol1.serving.kv`)
- vol2: Serving at Scale — KV cache (`vol2.serving.kv-cache`)
- vol3: KV Cache Management — Retained KV state (`vol3.kv`)

#### Sparse &amp; conditional computation

Identity: `conditional-computation`

Model designs in which the computation performed varies per input, such as mixture-of-experts routing and early exit.

- **Activities:** Specify and design; Learn and adapt; Deploy and integrate.
- **Layers:** Model and algorithm; Numerical software and runtime.
- **Concerns:** Latency, throughput, and resources; Monetary cost.
- **Contexts:** Deep learning; Generative systems; Distributed systems.

Includes: Mixture of experts, routing, capacity, load balancing; Active vs. total parameters; Early exit.

Excludes: Weight pruning and static sparsity (Model compression); Cascades across separate models (Serving architectures).

Design warrants: `cs336`, `llmsys-cmu`.

Candidate home concepts:

- **Mixture of experts** (`mlsys.mixture-of-experts`): An architecture in which a learned router activates a small subset of expert subnetworks per token, decoupling total parameter count from per-token compute.
- **Expert routing** (`mlsys.expert-routing`): The gating function that assigns each token to one or more experts, such as top-k softmax gating or expert-choice routing.
- **Expert capacity** (`mlsys.expert-capacity`): The maximum number of tokens an expert processes per batch, set by a capacity factor; tokens beyond it are dropped or rerouted.
- **Load-balancing loss** (`mlsys.load-balancing-loss`): An auxiliary training objective that penalizes uneven token assignment across experts so routing stays balanced.
- **Active parameters** (`mlsys.active-parameters`): The parameters actually used to process one token or example, as distinct from total stored parameters in conditionally computed models.
- **Early exit** (`mlsys.early-exit`): Terminating inference at an intermediate layer once a confidence criterion is met, so per-input compute varies with input difficulty.

Secondary concept connections (not additional homes):

- **Expert parallelism** (`mlsys.expert-parallel`)
- **Sparsity** (`mlsys.sparsity`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Model Compression — Weight sparsity (`vol1.compression.sparsity`)

#### Long-context &amp; efficient sequence models

Identity: `efficient-sequence-models`

Sequence architectures that change how cost scales with context length, co-designed with the memory and compute they demand.

- **Activities:** Specify and design; Learn and adapt; Deploy and integrate.
- **Layers:** Model and algorithm.
- **Concerns:** Latency, throughput, and resources; Monetary cost; Task and predictive quality.
- **Contexts:** Generative systems; Deep learning.

Includes: Sparse, sliding-window, and linear attention; State-space models; grouped-query attention; Context-length extension.

Excludes: Exact attention kernels such as FlashAttention (Tensor operators &amp; memory layout); KV-cache management (LLM &amp; generative inference).

Design warrants: `ut378`, `mit-65940`.

Candidate home concepts:

- **Sparse and sliding-window attention** (`mlsys.sliding-window-attention`): Attention restricted to local windows or fixed sparse patterns so cost grows linearly rather than quadratically with sequence length.
- **Linear attention** (`mlsys.linear-attention`): Attention reformulated with kernel feature maps so it is computed in linear time with a fixed-size recurrent state.
- **State-space sequence model** (`mlsys.state-space-model`): Sequence models built on discretized linear state-space recurrences, trainable in parallel and run with constant-size state per step at inference.
- **Grouped-query attention** (`mlsys.grouped-query-attention`): Attention in which groups of query heads share key and value heads, shrinking the key-value cache and decode memory traffic.
- **Context-length extension** (`mlsys.context-extension`): Techniques that extend a trained model's usable sequence length, such as position-encoding interpolation followed by long-sequence fine-tuning.
- **Context window** (`mlsys.context`): The maximum token span a model supports in a stated execution mode, bounding the history it can use and the longest sequences the system must process.
- **Long-context degradation** (`mlsys.context-degradation`): Decline in a model's use of relevant information as context grows or as that information sits away from the context's start or end.
- **Dynamic sparse attention** (`mlsys.dynamic-sparse-attention`): Attention that computes only the query-key blocks predicted important for each input at runtime, rather than a fixed pattern, to cut long-context cost.
- **Multi-head latent attention** (`mlsys.multi-head-latent-attention`): Attention that caches one low-rank latent vector per token in place of per-head keys and values, reconstructing them or absorbing the up-projections into the query and output weights, which shrinks the KV cache and decode memory traffic.

Secondary concept connections (not additional homes):

- **KV cache** (`mlsys.kv-cache`)
- **IO-aware attention kernel** (`mlsys.io-aware-attention`)
- **Sequence and context parallelism** (`mlsys.sequence-parallel`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Model Serving — KV cache per conversation (`vol1.serving.kv`)
- vol2: Serving at Scale — KV cache (`vol2.serving.kv-cache`)
- vol3: KV Cache Management — Retained KV state (`vol3.kv`)
- vol3: Context Engineering — Working context (`vol3.working-context`)

#### Model artifacts &amp; formats

Identity: `model-artifacts`

The weights, graphs, metadata, and interface contracts that make a trained model a portable, loadable, deployable artifact.

- **Activities:** Deploy and integrate; Operate and change.
- **Layers:** Model and algorithm; Numerical software and runtime.
- **Concerns:** Reliability and recovery; Reproducibility; Security.
- **Contexts:** Edge and device.

Includes: Weight serialization and exchange formats; Model export and signatures.

Excludes: Versioning and registries (Versioning, rollout &amp; rollback); Training checkpoints for recovery (Reliability &amp; fault tolerance).

Design warrants: `mlops`, `mlflow-dependencies`.

Candidate home concepts:

- **Model artifact** (`mlsys.model-artifact`): The learned parameters, structures, example stores, and metadata required to execute a model.
- **Weight serialization format** (`mlsys.weight-serialization`): The on-disk encoding of a model's parameter tensors and metadata, governing load time, memory mapping, and safety of deserialization.
- **Model exchange format** (`mlsys.model-exchange-format`): A framework-neutral representation of a model's operator graph and weights used to move models between training frameworks and inference runtimes.
- **Model export** (`mlsys.model-export`): Converting a framework program into a self-contained deployable artifact by tracing or capturing its graph and fixing operators, shapes, and weights.
- **Model signature** (`mlsys.model-signature`): The declared names, types, and shapes of a model artifact's inputs and outputs, forming its interface contract with callers.

Secondary concept connections (not additional homes):

- **Artifact versioning** (`mlsys.artifact-version`)
- **Recovery checkpoint** (`mlsys.checkpoint`)

Related MLSysBook passages (teaching depth not assessed):

- vol2: Data Storage — Durable checkpoint (`vol2.storage.durable-checkpoint`)
- vol3: Durable Execution — Recovery checkpoint (`vol3.checkpoint`)
- vol2: Data Storage — Versioned artifact (`vol2.storage.versioned-artifact`)
- vol4: Deployment Release — Release manifest (`vol4.assurance.manifest`)

### Quantitative system models

Which simple models predict where time, memory, and money go?

Analytical and empirical models used to bound performance, capacity, and training cost before or instead of measuring.

#### Roofline, performance models &amp; simulation

Identity: `roofline-locality`

Bound-based reasoning about whether computation is limited by arithmetic or data movement, and how reuse changes that limit.

- **Activities:** Specify and design; Evaluate and assure.
- **Layers:** Numerical software and runtime; Compute and communication infrastructure.
- **Concerns:** Latency, throughput, and resources; Energy and environmental impacts.
- **Contexts:** Deep learning.

Includes: Ridge point and compute- vs. memory-bound regimes; Energy cost of data movement.

Excludes: Measured profiles and MFU (Profiling &amp; utilization); Hardware memory organization (Memory hierarchy &amp; bandwidth).

Design warrants: `roofline`, `cs149`.

Candidate home concepts:

- **Arithmetic intensity** (`mlsys.arithmetic-intensity`): Arithmetic work per byte transferred at a specified memory boundary.
- **Roofline bound** (`mlsys.roofline`): A performance bound combining a compute ceiling with an intensity-dependent bandwidth ceiling.
- **Data locality** (`mlsys.locality`): Reuse of data near its consumers across time or execution locations.
- **Ridge point** (`mlsys.ridge-point`): The arithmetic intensity at which a roofline's bandwidth and compute ceilings meet, equal to peak compute throughput divided by peak memory bandwidth.
- **Compute-bound vs. memory-bound** (`mlsys.bound-regime`): Classifying a kernel or phase by which roofline ceiling limits it, indicating whether more arithmetic throughput or more bandwidth would help.
- **Data-movement energy** (`mlsys.data-movement-energy`): The energy cost of moving operands between storage levels, which grows with distance and typically exceeds the energy of the arithmetic performed on them.
- **Analytical performance model** (`mlsys.analytical-performance-model`): A closed-form model composing compute, memory, communication, and queueing terms to predict a workload's time, memory, and cost on a configuration without running it.
- **Performance simulation** (`mlsys.performance-simulation`): Predicting system behavior by executing a model of compute, memory, network, or scheduling events at a chosen granularity, trading simulation speed against detail.
- **Simulator fidelity** (`mlsys.model-fidelity`): Agreement between a performance model's or simulator's predictions and measured behavior on reference workloads, reported as validation error over stated configurations.
- **Operator execution trace** (`mlsys.execution-trace`): A recorded graph of a training or inference step's compute and communication operators with their dependencies and timings, replayable for simulation and benchmarking.

Secondary concept connections (not additional homes):

- **Tiling** (`mlsys.tiling`)
- **Memory bandwidth** (`mlsys.memory-bandwidth`)
- **Compute throughput** (`mlsys.compute-throughput`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Hardware Acceleration — Memory bandwidth (`vol1.hardware.bandwidth`)
- vol1: Hardware Acceleration — Arithmetic intensity (`vol1.hardware.intensity`)
- vol1: System Paradigms — Data locality (`vol1.system.locality`)
- vol1: Hardware Acceleration — Tiling (`vol1.hardware.tiling`)
- vol1: Hardware Acceleration — Operand transfer time (`vol1.hardware.transfer`)
- vol4: The Cognitive Brain — Weight bandwidth (`vol4.inference.bandwidth`)

#### Amdahl's law &amp; scaling efficiency

Identity: `speedup-limits`

Models of how much speedup added parallel or accelerated resources can deliver, given serial work and fixed or growing problem sizes.

- **Activities:** Specify and design; Evaluate and assure.
- **Layers:** Numerical software and runtime; Compute and communication infrastructure.
- **Concerns:** Latency, throughput, and resources; Monetary cost.
- **Contexts:** Distributed systems.

Includes: Amdahl's law and serial fractions; Strong and weak scaling; Parallel efficiency.

Excludes: Communication cost models (Collective communication).

Design warrants: `cs149`, `dlsys`.

Candidate home concepts:

- **Serial fraction** (`mlsys.serial-fraction`): The portion of execution that a proposed parallel or accelerated transformation leaves unchanged.
- **Amdahl's law** (`mlsys.amdahls-law`): The bound that overall speedup from accelerating part of an execution is limited by the fraction left unaccelerated.
- **Strong scaling** (`mlsys.strong-scaling`): How time to solution changes as resources increase for a fixed total problem size.
- **Weak scaling** (`mlsys.weak-scaling`): How time to solution changes as resources and problem size grow proportionally, keeping work per resource fixed.
- **Parallel scaling efficiency** (`mlsys.parallel-efficiency`): Achieved speedup divided by the factor of added resources, measuring how much additional hardware turns into useful progress.

Secondary concept connections (not additional homes):

- **Stragglers** (`mlsys.stragglers`)
- **Communication overlap** (`mlsys.communication-overlap`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Hardware Acceleration — Accelerated fraction (`vol1.hardware.amdahl`)
- vol2: Collective Communication — Communication overlap (`vol2.collective.overlap`)
- vol1: Hardware Acceleration — Accelerated fraction (`vol1.hardware.amdahl`)
- vol2: Introduction — Distributed step time (`vol2.fleet.step-time`)

#### Queueing &amp; capacity models

Identity: `queueing-models`

Models relating arrivals, service rates, utilization, and waiting, used to predict latency and capacity under load.

- **Activities:** Specify and design; Deploy and integrate; Operate and change.
- **Layers:** Numerical software and runtime; Compute and communication infrastructure.
- **Concerns:** Latency, throughput, and resources; Monetary cost.
- **Contexts:** Interactive services; Streaming systems.

Includes: Little's law; Latency-throughput curves; Burstiness and variability.

Excludes: Batching and admission mechanisms (Batching, admission &amp; autoscaling); Measuring a specific serving system's latency and throughput (Latency, throughput &amp; goodput).

Design warrants: `sre`, `cs329s`.

Candidate home concepts:

- **Arrival rate** (`mlsys.arrival-rate`): The frequency and distribution of requests reaching a service boundary.
- **Service capacity** (`mlsys.service-capacity`): The completion rate supported by a service under a stated workload and operating point.
- **Queueing delay** (`mlsys.queueing`): Time a request waits for a resource before receiving service.
- **Little's law** (`mlsys.littles-law`): The relation that the mean number of items in a stable system equals mean arrival rate times mean time in the system.
- **Latency-throughput curve** (`mlsys.latency-throughput-curve`): The relationship between offered load and latency, in which queueing delay grows sharply as utilization approaches capacity.
- **Arrival burstiness** (`mlsys.arrival-burstiness`): Variability of request inter-arrival times or sizes beyond a Poisson baseline, which raises queueing delay at a given mean utilization.
- **Finite buffers and overflow** (`mlsys.finite-buffer`): A queue with bounded capacity drops or blocks arrivals once full, so its size trades memory and added delay against the loss rate under bursty input.
- **Bandwidth-delay product** (`mlsys.bandwidth-delay-product`): A path's bottleneck bandwidth times its round-trip time, the amount of outstanding data a window-controlled flow must sustain to keep the path full.

Secondary concept connections (not additional homes):

- **Tail latency** (`mlsys.tail-latency`)
- **Request batching** (`mlsys.batching`)
- **Resource utilization** (`mlsys.utilization`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Model Serving — Utilization (`vol1.serving.utilization`)
- vol1: Model Serving — Arrival rate (`vol1.serving.arrivals`)
- vol1: Model Serving — Serving capacity (`vol1.serving.capacity`)
- vol1: Model Serving — Queueing delay (`vol1.serving.queue`)
- vol1: Model Serving — Request latency (`vol1.serving.latency`)
- vol1: Model Serving — Batching (`vol1.serving.batching`)
- vol2: Serving at Scale — Request batching (`vol2.serving.batching`)
- vol1: Model Serving — Queueing delay (`vol1.serving.queue`)
- vol1: Model Serving — Batch formation delay (`vol1.serving.formation`)

#### Scaling laws &amp; compute-optimal allocation

Identity: `scaling-laws`

Empirical relationships among parameters, data, and compute and observed quality, used to allocate training budgets.

- **Activities:** Specify and design; Learn and adapt.
- **Layers:** Model and algorithm.
- **Concerns:** Monetary cost; Task and predictive quality.
- **Contexts:** Generative systems; Deep learning.

Includes: Loss scaling laws; Compute-optimal model/data allocation; Training compute estimates.

Excludes: Hyperparameter search (Hyperparameter search &amp; experiment management); Test-time compute scaling (Test-time compute).

Design warrants: `cs336`.

Candidate home concepts:

- **Empirical scaling laws** (`mlsys.scaling-laws`): Fitted relationships between resources, data, model size, and observed model performance.
- **Compute-optimal allocation** (`mlsys.compute-optimal-allocation`): Choosing model size and training data quantity to minimize loss under a fixed training compute budget, using fitted scaling laws.
- **Training compute estimate** (`mlsys.training-compute-estimate`): Approximating dense transformer training compute as roughly six floating-point operations per parameter per training token.

Secondary concept connections (not additional homes):

- **Experiment-budget allocation** (`mlsys.experiment-budget`)

Related MLSysBook passages (teaching depth not assessed):

- vol2: Introduction — Empirical scaling laws (`vol2.fleet.scaling-laws`)
- vol1: Model Training — Time to target quality (`vol1.training.quality-time`)
- vol1: Data Selection — Model-data allocation (`vol1.selection.balance`)

## Stack

What does the system run on, from silicon to the fleet?

### Hardware &amp; accelerators

How do processors, memory, and chip-level power shape what ML workloads can achieve?

Compute, memory, interconnect, and power characteristics of ML hardware from datacenter accelerators to microcontrollers.

#### Accelerator architecture

Identity: `accelerator-architecture`

The organization of processors specialized for ML: execution units, threading, dataflow, and the throughput they provide.

- **Activities:** Learn and adapt; Deploy and integrate.
- **Layers:** Compute and communication infrastructure.
- **Concerns:** Latency, throughput, and resources; Energy and environmental impacts.
- **Contexts:** Deep learning.

Includes: GPUs, TPUs, NPUs; SIMT and matrix engines; Systolic arrays and dataflows; Domain-specific architectures.

Excludes: On-device and embedded accelerators (Edge &amp; embedded hardware); Compiler mapping onto hardware (ML compilers &amp; kernel optimization).

Design warrants: `cuda-model`, `book-hardware`, `cs149`.

Candidate home concepts:

- **Accelerator execution architecture** (`mlsys.accelerator-execution-architecture`): Organization of arithmetic units, execution lanes or threads, scheduling resources, and local storage that determines how supported operations run. SIMD/SIMT, matrix units, and systolic arrays are distinct examples.
- **Compute throughput** (`mlsys.compute-throughput`): The sustainable rate of a specified arithmetic workload on an execution resource.
- **Systolic array** (`mlsys.systolic-array`): A grid of processing elements that pass operands rhythmically to neighbors, performing matrix multiplication with high operand reuse and little register-file traffic.
- **Matrix engine** (`mlsys.matrix-unit`): A functional unit that executes small dense matrix multiply-accumulate tiles per instruction, supplying most of an accelerator's peak low-precision throughput.
- **SIMT execution** (`mlsys.simt`): An execution model in which groups of threads issue common instructions to active lanes, with divergent paths executed under lane masks and architecture-specific reconvergence.
- **Accelerator dataflow** (`mlsys.dataflow`): The mapping that decides which operands stay resident in local storage while others stream, such as weight-, output-, or row-stationary schedules.
- **Domain-specific architecture** (`mlsys.domain-specific-architecture`): Hardware specialized to a class of workloads, trading programmability for higher performance per watt on its target operations.
- **Compute-in-memory** (`mlsys.compute-in-memory`): Performing matrix-vector operations inside or beside memory arrays, often in the analog domain, so weights need not move to separate compute units.
- **CPU execution architecture** (`mlsys.cpu-execution-architecture`): Organization of general-purpose processor cores, SIMD vector units, and multilevel caches that runs host code, preprocessing, classical models, and CPU inference.
- **Advanced accelerator packaging** (`mlsys.advanced-accelerator-packaging`): Integrating compute and memory dies through chiplets, interposers, or three-dimensional bonding to trade local bandwidth and manufacturing flexibility against power-delivery and heat-removal constraints.
- **FPGA-based ML acceleration** (`mlsys.fpga-acceleration`): Mapping ML operations to reconfigurable logic, arithmetic blocks, and local memories so datapaths can match a workload at the cost of hardware compilation and resource constraints.
- **Wafer-scale acceleration** (`mlsys.wafer-scale-acceleration`): Executing ML on a compute-and-memory array spanning most of a semiconductor wafer, making spatial placement and communication between tiles central to performance.

Secondary concept connections (not additional homes):

- **Matrix multiplication** (`mlsys.matrix-multiply`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Neural Computation — Matrix multiplication (`vol1.computation.matmul`)
- vol1: Hardware Acceleration — Operand reuse (`vol1.hardware.reuse`)

#### Memory hierarchy &amp; bandwidth

Identity: `memory-systems`

The storage levels of an ML processor, from registers and SRAM to HBM, and the capacity and bandwidth limits they impose.

- **Activities:** Learn and adapt; Deploy and integrate.
- **Layers:** Compute and communication infrastructure.
- **Concerns:** Latency, throughput, and resources; Monetary cost.
- **Contexts:** Deep learning; Generative systems; Recommendation and ranking.

Includes: On-chip SRAM and scratchpads; HBM; The memory wall.

Excludes: Persistent and distributed storage (Storage &amp; data delivery); Software memory allocation (Runtime execution &amp; dispatch).

Design warrants: `cuda-model`, `book-hardware`.

Candidate home concepts:

- **Memory hierarchy** (`mlsys.memory-hierarchy`): Arrangement of storage levels and transfer paths with differing capacity, latency, bandwidth, access control, and management mechanisms. The available levels depend on the platform.
- **Memory capacity** (`mlsys.memory-capacity`): Available storage for concurrently resident state and intermediates.
- **Memory bandwidth** (`mlsys.memory-bandwidth`): The sustainable rate at which bytes move across a specified memory interface.
- **High-bandwidth memory** (`mlsys.hbm`): Stacked DRAM placed beside the processor on an interposer, providing very high bandwidth with limited capacity per stack.
- **On-chip SRAM and scratchpads** (`mlsys.on-chip-sram`): Small, fast on-die storage, such as caches, shared memory, and register files, used to stage reused operands near compute units.
- **Memory wall** (`mlsys.memory-wall`): The widening gap between processor arithmetic throughput and memory bandwidth or latency, which makes data movement the limiting factor.
- **Embedding cache and tiering** (`mlsys.embedding-cache`): Keeping frequently accessed embedding rows in accelerator memory and cold rows in host memory or SSD, exploiting the skew of lookups.
- **Emerging non-volatile memory** (`mlsys.emerging-nvm`): Resistive, phase-change, and magnetic memory devices that retain state without power and can store weights densely or act as analog compute elements.

Secondary concept connections (not additional homes):

- **Data locality** (`mlsys.locality`)
- **Tiling** (`mlsys.tiling`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: System Paradigms — Memory capacity (`vol1.system.capacity`)
- vol2: Compute Infrastructure — Memory capacity (`vol2.hardware.memory-capacity`)
- vol1: Hardware Acceleration — Memory bandwidth (`vol1.hardware.bandwidth`)
- vol1: System Paradigms — Data locality (`vol1.system.locality`)
- vol1: Hardware Acceleration — Tiling (`vol1.hardware.tiling`)
- vol1: Hardware Acceleration — Memory bandwidth (`vol1.hardware.bandwidth`)
- vol4: The Cognitive Brain — Weight bandwidth (`vol4.inference.bandwidth`)

#### Host-device &amp; scale-up interconnect

Identity: `node-interconnect`

Links that move data between hosts and accelerators and among accelerators within a node or rack-scale domain.

- **Activities:** Learn and adapt; Deploy and integrate.
- **Layers:** Compute and communication infrastructure.
- **Concerns:** Latency, throughput, and resources.
- **Contexts:** Distributed systems.

Includes: PCIe host links; NVLink-class scale-up fabrics; CXL and peer-to-peer transfers.

Excludes: Datacenter scale-out networks (Network fabrics).

Design warrants: `book-hardware`, `cse599o`.

Candidate home concepts:

- **Host-device interconnect** (`mlsys.host-device-interconnect`): Communication path through which a host and accelerator exchange data and control, with transport, topology, and memory-access semantics that affect transfers.
- **Scale-up interconnect** (`mlsys.scale-up-interconnect`): High-bandwidth links and switches, such as NVLink-class fabrics, joining accelerators within a node or rack into one tightly coupled domain.
- **Cache-coherent interconnect** (`mlsys.coherent-interconnect`): A link protocol that lets hosts and devices share or pool memory with hardware coherence, such as CXL over a PCIe physical layer.
- **Peer-to-peer device transfer** (`mlsys.peer-to-peer-transfer`): Direct memory access between accelerators, or between accelerators and NICs or storage, without staging data through host memory.

Secondary concept connections (not additional homes):

- **Network topology** (`mlsys.network-topology`)
- **Collective communication** (`mlsys.collectives`)

Related MLSysBook passages (teaching depth not assessed):

- vol2: Collective Communication — AllReduce (`vol2.collective.allreduce`)
- vol2: Network Fabrics — Bisection bandwidth (`vol2.network.bisection-bandwidth`)

#### Chip power &amp; thermal limits

Identity: `chip-power-thermal`

Electrical and thermal limits at the chip and board level, and the runtime controls that keep execution within them.

- **Activities:** Deploy and integrate; Operate and change.
- **Layers:** Compute and communication infrastructure; People and operating environment.
- **Concerns:** Latency, throughput, and resources; Energy and environmental impacts; Reliability and recovery.
- **Contexts:** Edge and device; Physical and control systems.

Includes: TDP and thermal throttling; DVFS and power capping; Dark silicon.

Excludes: Facility power and cooling (Datacenter power, cooling &amp; facilities); Carbon accounting (Sustainability &amp; cost).

Design warrants: `nvidia-smi`, `book-hardware`.

Candidate home concepts:

- **Power delivery and integrity** (`mlsys.power-delivery`): Supply paths, conversion stages, limits, and transient behavior that must maintain required voltages while workloads and actuators draw or return power.
- **Thermal behavior and throttling** (`mlsys.thermal-dynamics`): Evolution of temperature under heat generation and removal, and the performance controls or operating restrictions that follow under a platform thermal policy.
- **Thermal design power** (`mlsys.tdp`): The sustained heat dissipation a cooling solution is designed to remove, used as a chip's nominal power envelope.
- **Dynamic voltage and frequency scaling** (`mlsys.dvfs`): Adjusting supply voltage and clock frequency at runtime to trade performance for power, since dynamic power grows superlinearly with voltage.
- **Dark silicon** (`mlsys.dark-silicon`): The fraction of a chip that cannot be powered at full frequency simultaneously within its power and thermal budget.
- **Power capping** (`mlsys.power-capping`): Enforcing a maximum power draw on a device or group by limiting clocks or activity, trading peak throughput for bounded power.
- **Duty cycling and sleep states** (`mlsys.duty-cycling`): Keeping a device in a low-power sleep state most of the time and waking it briefly to sense and infer, so average power is set by the active fraction and the cost of each wake-up.

Secondary concept connections (not additional homes):

- **Operating energy** (`mlsys.energy`)

Related MLSysBook passages (teaching depth not assessed):

- vol2: Sustainable AI — Operating energy (`vol2.sustainability.energy`)
- vol4: The Physical Body — Supply integrity (`vol4.plant.supply`)
- vol4: Workload Placement — Thermal throttling (`vol4.placement.throttling`)

#### Edge &amp; embedded hardware

Identity: `edge-embedded-hardware`

Hardware for on-device ML, from mobile SoCs to microcontrollers, where memory and energy per inference bind.

- **Activities:** Deploy and integrate.
- **Layers:** Compute and communication infrastructure; People and operating environment.
- **Concerns:** Latency, throughput, and resources; Energy and environmental impacts; Monetary cost.
- **Contexts:** Edge and device; Physical and control systems.

Includes: Microcontrollers and TinyML devices; Heterogeneous mobile SoCs and NPUs; Energy and memory budgets.

Excludes: On-device runtimes and OTA updates (On-device &amp; edge deployment); Robot-specific real-time platforms (Real-time &amp; mixed-criticality computing).

Design warrants: `mit-65940`, `book-hardware`.

Candidate home concepts:

- **Microcontroller-class device** (`mlsys.microcontroller`): A low-power processor with on-chip flash and kilobytes to a few megabytes of SRAM, typically without an operating system or virtual memory.
- **Heterogeneous mobile SoC** (`mlsys.heterogeneous-soc`): A system-on-chip combining CPU, GPU, DSP, and NPU blocks under shared memory and power limits, across which model operators are partitioned.
- **On-device neural accelerator** (`mlsys.edge-npu`): An accelerator integrated in mobile or embedded silicon for low-precision neural inference, usually supporting a restricted set of operators.
- **On-device memory budget** (`mlsys.on-device-memory-budget`): The flash and SRAM limits that bound a deployed model's stored weights and peak activation memory on embedded hardware.
- **Energy-harvesting intermittent computing** (`mlsys.intermittent-computing`): Running inference on devices powered by harvested energy that lose power unpredictably, which requires checkpointing progress to nonvolatile memory so computation resumes rather than restarts.
- **Sensor interconnect bandwidth** (`mlsys.sensor-interconnect`): The data rate of the links that carry sensor data into an embedded processor (MIPI CSI-2 for cameras, I2S for audio, SPI and I2C for low-rate sensors), which bounds the resolution, frame rate, and number of sensors a device can ingest.
- **Execute-in-place weights** (`mlsys.execute-in-place`): Reading immutable model weights from memory-mapped flash during inference without first copying them into SRAM, reducing working memory while exposing flash-access latency.
- **Always-on sensing cascade** (`mlsys.always-on-cascade`): Using a low-power sensing or inference stage to wake a more expensive stage only when needed, cutting continuous-monitoring energy at the risk of missed triggers and wake latency.
- **DMA cache maintenance** (`mlsys.dma-cache-maintenance`): Cleaning and invalidating processor caches around DMA transfers on hardware without coherent DMA, with buffers aligned to cache lines, so a device and the CPU do not read stale copies of a shared buffer; each maintenance step adds latency.

Secondary concept connections (not additional homes):

- **Quantization** (`mlsys.quantization`)
- **Execution placement** (`mlsys.placement`)
- **Energy per inference** (`mlsys.energy-per-inference`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Model Compression — Quantization (`vol1.compression.quantization`)
- vol1: System Paradigms — Power budget (`vol1.system.power`)

### Frameworks, compilers &amp; runtimes

How is a model program expressed, transformed, and executed efficiently on hardware?

The software layers between model code and hardware: programming models, operators, compilers, runtimes, and numerics.

#### ML frameworks &amp; programming models

Identity: `programming-models`

The abstractions with which ML programs are written: tensors, computation graphs, execution modes, and automatic differentiation.

- **Activities:** Learn and adapt; Deploy and integrate.
- **Layers:** Numerical software and runtime.
- **Concerns:** Latency, throughput, and resources; Reproducibility.
- **Contexts:** Deep learning.

Includes: Tensors and computation graphs; Eager vs. graph execution; tracing JIT; Automatic differentiation.

Excludes: Compiler optimization passes (ML compilers &amp; kernel optimization); Framework product catalogs.

Design warrants: `dlsys`.

Candidate home concepts:

- **Automatic differentiation** (`mlsys.autodiff`): Computing derivatives by composing local derivative rules through an execution graph.
- **Tensor** (`mlsys.tensor`): A multidimensional array carrying shape, element type, device, and layout metadata, the core data abstraction of ML frameworks.
- **Computation graph** (`mlsys.computation-graph`): A directed graph of tensor operations and their data dependencies representing a model program, used for differentiation, optimization, and scheduling.
- **Eager vs. graph execution** (`mlsys.eager-execution`): The choice between running each operation as the host program issues it and building a whole graph first for later optimization and execution.
- **Tracing and JIT capture** (`mlsys.tracing-jit`): Recording operations executed on example inputs to build a graph compiled just in time, with guards or retracing when shapes or control flow change.
- **Composable function transformations** (`mlsys.function-transformations`): Higher-order operations that map pure array functions to new functions, such as differentiation, vectorization, compilation, and sharding, and compose in any order.
- **Distributed task and actor runtime** (`mlsys.task-actor-runtime`): A runtime exposing stateless remote tasks and stateful actors as future-returning calls, scheduled across a cluster with a shared object store and lineage-based fault tolerance.
- **Single- vs. multi-controller execution** (`mlsys.single-vs-multi-controller`): Whether one coordinating process issues work to all devices or every host runs the same program independently, trading orchestration flexibility against dispatch scalability.
- **Embedded ML DSL** (`mlsys.embedded-dsl`): A domain-specific language hosted inside a general language such as Python, whose operations build a program representation that a separate compiler or runtime executes.
- **Differentiable programming** (`mlsys.differentiable-programming`): Writing general programs, including control flow, recursion, and data structures, that the language or framework differentiates end to end by automatic differentiation.
- **Large-model training framework** (`mlsys.llm-training-framework`): A library that composes data, tensor, pipeline, and sharded-state parallelism with memory optimizations behind one training configuration for models exceeding one device.
- **Python GIL and host-language concurrency** (`mlsys.python-gil`): The CPython global interpreter lock serializes Python bytecode across threads, so ML host code achieves concurrency through native kernels that release the lock, multiple processes, or asynchronous I/O.

Secondary concept connections (not additional homes):

- **Tensor layout** (`mlsys.tensor-layout`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: ML Frameworks — Reverse-mode autodiff (`vol1.framework.autodiff`)

#### Tensor operators &amp; memory layout

Identity: `tensor-operators`

Implementations of core tensor operations, such as matrix multiply, convolution, and attention, and the layouts that govern their efficiency.

- **Activities:** Learn and adapt; Deploy and integrate.
- **Layers:** Numerical software and runtime.
- **Concerns:** Latency, throughput, and resources.
- **Contexts:** Deep learning; Generative systems.

Includes: GEMM and convolution lowering; IO-aware attention kernels; Vendor kernel libraries.

Excludes: Compiler-generated kernels and autotuning (ML compilers &amp; kernel optimization); Architectural changes to attention (Long-context &amp; efficient sequence models).

Design warrants: `dlsys`, `flash`.

Candidate home concepts:

- **Tensor layout** (`mlsys.tensor-layout`): The mapping from tensor indices to storage locations, including strides and materialization.
- **Matrix multiplication** (`mlsys.matrix-multiply`): A dense linear-algebra operation whose tiling, shape, and precision affect execution cost.
- **IO-aware attention kernel** (`mlsys.io-aware-attention`): An exact attention implementation that tiles queries, keys, and values through on-chip memory to avoid materializing the full score matrix in off-chip memory.
- **Convolution lowering** (`mlsys.convolution-lowering`): Implementing convolution by rearranging input patches into matrices for GEMM, or by direct, FFT, or Winograd algorithms with different memory costs.
- **Vendor kernel library** (`mlsys.kernel-library`): Tuned implementations of common operators supplied for a hardware platform, which frameworks call rather than generating their own kernels.
- **Sparse tensor kernels** (`mlsys.sparse-tensor-kernels`): Kernels such as sparse-dense matrix multiply, sampled dense-dense multiply, and sparse convolution that compute over compressed nonzero structure, trading irregular access for skipped work.
- **Sparse and ragged tensor representation** (`mlsys.sparse-ragged-tensors`): Storing tensors with mostly zero entries or variable-length dimensions in compressed formats such as COO, CSR, or offset-indexed ragged layouts instead of padded dense arrays.

Secondary concept connections (not additional homes):

- **Kernel fusion** (`mlsys.fusion`)
- **Tiling** (`mlsys.tiling`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: ML Frameworks — Kernel fusion (`vol1.framework.fusion`)
- vol1: Hardware Acceleration — Tiling (`vol1.hardware.tiling`)
- vol1: Neural Computation — Matrix multiplication (`vol1.computation.matmul`)

#### Kernel &amp; GPU programming model

Identity: `gpu-programming`

How kernels are written for parallel accelerators: the thread hierarchy, memory access patterns, occupancy, asynchronous data movement, and programming the matrix units.

- **Activities:** Specify and design; Learn and adapt; Deploy and integrate.
- **Layers:** Numerical software and runtime; Compute and communication infrastructure.
- **Concerns:** Latency, throughput, and resources.
- **Contexts:** Deep learning.

Includes: Writing and tuning kernels in CUDA-class or tile-level languages such as Triton; Shared memory, synchronization, and latency hiding.

Excludes: Choosing kernels for operators (Tensor operators &amp; memory layout); Automatic kernel generation and fusion (ML compilers &amp; kernel optimization).

Design warrants: `dlsys`, `cuda-model`.

Candidate home concepts:

- **Thread hierarchy** (`mlsys.thread-hierarchy`): The decomposition of a kernel into grids of thread blocks and warps that share on-chip memory and synchronize at defined scopes.
- **Memory coalescing and bank conflicts** (`mlsys.memory-coalescing`): Access patterns that merge a warp's loads into few memory transactions, or that serialize on shared-memory banks.
- **Occupancy and latency hiding** (`mlsys.occupancy`): The ratio of resident to maximum warps, and how oversubscribing threads hides memory latency at the cost of registers and shared memory per thread.
- **Asynchronous data movement** (`mlsys.async-data-movement`): Hardware-assisted bulk copies into on-chip memory and producer-consumer warp roles that overlap loads with matrix-engine compute.
- **Matrix-engine programming** (`mlsys.matrix-engine-programming`): Issuing tile-level matrix-multiply-accumulate instructions with the layouts, fragment shapes, and precisions a tensor-core-class unit requires.
- **Register pressure** (`mlsys.register-pressure`): Demand for registers from simultaneously live kernel values, which can reduce resident threads or force values to spill into slower memory.

Secondary concept connections (not additional homes):

- **SIMT execution** (`mlsys.simt`)
- **Tiling** (`mlsys.tiling`)
- **IO-aware attention kernel** (`mlsys.io-aware-attention`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Hardware Acceleration — Tiling (`vol1.hardware.tiling`)

#### ML compilers &amp; kernel optimization

Identity: `ml-compilers`

Transforming model programs through intermediate representations into optimized kernels for target hardware.

- **Activities:** Learn and adapt; Deploy and integrate.
- **Layers:** Numerical software and runtime.
- **Concerns:** Latency, throughput, and resources.
- **Contexts:** Deep learning; Edge and device.

Includes: Graph and loop IRs; lowering; Fusion, tiling, autotuning, kernel DSLs; Static memory planning.

Excludes: Quantization as a compression technique (Model compression).

Design warrants: `dlsys`, `ut378`.

Candidate home concepts:

- **Graph compilation** (`mlsys.compilation`): Transforming a model or execution graph into an executable representation.
- **Kernel fusion** (`mlsys.fusion`): Combining compatible operations to reduce intermediate materialization or dispatch.
- **Tiling** (`mlsys.tiling`): Partitioning computation so working data can be reused within a chosen storage level.
- **ML compiler IR** (`mlsys.intermediate-representation`): Program representations at graph, tensor, and loop levels on which compilers apply lowering and optimization passes.
- **Kernel autotuning** (`mlsys.autotuning`): Searching kernel schedule parameters, such as tile sizes and loop orders, by measurement or learned cost models to find fast implementations for a target.
- **Kernel DSL** (`mlsys.kernel-dsl`): A domain-specific language for writing tiled tensor kernels above the level of vendor assembly, with the compiler handling memory and thread mapping.
- **Static memory planning** (`mlsys.memory-planning`): Compile-time assignment of tensor buffers to memory with lifetime-based reuse, reducing peak memory and runtime allocation.
- **Persistent kernel** (`mlsys.persistent-kernel`): A long-running kernel that stays resident on the device and executes many operators, or a whole forward pass, from an in-kernel work queue.
- **Tensor graph superoptimization** (`mlsys.graph-superoptimization`): Automatically generating and verifying semantics-preserving rewrites of tensor computation graphs or kernels, then searching the rewritten space under a cost model.
- **Dynamic shapes and shape specialization** (`mlsys.dynamic-shapes`): Handling inputs whose tensor dimensions vary across calls by specializing compiled code per shape, padding into buckets, or compiling with symbolic dimensions.
- **Rule-based graph rewriting** (`mlsys.rule-based-graph-rewriting`): Applying hand-written semantics-preserving transformations to a computation graph, such as constant folding, dead-code elimination, operator substitution, and algebraic simplification.
- **Software pipelining** (`mlsys.software-pipelining`): Scheduling independent operations from different loop iterations to overlap, such as loading the next tile while computing the current one, hiding latency at the cost of extra live state and buffering.

Secondary concept connections (not additional homes):

- **Host dispatch overhead** (`mlsys.dispatch`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: ML Frameworks — Kernel fusion (`vol1.framework.fusion`)
- vol1: ML Frameworks — Dispatch overhead (`vol1.framework.dispatch`)
- vol1: Hardware Acceleration — Tiling (`vol1.hardware.tiling`)

#### Runtime execution &amp; dispatch

Identity: `runtime-dispatch`

How a runtime launches and orders device work and manages device memory during execution.

- **Activities:** Learn and adapt; Deploy and integrate.
- **Layers:** Numerical software and runtime.
- **Concerns:** Latency, throughput, and resources; Reliability and recovery.
- **Contexts:** Deep learning; Interactive services.

Includes: Launch overhead and graph replay; Streams and asynchronous execution; Caching allocators.

Excludes: Request-level serving schedulers (Batching, admission &amp; autoscaling).

Design warrants: `dlsys`, `cuda-model`.

Candidate home concepts:

- **Host dispatch overhead** (`mlsys.dispatch`): Host-side work required to launch or orchestrate device operations.
- **Device streams** (`mlsys.device-streams`): Ordered queues of device work that the host enqueues asynchronously, with events expressing dependencies and allowing overlap across queues.
- **Caching memory allocator** (`mlsys.caching-allocator`): A runtime allocator that retains and reuses freed device memory blocks to avoid costly allocation calls, at the risk of fragmentation.
- **Captured graph replay** (`mlsys.graph-replay`): Recording a fixed sequence of kernel launches once and replaying it as a unit to remove per-launch host overhead.
- **CPU threading and NUMA placement** (`mlsys.cpu-threading-numa`): Configuring thread pools, core affinity, and memory placement across NUMA domains so CPU operators parallelize without oversubscription or remote-memory penalties.
- **Execution providers and operator fallback** (`mlsys.execution-providers`): Partitioning a model graph across runtime backends by operator support, running supported subgraphs on accelerators and falling back to a CPU backend otherwise.
- **Runtime memory fragmentation** (`mlsys.memory-fragmentation`): Free device memory split into blocks too small to satisfy new allocations, so a job runs out of memory while reserved memory exceeds what live tensors use.
- **Memory-mapped model weights** (`mlsys.mmap-weights`): Mapping a weight file into virtual memory so pages load on demand and read-only weights are shared across processes, cutting load time and duplicate memory at the cost of page faults.
- **OS memory pressure and paging** (`mlsys.os-memory-pressure`): How an operating system reclaims memory from a model process under pressure, by evicting file-backed pages, swapping or compressing memory, or killing the app, which shows up as page thrashing, latency spikes, or abrupt termination.
- **Managed-runtime interop and garbage collection** (`mlsys.managed-runtime-interop`): Crossing between a managed language runtime (Java, Kotlin, Swift, C\#) and native inference code, where buffer copies, object pinning, and garbage-collection pauses add latency and memory.

Secondary concept connections (not additional homes):

- **Communication overlap** (`mlsys.communication-overlap`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: ML Frameworks — Dispatch overhead (`vol1.framework.dispatch`)
- vol2: Collective Communication — Communication overlap (`vol2.collective.overlap`)

#### Data loading &amp; input pipelines

Identity: `data-loading`

Delivering training and inference inputs to accelerators fast enough: host-side loading and preprocessing, prefetching, and sampling minibatches from large or irregular data such as graphs.

- **Activities:** Learn and adapt; Deploy and integrate.
- **Layers:** Numerical software and runtime.
- **Concerns:** Latency, throughput, and resources; Reliability and recovery.
- **Contexts:** Deep learning; Interactive services.

Includes: Data-loader workers, decoding and augmentation on the host, pinned memory, and prefetching; Minibatch construction for irregular data, such as graph neighborhood sampling.

Excludes: Building datasets and transformation pipelines (Data pipelines &amp; storage formats); Storage systems and parallel file systems that hold the data (Storage &amp; data delivery).

Design warrants: `dlsys`, `cuda-model`.

Candidate home concepts:

- **Training input pipeline** (`mlsys.input-pipeline`): The read, decode, transform, shuffle, and prefetch stages that deliver batches to accelerators; when slower than compute, they cause data stalls.
- **Input prefetching** (`mlsys.input-prefetching`): Preparing upcoming batches in background workers and buffers while the accelerator processes the current batch, overlapping input delivery with computation.
- **Neighbor sampling** (`mlsys.neighbor-sampling`): Building GNN mini-batches from sampled subsets of each node's multi-hop neighborhood, bounding the exponential neighbor growth that makes full-neighborhood computation exceed memory.
- **Host-side data loading** (`mlsys.host-data-loader`): Reading, decoding, augmenting, and collating examples on host CPU worker processes and staging batches in pinned memory so that a single machine's accelerators are not left waiting for input.
- **Zero-copy sensor input via DMA** (`mlsys.zero-copy-dma`): Having sensors or peripherals write input directly into model input buffers by direct memory access, avoiding CPU copies between capture and inference.
- **Sequence packing** (`mlsys.sequence-packing`): Combining variable-length training examples into packed token buffers with explicit example boundaries and attention semantics to reduce padding work without unintended cross-example attention.

Related MLSysBook passages (teaching depth not assessed):

- vol1: Data Engineering — Input pipeline (`vol1.data.pipeline`)
- vol2: Data Storage — Input pipeline (`vol2.storage.input-pipeline`)
- vol2: Data Storage — Prefetching (`vol2.storage.prefetch`)

#### Numerical formats &amp; precision

Identity: `numerics`

Number formats used in ML computation and the techniques that keep training and inference accurate at reduced precision.

- **Activities:** Learn and adapt; Deploy and integrate.
- **Layers:** Numerical software and runtime; Model and algorithm.
- **Concerns:** Latency, throughput, and resources; Task and predictive quality.
- **Contexts:** Deep learning; Generative systems.

Includes: FP32/BF16/FP8/INT formats and block scaling; Mixed precision and loss scaling; Accumulation precision.

Excludes: Post-training quantization as compression (Model compression).

Design warrants: `dlsys`, `cs336`.

Candidate home concepts:

- **Numerical precision** (`mlsys.precision`): The representable values and arithmetic behavior of a numerical format.
- **Mixed-precision training** (`mlsys.mixed-precision`): Training with multiple numerical formats and explicit handling of their range and accumulation behavior.
- **Numerical stability** (`mlsys.numerical-stability`): Sensitivity of a computed result to finite-precision operations and intermediate representations.
- **Loss scaling** (`mlsys.loss-scaling`): Multiplying the loss before backpropagation so small gradients remain representable in reduced-precision formats, then unscaling before the optimizer step.
- **Block-scaled number formats** (`mlsys.block-scaled-format`): Low-precision formats in which small blocks of values share a scale factor, extending dynamic range for FP8, FP6, FP4, or INT8 elements.
- **Accumulation precision** (`mlsys.accumulation-precision`): The numerical format used to sum products in dot products and reductions, which may differ from input precision and governs rounding-error growth.
- **Numerical nondeterminism** (`mlsys.numerical-nondeterminism`): Run-to-run variation in floating-point results because non-associative reductions change order with atomics, kernel selection, parallel layout, batch composition, or hardware.
- **Batch invariance** (`mlsys.batch-invariance`): A kernel property that each request's output is bitwise independent of batch size and co-batched requests, required for deterministic serving and matching training to inference numerics.

Secondary concept connections (not additional homes):

- **Quantization** (`mlsys.quantization`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Model Compression — Weight precision (`vol1.compression.precision`)
- vol1: Model Compression — Quantization (`vol1.compression.quantization`)

### Parallelism &amp; distributed execution

How is one workload split across many devices, and what does coordination cost?

Partitioning ML work across devices and participants, the communication it requires, and the coordination that keeps it consistent.

#### Parallelism strategies

Identity: `parallelism`

Ways of partitioning a model's computation and state across devices, and how those dimensions are combined.

- **Activities:** Learn and adapt; Deploy and integrate.
- **Layers:** Numerical software and runtime; Compute and communication infrastructure.
- **Concerns:** Latency, throughput, and resources; Monetary cost.
- **Contexts:** Distributed systems; Generative systems; Recommendation and ranking.

Includes: Data, tensor, pipeline, sequence, expert parallelism; ZeRO/FSDP state sharding; Hybrid 3D layouts and pipeline bubbles.

Excludes: Collective algorithms (Collective communication); Job placement on the cluster (Cluster scheduling &amp; orchestration); Searching for and placing a parallel plan (Parallelization planning &amp; placement).

Design warrants: `zero`, `cs336`.

Candidate home concepts:

- **Data parallelism** (`mlsys.data-parallel`): Executing model replicas on different examples while coordinating learning state.
- **Tensor parallelism** (`mlsys.tensor-parallel`): Partitioning a layer's tensors and operations across devices so each computes a slice, reducing per-device state while adding communication along the partitioned dimensions, usually within a fast scale-up domain.
- **Pipeline parallelism** (`mlsys.pipeline-parallel`): Partitioning a computation into stages that process different microbatches concurrently.
- **Training-state sharding** (`mlsys.state-sharding`): Partitioning parameters, gradients, or optimizer state across workers.
- **Expert parallelism** (`mlsys.expert-parallel`): Placing different experts of a mixture-of-experts layer on different devices and exchanging routed tokens among them.
- **Sequence and context parallelism** (`mlsys.sequence-parallel`): Partitioning activations along the sequence dimension across devices, exchanging keys, values, or partial results so long sequences fit.
- **Hybrid parallelism** (`mlsys.hybrid-parallel`): Combining data, tensor, pipeline, and other parallelism dimensions, mapped onto the hardware hierarchy so the most communication-intensive dimension uses the fastest links.
- **Pipeline bubble** (`mlsys.pipeline-bubble`): Idle device time during pipeline fill and drain, whose fraction shrinks as the number of microbatches grows relative to pipeline stages.
- **Embedding-table sharding** (`mlsys.embedding-table-sharding`): Partitioning very large sparse embedding tables row-, column-, or table-wise across devices, usually model-parallel beside data-parallel dense layers.
- **Parameter server** (`mlsys.parameter-server`): An architecture in which workers pull and push model parameters from sharded server processes, common for sparse recommendation models and asynchronous training.
- **Synchronous vs. asynchronous training** (`mlsys.sync-async-training`): Choosing whether workers apply an update only after every gradient for a step is combined, or apply updates independently on possibly stale parameters.
- **Graph partitioning for GNN training** (`mlsys.gnn-graph-partitioning`): Splitting a large graph across workers to balance computation while minimizing cut edges, whose cross-partition neighbor features must be fetched remotely.
- **Pipeline schedule** (`mlsys.pipeline-schedule`): The order in which pipeline stages run the forward and backward passes of microbatches (fill-drain, one-forward-one-backward, interleaved), which sets the bubble fraction and how many microbatches' activations each stage holds.
- **Graph pipeline parallelism** (`mlsys.graph-pipeline-parallelism`): Pipeline parallelism that partitions a model whose layers form a branching graph into stages that follow the graph, so independent branches, such as separate modality encoders, run concurrently in parallel stages instead of being serialized into one chain.

Secondary concept connections (not additional homes):

- **Collective communication** (`mlsys.collectives`)
- **Training state** (`mlsys.training-state`)
- **Neighbor sampling** (`mlsys.neighbor-sampling`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Neural Computation — Training state (`vol1.computation.state`)
- vol2: Collective Communication — AllReduce (`vol2.collective.allreduce`)
- vol2: Distributed Training — Data parallelism (`vol2.training.data-parallelism`)
- vol2: Distributed Training — Tensor parallelism (`vol2.training.tensor-parallelism`)
- vol2: Distributed Training — Pipeline parallelism (`vol2.training.pipeline-parallelism`)
- vol2: Distributed Training — Training-state sharding (`vol2.training.state-sharding`)
- vol2: Distributed Training — Pipeline parallelism (`vol2.training.pipeline-parallelism`)
- vol2: Distributed Training — Pipeline parallelism (`vol2.training.pipeline-parallelism`)

#### Parallelization planning &amp; placement

Identity: `parallelism-planning`

Choosing and changing how a model is partitioned across devices: describing devices as a mesh, searching or propagating sharding decisions, placing experts and stages on heterogeneous hardware, and resharding at run time.

- **Activities:** Learn and adapt; Deploy and integrate.
- **Layers:** Numerical software and runtime; Compute and communication infrastructure.
- **Concerns:** Latency, throughput, and resources; Monetary cost.
- **Contexts:** Distributed systems; Generative systems; Recommendation and ranking.

Includes: Automatic parallelization and sharding propagation; Device meshes, expert placement, and dynamic resharding.

Excludes: The parallelism strategies themselves (Parallelism strategies).

Design warrants: `zero`, `cs336`.

Candidate home concepts:

- **Automatic parallelization** (`mlsys.auto-parallelization`): Compiler-derived multi-device partitioning of a single-device program, found by propagating sharding annotations or by searching parallelism strategies under a cost model.
- **Sharding annotation and propagation** (`mlsys.sharding-propagation`): Annotating a few tensors with layouts over a device mesh and letting the compiler infer the remaining layouts and insert required collectives, yielding one SPMD program.
- **Device mesh** (`mlsys.device-mesh`): A logical multidimensional arrangement of devices whose axes are assigned to parallelism dimensions, usually aligned so the busiest axis maps to the fastest interconnect.
- **Heterogeneous-accelerator training and serving** (`mlsys.heterogeneous-accelerator-execution`): Partitioning one training or serving job across accelerators of different types or generations, sizing stages and batch shares to each device's compute, memory, and links.
- **Dynamic re-parallelization** (`mlsys.dynamic-resharding`): Redistributing model weights among devices at runtime so successive phases, such as generation and training or prefill and decode, each run under a better parallel layout.
- **Expert replication and placement** (`mlsys.expert-placement`): Assigning mixture-of-experts experts to devices, and replicating frequently routed ones, to balance per-device token load under skewed routing.
- **Variable-length parallelism planning** (`mlsys.variable-length-parallelism`): Choosing, for each training step, sequence or context parallel degrees and how packed documents are sharded across ranks from that step's sequence lengths and attention patterns, so ranks receive balanced attention work.

Secondary concept connections (not additional homes):

- **Hybrid parallelism** (`mlsys.hybrid-parallel`)

Related MLSysBook passages (teaching depth not assessed):

No passage link is recorded. This is not evidence of absence from the books.

#### Collective communication

Identity: `collectives`

Group communication primitives and algorithms used by parallel ML, their cost models, and techniques to reduce or hide their cost.

- **Activities:** Learn and adapt; Deploy and integrate.
- **Layers:** Numerical software and runtime; Compute and communication infrastructure.
- **Concerns:** Latency, throughput, and resources.
- **Contexts:** Distributed systems; Recommendation and ranking.

Includes: All-reduce, all-gather, all-to-all; Ring and tree algorithms; alpha-beta cost model; Overlap and gradient compression.

Excludes: Physical fabric design (Network fabrics).

Design warrants: `cs336`, `ut378`.

Candidate home concepts:

- **Collective communication** (`mlsys.collectives`): Coordinated data exchange and reduction across a process group.
- **Communication overlap** (`mlsys.communication-overlap`): Scheduling communication concurrently with independent computation.
- **Stragglers** (`mlsys.stragglers`): Workers or tasks whose delayed completion holds back coordinated work.
- **Ring all-reduce** (`mlsys.ring-allreduce`): An all-reduce performed as reduce-scatter then all-gather around a ring, bandwidth-optimal for large messages but with latency linear in participants.
- **Tree and hierarchical all-reduce** (`mlsys.tree-allreduce`): All-reduce schedules over trees or hierarchical groups, with latency logarithmic in participants, suited to small messages and multi-level networks.
- **All-to-all exchange** (`mlsys.all-to-all`): A collective in which every participant sends distinct data to every other, as in expert routing; its throughput is sensitive to bisection bandwidth.
- **Alpha-beta communication model** (`mlsys.alpha-beta-model`): A cost model estimating message time as a fixed latency term plus message size times a per-byte cost, used to compare collective algorithms.
- **Gradient compression** (`mlsys.gradient-compression`): Reducing gradient communication through quantization, sparsification, or low-rank approximation, often with error feedback to preserve convergence.
- **Local SGD** (`mlsys.local-sgd`): Data-parallel training in which workers take several local optimizer steps between parameter synchronizations, cutting communication frequency for low-bandwidth or geo-distributed settings.
- **Distributed job launch and process groups** (`mlsys.process-group-rendezvous`): Starting the processes of a distributed job, having them discover each other through a rendezvous, assigning ranks, and forming the process groups and communicators that collectives run over.
- **Device-initiated communication** (`mlsys.device-initiated-communication`): Communication in which kernels on the accelerator issue and synchronize inter-device transfers themselves, without a host call per exchange, allowing fine-grained fusion of compute and communication at the cost of explicit ordering and completion handling.
- **Collective algorithm selection** (`mlsys.collective-algorithm-selection`): Choosing, per call, among collective algorithms (ring, tree, switch-offloaded) and wire protocols by message size, participant count, and link hierarchy, to minimize latency for small messages and maximize bandwidth for large ones.

Secondary concept connections (not additional homes):

- **Network topology** (`mlsys.network-topology`)
- **Bisection bandwidth** (`mlsys.bisection-bandwidth`)
- **In-network reduction** (`mlsys.in-network-reduction`)

Related MLSysBook passages (teaching depth not assessed):

- vol2: Collective Communication — AllReduce (`vol2.collective.allreduce`)
- vol2: Collective Communication — Communication overlap (`vol2.collective.overlap`)
- vol2: Network Fabrics — Bisection bandwidth (`vol2.network.bisection-bandwidth`)
- vol2: Network Fabrics — Message startup latency (`vol2.network.startup-latency`)
- vol2: Network Fabrics — Bisection bandwidth (`vol2.network.bisection-bandwidth`)
- vol2: Collective Communication — Gradient compression (`vol2.collective.gradient-compression`)

#### Coordination &amp; consensus

Identity: `coordination`

Mechanisms by which distributed components agree on authoritative control state and tolerate stale or failed participants.

- **Activities:** Learn and adapt; Operate and change.
- **Layers:** Numerical software and runtime; Compute and communication infrastructure.
- **Concerns:** Reliability and recovery.
- **Contexts:** Distributed systems; Agentic systems.

Includes: Consensus and leader election; Leases and fencing; Bounded-staleness consistency.

Excludes: Checkpoint and recovery protocols (Reliability &amp; fault tolerance); Retry and compensation semantics (Failure models, retries &amp; degradation).

Design warrants: `sre`.

Candidate home concepts:

- **Replicated-log consensus** (`mlsys.consensus`): Agreement on an ordered state history under specified timing and failure assumptions.
- **Fencing** (`mlsys.fencing`): Rejecting effects from an actor whose epoch or authority is stale.
- **Lease** (`mlsys.lease`): A time-bounded grant of authority or cached state that expires unless renewed, letting a system recover from failed holders without explicit release.
- **Leader election** (`mlsys.leader-election`): Selecting a single coordinator among replicas, typically through consensus and leases, so control decisions have one authoritative source.
- **Bounded staleness** (`mlsys.bounded-staleness`): A consistency model in which workers may compute on parameters up to a fixed number of updates old, trading freshness for less waiting.
- **Consistency-availability trade-off** (`mlsys.cap-tradeoff`): Under a network partition, a replicated system must choose between answering with possibly stale state and refusing requests until replicas agree.

Secondary concept connections (not additional homes):

- **Recovery checkpoint** (`mlsys.checkpoint`)
- **Consistent checkpoint cut** (`mlsys.consistent-cut`)
- **Idempotency** (`mlsys.idempotency`)

Related MLSysBook passages (teaching depth not assessed):

- vol2: Data Storage — Durable checkpoint (`vol2.storage.durable-checkpoint`)
- vol3: Durable Execution — Recovery checkpoint (`vol3.checkpoint`)
- vol2: Fault Tolerance — Consistent checkpoint cut (`vol2.reliability.consistent-cut`)
- vol3: Tool Calling — Operation identity (`vol3.operation`)
- vol2: Fault Tolerance — Recovery (`vol2.reliability.recovery`)
- vol4: The Nervous System — Chunk lease (`vol4.authority.chunk-lease`)

#### Federated &amp; decentralized learning

Identity: `federated-learning`

Training across devices or organizations that keep their data local, coordinated through a shared model.

- **Activities:** Learn and adapt; Acquire and prepare data.
- **Layers:** Model and algorithm; Compute and communication infrastructure.
- **Concerns:** Privacy; Latency, throughput, and resources; Fairness.
- **Contexts:** Federated settings; Edge and device.

Includes: Federated averaging; Client heterogeneity and selection; Cross-device vs. cross-silo.

Excludes: Secure aggregation and differential privacy (Security &amp; privacy).

Design warrants: `fl-open`.

Candidate home concepts:

- **Federated learning** (`mlsys.federated-learning`): Coordinating model learning across separately controlled data holders.
- **Federated averaging** (`mlsys.federated-averaging`): A federated algorithm in which sampled clients run several local optimization steps and a server averages their model updates, weighted by data size.
- **Statistical heterogeneity** (`mlsys.non-iid-data`): Differences in data distribution and quantity across federated clients, which slow convergence and can bias a shared model.
- **Client selection** (`mlsys.client-sampling`): Choosing which available devices participate in each federated round, subject to availability, connectivity, and power constraints.
- **Cross-device vs. cross-silo federation** (`mlsys.federation-setting`): The distinction between federations of many intermittently available devices and federations of a few reliable organizational data holders.
- **Decentralized gossip averaging** (`mlsys.gossip-averaging`): Training in which each worker averages its model with a few neighbors instead of a central server or global all-reduce, so updates diffuse through the communication graph.

Secondary concept connections (not additional homes):

- **Differential privacy** (`mlsys.differential-privacy`)
- **Privacy requirements** (`mlsys.privacy`)

Related MLSysBook passages (teaching depth not assessed):

- vol2: Edge Intelligence — Federated averaging (`vol2.federated.fedavg`)
- vol2: Security &amp; Privacy — Differential privacy (`vol2.privacy.dp`)
- vol2: Edge Intelligence — Federated averaging (`vol2.federated.fedavg`)
- vol2: Edge Intelligence — Client participation (`vol2.federated.participation`)

### Datacenter infrastructure &amp; fleet

How are AI datacenters built, connected, fed, and shared across many jobs?

Facilities, networks, storage, and cluster management that turn many accelerators into a shared ML fleet.

#### Datacenter power, cooling &amp; facilities

Identity: `datacenter-facilities`

Engineering of the buildings that house ML fleets: power provisioning, cooling, density, and facility efficiency.

- **Activities:** Deploy and integrate; Operate and change.
- **Layers:** Compute and communication infrastructure; People and operating environment.
- **Concerns:** Energy and environmental impacts; Monetary cost; Reliability and recovery.
- **Contexts:** Distributed systems.

Includes: Facility power and oversubscription; Air and liquid cooling; rack density; PUE.

Excludes: Carbon and water accounting (Sustainability &amp; cost); Chip-level power limits (Chip power &amp; thermal limits).

Design warrants: `dc-computer`, `green`.

Candidate home concepts:

- **Power usage effectiveness** (`mlsys.pue`): Total facility energy divided by IT equipment energy, measuring overhead from cooling, power conversion, and other non-compute loads.
- **Liquid cooling** (`mlsys.liquid-cooling`): Removing heat with liquid through cold plates or immersion, enabling rack power densities beyond the practical limits of air cooling.
- **Rack power density** (`mlsys.rack-power-density`): Electrical power drawn per rack, which determines cooling technology, floor layout, and how many accelerators fit within a facility's power envelope.
- **Power oversubscription** (`mlsys.power-oversubscription`): Deploying more equipment than the sum of nameplate power a feed supports, relying on non-coincident peaks and power capping to stay within limits.
- **Warehouse-scale computer** (`mlsys.warehouse-scale-computer`): A datacenter treated as a single computer whose power, cooling, network, and software are co-designed for a common workload.
- **Synchronous training power swings** (`mlsys.training-power-swings`): Rapid cluster-wide swings in power draw as synchronized accelerators alternate between compute and communication or stall together at checkpoints and failures.
- **UPS and battery energy storage** (`mlsys.facility-energy-storage`): Facility batteries that carry load through outages until generators start and can also buffer fast load swings or shave demand peaks.
- **Grid interconnection** (`mlsys.grid-interconnection`): The utility connection that fixes how much power a site can draw; its capacity and multi-year lead times increasingly decide where AI datacenters are built.
- **On-site power generation** (`mlsys.on-site-generation`): Generators, turbines, fuel cells, or renewables at the datacenter that provide backup power or supply load alongside or instead of the grid.

Secondary concept connections (not additional homes):

- **Operating energy** (`mlsys.energy`)
- **Lifecycle carbon** (`mlsys.carbon`)
- **Power delivery and integrity** (`mlsys.power-delivery`)
- **Thermal behavior and throttling** (`mlsys.thermal-dynamics`)

Related MLSysBook passages (teaching depth not assessed):

- vol2: Sustainable AI — Operating energy (`vol2.sustainability.energy`)
- vol1: Responsible Engineering — Operational carbon (`vol1.responsibility.carbon`)
- vol2: Sustainable AI — Embodied carbon (`vol2.sustainability.embodied-carbon`)
- vol2: Sustainable AI — Operational carbon (`vol2.sustainability.operational-carbon`)
- vol2: Compute Infrastructure — Rack power envelope (`vol2.hardware.power-envelope`)
- vol2: Sustainable AI — Operational carbon (`vol2.sustainability.operational-carbon`)
- vol4: The Physical Body — Supply integrity (`vol4.plant.supply`)
- vol4: Workload Placement — Thermal throttling (`vol4.placement.throttling`)

#### Network fabrics

Identity: `network-fabrics`

Datacenter networks that connect accelerator nodes: transports, topologies, flow and congestion control, and in-network compute.

- **Activities:** Learn and adapt; Deploy and integrate; Operate and change.
- **Layers:** Compute and communication infrastructure.
- **Concerns:** Latency, throughput, and resources; Reliability and recovery; Monetary cost.
- **Contexts:** Distributed systems.

Includes: RDMA over InfiniBand and RoCE; lossless flow control; Fat-tree, rail-optimized, torus topologies; Congestion control; in-network reduction.

Excludes: Intra-node links (Host-device &amp; scale-up interconnect); Collective algorithms (Collective communication).

Design warrants: `dc-computer`, `cse599o`.

Candidate home concepts:

- **Network topology** (`mlsys.network-topology`): The connectivity and capacity structure that constrains communication paths.
- **Remote direct memory access** (`mlsys.rdma`): Network transfers that read or write remote memory directly through the network interface without remote CPU involvement, giving low latency and high throughput.
- **Lossless network fabric** (`mlsys.lossless-fabric`): A network that avoids packet drops through credit-based or priority flow control, as RDMA transports expect, at the risk of head-of-line blocking.
- **Datacenter congestion control** (`mlsys.congestion-control`): End-host and switch mechanisms, such as ECN-driven rate control, that limit queue buildup from incast and synchronized collective traffic.
- **Fat-tree (Clos) topology** (`mlsys.fat-tree`): A multistage switch topology providing many equal-cost paths and, without oversubscription, full bisection bandwidth between endpoints.
- **Rail-optimized topology** (`mlsys.rail-optimized`): A fabric design connecting the same-index accelerator of every node to a common leaf switch, keeping most collective traffic within one switch hop.
- **Bisection bandwidth** (`mlsys.bisection-bandwidth`): The minimum total bandwidth across any cut dividing a network into two equal halves, bounding throughput of all-to-all and global exchanges.
- **In-network reduction** (`mlsys.in-network-reduction`): Performing collective reductions inside network switches so data crosses each link fewer times, lowering all-reduce traffic and latency.
- **Adaptive routing** (`mlsys.adaptive-routing`): Choosing network paths per packet or flowlet from current congestion, avoiding the hash collisions static ECMP suffers with few large, synchronized training flows.
- **Incast congestion** (`mlsys.incast`): Congestion that occurs when many senders transmit to one receiver at once, overflowing its switch port buffer and causing queueing, drops, or pause propagation.
- **Network propagation delay** (`mlsys.network-propagation-delay`): The minimum latency set by the distance a signal travels at the speed of light in fiber, which bounds round trips between regions regardless of bandwidth.
- **Optical circuit switching** (`mlsys.optical-circuit-switching`): Reconfiguring optical connections between endpoints to match communication demand, trading switching delay and topology coordination against delivered bandwidth and network power.
- **Co-packaged optics** (`mlsys.co-packaged-optics`): Placing optical transceivers inside the switch or accelerator package beside the silicon, so electrical paths shrink and energy per bit falls, at the cost of field serviceability and added package heat.
- **Forward error correction** (`mlsys.forward-error-correction`): Adding redundant bits to transmitted data so the receiver corrects supported error patterns without retransmission, at the cost of coding overhead and decoding delay.
- **Network traffic classes and QoS** (`mlsys.traffic-classes`): Classifying traffic on a shared network into separately scheduled queues with configured priorities or bandwidth shares, limiting interference between collective, storage, and control traffic.

Secondary concept connections (not additional homes):

- **Collective communication** (`mlsys.collectives`)
- **All-to-all exchange** (`mlsys.all-to-all`)

Related MLSysBook passages (teaching depth not assessed):

- vol2: Collective Communication — AllReduce (`vol2.collective.allreduce`)
- vol2: Network Fabrics — Bisection bandwidth (`vol2.network.bisection-bandwidth`)
- vol2: Network Fabrics — Bisection bandwidth (`vol2.network.bisection-bandwidth`)
- vol2: Network Fabrics — Network congestion (`vol2.network.congestion`)
- vol2: Network Fabrics — Network congestion (`vol2.network.congestion`)

#### Storage &amp; data delivery

Identity: `storage-systems`

Storage systems that hold training data, checkpoints, and artifacts at scale and deliver them to accelerators fast enough.

- **Activities:** Acquire and prepare data; Learn and adapt; Operate and change.
- **Layers:** Compute and communication infrastructure.
- **Concerns:** Latency, throughput, and resources; Monetary cost; Reliability and recovery.
- **Contexts:** Distributed systems; Batch and offline.

Includes: Parallel file systems and object stores; Input pipelines and data stalls; Checkpoint I/O and storage tiering.

Excludes: Dataset formats and schema (Data pipelines &amp; storage formats); Checkpoint frequency and recovery semantics (Reliability &amp; fault tolerance); Host-side loading and prefetching of batches (Data loading &amp; input pipelines).

Design warrants: `dc-computer`, `cse599o`.

Candidate home concepts:

- **Parallel file system** (`mlsys.parallel-file-system`): A distributed file system that stripes data across many storage servers to deliver aggregate bandwidth to many concurrent clients.
- **Object storage** (`mlsys.object-storage`): Storage of immutable named objects behind an HTTP-style interface, offering scalable capacity and throughput with higher per-request latency than file systems.
- **Checkpoint I/O** (`mlsys.checkpoint-io`): Writing and reading large training-state snapshots, where storage bandwidth, asynchronous persistence, and sharded layout determine checkpoint cost.
- **Storage tiering and caching** (`mlsys.storage-tiering`): Placing data across tiers, from memory and local NVMe caches to shared and archival storage, according to access frequency and cost.
- **Erasure coding** (`mlsys.erasure-coding`): Encoding stored data into data and parity fragments so it can be reconstructed after a bounded number of fragment losses, trading storage overhead against encoding and recovery work.

Secondary concept connections (not additional homes):

- **Recovery checkpoint** (`mlsys.checkpoint`)
- **Dataset storage layout** (`mlsys.dataset-storage-layout`)
- **Data ingestion** (`mlsys.data-ingestion`)
- **Training input pipeline** (`mlsys.input-pipeline`)

Related MLSysBook passages (teaching depth not assessed):

- vol2: Data Storage — Durable checkpoint (`vol2.storage.durable-checkpoint`)
- vol3: Durable Execution — Recovery checkpoint (`vol3.checkpoint`)
- vol1: Data Engineering — Input pipeline (`vol1.data.pipeline`)
- vol2: Data Storage — Input pipeline (`vol2.storage.input-pipeline`)

#### Cluster scheduling &amp; orchestration

Identity: `cluster-scheduling`

Allocating a shared cluster's accelerators to jobs over time: placement, gang allocation, fairness, preemption, and elasticity.

- **Activities:** Learn and adapt; Deploy and integrate; Operate and change.
- **Layers:** Compute and communication infrastructure.
- **Concerns:** Latency, throughput, and resources; Monetary cost; Fairness.
- **Contexts:** Distributed systems; Batch and offline.

Includes: Gang scheduling and backfill; Fair sharing, preemption, elastic jobs; Topology-aware placement and bin packing.

Excludes: Request-level serving autoscaling (Batching, admission &amp; autoscaling); Workflow DAG orchestration (ML pipelines &amp; reproducibility); Packaging code and dependencies as images (ML pipelines &amp; reproducibility).

Design warrants: `dc-computer`, `sre`.

Candidate home concepts:

- **Gang scheduling** (`mlsys.gang-scheduling`): Allocating all resources a parallel job needs at once, or none, so tightly coupled workers start together and partial allocations cannot deadlock.
- **Job preemption** (`mlsys.job-preemption`): Suspending or evicting a running job to reclaim its resources for higher-priority work, at a cost set by checkpointing and lost progress.
- **Dominant resource fairness** (`mlsys.dominant-resource-fairness`): A multi-resource fair-sharing policy that equalizes each user's largest share across resource types such as accelerators, CPUs, and memory.
- **Elastic training** (`mlsys.elastic-training`): Training that continues while the number of workers changes, resharding state and adjusting batch size or learning rate as resources join or leave.
- **Topology-aware placement** (`mlsys.topology-aware-placement`): Assigning a job's workers to devices and nodes so that heavily communicating groups share fast links or switches.
- **Bin packing** (`mlsys.bin-packing`): Placing jobs onto nodes to minimize fragmentation of resources such as accelerators, trading packing density against interference and locality.
- **ML-aware cluster scheduling** (`mlsys.ml-aware-scheduling`): Allocating cluster accelerators using ML job properties, such as scaling curves, iteration time, and parallelism choice, to optimize objectives like job completion time or goodput.
- **Cluster manager** (`mlsys.cluster-manager`): The control plane that admits jobs, places their tasks on machines, enforces quotas and priorities, and reconciles running state toward declared intent across a cluster.
- **Backfill scheduling** (`mlsys.backfill-scheduling`): Running lower-priority jobs in idle resource-time gaps without delaying the reserved start of a higher-priority job, which raises cluster utilization.

Secondary concept connections (not additional homes):

- **Admission control** (`mlsys.admission`)
- **Resource utilization** (`mlsys.utilization`)
- **Recovery checkpoint** (`mlsys.checkpoint`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Model Serving — Utilization (`vol1.serving.utilization`)
- vol2: Data Storage — Durable checkpoint (`vol2.storage.durable-checkpoint`)
- vol3: Durable Execution — Recovery checkpoint (`vol3.checkpoint`)
- vol2: Fleet Orchestration — Gang scheduling (`vol2.scheduling.gang`)
- vol2: Fleet Orchestration — Topology-aware placement (`vol2.scheduling.placement`)
- vol2: Fleet Orchestration — Resource fairness (`vol2.scheduling.fairness`)

#### Capacity planning &amp; multi-tenancy

Identity: `capacity-provisioning`

Provisioning fleet capacity and sharing it among tenants through quotas, accelerator sharing, and interference control.

- **Activities:** Deploy and integrate; Operate and change.
- **Layers:** Compute and communication infrastructure.
- **Concerns:** Monetary cost; Latency, throughput, and resources; Security.
- **Contexts:** Distributed systems; Interactive services.

Includes: Capacity planning and quotas; Accelerator sharing and partitioning; Spot capacity and interference.

Excludes: Security isolation guarantees (Access control &amp; isolation); Unit-cost economics (Sustainability &amp; cost).

Design warrants: `sre`, `dc-computer`.

Candidate home concepts:

- **Capacity planning** (`mlsys.capacity-planning`): Provisioning resources for a workload distribution and service objectives.
- **Multi-tenancy** (`mlsys.multi-tenancy`): Sharing infrastructure among teams, jobs, or customers under policies for allocation, isolation, priority, and accounting.
- **Resource quota** (`mlsys.resource-quota`): A limit on the resources a tenant may hold or request, sometimes with guaranteed minimums and borrowing of idle capacity.
- **Accelerator sharing and partitioning** (`mlsys.gpu-partitioning`): Running several workloads on one accelerator by time-slicing, concurrent contexts, or hardware partitions with dedicated memory and compute slices.
- **Spot and preemptible capacity** (`mlsys.spot-capacity`): Discounted compute that the provider may reclaim at short notice, usable by workloads that tolerate interruption.
- **Performance interference** (`mlsys.performance-interference`): Slowdown of one workload caused by co-located workloads contending for shared caches, memory bandwidth, network, or power.
- **Cloud service models** (`mlsys.cloud-service-model`): Divisions of responsibility between cloud provider and customer, from rented virtual infrastructure to managed platforms to hosted model APIs.
- **Fine-grained accelerator sharing** (`mlsys.fine-grained-accelerator-sharing`): Scheduling co-located workloads on one accelerator in small units of a kernel, so latency-sensitive work can preempt background work quickly and interference stays bounded.

Secondary concept connections (not additional homes):

- **Execution isolation** (`mlsys.isolation`)
- **Autoscaling** (`mlsys.autoscaling`)
- **Cost per useful outcome** (`mlsys.unit-cost`)

Related MLSysBook passages (teaching depth not assessed):

- vol3: Agent Sandboxes — Isolation envelope (`vol3.isolation`)
- vol4: Workload Placement — Resource isolation (`vol4.placement.isolation`)
- vol3: Agent Economics — Cost per accepted task (`vol3.cost`)
- vol1: Model Serving — Serving capacity (`vol1.serving.capacity`)

## Lifecycle

What do engineers do to build, ship, and keep the system working?

### Data engineering

How is data acquired, labeled, curated, validated, stored, and kept trustworthy?

Acquisition and labeling, corpus curation, validation and features, pipelines and storage formats, lineage, and temporal correctness of training and serving data.

#### Data collection &amp; labeling

Identity: `data-collection`

Obtaining observations and supervisory targets whose sampling, coverage, and label quality are known and fit the task.

- **Activities:** Specify and design; Acquire and prepare data.
- **Layers:** Application and workflow; People and operating environment.
- **Concerns:** Task and predictive quality; Fairness; Privacy; Monetary cost.
- **Contexts:** Classical ML; Deep learning; Physical and control systems.

Includes: Sampling design, human and programmatic labeling, labeling budgets.; Annotation quality and agreement..

Excludes: Sensor design and calibration (Sensing, perception &amp; fusion); Corpus filtering and mixtures (Data selection &amp; curation).

Design warrants: `se4ml`, `dmls`.

Candidate home concepts:

- **Data acquisition and sampling** (`mlsys.data-acquisition`): The observation and selection process that determines which examples and populations enter a dataset.
- **Label quality** (`mlsys.labels`): The correctness, consistency, and coverage of recorded supervision relative to the intended task, which bound the training signal and the trustworthiness of evaluation.
- **Weak supervision** (`mlsys.weak-supervision`): Producing training labels from noisy programmatic sources such as heuristics, rules, or distant supervision, then modeling and combining their errors.
- **Active learning** (`mlsys.active-learning`): Selecting which unlabeled examples to annotate next according to their expected value to the model.
- **Inter-annotator agreement** (`mlsys.annotator-agreement`): The degree to which independent annotators assign the same label to the same item, used to measure label reliability and task ambiguity.
- **Class imbalance** (`mlsys.class-imbalance`): A skewed label distribution in which some classes are rare, so aggregate accuracy misleads and training favors the majority classes.
- **Class rebalancing** (`mlsys.rebalancing`): Changing per-class sampling rates, synthesizing minority examples, or weighting the loss so rare classes contribute adequately to training.
- **Model-assisted labeling** (`mlsys.model-assisted-labeling`): Using a model's predictions as draft labels that annotators accept or correct, which raises labeling throughput but can anchor annotators to the model's errors.

Secondary concept connections (not additional homes):

- **Sensor calibration** (`mlsys.sensing`)
- **Privacy requirements** (`mlsys.privacy`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Data Engineering — Labels (`vol1.data.labels`)
- vol1: Data Engineering — Labels (`vol1.data.labels`)

#### Data selection &amp; curation

Identity: `data-curation`

Choosing, filtering, deduplicating, mixing, augmenting, and synthesizing training examples to maximize learning value per unit of compute and labeling cost.

- **Activities:** Acquire and prepare data; Learn and adapt.
- **Layers:** Model and algorithm; Application and workflow.
- **Concerns:** Task and predictive quality; Monetary cost; Fairness.
- **Contexts:** Deep learning; Generative systems; Agentic systems; Multimodal systems.

Includes: Web-scale corpus filtering, deduplication, and domain mixtures.; Augmentation and synthetic or model-generated data.; Trajectory and demonstration curation for agents and robots..

Excludes: Labeling processes (Data collection &amp; labeling); Teacher-student compression (Model compression).

Design warrants: `cs336`, `dmls`.

Candidate home concepts:

- **Data selection** (`mlsys.data-selection`): Allocating learning effort to examples according to value, coverage, and acquisition cost.
- **Dataset deduplication** (`mlsys.deduplication`): Identifying and managing repeated or near-duplicate examples.
- **Quality filtering** (`mlsys.quality-filtering`): Removing or down-weighting training examples judged low quality by heuristics, classifiers, or model scores.
- **Data mixture** (`mlsys.data-mixture`): The proportions in which source domains or datasets are sampled during training.
- **Synthetic data** (`mlsys.synthetic-data`): Training or evaluation examples generated by simulators, programs, or models rather than collected from the target environment.
- **Data augmentation** (`mlsys.data-augmentation`): Creating additional training examples by applying label-preserving transformations to existing ones.
- **Data valuation** (`mlsys.data-valuation`): Estimating each training example's or data source's contribution to model quality, for example through influence or Shapley-style attribution, to guide selection, pricing, or removal.
- **Model collapse** (`mlsys.model-collapse`): Loss of coverage of the original data distribution across generations trained on model-generated data, making provenance and real-versus-synthetic mixture policies part of quality control.
- **Data curriculum and annealing** (`mlsys.data-curriculum`): Changing the training data distribution over a run, such as raising the share of high-quality or long-sequence data during the final learning-rate decay, to improve final capability per unit of compute.

Secondary concept connections (not additional homes):

- **Empirical scaling laws** (`mlsys.scaling-laws`)
- **Knowledge distillation** (`mlsys.distillation`)
- **Benchmark contamination** (`mlsys.benchmark-contamination`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Data Selection — Training subset (`vol1.selection.subset`)
- vol2: Introduction — Empirical scaling laws (`vol2.fleet.scaling-laws`)
- vol1: Data Selection — Selection overhead (`vol1.selection.scoring`)

#### Data validation &amp; feature engineering

Identity: `data-quality`

Checking that recorded data is fit for use and transforming it into model inputs whose semantics and availability match training and serving.

- **Activities:** Acquire and prepare data; Learn and adapt; Deploy and integrate.
- **Layers:** Application and workflow; Model and algorithm; Compute and communication infrastructure.
- **Concerns:** Task and predictive quality; Reliability and recovery.
- **Contexts:** Classical ML; Batch and offline; Streaming systems; Recommendation and ranking.

Includes: Schema and distribution checks before training and serving.; Feature construction and shared feature stores..

Excludes: Learned representations inside the model (Models &amp; architectures); Production drift monitoring (Monitoring &amp; observability).

Design warrants: `ml-test-score`, `rules`.

Candidate home concepts:

- **Data quality** (`mlsys.data-quality`): Fitness of recorded examples for the intended task, including validity, coverage, and error.
- **Feature engineering** (`mlsys.features`): Constructing model inputs whose semantics and availability match the task.
- **Missing data** (`mlsys.missing-data`): Absent observations whose pattern and treatment can change learned behavior.
- **Data validation** (`mlsys.data-validation`): Automated checks of incoming data against expected schema, value ranges, and distributions before it is used for training or serving.
- **Feature store** (`mlsys.feature-store`): A system that defines, computes, stores, and serves features consistently for offline training and online inference.
- **Data cleaning** (`mlsys.data-cleaning`): Detecting and correcting erroneous, inconsistent, duplicated, or outlying records with rules, statistics, or learned repair models before the data is used for training.
- **Feature hashing** (`mlsys.feature-hashing`): Mapping high-cardinality or open-vocabulary categorical values into a fixed number of buckets with a hash function, bounding table size at the cost of collisions.
- **Feature cross** (`mlsys.feature-cross`): A synthetic feature formed from the conjunction of two or more categorical features so a model, especially a linear one, can learn their interaction.
- **Exploratory data analysis** (`mlsys.exploratory-data-analysis`): Summarizing and plotting distributions, correlations, and anomalies in a dataset before modeling to inform cleaning, feature design, and validation rules.
- **Data cascade** (`mlsys.data-cascade`): Compounding downstream harm from upstream data problems, such as poor collection or labeling, that surface late, often after deployment, and are costly to trace.

Secondary concept connections (not additional homes):

- **Data contract** (`mlsys.data-contract`)
- **Training-serving skew** (`mlsys.train-serve-skew`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Data Engineering — Training-serving parity (`vol1.data.parity`)
- vol1: ML Lifecycle — Stage interface contract (`vol1.lifecycle.contract`)

#### Data pipelines &amp; storage formats

Identity: `data-pipelines`

Engineering ingestion, batch and stream transformation, event-time semantics, storage formats, and schemas for training and serving data.

- **Activities:** Acquire and prepare data; Learn and adapt; Operate and change.
- **Layers:** Compute and communication infrastructure; Numerical software and runtime.
- **Concerns:** Latency, throughput, and resources; Reliability and recovery; Monetary cost.
- **Contexts:** Batch and offline; Streaming systems; Distributed systems.

Includes: ETL/ELT, batch and stream processing, columnar formats.; Data loaders that keep accelerators fed..

Excludes: Tensor layout in working memory (Tensor operators &amp; memory layout); Delivering batches to accelerators (Data loading &amp; input pipelines).

Design warrants: `parquet-layout`, `beam-io`, `iceberg-schema`.

Candidate home concepts:

- **Dataset storage layout** (`mlsys.dataset-storage-layout`): Organization of stored dataset records, columns, partitions, and metadata that determines which data an access reads or decodes.
- **Data ingestion** (`mlsys.data-ingestion`): Reading externally produced records into a processing or learning system under declared delivery, ordering, and timing assumptions.
- **Schema evolution** (`mlsys.schema-evolution`): Changing a data schema while explicitly managing how existing data and producers or consumers remain interpretable and compatible.
- **Stream processing** (`mlsys.stream-processing`): Continuously transforming unbounded event streams with low latency and incrementally maintained state.
- **Feature backfill** (`mlsys.feature-backfill`): Recomputing a feature's historical values under a new or corrected definition so training data matches what serving will produce.
- **Distributed batch data processing** (`mlsys.batch-data-processing`): Transforming bounded datasets with dataflow engines that partition records across workers, shuffle between stages, and re-execute failed tasks, for large-scale preprocessing and feature generation.
- **Data warehouse and lakehouse** (`mlsys.lakehouse`): Analytical stores of governed tables queried with SQL; a lakehouse adds transactional table formats over open files in object storage.
- **Event-time windowing and watermarks** (`mlsys.event-time-windowing`): Grouping stream events into windows by when they occurred, using watermarks to estimate completeness and explicit policies for events arriving after a window closes.
- **Processing delivery semantics** (`mlsys.delivery-semantics`): The guarantee a data pipeline gives about how often each record takes effect (at most once, at least once, or effectively once) under retries, replays, and failures, and the idempotent or transactional sinks that achieve it.
- **Sharded streaming dataset format** (`mlsys.streaming-dataset-format`): Organizing training examples into sequentially readable shard files so workers can stream records from remote storage with few small-object reads and bounded local staging.

Secondary concept connections (not additional homes):

- **Workflow orchestration** (`mlsys.workflow-orchestration`)
- **Training input pipeline** (`mlsys.input-pipeline`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Data Engineering — Input pipeline (`vol1.data.pipeline`)
- vol2: Data Storage — Input pipeline (`vol2.storage.input-pipeline`)

#### Data contracts, versioning &amp; lineage

Identity: `data-lineage`

Specifying, versioning, and tracing the datasets a system consumes so that dependencies, provenance, and changes are explicit.

- **Activities:** Acquire and prepare data; Operate and change; Retire and decommission.
- **Layers:** Application and workflow; Compute and communication infrastructure.
- **Concerns:** Reproducibility; Reliability and recovery; Privacy.
- **Contexts:** .

Includes: Dataset versions, snapshots, and provenance records.; Producer-consumer contracts and dependency hygiene..

Excludes: Model and code artifact versioning (Versioning, rollout &amp; rollback); Dataset documentation such as datasheets (Explainability &amp; documentation).

Design warrants: `mlops`, `debt`.

Candidate home concepts:

- **Data lineage** (`mlsys.lineage`): Recorded dependencies that identify the sources and transformations of a dataset or artifact.
- **Data contract** (`mlsys.data-contract`): A specification of schema, semantics, validity rules, and availability for data exchanged between components.
- **Dataset versioning** (`mlsys.data-versioning`): Recording immutable, addressable versions of datasets so that a training run or evaluation can identify exactly which data it used.
- **Data dependency debt** (`mlsys.data-dependency`): Undeclared, unstable, or underused input signals on which a model depends, which can change or break its behavior silently.
- **Data governance** (`mlsys.data-governance`): Policies, roles, and controls that decide who may collect, access, use, and share which data, for which purposes, and under what accountability.
- **Consent and purpose limitation** (`mlsys.data-consent`): Restricting collection and use of personal data to purposes the data subject agreed to or a legal basis allows, recorded so downstream uses can be checked.
- **Data and model licensing** (`mlsys.data-model-licensing`): Legal terms governing how datasets and model weights may be used, modified, redistributed, or deployed, which constrain acquisition, fine-tuning, and release decisions.

Secondary concept connections (not additional homes):

- **Artifact versioning** (`mlsys.artifact-version`)
- **Experiment reproducibility** (`mlsys.experiment-reproducibility`)

Related MLSysBook passages (teaching depth not assessed):

- vol3: Long-Term Memory — Source provenance (`vol3.provenance`)
- vol2: Data Storage — Versioned artifact (`vol2.storage.versioned-artifact`)
- vol4: Deployment Release — Release manifest (`vol4.assurance.manifest`)
- vol1: ML Lifecycle — Stage interface contract (`vol1.lifecycle.contract`)

#### Leakage &amp; training-serving skew

Identity: `data-leakage-skew`

Ensuring examples use only information available at prediction time and that training and serving compute features identically and equally fresh.

- **Activities:** Acquire and prepare data; Learn and adapt; Evaluate and assure; Deploy and integrate.
- **Layers:** Application and workflow; Model and algorithm.
- **Concerns:** Task and predictive quality; Reliability and recovery.
- **Contexts:** Classical ML; Streaming systems.

Includes: Point-in-time joins and temporal leakage.; Feature staleness and skew between offline and online paths..

Excludes: Deployment distribution change (Drift, feedback loops &amp; retraining).

Design warrants: `rules`, `debt`.

Candidate home concepts:

- **Data leakage** (`mlsys.leakage`): Information unavailable at the intended prediction boundary entering development or evaluation.
- **Point-in-time correctness** (`mlsys.point-in-time`): Constructing examples using only information available at their stated prediction time.
- **Training-serving skew** (`mlsys.train-serve-skew`): A mismatch between the data transformations or feature semantics used in training and serving.
- **Feature freshness** (`mlsys.feature-freshness`): The age of a feature value at the time it is used for a prediction, relative to the freshness assumed during training.
- **Online and offline feature consistency** (`mlsys.online-feature-consistency`): Keeping the feature values materialized in the low-latency online store consistent with the offline values used for training, as features are recomputed, backfilled, and updated.

Secondary concept connections (not additional homes):

- **Feature store** (`mlsys.feature-store`)
- **Benchmark contamination** (`mlsys.benchmark-contamination`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Data Engineering — Data leakage (`vol1.data.leakage`)
- vol1: Data Engineering — Training-serving parity (`vol1.data.parity`)

### Training &amp; adaptation

How are models trained and adapted, and what does that demand from the system?

Learning paradigms, optimization dynamics, training memory, experiment management, fine-tuning, post-training, and continual learning.

#### Learning paradigms

Identity: `learning-paradigms`

The ways supervision, self-generated targets, interaction, and demonstration produce learning signals, each with distinct data and compute demands.

- **Activities:** Specify and design; Learn and adapt.
- **Layers:** Model and algorithm.
- **Concerns:** Task and predictive quality; Monetary cost.
- **Contexts:** Classical ML; Deep learning; Generative systems; Physical and control systems.

Includes: Supervised, self-supervised, unsupervised, reinforcement, and imitation learning..

Excludes: Preference and RL post-training of pretrained models (Preference &amp; RL post-training).

Design warrants: `sklearn`, `cs336`.

Candidate home concepts:

- **Supervised learning** (`mlsys.supervised-learning`): Learning a prediction rule from examples with target outputs.
- **Self-supervised learning** (`mlsys.self-supervised-learning`): Learning from targets constructed from the observed data itself.
- **Unsupervised learning** (`mlsys.unsupervised-learning`): Learning structure in data without externally supplied example-level targets.
- **Reinforcement learning** (`mlsys.reinforcement-learning`): Learning a policy from action-dependent observations and reward signals.
- **Imitation learning** (`mlsys.imitation-learning`): Learning a policy from demonstrated behavior and possibly corrective interaction.
- **Pretraining** (`mlsys.pretraining`): Training a model from scratch on a large, broad corpus with a self-supervised objective to produce a general base model for later adaptation.
- **Causal language modeling** (`mlsys.causal-lm`): Training a model to predict each token from the tokens before it, maximizing the likelihood of text under a left-to-right factorization.
- **Compounding error in sequential decisions** (`mlsys.compounding-error`): Per-step errors of a learned sequential policy or generator push later inputs away from the training distribution, so failure probability grows with horizon length.
- **Interactive imitation learning** (`mlsys.interactive-imitation`): Imitation learning that queries an expert for corrective labels on states the learner's own policy visits and aggregates them into training data over rounds.
- **Offline reinforcement learning** (`mlsys.offline-rl`): Learning a policy from a fixed dataset of previously logged interactions without further environment access, limited by how well the data covers states the policy would visit.
- **Multi-task learning** (`mlsys.multi-task-learning`): Jointly learning several tasks through shared parameters, often a shared backbone with task-specific heads, so their training signals can help or interfere with one another.

Secondary concept connections (not additional homes):

- **Federated learning** (`mlsys.federated-learning`)
- **In-context learning** (`mlsys.in-context-learning`)

Related MLSysBook passages (teaching depth not assessed):

- vol2: Edge Intelligence — Federated averaging (`vol2.federated.fedavg`)
- vol3: Trajectory Fine-Tuning — Exposure bias (`vol3.exposure-bias`)
- vol4: Physical Data — Compounding error (`vol4.learning.compounding-error`)
- vol4: Physical Data — Recovery demonstrations (`vol4.learning.recovery`)

#### Optimization &amp; training dynamics

Identity: `training-optimization`

The objectives, update rules, schedules, and batch regimes that determine whether and how efficiently training converges.

- **Activities:** Learn and adapt.
- **Layers:** Model and algorithm; Numerical software and runtime.
- **Concerns:** Task and predictive quality; Latency, throughput, and resources; Monetary cost.
- **Contexts:** Deep learning; Distributed systems.

Includes: Optimizers, learning-rate warmup and decay, large-batch scaling.; Loss spikes, divergence, and their mitigations..

Excludes: Floating-point formats and overflow (Numerical formats &amp; precision).

Design warrants: `cs336`, `dlsys`.

Candidate home concepts:

- **Training loss** (`mlsys.loss`): A computable criterion optimized on training examples.
- **Optimization algorithm** (`mlsys.optimizer`): A rule that uses training information to update model parameters and optimizer state.
- **Regularization** (`mlsys.regularization`): Constraints or training procedures used to control sensitivity and generalization behavior.
- **Training batch size** (`mlsys.batch-size`): The number of examples contributing to an optimization update or microstep.
- **Learning-rate schedule** (`mlsys.learning-rate-schedule`): The planned variation of the optimizer step size over training, such as warmup followed by decay.
- **Critical batch size** (`mlsys.critical-batch-size`): The batch size beyond which increasing it further yields diminishing reductions in optimization steps to reach a target loss.
- **Training instability** (`mlsys.training-instability`): Loss spikes, divergence, or collapse during training, often mitigated by gradient clipping, schedule changes, normalization, or restarting from a checkpoint.
- **Mode collapse** (`mlsys.mode-collapse`): A generative-training failure in which the model covers only a few modes of the data distribution and produces low-diversity samples.
- **Gradient clipping** (`mlsys.gradient-clipping`): Rescaling or truncating gradients whose norm exceeds a threshold so that a single large update cannot destabilize training.
- **Weight initialization** (`mlsys.weight-initialization`): Choosing the scale and distribution of initial parameters so that activations and gradients keep stable variance through depth at the start of training.

Secondary concept connections (not additional homes):

- **Numerical stability** (`mlsys.numerical-stability`)
- **Mixed-precision training** (`mlsys.mixed-precision`)
- **Data parallelism** (`mlsys.data-parallel`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Model Training — Training batch size (`vol1.training.batch`)
- vol2: Distributed Training — Data parallelism (`vol2.training.data-parallelism`)
- vol1: Neural Computation — Parameter gradients (`vol1.computation.gradients`)
- vol1: Neural Computation — Parameter gradients (`vol1.computation.gradients`)
- vol2: Distributed Training — Critical batch size (`vol2.training.critical-batch`)

#### Training memory &amp; state

Identity: `training-memory`

Accounting for and reducing the parameters, gradients, optimizer state, and activations a training step must hold.

- **Activities:** Learn and adapt.
- **Layers:** Numerical software and runtime; Model and algorithm.
- **Concerns:** Latency, throughput, and resources; Monetary cost.
- **Contexts:** Deep learning; Distributed systems.

Includes: Activation recomputation, gradient accumulation, CPU/NVMe offload..

Excludes: Sharding state across devices (Parallelism strategies); Persisting state for recovery (Checkpointing &amp; elastic recovery).

Design warrants: `zero`, `cs336`.

Candidate home concepts:

- **Training state** (`mlsys.training-state`): The parameters, gradients, optimizer variables, and other state needed to continue training.
- **Activation recomputation** (`mlsys.activation-recompute`): Recreating forward intermediates during backward execution to reduce retained memory.
- **Gradient accumulation** (`mlsys.gradient-accumulation`): Summing gradients over several micro-batches before one optimizer update to emulate a larger batch within limited memory.
- **Training-state offloading** (`mlsys.memory-offloading`): Moving optimizer state, parameters, or activations from accelerator memory to host memory or storage between uses.
- **Memory-efficient optimizer** (`mlsys.memory-efficient-optimizer`): Optimizers that shrink per-parameter state through factorization, low-rank projection, or low-precision storage, cutting the memory adaptive methods such as Adam otherwise require.

Secondary concept connections (not additional homes):

- **Training-state sharding** (`mlsys.state-sharding`)
- **Recovery checkpoint** (`mlsys.checkpoint`)
- **Memory capacity** (`mlsys.memory-capacity`)
- **Neighbor sampling** (`mlsys.neighbor-sampling`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Neural Computation — Training state (`vol1.computation.state`)
- vol1: Model Training — Activation checkpointing (`vol1.training.checkpoint`)
- vol1: System Paradigms — Memory capacity (`vol1.system.capacity`)
- vol2: Compute Infrastructure — Memory capacity (`vol2.hardware.memory-capacity`)
- vol2: Distributed Training — Training-state sharding (`vol2.training.state-sharding`)
- vol2: Data Storage — Durable checkpoint (`vol2.storage.durable-checkpoint`)
- vol3: Durable Execution — Recovery checkpoint (`vol3.checkpoint`)

#### Hyperparameter search &amp; experiment management

Identity: `experiment-management`

Allocating a finite budget of training runs across configurations, and recording runs so their outcomes can be compared and reproduced.

- **Activities:** Learn and adapt; Evaluate and assure.
- **Layers:** Model and algorithm; Application and workflow.
- **Concerns:** Task and predictive quality; Monetary cost; Reproducibility.
- **Contexts:** Deep learning.

Includes: Random, Bayesian, and multi-fidelity search; small-scale proxies.; Run metadata and experiment tracking..

Excludes: Pipeline orchestration (ML pipelines &amp; reproducibility).

Design warrants: `random-search`, `hyperband`, `mlops`.

Candidate home concepts:

- **Hyperparameter search** (`mlsys.hyperparameter-search`): Choosing model or training settings through a sequence of evaluations.
- **Experiment-budget allocation** (`mlsys.experiment-budget`): Dividing a finite evaluation budget across configurations, repetitions, and fidelities.
- **Experiment tracking** (`mlsys.experiment-tracking`): Recording each run's code, data, configuration, metrics, and artifacts in a queryable store for comparison and reproduction.
- **Hyperparameter transfer** (`mlsys.hyperparameter-transfer`): Tuning hyperparameters on smaller proxy models and reusing them at larger scale under a parameterization designed to keep them stable.
- **Model selection** (`mlsys.model-selection`): Choosing among candidate model families, architectures, and hyperparameters by validated performance under the system's accuracy, cost, and latency constraints.

Secondary concept connections (not additional homes):

- **Model-selection optimism** (`mlsys.selection-bias`)
- **Experiment reproducibility** (`mlsys.experiment-reproducibility`)
- **Empirical scaling laws** (`mlsys.scaling-laws`)

Related MLSysBook passages (teaching depth not assessed):

- vol2: Introduction — Empirical scaling laws (`vol2.fleet.scaling-laws`)

#### Adaptation: fine-tuning, prompting &amp; transfer

Identity: `fine-tuning`

Adapting a pretrained model to a new task or behavior, by continued training of all or part of its parameters or by prompting without parameter updates.

- **Activities:** Learn and adapt.
- **Layers:** Model and algorithm.
- **Concerns:** Task and predictive quality; Monetary cost.
- **Contexts:** Deep learning; Generative systems; Agentic systems; Edge and device.

Includes: Instruction and supervised fine-tuning, adapters and LoRA..

Excludes: Preference or reward-driven training (Preference &amp; RL post-training); Assembling context for agents (Context engineering &amp; agent memory).

Design warrants: `cs336`, `fsdl2022`.

Candidate home concepts:

- **Transfer learning** (`mlsys.transfer-learning`): Adapting representations or parameters learned from one setting to another.
- **Parameter-efficient adaptation** (`mlsys.peft`): Adapting a model by changing a limited parameter subset or added parameterization.
- **Supervised fine-tuning** (`mlsys.supervised-fine-tuning`): Continued training of a pretrained model on input-output demonstrations, such as instruction-response pairs, to shape task behavior.
- **Low-rank adaptation** (`mlsys.lora`): Fine-tuning by learning low-rank update matrices added to frozen pretrained weights.
- **Prompt engineering** (`mlsys.prompt-engineering`): Designing instructions, examples, and input layout for a model call to elicit desired behavior without changing the model's weights.
- **Loss masking and token normalization** (`mlsys.loss-masking`): Excluding prompt, tool-output, and other non-target tokens from the training loss, and choosing how remaining token losses are averaged across turns and sequences.
- **In-context learning** (`mlsys.in-context-learning`): A pretrained model performing a task from instructions or examples supplied in its input at inference time, with no weight updates.
- **Model merging** (`mlsys.model-merging`): Combining compatible model parameters or parameter updates, by averaging or arithmetic on weight differences, into one model that consolidates capabilities without further training or serving the sources separately.

Secondary concept connections (not additional homes):

- **Knowledge distillation** (`mlsys.distillation`)
- **Quantization** (`mlsys.quantization`)
- **Catastrophic forgetting** (`mlsys.catastrophic-forgetting`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Model Compression — Quantization (`vol1.compression.quantization`)
- vol3: Trajectory Fine-Tuning — Loss mask (`vol3.mask`)

#### Preference &amp; RL post-training

Identity: `post-training`

Training pretrained models against preference, reward-model, or verifiable-reward signals, and the rollout, scoring, and weight-synchronization systems that loop requires.

- **Activities:** Learn and adapt; Evaluate and assure.
- **Layers:** Model and algorithm; Numerical software and runtime; Compute and communication infrastructure.
- **Concerns:** Task and predictive quality; Latency, throughput, and resources; Safety; Monetary cost.
- **Contexts:** Generative systems; Agentic systems; Distributed systems.

Includes: RLHF, DPO-style preference optimization, RL with verifiable rewards.; Rollout generation, reward scoring, colocated vs disaggregated actor/learner, weight sync..

Excludes: General RL theory (Learning paradigms); Reward exploitation as a specification failure (Objectives, proxies &amp; decision policies).

Design warrants: `cs336`, `instructgpt`.

Candidate home concepts:

- **RL from human feedback** (`mlsys.rlhf`): Optimizing a model's policy against a reward model trained on human preference comparisons, usually with a penalty for drifting from a reference model.
- **Reward model** (`mlsys.reward-model`): A learned model that scores outputs or trajectories to supply the training signal for policy optimization.
- **Direct preference optimization** (`mlsys.preference-optimization`): Training a policy directly on preferred-versus-rejected output pairs with a classification-style loss, without a separate reward model or RL loop.
- **RL with verifiable rewards** (`mlsys.rlvr`): Reinforcement learning whose rewards come from programmatic outcome checks, making verifier correctness and reward computation part of the training system.
- **RL rollout generation** (`mlsys.rl-rollout`): Sampling responses or trajectories from the current policy, often with an inference engine, to produce experience for policy updates.
- **Policy weight synchronization** (`mlsys.weight-sync`): Transferring updated policy weights from training workers to rollout workers, whose cost and frequency shape RL post-training throughput.
- **Policy staleness** (`mlsys.policy-staleness`): The lag between the policy version that generated rollouts and the version being updated, introduced by asynchronous rollout and training.
- **Credit assignment** (`mlsys.credit-assignment`): Attributing a delayed or episode-level reward to the individual actions, turns, or tokens that produced it, so updates reinforce the responsible decisions.
- **Reward hacking** (`mlsys.reward-hacking`): Improving a measured reward through behavior that fails the intended task objective.
- **Group relative policy optimization** (`mlsys.group-relative-policy-optimization`): A policy-gradient method that scores each sampled response against the mean reward of a group of responses to the same prompt, removing the learned critic at the cost of generating several rollouts per prompt.
- **Agent training environment** (`mlsys.agent-training-environment`): A task system that supplies observations, accepts actions, and manages episode state for training tool-using policies, whose throughput and reset fidelity limit rollout collection.

Secondary concept connections (not additional homes):

- **Reinforcement learning** (`mlsys.reinforcement-learning`)
- **Verifier** (`mlsys.verifier`)
- **Speculative decoding** (`mlsys.speculative-decoding`)

Related MLSysBook passages (teaching depth not assessed):

- vol3: Reinforcement Learning from Verifiable Rewards — Reward hacking (`vol3.reward-hacking`)
- vol3: Test-Time Compute — Verifier precision (`vol3.precision`)
- vol3: Reinforcement Learning from Verifiable Rewards — Verifiable reward (`vol3.reward`)
- vol3: Reinforcement Learning from Verifiable Rewards — Credit assignment (`vol3.credit`)

#### Continual &amp; online learning

Identity: `continual-learning`

Updating deployed models incrementally from new or streaming data while retaining prior competence.

- **Activities:** Learn and adapt; Operate and change.
- **Layers:** Model and algorithm; Numerical software and runtime.
- **Concerns:** Task and predictive quality; Reliability and recovery; Monetary cost.
- **Contexts:** Streaming systems; Edge and device; Federated settings.

Includes: Online and incremental updates, forgetting and replay.; Personalization and adaptation on device..

Excludes: Deciding when to retrain (Drift, feedback loops &amp; retraining); Federated aggregation protocols (Federated &amp; decentralized learning).

Design warrants: `dmls`, `cs329s`.

Candidate home concepts:

- **Online learning** (`mlsys.online-learning`): Updating a model incrementally as each example or small batch arrives, rather than retraining on a fixed dataset.
- **Incremental training** (`mlsys.incremental-training`): Continuing training of an existing model from its current weights on new data instead of training from scratch.
- **Catastrophic forgetting** (`mlsys.catastrophic-forgetting`): Loss of previously learned capability when a model is trained on new data or tasks.
- **Experience replay** (`mlsys.experience-replay`): Mixing stored past examples into updates on new data to reduce forgetting or reuse experience.
- **On-device training** (`mlsys.on-device-training`): Updating model parameters on the deployment device using local data under its memory, energy, and compute limits.

Secondary concept connections (not additional homes):

- **Retraining** (`mlsys.retraining`)
- **Distribution shift** (`mlsys.distribution-shift`)
- **Federated learning** (`mlsys.federated-learning`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: ML Operations — Distribution shift (`vol1.operations.drift`)
- vol2: Robust AI — Distribution shift (`vol2.robustness.shift`)
- vol4: Physical Data — Endogenous covariate shift (`vol4.learning.shift`)
- vol2: Edge Intelligence — Federated averaging (`vol2.federated.fedavg`)

### Model evaluation &amp; testing

How do we know a model and the system around it behave well enough, and for whom?

Evaluation design, metrics and uncertainty, slices and robustness, evaluation of generative and agentic outputs, and ML testing and debugging.

#### Evaluation design &amp; experiments

Identity: `evaluation-protocols`

Constructing offline splits and online experiments whose comparisons support the conclusions drawn from them.

- **Activities:** Evaluate and assure; Deploy and integrate.
- **Layers:** Application and workflow; Model and algorithm.
- **Concerns:** Task and predictive quality; Reproducibility.
- **Contexts:** Classical ML; Interactive services.

Includes: Holdout, temporal, and cross-validation splits.; A/B tests, sequential monitoring, interference between units..

Excludes: Release staging mechanics (Versioning, rollout &amp; rollback).

Design warrants: `selection-bias`, `sklearn`, `cs329s`.

Candidate home concepts:

- **Evaluation split** (`mlsys.evaluation-split`): A partitioning protocol that separates learning, selection, and final assessment.
- **Model-selection optimism** (`mlsys.selection-bias`): Overstated performance caused by choosing a winner using noisy evaluation results.
- **Online experimentation** (`mlsys.online-experiment`): A controlled intervention used to estimate effects on deployed-system outcomes.
- **Cross-validation** (`mlsys.cross-validation`): Estimating performance by repeatedly training on some data partitions and evaluating on the held-out remainder.
- **Sequential testing** (`mlsys.sequential-testing`): Experimental analysis that permits repeated looks at accumulating data while controlling error rates, unlike fixed-horizon tests.
- **Experiment interference** (`mlsys.experiment-interference`): Treatment of one experimental unit affecting outcomes of another, violating the no-interference assumption (SUTVA) of standard A/B analysis.
- **Evaluation harness** (`mlsys.evaluation-harness`): A system running versioned benchmark suites against model checkpoints with fixed prompts, decoding settings, and scorers, sharding and caching work so results stay reproducible and comparable.
- **Paired comparison and noninferiority testing** (`mlsys.paired-comparison`): Comparing two systems on the same items or units and testing per-item differences, including whether a candidate is no worse than a baseline by a preset margin.
- **Off-policy evaluation** (`mlsys.off-policy-evaluation`): Estimating a proposed policy's expected outcomes from interactions collected under another policy, by propensity reweighting, outcome modeling, or doubly robust estimators, subject to assumptions about action coverage and confounding.
- **Multiple comparisons correction** (`mlsys.multiple-comparisons`): Adjusting significance thresholds when many hypotheses, metrics, or slices are tested together, so the chance of a false declared effect, by family-wise error or false discovery rate, stays controlled.

Secondary concept connections (not additional homes):

- **Data leakage** (`mlsys.leakage`)
- **Shadow deployment** (`mlsys.shadow-deployment`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Data Engineering — Data leakage (`vol1.data.leakage`)
- vol3: Trajectory Curation — Task-group split (`vol3.split`)

#### Metrics, calibration &amp; uncertainty

Identity: `metrics-uncertainty`

Specifying what is measured and quantifying how far estimates and predicted probabilities can be trusted.

- **Activities:** Evaluate and assure.
- **Layers:** Model and algorithm.
- **Concerns:** Task and predictive quality; Reliability and recovery; Human control and accountability.
- **Contexts:** .

Includes: Task metrics, calibration, confidence intervals, abstention.; Distribution-free prediction sets..

Excludes: Choosing what to optimize (Objectives, proxies &amp; decision policies).

Design warrants: `metrics`, `calibration`.

Candidate home concepts:

- **Generalization** (`mlsys.generalization`): Performance on relevant examples or conditions beyond those used to fit the system.
- **Evaluation metrics** (`mlsys.metrics`): Specified measurements of predictive or task behavior, with explicit populations and aggregation.
- **Probability calibration** (`mlsys.calibration`): Agreement between predicted probabilities and observed event frequencies under an evaluation distribution.
- **Statistical uncertainty** (`mlsys.statistical-uncertainty`): Uncertainty in an estimated quantity due to finite and dependent observations.
- **Selective prediction** (`mlsys.selective-prediction`): Predicting on admitted cases and abstaining or deferring the rest.
- **Conformal prediction** (`mlsys.conformal-prediction`): Constructing prediction sets with finite-sample coverage guarantees under exchangeability, using held-out calibration scores.
- **Out-of-distribution detection** (`mlsys.ood-detection`): Flagging individual inputs that differ from the training distribution so the system can abstain, defer, or fall back instead of trusting the prediction.

Secondary concept connections (not additional homes):

- **Proxy objective** (`mlsys.proxy-objective`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Benchmarking — Model quality (`vol1.measurement.quality`)
- vol2: Robust AI — Selective prediction (`vol2.robustness.selective-prediction`)
- vol2: Robust AI — Selective prediction (`vol2.robustness.selective-prediction`)

#### Sliced &amp; robustness evaluation

Identity: `robustness-slices`

Measuring behavior separately across populations and conditions, and under deliberate shift or perturbation, rather than only in aggregate.

- **Activities:** Evaluate and assure.
- **Layers:** Model and algorithm; Application and workflow.
- **Concerns:** Task and predictive quality; Fairness; Reliability and recovery.
- **Contexts:** .

Includes: Subgroup and condition slices, shift and perturbation tests..

Excludes: Fairness criteria themselves (Fairness &amp; bias); Adversarial attack mechanics (Security &amp; privacy).

Design warrants: `ml-test-score`, `cs329s`.

Candidate home concepts:

- **Sliced evaluation** (`mlsys.sliced-evaluation`): Measuring behavior separately on relevant populations, conditions, or workload groups.
- **Robustness evaluation** (`mlsys.robustness-evaluation`): Testing behavior under explicitly chosen shifts, perturbations, faults, or operating stresses.
- **Slice discovery** (`mlsys.slice-discovery`): Automatically finding coherent data subsets on which a model underperforms, without predefined slice labels.

Secondary concept connections (not additional homes):

- **Fairness criteria** (`mlsys.fairness`)
- **Distribution shift** (`mlsys.distribution-shift`)
- **Adversarial evasion** (`mlsys.evasion`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: ML Operations — Distribution shift (`vol1.operations.drift`)
- vol2: Robust AI — Distribution shift (`vol2.robustness.shift`)
- vol4: Physical Data — Endogenous covariate shift (`vol4.learning.shift`)
- vol1: Responsible Engineering — Subgroup fairness (`vol1.responsibility.fairness`)
- vol2: Responsible AI — Fairness criterion (`vol2.governance.fairness`)
- vol1: Benchmarking — Model quality (`vol1.measurement.quality`)

#### Evaluating generative &amp; agentic systems

Identity: `generative-agent-eval`

Assessing open-ended outputs and multi-step runs where no single reference answer exists, using graders, sampling-aware metrics, and contamination controls.

- **Activities:** Evaluate and assure.
- **Layers:** Model and algorithm; Application and workflow.
- **Concerns:** Task and predictive quality; Safety; Reproducibility.
- **Contexts:** Generative systems; Agentic systems.

Includes: Model graders, pairwise and human preference, pass@k.; Contamination of benchmarks by training data..

Excludes: Verifiers used to select outputs at inference (Test-time compute).

Design warrants: `agents`, `cs336`.

Candidate home concepts:

- **Hallucination** (`mlsys.hallucination`): Generated factual assertions or summaries not supported by the relevant source or evidence.
- **Trajectory evaluation** (`mlsys.agent-evaluation`): Assessing task outcomes, intermediate effects, and constraint compliance across a complete agent run.
- **Model-based grading** (`mlsys.llm-judge`): Using a language model to score or compare generated outputs against a rubric, with its own biases and error rates to be measured.
- **pass@k** (`mlsys.pass-at-k`): The probability that at least one of k sampled outputs satisfies a correctness check.
- **Pairwise preference evaluation** (`mlsys.pairwise-preference-eval`): Ranking models from judgments of which of two outputs is better, typically aggregated with a Bradley-Terry or Elo-style model.
- **Human evaluation** (`mlsys.human-evaluation`): Assessment of system outputs by people following a protocol, used where automatic metrics or model graders are not trusted.
- **Benchmark contamination** (`mlsys.benchmark-contamination`): Presence of evaluation items, or close variants, in a model's training data, inflating measured performance.
- **Dangerous-capability evaluation** (`mlsys.dangerous-capability-eval`): Evaluating whether a model has capabilities, such as cyberoffense, biological uplift, or autonomous replication, that cross thresholds requiring added safeguards.
- **Perplexity** (`mlsys.perplexity`): The exponentiated average negative log-likelihood per token of held-out text, measuring how well a language model predicts that text.
- **Reference-match scoring** (`mlsys.reference-match-scoring`): Scoring a generated output by exact or token-overlap agreement with reference answers; cheap and reproducible but blind to valid answers phrased differently.
- **Model-grader bias** (`mlsys.judge-bias`): Systematic scoring errors in model-based graders, such as preferences for response position, length, or style, that distort comparisons independently of task quality.

Secondary concept connections (not additional homes):

- **Verifier** (`mlsys.verifier`)
- **Statistical uncertainty** (`mlsys.statistical-uncertainty`)

Related MLSysBook passages (teaching depth not assessed):

- vol3: Agent Evaluation — Task acceptance (`vol3.acceptance`)
- vol3: Test-Time Compute — Candidate selection (`vol3.selection`)
- vol3: Test-Time Compute — Verifier precision (`vol3.precision`)
- vol3: Agent Evaluation — Repeated-run consistency (`vol3.repeatability`)

#### ML testing &amp; debugging

Identity: `ml-testing`

Testing data, models, pipelines, and their integration against stated expectations when exact expected outputs are often unknown.

- **Activities:** Evaluate and assure; Deploy and integrate.
- **Layers:** Application and workflow; Model and algorithm; Numerical software and runtime.
- **Concerns:** Task and predictive quality; Reliability and recovery.
- **Contexts:** .

Includes: Oracles, metamorphic and behavioral tests, readiness rubrics..

Excludes: Fault injection into infrastructure (Fault injection &amp; resilience testing).

Design warrants: `ml-test-score`, `oracle-survey`.

Candidate home concepts:

- **Test oracle** (`mlsys.test-oracle`): A specification, procedure, reference, or informed judgment used to assess whether observed behavior satisfies stated expectations for a test.
- **Integration testing** (`mlsys.integration-testing`): Testing interactions among assembled components against stated interface and behavior expectations.
- **Metamorphic testing** (`mlsys.metamorphic-testing`): Testing by checking that known relations hold between outputs for related inputs, when correct outputs for single inputs are unknown.
- **Behavioral testing** (`mlsys.behavioral-testing`): Testing model capabilities with targeted input suites for invariance, directional change, and minimum functionality, independent of aggregate accuracy.
- **Production-readiness assessment** (`mlsys.production-readiness`): A structured rubric of data, model, infrastructure, and monitoring tests used to judge whether an ML system is ready to operate.
- **Model and prediction visualization** (`mlsys.model-visualization`): Plotting training curves, activations, embeddings, and prediction errors such as confusion matrices to inspect what a model learned and where it fails.
- **Error analysis** (`mlsys.error-analysis`): Inspecting a model's mistakes, grouping them into categories, and ranking categories by frequency and fixability to decide the next data or model change.
- **Training diagnostics** (`mlsys.training-diagnostics`): Checks such as overfitting a tiny batch, reading learning curves, and tracking gradient and activation statistics that localize bugs in training code, data, and configuration.

Secondary concept connections (not additional homes):

- **Data validation** (`mlsys.data-validation`)
- **Failure injection** (`mlsys.failure-injection`)

Related MLSysBook passages (teaching depth not assessed):

- vol4: Adversarial Verification — Fault injection (`vol4.verification.fault`)
- vol4: Adversarial Verification — Test oracle (`vol4.verification.oracle`)

### Deployment &amp; serving

How are trained models delivered to users within latency, throughput, and cost targets?

Serving architectures, latency and goodput, request scheduling, LLM inference mechanisms, structured generation, test-time compute, retrieval, and on-device deployment.

#### Serving architectures

Identity: `serving-architecture`

The structure through which predictions are produced and delivered, from offline batch scoring to online and serverless model servers, multi-model and multi-adapter hosting, and streaming speech services.

- **Activities:** Deploy and integrate; Operate and change.
- **Layers:** Application and workflow; Numerical software and runtime.
- **Concerns:** Latency, throughput, and resources; Monetary cost; Reliability and recovery.
- **Contexts:** Batch and offline; Interactive services; Streaming systems; Generative systems; Recommendation and ranking.

Includes: Batch vs online prediction, inference servers and APIs..

Excludes: Replica load balancing (Batching, admission &amp; autoscaling); Routing and composing calls across models (Compound AI systems &amp; model routing); Second-stage reranking inside retrieval (Retrieval &amp; vector search).

Design warrants: `cs329s`, `dmls`.

Candidate home concepts:

- **Online inference** (`mlsys.online-inference`): Producing predictions on demand for individual requests within a latency target.
- **Batch inference** (`mlsys.batch-inference`): Computing predictions for many inputs in scheduled jobs and storing them for later lookup.
- **Model server** (`mlsys.model-server`): A service that loads model artifacts and executes inference requests behind a network interface, managing versions, batching, and resources.
- **Multi-model serving** (`mlsys.multi-model-serving`): Hosting many models or adapters on shared serving resources, trading isolation and load time against utilization.
- **Multi-stage retrieval and ranking** (`mlsys.ranking-funnel`): Candidate generation followed by progressively costlier ranking stages under one latency budget, the dominant architecture for recommendation and search.
- **Multi-adapter serving** (`mlsys.multi-adapter-serving`): Serving many low-rank adapters over one shared base model by batching requests for different adapters with heterogeneous kernels and paging adapter weights on demand.
- **Request correlation keys** (`mlsys.keyed-predictions`): Passing a client-supplied key through inference unchanged so outputs of batch or asynchronous requests can be matched to their inputs regardless of order.
- **Stateless model serving** (`mlsys.stateless-serving-function`): Exporting a model as a stateless function from request inputs to prediction, so any replica can serve any request and replicas scale horizontally.
- **Serverless inference** (`mlsys.serverless-inference`): Running models on a managed platform that provisions replicas on demand, bills per use, and can scale to zero, exposing requests to cold-start latency.
- **Streaming speech recognition** (`mlsys.streaming-asr`): Serving speech recognition incrementally: consuming audio in chunks and emitting partial transcripts within a fixed lag, with encoder state carried between chunks.
- **Voice interaction pipeline** (`mlsys.voice-pipeline`): Serving a spoken interaction as one latency-budgeted pipeline of streaming speech recognition, a language model, and speech synthesis, with turn detection and interruption handling between stages.
- **Streaming inference** (`mlsys.streaming-inference`): Serving a model over an open stream of input, producing outputs incrementally from partial input while carrying state between chunks, rather than answering one complete request at a time.
- **Inference API and payload serialization** (`mlsys.inference-api`): The request and response protocol of a model server, including the transport (HTTP, gRPC, streaming connections), the tensor and payload encoding, and the cost of serializing inputs and outputs.

Secondary concept connections (not additional homes):

- **Retrieval** (`mlsys.retrieval`)
- **Cost per useful outcome** (`mlsys.unit-cost`)
- **Selective prediction** (`mlsys.selective-prediction`)

Related MLSysBook passages (teaching depth not assessed):

- vol2: Robust AI — Selective prediction (`vol2.robustness.selective-prediction`)
- vol3: Agent Economics — Cost per accepted task (`vol3.cost`)
- vol3: Long-Term Memory — Retrieval contract (`vol3.retrieval`)

#### Compound AI systems &amp; model routing

Identity: `compound-ai`

Composing several models, retrievers, and tools into one application, and routing, cascading, caching, and gating requests across models to meet quality and cost targets.

- **Activities:** Deploy and integrate; Operate and change.
- **Layers:** Application and workflow; Numerical software and runtime.
- **Concerns:** Latency, throughput, and resources; Monetary cost; Reliability and recovery.
- **Contexts:** Batch and offline; Interactive services; Streaming systems; Generative systems; Recommendation and ranking.

Includes: Language-model programs and their optimization; Model routing, cascades, gateways, and semantic caches.

Excludes: Single-model serving infrastructure (Serving architectures); Agent loops that plan and act through tools (Agent harness &amp; control loop).

Design warrants: `cs329s`, `dmls`.

Candidate home concepts:

- **Compound AI system** (`mlsys.compound-ai-system`): An application composing multiple model calls, retrievers, and tools into one program whose components and control flow are tuned and served jointly for quality and cost.
- **LM program optimization** (`mlsys.lm-program-optimization`): Automatically tuning the prompts, demonstrations, or weights of each model call in a declared pipeline against an end-to-end metric rather than per call.
- **Model routing** (`mlsys.model-routing`): Choosing which model serves each request from request features, predicted quality, and cost.
- **Model gateway** (`mlsys.model-gateway`): A proxy in front of one or more model providers that centralizes authentication, routing, rate limits, fallbacks, and usage logging for model calls.
- **Semantic cache** (`mlsys.semantic-cache`): Returning a stored model response when a new request is close in embedding space to an earlier one, trading answer fidelity for latency and cost.
- **Model cascade** (`mlsys.model-cascade`): Sending a request to cheaper models first and escalating to more capable models only when a confidence or quality criterion is not met.

Secondary concept connections (not additional homes):

- **Retrieval** (`mlsys.retrieval`)
- **Cost per useful outcome** (`mlsys.unit-cost`)

Related MLSysBook passages (teaching depth not assessed):

- vol3: Agent Economics — Cost per accepted task (`vol3.cost`)
- vol3: Long-Term Memory — Retrieval contract (`vol3.retrieval`)

#### Latency, throughput &amp; goodput

Identity: `latency-goodput`

Defining and meeting per-request latency and useful-throughput targets, including tail behavior and token-level latency for generative services.

- **Activities:** Deploy and integrate; Operate and change.
- **Layers:** Numerical software and runtime; Application and workflow.
- **Concerns:** Latency, throughput, and resources; Monetary cost.
- **Contexts:** Interactive services; Generative systems.

Includes: Latency budgets, tail latency, TTFT and inter-token latency.; Tail-tolerant request techniques..

Excludes: Analytic queueing models (Queueing &amp; capacity models); General measurement methodology and profiling (Profiling &amp; utilization); analytic queueing models (Queueing &amp; capacity models).

Design warrants: `sre`, `cs329s`.

Candidate home concepts:

- **Tail latency** (`mlsys.tail-latency`): A high quantile of a defined request-latency distribution.
- **SLO goodput** (`mlsys.slo-goodput`): The rate of useful completed work that satisfies the specified quality and service criteria.
- **Time to first token** (`mlsys.ttft`): Latency from request arrival until the first generated token is returned, dominated by queueing and prefill.
- **Time per output token** (`mlsys.tpot`): For a request producing multiple tokens, its elapsed decoding time after the first output divided by its remaining generated-token count, summarizing average decoding speed at a declared measurement boundary.
- **Hedged requests** (`mlsys.hedged-requests`): Issuing a duplicate request to another replica when the first has not responded within a delay, and using whichever returns first.
- **Request throughput** (`mlsys.request-throughput`): The number of requests, or tokens, a serving system completes per unit time at a stated load and latency target.
- **Latency percentiles** (`mlsys.latency-percentiles`): Summarizing a latency distribution by percentiles such as p50, p95, and p99, so that a service is judged by the experience of most requests and of the slow tail rather than by the mean.
- **Inter-token latency** (`mlsys.inter-token-latency`): The time between consecutive output tokens or chunks delivered to a streaming client, whose distribution, not only its mean, exposes decode stalls that an average time per output token hides.
- **Fan-out tail amplification** (`mlsys.fan-out-tail-amplification`): The growth in a request's latency tail when it waits on many parallel subrequests, because the chance that at least one subrequest is slow rises with the fan-out.

Secondary concept connections (not additional homes):

- **Service-level objective** (`mlsys.slo`)
- **Queueing delay** (`mlsys.queueing`)
- **Latency budget** (`mlsys.latency-budget`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Benchmarking — Latency SLO (`vol1.measurement.slo`)
- vol2: ML Operations at Scale — Evaluation contract (`vol2.operations.evaluation-contract`)
- vol1: Model Serving — Queueing delay (`vol1.serving.queue`)
- vol1: Model Serving — Request latency (`vol1.serving.latency`)
- vol2: Serving at Scale — SLO goodput (`vol2.serving.slo-goodput`)
- vol1: Benchmarking — Latency SLO (`vol1.measurement.slo`)
- vol1: Model Serving — Request latency (`vol1.serving.latency`)
- vol1: Model Serving — Time to first token (`vol1.serving.ttft`)
- vol1: Model Serving — Time per output token (`vol1.serving.tpot`)

#### Batching, admission &amp; autoscaling

Identity: `request-scheduling`

Policies that group, admit, distribute, and provision for requests so a service meets its targets under varying load.

- **Activities:** Deploy and integrate; Operate and change.
- **Layers:** Numerical software and runtime; Compute and communication infrastructure.
- **Concerns:** Latency, throughput, and resources; Monetary cost; Reliability and recovery.
- **Contexts:** Interactive services; Generative systems; Distributed systems.

Includes: Dynamic and iteration-level batching, admission, load balancing.; Autoscaling and model cold start..

Excludes: Cluster-level job scheduling (Cluster scheduling &amp; orchestration).

Design warrants: `sre`, `paged`.

Candidate home concepts:

- **Request batching** (`mlsys.batching`): Executing requests together to share work or improve resource use.
- **Admission control** (`mlsys.admission`): Accepting, delaying, or rejecting work according to capacity, policy, and service constraints.
- **Autoscaling** (`mlsys.autoscaling`): Changing provisioned service resources in response to measured or predicted demand.
- **Backpressure** (`mlsys.backpressure`): A mechanism that signals downstream saturation to limit or slow upstream production.
- **Continuous batching** (`mlsys.continuous-batching`): Scheduling at each generation iteration so finished sequences leave and new requests join a running batch.
- **Request load balancing** (`mlsys.load-balancing`): Distributing requests across replicas according to load, locality, or cached state.
- **Model cold start** (`mlsys.cold-start`): Delay before a newly started replica can serve, from provisioning, loading weights, and warming runtime state.
- **LLM request scheduling and preemption** (`mlsys.llm-request-scheduling`): Ordering, prioritizing, and preempting in-flight generation requests at iteration granularity, swapping or recomputing their KV state, to meet latency targets and fairness goals.
- **Cache-aware request routing** (`mlsys.cache-aware-routing`): Sending each request to the replica that already holds a reusable prefix or KV cache for it, trading load balance against cache reuse.

Secondary concept connections (not additional homes):

- **Service capacity** (`mlsys.service-capacity`)
- **Capacity planning** (`mlsys.capacity-planning`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Model Serving — Serving capacity (`vol1.serving.capacity`)
- vol1: Model Serving — Batching (`vol1.serving.batching`)
- vol2: Serving at Scale — Request batching (`vol2.serving.batching`)
- vol1: Model Serving — Batch formation delay (`vol1.serving.formation`)
- vol1: Model Serving — Serving capacity (`vol1.serving.capacity`)
- vol1: Model Serving — Continuous batching (`vol1.serving.continuous`)

#### LLM &amp; generative inference

Identity: `llm-inference`

Mechanisms specific to generative inference: managing attention state, separating prompt processing from token generation, accelerating autoregressive decoding, and reducing iterative denoising cost.

- **Activities:** Deploy and integrate; Operate and change.
- **Layers:** Numerical software and runtime; Model and algorithm; Compute and communication infrastructure.
- **Concerns:** Latency, throughput, and resources; Monetary cost.
- **Contexts:** Generative systems; Interactive services; Distributed systems; Multimodal systems.

Includes: KV cache management, prefix reuse, speculative decoding.; Chunked prefill and prefill/decode disaggregation.; Denoising-step reduction and feature caching for diffusion models..

Excludes: Weight quantization (Model compression); Context selection and compaction (Context engineering &amp; agent memory).

Design warrants: `paged`, `cs336`.

Candidate home concepts:

- **KV cache** (`mlsys.kv-cache`): Retained attention keys and values reused during autoregressive execution.
- **Prefill and decode** (`mlsys.prefill-decode`): The prompt-processing and incremental token-generation phases of autoregressive inference.
- **Paged KV allocation** (`mlsys.paged-kv`): Managing attention state in blocks to reduce allocation waste and support sharing.
- **Prefix caching** (`mlsys.prefix-cache`): Reusing compatible computed state for an identical request prefix.
- **Speculative decoding** (`mlsys.speculative-decoding`): Using proposed tokens and a validating procedure to accelerate generation while preserving a specified output distribution.
- **Chunked prefill** (`mlsys.chunked-prefill`): Splitting a long prompt's prefill into chunks interleaved with decode steps to bound per-iteration latency.
- **Prefill-decode disaggregation** (`mlsys.pd-disaggregation`): Running prefill and decode on separate resource pools, transferring KV state between them, so each phase is provisioned and batched independently.
- **Multimodal encoder disaggregation** (`mlsys.encoder-disaggregation`): Scheduling modality encoders separately from prefill and decode and caching encoder outputs so image or audio inputs do not stall token generation.
- **KV-cache offloading and tiering** (`mlsys.kv-cache-offload`): Moving KV-cache blocks from accelerator memory to host memory or SSD and reloading them on reuse, extending cache capacity at transfer bandwidth and latency cost.
- **KV-cache transfer** (`mlsys.kv-cache-transfer`): Shipping KV state between instances or nodes, as in disaggregated prefill and decode or cross-instance prefix reuse, overlapped with compute to hide network latency.
- **KV-cache eviction** (`mlsys.kv-cache-eviction`): Retaining only a selected subset of cached key-value entries, chosen by attention scores, recency, or learned importance, to bound KV memory during long generation.
- **CPU-GPU hybrid inference** (`mlsys.cpu-gpu-hybrid-inference`): Splitting inference so the host CPU computes some operations, such as attention or rarely active neurons, from host memory when model or KV state exceeds accelerator memory.
- **Non-prefix KV reuse** (`mlsys.non-prefix-kv-reuse`): Reusing precomputed KV entries for text chunks appearing at arbitrary positions across requests, selectively recomputing cross-chunk attention instead of the full prefill.
- **Token streaming** (`mlsys.token-streaming`): Delivering generated output to the client incrementally as tokens are produced, holding per-stream state and stopping generation when the client cancels or disconnects.
- **Denoising-step reduction** (`mlsys.denoising-step-reduction`): Lowering network evaluations per diffusion sample with faster samplers, step distillation, or consistency models, trading sample quality against latency and cost.
- **Cross-step feature caching** (`mlsys.diffusion-feature-caching`): Reusing intermediate activations that change little between adjacent denoising steps instead of recomputing them, cutting per-step cost at bounded quality loss.
- **Masked diffusion decoding** (`mlsys.masked-diffusion-decoding`): Generating text by repeatedly predicting and committing masked positions in parallel, trading refinement passes for token parallelism while weakening the attention-state reuse that autoregressive decoding relies on.
- **Expert offloading** (`mlsys.expert-offloading`): Keeping some mixture-of-experts weights in host memory or storage and fetching the experts a batch needs into accelerator memory, often prefetching predicted ones, which lets a model larger than accelerator memory run at the cost of transfers.
- **Intra-device overlap of bottlenecks** (`mlsys.intra-device-overlap`): Scheduling generative-inference operations with complementary compute, memory, or communication demands concurrently on one accelerator, using small batches or fused kernels so several hardware resources stay busy together.

Secondary concept connections (not additional homes):

- **Continuous batching** (`mlsys.continuous-batching`)
- **Quantization** (`mlsys.quantization`)
- **Tensor parallelism** (`mlsys.tensor-parallel`)
- **Attention** (`mlsys.attention`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Network Architectures — Self-attention (`vol1.architecture.attention`)
- vol1: Model Compression — Quantization (`vol1.compression.quantization`)
- vol2: Distributed Training — Tensor parallelism (`vol2.training.tensor-parallelism`)
- vol1: Model Serving — KV cache per conversation (`vol1.serving.kv`)
- vol2: Serving at Scale — KV cache (`vol2.serving.kv-cache`)
- vol3: KV Cache Management — Retained KV state (`vol3.kv`)
- vol1: Model Serving — Prefill (`vol1.serving.prefill`)
- vol1: Model Serving — Decode (`vol1.serving.decode`)
- vol1: Model Serving — Paged KV allocation (`vol1.serving.paging`)
- vol1: Model Serving — Prefix caching (`vol1.serving.prefix`)
- vol3: KV Cache Management — Prefix reuse (`vol3.prefix-reuse`)
- vol1: Model Serving — Continuous batching (`vol1.serving.continuous`)
- vol3: KV Cache Management — Paused-state residency (`vol3.residency`)

#### Decoding, sampling &amp; structured output

Identity: `structured-generation`

Controlling how tokens are selected and constraining generated output to required syntax or schemas.

- **Activities:** Deploy and integrate.
- **Layers:** Numerical software and runtime; Application and workflow.
- **Concerns:** Task and predictive quality; Reliability and recovery.
- **Contexts:** Generative systems; Agentic systems.

Includes: Temperature and nucleus sampling, grammar-constrained decoding, schema checks..

Excludes: Tool authority and effects (Tool use &amp; action interfaces).

Design warrants: `guided-generation`.

Candidate home concepts:

- **Structured output constraint** (`mlsys.grammar`): Restricting the syntax or schema of model outputs without establishing factual correctness or authority.
- **Decoding and sampling controls** (`mlsys.sampling-controls`): Parameters and rules, such as greedy, temperature, top-k, and nucleus sampling, that determine how tokens are chosen from a model's distribution.
- **Constrained decoding** (`mlsys.constrained-decoding`): Masking disallowed tokens at each step, typically from a grammar or schema automaton, so output satisfies a structural constraint.
- **Output validation** (`mlsys.output-validation`): Checking generated output against a schema or rules after generation and repairing, retrying, or rejecting it on failure.

Secondary concept connections (not additional homes):

- **Tool contract** (`mlsys.tool-contract`)
- **Hallucination** (`mlsys.hallucination`)

Related MLSysBook passages (teaching depth not assessed):

- vol3: The Foundation Model — Grammar constraint (`vol3.grammar`)

#### Test-time compute

Identity: `test-time-compute`

Spending additional inference computation on reasoning, sampling, search, and verification to raise output quality under a budget.

- **Activities:** Deploy and integrate; Evaluate and assure.
- **Layers:** Numerical software and runtime; Model and algorithm.
- **Concerns:** Task and predictive quality; Monetary cost; Latency, throughput, and resources.
- **Contexts:** Generative systems; Agentic systems.

Includes: Reasoning budgets, parallel sampling and selection, verifiers, tree search.; Trading inference compute against model size..

Excludes: Agent control loops and tool use (Agent harness &amp; control loop); Pass@k as an evaluation metric (Evaluating generative &amp; agentic systems).

Design warrants: `ttc-scaling`, `cs336`.

Candidate home concepts:

- **Reasoning budget** (`mlsys.reasoning-budget`): A limit on the intermediate tokens or steps a model may spend before answering, set to trade quality against latency and cost.
- **Best-of-n sampling** (`mlsys.best-of-n`): Generating n candidate outputs and returning the one ranked highest by a scorer or verifier.
- **Self-consistency voting** (`mlsys.self-consistency`): Sampling multiple reasoning paths and returning the most frequent final answer.
- **Verifier** (`mlsys.verifier`): A model or program that judges whether a candidate output or reasoning step is correct, used for selection, search, or reward.
- **Process reward model** (`mlsys.process-reward-model`): A verifier that scores intermediate reasoning steps rather than only final answers.
- **Inference-time search** (`mlsys.inference-time-search`): Exploring a tree of partial solutions, as in beam or tree search, guided by a scorer, to find a higher-quality output.
- **Test-time scaling** (`mlsys.test-time-scaling`): The empirical relation between inference compute spent per query and output quality, used to allocate compute between model size and inference.

Secondary concept connections (not additional homes):

- **pass@k** (`mlsys.pass-at-k`)
- **Reward model** (`mlsys.reward-model`)
- **Agent planning** (`mlsys.planning`)
- **Cost per useful outcome** (`mlsys.unit-cost`)

Related MLSysBook passages (teaching depth not assessed):

- vol3: Agent Economics — Cost per accepted task (`vol3.cost`)
- vol3: Test-Time Compute — Test-time search (`vol3.search`)
- vol3: Test-Time Compute — Candidate selection (`vol3.selection`)
- vol3: Test-Time Compute — Verifier precision (`vol3.precision`)
- vol3: Agent Evaluation — Repeated-run consistency (`vol3.repeatability`)

#### Retrieval &amp; vector search

Identity: `retrieval-rag`

Selecting relevant records from an external corpus by lexical, dense, or hybrid search and supplying them to a model at inference.

- **Activities:** Deploy and integrate; Acquire and prepare data; Operate and change.
- **Layers:** Application and workflow; Compute and communication infrastructure.
- **Concerns:** Task and predictive quality; Latency, throughput, and resources; Privacy.
- **Contexts:** Generative systems; Agentic systems; Interactive services.

Includes: Retrieval-augmented generation, ANN indexes and vector stores, reranking..

Excludes: Agent memory write and consolidation policy (Context engineering &amp; agent memory).

Design warrants: `ann`.

Candidate home concepts:

- **Retrieval** (`mlsys.retrieval`): Selecting relevant records from an external collection using an explicit query and ranking process.
- **Approximate nearest-neighbor search** (`mlsys.ann`): Trading exact neighbor recovery for reduced search cost using an index or approximation.
- **Retrieval-augmented generation** (`mlsys.rag`): Conditioning generation on documents retrieved for the request so outputs can draw on information outside the model's parameters.
- **Dense retrieval** (`mlsys.dense-retrieval`): Retrieving records by similarity between learned embeddings of query and records.
- **Lexical retrieval** (`mlsys.lexical-retrieval`): Retrieving records by weighted term overlap with the query using an inverted index, as in BM25.
- **Reranking** (`mlsys.reranking`): Reordering a retrieved candidate set with a more expensive relevance model before use.
- **Graph-based vector index** (`mlsys.graph-vector-index`): An approximate nearest-neighbor index linking vectors into a proximity graph, often hierarchical, searched by greedy traversal for high recall at high memory cost.
- **Partitioned and quantized vector index** (`mlsys.quantized-vector-index`): An index that clusters vectors into partitions probed selectively and compresses them with product quantization, trading recall for memory footprint and scan cost.
- **Filtered vector search** (`mlsys.filtered-vector-search`): Nearest-neighbor search restricted to records satisfying metadata predicates, where naive pre- or post-filtering loses recall or cost under selective filters.
- **Vector index freshness** (`mlsys.index-freshness`): The delay and recall cost of reflecting inserts, updates, and deletes in an ANN index, managed by incremental insertion, delta indexes, tombstones, and periodic rebuilds.
- **Iterative multi-hop retrieval** (`mlsys.multi-hop-retrieval`): Retrieval that issues successive queries, each formulated from previously retrieved evidence or reasoning steps, to answer questions no single lookup resolves.
- **Document chunking** (`mlsys.document-chunking`): Dividing source documents into retrievable units with chosen boundaries, overlap, and source identifiers so retrieval granularity fits embedding and generation budgets.
- **Hybrid retrieval** (`mlsys.hybrid-retrieval`): Combining lexical and dense retrieval results through score or rank fusion to recover complementary matches while managing additional search cost and ranking calibration.
- **Embedding model migration** (`mlsys.embedding-migration`): Changing the encoder behind a vector index while keeping queries and stored vectors comparable, by re-encoding the corpus, validated mappings, or backward-compatible training, with a controlled serving cutover.
- **RAG pipeline scheduling** (`mlsys.rag-pipeline-scheduling`): Coordinating the retrieval and generation work of retrieval-augmented requests across requests and processors, overlapping and batching work across the stage boundary and optionally adapting retrieval settings per query, to meet latency and answer-quality targets.

Secondary concept connections (not additional homes):

- **Embeddings** (`mlsys.embeddings`)
- **Context window** (`mlsys.context`)

Related MLSysBook passages (teaching depth not assessed):

- vol3: Context Engineering — Working context (`vol3.working-context`)
- vol3: Long-Term Memory — Retrieval contract (`vol3.retrieval`)

#### On-device &amp; edge deployment

Identity: `on-device-deployment`

Converting, packaging, executing, and updating models on phones, embedded devices, and microcontrollers under memory, energy, and connectivity limits.

- **Activities:** Deploy and integrate; Operate and change.
- **Layers:** Numerical software and runtime; Compute and communication infrastructure; People and operating environment.
- **Concerns:** Latency, throughput, and resources; Reliability and recovery; Privacy; Energy and environmental impacts.
- **Contexts:** Edge and device; Physical and control systems.

Includes: Export and conversion, on-device runtimes, device/cloud split.; Over-the-air model updates across heterogeneous fleets..

Excludes: Edge silicon (Edge &amp; embedded hardware); Compression techniques (Model compression).

Design warrants: `tflm`, `dlsys`.

Candidate home concepts:

- **Model conversion** (`mlsys.model-conversion`): Translating a trained model into the format and operator set of a target runtime, possibly changing numerics and supported operations.
- **On-device inference runtime** (`mlsys.on-device-runtime`): An interpreter or compiled engine that executes models locally on personal or embedded devices, including inside browsers, under the target environment's memory, operator, and execution limits.
- **Over-the-air model update** (`mlsys.ota-model-update`): Delivering new model versions to deployed devices remotely, with staging, verification, and rollback across the device fleet.
- **Device-cloud partitioning** (`mlsys.device-cloud-partitioning`): Splitting inference between device and server, by layer or by model, to balance latency, energy, bandwidth, and privacy.
- **Device heterogeneity** (`mlsys.device-heterogeneity`): Variation in hardware, operating system, and runtime capability across a deployed device fleet that a single model release must accommodate.
- **Asynchronous on-device inference** (`mlsys.ui-thread-inference`): Running inference off the application's main thread, so a model call does not block rendering and input handling, with results delivered back by callback or message.

Secondary concept connections (not additional homes):

- **Quantization** (`mlsys.quantization`)
- **Execution placement** (`mlsys.placement`)
- **On-device training** (`mlsys.on-device-training`)
- **Memory capacity** (`mlsys.memory-capacity`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Model Compression — Quantization (`vol1.compression.quantization`)
- vol1: System Paradigms — Memory capacity (`vol1.system.capacity`)
- vol2: Compute Infrastructure — Memory capacity (`vol2.hardware.memory-capacity`)

### MLOps &amp; operations

How is a deployed ML system versioned, observed, changed, and eventually retired?

Release management, monitoring, drift and retraining, incident response, pipelines and reproducibility, platforms and teams, and retirement.

#### Versioning, rollout &amp; rollback

Identity: `release-management`

Identifying, gating, and progressively exposing model releases, with the ability to return to a known-good configuration.

- **Activities:** Deploy and integrate; Operate and change.
- **Layers:** Application and workflow; Compute and communication infrastructure.
- **Concerns:** Reliability and recovery; Reproducibility; Safety.
- **Contexts:** .

Includes: Registries, CI/CD for models, release gates.; Shadow, canary, and staged rollout; rollback..

Excludes: Statistical design of experiments (Evaluation design &amp; experiments).

Design warrants: `mlops`, `ml-test-score`.

Candidate home concepts:

- **Artifact versioning** (`mlsys.artifact-version`): Identifying the exact model, code, data, configuration, and dependencies in an execution or release.
- **Staged rollout** (`mlsys.rollout`): Increasing exposure to a candidate system through bounded release stages.
- **Rollback** (`mlsys.rollback`): Returning to a previously accepted deployment configuration with compatible state.
- **Model registry** (`mlsys.model-registry`): A versioned store of model artifacts with metadata, lineage, and lifecycle stage used to govern promotion and deployment.
- **Shadow deployment** (`mlsys.shadow-deployment`): Running a candidate model on live traffic without serving its outputs, to compare it against the production model.
- **CI/CD for ML** (`mlsys.ml-ci-cd`): Automated building, testing, and deployment of ML code, data checks, and model artifacts on each change.
- **Release gate** (`mlsys.release-gate`): A predeclared set of checks a candidate model must pass before it advances to a wider deployment stage.
- **Model and feature compatibility** (`mlsys.model-feature-compatibility`): Ensuring that a model version, its tokenizer and feature transformations, the schemas it reads, and the consumers of its outputs remain compatible as each is versioned and released independently.

Secondary concept connections (not additional homes):

- **Online experimentation** (`mlsys.online-experiment`)
- **Model artifact** (`mlsys.model-artifact`)
- **Over-the-air model update** (`mlsys.ota-model-update`)

Related MLSysBook passages (teaching depth not assessed):

- vol2: Data Storage — Versioned artifact (`vol2.storage.versioned-artifact`)
- vol4: Deployment Release — Release manifest (`vol4.assurance.manifest`)
- vol2: ML Operations at Scale — Staged rollout (`vol2.operations.rollout`)
- vol3: Operating Agents — Staged release (`vol3.release`)
- vol1: ML Operations — Rollback (`vol1.operations.rollback`)
- vol1: ML Lifecycle — Release gate (`vol1.lifecycle.gate`)
- vol2: ML Operations at Scale — Evaluation contract (`vol2.operations.evaluation-contract`)
- vol2: ML Operations at Scale — Dependency registry (`vol2.operations.registry`)

#### Monitoring &amp; observability

Identity: `monitoring`

Instrumenting deployed systems and detecting, from logged signals, changes in inputs, outputs, quality, and health.

- **Activities:** Operate and change.
- **Layers:** Application and workflow; Numerical software and runtime; Compute and communication infrastructure.
- **Concerns:** Reliability and recovery; Task and predictive quality; Privacy.
- **Contexts:** .

Includes: Prediction logging, drift statistics, alerting and alert fatigue.; Metrics, logs, traces for model services and agents..

Excludes: Responding to detected incidents (Incident response).

Design warrants: `sre`, `cs329s`.

Candidate home concepts:

- **Production monitoring** (`mlsys.monitoring`): Collecting and interpreting signals about deployed system behavior and health.
- **System observability** (`mlsys.observability`): Instrumentation that permits investigation of internal execution from recorded outputs, traces, and metrics.
- **Drift detection** (`mlsys.drift-detection`): Statistical tests or distances comparing production and reference distributions of inputs, predictions, or outcomes to flag change.
- **Prediction logging** (`mlsys.inference-logging`): Recording requests, model versions, outputs, and later outcomes so production behavior can be analyzed, debugged, and reused as data.
- **Alerting** (`mlsys.alerting`): Rules that turn monitored signals into notifications for responders, tuned to balance missed incidents against alert fatigue.
- **Distributed tracing** (`mlsys.distributed-tracing`): Recording request-scoped, causally linked spans across services so that one request's latency and failures can be attributed to the components it traversed.
- **Model and agent call tracing** (`mlsys.llm-call-tracing`): Tracing model calls, retrievals, and tool invocations as spans annotated with prompts, token counts, latency, and cost, so quality and spend are attributable per step.
- **Misuse monitoring** (`mlsys.misuse-monitoring`): Detecting policy-violating use of a deployed model across requests, sessions, and accounts, such as coordinated abuse or capability probing, and responding with throttling, review, or account action.
- **Telemetry sampling and cardinality control** (`mlsys.telemetry-sampling`): Choosing which traces, events, and metric label combinations to keep, by sampling and by bounding or aggregating label values, so fleet telemetry stays affordable at the risk of losing rare failures and fine-grained detail.

Secondary concept connections (not additional homes):

- **Distribution shift** (`mlsys.distribution-shift`)
- **Delayed outcomes** (`mlsys.delayed-outcomes`)
- **Privacy leakage** (`mlsys.privacy-leakage`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: ML Operations — Distribution shift (`vol1.operations.drift`)
- vol2: Robust AI — Distribution shift (`vol2.robustness.shift`)
- vol4: Physical Data — Endogenous covariate shift (`vol4.learning.shift`)
- vol1: ML Operations — Production monitoring (`vol1.operations.monitor`)
- vol1: ML Operations — Distribution shift (`vol1.operations.drift`)
- vol3: Operating Agents — Trajectory trace (`vol3.trace`)

#### Drift, feedback loops &amp; retraining

Identity: `drift-retraining`

Understanding how deployment data and outcomes change, including changes the system itself causes, and deciding when and how to update the model.

- **Activities:** Operate and change; Learn and adapt.
- **Layers:** Application and workflow; Model and algorithm; People and operating environment.
- **Concerns:** Task and predictive quality; Reliability and recovery; Fairness; Monetary cost.
- **Contexts:** Classical ML; Streaming systems.

Includes: Covariate and concept drift, delayed labels, hidden feedback loops.; Scheduled and triggered retraining, continuous training..

Excludes: Incremental learning algorithms (Continual &amp; online learning).

Design warrants: `debt`, `dmls`.

Candidate home concepts:

- **Distribution shift** (`mlsys.distribution-shift`): A difference between development and deployment data-generating distributions.
- **Concept drift** (`mlsys.concept-drift`): Change over time in the relationship between inputs and targets, so the same input warrants a different prediction.
- **Delayed outcomes** (`mlsys.delayed-outcomes`): Target outcomes that become observable only after a delay or for a selected subset of cases.
- **Deployment feedback loop** (`mlsys.feedback-loop`): A deployed system's outputs changing the data or behavior encountered by future learning or evaluation.
- **Retraining** (`mlsys.retraining`): Training a replacement version of a deployed model on updated or revised data, with timing and data chosen to balance staleness against training and validation cost.
- **Continuous training** (`mlsys.continuous-training`): An automated pipeline that retrains, validates, and proposes models on a schedule or when triggered by data or performance signals.
- **User feedback collection** (`mlsys.user-feedback`): Capturing explicit signals such as ratings and corrections and implicit signals such as clicks, edits, and abandonment, for use as labels or quality signals.
- **Fleet data engine** (`mlsys.fleet-data-engine`): Collecting training data from deployed devices through on-device triggers that flag rare or uncertain events, uploading selected samples under bandwidth budgets for labeling and retraining.

Secondary concept connections (not additional homes):

- **Online learning** (`mlsys.online-learning`)
- **Drift detection** (`mlsys.drift-detection`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: ML Operations — Distribution shift (`vol1.operations.drift`)
- vol2: Robust AI — Distribution shift (`vol2.robustness.shift`)
- vol4: Physical Data — Endogenous covariate shift (`vol4.learning.shift`)
- vol1: Responsible Engineering — Bias feedback (`vol1.responsibility.feedback`)
- vol1: ML Operations — Distribution shift (`vol1.operations.drift`)
- vol2: Robust AI — Distribution shift (`vol2.robustness.shift`)

#### Incident response

Identity: `incident-response`

Detecting, triaging, mitigating, and learning from production failures, including attributing them to data, model, or infrastructure causes.

- **Activities:** Operate and change.
- **Layers:** Application and workflow; Numerical software and runtime; Compute and communication infrastructure.
- **Concerns:** Reliability and recovery; Safety.
- **Contexts:** .

Includes: Triage, mitigation, blameless postmortems, on-call..

Excludes: Failure semantics and recovery mechanisms (Reliability &amp; fault tolerance).

Design warrants: `sre`.

Candidate home concepts:

- **Incident response** (`mlsys.incident-response`): Detecting, containing, diagnosing, and recovering from unexpected system behavior.
- **Blameless postmortem** (`mlsys.postmortem`): A written analysis after an incident of its timeline, causes, and corrective actions, focused on system fixes rather than blame.
- **ML failure attribution** (`mlsys.failure-attribution`): Determining whether a production failure stems from data, model, code, configuration, or infrastructure, or from an adversary.

Secondary concept connections (not additional homes):

- **Rollback** (`mlsys.rollback`)
- **System observability** (`mlsys.observability`)
- **Failure model** (`mlsys.failure-model`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: ML Operations — Rollback (`vol1.operations.rollback`)
- vol3: Operating Agents — Causal diagnosis (`vol3.diagnosis`)

#### ML pipelines &amp; reproducibility

Identity: `pipelines-reproducibility`

Automating the steps that produce models and controlling their environments and configuration so runs can be repeated.

- **Activities:** Learn and adapt; Deploy and integrate; Operate and change.
- **Layers:** Compute and communication infrastructure; Numerical software and runtime.
- **Concerns:** Reproducibility; Reliability and recovery.
- **Contexts:** Batch and offline.

Includes: Orchestrated pipelines, environment pinning, configuration as code..

Excludes: Experiment search strategy (Hyperparameter search &amp; experiment management); Scheduling containers across a cluster (Cluster scheduling &amp; orchestration).

Design warrants: `kfp-pipeline`, `mlflow-dependencies`, `pytorch-reproducibility`.

Candidate home concepts:

- **Dependency environment** (`mlsys.dependency-environment`): The software packages, versions, native libraries, and runtime configuration on which an execution depends.
- **Workflow orchestration** (`mlsys.workflow-orchestration`): Coordinating dependent computation steps, their data or control flow, execution conditions, and failure handling.
- **Experiment reproducibility** (`mlsys.experiment-reproducibility`): The ability to repeat an experimental procedure and obtain results consistent with declared criteria when its inputs, environment, randomness, and execution conditions are specified.
- **ML configuration management** (`mlsys.configuration-management`): Treating model, feature, and pipeline configuration as versioned, reviewed, and tested code.
- **Infrastructure as code** (`mlsys.infrastructure-as-code`): Declaring compute, storage, and service infrastructure in versioned machine-readable files from which environments are provisioned.
- **Containerization** (`mlsys.containerization`): Packaging a model, its code, and its dependencies as an immutable image that runs unchanged across machines. Scheduling those containers across a cluster belongs to the cluster manager.

Secondary concept connections (not additional homes):

- **Data lineage** (`mlsys.lineage`)
- **Experiment tracking** (`mlsys.experiment-tracking`)

Related MLSysBook passages (teaching depth not assessed):

- vol3: Long-Term Memory — Source provenance (`vol3.provenance`)

#### ML platforms, teams &amp; technical debt

Identity: `ml-platforms`

Shared internal platforms, team structures, and debt management that let an organization operate many models at sustainable cost.

- **Activities:** Operate and change; Specify and design.
- **Layers:** Application and workflow; Compute and communication infrastructure.
- **Concerns:** Monetary cost; Reliability and recovery.
- **Contexts:** .

Includes: Platform-as-product, maturity levels, ownership and roles.; ML-specific technical debt and anti-patterns..

Excludes: Fleet hardware and scheduling (Datacenter infrastructure &amp; fleet); Governance and regulation (AI governance, audit &amp; regulation).

Design warrants: `debt`, `mlops`, `se4ml`.

Candidate home concepts:

- **ML platform** (`mlsys.ml-platform`): Shared internal infrastructure and services that standardize how teams train, deploy, and operate models.
- **ML technical debt** (`mlsys.technical-debt`): Accumulated system costs from shortcuts in ML code, data dependencies, configuration, and pipelines that make future change slower and riskier.
- **Entanglement** (`mlsys.entanglement`): The property that changing any input, feature, or hyperparameter of a model can change its behavior on everything else.
- **Undeclared consumers** (`mlsys.undeclared-consumers`): Systems that use a model's outputs without the model owner's knowledge, creating hidden coupling.
- **MLOps maturity** (`mlsys.mlops-maturity`): Staged levels of automation, from manual training and deployment to automated continuous training and delivery.
- **ML team roles** (`mlsys.ml-roles`): The division of responsibilities among data, modeling, platform, and operations roles that owns an ML system across its lifecycle.
- **Glue code** (`mlsys.glue-code`): Supporting code written to move data into and out of general-purpose ML packages, often far larger than the model code and binding the system to package specifics.
- **Pipeline jungle** (`mlsys.pipeline-jungle`): Data preparation grown incrementally into a tangle of scrapes, joins, and intermediate outputs that is hard to test, trace, and change.
- **Build-versus-buy decision** (`mlsys.build-vs-buy`): Choosing whether to build an ML platform component in house, adopt open source, or buy a managed service, weighing fit, cost, lock-in, and team capacity.

Secondary concept connections (not additional homes):

- **Feature store** (`mlsys.feature-store`)
- **Model registry** (`mlsys.model-registry`)
- **Data dependency debt** (`mlsys.data-dependency`)

Related MLSysBook passages (teaching depth not assessed):

- vol2: ML Operations at Scale — Dependency registry (`vol2.operations.registry`)

#### Model retirement

Identity: `retirement`

Ending a model's service by migrating its consumers, archiving what must be kept, and deleting data and resources that are no longer justified.

- **Activities:** Retire and decommission.
- **Layers:** Application and workflow; Compute and communication infrastructure.
- **Concerns:** Monetary cost; Privacy; Reproducibility.
- **Contexts:** .

Includes: Deprecation with consumer migration, archival for audit, retention and deletion..

Excludes: Removing learned influence of specific records (Privacy leakage &amp; differential privacy).

Design warrants: `mlops`, `debt`.

Candidate home concepts:

- **Model deprecation** (`mlsys.model-deprecation`): Planned withdrawal of a model from service, including notifying and migrating its consumers before removing it.
- **Artifact archival** (`mlsys.artifact-archival`): Preserving retired models, data references, and evaluation records needed for audit, reproduction, or rollback after service ends.
- **Data retention** (`mlsys.data-retention`): Policy and mechanisms that set how long data, logs, and derived artifacts are kept and when they are deleted.
- **Data deletion propagation** (`mlsys.deletion-propagation`): Carrying a deletion request through every copy and derivative of the data, including datasets, features, caches, replicas, backups, and models trained on it.

Secondary concept connections (not additional homes):

- **Undeclared consumers** (`mlsys.undeclared-consumers`)
- **Privacy requirements** (`mlsys.privacy`)
- **Data lineage** (`mlsys.lineage`)

Related MLSysBook passages (teaching depth not assessed):

- vol3: Long-Term Memory — Source provenance (`vol3.provenance`)

## Qualities

By which properties is the system judged, and how are they engineered?

### Performance, efficiency &amp; benchmarking

How do we measure, compare, and reduce the time, memory, and energy a workload consumes?

Measuring and benchmarking where time, memory, and energy go, and the cross-layer techniques whose primary purpose is to reduce them, such as compression and hardware-aware design.

#### Profiling &amp; utilization

Identity: `profiling-utilization`

Measuring where execution time and resources go under a stated workload, and expressing how much of the hardware's capacity does useful model work.

- **Activities:** Learn and adapt; Evaluate and assure; Deploy and integrate; Operate and change.
- **Layers:** Numerical software and runtime; Compute and communication infrastructure.
- **Concerns:** Latency, throughput, and resources; Monetary cost; Reproducibility.
- **Contexts:** Deep learning; Distributed systems.

Includes: Profilers and execution-time breakdowns; Utilization metrics such as MFU and training goodput; Bottleneck attribution across compute, memory, I/O, and communication.

Excludes: Standardized benchmark suites and reporting rules (System benchmarking); Worst-case execution-time bounds (Real-time &amp; mixed-criticality computing); Serving-specific latency and goodput targets (Latency, throughput &amp; goodput).

Design warrants: `dlsys`, `roofline`.

Candidate home concepts:

- **Profiling** (`mlsys.profiling`): Measuring execution to attribute time, work, stalls, and resource use.
- **Execution time** (`mlsys.execution-time`): Elapsed time for a specified execution boundary, workload, and operating condition.
- **Resource utilization** (`mlsys.utilization`): The fraction of a resource's defined service capacity that is occupied over a stated interval.
- **Model FLOPs utilization** (`mlsys.model-flops-utilization`): Model-required floating-point operations per second achieved, divided by the aggregate peak FLOP rate of the hardware used, excluding recomputation.
- **Bottleneck analysis** (`mlsys.bottleneck-analysis`): Identifying the resource or stage, such as compute, memory bandwidth, I/O, or communication, whose capacity limits end-to-end throughput or latency for a workload.
- **Training goodput** (`mlsys.training-goodput`): The fraction of allocated accelerator time that produces retained training progress, after losses to scheduling, failures, restarts, and lost work.
- **Cross-stack profiling** (`mlsys.cross-stack-profiling`): Profiling that correlates timestamped events and calling contexts from model code, frameworks, native libraries, and accelerator kernels, so that device time and stalls can be attributed to the operations that launched them.

Secondary concept connections (not additional homes):

- **Roofline bound** (`mlsys.roofline`)
- **Benchmark protocol** (`mlsys.benchmark`)
- **SLO goodput** (`mlsys.slo-goodput`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Model Serving — Utilization (`vol1.serving.utilization`)
- vol2: Serving at Scale — SLO goodput (`vol2.serving.slo-goodput`)
- vol1: Benchmarking — Benchmark protocol (`vol1.measurement.protocol`)
- vol1: Model Training — Model FLOPs utilization (`vol1.training.mfu`)
- vol1: Hardware Acceleration — Operand transfer time (`vol1.hardware.transfer`)
- vol1: Model Serving — Host-path overhead (`vol1.serving.host`)
- vol2: Fleet Efficiency — ML goodput (`vol2.goodput.ml`)
- vol2: Fleet Efficiency — Program goodput (`vol2.goodput.program`)

#### System benchmarking

Identity: `system-benchmarking`

Measuring training and inference system performance under specified workloads, quality targets, load scenarios, and run rules so results are comparable.

- **Activities:** Evaluate and assure.
- **Layers:** Numerical software and runtime; Compute and communication infrastructure.
- **Concerns:** Latency, throughput, and resources; Reproducibility; Energy and environmental impacts.
- **Contexts:** Distributed systems; Edge and device.

Includes: MLPerf-style time-to-quality and scenario-based inference metrics..

Excludes: Profiling a single workload (Profiling &amp; utilization).

Design warrants: `mlperf-inference`, `roofline`.

Candidate home concepts:

- **Benchmark protocol** (`mlsys.benchmark`): A reproducible workload, measurement boundary, and comparison procedure.
- **Time to train** (`mlsys.time-to-train`): Wall-clock time for a system to train a specified model to a specified quality target under fixed run rules.
- **Inference load scenario** (`mlsys.inference-load-scenario`): A specified query-arrival pattern and latency constraint, such as single-stream, server, or offline, under which inference performance is measured.
- **Benchmark representativeness** (`mlsys.benchmark-representativeness`): The degree to which a benchmark's workload, scale, and measurement boundary predict production behavior, from isolated microbenchmarks to end-to-end application runs.
- **Performance regression testing** (`mlsys.performance-regression-testing`): Benchmarking every change to code, kernels, models, or configuration continuously and failing the change when throughput, latency, or memory degrades beyond a set tolerance.
- **Performance measurement variability** (`mlsys.measurement-variability`): Run-to-run variation in measured time and throughput from warm-up, asynchronous execution, clock and thermal state, and contention, handled by synchronizing, repeating runs, and reporting spread.

Secondary concept connections (not additional homes):

- **Profiling** (`mlsys.profiling`)
- **Operating energy** (`mlsys.energy`)
- **SLO goodput** (`mlsys.slo-goodput`)
- **Statistical uncertainty** (`mlsys.statistical-uncertainty`)

Related MLSysBook passages (teaching depth not assessed):

- vol2: Serving at Scale — SLO goodput (`vol2.serving.slo-goodput`)
- vol1: Benchmarking — Benchmark protocol (`vol1.measurement.protocol`)
- vol2: Sustainable AI — Operating energy (`vol2.sustainability.energy`)
- vol1: Model Training — Time to target quality (`vol1.training.quality-time`)

#### Model compression

Identity: `model-compression`

Techniques that shrink a trained or training model's parameters, precision, or rank to reduce memory, compute, and latency at a measured cost in quality.

- **Activities:** Learn and adapt; Evaluate and assure; Deploy and integrate.
- **Layers:** Model and algorithm; Numerical software and runtime.
- **Concerns:** Task and predictive quality; Latency, throughput, and resources; Monetary cost; Energy and environmental impacts.
- **Contexts:** Deep learning; Generative systems; Edge and device.

Includes: Quantization (post-training and quantization-aware); Pruning (structured and unstructured) and sparsity exploitation; Knowledge distillation and low-rank factorization.

Excludes: Numeric format definitions and mixed-precision training (Numerical formats &amp; precision); Parameter-efficient fine-tuning such as LoRA, whose purpose is adaptation (Adaptation: fine-tuning, prompting &amp; transfer).

Design warrants: `int8`, `mit-65940`.

Candidate home concepts:

- **Quantization** (`mlsys.quantization`): Representing values using a reduced discrete set with a specified encoding and approximation rule.
- **Sparsity** (`mlsys.sparsity`): A pattern of absent or zero values whose execution benefit depends on representation and support.
- **Knowledge distillation** (`mlsys.distillation`): Training a student model using signals from a teacher model or ensemble.
- **Pruning** (`mlsys.pruning`): Removing weights, neurons, channels, or other structures judged unimportant from a network, usually followed by fine-tuning to recover quality.
- **Structured pruning** (`mlsys.structured-pruning`): Pruning whole channels, filters, heads, or layers so that the remaining model stays dense and runs faster without sparse-execution support.
- **Low-rank factorization** (`mlsys.low-rank-factorization`): Approximating a weight matrix or tensor by a product of smaller factors of reduced rank to cut parameters and operations.
- **Quantization-aware training** (`mlsys.quantization-aware-training`): Training or fine-tuning with simulated quantization in the forward pass so the model adapts to reduced precision, in contrast to post-training quantization.
- **Post-training quantization** (`mlsys.post-training-quantization`): Quantizing a trained model without retraining, choosing scales and rounding from a small calibration set of representative inputs.
- **Weight-only quantization** (`mlsys.weight-only-quantization`): Storing weights at low bit width while computing in higher-precision activations, cutting memory traffic for bandwidth-bound decoding, with error-compensating or activation-aware rounding.
- **Activation outliers** (`mlsys.activation-outliers`): Rare, large-magnitude activation channels in large transformers that inflate quantization ranges, handled by mixed-precision decomposition or by migrating scale into weights.
- **KV-cache quantization** (`mlsys.kv-cache-quantization`): Storing attention keys and values at reduced precision, often with per-channel keys and per-token values, to raise batch size or context length under fixed memory.
- **N:M structured sparsity** (`mlsys.nm-sparsity`): A fine-grained pattern keeping at most N nonzero values in every group of M, such as 2:4, that matrix engines can skip for real speedup.
- **Contextual activation sparsity** (`mlsys.activation-sparsity`): Skipping neurons, heads, or channels whose activations are predicted negligible for the current input, reducing per-token compute and weight reads at inference.
- **Integer-only inference** (`mlsys.integer-only-inference`): Executing a quantized model with integer arithmetic throughout its operator graph, including accumulation and rescaling, so no floating-point operations are needed at inference time.
- **Quantization granularity** (`mlsys.quantization-granularity`): The grouping of values that share quantization parameters, such as a scale and an optional zero point, from per tensor to per channel to small groups, trading approximation error against metadata and kernel cost.

Secondary concept connections (not additional homes):

- **Numerical precision** (`mlsys.precision`)
- **Mixed-precision training** (`mlsys.mixed-precision`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Model Compression — Weight precision (`vol1.compression.precision`)
- vol1: Model Compression — Quantization (`vol1.compression.quantization`)
- vol1: Model Compression — Weight sparsity (`vol1.compression.sparsity`)
- vol1: Model Compression — Quantization (`vol1.compression.quantization`)
- vol1: Model Compression — Quantization (`vol1.compression.quantization`)
- vol1: Model Compression — Weight sparsity (`vol1.compression.sparsity`)

#### Hardware-aware model design

Identity: `hardware-aware-design`

Choosing or searching model architectures against a target device's measured cost, so that quality and efficiency are optimized together rather than compressed after the fact.

- **Activities:** Specify and design; Learn and adapt; Evaluate and assure.
- **Layers:** Model and algorithm; Numerical software and runtime; Compute and communication infrastructure.
- **Concerns:** Latency, throughput, and resources; Task and predictive quality; Monetary cost.
- **Contexts:** Deep learning; Edge and device.

Includes: Neural architecture search, including hardware-aware objectives; Efficient building blocks and the accuracy-efficiency frontier; Joint model and hardware co-design.

Excludes: Mixture-of-experts and early exit (Sparse &amp; conditional computation); Efficient attention and state-space models (Long-context &amp; efficient sequence models).

Design warrants: `mit-65940`, `dlsys`.

Candidate home concepts:

- **Neural architecture search** (`mlsys.neural-architecture-search`): Automated search over a defined space of network architectures for designs that optimize a stated objective such as accuracy under a cost budget.
- **Hardware-aware NAS** (`mlsys.hardware-aware-nas`): Architecture search whose objective includes latency, energy, or memory measured or predicted on the target device rather than proxy counts such as FLOPs.
- **Accuracy-efficiency frontier** (`mlsys.efficiency-frontier`): The set of model configurations for which no alternative improves quality without worsening a stated cost measure, such as latency, memory, or energy, on a stated platform.
- **Model-hardware co-design** (`mlsys.model-hardware-co-design`): Jointly choosing model architecture and hardware or kernel design so that each exploits the other's structure.

Secondary concept connections (not additional homes):

- **Accelerator execution architecture** (`mlsys.accelerator-execution-architecture`)
- **Arithmetic intensity** (`mlsys.arithmetic-intensity`)
- **Depthwise separable convolution** (`mlsys.depthwise-separable-convolution`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Hardware Acceleration — Arithmetic intensity (`vol1.hardware.intensity`)

### Reliability &amp; fault tolerance

How does the system keep working, or fail safely, when components fail at scale?

Infrastructure and service fault tolerance, from failure assumptions and recovery semantics to fleet failure rates, checkpoint economics, and resilience testing.

#### Failure models, retries &amp; degradation

Identity: `failure-semantics`

Declared failure assumptions and the service-level mechanisms (retries, idempotency, replication, circuit breaking, degradation) that keep requests correct and available when components fail.

- **Activities:** Specify and design; Deploy and integrate; Operate and change.
- **Layers:** Application and workflow; Numerical software and runtime; Compute and communication infrastructure.
- **Concerns:** Reliability and recovery; Latency, throughput, and resources; Task and predictive quality.
- **Contexts:** Interactive services; Distributed systems; Agentic systems.

Includes: Failure models and recovery semantics (idempotency, compensation); Retries, circuit breakers, replication, and failover; Graceful degradation of quality or features under failure.

Excludes: Physical safety fallbacks and stopping envelopes (Runtime assurance &amp; safety filters); Durable agent execution and replay (Durable execution &amp; agent state).

Design warrants: `sre`.

Candidate home concepts:

- **Failure model** (`mlsys.failure-model`): The declared classes of faults and timing assumptions under which a mechanism is analyzed.
- **Idempotency** (`mlsys.idempotency`): A specified operation's repeated execution having the same intended effect as one execution.
- **Compensating action** (`mlsys.compensation`): An explicit action intended to remedy an earlier effect whose direct reversal is unavailable.
- **Retry with backoff** (`mlsys.retry-backoff`): Reissuing a failed request after increasing, typically jittered, delays and within a retry budget, so that transient faults recover without overloading dependencies.
- **Circuit breaker** (`mlsys.circuit-breaker`): A client-side guard that stops sending requests to a failing dependency after an error threshold and probes it before resuming traffic.
- **Replication and failover** (`mlsys.replication-failover`): Maintaining redundant replicas of a service or state and redirecting work to a healthy replica when one fails.
- **Graceful degradation** (`mlsys.graceful-degradation`): Continuing to serve with reduced quality, features, or freshness under failure or overload instead of failing outright.
- **Health checks and readiness probes** (`mlsys.health-checks`): Periodic liveness and readiness checks that let an orchestrator or load balancer stop routing traffic to, or restart, a replica that is unhealthy or still loading its model.
- **Metastable failure** (`mlsys.metastable-failure`): A state in which a transient trigger, such as a load spike or slowdown, starts a self-sustaining loop of retries, cache misses, or timeouts that keeps a system overloaded after the trigger is gone.

Secondary concept connections (not additional homes):

- **Backpressure** (`mlsys.backpressure`)
- **Admission control** (`mlsys.admission`)
- **Runtime assurance** (`mlsys.runtime-assurance`)

Related MLSysBook passages (teaching depth not assessed):

- vol3: Tool Calling — Operation identity (`vol3.operation`)
- vol3: Failure Recovery — Semantic compensation (`vol3.compensation`)
- vol4: Safety Enforcement — Runtime safety filter (`vol4.enforcement.filter`)

#### Checkpointing &amp; elastic recovery

Identity: `checkpoint-recovery`

Persisting recoverable state for long-running jobs and choosing checkpoint frequency and recovery strategy to minimize lost work and restart cost under a given failure rate.

- **Activities:** Learn and adapt; Operate and change.
- **Layers:** Numerical software and runtime; Compute and communication infrastructure.
- **Concerns:** Reliability and recovery; Monetary cost; Latency, throughput, and resources.
- **Contexts:** Distributed systems; Deep learning.

Includes: Checkpoints and consistent cuts for distributed state; Optimal checkpoint interval (Young-Daly) and asynchronous checkpointing; Elastic recovery on a changed set of workers.

Excludes: Activation recomputation, sometimes called gradient checkpointing (Training memory &amp; state); Checkpoint storage bandwidth and I/O systems (Storage &amp; data delivery).

Design warrants: `sre`, `daly-checkpoint`.

Candidate home concepts:

- **Recovery checkpoint** (`mlsys.checkpoint`): A persisted state from which a specified computation can resume.
- **Consistent checkpoint cut** (`mlsys.consistent-cut`): A recoverable collection of distributed states with compatible dependencies.
- **Optimal checkpoint interval** (`mlsys.checkpoint-interval`): The checkpoint period that minimizes expected lost time, balancing checkpoint write cost against expected rework, approximately the square root of twice the write cost times MTBF.
- **Asynchronous checkpointing** (`mlsys.async-checkpointing`): Snapshotting training state quickly to host memory and persisting it in the background so computation resumes before the durable write completes.
- **Checkpoint resharding** (`mlsys.checkpoint-resharding`): Loading a checkpoint saved under one parallel layout or world size into another by remapping sharded tensors, needed for elastic restarts and training-to-serving handoff.
- **Incremental checkpointing** (`mlsys.incremental-checkpointing`): Persisting only the state that changed since the preceding checkpoint, with periodic full checkpoints as bases, which cuts write volume when updates are sparse at the cost of reconstructing state from a chain on restore.

Secondary concept connections (not additional homes):

- **Training state** (`mlsys.training-state`)
- **Training goodput** (`mlsys.training-goodput`)
- **Elastic training** (`mlsys.elastic-training`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Neural Computation — Training state (`vol1.computation.state`)
- vol2: Data Storage — Durable checkpoint (`vol2.storage.durable-checkpoint`)
- vol3: Durable Execution — Recovery checkpoint (`vol3.checkpoint`)
- vol2: Fault Tolerance — Consistent checkpoint cut (`vol2.reliability.consistent-cut`)
- vol2: Fleet Efficiency — ML goodput (`vol2.goodput.ml`)

#### Hardware faults &amp; silent data corruption

Identity: `hardware-faults`

How hardware fails in a fleet, from fail-stop crashes to gray and silent faults, how often it fails, and how such faults are detected and contained.

- **Activities:** Learn and adapt; Deploy and integrate; Operate and change.
- **Layers:** Compute and communication infrastructure; Numerical software and runtime.
- **Concerns:** Reliability and recovery; Task and predictive quality.
- **Contexts:** Distributed systems.

Includes: Fail-stop, gray, and silent failures and their detection; Fleet failure rates (MTBF composition) and availability; Correlated failures and failure domains.

Excludes: Distribution shift and adversarial robustness (Drift, feedback loops &amp; retraining; Threat models &amp; adversarial attacks); Chip power and thermal limits as design constraints (Chip power &amp; thermal limits).

Design warrants: `sre`, `sdc-scale`.

Candidate home concepts:

- **Silent data corruption** (`mlsys.silent-data-corruption`): Hardware producing wrong computational results without raising an error or crash, so the corruption propagates undetected.
- **Mean time between failures** (`mlsys.mtbf`): The expected operating time between failures of a component or system; for independent components, system MTBF shrinks roughly in inverse proportion to component count.
- **Availability** (`mlsys.availability`): The fraction of time or requests for which a system performs its specified function, approximately MTBF divided by MTBF plus mean time to repair.
- **Gray failure** (`mlsys.gray-failure`): A partial or degraded failure that some observers detect and others do not, such as a slow or lossy node that still passes health checks.
- **Failure domain** (`mlsys.failure-domain`): A set of components that can fail together from a shared cause, such as a rack, power feed, switch, or software release.
- **Algorithm-based fault tolerance** (`mlsys.abft`): Detecting or correcting computational errors by carrying checksums through an algorithm, such as matrix multiplication, and verifying invariants on its outputs.
- **Distributed job hang diagnosis** (`mlsys.distributed-hang-diagnosis`): Detecting and localizing stalled distributed jobs through collective timeouts, heartbeats, per-rank progress traces, and node health checks, so faulty ranks can be isolated.
- **Flash write endurance** (`mlsys.flash-endurance`): Flash memory cells wear out after a limited number of program and erase cycles, which bounds how often embedded and edge devices can write logs, checkpoints, or model updates; wear leveling spreads writes to extend life.
- **Error-correcting code memory** (`mlsys.ecc-memory`): Memory protected by redundant check bits that correct supported error patterns and report detectable uncorrectable errors, reducing the risk of silent corruption.

Secondary concept connections (not additional homes):

- **Stragglers** (`mlsys.stragglers`)
- **Thermal behavior and throttling** (`mlsys.thermal-dynamics`)

Related MLSysBook passages (teaching depth not assessed):

- vol2: Fault Tolerance — Job failure rate (`vol2.reliability.failure-rate`)
- vol4: Workload Placement — Thermal throttling (`vol4.placement.throttling`)

#### Fault injection &amp; resilience testing

Identity: `fault-injection`

Deliberately introducing software, infrastructure, or hardware faults under controlled scope to verify that detection, recovery, and degradation work as designed.

- **Activities:** Evaluate and assure; Operate and change.
- **Layers:** Application and workflow; Numerical software and runtime; Compute and communication infrastructure.
- **Concerns:** Reliability and recovery; Safety.
- **Contexts:** Distributed systems; Interactive services; Physical and control systems.

Includes: Chaos engineering and failure injection in production-like settings; Game days and recovery rehearsals; Hardware fault injection to estimate model error resilience.

Excludes: Functional ML tests and test oracles (ML testing &amp; debugging); Scenario-based safety validation (Safety verification &amp; validation).

Design warrants: `chaos-principles`, `sre-testing`.

Candidate home concepts:

- **Failure injection** (`mlsys.failure-injection`): Deliberately introducing selected faults or disruptive conditions to observe system response under controlled experimental boundaries.
- **Chaos engineering** (`mlsys.chaos-engineering`): Running controlled experiments that inject real-world failures into production-like systems to test a hypothesis about steady-state behavior.
- **Game day** (`mlsys.game-day`): A scheduled exercise in which teams deliberately trigger failure scenarios to rehearse detection, response, and recovery procedures.
- **Hardware fault injection** (`mlsys.hardware-fault-injection`): Emulating hardware faults such as bit flips in weights, activations, or registers to measure how errors propagate to model outputs.
- **Blast radius** (`mlsys.blast-radius`): The extent of users, components, or effects that a single fault, change, or experiment can impact.

Secondary concept connections (not additional homes):

- **Incident response** (`mlsys.incident-response`)
- **Integration testing** (`mlsys.integration-testing`)

Related MLSysBook passages (teaching depth not assessed):

- vol4: Adversarial Verification — Fault injection (`vol4.verification.fault`)

### Security &amp; privacy

How do we protect ML systems and the data they touch from adversaries and leakage?

Adversaries and defenses across data, model, artifact, and execution boundaries, and the leakage of information about people or records through learned systems.

#### Threat models &amp; adversarial attacks

Identity: `ml-attacks`

Specifying adversaries against learned systems and the attacks they mount on training data, model inputs, and instruction boundaries, with the defenses aimed at each.

- **Activities:** Specify and design; Acquire and prepare data; Learn and adapt; Evaluate and assure; Deploy and integrate; Operate and change.
- **Layers:** Application and workflow; Model and algorithm; People and operating environment.
- **Concerns:** Security; Reliability and recovery; Safety.
- **Contexts:** Deep learning; Generative systems; Agentic systems.

Includes: Threat models for ML assets and interfaces; Poisoning, backdoors, evasion, jailbreaks, and prompt injection; Defenses such as adversarial training.

Excludes: Robustness evaluation methodology (Sliced &amp; robustness evaluation); Theft and integrity of model artifacts (Model &amp; supply-chain security).

Design warrants: `aml`, `mitre-atlas`.

Candidate home concepts:

- **Threat modeling** (`mlsys.threat-model`): Specifying adversaries, assets, interfaces, capabilities, and failure consequences.
- **Data poisoning** (`mlsys.poisoning`): Adversarial manipulation of learning data or signals to alter learned behavior.
- **Adversarial evasion** (`mlsys.evasion`): Adversarially chosen inference inputs intended to cause harmful model behavior.
- **Prompt injection** (`mlsys.prompt-injection`): Untrusted content attempting to redirect model-driven behavior across an instruction or authority boundary.
- **Backdoor attack** (`mlsys.backdoor`): Implanting a hidden behavior in a model, via poisoned data or modified weights, that activates only on inputs containing an attacker-chosen trigger.
- **Jailbreak** (`mlsys.jailbreak`): A user-crafted input that induces a model to bypass its safety training or usage policy and produce disallowed output.
- **Adversarial training** (`mlsys.adversarial-training`): Training on adversarially perturbed inputs generated during training to increase robustness within a specified perturbation set.
- **Red teaming** (`mlsys.red-teaming`): Structured adversarial probing of a model or system, by people or automated attacker models, to elicit harmful, insecure, or policy-violating behavior before and after release.
- **Byzantine-robust aggregation** (`mlsys.byzantine-robust-aggregation`): Combining worker or client updates with robust statistics, such as medians, trimmed means, or Krum, so a bounded fraction of arbitrary malicious updates cannot steer training.
- **Sensor spoofing** (`mlsys.sensor-spoofing`): Attacks that inject false signals into a sensor's physical channel, such as fabricated LiDAR returns or GPS and acoustic spoofing, so perception receives counterfeit measurements.
- **Certified robustness by randomized smoothing** (`mlsys.randomized-smoothing`): Classifying many noise-perturbed copies of an input and taking the majority vote, which yields a provable radius within which no adversarial perturbation can change the prediction, at the cost of many inferences per input.
- **Resource-exhaustion attack** (`mlsys.resource-exhaustion-attack`): Crafting inputs that maximize a model service's compute, energy, output length, or tool calls, degrading service for other users or running up the operator's cost.

Secondary concept connections (not additional homes):

- **Robustness evaluation** (`mlsys.robustness-evaluation`)

Related MLSysBook passages (teaching depth not assessed):

- vol2: Security &amp; Privacy — Data poisoning (`vol2.security.poisoning`)
- vol3: Agent Security — Prompt injection (`vol3.injection`)

#### Model &amp; supply-chain security

Identity: `model-supply-chain`

Protecting models as valuable artifacts from theft and tampering, and protecting the chain of datasets, pretrained weights, libraries, and builds that produce them.

- **Activities:** Learn and adapt; Deploy and integrate; Operate and change.
- **Layers:** Model and algorithm; Numerical software and runtime; Compute and communication infrastructure.
- **Concerns:** Security; Reproducibility.
- **Contexts:** Deep learning; Generative systems.

Includes: Model extraction and ownership evidence (watermarking); Artifact signing and provenance; Malicious weights, serialization formats, and dependencies.

Excludes: Artifact versioning and registries as release tooling (Versioning, rollout &amp; rollback); Data lineage as a data-engineering practice (Data contracts, versioning &amp; lineage).

Design warrants: `aml`, `mitre-atlas`.

Candidate home concepts:

- **Model extraction** (`mlsys.model-extraction`): Reconstructing a functionally equivalent copy or the parameters of a model by querying its prediction interface.
- **Model watermarking** (`mlsys.model-watermarking`): Embedding a verifiable signature in a model's parameters or behavior so that ownership of copies or derivatives can later be demonstrated.
- **Model artifact signing** (`mlsys.artifact-signing`): Cryptographically signing model files and their provenance metadata so that consumers can verify integrity and origin before loading.
- **Unsafe model deserialization** (`mlsys.unsafe-deserialization`): Code execution triggered by loading a model stored in an executable serialization format, such as pickle, from an untrusted source.
- **ML supply-chain compromise** (`mlsys.supply-chain-compromise`): Introducing malicious or vulnerable components into an ML system through third-party datasets, pretrained weights, libraries, build tools, or hardware.

Secondary concept connections (not additional homes):

- **Model artifact** (`mlsys.model-artifact`)
- **Data lineage** (`mlsys.lineage`)
- **Dependency environment** (`mlsys.dependency-environment`)

Related MLSysBook passages (teaching depth not assessed):

- vol3: Long-Term Memory — Source provenance (`vol3.provenance`)

#### Privacy leakage &amp; differential privacy

Identity: `privacy-dp`

Privacy requirements for data about people, the attacks that extract it from models and outputs, and formal or architectural mechanisms that bound or remove that leakage.

- **Activities:** Specify and design; Acquire and prepare data; Learn and adapt; Evaluate and assure; Retire and decommission.
- **Layers:** Application and workflow; Model and algorithm; People and operating environment.
- **Concerns:** Privacy; Security.
- **Contexts:** Federated settings; Generative systems; Edge and device.

Includes: Membership inference and model inversion; Differential privacy and DP-SGD; Machine unlearning and secure aggregation.

Excludes: Federated training protocols and client scheduling (Federated &amp; decentralized learning); Privacy regulation and governance processes (AI governance, audit &amp; regulation).

Design warrants: `dp`, `nist-rmf`.

Candidate home concepts:

- **Privacy requirements** (`mlsys.privacy`): Requirements governing the collection, use, retention, and disclosure of information about people or records.
- **Differential privacy** (`mlsys.differential-privacy`): A parameterized bound on changes in output distributions under a specified neighboring-dataset relation.
- **Privacy leakage** (`mlsys.privacy-leakage`): Information about people or records revealed through data, artifacts, or model interaction.
- **Membership inference** (`mlsys.membership-inference`): Inferring whether a specific record was in a model's training data from the model's outputs or parameters.
- **Model inversion** (`mlsys.model-inversion`): Reconstructing sensitive attributes or representative training inputs from a model's outputs, confidence scores, or shared parameter gradients.
- **DP-SGD** (`mlsys.dp-sgd`): Stochastic gradient descent with per-example gradient clipping and calibrated noise, with privacy loss tracked across steps by an accountant.
- **Machine unlearning** (`mlsys.machine-unlearning`): Removing the influence of specified training records from a trained model, either exactly by retraining or approximately at lower cost.
- **Secure aggregation** (`mlsys.secure-aggregation`): A cryptographic protocol that lets a server compute the sum of client updates without seeing any individual client's update.
- **Homomorphic-encryption inference** (`mlsys.he-inference`): Evaluating a model on encrypted inputs with homomorphic encryption so the server never sees plaintext, at large compute overhead and with polynomial approximations of nonlinearities.
- **Secure multi-party computation inference** (`mlsys.secure-computation-inference`): Jointly evaluating a model over secret-shared inputs or weights so no party learns another's data, with cost dominated by communication rounds for nonlinear layers.
- **Training data extraction** (`mlsys.training-data-extraction`): Recovering particular training examples or their sensitive contents through a trained model's outputs, exposing information that was not intended for release.
- **PII detection and de-identification** (`mlsys.pii-deidentification`): Finding personal identifiers in training corpora, logs, and prompts and removing, masking, or pseudonymizing them before the data is stored, trained on, or shared.

Secondary concept connections (not additional homes):

- **Federated learning** (`mlsys.federated-learning`)

Related MLSysBook passages (teaching depth not assessed):

- vol2: Edge Intelligence — Federated averaging (`vol2.federated.fedavg`)
- vol2: Security &amp; Privacy — Differential privacy (`vol2.privacy.dp`)
- vol2: Edge Intelligence — Secure aggregation (`vol2.federated.secure-aggregation`)
- vol2: Security &amp; Privacy — Privacy budget (`vol2.privacy.budget`)

#### Access control &amp; isolation

Identity: `access-isolation`

Constraining who may act, what code may reach, and which parties can observe data in use, through authorization, least privilege, isolation, and hardware-rooted trust.

- **Activities:** Specify and design; Deploy and integrate; Operate and change.
- **Layers:** Application and workflow; Numerical software and runtime; Compute and communication infrastructure.
- **Concerns:** Security; Privacy; Human control and accountability.
- **Contexts:** Agentic systems; Distributed systems.

Includes: Authorization, least privilege, and sandboxed execution; Confidential computing and remote attestation; Side channels across shared accelerators and caches.

Excludes: Tenant quotas and resource isolation for performance (Capacity planning &amp; multi-tenancy); Agent tool contracts and action approval (Tool use &amp; action interfaces).

Design warrants: `sre`, `agents`.

Candidate home concepts:

- **Authorization** (`mlsys.authorization`): Determining whether an actor may perform a specified action on a resource.
- **Execution isolation** (`mlsys.isolation`): Restricting which resources and effects a computation can access.
- **Least privilege** (`mlsys.least-privilege`): Granting each component or actor only the permissions its task requires, for only as long as it requires them.
- **Confidential computing** (`mlsys.confidential-computing`): Protecting data and models while in use by running computation inside hardware-isolated, attestable trusted execution environments.
- **Remote attestation** (`mlsys.remote-attestation`): A procedure by which a platform proves to a remote verifier the identity and measured state of its hardware and software before receiving secrets.
- **Side-channel attack** (`mlsys.side-channel`): Inferring secrets from observable resource behavior, such as timing, power, memory, or cache contention, across an isolation boundary.
- **Capability-based authorization** (`mlsys.capability-based-security`): Authorization in which holding an unforgeable, scoped reference grants a specific right that can be delegated with attenuated scope and later revoked.
- **Ephemeral sandbox** (`mlsys.ephemeral-sandbox`): An isolated execution environment created per task from a snapshot, often copy-on-write, and destroyed or reset afterward so no state carries across tasks or tenants.
- **Network egress control** (`mlsys.egress-control`): Restricting which external network destinations a workload or sandbox may contact, typically through default-deny allowlists, to block exfiltration and unintended remote actions.
- **Information-flow control** (`mlsys.information-flow-control`): Labeling data with confidentiality and integrity tags and propagating them through computation so policy can block untrusted inputs from steering privileged actions or secrets from leaking.
- **Workload identity and secrets management** (`mlsys.secrets-management`): Giving training jobs, model servers, and pipelines verifiable identities and provisioning, rotating, and revoking the keys and credentials they use, without embedding secrets in code or images.
- **Confused deputy** (`mlsys.confused-deputy`): A security failure in which a component uses its legitimate authority for an untrusted party's purpose because it does not distinguish the requester's authority from its own.

Secondary concept connections (not additional homes):

- **Tool contract** (`mlsys.tool-contract`)

Related MLSysBook passages (teaching depth not assessed):

- vol3: Agent Sandboxes — Capability grant (`vol3.capability`)
- vol3: Agent Sandboxes — Isolation envelope (`vol3.isolation`)
- vol4: Workload Placement — Resource isolation (`vol4.placement.isolation`)
- vol3: Agent Sandboxes — Capability grant (`vol3.capability`)
- vol3: Agent Sandboxes — Egress control (`vol3.egress`)
- vol3: Agent Security — Information-flow policy (`vol3.flow`)

### Safety &amp; assurance

How do we argue, enforce, and verify that a system will not cause unacceptable harm?

Identifying hazards, bounding the operating domain, enforcing safety at runtime, verifying and validating behavior, and arguing acceptable residual risk against recognized standards.

#### Hazard analysis &amp; operating domains

Identity: `hazards-operating-domain`

Identifying how a system can contribute to harm, including hazards from functional insufficiency rather than faults, and fixing the operating conditions within which safety claims are made.

- **Activities:** Specify and design; Evaluate and assure; Operate and change.
- **Layers:** Application and workflow; Model and algorithm; People and operating environment.
- **Concerns:** Safety; Reliability and recovery; Human control and accountability.
- **Contexts:** Physical and control systems; Agentic systems; Deep learning.

Includes: Hazard identification and analysis (STPA, FMEA, fault trees) applied to learned components.; Operational design domains and triggering conditions for functional insufficiency (SOTIF)..

Excludes: Adversary-driven threat modeling (Threat models &amp; adversarial attacks); Infrastructure fault models (Failure models, retries &amp; degradation).

Design warrants: `stpa-leveson`, `iso21448`, `concrete-safety`.

Candidate home concepts:

- **Operational design domain** (`mlsys.operating-domain`): The conditions under which a physical system's behavior and evidence are intended to apply.
- **Hazard analysis** (`mlsys.hazard-analysis`): Systematically identifying conditions under which a system could contribute to harm, with their causes, severity, and required controls.
- **System-theoretic process analysis** (`mlsys.stpa`): Hazard analysis that models a system as a control structure and identifies unsafe control actions and the flawed feedback or process models that cause them.
- **Failure modes and effects analysis** (`mlsys.fmea`): Enumerating each component's possible failure modes and tracing their effects on system behavior to rank and mitigate them.
- **Functional insufficiency (SOTIF)** (`mlsys.functional-insufficiency`): Hazardous behavior arising without any fault, from limits of a specification or of a component's performance, such as perception errors under particular triggering conditions.
- **Fault tree analysis** (`mlsys.fault-tree-analysis`): Top-down hazard analysis that decomposes an undesired top event into Boolean combinations of contributing faults, so minimal cut sets and their probabilities can be computed.

Secondary concept connections (not additional homes):

- **Failure model** (`mlsys.failure-model`)
- **Threat modeling** (`mlsys.threat-model`)
- **Distribution shift** (`mlsys.distribution-shift`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: ML Operations — Distribution shift (`vol1.operations.drift`)
- vol2: Robust AI — Distribution shift (`vol2.robustness.shift`)
- vol4: Physical Data — Endogenous covariate shift (`vol4.learning.shift`)
- vol4: Physical Data — Operational design domain (`vol4.deployment.odd`)

#### Runtime assurance &amp; safety filters

Identity: `runtime-enforcement`

Mechanisms that monitor a learned component at run time and override, constrain, or replace its outputs so the system stays within a verified safe set.

- **Activities:** Specify and design; Deploy and integrate; Operate and change.
- **Layers:** Application and workflow; Numerical software and runtime; People and operating environment.
- **Concerns:** Safety; Reliability and recovery; Human control and accountability.
- **Contexts:** Physical and control systems; Agentic systems; Generative systems; Interactive services.

Includes: Safety filters, control barrier functions, Simplex switching, and fallback to a minimal-risk condition.; Input and output guardrails that block or rewrite unsafe generated content or actions..

Excludes: Human approval of agent actions (Agent harness &amp; control loop); Retry, failover, and graceful degradation for availability (Reliability &amp; fault tolerance).

Design warrants: `safety-filter`, `cbf`, `simplex`.

Candidate home concepts:

- **Runtime assurance** (`mlsys.runtime-assurance`): Monitoring and constraining execution using explicitly stated safety premises and fallback mechanisms.
- **Stopping envelope** (`mlsys.stopping-envelope`): The state and clearance conditions within which a specified stopping response remains feasible.
- **Safety filter** (`mlsys.safety-filter`): A runtime component that passes a proposed action when it is certified safe and otherwise replaces it with a minimally different safe action.
- **Control barrier function** (`mlsys.control-barrier-function`): A function whose sign defines a safe set and whose derivative condition yields input constraints that keep the system state inside that set.
- **Simplex architecture** (`mlsys.simplex-architecture`): Pairing an unverified high-performance controller with a verified safety controller and a decision module that switches to the safety controller before safety is lost.
- **Minimal-risk condition** (`mlsys.minimal-risk-condition`): A predefined reduced-capability state the system reaches when it can no longer operate safely, often via an ordered ladder of fallback behaviors.
- **Guardrails** (`mlsys.guardrails`): Runtime checks on model inputs and outputs that detect and block, rewrite, or escalate content or actions violating a stated safety policy.
- **Over-refusal** (`mlsys.over-refusal`): Refusing benign requests that resemble harmful ones, the false-positive side of a safety filter or refusal policy, which is traded against missed harmful requests when setting the operating point.

Secondary concept connections (not additional homes):

- **Trajectory feasibility** (`mlsys.trajectory-feasibility`)
- **Selective prediction** (`mlsys.selective-prediction`)
- **Action approval gate** (`mlsys.approval-gate`)

Related MLSysBook passages (teaching depth not assessed):

- vol2: Robust AI — Selective prediction (`vol2.robustness.selective-prediction`)
- vol4: Trajectory Planning — Kinodynamic feasibility (`vol4.motion.feasibility`)
- vol4: Safety Enforcement — Runtime safety filter (`vol4.enforcement.filter`)
- vol4: Safety Enforcement — Stopping envelope (`vol4.enforcement.envelope`)
- vol3: Agent Security — Action-bound approval (`vol3.approval`)
- vol4: The Causal Boundary — Stopping distance (`vol4.stopping.distance`)
- vol4: Safety Enforcement — Runtime safety filter (`vol4.enforcement.filter`)

#### Safety verification &amp; validation

Identity: `verification-validation`

Producing evidence that a system meets its safety requirements across its operating domain, through scenario testing, falsification, formal proof, and statistical validation of rare failures.

- **Activities:** Evaluate and assure; Deploy and integrate.
- **Layers:** Application and workflow; Model and algorithm; People and operating environment.
- **Concerns:** Safety; Reliability and recovery; Task and predictive quality.
- **Contexts:** Physical and control systems; Deep learning.

Includes: Scenario coverage, simulation-based falsification, and software/hardware-in-the-loop qualification.; Formal and statistical guarantees, including verification of neural networks and rare-event estimation..

Excludes: Average-case model metrics and benchmarks (Model evaluation &amp; testing).

Design warrants: `safety-validation`, `nn-verification`, `driving-to-safety`.

Candidate home concepts:

- **Scenario-based testing** (`mlsys.scenario-based-testing`): Validating a system against a structured catalog of operating scenarios and their parameter ranges, with coverage measured over the operating domain.
- **Simulation-based falsification** (`mlsys.falsification`): Searching scenario parameters, often guided by temporal-logic robustness, to find counterexamples that violate a safety specification.
- **Formal verification** (`mlsys.formal-verification`): Mathematically proving that a model of a system satisfies a formal specification for all behaviors within stated assumptions.
- **Neural network verification** (`mlsys.neural-network-verification`): Proving or refuting input-output properties of a trained network over a bounded input region, such as local robustness, using exact or bound-propagation methods.
- **Rare-event safety validation** (`mlsys.rare-event-validation`): Estimating or bounding failure probabilities too small to observe in feasible exposure, using importance sampling, extreme-value models, or zero-failure statistical bounds.
- **X-in-the-loop testing** (`mlsys.xil-testing`): A qualification ladder that runs the real software, then real hardware, against simulated or recorded environments before physical trials.
- **Closed-loop policy evaluation** (`mlsys.closed-loop-evaluation`): Evaluating a policy by letting its own actions determine subsequent observations, in simulation or deployment, rather than scoring its predictions against logged data.

Secondary concept connections (not additional homes):

- **Robustness evaluation** (`mlsys.robustness-evaluation`)
- **Failure injection** (`mlsys.failure-injection`)
- **Simulation-to-real transfer** (`mlsys.sim-to-real`)
- **Test oracle** (`mlsys.test-oracle`)

Related MLSysBook passages (teaching depth not assessed):

- vol4: Policy Training — Reality gap (`vol4.learning.reality-gap`)
- vol3: Trajectory Fine-Tuning — Closed-loop evaluation (`vol3.closed-loop`)
- vol4: Closed-Loop Evaluation — Closed-loop trial (`vol4.evaluation.trial`)
- vol4: Closed-Loop Evaluation — Failure-rate bound (`vol4.evaluation.confidence`)
- vol4: Adversarial Verification — Fault injection (`vol4.verification.fault`)
- vol4: Adversarial Verification — Test oracle (`vol4.verification.oracle`)

#### Safety cases, standards &amp; certification

Identity: `safety-cases`

Structured arguments that a system is acceptably safe for a defined use, the standards that set what evidence is required, and the field indicators that keep the argument valid after release.

- **Activities:** Specify and design; Evaluate and assure; Deploy and integrate; Operate and change.
- **Layers:** Application and workflow; People and operating environment.
- **Concerns:** Safety; Human control and accountability; Reliability and recovery.
- **Contexts:** Physical and control systems; Agentic systems.

Includes: Safety-case structure and notation (claims, arguments, evidence; GSN).; Functional-safety and autonomy standards (ISO 26262, IEC 61508, ISO 21448, UL 4600) and integrity levels..

Excludes: Regulatory compliance not specific to safety, such as the EU AI Act (AI governance, audit &amp; regulation).

Design warrants: `gsn`, `iso26262`, `ul4600`.

Candidate home concepts:

- **Safety case** (`mlsys.safety-case`): A structured argument connecting claims, evidence, assumptions, and defeaters for a defined system.
- **Goal Structuring Notation** (`mlsys.goal-structuring-notation`): A graphical notation for safety cases that links goals, strategies, solutions (evidence), context, and assumptions into an explicit argument structure.
- **Functional safety** (`mlsys.functional-safety`): Absence of unreasonable risk from malfunctioning electrical or electronic systems, achieved through a standardized safety lifecycle such as IEC 61508 or ISO 26262.
- **Safety integrity level** (`mlsys.safety-integrity-level`): A graded risk class assigned to a safety function that sets the rigor of development, verification, and tolerable failure rate required.
- **Safety performance indicator** (`mlsys.safety-performance-indicator`): A field-measured metric tied to a safety-case claim whose threshold violation signals that the argument may no longer hold.
- **Event data recorder** (`mlsys.event-data-recorder`): Tamper-evident onboard logging of inputs, decisions, control authority, and safety interventions around events, retained for incident reconstruction and assurance evidence.
- **Safety change-impact analysis** (`mlsys.change-impact-analysis`): Determining which hazards, safety-case claims, and evidence a change to a model, software, data, or operating domain invalidates, and what must be re-verified.

Secondary concept connections (not additional homes):

- **Operational design domain** (`mlsys.operating-domain`)
- **Staged rollout** (`mlsys.rollout`)
- **Artifact versioning** (`mlsys.artifact-version`)

Related MLSysBook passages (teaching depth not assessed):

- vol2: Data Storage — Versioned artifact (`vol2.storage.versioned-artifact`)
- vol4: Deployment Release — Release manifest (`vol4.assurance.manifest`)
- vol2: ML Operations at Scale — Staged rollout (`vol2.operations.rollout`)
- vol3: Operating Agents — Staged release (`vol3.release`)
- vol4: Physical Data — Operational design domain (`vol4.deployment.odd`)
- vol4: Deployment Release — Safety case (`vol4.assurance.case`)
- vol2: Responsible AI — Assurance case (`vol2.governance.assurance`)
- vol4: Safety Enforcement — Certified safety function (`vol4.enforcement.certified-floor`)
- vol4: Safety Enforcement — Certified safety function (`vol4.enforcement.certified-floor`)

### Responsible AI

How do we ensure the system's outcomes are fair, accountable, transparent, and governed?

Measuring and mitigating unfair outcomes, explaining and documenting models and data, keeping people able to oversee and contest decisions, and governing AI risk under organizational and legal regimes.

#### Fairness &amp; bias

Identity: `fairness-bias`

Specifying, measuring, and mitigating how a system's outcomes and errors differ across people and groups.

- **Activities:** Specify and design; Acquire and prepare data; Learn and adapt; Evaluate and assure; Operate and change.
- **Layers:** Application and workflow; Model and algorithm; People and operating environment.
- **Concerns:** Fairness; Task and predictive quality; Human control and accountability.
- **Contexts:** Classical ML; Deep learning; Generative systems.

Includes: Group and individual fairness criteria and the metrics that operationalize them.; Pre-, in-, and post-processing mitigation and its accuracy cost..

Excludes: Per-slice measurement machinery in general (Sliced &amp; robustness evaluation); Scheduler fair sharing (Cluster scheduling &amp; orchestration).

Design warrants: `fairmlbook`, `fair-tradeoffs`, `nist-rmf`.

Candidate home concepts:

- **Fairness criteria** (`mlsys.fairness`): Explicit requirements for how system outcomes or errors are distributed across affected groups.
- **Group fairness metrics** (`mlsys.group-fairness-metric`): Statistics comparing selection rates, error rates, or calibration of a model's decisions across protected groups.
- **Individual fairness** (`mlsys.individual-fairness`): A requirement that individuals who are similar with respect to the task receive similar outcomes under a stated similarity metric.
- **Fairness criteria incompatibility** (`mlsys.fairness-incompatibility`): The result that calibration and error-rate parity across groups cannot generally hold together when base rates differ, forcing an explicit choice among criteria.
- **Bias mitigation** (`mlsys.bias-mitigation`): Interventions on data (pre-processing), training objective (in-processing), or decisions (post-processing) that reduce measured disparities, usually at some cost in other metrics.
- **Representational bias evaluation** (`mlsys.representational-bias-evaluation`): Measuring stereotyped, demeaning, or unequal portrayals of groups in generated outputs, revealing representational harms that aggregate quality or decision-rate metrics can miss.
- **Proxy discrimination** (`mlsys.proxy-discrimination`): Decisions driven by features that encode protected-group membership, so group disparities persist even when the protected attribute itself is omitted.

Secondary concept connections (not additional homes):

- **Sliced evaluation** (`mlsys.sliced-evaluation`)
- **Deployment feedback loop** (`mlsys.feedback-loop`)
- **Proxy objective** (`mlsys.proxy-objective`)
- **Data acquisition and sampling** (`mlsys.data-acquisition`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Responsible Engineering — Bias feedback (`vol1.responsibility.feedback`)
- vol1: Responsible Engineering — Subgroup fairness (`vol1.responsibility.fairness`)
- vol2: Responsible AI — Fairness criterion (`vol2.governance.fairness`)
- vol1: Benchmarking — Model quality (`vol1.measurement.quality`)
- vol1: Responsible Engineering — Subgroup fairness (`vol1.responsibility.fairness`)
- vol1: Responsible Engineering — Subgroup fairness (`vol1.responsibility.fairness`)
- vol2: Responsible AI — Fairness criterion (`vol2.governance.fairness`)

#### Explainability &amp; documentation

Identity: `transparency-explainability`

Making model behavior understandable to the people who build, deploy, and are affected by it, through explanations, interpretable models, and standardized documentation of models and data.

- **Activities:** Acquire and prepare data; Learn and adapt; Evaluate and assure; Deploy and integrate.
- **Layers:** Application and workflow; Model and algorithm.
- **Concerns:** Human control and accountability; Fairness; Task and predictive quality.
- **Contexts:** Classical ML; Deep learning; Generative systems.

Includes: Post-hoc explanations (feature attribution, counterfactuals) and inherently interpretable models.; Model cards, datasheets, and the serving cost of producing explanations..

Excludes: Debugging-oriented model inspection (ML testing &amp; debugging); Data provenance records (Data contracts, versioning &amp; lineage).

Design warrants: `iml-book`, `model-cards`, `datasheets`.

Candidate home concepts:

- **Feature attribution** (`mlsys.feature-attribution`): Post-hoc explanation assigning each input feature a contribution to a specific prediction.
- **Counterfactual explanation** (`mlsys.counterfactual-explanation`): An explanation stating the smallest feasible change to an input that would change the model's decision.
- **Inherently interpretable model** (`mlsys.interpretable-model`): A model whose structure, such as a sparse linear model, short rule list, or shallow tree, can be directly inspected to understand its predictions.
- **Model card** (`mlsys.model-card`): A standardized document reporting a model's intended use, evaluation results across conditions and groups, limitations, and ethical considerations.
- **Datasheet for datasets** (`mlsys.datasheet`): A standardized document recording a dataset's motivation, composition, collection process, preprocessing, intended uses, and maintenance.

Secondary concept connections (not additional homes):

- **Data lineage** (`mlsys.lineage`)
- **Probability calibration** (`mlsys.calibration`)

Related MLSysBook passages (teaching depth not assessed):

- vol3: Long-Term Memory — Source provenance (`vol3.provenance`)

#### Human oversight &amp; contestability

Identity: `human-oversight`

Arrangements that let people supervise, intervene in, take over from, and challenge consequential system decisions, and the human-factors limits on doing so.

- **Activities:** Specify and design; Deploy and integrate; Operate and change.
- **Layers:** Application and workflow; People and operating environment.
- **Concerns:** Human control and accountability; Safety; Fairness.
- **Contexts:** Classical ML; Agentic systems; Physical and control systems; Interactive services.

Includes: Human-in-the-loop and on-the-loop control, recourse and appeal, and control handover.; Automation bias, complacency, and the ironies of automation..

Excludes: The mechanics of agent approval gates (Agent harness &amp; control loop).

Design warrants: `nist-rmf`, `ironies`, `automation-misuse`.

Candidate home concepts:

- **Human oversight** (`mlsys.human-oversight`): An arrangement in which people can understand, intervene in, and contest consequential system behavior.
- **Contestability and recourse** (`mlsys.contestability`): The ability of an affected person to understand, challenge, and obtain review or reversal of a consequential automated decision.
- **Automation bias** (`mlsys.automation-bias`): The tendency of human supervisors to over-rely on automated outputs and to lose the vigilance and skill needed to catch their errors.
- **Authority handover** (`mlsys.authority-handover`): Transfer of control between an automated system and a human operator, bounded by human reaction time, situational awareness, and mode clarity.
- **Human-AI interaction design** (`mlsys.human-ai-interaction`): Designing how a system conveys its capabilities, uncertainty, and errors to users and lets them correct, override, or dismiss its outputs.

Secondary concept connections (not additional homes):

- **Action approval gate** (`mlsys.approval-gate`)
- **Selective prediction** (`mlsys.selective-prediction`)
- **Irreversible effects** (`mlsys.irreversible-effects`)

Related MLSysBook passages (teaching depth not assessed):

- vol2: Robust AI — Selective prediction (`vol2.robustness.selective-prediction`)
- vol4: Responsible Physical AI — Human oversight (`vol4.accountability.oversight`)
- vol4: The Causal Boundary — Irreversibility (`vol4.boundary.irreversibility`)
- vol2: Responsible AI — Contestability (`vol2.governance.contestability`)
- vol3: Agent Security — Action-bound approval (`vol3.approval`)
- vol4: Supervisory Intervention — Takeover delay (`vol4.intervention.delay`)
- vol4: Supervisory Intervention — Single-writer handoff (`vol4.intervention.handoff`)

#### AI governance, audit &amp; regulation

Identity: `governance-regulation`

Organizational and legal mechanisms that identify, assess, and control AI risk across a system's life, including risk-management frameworks, impact assessments, audits, and regulation.

- **Activities:** Specify and design; Evaluate and assure; Deploy and integrate; Operate and change; Retire and decommission.
- **Layers:** Application and workflow; People and operating environment.
- **Concerns:** Human control and accountability; Fairness; Safety; Privacy.
- **Contexts:** Classical ML; Deep learning; Generative systems; Agentic systems.

Includes: NIST AI RMF and ISO/IEC 42001 management systems; internal and external audits.; Risk-tiered regulation (EU AI Act) and frontier-model capability thresholds..

Excludes: Data-protection mechanisms such as differential privacy (Security &amp; privacy); Safety-standard certification (Safety cases, standards &amp; certification).

Design warrants: `nist-rmf`, `eu-ai-act`, `ai-audit`.

Candidate home concepts:

- **AI risk management** (`mlsys.ai-risk-management`): An organizational process to govern, map, measure, and manage risks of AI systems throughout their lifecycle.
- **Algorithmic impact assessment** (`mlsys.impact-assessment`): A structured pre-deployment analysis of who an AI system could affect, how, and with what severity, used to decide mitigations or whether to deploy.
- **Algorithmic audit** (`mlsys.algorithmic-audit`): An internal or independent examination of an AI system's development records and behavior against stated requirements, standards, or law.
- **Risk-tiered AI regulation** (`mlsys.risk-tiered-regulation`): Law that classifies AI uses by risk level and imposes obligations, such as conformity assessment, documentation, and human oversight, scaled to that level.
- **Frontier safety framework** (`mlsys.frontier-safety-framework`): A developer policy that ties dangerous-capability evaluation thresholds to required safeguards before further training or deployment of frontier models.
- **Decision accountability record** (`mlsys.decision-accountability`): A record of who authorized each consequential decision, policy, or automated action and on what evidence, so responsibility can be assigned and reviewed later.
- **Content provenance** (`mlsys.content-provenance`): Cryptographically signed metadata attached to media that records its origin and edit history, including model generation, so recipients can verify source claims.
- **Generated-content watermarking** (`mlsys.generated-content-watermarking`): Embedding a statistically detectable signal into generated text, images, or audio at generation time, such as by biasing token sampling, so that content can later be identified as model-generated.

Secondary concept connections (not additional homes):

- **Safety case** (`mlsys.safety-case`)
- **Privacy requirements** (`mlsys.privacy`)
- **Incident response** (`mlsys.incident-response`)

Related MLSysBook passages (teaching depth not assessed):

- vol4: Deployment Release — Safety case (`vol4.assurance.case`)
- vol2: Responsible AI — Assurance case (`vol2.governance.assurance`)
- vol4: Responsible Physical AI — Approval record (`vol4.accountability.approval`)

### Sustainability &amp; cost

What does the system cost in energy, carbon, water, and money over its lifetime?

Accounting for environmental and monetary consequences of ML systems under declared boundaries, and the decisions that reduce them per useful outcome.

#### Energy &amp; carbon accounting

Identity: `energy-carbon`

Measuring the energy a workload consumes and converting it to operational emissions under declared boundaries, and scheduling work to reduce both.

- **Activities:** Learn and adapt; Evaluate and assure; Deploy and integrate; Operate and change.
- **Layers:** Compute and communication infrastructure; People and operating environment.
- **Concerns:** Energy and environmental impacts; Monetary cost.
- **Contexts:** Distributed systems; Generative systems; Batch and offline.

Includes: Energy measurement and energy efficiency (work per joule); Grid carbon intensity and operational carbon; Carbon-aware placement and scheduling.

Excludes: Facility power, cooling, and PUE (Datacenter power, cooling &amp; facilities); Manufacturing emissions and hardware end of life (Embodied carbon &amp; lifecycle impact).

Design warrants: `green`, `bloom-carbon`.

Candidate home concepts:

- **Operating energy** (`mlsys.energy`): Energy consumed by a specified system boundary over a defined workload or interval.
- **Lifecycle carbon** (`mlsys.carbon`): Emissions attributed to operation and embodied system resources under declared accounting boundaries.
- **Energy efficiency** (`mlsys.energy-efficiency`): Useful work completed per unit energy, such as inferences or tokens per joule, measured within a stated system boundary.
- **Grid carbon intensity** (`mlsys.carbon-intensity`): Greenhouse-gas emissions per unit of electricity supplied, varying by region, time, and generation mix, in grams CO2-equivalent per kilowatt-hour.
- **Operational carbon** (`mlsys.operational-carbon`): Emissions from electricity consumed while operating a system, computed from energy use and the carbon intensity of its supply.
- **Emission scopes** (`mlsys.emission-scopes`): The GHG Protocol classification of emissions into direct (Scope 1), purchased energy (Scope 2), and value-chain (Scope 3) categories.
- **Carbon-aware scheduling** (`mlsys.carbon-aware-scheduling`): Shifting flexible workloads in time or location toward periods and regions of lower grid carbon intensity, subject to deadlines and capacity.
- **Energy per inference** (`mlsys.energy-per-inference`): Energy consumed to process one input end to end on a device, the binding metric for battery-powered and energy-harvesting deployments.
- **Energy measurement methodology** (`mlsys.energy-measurement`): The declared sensor sources, sampling rate, measurement boundary, and attribution rules by which a workload's energy is measured, from on-chip counters to facility meters.
- **Idle vs. dynamic power** (`mlsys.idle-dynamic-power`): Separating power drawn by provisioned but idle hardware from the incremental power of active work, which decides how shared energy is attributed to workloads.
- **Datacenter demand response** (`mlsys.demand-response`): Shifting, shedding, or capping datacenter power in response to grid signals, made practical by deferrable training and batch work.
- **Rebound effect** (`mlsys.rebound-effect`): Efficiency gains lower the cost of a service enough that total demand, and with it total energy or resource use, grows, partly or fully offsetting the savings.

Secondary concept connections (not additional homes):

- **Power usage effectiveness** (`mlsys.pue`)
- **Power delivery and integrity** (`mlsys.power-delivery`)

Related MLSysBook passages (teaching depth not assessed):

- vol2: Sustainable AI — Operating energy (`vol2.sustainability.energy`)
- vol1: Responsible Engineering — Operational carbon (`vol1.responsibility.carbon`)
- vol2: Sustainable AI — Embodied carbon (`vol2.sustainability.embodied-carbon`)
- vol2: Sustainable AI — Operational carbon (`vol2.sustainability.operational-carbon`)
- vol1: System Paradigms — Power budget (`vol1.system.power`)
- vol1: Responsible Engineering — Operational carbon (`vol1.responsibility.carbon`)
- vol2: Sustainable AI — Operational carbon (`vol2.sustainability.operational-carbon`)
- vol2: Sustainable AI — Operational carbon (`vol2.sustainability.operational-carbon`)
- vol2: Sustainable AI — Carbon-aware scheduling (`vol2.sustainability.carbon-scheduling`)
- vol4: The Physical Body — Supply integrity (`vol4.plant.supply`)

#### Embodied carbon &amp; lifecycle impact

Identity: `embodied-lifecycle`

Environmental impacts beyond operating energy, from manufacturing emissions and water to hardware lifetime and disposal, assessed over the full hardware lifecycle.

- **Activities:** Specify and design; Deploy and integrate; Retire and decommission.
- **Layers:** Compute and communication infrastructure; People and operating environment.
- **Concerns:** Energy and environmental impacts; Monetary cost.
- **Contexts:** Edge and device; Distributed systems.

Includes: Embodied carbon and lifecycle assessment; Water consumption of training and serving; Hardware lifetime, refresh, and e-waste.

Excludes: Operational emissions from electricity (Energy &amp; carbon accounting); Model retirement and decommissioning (Model retirement).

Design warrants: `chasing-carbon`, `sustainable-ai`.

Candidate home concepts:

- **Embodied carbon** (`mlsys.embodied-carbon`): Emissions from manufacturing, transporting, and disposing of hardware and facilities, attributed to a system independent of its operation.
- **Life cycle assessment** (`mlsys.lifecycle-assessment`): A standardized method for compiling and evaluating a product's environmental inputs, outputs, and impacts from raw materials through disposal.
- **Water footprint** (`mlsys.water-footprint`): Water withdrawn or consumed by a system, on site for cooling and off site for electricity generation, per unit of work or over its lifetime.
- **Hardware service lifetime** (`mlsys.hardware-lifetime`): The period a hardware asset stays in service before replacement, over which its embodied impacts and capital cost are amortized.
- **Electronic waste** (`mlsys.e-waste`): Discarded electronic equipment and components, including retired accelerators and servers, and the hazards and lost materials of their disposal.

Related MLSysBook passages (teaching depth not assessed):

- vol2: Sustainable AI — Embodied carbon (`vol2.sustainability.embodied-carbon`)

#### Cost, TCO &amp; unit economics

Identity: `cost-economics`

Monetary accounting for ML systems, from total cost of ownership to cost per token or accepted outcome, and the architectural choices that lower cost at fixed quality.

- **Activities:** Specify and design; Evaluate and assure; Deploy and integrate; Operate and change.
- **Layers:** Application and workflow; Compute and communication infrastructure.
- **Concerns:** Monetary cost; Latency, throughput, and resources.
- **Contexts:** Generative systems; Agentic systems; Interactive services.

Includes: Total cost of ownership and cost attribution (FinOps); Cost per token and cost per accepted outcome; Model routing and cascades chosen for cost.

Excludes: Capacity provisioning and quotas (Capacity planning &amp; multi-tenancy); Spot and preemption scheduling mechanics (Cluster scheduling &amp; orchestration).

Design warrants: `dc-computer`, `cs336`.

Candidate home concepts:

- **Cost per useful outcome** (`mlsys.unit-cost`): Total relevant expenditure divided by outcomes meeting a specified acceptance criterion.
- **Total cost of ownership** (`mlsys.tco`): The full cost of acquiring and operating a system over its life, including capital, energy, facilities, networking, staff, and amortization.
- **Cost per token** (`mlsys.cost-per-token`): Serving or API expenditure divided by tokens processed or generated, often priced separately for input and output tokens.
- **Cost attribution** (`mlsys.cost-attribution`): Allocating shared infrastructure spend to teams, models, or workloads so that owners see and act on their costs.
- **Data transfer cost** (`mlsys.data-transfer-cost`): The money charged for moving data across network, zone, region, or provider boundaries, such as cloud egress fees, which can dominate the cost of distributing data, checkpoints, or responses.

Secondary concept connections (not additional homes):

- **Capacity planning** (`mlsys.capacity-planning`)
- **Prefix caching** (`mlsys.prefix-cache`)
- **Model routing** (`mlsys.model-routing`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Model Serving — Prefix caching (`vol1.serving.prefix`)
- vol3: KV Cache Management — Prefix reuse (`vol3.prefix-reuse`)
- vol3: Agent Economics — Cost per accepted task (`vol3.cost`)
- vol1: Model Serving — Serving capacity (`vol1.serving.capacity`)

## System types

What is distinctive when the system acts as an agent or in the physical world?

### Agentic systems

What does it take to engineer models that plan and act through tools over many steps?

Context engineering and agent memory, tool and action interfaces, the agent harness and its control loop, durable agent state, and multi-agent coordination.

#### Context engineering &amp; agent memory

Identity: `context-memory`

Deciding what information enters each model call over a long task, and maintaining persistent memory the agent can write, consolidate, and retrieve across calls and sessions.

- **Activities:** Specify and design; Deploy and integrate; Operate and change.
- **Layers:** Application and workflow; Model and algorithm; Numerical software and runtime.
- **Concerns:** Task and predictive quality; Monetary cost; Latency, throughput, and resources; Security.
- **Contexts:** Agentic systems; Generative systems; Interactive services.

Includes: Working-set selection, ordering for prefix reuse, compaction, and long-context degradation.; Long-term memory write paths, consolidation, expiry, and invalidation..

Excludes: Retrieval and vector indexes in general (Retrieval &amp; vector search); KV-cache management (LLM &amp; generative inference).

Design warrants: `context-engineering`, `memgpt`, `agents`.

Candidate home concepts:

- **Context compaction** (`mlsys.context-compaction`): Replacing older parts of an agent's context with summaries or structured records so a long task fits the model's input budget.
- **Long-term agent memory** (`mlsys.agent-memory`): A persistent store an agent writes to and retrieves from across calls and sessions, with policies for consolidation, expiry, and invalidation.
- **Context selection** (`mlsys.context-selection`): Choosing which instructions, tool descriptions, retrieved items, state, and observations enter a model's limited context at each step, within a token budget.
- **Context staleness** (`mlsys.context-staleness`): Observations held in an agent's context or memory that no longer reflect the environment after later actions or external changes, yet still steer decisions unless invalidated.

Secondary concept connections (not additional homes):

- **Retrieval** (`mlsys.retrieval`)
- **Prefix caching** (`mlsys.prefix-cache`)
- **KV cache** (`mlsys.kv-cache`)
- **Prompt injection** (`mlsys.prompt-injection`)
- **Context window** (`mlsys.context`)
- **Long-context degradation** (`mlsys.context-degradation`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: Model Serving — KV cache per conversation (`vol1.serving.kv`)
- vol2: Serving at Scale — KV cache (`vol2.serving.kv-cache`)
- vol3: KV Cache Management — Retained KV state (`vol3.kv`)
- vol1: Model Serving — Prefix caching (`vol1.serving.prefix`)
- vol3: KV Cache Management — Prefix reuse (`vol3.prefix-reuse`)
- vol3: Context Engineering — Working context (`vol3.working-context`)
- vol3: Long-Term Memory — Retrieval contract (`vol3.retrieval`)
- vol3: Agent Security — Prompt injection (`vol3.injection`)
- vol3: Context Engineering — Working context (`vol3.working-context`)
- vol3: Context Engineering — Context compaction (`vol3.compaction`)
- vol3: Context Engineering — Evidence invalidation (`vol3.invalidation`)
- vol3: Long-Term Memory — Persistent memory record (`vol3.memory`)

#### Tool use &amp; action interfaces

Identity: `tool-interfaces`

The interfaces through which a model acts on the world, including structured tool calls, tool discovery protocols, computer use, and the effects and authority those actions carry.

- **Activities:** Specify and design; Deploy and integrate; Operate and change.
- **Layers:** Application and workflow; Numerical software and runtime; People and operating environment.
- **Concerns:** Reliability and recovery; Security; Safety; Human control and accountability.
- **Contexts:** Agentic systems; Generative systems.

Includes: Tool schemas, preconditions, function calling, and MCP-style discovery.; Computer-use agents and the handling of irreversible effects..

Excludes: Sandboxing and capability isolation (Access control &amp; isolation); Prompt injection (Threat models &amp; adversarial attacks).

Design warrants: `mcp`, `agents`, `building-agents`.

Candidate home concepts:

- **Tool contract** (`mlsys.tool-contract`): A specification of tool inputs, outputs, authority, effects, and failure behavior.
- **Irreversible effects** (`mlsys.irreversible-effects`): External consequences that cannot be fully undone by restoring software state, retrying, or performing a compensating action.
- **Function calling** (`mlsys.function-calling`): A model emitting a structured invocation of a declared tool, which the harness validates, executes, and returns as an observation.
- **Tool discovery protocol** (`mlsys.model-context-protocol`): A shared client-server interface through which agent hosts discover the tools, resources, and prompts external services expose, with their schemas, reducing custom integration.
- **Computer use** (`mlsys.computer-use`): An agent operating graphical software through screenshots or accessibility trees and synthesized keyboard and pointer input.
- **Agent credentials and delegated authority** (`mlsys.agent-credentials`): Issuing tool-using agents scoped, short-lived credentials that act on a user's behalf, brokering secrets outside model context so authority stays bounded and revocable.

Secondary concept connections (not additional homes):

- **Authorization** (`mlsys.authorization`)
- **Execution isolation** (`mlsys.isolation`)
- **Idempotency** (`mlsys.idempotency`)
- **Prompt injection** (`mlsys.prompt-injection`)
- **Structured output constraint** (`mlsys.grammar`)

Related MLSysBook passages (teaching depth not assessed):

- vol3: Agent Sandboxes — Capability grant (`vol3.capability`)
- vol3: Agent Sandboxes — Isolation envelope (`vol3.isolation`)
- vol4: Workload Placement — Resource isolation (`vol4.placement.isolation`)
- vol3: Tool Calling — Operation identity (`vol3.operation`)
- vol3: Agent Security — Prompt injection (`vol3.injection`)
- vol4: The Causal Boundary — Irreversibility (`vol4.boundary.irreversibility`)

#### Agent harness &amp; control loop

Identity: `agent-harness`

The runtime that owns an agent's loop of model calls and tool actions, enforcing budgets, stopping rules, and approval gates and recording the trajectory.

- **Activities:** Specify and design; Deploy and integrate; Operate and change.
- **Layers:** Application and workflow; Numerical software and runtime.
- **Concerns:** Reliability and recovery; Monetary cost; Human control and accountability; Safety.
- **Contexts:** Agentic systems; Generative systems; Interactive services.

Includes: Reason-act loops, workflows versus model-directed loops, and plan revision.; Turn, token, time, and spend budgets; stopping rules; human approval gates..

Excludes: Inference-time search and reasoning budgets per call (Test-time compute); Trajectory-level evaluation (Evaluating generative &amp; agentic systems).

Design warrants: `react`, `building-agents`, `agents`.

Candidate home concepts:

- **Agent trajectory** (`mlsys.trajectory`): A sequence of observations, model proposals, tool actions, and external outcomes for a task.
- **Agent planning** (`mlsys.planning`): Constructing and revising prospective actions in response to goals, dependencies, and observations.
- **Agent harness** (`mlsys.agent-harness`): The runtime that repeatedly calls a model, dispatches its tool actions, appends observations, and decides whether the trajectory continues, pauses, or ends.
- **Agent execution budget** (`mlsys.agent-budget`): Limits on turns, tokens, wall-clock time, or spend for a trajectory, with stopping rules applied when a limit or completion criterion is reached.
- **Action approval gate** (`mlsys.approval-gate`): A harness checkpoint that holds a specific proposed action until an authorized human or policy approves it, binding the approval to that exact action.
- **Agent interruption and cancellation** (`mlsys.agent-interruption`): Operator controls that pause, redirect, cancel, or terminate a running agent at defined boundaries while settling in-flight tool calls into a known state.
- **Agent loop detection** (`mlsys.agent-loop-detection`): Monitoring an agent trajectory for repeated actions, cycling states, or absent progress, and halting or escalating when the agent appears stuck.
- **Agentic workload** (`mlsys.agentic-workload`): The load an agent places on the serving stack, including context that grows each turn, idle gaps while tools run, many short dependent model calls, and reuse of KV state across steps.
- **Model-directed control flow** (`mlsys.model-directed-control-flow`): A control structure in which model outputs select the next step, tool call, or stopping decision within runtime-enforced limits, increasing flexibility while making cost and behavior harder to predict and test.

Secondary concept connections (not additional homes):

- **Trajectory evaluation** (`mlsys.agent-evaluation`)
- **Human oversight** (`mlsys.human-oversight`)
- **Cost per useful outcome** (`mlsys.unit-cost`)
- **Decision policy** (`mlsys.decision-policy`)

Related MLSysBook passages (teaching depth not assessed):

- vol4: Responsible Physical AI — Human oversight (`vol4.accountability.oversight`)
- vol3: Agent Economics — Cost per accepted task (`vol3.cost`)
- vol3: Introduction — Trajectory (`vol3.trajectory`)
- vol3: Agent Evaluation — Task acceptance (`vol3.acceptance`)
- vol3: Agent Security — Action-bound approval (`vol3.approval`)
- vol3: The Agent Harness — Transition guard (`vol3.guard`)
- vol3: The Agent Harness — Trajectory budget (`vol3.budget`)
- vol3: Failure Recovery — Semantic watchdog (`vol3.watchdog`)

#### Durable execution &amp; agent state

Identity: `durable-execution`

Persisting an agent's progress so long-running trajectories survive crashes, pauses, and redeployments without repeating completed effects, and can be replayed for debugging.

- **Activities:** Deploy and integrate; Operate and change.
- **Layers:** Numerical software and runtime; Compute and communication infrastructure.
- **Concerns:** Reliability and recovery; Reproducibility; Monetary cost.
- **Contexts:** Agentic systems; Distributed systems.

Includes: Event-sourced trajectory logs, resume protocols, and record-and-replay.; Pausing for human or external events and resuming safely..

Excludes: General checkpointing, idempotency, and sagas (Reliability &amp; fault tolerance).

Design warrants: `temporal`, `agents`.

Candidate home concepts:

- **Durable execution** (`mlsys.durable-execution`): Execution that persists workflow progress and reuses recorded results on recovery, with external-effect guarantees determined by endpoint idempotency, transactional boundaries, and reconciliation.
- **Record and replay** (`mlsys.deterministic-replay`): Re-executing a trajectory from its log by substituting recorded model outputs and tool results, to reproduce, debug, or test a divergence.
- **Suspend and resume** (`mlsys.suspend-resume`): Persisting a trajectory while it waits on a human, timer, or external event, then resuming with a validated state and authority.

Secondary concept connections (not additional homes):

- **Recovery checkpoint** (`mlsys.checkpoint`)
- **Idempotency** (`mlsys.idempotency`)
- **Compensating action** (`mlsys.compensation`)
- **Fencing** (`mlsys.fencing`)
- **Workflow orchestration** (`mlsys.workflow-orchestration`)

Related MLSysBook passages (teaching depth not assessed):

- vol2: Data Storage — Durable checkpoint (`vol2.storage.durable-checkpoint`)
- vol3: Durable Execution — Recovery checkpoint (`vol3.checkpoint`)
- vol3: Tool Calling — Operation identity (`vol3.operation`)
- vol3: Failure Recovery — Semantic compensation (`vol3.compensation`)
- vol2: Fault Tolerance — Recovery (`vol2.reliability.recovery`)
- vol3: Durable Execution — Trajectory log (`vol3.log`)

#### Multi-agent orchestration

Identity: `multi-agent`

Decomposing a task across several model-driven agents and coordinating their delegation, shared state, and results, including when coordination costs outweigh the benefit.

- **Activities:** Specify and design; Deploy and integrate; Operate and change.
- **Layers:** Application and workflow; Numerical software and runtime.
- **Concerns:** Reliability and recovery; Monetary cost; Task and predictive quality.
- **Contexts:** Agentic systems; Generative systems; Distributed systems.

Includes: Orchestrator-worker and peer topologies, typed task envelopes, and delegation.; Coordination failures and correlated errors among agents sharing a base model..

Excludes: Distributed consensus for infrastructure state (Coordination &amp; consensus).

Design warrants: `mast`, `autogen`.

Candidate home concepts:

- **Multi-agent orchestration** (`mlsys.multi-agent-orchestration`): Structuring a task as delegated subtasks among agents under a topology, such as orchestrator-worker or peer, with typed handoffs and result merging.
- **Shared agent state** (`mlsys.shared-agent-state`): Task state read and written by several agents, requiring concurrency control such as leases or optimistic versioning to avoid lost or conflicting updates.
- **Multi-agent coordination failure** (`mlsys.coordination-failure`): Task failure arising from interaction among agents, such as mis-specified delegation, lost context at handoff, conflicting actions, or premature termination.
- **Correlated agent errors** (`mlsys.correlated-agent-errors`): Errors shared across agents built on the same model or prompts, which make voting, debate, or cross-checking less independent than assumed.

Secondary concept connections (not additional homes):

- **Baseline** (`mlsys.baseline`)
- **Trajectory evaluation** (`mlsys.agent-evaluation`)
- **Fencing** (`mlsys.fencing`)

Related MLSysBook passages (teaching depth not assessed):

- vol3: Agent Evaluation — Task acceptance (`vol3.acceptance`)
- vol2: Fault Tolerance — Recovery (`vol2.reliability.recovery`)
- vol3: Multi-Agent Coordination — Task delegation (`vol3.delegation`)

### Embodied &amp; physical systems

What changes when ML closes the loop with sensors, actuators, and the physical world?

Sensing and perception, state estimation and mapping, motion planning and control, real-time and mixed-criticality computing, and robot learning from demonstration and simulation.

#### Sensing, perception &amp; fusion

Identity: `perception-sensing`

Turning time-stamped measurements from physical sensors into task-relevant estimates of the scene, including calibration, synchronization, and fusion across modalities.

- **Activities:** Acquire and prepare data; Learn and adapt; Deploy and integrate; Operate and change.
- **Layers:** Model and algorithm; Numerical software and runtime; People and operating environment.
- **Concerns:** Task and predictive quality; Latency, throughput, and resources; Safety.
- **Contexts:** Physical and control systems; Edge and device; Streaming systems.

Includes: Camera, LiDAR, radar, event, and tactile sensing; calibration chains.; Detection, segmentation, learned depth, and multi-sensor fusion..

Excludes: Sensor spoofing (Threat models &amp; adversarial attacks).

Design warrants: `slam-survey`, `bevfusion`, `manipulation`.

Candidate home concepts:

- **Sensor calibration** (`mlsys.sensing`): Establishing how observations relate to physical quantities, coordinates, and time.
- **Robot perception** (`mlsys.robot-perception`): Models that convert raw sensor streams into task-relevant scene estimates, such as detections, segmentation, depth, and pose, under latency limits.
- **Sensor fusion** (`mlsys.sensor-fusion`): Combining measurements from multiple sensors or modalities, at raw, feature, or decision level, into one estimate more accurate or robust than any single source.
- **Sensor time synchronization** (`mlsys.time-synchronization`): Aligning clocks and timestamps across sensors, compute, and actuators so measurements can be associated with the physical instant they describe.
- **Perception latency budget** (`mlsys.perception-latency-budget`): The share of a robot's sense-to-act deadline allotted to sensing and perception, which bounds model size, input resolution, and frame rate on the onboard computer.
- **Active sensor interference** (`mlsys.sensor-interference`): Crosstalk between active sensors, such as lidars, radars, or structured-light cameras, that emit into the same environment and corrupt each other's measurements.
- **Coordinate frames and transforms** (`mlsys.coordinate-frames`): Reference frames and the spatial transforms between them that express sensor observations, robot geometry, and actions in consistent coordinates.
- **Sensor degradation** (`mlsys.sensor-degradation`): Loss of sensor measurement fidelity under environmental or operating conditions, such as glare, rain, fog, dust, or a dirty lens, that degrades perception even when the sensing hardware has not failed.

Secondary concept connections (not additional homes):

- **Observation age** (`mlsys.observation-age`)
- **Distribution shift** (`mlsys.distribution-shift`)
- **Streaming speech recognition** (`mlsys.streaming-asr`)
- **Voice interaction pipeline** (`mlsys.voice-pipeline`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: ML Operations — Distribution shift (`vol1.operations.drift`)
- vol2: Robust AI — Distribution shift (`vol2.robustness.shift`)
- vol4: Physical Data — Endogenous covariate shift (`vol4.learning.shift`)
- vol4: Sensor Perception — Observation age (`vol4.sensing.age`)

#### State estimation, localization &amp; mapping

Identity: `state-estimation`

Inferring a robot's and its environment's latent state and uncertainty over time, through recursive filtering, localization and mapping, and learned predictive world models.

- **Activities:** Learn and adapt; Deploy and integrate; Operate and change.
- **Layers:** Model and algorithm; Numerical software and runtime; People and operating environment.
- **Concerns:** Task and predictive quality; Reliability and recovery; Safety.
- **Contexts:** Physical and control systems; Edge and device; Streaming systems.

Includes: Kalman and particle filters, tracking and data association, SLAM, metric and semantic maps.; Learned world models used for prediction and planning..

Excludes: Software observability of services (Monitoring &amp; observability).

Design warrants: `slam-survey`, `control`, `world-models`.

Candidate home concepts:

- **State estimation** (`mlsys.state-estimation`): Inferring latent physical state and uncertainty from observations and a model.
- **State observability** (`mlsys.state-observability`): Whether distinct internal states can be distinguished through the available observations and model.
- **Recursive state filtering** (`mlsys.recursive-filtering`): Updating a state estimate and its uncertainty by alternating model-based prediction with measurement correction at each step.
- **Simultaneous localization and mapping** (`mlsys.slam`): Jointly estimating a platform's pose and a map of an unknown environment from its own sensor observations.
- **Learned world model** (`mlsys.world-model`): A learned model that predicts future observations or latent states conditioned on actions, used for planning, policy training, or simulation.
- **Multi-object tracking** (`mlsys.multi-object-tracking`): Maintaining identities and states of multiple moving objects over time by associating new detections with existing tracks and starting or ending tracks.
- **Motion forecasting** (`mlsys.motion-forecasting`): Predicting the future trajectories of other moving agents, usually as several weighted hypotheses, from their tracked history and the map, so a planner can avoid where they will be rather than where they are.

Secondary concept connections (not additional homes):

- **Sensor calibration** (`mlsys.sensing`)
- **Probability calibration** (`mlsys.calibration`)

Related MLSysBook passages (teaching depth not assessed):

- vol4: Spatial Memory — Belief state (`vol4.estimation.belief`)
- vol4: Spatial Memory — Belief state (`vol4.estimation.belief`)
- vol4: Engineering Physical AI — Observational indistinguishability (`vol4.limits.indistinguishability`)

#### Motion planning &amp; control

Identity: `planning-control`

Generating feasible, collision-free motions and closing feedback loops that drive a physical system along them under dynamics, contact, and actuation limits.

- **Activities:** Specify and design; Deploy and integrate; Operate and change.
- **Layers:** Application and workflow; Numerical software and runtime; People and operating environment.
- **Concerns:** Latency, throughput, and resources; Safety; Reliability and recovery.
- **Contexts:** Physical and control systems; Edge and device.

Includes: Sampling- and optimization-based motion planning, task and motion planning.; Feedback control, MPC, and contact-rich manipulation..

Excludes: Runtime safety filters over a planner's output (Runtime assurance &amp; safety filters).

Design warrants: `control`, `planning-lavalle`, `mpc-book`.

Candidate home concepts:

- **Physical dynamics** (`mlsys.dynamics`): The evolution of a physical system under inputs, disturbances, and constraints.
- **Feedback control** (`mlsys.feedback-control`): Selecting inputs using measured or estimated deviation from desired behavior.
- **Trajectory feasibility** (`mlsys.trajectory-feasibility`): Satisfaction of motion, state, input, and dynamic constraints along a proposed trajectory.
- **Contact and compliance** (`mlsys.contact-compliance`): Mechanical interaction whose forces depend on geometry, motion, stiffness, and damping.
- **Motion planning** (`mlsys.motion-planning`): Computing a collision-free path or trajectory from a start to a goal configuration using search, sampling, or trajectory optimization.
- **Model predictive control** (`mlsys.model-predictive-control`): Repeatedly solving a finite-horizon constrained optimal control problem from the current state and applying only the first input.
- **Actuator limits** (`mlsys.actuator-limits`): Bounds on the force, torque, speed, and sustained duty an actuator can deliver, set by drive current limits, gearing, and motor heating.
- **Task and motion planning** (`mlsys.task-and-motion-planning`): Planning that jointly searches discrete symbolic action sequences and continuous motions, accepting a task plan only when each step admits a feasible motion.
- **System identification** (`mlsys.system-identification`): Estimating the parameters of a dynamics model, such as masses, friction, delays, and actuator gains, from measured input-output data of the physical system.
- **Robot kinematics** (`mlsys.robot-kinematics`): The geometric mapping between joint positions and velocities and the pose and velocity of a robot's links and end effector, including inverse solutions and singular configurations where motion authority is lost.

Secondary concept connections (not additional homes):

- **Stopping envelope** (`mlsys.stopping-envelope`)
- **Agent planning** (`mlsys.planning`)

Related MLSysBook passages (teaching depth not assessed):

- vol4: The Physical Body — Braking authority (`vol4.plant.braking`)
- vol4: The Nervous System — Phase margin (`vol4.control.phase-margin`)
- vol4: Trajectory Planning — Kinodynamic feasibility (`vol4.motion.feasibility`)
- vol4: Safety Enforcement — Contact margin (`vol4.enforcement.contact`)
- vol4: Safety Enforcement — Stopping envelope (`vol4.enforcement.envelope`)
- vol4: The Causal Boundary — Stopping distance (`vol4.stopping.distance`)
- vol4: The Physical Body — Braking authority (`vol4.plant.braking`)
- vol4: The Physical Body — Thermal headroom (`vol4.plant.thermal`)

#### Real-time &amp; mixed-criticality computing

Identity: `real-time-systems`

Guaranteeing that sensing, inference, and control complete within deadlines on shared embedded hardware, and isolating safety-critical functions from less trusted ones.

- **Activities:** Specify and design; Evaluate and assure; Deploy and integrate; Operate and change.
- **Layers:** Numerical software and runtime; Compute and communication infrastructure; People and operating environment.
- **Concerns:** Latency, throughput, and resources; Safety; Reliability and recovery.
- **Contexts:** Physical and control systems; Edge and device; Streaming systems.

Includes: Multi-rate scheduling, response-time analysis, WCET, and real-time buses.; Mixed-criticality partitioning and hypervisors on shared silicon..

Excludes: Throughput-oriented profiling and utilization (Profiling &amp; utilization).

Design warrants: `wcet-survey`, `liu-layland`, `mixed-crit-vestal`.

Candidate home concepts:

- **Observation age** (`mlsys.observation-age`): Elapsed time between a physical observation and its use in a decision.
- **Feedback delay** (`mlsys.feedback-delay`): Delay around a sensing, computation, communication, and actuation loop.
- **Worst-case execution time** (`mlsys.wcet`): A justified execution-time upper bound under specified hardware and execution assumptions.
- **Real-time scheduling** (`mlsys.real-time-scheduling`): Assigning processor time to periodic and sporadic tasks so that each meets its deadline, with schedulability established by analysis.
- **Mixed-criticality partitioning** (`mlsys.mixed-criticality`): Running functions of different assurance levels on shared hardware while preventing lower-criticality work from delaying or corrupting higher-criticality functions.
- **Watchdog timer** (`mlsys.watchdog-timer`): A timer that a monitored component must refresh within a deadline or window; expiry triggers a defined safe response such as reset or revoking control authority.
- **Robotics middleware and dataflow** (`mlsys.robotics-middleware`): Software connecting sensing, perception, planning, and control processes as nodes that exchange typed messages over publish-subscribe topics with configurable delivery and timing policies.
- **Onboard compute partitioning** (`mlsys.onboard-compute-partitioning`): Assigning perception, planning, and control workloads to the CPUs, accelerators, and microcontrollers of a robot, or offloading them, so each meets its deadline and criticality level.
- **Interrupt latency** (`mlsys.interrupt-latency`): The delay between a hardware event and its interrupt handler running, plus the handler's own run time, which must fit inside a real-time sensing or control budget alongside inference.
- **Priority inversion** (`mlsys.priority-inversion`): A high-priority task waiting on a resource held by a lower-priority task, which a medium-priority task then preempts, so the high-priority task misses its deadline unless priority inheritance or ceilings are used.

Secondary concept connections (not additional homes):

- **Execution time** (`mlsys.execution-time`)
- **Execution placement** (`mlsys.placement`)
- **Thermal behavior and throttling** (`mlsys.thermal-dynamics`)
- **Execution isolation** (`mlsys.isolation`)

Related MLSysBook passages (teaching depth not assessed):

- vol3: Agent Sandboxes — Isolation envelope (`vol3.isolation`)
- vol4: Workload Placement — Resource isolation (`vol4.placement.isolation`)
- vol4: Sensor Perception — Observation age (`vol4.sensing.age`)
- vol4: The Nervous System — Feedback-loop delay (`vol4.timing.feedback-delay`)
- vol4: Workload Placement — Worst-case execution time (`vol4.timing.wcet`)
- vol4: The Nervous System — Response-time bound (`vol4.timing.response-bound`)
- vol4: Workload Placement — Thermal throttling (`vol4.placement.throttling`)

#### Robot learning &amp; sim-to-real

Identity: `robot-learning`

Learning robot policies from teleoperated demonstrations, simulation, and pretrained vision-language backbones, and transferring them to physical hardware.

- **Activities:** Acquire and prepare data; Learn and adapt; Evaluate and assure; Deploy and integrate.
- **Layers:** Model and algorithm; People and operating environment.
- **Concerns:** Task and predictive quality; Safety; Monetary cost.
- **Contexts:** Physical and control systems; Deep learning; Edge and device.

Includes: Vision-language-action models, action chunking, and cross-embodiment pretraining.; Teleoperation data collection, domain randomization, and sim-to-real transfer..

Excludes: Imitation and reinforcement learning as general paradigms (Learning paradigms); Closed-loop robot evaluation statistics (Model evaluation &amp; testing).

Design warrants: `pi0`, `rt2`, `domain-rand`.

Candidate home concepts:

- **Simulation-to-real transfer** (`mlsys.sim-to-real`): Deployment of a learned policy under differences between simulated and physical conditions.
- **Vision-language-action model** (`mlsys.vla-model`): A policy built on a pretrained vision-language backbone that maps images and language instructions to robot actions.
- **Teleoperation data** (`mlsys.teleoperation-data`): Robot demonstrations collected by humans driving the robot remotely, whose cost, coverage, and action-label fidelity bound imitation-learned policies.
- **Action chunking** (`mlsys.action-chunking`): Predicting a short sequence of future actions per policy inference, decoupling model invocation rate from control rate and reducing compounding error.
- **Domain randomization** (`mlsys.domain-randomization`): Training in simulation over randomized visual and physical parameters so the policy treats the real world as one more variation.
- **Diffusion or flow-matching policy** (`mlsys.diffusion-policy`): Robot policies that generate continuous actions through iterative denoising or learned-flow integration, representing multiple action modes while spending an inference-iteration budget.
- **Parallel physics simulation** (`mlsys.parallel-simulation`): Running many simulated robot environments concurrently, often batched on accelerators, so policy learning and data generation gather experience faster than physical collection.
- **Action tokenization** (`mlsys.action-tokenization`): Encoding continuous robot actions or action sequences as discrete model tokens, making reconstruction error and token count determinants of policy fidelity and execution cost.
- **Cross-embodiment training** (`mlsys.cross-embodiment-training`): Training one policy across datasets from several robot embodiments, aligning observations and action interfaces so learning is shared despite differences in sensors and control.
- **Dual-rate hierarchical policy** (`mlsys.dual-rate-hierarchical-policy`): Splitting a robot policy into a slow high-level model that sets goals or latent commands and a fast low-level controller that tracks them at control rate, so large-model latency does not set the control loop's rate.

Secondary concept connections (not additional homes):

- **Imitation learning** (`mlsys.imitation-learning`)
- **Reinforcement learning** (`mlsys.reinforcement-learning`)
- **Distribution shift** (`mlsys.distribution-shift`)

Related MLSysBook passages (teaching depth not assessed):

- vol1: ML Operations — Distribution shift (`vol1.operations.drift`)
- vol2: Robust AI — Distribution shift (`vol2.robustness.shift`)
- vol4: Physical Data — Endogenous covariate shift (`vol4.learning.shift`)
- vol4: Policy Training — Reality gap (`vol4.learning.reality-gap`)
- vol4: The Cognitive Brain — Action chunk (`vol4.proposal.chunk`)
- vol4: Policy Training — Reality gap (`vol4.learning.reality-gap`)

## Candidate scope gaps

Candidates for editorial investigation, not verified omissions.

- **Photonic compute, in Accelerator architecture** (`gap-emerging-compute-substrates`): Kept open in 0.3 (2026-10 review): photonic arithmetic is not yet in production ML; compute-in-memory, wafer-scale execution, FPGAs, and advanced packaging are now covered.

- **Attention-FFN disaggregation, in LLM &amp; generative inference** (`gap-attention-ffn-disaggregation`): Proposed in the 2026-10 review from MegaScale-Infer (SIGCOMM 2025); admit with independent supporting evidence or a qualifying recorded coverage miss.

- **Decoded-video view reuse, in Data loading &amp; input pipelines** (`gap-decoded-video-reuse`): Surfaced by the 2025 held-out test (SAND, SOSP 2025); one venue lineage so far, and basic decoding is covered by the input-pipeline concepts. Admit with a second independent source.

## Design references

These references inform design and scope; they are not assertion-level validation.

- [W3C SKOS Reference](https://www.w3.org/TR/skos-reference/): Representation and relation semantics; not an ML systems subject inventory.
- [ISO/IEC 23053:2022 official description](https://www.iso.org/standard/74438.html): Public description of ML systems framework and vocabulary; paid full standard not inspected.
- [Software Engineering for Machine Learning: A Case Study](https://www.microsoft.com/en-us/research/publication/software-engineering-for-machine-learning-a-case-study/): Empirical workflow and software/data engineering comparison.
- [The ML Test Score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/): Production-readiness evidence and testing comparison; not learner proficiency.
- [Towards CRISP-ML(Q)](https://arxiv.org/abs/2003.05155): Lifecycle and quality-assurance comparison.
- [Machine Learning Operations: Overview, Definition, and Architecture](https://arxiv.org/abs/2205.02302): Operational architecture, workflow, and role comparison.
- [NIST AI RMF 1.0](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf): Governance and risk concerns; pin edition; not technical taxonomy conformance.
- [CS2023 Knowledge Areas](https://csed.acm.org/knowledge-areas/): Adjacent computing and curriculum boundary comparison.
- [Deep Learning Systems](https://dlsyscourse.org/): Numerical software and hardware topic depth comparison.
- [Full Stack Deep Learning 2022](https://fullstackdeeplearning.com/course/2022/): End-to-end application engineering and curricular comparison.
- [A Taxonomy of ML for Systems Problems](https://research.google/pubs/a-taxonomy-of-ml-for-systems-problems/): Related taxonomy with a distinct ML-for-systems scope.
- [Machine Learning Systems: A Survey from a Data-Oriented Perspective](https://doi.org/10.1145/3769292): Related architectural lens; detailed mappings pending full-paper inspection.
- [Stanford CS329S, 2021 syllabus](https://stanford-cs329s.github.io/2021/syllabus.html): Reading-scope warrant imported from the field model; not verification of this organizing row or exact semantic correspondence.
- [Stanford CS336, Spring 2025](https://cs336.stanford.edu/spring2025/index.html): Reading-scope warrant imported from the field model; not verification of this organizing row or exact semantic correspondence.
- [scikit-learn supervised learning guide](https://scikit-learn.org/stable/supervised_learning.html): Reading-scope warrant imported from the field model; not verification of this organizing row or exact semantic correspondence.
- [scikit-learn model evaluation guide](https://scikit-learn.org/stable/modules/model_evaluation.html): Reading-scope warrant imported from the field model; not verification of this organizing row or exact semantic correspondence.
- [Google Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml): Reading-scope warrant imported from the field model; not verification of this organizing row or exact semantic correspondence.
- [Hidden Technical Debt in Machine Learning Systems](https://research.google/pubs/hidden-technical-debt-in-machine-learning-systems/): Reading-scope warrant imported from the field model; not verification of this organizing row or exact semantic correspondence.
- [Roofline performance model](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html): Reading-scope warrant imported from the field model; not verification of this organizing row or exact semantic correspondence.
- [Random Search for Hyper-Parameter Optimization](https://jmlr.org/papers/v13/bergstra12a.html): Reading-scope warrant imported from the field model; not verification of this organizing row or exact semantic correspondence.
- [Hyperband](https://jmlr.org/papers/v18/16-558.html): Reading-scope warrant imported from the field model; not verification of this organizing row or exact semantic correspondence.
- [Over-fitting in Model Selection and Subsequent Selection Bias](https://jmlr.org/papers/v11/cawley10a.html): Reading-scope warrant imported from the field model; not verification of this organizing row or exact semantic correspondence.
- [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html): Reading-scope warrant imported from the field model; not verification of this organizing row or exact semantic correspondence.
- [FlashAttention](https://arxiv.org/abs/2205.14135): Reading-scope warrant imported from the field model; not verification of this organizing row or exact semantic correspondence.
- [PagedAttention](https://arxiv.org/abs/2309.06180): Reading-scope warrant imported from the field model; not verification of this organizing row or exact semantic correspondence.
- [ZeRO](https://arxiv.org/abs/1910.02054): Reading-scope warrant imported from the field model; not verification of this organizing row or exact semantic correspondence.
- [Berkeley LLM Agents, Fall 2024](https://llmagents-learning.org/f24): Reading-scope warrant imported from the field model; not verification of this organizing row or exact semantic correspondence.
- [MIT Underactuated Robotics](https://underactuated.mit.edu/index.html): Reading-scope warrant imported from the field model; not verification of this organizing row or exact semantic correspondence.
- [The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf): Reading-scope warrant imported from the field model; not verification of this organizing row or exact semantic correspondence.
- [Google Site Reliability Engineering](https://sre.google/sre-book/table-of-contents/): Reading-scope warrant imported from the field model; not verification of this organizing row or exact semantic correspondence.
- [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734): Reading-scope warrant imported from the field model; not verification of this organizing row or exact semantic correspondence.
- [NIST Adversarial ML taxonomy, 2025](https://csrc.nist.gov/pubs/ai/100/2/e2025/final): Reading-scope warrant imported from the field model; not verification of this organizing row or exact semantic correspondence.
- [Green Algorithms: Quantifying the Carbon Footprint of Computation](https://arxiv.org/abs/2007.07610): Reading-scope warrant imported from the field model; not verification of this organizing row or exact semantic correspondence.
- [Estimating the Carbon Footprint of BLOOM](https://arxiv.org/abs/2211.02001): Reading-scope warrant imported from the field model; not verification of this organizing row or exact semantic correspondence.
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877): Reading-scope warrant imported from the field model; not verification of this organizing row or exact semantic correspondence.
- [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702v4): Abstract inspected: finite-state constrained generation with regular expressions and context-free grammars; not semantic correctness.
- [Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977v3): Federated learning survey; source scope and abstract inspected, not an assertion-level crosswalk.
- [The worst-case execution-time problem: overview of methods and survey of tools](https://doi.org/10.1145/1347375.1347389): Execution-time bounds and timing-analysis assumptions; narrow timing reference, not a whole physical-system safety case.
- [Apache Parquet file format](https://parquet.apache.org/docs/file-format/): Describes row groups, column chunks, metadata, and selective sequential column reading; one concrete storage-layout design. Reading basis, not independent assertion validation.
- [Apache Beam programming guide: Pipeline I/O](https://beam.apache.org/documentation/programming-guide/#pipeline-io): Defines reading from and writing to external sources through pipeline I/O transforms; adapters differ in behavior. Reading basis, not independent assertion validation.
- [Apache Iceberg specification: Schema evolution](https://iceberg.apache.org/spec/#schema-evolution): Describes adding/deleting/renaming/reordering fields, permitted type promotions, schema IDs, and compatibility mechanisms for that format. Reading basis, not independent assertion validation.
- [The Oracle Problem in Software Testing: A Survey](https://philmcminn.com/publications/barr2015.pdf): Author-hosted paper distinguishing desired from incorrect observed behavior; includes specifications, partial oracles, metamorphic relations, and human judgment. Reading basis, not independent assertion validation.
- [Google SRE: Testing for Reliability](https://sre.google/sre-book/testing-reliability/): Defines integration/system tests and discusses statistical fault-oriented testing; passing tests does not prove reliability. Reading basis, not independent assertion validation.
- [Principles of Chaos Engineering](https://principlesofchaos.org/): Authors describe experiments with disruptive events, observable steady-state hypotheses, and containing experimental impact; production experiments are not required by our concept. Reading basis, not independent assertion validation.
- [MLflow: Managing Dependencies in Models](https://mlflow.org/docs/latest/ml/model/dependencies/): Documents model environment metadata, dependencies, direct/transitive version locking, and inference limitations; locking does not guarantee identical execution. Reading basis, not independent assertion validation.
- [Kubeflow Pipelines: Pipeline concepts](https://www.kubeflow.org/docs/components/pipelines/concepts/pipeline/): Describes component ordering, conditions, parameters/artifacts, caching, retries, and orchestration; an implementation example rather than a universal DAG requirement. Reading basis, not independent assertion validation.
- [PyTorch: Reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html): States limits across versions/platforms and distinguishes randomness controls, algorithm determinism, and associated performance tradeoffs. Reading basis, not independent assertion validation.
- [CUDA Programming Guide: Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html): A vendor example of accelerator execution and memory organization; not a universal accelerator definition.
- [NVIDIA System Management Interface documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html): Vendor-specific power limits, thermal slowdown, and operating telemetry; not a physical-system safety qualification.
- [MLSysBook hardware acceleration and physical placement](https://mlsysbook.ai/vol1/): Internal manuscript reading basis: Volume I hardware acceleration, Volume II compute infrastructure, and Volume IV placement. Electrical and thermal scope remains draft.
- [Google SRE book: Service Level Objectives](https://sre.google/sre-book/service-level-objectives/): SLI/SLO/error-budget vocabulary for requirements rows.
- [The Datacenter as a Computer: Designing Warehouse-Scale Machines, 3rd ed.](https://doi.org/10.1007/978-3-031-01761-2): Canonical textbook for warehouse-scale facilities, power, cooling, networking, storage, and cluster management.
- [MIT 6.5940 TinyML and Efficient Deep Learning Computing, Fall 2024](https://hanlab.mit.edu/courses/2024-fall-65940): Curriculum warrant for efficient models, long context, and edge/embedded hardware.
- [Stanford CS149 Parallel Computing, Fall 2025](https://gfxcourses.stanford.edu/cs149/fall25): Curriculum warrant for parallel speedup reasoning, hardware specialization, and mapping AI to datacenters.
- [CMU 11-868 LLM Systems, Spring 2025](https://llmsystem.github.io/llmsystem2025spring/docs/Syllabus/): Curriculum warrant for MoE, efficient attention, and LLM training/inference systems.
- [UT Austin CS 378 Systems for ML, Fall 2024](https://www.cs.utexas.edu/~akella/CS378/F24/lectures.html): Curriculum warrant for collectives, compilers, linear attention, and SSMs.
- [UW CSE 599o, Autumn 2025](https://courses.cs.washington.edu/courses/cse599o/25au/calendar/calendar.html): Curriculum warrant for GPU communication, interconnects, data loading, and N-D parallelism.
- [Huyen, Designing Machine Learning Systems](https://github.com/chiphuyen/dmls-book): Production-ML lifecycle comparison (data, deployment, monitoring, continual learning); table of contents inspected.
- [Kästner, Machine Learning in Production](https://mlip-cmu.github.io/book/): Software-engineering lifecycle and operations comparison.
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155): Canonical description of the SFT-reward-model-RLHF post-training pipeline.
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314): Defines the test-time compute allocation problem and its main mechanisms.
- [TensorFlow Lite Micro: Embedded Machine Learning on TinyML Systems](https://arxiv.org/abs/2010.08678): On-device runtime design and deployment constraints.
- [MLPerf Inference Benchmark](https://arxiv.org/abs/1911.02549): System benchmarking methodology (scenarios, run rules).
- [MITRE ATLAS](https://atlas.mitre.org/): Adversary technique taxonomy for AI systems; comparison for attack and supply-chain rows.
- [Silent Data Corruptions at Scale](https://arxiv.org/abs/2102.11245): Reading-scope warrant for hardware-fault and SDC row.
- [A higher order estimate of the optimum checkpoint interval for restart dumps](https://doi.org/10.1016/j.future.2004.11.016): Reading-scope warrant for checkpoint interval and restart economics.
- [Chasing Carbon: The Elusive Environmental Footprint of Computing](https://arxiv.org/abs/2011.02839): Reading-scope warrant for embodied carbon and hardware lifecycle.
- [Sustainable AI: Environmental Implications, Challenges and Opportunities](https://arxiv.org/abs/2111.00364): Reading-scope warrant for ML-specific environmental accounting across the lifecycle.
- [Leveson, Engineering a Safer World: Systems Thinking Applied to Safety](https://doi.org/10.7551/mitpress/8179.001.0001): Canonical text for system-theoretic hazard analysis (STPA).
- [ISO 26262-1:2018 Road vehicles: Functional safety](https://www.iso.org/standard/68383.html): Functional-safety standard scoping hazard analysis, integrity levels, and the safety lifecycle.
- [ISO 21448:2022 Road vehicles: Safety of the intended functionality](https://www.iso.org/standard/77490.html): Standard scoping hazards from functional insufficiency of perception and learned components.
- [UL 4600 Standard for Safety for the Evaluation of Autonomous Products (Koopman overview)](https://users.ece.cmu.edu/~koopman/ul4600/index.html): Autonomy safety-case standard scoping safety cases and field indicators.
- [Goal Structuring Notation Community Standard](https://scsc.uk/gsn): Reference notation for structured safety arguments.
- [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565): Widely taught statement of ML-specific accident risks.
- [The Safety Filter: A Unified View of Safety-Critical Control in Autonomous Systems](https://arxiv.org/abs/2309.05837): Survey scoping runtime safety filters.
- [Control Barrier Functions: Theory and Applications](https://arxiv.org/abs/1903.11199): Reference for barrier-function enforcement.
- [Sha, Using Simplicity to Control Complexity](https://doi.org/10.1109/MS.2001.936213): Origin of the Simplex runtime-assurance architecture.
- [A Survey of Algorithms for Black-Box Safety Validation of Cyber-Physical Systems](https://arxiv.org/abs/2005.02979): Survey scoping falsification and rare-failure validation.
- [Algorithms for Verifying Deep Neural Networks](https://arxiv.org/abs/1903.06758): Survey scoping neural-network verification.
- [Kalra and Paddock, Driving to Safety (RAND RR-1478)](https://www.rand.org/pubs/research_reports/RR1478.html): Standard argument on the statistics of demonstrating rare-failure safety.
- [Barocas, Hardt, Narayanan, Fairness and Machine Learning](https://fairmlbook.org/): Textbook scoping fairness criteria and mitigation.
- [Inherent Trade-Offs in the Fair Determination of Risk Scores](https://arxiv.org/abs/1609.05807): Canonical incompatibility result for fairness criteria.
- [Molnar, Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/): Widely used reference scoping explainability methods.
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993): Origin of model-documentation practice.
- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010): Origin of dataset-documentation practice.
- [Bainbridge, Ironies of Automation](https://doi.org/10.1016/0005-1098(83)90046-8): Canonical human-factors account of supervising automation.
- [Parasuraman and Riley, Humans and Automation: Use, Misuse, Disuse, Abuse](https://doi.org/10.1518/001872097778543886): Canonical account of automation bias and complacency.
- [Regulation (EU) 2024/1689 (Artificial Intelligence Act)](https://eur-lex.europa.eu/eli/reg/2024/1689/oj): Primary risk-tiered AI regulation.
- [Closing the AI Accountability Gap: Defining an End-to-End Framework for Internal Algorithmic Auditing](https://arxiv.org/abs/2001.00973): Reference framework for algorithmic audits.
- [Anthropic, Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): Practitioner reference scoping context engineering.
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560): Reference design for tiered agent memory.
- [Model Context Protocol specification](https://modelcontextprotocol.io/specification): De facto standard tool-discovery protocol.
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents): Practitioner reference scoping agent loops, workflows, and orchestration.
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629): Canonical agent-loop pattern.
- [Temporal documentation](https://docs.temporal.io/): Reference implementation of durable execution and replay.
- [Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657): Empirical taxonomy scoping multi-agent failure.
- [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155): Widely used multi-agent orchestration framework.
- [Past, Present, and Future of Simultaneous Localization and Mapping](https://arxiv.org/abs/1606.05830): Survey scoping state estimation and mapping.
- [BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation](https://arxiv.org/abs/2205.13542): Representative learned sensor-fusion system.
- [MIT Robotic Manipulation (Tedrake)](https://manipulation.csail.mit.edu/): Course scoping robot perception, planning, and control.
- [Mastering Diverse Domains through World Models (DreamerV3)](https://arxiv.org/abs/2301.04104): Representative learned world model.
- [LaValle, Planning Algorithms](https://lavalle.pl/planning/): Textbook scoping motion planning.
- [Rawlings, Mayne, Diehl, Model Predictive Control: Theory, Computation, and Design](https://sites.engineering.ucsb.edu/~jbraw/mpc/): Textbook scoping MPC.
- [Scheduling Algorithms for Multiprogramming in a Hard-Real-Time Environment](https://doi.org/10.1145/321738.321743): Foundational real-time scheduling result.
- [Preemptive Scheduling of Multi-criticality Systems with Varying Degrees of Execution Time Assurance](https://doi.org/10.1109/RTSS.2007.47): Origin of mixed-criticality scheduling.
- [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818): Origin of the vision-language-action model category.
- [pi0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164): Representative generalist robot foundation model.
- [Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907): Canonical sim-to-real technique.
