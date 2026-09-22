#!/usr/bin/env python3
"""
apply_pedagogical_openings.py

Transforms raw/stub introductions into the publication-grade 5-beat Hourglass opening
in TinyTorch: From Tensors to Transformers.

Implements the two-part pedagogical progression on page 1 of every chapter:
1. The Deep Learning Connection ("As an ML person, I know this"):
   Why this mathematical abstraction is necessary, what modeling problem it solves,
   and why simpler or linear alternatives fail.
2. The Systems & Framework Standpoint ("From a framework standpoint, what do I have to think about?"):
   The physical hardware and runtime engineering reality: memory buffers, cache
   hierarchies, compute-vs-memory-bandwidth walls, operator dispatch, and graph recording.
3. The Bridge:
   Connecting seamlessly from prior modules to upcoming modules, leading directly
   into the chapter's opening figure and worked problem.
"""

import re
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Dictionary of opening prose bridges (excluding the opening figure, which is preserved from the original file)
PROSE_BRIDGES = {
    "02_activations.qmd": r"""The core premise of deep learning is the ability to learn complex, non-linear representations from data. Yet if a neural network consists solely of linear transformations—matrix multiplications and bias additions—it suffers a fatal mathematical collapse: the composition of any number of linear maps is itself just a single linear map ($\mathbf{W}_2(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = \mathbf{W}_{\text{eff}}\mathbf{x} + \mathbf{b}_{\text{eff}}$). Without non-linearity, a 100-layer deep network has no more expressive power than a single linear layer. To bend representation space and learn intricate decision boundaries, deep architectures require **activation functions**.

From a framework and systems standpoint, activation functions introduce a distinct operational class. While tensor operations in @sec-tensors established physical 1D memory buffers and strided coordinate views, activations are shape-preserving, elementwise or reduction operators executed at immense scale. Building an activation engine forces the systems architect to resolve three core concerns:
1. **Operator abstraction**: How does the runtime apply mathematical transformations to underlying buffers through a uniform, composable boundary (`Function.apply`)?
2. **Memory bandwidth efficiency**: Because elementwise arithmetic is computationally inexpensive, activations are almost always *memory-bandwidth bound*—moving arrays across the memory bus costs far more than computing the functions themselves.
3. **The autograd contract**: Each activation must preserve the necessary forward state so that reverse-mode automatic differentiation (@sec-autograd) can compute local derivatives during backpropagation.

This chapter builds the non-linear foundation of TinyTorch, implementing the core activation zoo—ReLU, Sigmoid, Tanh, GELU, and Softmax—and establishing the contracts that allow @sec-layers to stack them into deep, trainable neural networks.""",

    "03_layers.qmd": r"""Deep learning models are not solitary mathematical equations; they are modular compositions of stateful, parameterized building blocks. As architectures grow in depth, raw tensor math must be organized into clean modules that bundle weights, biases, and forward computation rules. Furthermore, deep networks require systematic numerical initialization to prevent signal variance from vanishing or exploding across layers, as well as distinct operational behaviors during training versus inference (such as stochastic zeroing in Dropout).

From a framework and systems perspective, a **layer** represents the critical boundary between *persistent model state* and *ephemeral activations*. Trainable parameters live across iterations in persistent memory buffers, while input and output activations are temporary tensors allocated and freed during each pass. A framework must solve three architectural obligations:
1. **State management and parameter discovery**: How to hierarchically register child parameters (`parameters()`) so optimizers can locate every trainable weight without duplicate references or memory leaks.
2. **Modular container orchestration**: How to pipeline layers sequentially (`Sequential`), automatically threading forward arguments and training mode flags through child layers.
3. **Variance-preserving initialization**: How to initialize weight matrices (e.g., He and Xavier scaling) so that activation and gradient variances remain stable regardless of network depth.

Building upon the memory buffers of @sec-tensors and activations of @sec-activations, this chapter encapsulates stateful parameters into reusable, composable layer blocks.""",

    "04_losses.qmd": r"""A neural network produces raw, unbounded scores across possible outputs, but learning requires a scalar objective that quantifies prediction error: how far is the model's prediction from ground truth? Different learning tasks require distinct mathematical loss formulations—mean squared error for continuous regression, binary cross-entropy for probabilities, and categorical cross-entropy for multi-class classification.

From a framework and systems standpoint, a loss function is a tensor reduction operator: it ingests high-dimensional batch predictions and ground-truth targets and collapses them into a single scalar that drives gradient descent. However, implementing loss functions on physical hardware exposes the harsh reality of IEEE 754 floating-point arithmetic. Textbook mathematical formulas frequently suffer from catastrophic numerical instability:
1. **Exponential overflow**: Exponentiating large positive raw scores ($e^{1000}$) overflows float32 dynamic range to `+inf`, causing division to produce `nan`.
2. **Logarithmic underflow**: Taking the logarithm of probabilities near zero ($\log(0)$) underflows to `-inf`.
3. **Operator fusion for stability**: To prevent numerical breakdown, frameworks must mathematically fuse separate operations—combining LogSoftmax with Negative Log-Likelihood into a single numerically stable kernel using the max-subtraction shift.

This chapter constructs TinyTorch's loss library, defusing the numerical traps of cross-entropy to provide stable scalar objectives that @sec-autograd and @sec-optimizers turn into parameter updates.

@Fig-loss-stability contrasts a direct exponential with the max-shifted path. The worked trace will follow each intermediate value through the implementation.""",

    "05_dataloader.qmd": r"""Training neural networks with Stochastic Gradient Descent (SGD) requires balancing gradient accuracy with computational tractability. Computing the true gradient across an entire dataset of millions of samples is computationally prohibitive and prone to trapping models in sharp local minima. Conversely, updating parameters on single samples produces high-variance, noisy gradient trajectories. Mini-batching resolves this tension: it yields stable empirical gradient estimates while random shuffling breaks autocorrelation between consecutive training steps.

From a systems and framework perspective, the data ingestion pipeline is often the primary bottleneck of modern deep learning. Processors and accelerators spend immense time starved of operands if data loading cannot continuously saturate the memory bus. An ML framework must address three core architectural responsibilities:
1. **Separation of concerns**: Decouple dataset storage and random-access sample indexing (`Dataset`) from batching, shuffling, and memory collation logic (`DataLoader`).
2. **Memory packing and tensor collation**: Pack individual scattered samples into contiguous, row-major tensor batches $(B, \dots)$ that modern BLAS and GEMM compute kernels require for high throughput.
3. **Epoch orchestration**: Generate deterministic random permutation arrays to ensure reproducible shuffling, and handle edge-case boundary conditions such as remainder batches (`drop_last`).

This chapter builds TinyTorch's data loading pipeline, establishing the contiguous batch stream that feeds the forward passes of @sec-layers and powers the end-to-end training loop in @sec-training.""",

    "06_autograd.qmd": r"""The foundational algorithm of modern deep learning is backpropagation: computing the exact partial derivative of a scalar loss with respect to every trainable weight in a network via the multivariable chain rule. These gradients tell the optimizer precisely how to adjust millions of parameters to decrease error.

From a framework and systems standpoint, calculating derivatives manually or through symbolic algebra does not scale to dynamic neural networks with complex branching and control flow. Numerical finite differences are equally impractical, requiring $O(N)$ forward passes that would make training computationally impossible. A production autograd engine must solve several critical systems challenges:
1. **Dynamic tape recording**: Record an execution Directed Acyclic Graph (DAG) during the forward pass, capturing operations (`Function.apply`), input-output tensor relationships, and saved intermediate activations.
2. **Reverse topological traversal**: Walk the recorded computation tape in exact reverse topological order, ensuring every consumer contributes its gradient before an intermediate tensor passes its accumulated derivative upstream.
3. **Memory lifetime and mutation safety**: Accumulate gradients into leaf parameter buffers (`param.grad += ...`) while promptly freeing intermediate graph nodes to prevent memory leaks, and guard against silent in-place tensor mutations that corrupt saved forward values.

This chapter constructs TinyTorch's reverse-mode automatic differentiation engine, connecting the forward computations of @sec-tensors through @sec-dataloader with the backward gradient updates required by @sec-optimizers.""",

    "07_optimizers.qmd": r"""Gradient descent is the engine that drives neural learning, but raw gradients alone are often insufficient to train deep models effectively. High-dimensional loss landscapes are characterized by ill-conditioned ravines, saddle points, and extreme curvature variations where gradients are steep in some directions and nearly flat in others. Plain gradient descent oscillates uncontrollably across canyon walls while making glacial progress along the valley floor toward the optimum.

From a systems and framework perspective, an optimizer is an in-place stateful memory manager. While automatic differentiation (@sec-autograd) computes raw gradient vectors, the optimizer must translate those derivatives into stable parameter trajectories:
1. **State buffer allocation**: Track persistent auxiliary moment buffers for each parameter. Momentum stores velocity vectors ($m_t$), while adaptive algorithms like Adam and AdamW maintain running estimates of both first and uncentered second moments ($v_t$), doubling or tripling the runtime memory footprint of model parameters.
2. **Decoupled weight decay (AdamW)**: Separate $L_2$ regularization weight shrinkage from adaptive gradient moment scaling, preventing weights with large historical gradients from evading regularization.
3. **In-place memory mutation**: Apply trajectory updates directly to parameter memory buffers without allocating new tensors or breaking existing references.

This chapter builds TinyTorch's optimization suite—SGD with momentum, Adam, and AdamW—providing the parameter update engine that drives the unified training loop in @sec-training.""",

    "08_training.qmd": r"""Supervised machine learning is the practical execution of empirical risk minimization: repeatedly exposing a parameterized model to training data, evaluating loss, computing gradients, and steering weights toward optimal generalization. However, learning is not a loose script; it is a precisely choreographed state machine where data pipelines, forward models, autograd graphs, and optimizers interact in lockstep.

From a framework and systems standpoint, the training engine coordinates execution lifecycles and memory boundaries across five non-negotiable phases:
1. **Gradient zeroing (`zero_grad()`)**: Clearing accumulated gradients from prior steps to prevent catastrophic, silent gradient accumulation.
2. **Forward evaluation**: Pipelining batch inputs through model layers to compute predictions and scalar losses.
3. **Backward tape execution**: Traversing the autograd DAG to populate parameter `.grad` fields, followed by immediate graph disposal to release gigabytes of intermediate activation memory.
4. **Gradient clipping and optimization**: Bounding global gradient $L_2$ norms to prevent numerical divergence, followed by in-place optimizer parameter updates.
5. **Epoch and batch boundary accounting**: Correctly weighting partial remainder batches so update trajectories reflect true sample-weighted empirical risk.

This chapter synthesizes the foundational modules of @sec-tensors through @sec-optimizers into the unified `Trainer` engine, establishing the end-to-end execution loop of TinyTorch.""",

    "milestone_01.qmd": r"""In 1969, Marvin Minsky and Seymour Papert published their famous proof that single-layer perceptrons cannot learn the non-linear XOR (exclusive or) decision boundary, precipitating the first "AI Winter." The limitation was not in learning algorithms, but in representation: a single linear boundary cannot separate points whose positive labels sit on opposite diagonals. Seventeen years later, Rumelhart, Hinton, and Williams demonstrated that multi-layer perceptrons with hidden units trained via backpropagation can bend representation space, cleanly resolving the XOR crisis.

From a systems and framework perspective, Milestone I represents the comprehensive integration test of Part I. It is where eight distinct subsystems—tensors, strided memory, activation functions, modular layers, stable losses, data loaders, autograd DAG recording, and AdamW optimization—must operate as a unified, coherent runtime. Building an end-to-end learning machine requires validating that:
1. **Multi-layer gradient propagation**: Backward gradients propagate seamlessly across multiple affine and non-linear layer boundaries without vanishing or diverging.
2. **Symmetry breaking and initialization**: Random weight initialization breaks hidden unit symmetry, allowing distinct neurons to learn complementary decision boundaries.
3. **End-to-end convergence**: The five-step training loop drives empirical loss to zero on XOR and converges successfully on real multiclass image data (TinyDigits).

This milestone synthesizes the foundations of TinyTorch into a complete working deep learning system, demonstrating that our modular kernel can learn non-linear representations from scratch.""",

    "09_convolutions.qmd": r"""Visual and spatial data exhibit two fundamental physical properties: spatial locality and translation equivariance. Neighboring pixels share strong local correlations, and a visual feature—such as an edge, corner, or texture—maintains identical semantic meaning regardless of where it appears in an image. Fully connected dense layers completely ignore this spatial structure: flattening an image into a 1D vector requires millions of redundant parameters that overfit and fail to generalize across spatial shifts.

From a framework and systems standpoint, convolutional layers solve this by sliding compact, weight-shared receptive field kernels across 2D spatial dimensions. However, implementing 2D sliding convolutions using naive nested loops results in 6-deep nested loops that thrash CPU/GPU caches and run disastrously slowly. The systems architect must resolve key implementation challenges:
1. **The `im2col` lowering transformation**: Rearranging overlapping 2D image patches into matrix rows or columns, lowering the spatial convolution into a single high-performance BLAS matrix multiplication (GEMM) at the cost of temporary memory duplication.
2. **Padding and stride geometry**: Calculating exact output spatial feature map dimensions while maintaining boundary information.
3. **Autograd gradient routing**: Propagating gradients through shared kernel weights and routing spatial derivatives back to input feature maps via transposed convolution routines (`col2im`).

This chapter expands TinyTorch into spatial representation learning, implementing 2D convolutions, max pooling, and the `im2col` engine within our modular autograd framework.""",

    "10_tokenization.qmd": r"""Language models operate over discrete symbolic text, yet neural networks compute strictly over continuous floating-point vectors. The fundamental challenge of natural language processing is bridging this representational divide: how do we convert variable-length human text into structured numerical sequences? Character-level tokenization produces cripplingly long sequences that crush downstream attention mechanisms ($O(S^2)$ memory), while word-level tokenization creates massive, sparse vocabularies that fail completely on unseen words.

From a systems and framework perspective, Byte-Pair Encoding (BPE) provides the optimal engineering compromise: iteratively merging the most frequent adjacent character pairs into subwords, compressing common words into single tokens while preserving subword components for rare terms. Implementing a production-grade tokenizer requires solving three systems problems:
1. **Vocabulary and merge table data structures**: Efficiently tracking frequency counts, priority queues, and ordered merge dictionaries for fast text encoding.
2. **Lossless byte-level fallback**: Ensuring that every possible Unicode character sequence can be represented, eliminating fatal out-of-vocabulary (`<unk>`) token crashes.
3. **Contiguous tensor collation**: Converting variable-length tokenized text into fixed-size, padded integer tensor arrays $(B, S)$ ready to feed downstream hardware memory buses.

This chapter constructs TinyTorch's BPE tokenization engine, establishing the text ingestion bridge that feeds the embedding tables in @sec-embeddings and powers language modeling in @sec-transformers.""",

    "11_embeddings.qmd": r"""Discrete token IDs are arbitrary categorical integers with no intrinsic geometric relationship; token 1542 is not mathematically "closer" to token 1543 than to token 50000. Feeding raw integers into linear layers incorrectly imposes ordinal magnitude on categorical labels. To enable semantic reasoning, discrete token IDs must be mapped into continuous, high-dimensional vector spaces where geometric distance reflects semantic similarity, complemented by positional encodings that supply critical sequence order.

From a framework and systems standpoint, an **embedding layer** is functionally a hardware memory gather operation: a sparse lookup into a large weight table of shape $(V, D)$. Implementing embedding layers in an ML runtime introduces distinct architectural considerations:
1. **Sparse gather vs. one-hot projection**: Looking up embedding rows directly rather than multiplying by massive, sparse one-hot matrices, saving gigabytes of memory.
2. **Gradient routing via scatter-add**: During the backward pass, gradients for repeated token IDs within a batch must be accumulated into shared embedding rows via atomic scatter-add operations.
3. **Positional encoding injection**: Adding learned or fixed sinusoidal coordinate vectors to token representations to restore sequence position information destroyed by permutation-invariant architectures.

This chapter constructs TinyTorch's embedding subsystem, converting discrete token sequences into continuous representation tensors $(B, S, D)$ ready for self-attention in @sec-attention.""",

    "12_attention.qmd": r"""Fixed-window convolutions and sequential recurrent networks struggle to model long-range context and dynamic token interactions. A word's meaning in natural language depends heavily on distant context: in "the server crashed because it overheated," resolving what "it" refers to requires the model to route information dynamically across the sequence. **Scaled dot-product attention** solves this by allowing every token to dynamically query, weigh, and aggregate representations from all other tokens in parallel.

From a systems and framework perspective, multi-head attention introduces one of the most computationally demanding and memory-intensive operations in modern deep learning:
1. **The quadratic memory cliff**: Computing all-to-all query-key similarity matrices incurs an $O(S^2)$ memory and compute footprint, making sequence length $S$ a dominant factor in GPU memory allocation.
2. **Autoregressive causal masking**: Enforcing causality in generative decoders by setting future attention weights to $-\infty$ prior to softmax, ensuring tokens cannot look ahead during training.
3. **Multi-head projection splitting and concatenation**: Projecting input vectors into query, key, and value subspaces, reshaping tensors into multi-head views $(B, H, S, d_k)$, computing parallel scaled dot-products, and projecting concatenated outputs back to hidden dimension $D$.

This chapter builds TinyTorch's multi-head causal attention engine, laying the core communication mechanism that powers the Transformer architecture in @sec-transformers.""",

    "13_transformers.qmd": r"""Stacking attention layers directly leads to gradient instability and signal degradation as network depth increases. To build scalable, multi-layer foundation models, attention must be organized into a robust, repeating architectural block. The **Transformer decoder block** stabilizes deep sequence processing by combining causal multi-head attention with position-wise feed-forward networks (MLPs), Pre-Layer Normalization (Pre-LN), and residual skip connections.

From a framework and systems standpoint, the Transformer block acts as an information highway: the residual connection passes signals directly through network depth without attenuation, while sub-layers compute additive feature updates. Implementing a full Transformer (TinyGPT) requires coordinating multiple runtime contracts:
1. **Pre-LN numerical stability**: Applying layer normalization before attention and MLP sub-layers, ensuring clean gradient backpropagation through the residual stream without exploding variance.
2. **Balanced parameter allocation**: Allocating parameter and FLOP budgets between attention projections ($4D^2$) and MLP feed-forward expansions ($8D^2$).
3. **End-to-end model composition**: Stacking $N$ identical blocks, connecting token and position embeddings at the input, and projecting final hidden states through a vocabulary head to produce next-token logit distributions of shape $(B, S, V)$.

This chapter synthesizes the components of Part II into TinyGPT, TinyTorch's flagship generative language model, setting the stage for text generation in @sec-milestone-2.

::: {#fig-transformer-arch fig-env="figure" fig-pos="htb" fig-cap="**Composing a transformer**: Each block adds two learned updates to the residual stream. The complete GPT stacks these blocks before a final normalization and a separate vocabulary projection." fig-alt="Two panels show a Pre-LN block with two bypass arrows around attention and MLP branches, and a GPT path from token and position embeddings through N blocks to logits of shape B by S by V."}
![](assets/images/diagrams/13_transformers-diag-1.svg)
:::

Module 13 names this model `GPT`, with `TinyGPT` retained as an alias. Its caller supplies token IDs of shape $(B, S)$ and receives logits of shape $(B, S, V)$, one row of next-token scores per position. The training and sampling loop in @sec-milestone-2 will use that contract.""",

    "milestone_02.qmd": r"""A trained language model outputs raw next-token logit distributions, but generating coherent text requires closing the autoregressive loop: predicting the next token, appending it to the prompt, and feeding the extended sequence back into the model. However, naive generation strategies fail: greedy argmax selection produces repetitive, robotic loops, while unconstrained random sampling produces incoherent gibberish. Controllable text generation requires shaping probability distributions through temperature scaling and top-$k$ filtering.

From a systems and framework perspective, Milestone II unites the entire natural language processing stack developed across Part II (tokenization, embeddings, causal self-attention, and Transformer blocks) with the Part I training engine. The runtime must orchestrate several sequential steps:
1. **Self-supervised text modeling**: Formatting plain text corpora into shifted input-target token pairs ($x_{1:t} \to x_{2:t+1}$) and computing cross-entropy loss across all sequence positions simultaneously.
2. **Distribution shaping and sampling**: Implementing numerically stable temperature division and top-$k$ truncation over vocabulary probability vectors in float32.
3. **Autoregressive context management**: Managing the token generation loop, iteratively feeding expanding context prefixes through TinyGPT and streaming generated text back through the tokenizer.

This milestone demonstrates end-to-end generative capability in TinyTorch, taking raw text, training TinyGPT from scratch, and sampling coherent language.

::: {.column-margin}
**Milestone Scope:**
Milestone II integrates Modules 09 through 13 (`tinytorch/src/09_convolutions` through `tinytorch/src/13_transformers`) with the Part I engine. The convolution module from @sec-convolutions is exercised by the course's CNN milestone; this milestone uses the tokenizer, embeddings, attention, and transformer modules to train and sample a small GPT.
:::""",

    "14_profiling.qmd": r"""In machine learning systems, optimizing without measurement is guesswork. As models scale from millions to billions of parameters, understanding where execution time and memory are spent is essential. A practitioner must be able to inspect a model architecture and quantitatively predict its resource requirements: how many FLOPs does a forward pass consume, how much memory is dedicated to parameters versus activations, and what hardware limits govern execution speed?

From a framework and systems standpoint, profiling provides the formal accounting tools necessary to diagnose computational bottlenecks:
1. **Parameter and memory accounting**: Calculating exact byte footprints across model parameters, gradients, optimizer states, and forward activation tensors.
2. **FLOP arithmetic and operational intensity**: Deriving the exact theoretical floating-point operations per layer and computing operational intensity (FLOPs per byte moved).
3. **The Roofline model**: Plotting workload operational intensity against machine hardware ceilings to determine definitively whether an operation is **memory-bandwidth bound** (limited by memory bus throughput) or **compute-bound** (limited by arithmetic ALUs).

This chapter establishes TinyTorch's profiling harness, providing the analytical and empirical diagnostic foundations that guide all performance optimizations in Part III.""",

    "15_quantization.qmd": r"""Standard neural network training uses 32-bit floating-point (FP32) arithmetic, but serving large models on edge devices and production servers is heavily constrained by memory capacity, memory bandwidth, and power consumption. Post-training quantization (PTQ) addresses these limits by converting 32-bit floating-point weights and activations into compact 8-bit integers (INT8), slashing storage requirements by $4\times$ and drastically accelerating inference while preserving accuracy.

From a systems and framework perspective, quantization requires mapping continuous real numbers onto discrete integer grids through rigorous affine transformations:
1. **Affine quantization arithmetic**: Deriving scale ($S$) and zero-point ($Z$) parameters ($q = \text{round}(x/S) + Z$) across symmetric and asymmetric mapping schemes.
2. **Integer GEMM execution**: Executing matrix multiplications using fast integer arithmetic while accumulating intermediate products in 32-bit integer accumulators to prevent overflow.
3. **Dequantization and precision calibration**: Scaling accumulated integers back into floating-point representation, balancing numerical dynamic range against quantization noise.

This chapter constructs TinyTorch's quantization engine, demonstrating how to compress weights to INT8 and measure the resulting storage savings and numerical fidelity.""",

    "16_compression.qmd": r"""Deploying multi-layer transformers on resource-constrained devices often requires reducing model parameter counts beyond what quantization alone can achieve. However, training small models from scratch frequently yields poor generalization. Knowledge distillation resolves this challenge by transferring "dark knowledge"—the subtle, rich probability distributions over incorrect classes learned by a high-capacity teacher model—into a compact, efficient student model.

From a framework and systems standpoint, knowledge distillation introduces unique runtime orchestration requirements:
1. **Dual-model graph execution**: Running forward passes through both teacher and student models concurrently, freezing teacher parameters while propagating gradients exclusively through the student.
2. **Temperature-scaled soft loss**: Softening teacher logits via temperature scaling ($T > 1$) and computing Kullback-Leibler (KL) divergence alongside standard task cross-entropy.
3. **Gradient scale normalization**: Scaling the distillation loss component by $T^2$ to maintain gradient magnitude equilibrium between hard and soft objectives.

This chapter implements TinyTorch's model compression pipeline, demonstrating how knowledge distillation enables compact student networks to retain high accuracy at a fraction of the original computational cost.""",

    "17_acceleration.qmd": r"""In modern deep learning execution, arithmetic operations like addition, scaling, and non-linear activations execute on compute units in fractions of a microsecond. However, in eager execution runtimes, chaining separate operations forces intermediate tensors to make round-trip journeys across the slow off-chip memory bus (DRAM/HBM), incurring devastating latency and memory bandwidth overhead.

From a framework and systems standpoint, **operator fusion** is the primary compiler optimization used to break the memory bandwidth wall:
1. **Eliminating memory round-trips**: Combining sequential elementwise operations (such as linear bias addition followed by GELU activation) into a single compiled kernel execution pass.
2. **On-chip register reuse**: Retaining intermediate values in ultra-fast on-chip processor registers or SRAM rather than allocating temporary tensors in global DRAM.
3. **The memory traffic ledger**: Quantifying exact memory traffic reductions, proving how fusing an 8-operation GELU expression slashes DRAM bytes moved by up to $9\times$.

This chapter implements TinyTorch's operator fusion engine, demonstrating how kernel fusion transforms memory-bound operator sequences into peak hardware performance.""",

    "18_memoization.qmd": r"""Autoregressive text generation produces output one token at a time: each generated token is appended to the context, and the extended sequence is passed back to the model. In a naive implementation, generating $S$ tokens requires re-running attention across all preceding tokens at every single step, creating an $O(S^2)$ computational disaster where 99% of operations redundantly recompute identical key and value representations.

From a systems and framework perspective, the **Key-Value (KV) cache** is the essential serving optimization that eliminates this recomputation:
1. **Static buffer pre-allocation**: Pre-allocating persistent tensor memory buffers for attention keys and values up to the maximum sequence length, avoiding costly dynamic reallocations.
2. **Moving write cursor**: Incrementing a sequential write pointer to store only the new token's key and value projections during each decode step.
3. **Asymptotic step reduction**: Transforming per-token attention computation from $O(S)$ back down to $O(1)$, unlocking massive serving speedups for long context windows.

This chapter constructs TinyTorch's KV cache subsystem, proving mathematical equivalence with full-prefix attention while demonstrating dramatic reductions in generation latency.""",

    "19_benchmarking.qmd": r"""Performance claims in machine learning systems—such as "this optimization provides a $3\times$ speedup"—are completely meaningless without rigorous, reproducible benchmarking methodology. Naive timing using basic wall-clock timers produces misleading results contaminated by Python interpreter overhead, cache cold starts, asynchronous execution queues, and operating system noise.

From a framework and systems standpoint, building a publication-grade benchmarking harness requires strict scientific controls:
1. **Warmup phase discipline**: Executing un-timed warmup iterations to prime instruction caches, pre-allocate heap memory, and trigger JIT compilation before taking measurements.
2. **High-resolution monotonic timing**: Measuring isolated execution windows with microsecond-resolution monotonic clocks (`time.perf_counter`) and ensuring hardware synchronization.
3. **Statistical distribution reporting**: Going beyond misleading mean averages to measure and report variance, standard deviations, and latency percentiles (P50, P90, P99), alongside system throughput (tokens/sec).

This chapter builds TinyTorch's benchmarking suite, establishing the rigorous empirical measurement standards required for the Capstone evaluation in @sec-capstone.""",

    "20_capstone.qmd": r"""Real-world machine learning systems engineering is never about applying a single optimization in isolation; it is an exercise in multi-objective trade-offs. An optimization that doubles inference speed is unacceptable if it degrades model accuracy below acceptable limits. Similarly, a compression technique that reduces storage is worthless if it introduces excessive runtime latency. The systems engineer must synthesize multiple optimizations into a coherent, balanced production pipeline.

From a framework and systems standpoint, the Capstone integrates the full optimization arsenal developed across Part III:
1. **End-to-end optimization pipeline**: Systematically combining profiling, INT8 post-training quantization, knowledge distillation, and KV cache memoization on TinyGPT.
2. **Pareto frontier analysis**: Mapping the multi-dimensional trade-offs between model accuracy retention, storage compression ratios, and inference latency speedups.
3. **Verification and scorecard reporting**: Enforcing strict verification gates to ensure optimized models produce valid predictions that pass classroom and production quality thresholds.

This chapter guides the student through the complete optimization and benchmarking of TinyGPT, synthesizing the principles of the entire TinyTorch curriculum.""",

    "milestone_03.qmd": r"""In competitive systems benchmarks like MLPerf, raw speedup numbers and compression ratios mean nothing if the candidate model compromises accuracy. A system optimization is only valid if it satisfies strict accuracy retention floors while delivering verifiable reductions in memory footprint and latency under concrete edge hardware constraints.

From a systems and framework perspective, Milestone III represents the ultimate competitive arena of TinyTorch: The Torch Olympics. Students deploy their optimized TinyGPT and DigitMLP models against strict scoring gates:
1. **The three-axis scorecard**: Evaluating candidate submissions across accuracy retention ($R \ge 0.99$), stored size compression ratio ($C$), and runtime speedup ($S$).
2. **Edge deployment constraints**: Operating under strict memory ceilings where unoptimized models trigger out-of-memory crashes, requiring quantized weights and cached attention to fit within resource budgets.
3. **Rigorous empirical verification**: Generating auditable benchmark reports that compare optimized candidates against frozen baseline models on identical workloads.

This milestone concludes the TinyTorch journey, proving that students have mastered not just deep learning theory, but the systems engineering discipline required to build and deploy efficient ML runtimes.

::: {.column-margin}
**Milestone Scope:**
Milestone III integrates Modules 14 through 19 (`tinytorch/src/14_profiling` through `tinytorch/src/19_benchmarking`) with the models of Part I and Part II. The course numbers this milestone 06 and keeps it in `tinytorch/milestones/06_2018_mlperf/`; it has two scripts. `01_optimization_olympics.py` trains the `DigitMLP` of @sec-milestone-1 on TinyDigits, then profiles, quantizes, prunes, and benchmarks it. `02_generation_speedup.py` checks per-position logits before timing full-prefix and cached GPT execution. Neither script is a line-by-line match for the listings here; the listings are a pure-Python model of the same scorecard, small enough to check by hand.
:::""",

    "21_extensions.qmd": r"""While TinyTorch implements an educational reference runtime in pure Python and NumPy, production deep learning systems at hyperscale operate through sophisticated compiler stacks and dedicated custom silicon. Modern ML systems do not execute tensor operations one by one in an interpreter; they capture computational graphs, generate optimized GPU machine code, and dispatch workloads to specialized matrix acceleration hardware.

From a systems and architecture perspective, this chapter reaches beyond the educational interpreter to explore the frontiers of modern ML systems:
1. **Computation graph capture**: Tracing dynamic Python programs into intermediate representations (IR) using graph capture engines like TorchDynamo and JAX JIT.
2. **Automated kernel compilation**: Lowering tensor operations to high-performance GPU machine code via domain-specific compiler frameworks like OpenAI Triton and PyTorch Inductor.
3. **Custom domain-specific hardware**: Analyzing weight-stationary systolic arrays (such as the Google TPU Matrix Multiply Unit) that achieve massive energy efficiency by streaming operands across 2D grids of physical arithmetic units.
4. **Distributed data parallelism**: Scaling training across clusters of GPUs using collective communication algorithms (Ring-AllReduce) to synchronize parameter gradients across the network.

This final chapter completes the bridge from the educational xv6 kernel of TinyTorch to the high-performance compiler and hardware architectures powering modern artificial intelligence."""
}

def get_original_fig(filepath: Path) -> str:
    """Extracts original figure block from the introductory section of the file from git HEAD."""
    try:
        out = subprocess.check_output(
            ['git', 'show', f'HEAD:tinytorch/book/{filepath.name}'],
            text=True, encoding="utf-8", errors="replace"
        )
    except Exception:
        out = filepath.read_text(encoding="utf-8")
        
    m_margin = re.search(r'(:::\s*\{\.column-margin\}[\s\S]*?:::)\s*\n', out)
    if not m_margin:
        return ""
    rest = out[m_margin.end():]
    m_boundary = re.search(r'\n(---|## The Problem)\n', rest)
    if not m_boundary:
        return ""
    intro_block = rest[:m_boundary.start()].strip()
    
    # Check if there is an ending figure
    m_fig = re.search(r'(!\[[\s\S]*?\]\([^\)]+\)\{[^\}]+\}|:::\s*\{#fig-[\s\S]*?:::)\s*$', intro_block)
    if m_fig:
        return m_fig.group(0).strip()
    return ""

def update_file(filepath: Path, new_prose: str):
    text = filepath.read_text(encoding="utf-8")
    
    m_margin = re.search(r'(:::\s*\{\.column-margin\}[\s\S]*?:::)\s*\n', text)
    if not m_margin:
        print(f"ERROR: Could not find column-margin blueprint in {filepath.name}")
        return False
    
    margin_end = m_margin.end()
    rest = text[margin_end:]
    
    m_next = re.search(r'\n(---|## The Problem)\n', rest)
    if not m_next:
        print(f"ERROR: Could not find section boundary in {filepath.name}")
        return False
    
    intro_end = margin_end + m_next.start()
    
    # Get original figure from HEAD (if any)
    orig_fig = get_original_fig(filepath)
    
    # Assemble replacement
    if orig_fig:
        # Check if new_prose already includes this figure
        if orig_fig in new_prose:
            full_intro = new_prose.strip()
        else:
            full_intro = f"{new_prose.strip()}\n\n{orig_fig}"
    else:
        full_intro = new_prose.strip()
        
    new_text = text[:margin_end] + "\n" + full_intro + "\n" + text[intro_end:]
    filepath.write_text(new_text, encoding="utf-8")
    print(f"SUCCESS: Updated {filepath.name}")
    return True

def main():
    print(f"Applying pedagogical openings across {len(PROSE_BRIDGES)} files...")
    count = 0
    for filename, prose in PROSE_BRIDGES.items():
        fp = BASE_DIR / filename
        if not fp.exists():
            print(f"ERROR: {fp} missing!")
            continue
        if update_file(fp, prose):
            count += 1
    print(f"Done. Successfully updated {count}/{len(PROSE_BRIDGES)} files.")

if __name__ == "__main__":
    main()
