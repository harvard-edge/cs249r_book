# Part I: The Model - Gap Analysis Report

## Chapter 2: Foundation Model

### Covered Topics
- **Model Invocation Boundary & Representation**: Discrete token representation, context budgeting, KV cache capacity bounds.
- **Autoregressive Generation Mechanics**: Causal serialization barrier, logit scaling and truncation, vocabulary simplex sampling.
- **Candidate Sequence Verification**: The epistemic gap, memory escrow, fallacy of autoregressive self-checking, deterministic verification perimeters.
- **The Invocation Contract**: Formal request specification, streaming execution mechanics, early stream cancellation reclamation, normalized status envelopes.
- **Grammar-Guided Decoding**: Automata-driven token masking, accelerator bitmask execution latency footprints.
- **Accelerator Serving Latency**: The Roofline model, compute-bound prefill vs. memory-bound decode, the memory bandwidth shuttle, multi-turn cost accumulation.
- **Interface Benchmarking**: Controlled benchmarking under invariant resource envelopes, evaluation metrics from syntactic compliance to verified task closure.

### Strengths
- Provides a rigorous, systems-level mathematical treatment of inference costs (e.g., Roofline model, explicit FLOP and memory bandwidth bounds).
- Clearly distinguishes syntactic correctness (e.g., valid JSON) from semantic/end-to-end task success, correctly identifying over-optimization traps in structured generation.
- Thoroughly covers the mechanics and memory/latency implications of grammar-guided decoding and early stream cancellation.

### Missing Topics
- **Speculative Decoding**: While the chapter thoroughly establishes that decode is severely bottlenecked by memory bandwidth, it completely misses the primary systems technique deployed to overcome this limit—speculative decoding (using small draft models or Medusa heads to generate candidate tokens for parallel verification). Given the chapter's focus on verification perimeters and memory shuttling, this is a critical systems omission.
  - *Priority*: Critical
- **Continuous Batching and Chunked Prefill**: The chapter models generation latency at batch size $B=1$. Production agent runtimes rely heavily on iteration-level continuous batching (Orca) to achieve high throughput, and chunked prefill to prevent massive context uploads from starving decode iterations. This dynamic is fundamental to agent systems engineering where output lengths are highly unpredictable.
  - *Priority*: Critical
- **Multi-modal Context Processing**: Agentic systems increasingly rely on vision capabilities (e.g., computer use, chart reading, screenshot analysis). The chapter currently focuses purely on text tokens. The mechanics of spatial patch encoding, vision encoders, and how images consume context budgets (e.g., a 1024x1024 image costing ~1-3k tokens depending on architecture) must be addressed.
  - *Priority*: Critical
- **Mixture of Experts (MoE) and Quantization Constraints**: The algebraic latency and parameter bounds presented assume dense, full-precision models. Production agentic inference heavily utilizes MoE (decoupling active parameter count from total capacity) and KV/weight quantization (FP8, INT8, AWQ).
  - *Priority*: Important

### Under-Developed Areas
- **Native Tool-Calling via Fine-Tuning vs. Inference Masks**: While grammar masks are covered deeply, the chapter lacks a comparative systems evaluation against models natively fine-tuned for structured tool emission (e.g., OpenAI's function calling) which often bypass the need for expensive logit masking at decode time.

### Suggested Additions
- *Breaking the Memory Barrier: Speculative Decoding and Parallel Verification*
- *Serving Throughput: Continuous Batching and Chunked Prefill*
- *Multi-modal Context: Vision Encoding and Spatial Budgets*
- *Architectural Alterations: Mixture of Experts and Precision Scaling*

---

## Chapter 3: Test-Time Compute

### Covered Topics
- **Compute Allocation Axes**: Depth (Sequential token extension, reasoning models), Breadth (Parallel candidate sampling), Feedback (Observation-conditioned revision).
- **Candidate Selection Mechanisms**: Diversity dynamics, architectures (self-consistency, learned verifier scoring, deterministic execution filtering), Pass@$k$ ceilings.
- **Process Verification**: Verifier taxonomies, the asymmetry of verification error (false acceptance vs. false rejection), robust verification topologies (PRM ensembles, deterministic structural guards, property-based invariants).
- **Plans as Revisable State**: Plan DAG representations, precondition verification, hierarchical revision mechanics (local repair, subgraph resynthesis, global escalation).
- **Search Stopping Criteria**: Search topologies (MCTS, Best-of-$N$), unified resource accounting models, multicriteria stopping invariants.
- **Deliberation Strategy Evaluation**: Controlled systems evaluation (resource equivalence), multi-dimensional metrics (Pass@$k$ vs wall-clock latency distributions), mapping the deliberation Pareto frontier.

### Strengths
- Exceptional formulation of hierarchical plan revision, treating plans as DAGs and applying structured systems principles (e.g., Saltzer and Kaashoek's modular error containment) to prompt-based planning.
- Rigorous approach to search stopping and Pareto frontier evaluation, forcing different deliberation strategies (ReAct, MCTS, Best-of-N) into a matched-token cost accounting ledger.
- Correctly identifies that environmental grounding (sandbox execution) is far superior to ungrounded internal neural search for scaling test-time compute securely.

### Missing Topics
- **Ephemeral Test-Time Training (TTT)**: Recent advances in test-time scaling extend beyond prompt context and token generation to include ephemeral gradient updates during reasoning. Adapting the model weights on the fly based on intermediate verification represents a new axis of test-time compute.
  - *Priority*: Important
- **Systems Architecture of Learned Verifiers**: While PRM ensembles are mentioned conceptually, the chapter ignores the physical deployment topology. Do the generator and PRM share weights? Are they co-located on the same node to avoid network hops during tight tree-search loops?
  - *Priority*: Important

### Under-Developed Areas
- **Reward Hacking and Verifier Exploitation**: The chapter briefly notes an "$11\%$ false acceptance rate from verifier Goodharting" in a table. The dynamics of how agent policies iteratively search for and exploit edge cases in learned verifiers (Reward Hacking)—a massive reliability issue in reasoning models like o1 or DeepSeek-R1—requires a dedicated subsection under the verification asymmetry.
  - *Priority*: Critical
- **Scaffolding Frameworks**: The orchestration overhead of maintaining search state (MCTS, Best-of-N) across distributed microservices (e.g., using DSPy, LangGraph) is hinted at but lacks concrete runtime patterns. (Note: This may be acceptable if heavily covered in Part IV: Orchestration, but a forward-reference is needed).

### Suggested Additions
- *Reward Hacking and Verifier Exploitation*
- *Systems Co-location of Generators and Learned Verifiers*
- *Beyond Activations: Ephemeral Test-Time Training*
