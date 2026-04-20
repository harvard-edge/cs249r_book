# MLPerf EDU: Design Philosophy

## The SPEC for Machine Learning Pedagogy
Machine learning systems are notoriously difficult to profile without deep C++ and CUDA expertise. While the original **MLPerf (MLCommons)** is the enterprise gold standard for benchmarking datacenters, it is heavily bound by vendor-specific optimized submodules (e.g., cuDNN, OneDNN, TensorRT). It is not designed to be easily analyzed inside a classroom.

**MLPerf EDU** is built on the philosophy of the venerable **SPEC CPU Benchmarks**. Its primary purpose is not to reward the absolute fastest hardware, but to provide a canonical vessel for pedagogical analysis, architectural profiling, and academic research.

### Core Tenets

#### 1. White-Box Algorithms
Every benchmark is written in sub-300 lines of pure Python/PyTorch. We explicitly forbid opaque C++ hardware bindings for core algorithmic loops during training/inference references. If a student wants to see exactly how KV-Caching is bottlenecking a transformer, they can introspect the raw Python dictionary carrying the cache matrix.

#### 2. Canonical Provenance
We do not invent random ML topologies. Every workload correlates explicitly with a foundational paper:
- **Cloud/LLM:** GPT-2 (Radford et al., 2019)
- **Edge/Vision:** ResNet (He et al., 2015)
- **Mobile/Detection:** MobileNetV2 (Sandler et al., 2018)
- **TinyML/KWS:** DS-CNN (Zhang et al., 2017)

#### 3. Surgical Telemetry over Anti-Cheating
Traditional MLPerf goes to extraordinary lengths (cryptographic dataset hashing, strict PRNG enforcement) to prevent hyper-scalers from "cheating" the benchmark. Because we operate in an academic/pedagogical setting, we trade draconian anti-cheating measures for **Introspection Hooking**. Our `Referee` captures Roofline Arithmetic Intensity (FLOPs/Byte), Dataloader I/O blocking percentages, and localized energy (Joules) without breaking the execution flow.

#### 4. The Full Provenance Loop
The benchmark integrates Training explicitly with Inference. A student trains an architecture until they hit the YAML-defined generic target accuracy. Upon success, the system emits a `.provd` (Provenance Artifact) capturing the frozen `state_dict` and its SHA-256 hash. The Inference lab enforces the ingestion of this specific `.provd` artifact, closing the pedagogical loop.

#### 5. Canonical Hydration
Acknowledging that students lack datacenter compute to pre-train LLMs locally, the framework introduces the **Hydration Layer**. Using `mlperf hydrate`, the framework autonomously downloads canonical industry weights (e.g. HuggingFace GPT-2), hashes them, and packages them securely into a `.provd` artifact. This ensures students always interact with mathematically validated tensors during Inference, maintaining strict academic provenance.

#### 6. The Systems Under Test (SUT) Plugin Protocol
MLPerf EDU enforces a rigid separation between the LoadGen Referee and the SUT Implementation. Students never modify the core framework. To submit a custom CUDA optimization, optimized Dataloader, or fused C++ Attention mechanism, they inherit the `SUT_Interface` in an isolated `.py` file. The CLI dynamically intercepts and evaluates their plugin (`--sut student_hw.py`), emitting reproducible MLCommons-style JSON dumps tracking their optimization gains against the host's hardware boundaries.

---
*MLPerf EDU enables researchers to run deep architectural experiments on arbitrary local hardware with zero setup, generating publication-ready telemetry out-of-the-box.*
