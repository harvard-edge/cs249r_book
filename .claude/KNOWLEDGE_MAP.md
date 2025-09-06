# Knowledge Progression Map

This document tracks what concepts are introduced in each chapter of the ML Systems textbook. Chapters should primarily explain their own concepts while building on previous chapters.

## Guiding Principles

1. **Historical Context is Always Acceptable**: Terms like "deep learning", "AlexNet", "GPT-3" can be mentioned as historical facts or examples in any chapter
2. **Technical Details Must Wait**: The actual mechanics (how things work) should only be explained in their designated chapter
3. **Use Footnotes for Forward References**: If mentioning a future concept, add a footnote like "explained in detail in Chapter X"
4. **Preserve Accuracy**: Never change technical terms to incorrect alternatives (e.g., don't replace "deep learning" with "hierarchical learning")

## Part I: Systems Foundations

### Chapter 1: Introduction
**Introduces:**
- What is ML Systems Engineering
- Historical evolution of AI (symbolic → statistical → deep learning)
- Why ML systems are challenging
- Systems thinking for ML
- Real-world deployment considerations
- Cross-functional collaboration needs
- Case studies: FarmBeats, AlphaFold, Waymo

### Chapter 2: ML Systems
**Introduces:**
- Deployment tiers (Cloud, Edge, Mobile, TinyML)
- Resource constraints (memory, compute, power)
- System architectures for ML
- Latency vs throughput trade-offs
- Infrastructure requirements
- ML system design patterns
- Scalability considerations

### Chapter 3: Deep Learning Primer
**Introduces:**
- How neural networks actually work
- Neurons, weights, biases, connections
- Activation functions
- Forward and backward propagation
- Gradient descent and optimization
- Loss functions
- Training vs inference
- Overfitting, underfitting, regularization

### Chapter 4: DNN Architectures
**Introduces:**
- Multi-Layer Perceptrons (MLPs)
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Transformers and attention mechanisms
- Architectural patterns and design choices
- Layer types and configurations
- Model depth vs width trade-offs
- Skip connections and residual blocks

## Part II: Design Principles

### Chapter 5: Workflow
**Introduces:**
- ML development workflow stages
- Data collection and preparation
- Model development pipeline
- Experiment tracking
- Version control for ML
- Hyperparameter tuning
- Model validation strategies
- Deployment pipelines

### Chapter 6: Data Engineering
**Introduces:**
- Data pipelines
- Data quality and validation
- Feature engineering
- Data preprocessing techniques
- Data storage formats
- Batch vs streaming data
- Data versioning
- ETL/ELT processes
- Data governance

### Chapter 7: Frameworks
**New Concepts Introduced:**
- ML frameworks (TensorFlow, PyTorch, etc.)
- Computational graphs
- Automatic differentiation
- Framework APIs and abstractions
- Model serialization formats
- Framework interoperability
- Hardware backend support
- Distributed training support

**Can Use:** Chapters 1-6 concepts
**Cannot Use:** Hardware acceleration specifics, quantization

### Chapter 8: Training
**New Concepts Introduced:**
- Training algorithms in detail
- Optimization techniques
- Learning rate scheduling
- Batch processing strategies
- Distributed training methods
- Data parallelism
- Model parallelism
- Mixed precision training
- Checkpointing and recovery

**Can Use:** Chapters 1-7 concepts
**Cannot Use:** Deployment optimizations, quantization

## Part III: Performance Engineering

### Chapter 9: Efficient AI
**New Concepts Introduced:**
- AI scaling laws
- Power-law relationships
- Algorithmic efficiency
- Compute efficiency
- Data efficiency
- Efficiency trade-offs
- Sustainability considerations
- Resource optimization strategies

**Can Use:** Chapters 1-8 concepts
**Cannot Use:** Specific compression techniques

### Chapter 10: Model Optimizations
**New Concepts Introduced:**
- Quantization
- Pruning
- Knowledge distillation
- Model compression
- Sparsity
- Neural Architecture Search (NAS)
- Hardware-aware optimization
- Numerical precision formats (FP16, INT8)

**Can Use:** Chapters 1-9 concepts
**Cannot Use:** Hardware accelerators specifics

### Chapter 11: AI Acceleration
**New Concepts Introduced:**
- Hardware accelerators (GPUs, TPUs, NPUs)
- Specialized AI chips
- Memory hierarchies
- Hardware-software co-design
- Kernel optimization
- Compiler optimizations for ML
- SIMD/SIMT architectures
- Tensor cores

**Can Use:** Chapters 1-10 concepts

### Chapter 12: Benchmarking AI
**New Concepts Introduced:**
- Performance metrics
- Benchmarking methodologies
- MLPerf and other benchmarks
- System-level evaluation
- Throughput and latency measurement
- Power and energy profiling
- Reproducibility in benchmarking
- Statistical significance testing

**Can Use:** Chapters 1-11 concepts

## Part IV: Robust Deployment

### Chapter 13: ML Operations
**New Concepts Introduced:**
- MLOps principles
- CI/CD for ML
- Model monitoring
- Data drift and concept drift
- Model versioning and registry
- A/B testing for ML
- Feature stores
- Model observability

**Can Use:** Chapters 1-12 concepts

### Chapter 14: On-Device Learning
**New Concepts Introduced:**
- Edge training
- Federated learning
- Transfer learning
- Fine-tuning
- Continual learning
- Privacy-preserving ML
- Model personalization
- Resource-constrained training

**Can Use:** Chapters 1-13 concepts

### Chapter 15: Robust AI
**New Concepts Introduced:**
- Model robustness
- Adversarial attacks and defenses
- Out-of-distribution detection
- Uncertainty quantification
- Model calibration
- Failure modes and mitigation
- Safety-critical ML systems

**Can Use:** Chapters 1-14 concepts

### Chapter 16: Privacy & Security
**New Concepts Introduced:**
- Differential privacy
- Secure multi-party computation
- Homomorphic encryption
- Privacy attacks (membership, reconstruction)
- Model stealing and extraction
- Backdoor attacks
- Secure enclaves
- Privacy regulations (GDPR, HIPAA)

**Can Use:** Chapters 1-15 concepts

## Part V: Trustworthy Systems

### Chapter 17: Responsible AI
**New Concepts Introduced:**
- AI ethics frameworks
- Fairness metrics and definitions
- Bias detection and mitigation
- Algorithmic transparency
- Explainable AI (XAI)
- Model interpretability
- Accountability in AI systems
- Regulatory compliance

**Can Use:** Chapters 1-16 concepts

### Chapter 18: Sustainable AI
**New Concepts Introduced:**
- Carbon footprint of ML
- Energy-efficient computing
- Green AI initiatives
- Lifecycle assessment
- Sustainable hardware design
- Carbon-aware computing
- Environmental impact metrics

**Can Use:** Chapters 1-17 concepts

### Chapter 19: AI for Good
**New Concepts Introduced:**
- Social impact applications
- Healthcare AI systems
- Environmental monitoring
- Educational technology
- Accessibility features
- Humanitarian applications
- Development considerations

**Can Use:** Chapters 1-18 concepts

## Part VI: Frontiers

### Chapter 20: Conclusion
**New Concepts Introduced:**
- Future trends and directions
- Emerging architectures
- Research frontiers
- Industry evolution
- Societal implications

**Can Use:** All previous chapters (1-19)

## Usage Guidelines

When reviewing or improving any chapter:

1. **Check chapter number** - Identify which chapter you're working on
2. **List available concepts** - All concepts from chapters 1 through (N-1)
3. **List forbidden concepts** - Any concepts from chapter N+1 onwards
4. **Flag violations** - Identify any forward references
5. **Suggest alternatives** - Replace with concepts already introduced

## Common Forward Reference Violations

| Term | First Introduced | Safe Alternative Before Introduction |
|------|-----------------|--------------------------------------|
| Quantization | Chapter 10 | "optimization techniques" or "efficiency methods" |
| GPUs/TPUs | Chapter 11 | "specialized hardware" or "accelerators" |
| MLOps | Chapter 13 | "operational practices" or "deployment processes" |
| Federated Learning | Chapter 14 | "distributed approaches" or "collaborative methods" |
| Differential Privacy | Chapter 16 | "privacy techniques" or "data protection methods" |
| Fairness/Bias | Chapter 17 | "model behavior" or "system characteristics" |

## Progressive Building Examples

### Good Progression:
- Ch 3: "Neural networks process data through layers"
- Ch 4: "CNNs use convolutional layers for image processing"
- Ch 10: "We can quantize CNN weights to reduce memory"

### Bad Progression (Forward Reference):
- Ch 3: "Neural networks can be quantized for efficiency" ❌
- Ch 4: "CNNs run efficiently on GPUs" ❌
- Ch 8: "Training uses MLOps pipelines" ❌

This knowledge map ensures that the textbook builds concepts progressively, never using undefined terms or assuming knowledge not yet introduced.