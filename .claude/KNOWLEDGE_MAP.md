# Knowledge Progression Map

This document tracks what concepts are taught in each chapter. The content listing comes from our design and has been validated against actual chapter content.

## Core Principles

1. **Historical Context is Always Acceptable**: Terms like "deep learning", "AlexNet", "GPT-3" can be mentioned as historical facts or examples in any chapter
2. **Technical Details Must Wait**: The actual mechanics (how things work) should only be explained in their designated chapter  
3. **Use Footnotes for Forward References**: If mentioning a future concept, add a footnote like "explained in detail in Chapter X"
4. **Preserve Accuracy**: Never change technical terms to incorrect alternatives

## Part I: Systems Foundations

### Chapter 1: Introduction
**Teaches:**
- What is ML Systems Engineering
- Historical evolution of AI (symbolic → statistical → deep learning)
- Why ML systems are challenging
- Systems thinking for ML
- Real-world deployment considerations
- Cross-functional collaboration needs
- Case studies: FarmBeats, AlphaFold, Waymo

**Mentions (with footnotes):** Neural networks, deep learning, AlexNet, GPT-3, backpropagation (as historical milestones)

### Chapter 2: ML Systems
**Teaches:**
- Deployment tiers (Cloud, Edge, Mobile, TinyML)
- Resource constraints (memory, compute, power)
- System architectures for ML
- Latency vs throughput trade-offs
- Infrastructure requirements
- ML system design patterns
- Scalability considerations

### Chapter 3: Deep Learning Primer
**Teaches:**
- How neural networks actually work
- Neurons, weights, biases, connections
- Activation functions (what they are, why needed)
- Forward and backward propagation algorithms
- Gradient descent and optimization
- Loss functions
- Training vs inference
- Overfitting, underfitting, regularization

### Chapter 4: DNN Architectures
**Teaches:**
- Multi-Layer Perceptrons (MLPs)
- Convolutional Neural Networks (CNNs) - layers, filters, pooling
- Recurrent Neural Networks (RNNs) - sequential processing
- Transformers and attention mechanisms
- Architectural patterns and design choices
- Layer types and configurations
- Model depth vs width trade-offs
- Skip connections and residual blocks

## Part II: Design Principles

### Chapter 5: Workflow
**Teaches:**
- ML development workflow stages
- Data collection and preparation
- Model development pipeline
- Experiment tracking
- Version control for ML
- Hyperparameter tuning
- Model validation strategies
- Deployment pipelines

### Chapter 6: Data Engineering
**Teaches:**
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
**Teaches:**
- ML frameworks (TensorFlow, PyTorch, JAX)
- Computational graphs
- Automatic differentiation
- Framework APIs and abstractions
- Model serialization formats
- Framework interoperability
- Hardware backend support
- Distributed training support

### Chapter 8: Training
**Teaches:**
- Training algorithms in detail
- Optimization techniques (SGD, Adam, etc.)
- Learning rate scheduling
- Batch processing strategies
- Distributed training methods
- Data parallelism
- Model parallelism
- Mixed precision training
- Checkpointing and recovery

## Part III: Performance Engineering

### Chapter 9: Efficient AI
**Teaches:**
- AI scaling laws
- Power-law relationships
- Algorithmic efficiency
- Compute efficiency
- Data efficiency
- Efficiency trade-offs
- Sustainability considerations
- Resource optimization strategies

### Chapter 10: Model Optimizations
**Teaches:**
- Quantization (INT8, FP16)
- Pruning (structured, unstructured)
- Knowledge distillation
- Model compression
- Sparsity
- Neural Architecture Search (NAS)
- Hardware-aware optimization

### Chapter 11: AI Acceleration
**Teaches:**
- Hardware accelerators (GPUs, TPUs, NPUs)
- Specialized AI chips
- Memory hierarchies
- Hardware-software co-design
- Kernel optimization
- Compiler optimizations for ML
- SIMD/SIMT architectures
- Tensor cores

### Chapter 12: Benchmarking AI
**Teaches:**
- Performance metrics
- Benchmarking methodologies
- Standard benchmarks (MLPerf, etc.)
- Profiling and analysis
- Performance debugging
- System-level metrics
- Energy efficiency measurement

## Part IV: Deployment Engineering

### Chapter 13: ML Operations
**Teaches:**
- MLOps principles
- CI/CD for ML
- Model monitoring
- A/B testing
- Model versioning
- Data drift detection
- Deployment strategies
- Production debugging

### Chapter 14: On-Device Learning
**Teaches:**
- Federated learning
- Transfer learning
- Edge training techniques
- Privacy-preserving ML
- Continual learning
- Model personalization
- Resource-constrained training

### Chapter 15: Robust AI
**Teaches:**
- Adversarial robustness
- Distribution shifts
- Out-of-distribution detection
- Model uncertainty
- Failure modes
- Safety engineering
- Testing strategies

### Chapter 16: Privacy & Security
**Teaches:**
- Privacy-preserving techniques
- Differential privacy
- Secure multi-party computation
- Model extraction attacks
- Data poisoning
- Membership inference
- Homomorphic encryption

## Part V: Societal Impact

### Chapter 17: Responsible AI
**Teaches:**
- Fairness metrics
- Bias detection and mitigation
- Explainability methods
- Accountability frameworks
- Ethical considerations
- Regulatory compliance
- Impact assessment

### Chapter 18: Sustainable AI
**Teaches:**
- Carbon footprint of AI
- Green AI practices
- Energy-efficient training
- Sustainable deployment
- Lifecycle assessment
- Environmental impact

### Chapter 19: AI for Good
**Teaches:**
- Healthcare applications
- Climate change solutions
- Education technology
- Accessibility
- Social good applications
- Global development

### Chapter 20: Conclusion
**Synthesizes:**
- Key principles recap
- Future directions
- Emerging challenges
- Call to action

## Usage Guidelines

### For Review/Edit Agents:
- Historical mentions → Always OK
- Technical explanations → Only in designated chapter
- When in doubt → Add footnote, don't replace

### Key Distinction:
- **Mentioning** "deep learning revolutionized computer vision" → OK anywhere
- **Explaining** "deep learning uses backpropagation to adjust weights" → Only Chapter 3+