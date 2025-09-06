# Knowledge Map V2 - Based on Actual Content

*Generated from actual chapter content, not assumptions*

## Guiding Principles

1. **Historical Context is Always Acceptable**: Terms like "deep learning", "AlexNet", "GPT-3" can be mentioned as historical facts in any chapter
2. **Technical Details Must Wait**: The mechanics of how things work should be explained in their designated chapter
3. **Use Footnotes for Forward References**: When mentioning future concepts, add "details in Chapter X"
4. **Preserve Technical Accuracy**: Never replace terms with incorrect alternatives

## Part I: Systems Foundations

### Chapter 1: Introduction
**Main Sections:**
- AI Evolution (Symbolic → Expert Systems → Statistical → Deep Learning)
- ML Systems Engineering
- Lifecycle of ML Systems
- Case Studies: FarmBeats, AlphaFold, Autonomous Vehicles
- Challenges in ML Systems

**Key Terms Introduced:**
- Machine Learning Systems Engineering (the field)
- ML System Lifecycle
- Systems thinking for ML
- Cross-functional collaboration

**Historical Context Used:**
- Deep learning revolution (2012)
- AlexNet, GPT-3, backpropagation (as historical milestones)
- Neural networks (as concept, not mechanics)

### Chapter 2: ML Systems
**Main Sections:**
- Cloud ML, Edge ML, Mobile ML, TinyML
- Deployment Decision Framework
- System Comparison

**Key Terms Introduced:**
- Deployment tiers and their characteristics
- Resource constraints (memory, compute, power)
- Latency vs throughput trade-offs
- Hybrid architectures

### Chapter 3: Deep Learning Primer
**Main Sections:**
- Biological to Artificial Neurons
- Neural Network Fundamentals
- Learning Process (Forward/Backward Propagation)
- Training vs Inference

**Key Terms Introduced:**
- How neural networks actually work
- Neurons, weights, biases, activation functions
- Backpropagation algorithm
- Gradient descent
- Loss functions

### Chapter 4: DNN Architectures
**Main Sections:**
- Multi-Layer Perceptrons (MLPs)
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Attention Mechanisms

**Key Terms Introduced:**
- Specific architectures and when to use them
- Convolution operations
- Sequential processing
- Attention and transformers

## Part II: Design Principles

### Chapter 5: Workflow
**Main Sections:**
- ML Lifecycle Stages
- Problem Definition → Data → Development → Deployment → Maintenance
- Roles and Responsibilities

### Chapter 6: Data Engineering
**Main Sections:**
- Data Pipelines
- Data Processing and Labeling
- Data Storage and Governance

### Chapter 7: Frameworks
**Main Sections:**
- Framework Evolution
- Computational Graphs
- Automatic Differentiation
- Major Frameworks (TensorFlow, PyTorch, JAX)

### Chapter 8: Training
**Main Sections:**
- Training Systems Architecture
- Distributed Training
- Optimization Techniques
- Hardware Acceleration

### Chapter 9: Efficient AI
**Main Sections:**
- Efficiency Metrics
- Model Compression
- Efficient Architectures

### Chapter 10: Model Optimizations
**Main Sections:**
- Quantization
- Pruning
- Knowledge Distillation
- Neural Architecture Search

### Chapter 11: AI Acceleration
**Main Sections:**
- Hardware Accelerators (GPUs, TPUs, FPGAs)
- Custom ASICs
- Hardware-Software Co-design

## Usage Guidelines for Agents

### For Reviewer Agent:
- Check if technical explanations appear before their designated chapter
- Historical mentions are fine, technical details are not
- Suggest footnotes over replacements when possible

### For Editor Agent:
- Never change "deep learning" to "hierarchical learning" (wrong!)
- Keep historical accuracy
- Add footnotes for forward references

### For Footnote Agent:
- Focus on historical context (highest value)
- Bridge CS knowledge to ML concepts
- Be selective - quality over quantity