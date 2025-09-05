# ML Systems Book - Progressive Knowledge Map

## Chapter-by-Chapter Concept Introduction

### Chapter 1: Introduction
**New Concepts Introduced:**
- Machine Learning Systems Engineering
- Production deployment (general concept)
- Model development lifecycle
- Infrastructure orchestration
- API design
- Containerization (basic mention)
- Data pipelines (basic concept)
- Model versioning
- A/B testing (basic mention)
- Model drift (basic concept, not detailed)
- Sub-100ms latency (as example)

**Can Use:** Basic CS/systems concepts
**Cannot Use:** Any ML-specific algorithms, neural networks, etc.

---

### Chapter 2: ML Systems
**New Concepts Introduced:**
- Deployment spectrum/tiers
- Cloud ML
- Edge ML  
- Mobile ML
- TinyML
- Resource constraints (memory, compute, power)
- Latency vs throughput trade-offs
- Privacy considerations in deployment
- Batch vs real-time processing

**Can Use:** Everything from Ch 1
**Cannot Use:** Neural networks, specific architectures, optimization techniques

---

### Chapter 3: DL Primer
**New Concepts Introduced:**
- Neural networks
- Neurons, weights, biases, connections
- Activation functions
- Forward propagation
- Backward propagation (backprop)
- Gradient descent
- Loss functions
- Training vs inference
- Overfitting/underfitting
- Biological inspiration for AI

**Can Use:** Everything from Ch 1-2
**Cannot Use:** Specific architectures (CNNs, RNNs), attention, transformers

---

### Chapter 4: DNN Architectures
**New Concepts Introduced:**
- Multi-Layer Perceptrons (MLPs)
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Transformers
- Attention mechanisms
- Spatial patterns
- Temporal patterns
- Dense connectivity
- Convolutional filters
- Pooling layers

**Can Use:** Everything from Ch 1-3
**Cannot Use:** Specific training techniques, hardware accelerators, quantization

---

### Chapter 5: AI Workflow
**New Concepts Introduced:**
- Data collection and curation
- Data labeling
- Feature engineering
- Model selection
- Hyperparameter tuning
- Cross-validation
- Train/validation/test splits

**Can Use:** Everything from Ch 1-4
**Cannot Use:** Advanced optimization, hardware-specific optimizations

---

### Chapter 6: Data Engineering
**New Concepts Introduced:**
- Data lakes vs data warehouses
- ETL/ELT pipelines
- Feature stores
- Data versioning
- Schema management
- Data quality monitoring

**Can Use:** Everything from Ch 1-5
**Cannot Use:** Distributed training, model compression

---

### Chapter 7: AI Frameworks
**New Concepts Introduced:**
- TensorFlow
- PyTorch  
- JAX
- Framework APIs
- Computational graphs
- Automatic differentiation

**Can Use:** Everything from Ch 1-6
**Cannot Use:** Hardware acceleration specifics, quantization

---

### Chapter 8: AI Training
**New Concepts Introduced:**
- Distributed training
- Data parallelism
- Model parallelism
- Pipeline parallelism
- Gradient accumulation
- Mixed precision training

**Can Use:** Everything from Ch 1-7
**Cannot Use:** Deployment optimizations, quantization

---

### Chapter 9: Efficient AI
**New Concepts Introduced:**
- Model efficiency metrics
- FLOPs and MACs
- Memory bandwidth
- Roofline model
- Latency vs throughput optimization

**Can Use:** Everything from Ch 1-8
**Cannot Use:** Specific compression techniques

---

### Chapter 10: Model Optimizations
**New Concepts Introduced:**
- Quantization (INT8, INT4, etc.)
- Pruning (structured, unstructured)
- Knowledge distillation
- Model compression
- Neural Architecture Search (NAS)
- Low-rank factorization

**Can Use:** Everything from Ch 1-9
**Cannot Use:** Hardware accelerators specifics

---

### Chapter 11: AI Acceleration
**New Concepts Introduced:**
- GPUs for ML
- TPUs
- NPUs/Neural Processing Units
- FPGAs for ML
- ASIC accelerators
- Hardware-software co-design

**Can Use:** Everything from Ch 1-10

---

## Critical Rules for Progressive Review

1. **NEVER** use a term before it's introduced
2. **NEVER** reference a concept from a future chapter
3. **ALWAYS** check this map before making improvements
4. **ALWAYS** use simpler language if the precise term hasn't been introduced
5. **WHEN IN DOUBT** use general descriptions rather than specific terms

## Examples of Progressive Language

### Before Chapter 10:
❌ "This can be optimized through quantization"
✅ "This can be optimized through techniques we'll explore later"

### Before Chapter 11:
❌ "GPUs and TPUs can accelerate this"
✅ "Specialized hardware can accelerate this"

### Before Chapter 4:
❌ "CNNs are better for image processing"
✅ "Specialized network structures handle images better"

### Before Chapter 3:
❌ "Neural networks can learn these patterns"
✅ "Machine learning systems can identify these patterns"