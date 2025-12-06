# 🏅 Torch Olympics (Module 20)

**The ultimate test: Build a complete, competition-ready ML system.**

---

## What Is the Torch Olympics?

The Torch Olympics is TinyTorch's **capstone experience**—a comprehensive challenge where you integrate everything you've learned across 19 modules to build, optimize, and compete with a complete ML system.

This isn't a traditional homework assignment. It's a **systems engineering competition** where you'll:

- Design and implement a complete neural architecture
- Train it on real datasets with YOUR framework
- Optimize for production deployment
- Benchmark against other students
- Submit to the TinyTorch Leaderboard

**Think of it as**: MLPerf meets academic research meets systems engineering—all using the framework YOU built.

---

## What You'll Build

```{mermaid}
graph TB
    FOUNDATION[🏗 Foundation<br/>Tensor, Autograd, Training]
    ARCHITECTURE[🏛️ Architecture<br/>CNNs, Transformers]
    OPTIMIZATION[⏱️ Optimization<br/>Quantization, Acceleration]

    FOUNDATION --> SYSTEM[🏅 Production System]
    ARCHITECTURE --> SYSTEM
    OPTIMIZATION --> SYSTEM

    SYSTEM --> CHALLENGES[Competition Challenges]

    CHALLENGES --> C1[Vision: CIFAR-10<br/>Goal: 80%+ accuracy]
    CHALLENGES --> C2[Language: TinyTalks<br/>Goal: Coherent generation]
    CHALLENGES --> C3[Optimization: Speed<br/>Goal: 100 tokens/sec]
    CHALLENGES --> C4[Compression: Size<br/>Goal: <10MB model]

    C1 --> LEADERBOARD[🏆 TinyTorch Leaderboard]
    C2 --> LEADERBOARD
    C3 --> LEADERBOARD
    C4 --> LEADERBOARD

    style FOUNDATION fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style ARCHITECTURE fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style OPTIMIZATION fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style SYSTEM fill:#fef3c7,stroke:#f59e0b,stroke-width:4px
    style LEADERBOARD fill:#c8e6c9,stroke:#388e3c,stroke-width:4px
```

---

## Competition Tracks

### Track 1: Computer Vision Excellence

**Challenge**: Achieve the highest accuracy on CIFAR-10 (color images) using YOUR Conv2d implementation.

**Constraints**:
- Must use YOUR TinyTorch implementation (no PyTorch/TensorFlow)
- Training time: <2 hours on standard hardware
- Model size: <50MB

**Skills tested**:
- CNN architecture design
- Data augmentation strategies
- Hyperparameter tuning
- Training loop optimization

**Current record**: 82% accuracy (can you beat it?)

---

### Track 2: Language Generation Quality

**Challenge**: Build the best text generation system using YOUR transformer implementation.

**Evaluation**:
- Coherence: Do responses make sense?
- Relevance: Does the model stay on topic?
- Fluency: Is the language natural?
- Perplexity: Lower is better

**Constraints**:
- Must use YOUR attention + transformer code
- Trained on TinyTalks dataset
- Context length: 512 tokens

**Skills tested**:
- Transformer architecture design
- Tokenization strategy
- Training stability
- Generation sampling techniques

---

### Track 3: Inference Speed Championship

**Challenge**: Achieve the highest throughput (tokens/second) for transformer inference.

**Optimization techniques**:
- KV-cache implementation quality
- Batching efficiency
- Operation fusion
- Memory management

**Constraints**:
- Must maintain >95% of baseline accuracy
- Measured on standard hardware (CPU or GPU)
- Single-thread or multi-thread allowed

**Current record**: 250 tokens/sec (can you go faster?)

**Skills tested**:
- Profiling and bottleneck identification
- Cache management
- Systems-level optimization
- Performance benchmarking

---

### Track 4: Model Compression Masters

**Challenge**: Build the smallest model that maintains competitive accuracy.

**Optimization techniques**:
- Quantization (INT8, INT4)
- Structured pruning
- Knowledge distillation
- Architecture search

**Constraints**:
- Accuracy drop: <3% from baseline
- Target: <10MB model size
- Must run on CPU (no GPU required)

**Current record**: 8.2MB model with 92% CIFAR-10 accuracy

**Skills tested**:
- Quantization strategy
- Pruning methodology
- Accuracy-efficiency trade-offs
- Edge deployment considerations

---

## How It Works

### 1. Choose Your Challenge

Pick one or more competition tracks based on your interests:
- Vision (CNNs)
- Language (Transformers)
- Speed (Inference optimization)
- Size (Model compression)

### 2. Design Your System

Use all 19 modules you've completed:

```python
from tinytorch import Tensor, Linear, Conv2d, Attention  # YOUR code
from tinytorch import Adam, CrossEntropyLoss             # YOUR optimizers
from tinytorch import DataLoader, train_loop             # YOUR infrastructure

# Design your architecture
model = YourCustomArchitecture()  # Your design choices matter!

# Train with YOUR framework
optimizer = Adam(model.parameters(), lr=0.001)
train_loop(model, train_loader, optimizer, epochs=50)

# Optimize for production
quantized_model = quantize(model)  # YOUR quantization
pruned_model = prune(quantized_model, sparsity=0.5)  # YOUR pruning
```

### 3. Benchmark Rigorously

Use Module 19's benchmarking tools:

```bash
# Accuracy
tito benchmark accuracy --model your_model.pt --dataset cifar10

# Speed (tokens/sec)
tito benchmark speed --model your_transformer.pt --input-length 512

# Size (MB)
tito benchmark size --model your_model.pt

# Memory (peak usage)
tito benchmark memory --model your_model.pt
```

### 4. Submit to Leaderboard

```bash
# Package your submission
tito olympics submit \
  --track vision \
  --model your_model.pt \
  --code your_training.py \
  --report your_analysis.md

# View leaderboard
tito olympics leaderboard --track vision
```

---

## Leaderboard Dimensions

Your submission is evaluated across **multiple dimensions**:

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| **Accuracy** | 40% | Primary task performance |
| **Speed** | 20% | Inference throughput (tokens/sec or images/sec) |
| **Size** | 20% | Model size in MB |
| **Code Quality** | 10% | Implementation clarity and documentation |
| **Innovation** | 10% | Novel techniques or insights |

**Final score**: Weighted combination of all dimensions. This mirrors real-world ML where you optimize for multiple objectives simultaneously.

---

## Learning Objectives

The Torch Olympics integrates everything you've learned:

### Systems Engineering Skills
- **Architecture design**: Making trade-offs between depth, width, and complexity
- **Hyperparameter tuning**: Systematic search vs intuition
- **Performance optimization**: Profiling → optimization → validation loop
- **Benchmarking**: Rigorous measurement and comparison

### Production Readiness
- **Deployment constraints**: Size, speed, memory limits
- **Quality assurance**: Testing, validation, error analysis
- **Documentation**: Explaining your design choices
- **Reproducibility**: Others can run your code

### Research Skills
- **Experimentation**: Hypothesis → experiment → analysis
- **Literature review**: Understanding SOTA techniques
- **Innovation**: Trying new ideas and combinations
- **Communication**: Writing clear technical reports

---

## Grading (For Classroom Use)

Instructors can use the Torch Olympics as a capstone project:

**Deliverables**:
1. **Working Implementation** (40%): Model trains and achieves target metrics
2. **Technical Report** (30%): Design choices, experiments, analysis
3. **Code Quality** (20%): Clean, documented, reproducible
4. **Leaderboard Performance** (10%): Relative ranking

**Example rubric**:
- 90-100%: Top 10% of leaderboard + excellent report
- 80-89%: Top 25% + good report
- 70-79%: Baseline metrics met + complete report
- 60-69%: Partial completion
- <60%: Incomplete submission

---

## Timeline

**Recommended schedule** (8-week capstone):

- **Weeks 1-2**: Challenge selection and initial implementation
- **Weeks 3-4**: Training and baseline experiments
- **Weeks 5-6**: Optimization and experimentation
- **Week 7**: Benchmarking and final tuning
- **Week 8**: Report writing and submission

**Intensive schedule** (2-week sprint):
- Days 1-3: Baseline implementation
- Days 4-7: Optimization sprint
- Days 8-10: Benchmarking
- Days 11-14: Documentation and submission

---

## Support and Resources

### Reference Implementations

Starter code is provided for each track:

```bash
# Vision track starter
tito olympics init --track vision --output ./my_vision_project

# Language track starter
tito olympics init --track language --output ./my_language_project
```

### Community

- **Discord**: Get help from other students and instructors
- **Office Hours**: Weekly video calls for Q&A
- **Leaderboard**: See what others are achieving
- **Forums**: Share insights and techniques

### Documentation

- **[MLPerf Milestone](../chapters/milestones)**: Historical context
- **[Benchmarking Guide](../modules/19_benchmarking_ABOUT)**: Measurement methodology
- **[Optimization Techniques](../tiers/optimization)**: Compression and acceleration strategies

---

## Prerequisites

**Required**:
- ✅ **All 19 modules completed** (Foundation + Architecture + Optimization)
- ✅ Experience training models on real datasets
- ✅ Understanding of profiling and benchmarking
- ✅ Comfort with YOUR TinyTorch codebase

**Highly recommended**:
- Complete all 6 historical milestones (1957-2018)
- Review optimization tier (Modules 14-19)
- Practice with profiling tools

---

## Time Commitment

**Minimum**: 20-30 hours for single track completion

**Recommended**: 40-60 hours for multi-track competition + excellent report

**Intensive**: 80+ hours for top leaderboard performance + research-level analysis

This is a capstone project—expect it to be challenging and rewarding!

---

## What You'll Take Away

By completing the Torch Olympics, you'll have:

1. **Portfolio piece**: A complete ML system you built from scratch
2. **Systems thinking**: Deep understanding of ML engineering trade-offs
3. **Benchmarking skills**: Ability to measure and optimize systematically
4. **Production experience**: End-to-end ML system development
5. **Competition experience**: Leaderboard ranking and peer comparison

**This is what sets TinyTorch apart**: You didn't just learn to use ML frameworks—you built one, optimized it, and competed with it.

---

## Next Steps

**Ready to compete?**

```bash
# Initialize your Torch Olympics project
tito olympics init --track vision

# Review the rules
tito olympics rules

# View current leaderboard
tito olympics leaderboard
```

**Or review prerequisites:**

- **[🏗 Foundation Tier](foundation)** (Modules 01-07)
- **[🏛️ Architecture Tier](architecture)** (Modules 08-13)
- **[⏱️ Optimization Tier](optimization)** (Modules 14-19)

---

**[← Back to Home](../intro)**
