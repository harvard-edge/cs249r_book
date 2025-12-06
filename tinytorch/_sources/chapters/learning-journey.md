# The Learning Journey: From Atoms to Intelligence

**Understand the pedagogical narrative connecting modules 01-20 into a complete learning story from atomic components to production AI systems.**

---

## What This Page Is About

This page tells the **pedagogical story** behind TinyTorch's module progression. While other pages explain:
- **WHAT you'll build** ([Three-Tier Structure](00-introduction.md)) - organized module breakdown
- **WHEN in history** ([Milestones](milestones.md)) - recreating ML breakthroughs
- **WHERE you are** ([Student Workflow](../student-workflow.md)) - development workflow and progress

This page explains **WHY modules flow this way** - the learning narrative that transforms 20 individual modules into a coherent journey from mathematical foundations to production AI systems.

### How to Use This Narrative

- **Starting TinyTorch?** Read this to understand the complete arc before diving into modules
- **Mid-journey?** Return here when wondering "Why am I building DataLoader now?"
- **Planning your path?** Use this to understand how modules build on each other pedagogically
- **Teaching TinyTorch?** Share this narrative to help students see the big picture

---

## The Six-Act Learning Story

TinyTorch's 20 modules follow a carefully crafted six-act narrative arc. Each act represents a fundamental shift in what you're learning and what you can build.

```{mermaid}
graph LR
    Act1["Act I: Foundation<br/>01-04<br/>Atomic Components"] --> Act2["Act II: Learning<br/>05-07<br/>Gradient Revolution"]
    Act2 --> Act3["Act III: Data & Scale<br/>08-09<br/>Real Complexity"]
    Act3 --> Act4["Act IV: Language<br/>10-13<br/>Sequential Data"]
    Act4 --> Act5["Act V: Production<br/>14-19<br/>Optimization"]
    Act5 --> Act6["Act VI: Integration<br/>20<br/>Complete Systems"]

    style Act1 fill:#e3f2fd
    style Act2 fill:#fff8e1
    style Act3 fill:#e8f5e9
    style Act4 fill:#f3e5f5
    style Act5 fill:#fce4ec
    style Act6 fill:#fff3e0
```

---

### Act I: Foundation (Modules 01-04) - Building the Atomic Components

**The Beginning**: You start with nothing but Python and NumPy. Before you can build intelligence, you need the atoms.

<div style="background: #f8f9fa; border-left: 4px solid #007bff; padding: 1.5rem; margin: 2rem 0;">

**What You Learn**: Mathematical infrastructure that powers all neural networks - data structures, nonlinearity, composable transformations, and error measurement.

**What You Build**: The fundamental building blocks that everything else depends on.

</div>

#### Module 01: Tensor - The Universal Data Structure
You begin by building the Tensor class - the fundamental container for all ML data. Tensors are to ML what integers are to programming: the foundation everything else is built on. You implement arithmetic, matrix operations, reshaping, slicing, and broadcasting. Every component you build afterward will use Tensors.

**Systems Insight**: Understanding tensor memory layout, contiguous storage, and view semantics prepares you for optimization in Act V.

#### Module 02: Activations - Adding Intelligence
With Tensors ready, you add nonlinearity. You implement ReLU, Sigmoid, Tanh, and Softmax - the functions that give neural networks their power to approximate any function. Without activations, networks are just linear algebra. With them, they can learn complex patterns.

**Systems Insight**: Each activation has different computational and numerical stability properties - knowledge critical for debugging training later.

#### Module 03: Layers - Composable Building Blocks
Now you construct layers - reusable components that transform inputs to outputs. Linear layers perform matrix multiplication, LayerNorm stabilizes training, Dropout prevents overfitting. Each layer encapsulates transformation logic with a clean forward() interface.

**Systems Insight**: The layer abstraction teaches composability and modularity - how complex systems emerge from simple, well-designed components.

#### Module 04: Losses - Measuring Success
How do you know if your model is learning? Loss functions measure the gap between predictions and truth. MSELoss for regression, CrossEntropyLoss for classification, ContrastiveLoss for embeddings. Losses convert abstract predictions into concrete numbers you can minimize.

**Systems Insight**: Loss functions shape the optimization landscape - understanding their properties explains why some problems train easily while others struggle.

**🎯 Act I Achievement**: You've built the atomic components. But they're static - they can compute forward passes but cannot learn. You're ready for the revolution...

**Connection to Act II**: Static components are useful, but the real power comes when they can LEARN from data. That requires gradients.

---

### Act II: Learning (Modules 05-07) - The Gradient Revolution

**The Breakthrough**: Your static components awaken. Automatic differentiation transforms computation into learning.

<div style="background: #fff8e1; border-left: 4px solid #ffa726; padding: 1.5rem; margin: 2rem 0;">

**What You Learn**: The mathematics and systems engineering that enable learning - computational graphs, reverse-mode differentiation, gradient-based optimization, and training loops.

**What You Build**: A complete training system that can optimize any neural network architecture.

</div>

#### Module 05: Autograd - The Gradient Engine
This is the magic. You enhance Tensors with automatic differentiation - the ability to compute gradients automatically by building a computation graph. You implement backward() and the Function class. Now your Tensors remember their history and can propagate gradients through any computation.

**Systems Insight**: Understanding computational graphs explains memory growth during training and why checkpointing saves memory - critical for scaling to large models.

**Pedagogical Note**: This is the moment everything clicks. Students realize that `.backward()` isn't magic - it's a carefully designed system they can understand and modify.

#### Module 06: Optimizers - Following the Gradient Downhill
Gradients tell you which direction to move, but how far? You implement optimization algorithms: SGD takes simple steps, SGDMomentum adds velocity, RMSprop adapts step sizes, Adam combines both. Each optimizer is a strategy for navigating the loss landscape.

**Systems Insight**: Optimizers have different memory footprints (Adam needs 3× parameter memory) and convergence properties - trade-offs that matter in production.

#### Module 07: Training - The Learning Loop
You assemble everything into the training loop - the heartbeat of machine learning. Trainer orchestrates forward passes, loss computation, backward passes, and optimizer steps. You add learning rate schedules, checkpointing, and validation. This is where learning actually happens.

**Systems Insight**: The training loop reveals how all components interact - a systems view that's invisible when just calling model.fit().

**🎯 Act II Achievement**: You can now train neural networks to learn from data! MLPs achieve 95%+ accuracy on MNIST using 100% your own implementations.

**Connection to Act III**: Your learning system works beautifully on clean datasets that fit in memory. But real ML means messy data at scale.

---

### Act III: Data & Scale (Modules 08-09) - Handling Real-World Complexity

**The Challenge**: Laboratory ML meets production reality. Real data is large, messy, and requires specialized processing.

<div style="background: #e8f5e9; border-left: 4px solid #66bb6a; padding: 1.5rem; margin: 2rem 0;">

**What You Learn**: How to handle real-world data and spatial structure - the bridge from toy problems to production systems.

**What You Build**: Data pipelines and computer vision capabilities that work on real image datasets.

</div>

#### Module 08: DataLoader - Feeding the Training Loop
Real datasets don't fit in memory. DataLoader provides batching, shuffling, and efficient iteration over large datasets. It separates data handling from model logic, enabling training on datasets larger than RAM through streaming and mini-batch processing.

**Systems Insight**: Understanding batch processing, memory hierarchies, and I/O bottlenecks - the data pipeline is often the real bottleneck in production systems.

#### Module 09: Spatial - Seeing the World in Images
Neural networks need specialized operations for spatial data. Conv2D applies learnable filters, MaxPool2D reduces dimensions while preserving features, Flatten converts spatial features to vectors. These are the building blocks of computer vision.

**Systems Insight**: Convolutions exploit weight sharing and local connectivity - architectural choices that reduce parameters 100× compared to fully connected layers while improving performance.

**🎯 Act III Achievement**: CNNs achieve 75%+ accuracy on CIFAR-10 natural images - real computer vision with YOUR spatial operations!

**Connection to Act IV**: You've mastered vision. But the most exciting ML breakthroughs are happening in language. Time to understand sequential data.

---

### Act IV: Language (Modules 10-13) - Understanding Sequential Data

**The Modern Era**: From pixels to words. You implement the architectures powering the LLM revolution.

<div style="background: #f3e5f5; border-left: 4px solid #ab47bc; padding: 1.5rem; margin: 2rem 0;">

**What You Learn**: How to process language and implement the attention mechanisms that revolutionized AI - the path to GPT, BERT, and modern LLMs.

**What You Build**: Complete transformer architecture capable of understanding and generating language.

</div>

#### Module 10: Tokenization - Text to Numbers
Language models need numbers, not words. You implement character-level and BPE tokenization - converting text into sequences of integers. This is the bridge from human language to neural network inputs.

**Systems Insight**: Tokenization choices (vocabulary size, subword splitting) directly impact model size and training efficiency - crucial decisions for production systems.

#### Module 11: Embeddings - Learning Semantic Representations
Token IDs are just indices - they carry no meaning. Embeddings transform discrete tokens into continuous vectors where similar words cluster together. You add positional embeddings so models know word order.

**Systems Insight**: Embeddings are often the largest single component in language models - understanding their memory footprint matters for deployment.

#### Module 12: Attention - Dynamic Context Weighting
Not all words matter equally. Attention mechanisms let models focus on relevant parts of the input. You implement scaled dot-product attention and multi-head attention - the core innovation that powers modern language models.

**Systems Insight**: Attention scales O(n²) with sequence length - understanding this limitation explains why context windows are limited and why KV-caching matters (Act V).

**Pedagogical Note**: This is often the "aha!" moment for students - seeing attention as a differentiable dictionary lookup demystifies transformers.

#### Module 13: Transformers - The Complete Architecture
You assemble attention, embeddings, and feed-forward layers into the Transformer architecture. TransformerBlock stacks self-attention with normalization and residual connections. This is the architecture that revolutionized NLP and enabled GPT, BERT, and modern AI.

**Systems Insight**: Transformers are highly parallelizable (unlike RNNs) but memory-intensive - architectural trade-offs that shaped the modern ML landscape.

**🎯 Act IV Achievement**: Your transformer generates coherent text! You've implemented the architecture powering ChatGPT, GPT-4, and the modern AI revolution.

**Connection to Act V**: Your transformer works, but it's slow and memory-hungry. Time to optimize for production.

---

### Act V: Production (Modules 14-19) - Optimization & Deployment

**The Engineering Challenge**: Research models meet production constraints. You transform working prototypes into deployable systems.

<div style="background: #e0f7fa; border-left: 4px solid #26c6da; padding: 1.5rem; margin: 2rem 0;">

**What You Learn**: The systems engineering that makes ML production-ready - profiling, quantization, compression, caching, acceleration, and benchmarking.

**What You Build**: Optimized systems competitive with industry implementations, ready for real-world deployment.

</div>

#### Module 14: Profiling - Measuring Before Optimizing
You can't optimize what you don't measure. Profiler tracks memory usage, execution time, parameter counts, and FLOPs. You identify bottlenecks and validate that optimizations actually work.

**Systems Insight**: Premature optimization is the root of all evil. Profiling reveals that the bottleneck is rarely where you think it is.

#### Module 15: Quantization - Reduced Precision for Efficiency
Models use 32-bit floats by default, but 8-bit integers work almost as well. You implement INT8 quantization with calibration, reducing memory 4× and enabling 2-4× speedup on appropriate hardware.

**Systems Insight**: Quantization trades precision for efficiency - understanding this trade-off is essential for edge deployment (mobile, IoT) where memory and power are constrained.

#### Module 16: Compression - Removing Redundancy
Neural networks are over-parameterized. You implement magnitude pruning (removing small weights), structured pruning (removing neurons), low-rank decomposition (matrix factorization), and knowledge distillation (teacher-student training).

**Systems Insight**: Different compression techniques offer different trade-offs. Structured pruning enables real speedup (unstructured doesn't without sparse kernels).

#### Module 17: Memoization - Avoiding Redundant Computation
Why recompute what you've already calculated? You implement memoization with cache invalidation - dramatically speeding up recurrent patterns like autoregressive text generation.

**Systems Insight**: KV-caching in transformers reduces generation from O(n²) to O(n) - the optimization that makes real-time LLM interaction possible.

#### Module 18: Acceleration - Vectorization & Parallel Execution
Modern CPUs have SIMD instructions operating on multiple values simultaneously. You implement vectorized operations using NumPy's optimized routines and explore parallel execution patterns.

**Systems Insight**: Understanding hardware capabilities (SIMD width, cache hierarchy, instruction pipelining) enables 10-100× speedups through better code.

#### Module 19: Benchmarking - Rigorous Performance Measurement
You build comprehensive benchmarking tools with precise timing, statistical analysis, and comparison frameworks. Benchmarks let you compare implementations objectively and measure real-world impact.

**Systems Insight**: Benchmarking is a science - proper methodology (warmup, statistical significance, controlling variables) matters as much as the measurements themselves.

**🎯 Act V Achievement**: Production-ready systems competitive in Torch Olympics benchmarks! Models achieve <100ms inference latency with 4× memory reduction.

**Connection to Act VI**: You have all the pieces - foundation, learning, data, language, optimization. Time to assemble them into a complete AI system.

---

### Act VI: Integration (Module 20) - Building Real AI Systems

**The Culmination**: Everything comes together. You build TinyGPT - a complete language model from scratch.

<div style="background: #fce4ec; border-left: 4px solid #ec407a; padding: 1.5rem; margin: 2rem 0;">

**What You Learn**: Systems integration and end-to-end thinking - how all components work together to create functional AI.

**What You Build**: A complete transformer-based language model with training, optimization, and text generation.

</div>

#### Module 20: Capstone - TinyGPT End-to-End
Using all 19 previous modules, you build TinyGPT - a complete language model with:
- Text tokenization and embedding (Act IV)
- Multi-layer transformer architecture (Act IV)
- Training loop with optimization (Act II)
- Quantization and pruning for efficiency (Act V)
- Comprehensive benchmarking (Act V)
- Text generation with sampling (Act IV + V)

**Systems Insight**: Integration reveals emergent complexity. Individual components are simple, but their interactions create surprising behaviors - the essence of systems engineering.

**Pedagogical Note**: The capstone isn't about learning new techniques - it's about synthesis. Students discover that they've built something real, not just completed exercises.

**🎯 Act VI Achievement**: You've built a complete AI framework and deployed a real language model - entirely from scratch, from tensors to text generation!

---

## How This Journey Connects to Everything Else

### Journey (6 Acts) vs. Tiers (3 Levels)

**Acts** and **Tiers** are complementary views of the same curriculum:

| Perspective | Purpose | Granularity | Used For |
|-------------|---------|-------------|----------|
| **Tiers** (3) | Structural organization | Coarse-grained | Navigation, TOCs, planning |
| **Acts** (6) | Pedagogical narrative | Fine-grained | Understanding progression, storytelling |

**Mapping Acts to Tiers**:

```
🏗️ FOUNDATION TIER (Modules 01-07)
  ├─ Act I: Foundation (01-04) - Atomic components
  └─ Act II: Learning (05-07) - Gradient revolution

🏛️ ARCHITECTURE TIER (Modules 08-13)
  ├─ Act III: Data & Scale (08-09) - Real-world complexity
  └─ Act IV: Language (10-13) - Sequential understanding

⚡ OPTIMIZATION TIER (Modules 14-20)
  ├─ Act V: Production (14-19) - Deployment optimization
  └─ Act VI: Integration (20) - Complete systems
```

**When to use Tiers**: Navigating the website, planning your study schedule, understanding time commitment.

**When to use Acts**: Understanding why you're learning something now, seeing how modules connect, maintaining motivation through the narrative arc.

---

### Journey vs. Milestones: Two Dimensions of Progress

As you progress through TinyTorch, you advance along **two dimensions simultaneously**:

**Pedagogical Dimension (Acts)**: What you're LEARNING
- **Act I (01-04)**: Building atomic components - mathematical foundations
- **Act II (05-07)**: The gradient revolution - systems that learn
- **Act III (08-09)**: Real-world complexity - data and scale
- **Act IV (10-13)**: Sequential intelligence - language understanding
- **Act V (14-19)**: Production systems - optimization and deployment
- **Act VI (20)**: Complete integration - unified AI systems

**Historical Dimension (Milestones)**: What you CAN BUILD
- **1957: Perceptron** - Binary classification (after Act I)
- **1969: XOR** - Non-linear learning (after Act II)
- **1986: MLP** - Multi-class vision achieving 95%+ on MNIST (after Act II)
- **1998: CNN** - Spatial intelligence achieving 75%+ on CIFAR-10 (after Act III)
- **2017: Transformers** - Language generation (after Act IV)
- **2024: Systems** - Production optimization (after Act V)

**How They Connect**:

| Learning Act | Unlocked Milestone | Proof of Mastery |
|--------------|-------------------|------------------|
| **Act I: Foundation** | 🧠 1957 Perceptron | Your Linear layer recreates history |
| **Act II: Learning** | ⚡ 1969 XOR + 🔢 1986 MLP | Your autograd enables training (95%+ MNIST) |
| **Act III: Data & Scale** | 🖼️ 1998 CNN | Your Conv2d achieves 75%+ on CIFAR-10 |
| **Act IV: Language** | 🤖 2017 Transformers | Your attention generates coherent text |
| **Act V: Production** | ⚡ 2024 Systems Age | Your optimizations compete in benchmarks |
| **Act VI: Integration** | 🏆 TinyGPT Capstone | Your complete framework works end-to-end |

**Understanding Both Dimensions**: The **Acts** explain WHY you're building each component (pedagogical progression). The **Milestones** prove WHAT you've built actually works (historical validation).

**📖 See [Journey Through ML History](milestones.md)** for complete milestone details and how to run them.

---

### Journey vs. Capabilities: Tracking Your Skills

The learning journey also maps to **21 capability checkpoints** you can track:

**Foundation Capabilities (Act I-II)**:
- Checkpoint 01: Tensor manipulation ✓
- Checkpoint 02: Nonlinearity ✓
- Checkpoint 03: Network layers ✓
- Checkpoint 04: Loss measurement ✓
- Checkpoint 05: Gradient computation ✓
- Checkpoint 06: Parameter optimization ✓
- Checkpoint 07: Model training ✓

**Architecture Capabilities (Act III-IV)**:
- Checkpoint 08: Image processing ✓
- Checkpoint 09: Data loading ✓
- Checkpoint 10: Text processing ✓
- Checkpoint 11: Embeddings ✓
- Checkpoint 12: Attention mechanisms ✓
- Checkpoint 13: Transformers ✓

**Production Capabilities (Act V-VI)**:
- Checkpoint 14: Performance profiling ✓
- Checkpoint 15: Model quantization ✓
- Checkpoint 16: Network compression ✓
- Checkpoint 17: Computation caching ✓
- Checkpoint 18: Algorithm acceleration ✓
- Checkpoint 19: Competitive benchmarking ✓
- Checkpoint 20: Complete systems ✓

See [Student Workflow](../student-workflow.md) for the development workflow and progress tracking.

---

## Visualizing Your Complete Journey

Here's how the three views work together:

```
    PEDAGOGICAL NARRATIVE (6 Acts)
    ↓
Act I → Act II → Act III → Act IV → Act V → Act VI
01-04   05-07    08-09     10-13    14-19    20
  │       │        │         │        │       │
  └───────┴────────┴─────────┴────────┴───────┘
          │                  │                │
    STRUCTURE (3 Tiers)      │                │
    Foundation Tier ─────────┘                │
    Architecture Tier ────────────────────────┘
    Optimization Tier ────────────────────────┘
          │
    VALIDATION (Historical Milestones)
    │
    ├─ 1957 Perceptron (after Act I)
    ├─ 1969 XOR + 1986 MLP (after Act II)
    ├─ 1998 CNN 75%+ CIFAR-10 (after Act III)
    ├─ 2017 Transformers (after Act IV)
    ├─ 2024 Systems Age (after Act V)
    └─ TinyGPT Capstone (after Act VI)
```

**Use all three views**:
- **Tiers** help you navigate and plan
- **Acts** help you understand and stay motivated
- **Milestones** help you validate and celebrate

---

## Using This Journey: Student Guidance

### When Starting TinyTorch

**Read this page FIRST** (you're doing it right!) to understand:
- Where you're going (Act VI: complete AI systems)
- Why modules are ordered this way (pedagogical progression)
- How modules build on each other (each act enables the next)

### During Your Learning Journey

**Return to this page when**:
- Wondering "Why am I building DataLoader now?" (Act III: Real data at scale)
- Feeling lost in the details (zoom out to see which act you're in)
- Planning your next study session (understand what's coming next)
- Celebrating a milestone (see how it connects to the learning arc)

### Module-by-Module Orientation

As you work through modules, ask yourself:
- **Which act am I in?** (Foundation, Learning, Data & Scale, Language, Production, or Integration)
- **What did I learn in the previous act?** (Act I: atomic components)
- **What am I learning in this act?** (Act II: how they learn)
- **What will I unlock next act?** (Act III: real-world data)

**This narrative provides the context that makes individual modules meaningful.**

### When Teaching TinyTorch

**Share this narrative** to help students:
- See the big picture before diving into details
- Understand why prerequisites matter (each act builds on previous)
- Stay motivated through challenging modules (see where it's going)
- Appreciate the pedagogical design (not arbitrary order)

---

## The Pedagogical Arc: Why This Progression Works

### Bottom-Up Learning: From Atoms to Systems

TinyTorch follows a **bottom-up progression** - you build foundational components before assembling them into systems:

```
Act I: Atoms (Tensor, Activations, Layers, Losses)
  ↓
Act II: Learning (Autograd, Optimizers, Training)
  ↓
Act III: Scale (DataLoader, Spatial)
  ↓
Act IV: Intelligence (Tokenization, Embeddings, Attention, Transformers)
  ↓
Act V: Production (Profiling, Quantization, Compression, Acceleration)
  ↓
Act VI: Systems (Complete integration)
```

**Why bottom-up?**
- You can't understand training loops without understanding gradients
- You can't understand gradients without understanding computational graphs
- You can't understand computational graphs without understanding tensor operations

**Each act requires mastery of previous acts** - no forward references, no circular dependencies.

### Progressive Complexity: Scaffolded Learning

The acts increase in complexity while maintaining momentum:

**Act I (4 modules)**: Simple mathematical operations - build confidence
**Act II (3 modules)**: Core learning algorithms - consolidate understanding
**Act III (2 modules)**: Real-world data handling - practical skills
**Act IV (4 modules)**: Modern architectures - exciting applications
**Act V (6 modules)**: Production optimization - diverse techniques
**Act VI (1 module)**: Integration - synthesis and mastery

**The pacing is intentional**: shorter acts when introducing hard concepts (autograd), longer acts when students are ready for complexity (production optimization).

### Systems Thinking: See the Whole, Not Just Parts

Each act teaches **systems thinking** - how components interact to create emergent behavior:

- **Act I**: Components in isolation
- **Act II**: Components communicating (gradients flow backward)
- **Act III**: Components scaling (data pipelines)
- **Act IV**: Components specializing (attention routing)
- **Act V**: Components optimizing (trade-offs everywhere)
- **Act VI**: Complete system integration

**By Act VI, you think like a systems engineer** - not just "How do I implement this?" but "How does this affect memory? Compute? Training time? Accuracy?"

---

## FAQ: Understanding the Journey

### Why six acts instead of just three tiers?

**Tiers** are for organization. **Acts** are for learning.

Tiers group modules by theme (foundation, architecture, optimization). Acts explain pedagogical progression (why Module 08 comes after Module 07, not just that they're in the same tier).

Think of tiers as book chapters, acts as narrative arcs.

### Can I skip acts or jump around?

**No** - each act builds on previous acts with hard dependencies:

- Can't do Act II (Autograd) without Act I (Tensors)
- Can't do Act IV (Transformers) without Act II (Training) and Act III (DataLoader)
- Can't do Act V (Quantization) without Act IV (models to optimize)

**The progression is carefully designed** to avoid forward references and circular dependencies.

### Which act is the hardest?

**Act II (Autograd)** is conceptually hardest - automatic differentiation requires understanding computational graphs and reverse-mode differentiation.

**Act V (Production)** is breadth-wise hardest - six diverse optimization techniques, each with different trade-offs.

**Act IV (Transformers)** is most exciting - seeing attention generate text is the "wow" moment for many students.

### How long does each act take?

Typical time estimates (varies by background):

- **Act I**: 8-12 hours (2 weeks @ 4-6 hrs/week)
- **Act II**: 6-9 hours (1.5 weeks @ 4-6 hrs/week)
- **Act III**: 6-8 hours (1 week @ 6-8 hrs/week)
- **Act IV**: 12-15 hours (2-3 weeks @ 4-6 hrs/week)
- **Act V**: 18-24 hours (3-4 weeks @ 6-8 hrs/week)
- **Act VI**: 8-10 hours (1.5 weeks @ 5-7 hrs/week)

**Total**: ~60-80 hours over 14-18 weeks

### When do I unlock milestones?

**After completing acts**:
- Act I → Perceptron (1957)
- Act II → XOR (1969) + MLP (1986)
- Act III → CNN (1998)
- Act IV → Transformers (2017)
- Act V → Systems (2024)
- Act VI → TinyGPT (complete)

**📖 See [Milestones](milestones.md)** for details.

---

## What's Next?

**Ready to begin your journey?**

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h3 style="margin: 0 0 1rem 0; color: #495057;">Start Your Learning Journey</h3>
<p style="margin: 0 0 1.5rem 0; color: #6c757d;">Begin with Act I: Foundation - build the atomic components</p>
<a href="../quickstart-guide.html" style="display: inline-block; background: #007bff; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500; margin-right: 1rem;">15-Minute Quick Start →</a>
<a href="00-introduction.html" style="display: inline-block; background: #28a745; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500;">View Course Structure →</a>
</div>

**Related Resources**:
- **[Three-Tier Structure](00-introduction.md)** - Organized module breakdown with time estimates
- **[Journey Through ML History](milestones.md)** - Historical milestones you'll recreate
- **[Student Workflow](../student-workflow.md)** - Development workflow and progress tracking
- **[Quick Start Guide](../quickstart-guide.md)** - Hands-on setup and first module

---

**Remember**: You're not just learning ML algorithms. You're building ML systems - from mathematical foundations to production deployment. This journey transforms you from a framework user into a systems engineer who truly understands how modern AI works.

**Welcome to the learning journey. Let's build something amazing together.** 🚀
