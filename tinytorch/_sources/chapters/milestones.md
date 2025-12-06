# Journey Through ML History

**Experience the evolution of AI by rebuilding history's most important breakthroughs with YOUR TinyTorch implementations.**

---

## What Are Milestones?

Milestones are **proof-of-mastery demonstrations** that showcase what you can build after completing specific modules. Each milestone recreates a historically significant ML achievement using YOUR implementations.

### Why This Approach?

- **Deep Understanding**: Experience the actual challenges researchers faced
- **Progressive Learning**: Each milestone builds on previous foundations
- **Real Achievements**: Not toy examples - these are historically significant breakthroughs
- **Systems Thinking**: Understand WHY each innovation mattered for ML systems

---

## Two Dimensions of Your Progress

As you build TinyTorch, you're progressing along **TWO dimensions simultaneously**:

### Pedagogical Dimension (Acts): What You're LEARNING

**Act I (01-04)**: Building atomic components - mathematical foundations
**Act II (05-07)**: The gradient revolution - systems that learn
**Act III (08-09)**: Real-world complexity - data and scale
**Act IV (10-13)**: Sequential intelligence - language understanding
**Act V (14-19)**: Production systems - optimization and deployment
**Act VI (20)**: Complete integration - unified AI systems

See [The Learning Journey](learning-journey.md) for the complete pedagogical narrative explaining WHY modules flow this way.

### Historical Dimension (Milestones): What You CAN Build

**1957: Perceptron** - Binary classification
**1969: XOR** - Non-linear learning
**1986: MLP** - Multi-class vision
**1998: CNN** - Spatial intelligence
**2017: Transformers** - Language generation
**2018: Torch Olympics** - Production optimization

### How They Connect

```{mermaid}
graph TB
    subgraph "Pedagogical Acts (What You're Learning)"
        A1["Act I: Foundation<br/>Modules 01-04<br/>Atomic Components"]
        A2["Act II: Learning<br/>Modules 05-07<br/>Gradient Revolution"]
        A3["Act III: Data & Scale<br/>Modules 08-09<br/>Real-World Complexity"]
        A4["Act IV: Language<br/>Modules 10-13<br/>Sequential Intelligence"]
        A5["Act V: Production<br/>Modules 14-19<br/>Optimization"]
        A6["Act VI: Integration<br/>Module 20<br/>Complete Systems"]
    end

    subgraph "Historical Milestones (What You Can Build)"
        M1["1957: Perceptron<br/>Binary Classification"]
        M2["1969: XOR Crisis<br/>Non-linear Learning"]
        M3["1986: MLP<br/>Multi-class Vision<br/>95%+ MNIST"]
        M4["1998: CNN<br/>Spatial Intelligence<br/>75%+ CIFAR-10"]
        M5["2017: Transformers<br/>Language Generation"]
        M6["2018: Torch Olympics<br/>Production Speed"]
    end

    A1 --> M1
    A2 --> M2
    A2 --> M3
    A3 --> M4
    A4 --> M5
    A5 --> M6

    style A1 fill:#e3f2fd
    style A2 fill:#fff8e1
    style A3 fill:#e8f5e9
    style A4 fill:#f3e5f5
    style A5 fill:#fce4ec
    style A6 fill:#fff3e0
    style M1 fill:#ffcdd2
    style M2 fill:#f8bbd0
    style M3 fill:#e1bee7
    style M4 fill:#d1c4e9
    style M5 fill:#c5cae9
    style M6 fill:#bbdefb
```

| Learning Act | Unlocked Milestone | Proof of Mastery |
|--------------|-------------------|------------------|
| **Act I: Foundation (01-04)** | 1957 Perceptron | Your Linear layer recreates history |
| **Act II: Learning (05-07)** | 1969 XOR + 1986 MLP | Your autograd enables training (95%+ MNIST) |
| **Act III: Data & Scale (08-09)** | 1998 CNN | Your Conv2d achieves 75%+ on CIFAR-10 |
| **Act IV: Language (10-13)** | 2017 Transformers | Your attention generates coherent text |
| **Act V: Production (14-18)** | 2018 Torch Olympics | Your optimizations achieve production speed |
| **Act VI: Integration (19-20)** | Benchmarking + Capstone | Your complete framework competes |

**Understanding Both Dimensions**: The **Acts** explain WHY you're building each component (pedagogical progression). The **Milestones** prove WHAT you've built works (historical validation). Together, they show you're not just completing exercises - you're building something real.

---

## The Timeline

```{mermaid}
timeline
    title Journey Through ML History
    1957 : Perceptron : Binary classification with gradient descent
    1969 : XOR Crisis : Hidden layers solve non-linear problems
    1986 : MLP Revival : Backpropagation enables deep learning
    1998 : CNN Era : Spatial intelligence for computer vision
    2017 : Transformers : Attention revolutionizes language AI
    2018 : Torch Olympics : Production benchmarking and optimization
```

### 01. Perceptron (1957) - Rosenblatt

**After Modules 02-04**

```
Input → Linear → Sigmoid → Output
```

**The Beginning**: The first trainable neural network. Frank Rosenblatt proved machines could learn from data.

**What You'll Build**:
- Binary classification with gradient descent
- Simple but revolutionary architecture
- YOUR Linear layer recreates history

**Systems Insights**:
- Memory: O(n) parameters
- Compute: O(n) operations
- Limitation: Only linearly separable problems

```bash
cd milestones/01_1957_perceptron
python 01_rosenblatt_forward.py   # See the problem (random weights)
python 02_rosenblatt_trained.py   # See the solution (trained)
```

**Expected Results**: ~50% (untrained) → 95%+ (trained) accuracy

---

### 02. XOR Crisis (1969) - Minsky & Papert

**After Modules 02-06**

```
Input → Linear → ReLU → Linear → Output
```

**The Challenge**: Minsky proved perceptrons couldn't solve XOR. This crisis nearly ended AI research.

**What You'll Build**:
- Hidden layers enable non-linear solutions
- Multi-layer networks break through limitations
- YOUR autograd makes it possible

**Systems Insights**:
- Memory: O(n²) with hidden layers
- Compute: O(n²) operations
- Breakthrough: Hidden representations

```bash
cd milestones/02_1969_xor
python 01_xor_crisis.py   # Watch it fail (loss stuck at 0.69)
python 02_xor_solved.py   # Hidden layers solve it!
```

**Expected Results**: 50% (single layer) → 100% (multi-layer) on XOR

---

### 03. MLP Revival (1986) - Backpropagation Era

**After Modules 02-08**

```
Images → Flatten → Linear → ReLU → Linear → ReLU → Linear → Classes
```

**The Revolution**: Backpropagation enabled training deep networks on real datasets like MNIST.

**What You'll Build**:
- Multi-class digit recognition
- Complete training pipelines
- YOUR optimizers achieve 95%+ accuracy

**Systems Insights**:
- Memory: ~100K parameters for MNIST
- Compute: Dense matrix operations
- Architecture: Multi-layer feature learning

```bash
cd milestones/03_1986_mlp
python 01_rumelhart_tinydigits.py  # 8x8 digits (quick)
python 02_rumelhart_mnist.py       # Full MNIST
```

**Expected Results**: 95%+ accuracy on MNIST

---

### 04. CNN Revolution (1998) - LeCun's Breakthrough

**After Modules 02-09** • **🎯 North Star Achievement**

```
Images → Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → Linear → Classes
```

**The Game-Changer**: CNNs exploit spatial structure for computer vision. This enabled modern AI.

**What You'll Build**:
- Convolutional feature extraction
- Natural image classification (CIFAR-10)
- YOUR Conv2d + MaxPool2d unlock spatial intelligence

**Systems Insights**:
- Memory: ~1M parameters (weight sharing reduces vs dense)
- Compute: Convolution is intensive but parallelizable
- Architecture: Local connectivity + translation invariance

```bash
cd milestones/04_1998_cnn
python 01_lecun_tinydigits.py  # Spatial features on digits
python 02_lecun_cifar10.py     # CIFAR-10 @ 75%+ accuracy
```

**Expected Results**: **75%+ accuracy on CIFAR-10** ✨

---

### 05. Transformer Era (2017) - Attention Revolution

**After Modules 02-13**

```
Tokens → Embeddings → Attention → FFN → ... → Attention → Output
```

**The Modern Era**: Transformers + attention launched the LLM revolution (GPT, BERT, ChatGPT).

**What You'll Build**:
- Self-attention mechanisms
- Autoregressive text generation
- YOUR attention implementation generates language

**Systems Insights**:
- Memory: O(n²) attention requires careful management
- Compute: Highly parallelizable
- Architecture: Long-range dependencies

```bash
cd milestones/05_2017_transformer
python 01_vaswani_generation.py  # Q&A generation with TinyTalks
python 02_vaswani_dialogue.py    # Multi-turn dialogue
```

**Expected Results**: Loss < 1.5, coherent responses to questions

---

### 06. Torch Olympics Era (2018) - The Optimization Revolution

**After Modules 14-18**

```
Profile → Compress → Accelerate
```

**The Turning Point**: As models grew larger, MLCommons' Torch Olympics (2018) established systematic optimization as a discipline - profiling, compression, and acceleration became essential for deployment.

**What You'll Build**:
- Performance profiling and bottleneck analysis
- Model compression (quantization + pruning)
- Inference acceleration (KV-cache + batching)

**Systems Insights**:
- Memory: 4-16× compression through quantization/pruning
- Speed: 12-40× faster generation with KV-cache + batching
- Workflow: Systematic "measure → optimize → validate" methodology

```bash
cd milestones/06_2018_mlperf
python 01_baseline_profile.py   # Find bottlenecks
python 02_compression.py         # Reduce size (quantize + prune)
python 03_generation_opts.py    # Speed up inference (cache + batch)
```

**Expected Results**: 8-16× smaller models, 12-40× faster inference

---

## Learning Philosophy

### Progressive Capability Building

| Stage | Era | Capability | Your Tools |
|-------|-----|-----------|-----------|
| **1957** | Foundation | Binary classification | Linear + Sigmoid |
| **1969** | Depth | Non-linear problems | Hidden layers + Autograd |
| **1986** | Scale | Multi-class vision | Optimizers + Training |
| **1998** | Structure | Spatial understanding | Conv2d + Pooling |
| **2017** | Attention | Sequence modeling | Transformers + Attention |
| **2018** | Optimization | Production deployment | Profiling + Compression + Acceleration |

### Systems Engineering Progression

Each milestone teaches critical systems thinking:

1. **Memory Management**: From O(n) → O(n²) → O(n²) with optimizations
2. **Computational Trade-offs**: Accuracy vs efficiency
3. **Architectural Patterns**: How structure enables capability
4. **Production Deployment**: What it takes to scale

---

## How to Use Milestones

### 1. Complete Prerequisites

```bash
# Check which modules you've completed
tito checkpoint status

# Complete required modules
tito module complete 02_tensor
tito module complete 03_activations
# ... and so on
```

### 2. Run the Milestone

```bash
cd milestones/01_1957_perceptron
python 02_rosenblatt_trained.py
```

### 3. Understand the Systems

Each milestone includes:
- 📊 **Memory profiling**: See actual memory usage
- ⚡ **Performance metrics**: FLOPs, parameters, timing
- 🧠 **Architectural analysis**: Why this design matters
- 📈 **Scaling insights**: How performance changes with size

### 4. Reflect and Compare

**Questions to ask:**
- How does this compare to modern architectures?
- What were the computational constraints in that era?
- How would you optimize this for production?
- What patterns appear in PyTorch/TensorFlow?

---

## Quick Reference

### Milestone Prerequisites

| Milestone | After Module | Key Requirements |
|-----------|-------------|-----------------|
| 01. Perceptron (1957) | 04 | Tensor, Activations, Layers |
| 02. XOR (1969) | 06 | + Losses, Autograd |
| 03. MLP (1986) | 08 | + Optimizers, Training |
| 04. CNN (1998) | 09 | + Spatial, DataLoader |
| 05. Transformer (2017) | 13 | + Tokenization, Embeddings, Attention |
| 06. Torch Olympics (2018) | 18 | + Profiling, Quantization, Compression, Memoization, Acceleration |

### What Each Milestone Proves

- **Your implementations work** - Not just toy code
- **Historical significance** - These breakthroughs shaped modern AI
- **Systems understanding** - You know memory, compute, scaling
- **Production relevance** - Patterns used in real ML frameworks

---

## Further Learning

After completing milestones, explore:

- **Torch Olympics Competition**: Optimize your implementations
- **Leaderboard**: Compare with other students
- **Capstone Projects**: Build your own ML applications
- **Research Papers**: Read the original papers for each milestone

---

## Why This Matters

**Most courses teach you to USE frameworks.**  
**TinyTorch teaches you to UNDERSTAND them.**

By rebuilding ML history, you gain:
- 🧠 Deep intuition for how neural networks work
- 🔧 Systems thinking for production ML
- 🏆 Portfolio projects demonstrating mastery
- 💼 Preparation for ML systems engineering roles

---

**Ready to start your journey through ML history?**

```bash
cd milestones/01_1957_perceptron
python 02_rosenblatt_trained.py
```

**Build the future by understanding the past.** 🚀

