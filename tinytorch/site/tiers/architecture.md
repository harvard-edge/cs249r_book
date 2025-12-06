# 🏛️ Architecture Tier (Modules 08-13)

**Build modern neural architectures—from computer vision to language models.**

---

## What You'll Learn

The Architecture tier teaches you how to build the neural network architectures that power modern AI. You'll implement CNNs for computer vision, transformers for language understanding, and the data loading infrastructure needed to train on real datasets.

**By the end of this tier, you'll understand:**
- How data loaders efficiently feed training data to models
- Why convolutional layers are essential for computer vision
- How attention mechanisms enable transformers to understand sequences
- What embeddings do to represent discrete tokens as continuous vectors
- How modern architectures compose these components into powerful systems

---

## Module Progression

```{mermaid}
graph TB
    F[🏗 Foundation<br/>Tensor, Autograd, Training]

    F --> M08[08. DataLoader<br/>Efficient data pipelines]
    F --> M09[09. Spatial<br/>Conv2d + Pooling]

    M08 --> M09
    M09 --> VISION[💡 Computer Vision<br/>CNNs unlock spatial intelligence]

    F --> M10[10. Tokenization<br/>Text → integers]
    M10 --> M11[11. Embeddings<br/>Integers → vectors]
    M11 --> M12[12. Attention<br/>Context-aware representations]
    M12 --> M13[13. Transformers<br/>Complete architecture]

    M13 --> LLM[💡 Language Models<br/>Transformers generate text]

    style F fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style M08 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px
    style M09 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px
    style M10 fill:#e1bee7,stroke:#6a1b9a,stroke-width:3px
    style M11 fill:#e1bee7,stroke:#6a1b9a,stroke-width:3px
    style M12 fill:#ce93d8,stroke:#4a148c,stroke-width:3px
    style M13 fill:#ba68c8,stroke:#4a148c,stroke-width:4px
    style VISION fill:#fef3c7,stroke:#f59e0b,stroke-width:3px
    style LLM fill:#fef3c7,stroke:#f59e0b,stroke-width:3px
```

---

## Module Details

### 08. DataLoader - Efficient Data Pipelines

**What it is**: Infrastructure for loading, batching, and shuffling training data efficiently.

**Why it matters**: Real ML systems train on datasets that don't fit in memory. DataLoaders handle batching, shuffling, and parallel data loading—essential for efficient training.

**What you'll build**: A DataLoader that supports batching, shuffling, and dataset iteration with proper memory management.

**Systems focus**: Memory efficiency, batching strategies, I/O optimization

---

### 09. Spatial - Convolutional Neural Networks

**What it is**: Conv2d (convolutional layers) and pooling operations for processing images.

**Why it matters**: CNNs revolutionized computer vision by exploiting spatial structure. Understanding convolutions, kernels, and pooling is essential for image processing and beyond.

**What you'll build**: Conv2d, MaxPool2d, and related operations with proper gradient computation.

**Systems focus**: Spatial operations, memory layout (channels), computational intensity

**Historical impact**: This module enables **Milestone 04 (1998 CNN Revolution)** - achieving 75%+ accuracy on CIFAR-10 with YOUR implementations.

---

### 10. Tokenization - From Text to Numbers

**What it is**: Converting text into integer sequences that neural networks can process.

**Why it matters**: Neural networks operate on numbers, not text. Tokenization is the bridge between human language and machine learning—understanding vocabulary, encoding, and decoding is fundamental.

**What you'll build**: Character-level and subword tokenizers with vocabulary management and encoding/decoding.

**Systems focus**: Vocabulary management, encoding schemes, out-of-vocabulary handling

---

### 11. Embeddings - Learning Representations

**What it is**: Learned mappings from discrete tokens (words, characters) to continuous vectors.

**Why it matters**: Embeddings transform sparse, discrete representations into dense, semantic vectors. Understanding embeddings is crucial for NLP, recommendation systems, and any domain with categorical data.

**What you'll build**: Embedding layers with proper initialization and gradient computation.

**Systems focus**: Lookup tables, gradient backpropagation through indices, initialization

---

### 12. Attention - Context-Aware Representations

**What it is**: Self-attention mechanisms that let each token attend to all other tokens in a sequence.

**Why it matters**: Attention is the breakthrough that enabled modern LLMs. It allows models to capture long-range dependencies and contextual relationships that RNNs struggled with.

**What you'll build**: Scaled dot-product attention, multi-head attention, and causal masking for autoregressive generation.

**Systems focus**: O(n²) memory/compute, masking strategies, numerical stability

---

### 13. Transformers - The Modern Architecture

**What it is**: Complete transformer architecture combining embeddings, attention, and feedforward layers.

**Why it matters**: Transformers power GPT, BERT, and virtually all modern LLMs. Understanding their architecture—positional encodings, layer normalization, residual connections—is essential for AI engineering.

**What you'll build**: A complete decoder-only transformer (GPT-style) for autoregressive text generation.

**Systems focus**: Layer composition, residual connections, generation loop

**Historical impact**: This module enables **Milestone 05 (2017 Transformer Era)** - generating coherent text with YOUR attention implementation.

---

## What You Can Build After This Tier

```{mermaid}
timeline
    title Historical Achievements Unlocked
    1998 : CNN Revolution : 75%+ accuracy on CIFAR-10 with spatial intelligence
    2017 : Transformer Era : Text generation with attention mechanisms
```

After completing the Architecture tier, you'll be able to:

- **Milestone 04 (1998)**: Build CNNs that achieve 75%+ accuracy on CIFAR-10 (color images)
- **Milestone 05 (2017)**: Implement transformers that generate coherent text responses
- Train on real datasets (MNIST, CIFAR-10, text corpora)
- Understand why modern architectures (ResNets, Vision Transformers, LLMs) work

---

## Prerequisites

**Required**:
- **🏗 Foundation Tier** (Modules 01-07) completed
- Understanding of tensors, autograd, and training loops
- Basic understanding of images (height, width, channels)
- Basic understanding of text/language concepts

**Helpful but not required**:
- Computer vision concepts (convolution, feature maps)
- NLP concepts (tokens, vocabulary, sequence modeling)

---

## Time Commitment

**Per module**: 4-6 hours (implementation + exercises + datasets)

**Total tier**: ~30-40 hours for complete mastery

**Recommended pace**: 1 module per week (2 modules/week for intensive study)

---

## Learning Approach

Each module follows the **Build → Use → Reflect** cycle with **real datasets**:

1. **Build**: Implement the architecture component (Conv2d, attention, transformers)
2. **Use**: Train on real data (CIFAR-10 images, text corpora)
3. **Reflect**: Analyze systems trade-offs (memory vs accuracy, speed vs quality)

---

## Key Achievements

### 🎯 Milestone 04: CNN Revolution (1998)

**After Module 09**, you'll recreate Yann LeCun's breakthrough:

```bash
cd milestones/04_1998_cnn
python 02_lecun_cifar10.py  # 75%+ accuracy on CIFAR-10
```

**What makes this special**: You're not just importing `torch.nn.Conv2d`—you built the entire convolutional architecture from scratch.

### 🎯 Milestone 05: Transformer Era (2017)

**After Module 13**, you'll implement the attention revolution:

```bash
cd milestones/05_2017_transformer
python 01_vaswani_generation.py  # Text generation with YOUR transformer
```

**What makes this special**: Your attention implementation powers the same architecture behind GPT, ChatGPT, and modern LLMs.

---

## Two Parallel Tracks

The Architecture tier splits into two parallel paths that can be learned in any order:

**Vision Track (Modules 08-09)**:
- DataLoader → Spatial (Conv2d + Pooling)
- Enables computer vision applications
- Culminates in CNN milestone

**Language Track (Modules 10-13)**:
- Tokenization → Embeddings → Attention → Transformers
- Enables natural language processing
- Culminates in Transformer milestone

**Recommendation**: Complete both tracks in order (08→09→10→11→12→13), but you can prioritize the track that interests you more.

---

## Next Steps

**Ready to build modern architectures?**

```bash
# Start the Architecture tier
tito module start 08_dataloader

# Or jump to language models
tito module start 10_tokenization
```

**Or explore other tiers:**

- **[🏗 Foundation Tier](foundation)** (Modules 01-07): Mathematical foundations
- **[⏱️ Optimization Tier](optimization)** (Modules 14-19): Production-ready performance
- **[🏅 Torch Olympics](olympics)** (Module 20): Compete in ML systems challenges

---

**[← Back to Home](../intro)** • **[View All Modules](../chapters/00-introduction)** • **[Historical Milestones](../chapters/milestones)**
