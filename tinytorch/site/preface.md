# Preface

## Why This Book Exists

Machine learning has transformed from an academic curiosity into the infrastructure powering our digital world. Yet most ML education treats frameworks like PyTorch and TensorFlow as magic boxes—you import them, call their functions, and hope for the best. When things break, you're stuck. When performance matters, you're guessing.

This book takes a different approach: **build everything from scratch**.

## The Systems Gap

There's a critical gap in ML education. Students learn the mathematics—backpropagation, optimization algorithms, attention mechanisms. They learn to use libraries—`nn.Linear()`, `optim.Adam()`, `nn.Transformer()`. But they rarely understand the systems engineering between math and library:

- How does a tensor actually store and compute with multidimensional data?
- Why does Adam optimizer need 3× the memory of SGD?
- What makes attention mechanisms scale quadratically?
- When does data loading become the bottleneck?

These aren't academic questions—they're the daily reality of ML systems engineers debugging OOM errors, optimizing inference latency, and deploying models to production.

## Who This Book Is For

This book is designed for four audiences:

**Students & Researchers** who want to understand ML systems deeply, not just use them superficially. If you've taken an ML course and wondered "how does that actually work?", this book is for you.

**ML Engineers** who need to debug, optimize, and deploy models in production. Understanding the systems underneath makes you more effective at every stage of the ML lifecycle.

**Systems Programmers** curious about ML. You understand systems thinking—memory hierarchies, computational complexity, performance optimization—and want to apply it to ML.

**Educators** teaching ML systems. This book provides a complete pedagogical framework emphasizing systems thinking, with built-in assessment tools and clear learning outcomes.

## How This Book Works

Each module follows the same pattern:

1. **Conceptual Understanding** - Why does this component exist? What problem does it solve?
2. **Implementation** - Build it yourself in clean, readable Python
3. **Profiling** - Measure memory usage, computational complexity, performance characteristics
4. **Validation** - Comprehensive tests ensure your implementation works correctly
5. **Historical Context** - Recreate breakthrough results using your implementations

You'll start with tensors and build up through 20 modules, each adding a new capability to your growing ML framework. By the end, you'll have implemented a complete system capable of training transformer models—and you'll understand every line of code.

## Learning Approach

**Build → Profile → Optimize**

This isn't just a coding exercise. You'll develop the mindset of an ML systems engineer:

- Measure first, optimize second
- Understand memory access patterns and cache behavior
- Profile to find bottlenecks, don't guess
- Make informed trade-offs between speed, memory, and accuracy

## Prerequisites

You should be comfortable with:

- **Python programming** - Functions, classes, NumPy basics
- **Linear algebra** - Matrix operations, vector spaces
- **Calculus** - Derivatives, chain rule (for backpropagation)
- **Basic ML concepts** - Neural networks, training, loss functions

If you've taken an introductory ML course and can write Python code, you're ready.

## Course Structure

The book follows a three-tier structure that mirrors ML history:

**Foundation Tier (Modules 01-07)** builds the mathematical infrastructure: tensors, activation functions, layers, loss functions, automatic differentiation, optimizers, and training loops. This tier covers 1950s-1990s breakthroughs.

**Architecture Tier (Modules 08-13)** implements modern AI: data loading, convolutional networks for vision, tokenization, embeddings, attention mechanisms, and transformers. This tier covers 1990s-2017 innovations.

**Optimization Tier (Modules 14-20)** focuses on production deployment: profiling, memoization, quantization, compression, hardware acceleration, and benchmarking. This tier addresses modern systems challenges.

## Historical Milestones

As you complete modules, you'll unlock historical milestone demonstrations that prove your implementations work:

- **1957: Perceptron** - First trainable network
- **1969: XOR Solution** - Multi-layer networks with backpropagation
- **1986: MNIST MLP** - Digit recognition with 95%+ accuracy
- **1998: CIFAR-10 CNN** - Image classification with 75%+ accuracy
- **2017: Transformers** - Language generation with attention
- **2024: Systems Age** - Production optimization and deployment

These aren't toy examples—you'll recreate genuine breakthroughs using only the code you've written.

## Getting the Most from This Book

**Type every line of code yourself.** Don't copy-paste. The learning happens in the struggle of implementation.

**Profile your code.** Use the built-in profiling tools to understand memory and performance characteristics.

**Run the tests.** Every module includes comprehensive tests. When they pass, you've built something real.

**Compare with PyTorch.** Once your implementation works, compare it with PyTorch's equivalent to see what optimizations production frameworks add.

**Join the community.** Share your progress, ask questions, help others. Learning is more effective together.

## Acknowledgments

This book emerged from years of teaching ML systems at Harvard University. I'm grateful to:

- Students who challenged assumptions and pushed for deeper understanding
- The PyTorch, TensorFlow, and JAX teams whose frameworks inspired this educational approach
- The MLPerf community for benchmark standards that inform our performance discussions
- Open source contributors who built the tools (Jupyter, NumPy, NBGrader) that make this course possible

## Let's Begin

The difference between using a library and understanding a system is the difference between being limited by tools and being empowered to create them.

Let's build something real.

---

**Prof. Vijay Janapa Reddi**
Cambridge, Massachusetts
January 2025
