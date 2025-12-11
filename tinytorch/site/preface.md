# Welcome

Everyone wants to be an astronaut 🧑‍🚀. Very few want to be the rocket scientist 🚀.

In machine learning, we see the same pattern. Everyone wants to train models, run inference, deploy AI. Very few want to understand how the frameworks actually work. Even fewer want to build one.

The world is full of users. We do not have enough builders—people who can debug, optimize, and adapt systems when the black box breaks down.

This is the gap TinyTorch exists to fill.

## The Problem

Most people can use PyTorch or TensorFlow. They can import libraries, call functions, train models. But very few understand how these frameworks work: how memory is managed for tensors, how autograd builds computation graphs, how optimizers update parameters. And almost no one has a guided, structured way to learn that from the ground up.

Why does this matter? Because users hit walls that builders do not. When your model runs out of memory, you need to understand tensor allocation. When gradients explode, you need to understand the computation graph. When training is slow, you need to understand where the bottlenecks are. When you need to deploy on a microcontroller, you need to understand what can be stripped away and what cannot. The framework becomes a black box that you cannot debug, optimize, or adapt. You are stuck waiting for someone else to solve your problem.

Students cannot learn this from production code. PyTorch is too large, too complex, too optimized. Fifty thousand lines of C++ across hundreds of files. No one learns to build rockets by studying the Saturn V.

They also cannot learn it from toy scripts. A hundred-line neural network does not reveal the architecture of a framework. It hides it.

## The Solution

TinyTorch teaches you the AI bricks—the stable engineering foundations you can use to build any AI system. Small enough to learn from: bite-sized code that runs even on a Raspberry Pi. Big enough to matter: showing the real architecture of how frameworks are built.

If the [Machine Learning Systems](https://mlsysbook.ai) textbook teaches you the concepts of the rocket ship (propulsion, guidance, life support) then TinyTorch is where you actually build a small rocket with your own hands. Not a toy. A real framework with tensors, autograd, layers, optimizers, data loaders, and training loops. Twenty modules that take you from first principles to a working system.

This is how people move from *using* machine learning to *engineering* machine learning systems. This is how someone becomes an AI systems engineer rather than someone who only knows how to run code in a notebook.

## Who This Is For

**Students and Researchers** who want to understand ML systems deeply, not just use them superficially. If you have taken an ML course and wondered "how does that actually work?", this guide is for you.

**ML Engineers** who need to debug, optimize, and deploy models in production. Understanding the systems underneath makes you more effective at every stage of the ML lifecycle.

**Systems Programmers** curious about ML. You understand systems thinking: memory hierarchies, computational complexity, performance optimization. You want to apply it to ML.

**Self-taught Engineers** who can use frameworks but want to know how they work. You might be preparing for ML infrastructure roles and need systems-level understanding.

What you need is not another API tutorial. You need to build.[^pin]

[^pin]: My own background was in compilers, specifically just-in-time (JIT) compilation. But I did not become a systems engineer by reading papers alone. I became one by building [Pin](https://software.intel.com/content/www/us/en/develop/articles/pin-a-dynamic-binary-instrumentation-tool.html), a dynamic binary instrumentation engine that uses JIT technology. The lesson stayed with me: reading teaches concepts, but building deepens understanding.

## What You Will Build

By the end of TinyTorch, you will have implemented:

- A tensor library with broadcasting, reshaping, and matrix operations
- Activation functions with numerical stability considerations
- Neural network layers: linear, convolutional, normalization
- An autograd engine that builds computation graphs and computes gradients
- Optimizers that update parameters using those gradients
- Data loaders that handle batching, shuffling, and preprocessing
- A complete training loop that ties everything together
- Tokenizers, embeddings, attention, and transformer architectures
- Profiling, quantization, and optimization techniques

This is not a simulation. This is the actual architecture of modern ML frameworks, implemented at a scale you can fully understand.

## How to Use This Guide

Each module follows a Build-Use-Reflect cycle:

1. **Build**: Implement the component from scratch, understanding every line
2. **Use**: Apply it to real problems: training networks, processing data
3. **Reflect**: Connect what you built to production systems and understand the tradeoffs

The guide follows a three-tier structure:

**Foundation Tier (Modules 01-09)** builds the core infrastructure: tensors, activations, layers, losses, autograd, optimizers, training loops, data loading, and spatial operations.

**Architecture Tier (Modules 10-13)** implements modern deep learning: tokenization, embeddings, attention mechanisms, and transformers.

**Optimization Tier (Modules 14-19)** focuses on production: profiling, quantization, compression, memoization, acceleration, and benchmarking.

**Module 20: Capstone** brings everything together. It is designed as a launchpad for community competitions we plan to run, fostering lifelong learning and connection among builders who share this path.

Work through Foundation first. Then choose your path based on your interests.

## Learning Approach

**Type every line of code yourself.** Do not copy-paste. The learning happens in the struggle of implementation.

**Profile your code.** Use the built-in profiling tools to understand memory and performance characteristics. Measure first, optimize second.

**Run the tests.** Every module includes comprehensive tests. When they pass, you have built something real.

**Compare with PyTorch.** Once your implementation works, compare it with PyTorch's equivalent to see what optimizations production frameworks add.

Take your time. The goal is not to finish fast. The goal is to understand deeply.

## Prerequisites

You should be comfortable with:

- **Python programming**: functions, classes, NumPy basics
- **Linear algebra**: matrix operations, vector spaces
- **Calculus**: derivatives, chain rule (for backpropagation)
- **Basic ML concepts**: neural networks, training, loss functions

If you have taken an introductory ML course and can write Python code, you are ready.

## The Bigger Picture

TinyTorch is part of a larger effort to educate a million learners at the edge of AI. The [Machine Learning Systems](https://mlsysbook.ai) textbook provides the conceptual foundation. TinyTorch provides the hands-on implementation experience. Together, they form a complete path into ML systems engineering.

The next generation of engineers cannot rely on magic. They need to see how everything fits together, from tensors all the way to systems. They need to feel that the world of ML systems is not an unreachable tower but something they can open, shape, and build.

That is what TinyTorch offers: the confidence that comes from building something real.

---

*Prof. Vijay Janapa Reddi*
*Harvard University*
*2025*
