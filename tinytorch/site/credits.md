# Credits & Acknowledgments

**TinyTorch stands on the shoulders of giants.**

This project draws inspiration from pioneering educational ML frameworks and owes its existence to the open source community's commitment to accessible ML education.

---

## Core Inspirations

### MiniTorch
**[minitorch.github.io](https://minitorch.github.io/)** by Sasha Rush (Cornell Tech)

TinyTorch's pedagogical DNA comes from MiniTorch's brilliant "build a framework from scratch" approach. MiniTorch pioneered teaching ML through implementation rather than usage, proving students gain deeper understanding by building systems themselves.

**What MiniTorch teaches**: Automatic differentiation through minimal, elegant implementations

**How TinyTorch differs**: Extends to full systems engineering including optimization, profiling, and production deployment across Foundation → Architecture → Optimization tiers

**When to use MiniTorch**: Excellent complement for deep mathematical understanding of autodifferentiation

**Connection to TinyTorch**: Modules 05-07 (Autograd, Optimizers, Training) share philosophical DNA with MiniTorch's core pedagogy

---

### micrograd
**[github.com/karpathy/micrograd](https://github.com/karpathy/micrograd)** by Andrej Karpathy

Micrograd demonstrated that automatic differentiation—the heart of modern ML—can be taught in ~100 lines of elegant Python. Its clarity and simplicity inspired TinyTorch's emphasis on understandable implementations.

**What micrograd teaches**: Autograd engine in 100 beautiful lines of Python

**How TinyTorch differs**: Comprehensive framework covering vision, language, and production systems (20 modules vs. single-file implementation)

**When to use micrograd**: Perfect 2-hour introduction before starting TinyTorch

**Connection to TinyTorch**: Module 05 (Autograd) teaches the same core concepts with systems engineering focus

---

### nanoGPT
**[github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)** by Andrej Karpathy

nanoGPT's minimalist transformer implementation showed how to teach modern architectures without framework abstraction. TinyTorch's transformer modules (12, 13) follow this philosophy: clear, hackable implementations that reveal underlying mathematics.

**What nanoGPT teaches**: Clean transformer implementation for understanding GPT architecture

**How TinyTorch differs**: Build transformers from tensors up, understanding all dependencies from scratch

**When to use nanoGPT**: Complement to TinyTorch Modules 10-13 for transformer-specific deep-dive

**Connection to TinyTorch**: Module 13 (Transformers) culminates in similar architecture built from your own tensor operations

---

### tinygrad
**[github.com/geohot/tinygrad](https://github.com/geohot/tinygrad)** by George Hotz

Tinygrad proves educational frameworks can achieve impressive performance. While TinyTorch optimizes for learning clarity over speed, tinygrad's emphasis on efficiency inspired our Optimization Tier's production-focused modules.

**What tinygrad teaches**: Performance-focused educational framework with actual GPU acceleration

**How TinyTorch differs**: Pedagogy-first with explicit systems thinking and scaffolding (educational over performant)

**When to use tinygrad**: After TinyTorch for performance optimization deep-dive and GPU programming

**Connection to TinyTorch**: Modules 14-19 (Optimization Tier) share production systems focus

---


## What Makes TinyTorch Unique

TinyTorch combines inspiration from these projects into a comprehensive ML systems course:

- **Comprehensive Scope**: Only educational framework covering Foundation → Architecture → Optimization
- **Systems Thinking**: Every module includes profiling, complexity analysis, production context
- **Historical Validation**: Milestone system proving implementations through ML history (1957 → 2018)
- **Pedagogical Scaffolding**: Progressive disclosure, Build → Use → Reflect methodology
- **Production Context**: Direct connections to PyTorch, TensorFlow, and industry practices

---



## Community Contributors

TinyTorch is built by students, educators, and ML engineers who believe in accessible systems education.

**[View all contributors on GitHub](https://github.com/harvard-edge/TinyTorch/graphs/contributors)**

---

## How to Contribute

TinyTorch is open source and welcomes contributions:

- **Found a bug?** Report it on [GitHub Issues](https://github.com/harvard-edge/TinyTorch/issues)
- **Improved documentation?** Submit a pull request
- **Built something cool?** Share it in [GitHub Discussions](https://github.com/harvard-edge/TinyTorch/discussions)

**[See contribution guidelines](https://github.com/harvard-edge/TinyTorch/blob/main/CONTRIBUTING.md)**

---

## License

TinyTorch is released under the MIT License, ensuring it remains free and open for educational use.

---

**Thank you to everyone building the future of accessible ML education.**
