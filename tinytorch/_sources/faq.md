# Frequently Asked Questions

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h2 style="margin: 0 0 1rem 0; color: #495057;">Common Questions About TinyTorch</h2>
<p style="margin: 0; font-size: 1.1rem; color: #6c757d;">Why build from scratch? Why not just use PyTorch? All your questions answered.</p>
</div>

## General Questions

### What is TinyTorch?

TinyTorch is an educational ML systems framework where you build a complete neural network library from scratch. Instead of using PyTorch or TensorFlow as black boxes, you implement every component yourself—tensors, gradients, optimizers, attention mechanisms—gaining deep understanding of how modern ML frameworks actually work.

### Who is TinyTorch for?

TinyTorch is designed for:

- **Students** learning ML who want to understand what's happening under the hood
- **ML practitioners** who want to debug models more effectively
- **Systems engineers** building or optimizing ML infrastructure
- **Researchers** who need to implement novel architectures
- **Educators** teaching ML systems (not just ML algorithms)

If you've ever wondered "why does my model OOM?" or "how does autograd actually work?", TinyTorch is for you.

### How long does it take?

**Quick exploration**: 2-4 weeks focusing on Foundation Tier (Modules 01-07)
**Complete course**: 14-18 weeks implementing all three tiers (20 modules)
**Flexible approach**: Pick specific modules based on your learning goals

You control the pace. Some students complete it in intensive 8-week sprints, others spread it across a semester.

---

## Why TinyTorch vs. Alternatives?

### Why not just use PyTorch or TensorFlow directly?

**Short answer**: Because using a library doesn't teach you how it works.

**The problem with "just use PyTorch":**

When you write:
```python
import torch.nn as nn
model = nn.Linear(784, 10)
optimizer = torch.optim.Adam(model.parameters())
```

You're calling functions you don't understand. When things break (and they will), you're stuck:
- **OOM errors**: Why? How much memory does this need?
- **Slow training**: What's the bottleneck? Data loading? Computation?
- **NaN losses**: Where did gradients explode? How do you debug?

**What TinyTorch teaches:**

When you implement `Linear` yourself:
```python
class Linear:
    def __init__(self, in_features, out_features):
        # You understand EXACTLY what memory is allocated
        self.weight = randn(in_features, out_features) * 0.01  # Why 0.01?
        self.bias = zeros(out_features)  # Why zeros?

    def forward(self, x):
        self.input = x  # Why save input? (Hint: backward pass)
        return x @ self.weight + self.bias  # You know the exact operations

    def backward(self, grad):
        # You wrote this gradient! You can debug it!
        self.weight.grad = self.input.T @ grad
        return grad @ self.weight.T
```

Now you can:
- **Calculate memory requirements** before running
- **Profile and optimize** every operation
- **Debug gradient issues** by inspecting your own code
- **Implement novel architectures** with confidence

### Why TinyTorch instead of Andrej Karpathy's micrograd or nanoGPT?

We love micrograd and nanoGPT! They're excellent educational resources. Here's how TinyTorch differs:

**micrograd (100 lines)**
- **Scope**: Teaches autograd elegantly in minimal code
- **Limitation**: Doesn't cover CNNs, transformers, data loading, optimization
- **Use case**: Perfect introduction to automatic differentiation

**nanoGPT (300 lines)**
- **Scope**: Clean GPT implementation for understanding transformers
- **Limitation**: Doesn't teach fundamentals (tensors, layers, training loops)
- **Use case**: Excellent for understanding transformer architecture specifically

**TinyTorch (20 modules, complete framework)**
- **Scope**: Full ML systems course from mathematical primitives to production deployment
- **Coverage**:
  - Foundation (tensors, autograd, optimizers)
  - Architecture (CNNs for vision, transformers for language)
  - Optimization (profiling, quantization, benchmarking)
- **Outcome**: You build a unified framework supporting both vision AND language models
- **Systems focus**: Memory profiling, performance analysis, and production context built into every module

**Analogy:**
- **micrograd**: Learn how an engine works
- **nanoGPT**: Learn how a sports car works
- **TinyTorch**: Build a complete vehicle manufacturing plant (and understand engines, cars, AND the factory)

**When to use each:**
- **Start with micrograd** if you want a gentle introduction to autograd (1-2 hours)
- **Try nanoGPT** if you specifically want to understand GPT architecture (1-2 days)
- **Choose TinyTorch** if you want complete ML systems engineering skills (8-18 weeks)

### Why not just read PyTorch source code?

**Three problems with reading production framework code:**

1. **Complexity**: PyTorch has 350K+ lines optimized for production, not learning
2. **C++/CUDA**: Core operations are in low-level languages for performance
3. **No learning path**: Where do you even start?

**TinyTorch's pedagogical approach:**

1. **Incremental complexity**: Start with 2D matrices, build up to 4D tensors
2. **Pure Python**: Understand algorithms before optimization
3. **Guided curriculum**: Clear progression from basics to advanced
4. **Systems thinking**: Every module includes profiling and performance analysis

You learn the *concepts* in TinyTorch, then understand how PyTorch optimizes them for production.

---

## Technical Questions

### What programming background do I need?

**Required:**
- Python programming (functions, classes, basic NumPy)
- Basic calculus (derivatives, chain rule)
- Linear algebra (matrix multiplication)

**Helpful but not required:**
- Git version control
- Command-line comfort
- Previous ML course (though TinyTorch teaches from scratch)

### What hardware do I need?

**Minimum:**
- Any laptop with 8GB RAM
- Works on M1/M2 Macs, Intel, AMD

**No GPU required!** TinyTorch runs on CPU and teaches concepts that transfer to GPU optimization.

### Does TinyTorch replace a traditional ML course?

**No, it complements it.**

**Traditional ML course teaches:**
- Algorithms (gradient descent, backpropagation)
- Theory (loss functions, regularization)
- Applications (classification, generation)

**TinyTorch teaches:**
- Systems (how frameworks work)
- Implementation (building from scratch)
- Production (profiling, optimization, deployment)

**Best approach**: Take a traditional ML course for theory, use TinyTorch to deeply understand implementation.

### Can I use TinyTorch for research or production?

**Research**: Absolutely! Build novel architectures with full control
**Production**: TinyTorch is educational—use PyTorch/TensorFlow for production scale

**However:** Understanding TinyTorch makes you much better at using production frameworks. You'll:
- Write more efficient PyTorch code
- Debug issues faster
- Understand performance characteristics
- Make better architectural decisions

---

## Course Structure Questions

### Do I need to complete all 20 modules?

**No!** TinyTorch offers flexible learning paths:

**Three tiers:**
1. **Foundation (01-07)**: Core ML infrastructure—understand how training works
2. **Architecture (08-13)**: Modern AI architectures—CNNs and transformers
3. **Optimization (14-20)**: Production deployment—profiling and acceleration

**Suggested paths:**
- **ML student**: Foundation tier gives you deep understanding
- **Systems engineer**: All three tiers teach complete ML systems
- **Researcher**: Focus on Foundation + Architecture for implementation skills
- **Curious learner**: Pick modules that interest you

### What are the milestones?

Milestones are historical ML achievements you recreate with YOUR implementations:

- **M01: 1957 Perceptron** - First trainable neural network
- **M02: 1969 XOR** - Multi-layer networks solve XOR problem
- **M03: 1986 MLP** - Backpropagation achieves 95%+ on MNIST
- **M04: 1998 CNN** - LeNet-style CNN gets 75%+ on CIFAR-10
- **M05: 2017 Transformer** - GPT-style text generation
- **M06: 2018 Torch Olympics** - Production optimization benchmarking

Each milestone proves your framework works by running actual ML experiments.

**📖 See [Journey Through ML History](chapters/milestones.md)** for details.

### Are the checkpoints required?

**No, they're optional.**

**The essential workflow:**
```
1. Edit modules → 2. Export → 3. Validate with milestones
```

**Optional checkpoint system:**
- Tracks 21 capability checkpoints
- Helpful for self-assessment
- Use `tito checkpoint status` to view progress

**📖 See [Module Workflow](tito/modules.md)** for the core development cycle.

---

## Practical Questions

### How do I get started?

**Quick start (15 minutes):**

```bash
# 1. Clone repository
git clone https://github.com/mlsysbook/TinyTorch.git
cd TinyTorch

# 2. Automated setup
./setup-environment.sh
source activate.sh

# 3. Verify setup
tito system health

# 4. Start first module
cd modules/01_tensor
jupyter lab tensor_dev.py
```

**📖 See [Getting Started Guide](getting-started.md)** for detailed setup.

### What's the typical workflow?

```bash
# 1. Work on module source
cd modules/03_layers
jupyter lab layers_dev.py

# 2. Export when ready
tito module complete 03

# 3. Validate by running milestones
cd ../../milestones/01_1957_perceptron
python rosenblatt_forward.py  # Uses YOUR implementation!
```

**📖 See [Module Workflow](tito/modules.md)** for complete details.

### Can I use this in my classroom?

**Yes!** TinyTorch is designed for classroom use.

**Current status:**
- Students can work through modules individually
- [NBGrader](https://nbgrader.readthedocs.io/) integration coming soon for automated grading
- Instructor tooling under development

**📖 See [Classroom Use Guide](usage-paths/classroom-use.md)** for details.

### How do I get help?

**Resources:**
- **Documentation**: Comprehensive guides for every module
- **GitHub Issues**: Report bugs or ask questions
- **Community**: (Coming soon) Discord/forum for peer support

---

## Philosophy Questions

### Why build from scratch instead of using libraries?

**The difference between using and understanding:**

When you import a library, you're limited by what it provides. When you build from scratch, you understand the foundations and can create anything.

**Real-world impact:**
- **Debugging**: "My model won't train" → You know exactly where to look
- **Optimization**: "Training is slow" → You can profile and fix bottlenecks
- **Innovation**: "I need a novel architecture" → You build it confidently
- **Career**: ML systems engineers who understand internals are highly valued

### Isn't this reinventing the wheel?

**Yes, intentionally!**

**The best way to learn engineering:** Build it yourself.

- Car mechanics learn by taking apart engines
- Civil engineers build bridge models
- Software engineers implement data structures from scratch

**Then** they use production tools with deep understanding.

### Will I still use PyTorch/TensorFlow after this?

**Absolutely!** TinyTorch makes you *better* at using production frameworks.

**Before TinyTorch:**
```python
model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
# It works but... why 128? What's the memory usage? How does ReLU affect gradients?
```

**After TinyTorch:**
```python
model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
# I know: 784*128 + 128*10 params = ~100K params * 4 bytes = ~400KB
# I understand: ReLU zeros negative gradients, affects backprop
# I can optimize: Maybe use smaller hidden layer or quantize to INT8
```

You use the same tools, but with systems-level understanding.

---

## Community Questions

### Can I contribute to TinyTorch?

**Yes!** TinyTorch is open-source and welcomes contributions:

- Bug fixes and improvements
- Documentation enhancements
- Additional modules or extensions
- Educational resources

Check the GitHub repository for contribution guidelines.

### Is there a community?

**Growing!** TinyTorch is launching to the community in December 2024.

- GitHub Discussions for Q&A
- Optional leaderboard for module 20 competition
- Community showcase (coming soon)

### How is TinyTorch maintained?

TinyTorch is developed at the intersection of academia and education:
- Research-backed pedagogy
- Active development and testing
- Community feedback integration
- Regular updates and improvements

---

## Still Have Questions?

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h3 style="margin: 0 0 1rem 0; color: #495057;">Ready to Start Building?</h3>
<p style="margin: 0 0 1.5rem 0; color: #6c757d;">Jump in and start implementing ML systems from scratch</p>
<a href="getting-started.html" style="display: inline-block; background: #007bff; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500; margin-right: 1rem;">Getting Started →</a>
<a href="intro.html" style="display: inline-block; background: #28a745; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500;">Learn More →</a>
</div>

**Can't find your question?** Open an issue on [GitHub](https://github.com/mlsysbook/TinyTorch/issues) and we'll help!
