# Getting Started with TinyTorch

```{warning} Early Explorer Territory

You're ahead of the curve. TinyTorch is functional but still being refined. Expect rough edges, incomplete documentation, and things that might change. If you proceed, you're helping us shape this by finding what works and what doesn't.

**Best approach right now:** Browse the code and concepts. For hands-on building, check back when we announce classroom readiness (Summer/Fall 2026).

Questions or feedback? [Join the discussion](https://github.com/harvard-edge/cs249r_book/discussions/1076)
```

```{note} Prerequisites Check
This guide requires **Python programming** (classes, functions, NumPy basics) and **basic linear algebra** (matrix multiplication).
```

## The Journey

TinyTorch follows a simple pattern: **build modules, unlock milestones, recreate ML history**.

```{mermaid}
:align: center
graph LR
    A[Install] --> B[Setup]
    B --> C[Start Module]
    C --> D[Complete Module]
    D --> E[Run Milestone]
    E --> C

    style A fill:#e3f2fd
    style B fill:#e3f2fd
    style C fill:#fff3e0
    style D fill:#f0fdf4
    style E fill:#fce4ec
```

As you complete modules, you unlock milestones that recreate landmark moments in ML history—using YOUR code.

---

## Step 1: Install & Setup (2 Minutes)

```bash
# Install TinyTorch (run from a project folder like ~/projects)
curl -sSL tinytorch.ai/install | bash

# Activate and verify
cd tinytorch
source .venv/bin/activate
tito setup
```

**What this does:**
- Checks your system (Python 3.8+, git)
- Downloads TinyTorch to a `tinytorch/` folder
- Creates an isolated virtual environment
- Installs all dependencies
- Verifies installation

---

## Step 2: Your First Module (15 Minutes)

Let's build Module 01 (Tensor)—the foundation of all neural networks.

### Start the module

```bash
tito module start 01
```

This opens the module notebook and tracks your progress.

### Work in the notebook

Edit `modules/01_tensor/01_tensor.ipynb` in Jupyter:

```bash
jupyter lab modules/01_tensor/01_tensor.ipynb
```

You'll implement:
- N-dimensional array creation
- Mathematical operations (add, multiply, matmul)
- Shape manipulation (reshape, transpose)

### Complete the module

When your implementation is ready, export it to the TinyTorch package:

```bash
tito module complete 01
```

Your code is now importable:

```python
from tinytorch.core.tensor import Tensor  # YOUR implementation!
x = Tensor([1, 2, 3])
```

---

## Step 3: Your First Milestone

Now for the payoff! After completing the required modules (01-03), run a milestone:

```bash
tito milestone run perceptron
```

The milestone uses YOUR implementations to recreate Rosenblatt's 1957 Perceptron:

```
Checking prerequisites for Milestone 01...
All required modules completed!

Testing YOUR implementations...
  * Tensor import successful
  * Activations import successful
  * Layers import successful
YOUR TinyTorch is ready!

+----------------------- Milestone 01 (1957) -----------------------+
|  Milestone 01: Perceptron (1957)                                  |
|  Frank Rosenblatt's First Neural Network                          |
|                                                                   |
|  Running: milestones/01_1957_perceptron/01_rosenblatt_forward.py  |
|  All code uses YOUR TinyTorch implementations!                    |
+-------------------------------------------------------------------+

Starting Milestone 01...

Assembling perceptron with YOUR TinyTorch modules...
   * Linear layer: 2 -> 1 (YOUR Module 03!)
   * Activation: Sigmoid (YOUR Module 02!)

+-------------------- Achievement Unlocked --------------------+
|  MILESTONE ACHIEVED!                                         |
|                                                              |
|  You completed Milestone 01: Perceptron (1957)               |
|  Frank Rosenblatt's First Neural Network                     |
|                                                              |
|  What makes this special:                                    |
|  - Every tensor operation: YOUR Tensor class                 |
|  - Every layer: YOUR Linear implementation                   |
|  - Every activation: YOUR Sigmoid function                   |
+--------------------------------------------------------------+
```

You're recreating ML history with your own code. *By Module 19, you'll benchmark against MLPerf—the industry standard for ML performance.*

---

## The Pattern Continues

As you complete more modules, you unlock more milestones:

| Modules Completed | Milestone Unlocked | What You Recreate |
|-------------------|-------------------|-------------------|
| 01-03 | `perceptron` | The 1957 Perceptron |
| 01-05 | `backprop` | 1986 Backpropagation |
| 01-07 | `lenet` | 1989 LeNet CNN |
| 01-09 | `alexnet` | 2012 AlexNet |
| 01-13 | `transformer` | 2017 Transformer |
| 01-19 | `mlperf` | MLPerf Benchmarks |

See all milestones and their requirements:

```bash
tito milestone list
```

---

## Quick Reference

Here are the commands you'll use throughout your journey:

```bash
# Module workflow
tito module start <N>       # Start working on module N
tito module complete <N>    # Export module to package
tito module status          # See your progress across all modules

# Milestones
tito milestone list         # See all milestones & requirements
tito milestone run <name>   # Run a milestone with your code

# Utilities
tito setup                  # Verify installation
tito update                 # Update TinyTorch (your work is preserved)
tito --help                 # Full command reference
```

---

## Module Progression

TinyTorch has 20 modules organized in progressive tiers:

| Tier | Modules | Focus | Time Estimate |
|------|---------|-------|---------------|
| **Foundation** | 01-07 | Core ML infrastructure (tensors, autograd, training) | ~15-20 hours |
| **Architecture** | 08-13 | Neural architectures (data loading, CNNs, transformers) | ~18-24 hours |
| **Optimization** | 14-19 | Production optimization (profiling, quantization) | ~18-24 hours |
| **Capstone** | 20 | Torch Olympics Competition | ~8-10 hours |

**Total: ~60-80 hours** over 14-18 weeks (4-6 hours/week pace).

**See [Foundation Tier Overview](tiers/foundation)** for detailed module descriptions.

---

## Join the Community (Optional)

After setup, join the global TinyTorch community:

```bash
tito community login        # Join the community
```

See **[Community Guide](community.md)** for complete features.

---

## For Instructors & TAs

```{note}
Classroom support with NBGrader integration is coming (target: Summer/Fall 2026). TinyTorch works for self-paced learning today.
```

**What's Planned:**
- Automated assignment generation with solutions removed
- Auto-grading against test suites
- Progress tracking across all 20 modules
- Grade export to CSV for LMS integration

**Interested in early adoption?** [Join the discussion](https://github.com/harvard-edge/cs249r_book/discussions/1076) to share your use case.

---

**Ready to start?** Run `tito module start 01` and begin building!
