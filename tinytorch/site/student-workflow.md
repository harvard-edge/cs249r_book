# Student Workflow

This guide explains the actual day-to-day workflow for building your ML framework with TinyTorch.

## The Core Workflow

TinyTorch follows a simple three-step cycle:

```{mermaid}
:align: center
:caption: Architecture Overview
graph LR
    A[Work in Notebooks<br/>modules/NN_name.ipynb] --> B[Export to Package<br/>tito module complete N]
    B --> C[Validate with Milestones<br/>Run milestone scripts]
    C --> A

    style A fill:#e3f2fd
    style B fill:#f0fdf4
    style C fill:#fef3c7
```

### Step 1: Edit Modules

Work on module notebooks in `modules/`:

```bash
# Example: Working on Module 03 (Layers)
cd modules/03_layers
jupyter lab 03_layers.ipynb
```

Each module is a Jupyter notebook that you edit interactively. You'll:
- Implement the required functionality
- Add docstrings and comments
- Run and test your code inline
- See immediate feedback

### Step 2: Export to Package

Once your module implementation is complete, export it to the main TinyTorch package:

```bash
tito module complete MODULE_NUMBER
```

This command:
- Converts your source files to the `tinytorch/` package
- Validates [NBGrader](https://nbgrader.readthedocs.io/) metadata
- Makes your implementation available for import

**Example:**
```bash
tito module complete 03  # Export Module 03 (Layers)
```

After export, your code is importable:
```python
from tinytorch.layers import Linear  # YOUR implementation!
```

### Step 3: Validate with Milestones

Run milestone scripts to prove your implementation works:

```bash
cd milestones/01_1957_perceptron
python 01_rosenblatt_forward.py  # Uses YOUR Tensor (M01)
python 02_rosenblatt_trained.py  # Uses YOUR layers (M01-M07)
```

Each milestone has a README explaining:
- Required modules
- Historical context
- Expected results
- What you're learning

See [Milestones Guide](chapters/milestones.md) for the full progression.

## Testing Your Implementation

TinyTorch uses a **three-phase testing approach** to ensure your code works correctly at every level:

```bash
# Run comprehensive tests for a module
tito module test 03
```

### Three-Phase Testing

When you run `tito module test`, it executes three phases:

**Phase 1: Inline Unit Tests** (Yellow)
- Quick sanity checks from the module source file
- Tests the core functionality you just implemented
- Fast feedback loop

**Phase 2: Module Tests** (Blue)
- Runs pytest with educational output (`--tinytorch`)
- Shows **WHAT** each test checks
- Explains **WHY** it matters
- Provides **learning tips** when tests fail
- Groups tests by module for clarity

**Phase 3: Integration Tests** (Magenta)
- Verifies your module works with all previous modules
- Tests gradient flow, layer composition, training loops
- Catches "it works in isolation but fails in the system" bugs

### Testing Options

```bash
# Full three-phase testing (recommended)
tito module test 03

# Only inline unit tests (quick check)
tito module test 03 --unit-only

# Skip integration tests (faster feedback)
tito module test 03 --no-integration

# Verbose output with details
tito module test 03 -v
```

### Why Integration Tests Matter

A common mistake is implementing a module that passes its own tests but breaks when combined with others. For example:
- Your Layer might compute forward passes correctly but have wrong gradient shapes
- Your Optimizer might update weights but break the computation graph
- Your Attention might work for one head but fail with multiple heads

Integration tests catch these issues early, before you spend hours debugging in milestones.

## Module Progression

TinyTorch has 20 modules organized in three tiers:

### Foundation (Modules 01-07)
Core ML infrastructure - tensors, autograd, training loops

**Milestones unlocked:**
- M01: Perceptron (after Module 07)
- M02: XOR Crisis (after Module 07)

### Architecture (Modules 08-13)
Neural network architectures - data loading, CNNs, transformers

**Milestones unlocked:**
- M03: MLPs (after Module 08)
- M04: CNNs (after Module 09)
- M05: Transformers (after Module 13)

### Optimization (Modules 14-19)
Production optimization - profiling, quantization, benchmarking

**Milestones unlocked:**
- M06: Torch Olympics (after Module 18)

### Capstone Competition (Module 20)
Apply all optimizations in the Torch Olympics Competition

## Typical Development Session

Here's what a typical session looks like:

```bash
# 1. Work on a module
cd modules/05_autograd
jupyter lab autograd_dev.ipynb
# Edit your implementation interactively

# 2. Export when ready
tito module complete 05

# 3. Validate with existing milestones
cd ../milestones/01_1957_perceptron
python 01_rosenblatt_forward.py  # Should still work!

# 4. Continue to next module or milestone
```

## TITO Commands Reference

The most important commands you'll use:

```bash
# Export module to package
tito module complete MODULE_NUMBER

# Check module status
tito module status

# System information
tito system info

# Community and benchmark
tito community login
tito benchmark baseline
```

For complete command documentation, see [TITO CLI Reference](tito/overview.md).

## Progress Tracking (Optional)

TinyTorch includes progress tracking for modules:

```bash
tito module status  # View module completion status
```

This is helpful for self-assessment but **not required** for the core workflow. The essential cycle remains: edit → export → validate.

## Notebook Platform Options

TinyTorch notebooks work with multiple platforms, but **important distinction**:

### Online Notebooks (Viewing & Exploration)
- **Jupyter/MyBinder**: Click "Launch Binder" on any notebook page - great for viewing
- **Google Colab**: Click "Launch Colab" for GPU access - good for exploration
- **Marimo**: Click "~ Open in Marimo" for reactive notebooks - excellent for learning

**⚠ Important**: Online notebooks are for **viewing and learning**. They don't have the full TinyTorch package installed, so you can't:
- Run milestone validation scripts
- Import from `tinytorch.*` modules
- Execute full experiments
- Use the complete CLI tools

### Local Setup (Required for Full Package)
**To actually build and experiment**, you need a **local installation**:

```bash
# Clone and setup locally
git clone https://github.com/mlsysbook/TinyTorch.git
cd TinyTorch
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .  # Install TinyTorch package
```

**Why local?**
- ✓ Full `tinytorch.*` package available
- ✓ Run milestone validation scripts
- ✓ Use `tito` CLI commands
- ✓ Execute complete experiments
- ✓ Export modules to package
- ✓ Full development workflow

**Note for NBGrader assignments**: Submit `.ipynb` files (not Marimo's `.py` format) to preserve grading metadata.

## Community & Benchmarking

### Join the Community

After completing setup, join the global TinyTorch community:

```bash
# Log in to join the community
tito community login

# View your profile and progress
tito community profile

# Check your community status
tito community status
```

**Privacy:** All information is optional. Data is stored locally in `.tinytorch/` directory. See [Community Guide](community.md) for details.

### Benchmark Your Progress

Validate your setup and track performance:

```bash
# Quick baseline benchmark (after setup)
tito benchmark baseline

# Full capstone benchmarks (after Module 20)
tito benchmark capstone --track all
```

**Baseline Benchmark:** Quick validation that your setup works correctly - your "Hello World" moment!

**Capstone Benchmark:** Full performance evaluation across speed, compression, accuracy, and efficiency tracks.

See [Community Guide](community.md) for complete community and benchmarking features.

## Instructor Integration

TinyTorch supports [NBGrader](https://nbgrader.readthedocs.io/) for classroom use. See the [Instructor Guide](usage-paths/classroom-use.md) for complete setup and grading workflows.

For now, focus on the student workflow: building your implementations and validating them with milestones.

## What's Next?

1. **Start with Module 01**: See [Getting Started](intro.md)
2. **Follow the progression**: Each module builds on previous ones
3. **Run milestones**: Prove your implementations work
4. **Build intuition**: Understand ML systems from first principles

The goal isn't just to write code - it's to **understand** how modern ML frameworks work by building one yourself.
