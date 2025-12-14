# Quick Reference

**One-page cheatsheet for experienced developers**

## Essential Commands

### Setup & Verification
```bash
# Initial setup
git clone https://github.com/harvard-edge/cs249r_book.git
cd TinyTorch
./setup-environment.sh
source activate.sh

# Verify installation
tito system health

# System information
tito system info
```

### Core Workflow
```bash
# 1. Edit module in Jupyter
cd modules/NN_name
jupyter lab NN_name.ipynb

# 2. Export to package
tito module complete N

# 3. Validate (run milestone scripts)
cd milestones/MM_YYYY_name
python script.py
```

### Module Management
```bash
# Export specific module
tito module complete 01

# Check module status
tito module status

# Check system health
tito system health
```

### Community & Benchmarking
```bash
# Log in to community (optional)
tito community login

# Run baseline benchmark
tito benchmark baseline

# View profile
tito community profile
```

## Module Dependencies

### Foundation (Required for All)
**Modules 01-07**: Tensor, Activations, Layers, Losses, Autograd, Optimizers, Training

**Unlocks:**
- Milestone 01: Perceptron (1957)
- Milestone 02: XOR Crisis (1969)
- Milestone 03: MLP (1986)

### Architecture (Vision + Language)
**Modules 08-13**: DataLoader, Convolutions, Tokenization, Embeddings, Attention, Transformers

**Unlocks:**
- Milestone 04: CNN (1998) - requires M01-09
- Milestone 05: Transformer (2017) - requires M01-13

### Optimization (Production)
**Modules 14-19**: Profiling, Quantization, Compression, Memoization, Acceleration, Benchmarking

**Unlocks:**
- Milestone 06: Torch Olympics (2018) - requires M01-19

### Capstone
**Module 20**: Torch Olympics Competition

## Common Workflows

### Starting a New Module
```bash
# Navigate to module directory
cd modules/05_autograd

# Open in Jupyter
jupyter lab 05_autograd.ipynb

# Implement required functions
# Run inline tests
# Add docstrings

# Export when complete
cd ../..
tito module complete 05
```

### Debugging Module Errors
```bash
# Check system health
tito system health

# View detailed error logs
tito module complete N --verbose

# Reset module if needed
tito module reset N

# Reimport in Python
python
>>> from tinytorch import *
>>> # Test your implementation
```

### Running Milestones
```bash
# After completing Foundation (M01-07)
tito milestone run perceptron

# After completing Architecture (M01-09)
tito milestone run cnn

# After completing Optimization (M01-19)
tito milestone run mlperf

# List all milestones and their requirements
tito milestone list
```

## File Structure

```
TinyTorch/
├── modules/ # Source notebooks (edit these)
│ ├── 01_tensor/
│ │ └── 01_tensor.ipynb
│ └── ...
├── tinytorch/ # Exported package (auto-generated)
│ ├── core/
│ ├── nn/
│ └── ...
├── milestones/ # Validation scripts (run these)
│ ├── 01_1957_perceptron/
│ ├── 02_1969_xor/
│ └── ...
└── tito/ # CLI tool
```

## Import Patterns

```python
# After exporting modules
from tinytorch.core.tensor import Tensor # M01
from tinytorch.nn.activations import ReLU, Softmax # M02
from tinytorch.nn.layers import Linear # M03
from tinytorch.nn.losses import CrossEntropyLoss # M04
from tinytorch.autograd import backward # M05
from tinytorch.optim import SGD, Adam # M06
from tinytorch.training import Trainer # M07
```

## Troubleshooting Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| `ModuleNotFoundError: tinytorch` | Run `tito module complete N` to export |
| `tito: command not found` | Run `source activate.sh` |
| Import works in notebook, fails in Python | Restart Python kernel after export |
| Tests pass in notebook, fail in milestone | Check module dependencies (M01-07 required) |
| OOM errors | Profile memory usage, check for unnecessary copies |
| NaN losses | Check gradient flow, activation stability |

## Performance Expectations

### Baseline Benchmarks (Your Hardware May Vary)
- **M01-07**: Perceptron trains in <1 second
- **M01-09**: CIFAR-10 CNN trains in 10-30 minutes (CPU)
- **M01-13**: Small transformer inference in 1-5 seconds
- **M01-19**: Optimized models run 10-40× faster

### Memory Usage
- **M01**: Tensor operations ~100 MB
- **M01-09**: CNN training ~1-2 GB
- **M01-13**: Transformer training ~2-4 GB
- **M01-19**: Optimized models use 50-90% less memory

## NBGrader (For Students in Courses)

```bash
# If using NBGrader in classroom setting
# Submit your completed notebook
# Do NOT submit the exported package

# Grading components:
# - 70% Auto-graded (tests)
# - 30% Manual (systems thinking questions)
```

## Next Steps

- **New to TinyTorch?** Start with [Getting Started](../getting-started.md)
- **Stuck on a module?** Check [Troubleshooting](troubleshooting.md)
- **Need detailed docs?** See [CLI Documentation](overview.md)
- **Teaching a course?** See [For Instructors & TAs](../for-instructors.md)


** Pro Tip**: Bookmark this page for quick command reference while building!
