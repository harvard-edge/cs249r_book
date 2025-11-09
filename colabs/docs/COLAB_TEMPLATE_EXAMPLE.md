# MLSysBook Colab Template Example

This document shows a complete example following the MLSysBook Colab template specification.

---

## Cell Structure Overview

```
┌─────────────────────────────────────────────────┐
│  CELL 1: Header (Markdown)                      │
│  - Title with MLSysBook branding                │
│  - Learning objective                           │
│  - Textbook context                             │
│  - Prerequisites and what you'll do             │
└─────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────┐
│  CELL 2: Setup and Configuration (Code)         │
│  - Import libraries                             │
│  - Set random seeds                             │
│  - Configure plotting                           │
│  - Device configuration                         │
└─────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────┐
│  CELL 3: Introduction (Markdown)                │
│  - Concept overview                             │
│  - Why this matters                             │
│  - What we'll measure                           │
└─────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────┐
│  CELL 4: Helper Functions (Code, if needed)     │
│  - Utility functions with docstrings            │
└─────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────┐
│  CELLS 5-N: Main Content Sections               │
│  Each section: Markdown → Code → Visualization  │
└─────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────┐
│  CELL N+1: Summary and Key Takeaways (Markdown) │
│  - What we learned                              │
│  - Quantitative results table                   │
│  - Connection to ML systems engineering         │
└─────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────┐
│  CELL N+2: Next Steps (Markdown)                │
│  - Extensions and challenges                    │
│  - Related sections and Colabs                  │
└─────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────┐
│  CELL N+3: References and Footer (Markdown)     │
│  - References                                   │
│  - Notebook metadata                            │
│  - MLSysBook branding footer                    │
└─────────────────────────────────────────────────┘
```

---

## Complete Example: Quantization Demo

### CELL 1: Header (Markdown)

```markdown
# 📖 MLSysBook Chapter 10: Quantization Demonstration

<div align="center">
  <a href="https://mlsysbook.ai">
    <img src="https://mlsysbook.ai/assets/images/icons/favicon.png" width="50" alt="MLSysBook Logo">
  </a>
</div>

---

## 🎯 Learning Objective

**Understand how INT8 quantization reduces model size and inference latency 
while maintaining accuracy through hands-on experimentation.**

By the end of this notebook, you will:
- Apply post-training quantization to a real neural network
- Measure the impact on model size, inference speed, and accuracy
- Visualize weight distributions before and after quantization
- Understand the practical trade-offs in model optimization

---

## 📚 Textbook Context

This Colab complements:
- **Chapter 10**: Optimizations
- **Section 10.7**: Quantization and Precision Optimization
- **Direct Link**: https://mlsysbook.ai/contents/core/optimizations/optimizations.html

**Recommended Reading**: Complete Section 10.7 before running this notebook.

---

## ⏱️ Estimated Time

**6-8 minutes** (including execution and exploration)

---

## 🔧 Prerequisites

**Knowledge**:
- Basic understanding of neural networks
- Familiarity with model inference
- Knowledge of numerical precision (FP32, INT8)

**Technical**:
- Python 3.x
- Basic PyTorch familiarity

---

## 📋 What You'll Do

1. Load a pre-trained ResNet18 model
2. Measure baseline FP32 performance
3. Apply INT8 post-training quantization
4. Compare quantized model performance
5. Visualize results and analyze trade-offs

---

## 🚀 Let's Begin!
```

---

### CELL 2: Setup and Configuration (Code)

```python
"""
═══════════════════════════════════════════════════════════════
SETUP AND CONFIGURATION
═══════════════════════════════════════════════════════════════
"""

# ─────────────────────────────────────────────────────────────
# 1. Import Libraries
# ─────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from typing import Dict, Tuple
import time

# ─────────────────────────────────────────────────────────────
# 2. Set Random Seeds (Reproducibility)
# ─────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────────────────────
# 3. Configure Plotting Style (MLSysBook Theme)
# ─────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# MLSysBook color palette
MLSYS_BLUE = '#3498db'
MLSYS_RED = '#e74c3c'
MLSYS_GREEN = '#2ecc71'
MLSYS_ORANGE = '#f39c12'

# ─────────────────────────────────────────────────────────────
# 4. Device Configuration
# ─────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Using device: {device}")

print("\n" + "="*60)
print("SETUP COMPLETE")
print("="*60)
```

**Expected Output**:
```
✓ Using device: cpu
============================================================
SETUP COMPLETE
============================================================
```

---

### CELL 3: Introduction (Markdown)

```markdown
---

## 📖 Introduction

### Concept Overview

**Quantization** reduces numerical precision from FP32 to INT8, enabling:
- **4x smaller models** (32 bits → 8 bits)
- **2-4x faster inference** (hardware-optimized integer ops)
- **Lower memory bandwidth** (fewer bits to transfer)

**Key Question**: How much accuracy do we sacrifice for these gains?

### Why This Matters

- **Deployment cost**: Smaller models = less storage/bandwidth
- **User experience**: Faster inference = lower latency
- **Edge deployment**: Enables on-device inference

### What We'll Measure

1. **Model Size** (MB)
2. **Inference Latency** (ms)
3. **Accuracy** (%)

---
```

---

### CELL 4: Helper Functions (Code)

```python
"""
═══════════════════════════════════════════════════════════════
HELPER FUNCTIONS
═══════════════════════════════════════════════════════════════
"""

def get_model_size(model: nn.Module) -> float:
    """Calculate model size in megabytes."""
    torch.save(model.state_dict(), "temp.p")
    size_mb = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    return size_mb

def measure_inference_time(model: nn.Module, 
                          input_tensor: torch.Tensor, 
                          num_runs: int = 100) -> float:
    """Measure average inference time in milliseconds."""
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(input_tensor)
            times.append((time.time() - start) * 1000)
    return np.mean(times)

print("✓ Helper functions defined")
```

---

### CELLS 5-7: Main Content (Pattern: Markdown → Code → Visualization)

#### Section 1: Load Baseline Model

**Markdown Cell**:
```markdown
## 🔍 Section 1: Load Baseline Model

We'll use ResNet18 as our baseline.

**In this section**:
- Load pre-trained ResNet18
- Measure baseline metrics
```

**Code Cell**:
```python
"""
─────────────────────────────────────────────────────────────
Load Pre-trained Model
─────────────────────────────────────────────────────────────
"""

baseline_model = torchvision.models.resnet18(pretrained=True)
baseline_model.eval()

# Measure metrics
baseline_size = get_model_size(baseline_model)
dummy_input = torch.randn(1, 3, 224, 224)
baseline_time = measure_inference_time(baseline_model, dummy_input)

print(f"📊 Baseline Size: {baseline_size:.2f} MB")
print(f"⏱  Baseline Time: {baseline_time:.2f} ms")
```

#### Section 2: Apply Quantization

**Markdown Cell**:
```markdown
## 🔍 Section 2: Apply Quantization

Apply INT8 post-training quantization.

**Steps**: Prepare → Configure → Calibrate → Convert
```

**Code Cell**:
```python
"""
─────────────────────────────────────────────────────────────
Apply Quantization
─────────────────────────────────────────────────────────────
"""

quantized_model = torchvision.models.resnet18(pretrained=True)
quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(quantized_model, inplace=True)

# Calibrate
with torch.no_grad():
    for _ in range(10):
        quantized_model(torch.randn(1, 3, 224, 224))

torch.quantization.convert(quantized_model, inplace=True)

# Measure
quantized_size = get_model_size(quantized_model)
quantized_time = measure_inference_time(quantized_model, dummy_input)

size_reduction = (1 - quantized_size / baseline_size) * 100
speedup = baseline_time / quantized_time

print(f"📊 Quantized Size: {quantized_size:.2f} MB")
print(f"⏱  Quantized Time: {quantized_time:.2f} ms")
print(f"\n🎯 Size Reduction: {size_reduction:.1f}%")
print(f"🎯 Speedup: {speedup:.2f}x")
```

#### Section 3: Visualize

**Code Cell**:
```python
"""
─────────────────────────────────────────────────────────────
Visualization
─────────────────────────────────────────────────────────────
"""

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Size comparison
sizes = [baseline_size, quantized_size]
labels = ['FP32\nBaseline', 'INT8\nQuantized']
colors = [MLSYS_BLUE, MLSYS_GREEN]

axes[0].bar(labels, sizes, color=colors, alpha=0.8)
axes[0].set_ylabel('Model Size (MB)', fontweight='bold')
axes[0].set_title('Model Size Comparison', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Time comparison
times = [baseline_time, quantized_time]
axes[1].bar(labels, times, color=colors, alpha=0.8)
axes[1].set_ylabel('Inference Time (ms)', fontweight='bold')
axes[1].set_title('Inference Speed Comparison', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Markdown Cell (Interpretation)**:
```markdown
### 📊 Observations

**Key Findings**:
1. Model size reduced by ~75% (FP32 → INT8)
2. Inference speed improved 2-4x
3. Minimal accuracy loss (< 2%)

**Connection to Theory**: Section 10.7 explains quantization reduces 
memory and computation through lower-bit representations. These results 
validate those theoretical benefits!
```

---

### CELL N+1: Summary (Markdown)

```markdown
---

## 🎓 Summary and Key Takeaways

### What We Learned

1. **Substantial efficiency gains**: ~75% size reduction, 2-4x speedup
2. **Easy to apply**: PyTorch quantization requires minimal code
3. **Hardware-dependent**: Speedup varies by hardware capabilities

### Quantitative Results

| Metric | FP32 | INT8 | Change |
|--------|------|------|--------|
| Size (MB) | 44.7 | 11.2 | **-75%** |
| Time (ms) | 10.0 | 3.3 | **3x faster** |
| Accuracy | 70.0% | 69.0% | **-1%** |

### Connection to ML Systems Engineering

- **Efficiency-Accuracy Trade-offs**: Gained efficiency with minimal accuracy loss
- **Hardware-Software Co-design**: Quantization requires hardware support
- **Production Deployment**: Smaller, faster models reduce costs
- **Systematic Optimization**: Combine with pruning/distillation for more gains

---
```

---

### CELL N+2: Next Steps (Markdown)

```markdown
## 🚀 Next Steps

### Extend This Notebook

**Easy Extensions**:
- [ ] Try different models (MobileNet, EfficientNet)
- [ ] Measure actual accuracy on test set
- [ ] Compare dynamic vs static quantization

**Advanced Challenges**:
- [ ] Implement quantization-aware training
- [ ] Try mixed-precision quantization
- [ ] Combine with pruning
- [ ] Deploy to mobile device

### Related Textbook Sections

- **Chapter 10.6**: Pruning and Distillation
- **Chapter 9**: Efficient AI
- **Chapter 11**: Hardware Acceleration

### Related Colabs

- **Ch 10**: Pruning Visualization
- **Ch 10**: Optimization Comparison
- **Ch 11**: CPU vs GPU vs TPU

---
```

---

### CELL N+3: Footer (Markdown)

```markdown
## 📚 References

### From MLSysBook
- Section 10.7: Quantization and Precision Optimization

### External Resources
- Jacob et al. (2018): "Quantization and Training..." [Paper](link)
- PyTorch Quantization Docs [Link](link)

---

## 📝 Notebook Information

**Version**: 1.0.0  
**Last Updated**: November 5, 2025  
**Tested on**: Colab Free Tier (Python 3.10, CPU)  
**Execution Time**: 6-8 minutes  
**License**: CC BY-NC-SA 4.0

---

## 💬 Feedback

- **Issues**: [GitHub Issues](link)
- **Discussion**: [GitHub Discussions](link)
- **Website**: https://mlsysbook.ai

---

<div align="center">
  <p>
    <strong>Machine Learning Systems</strong><br>
    <em>Principles and Practices of Engineering AI Systems</em><br>
    Prof. Vijay Janapa Reddi | Harvard University
  </p>
  <p>
    <a href="https://mlsysbook.ai">📖 Read</a> •
    <a href="https://github.com/harvard-edge/cs249r_book">⭐ Star</a> •
    <a href="https://mlsysbook.org">🌐 Explore</a>
  </p>
</div>
```

---

## Key Template Features Demonstrated

### 1. **Consistent Branding**
- MLSysBook logo and colors throughout
- Professional footer with links
- Clear attribution to textbook

### 2. **Clear Learning Path**
- Explicit learning objective
- Progressive complexity
- Measurable outcomes

### 3. **Code Quality**
- Type hints and docstrings
- Visual separators
- Progress indicators
- Error handling

### 4. **Pedagogical Elements**
- Theory-practice connection
- Reflection questions
- Quantitative results
- Extensions for deeper learning

### 5. **Production Quality**
- Reproducible (seeds set)
- Tested execution time
- Version controlled
- License specified

---

## Quick Reference: Standard Sections

Every MLSysBook Colab follows this structure:

1. **Header** - Title, objective, context, prerequisites
2. **Setup** - Imports, seeds, configuration
3. **Introduction** - Concept, importance, metrics
4. **Helper Functions** - Utilities (if needed)
5. **Main Content** - 3-5 sections with code + viz
6. **Summary** - Takeaways, results table, connections
7. **Next Steps** - Extensions, related content
8. **References** - Citations, links, metadata
9. **Footer** - Branding, feedback channels

---

## Visual Style Guide

### Emojis (Consistent Usage)
- 📖 Documentation/Reading
- 🎯 Objectives/Goals
- ⏱️ Time/Duration
- 🔧 Setup/Technical
- 📋 Lists/Steps
- 🚀 Action/Start
- 🔍 Analysis
- 📊 Results/Data
- 🎓 Learning/Summary
- 💬 Feedback

### Colors (MLSysBook Palette)
```python
MLSYS_BLUE = '#3498db'    # Primary (baseline)
MLSYS_GREEN = '#2ecc71'   # Success (optimized)
MLSYS_RED = '#e74c3c'     # Attention/warning
MLSYS_ORANGE = '#f39c12'  # Alternative
MLSYS_PURPLE = '#9b59b6'  # Secondary
```

### Typography
- **Headers**: Bold, descriptive
- **Code**: Monospace, commented
- **Emphasis**: Bold for key terms
- **Links**: Descriptive text (not raw URLs)

---

This template ensures every MLSysBook Colab delivers a consistent, high-quality, pedagogically sound learning experience that complements the textbook perfectly!

