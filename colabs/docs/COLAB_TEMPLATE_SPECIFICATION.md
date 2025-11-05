# MLSysBook Colab Template Specification

This document defines the standardized structure, style, and conventions for all Google Colab notebooks affiliated with MLSysBook.

---

## Design Principles

1. **Pedagogical Clarity**: Each section has a clear learning purpose
2. **Self-Contained**: Works standalone but references textbook
3. **Minimal Dependencies**: Use standard ML stack, avoid exotic libraries
4. **Fast Execution**: Complete in 5-10 minutes on free Colab tier
5. **Visual Output**: Prioritize plots, tables, comparisons
6. **Progressive Complexity**: Start simple, build up
7. **Reproducibility**: Pin versions, set random seeds
8. **Professional Quality**: Publication-ready code and outputs

---

## Standard Colab Structure

### 1. Header Section (Markdown Cell)
```markdown
# 📖 MLSysBook Chapter [X]: [Colab Title]

<div align="center">
  <a href="https://mlsysbook.ai">
    <img src="https://mlsysbook.ai/assets/images/icons/favicon.png" width="50" alt="MLSysBook Logo">
  </a>
</div>

---

## 🎯 Learning Objective

[One clear, measurable learning objective]

**Example**: "Understand how INT8 quantization reduces model size and inference latency while maintaining accuracy"

---

## 📚 Textbook Context

This Colab complements:
- **Chapter [X]**: [Chapter Title]
- **Section**: [Section Title] 
- **Direct Link**: [https://mlsysbook.ai/contents/core/chapter/section.html]

**Recommended Reading**: Complete [Section Name] before running this notebook.

---

## ⏱️ Estimated Time

**[5-10] minutes** (including execution and exploration)

---

## 🔧 Prerequisites

**Knowledge**:
- [List required concepts from textbook]

**Technical**:
- Python 3.x
- Basic familiarity with [PyTorch/TensorFlow/etc.]

---

## 📋 What You'll Do

1. [High-level step 1]
2. [High-level step 2]
3. [High-level step 3]
4. [High-level step 4]

---

## 🚀 Let's Begin!
```

### 2. Setup and Configuration (Code Cell)
```python
"""
═══════════════════════════════════════════════════════════════
SETUP AND CONFIGURATION
═══════════════════════════════════════════════════════════════
Install dependencies and configure environment for reproducibility.
"""

# ─────────────────────────────────────────────────────────────
# 1. Install Dependencies (if needed)
# ─────────────────────────────────────────────────────────────
# Uncomment if additional packages needed beyond Colab defaults

# !pip install -q package_name==version

# ─────────────────────────────────────────────────────────────
# 2. Import Libraries
# ─────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from typing import Dict, List, Tuple

# Standard imports organized by category:
# - Standard library
# - Third-party scientific
# - ML frameworks
# - Utilities

# ─────────────────────────────────────────────────────────────
# 3. Set Random Seeds (Reproducibility)
# ─────────────────────────────────────────────────────────────
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ─────────────────────────────────────────────────────────────
# 4. Configure Plotting Style
# ─────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# ─────────────────────────────────────────────────────────────
# 5. Device Configuration
# ─────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Using device: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print("\n" + "="*60)
print("SETUP COMPLETE")
print("="*60)
```

### 3. Introduction Section (Markdown Cell)
```markdown
---

## 📖 Introduction

### Concept Overview

[2-3 paragraphs explaining the concept being demonstrated]

**Key Question**: [Pose a question this Colab answers]

### Why This Matters

[Practical importance and real-world relevance]

### What We'll Measure

[Specific metrics or observations we'll make]

---
```

### 4. Helper Functions (Code Cell - if needed)
```python
"""
═══════════════════════════════════════════════════════════════
HELPER FUNCTIONS
═══════════════════════════════════════════════════════════════
Utility functions used throughout the notebook.
"""

def helper_function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of what this function does.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Example:
        >>> result = helper_function_name(value1, value2)
    """
    # Implementation
    pass

# Add any other helper functions with clear docstrings
```

### 5. Main Content Sections (Multiple Cells)

Each main section follows this pattern:

**Section Header (Markdown)**:
```markdown
---

## 🔍 Section [N]: [Section Title]

[Brief introduction to what this section demonstrates]

**In this section**:
- [Bullet 1]
- [Bullet 2]
- [Bullet 3]
```

**Implementation (Code)**:
```python
"""
─────────────────────────────────────────────────────────────
Section [N]: [Title]
─────────────────────────────────────────────────────────────
"""

# Step 1: [Description]
# ─────────────────────────────────────────────────────────────
# Clear comments explaining what's happening

code_here()

# Step 2: [Description]
# ─────────────────────────────────────────────────────────────

more_code()

print(f"✓ Step completed: [brief result]")
```

**Visualization (Code)**:
```python
"""
─────────────────────────────────────────────────────────────
Visualization: [What we're visualizing]
─────────────────────────────────────────────────────────────
"""

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1
axes[0].plot(data)
axes[0].set_title('Title')
axes[0].set_xlabel('X Label')
axes[0].set_ylabel('Y Label')
axes[0].grid(True, alpha=0.3)

# Plot 2
axes[1].plot(other_data)
axes[1].set_title('Title')
axes[1].set_xlabel('X Label')
axes[1].set_ylabel('Y Label')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Interpretation (Markdown)**:
```markdown
### 📊 Observations

**Key Findings**:
1. [Finding 1 with specific numbers]
2. [Finding 2 with specific numbers]
3. [Finding 3 with specific numbers]

**Connection to Theory**: [How these results connect to textbook concepts]
```

### 6. Interactive Exploration (Optional - Code Cell)
```python
"""
═══════════════════════════════════════════════════════════════
INTERACTIVE EXPLORATION
═══════════════════════════════════════════════════════════════
Modify parameters below and re-run to explore different scenarios.
"""

# ─────────────────────────────────────────────────────────────
# 🎮 EXPERIMENT: Try changing these parameters!
# ─────────────────────────────────────────────────────────────

PARAM_1 = 0.01  # Try: 0.001, 0.01, 0.1
PARAM_2 = 32    # Try: 16, 32, 64, 128

# [Run experiment with parameters]

print("✓ Experiment complete! Try different values above.")
```

### 7. Summary and Key Takeaways (Markdown Cell)
```markdown
---

## 🎓 Summary and Key Takeaways

### What We Learned

1. **[Key Insight 1]**: [Explanation with specific evidence from notebook]

2. **[Key Insight 2]**: [Explanation with specific evidence from notebook]

3. **[Key Insight 3]**: [Explanation with specific evidence from notebook]

### Quantitative Results

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| [Metric 1] | [Value] | [Value] | [±X%] |
| [Metric 2] | [Value] | [Value] | [±X%] |
| [Metric 3] | [Value] | [Value] | [±X%] |

### Connection to ML Systems Engineering

[How this connects to broader ML systems principles from the textbook]

---
```

### 8. Further Exploration (Markdown Cell)
```markdown
## 🚀 Next Steps

### Extend This Notebook

**Easy Extensions**:
- [ ] [Suggestion 1]
- [ ] [Suggestion 2]

**Advanced Challenges**:
- [ ] [Challenge 1]
- [ ] [Challenge 2]

### Related Textbook Sections

- **[Chapter X.Y]**: [Section Title] - [Why relevant]
- **[Chapter Z.W]**: [Section Title] - [Why relevant]

### Related Colabs

- **[Chapter X]**: [Colab Title] - [Connection]
- **[Chapter Y]**: [Colab Title] - [Connection]

---
```

### 9. References and Resources (Markdown Cell)
```markdown
## 📚 References and Resources

### From MLSysBook

- [Specific section references with links]

### External Resources

- [Paper 1]: [Citation and link]
- [Tutorial 1]: [Link]
- [Documentation]: [Link]

### Implementation References

- [Library/Framework]: [Version used and documentation link]

---
```

### 10. Footer (Markdown Cell)
```markdown
---

## 📝 Notebook Information

**Version**: 1.0.0  
**Last Updated**: [Date]  
**Tested on**: Colab Free Tier (Runtime: Python 3.10, GPU: T4)  
**Estimated Execution Time**: [X] minutes  
**License**: CC BY-NC-SA 4.0 (Same as MLSysBook)

---

## 💬 Feedback and Questions

Found an issue or have suggestions? 

- **Open an issue**: [Link to GitHub Issues]
- **Discussion**: [Link to GitHub Discussions]
- **Book website**: https://mlsysbook.ai

---

<div align="center">
  <p>
    <strong>Machine Learning Systems</strong><br>
    <em>Principles and Practices of Engineering Artificially Intelligent Systems</em><br>
    Prof. Vijay Janapa Reddi | Harvard University
  </p>
  <p>
    <a href="https://mlsysbook.ai">📖 Read the Book</a> •
    <a href="https://github.com/harvard-edge/cs249r_book">⭐ Star on GitHub</a> •
    <a href="https://mlsysbook.org">🌐 Explore Ecosystem</a>
  </p>
</div>
```

---

## Code Style Standards

### Python Code Conventions

1. **Follow PEP 8** with these specifics:
   - Line length: 88 characters (Black formatter standard)
   - Use type hints for function signatures
   - Use descriptive variable names

2. **Documentation**:
   - Every function has a docstring
   - Complex code sections have explanatory comments
   - Use visual separators for major sections

3. **Visual Separators**:
```python
# ═══════════════════════════════════════════════════════════════
# MAJOR SECTION (using double lines)
# ═══════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# Subsection (using single lines)
# ─────────────────────────────────────────────────────────────
```

4. **Print Statements**:
```python
print("=" * 60)
print("SECTION TITLE")
print("=" * 60)

print(f"✓ Task completed successfully")
print(f"⚠ Warning: [message]")
print(f"✗ Error: [message]")
print(f"📊 Results: {value}")
print(f"⏱ Time elapsed: {elapsed:.2f}s")
```

5. **Progress Indication**:
```python
from tqdm import tqdm

for i in tqdm(range(n), desc="Processing"):
    # work
    pass
```

### Output Formatting

1. **Tables** (use pandas for clean display):
```python
import pandas as pd

results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Size', 'Speed'],
    'Baseline': [0.95, 100, 10],
    'Optimized': [0.94, 25, 40],
    'Change': ['-1%', '-75%', '+300%']
})

print(results_df.to_string(index=False))
```

2. **Comparison Boxes**:
```python
print("\n" + "="*60)
print(" "*20 + "COMPARISON")
print("="*60)
print(f"{'Metric':<20} {'Before':<15} {'After':<15} {'Change'}")
print("-"*60)
print(f"{'Model Size':<20} {size_before:<15} {size_after:<15} {change}")
print("="*60 + "\n")
```

### Plotting Standards

1. **Consistent Style**:
```python
def create_comparison_plot(data1, data2, labels, title):
    """Standard plotting function for comparisons."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, data1, width, label='Baseline', 
           alpha=0.8, color='#3498db')
    ax.bar(x + width/2, data2, width, label='Optimized', 
           alpha=0.8, color='#e74c3c')
    
    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig
```

2. **Color Palette** (MLSysBook themed):
```python
# Primary colors
MLSYS_BLUE = '#3498db'      # Primary
MLSYS_RED = '#e74c3c'       # Attention
MLSYS_GREEN = '#2ecc71'     # Success
MLSYS_ORANGE = '#f39c12'    # Warning
MLSYS_PURPLE = '#9b59b6'    # Alternative

# Use these consistently across all Colabs
```

---

## Dependency Management

### Requirements Specification

Create a `requirements.txt` snippet for each Colab:

```python
# At the top of notebook in a markdown cell
"""
## 📦 Dependencies

This notebook requires:
- `torch>=2.0.0`
- `numpy>=1.24.0`
- `matplotlib>=3.7.0`
- `pandas>=2.0.0`
- `tqdm>=4.65.0`

All dependencies are pre-installed in Google Colab Free Tier.
Additional packages (if any) are installed in the setup cell.
"""
```

### Version Pinning Strategy

```python
# For pip installs, always pin versions
!pip install -q torch==2.0.1 torchvision==0.15.2

# Document versions used
print("Library Versions:")
print(f"  PyTorch: {torch.__version__}")
print(f"  NumPy: {np.__version__}")
```

---

## Testing and Validation

### Pre-Publication Checklist

Before finalizing any Colab, verify:

- [ ] Runs successfully on Colab Free Tier (GPU: T4)
- [ ] Completes in < 10 minutes
- [ ] All outputs are visible and clear
- [ ] No hardcoded paths (use Colab-safe paths)
- [ ] Random seeds set for reproducibility
- [ ] All links work (textbook, GitHub)
- [ ] Markdown renders correctly
- [ ] Code follows style standards
- [ ] Docstrings present for all functions
- [ ] Visualizations have proper labels, titles, legends
- [ ] Learning objective is achieved and measured
- [ ] Key takeaways summarize findings

### Runtime Testing

```python
# Include timing information
import time

start_time = time.time()

# ... main computation ...

elapsed_time = time.time() - start_time
print(f"\n⏱ Total execution time: {elapsed_time:.2f} seconds")

# Verify it's < 10 minutes
assert elapsed_time < 600, "Notebook execution exceeded 10 minutes!"
```

---

## File Naming Conventions

### Directory Structure
```
colabs/
├── ch03_dl_primer/
│   ├── gradient_descent_visualization.ipynb
│   ├── activation_function_explorer.ipynb
│   └── README.md
├── ch06_data_engineering/
│   ├── data_quality_impact.ipynb
│   └── README.md
└── ...
```

### Naming Pattern
```
[chapter_number]_[chapter_shortname]/[descriptive_name].ipynb

Examples:
- ch03_dl_primer/gradient_descent_visualization.ipynb
- ch10_optimizations/quantization_demo.ipynb
- ch17_responsible_ai/fairness_bias_detection.ipynb
```

**Rules**:
- Lowercase with underscores
- Chapter number with leading zero (ch03, not ch3)
- Descriptive but concise (3-5 words max)
- No spaces or special characters

---

## Markdown Styling

### Headers
```markdown
# 📖 Top Level (Title)
## 🎯 Section Headers (with emoji for visual interest)
### 📊 Subsections
#### Details (sparingly)
```

### Emoji Usage (Consistent Meanings)
- 📖 Book/Reading/Documentation
- 🎯 Learning Objective/Goal
- ⏱️ Time/Duration
- 🔧 Prerequisites/Setup
- 📋 Checklist/Steps
- 🚀 Action/Start/Next Steps
- 🔍 Analysis/Investigation
- 📊 Results/Data/Visualization
- 🎓 Summary/Key Takeaway
- 💬 Communication/Feedback
- ⚠️ Warning/Caution
- ✓ Success/Completion
- ✗ Error/Failure
- 🎮 Interactive/Experiment

### Call-out Boxes
```markdown
> **💡 Tip**: [Helpful tip for readers]

> **⚠️ Warning**: [Important caution]

> **📌 Note**: [Additional context]

> **🤔 Think About It**: [Reflection question]
```

---

## Accessibility Considerations

1. **Color Blindness**: Never rely on color alone to convey information
   - Use different line styles (solid, dashed, dotted)
   - Add labels or annotations
   - Include patterns in bars/areas

2. **Alt Text**: Provide descriptions for complex visualizations
```markdown
![Description of what the plot shows and key findings](image.png)
```

3. **Clear Language**: 
   - Avoid jargon without explanation
   - Define technical terms
   - Use concrete examples

---

## Version Control and Metadata

### Notebook Metadata
Include at the top of each notebook:

```python
NOTEBOOK_INFO = {
    'title': 'Quantization Demonstration',
    'chapter': 10,
    'version': '1.0.0',
    'created': '2025-01-01',
    'last_updated': '2025-01-15',
    'authors': ['MLSysBook Team'],
    'tested_on': 'Colab Free Tier, GPU: T4, Python 3.10',
    'license': 'CC BY-NC-SA 4.0'
}
```

### Versioning Scheme
- **Major.Minor.Patch** (e.g., 1.0.0)
- **Major**: Structural changes, different learning outcomes
- **Minor**: Content updates, new examples
- **Patch**: Bug fixes, typo corrections

---

## Example: Complete Mini-Colab

See `COLAB_TEMPLATE_EXAMPLE.ipynb` for a fully implemented example following all these standards.

---

## Maintenance Guidelines

### Regular Updates
- **Quarterly**: Test all Colabs on current Colab environment
- **As Needed**: Update when libraries release breaking changes
- **Continuous**: Fix reported issues

### Deprecation Handling
```python
# Handle deprecated APIs gracefully
try:
    # New API
    result = new_function()
except AttributeError:
    # Fallback for older versions
    import warnings
    warnings.warn("Using deprecated API, consider updating library")
    result = old_function()
```

### Issue Reporting
Encourage users to report:
- Execution failures
- Unclear explanations
- Broken links
- Timing issues (> 10 minutes)

---

## Summary Checklist

Every MLSysBook Colab must have:

### Structure
- [ ] Standard header with learning objective
- [ ] Setup cell with reproducibility settings
- [ ] Clear section progression
- [ ] Visualizations for key findings
- [ ] Summary with quantitative results
- [ ] Standard footer with metadata

### Code Quality
- [ ] Type hints for functions
- [ ] Docstrings for functions
- [ ] Visual separators for sections
- [ ] Progress indicators for long operations
- [ ] Timing information

### Pedagogy
- [ ] Clear learning objective (measurable)
- [ ] Connection to textbook section
- [ ] Progressive complexity
- [ ] Reflection questions
- [ ] Concrete takeaways with numbers

### Technical
- [ ] Runs in < 10 minutes
- [ ] Reproducible (seeds set)
- [ ] Dependencies pinned
- [ ] Works on free tier
- [ ] Error handling

### Documentation
- [ ] Links to textbook work
- [ ] Related Colabs listed
- [ ] Version information
- [ ] License specified

---

**This template ensures every MLSysBook Colab delivers a consistent, high-quality learning experience.**

