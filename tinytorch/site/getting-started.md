# Getting Started with TinyTorch

```{warning} Early Explorer Territory

You're ahead of the curve. TinyTorch is functional but still being refined. Expect rough edges, incomplete documentation, and things that might change. If you proceed, you're helping us shape this by finding what works and what doesn't.

**Best approach right now:** Browse the code and concepts. For hands-on building, check back when we announce classroom readiness (Summer/Fall 2026).

Questions or feedback? [Join the discussion →](https://github.com/harvard-edge/cs249r_book/discussions/1076)
```

```{note} Prerequisites Check
This guide requires **Python programming** (classes, functions, NumPy basics) and **basic linear algebra** (matrix multiplication). Not sure if you're ready? Take the [Prerequisites Self-Assessment](prerequisites.md) first.
```

Welcome to TinyTorch! This comprehensive guide will get you started whether you're a student building ML systems, an instructor setting up a course, or a TA supporting learners.

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h2 style="margin: 0 0 1rem 0; color: #495057;">Choose Your Path</h2>
<p style="margin: 0 0 1.5rem 0; color: #6c757d;">Jump directly to your role-specific guide</p>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; max-width: 800px; margin: 0 auto;">

<a href="#students" style="display: block; background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #1976d2; text-decoration: none; transition: transform 0.2s;">
<div style="font-size: 2rem; margin-bottom: 0.5rem;"></div>
<div style="color: #0d47a1; font-weight: 600; font-size: 1.1rem;">Students</div>
<div style="color: #1565c0; font-size: 0.85rem; margin-top: 0.5rem;">Setup + Build Workflow</div>
</a>

<a href="#instructors" style="display: block; background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #7b1fa2; text-decoration: none; transition: transform 0.2s;">
<div style="font-size: 2rem; margin-bottom: 0.5rem;"></div>
<div style="color: #4a148c; font-weight: 600; font-size: 1.1rem;">Instructors & TAs</div>
<div style="color: #6a1b9a; font-size: 0.85rem; margin-top: 0.5rem;">Coming Soon</div>
</a>

</div>
</div>


<a id="students"></a>
## For Students: Build Your ML Framework

### Quick Setup (2 Minutes)

Get your development environment ready to build ML systems from scratch:

```bash
# One-line install (run from a project folder like ~/projects)
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

**Keeping up to date:**
```bash
tito update # Check for and install updates (your work is preserved)
```

### Join the Community (Optional)

After setup, join the global TinyTorch community and validate your installation:

```bash
# Log in to join the community
tito community login

# Run baseline benchmark to validate setup
tito benchmark baseline
```

All community data is stored locally in `.tinytorch/` directory. See **[Community Guide](community.md)** for complete features.

### The TinyTorch Build Cycle

TinyTorch follows a simple three-step workflow that you'll repeat for each module:

```{mermaid}
:align: center
:caption: Architecture Overview
graph LR
 A[1. Edit Module<br/>modules/NN_name.ipynb] --> B[2. Export to Package<br/>tito module complete N]
 B --> C[3. Validate with Milestones<br/>Run milestone scripts]
 C --> A

 style A fill:#fffbeb
 style B fill:#f0fdf4
 style C fill:#fef3c7
```

#### Step 1: Edit Modules

Work on module notebooks interactively:

```bash
# Example: Working on Module 01 (Tensor)
cd modules/01_tensor
jupyter lab 01_tensor.ipynb
```

Each module is a Jupyter notebook where you'll:
- Implement the required functionality from scratch
- Add docstrings and comments
- Run and test your code inline
- See immediate feedback

#### Step 2: Export to Package

Once your implementation is complete, export it to the main TinyTorch package:

```bash
tito module complete MODULE_NUMBER

# Example:
tito module complete 01 # Export Module 01 (Tensor)
```

After export, your code becomes importable:
```python
from tinytorch.core.tensor import Tensor # YOUR implementation!
```

#### Step 3: Validate with Milestones

Run milestone scripts to prove your implementation works:

```bash
tito milestone run perceptron  # Uses YOUR Tensor, Activations, Layers
```

Each milestone validates that your modules work together correctly. Use `tito milestone list` to see all available milestones and their required modules.

**What if validation fails?** If a milestone script produces errors:
1. Read the error message carefully—it usually points to the problem
2. Run module tests: `tito module test 01` to check your implementation
3. Return to your Jupyter notebook to debug and fix
4. Re-export with `tito module complete 01` and try again

**See [Historical Milestones](chapters/milestones.md)** for the complete progression through ML history.

### Your First Module (15 Minutes)

Start with Module 01 to build tensor operations - the foundation of all neural networks:

```bash
# Step 1: Edit the module
cd modules/01_tensor
jupyter lab 01_tensor.ipynb

# Step 2: Export when ready
tito module complete 01

# Step 3: Validate
from tinytorch.core.tensor import Tensor
x = Tensor([1, 2, 3]) # YOUR implementation!
```

**What you'll implement:**
- N-dimensional array creation
- Mathematical operations (add, multiply, matmul)
- Shape manipulation (reshape, transpose)
- Memory layout understanding

### Module Progression

TinyTorch has 20 modules organized in progressive tiers:

| Tier | Modules | Focus | Time Estimate |
|------|---------|-------|---------------|
| **Foundation** | 01-07 | Core ML infrastructure (tensors, autograd, training) | ~15-20 hours |
| **Architecture** | 08-13 | Neural architectures (data loading, CNNs, transformers) | ~18-24 hours |
| **Optimization** | 14-19 | Production optimization (profiling, quantization) | ~18-24 hours |
| **Capstone** | 20 | Torch Olympics Competition | ~8-10 hours |

**Total: ~60-80 hours** over 14-18 weeks (4-6 hours/week pace).

**See [Foundation Tier Overview](tiers/foundation.md)** for detailed module descriptions, or [Learning Journey](chapters/learning-journey.md) for the complete pedagogical narrative.

### Essential Commands Reference

The most important commands you'll use daily:

```bash
# Export module to package
tito module complete MODULE_NUMBER

# Check module status
tito module status

# System information
tito system info

# Community features
tito community login
tito benchmark baseline
```

**See [TITO CLI Reference](tito/overview.md)** for complete command documentation.

### Notebook Platform Options

**For Viewing & Exploration (Online):**
- Jupyter/MyBinder: Click "Launch Binder" on any notebook page
- Google Colab: Click "Launch Colab" for GPU access
- Marimo: Click "~ Open in Marimo" for reactive notebooks

**For Full Development (Local - Required):**

To actually build the framework, you need local installation:
- Full `tinytorch.*` package available
- Run milestone validation scripts
- Use `tito` CLI commands
- Execute complete experiments
- Export modules to package

**Note for NBGrader assignments**: Submit `.ipynb` files to preserve grading metadata.

### What's Next?

1. **Continue Building**: Follow the module progression (01 → 02 → 03...)
2. **Run Milestones**: Prove your implementations work with real ML history
3. **Build Intuition**: Understand ML systems from first principles

The goal isn't just to write code - it's to **understand** how modern ML frameworks work by building one yourself.


## For Instructors & TAs: Classroom Support Coming Soon

```{note}
We're building comprehensive classroom support with NBGrader integration. For hands-on building today, TinyTorch is fully functional for self-paced learning.
```

**What's Planned:**
- Automated assignment generation with solutions removed
- Auto-grading against test suites
- Manual review interface for ML Systems Thinking questions
- Progress tracking across all 20 modules
- Grade export to CSV for LMS integration

**Current Status:** TinyTorch works for self-paced learning today. For classroom deployment, we recommend waiting for the official NBGrader integration (target: Summer/Fall 2026).

**Interested in early adoption?** [Join the discussion](https://github.com/harvard-edge/cs249r_book/discussions/1076) to share your use case.

See the **[Instructor Guide](instructor-guide.md)** for detailed setup instructions and grading rubrics.


## Additional Resources

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin: 2rem 0;">

<div style="background: #f0f9ff; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #3b82f6;">
<h4 style="margin: 0 0 0.5rem 0; color: #1e40af;"> Course Documentation</h4>
<ul style="margin: 0.5rem 0; padding-left: 1.25rem; font-size: 0.9rem;">
<li><a href="chapters/learning-journey.html">Learning Journey</a></li>
<li><a href="chapters/milestones.html">Historical Milestones</a></li>
<li><a href="prerequisites.html">Prerequisites & Resources</a></li>
<li><a href="faq.html">Frequently Asked Questions</a></li>
</ul>
</div>

<div style="background: #f0fdf4; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #22c55e;">
<h4 style="margin: 0 0 0.5rem 0; color: #166534;"> CLI & Tools</h4>
<ul style="margin: 0.5rem 0; padding-left: 1.25rem; font-size: 0.9rem;">
<li><a href="tito/overview.html">TITO CLI Overview</a></li>
<li><a href="tito/modules.html">Module Workflow</a></li>
<li><a href="tito/milestones.html">Milestone System</a></li>
<li><a href="tito/troubleshooting.html">Troubleshooting</a></li>
</ul>
</div>

<div style="background: #fef3c7; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #eab308;">
<h4 style="margin: 0 0 0.5rem 0; color: #a16207;"> Community</h4>
<ul style="margin: 0.5rem 0; padding-left: 1.25rem; font-size: 0.9rem;">
<li><a href="community.html">Community Ecosystem</a></li>
<li><a href="resources.html">Learning Resources</a></li>
<li><a href="credits.html">Credits & Acknowledgments</a></li>
<li><a href="https://github.com/mlsysbook/TinyTorch/discussions">GitHub Discussions</a></li>
</ul>
</div>

</div>


**Ready to start building?** Choose your path above and dive into the most comprehensive ML systems course available!
