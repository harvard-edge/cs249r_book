# Getting Started with TinyTorch

```{warning} 🔬 Early Explorer Territory

You're ahead of the curve. TinyTorch is functional but still being refined. Expect rough edges, incomplete documentation, and things that might change. If you proceed, you're helping us shape this by finding what works and what doesn't.

**Best approach right now:** Browse the code and concepts. For hands-on building, check back when we announce classroom readiness (Summer/Fall 2026).

Questions or feedback? [Join the discussion →](https://github.com/harvard-edge/cs249r_book/discussions/1076)
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
##  For Students: Build Your ML Framework

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
tito update    # Check for and install updates (your work is preserved)
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
tito module complete 01  # Export Module 01 (Tensor)
```

After export, your code becomes importable:
```python
from tinytorch.core.tensor import Tensor  # YOUR implementation!
```

#### Step 3: Validate with Milestones

Run milestone scripts to prove your implementation works:

```bash
cd milestones/01_1957_perceptron
python 01_rosenblatt_forward.py  # Uses YOUR Tensor (M01)
python 02_rosenblatt_trained.py  # Uses YOUR implementation (M01-M07)
```

Each milestone has a README explaining:
- Required modules
- Historical context
- Expected results
- What you're learning

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
x = Tensor([1, 2, 3])  # YOUR implementation!
```

**What you'll implement:**
- N-dimensional array creation
- Mathematical operations (add, multiply, matmul)
- Shape manipulation (reshape, transpose)
- Memory layout understanding

### Module Progression

TinyTorch has 20 modules organized in progressive tiers:

- **Foundation (01-07)**: Core ML infrastructure - tensors, autograd, training
- **Architecture (08-13)**: Neural architectures - data loading, CNNs, transformers
- **Optimization (14-19)**: Production optimization - profiling, quantization, benchmarking
- **Capstone (20)**: Torch Olympics Competition

**See [Complete Course Structure](chapters/00-introduction.md)** for detailed module descriptions.

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


<a id="instructors"></a>
##  For Instructors & TAs: Classroom Support Coming Soon

<div style="background: #fff3cd; border: 1px solid #ffc107; padding: 1.5rem; border-radius: 0.5rem; margin: 1.5rem 0;">
<h4 style="margin: 0 0 0.5rem 0; color: #856404;">📢 Stay Tuned: NBGrader Integration In Development</h4>
<p style="margin: 0 0 1rem 0; color: #856404;">We're building comprehensive classroom support with NBGrader integration that will enable:</p>

<ul style="margin: 0; color: #664d03; padding-left: 1.5rem;">
<li><strong>Automated Assignment Generation</strong> - Create student assignments from TinyTorch modules with solutions removed</li>
<li><strong>Auto-Grading</strong> - Automatically grade student implementations against test suites</li>
<li><strong>Manual Review Interface</strong> - Grade ML Systems Thinking questions through a browser-based interface</li>
<li><strong>Progress Tracking</strong> - Monitor student progress across all 20 modules</li>
<li><strong>Grade Export</strong> - Export grades to CSV for LMS integration</li>
</ul>
</div>

### What's Planned

**Course Structure:**
- 14-16 week curriculum covering all 20 modules
- Progressive difficulty from tensors to transformers to optimization
- Historical milestones that validate student implementations
- Capstone competition (Torch Olympics)

**Grading Components:**
- **70% Auto-Graded**: Code implementation correctness via NBGrader test cells
- **30% Manual Review**: ML Systems Thinking questions (3 per module)

**Assessment Tools:**
- `tito grade generate` - Create instructor versions with solutions
- `tito grade release` - Generate student versions (solutions removed)
- `tito grade collect` - Collect student submissions
- `tito grade autograde` - Run automatic grading
- `tito grade feedback` - Generate student feedback
- `tito grade export` - Export grades to CSV

### Current Status

TinyTorch is fully functional for **self-paced learning** today. Students can:
- Work through all 20 modules independently
- Run milestone validation scripts
- Use the complete `tito` CLI for module management
- Join the community and run benchmarks

**For classroom deployment**, we recommend waiting for the official NBGrader integration announcement (target: Summer/Fall 2026).

### Interested in Early Adoption?

If you're considering using TinyTorch in your course before full classroom support is ready:

1. **Review the curriculum** - Browse modules and milestones to assess fit
2. **Test the workflow** - Complete a few modules yourself to understand the student experience
3. **Contact us** - [Join the discussion](https://github.com/harvard-edge/cs249r_book/discussions/1076) to share your use case

We're actively seeking instructor feedback to shape the classroom experience.

### Stay Updated

- **[GitHub Discussions](https://github.com/harvard-edge/cs249r_book/discussions/1076)** - Join the conversation
- **[Course Structure Overview](chapters/00-introduction.md)** - Full curriculum details
- **[Module Documentation](tito/modules.md)** - Technical module specifications


## Additional Resources

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin: 2rem 0;">

<div style="background: #f0f9ff; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #3b82f6;">
<h4 style="margin: 0 0 0.5rem 0; color: #1e40af;">📚 Course Documentation</h4>
<ul style="margin: 0.5rem 0; padding-left: 1.25rem; font-size: 0.9rem;">
<li><a href="chapters/00-introduction.html">Complete Course Structure</a></li>
<li><a href="chapters/milestones.html">Historical Milestones</a></li>
<li><a href="prerequisites.html">Prerequisites & Resources</a></li>
<li><a href="faq.html">Frequently Asked Questions</a></li>
</ul>
</div>

<div style="background: #f0fdf4; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #22c55e;">
<h4 style="margin: 0 0 0.5rem 0; color: #166534;">🛠 CLI & Tools</h4>
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
