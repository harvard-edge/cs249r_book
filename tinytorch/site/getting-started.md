# Getting Started with TinyTorch

Welcome to TinyTorch! This comprehensive guide will get you started whether you're a student building ML systems, an instructor setting up a course, or a TA supporting learners.

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h2 style="margin: 0 0 1rem 0; color: #495057;">Choose Your Path</h2>
<p style="margin: 0 0 1.5rem 0; color: #6c757d;">Jump directly to your role-specific guide</p>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; max-width: 800px; margin: 0 auto;">

<a href="#students" style="display: block; background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #1976d2; text-decoration: none; transition: transform 0.2s;">
<div style="font-size: 2rem; margin-bottom: 0.5rem;">🎓</div>
<div style="color: #0d47a1; font-weight: 600; font-size: 1.1rem;">Students</div>
<div style="color: #1565c0; font-size: 0.85rem; margin-top: 0.5rem;">Setup + Build Workflow</div>
</a>

<a href="#instructors" style="display: block; background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #7b1fa2; text-decoration: none; transition: transform 0.2s;">
<div style="font-size: 2rem; margin-bottom: 0.5rem;">👨‍🏫</div>
<div style="color: #4a148c; font-weight: 600; font-size: 1.1rem;">Instructors</div>
<div style="color: #6a1b9a; font-size: 0.85rem; margin-top: 0.5rem;">Course Setup + Grading</div>
</a>

<a href="#tas" style="display: block; background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #f57c00; text-decoration: none; transition: transform 0.2s;">
<div style="font-size: 2rem; margin-bottom: 0.5rem;">👥</div>
<div style="color: #e65100; font-weight: 600; font-size: 1.1rem;">Teaching Assistants</div>
<div style="color: #ef6c00; font-size: 0.85rem; margin-top: 0.5rem;">Student Support + Debugging</div>
</a>

</div>
</div>

---

<a id="students"></a>
## 🎓 For Students: Build Your ML Framework

### Quick Setup (2 Minutes)

Get your development environment ready to build ML systems from scratch:

```bash
# Clone repository
git clone https://github.com/mlsysbook/TinyTorch.git
cd TinyTorch

# Automated setup (handles everything!)
./setup-environment.sh

# Activate environment
source activate.sh

# Verify setup
tito system health
```

**What this does:**
- Creates optimized virtual environment
- Installs all dependencies (NumPy, Jupyter, Rich, PyTorch for validation)
- Configures TinyTorch in development mode
- Verifies installation with system diagnostics

### Join the Community (Optional)

After setup, join the global TinyTorch community and validate your installation:

```bash
# Join with optional information
tito community join

# Run baseline benchmark to validate setup
tito benchmark baseline
```

All community data is stored locally in `.tinytorch/` directory. See **[Community Guide](community.md)** for complete features.

### The TinyTorch Build Cycle

TinyTorch follows a simple three-step workflow that you'll repeat for each module:

```{mermaid}
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

**📖 See [Historical Milestones](chapters/milestones.md)** for the complete progression through ML history.

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

**📖 See [Complete Course Structure](chapters/00-introduction.md)** for detailed module descriptions.

### Essential Commands Reference

The most important commands you'll use daily:

```bash
# Export module to package
tito module complete MODULE_NUMBER

# Check module status (optional)
tito checkpoint status

# System information
tito system info

# Community features
tito community join
tito benchmark baseline
```

**📖 See [TITO CLI Reference](tito/overview.md)** for complete command documentation.

### Notebook Platform Options

**For Viewing & Exploration (Online):**
- Jupyter/MyBinder: Click "Launch Binder" on any notebook page
- Google Colab: Click "Launch Colab" for GPU access
- Marimo: Click "🍃 Open in Marimo" for reactive notebooks

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

---

<a id="instructors"></a>
## 👨‍🏫 For Instructors: Turn-Key ML Systems Course

### Course Overview

TinyTorch provides a complete ML systems engineering course with NBGrader integration, automated grading, and production-ready teaching materials.

<div style="background: #d4edda; border: 1px solid #28a745; padding: 1.5rem; border-radius: 0.5rem; margin: 1.5rem 0;">
<h4 style="margin: 0 0 0.5rem 0; color: #155724;">✅ Complete NBGrader Integration Available</h4>
<p style="margin: 0; color: #155724;">TinyTorch includes automated grading workflows, rubrics, and sample solutions ready for classroom use.</p>
</div>

**Course Duration:** 14-16 weeks (flexible pacing)
**Student Outcome:** Complete ML framework supporting vision AND language models
**Teaching Approach:** Systems-focused learning through building, not just using

### 30-Minute Instructor Setup

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin: 2rem 0;">

<div style="background: white; padding: 1.5rem; border-radius: 0.5rem; border: 1px solid #dee2e6;">
<h4 style="color: #495057; margin: 0 0 0.5rem 0;">1️⃣ Clone & Setup (10 min)</h4>
<div style="background: #f8f9fa; padding: 1rem; border-radius: 0.25rem; font-family: monospace; font-size: 0.85rem; margin: 0.5rem 0;">
git clone TinyTorch<br>
cd TinyTorch<br>
python -m venv .venv<br>
source .venv/bin/activate<br>
pip install -r requirements.txt<br>
pip install nbgrader
</div>
<p style="font-size: 0.9rem; margin: 0; color: #6c757d;">One-time environment setup</p>
</div>

<div style="background: white; padding: 1.5rem; border-radius: 0.5rem; border: 1px solid #dee2e6;">
<h4 style="color: #495057; margin: 0 0 0.5rem 0;">2️⃣ Initialize Grading (10 min)</h4>
<div style="background: #f8f9fa; padding: 1rem; border-radius: 0.25rem; font-family: monospace; font-size: 0.85rem; margin: 0.5rem 0;">
tito grade setup<br>
tito system health
</div>
<p style="font-size: 0.9rem; margin: 0; color: #6c757d;">NBGrader integration & health check</p>
</div>

<div style="background: white; padding: 1.5rem; border-radius: 0.5rem; border: 1px solid #dee2e6;">
<h4 style="color: #495057; margin: 0 0 0.5rem 0;">3️⃣ First Assignment (10 min)</h4>
<div style="background: #f8f9fa; padding: 1rem; border-radius: 0.25rem; font-family: monospace; font-size: 0.85rem; margin: 0.5rem 0;">
tito grade generate 01_tensor<br>
tito grade release 01_tensor
</div>
<p style="font-size: 0.9rem; margin: 0; color: #6c757d;">Ready to distribute to students!</p>
</div>

</div>

### Assignment Workflow

TinyTorch wraps NBGrader behind simple `tito grade` commands:

**1. Prepare Assignments**
```bash
# Generate instructor version (with solutions)
tito grade generate 01_tensor

# Create student version (solutions removed)
tito grade release 01_tensor
```

**2. Collect Submissions**
```bash
# Collect all students
tito grade collect 01_tensor

# Or specific student
tito grade collect 01_tensor --student student_id
```

**3. Auto-Grade**
```bash
# Grade all submissions
tito grade autograde 01_tensor

# Grade specific student
tito grade autograde 01_tensor --student student_id
```

**4. Manual Review**
```bash
# Open grading interface (browser-based)
tito grade manual 01_tensor
```

**5. Export Grades**
```bash
# Export all grades to CSV
tito grade export

# Or specific module
tito grade export --module 01_tensor --output grades_module01.csv
```

### Grading Components

**Auto-Graded (70%)**
- Code implementation correctness
- Test passing
- Function signatures
- Output validation

**Manually Graded (30%)**
- ML Systems Thinking questions (3 per module)
- Each question: 10 points
- Focus on understanding, not perfection

### Grading Rubric for ML Systems Questions

| Points | Criteria |
|--------|----------|
| 9-10 | Demonstrates deep understanding, references specific code, discusses systems implications |
| 7-8 | Good understanding, some code references, basic systems thinking |
| 5-6 | Surface understanding, generic response, limited systems perspective |
| 3-4 | Attempted but misses key concepts |
| 0-2 | No attempt or completely off-topic |

**What to Look For:**
- References to actual implemented code
- Memory/performance analysis
- Scaling considerations
- Production system comparisons
- Understanding of trade-offs

### Module Teaching Notes

**Module 01: Tensor**
- Focus: Memory layout, data structures
- Key Concept: Understanding memory is crucial for ML performance
- Demo: Show memory profiling, copying behavior

**Module 05: Autograd**
- Focus: Computational graphs, backpropagation
- Key Concept: Automatic differentiation enables deep learning
- Demo: Visualize computational graphs

**Module 09: Spatial (CNNs)**
- Focus: Algorithmic complexity, memory patterns
- Key Concept: O(N²) operations become bottlenecks
- Demo: Profile convolution memory usage

**Module 12: Attention**
- Focus: Attention mechanisms, scaling
- Key Concept: Attention is compute-intensive but powerful
- Demo: Profile attention with different sequence lengths

**Module 20: Capstone**
- Focus: End-to-end system integration
- Key Concept: Production requires optimization across all components
- Project: Torch Olympics Competition

### Sample Schedule (16 Weeks)

| Week | Module | Focus |
|------|--------|-------|
| 1 | 01 Tensor | Data Structures, Memory |
| 2 | 02 Activations | Non-linearity Functions |
| 3 | 03 Layers | Neural Network Components |
| 4 | 04 Losses | Optimization Objectives |
| 5 | 05 Autograd | Automatic Differentiation |
| 6 | 06 Optimizers | Training Algorithms |
| 7 | 07 Training | Complete Training Loop |
| 8 | Midterm Project | Build and Train Network |
| 9 | 08 DataLoader | Data Pipeline |
| 10 | 09 Spatial | Convolutions, CNNs |
| 11 | 10 Tokenization | Text Processing |
| 12 | 11 Embeddings | Word Representations |
| 13 | 12 Attention | Attention Mechanisms |
| 14 | 13 Transformers | Transformer Architecture |
| 15 | 14-19 Optimization | Profiling, Quantization |
| 16 | 20 Capstone | Torch Olympics |

### Assessment Strategy

**Continuous Assessment (70%)**
- Module completion: 4% each × 16 = 64%
- Checkpoint achievements: 6%

**Projects (30%)**
- Midterm: Build and train CNN (15%)
- Final: Torch Olympics Competition (15%)

### Instructor Resources

- **Complete grading rubrics** with sample solutions
- **Module-specific teaching notes** in each ABOUT.md file
- **Progress tracking tools** (`tito checkpoint status --student ID`)
- **System health monitoring** (`tito module status --comprehensive`)
- **Community support** via GitHub Issues

**📖 See [Complete Course Structure](chapters/00-introduction.md)** for full curriculum overview.

---

<a id="tas"></a>
## 👥 For Teaching Assistants: Student Support Guide

### TA Preparation

Develop deep familiarity with modules where students commonly struggle:

**Critical Modules:**
1. **Module 05: Autograd** - Most conceptually challenging
2. **Module 09: CNNs (Spatial)** - Complex nested loops and memory patterns
3. **Module 13: Transformers** - Attention mechanisms and scaling

**Preparation Process:**
1. Complete all three critical modules yourself
2. Introduce bugs intentionally to understand error patterns
3. Practice debugging common scenarios
4. Review past student submissions

### Common Student Errors

#### Module 05: Autograd

**Error 1: Gradient Shape Mismatches**
- Symptom: `ValueError: shapes don't match for gradient`
- Common Cause: Incorrect gradient accumulation or shape handling
- Debugging: Check gradient shapes match parameter shapes, verify accumulation logic

**Error 2: Disconnected Computational Graph**
- Symptom: Gradients are None or zero
- Common Cause: Operations not tracked in computational graph
- Debugging: Verify `requires_grad=True`, check operations create new Tensor objects

**Error 3: Broadcasting Failures**
- Symptom: Shape errors during backward pass
- Common Cause: Incorrect handling of broadcasted operations
- Debugging: Understand NumPy broadcasting, check gradient accumulation for broadcasted dims

#### Module 09: CNNs (Spatial)

**Error 1: Index Out of Bounds**
- Symptom: `IndexError` in convolution loops
- Common Cause: Incorrect padding or stride calculations
- Debugging: Verify output shape calculations, check padding logic

**Error 2: Memory Issues**
- Symptom: Out of memory errors
- Common Cause: Creating unnecessary intermediate arrays
- Debugging: Profile memory usage, look for unnecessary copies, optimize loop structure

#### Module 13: Transformers

**Error 1: Attention Scaling Issues**
- Symptom: Attention weights don't sum to 1
- Common Cause: Missing softmax or incorrect scaling
- Debugging: Verify softmax is applied, check scaling factor (1/sqrt(d_k))

**Error 2: Positional Encoding Errors**
- Symptom: Model doesn't learn positional information
- Common Cause: Incorrect positional encoding implementation
- Debugging: Verify sinusoidal patterns, check encoding is added correctly

### Debugging Strategies

When students ask for help, guide them with questions rather than giving answers:

1. **What error message are you seeing?** - Read full traceback
2. **What did you expect to happen?** - Clarify their mental model
3. **What actually happened?** - Compare expected vs actual
4. **What have you tried?** - Avoid repeating failed approaches
5. **Can you test with a simpler case?** - Reduce complexity

### Productive vs Unproductive Struggle

**Productive Struggle (encourage):**
- Trying different approaches
- Making incremental progress
- Understanding error messages
- Passing additional tests over time

**Unproductive Frustration (intervene):**
- Repeated identical errors
- Random code changes
- Unable to articulate the problem
- No progress after 30+ minutes

### Office Hour Patterns

**Expected Demand Spikes:**

- **Module 05 (Autograd)**: Highest demand
  - Schedule additional TA capacity
  - Pre-record debugging walkthroughs
  - Create FAQ document

- **Module 09 (CNNs)**: High demand
  - Focus on memory profiling
  - Loop optimization strategies
  - Padding/stride calculations

- **Module 13 (Transformers)**: Moderate-high demand
  - Attention mechanism debugging
  - Positional encoding issues
  - Scaling problems

### Manual Review Focus Areas

While NBGrader automates 70-80% of assessment, focus manual review on:

1. **Code Clarity and Design Choices**
   - Is code readable?
   - Are design decisions justified?
   - Is the implementation clean?

2. **Edge Case Handling**
   - Does code handle edge cases?
   - Are there appropriate checks?
   - Is error handling present?

3. **Systems Thinking Analysis**
   - Do students understand complexity?
   - Can they analyze their code?
   - Do they recognize bottlenecks?

### Teaching Tips

1. **Encourage Exploration** - Let students try different approaches
2. **Connect to Production** - Reference PyTorch equivalents and real-world scenarios
3. **Make Systems Visible** - Profile memory usage, analyze complexity together
4. **Build Confidence** - Acknowledge progress and validate understanding

### TA Resources

- Module-specific ABOUT.md files with common pitfalls
- Grading rubrics with sample excellent/good/acceptable solutions
- System diagnostics tools (`tito system health`)
- Progress tracking (`tito checkpoint status --student ID`)

---

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
<h4 style="margin: 0 0 0.5rem 0; color: #166534;">🛠️ CLI & Tools</h4>
<ul style="margin: 0.5rem 0; padding-left: 1.25rem; font-size: 0.9rem;">
<li><a href="tito/overview.html">TITO CLI Overview</a></li>
<li><a href="tito/modules.html">Module Workflow</a></li>
<li><a href="tito/milestones.html">Milestone System</a></li>
<li><a href="tito/troubleshooting.html">Troubleshooting</a></li>
</ul>
</div>

<div style="background: #fef3c7; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #eab308;">
<h4 style="margin: 0 0 0.5rem 0; color: #a16207;">🤝 Community</h4>
<ul style="margin: 0.5rem 0; padding-left: 1.25rem; font-size: 0.9rem;">
<li><a href="community.html">Community Ecosystem</a></li>
<li><a href="resources.html">Learning Resources</a></li>
<li><a href="credits.html">Credits & Acknowledgments</a></li>
<li><a href="https://github.com/mlsysbook/TinyTorch/discussions">GitHub Discussions</a></li>
</ul>
</div>

</div>

---

**Ready to start building?** Choose your path above and dive into the most comprehensive ML systems course available!
