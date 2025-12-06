# Module Workflow

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h2 style="margin: 0 0 1rem 0; color: #495057;">Build ML Systems from Scratch</h2>
<p style="margin: 0; font-size: 1.1rem; color: #6c757d;">The core workflow for implementing and exporting TinyTorch modules</p>
</div>

**Purpose**: Master the module development workflow - the heart of TinyTorch. Learn how to implement modules, export them to your package, and validate with tests.

## The Core Workflow

TinyTorch follows a simple build-export-validate cycle:

```{mermaid}
graph LR
    A[Start/Resume Module] --> B[Edit in Jupyter]
    B --> C[Complete & Export]
    C --> D[Test Import]
    D --> E[Next Module]

    style A fill:#e3f2fd
    style B fill:#fffbeb
    style C fill:#f0fdf4
    style D fill:#fef3c7
    style E fill:#f3e5f5
```

**The essential command**: `tito module complete XX` - exports your code to the TinyTorch package

See [Student Workflow](../student-workflow.md) for the complete development cycle and best practices.

---

## Essential Commands

<div style="display: grid; grid-template-columns: 1fr; gap: 1rem; margin: 2rem 0;">

<div style="background: #e3f2fd; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #2196f3;">
<h4 style="margin: 0 0 0.5rem 0; color: #1976d2;">Check Environment</h4>
<code style="background: #263238; color: #ffffff; padding: 0.5rem; border-radius: 0.25rem; display: block; margin: 0.5rem 0;">tito system health</code>
<p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #64748b;">Verify your setup is ready before starting</p>
</div>

<div style="background: #fffbeb; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #f59e0b;">
<h4 style="margin: 0 0 0.5rem 0; color: #d97706;">Start a Module (First Time)</h4>
<code style="background: #263238; color: #ffffff; padding: 0.5rem; border-radius: 0.25rem; display: block; margin: 0.5rem 0;">tito module start 01</code>
<p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #64748b;">Opens Jupyter Lab for Module 01 (Tensor)</p>
</div>

<div style="background: #f3e5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #9c27b0;">
<h4 style="margin: 0 0 0.5rem 0; color: #7b1fa2;">Resume Work (Continue Later)</h4>
<code style="background: #263238; color: #ffffff; padding: 0.5rem; border-radius: 0.25rem; display: block; margin: 0.5rem 0;">tito module resume 01</code>
<p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #64748b;">Continue working on Module 01 where you left off</p>
</div>

<div style="background: #f0fdf4; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #22c55e;">
<h4 style="margin: 0 0 0.5rem 0; color: #15803d;">Export & Complete (Essential)</h4>
<code style="background: #263238; color: #ffffff; padding: 0.5rem; border-radius: 0.25rem; display: block; margin: 0.5rem 0;">tito module complete 01</code>
<p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #64748b;">Export Module 01 to TinyTorch package - THE key command</p>
</div>

<div style="background: #fef3c7; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #f59e0b;">
<h4 style="margin: 0 0 0.5rem 0; color: #d97706;">Check Progress</h4>
<code style="background: #263238; color: #ffffff; padding: 0.5rem; border-radius: 0.25rem; display: block; margin: 0.5rem 0;">tito module status</code>
<p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #64748b;">See which modules you've completed</p>
</div>

</div>

---

## Typical Development Session

Here's what a complete session looks like:

<div style="background: #f8f9fa; padding: 1.5rem; border: 1px solid #dee2e6; border-radius: 0.5rem; margin: 1.5rem 0;">

**1. Start Session**
```bash
cd TinyTorch
source activate.sh
tito system health         # Verify environment
```

**2. Start or Resume Module**
```bash
# First time working on Module 03
tito module start 03

# OR: Continue from where you left off
tito module resume 03
```

This opens Jupyter Lab with the module notebook.

**3. Edit in Jupyter Lab**
```python
# In the generated notebook
class Linear:
    def __init__(self, in_features, out_features):
        # YOUR implementation here
        ...
```

Work interactively:
- Implement the required functionality
- Add docstrings and comments
- Run and test your code inline
- See immediate feedback

**4. Export to Package**
```bash
# From repository root
tito module complete 03
```

This command:
- Runs tests on your implementation
- Exports code to `tinytorch/nn/layers.py`
- Makes your code importable
- Tracks completion

**5. Test Your Implementation**
```bash
# Your code is now in the package!
python -c "from tinytorch import Linear; print(Linear(10, 5))"
```

**6. Check Progress**
```bash
tito module status
```

</div>

---

## System Commands

### Environment Health

<div style="background: #e3f2fd; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #2196f3; margin: 1.5rem 0;">

**Check Setup (Run This First)**
```bash
tito system health
```

Verifies:
- Virtual environment activated
- Dependencies installed (NumPy, Jupyter, Rich)
- TinyTorch in development mode
- All systems ready

**Output**:
```
✅ Environment validation passed
  • Virtual environment: Active
  • Dependencies: NumPy, Jupyter, Rich installed
  • TinyTorch: Development mode
```

**System Information**
```bash
tito system info
```

Shows:
- Python version
- Environment paths
- Package versions
- Configuration settings

**Start Jupyter Lab**
```bash
tito system jupyter
```

Convenience command to launch Jupyter Lab from the correct directory.

</div>

---

## Module Lifecycle Commands

### Start a Module (First Time)

<div style="background: #fffbeb; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #f59e0b; margin: 1.5rem 0;">

```bash
tito module start 01
```

**What this does**:
1. Opens Jupyter Lab for Module 01 (Tensor)
2. Shows module README and learning objectives
3. Provides clean starting point
4. Creates backup of any existing work

**Example**:
```bash
tito module start 05  # Start Module 05 (Autograd)
```

Jupyter Lab opens with the generated notebook for Module 05

</div>

### Resume Work (Continue Later)

<div style="background: #f3e5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #9c27b0; margin: 1.5rem 0;">

```bash
tito module resume 01
```

**What this does**:
1. Opens Jupyter Lab with your previous work
2. Preserves all your changes
3. Shows where you left off
4. No backup created (you're continuing)

**Use this when**: Coming back to a module you started earlier

</div>

### Complete & Export (Essential)

<div style="background: #f0fdf4; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #22c55e; margin: 1.5rem 0;">

```bash
tito module complete 01
```

**THE KEY COMMAND** - This is what makes your code real!

**What this does**:
1. **Tests** your implementation (inline tests)
2. **Exports** to `tinytorch/` package
3. **Tracks** completion in `.tito/progress.json`
4. **Validates** NBGrader metadata
5. **Makes read-only** exported files (protection)

**Example**:
```bash
tito module complete 05  # Export Module 05 (Autograd)
```

**After exporting**:
```python
# YOUR code is now importable!
from tinytorch.autograd import backward
from tinytorch import Tensor

# Use YOUR implementations
x = Tensor([[1.0, 2.0]], requires_grad=True)
y = x * 2
y.backward()
print(x.grad)  # Uses YOUR autograd!
```

</div>

### View Progress

<div style="background: #fef3c7; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #f59e0b; margin: 1.5rem 0;">

```bash
tito module status
```

**Shows**:
- Which modules (01-20) you've completed
- Completion dates
- Next recommended module

**Example Output**:
```
📦 Module Progress

✅ Module 01: Tensor (completed 2025-11-16)
✅ Module 02: Activations (completed 2025-11-16)
✅ Module 03: Layers (completed 2025-11-16)
🔒 Module 04: Losses (not started)
🔒 Module 05: Autograd (not started)

Progress: 3/20 modules (15%)

Next: Complete Module 04 to continue Foundation Tier
```

</div>

### Reset Module (Advanced)

<div style="background: #fff5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #e74c3c; margin: 1.5rem 0;">

```bash
tito module reset 01
```

**What this does**:
1. Creates backup of current work
2. Unexports from `tinytorch/` package
3. Restores module to clean state
4. Removes from completion tracking

**Use this when**: You want to start a module completely fresh

⚠️ **Warning**: This removes your implementation. Use with caution!

</div>

---

## Understanding the Export Process

When you run `tito module complete XX`, here's what happens:

<div style="background: #f8f9fa; padding: 1.5rem; border: 1px solid #dee2e6; border-radius: 0.5rem; margin: 1.5rem 0;">

**Step 1: Validation**
```
✓ Checking NBGrader metadata
✓ Validating Python syntax
✓ Running inline tests
```

**Step 2: Export**
```
✓ Converting src/XX_name/XX_name.py
  → modules/XX_name/XX_name.ipynb (notebook)
  → tinytorch/path/name.py (package)
✓ Adding "DO NOT EDIT" warning
✓ Making file read-only
```

**Step 3: Tracking**
```
✓ Recording completion in .tito/progress.json
✓ Updating module status
```

**Step 4: Success**
```
🎉 Module XX complete!
   Your code is now part of TinyTorch!

   Import with: from tinytorch import YourClass
```

</div>

---

## Module Structure

### Development Structure

```
src/                          ← Developer source code
├── 01_tensor/
│   └── 01_tensor.py         ← SOURCE OF TRUTH (devs edit)
├── 02_activations/
│   └── 02_activations.py    ← SOURCE OF TRUTH (devs edit)
└── 03_layers/
    └── 03_layers.py         ← SOURCE OF TRUTH (devs edit)

modules/                      ← Generated notebooks (students use)
├── 01_tensor/
│   └── 01_tensor.ipynb      ← AUTO-GENERATED for students
├── 02_activations/
│   └── 02_activations.ipynb ← AUTO-GENERATED for students
└── 03_layers/
    └── 03_layers.ipynb      ← AUTO-GENERATED for students
```

### Where Code Exports

```
tinytorch/
├── core/
│   └── tensor.py           ← AUTO-GENERATED (DO NOT EDIT)
├── nn/
│   ├── activations.py      ← AUTO-GENERATED (DO NOT EDIT)
│   └── layers.py           ← AUTO-GENERATED (DO NOT EDIT)
└── ...
```

**IMPORTANT**: Understanding the flow
- **Developers**: Edit `src/XX_name/XX_name.py` → Run `tito source export` → Generates notebooks & package
- **Students**: Work in generated `modules/XX_name/XX_name.ipynb` notebooks
- **Never edit** `tinytorch/` directly - it's auto-generated
- Changes in `tinytorch/` will be lost on re-export

---

## Troubleshooting

### Environment Not Ready

<div style="background: #fff5f5; padding: 1.5rem; border: 1px solid #fed7d7; border-radius: 0.5rem; margin: 1rem 0;">

**Problem**: `tito system health` shows errors

**Solution**:
```bash
# Re-run setup
./setup-environment.sh
source activate.sh

# Verify
tito system health
```

</div>

### Export Fails

<div style="background: #fff5f5; padding: 1.5rem; border: 1px solid #fed7d7; border-radius: 0.5rem; margin: 1rem 0;">

**Problem**: `tito module complete XX` fails

**Common causes**:
1. Syntax errors in your code
2. Failing tests
3. Missing required functions

**Solution**:
1. Check error message for details
2. Fix issues in `modules/XX_name/`
3. Test in Jupyter Lab first
4. Re-run `tito module complete XX`

</div>

### Import Errors

<div style="background: #fff5f5; padding: 1.5rem; border: 1px solid #fed7d7; border-radius: 0.5rem; margin: 1rem 0;">

**Problem**: `from tinytorch import X` fails

**Solution**:
```bash
# Re-export the module
tito module complete XX

# Test import
python -c "from tinytorch import Tensor"
```

</div>

See [Troubleshooting Guide](troubleshooting.md) for more issues and solutions.

---

## Next Steps

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h3 style="margin: 0 0 1rem 0; color: #495057;">Ready to Build Your First Module?</h3>
<p style="margin: 0 0 1.5rem 0; color: #6c757d;">Start with Module 01 (Tensor) and build the foundation of neural networks</p>
<a href="../tiers/foundation.html" style="display: inline-block; background: #007bff; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500; margin-right: 1rem;">Foundation Tier →</a>
<a href="milestones.html" style="display: inline-block; background: #9c27b0; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500;">Milestone System →</a>
</div>

---

*The module workflow is the heart of TinyTorch. Master these commands and you'll build ML systems with confidence. Every line of code you write becomes part of a real, working framework.*
