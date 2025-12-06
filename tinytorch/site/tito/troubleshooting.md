# Troubleshooting Guide

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h2 style="margin: 0 0 1rem 0; color: #495057;">Common Issues & Solutions</h2>
<p style="margin: 0; font-size: 1.1rem; color: #6c757d;">Quick fixes for the most common TinyTorch problems</p>
</div>

**Purpose**: Fast solutions to common issues. Get unstuck and back to building ML systems quickly.

---

## Quick Diagnostic: Start Here

<div style="background: #e3f2fd; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #2196f3; margin: 1.5rem 0;">

**First step for ANY issue**:

```bash
cd TinyTorch
source activate.sh
tito system health
```

This checks:
- ✅ Virtual environment activated
- ✅ Dependencies installed (NumPy, Jupyter, Rich)
- ✅ TinyTorch in development mode
- ✅ Data files intact
- ✅ All systems ready

**If doctor shows errors**: Follow the specific fixes below.

**If doctor shows all green**: Your environment is fine - issue is elsewhere.

</div>

---

## Environment Issues

### Problem: "tito: command not found"

<div style="background: #fff5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #e74c3c; margin: 1.5rem 0;">

**Symptom**:
```bash
$ tito module start 01
-bash: tito: command not found
```

**Cause**: Virtual environment not activated or TinyTorch not installed in development mode.

**Solution**:
```bash
# 1. Activate environment
cd TinyTorch
source activate.sh

# 2. Verify activation
which python  # Should show TinyTorch/venv/bin/python

# 3. Re-install TinyTorch in development mode
pip install -e .

# 4. Test
tito --help
```

**Prevention**: Always run `source activate.sh` before working.

</div>

### Problem: "No module named 'tinytorch'"

<div style="background: #fff5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #e74c3c; margin: 1.5rem 0;">

**Symptom**:
```python
>>> from tinytorch import Tensor
ModuleNotFoundError: No module named 'tinytorch'
```

**Cause**: TinyTorch not installed in development mode, or wrong Python interpreter.

**Solution**:
```bash
# 1. Verify you're in the right directory
pwd  # Should end with /TinyTorch

# 2. Activate environment
source activate.sh

# 3. Install in development mode
pip install -e .

# 4. Verify installation
pip show tinytorch
python -c "import tinytorch; print(tinytorch.__file__)"
```

**Expected output**:
```
/Users/YourName/TinyTorch/tinytorch/__init__.py
```

</div>

### Problem: "Virtual environment issues after setup"

<div style="background: #fff5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #e74c3c; margin: 1.5rem 0;">

**Symptom**:
```bash
$ source activate.sh
# No (venv) prefix appears, or wrong Python version
```

**Cause**: Virtual environment not created properly or corrupted.

**Solution**:
```bash
# 1. Remove old virtual environment
rm -rf venv/

# 2. Re-run setup
./setup-environment.sh

# 3. Activate
source activate.sh

# 4. Verify
python --version  # Should be 3.8+
which pip  # Should show TinyTorch/venv/bin/pip
```

**Expected**: `(venv)` prefix appears in terminal prompt.

</div>

---

## Module Issues

### Problem: "Module export fails"

<div style="background: #fff5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #e74c3c; margin: 1.5rem 0;">

**Symptom**:
```bash
$ tito module complete 03
❌ Export failed: SyntaxError in source file
```

**Causes**:
1. Python syntax errors in your code
2. Missing required functions
3. NBGrader metadata issues

**Solution**:

**Step 1: Check syntax**:
```bash
# Test Python syntax directly (for developers)
python -m py_compile src/03_layers/03_layers.py
```

**Step 2: Open in Jupyter and test**:
```bash
tito module resume 03
# In Jupyter: Run all cells, check for errors
```

**Step 3: Fix errors shown in output**

**Step 4: Re-export**:
```bash
tito module complete 03
```

**Common syntax errors**:
- Missing `:` after function/class definitions
- Incorrect indentation (use 4 spaces, not tabs)
- Unclosed parentheses or brackets
- Missing `return` statements

</div>

### Problem: "Tests fail during export"

<div style="background: #fff5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #e74c3c; margin: 1.5rem 0;">

**Symptom**:
```bash
$ tito module complete 05
Running tests...
❌ Test failed: test_backward_simple
```

**Cause**: Your implementation doesn't match expected behavior.

**Solution**:

**Step 1: See test details**:
```bash
# Tests are in the module file - look for cells marked "TEST"
tito module resume 05
# In Jupyter: Find test cells, run them individually
```

**Step 2: Debug your implementation**:
```python
# Add print statements to see what's happening
def backward(self):
    print(f"Debug: self.grad = {self.grad}")
    # ... your implementation
```

**Step 3: Compare with expected behavior**:
- Read test assertions carefully
- Check edge cases (empty tensors, zero values)
- Verify shapes and types

**Step 4: Fix and re-export**:
```bash
tito module complete 05
```

**Tip**: Run tests interactively in Jupyter before exporting.

</div>

### Problem: "Jupyter Lab won't start"

<div style="background: #fff5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #e74c3c; margin: 1.5rem 0;">

**Symptom**:
```bash
$ tito module start 01
# Jupyter Lab fails to launch or shows errors
```

**Cause**: Jupyter not installed or port already in use.

**Solution**:

**Step 1: Verify Jupyter installation**:
```bash
pip install jupyter jupyterlab jupytext
```

**Step 2: Check for port conflicts**:
```bash
# Kill any existing Jupyter instances
pkill -f jupyter

# Or try a different port
jupyter lab --port=8889 modules/01_tensor/
```

**Step 3: Clear Jupyter cache**:
```bash
jupyter lab clean
```

**Step 4: Restart**:
```bash
tito module start 01
```

</div>

### Problem: "Changes in Jupyter don't save"

<div style="background: #fff5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #e74c3c; margin: 1.5rem 0;">

**Symptom**: Edit in Jupyter Lab, but changes don't persist.

**Cause**: File permissions or save issues.

**Solution**:

**Step 1: Manual save**:
```
In Jupyter Lab:
File → Save File (or Cmd/Ctrl + S)
```

**Step 2: Check file permissions**:
```bash
ls -la modules/01_tensor/01_tensor.ipynb
# Should be writable (not read-only)
```

**Step 3: If read-only, fix permissions**:
```bash
chmod u+w modules/01_tensor/01_tensor.ipynb
```

**Step 4: Verify changes saved**:
```bash
# Check the notebook was updated
ls -l modules/01_tensor/01_tensor.ipynb
```

</div>

---

## Import Issues

### Problem: "Cannot import from tinytorch after export"

<div style="background: #fff5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #e74c3c; margin: 1.5rem 0;">

**Symptom**:
```python
>>> from tinytorch import Linear
ImportError: cannot import name 'Linear' from 'tinytorch'
```

**Cause**: Module not exported yet, or export didn't update `__init__.py`.

**Solution**:

**Step 1: Verify module completed**:
```bash
tito module status
# Check if module shows as ✅ completed
```

**Step 2: Check exported file exists**:
```bash
ls -la tinytorch/nn/layers.py
# File should exist and have recent timestamp
```

**Step 3: Re-export**:
```bash
tito module complete 03
```

**Step 4: Test import**:
```python
python -c "from tinytorch.nn import Linear; print(Linear)"
```

**Note**: Use full import path initially, then check if `from tinytorch import Linear` works (requires `__init__.py` update).

</div>

### Problem: "Circular import errors"

<div style="background: #fff5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #e74c3c; margin: 1.5rem 0;">

**Symptom**:
```python
>>> from tinytorch import Tensor
ImportError: cannot import name 'Tensor' from partially initialized module 'tinytorch'
```

**Cause**: Circular dependency in your imports.

**Solution**:

**Step 1: Check your import structure**:
```python
# In modules/XX_name/name_dev.py
# DON'T import from tinytorch in module development files
# DO import from dependencies only
```

**Step 2: Use local imports if needed**:
```python
# Inside functions, not at module level
def some_function():
    from tinytorch.core import Tensor  # Local import
    ...
```

**Step 3: Re-export**:
```bash
tito module complete XX
```

</div>

---

## Milestone Issues

### Problem: "Milestone says prerequisites not met"

<div style="background: #fff5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #e74c3c; margin: 1.5rem 0;">

**Symptom**:
```bash
$ tito milestone run 04
❌ Prerequisites not met
   Missing modules: 08, 09
```

**Cause**: You haven't completed required modules yet.

**Solution**:

**Step 1: Check requirements**:
```bash
tito milestone info 04
# Shows which modules are required
```

**Step 2: Complete required modules**:
```bash
tito module status  # See what's completed
tito module start 08  # Complete missing modules
# ... implement and export
tito module complete 08
```

**Step 3: Try milestone again**:
```bash
tito milestone run 04
```

**Tip**: Milestones unlock progressively. Complete modules in order (01 → 20) for best experience.

</div>

### Problem: "Milestone fails with import errors"

<div style="background: #fff5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #e74c3c; margin: 1.5rem 0;">

**Symptom**:
```bash
$ tito milestone run 03
Running: MLP Revival (1986)
ImportError: cannot import name 'ReLU' from 'tinytorch'
```

**Cause**: Required module not exported properly.

**Solution**:

**Step 1: Check which import failed**:
```
# Error message shows: 'ReLU' from 'tinytorch'
# This is from Module 02 (Activations)
```

**Step 2: Re-export that module**:
```bash
tito module complete 02
```

**Step 3: Test import manually**:
```python
python -c "from tinytorch import ReLU; print(ReLU)"
```

**Step 4: Run milestone again**:
```bash
tito milestone run 03
```

</div>

### Problem: "Milestone runs but shows errors"

<div style="background: #fff5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #e74c3c; margin: 1.5rem 0;">

**Symptom**:
```bash
$ tito milestone run 03
Running: MLP Revival (1986)
# Script runs but shows runtime errors or wrong output
```

**Cause**: Your implementation has bugs (not syntax errors, but logic errors).

**Solution**:

**Step 1: Run milestone script manually**:
```bash
python milestones/03_1986_mlp/03_mlp_mnist_train.py
# See full error output
```

**Step 2: Debug the specific module**:
```bash
# If error is in ReLU, for example
tito module resume 02
# Fix implementation in Jupyter
```

**Step 3: Re-export**:
```bash
tito module complete 02
```

**Step 4: Test milestone again**:
```bash
tito milestone run 03
```

**Tip**: Milestones test your implementations in realistic scenarios. They help find edge cases you might have missed.

</div>

---

## Data & Progress Issues

### Problem: ".tito folder deleted or corrupted"

<div style="background: #fff5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #e74c3c; margin: 1.5rem 0;">

**Symptom**:
```bash
$ tito module status
Error: .tito/progress.json not found
```

**Cause**: `.tito/` folder deleted or progress file corrupted.

**Solution**:

**Option 1: Let TinyTorch recreate it (fresh start)**:
```bash
tito system health
# Recreates .tito/ structure with empty progress
```

**Option 2: Restore from backup (if you have one)**:
```bash
# Check for backups
ls -la .tito_backup_*/

# Restore from latest backup
cp -r .tito_backup_20251116_143000/ .tito/
```

**Option 3: Manual recreation**:
```bash
mkdir -p .tito/backups
echo '{"version":"1.0","completed_modules":[],"completion_dates":{}}' > .tito/progress.json
echo '{"version":"1.0","completed_milestones":[],"completion_dates":{}}' > .tito/milestones.json
echo '{"logo_theme":"standard"}' > .tito/config.json
```

**Important**: Your code in `modules/` and `tinytorch/` is safe. Only progress tracking is affected.

</div>

### Problem: "Progress shows wrong modules completed"

<div style="background: #fff5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #e74c3c; margin: 1.5rem 0;">

**Symptom**:
```bash
$ tito module status
Shows modules as completed that you haven't done
```

**Cause**: Accidentally ran `tito module complete XX` without implementing, or manual `.tito/progress.json` edit.

**Solution**:

**Option 1: Reset specific module**:
```bash
tito module reset 05
# Clears completion for Module 05 only
```

**Option 2: Reset all progress**:
```bash
tito reset progress
# Clears all module completion
```

**Option 3: Manually edit `.tito/progress.json`**:
```bash
# Open in editor
nano .tito/progress.json

# Remove the module number from "completed_modules" array
# Remove the entry from "completion_dates" object
```

</div>

---

## Dependency Issues

### Problem: "NumPy import errors"

<div style="background: #fff5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #e74c3c; margin: 1.5rem 0;">

**Symptom**:
```python
>>> import numpy as np
ImportError: No module named 'numpy'
```

**Cause**: Dependencies not installed in virtual environment.

**Solution**:
```bash
# Activate environment
source activate.sh

# Install dependencies
pip install numpy jupyter jupyterlab jupytext rich

# Verify
python -c "import numpy; print(numpy.__version__)"
```

</div>

### Problem: "Rich formatting doesn't work"

<div style="background: #fff5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #e74c3c; margin: 1.5rem 0;">

**Symptom**: TITO output is plain text instead of colorful panels.

**Cause**: Rich library not installed or terminal doesn't support colors.

**Solution**:

**Step 1: Install Rich**:
```bash
pip install rich
```

**Step 2: Use color-capable terminal**:
- macOS: Terminal.app, iTerm2
- Linux: GNOME Terminal, Konsole
- Windows: Windows Terminal, PowerShell

**Step 3: Test**:
```bash
python -c "from rich import print; print('[bold green]Test[/bold green]')"
```

</div>

---

## Performance Issues

### Problem: "Jupyter Lab is slow"

<div style="background: #fffbeb; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #f59e0b; margin: 1.5rem 0;">

**Solutions**:

**1. Close unused notebooks**:
```
In Jupyter Lab:
Right-click notebook tab → Close
File → Shut Down All Kernels
```

**2. Clear output cells**:
```
In Jupyter Lab:
Edit → Clear All Outputs
```

**3. Restart kernel**:
```
Kernel → Restart Kernel
```

**4. Increase memory** (if working with large datasets):
```bash
# Check memory usage
top
# Close other applications if needed
```

</div>

### Problem: "Export takes a long time"

<div style="background: #fffbeb; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #f59e0b; margin: 1.5rem 0;">

**Cause**: Tests running on large data or complex operations.

**Solution**:

**This is normal for**:
- Modules with extensive tests
- Operations involving training loops
- Large tensor operations

**If export hangs**:
```bash
# Cancel with Ctrl+C
# Check for infinite loops in your code
# Simplify tests temporarily, then re-export
```

</div>

---

## Platform-Specific Issues

### macOS: "Permission denied"

<div style="background: #fff5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #e74c3c; margin: 1.5rem 0;">

**Symptom**:
```bash
$ ./setup-environment.sh
Permission denied
```

**Solution**:
```bash
chmod +x setup-environment.sh activate.sh
./setup-environment.sh
```

</div>

### Windows: "activate.sh not working"

<div style="background: #fff5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #e74c3c; margin: 1.5rem 0;">

**Solution**: Use Windows-specific activation:
```bash
# PowerShell
.\venv\Scripts\Activate.ps1

# Command Prompt
.\venv\Scripts\activate.bat

# Git Bash
source venv/Scripts/activate
```

</div>

### Linux: "Python version issues"

<div style="background: #fff5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #e74c3c; margin: 1.5rem 0;">

**Solution**: Specify Python 3.8+ explicitly:
```bash
python3.8 -m venv venv
source activate.sh
python --version  # Verify
```

</div>

---

## Getting More Help

### Debug Mode

<div style="background: #e3f2fd; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #2196f3; margin: 1.5rem 0;">

**Run commands with verbose output**:
```bash
# Most TITO commands support --verbose
tito module complete 03 --verbose

# See detailed error traces
python -m pdb milestones/03_1986_mlp/03_mlp_mnist_train.py
```

</div>

### Check Logs

<div style="background: #e3f2fd; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #2196f3; margin: 1.5rem 0;">

**Jupyter Lab logs**:
```bash
# Check Jupyter output in terminal where you ran tito module start
# Look for error messages, warnings
```

**Python traceback**:
```bash
# Full error context
python -c "from tinytorch import Tensor" 2>&1 | less
```

</div>

### Community Support

<div style="background: #f3e5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #9c27b0; margin: 1.5rem 0;">

**GitHub Issues**: Report bugs or ask questions
- Repository: [mlsysbook/TinyTorch](https://github.com/mlsysbook/TinyTorch)
- Search existing issues first
- Include error messages and OS details

**Documentation**: Check other guides
- [Module Workflow](modules.md)
- [Milestone System](milestones.md)
- [Progress & Data](data.md)

</div>

---

## Prevention: Best Practices

<div style="background: #f0fdf4; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #22c55e; margin: 1.5rem 0;">

**Avoid issues before they happen**:

1. **Always activate environment first**:
   ```bash
   source activate.sh
   ```

2. **Run `tito system health` regularly**:
   ```bash
   tito system health
   ```

3. **Test in Jupyter before exporting**:
   ```bash
   # Run all cells, verify output
   # THEN run tito module complete
   ```

4. **Keep backups** (automatic):
   ```bash
   # Backups happen automatically
   # Don't delete .tito/backups/ unless needed
   ```

5. **Use git for your code**:
   ```bash
   git commit -m "Working Module 05 implementation"
   ```

6. **Read error messages carefully**:
   - They usually tell you exactly what's wrong
   - Pay attention to file paths and line numbers

</div>

---

## Quick Reference: Fixing Common Errors

| Error Message | Quick Fix |
|--------------|-----------|
| `tito: command not found` | `source activate.sh` |
| `ModuleNotFoundError: tinytorch` | `pip install -e .` |
| `SyntaxError` in export | Fix Python syntax, test in Jupyter first |
| `ImportError` in milestone | Re-export required modules |
| `.tito/progress.json not found` | `tito system health` to recreate |
| `Jupyter Lab won't start` | `pkill -f jupyter && tito module start XX` |
| `Permission denied` | `chmod +x setup-environment.sh activate.sh` |
| `Tests fail` during export | Debug in Jupyter, check test assertions |
| `Prerequisites not met` | `tito milestone info XX` to see requirements |

---

## Still Stuck?

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h3 style="margin: 0 0 1rem 0; color: #495057;">Need More Help?</h3>
<p style="margin: 0 0 1.5rem 0; color: #6c757d;">Try these resources for additional support</p>
<a href="https://github.com/mlsysbook/TinyTorch/issues" style="display: inline-block; background: #28a745; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500; margin-right: 1rem;">Report Issue →</a>
<a href="overview.html" style="display: inline-block; background: #007bff; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500;">Command Reference →</a>
</div>

---

*Most issues have simple fixes. Start with `tito system health`, read error messages carefully, and remember: your code is always safe in `modules/` - only progress tracking can be reset.*
