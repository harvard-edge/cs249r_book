# 🩺 How Students Use Environment Validation

## Quick Health Check

**When to use**: Anytime you want to verify your TinyTorch environment is working.

```bash
tito system health
```

**What it shows**:
- Python version ✅
- Virtual environment status ✅
- Core packages (numpy, pytest, etc.) ✅
- Project structure ✅
- Module status ✅

**Takes**: ~1 second

---

## Comprehensive Validation

**When to use**:
- After running `tito setup`
- Before starting a new module
- When something isn't working
- Before asking a TA for help

```bash
tito system health
```

**What it shows**:
- 🧪 Beautiful header explaining the check
- 📊 Summary table (passed/failed/skipped)
- ✅ or ❌ Health status with clear messaging
- 📋 Detailed test output (if there are failures)
- 💡 Quick fixes for common issues

**What it tests** (60+ checks):
- ✅ Python environment (version, venv, pip)
- ✅ All packages from requirements.txt
- ✅ Packages actually work (not just installed)
- ✅ Jupyter/JupyterLab configuration
- ✅ TinyTorch package structure
- ✅ System resources (disk, memory)
- ✅ Git configuration
- ✅ No version conflicts

**Takes**: ~5 seconds

---

## Example Output

### When Everything Works ✅

```bash
$ tito system health

╭─────────────────────────────── TinyTorch Health Check ───────────────────────╮
│ 🧪 Running Comprehensive Environment Validation                              │
│                                                                              │
│ This will test 60+ aspects of your TinyTorch environment.                    │
│ Perfect for sharing with TAs if something isn't working!                     │
╰──────────────────────────────────────────────────────────────────────────────╯

Running validation tests...

                         Test Results Summary
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Category                       ┃      Count ┃ Status               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ Tests Passed                   │         65 │ ✅ OK                │
│ Tests Skipped                  │          3 │ ⏭️  Optional          │
└────────────────────────────────┴────────────┴──────────────────────┘

╭─────────────────────────────── Health Status ────────────────────────────────╮
│ ✅ Environment is HEALTHY!                                                   │
│                                                                              │
│ All 65 required checks passed.                                               │
│ 3 optional checks skipped.                                                   │
│                                                                              │
│ Your Tiny🔥Torch environment is ready to use! 🎉                               │
│                                                                              │
│ Next: tito module 01                                                         │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

### When Something Fails ❌

```bash
$ tito system health

╭─────────────────────────────── TinyTorch Health Check ───────────────────────╮
│ 🧪 Running Comprehensive Environment Validation                              │
│                                                                              │
│ This will test 60+ aspects of your TinyTorch environment.                    │
│ Perfect for sharing with TAs if something isn't working!                     │
╰──────────────────────────────────────────────────────────────────────────────╯

Running validation tests...

                         Test Results Summary
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Category                       ┃      Count ┃ Status               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ Tests Passed                   │         59 │ ✅ OK                │
│ Tests Failed                   │          3 │ ❌ Issues Found      │
│ Tests Skipped                  │          3 │ ⏭️  Optional          │
└────────────────────────────────┴────────────┴──────────────────────┘

╭─────────────────────────────── Health Status ────────────────────────────────╮
│ ❌ Found 3 issue(s)                                                          │
│                                                                              │
│ 59 checks passed, but some components need attention.                        │
│                                                                              │
│ What to share with your TA:                                                  │
│ 1. Copy the output above                                                     │
│ 2. Include the error messages below                                          │
│ 3. Mention what you were trying to do                                        │
│                                                                              │
│ Or try: tito setup to reinstall                                             │
╰──────────────────────────────────────────────────────────────────────────────╯

╭──────────────────────────────────────────────────────────────────────────────╮
│ 📋 Detailed Test Output                                                      │
╰──────────────────────────────────────────────────────────────────────────────╯

FAILED tests/environment/test_setup_validation.py::TestJupyterEnvironment::test_jupyterlab_import
    ModuleNotFoundError: No module named 'jupyterlab'

FAILED tests/environment/test_setup_validation.py::TestJupyterEnvironment::test_jupyter_lab_command
    AssertionError: jupyter lab command not found

FAILED tests/environment/test_all_requirements.py::TestRequiredPackages::test_package_installed[jupyterlab]
    ❌ jupyterlab cannot be imported
    Install: pip install jupyterlab>=4.2.0

╭─────────────────────────────── 💡 Quick Fixes ───────────────────────────────╮
│ Common Solutions:                                                            │
│                                                                              │
│ • Missing packages: pip install -r requirements.txt                          │
│ • Jupyter issues: pip install --upgrade jupyterlab                           │
│ • Import errors: pip install -e . (reinstall TinyTorch)                      │
│ • Still stuck: Run tito system health --verbose                    │
│                                                                              │
│ Then share the full output with your TA                                      │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

## Verbose Output

**When to use**: When you need even more details for debugging or sharing with TAs.

```bash
tito system health --verbose
```

**What it shows**:
- Everything from the basic health check
- Plus: Full pytest output with all test details
- Plus: Complete error messages and stack traces

---

## For TAs: How to Read Reports

When a student shares their `tito system health` output, look for:

### 1. **Test Results Summary Table**
- Shows passed/failed/skipped counts
- Quick overview of environment health

### 2. **Health Status Panel**
- ✅ Green = Environment is healthy, ready to use
- ❌ Red = Issues found, shows count

### 3. **Detailed Test Output** (if failures)
- Lists specific failed tests
- Shows error messages
- Indicates missing packages or configuration issues

### 4. **Common Patterns**

**Missing Jupyter**:
```
FAILED test_jupyterlab_import - ModuleNotFoundError: No module named 'jupyterlab'
```
**Fix**: `pip install jupyterlab`

**Wrong NumPy version**:
```
FAILED test_package_installed[numpy] - numpy version 1.20.0 does not match >=1.24.0
```
**Fix**: `pip install --upgrade numpy`

**Package conflicts**:
```
FAILED test_no_conflicting_versions - Found conflicting version specifications
```
**Fix**: Standardize requirements files or use the higher version requirement

**TinyTorch not installed**:
```
FAILED test_tinytorch_import - ModuleNotFoundError: No module named 'tinytorch'
```
**Fix**: `pip install -e .` from the TinyTorch root directory

---

## Integration with Student Workflow

### First Time Setup
```bash
# 1. Clone repository
git clone https://github.com/harvard-edge/cs249r_book.git
cd TinyTorch

# 2. Run setup
tito setup

# 3. Verify everything works
tito system health

# If all ✅ green, you're ready!
tito module start 01
```

### Before Starting a Module
```bash
# Quick health check
tito system health

# If you see any ❌ red, run with verbose for details
tito system health --verbose
```

### When Something Breaks
```bash
# 1. Run full verification
tito system health --verbose

# 2. Copy the entire output

# 3. Share with TA along with:
#    - What you were trying to do
#    - What error you saw
#    - What you've tried so far
```

---

## Common Student Questions

### Q: How often should I run this?
**A**:
- Quick check (`tito system health`): Anytime, it's fast
- Verbose output (`--verbose`): After setup, when issues occur, before asking for help

### Q: What if tests are failing?
**A**:
1. Try the suggested fixes in the "💡 Quick Fixes" panel
2. Run `tito setup` to reinstall everything
3. If still failing, run with `--verbose` and share with TA

### Q: What does "Tests Skipped" mean?
**A**: Optional components (like matplotlib) that aren't required for core functionality. You can ignore these.

### Q: Can I share this output with TAs?
**A**: Yes! That's exactly what it's designed for. The output includes everything a TA needs to help debug your issue.

### Q: What if the validation says I'm healthy but I still have issues?
**A**:
1. Try `tito system health --verbose` for more details
2. The validation tests core environment - your specific issue might be module-specific
3. Run `tito module test N` to test a specific module
4. Share both outputs with your TA

---

## Direct pytest Access (Advanced)

If you want to run the tests directly with pytest (not through TITO):

```bash
# Run all environment tests
pytest tests/environment/ -v

# Run just setup validation
pytest tests/environment/test_setup_validation.py -v

# Run just requirements validation
pytest tests/environment/test_all_requirements.py -v

# Run a specific test class
pytest tests/environment/test_setup_validation.py::TestPythonEnvironment -v
```

But for students, we recommend using `tito system health` instead - it has prettier output! 🎨
