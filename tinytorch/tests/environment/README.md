# 🧪 Environment Validation Tests

Comprehensive tests to ensure TinyTorch environment is correctly configured and all dependencies work.

## 🎯 For Students

**Easy-to-use command with beautiful output:**

```bash
# Quick health check (1 second)
tito system health

# Comprehensive validation (5 seconds)
tito system health

# Verbose output for debugging
tito system health --verbose
```

**Perfect for**:
- ✅ Verifying your environment after setup
- ✅ Checking everything works before starting a module
- ✅ Debugging when something isn't working
- ✅ Sharing with TAs when you need help

**See**: [HOW_TO_USE.md](HOW_TO_USE.md) for complete student guide with examples.

---

## 🔬 For Developers

### Run All Validation Tests
```bash
# Via TITO (recommended - beautiful output)
tito system health

# Via pytest (raw test output)
pytest tests/environment/ -v
```

### Run Specific Test Suites

**Setup Validation** (comprehensive environment check):
```bash
pytest tests/environment/test_setup_validation.py -v
```

**Requirements Validation** (all packages from requirements.txt):
```bash
pytest tests/environment/test_all_requirements.py -v
```

## Test Suites

### 1. Setup Validation (`test_setup_validation.py`)

**Tests 50+ environment checks** organized into categories:

#### Python Environment
- ✅ Python version (3.8+)
- ✅ Virtual environment active
- ✅ pip available

#### Core Dependencies
- ✅ NumPy: import, arrays, matrix operations
- ✅ Matplotlib: import, plotting, save figures
- ✅ pytest: available for testing
- ✅ PyYAML: import, YAML serialization
- ✅ Rich: console rendering

#### Jupyter Environment
- ✅ Jupyter installed
- ✅ JupyterLab available
- ✅ jupyter command available
- ✅ jupyter lab command works
- ✅ Python3 kernel configured
- ✅ Jupytext for .py ↔ .ipynb conversion

#### TinyTorch Package
- ✅ tinytorch package importable
- ✅ tinytorch.core available
- ✅ Version info defined
- ✅ Tensor class (if Module 01 completed)

#### Project Structure
- ✅ tinytorch/ package directory
- ✅ modules/ student workspace
- ✅ src/ source modules
- ✅ tests/ test directory
- ✅ TITO CLI available

#### System Resources
- ✅ Adequate disk space (1GB+)
- ✅ Adequate memory (checks available)
- ✅ Python architecture (warns about Rosetta on M1/M2)

#### Git Configuration
- ✅ Git available
- ✅ Git user configured
- ✅ Repository initialized

### 2. Requirements Validation (`test_all_requirements.py`)

**Automatically discovers and tests ALL packages** from requirements files:

#### Auto-Discovery
- 📁 Finds all requirements*.txt files in project
- 📋 Parses package specifications (handles >=, ==, <, etc.)
- 🔍 Converts package names to import names (PyYAML → yaml, etc.)

#### Package Tests
- ✅ **Installation**: Package can be imported
- ✅ **Version**: Installed version matches specification
- ✅ **Functionality**: Package actually works (not just installed)

#### Functionality Tests Include:
- **numpy**: Array creation and operations
- **matplotlib**: Plot creation and saving
- **pytest**: Command availability
- **jupyterlab**: Command availability
- **jupytext**: Notebook parsing
- **PyYAML**: YAML serialization
- **rich**: Console rendering
- **Generic**: Import test for other packages

#### Consistency Checks
- ✅ No conflicting version specs across files
- ✅ Requirements files are readable
- ✅ Requirements files are parseable

## Example Output

### Successful Run
```bash
$ pytest tests/environment/ -v

tests/environment/test_setup_validation.py::TestPythonEnvironment::test_python_version PASSED
✅ Python 3.10.8
tests/environment/test_setup_validation.py::TestPythonEnvironment::test_virtual_environment_active PASSED
✅ Virtual environment active: /Users/student/TinyTorch/.venv
tests/environment/test_setup_validation.py::TestCoreDependencies::test_numpy_import PASSED
✅ NumPy 1.24.3 imported
tests/environment/test_setup_validation.py::TestCoreDependencies::test_numpy_operations PASSED
✅ NumPy operations work correctly
...

tests/environment/test_all_requirements.py::TestRequiredPackages::test_package_installed[numpy] PASSED
✅ numpy v1.24.3 installed
tests/environment/test_all_requirements.py::TestRequiredPackages::test_package_functionality[numpy] PASSED
✅ numpy: Array operations work
...

============================== 75 passed in 2.5s ==============================
🎉 All validation tests passed!
✅ TinyTorch environment is correctly configured
💡 Next: tito module 01
```

### Failed Run (with helpful errors)
```bash
$ pytest tests/environment/ -v

tests/environment/test_all_requirements.py::TestRequiredPackages::test_package_installed[matplotlib] FAILED
❌ matplotlib cannot be imported
   Import name: matplotlib
   Required by: requirements.txt
   Install: pip install matplotlib>=3.9.0
   Error: No module named 'matplotlib'

tests/environment/test_setup_validation.py::TestJupyterEnvironment::test_jupyter_lab_command FAILED
❌ jupyter lab command not found
   Fix: pip install jupyterlab

============================== 2 failed, 73 passed in 2.3s ==============================
❌ Some validation tests failed
🔧 Install missing packages: pip install -r requirements.txt
```

## Integration with TITO

### `tito system health`
Basic environment check (quick):
```bash
tito system health

# Shows:
# ✅ Python 3.10.8
# ✅ Virtual environment active
# ✅ NumPy v1.24.3
# ✅ Matplotlib v3.7.1
# ✅ Jupyter available
```

### `tito system health`
Comprehensive validation (runs all tests):
```bash
tito system health

# Runs both test suites:
# 1. test_setup_validation.py (50+ checks)
# 2. test_all_requirements.py (all packages)
#
# Takes ~5 seconds
# Shows detailed results for each check
```

### `tito system health`
Quick validation (essential checks only):
```bash
tito system health

# Runs:
# - Python environment
# - Core dependencies (numpy, jupyter)
# - TinyTorch package
#
# Takes ~1 second
# Good for "is everything basically working?"
```

## Adding New Tests

### For New Dependencies
Add to `test_package_functionality()` in `test_all_requirements.py`:
```python
elif package_name.lower() == 'mypackage':
    import mypackage
    # Test basic functionality
    result = mypackage.do_something()
    return result is not None, "Basic function works"
```

### For New Environment Checks
Add new test to `test_setup_validation.py`:
```python
class TestMyComponent:
    """Test my new component."""

    def test_my_check(self):
        """Description of what is tested."""
        # Your test logic
        assert something_works, "Error message"
        print("✅ My component works")
```

## CI/CD Integration

### GitHub Actions
```yaml
- name: Validate Environment
  run: |
    pip install -r requirements.txt
    pytest tests/environment/ -v
```

### Pre-commit Hook
```bash
# .git/hooks/pre-commit
#!/bin/bash
pytest tests/environment/test_all_requirements.py -q
```

## Troubleshooting

### Tests fail with "No module named 'X'"
```bash
# Install missing package
pip install -r requirements.txt

# Or specific package
pip install X
```

### Tests fail with version mismatch
```bash
# Upgrade package to required version
pip install --upgrade X

# Or reinstall everything
pip install -r requirements.txt --force-reinstall
```

### Virtual environment not detected
```bash
# Activate virtual environment
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows

# Then run tests again
pytest tests/environment/ -v
```

### Jupyter tests fail
```bash
# Reinstall Jupyter
pip install --upgrade jupyter jupyterlab

# Check kernel
jupyter kernelspec list

# Install kernel if missing
python -m ipykernel install --user
```

## Best Practices

1. **Run before starting work**: `tito system health`
2. **Run after setup**: Automatically runs at end of `tito setup`
3. **Run after package updates**: `pip install -r requirements.txt && tito system health`
4. **Include in CI/CD**: Ensures environment consistency
5. **Add tests for new dependencies**: Keep validation comprehensive

## Performance

- **Quick check** (~1s): Basic imports and versions
- **Full validation** (~5s): All functionality tests
- **Cached results**: Pytest caches successful imports

## What Gets Tested

✅ **60+ automated checks** across:
- Python environment (3 checks)
- Core dependencies (7 checks)
- Jupyter environment (6 checks)
- TinyTorch package (4 checks)
- Project structure (7 checks)
- System resources (3 checks)
- Git configuration (3 checks)
- All requirements.txt packages (N checks)
- Package version consistency (1 check)
- Requirements file validity (2 checks)

**Result**: Complete confidence that environment works before students start!
