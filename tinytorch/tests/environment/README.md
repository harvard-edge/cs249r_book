# Environment Validation Tests

Comprehensive tests to ensure TinyTorch environment is correctly configured.

## For Students

```bash
# Quick health check
tito system health

# Verbose output for debugging
tito system health --verbose
```

**Use when**:
- After running `tito setup`
- Before starting a new module
- When something isn't working
- Sharing output with TAs for help

## Test Suites

### Setup Validation (`test_setup_validation.py`)

Tests 50+ environment checks:

- **Python Environment**: Version (3.8+), virtual environment, pip
- **Core Dependencies**: NumPy, Matplotlib, pytest, PyYAML, Rich
- **Jupyter Environment**: Jupyter, JupyterLab, kernels, Jupytext
- **TinyTorch Package**: Import, core modules, version
- **Project Structure**: tinytorch/, modules/, src/, tests/, TITO CLI
- **System Resources**: Disk space, memory
- **Git Configuration**: Git available, user configured

### Requirements Validation (`test_all_requirements.py`)

Auto-discovers and tests ALL packages from requirements files:

- Package installation (can be imported)
- Version matching (meets specification)
- Functionality (actually works, not just installed)

## Running Tests

```bash
# Via TITO (recommended)
tito system health

# Via pytest
pytest tests/environment/ -v

# Specific suite
pytest tests/environment/test_setup_validation.py -v
```

## Troubleshooting

**"No module named 'X'"**:
```bash
pip install -r requirements.txt
```

**Version mismatch**:
```bash
pip install --upgrade X
```

**Virtual environment not detected**:
```bash
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
```

**Jupyter tests fail**:
```bash
pip install --upgrade jupyter jupyterlab
python -m ipykernel install --user
```

## CI Integration

```yaml
- name: Validate Environment
  run: pytest tests/environment/ -v
```
