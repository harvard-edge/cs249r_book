# ISCA Tutorial: Prerequisites and Setup

Everything you need to run the hands-on exercises.

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.12+ |
| OS | macOS, Linux, Windows | Any |
| RAM | 2 GB | 4 GB |
| Disk | 50 MB | 100 MB |
| GPU | **Not required** | Not required |

mlsysim is a first-principles infrastructure modeling tool -- it models
hardware performance analytically without executing any GPU kernels.
All exercises run on a laptop CPU in seconds.

---

## Installation

### Option A: pip install (recommended)

```bash
pip install mlsysim
```

### Option B: Install from source (for tutorial development)

```bash
git clone https://github.com/harvard-edge/cs249r_book.git
cd cs249r_book/mlsysim
pip install -e ".[full]"
```

The `[full]` extra includes visualization libraries (plotly, matplotlib)
and optimization solvers (scipy, ortools) used in some exercises.

### Option C: Minimal install (exercises only)

```bash
pip install mlsysim
```

Core dependencies installed automatically:
- `pint` (physical units)
- `pydantic` (data validation)
- `numpy` (numerical computation)
- `typer` and `rich` (CLI interface)
- `pyyaml` (configuration files)

---

## Verification

Run this command to verify your installation:

```bash
python3 -c "
import mlsysim
print(f'mlsysim v{mlsysim.__version__}')

# Quick smoke test: ResNet-50 on A100
from mlsysim import Engine, Hardware, Models
p = Engine.solve(Models.ResNet50, Hardware.A100, batch_size=1)
print(f'ResNet-50 on A100: {p.latency:~P.2f}, bottleneck={p.bottleneck}')

# Verify registries
print(f'Hardware: {len(list(Hardware))} accelerators')
print(f'Models:   {len(list(Models))} workloads')
print('All checks passed.')
"
```

Expected output (approximate):

```
mlsysim v0.1.0
ResNet-50 on A100: 0.XX ms, bottleneck=memory
Hardware: XX accelerators
Models:   XX workloads
All checks passed.
```

If the import succeeds and the smoke test prints a latency value, you are
ready for the tutorial.

---

## Troubleshooting FAQ

### 1. `ModuleNotFoundError: No module named 'mlsysim'`

**Cause:** mlsysim is not installed in your active Python environment.

**Fix:**
```bash
# Check which Python you are using
which python3
python3 --version

# Install in the active environment
pip install mlsysim

# If using conda:
conda activate your_env
pip install mlsysim
```

### 2. `ModuleNotFoundError: No module named 'pint'`

**Cause:** Core dependency not installed (rare with pip, common with
manual source installs).

**Fix:**
```bash
pip install pint>=0.23 pydantic>=2.0 numpy>=1.24
```

### 3. `ImportError: cannot import name 'Hardware' from 'mlsysim'`

**Cause:** You have an outdated version of mlsysim or a name collision
with another package.

**Fix:**
```bash
# Check version
python3 -c "import mlsysim; print(mlsysim.__version__)"

# Upgrade to latest
pip install --upgrade mlsysim

# If name collision, check for local files named mlsysim.py
ls mlsysim.py 2>/dev/null && echo "Remove or rename this file!"
```

### 4. `pint.DimensionalityError` or unexpected unit errors

**Cause:** Mixing raw numbers with Pint Quantities. mlsysim uses
physical units throughout -- all inputs and outputs carry dimensions.

**Fix:**
```python
# Wrong: passing a raw number where a Quantity is expected
bandwidth = 100  # missing units!

# Right: use the unit registry
from mlsysim import ureg
bandwidth = 100 * ureg.GB / ureg.s
```

**General rule:** If a function expects bandwidth, memory, or time,
pass a Pint Quantity with explicit units. The error message will
tell you which units are expected.

---

## Optional: Visualization Setup

Some exercises include optional visualization. Install plotly for
interactive charts:

```bash
pip install mlsysim[viz]
```

This adds `plotly` and `matplotlib` for `plot_roofline()` and
`plot_evaluation_scorecard()`.

---

## Tutorial Files

After setup, the exercise file is at:

```
mlsysim/tutorial/exercises.md
```

Each exercise is self-contained. You can copy code blocks into a
Python REPL, Jupyter notebook, or any IDE. No special notebook
infrastructure is required.
