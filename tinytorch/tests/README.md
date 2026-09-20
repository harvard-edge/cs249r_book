# TinyTorch Test Suite

The TinyTorch test suite validates the framework at every level of abstraction:
from individual mathematical operations in a single module, to the seams where
modules connect, to end-to-end training loops and historical milestone reproductions.

The suite serves two audiences simultaneously:
1. **Students**: Delivers immediate, educational feedback through rich terminal formatting
   and clear `WHAT` / `WHY` explanations when tests fail.
2. **Framework Engineers**: Enforces mathematical precision, autograd graph integrity,
   and zero regressions across all 20 modules.

---

## 🏛️ Test Architecture & Directory Layout

```
tests/
├── 01_tensor/ ... 20_capstone/   # Module-level unit, progressive, and contract tests
├── integration/                  # Cross-module seams, end-to-end gradient flow, pipelines
├── regression/                   # Past defect reproductions, hardware parity, schema checks
├── milestones/                   # Historical milestone verification (smoke & training runs)
├── cli/                          # The `tito` developer tool and command registry
├── environment/                  # Platform, interpreter, dependencies, and workspace health
├── e2e/                          # Full student journey simulations from start to capstone
├── conftest.py                   # Pytest configuration, package validator, and educational plugin
└── test_utils.py                 # Shared test helpers, assertions, and tensor factories
```

### Directory Scope & Responsibilities

| Directory | Scope | What Belongs Here | Key Test Files |
|---|---|---|---|
| `NN_modulename/` | One module (01–20) | Authoritative unit tests, progressive notebook checks, and in-module contracts | `test_NN_progressive.py`, `test_*_core.py`, `test_*_contracts.py` |
| `integration/` | Cross-module boundaries | Backpropagation across modules, training loops, CNN/NLP end-to-end pipelines | `test_integration_gradient_flow.py`, `test_training_flow.py`, `test_nlp_pipeline_flow.py`, `test_layers_composition.py` |
| `regression/` | Specific past defects | Tests named for or documenting bugs preventing silent regressions | `test_gradient_flow_fixes.py`, `test_conv_linear_dimensions.py`, `test_transformer_reshaping.py`, `test_hardware_extensions.py` |
| `milestones/` | Milestone scripts | Historical milestones (1958–2020) verifying the framework can train real models | `test_milestones_smoke.py`, `test_milestones_run.py` |
| `cli/` | `tito` CLI tool | Command registry validation, help text consistency, nbgrader workflow, progress sync | `test_cli_registry.py`, `test_cli_execution.py`, `test_nbgrader_command.py` |
| `environment/` | Runtime & machine | Python interpreter version, package requirements, Jupyter kernel, directory layout | `test_setup_validation.py`, `test_all_requirements.py`, `test_conftest_validation.py` |
| `e2e/` | Student journey | Simulates student path: `tito setup` → `module start` → `module complete` → milestone unlocks | `test_user_journey.py`, `validate_release.sh` |

---

## 📦 Anatomy of a Module Test Directory (`tests/NN_modulename/`)

Every module directory follows a predictable, disciplined pattern designed to support both progressive learning and rigorous verification:

1. **Progressive Tests (`test_NN_<name>_progressive.py`)**:
   - Mirrors the step-by-step exercise flow in the student notebook.
   - Verifies that prior dependencies (Modules 01 through $N-1$) remain stable before testing Module $N$.
   - Validates student-implemented functions against progressive checkpoints.

2. **Core Unit Tests (`test_<name>_core.py`)**:
   - Tests every exported class and function in isolation.
   - Uses the educational docstring format (`WHAT`, `WHY`, `STUDENT LEARNING`).
   - Asserts mathematical correctness against exact values and known closed forms.

3. **In-Module Contract Tests (`test_<name>_contracts.py` or `test_<name>_integration.py`)**:
   - Validates intra-module contracts, edge cases, and immediate dependency seams.
   - Examples: `test_batch_contract.py` (DataLoader), `test_batched_matmul_backward.py` (Autograd), `test_spatial_contracts.py` (Convolutions), `test_transformer_contracts.py` (Transformers).

---

## 🔄 Three-Phase Testing Workflow (`tito module test`)

When a student runs `tito module test <NN>`, `tito` executes a three-phase validation pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Inline Unit Tests                                  │
│ Fast sanity checks embedded directly in the notebook cells  │
└──────────────────────────────┬──────────────────────────────┘
                               │ passes
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Module Tests (Educational Mode)                    │
│ Pytest runs tests/NN_<modulename>/ with --tinytorch flag    │
└──────────────────────────────┬──────────────────────────────┘
                               │ passes
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Cumulative Cross-Module Integration Tests          │
│ Runs all integration tests unlocked up through Module NN    │
└─────────────────────────────────────────────────────────────┘
```

- **Phase 1 (Inline Unit Tests)**: Validates that code inside the notebook's solution cells executes and passes immediate assertions.
- **Phase 2 (Module Tests)**: Runs pytest on the module's test folder. If any test fails, the runner displays the `WHAT` and `WHY` docstrings to explain the failure.
- **Phase 3 (Cumulative Integration Tests)**: Dispatches integration tests mapped in `tito/commands/module/test.py:integration_test_map`. As students advance, they inherit integration tests from earlier modules, guaranteeing that changes to later modules do not break earlier foundations.

---

## 💡 Educational Mode (`--tinytorch`)

Passing the `--tinytorch` flag to pytest activates the educational test reporter defined in [`conftest.py`](conftest.py).
When a test fails in educational mode, pytest prints the structured pedagogical purpose of the test instead of a raw stack trace:

```python
def test_tensor_addition(self):
    """
    WHAT: Element-wise tensor addition.

    WHY: Addition is used everywhere in neural networks:
    - Adding bias to layer output: y = Wx + b
    - Residual connections: output = layer(x) + x

    STUDENT LEARNING: Operations return new Tensors (functional style).
    """
    ...
```

The plugin extracts `WHAT:`, `WHY:`, and `STUDENT LEARNING:` (or `HOW:`) sections to show learners why the concept matters and where to look to fix it.

---

## 🏃 Running Tests

### Fast Developer Test Suite (Recommended)
Skips slow training loops and long benchmark runs (< 2 minutes, ~1,400 tests):
```bash
pytest -m "not slow" -q
```

### Full Test Suite
Runs all tests including full milestone training scripts:
```bash
pytest tests/
```

### Single Module Test Suite
Run tests for a single module with educational output:
```bash
pytest tests/06_autograd/ --tinytorch
```
Or via the `tito` CLI:
```bash
tito module test 06
```

### Cross-Module Integration Seams
```bash
pytest tests/integration/ -v
```

### Pre-Release Quality Gates
Evaluates all 33 architectural, pedagogical, and structural release gates:
```bash
python tools/release_check.py --fast
```

---

## 🎯 Authoring Rules: What Makes a Valid Test

Every test in TinyTorch earns its place by being able to catch real defects. Three core rules apply:

### 1. Assert the Value, Not Just the Shape
Shape checks pass for transposed matrix products, inverted broadcast dimensions, and subtractions that lost their sign. Always assert numerical values against hand-calculated results, analytical derivatives, or NumPy reference computations:
```python
# ❌ Bad: shape check passes for wrong gradients
assert x.grad.shape == (2, 3)

# ✅ Good: asserts exact numerical values
np.testing.assert_allclose(x.grad.data, expected_grad, rtol=1e-5, atol=1e-7)
```

### 2. Never Swallow Failures
- No bare `except: pass` or `except Exception: print(...)`.
- If a test cannot fail when the code is broken, it is not a test.
- Two release gates (`g_no_bare_return`, `g_no_bare_except`) enforce this statically.

### 3. No Mock Implementations in Integration Tests
- Integration tests must import and exercise real components from `tinytorch.core.*` and `tinytorch.perf.*`.
- Do not create mock layers (`class MockSequential: ...`) in integration tests; test the actual components students build.

### 4. Deterministic RNG
- Tests must be order-independent.
- Module-level random number generators are reseeded automatically before every test via the `_reset_module_rngs` autouse fixture in `conftest.py`.

---

## 🔍 Critical Watchpoints

- **[`integration/test_integration_gradient_flow.py`](integration/test_integration_gradient_flow.py)**: The central health check for autograd and backpropagation. If gradients stop transmitting through stacked layers, the entire framework is broken regardless of unit test results.
- **[`milestones/test_milestones_smoke.py`](milestones/test_milestones_smoke.py)**: Validates that historical milestone models (Perceptron, XOR, MLP, CNN, Transformer, MLPerf) instantiate and run forward passes cleanly against exported code.
- **[`../tools/release_check.py`](../tools/release_check.py)**: The ultimate arbiter of release readiness across all 20 modules.
