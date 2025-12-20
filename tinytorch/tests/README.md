# TinyTorch Test Suite

Comprehensive testing organized by purpose and scope.

## Test Organization

### 📦 Module Tests (`XX_modulename/`)
**Purpose**: Test individual module functionality
**Scope**: Single module, isolated behavior
**Example**: `01_tensor/test_progressive_integration.py`

These tests validate that each module works correctly in isolation.

### 🔗 Integration Tests (`integration/`)
**Purpose**: Test cross-module interactions
**Scope**: Multiple modules working together
**Files**:
- `test_gradient_flow.py` - **CRITICAL**: Validates gradients flow through entire training stack
- `test_end_to_end_training.py` - Full training loops (TODO)
- `test_module_compatibility.py` - Module interfaces (TODO)

**Why this matters**:
- Catches bugs that unit tests miss
- Validates the "seams" between modules
- Ensures training actually works end-to-end

### 🐛 Debugging Tests (`debugging/`)
**Purpose**: Catch common student pitfalls
**Scope**: Pedagogical - teaches debugging
**Files**:
- `test_gradient_vanishing.py` - Detect/diagnose vanishing gradients (TODO)
- `test_gradient_explosion.py` - Detect/diagnose exploding gradients (TODO)
- `test_common_mistakes.py` - "Did you forget backward()?" style tests (TODO)

**Philosophy**: When these tests fail, the error message should teach the student what went wrong and how to fix it.

### ⚡ Autograd Edge Cases (`06_autograd/`)
**Purpose**: Stress-test autograd system
**Scope**: Autograd internals and edge cases
**Files**:
- `test_broadcasting.py` - Broadcasting gradient bugs (TODO)
- `test_computation_graph.py` - Graph construction edge cases (TODO)
- `test_backward_edge_cases.py` - Numerical stability, etc. (TODO)

## Running Tests

### Standard Mode
```bash
pytest tests/ -v                    # All tests
pytest tests/integration/ -v        # Integration tests only
pytest tests/01_tensor/ -v          # Specific module
```

### 🎓 Educational Mode (Recommended for Students)
```bash
pytest tests/ --tinytorch           # Rich output with WHAT/WHY context
pytest tests/01_tensor/ --tinytorch # Single module with education
```

**Educational mode shows:**
- Module groupings before running
- What each test does (WHAT)
- Why it matters (WHY)
- Learning tips on failure (STUDENT LEARNING)
- Clear pass/fail indicators with Rich formatting

### Run without pytest
```bash
python tests/integration/test_gradient_flow.py
```

## Test Philosophy

1. **Integration tests catch real bugs**: The gradient flow test caught the exact bugs that prevented training
2. **Descriptive names**: Test names should explain what they test
3. **Good error messages**: When tests fail, students should understand why
4. **Pedagogical value**: Tests teach correct usage patterns

## Educational Test Docstrings

All `*_core.py` test files use a structured docstring format:

```python
def test_tensor_addition(self):
    """
    WHAT: Element-wise tensor addition.

    WHY: Addition is used everywhere in neural networks:
    - Adding bias to layer output: y = Wx + b
    - Residual connections: output = layer(x) + x

    STUDENT LEARNING: Operations return new Tensors (functional style).
    """
```

This format enables the `--tinytorch` flag to show educational context when tests run.

## Adding New Tests

When adding a test, ask:
- **Is it testing one module?** → Put in `XX_modulename/`
- **Is it testing modules working together?** → Put in `integration/`
- **Is it teaching debugging?** → Put in `debugging/`
- **Is it an autograd edge case?** → Put in `06_autograd/`

## Most Important Tests

🔥 **Must pass before merging**:
- `integration/test_gradient_flow.py` - If this fails, training is broken

📚 **Module validation**:
- Each module's inline tests (in `modules/`)
- Module-specific tests in `tests/XX_modulename/`

## Test Coverage Goals

- ✅ All tensor operations have gradient tests
- ✅ All layers compute gradients correctly
- ✅ All activations integrate with autograd
- ✅ All loss functions compute gradients
- ✅ All optimizers update parameters
- ⏳ End-to-end training converges (TODO)
- ⏳ Common pitfalls are detected (TODO)
