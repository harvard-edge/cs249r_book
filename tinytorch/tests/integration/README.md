# Integration Tests

## Philosophy

Integration tests catch bugs that **unit tests miss** - specifically bugs at **module boundaries** where one module's output becomes another module's input.

### The Gradient Flow Pattern

The gold standard is `test_integration_gradient_flow.py`. It verifies:
1. **Gradients exist** (not None)
2. **Gradients are non-zero** (actually computed)
3. **Gradients flow through each layer** (chain not broken)
4. **Training actually works** (loss decreases)

This pattern catches the most common and frustrating bugs students encounter.

## Test Categories

### 🔥 Critical (Must Pass)

<table width="100%">
  <thead>
<tr>
<th width="35%"><b>Test File</b></th>
<th width="45%">What It Catches</th>
<th width="20%">Modules</th>
</tr>
</thead>
<tbody>
<tr><td><b>`test_integration_gradient_flow.py`</b></td><td>Broken backpropagation</td><td>01-08</td></tr>
<tr><td><b>`test_training_flow.py`</b></td><td>Training loop failures</td><td>05-07</td></tr>
<tr><td><b>`test_nlp_pipeline_flow.py`</b></td><td>NLP stack issues</td><td>10-13</td></tr>
<tr><td><b>`test_cnn_integration.py`</b></td><td>CNN gradient issues</td><td>09</td></tr>
</tbody>
</table>

### 📋 Standard (Should Pass)

<table width="100%">
  <thead>
<tr>
<th width="45%"><b>Test File</b></th>
<th width="35%">What It Catches</th>
<th width="20%">Modules</th>
</tr>
</thead>
<tbody>
<tr><td><b>`test_dataloader_integration.py`</b></td><td>Data pipeline issues</td><td>05</td></tr>
<tr><td><b>`test_module_dependencies.py`</b></td><td>Module dependency drift</td><td>All</td></tr>
<tr><td><b>`test_optimizers_integration.py`</b></td><td>Optimizer/training interactions</td><td>06-08</td></tr>
</tbody>
</table>

### 🔬 Scenario Tests

These test complete use cases:
- `test_xor_thorough.py` - XOR learning (classic test)
- `test_cnn_integration.py` - CNN on images
- `test_nlp_pipeline_flow.py` - Language model pipeline flow
- `test_training_capabilities.py` - End-to-end training capabilities

## What Makes a Good Integration Test

### ✅ Good Integration Test
```python
def test_gradients_flow_through_mlp():
    """Gradients must reach all layers"""
    layers = [Linear(4, 4) for _ in range(5)]

    x = Tensor(np.random.randn(1, 4), requires_grad=True)
    h = x
    for layer in layers:
        h = relu(layer(h))
    loss = mse_loss(h, target)
    loss.backward()

    # ALL layers must have gradients
    for i, layer in enumerate(layers):
        assert layer.weight.grad is not None, f"Layer {i} has no gradient!"
```

**Why it's good:**
- Tests the **boundary** between layers
- Catches gradient chain breaks
- Clear error message tells you WHERE it broke

### ❌ Bad Integration Test
```python
def test_linear_layer():
    """Test linear layer works"""
    layer = Linear(2, 3)
    x = Tensor([[1, 2]])
    y = layer(x)
    assert y.shape == (1, 3)
```

**Why it's bad:**
- This is a **unit test**, not integration
- Doesn't test interaction with other modules
- Belongs in `tests/03_layers/`

## Running Tests

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run only gradient flow tests
pytest tests/integration/test_integration_gradient_flow.py -v

# Run only training flow tests
pytest tests/integration/test_training_flow.py -v

# Run quick smoke tests (for CI)
pytest tests/integration/ -v -k quick

# Run with detailed output on failure
pytest tests/integration/ -v --tb=long
```

## Adding New Integration Tests

When adding a new module (e.g., Module 14: Profiling), ask:

1. **What other modules does it interact with?**
   - Profiling interacts with training loops (07) and models (03)

2. **What could break at the boundary?**
   - Profiling hooks might interfere with autograd
   - Timing might change tensor operations

3. **Write a test that exercises the boundary:**
```python
def test_profiling_does_not_break_training():
    """Profiling should not interfere with gradient flow"""
    with profiler.profile():
        loss = model(x)
        loss.backward()  # Should still work!

    assert model.weight.grad is not None
```

## Coverage Gaps

### Currently Missing

<table width="100%">
  <thead>
<tr>
<th width="30%"><b>Module</b></th>
<th width="70%">Integration Test Needed</th>
</tr>
</thead>
<tbody>
<tr><td><b>14 Profiling</b></td><td>Profiler + training loop</td></tr>
<tr><td><b>15 Quantization</b></td><td>Quantized model accuracy</td></tr>
<tr><td><b>16 Compression</b></td><td>Compressed model still trains</td></tr>
<tr><td><b>17 Acceleration</b></td><td>Accelerated ops match baseline</td></tr>
<tr><td><b>18 Memoization</b></td><td>Full model-level generation with cache enabled</td></tr>
</tbody>
</table>

### How to Fill Gaps

For each gap, create a test that:
1. Uses the module in a **realistic scenario**
2. Verifies **correctness** (not just "doesn't crash")
3. Checks **boundaries** with connected modules
