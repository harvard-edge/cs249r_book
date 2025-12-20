# Progressive Testing Framework

## Philosophy

TinyTorch uses **progressive testing** - when you complete Module N, we verify:
1. **Module N works correctly** (your new implementation)
2. **Modules 1 to N-1 still work** (no regressions)
3. **Modules integrate properly** (components work together)

## Why Progressive Testing?

```
Module 01: Tensor        ← Foundation: if this breaks, everything breaks
Module 02: Activations   ← Builds on Tensor
Module 03: Layers        ← Uses Tensor + Activations
Module 04: Losses        ← Uses Tensor + Layers
Module 05: DataLoader    ← Data pipelines for training
Module 06: Autograd      ← Core: patches Tensor with gradient tracking
...and so on
```

When you're working on Module 06 (Autograd), a bug could:
- Break Autograd itself (Module 06 tests catch this)
- Break Tensor operations (Module 01 regression tests catch this)
- Break how Layers integrate with Autograd (integration tests catch this)

## Test Structure

Each module has three test categories:

### 1. Capability Tests (`test_XX_capabilities.py`)
**What**: Tests that the module provides its core functionality
**Educational Value**: Shows students exactly what they need to implement

```python
class TestLinearCapability:
    """
    🎯 LEARNING OBJECTIVE: Linear layer performs y = xW + b

    A Linear layer is the fundamental building block of neural networks.
    It applies a linear transformation to input data.
    """

    def test_linear_forward_computes_affine_transformation(self):
        """
        ✅ WHAT WE'RE TESTING: y = xW + b computation

        Your Linear layer should:
        1. Store weight matrix W of shape (in_features, out_features)
        2. Store bias vector b of shape (out_features,)
        3. Compute output = input @ W + b

        🔍 IF THIS FAILS: Check your forward() method
        """
```

### 2. Regression Tests (`test_XX_regression.py`)
**What**: Verifies earlier modules still work after changes
**Educational Value**: Teaches defensive programming and integration

```python
class TestModule05DoesNotBreakFoundation:
    """
    🛡️ REGRESSION CHECK: Ensure Autograd doesn't break earlier modules

    Autograd patches Tensor operations. This can accidentally break
    basic tensor functionality if not done carefully.
    """

    def test_tensor_creation_still_works(self):
        """After enabling autograd, basic tensor creation must still work"""

    def test_tensor_arithmetic_still_works(self):
        """After enabling autograd, tensor +, -, *, / must still work"""
```

### 3. Integration Tests (`test_XX_integration.py`)
**What**: Tests that modules work together correctly
**Educational Value**: Shows how ML systems connect

```python
class TestLayerAutogradIntegration:
    """
    🔗 INTEGRATION CHECK: Layers + Autograd work together

    Neural network training requires:
    - Layers compute forward pass
    - Loss measures error
    - Autograd computes gradients
    - Optimizer updates weights

    This tests the Layer ↔ Autograd connection.
    """
```

## Running Progressive Tests

```bash
# Test single module (also runs regression tests for earlier modules)
tito module test 05

# What actually runs:
# 1. Module 01 regression tests (is Tensor still OK?)
# 2. Module 02 regression tests (are Activations still OK?)
# 3. Module 03 regression tests (are Layers still OK?)
# 4. Module 04 regression tests (are Losses still OK?)
# 5. Module 05 capability tests (does DataLoader work?)
# 6. Module 06 capability tests (does Autograd work?)
# 6. Integration tests (do they all work together?)
```

## Educational Test Naming

Tests should be self-documenting:

```python
# ❌ BAD: Unclear what's being tested
def test_forward(self):

# ✅ GOOD: Clear learning objective
def test_forward_pass_produces_correct_output_shape(self):

# ✅ BETTER: Includes the concept being taught
def test_linear_layer_output_shape_is_batch_size_by_out_features(self):
```

## Failure Messages Should Teach

```python
# ❌ BAD: Unhelpful error
assert output.shape == expected, "Wrong shape"

# ✅ GOOD: Educational error message
assert output.shape == expected, (
    f"Linear layer output shape incorrect!\n"
    f"  Input shape: {input.shape}\n"
    f"  Weight shape: {layer.weight.shape}\n"
    f"  Expected output: {expected}\n"
    f"  Got: {output.shape}\n"
    f"\n"
    f"💡 HINT: For y = xW + b:\n"
    f"   x has shape (batch, in_features)\n"
    f"   W has shape (in_features, out_features)\n"
    f"   y should have shape (batch, out_features)"
)
```
