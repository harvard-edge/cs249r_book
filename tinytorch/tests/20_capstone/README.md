# Capstone Integration Tests - Module 20

This directory contains comprehensive integration tests for the **Capstone module**, which validates the ENTIRE 100+ hour TinyTorch learning journey.

## Overview

The capstone tests verify that all 19 previous modules work together to build production-ready ML systems. This is the most important test suite in TinyTorch.

## Test Coverage

### Priority 1: Complete ML Pipeline (CRITICAL)
- **test_complete_ml_pipeline_end_to_end**: Full data → model → training → evaluation workflow
- Validates: Modules 01-08 integration

### Priority 2: Model Architecture
- **test_mlp_architecture_integration**: Multi-layer perceptron with all components
- **test_cnn_architecture_integration**: CNN with Conv2d, pooling, flatten
- **test_transformer_architecture_integration**: Attention, embeddings, positional encoding
- Validates: Modules 01-03, 09, 11-12 integration

### Priority 3: Training Convergence
- **test_xor_convergence**: Classic XOR problem (non-linearly separable)
- **test_binary_classification_convergence**: Real binary classification task
- Validates: Training pipeline actually learns

### Priority 4: Inference Pipeline
- **test_inference_pipeline**: Trained model performs inference correctly
- Validates: Deployment readiness

### Priority 5: Optimization & Deployment
- **test_quantization_pipeline**: INT8 quantization for deployment
- **test_pruning_pipeline**: Weight pruning for compression
- **test_combined_optimization_deployment**: Quantization + pruning together
- Validates: Modules 16-17 optimization techniques

### Priority 6: Gradient Flow
- **test_deep_network_gradient_flow**: Gradients flow through all layer types
- **test_gradient_accumulation_correctness**: Shared parameters accumulate gradients
- Validates: Module 06 autograd across all modules

### Priority 7: Memory & Performance
- **test_memory_efficiency**: Memory usage is reasonable
- **test_training_performance**: Training speed meets expectations
- Validates: System efficiency

## Running Tests

### Run all capstone tests:
```bash
python tests/20_capstone/test_capstone_integration.py
```

### Run with pytest:
```bash
pytest tests/20_capstone/test_capstone_integration.py -v
```

### Run specific test class:
```bash
pytest tests/20_capstone/test_capstone_integration.py::TestCompleteMLPipeline -v
```

## Current Status

**Total Tests**: 14 comprehensive integration tests
- **Passing**: 1 (Memory Efficiency)
- **Framework Bugs**: 8 (optimizer/gradient issues - not test bugs)
- **Skipped**: 5 (components not yet implemented)

### Known Framework Issues (Not Test Issues)

The following tests expose real bugs in the TinyTorch framework:

1. **Optimizer bug**: `unsupported operand type(s) for *: 'float' and 'memoryview'`
   - Affects: SGD, Adam optimizers
   - Impact: Training loops fail
   - Tests affected: 6 tests

2. **Gradient accumulation bug**: `Cannot cast ufunc 'add' output from dtype('O') to dtype('float32')`
   - Affects: Backward pass with multiple uses
   - Impact: Shared parameters don't work
   - Tests affected: 2 tests

3. **Missing gradient tracking**: Gradients not computed for some layers
   - Affects: Deep networks
   - Impact: Some layers don't get gradients
   - Tests affected: 1 test

## Test Philosophy

These tests follow **production ML workflow patterns**:

1. **Data Creation** → Representative datasets (not toy examples)
2. **Model Building** → Real architectures (MLP, CNN, Transformer)
3. **Training** → Actual convergence (loss decreases, accuracy improves)
4. **Evaluation** → Real metrics (accuracy, loss reduction)
5. **Optimization** → Production techniques (quantization, pruning)
6. **Validation** → Strong assertions (models must actually learn)

## Expected Behavior After Framework Fixes

Once the framework bugs are fixed, all 14 tests should:

1. **Pass completely** (no skips due to implementation)
2. **Run in < 60 seconds** (performance test validates this)
3. **Demonstrate learning** (loss decreases, accuracy improves)
4. **Validate integration** (all modules work together)

## Adding New Capstone Tests

When adding new tests, follow this pattern:

```python
class TestNewCapability:
    """
    Tests new ML capability integration.
    Validates Modules X, Y, Z work together.
    """

    def test_capability_name(self):
        """Test specific capability works end-to-end."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required imports not available")

        print("\\n" + "="*80)
        print("CAPSTONE TEST X: CAPABILITY NAME")
        print("="*80)

        # 1. Setup (data, model, optimizer)
        # 2. Training loop
        # 3. Validation with strong assertions
        # 4. Print clear success message

        assert strong_condition, "Descriptive error message"

        print("✅ Capability test passed!")
        print("="*80)
```

## Success Criteria

For capstone tests to pass, students must have:

1. **Built all 19 modules correctly**
2. **Integrated modules properly** (no breaking changes)
3. **Implemented autograd correctly** (gradients flow everywhere)
4. **Created working optimizers** (parameters update properly)
5. **Validated on real tasks** (models actually learn)

This validates the **100+ hour learning journey is complete and successful**.

## What This Tests That Unit Tests Don't

| Aspect | Unit Tests | Capstone Tests |
|--------|------------|----------------|
| Scope | Single module | All 19 modules together |
| Integration | Module isolation | Cross-module integration |
| Real workflows | Synthetic checks | Production ML pipelines |
| Learning | Correctness only | Models must converge |
| Performance | Not tested | Memory & speed validated |
| Deployment | Not tested | Quantization, pruning tested |

## Framework Maintainers

If capstone tests fail:

1. **Check unit tests first** - Individual modules should pass
2. **Fix integration bugs** - Tests expose real framework issues
3. **Don't modify tests** - Tests define correct behavior
4. **Fix the framework** - Make TinyTorch match production ML patterns

The capstone tests are **specification tests** - they define what must work for students to succeed.
