# TinyTorch Regression Tests
## Ensuring Core Infrastructure Works Correctly

This directory contains regression tests that ensure TinyTorch's core functionality, past defect fixes, and cross-subsystem contracts remain reliable so students do not encounter infrastructure failures.

---

## 📋 Test Coverage

### Shape Compatibility Regressions
- **File**: `test_conv_linear_dimensions.py`
- **What it tests**: Convolution output dimensions match Linear layer expectations after flattening.
- **Why it matters**: Students should not debug dimension calculation mismatches in their CNN architectures.

### Tensor Reshaping Regressions
- **File**: `test_transformer_reshaping.py`
- **What it tests**: 3D sequence outputs from Transformer blocks feed cleanly into 2D Linear projection layers.
- **Why it matters**: Autoregressive LM head projections must compose without manual reshape boilerplate.

### Gradient Flow Bug Regressions
- **File**: `test_gradient_flow_fixes.py`
- **What it tests**: Specific backpropagation fixes from milestone implementation:
  1. Batched 3D matmul (`np.matmul` vs `np.dot`)
  2. `transpose()` preserving `requires_grad` and autograd graph history
  3. `SubBackward` and `DivBackward` autograd operations
  4. `Softmax` and `Dropout` executing via Tensor operations
  5. `Embedding` parameter gradient retention
  6. `MultiHeadAttention` batched operations without `.data` extraction
  7. `LayerNorm` maintaining autograd gradients
- **Why it matters**: Prevents silent gradient disconnection bugs from returning.

### NLP Components Gradient Flow
- **File**: `test_nlp_components_gradient_flow.py`
- **What it tests**: End-to-end gradient transmission through the NLP module stack:
  - Tokenization index validation
  - Embedding and PositionalEncoding gradient accumulation
  - Scaled dot-product attention with and without causal masks
  - Multi-head attention projection weight and bias gradients
  - LayerNorm and MLP gradient flow
  - Full TinyGPT training step parameter updates
- **Why it matters**: Verifies that every learnable tensor across Modules 10-13 receives non-zero gradients during training.

### Hardware Extension Parity and Acceleration
- **File**: `test_hardware_extensions.py`
- **What it tests**:
  - C++ SIMD GEMM (AVX2/NEON + OpenMP) numerical parity against NumPy reference
  - OpenAI Triton fused Bias + GELU kernel parity and CPU fallback
  - Apple Metal MPS matrix multiplication numerical parity
  - Dimension mismatch validation on hardware-accelerated kernels
- **Why it matters**: Ensures hardware acceleration produces numerically identical results to reference NumPy implementations.

### Capstone & Olympics Infrastructure
- **File**: `test_olympics.py`
- **What it tests**:
  - `SimpleMLP` architecture and parameter accounting
  - `BenchmarkReport` metrics calculation (latency, throughput, memory, accuracy)
  - `OlympicEvent` qualification threshold rules
  - Submission JSON schema generation, serialization, and round-trip validation
  - `tito olympics` CLI command execution
- **Why it matters**: Protects the final student benchmark harness and qualification scoring.

---

## 🧪 Running Regression Tests

### Run All Regression Tests
```bash
pytest tests/regression/ -v
```

### Run Specific Bug Regressions
```bash
pytest tests/regression/test_gradient_flow_fixes.py -v
```

---

## 📝 Adding New Regression Tests

When fixing a bug:

1. **Create Test File**: `test_issue_YYYYMMDD_description.py` (or add to `test_gradient_flow_fixes.py`).
2. **Document the Defect**: Include description, reproduction steps, root cause, and fix.
3. **Verify Test Catches Bug**: Ensure the test fails without the fix and passes with it.
4. **Update This README**: Add an entry under Test Coverage.
