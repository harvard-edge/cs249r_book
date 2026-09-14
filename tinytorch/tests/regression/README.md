# TinyTorch Regression Tests
## Ensuring Core Infrastructure Works Correctly

This directory contains regression tests that ensure TinyTorch's core functionality works correctly so students don't get stuck on infrastructure issues.

---

## 📋 Test Coverage

### Shape Compatibility Tests
**File**: `test_conv_linear_dimensions.py`
**What it tests**: Convolution output dimensions match Linear layer expectations
**Why it matters**: Students shouldn't debug dimension mismatches in their CNNs

### Tensor Reshaping Tests
**File**: `test_transformer_reshaping.py`
**What it tests**: Transformer 3D outputs work with Linear 2D layers
**Why it matters**: Language model architectures should "just work"

---

## 🧪 Running Regression Tests

### Run All Regression Tests
```bash
pytest tests/regression/
```

### Run Specific Bug Test
```bash
pytest tests/regression/test_issue_20241125_conv_fc_shapes.py -v
```

### Run with Coverage
```bash
pytest tests/regression/ --cov=tinytorch --cov-report=html
```

---

## 📝 Adding New Regression Tests

When you discover a bug:

1. **Create Test File**: `test_issue_YYYYMMDD_description.py`

2. **Use Bug Tracking Template**:
```python
"""
BUG TRACKING:
============
Bug ID: BUG-YYYY-MM-DD-XXX
Date Found: YYYY-MM-DD
Found By: [Name/System]
Severity: [Critical/High/Medium/Low]

DESCRIPTION:
[What broke and under what conditions]

REPRODUCTION:
[Exact steps to reproduce]

ROOT CAUSE:
[Why it happened]

FIX:
[What was changed to fix it]

PREVENTION:
[How this test prevents recurrence]
"""
```

3. **Write Specific Test**: Test the EXACT scenario that failed

4. **Verify Test Catches Bug**:
   - Test should FAIL without the fix
   - Test should PASS with the fix

5. **Update This README**: Add entry to Bug Index

---

## 🎯 Testing Philosophy

**Every bug tells a story about a gap in our testing.**

When we find a bug, we ask:
1. Why didn't existing tests catch this?
2. What test would have prevented it?
3. Are there similar bugs we haven't found yet?

**The goal**: Build a test suite so comprehensive that bugs become impossible.

---

## 📊 Regression Test Statistics

- **Total Bugs Found**: 2
- **Bugs with Regression Tests**: 2 (100%)
- **Test Coverage**: 100% of discovered issues
- **Last Updated**: 2024-11-25

---

## 🔄 Integration with CI/CD

These regression tests run automatically on:
- Every commit to main branch
- Every pull request
- Nightly comprehensive test suite

Failures in regression tests block deployment to ensure fixed bugs never return.

---

## 🏆 Success Metrics

We measure success by:
1. **Zero Regressions**: No bug returns after being fixed
2. **Fast Detection**: Regression tests catch issues immediately
3. **Clear Documentation**: Every test explains the bug it prevents
4. **Continuous Growth**: New bugs always get new tests

---

## 📚 Learning from Bugs

Each bug teaches us something:

- **Conv Shape Mismatch**: Always calculate dimensions programmatically, never manually
- **Transformer Reshape**: Consider tensor dimensionality at module boundaries
- **[Future bugs will add lessons here]**

---

## 🚀 Future Improvements

- [ ] Add performance regression tests
- [ ] Create fuzz testing for edge cases
- [ ] Build automatic bug report generation
- [ ] Implement regression test metrics dashboard

---

Remember: **A bug fixed without a test is a bug waiting to return.**
