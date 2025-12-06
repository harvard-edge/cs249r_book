# Module 01: Tensor Foundation

## Overview
Build the foundational Tensor class that powers all machine learning operations in TinyTorch.

## Time Estimate
**2-3 hours**

## Difficulty
⭐⭐☆☆☆ (Beginner)

## Prerequisites
- **Python basics**: Variables, functions, classes, operators
- **NumPy fundamentals**: Array creation, indexing, basic operations
- **Linear algebra**: Matrix multiplication concept, vectors vs matrices

## Learning Outcomes

By completing this module, you will be able to:

1. **Implement a complete Tensor class** with arithmetic operations (+, -, *, /), matrix multiplication, and shape manipulation that mirrors PyTorch's design patterns

2. **Understand tensor broadcasting semantics** and how automatic shape alignment enables efficient batch processing across different dimensional data

3. **Design classes with dormant features** that activate in future modules, learning PyTorch's evolution from Variable to unified Tensor with built-in autograd

4. **Analyze memory layout and cache behavior** to understand why certain operations (row-wise access) are significantly faster than others (column-wise access)

5. **Build production-ready APIs** with proper error handling, clear error messages, and input validation that guides users toward correct usage

## Key Concepts

### Tensors: The Universal ML Data Structure
Tensors are multi-dimensional arrays that serve as the fundamental data structure in machine learning:
- **0D (scalar)**: Single number (e.g., loss value)
- **1D (vector)**: List of numbers (e.g., bias terms)
- **2D (matrix)**: Grid of numbers (e.g., weight matrices, images)
- **3D+**: Higher dimensions (e.g., batches of images, sequence data)

### Broadcasting: Automatic Shape Alignment
NumPy-style broadcasting automatically aligns tensors of different shapes for operations:
```python
matrix = [[1, 2], [3, 4]]  # Shape: (2, 2)
vector = [10, 20]           # Shape: (2,)
result = matrix + vector    # Broadcasting: (2,2) + (2,) → (2,2)
# Result: [[11, 22], [13, 24]]
```

### Memory Layout and Cache Effects
Understanding row-major (C-style) storage explains why sequential access is faster:
- **Row-wise access**: Sequential memory, excellent cache locality (~2-3× faster)
- **Column-wise access**: Strided memory, poor cache locality
- **Real impact**: Same O(n) algorithm, dramatically different wall-clock time

### Dormant Gradient Features
Our Tensor includes gradient tracking attributes (`requires_grad`, `grad`, `backward()`) from the start, but they remain inactive until Module 05. This design:
- Maintains consistent API throughout the course (no Variable vs Tensor confusion)
- Follows PyTorch 2.0's unified Tensor design
- Enables progressive disclosure of complexity

## Module Structure

1. **Introduction**: What is a Tensor? (Concept + ML context)
2. **Foundations**: Mathematical Background (Broadcasting, memory layout)
3. **Implementation**: Building Tensor class with immediate unit testing
4. **Integration**: Neural network layer simulation
5. **Systems Analysis**: Memory layout and cache performance
6. **Module Test**: Comprehensive validation

## What You'll Build

```python
# Your complete Tensor class will support:
x = Tensor([[1, 2, 3], [4, 5, 6]])
y = Tensor([[7, 8, 9], [10, 11, 12]])

# Arithmetic operations with broadcasting
z = x + y              # Element-wise addition
scaled = x * 2         # Scalar broadcasting
normalized = (x - x.mean()) / x.std()  # Chaining operations

# Matrix operations
W = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
output = x.matmul(W)   # Matrix multiplication: (2,3) @ (3,2) → (2,2)

# Shape manipulation
reshaped = x.reshape(3, 2)     # (2,3) → (3,2)
transposed = x.transpose()     # (2,3) → (3,2) with data rearrangement

# Reduction operations
total = x.sum()                # Sum all elements
col_means = x.mean(axis=0)     # Average per column
```

## Connection to Production ML

This module teaches patterns used in production frameworks:
- **PyTorch's Tensor class**: Same API design with unified gradients
- **NumPy broadcasting**: Industry-standard automatic shape alignment
- **Memory efficiency**: Row-major storage, cache-aware algorithms
- **Error handling**: Clear messages that guide users toward solutions

## Files in This Module

- `tensor_dev.py`: Your working implementation (Jupyter notebook format)
- `test_tensor.py`: Comprehensive test suite (run with pytest)
- `README.md`: This file

## Next Steps

After completing this module:

**→ Module 02: Activations**
- Build activation functions (ReLU, Sigmoid, GELU)
- Learn how nonlinearity enables neural networks to learn complex patterns
- Understand vanishing/exploding gradients through activation analysis

Your Tensor class becomes the foundation that all future modules build upon!

## Common Pitfalls to Avoid

1. **Matrix multiplication vs element-wise multiplication**
   - Use `.matmul()` or `@` for matrix multiplication (dot product)
   - Use `*` for element-wise multiplication (Hadamard product)

2. **Shape compatibility in broadcasting**
   - Inner dimensions must match for matmul: (M,K) @ (K,N) ✓
   - Broadcasting aligns from rightmost dimension
   - Clear error messages help debug shape mismatches

3. **Reshape vs transpose confusion**
   - Reshape: Same memory layout, different interpretation (fast, O(1))
   - Transpose: Data rearrangement in memory (slower, O(n))

4. **Gradient features are dormant**
   - `requires_grad`, `grad`, `backward()` exist but don't function yet
   - They activate in Module 05 - ignore them for now
   - Don't try to implement gradients manually

## Resources

- **NumPy documentation**: https://numpy.org/doc/stable/
- **PyTorch Tensor API**: https://pytorch.org/docs/stable/tensors.html
- **Broadcasting semantics**: https://numpy.org/doc/stable/user/basics.broadcasting.html

## Getting Help

If you're stuck:
1. Read the error messages carefully - they include hints
2. Check the ASCII diagrams in `tensor_dev.py` for visual explanations
3. Run unit tests individually to isolate issues
4. Review the module integration test for end-to-end examples

Happy learning! 🚀
