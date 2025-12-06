# Module 05: Autograd - The Gradient Engine

**Time Estimate**: 3-4 hours
**Difficulty**: ⭐⭐⭐⭐☆
**Prerequisites**: Modules 01-04 must be complete

## Overview

Welcome to Module 05! This module brings gradients to life by implementing automatic differentiation (autograd). You'll enhance the existing Tensor class with backward() capabilities, build computation graphs, and implement the chain rule that makes neural networks trainable.

This is where the dormant gradient features from Module 01 (requires_grad, grad, backward) become fully functional!

## Learning Outcomes

By completing this module, you will:

1. **Understand Automatic Differentiation**
   - Grasp how computation graphs track operations for gradient flow
   - Understand reverse-mode differentiation (backpropagation)
   - See how the chain rule connects gradients through complex networks

2. **Implement Gradient Functions**
   - Build Function base class for differentiable operations
   - Implement backward passes for core operations (Add, Mul, Matmul, Sum)
   - Create gradient rules for activations (ReLU, Sigmoid, Softmax, GELU)
   - Implement loss function gradients (MSE, BCE, CrossEntropy)

3. **Master Computation Graphs**
   - Track parent operations for gradient propagation
   - Handle gradient accumulation for shared parameters
   - Manage memory during forward and backward passes

4. **Enhance Tensor with Autograd**
   - Implement the backward() method for reverse-mode differentiation
   - Enable gradient tracking via requires_grad flag
   - Handle gradient broadcasting and shape matching
   - Support zero_grad() for gradient reset between iterations

5. **Build Production-Ready Autograd**
   - Use monkey-patching to enhance existing Tensor operations
   - Maintain backward compatibility with previous modules
   - Follow PyTorch 2.0 style (single Tensor class, no Variable wrapper)

## Why Monkey Patching?

This module uses **monkey patching** to enhance the existing Tensor class with autograd capabilities. Here's why this approach is powerful and educational:

### What is Monkey Patching?

Monkey patching means dynamically modifying a class at runtime by replacing or adding methods after the class is already defined. In our case, we enhance Tensor's operations to track gradients.

**Before enable_autograd():**
```python
x = Tensor([2.0])
y = x * 3  # Simple multiplication, no gradient tracking
```

**After enable_autograd():**
```python
enable_autograd()  # Enhances Tensor class
x = Tensor([2.0], requires_grad=True)
y = x * 3  # Now tracks computation graph!
y.backward()  # Computes gradients
print(x.grad)  # [3.0]
```

### Why This Approach?

**Educational Benefits:**
- **Progressive Disclosure**: Module 01 introduces Tensor simply, Module 05 adds complexity
- **Single Mental Model**: One Tensor class that grows with student knowledge
- **No Confusion**: No separate Variable class like old PyTorch (pre-0.4)
- **Realistic**: Matches how PyTorch 2.0 actually works internally

**Technical Benefits:**
- **Backward Compatible**: All previous modules continue working unchanged
- **Opt-In Gradients**: Only tensors with requires_grad=True track graphs
- **Clean Separation**: Core operations in Module 01, gradients in Module 05
- **No Import Changes**: All existing code imports Tensor the same way

### The Pattern

```python
# 1. Store original operation
_original_add = Tensor.__add__

# 2. Create enhanced version
def tracked_add(self, other):
    result = _original_add(self, other)  # Call original
    if self.requires_grad or other.requires_grad:
        result.requires_grad = True
        result._grad_fn = AddBackward(self, other)  # Track computation
    return result

# 3. Replace operation
Tensor.__add__ = tracked_add
```

### PyTorch 2.0 Alignment

This follows PyTorch's actual design:
- ✅ **Single Tensor class** with built-in autograd
- ✅ **No Variable wrapper** (removed in PyTorch 0.4)
- ✅ **requires_grad flag** controls gradient tracking
- ✅ **Clean API** that's easy to understand and use

### Alternative Approaches (Why Not These?)

❌ **Subclassing (AutogradTensor extends Tensor)**: Creates two tensor types, confuses students
❌ **Variable wrapper (old PyTorch)**: Deprecated, adds complexity, harder to understand
❌ **Redefining Tensor**: Breaks previous modules, forces rewrites, creates inconsistency
❌ **Separate gradient system**: Requires manual wiring, defeats purpose of "automatic" differentiation

### What You'll Learn

The monkey patching pattern teaches:
- How to enhance existing code without breaking it
- How PyTorch actually implements autograd internally
- How to build production-ready ML systems with clean APIs
- How to progressively add complexity to educational systems

## Module Structure

### Part 1: Introduction
- What is automatic differentiation?
- Why computation graphs enable training
- Visualization of forward and backward passes

### Part 2: Foundations
- Mathematical chain rule
- Gradient flow through operations
- Memory layout during backpropagation

### Part 3: Implementation
- Function base class for differentiable operations
- Gradient rules for core operations (Add, Mul, Matmul, Sum, etc.)
- Activation gradients (ReLU, Sigmoid, Softmax, GELU)
- Loss function gradients (MSE, BCE, CrossEntropy)
- The enable_autograd() enhancement function

### Part 4: Integration
- Testing gradient correctness
- Multi-layer computation graphs
- Gradient accumulation patterns
- Complex operation chaining

### Part 5: Module Test & Summary
- Comprehensive integration testing
- Verification of all gradient functions
- End-to-end gradient flow validation

## Key Concepts

### Computational Graphs
```
Forward Pass:  x → Linear₁ → ReLU → Linear₂ → Loss
               (track operations)
Backward Pass: ∇x ← ∇Linear₁ ← ∇ReLU ← ∇Linear₂ ← ∇Loss
               (chain rule flows gradients)
```

### Chain Rule
For composite functions f(g(x)), the derivative is:
```
df/dx = (df/dg) × (dg/dx)
```

The autograd engine automatically applies this rule through the entire computation graph.

### Gradient Accumulation
When parameters appear multiple times in a computation (like shared embeddings), gradients accumulate:
```python
self.grad = self.grad + new_grad  # Not: self.grad = new_grad
```

### Memory Pattern
```
Computation Graph Memory:
┌─────────────────────────────────┐
│ Forward Pass (stored)           │
├─────────────────────────────────┤
│ x (leaf, requires_grad=True)    │
│ y = x * 2 (MulFunction)         │
│     saved: (x=..., 2)           │
│ z = y + 1 (AddFunction)         │
│     saved: (y=..., 1)           │
└─────────────────────────────────┘
         ↓ backward()
┌─────────────────────────────────┐
│ Backward Pass (compute grads)   │
├─────────────────────────────────┤
│ z.grad = 1 (initialized)        │
│ y.grad = 1 (from AddBackward)   │
│ x.grad = 2 (from MulBackward)   │
└─────────────────────────────────┘
```

## Testing Strategy

Each gradient function is tested immediately after implementation:
1. **Unit tests** verify individual operations compute correct gradients
2. **Integration tests** validate multi-layer computation graphs
3. **Edge cases** test gradient accumulation, broadcasting, and shape handling

## Common Pitfalls

1. **Forgetting zero_grad()**: Gradients accumulate by default
   ```python
   for batch in data:
       x.zero_grad()  # Reset gradients!
       loss = forward(x)
       loss.backward()
   ```

2. **Shape Mismatches**: Gradients must match tensor shapes
   - Broadcasting in forward requires "unbroadcasting" in backward

3. **Graph Retention**: Computation graphs consume memory
   - Clear graphs between iterations for long-running training

4. **Backward on Non-Scalars**: backward() requires gradient argument for non-scalar outputs
   ```python
   loss.backward()  # OK: loss is scalar
   y.backward(grad_output)  # Required: y is non-scalar
   ```

## Next Steps

After completing this module:
- **Module 06: Optimizers** - Use gradients to update parameters (SGD, Adam)
- **Module 07: Training** - Build complete training loops
- **Module 08: Spatial Operations** - Add Conv2d and Pooling with gradients

## Files

- `autograd_dev.py` - Your implementation workspace (Jupytext-compatible)
- `test_autograd.py` - Comprehensive test suite
- `README.md` - This file

## Export

When all tests pass:
```bash
tito module complete 05_autograd
```

This exports your implementation to `tinytorch.core.autograd` for use in future modules.

---

**Remember**: This module activates the gradient features that were dormant in Module 01. The Tensor class grows with your understanding - this is the power of progressive disclosure in educational systems!

Happy gradient tracking! ⚡
