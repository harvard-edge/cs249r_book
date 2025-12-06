"""
Autograd-specific edge case tests.

These tests focus on the autograd module's internal behavior:
- Broadcasting in gradients (common bug source)
- Computation graph construction
- Numerical stability in backward pass
- Memory management in gradient accumulation
- Edge cases students encounter

Complements the inline tests in the autograd module with
focused edge case validation.
"""
