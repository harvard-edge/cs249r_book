"""
Debugging tests for common student pitfalls.

These tests identify and diagnose common issues students encounter:
- Vanishing gradients (ReLU dying, sigmoid saturation)
- Exploding gradients (unstable initialization)
- Silent failures (forgot backward(), forgot zero_grad())
- Common mistakes (wrong loss function, learning rate issues)

Goal: When a test fails, the error message should guide students
to the solution. These are pedagogical tests that teach debugging.
"""
