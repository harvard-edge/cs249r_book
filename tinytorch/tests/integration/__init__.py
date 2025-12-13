"""
Integration tests for TinyTorch.

These tests validate that multiple modules work together correctly.
They catch issues that unit tests miss, like:
- Gradient flow through entire training pipelines
- Module compatibility and interface contracts
- End-to-end training scenarios

Critical for catching bugs like:
- Missing autograd integration
- Shape mismatches in broadcasting
- Optimizer parameter updates
"""
