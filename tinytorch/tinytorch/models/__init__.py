"""
TinyTorch Models
================
Canonical neural network architectures assembled from TinyTorch LEGO bricks.
"""

try:
    from .transformer import GPT, TinyGPT
except ImportError:
    GPT = TinyGPT = None

__all__ = ["GPT", "TinyGPT"]
