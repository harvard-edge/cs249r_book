"""
Progressive Testing Framework for TinyTorch

This module provides educational, progressive testing that:
1. Verifies module capabilities (what students implement)
2. Checks for regressions (earlier modules still work)
3. Tests integration (modules work together)

Tests are designed to be educational - failure messages teach students
what went wrong and how to fix it.
"""

from pathlib import Path

# Module dependencies - when testing Module N, also test these earlier modules
MODULE_DEPENDENCIES = {
    "01": [],                                    # Tensor has no dependencies
    "02": ["01"],                                # Activations need Tensor
    "03": ["01", "02"],                          # Layers need Tensor, Activations
    "04": ["01", "02", "03"],                    # Losses need Tensor, Activations, Layers
    "05": ["01"],                                # DataLoader mainly needs Tensor
    "06": ["01", "02", "03", "04", "05"],        # Autograd needs foundation + DataLoader
    "07": ["01", "02", "03", "04", "05", "06"],  # Optimizers need Autograd
    "08": ["01", "02", "03", "04", "05", "06", "07"],  # Training needs Optimizers
    "09": ["01", "02", "03", "06"],              # Convolutions needs Tensor, Layers, Autograd
    "10": ["01"],                                # Tokenization mainly needs Tensor
    "11": ["01", "06", "10"],                    # Embeddings need Tensor, Autograd, Tokenization
    "12": ["01", "03", "06", "11"],              # Attention needs Layers, Autograd, Embeddings
    "13": ["01", "03", "06", "11", "12"],        # Transformers need Attention
    "14": ["01"],                                # Profiling is mostly standalone
    "15": ["01", "03"],                          # Quantization needs Tensor, Layers
    "16": ["01", "03"],                          # Compression needs Tensor, Layers
    "17": ["01"],                                # Acceleration is mostly standalone
    "18": ["01", "12", "13"],                    # Memoization (KV-cache) needs Attention, Transformers
    "19": ["01"],                                # Benchmarking is mostly standalone
    "20": ["01", "02", "03", "04", "05", "06", "07", "08"],  # Capstone needs core modules
}

# What each module should provide (for capability testing)
MODULE_CAPABILITIES = {
    "01": {
        "name": "Tensor",
        "exports": ["Tensor"],
        "capabilities": [
            "Create tensors from lists and numpy arrays",
            "Perform element-wise operations (+, -, *, /)",
            "Perform matrix multiplication (matmul)",
            "Reshape and transpose tensors",
            "Support broadcasting",
        ],
    },
    "02": {
        "name": "Activations",
        "exports": ["Sigmoid", "ReLU", "Tanh", "GELU", "Softmax"],
        "capabilities": [
            "Apply non-linear transformations",
            "Preserve tensor shapes",
            "Handle batch dimensions",
        ],
    },
    "03": {
        "name": "Layers",
        "exports": ["Layer", "Linear", "Dropout"],
        "capabilities": [
            "Linear transformation: y = xW + b",
            "Xavier weight initialization",
            "Parameter collection for optimization",
        ],
    },
    "04": {
        "name": "Losses",
        "exports": ["MSELoss", "CrossEntropyLoss", "BinaryCrossEntropyLoss"],
        "capabilities": [
            "Compute scalar loss from predictions and targets",
            "Handle batch inputs",
            "Numerical stability (log-sum-exp trick)",
        ],
    },
    "05": {
        "name": "Autograd",
        "exports": ["enable_autograd"],
        "capabilities": [
            "Track computation graph",
            "Compute gradients via backpropagation",
            "Support requires_grad flag",
        ],
    },
    # ... continue for other modules
}


def get_dependencies(module_num: str) -> list:
    """Get list of modules that must work for module_num to work."""
    return MODULE_DEPENDENCIES.get(module_num, [])


def get_capabilities(module_num: str) -> dict:
    """Get capability information for a module."""
    return MODULE_CAPABILITIES.get(module_num, {})
