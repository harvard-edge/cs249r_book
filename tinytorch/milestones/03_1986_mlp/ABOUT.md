# Milestone 03: The MLP Revival (1986)

**FOUNDATION TIER** | Difficulty: 2/4 | Time: 45-90 min | Prerequisites: Modules 01-08

## Overview

In 1986, **Rumelhart, Hinton, and Williams** published "Learning representations by back-propagating errors," proving that multi-layer networks could be trained effectively on real-world data. This **ended the AI Winter** and launched modern deep learning.

This milestone recreates that breakthrough using YOUR TinyTorch on image classification.

## What You'll Build

Multi-layer perceptrons (MLPs) for digit recognition:
1. **TinyDigits**: Quick proof-of-concept on 8x8 images
2. **MNIST**: The classic benchmark (95%+ accuracy!)

```
Images --> Flatten --> Linear --> ReLU --> Linear --> ReLU --> Linear --> Classes
```

## Prerequisites

| Module | Component | What It Provides |
|--------|-----------|------------------|
| 01-04 | Foundation | Tensor, Activations, Layers, Losses |
| 05-07 | Training | Autograd, Optimizers, Training loops |
| 08 | DataLoader | YOUR batching and data pipeline |

## Running the Milestone

```bash
cd milestones/03_1986_mlp

# Part 1: Quick validation (3-5 min)
python 01_rumelhart_tinydigits.py
# Expected: 75-85% accuracy

# Part 2: Full MNIST benchmark (10-15 min)
python 02_rumelhart_mnist.py
# Expected: 94-97% accuracy
```

## Expected Results

| Script | Dataset | Parameters | Accuracy | Training Time |
|--------|---------|------------|----------|---------------|
| 01 (TinyDigits) | 1K train, 8x8 | ~2.4K | 75-85% | 3-5 min |
| 02 (MNIST) | 60K train, 28x28 | ~100K | 94-97% | 10-15 min |

## Key Learning

**Hidden layers learn useful representations automatically.** The network discovers edge detectors, curve patterns, and digit-specific features without being told what to look for.

This is **representation learning** - the foundation of deep learning's power:
- Manual feature engineering --> Automatic feature learning
- Domain expertise --> Data-driven discovery

## Systems Insights

- **Memory**: ~100K parameters for MNIST (reasonable for 1986 hardware)
- **Compute**: Dense matrix operations dominate training time
- **Architecture**: Each hidden layer learns increasingly abstract features

## Historical Context

MNIST (1998) became THE benchmark for evaluating learning algorithms. MLPs hitting 95%+ proved neural networks were viable for real problems.

The backpropagation paper has been cited over 50,000 times and is considered one of the most influential papers in computer science.

## What's Next

MLPs treat images as flat vectors, ignoring spatial structure. A 28x28 image has 784 pixels - the MLP doesn't know that pixel (0,0) is near pixel (0,1). Milestone 04 (CNN) shows why **convolutional** layers dramatically improve image recognition.

## Further Reading

- **The Backprop Paper**: Rumelhart, Hinton, Williams (1986). "Learning representations by back-propagating errors"
- **MNIST Dataset**: LeCun et al. (1998). "Gradient-based learning applied to document recognition"
- **Universal Approximation**: Cybenko (1989). "Approximation by superpositions of a sigmoidal function"
