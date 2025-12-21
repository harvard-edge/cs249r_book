# Milestone 03: The MLP Revival (1986)

```{tip} What You'll Learn
- How networks automatically discover features (edges, patterns) you never programmed
- Why representation learning is the foundation of deep learning's power
- That YOUR code can achieve 95%+ accuracy on a real benchmark
```

## Overview

**For 17 years, neural networks were considered dead.**

After Minsky's XOR proof (Milestone 02), funding dried up, researchers moved on, and "neural network" became a dirty word in AI. The field was stuck in the AI Winter.

Then in 1986, **Rumelhart, Hinton, and Williams** published a single paper that changed everything: "Learning representations by back-propagating errors." They proved that multi-layer networks could learn *automatically*—no hand-crafted features, no expert rules. Just data in, patterns out.

This milestone recreates that breakthrough. You'll train YOUR TinyTorch implementation on real images and watch it discover features you never programmed.

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
| 05 | DataLoader | YOUR batching and data pipeline |
| 06-08 | Training Infrastructure | Autograd, Optimizers, Training loops |

## Running the Milestone

Before running, ensure you have completed Modules 01-08. You can check your progress:

```bash
tito module status
```

```bash
cd milestones/03_1986_mlp

# Part 1: Quick validation
python 01_rumelhart_tinydigits.py
# Expected: 75-85% accuracy

# Part 2: Full MNIST benchmark
python 02_rumelhart_mnist.py
# Expected: 94-97% accuracy
```

## Expected Results

| Script | Dataset | Parameters | Accuracy | Training Time |
|--------|---------|------------|----------|---------------|
| 01 (TinyDigits) | 1K train, 8x8 | ~2.4K | 75-85% | 3-5 min |
| 02 (MNIST) | 60K train, 28x28 | ~100K | 94-97% | 10-15 min |

## The Aha Moment: Automatic Feature Discovery

**Watch YOUR network learn something you never taught it.**

After training, examine the first hidden layer weights. You'll see edge detectors—horizontal, vertical, diagonal patterns. Nobody programmed these. The network discovered them because edges are useful for recognizing digits.

This is **representation learning**, the foundation of deep learning's power:
- Manual feature engineering → Automatic feature discovery
- Domain expertise → Data-driven patterns
- Hand-crafted rules → Emergent intelligence

**The moment you realize**: Your ~100 lines of TinyTorch code just replicated the breakthrough that ended the AI Winter.

## YOUR Code Powers This

Every component comes from YOUR implementations:

| Component | Your Module | What It Does |
|-----------|-------------|--------------|
| `Tensor` | Module 01 | Stores images and weights |
| `Linear` | Module 03 | YOUR fully-connected layers |
| `ReLU` | Module 02 | YOUR activation functions |
| `CrossEntropyLoss` | Module 04 | YOUR loss computation |
| `DataLoader` | Module 05 | YOUR batching pipeline |
| `backward()` | Module 06 | YOUR autograd engine |
| `SGD` | Module 07 | YOUR optimizer |

**No PyTorch. No TensorFlow. Just YOUR code learning to read handwritten digits.**

## Historical Context

MNIST (1998) became THE benchmark for evaluating learning algorithms. MLPs hitting 95%+ proved neural networks were viable for real problems.

The backpropagation paper has been cited over 50,000 times and is considered one of the most influential papers in computer science.

## Systems Insights

- **Memory**: ~100K parameters for MNIST (reasonable for 1986 hardware)
- **Compute**: Dense matrix operations dominate training time
- **Architecture**: Each hidden layer learns increasingly abstract features

## What's Next

MLPs treat images as flat vectors, ignoring spatial structure. A 28x28 image has 784 pixels - the MLP doesn't know that pixel (0,0) is near pixel (0,1). Milestone 04 (CNN) shows why **convolutional** layers dramatically improve image recognition.

## Further Reading

- **The Backprop Paper**: Rumelhart, Hinton, Williams (1986). "Learning representations by back-propagating errors"
- **MNIST Dataset**: LeCun et al. (1998). "Gradient-based learning applied to document recognition"
- **Universal Approximation**: Cybenko (1989). "Approximation by superpositions of a sigmoidal function"
