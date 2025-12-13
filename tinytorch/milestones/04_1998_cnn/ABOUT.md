# Milestone 04: The CNN Revolution (1998)

**ARCHITECTURE TIER** | Difficulty: 3/4 | Time: 60-120 min | Prerequisites: Modules 01-09

## Overview

In 1998, **Yann LeCun's LeNet-5** revolutionized computer vision with Convolutional Neural Networks (CNNs). By exploiting spatial structure through shared weights and local connectivity, CNNs achieved unprecedented accuracy with 100x fewer parameters than equivalent MLPs.

This milestone recreates that revolution using YOUR spatial implementations.

## What You'll Build

CNNs that exploit image structure:
1. **TinyDigits**: Prove convolution beats MLPs on 8x8 images
2. **CIFAR-10**: Scale to natural color images (32x32 RGB)

```
Images --> Conv --> ReLU --> Pool --> Conv --> ReLU --> Pool --> Flatten --> Linear --> Classes
```

## Prerequisites

| Module | Component | What It Provides |
|--------|-----------|------------------|
| 01-08 | Foundation + Training | Complete training pipeline |
| **09** | **Spatial** | **YOUR Conv2d + MaxPool2d** |

## Running the Milestone

```bash
cd milestones/04_1998_cnn

# Part 1: TinyDigits (works offline, 5-7 min)
python 01_lecun_tinydigits.py
# Expected: ~90% accuracy (vs ~80% MLP)

# Part 2: CIFAR-10 (requires download, 30-60 min)
python 02_lecun_cifar10.py
# Expected: 65-75% accuracy
```

## Expected Results

| Script | Dataset | Architecture | Accuracy | vs MLP |
|--------|---------|--------------|----------|--------|
| 01 (TinyDigits) | 1K train, 8x8 | Simple CNN | ~90% | +10% improvement |
| 02 (CIFAR-10) | 50K train, 32x32 RGB | Deeper CNN | 65-75% | MLPs struggle here |

## Key Learning

**Convolution exploits spatial structure.** Three principles make CNNs dominant for vision:

1. **Local Connectivity**: Only nearby pixels connect (instead of all-to-all)
2. **Weight Sharing**: Same filter detects features anywhere in the image
3. **Translation Invariance**: "Cat in corner" = "Cat in center"

The result: 100x fewer parameters with BETTER accuracy.

## Systems Insights

- **Memory**: ~1M parameters (weight sharing dramatically reduces vs dense)
- **Compute**: Convolution is compute-intensive but highly parallelizable
- **Architecture**: Hierarchical feature learning (edges --> textures --> objects)

## What Part 2 Showcases

- **YOUR DataLoader (Module 08)** batches 50,000 images efficiently
- **YOUR Conv2d + MaxPool2d (Module 09)** enable spatial intelligence
- First real test of your data pipeline at scale

## Historical Context

LeNet-5 was deployed by the US Postal Service for handwritten zip code recognition, processing millions of checks. This proved neural networks could work in production.

CIFAR-10 (2009) became the standard benchmark before ImageNet. CNNs achieving 70%+ on CIFAR demonstrated the architecture was ready for larger challenges.

The 2012 "ImageNet moment" (AlexNet) used the same CNN principles, just scaled up with GPUs.

## What's Next

CNNs excel at vision, but what about sequences (text, audio, time series)? Milestone 05 introduces **Transformers** - the architecture that unified vision AND language.

## Further Reading

- **LeNet Paper**: LeCun et al. (1998). "Gradient-based learning applied to document recognition"
- **CIFAR-10**: Krizhevsky (2009). "Learning Multiple Layers of Features from Tiny Images"
- **AlexNet**: Krizhevsky et al. (2012). "ImageNet Classification with Deep CNNs"
