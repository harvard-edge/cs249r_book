# Milestone 04: The CNN Revolution (1998)

```{tip} What You'll Learn
- Why spatial structure matters: 100× fewer parameters, 50% better accuracy
- How weight sharing enables translation invariance
- The hierarchical feature learning that powers all computer vision
```

## Overview

1998. The US Postal Service processes millions of handwritten checks daily - all by hand. Then Yann LeCun deploys LeNet-5: a convolutional neural network that reads zip codes with superhuman accuracy. It's not a research demo - it's processing real checks, in production, saving millions of dollars.

The breakthrough? LeCun realized images have STRUCTURE. Nearby pixels matter more than distant ones. The same edge detector works in the corner or the center. By exploiting these insights - local connectivity and weight sharing - CNNs achieve better accuracy with 100× fewer parameters.

You're about to prove those same principles work using YOUR spatial implementations on real natural images. If you hit 75% on CIFAR-10, you've built computer vision that would have been publishable a decade ago.

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
| **09** | **Convolutions** | **YOUR Conv2d + MaxPool2d** |

## Running the Milestone

Before running, ensure you have completed Modules 01-09. You can check your progress:

```bash
tito module status
```

```bash
cd milestones/04_1998_cnn

# Part 1: TinyDigits (works offline)
python 01_lecun_tinydigits.py
# Expected: ~90% accuracy (vs ~80% MLP)

# Part 2: CIFAR-10 (requires download)
python 02_lecun_cifar10.py
# Expected: 65-75% accuracy
```

## Expected Results

| Script | Dataset | Architecture | Accuracy | vs MLP |
|--------|---------|--------------|----------|--------|
| 01 (TinyDigits) | 1K train, 8x8 | Simple CNN | ~90% | +10% improvement |
| 02 (CIFAR-10) | 50K train, 32x32 RGB | Deeper CNN | 65-75% | MLPs struggle here |

## The Aha Moment: Structure Matches Reality

An MLP sees an image as 3,072 random numbers. It doesn't know pixel (0,0) is next to pixel (0,1). It learns brittle patterns like "if pixel 1,234 is bright AND pixel 2,891 is dark..." - unscalable and fragile.

A CNN understands spatial structure:

1. **Local Connectivity**: Each neuron only looks at nearby pixels (3×3 or 5×5 regions). Edges, corners, textures are all LOCAL patterns.
2. **Weight Sharing**: The SAME filter detects edges everywhere. "Cat in top-left" and "cat in bottom-right" use the same feature detector.
3. **Translation Invariance**: The network doesn't care WHERE the cat appears - only THAT it appears.

**The result?**
- MLP on CIFAR-10: ~100M parameters, ~50% accuracy (barely better than random)
- YOUR CNN: ~1M parameters, 75%+ accuracy (real computer vision)

**100× fewer parameters. 50% better accuracy.** That's what happens when architecture matches reality.

Part 1 validates YOUR implementations on TinyDigits. But Part 2 is where everything comes together at scale. 50,000 natural color images. 32×32×3 = 3,072 dimensions per image. 10 diverse categories (airplanes, cars, birds, cats, ships...). This is HARD.

Watch YOUR DataLoader stream batches from disk. Watch YOUR Conv2d layers extract features: first layer finds edges, second layer finds textures, third layer finds object parts. Watch YOUR MaxPool2d reduce dimensions while preserving features.

When you see "Test Accuracy: 72%," realize what just happened: YOUR implementations, running on YOUR computer, just achieved computer vision that rivals early ImageNet-era results. You didn't download a pretrained model. You built the framework AND trained the model. That's systems engineering.

## YOUR Code Powers This

| Component | Your Module | What It Does |
|-----------|-------------|--------------|
| `Tensor` | Module 01 | Stores images and feature maps |
| `Conv2d` | Module 09 | YOUR convolutional layers |
| `MaxPool2d` | Module 09 | YOUR pooling layers |
| `ReLU` | Module 02 | YOUR activation functions |
| `Linear` | Module 03 | YOUR classifier head |
| `CrossEntropyLoss` | Module 04 | YOUR loss computation |
| `DataLoader` | Module 05 | YOUR batching pipeline |
| `backward()` | Module 06 | YOUR autograd engine |

## Historical Context

LeNet-5 was deployed by the US Postal Service for handwritten zip code recognition, processing millions of checks. This proved neural networks could work in production.

CIFAR-10 (2009) became the standard benchmark before ImageNet. CNNs achieving 70%+ on CIFAR demonstrated the architecture was ready for larger challenges.

The 2012 "ImageNet moment" (AlexNet) used the same CNN principles, just scaled up with GPUs.

## Systems Insights

- **Memory**: ~1M parameters (weight sharing dramatically reduces vs dense)
- **Compute**: Convolution is compute-intensive but highly parallelizable
- **Architecture**: Hierarchical feature learning (edges → textures → objects)

## What's Next

CNNs excel at vision, but what about sequences (text, audio, time series)? Milestone 05 introduces **Transformers** - the architecture that unified vision AND language.

## Further Reading

- **LeNet Paper**: LeCun et al. (1998). "Gradient-based learning applied to document recognition"
- **CIFAR-10**: Krizhevsky (2009). "Learning Multiple Layers of Features from Tiny Images"
- **AlexNet**: Krizhevsky, A., Sutskever, I., & Hinton, G. (2012). "ImageNet Classification with Deep Convolutional Neural Networks"
