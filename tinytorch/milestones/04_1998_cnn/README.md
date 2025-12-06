# Milestone 04: The CNN Revolution (1998)

## Historical Context

After backpropagation revived neural networks (1986), researchers still struggled with image recognition. MLPs treated pixels independently, requiring millions of parameters and ignoring spatial structure.

Then in 1998, **Yann LeCun's LeNet-5** revolutionized computer vision with **Convolutional Neural Networks (CNNs)**. By using:
- **Shared weights** (convolution) → 100× fewer parameters
- **Local connectivity** → preserves spatial structure
- **Pooling** → translation invariance

LeNet achieved 99%+ accuracy on handwritten digits, launching the deep learning revolution that led to ImageNet (2012), object detection, and modern computer vision.

## What You're Building

CNNs that exploit spatial structure in images:
1. **TinyDigits** - Prove convolution works on 8×8 digits
2. **CIFAR-10** - Scale to natural color images (32×32)

## Required Modules

**Run after Module 09** (Spatial operations: Conv2d + Pooling)

| Module | Component | What It Provides |
|--------|-----------|------------------|
| Module 01 | Tensor | YOUR data structure |
| Module 02 | Activations | YOUR ReLU activation |
| Module 03 | Layers | YOUR Linear layers |
| Module 04 | Losses | YOUR CrossEntropyLoss |
| Module 05 | Autograd | YOUR automatic differentiation |
| Module 06 | Optimizers | YOUR SGD/Adam optimizers |
| Module 07 | Training | YOUR end-to-end training loop |
| Module 08 | DataLoader | YOUR data batching |
| **Module 09** | **Spatial** | **YOUR Conv2d + MaxPool2d** |

## Milestone Structure

This milestone uses **spatial architecture progression** with 2 scripts:

### 01_lecun_tinydigits.py
**Purpose:** Prove CNNs > MLPs on same data

- **Dataset:** TinyDigits (8×8 handwritten digits)
- **Architecture:** Conv(1→8) → Pool → Conv(8→16) → Pool → Linear(→10)
- **Comparison:** CNN ~90% vs MLP ~80% (Milestone 03)
- **Key Learning:** "Convolution preserves spatial structure!"

**Why This Comparison Matters:**
- Same dataset, different architecture
- Direct proof that spatial operations help
- ~10% accuracy gain from exploiting locality

### 02_lecun_cifar10.py
**Purpose:** Scale to natural color images

- **Dataset:** CIFAR-10 (60K images, 32×32 RGB, 10 classes)
- **Architecture:** Deeper CNN with multiple conv blocks
- **Expected:** 65-75% accuracy (decent for pure Python!)
- **Key Learning:** "CNNs scale to realistic vision tasks!"

**Historical Note:** CIFAR-10 (2009) became the benchmark for evaluating CNN architectures before ImageNet.

## Expected Results

| Script | Dataset | Image Size | Architecture | Accuracy | Training Time | vs MLP |
|--------|---------|------------|--------------|----------|---------------|--------|
| 01 (TinyDigits) | 1K train | 8×8 gray | Simple CNN | ~90% | 5-7 min | +10% improvement |
| 02 (CIFAR-10) | 50K train | 32×32 RGB | Deeper CNN | 65-75% | 30-60 min | MLPs struggle here |

## Key Learning: Why Convolution Dominates Vision

CNNs exploit three key principles:

### 1. Local Connectivity
**MLP:** Every pixel connects to every neuron (millions of parameters)
**CNN:** Only local regions connect (shared filters, 100× fewer params)

### 2. Translation Invariance
**MLP:** "Cat in top-left" ≠ "Cat in bottom-right" (different weights!)
**CNN:** Same filter detects features anywhere (shared weights)

### 3. Hierarchical Features
**Layer 1:** Edge detectors (vertical, horizontal, diagonal)
**Layer 2:** Texture patterns (combinations of edges)
**Layer 3:** Object parts (wheels, faces, legs)
**Output:** Full objects (cars, cats, planes)

This is why CNNs remained state-of-the-art for vision until Vision Transformers (2020)!

## Running the Milestone

```bash
cd milestones/04_1998_cnn

# Step 1: Prove CNNs > MLPs (run after Module 09)
python 01_lecun_tinydigits.py

# Step 2: Scale to natural images (run after Module 09)
python 02_lecun_cifar10.py
```

## Further Reading

- **LeNet-5 Paper**: LeCun et al. (1998). "Gradient-based learning applied to document recognition"
- **CIFAR-10**: Krizhevsky (2009). "Learning Multiple Layers of Features from Tiny Images"
- **ImageNet Moment**: Krizhevsky et al. (2012). "ImageNet Classification with Deep CNNs" (AlexNet)
- **Modern Survey**: [A guide to convolution arithmetic](https://arxiv.org/abs/1603.07285)

## Achievement Unlocked

After completing this milestone, you'll understand:
- Why convolution works better than dense layers for images
- How local connectivity + weight sharing reduce parameters
- What CNNs learn at each layer (edges → textures → parts → objects)
- Why spatial operations dominated vision until transformers

**You've recreated the architecture that launched modern computer vision!**

---

**Note for Next Milestone:** CNNs excel at vision, but what about sequences (text, audio, time series)? Milestone 05 introduces **Transformers** - the architecture that unified vision AND language!
