# Milestone 03: The MLP Revival (1986)

## Historical Context

The 1969 XOR crisis had killed neural network research. Then in 1986, **Rumelhart, Hinton, and Williams** published "Learning representations by back-propagating errors," showing that:
1. Multi-layer networks CAN solve complex problems
2. Backpropagation makes them trainable
3. They work on REAL-WORLD data (not just toy problems)

This paper **ended the AI Winter** and launched modern deep learning. Now it's your turn to recreate that breakthrough using YOUR TinyTorch!

## What You're Building

Multi-layer perceptrons (MLPs) on real image classification tasks:
1. **TinyDigits** - Learn hierarchical features on 8×8 handwritten digits
2. **MNIST** - Scale up to the full 28×28 benchmark dataset

## Required Modules

**Run after Module 08** (Full training pipeline with data loading)

| Module | Component | What It Provides |
|--------|-----------|------------------|
| Module 01 | Tensor | YOUR data structure with autograd |
| Module 02 | Activations | YOUR ReLU activation |
| Module 03 | Layers | YOUR Linear layers |
| Module 04 | Losses | YOUR CrossEntropyLoss |
| Module 05 | Autograd | YOUR automatic differentiation |
| Module 06 | Optimizers | YOUR SGD optimizer |
| Module 07 | Training | YOUR end-to-end training loop |
| Module 08 | DataLoader | YOUR batching and data pipeline |

## Milestone Structure

This milestone uses **progressive scaling** with 2 scripts:

### 01_rumelhart_tinydigits.py
**Purpose:** Prove MLPs work on real images (fast iteration)

- **Dataset:** TinyDigits (1000 train + 200 test, 8×8 images)
- **Architecture:** Input(64) → Linear(64→32) → ReLU → Linear(32→10)
- **Expected:** 75-85% accuracy in 3-5 minutes
- **Key Learning:** "MLPs can learn hierarchical features from images!"

**Why TinyDigits First?**
- Fast training = quick feedback loop
- Small size = easy to understand what's happening
- Decent accuracy = proves concept works
- Ships with TinyTorch = no downloads needed

### 02_rumelhart_mnist.py
**Purpose:** Scale to the classic benchmark

- **Dataset:** MNIST (60K train + 10K test, 28×28 images)
- **Architecture:** Input(784) → Linear(784→128) → ReLU → Linear(128→10)
- **Expected:** 94-97% accuracy (competitive for MLPs!)
- **Key Learning:** "Same principles scale to larger problems!"

**Historical Note:** MNIST (1998) became THE benchmark for evaluating learning algorithms. MLPs hitting 95%+ proved neural networks were back!

## Expected Results

| Script | Dataset | Image Size | Parameters | Loss | Accuracy | Training Time |
|--------|---------|------------|------------|------|----------|---------------|
| 01 (TinyDigits) | 1K train | 8×8 | ~2.4K | < 0.5 | 75-85% | 3-5 min |
| 02 (MNIST) | 60K train | 28×28 | ~100K | < 0.2 | 94-97% | 10-15 min |

## Key Learning: Hierarchical Feature Learning

MLPs don't just memorize - they learn useful internal representations:

**Hidden Layer Discovers:**
- Edge detectors (low-level features)
- Curve patterns (mid-level features)
- Digit-specific combinations (high-level features)

This is **representation learning** - the foundation of deep learning's power.

**Why This Matters:**
- Manual feature engineering → Automatic feature learning
- Domain expertise → Data-driven discovery
- This shift enabled modern AI

## Running the Milestone

```bash
cd milestones/03_1986_mlp

# Step 1: Quick validation on TinyDigits (run after Module 08)
python 01_rumelhart_tinydigits.py

# Step 2: Scale to MNIST benchmark (run after Module 08)
python 02_rumelhart_mnist.py
```

## Further Reading

- **The Backprop Paper**: Rumelhart, Hinton, Williams (1986). "Learning representations by back-propagating errors"
- **MNIST Dataset**: LeCun et al. (1998). "Gradient-based learning applied to document recognition"
- **Why MLPs Work**: Cybenko (1989). "Approximation by superpositions of a sigmoidal function" (Universal Approximation Theorem)

## Achievement Unlocked

After completing this milestone, you'll understand:
- How MLPs learn hierarchical features from raw pixels
- Why hidden layers discover useful representations
- The power of backpropagation for multi-layer training
- How to scale from toy datasets to real benchmarks

**You've recreated the breakthrough that ended the AI Winter!**

---

**Note for Next Milestone:** MLPs treat images as flat vectors, ignoring spatial structure. Milestone 04 (CNN) will show why **convolutional** layers dramatically improve image recognition!
