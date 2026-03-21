# Milestone 03: The MLP Revival (1986)

## Historical Context

The 1969 XOR crisis had killed neural network research. Then in 1986, **Rumelhart, Hinton, and Williams** published "Learning representations by back-propagating errors," showing that:
1. Multi-layer networks CAN solve complex problems
2. Backpropagation makes them trainable
3. They work on REAL-WORLD data (not just toy problems)

This paper **ended the AI Winter** and launched modern deep learning. Now it's your turn to recreate that breakthrough using YOUR Tiny🔥Torch!

## What You're Building

Multi-layer perceptrons (MLPs) on real image classification tasks:
1. **TinyDigits** - Learn hierarchical features on 8×8 handwritten digits
2. **MNIST** - Scale up to the full 28×28 benchmark dataset

## Required Modules

**Run after Module 08** (Full training pipeline with data loading)

<table>
<thead>
<tr>
<th width="25%"><b>Module</b></th>
<th width="25%">Component</th>
<th width="50%">What It Provides</th>
</tr>
</thead>
<tbody>
<tr><td><b>Module 01</b></td><td>Tensor</td><td>YOUR data structure with autograd</td></tr>
<tr><td><b>Module 02</b></td><td>Activations</td><td>YOUR ReLU activation</td></tr>
<tr><td><b>Module 03</b></td><td>Layers</td><td>YOUR Linear layers</td></tr>
<tr><td><b>Module 04</b></td><td>Losses</td><td>YOUR CrossEntropyLoss</td></tr>
<tr><td><b>Module 05</b></td><td>DataLoader</td><td>YOUR batching and data pipeline</td></tr>
<tr><td><b>Module 06</b></td><td>Autograd</td><td>YOUR automatic differentiation</td></tr>
<tr><td><b>Module 07</b></td><td>Optimizers</td><td>YOUR SGD optimizer</td></tr>
<tr><td><b>Module 08</b></td><td>Training</td><td>YOUR end-to-end training loop</td></tr>
</tbody>
</table>

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

<table>
<thead>
<tr>
<th width="18%"><b>Script</b></th>
<th width="12%">Dataset</th>
<th width="12%">Image Size</th>
<th width="15%">Parameters</th>
<th width="12%">Loss</th>
<th width="15%">Accuracy</th>
<th width="16%">Training Time</th>
</tr>
</thead>
<tbody>
<tr><td><b>01 (TinyDigits)</b></td><td>1K train</td><td>8×8</td><td>~2.4K</td><td>< 0.5</td><td>75-85%</td><td>3-5 min</td></tr>
<tr><td><b>02 (MNIST)</b></td><td>60K train</td><td>28×28</td><td>~100K</td><td>< 0.2</td><td>94-97%</td><td>10-15 min</td></tr>
</tbody>
</table>

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
