# Milestone 03: The MLP Revival (1986)

## Historical Context

The 1969 XOR crisis had killed neural network research. Then in 1986, **Rumelhart, Hinton, and Williams** published "Learning representations by back-propagating errors," showing that:
1. Multi-layer networks CAN solve complex problems
2. Backpropagation makes them trainable
3. They work on REAL-WORLD data (not just toy problems)

This paper **ended the AI Winter** and launched modern deep learning. Now it's your turn to recreate that breakthrough using YOUR Tiny🔥Torch!

## What You're Building

Multi-layer perceptrons (MLPs) on non-linear and image classification tasks:
1. **XOR Solved** - Use hidden layers plus backpropagation to solve the 1969 crisis
2. **TinyDigits** - Learn hierarchical features on 8×8 handwritten digits

## Required Modules

**Run after Module 08** (Full training pipeline with data loading)

<table width="100%">
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

### ../02_1969_xor/02_xor_solved.py
**Purpose:** Prove hidden layers plus backpropagation solve XOR

- **Dataset:** XOR truth table
- **Architecture:** Input(2) → Linear → ReLU → Linear → Sigmoid
- **Expected:** 100% accuracy
- **Key Learning:** "Depth plus gradients solves non-linear problems!"

### 01_rumelhart_tinydigits.py
**Purpose:** Prove MLPs work on real images (fast iteration)

- **Dataset:** TinyDigits (1000 train + 200 test, 8×8 images)
- **Architecture:** Input(64) → Linear(64→32) → ReLU → Linear(32→10)
- **Expected:** 85%+ accuracy in a few minutes
- **Key Learning:** "MLPs can learn hierarchical features from images!"

**Why TinyDigits First?**
- Fast training = quick feedback loop
- Small size = easy to understand what's happening
- Decent accuracy = proves concept works
- Ships with TinyTorch = no downloads needed

## Expected Results

<table width="100%">
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
<tr><td><b>01 (XOR Solved)</b></td><td>4 examples</td><td>N/A</td><td>small</td><td>→ 0.0</td><td>100%</td><td>&lt;1 min</td></tr>
<tr><td><b>02 (TinyDigits)</b></td><td>1K train</td><td>8×8</td><td>~2.4K</td><td>&lt; 0.5</td><td>85%+</td><td>3-5 min</td></tr>
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

# Quick validation on TinyDigits (run after Module 08)
python 01_rumelhart_tinydigits.py

# Or run the full milestone from the TinyTorch project root:
tito milestone run 03

# Run individual parts:
tito milestone run 03 --part 1  # XOR Solved
tito milestone run 03 --part 2  # TinyDigits
```

## Further Reading

- **The Backprop Paper**: Rumelhart, Hinton, Williams (1986). "Learning representations by back-propagating errors"
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
