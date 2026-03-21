# Milestone 02: The XOR Crisis (1969)

## Historical Context

In 1969, Marvin Minsky and Seymour Papert published **"Perceptrons,"** a book that mathematically proved single-layer perceptrons **CANNOT solve the XOR problem**. This revelation killed neural network research funding for over a decade - triggering the infamous **"AI Winter."**

The proof was devastating: no matter how much you train, a single layer cannot learn XOR. This milestone recreates that crisis... and then shows how multi-layer networks solved it.

## What You're Building

A demonstration of perceptron limitations and the multi-layer solution:
1. **The Crisis** - Watch a perceptron fail to learn XOR despite training
2. **The Solution** - See how adding a hidden layer solves the "impossible" problem

## Required Modules

**Run after Module 08** (Training capability)

<table>
<thead>
<tr>
<th width="25%"><b>Module</b></th>
<th width="25%">Component</th>
<th width="50%">What It Provides</th>
</tr>
</thead>
<tbody>
<tr><td><b>Module 01</b></td><td>Tensor</td><td>YOUR data structure</td></tr>
<tr><td><b>Module 02</b></td><td>Activations</td><td>YOUR sigmoid/ReLU activations</td></tr>
<tr><td><b>Module 03</b></td><td>Layers</td><td>YOUR Linear layers</td></tr>
<tr><td><b>Module 04</b></td><td>Losses</td><td>YOUR loss functions</td></tr>
<tr><td><b>Module 06</b></td><td>Autograd</td><td>YOUR automatic differentiation</td></tr>
<tr><td><b>Module 07</b></td><td>Optimizers</td><td>YOUR SGD optimizer</td></tr>
<tr><td><b>Module 08</b></td><td>Training</td><td>YOUR end-to-end training loop</td></tr>
</tbody>
</table>

## Milestone Structure

This milestone uses **crisis → solution** narrative with 2 scripts:

### 01_xor_crisis.py
**Purpose:** Demonstrate the fundamental limitation

- Train a single-layer perceptron on XOR
- Watch loss stay high (~0.69) and accuracy stuck at 50%
- No matter how long you train, it CANNOT learn
- **Key Learning:** "Minsky was right - single layers can't solve XOR"

**The XOR Problem:**
```
Inputs    Output
x1  x2    XOR
0   0  →  0   (same)
0   1  →  1   (different)
1   0  →  1   (different)
1   1  →  0   (same)
```

These 4 points CANNOT be separated by a single line!

### 02_xor_solved.py
**Purpose:** Show how multi-layer networks solve it

- Add ONE hidden layer (2-layer network)
- Same XOR problem, now solvable
- Watch accuracy reach 100%
- **Key Learning:** "Hidden layers unlock non-linear problems!"

**The Solution:**
```
Input → Hidden Layer → Output
        (learns useful features)
```

The hidden layer learns to transform the space so XOR becomes linearly separable!

## Expected Results

<table>
<thead>
<tr>
<th width="25%"><b>Script</b></th>
<th width="10%">Layers</th>
<th width="20%">Loss</th>
<th width="15%">Accuracy</th>
<th width="30%">What It Shows</th>
</tr>
</thead>
<tbody>
<tr><td><b>01 (Single Layer)</b></td><td>1</td><td>~0.69 (stuck!)</td><td>~50%</td><td>Cannot learn XOR (Minsky was right)</td></tr>
<tr><td><b>02 (Multi-Layer)</b></td><td>2</td><td>→ 0.0</td><td>100%</td><td>Hidden layers solve the problem!</td></tr>
</tbody>
</table>

## Key Learning: The Power of Depth

This milestone teaches the **fundamental reason why deep learning works**:

- **Single layers** = Only linear decision boundaries (limited expressiveness)
- **Multiple layers** = Can learn ANY decision boundary (universal approximation)

The XOR crisis wasn't about perceptrons being broken - it was about needing **depth** to solve complex problems. This realization (via backpropagation in 1986) ended the AI Winter.

## Running the Milestone

```bash
cd milestones/02_1969_xor

# Step 1: Experience the crisis (run after Module 08)
python 01_xor_crisis.py

# Step 2: See the solution (run after Module 08)
python 02_xor_solved.py
```

## Further Reading

- **The Crisis**: Minsky, M., & Papert, S. (1969). "Perceptrons"
- **The Solution**: Rumelhart, Hinton, Williams (1986). "Learning representations by back-propagating errors"
- **Historical Context**: [AI Winter on Wikipedia](https://en.wikipedia.org/wiki/AI_winter)

## Achievement Unlocked

After completing this milestone, you'll understand:
- Why single-layer networks have fundamental limitations
- What "linear separability" means (and why it matters)
- How hidden layers enable non-linear decision boundaries
- The historical importance of this problem (caused AI Winter!)

**You've experienced the crisis that shaped neural network history!**
