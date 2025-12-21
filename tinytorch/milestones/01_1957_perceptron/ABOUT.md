# Milestone 01: The Perceptron (1957)

```{tip} What You'll Learn
- Why random weights produce random results (and training fixes this)
- How gradient descent transforms guessing into learning
- The fundamental loop that powers all neural network training
```

## Overview

It's 1957. Computers fill entire rooms and can barely add numbers. Then Frank Rosenblatt makes an outrageous claim: he's built a machine that can LEARN. Not through programming - through experience, like a human child.

The press goes wild. The Navy funds research expecting machines that will "walk, talk, see, write, reproduce itself and be conscious of its existence." The New York Times runs the headline: *"New Navy Device Learns by Doing."*

The optimism was premature - but the core insight was revolutionary. You're about to recreate that moment - the exact moment machine learning was born - using components YOU built yourself.

## What You'll Build

A single-layer perceptron for binary classification that demonstrates:
1. **The Problem**: Random weights produce random predictions (~50% accuracy)
2. **The Solution**: Training transforms random weights into learned patterns (95%+ accuracy)

```
Input (features) --> Linear --> Sigmoid --> Output (0 or 1)
```

## Prerequisites

| Module | Component | What It Provides |
|--------|-----------|------------------|
| 01 | Tensor | YOUR data structure |
| 02 | Activations | YOUR sigmoid activation |
| 03 | Layers | YOUR Linear layer |
| 04 | Losses | YOUR loss functions |
| 06-08 | Training Infrastructure | YOUR autograd + optimizer (Part 2 only) |

## Running the Milestone

Before running, ensure you have completed the prerequisite modules. Part 1 requires Modules 01-04, Part 2 requires Modules 01-08. You can check your progress:

```bash
tito module status
```

```bash
cd milestones/01_1957_perceptron

# Part 1: See the problem
python 01_rosenblatt_forward.py
# Expected: ~50% accuracy (random guessing)

# Part 2: See the solution
python 02_rosenblatt_trained.py
# Expected: 95%+ accuracy (learned pattern)
```

## Expected Results

| Script | Accuracy | What It Shows |
|--------|----------|---------------|
| 01 (Forward Only) | ~50% | Random weights = random guessing |
| 02 (Trained) | 95%+ | Training learns the pattern |

## The Aha Moment: Learning IS the Intelligence

You'll run two scripts. Both use the same architecture - YOUR Linear layer, YOUR sigmoid. But one achieves 50% accuracy (random chance), the other 95%+.

**What's the difference?** Not the model. Not the data. The learning loop.

```python
# Script 01: Forward-only (50% accuracy)
output = model(input)           # YOUR code computes
loss = loss_fn(output, target)  # YOUR code measures
# No backward(), no optimization, no learning
# Result: Random weights stay random

# Script 02: Complete training (95%+ accuracy)
output = model(input)           # Same YOUR code
loss = loss_fn(output, target)  # Same YOUR code
loss.backward()                 # YOUR autograd computes gradients
optimizer.step()                # YOUR optimizer learns from mistakes
# Result: Random weights become intelligent
```

Run script 01 and watch YOUR Linear layer make random guesses - 50% accuracy, no better than a coin flip. Now run script 02. Same architecture. Same data. But now YOUR autograd engine computes gradients, YOUR optimizer updates weights. Within seconds, accuracy climbs: 60%... 75%... 85%... 95%+.

**You just watched YOUR implementation learn.** This is the moment Rosenblatt proved machines could improve through experience. And you recreated it with your own code.

## YOUR Code Powers This

| Component | Your Module | What It Does |
|-----------|-------------|--------------|
| `Tensor` | Module 01 | Stores inputs and weights |
| `Sigmoid` | Module 02 | YOUR activation function |
| `Linear` | Module 03 | YOUR fully-connected layer |
| `BCELoss` | Module 04 | YOUR loss computation |
| `backward()` | Module 06 | YOUR autograd engine |
| `SGD` | Module 07 | YOUR optimizer |

## Historical Context

The Perceptron was funded by the US Navy and received enormous media attention. Rosenblatt's 1958 paper introduced the core concepts of trainable weights and gradient-based learning that still power modern neural networks. The initial hype was followed by the "AI Winter" when limitations became apparent, but the fundamental insight that machines could learn from data proved correct.

## Systems Insights

- **Memory**: O(n) parameters for n input features
- **Compute**: O(n) operations per sample
- **Limitation**: Can only solve linearly separable problems

## What's Next

The Perceptron's limitation (linear separability) would become a crisis. Milestone 02 shows what happens when you try to learn XOR - and how hidden layers solve it.

## Further Reading

- **Original Paper**: Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"
- **Wikipedia**: [Perceptron](https://en.wikipedia.org/wiki/Perceptron)
