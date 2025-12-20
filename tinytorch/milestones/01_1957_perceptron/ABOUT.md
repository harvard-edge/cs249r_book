# Milestone 01: The Perceptron (1957)

**FOUNDATION TIER** | Difficulty: 1/4 | Time: 30-60 min | Prerequisites: Modules 01-04

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

```bash
cd milestones/01_1957_perceptron

# Part 1: See the problem (after Module 04)
python 01_rosenblatt_forward.py
# Expected: ~50% accuracy (random guessing)

# Part 2: See the solution (after Module 08)
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

## Systems Insights

- **Memory**: O(n) parameters for n input features
- **Compute**: O(n) operations per sample
- **Limitation**: Can only solve linearly separable problems

## What's Next

The Perceptron's limitation (linear separability) would become a crisis. Milestone 02 shows what happens when you try to learn XOR - and how hidden layers solve it.

## Further Reading

- **Original Paper**: Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"
- **Wikipedia**: [Perceptron](https://en.wikipedia.org/wiki/Perceptron)
