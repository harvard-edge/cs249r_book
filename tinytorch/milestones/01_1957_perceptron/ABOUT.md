# Milestone 01: The Perceptron (1957)

**FOUNDATION TIER** | Difficulty: 1/4 | Time: 30-60 min | Prerequisites: Modules 01-04

## Overview

Frank Rosenblatt's Perceptron was the **first trainable artificial neural network**. In 1957, he demonstrated that machines could learn from examples, launching the neural network revolution.

This milestone recreates that pivotal moment using YOUR Tiny🔥Torch implementations.

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
| 05-07 | Training | YOUR autograd + optimizer (Part 2 only) |

## Running the Milestone

```bash
cd milestones/01_1957_perceptron

# Part 1: See the problem (after Module 04)
python 01_rosenblatt_forward.py
# Expected: ~50% accuracy (random guessing)

# Part 2: See the solution (after Module 07)
python 02_rosenblatt_trained.py
# Expected: 95%+ accuracy (learned pattern)
```

## Expected Results

| Script | Accuracy | What It Shows |
|--------|----------|---------------|
| 01 (Forward Only) | ~50% | Random weights = random guessing |
| 02 (Trained) | 95%+ | Training learns the pattern |

## Key Learning

**Forward pass is not intelligence.** The architecture alone doesn't solve problems - training does. This milestone demonstrates the fundamental learning loop:

```
forward --> loss --> backward --> update --> repeat
```

## Systems Insights

- **Memory**: O(n) parameters for n input features
- **Compute**: O(n) operations per sample
- **Limitation**: Can only solve linearly separable problems

## Historical Context

The New York Times headline (1958): *"New Navy Device Learns by Doing"*

Rosenblatt's Perceptron sparked the first AI boom. The US Navy funded research expecting perceptrons to "walk, talk, see, write, reproduce itself and be conscious of its existence."

The optimism was premature - but the core insight was revolutionary: machines can learn from data.

## What's Next

The Perceptron's limitation (linear separability) would become a crisis. Milestone 02 shows what happens when you try to learn XOR - and how hidden layers solve it.

## Further Reading

- **Original Paper**: Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"
- **Wikipedia**: [Perceptron](https://en.wikipedia.org/wiki/Perceptron)
