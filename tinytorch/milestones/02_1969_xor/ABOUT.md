# Milestone 02: The XOR Crisis (1969)

**FOUNDATION TIER** | Difficulty: 2/4 | Time: 30-60 min | Prerequisites: Modules 01-06

## Overview

In 1969, Minsky and Papert's book "Perceptrons" mathematically proved that single-layer networks **cannot solve XOR**. This revelation killed neural network funding for over a decade - the infamous **AI Winter**.

This milestone recreates the crisis... and shows how multi-layer networks solved it.

## What You'll Build

Two demonstrations of perceptron limitations and the multi-layer solution:
1. **The Crisis**: Watch a perceptron fail on XOR despite training
2. **The Solution**: Add a hidden layer and solve the "impossible" problem

```
Crisis:   Input --> Linear --> Output (FAILS)
Solution: Input --> Linear --> ReLU --> Linear --> Output (100%!)
```

## The XOR Problem

```
Inputs    Output
x1  x2    XOR
0   0  -->  0   (same)
0   1  -->  1   (different)
1   0  -->  1   (different)
1   1  -->  0   (same)
```

These 4 points **cannot** be separated by a single line. No amount of training can make a single-layer network learn XOR.

## Prerequisites

| Module | Component | What It Provides |
|--------|-----------|------------------|
| 01 | Tensor | YOUR data structure |
| 02 | Activations | YOUR sigmoid/ReLU |
| 03 | Layers | YOUR Linear layers |
| 04 | Losses | YOUR loss functions |
| 05 | Autograd | YOUR automatic differentiation |
| 06 | Optimizers | YOUR SGD optimizer |

## Running the Milestone

```bash
cd milestones/02_1969_xor

# Part 1: Experience the crisis
python 01_xor_crisis.py
# Expected: Loss stuck at ~0.69, accuracy ~50%

# Part 2: See the solution
python 02_xor_solved.py
# Expected: Loss --> 0.0, accuracy 100%
```

## Expected Results

| Script | Layers | Loss | Accuracy | What It Shows |
|--------|--------|------|----------|---------------|
| 01 (Single Layer) | 1 | ~0.69 (stuck!) | ~50% | Cannot learn XOR |
| 02 (Multi-Layer) | 2 | --> 0.0 | 100% | Hidden layers solve it |

## Key Learning

**Depth enables non-linear decision boundaries.** The hidden layer learns to transform the input space so XOR becomes linearly separable.

Single layers can only draw straight lines. Multiple layers can draw any shape.

## Systems Insights

- **Memory**: O(n^2) with hidden layers (vs O(n) for perceptron)
- **Compute**: O(n^2) operations
- **Breakthrough**: Hidden representations unlock non-linear problems

## Historical Context

Minsky and Papert's proof was mathematically correct but missed the bigger picture. The solution (multi-layer networks with backpropagation) existed but wasn't well understood until Rumelhart, Hinton, and Williams (1986).

The AI Winter lasted ~17 years. When funding returned, progress accelerated rapidly.

## What's Next

Hidden layers solve XOR, but can they scale to real problems? Milestone 03 proves MLPs work on actual image data (MNIST).

## Further Reading

- **The Crisis**: Minsky, M., & Papert, S. (1969). "Perceptrons"
- **The Solution**: Rumelhart, Hinton, Williams (1986). "Learning representations by back-propagating errors"
- **Wikipedia**: [AI Winter](https://en.wikipedia.org/wiki/AI_winter)
