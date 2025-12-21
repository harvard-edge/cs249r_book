# Milestone 02: The XOR Crisis (1969)

```{tip} What You'll Learn
- Why single-layer networks have fundamental mathematical limits
- How hidden layers enable non-linear decision boundaries
- Why "deep" learning is called DEEP
```

## Overview

It's 1969. Neural networks are the hottest thing in AI. Funding is pouring in. Then Marvin Minsky and Seymour Papert publish a 308-page mathematical proof that destroys everything: perceptrons **cannot solve XOR**. Not "struggle with" - CANNOT. Mathematically impossible.

Funding evaporates overnight. Research labs shut down. The field dies for 17 years - the infamous **AI Winter**.

You're about to experience that crisis firsthand. You'll watch YOUR perceptron fail on XOR despite perfect training. Loss stuck at 0.69. Accuracy frozen at 50%. Epoch after epoch of futility. Then you'll discover what Minsky missed: add ONE hidden layer, and the impossible becomes trivial.

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
| 05 | DataLoader | YOUR data pipeline |
| 06 | Autograd | YOUR automatic differentiation |
| 07 | Optimizers | YOUR SGD optimizer |
| 08 | Training | YOUR training loop |

## Running the Milestone

Before running, ensure you have completed Modules 01-08. You can check your progress:

```bash
tito module status
```

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

## The Aha Moment: Depth Changes Everything

Script 01 starts training. Loss: 0.69... 0.69... 0.69. Still 0.69. Why isn't it learning? Did you break something?

You check the code. Everything's correct. YOUR Linear layer works. YOUR autograd computes gradients. YOUR optimizer updates weights. But accuracy stays at 50%.

The realization hits: it's not broken. It's IMPOSSIBLE. This is what Minsky proved. This is why funding died. YOUR code is hitting the same mathematical wall that nearly ended AI research. YOUR Linear layer, YOUR autograd, YOUR optimizer, all working perfectly, all completely useless against XOR's geometry.

Then you run script 02. Add one hidden layer. Loss drops immediately: 0.5... 0.3... 0.1... 0.01... 0.0. Accuracy: 100%.

**Depth enables non-linear decision boundaries.** The hidden layer learns to transform the input space so XOR becomes linearly separable. Single layers can only draw straight lines. Multiple layers can draw any shape.

Same YOUR implementations, same training loop. Suddenly the impossible becomes trivial. YOUR multi-layer network just solved what YOUR single layer couldn't. This is the moment you truly understand why "deep" learning is called DEEP.

## YOUR Code Powers This

| Component | Your Module | What It Does |
|-----------|-------------|--------------|
| `Tensor` | Module 01 | Stores inputs and weights |
| `ReLU` | Module 02 | YOUR activation for hidden layer |
| `Linear` | Module 03 | YOUR fully-connected layers |
| `BCELoss` | Module 04 | YOUR loss computation |
| `DataLoader` | Module 05 | YOUR data pipeline |
| `backward()` | Module 06 | YOUR autograd engine |
| `SGD` | Module 07 | YOUR optimizer |
| Training loop | Module 08 | YOUR training orchestration |

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

- **The Crisis**: Minsky, M., & Papert, S. (1969). "Perceptrons: An Introduction to Computational Geometry"
- **The Solution**: Rumelhart, Hinton, Williams (1986). "Learning representations by back-propagating errors"
- **Wikipedia**: [AI Winter](https://en.wikipedia.org/wiki/AI_winter)
