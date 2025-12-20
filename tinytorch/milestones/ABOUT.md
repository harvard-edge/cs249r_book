# Historical Milestones

**Proof-of-Mastery Demonstrations** | 6 Milestones | Prerequisites: Varies by Milestone

## Overview

You've been building TinyTorch components for weeks. But does your code actually work? Can YOUR tensor class, YOUR autograd engine, YOUR attention mechanism recreate what took the world's brightest researchers decades to discover?

There's only one way to find out: **rebuild history**.

These six milestones are your proof - not just that you understand the theory, but that you built something real. Every line of code executing in these milestones is YOURS. When the Perceptron learns, it's using YOUR gradient descent. When the transformer generates text, it's YOUR attention mechanism routing information. When you hit 75% on CIFAR-10, those are YOUR convolutional layers extracting features.

This isn't a demo - it's proof that you understand ML systems engineering from the ground up.

## The Journey

| Year | Milestone | What You'll Build | Unlocked After |
|------|-----------|-------------------|----------------|
| **1957** | Perceptron | Binary classification with gradient descent | Module 04 |
| **1969** | XOR Crisis | Hidden layers solve non-linear problems | Module 08 |
| **1986** | MLP Revival | Multi-class vision (95%+ MNIST) | Module 08 |
| **1998** | CNN Revolution | Convolutions (70%+ CIFAR-10) | Module 09 |
| **2017** | Transformers | Language generation with attention | Module 13 |
| **2018** | MLPerf | Production optimization pipeline | Module 19 |

## Why Milestones Transform Learning

**You'll Feel the Historical Struggle**: When your single-layer perceptron hits 50% accuracy on XOR and refuses to budge - loss stuck at 0.69, epoch after epoch - you'll viscerally understand why Minsky's proof nearly killed neural network research. The AI Winter wasn't abstract skepticism; it was researchers watching their perceptrons fail exactly like yours just did.

**You'll Experience the Breakthrough**: Then you add one hidden layer. Same data, same problem. Suddenly: 100% accuracy. Loss plummets to zero. You didn't just read about how depth enables non-linear representations - you watched YOUR two-layer network solve what YOUR single layer couldn't. That's not textbook knowledge; that's lived experience.

**You'll Build Something Real**: By Milestone 04, you're not running toy demos anymore. You're processing 50,000 natural images through YOUR DataLoader, extracting features with YOUR convolutional layers, and achieving 75%+ accuracy on CIFAR-10. That's better than many published results from the early 2010s. With code you wrote yourself.

## How to Use Milestones

```bash
# Check your module progress
tito module status

# Run a milestone after completing prerequisites
tito milestone run 01

# Or run directly
cd milestones/01_1957_perceptron
python 02_rosenblatt_trained.py
```

Each milestone folder contains:
- **README.md** - Full historical context and instructions
- **Python scripts** - Progressive demonstrations (e.g., "see the problem" then "see the solution")

## Learning Philosophy

```
Module teaches: HOW to build the component
Milestone proves: WHAT you can build with it
```

The combination of modules + milestones ensures you don't just complete exercises - you build something historically significant that works.

## Further Reading

See the individual milestone pages for detailed technical requirements and learning objectives.

---

**Build the future by understanding the past.**
