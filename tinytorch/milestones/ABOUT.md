# Historical Milestones

**Proof-of-Mastery Demonstrations** | 6 Milestones | Prerequisites: Varies by Milestone

## Overview

Milestones are **hands-on recreations of ML breakthroughs** using YOUR TinyTorch implementations. Each milestone proves your code works by recreating a historically significant achievement - from Rosenblatt's 1957 Perceptron to modern optimization techniques.

Unlike modules (which teach concepts), milestones demonstrate mastery by showing what you can BUILD.

## The Journey

| Year | Milestone | What You'll Build | Unlocked After |
|------|-----------|-------------------|----------------|
| **1957** | Perceptron | Binary classification with gradient descent | Module 04 |
| **1969** | XOR Crisis | Hidden layers solve non-linear problems | Module 06 |
| **1986** | MLP Revival | Multi-class vision (95%+ MNIST) | Module 08 |
| **1998** | CNN Revolution | Spatial intelligence (70%+ CIFAR-10) | Module 09 |
| **2017** | Transformers | Language generation with attention | Module 13 |
| **2018** | MLPerf | Production optimization pipeline | Module 18 |

## Why Milestones Matter

**Deep Understanding**: Experience the actual challenges researchers faced. When your single-layer perceptron fails on XOR, you understand WHY Minsky's critique nearly ended AI research.

**Progressive Building**: Each milestone builds on previous foundations. The perceptron's limitations motivate hidden layers; hidden layers enable CNNs; CNNs inspire transformers.

**Real Achievements**: These aren't toy examples. You're recreating breakthroughs that shaped the entire field of machine learning.

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

For the complete narrative connecting modules to milestones, see:
- [Journey Through ML History](../site/chapters/milestones.md) - Full milestone guide with systems insights
- [The Learning Journey](../site/chapters/learning-journey.md) - Pedagogical progression explanation

---

**Build the future by understanding the past.**
