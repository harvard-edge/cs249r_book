# Milestone 01: The Perceptron (1957)

## Historical Context

Frank Rosenblatt's Perceptron was the **first trainable artificial neural network** that could learn from examples. Demonstrated in 1957 and published in 1958, it sparked the first AI boom and demonstrated that machines could actually learn to recognize patterns, launching the neural network revolution.

This milestone recreates that pivotal moment using YOUR Tiny🔥Torch implementations.

## What You're Building

A single-layer perceptron for binary classification, demonstrating:
1. **The Problem** - Why random weights don't work (forward pass only)
2. **The Solution** - How training makes the model learn (with gradient descent)

## Required Modules

**Progressive Requirements:**
- **Part 1 (Forward Only):** Run after Module 04 (building blocks)
- **Part 2 (Trained):** Run after Module 08 (training capability)

| Module | Component | What It Provides |
|--------|-----------|------------------|
| Module 01 | Tensor | YOUR data structure |
| Module 02 | Activations | YOUR sigmoid activation |
| Module 03 | Layers | YOUR Linear layer |
| Module 04 | Losses | YOUR loss functions |
| Module 06 | Autograd | YOUR automatic differentiation (Part 2 only) |
| Module 07 | Optimizers | YOUR SGD optimizer (Part 2 only) |
| Module 08 | Training | YOUR end-to-end training loop (Part 2 only) |

## Milestone Structure

This milestone uses **progressive revelation** with 2 scripts:

### 01_rosenblatt_forward.py
**Purpose:** Demonstrate the problem (untrained model)

- Build perceptron with random weights
- Run forward pass on linearly separable data
- Show that random weights = random predictions (~50% accuracy)
- **Key Learning:** "My model doesn't work... yet!"

**When to run:** After Module 04 (before learning training)

### 02_rosenblatt_trained.py
**Purpose:** Demonstrate the solution (trained model)

- Same architecture, but WITH training
- Apply gradient descent (YOUR autograd + optimizer)
- Watch accuracy improve from ~50% to 95%+
- **Key Learning:** "Training makes it work!"

**When to run:** After Module 08 (after learning training)

## Expected Results

| Script | Accuracy | What It Shows |
|--------|----------|---------------|
| 01 (Forward Only) | ~50% | Random weights = random guessing |
| 02 (Trained) | 95%+ | Training learns the pattern |

## Key Learning: Forward Pass ≠ Intelligence

The architecture isn't enough - the model only becomes "intelligent" through training. This milestone drives home the distinction between:
- **Building the model** (easy - just connect layers)
- **Making it learn** (the hard part - requires training)

This is the foundation for understanding all of deep learning!

## Running the Milestone

```bash
cd milestones/01_1957_perceptron

# Step 1: See the problem (run after Module 04)
python 01_rosenblatt_forward.py

# Step 2: See the solution (run after Module 08)
python 02_rosenblatt_trained.py
```

## Further Reading

- **Original Paper**: Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"
- **Historical Context**: [Perceptron on Wikipedia](https://en.wikipedia.org/wiki/Perceptron)

## Achievement Unlocked

After completing this milestone, you'll understand:
- How perceptrons work (forward pass)
- Why random weights fail
- How training transforms random weights into learned patterns
- The fundamental learning loop: forward → loss → backward → update

**You've recreated the birth of neural networks!**
