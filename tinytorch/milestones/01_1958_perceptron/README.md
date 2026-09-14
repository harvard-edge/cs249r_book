# Milestone 01: The Perceptron (1958)

## Historical Context

Frank Rosenblatt's Perceptron was the **first trainable artificial neural network** that could learn from examples. Demonstrated in 1957 and published in 1958, it sparked the first AI boom and demonstrated that machines could actually learn to recognize patterns, launching the neural network revolution.

This milestone recreates that pivotal moment using YOUR Tiny🔥Torch implementations.

## What You're Building

A single-layer perceptron for binary classification, demonstrating the forward pass before training enters the curriculum.

## Required Modules

**Run after Module 03** (Tensor, activations, and layers)

<table width="100%">
  <thead>
<tr>
<th width="25%"><b>Module</b></th>
<th width="25%">Component</th>
<th width="50%">What It Provides</th>
</tr>
</thead>
<tbody>
<tr><td><b>Module 01</b></td><td>Tensor</td><td>YOUR data structure</td></tr>
<tr><td><b>Module 02</b></td><td>Activations</td><td>YOUR sigmoid activation</td></tr>
<tr><td><b>Module 03</b></td><td>Layers</td><td>YOUR Linear layer</td></tr>
</tbody>
</table>

## Milestone Structure

This milestone uses a forward-only script:

### 01_rosenblatt_forward.py
**Purpose:** Demonstrate the problem (untrained model)

- Build perceptron with random weights
- Run forward pass on linearly separable data
- Show that random weights = random predictions (~50% accuracy)
- **Key Learning:** "My model doesn't work... yet!"

**When to run:** After Module 03 (before learning losses, autograd, and training)

## Expected Results

<table width="100%">
  <thead>
<tr>
<th width="30%"><b>Script</b></th>
<th width="20%">Accuracy</th>
<th width="50%">What It Shows</th>
</tr>
</thead>
<tbody>
<tr><td><b>01 (Forward Only)</b></td><td>~50%</td><td>Random weights = random guessing</td></tr>
</tbody>
</table>

## Key Learning: Forward Pass ≠ Intelligence

The architecture isn't enough - the model only becomes "intelligent" through training. This milestone drives home the distinction between:
- **Building the model** (easy - just connect layers)
- **Making it learn** (the hard part - requires training)

This is the foundation for understanding all of deep learning!

## Running the Milestone

```bash
cd milestones/01_1958_perceptron

# See the problem (run after Module 03)
python 01_rosenblatt_forward.py

# Or from the TinyTorch project root:
tito milestone run 01
```

## Further Reading

- **Original Paper**: Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"
- **Historical Context**: [Perceptron on Wikipedia](https://en.wikipedia.org/wiki/Perceptron)

## Achievement Unlocked

After completing this milestone, you'll understand:
- How perceptrons work (forward pass)
- Why random weights fail
- Why training infrastructure is needed before models can learn

**You've recreated the birth of neural networks!**
