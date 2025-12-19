# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Module 04: Losses - Measuring How Wrong We Are

Welcome to Module 04! Today you'll implement the mathematical functions that measure how wrong your model's predictions are - the essential feedback signal that enables all machine learning.

## 🔗 Prerequisites & Progress
**You've Built**: Tensors (data), Activations (intelligence), Layers (architecture)
**You'll Build**: Loss functions that measure prediction quality
**You'll Enable**: The feedback signal needed for training (Module 06: Autograd)

**Connection Map**:
```
Layers → Losses → Autograd
(predictions) (error measurement) (learning signals)
```

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement MSELoss for regression problems
2. Implement CrossEntropyLoss for classification problems
3. Implement BinaryCrossEntropyLoss for binary classification
4. Understand numerical stability in loss computation
5. Test all loss functions with realistic examples

Let's measure prediction quality!
"""

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `src/04_losses/04_losses.py`
**Building Side:** Code exports to `tinytorch.core.losses`

```python
# Final package structure:
from tinytorch.core.losses import MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss, log_softmax  # This module
```

**Why this matters:**
- **Learning:** Complete loss function system in one focused module
- **Production:** Proper organization like PyTorch's torch.nn functional losses
- **Consistency:** All loss computations and numerical stability in core.losses
- **Integration:** Works seamlessly with layers for complete prediction-to-error workflow
"""

# %% [markdown]
"""
## 📋 Module Dependencies

**Prerequisites**: Modules 01 (Tensor), 02 (Activations), and 03 (Layers) must be completed

**External Dependencies**:
- `numpy` (for numerical operations)

**TinyTorch Dependencies**:
- **Module 01 (Tensor)**: Foundation for all loss computations
  - Used for: Input/output data structures, shape operations, element-wise operations
  - Required: Yes - losses operate on Tensor objects
- **Module 02 (Activations)**: Activation functions for testing
  - Used for: ReLU for building test networks that generate realistic outputs
  - Required: Yes - for testing loss functions with realistic predictions
- **Module 03 (Layers)**: Layer components for testing
  - Used for: Linear layer for testing loss functions with realistic predictions
  - Required: Yes - for building test networks

**Dependency Flow**:
```
Module 01 (Tensor) → Module 02 (Activations) → Module 03 (Layers) → Module 04 (Losses) → Module 05 (DataLoader) → Module 06 (Autograd)
     ↓                      ↓                         ↓                    ↓                    ↓                      ↓
  Foundation          Nonlinearity              Architecture        Error Measurement      Data Pipelines       Gradient Flow
```

**Import Strategy**:
This module imports directly from the TinyTorch package (`from tinytorch.core.*`).
**Assumption**: Modules 01 (Tensor), 02 (Activations), and 03 (Layers) have been completed and exported to the package.
If you see import errors, ensure you've run `tito export` after completing previous modules.
"""

# %% nbgrader={"grade": false, "grade_id": "setup", "solution": true}
#| default_exp core.losses
#| export

import numpy as np
from typing import Optional

# Import from TinyTorch package (previous modules must be completed and exported)
from tinytorch.core.tensor import Tensor
from tinytorch.core.activations import ReLU
from tinytorch.core.layers import Linear

# Constants for numerical stability
EPSILON = 1e-7  # Small value to prevent log(0) and numerical instability

# %% [markdown]
"""
## 💡 Introduction - What Are Loss Functions?

Loss functions are the mathematical conscience of machine learning. They measure the distance between what your model predicts and what actually happened. Without loss functions, models have no way to improve - they're like athletes training without knowing their score.

## 💡 The Three Essential Loss Functions

Think of loss functions as different ways to measure "wrongness" - each optimized for different types of problems:

**MSELoss (Mean Squared Error)**: "How far off are my continuous predictions?"
- Used for: Regression (predicting house prices, temperature, stock values)
- Calculation: Average of squared differences between predictions and targets
- Properties: Heavily penalizes large errors, smooth gradients

```
Loss Landscape for MSE:
     Loss
      ^
      |
   4  |     *
      |    / \
   2  |   /   \
      |  /     \
   0  |_/_______\\____> Prediction Error
      0  -2  0  +2

Quadratic growth: small errors → small penalty, large errors → huge penalty
```

**CrossEntropyLoss**: "How confident am I in the wrong class?"
- Used for: Multi-class classification (image recognition, text classification)
- Calculation: Negative log-likelihood of correct class probability
- Properties: Encourages confident correct predictions, punishes confident wrong ones

```
Cross-Entropy Penalty Curve:
     Loss
      ^
   10 |*
      ||
    5 | \
      |  \
    2 |   \
      |    \
    0 |_____\\____> Predicted Probability of Correct Class
      0   0.5   1.0

Logarithmic: wrong confident predictions get severe penalty
```

**BinaryCrossEntropyLoss**: "How wrong am I about yes/no decisions?"
- Used for: Binary classification (spam detection, medical diagnosis)
- Calculation: Cross-entropy specialized for two classes
- Properties: Symmetric penalty for false positives and false negatives

```
Binary Decision Boundary:
     Target=1 (Positive)    Target=0 (Negative)
     ┌─────────────────┬─────────────────┐
     │  Pred → 1.0     │  Pred → 1.0     │
     │  Loss → 0       │  Loss → ∞       │
     ├─────────────────┼─────────────────┤
     │  Pred → 0.0     │  Pred → 0.0     │
     │  Loss → ∞       │  Loss → 0       │
     └─────────────────┴─────────────────┘
```

Each loss function creates a different "error landscape" that guides learning in different ways.
"""

# %% [markdown]
"""
## 📐 Mathematical Foundations

## 📐 Mean Squared Error (MSE)
The foundation of regression, MSE measures the average squared distance between predictions and targets:

```
MSE = (1/N) * Σ(prediction_i - target_i)²
```

**Why square the differences?**
- Makes all errors positive (no cancellation between positive/negative errors)
- Heavily penalizes large errors (error of 2 becomes 4, error of 10 becomes 100)
- Creates smooth gradients for optimization

## 📐 Cross-Entropy Loss
For classification, we need to measure how wrong our probability distributions are:

```
CrossEntropy = -Σ target_i * log(prediction_i)
```

**The Log-Sum-Exp Trick**:
Computing softmax directly can cause numerical overflow. The log-sum-exp trick provides stability:
```
log_softmax(x) = x - log(Σ exp(x_i))
                = x - max(x) - log(Σ exp(x_i - max(x)))
```

This prevents exp(large_number) from exploding to infinity.

## 📐 Binary Cross-Entropy
A specialized case where we have only two classes:
```
BCE = -(target * log(prediction) + (1-target) * log(1-prediction))
```

The mathematics naturally handles both "positive" and "negative" cases in a single formula.
"""

# %% [markdown]
"""
## 🏗️ Implementation - Building Loss Functions

Let's implement our loss functions with proper numerical stability and clear educational structure.
"""

# %% [markdown]
"""
## 🏗️ Log-Softmax - The Numerically Stable Foundation

Before implementing loss functions, we need a reliable way to compute log-softmax. This function is the numerically stable backbone of classification losses.

### Why Log-Softmax Matters

Naive softmax can explode with large numbers:
```
Naive approach:
  logits = [100, 200, 300]
  exp(300) = 1.97 × 10^130  ← This breaks computers!

Stable approach:
  max_logit = 300
  shifted = [-200, -100, 0]  ← Subtract max
  exp(0) = 1.0  ← Manageable numbers
```

### The Log-Sum-Exp Trick Visualization

```
Original Computation:           Stable Computation:

logits: [a, b, c]              logits: [a, b, c]
   ↓                              ↓
exp(logits)                    max_val = max(a,b,c)
   ↓                              ↓
sum(exp(logits))               shifted = [a-max, b-max, c-max]
   ↓                              ↓
log(sum)                       exp(shifted)  ← All ≤ 1.0
   ↓                              ↓
logits - log(sum)              sum(exp(shifted))
                                  ↓
                               log(sum) + max_val
                                  ↓
                               logits - (log(sum) + max_val)
```

Both give the same result, but the stable version never overflows!
"""

# %% nbgrader={"grade": false, "grade_id": "log_softmax", "solution": true}
#| export
def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Compute log-softmax with numerical stability.

    TODO: Implement numerically stable log-softmax using the log-sum-exp trick

    APPROACH:
    1. Find maximum along dimension (for stability)
    2. Subtract max from input (prevents overflow)
    3. Compute log(sum(exp(shifted_input)))
    4. Return input - max - log_sum_exp

    EXAMPLE:
    >>> logits = Tensor([[1.0, 2.0, 3.0], [0.1, 0.2, 0.9]])
    >>> result = log_softmax(logits, dim=-1)
    >>> print(result.shape)
    (2, 3)

    HINT: Use np.max(x.data, axis=dim, keepdims=True) to preserve dimensions
    """
    ### BEGIN SOLUTION
    # Step 1: Find max along dimension for numerical stability
    max_vals = np.max(x.data, axis=dim, keepdims=True)

    # Step 2: Subtract max to prevent overflow
    shifted = x.data - max_vals

    # Step 3: Compute log(sum(exp(shifted)))
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))

    # Step 4: Return log_softmax = input - max - log_sum_exp
    result = x.data - max_vals - log_sum_exp

    return Tensor(result)
    ### END SOLUTION

# %% nbgrader={"grade": true, "grade_id": "test_log_softmax", "locked": true, "points": 10}
def test_unit_log_softmax():
    """🔬 Test log_softmax numerical stability and correctness."""
    print("🔬 Unit Test: Log-Softmax...")

    # Test basic functionality
    x = Tensor([[1.0, 2.0, 3.0], [0.1, 0.2, 0.9]])
    result = log_softmax(x, dim=-1)

    # Verify shape preservation
    assert result.shape == x.shape, f"Shape mismatch: expected {x.shape}, got {result.shape}"

    # Verify log-softmax properties: exp(log_softmax) should sum to 1
    softmax_result = np.exp(result.data)
    row_sums = np.sum(softmax_result, axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), f"Softmax doesn't sum to 1: {row_sums}"

    # Test numerical stability with large values
    large_x = Tensor([[100.0, 101.0, 102.0]])
    large_result = log_softmax(large_x, dim=-1)
    assert not np.any(np.isnan(large_result.data)), "NaN values in result with large inputs"
    assert not np.any(np.isinf(large_result.data)), "Inf values in result with large inputs"

    print("✅ log_softmax works correctly with numerical stability!")

if __name__ == "__main__":
    test_unit_log_softmax()

# %% [markdown]
"""
## 🏗️ MSELoss - Measuring Continuous Prediction Quality

Mean Squared Error is the workhorse of regression problems. It measures how far your continuous predictions are from the true values.

### When to Use MSE

**Perfect for:**
- House price prediction ($200k vs $195k)
- Temperature forecasting (25°C vs 23°C)
- Stock price prediction ($150 vs $148)
- Any continuous value where "distance" matters

### How MSE Shapes Learning

```
Prediction vs Target Visualization:

Target = 100

Prediction: 80   90   95   100  105  110  120
Error:     -20  -10   -5    0   +5  +10  +20
MSE:       400  100   25    0   25  100  400

Loss Curve:
     MSE
      ^
  400 |*           *
      |
  100 | *         *
      |  \
   25 |   *     *
      |    \\   /
    0 |_____*_____> Prediction
       80   100   120

Quadratic penalty: Large errors are MUCH more costly than small errors
```

### Why Square the Errors?

1. **Positive penalties**: (-10)² = 100, same as (+10)² = 100
2. **Heavy punishment for large errors**: Error of 20 → penalty of 400
3. **Smooth gradients**: Quadratic function has nice derivatives for optimization
4. **Statistical foundation**: Maximum likelihood for Gaussian noise

### MSE vs Other Regression Losses

```
Error Sensitivity Comparison:

 Error:   -10    -5     0     +5    +10
 MSE:     100    25     0     25    100  ← Quadratic growth
 MAE:      10     5     0      5     10  ← Linear growth
 Huber:    50    12.5   0    12.5    50  ← Hybrid approach

 MSE: More sensitive to outliers
 MAE: More robust to outliers
 Huber: Best of both worlds
```
"""

# %% nbgrader={"grade": false, "grade_id": "mse_loss", "solution": true}
#| export
class MSELoss:
    """Mean Squared Error loss for regression tasks."""

    def __init__(self):
        """Initialize MSE loss function."""
        pass

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute mean squared error between predictions and targets.

        TODO: Implement MSE loss calculation

        APPROACH:
        1. Compute difference: predictions - targets
        2. Square the differences: diff²
        3. Take mean across all elements

        EXAMPLE:
        >>> loss_fn = MSELoss()
        >>> predictions = Tensor([1.0, 2.0, 3.0])
        >>> targets = Tensor([1.5, 2.5, 2.8])
        >>> loss = loss_fn(predictions, targets)
        >>> print(f"MSE Loss: {loss.data:.4f}")
        MSE Loss: 0.1467

        HINTS:
        - Use (predictions.data - targets.data) for element-wise difference
        - Square with **2 or np.power(diff, 2)
        - Use np.mean() to average over all elements
        """
        ### BEGIN SOLUTION
        # Step 1: Compute element-wise difference
        diff = predictions.data - targets.data

        # Step 2: Square the differences
        squared_diff = diff ** 2

        # Step 3: Take mean across all elements
        mse = np.mean(squared_diff)

        return Tensor(mse)
        ### END SOLUTION

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Allows the loss function to be called like a function."""
        return self.forward(predictions, targets)

    def backward(self) -> Tensor:
        """
        Compute gradients (implemented in Module 06: Autograd).

        For now, this is a stub that students can ignore.
        """
        pass

# %% nbgrader={"grade": true, "grade_id": "test_mse_loss", "locked": true, "points": 10}
def test_unit_mse_loss():
    """🔬 Test MSELoss implementation and properties."""
    print("🔬 Unit Test: MSE Loss...")

    loss_fn = MSELoss()

    # Test perfect predictions (loss should be 0)
    predictions = Tensor([1.0, 2.0, 3.0])
    targets = Tensor([1.0, 2.0, 3.0])
    perfect_loss = loss_fn.forward(predictions, targets)
    assert np.allclose(perfect_loss.data, 0.0, atol=EPSILON), f"Perfect predictions should have 0 loss, got {perfect_loss.data}"

    # Test known case
    predictions = Tensor([1.0, 2.0, 3.0])
    targets = Tensor([1.5, 2.5, 2.8])
    loss = loss_fn.forward(predictions, targets)

    # Manual calculation: ((1-1.5)² + (2-2.5)² + (3-2.8)²) / 3 = (0.25 + 0.25 + 0.04) / 3 = 0.18
    expected_loss = (0.25 + 0.25 + 0.04) / 3
    assert np.allclose(loss.data, expected_loss, atol=1e-6), f"Expected {expected_loss}, got {loss.data}"

    # Test that loss is always non-negative
    random_pred = Tensor(np.random.randn(10))
    random_target = Tensor(np.random.randn(10))
    random_loss = loss_fn.forward(random_pred, random_target)
    assert random_loss.data >= 0, f"MSE loss should be non-negative, got {random_loss.data}"

    print("✅ MSELoss works correctly!")

if __name__ == "__main__":
    test_unit_mse_loss()

# %% [markdown]
"""
## 🏗️ CrossEntropyLoss - Measuring Classification Confidence

Cross-entropy loss is the gold standard for multi-class classification. It measures how wrong your probability predictions are and heavily penalizes confident mistakes.

### When to Use Cross-Entropy

**Perfect for:**
- Image classification (cat, dog, bird)
- Text classification (spam, ham, promotion)
- Language modeling (next word prediction)
- Any problem with mutually exclusive classes

### Understanding Cross-Entropy Through Examples

```
Scenario: Image Classification (3 classes: cat, dog, bird)

Case 1: Correct and Confident
Model Output (logits): [5.0, 1.0, 0.1]  ← Very confident about "cat"
After Softmax:        [0.95, 0.047, 0.003]
True Label:           cat (class 0)
Loss: -log(0.95) = 0.05  ← Very low loss ✅

Case 2: Correct but Uncertain
Model Output:         [1.1, 1.0, 0.9]  ← Uncertain between classes
After Softmax:        [0.4, 0.33, 0.27]
True Label:           cat (class 0)
Loss: -log(0.4) = 0.92  ← Higher loss (uncertainty penalized)

Case 3: Wrong and Confident
Model Output:         [0.1, 5.0, 1.0]  ← Very confident about "dog"
After Softmax:        [0.003, 0.95, 0.047]
True Label:           cat (class 0)
Loss: -log(0.003) = 5.8  ← Very high loss ❌
```

### Cross-Entropy's Learning Signal

```
What Cross-Entropy Teaches the Model:

┌─────────────────┬─────────────────┬───────────────────────────┐
│ Prediction      │ True Label      │ Learning Signal           │
├─────────────────┼─────────────────┼───────────────────────────┤
│ Confident ✅    │ Correct ✅      │ "Keep doing this"         │
│ Uncertain ⚠️    │ Correct ✅      │ "Be more confident"       │
│ Confident ❌    │ Wrong ❌        │ "STOP! Change everything" │
│ Uncertain ⚠️    │ Wrong ❌        │ "Learn the right answer"  │
└─────────────────┴─────────────────┴───────────────────────────┘

Loss Landscape by Confidence:
     Loss
      ^
    5 |*
      ||
    3 | *
      |  \
    1 |   *
      |    \\
    0 |______**____> Predicted Probability (correct class)
      0   0.5   1.0

Message: "Be confident when you're right!"
```

### Why Cross-Entropy Works So Well

1. **Probabilistic interpretation**: Measures quality of probability distributions
2. **Strong gradients**: Large penalty for confident mistakes drives fast learning
3. **Smooth optimization**: Log function provides nice gradients
4. **Information theory**: Minimizes "surprise" about correct answers

### Multi-Class vs Binary Classification

```
Multi-Class (3+ classes):          Binary (2 classes):

Classes: [cat, dog, bird]         Classes: [spam, not_spam]
Output:  [0.7, 0.2, 0.1]         Output:  0.8 (spam probability)
Must sum to 1.0 ✅               Must be between 0 and 1 ✅
Uses: CrossEntropyLoss            Uses: BinaryCrossEntropyLoss
```
"""

# %% nbgrader={"grade": false, "grade_id": "cross_entropy_loss", "solution": true}
#| export
class CrossEntropyLoss:
    """Cross-entropy loss for multi-class classification."""

    def __init__(self):
        """Initialize cross-entropy loss function."""
        pass

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute cross-entropy loss between logits and target class indices.

        TODO: Implement cross-entropy loss with numerical stability

        APPROACH:
        1. Compute log-softmax of logits (numerically stable)
        2. Select log-probabilities for correct classes
        3. Return negative mean of selected log-probabilities

        EXAMPLE:
        >>> loss_fn = CrossEntropyLoss()
        >>> logits = Tensor([[2.0, 1.0, 0.1], [0.5, 1.5, 0.8]])  # 2 samples, 3 classes
        >>> targets = Tensor([0, 1])  # First sample is class 0, second is class 1
        >>> loss = loss_fn(logits, targets)
        >>> print(f"Cross-Entropy Loss: {loss.data:.4f}")

        HINTS:
        - Use log_softmax() for numerical stability
        - targets.data.astype(int) ensures integer indices
        - Use np.arange(batch_size) for row indexing: log_probs[np.arange(batch_size), targets]
        - Return negative mean: -np.mean(selected_log_probs)
        """
        ### BEGIN SOLUTION
        # Step 1: Compute log-softmax for numerical stability
        log_probs = log_softmax(logits, dim=-1)

        # Step 2: Select log-probabilities for correct classes
        batch_size = logits.shape[0]
        target_indices = targets.data.astype(int)

        # Select correct class log-probabilities using advanced indexing
        selected_log_probs = log_probs.data[np.arange(batch_size), target_indices]

        # Step 3: Return negative mean (cross-entropy is negative log-likelihood)
        cross_entropy = -np.mean(selected_log_probs)

        return Tensor(cross_entropy)
        ### END SOLUTION

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Allows the loss function to be called like a function."""
        return self.forward(logits, targets)

    def backward(self) -> Tensor:
        """
        Compute gradients (implemented in Module 06: Autograd).

        For now, this is a stub that students can ignore.
        """
        pass

# %% nbgrader={"grade": true, "grade_id": "test_cross_entropy_loss", "locked": true, "points": 10}
def test_unit_cross_entropy_loss():
    """🔬 Test CrossEntropyLoss implementation and properties."""
    print("🔬 Unit Test: Cross-Entropy Loss...")

    loss_fn = CrossEntropyLoss()

    # Test perfect predictions (should have very low loss)
    perfect_logits = Tensor([[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]])  # Very confident predictions
    targets = Tensor([0, 1])  # Matches the confident predictions
    perfect_loss = loss_fn.forward(perfect_logits, targets)
    assert perfect_loss.data < 0.01, f"Perfect predictions should have very low loss, got {perfect_loss.data}"

    # Test uniform predictions (should have loss ≈ log(num_classes))
    uniform_logits = Tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])  # Equal probabilities
    uniform_targets = Tensor([0, 1])
    uniform_loss = loss_fn.forward(uniform_logits, uniform_targets)
    expected_uniform_loss = np.log(3)  # log(3) ≈ 1.099 for 3 classes
    assert np.allclose(uniform_loss.data, expected_uniform_loss, atol=0.1), f"Uniform predictions should have loss ≈ log(3) = {expected_uniform_loss:.3f}, got {uniform_loss.data:.3f}"

    # Test that wrong confident predictions have high loss
    wrong_logits = Tensor([[10.0, -10.0, -10.0], [-10.0, -10.0, 10.0]])  # Confident but wrong
    wrong_targets = Tensor([1, 1])  # Opposite of confident predictions
    wrong_loss = loss_fn.forward(wrong_logits, wrong_targets)
    assert wrong_loss.data > 5.0, f"Wrong confident predictions should have high loss, got {wrong_loss.data}"

    # Test numerical stability with large logits
    large_logits = Tensor([[100.0, 50.0, 25.0]])
    large_targets = Tensor([0])
    large_loss = loss_fn.forward(large_logits, large_targets)
    assert not np.isnan(large_loss.data), "Loss should not be NaN with large logits"
    assert not np.isinf(large_loss.data), "Loss should not be infinite with large logits"

    print("✅ CrossEntropyLoss works correctly!")

if __name__ == "__main__":
    test_unit_cross_entropy_loss()

# %% [markdown]
"""
## 🏗️ BinaryCrossEntropyLoss - Measuring Yes/No Decision Quality

Binary Cross-Entropy is specialized for yes/no decisions. It's like regular cross-entropy but optimized for the special case of exactly two classes.

### When to Use Binary Cross-Entropy

**Perfect for:**
- Spam detection (spam vs not spam)
- Medical diagnosis (disease vs healthy)
- Fraud detection (fraud vs legitimate)
- Content moderation (toxic vs safe)
- Any two-class decision problem

### Understanding Binary Cross-Entropy

```
Binary Classification Decision Matrix:

                 TRUE LABEL
              Positive  Negative
PREDICTED  P    TP       FP     ← Model says "Yes"
           N    FN       TN     ← Model says "No"

BCE Loss for each quadrant:
- True Positive (TP): -log(prediction)    ← Reward confident correct "Yes"
- False Positive (FP): -log(1-prediction) ← Punish confident wrong "Yes"
- False Negative (FN): -log(prediction)   ← Punish confident wrong "No"
- True Negative (TN): -log(1-prediction)  ← Reward confident correct "No"
```

### Binary Cross-Entropy Behavior Examples

```
Scenario: Spam Detection

Case 1: Perfect Spam Detection
Email: "Buy now! 50% off! Limited time!"
Model Prediction: 0.99 (99% spam probability)
True Label: 1 (actually spam)
Loss: -log(0.99) = 0.01  ← Very low loss ✅

Case 2: Uncertain About Spam
Email: "Meeting rescheduled to 2pm"
Model Prediction: 0.51 (slightly thinks spam)
True Label: 0 (actually not spam)
Loss: -log(1-0.51) = -log(0.49) = 0.71  ← Moderate loss

Case 3: Confident Wrong Prediction
Email: "Hi mom, how are you?"
Model Prediction: 0.95 (very confident spam)
True Label: 0 (actually not spam)
Loss: -log(1-0.95) = -log(0.05) = 3.0  ← High loss ❌
```

### Binary vs Multi-Class Cross-Entropy

```
Binary Cross-Entropy:              Regular Cross-Entropy:

Single probability output         Probability distribution output
Predict: 0.8 (spam prob)         Predict: [0.1, 0.8, 0.1] (3 classes)
Target: 1.0 (is spam)            Target: 1 (class index)

Formula:                         Formula:
-[y*log(p) + (1-y)*log(1-p)]    -log(p[target_class])

Handles class imbalance well     Assumes balanced classes
Optimized for 2-class case      General for N classes
```

### Why Binary Cross-Entropy is Special

1. **Symmetric penalties**: False positives and false negatives treated equally
2. **Probability calibration**: Output directly interpretable as probability
3. **Efficient computation**: Simpler than full softmax for binary cases
4. **Medical-grade**: Well-suited for safety-critical binary decisions

### Loss Landscape Visualization

```
Binary Cross-Entropy Loss Surface:

     Loss
      ^
   10 |*                    *     ← Wrong confident predictions
      ||
    5 | *                 *
      |  \\               /
    2 |   *             *          ← Uncertain predictions
      |    \\           /
    0 |_____*_______*_____> Prediction
      0    0.2     0.8    1.0

      Target = 1.0 (positive class)

Message: "Be confident about positive class, uncertain is okay,
         but don't be confident about wrong class!"
```
"""

# %% nbgrader={"grade": false, "grade_id": "binary_cross_entropy_loss", "solution": true}
#| export
class BinaryCrossEntropyLoss:
    """Binary cross-entropy loss for binary classification."""

    def __init__(self):
        """Initialize binary cross-entropy loss function."""
        pass

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute binary cross-entropy loss.

        TODO: Implement binary cross-entropy with numerical stability

        APPROACH:
        1. Clamp predictions to avoid log(0) and log(1)
        2. Compute: -(targets * log(predictions) + (1-targets) * log(1-predictions))
        3. Return mean across all samples

        EXAMPLE:
        >>> loss_fn = BinaryCrossEntropyLoss()
        >>> predictions = Tensor([0.9, 0.1, 0.7, 0.3])  # Probabilities between 0 and 1
        >>> targets = Tensor([1.0, 0.0, 1.0, 0.0])      # Binary labels
        >>> loss = loss_fn(predictions, targets)
        >>> print(f"Binary Cross-Entropy Loss: {loss.data:.4f}")

        HINTS:
        - Use np.clip(predictions.data, 1e-7, 1-1e-7) to prevent log(0)
        - Binary cross-entropy: -(targets * log(preds) + (1-targets) * log(1-preds))
        - Use np.mean() to average over all samples
        """
        ### BEGIN SOLUTION
        # Step 1: Clamp predictions to avoid numerical issues with log(0) and log(1)
        eps = EPSILON
        clamped_preds = np.clip(predictions.data, eps, 1 - eps)

        # Step 2: Compute binary cross-entropy
        # BCE = -(targets * log(preds) + (1-targets) * log(1-preds))
        log_preds = np.log(clamped_preds)
        log_one_minus_preds = np.log(1 - clamped_preds)

        bce_per_sample = -(targets.data * log_preds + (1 - targets.data) * log_one_minus_preds)

        # Step 3: Return mean across all samples
        bce_loss = np.mean(bce_per_sample)

        return Tensor(bce_loss)
        ### END SOLUTION

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Allows the loss function to be called like a function."""
        return self.forward(predictions, targets)

    def backward(self) -> Tensor:
        """
        Compute gradients (implemented in Module 06: Autograd).

        For now, this is a stub that students can ignore.
        """
        pass

# %% nbgrader={"grade": true, "grade_id": "test_binary_cross_entropy_loss", "locked": true, "points": 10}
def test_unit_binary_cross_entropy_loss():
    """🔬 Test BinaryCrossEntropyLoss implementation and properties."""
    print("🔬 Unit Test: Binary Cross-Entropy Loss...")

    loss_fn = BinaryCrossEntropyLoss()

    # Test perfect predictions
    perfect_predictions = Tensor([0.9999, 0.0001, 0.9999, 0.0001])
    targets = Tensor([1.0, 0.0, 1.0, 0.0])
    perfect_loss = loss_fn.forward(perfect_predictions, targets)
    assert perfect_loss.data < 0.01, f"Perfect predictions should have very low loss, got {perfect_loss.data}"

    # Test worst predictions
    worst_predictions = Tensor([0.0001, 0.9999, 0.0001, 0.9999])
    worst_targets = Tensor([1.0, 0.0, 1.0, 0.0])
    worst_loss = loss_fn.forward(worst_predictions, worst_targets)
    assert worst_loss.data > 5.0, f"Worst predictions should have high loss, got {worst_loss.data}"

    # Test uniform predictions (probability = 0.5)
    uniform_predictions = Tensor([0.5, 0.5, 0.5, 0.5])
    uniform_targets = Tensor([1.0, 0.0, 1.0, 0.0])
    uniform_loss = loss_fn.forward(uniform_predictions, uniform_targets)
    expected_uniform = -np.log(0.5)  # Should be about 0.693
    assert np.allclose(uniform_loss.data, expected_uniform, atol=0.01), f"Uniform predictions should have loss ≈ {expected_uniform:.3f}, got {uniform_loss.data:.3f}"

    # Test numerical stability at boundaries
    boundary_predictions = Tensor([0.0, 1.0, 0.0, 1.0])
    boundary_targets = Tensor([0.0, 1.0, 1.0, 0.0])
    boundary_loss = loss_fn.forward(boundary_predictions, boundary_targets)
    assert not np.isnan(boundary_loss.data), "Loss should not be NaN at boundaries"
    assert not np.isinf(boundary_loss.data), "Loss should not be infinite at boundaries"

    print("✅ BinaryCrossEntropyLoss works correctly!")

if __name__ == "__main__":
    test_unit_binary_cross_entropy_loss()

# %% [markdown]
"""
## 🔧 Integration - Bringing It Together

Now let's test how our loss functions work together with real data scenarios and explore their behavior with different types of predictions.

## 🔧 Real-World Loss Function Usage Patterns

Understanding when and why to use each loss function is crucial for ML engineering success:

```
Problem Type Decision Tree:

What are you predicting?
         │
    ┌────┼────┐
    │         │
Continuous   Categorical
 Values       Classes
    │         │
    │    ┌───┼───┐
    │    │       │
    │   2 Classes  3+ Classes
    │       │       │
 MSELoss   BCE Loss  CE Loss

Examples:
MSE: House prices, temperature, stock values
BCE: Spam detection, fraud detection, medical diagnosis
CE:  Image classification, language modeling, multiclass text classification
```

## 🔧 Loss Function Behavior Comparison

Each loss function creates different learning pressures on your model:

```
Error Sensitivity Comparison:

Small Error (0.1):     Medium Error (0.5):     Large Error (2.0):

MSE:     0.01         MSE:     0.25           MSE:     4.0
BCE:     0.11         BCE:     0.69           BCE:     ∞ (clips to large)
CE:      0.11         CE:      0.69           CE:      ∞ (clips to large)

MSE: Quadratic growth, manageable with outliers
BCE/CE: Logarithmic growth, explodes with confident wrong predictions
```
"""

# %% nbgrader={"grade": false, "grade_id": "loss_comparison", "solution": true}
def analyze_loss_behaviors():
    """
    📊 Compare how different loss functions behave with various prediction patterns.

    This helps students understand when to use each loss function.
    """
    print("📊 Analysis: Loss Function Behavior Comparison...")

    # Initialize loss functions
    mse_loss = MSELoss()
    ce_loss = CrossEntropyLoss()
    bce_loss = BinaryCrossEntropyLoss()

    print("\n1. Regression Scenario (House Price Prediction)")
    print("   Predictions: [200k, 250k, 300k], Targets: [195k, 260k, 290k]")
    house_pred = Tensor([200.0, 250.0, 300.0])  # In thousands
    house_target = Tensor([195.0, 260.0, 290.0])
    mse = mse_loss.forward(house_pred, house_target)
    print(f"   MSE Loss: {mse.data:.2f} (thousand²)")

    print("\n2. Multi-Class Classification (Image Recognition)")
    print("   Classes: [cat, dog, bird], Predicted: confident about cat, uncertain about dog")
    # Logits: [2.0, 0.5, 0.1] suggests model is most confident about class 0 (cat)
    image_logits = Tensor([[2.0, 0.5, 0.1], [0.3, 1.8, 0.2]])  # Two samples
    image_targets = Tensor([0, 1])  # First is cat (0), second is dog (1)
    ce = ce_loss.forward(image_logits, image_targets)
    print(f"   Cross-Entropy Loss: {ce.data:.3f}")

    print("\n3. Binary Classification (Spam Detection)")
    print("   Predictions: [0.9, 0.1, 0.7, 0.3] (spam probabilities)")
    spam_pred = Tensor([0.9, 0.1, 0.7, 0.3])
    spam_target = Tensor([1.0, 0.0, 1.0, 0.0])  # 1=spam, 0=not spam
    bce = bce_loss.forward(spam_pred, spam_target)
    print(f"   Binary Cross-Entropy Loss: {bce.data:.3f}")

    print("\n💡 Key Insights:")
    print("   - MSE penalizes large errors heavily (good for continuous values)")
    print("   - Cross-Entropy encourages confident correct predictions")
    print("   - Binary Cross-Entropy balances false positives and negatives")

    return mse.data, ce.data, bce.data


# %% nbgrader={"grade": false, "grade_id": "loss_sensitivity", "solution": true}
def analyze_loss_sensitivity():
    """
    📊 Analyze how sensitive each loss function is to prediction errors.

    This demonstrates the different error landscapes created by each loss.
    """
    print("\n📊 Analysis: Loss Function Sensitivity to Errors...")

    # Create a range of prediction errors for analysis
    true_value = 1.0
    predictions = np.linspace(0.1, 1.9, 50)  # From 0.1 to 1.9

    # Initialize loss functions
    mse_loss = MSELoss()
    bce_loss = BinaryCrossEntropyLoss()

    mse_losses = []
    bce_losses = []

    for pred in predictions:
        # MSE analysis
        pred_tensor = Tensor([pred])
        target_tensor = Tensor([true_value])
        mse = mse_loss.forward(pred_tensor, target_tensor)
        mse_losses.append(mse.data)

        # BCE analysis (clamp prediction to valid probability range)
        clamped_pred = max(0.01, min(0.99, pred))
        bce_pred_tensor = Tensor([clamped_pred])
        bce_target_tensor = Tensor([1.0])  # Target is "positive class"
        bce = bce_loss.forward(bce_pred_tensor, bce_target_tensor)
        bce_losses.append(bce.data)

    # Find minimum losses
    min_mse_idx = np.argmin(mse_losses)
    min_bce_idx = np.argmin(bce_losses)

    print(f"MSE Loss:")
    print(f"  Minimum at prediction = {predictions[min_mse_idx]:.2f}, loss = {mse_losses[min_mse_idx]:.4f}")
    print(f"  At prediction = 0.5: loss = {mse_losses[24]:.4f}")  # Middle of range
    print(f"  At prediction = 0.1: loss = {mse_losses[0]:.4f}")

    print(f"\nBinary Cross-Entropy Loss:")
    print(f"  Minimum at prediction = {predictions[min_bce_idx]:.2f}, loss = {bce_losses[min_bce_idx]:.4f}")
    print(f"  At prediction = 0.5: loss = {bce_losses[24]:.4f}")
    print(f"  At prediction = 0.1: loss = {bce_losses[0]:.4f}")

    print(f"\n💡 Sensitivity Insights:")
    print("   - MSE grows quadratically with error distance")
    print("   - BCE grows logarithmically, heavily penalizing wrong confident predictions")
    print("   - Both encourage correct predictions but with different curvatures")

# Run integration analysis when developing
if __name__ == "__main__":
    analyze_loss_behaviors()
    analyze_loss_sensitivity()

# %% [markdown]
"""
## 📊 Systems Analysis - Understanding Loss Function Performance

Loss functions seem simple, but they have important computational and numerical properties that affect training performance. Let's analyze the systems aspects.

## 📊 Computational Complexity Analysis

Different loss functions have different computational costs, especially at scale:

```
Computational Cost Comparison (Batch Size B, Classes C):

MSELoss:
┌────────────────┬────────────────┐
│ Operation      │ Complexity     │
├────────────────┼────────────────┤
│ Subtraction    │ O(B)           │
│ Squaring       │ O(B)           │
│ Mean           │ O(B)           │
│ Total          │ O(B)           │
└────────────────┴────────────────┘

CrossEntropyLoss:
┌────────────────┬────────────────┐
│ Operation      │ Complexity     │
├────────────────┼────────────────┤
│ Max (stability)│ O(B*C)         │
│ Exponential    │ O(B*C)         │
│ Sum            │ O(B*C)         │
│ Log            │ O(B)           │
│ Indexing       │ O(B)           │
│ Total          │ O(B*C)         │
└────────────────┴────────────────┘

Cross-entropy is C times more expensive than MSE!
For ImageNet (C=1000), CE is 1000x more expensive than MSE.
```

## 📊 Memory Layout and Access Patterns

```
Memory Usage Patterns:

MSE Forward Pass:              CE Forward Pass:

Input:  [B] predictions       Input:  [B, C] logits
       │                             │
       │ subtract                    │ subtract max
       v                             v
Temp:  [B] differences        Temp1: [B, C] shifted
       │                             │
       │ square                      │ exponential
       v                             v
Temp:  [B] squared            Temp2: [B, C] exp_vals
       │                             │
       │ mean                        │ sum along C
       v                             v
Output: [1] scalar            Temp3: [B] sums
                                     │
Memory: 3*B*sizeof(float)            │ log + index
                                     v
                              Output: [1] scalar

                              Memory: (3*B*C + 2*B)*sizeof(float)
```
"""

# %% nbgrader={"grade": false, "grade_id": "analyze_numerical_stability", "solution": true}
def analyze_numerical_stability():
    """
    📊 Demonstrate why numerical stability matters in loss computation.

    Shows the difference between naive and stable implementations.
    """
    print("📊 Analysis: Numerical Stability in Loss Functions...")

    # Test with increasingly large logits
    test_cases = [
        ("Small logits", [1.0, 2.0, 3.0]),
        ("Medium logits", [10.0, 20.0, 30.0]),
        ("Large logits", [100.0, 200.0, 300.0]),
        ("Very large logits", [500.0, 600.0, 700.0])
    ]

    print("\nLog-Softmax Stability Test:")
    print("Case                 | Max Input | Log-Softmax Min | Numerically Stable?")
    print("-" * 70)

    for case_name, logits in test_cases:
        x = Tensor([logits])

        # Our stable implementation
        stable_result = log_softmax(x, dim=-1)

        max_input = np.max(logits)
        min_output = np.min(stable_result.data)
        is_stable = not (np.any(np.isnan(stable_result.data)) or np.any(np.isinf(stable_result.data)))

        print(f"{case_name:20} | {max_input:8.0f} | {min_output:15.3f} | {'✅ Yes' if is_stable else '❌ No'}")

    print(f"\n💡 Key Insight: Log-sum-exp trick prevents overflow")
    print("   Without it: exp(700) would cause overflow in standard softmax")
    print("   With it: We can handle arbitrarily large logits safely")


# %% nbgrader={"grade": false, "grade_id": "analyze_loss_memory", "solution": true}
def analyze_loss_memory():
    """
    📊 Analyze memory usage patterns of different loss functions.

    Understanding memory helps with batch size decisions.
    """
    print("\n📊 Analysis: Loss Function Memory Usage...")

    batch_sizes = [32, 128, 512, 1024]
    num_classes = 1000  # Like ImageNet

    print("\nMemory Usage by Batch Size:")
    print("Batch Size | MSE (MB) | CrossEntropy (MB) | BCE (MB) | Notes")
    print("-" * 75)

    for batch_size in batch_sizes:
        # Memory calculations (assuming float32 = 4 bytes)
        bytes_per_float = 4

        # MSE: predictions + targets (both same size as output)
        mse_elements = batch_size * 1  # Regression usually has 1 output
        mse_memory = mse_elements * bytes_per_float * 2 / 1e6  # Convert to MB

        # CrossEntropy: logits + targets + softmax + log_softmax
        ce_logits = batch_size * num_classes
        ce_targets = batch_size * 1  # Target indices
        ce_softmax = batch_size * num_classes  # Intermediate softmax
        ce_total_elements = ce_logits + ce_targets + ce_softmax
        ce_memory = ce_total_elements * bytes_per_float / 1e6

        # BCE: predictions + targets (binary, so smaller)
        bce_elements = batch_size * 1
        bce_memory = bce_elements * bytes_per_float * 2 / 1e6

        notes = "Linear scaling" if batch_size == 32 else f"{batch_size//32}× first"

        print(f"{batch_size:10} | {mse_memory:8.2f} | {ce_memory:13.2f} | {bce_memory:7.2f} | {notes}")

    print(f"\n💡 Memory Insights:")
    print("   - CrossEntropy dominates due to large vocabulary (num_classes)")
    print("   - Memory scales linearly with batch size")
    print("   - Intermediate activations (softmax) double CE memory")
    print(f"   - For batch=1024, CE needs {ce_memory:.1f}MB just for loss computation")

# Run systems analysis when developing
if __name__ == "__main__":
    analyze_numerical_stability()
    analyze_loss_memory()

# %% [markdown]
"""
## 📊 Production Context - How Loss Functions Scale

Understanding how loss functions behave in production helps make informed engineering decisions about model architecture and training strategies.

## 📊 Loss Function Scaling Challenges

As models grow larger, loss function bottlenecks become critical:

```
Scaling Challenge Matrix:

                    │ Small Model     │ Large Model      │ Production Scale  │
                    │ (MNIST)         │ (ImageNet)       │ (GPT/BERT)        │
────────────────────┼─────────────────┼──────────────────┼───────────────────┤
Classes (C)         │ 10              │ 1,000            │ 50,000+           │
Batch Size (B)      │ 64              │ 256              │ 2,048             │
Memory (CE)         │ 2.5 KB          │ 1 MB             │ 400 MB            │
Memory (MSE)        │ 0.25 KB         │ 1 KB             │ 8 KB              │
Bottleneck          │ None            │ Softmax compute  │ Vocabulary memory │

Memory grows as B*C for cross-entropy!
At scale, vocabulary (C) dominates everything.
```

## 📊 Engineering Optimizations in Production

```
Common Production Optimizations:

1. Hierarchical Softmax:
   ┌─────────────────────┐     ┌─────────────────────┐
   │ Full Softmax:       │     │ Hierarchical:       │
   │ O(V) per sample     │ →   │ O(log V) per sample │
   │ 50k classes = 50k   │     │ 50k classes = 16    │
   │ operations          │     │ operations          │
   └─────────────────────┘     └─────────────────────┘

2. Sampled Softmax:
   Instead of computing over all 50k classes,
   sample 1k negative classes + correct class.
   50× speedup for training!

3. Label Smoothing:
   Instead of hard targets [0, 0, 1, 0],
   use soft targets [0.1, 0.1, 0.7, 0.1].
   Improves generalization.

4. Mixed Precision:
   Use FP16 for forward pass, FP32 for loss.
   2× memory reduction, same accuracy.
```
"""

# %% nbgrader={"grade": false, "grade_id": "analyze_production_patterns", "solution": true}
def analyze_production_patterns():
    """
    🚀 Analyze loss function patterns in production ML systems.

    Real insights from systems perspective.
    """
    print("🚀 Production Analysis: Loss Function Engineering Patterns...")

    print("\n1. Loss Function Choice by Problem Type:")

    scenarios = [
        ("Recommender Systems", "BCE/MSE", "User preference prediction", "Billions of interactions"),
        ("Computer Vision", "CrossEntropy", "Image classification", "1000+ classes, large batches"),
        ("NLP Translation", "CrossEntropy", "Next token prediction", "50k+ vocabulary"),
        ("Medical Diagnosis", "BCE", "Disease probability", "Class imbalance critical"),
        ("Financial Trading", "MSE/Huber", "Price prediction", "Outlier robustness needed")
    ]

    print("System Type          | Loss Type    | Use Case              | Scale Challenge")
    print("-" * 80)
    for system, loss_type, use_case, challenge in scenarios:
        print(f"{system:20} | {loss_type:12} | {use_case:20} | {challenge}")

    print("\n2. Engineering Trade-offs:")

    trade_offs = [
        ("CrossEntropy vs Label Smoothing", "Stability vs Confidence", "Label smoothing prevents overconfident predictions"),
        ("MSE vs Huber Loss", "Sensitivity vs Robustness", "Huber is less sensitive to outliers"),
        ("Full Softmax vs Sampled", "Accuracy vs Speed", "Hierarchical softmax for large vocabularies"),
        ("Per-Sample vs Batch Loss", "Accuracy vs Memory", "Batch computation is more memory efficient")
    ]

    print("\nTrade-off                    | Spectrum              | Production Decision")
    print("-" * 85)
    for trade_off, spectrum, decision in trade_offs:
        print(f"{trade_off:28} | {spectrum:20} | {decision}")

    print("\n💡 Production Insights:")
    print("   - Large vocabularies (50k+ tokens) dominate memory in CrossEntropy")
    print("   - Batch computation is 10-100× more efficient than per-sample")
    print("   - Numerical stability becomes critical at scale (FP16 training)")
    print("   - Loss computation is often <5% of total training time")

# Run production analysis when developing
if __name__ == "__main__":
    analyze_production_patterns()

# %% [markdown]
"""
## 🧪 Module Integration Test

Final validation that everything works together correctly.
"""


# %% nbgrader={"grade": true, "grade_id": "test_module", "locked": true, "points": 20}
def test_module():
    """🧪 Module Test: Complete Integration

    Comprehensive test of entire losses module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_log_softmax()
    test_unit_mse_loss()
    test_unit_cross_entropy_loss()
    test_unit_binary_cross_entropy_loss()

    print("\nRunning integration scenarios...")

    # Test realistic end-to-end scenario with previous modules
    print("🔬 Integration Test: Realistic training scenario...")

    # Simulate a complete prediction -> loss computation pipeline

    # 1. MSE for regression (house price prediction)
    house_predictions = Tensor([250.0, 180.0, 320.0, 400.0])  # Predicted prices in thousands
    house_actual = Tensor([245.0, 190.0, 310.0, 420.0])       # Actual prices
    mse_loss = MSELoss()
    house_loss = mse_loss.forward(house_predictions, house_actual)
    assert house_loss.data > 0, "House price loss should be positive"
    assert house_loss.data < 1000, "House price loss should be reasonable"

    # 2. CrossEntropy for classification (image recognition)
    image_logits = Tensor([[2.1, 0.5, 0.3], [0.2, 2.8, 0.1], [0.4, 0.3, 2.2]])  # 3 images, 3 classes
    image_labels = Tensor([0, 1, 2])  # Correct class for each image
    ce_loss = CrossEntropyLoss()
    image_loss = ce_loss.forward(image_logits, image_labels)
    assert image_loss.data > 0, "Image classification loss should be positive"
    assert image_loss.data < 5.0, "Image classification loss should be reasonable"

    # 3. BCE for binary classification (spam detection)
    spam_probabilities = Tensor([0.85, 0.12, 0.78, 0.23, 0.91])
    spam_labels = Tensor([1.0, 0.0, 1.0, 0.0, 1.0])  # True spam labels
    bce_loss = BinaryCrossEntropyLoss()
    spam_loss = bce_loss.forward(spam_probabilities, spam_labels)
    assert spam_loss.data > 0, "Spam detection loss should be positive"
    assert spam_loss.data < 5.0, "Spam detection loss should be reasonable"

    # 4. Test numerical stability with extreme values
    extreme_logits = Tensor([[100.0, -100.0, 0.0]])
    extreme_targets = Tensor([0])
    extreme_loss = ce_loss.forward(extreme_logits, extreme_targets)
    assert not np.isnan(extreme_loss.data), "Loss should handle extreme values"
    assert not np.isinf(extreme_loss.data), "Loss should not be infinite"

    print("✅ End-to-end loss computation works!")
    print("✅ All loss functions handle edge cases!")
    print("✅ Numerical stability verified!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 04")


# %%
# Run comprehensive module test
if __name__ == "__main__":
    test_module()


# %% [markdown]
"""
## 🤔 ML Systems Questions - Testing Your Understanding

Before we finish, let's reflect on what you've learned about loss functions from a systems perspective.

### Memory and Performance

**Question 1: Loss Function Selection for Large Vocabulary**

You're building a language model with a 50,000 word vocabulary. Your GPU has 16GB of memory, and you want to use batch size 128.

Calculate:
- How much memory does CrossEntropyLoss need for one forward pass? (Hint: B=128, C=50,000, float32)
- If this exceeds your budget, what are three strategies to reduce memory usage?

<details>
<summary>💡 Hint</summary>

Memory for logits = Batch_Size × Num_Classes × 4 bytes (float32) = 128 × 50,000 × 4 = 25.6 MB

For full forward pass with intermediate tensors (softmax, log_softmax), multiply by ~3 = 76.8 MB

Strategies to reduce memory:
1. **Sampled softmax**: Only compute softmax over subset of vocabulary (1000 samples)
2. **Hierarchical softmax**: Use tree structure, O(log V) instead of O(V)
3. **Mixed precision**: Use FP16 for forward pass (2 bytes instead of 4)
4. **Gradient checkpointing**: Recompute intermediate activations instead of storing
</details>

---

**Question 2: Loss Function Performance Bottleneck**

You profile your training loop and find:
- Forward pass (model): 80ms
- Loss computation: 120ms
- Backward pass: 150ms

Your model has 1000 output classes. What's the bottleneck and how would you fix it?

<details>
<summary>💡 Hint</summary>

**Bottleneck**: Loss computation (120ms) taking longer than forward pass (80ms) is unusual.

**Root Cause**: Softmax computation in CrossEntropyLoss is O(B×C). With C=1000, this dominates.

**Solutions**:
1. **Hierarchical softmax**: Reduces complexity from O(C) to O(log C)
2. **Sampled softmax**: Only compute over subset of classes during training
3. **Optimize softmax kernel**: Use fused operations (PyTorch does this automatically)
4. **Check batch size**: Very small batches don't utilize GPU well

**Reality Check**: In well-optimized PyTorch, loss should be ~5-10% of training time, not 35%!
</details>

---

### Numerical Stability

**Question 3: Debugging Exploding Loss**

During training, you see:
```
Epoch 1: Loss = 2.3
Epoch 2: Loss = 1.8
Epoch 3: Loss = inf
```

The model uses CrossEntropyLoss with raw logits reaching values like [150, -80, 200].

Why did loss become infinite? What code change fixes this?

<details>
<summary>💡 Hint</summary>

**Root Cause**: Without the log-sum-exp trick, computing softmax directly causes:
```python
exp(200) = 7.2 × 10^86  # Overflows to infinity in float32
```

**The Fix**: Use log_softmax with max subtraction (already implemented in your code!):
```python
# ❌ Naive approach (causes overflow)
softmax = np.exp(logits) / np.sum(np.exp(logits))
loss = -np.log(softmax[target])

# ✅ Stable approach (your implementation)
log_softmax = logits - np.max(logits) - np.log(np.sum(np.exp(logits - np.max(logits))))
loss = -log_softmax[target]
```

**Verification**: Your `log_softmax()` function handles this automatically. Check that you're using it in `CrossEntropyLoss.forward()`.

**Prevention**: Always use log-space computations for probabilities!
</details>

---

### Production Considerations

**Question 4: Real-Time Inference Latency**

Your spam filter needs to classify emails in <10ms. Currently:
- Model inference: 3ms
- Loss computation: 8ms (❓ Why are we computing loss?)

Your inference code looks like:
```python
prediction = model(email)
confidence = bce_loss(prediction, threshold)  # Using loss for confidence?
```

What's wrong with this approach, and how would you fix it?

<details>
<summary>💡 Hint</summary>

**Critical Mistake**: Loss functions are for **training**, not **inference**!

**Why it's wrong**:
- Loss requires ground truth labels (not available at inference time)
- Loss computation adds unnecessary overhead
- You already have the prediction probability!

**Correct inference code**:
```python
prediction = model(email)  # Returns probability between 0 and 1
is_spam = prediction.data > 0.5  # Simple threshold

# If you need confidence score:
confidence = abs(prediction.data - 0.5) * 2  # Distance from decision boundary
# Or just use the raw probability: prediction.data
```

**Performance gain**: 3ms (73% faster!) just by removing unnecessary loss computation.

**Key insight**: Loss functions measure "wrongness" during training. At inference, you already have the model's output - use it directly!
</details>

---

**Question 5: Class Imbalance in Medical Diagnosis**

You're building a cancer detection system:
- 95% of samples are negative (healthy)
- 5% are positive (cancer)

Using vanilla BinaryCrossEntropyLoss, your model achieves 95% accuracy by always predicting "healthy."

What are three ways to handle this with loss functions?

<details>
<summary>💡 Hint</summary>

**The Problem**: Model learned to exploit class imbalance - always predict majority class!

**Solution 1: Weighted Loss**
```python
class WeightedBCELoss:
    def __init__(self, pos_weight=19.0):  # 95/5 = 19
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        loss = -(self.pos_weight * target * np.log(pred) +
                 (1-target) * np.log(1-pred))
        return np.mean(loss)
```
Penalize missed cancer cases 19× more than false alarms.

**Solution 2: Focal Loss**
```python
# Focuses on hard examples (misclassified samples)
focal_loss = -(1 - p_correct)^gamma * log(p_correct)
```
Automatically downweights easy examples (majority class).

**Solution 3: Resampling**
- Oversample minority class (duplicate cancer cases)
- Undersample majority class (fewer healthy samples)
- SMOTE (Synthetic Minority Over-sampling Technique)

**Medical Reality**: Weighted loss is most common. False negatives (missed cancer) are MUCH worse than false positives (unnecessary tests).

**Critical Insight**: 95% accuracy is meaningless! Track precision, recall, F1, and AUC instead.
</details>

---

### Systems Thinking

**Question 6: Batch Size and Loss Computation**

You're training on a GPU with 24GB memory. With batch size 32, memory usage is 8GB. You increase batch size to 128.

Will memory usage be 32GB (4× increase)? Why or why not?

What happens to:
- Loss computation time?
- Loss value (the actual number)?
- Gradient quality?

<details>
<summary>💡 Hint</summary>

**Memory Usage**: YES, approximately 32GB (4× increase) - **EXCEEDS GPU MEMORY! Training will crash.**

**Why linear scaling?**
```
Memory = Model_Params + Batch_Size × (Activations + Gradients + Optimizer_State)
         ↑              ↑
      Fixed (1GB)     Scales linearly (7GB → 28GB)
```

**Loss computation time**: ~4× slower (linear with batch size)
- 32 samples: 0.5ms
- 128 samples: 2.0ms

**Loss value**: **SAME** (we take mean over batch)
```python
# Both compute the same thing:
batch_32_loss = np.mean(losses[:32])   # Mean of 32 samples
batch_128_loss = np.mean(losses[:128]) # Mean of 128 samples
```

**Gradient quality**: **BETTER** - larger batch = more stable gradient estimate
- Batch 32: High variance, noisy gradients
- Batch 128: Lower variance, smoother convergence

**The Trade-off**:
- Larger batch = better gradients but more memory
- Smaller batch = less memory but noisier training
- Sweet spot: Usually 64-256 depending on GPU memory

**Production Solution**: Gradient accumulation
```python
# Simulate batch_size=128 with only batch_size=32 memory:
for micro_batch in range(4):  # 4 × 32 = 128
    loss = compute_loss(micro_batch)
    loss.backward()  # Accumulate gradients
optimizer.step()  # Update once with accumulated gradients
```
</details>

---

These questions test your systems understanding of loss functions - not just "how do they work" but "how do they behave in production at scale." Keep these considerations in mind as you build real ML systems!
"""

# %% [markdown]
"""
## ⭐ Aha Moment: Loss Guides Learning

**What you built:** Loss functions that measure how wrong predictions are.

**Why it matters:** Without loss, there's no learning. The loss function is the "coach"
that tells the network whether its predictions are good or bad. Lower loss = better
predictions. Every training step aims to reduce this number.

In the next module, you'll add autograd which computes gradients of this loss—the
direction to adjust weights to make predictions better!
"""

# %%
def demo_losses():
    """🎯 See how loss responds to prediction quality."""
    print("🎯 AHA MOMENT: Loss Guides Learning")
    print("=" * 45)

    loss_fn = MSELoss()
    target = Tensor(np.array([1.0, 0.0, 0.0]))

    # Perfect prediction
    perfect = Tensor(np.array([1.0, 0.0, 0.0]))
    loss_perfect = loss_fn(perfect, target)

    # Close prediction
    close = Tensor(np.array([0.9, 0.1, 0.1]))
    loss_close = loss_fn(close, target)

    # Wrong prediction
    wrong = Tensor(np.array([0.0, 1.0, 1.0]))
    loss_wrong = loss_fn(wrong, target)

    print(f"Perfect prediction → Loss: {float(loss_perfect.data):.4f}")
    print(f"Close prediction   → Loss: {float(loss_close.data):.4f}")
    print(f"Wrong prediction   → Loss: {float(loss_wrong.data):.4f}")

    print("\n✨ Lower loss = better predictions! Training minimizes this.")

# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_losses()

# %% [markdown]
"""
## 🚀 MODULE SUMMARY: Losses

Congratulations! You've built the measurement system that enables all machine learning!

### Key Accomplishments
- Built 3 essential loss functions: MSE, CrossEntropy, and BinaryCrossEntropy ✅
- Implemented numerical stability with log-sum-exp trick ✅
- Discovered memory scaling patterns with batch size and vocabulary ✅
- Analyzed production trade-offs between different loss function choices ✅
- All tests pass ✅ (validated by `test_module()`)

### Ready for Next Steps
Your loss functions provide the essential feedback signal for learning. These "error measurements" will become the starting point for backpropagation in Module 06 (Autograd)!
Export with: `tito module complete 04`

**Next**: Module 05 will add DataLoader for efficient data pipelines, then Module 06 adds automatic differentiation!
"""
