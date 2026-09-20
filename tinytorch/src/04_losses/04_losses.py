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
r"""
# Module 04: Losses - Measuring How Wrong We Are

Welcome to Module 04! Today you'll implement the mathematical functions that measure how wrong your model's predictions are - the essential feedback signal that enables all machine learning.

## 🔗 Prerequisites & Progress
**You've Built**: Tensors (data), Activations (intelligence), Layers (architecture)
**You'll Build**: Loss functions that measure prediction quality
**You'll Enable**: The feedback signal needed for training

**Connection Map**:

$$\underbrace{\text{Layers}}_{\text{predictions } \mathbf{\hat{y}}} \longrightarrow \underbrace{\mathbf{\text{Losses}}}_{\text{error measurement } \mathcal{L}(\mathbf{\hat{y}}, \mathbf{y})} \longrightarrow \underbrace{\text{Autograd}}_{\text{learning signals } \nabla_{\boldsymbol{\theta}}\mathcal{L}}$$

<div align="center">
  <img src="losses_blueprint.svg" width="360" alt="Framework Blueprint: Loss Functions">
</div>

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement MSELoss for regression problems
2. Implement CrossEntropyLoss for classification problems
3. Implement BinaryCrossEntropyLoss for binary classification
4. Understand numerical stability in loss computation
5. Test all loss functions with realistic examples

Let's measure prediction quality!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/04_losses/losses.ipynb`
**Building Side:** Code exports to `tinytorch.core.losses`

```python
# Final package structure:
from tinytorch.core.losses import MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss, log_softmax  # This module
```

**Why this matters:**
- **Learning:** Complete loss function system in one focused module
- **Production:** Proper organization like PyTorch's `torch.nn` loss modules (`MSELoss`, `CrossEntropyLoss`, `BCELoss`)
- **Consistency:** All loss computations and numerical stability in core.losses
- **Integration:** Works seamlessly with layers for complete prediction-to-error workflow
"""

# %% [markdown]
r"""
## 📋 Module Dependencies

**Prerequisites**: Module 01 (Tensor) must be completed

**External Dependencies**:
- `numpy` (for numerical operations)

**TinyTorch Dependencies**:
- **Module 01 (Tensor)**: Foundation for all loss computations
  - Used for: Input/output data structures, shape operations, element-wise operations
  - Required: Yes - losses operate on Tensor objects

A loss function only needs predictions and targets, so this module imports nothing
from Activations (02) or Layers (03). Those modules produce the predictions a loss
scores, and you will wire all four together in Module 08 (Training), but
`tinytorch.core.losses` itself depends on the Tensor alone.

**Dependency Flow**:

$$\begin{array}{ccc}
\text{Module 01 (Tensor)} & \longrightarrow & \mathbf{\text{Module 04 (Losses)}} \\
\downarrow & & \downarrow \\
\text{Memory and Strides} & & \text{Error Measurement}
\end{array}$$
"""

# %% nbgrader={"grade": false, "grade_id": "setup", "solution": false}
#| default_exp core.losses
#| export

import numpy as np
rng = np.random.default_rng(7)

# Import from TinyTorch package (previous modules must be completed and exported)
from tinytorch.core.tensor import Tensor, Function

# Constants for numerical stability
EPSILON = 1e-7  # Small value to prevent log(0) and numerical instability

# %% [markdown]
r"""
## 💡 Introduction: What Are Loss Functions?

Loss functions are the mathematical conscience of machine learning. They measure the distance between what your model predicts and what actually happened. Without loss functions, models have no way to improve - they're like athletes training without knowing their score.

### The Three Essential Loss Functions

Think of loss functions as different ways to measure "wrongness" - each optimized for different types of problems:

**MSELoss (Mean Squared Error)**: "How far off are my continuous predictions?"
- Used for: Regression (predicting house prices, temperature, stock values)
- Calculation: Average of squared differences between predictions and targets
- Properties: Heavily penalizes large errors, smooth gradients

$$\mathcal{L}_{\text{MSE}}(e) = e^2 \quad \text{where } e = (\hat{y} - y)$$

| Prediction Error ($e$) | $-2.0$ | $-1.0$ | $-0.5$ | $0.0$ | $+0.5$ | $+1.0$ | $+2.0$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Squared Loss ($e^2$)** | $4.00$ | $1.00$ | $0.25$ | $0.00$ | $0.25$ | $1.00$ | $4.00$ |
| **Gradient ($\frac{\partial \mathcal{L}}{\partial e}$)** | $-4.0$ | $-2.0$ | $-1.0$ | $0.0$ | $+1.0$ | $+2.0$ | $+4.0$ |

*Quadratic growth: small errors yield modest penalties, while large outlier errors yield aggressively growing gradients.*

**CrossEntropyLoss**: "How confident am I in the wrong class?"
- Used for: Multi-class classification (image recognition, text classification)
- Calculation: Negative log-likelihood of correct class probability
- Properties: Encourages confident correct predictions, punishes confident wrong ones

$$\mathcal{L}_{\text{CE}}(p) = -\ln(p) \quad \text{where } p = P(y = y^* \mid \mathbf{x}) \in (0, 1]$$

<div align="center">
  <img src="exp_overflow_cliff.svg" width="340" alt="Float32 Exp Range Cliff">
</div>

| Predicted Probability ($p$) | $0.99$ | $0.90$ | $0.50$ | $0.10$ | $0.01$ | $10^{-4}$ | $\to 0^+$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Loss ($-\ln p$)** | $0.010$ | $0.105$ | $0.693$ | $2.303$ | $4.605$ | $9.210$ | $+\infty$ |
| **Surprise / Penalty** | Negligible | Low | Moderate | Severe | Massive | Catastrophic | Asymptote |

*Logarithmic penalty: wrong confident predictions face near-vertical asymptotic gradients.*

**BinaryCrossEntropyLoss**: "How wrong am I about yes/no decisions?"
- Used for: Binary classification (spam detection, medical diagnosis)
- Calculation: Cross-entropy specialized for two classes
- Properties: Symmetric penalty for false positives and false negatives

| Prediction $\hat{y}$ | True Target $y = 1$ (Positive) | True Target $y = 0$ (Negative) | Decision Consequence |
| :--- | :--- | :--- | :--- |
| **$\hat{y} \to 1.0$ (High Conf)** | $\mathcal{L} \to 0$ (Near Zero Loss) | $\mathcal{L} \to +\infty$ (Severe Penalty) | Confident True Positive vs False Alarm |
| **$\hat{y} = 0.5$ (Uncertain)** | $\mathcal{L} = -\ln(0.5) \approx 0.693$ | $\mathcal{L} = -\ln(0.5) \approx 0.693$ | Maximum entropy / uninformative |
| **$\hat{y} \to 0.0$ (Low Conf)** | $\mathcal{L} \to +\infty$ (Severe Penalty) | $\mathcal{L} \to 0$ (Near Zero Loss) | Missed Detection vs Confident Rejection |

Each loss function creates a distinct geometric error landscape that guides optimization in different ways.
"""

# %% [markdown]
r"""
## 📐 Foundations: Mathematical Background

### Mean Squared Error (MSE)
The foundation of regression, MSE measures the average squared distance between predictions and targets across all $N$ elements:

$$\mathcal{L}_{\text{MSE}}(\mathbf{\hat{y}}, \mathbf{y}) = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

**Why square the differences?**
- Makes all errors strictly positive (no cancellation between positive and negative residuals)
- Heavily penalizes large errors (error of 2 produces penalty of 4; error of 10 produces penalty of 100)
- Yields a linear, smooth gradient $\nabla_{\hat{y}} \mathcal{L} = \frac{2}{N}(\mathbf{\hat{y}} - \mathbf{y})$ ideal for gradient descent

### Cross-Entropy Loss
For multi-class classification over $C$ mutually exclusive classes, we measure the negative log-likelihood of the true target class $y^* \in \{0, \dots, C-1\}$:

$$\mathcal{L}_{\text{CE}}(\mathbf{z}, y^*) = -\log\left(\frac{e^{z_{y^*}}}{\sum_{j=0}^{C-1} e^{z_j}}\right) = -z_{y^*} + \log\left(\sum_{j=0}^{C-1} e^{z_j}\right)$$

**The Log-Sum-Exp Trick**:
Direct computation of softmax risks immediate IEEE 754 float32 overflow when $z_j > 88.72$. Factoring out the maximum logit $c = \max_k z_k$ guarantees numerical stability:

$$\log \sum_{j=0}^{C-1} e^{z_j} = \log \sum_{j=0}^{C-1} e^{z_j - c} \cdot e^c = c + \log \left(\sum_{j=0}^{C-1} e^{z_j - c}\right)$$

$$\operatorname{log\_softmax}(\mathbf{z})_i = (z_i - c) - \log \left(\sum_{j=0}^{C-1} e^{z_j - c}\right)$$

Because $z_j - c \le 0$ for all $j$, every exponent $e^{z_j - c} \in (0, 1]$, completely eliminating the possibility of overflow.

### Binary Cross-Entropy
A specialized formulation where targets are binary labels $y \in \{0, 1\}$ and predictions are probabilities $\hat{y} \in [0, 1]$:

$$\mathcal{L}_{\text{BCE}}(\hat{y}, y) = -\big[y \ln(\hat{y}) + (1 - y) \ln(1 - \hat{y})\big]$$

The mathematics naturally handles both positive and negative cases in a single unified equation.
"""

# %% [markdown]
r"""
## 🏗️ Implementation: Building Loss Functions

Let's implement our loss functions with proper numerical stability and clear educational structure.

### Log-Softmax: The Numerically Stable Foundation

Before implementing loss functions, we need a reliable way to compute log-softmax. This function is the numerically stable backbone of classification losses.

### Why Log-Softmax Matters

Naive softmax exponentiates raw logits directly, causing catastrophic float32 overflow:

| Metric / Stage | Naive Direct Exponentiation | Numerically Stable Log-Sum-Exp |
| :--- | :--- | :--- |
| **Input Logits $\mathbf{z}$** | $[100, 200, 300]$ | $[100, 200, 300]$ |
| **Maximum $c = \max(\mathbf{z})$** | *(Unused)* | $c = 300$ |
| **Shifted Logits $\mathbf{z} - c$** | *(Unused)* | $[-200, -100, 0]$ |
| **Exponentials $e^{\mathbf{z}}$** | $[e^{100}, e^{200}, e^{300}] \to [\infty, \infty, \infty]$ | $[e^{-200}, e^{-100}, e^0] \to [0, 0, 1.0]$ |
| **Sum of Exponentials** | $\sum = \infty$ | $\sum \approx 1.0$ |
| **Normalization / Log** | $\infty / \infty \to \mathbf{\text{NaN}}$ ❌ | $0 - \ln(1.0) = 0$ ✅ |

<div align="center">
  <img src="numerical_stability_flow.svg" width="680" alt="Numerical Stability Flow: Exponentiation vs Max Shift">
</div>

Both yield mathematically identical results in exact arithmetic, but the stabilized version never encounters exponential overflow!
"""

# %% nbgrader={"grade": false, "grade_id": "log-softmax", "solution": true}
#| export
class LogSoftmax(Function):
    """
    The log-softmax operation along an axis. forward() works on NumPy arrays; Module 06 adds backward().
    """
    dim = -1

    def forward(self, x):
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

        HINT: Use np.max(x, axis=dim, keepdims=True) to preserve dimensions
        """
        ### BEGIN SOLUTION role="scaffold"
        # Step 1: Find max along dimension for numerical stability
        max_vals = np.max(x, axis=self.dim, keepdims=True)

        # Step 2: Subtract max to prevent overflow
        shifted = x - max_vals

        # Step 3: Compute log(sum(exp(shifted)))
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=self.dim, keepdims=True))

        # Step 4: Return log_softmax = input - max - log_sum_exp
        result = x - max_vals - log_sum_exp

        return result
        ### END SOLUTION


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Compute log-softmax of a Tensor along dim (see LogSoftmax)."""
    return LogSoftmax.apply(x, dim=dim)

# %% [markdown]
"""
### 🧪 Unit Test: Log-Softmax

This test validates our log_softmax function works correctly with numerical stability.

**What we're testing**: Numerical stability and correctness of log-softmax computation
**Why it matters**: Foundation for cross-entropy loss - must handle large values without overflow
**Expected**: Stable results even with extreme inputs, softmax sums to 1
"""

# %% nbgrader={"grade": true, "grade_id": "test-log-softmax", "locked": true, "points": 10}
def test_unit_log_softmax():
    """🧪 Test log_softmax numerical stability and correctness."""
    print("🧪 Unit Test: Log-Softmax...")

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
r"""
### MSELoss: Measuring Continuous Prediction Quality

Mean Squared Error is the workhorse of regression problems. It measures how far your continuous predictions are from the true values.

### When to Use MSE

**Perfect for:**
- House price prediction (200k vs 195k USD)
- Temperature forecasting (25°C vs 23°C)
- Stock price prediction (150 vs 148 USD)
- Any continuous value where "distance" matters

### How MSE Shapes Learning

$$\mathcal{L}_{\text{MSE}}(\hat{y}, y^*) = (\hat{y} - y^*)^2 \quad \text{with target } y^* = 100$$

| Candidate Prediction $\hat{y}$ | $80$ | $90$ | $95$ | $100$ | $105$ | $110$ | $120$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Residual Error ($e = \hat{y} - y^*$)** | $-20$ | $-10$ | $-5$ | $0$ | $+5$ | $+10$ | $+20$ |
| **Squared Loss ($e^2$)** | $400$ | $100$ | $25$ | $0$ | $25$ | $100$ | $400$ |
| **Gradient Magnitude ($\lvert 2e \rvert$)** | $40$ | $20$ | $10$ | $0$ | $10$ | $20$ | $40$ |

*Quadratic penalty: Large errors are punished quadratically harder than small errors, providing strong restoring forces during gradient descent.*

### Why Square the Errors?

1. **Positive penalties**: $(-10)^2 = 100$, strictly identical to $(+10)^2 = 100$
2. **Heavy punishment for large errors**: Error of $20 \to$ penalty of $400$
3. **Smooth gradients**: Quadratic function $\nabla_e (e^2) = 2e$ is everywhere differentiable and continuous
4. **Statistical foundation**: Corresponds to maximum likelihood estimation under Gaussian noise $\mathcal{N}(0, \sigma^2)$

### MSE vs Other Regression Losses

| Loss Function | Mathematical Formulation | $e = \pm 1$ | $e = \pm 5$ | $e = \pm 10$ | Outlier Robustness |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Mean Squared Error (MSE)** | $\mathcal{L}(e) = e^2$ | $1.0$ | $25.0$ | $100.0$ | Low (strongly influenced by outliers) |
| **Mean Absolute Error (MAE)** | $\mathcal{L}(e) = \lvert e \rvert$ | $1.0$ | $5.0$ | $10.0$ | High (constant gradient magnitude $\pm 1$) |
| **Huber Loss ($\delta = 1.0$)** | $\frac{1}{2}e^2 \text{ if } \lvert e \rvert \le 1 \text{ else } \lvert e \rvert - \frac{1}{2}$ | $0.5$ | $4.5$ | $9.5$ | Balanced (quadratic near 0, linear tails) |
"""

# %% nbgrader={"grade": false, "grade_id": "mse-loss", "solution": true}
#| export
class MSEFunction(Function):
    """
    The MSELoss operation. forward() works on NumPy arrays; Module 06 adds backward().
    """

    def forward(self, predictions, targets):
        """
        Compute mean squared error between predictions and targets.

        TODO: Implement MSE loss calculation

        APPROACH:
        1. Require matching, nonempty shapes: each prediction has one target
        2. Compute difference: predictions - targets
        3. Square the differences: diff²
        4. Take mean across all elements

        EXAMPLE:
        >>> loss_fn = MSELoss()
        >>> predictions = Tensor([1.0, 2.0, 3.0])
        >>> targets = Tensor([1.5, 2.5, 2.8])
        >>> loss = loss_fn(predictions, targets)
        >>> print(f"MSE Loss: {loss.data:.4f}")
        MSE Loss: 0.1800

        HINTS:
        - Use (predictions - targets) for element-wise difference
        - Square with **2 or np.power(diff, 2)
        - Use np.mean() to average over all elements
        """
        ### BEGIN SOLUTION
        if predictions.shape != targets.shape or predictions.size == 0:
            raise ValueError("MSELoss requires matching, nonempty prediction and target shapes")

        # Step 1: Compute element-wise difference
        diff = predictions - targets

        # Step 2: Square the differences
        squared_diff = diff ** 2

        # Step 3: Take mean across all elements
        mse = np.mean(squared_diff)

        return mse
        ### END SOLUTION


class MSELoss:
    """Mean Squared Error loss for regression tasks."""

    def __init__(self):
        """Initialize the loss function."""
        pass

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute the loss through its operation, so Module 06 will be able to record it for gradients."""
        return MSEFunction.apply(predictions, targets)

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Allows the loss function to be called like a function."""
        return self.forward(predictions, targets)


# %% [markdown]
"""
### 🧪 Unit Test: MSE Loss

This test validates our MSELoss implementation with various prediction scenarios.

**What we're testing**: Mean squared error calculation with perfect and imperfect predictions
**Why it matters**: MSE is the foundation for regression - must be mathematically correct
**Expected**: Zero loss for perfect predictions, positive loss for errors, non-negative always
"""

# %% nbgrader={"grade": true, "grade_id": "test-mse-loss", "locked": true, "points": 10}
def test_unit_mse_loss():
    """🧪 Test MSELoss implementation and properties."""
    print("🧪 Unit Test: MSE Loss...")

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
    random_pred = Tensor(rng.standard_normal(10))
    random_target = Tensor(rng.standard_normal(10))
    random_loss = loss_fn.forward(random_pred, random_target)
    assert random_loss.data >= 0, f"MSE loss should be non-negative, got {random_loss.data}"

    # Broadcasting would compare each prediction with every target by mistake.
    with np.testing.assert_raises(ValueError):
        loss_fn(Tensor([[1.0], [2.0]]), Tensor([1.0, 2.0]))
    with np.testing.assert_raises(ValueError):
        loss_fn(Tensor([]), Tensor([]))

    print("✅ MSELoss works correctly!")

if __name__ == "__main__":
    test_unit_mse_loss()

# %% [markdown]
r"""
### CrossEntropyLoss: Measuring Classification Confidence

Cross-entropy loss is the gold standard for multi-class classification. It measures how wrong your probability predictions are and heavily penalizes confident mistakes.

### When to Use Cross-Entropy

**Perfect for:**
- Image classification (cat, dog, bird)
- Text classification (spam, ham, promotion)
- Language modeling (next word prediction)
- Any problem with mutually exclusive classes

### Understanding Cross-Entropy Through Examples

Consider a 3-class vision task across classes: $\text{Class } 0 \to \text{Cat}$, $\text{Class } 1 \to \text{Dog}$, $\text{Class } 2 \to \text{Bird}$, where the true target is $y^* = 0$ (Cat):

| Scenario Case | Logits $\mathbf{z}$ | Softmax $\mathbf{p} = \sigma(\mathbf{z})$ | True Label $y^*$ | Cross-Entropy Loss $-\ln(p_0)$ | Learning Consequence |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Case 1: Correct & Confident** | $[5.0, 1.0, 0.1]$ | $[0.975, 0.018, 0.007]$ | Cat (0) | $\mathbf{0.025}$ ✅ | Negligible error signal; weights remain intact |
| **Case 2: Correct but Uncertain** | $[1.1, 1.0, 0.9]$ | $[0.367, 0.332, 0.301]$ | Cat (0) | $\mathbf{1.002}$ ⚠️ | Significant push ($p_0 - 1 \approx -0.633$) to increase confidence |
| **Case 3: Wrong & Confident** | $[0.1, 5.0, 1.0]$ | $[0.007, 0.975, 0.018]$ | Cat (0) | $\mathbf{4.925}$ ❌ | Massive error signal ($p_0 - 1 \approx -0.993$) driving rapid correction |

*Every loss above is computed from the exact softmax, not from the rounded probability column. Case 3's $p_0$ is $0.0072596$, so $-\ln p_0 = 4.925$; rounding to $0.007$ first would report $4.962$. Round for display, never before the arithmetic.*

### Cross-Entropy's Learning Signal

$$\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z_i} = p_i - y_i \quad \text{where } y_i = \mathbf{1}[i = y^*]$$

| Prediction State | True Class Alignment | Gradient Push $(p_i - y_i)$ | Learning Signal Directive |
| :--- | :--- | :---: | :--- |
| **Confident High Probability** | Correct ($y_i = 1, p_i \approx 1$) | $\approx 0$ | "Converged, preserve current weights" |
| **Uncertain Probability** | Correct ($y_i = 1, p_i \approx 0.33$) | $\approx -0.67$ | "Step aggressively toward higher confidence" |
| **Confident High Probability** | Wrong ($y_i = 0, p_i \approx 1$) | $\approx +1.0$ | "Maximum emergency suppression of this logit" |
| **Uncertain Probability** | Wrong ($y_i = 0, p_i \approx 0.33$) | $\approx +0.33$ | "Gradually suppress probability" |

### Why Cross-Entropy Works So Well

1. **Probabilistic interpretation**: Directly measures negative log-likelihood under multinomial distribution
2. **Linear gradient in logit space**: Softmax and log-likelihood cancel elegantly: $\nabla_{\mathbf{z}} \mathcal{L} = \mathbf{p} - \mathbf{y}$ (no vanishing gradient near saturation!)
3. **Smooth convex optimization**: Log-sum-exp is globally convex in logit space
4. **Information theory**: Minimizes the Kullback-Leibler divergence $D_{\text{KL}}(p_{\text{data}} \parallel p_{\text{model}})$

### Multi-Class vs Binary Classification

| Architectural Property | Multi-Class Classification ($C \ge 3$) | Binary Classification ($C = 2$) |
| :--- | :--- | :--- |
| **Target Representation** | Class index $y^* \in \{0, \dots, C-1\}$ (int64) | Binary indicator $y \in \{0.0, 1.0\}$ (float32) |
| **Model Output Shape** | $(B, C)$ unnormalized logits | $(B,)$ or $(B, 1)$ probability |
| **Activation Function** | $\operatorname{Softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$ | $\operatorname{Sigmoid}(z) = \frac{1}{1 + e^{-z}}$ |
| **Normalization Constraint** | $\sum_{c=1}^C p_c = 1.0$ | $p + (1 - p) = 1.0$ |
| **TinyTorch Module Class** | `CrossEntropyLoss` | `BinaryCrossEntropyLoss` |
"""

# %% nbgrader={"grade": false, "grade_id": "cross-entropy-loss", "solution": true}
#| export
class CrossEntropyFunction(Function):
    """
    The CrossEntropyLoss operation. forward() works on NumPy arrays; Module 06 adds backward().
    """

    def forward(self, logits, targets):
        """
        Compute cross-entropy loss between logits and target class indices.

        TODO: Implement cross-entropy loss with numerical stability

        APPROACH:
        1. Require nonempty logits (batch, classes) and targets (batch,)
        2. Check targets are finite integers with 0 <= t < num_classes,
           then compute log-softmax of logits (numerically stable)
        3. Select log-probabilities for correct classes
        4. Return negative mean of selected log-probabilities

        EXAMPLE:
        >>> loss_fn = CrossEntropyLoss()
        >>> logits = Tensor([[2.0, 1.0, 0.1], [0.5, 1.5, 0.8]])  # 2 samples, 3 classes
        >>> targets = Tensor([0, 1])  # First sample is class 0, second is class 1
        >>> loss = loss_fn(logits, targets)
        >>> print(f"Cross-Entropy Loss: {loss.data:.4f}")

        HINTS:
        - Use LogSoftmax().forward(logits) for numerical stability (the array-level log-softmax you wrote above)
        - Tensor stores float32; validate whole-number labels before targets.astype(int)
        - num_classes is logits.shape[-1]; validate before indexing, because
        NumPy would let a negative target silently select the wrong class
        and would raise a bare IndexError for one that is too large
        - Use np.arange(batch_size) for row indexing: log_probs[np.arange(batch_size), targets]
        - Return negative mean: -np.mean(selected_log_probs)
        """
        ### BEGIN SOLUTION
        if logits.ndim != 2 or logits.size == 0 or targets.shape != (logits.shape[0],):
            raise ValueError("CrossEntropyLoss requires nonempty logits (batch, classes) and targets (batch,)")
        if not np.all(np.isfinite(targets)) or np.any(targets != np.floor(targets)):
            raise ValueError("CrossEntropyLoss targets must be finite integer class indices")

        batch_size, num_classes = logits.shape
        out_of_range = (targets < 0) | (targets >= num_classes)
        if np.any(out_of_range):
            bad_values = np.unique(targets[out_of_range])
            raise ValueError(
                f"CrossEntropyLoss target index out of range: {bad_values.tolist()}\n"
                f"  Valid range for {num_classes} classes is [0, {num_classes - 1}]"
            )

        # Validate before casting: conversion would silently truncate fractional labels.
        target_indices = targets.astype(int)
        log_probs = LogSoftmax().forward(logits)

        # Select correct class log-probabilities using advanced indexing
        selected_log_probs = log_probs[np.arange(batch_size), target_indices]

        # Step 3: Return negative mean (cross-entropy is negative log-likelihood)
        cross_entropy = -np.mean(selected_log_probs)

        return cross_entropy
        ### END SOLUTION


class CrossEntropyLoss:
    """Cross-entropy loss for multi-class classification."""

    def __init__(self):
        """Initialize the loss function."""
        pass

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute the loss through its operation, so Module 06 will be able to record it for gradients."""
        return CrossEntropyFunction.apply(logits, targets)

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Allows the loss function to be called like a function."""
        return self.forward(logits, targets)


# %% [markdown]
"""
### 🧪 Unit Test: Cross-Entropy Loss

This test validates our CrossEntropyLoss implementation with various confidence levels.

**What we're testing**: Cross-entropy loss with confident correct, uncertain, and confident wrong predictions
**Why it matters**: CrossEntropy is the gold standard for classification - must handle all confidence levels
**Expected**: Low loss for confident correct, high loss for confident wrong, numerical stability
"""

# %% nbgrader={"grade": true, "grade_id": "test-cross-entropy-loss", "locked": true, "points": 10}
def test_unit_cross_entropy_loss():
    """🧪 Test CrossEntropyLoss implementation and properties."""
    print("🧪 Unit Test: Cross-Entropy Loss...")

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

    for invalid_targets in ([0.5, 1], [-1, 1], [[0], [1]]):
        with np.testing.assert_raises(ValueError):
            loss_fn(uniform_logits, Tensor(invalid_targets))

    print("✅ CrossEntropyLoss works correctly!")

if __name__ == "__main__":
    test_unit_cross_entropy_loss()

# %% [markdown]
r"""
### BinaryCrossEntropyLoss: Measuring Yes/No Decision Quality

Binary Cross-Entropy is specialized for yes/no decisions. It's like regular cross-entropy but optimized for the special case of exactly two classes.

### When to Use Binary Cross-Entropy

**Perfect for:**
- Spam detection (spam vs not spam)
- Medical diagnosis (disease vs healthy)
- Fraud detection (fraud vs legitimate)
- Content moderation (toxic vs safe)
- Any two-class decision problem

### Understanding Binary Cross-Entropy

$$\mathcal{L}_{\text{BCE}}(\hat{y}, y) = - \big[ y \ln(\hat{y}) + (1 - y) \ln(1 - \hat{y}) \big]$$

| Prediction ($\hat{y}$) | True Target $y = 1$ (Positive Class) | True Target $y = 0$ (Negative Class) |
| :--- | :--- | :--- |
| **Model predicts $\hat{y} \to 1.0$ ("Yes")** | **True Positive (TP)**: $\mathcal{L} = -\ln(1.0) = 0$ (Reward) | **False Positive (FP)**: $\mathcal{L} = -\ln(0) \to +\infty$ (Severe Penalty) |
| **Model predicts $\hat{y} \to 0.0$ ("No")** | **False Negative (FN)**: $\mathcal{L} = -\ln(0) \to +\infty$ (Severe Penalty) | **True Negative (TN)**: $\mathcal{L} = -\ln(1.0) = 0$ (Reward) |

### Binary Cross-Entropy Behavior Examples

Consider email spam detection where label $y = 1$ denotes spam and $y = 0$ denotes legitimate ham:

| Email Prediction Scenario | Predicted $\hat{y}$ | True Label $y$ | Evaluated BCE Loss | Optimization Feedback |
| :--- | :---: | :---: | :---: | :--- |
| **Case 1: Confident Spam Detection** | $0.99$ | $1$ (Spam) | $-\ln(0.99) \approx \mathbf{0.010}$ ✅ | Negligible error signal; spam filter verified |
| **Case 2: Uncertain Classification** | $0.51$ | $0$ (Ham) | $-\ln(1 - 0.51) \approx \mathbf{0.713}$ ⚠️ | Substantial gradient push to classify email as ham |
| **Case 3: Catastrophic False Alarm** | $0.95$ | $0$ (Ham) | $-\ln(1 - 0.95) \approx \mathbf{2.996}$ ❌ | Massive error signal; urgent suppression of spam logit |

### Binary vs Multi-Class Cross-Entropy

| Property | Binary Cross-Entropy (`BCE`) | Multi-Class Cross-Entropy (`CE`) |
| :--- | :--- | :--- |
| **Output Dimension** | Single probability scalar $\hat{y} \in [0, 1]$ | Vector of probabilities $\mathbf{p} \in [0, 1]^C$ |
| **Mathematical Formula** | $-\big[y \ln(\hat{y}) + (1-y)\ln(1-\hat{y})\big]$ | $-\sum_{c=1}^C y_c \ln(p_c) = -\ln(p_{y^*})$ |
| **Class Topology** | Independent binary Bernoulli trials | Mutually exclusive Multinomial distribution |
| **Activation Pairing** | Logistic Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$ | Softmax $\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$ |

### Loss Landscape and Epsilon Boundary Clamping

To prevent $\ln(0) \to -\infty$ and floating-point `NaN` values, our implementation clamps predicted probabilities to the safe range $[\varepsilon, 1 - \varepsilon]$ where $\varepsilon = 10^{-7}$:

$$\hat{y}_{\text{clamped}} = \operatorname{clip}(\hat{y}, \, \varepsilon, \, 1 - \varepsilon)$$

| Unclamped Prediction $\hat{y}$ | Target $y = 1$ Clamped Loss | Target $y = 0$ Clamped Loss | Float Status |
| :---: | :---: | :---: | :---: |
| $1.0000$ | $-\ln(1 - 10^{-7}) \approx 10^{-7}$ | $-\ln(1.192 \times 10^{-7}) \approx 15.942$ | Finite IEEE 754 float32 ✅ |
| $0.9000$ | $-\ln(0.9) \approx 0.105$ | $-\ln(0.1) \approx 2.303$ | Finite IEEE 754 float32 ✅ |
| $0.5000$ | $-\ln(0.5) \approx 0.693$ | $-\ln(0.5) \approx 0.693$ | Finite IEEE 754 float32 ✅ |
| $0.1000$ | $-\ln(0.1) \approx 2.303$ | $-\ln(0.9) \approx 0.105$ | Finite IEEE 754 float32 ✅ |
| $0.0000$ | $-\ln(10^{-7}) \approx 16.118$ | $-\ln(1 - 10^{-7}) \approx 10^{-7}$ | Finite IEEE 754 float32 ✅ |

The first and last rows look like mirror images but are not. Tensor stores float32, whose
spacing just below $1.0$ is about $6 \times 10^{-8}$, so the upper clamp $1 - \varepsilon$
rounds to $0.9999999$ and its complement comes back as $1.192 \times 10^{-7}$ rather than
$10^{-7}$. The maximum loss is therefore $15.942$ against $y = 0$ and $16.118$ against
$y = 1$. The clamp bounds the loss in both directions, which is the point, but the bound
you get is set by the format, not by the $\varepsilon$ you wrote down.
"""

# %% nbgrader={"grade": false, "grade_id": "binary-cross-entropy-loss", "solution": true}
#| export
class BinaryCrossEntropyFunction(Function):
    """
    The BinaryCrossEntropyLoss operation. forward() works on NumPy arrays; Module 06 adds backward().
    """

    def forward(self, predictions, targets):
        """
        Compute binary cross-entropy loss.

        TODO: Implement binary cross-entropy with numerical stability

        APPROACH:
        1. Require matching, nonempty shapes and probabilities/targets in [0, 1]
        2. Clamp predictions to keep both logarithms finite
        3. Compute: -(targets * log(predictions) + (1-targets) * log(1-predictions))
        4. Return mean across all elements

        EXAMPLE:
        >>> loss_fn = BinaryCrossEntropyLoss()
        >>> predictions = Tensor([0.9, 0.1, 0.7, 0.3])  # Probabilities between 0 and 1
        >>> targets = Tensor([1.0, 0.0, 1.0, 0.0])      # Binary labels
        >>> loss = loss_fn(predictions, targets)
        >>> print(f"Binary Cross-Entropy Loss: {loss.data:.4f}")

        HINTS:
        - Use np.clip(predictions, 1e-7, 1-1e-7) to prevent log(0)
        - Clipping makes the loss constant outside that interval; Module 06 will give
          those regions zero prediction gradient; at each clipping
          boundary it will choose the derivative from inside the interval.
        - Binary cross-entropy: -(targets * log(preds) + (1-targets) * log(1-preds))
        - Use np.mean() to average over all samples
        """
        ### BEGIN SOLUTION role="scaffold"
        if predictions.shape != targets.shape or predictions.size == 0:
            raise ValueError("BinaryCrossEntropyLoss requires matching, nonempty prediction and target shapes")
        for values in (predictions, targets):
            if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
                raise ValueError("BinaryCrossEntropyLoss predictions and targets must be finite values in [0, 1]")

        # Step 1: Clamp predictions to avoid numerical issues with log(0) and log(1)
        eps = EPSILON
        clamped_preds = np.clip(predictions, eps, 1 - eps)

        # Step 2: Compute binary cross-entropy
        # BCE = -(targets * log(preds) + (1-targets) * log(1-preds))
        log_preds = np.log(clamped_preds)
        log_one_minus_preds = np.log(1 - clamped_preds)

        bce_per_sample = -(targets * log_preds + (1 - targets) * log_one_minus_preds)

        # Step 3: Return mean across all samples
        bce_loss = np.mean(bce_per_sample)

        return bce_loss
        ### END SOLUTION


class BinaryCrossEntropyLoss:
    """Binary cross-entropy loss for binary classification."""

    def __init__(self):
        """Initialize the loss function."""
        pass

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute the loss through its operation, so Module 06 will be able to record it for gradients."""
        return BinaryCrossEntropyFunction.apply(predictions, targets)

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Allows the loss function to be called like a function."""
        return self.forward(predictions, targets)


# %% [markdown]
"""
### 🧪 Unit Test: Binary Cross-Entropy Loss

This test validates our BinaryCrossEntropyLoss implementation with binary classification scenarios.

**What we're testing**: Binary cross-entropy with perfect, worst, and boundary predictions
**Why it matters**: BCE is essential for binary classification - must handle edge cases
**Expected**: Low loss for correct predictions, high loss for wrong predictions, numerical stability at boundaries
"""

# %% nbgrader={"grade": true, "grade_id": "test-binary-cross-entropy-loss", "locked": true, "points": 10}
def test_unit_binary_cross_entropy_loss():
    """🧪 Test BinaryCrossEntropyLoss implementation and properties."""
    print("🧪 Unit Test: Binary Cross-Entropy Loss...")

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

    with np.testing.assert_raises(ValueError):
        loss_fn(Tensor([[0.2], [0.8]]), Tensor([0.0, 1.0]))
    with np.testing.assert_raises(ValueError):
        loss_fn(Tensor([1.2]), Tensor([1.0]))

    print("✅ BinaryCrossEntropyLoss works correctly!")

if __name__ == "__main__":
    test_unit_binary_cross_entropy_loss()

# %% [markdown]
r"""
## 🔧 Integration: Bringing It Together

Now let's test how our loss functions work together with real data scenarios and explore their behavior with different types of predictions.

### Real-World Loss Function Usage Patterns

Understanding when and why to use each loss function is crucial for ML engineering success:

$$\mathbf{\text{Supervised Learning Problem Formulation}}$$
$$\begin{array}{ccc}
\swarrow & & \searrow \\
\mathbf{\text{Continuous Regression}} & & \mathbf{\text{Categorical Classification}} \\
\downarrow & & \swarrow \qquad\qquad\qquad\qquad \searrow \\
\mathbf{\text{MSELoss}} & \mathbf{\text{Binary (2 Mutually Exclusive Classes)}} & \mathbf{\text{Multi-Class (} C \ge 3 \text{ Classes)}} \\
(\text{Linear Output } \hat{y} \in \mathbb{R}) & \downarrow & \downarrow \\
& \mathbf{\text{BinaryCrossEntropyLoss}} & \mathbf{\text{CrossEntropyLoss}} \\
& (\text{Sigmoid Probability } \hat{y} \in [0, 1]) & (\text{Log-Softmax Logits } \mathbf{z} \in \mathbb{R}^C)
\end{array}$$

| Problem Domain | Output Modality | Recommended TinyTorch Loss | Example Task |
| :--- | :--- | :--- | :--- |
| **Continuous Regression** | Unconstrained scalar $\hat{y} \in \mathbb{R}$ | `MSELoss` | House pricing, temperature forecasting, trajectory regression |
| **Binary Classification** | Single probability $\hat{y} \in [0, 1]$ | `BinaryCrossEntropyLoss` | Spam filtering, fraud detection, medical anomaly detection |
| **Multi-Class Classification** | $C$-class logits $\mathbf{z} \in \mathbb{R}^C$ | `CrossEntropyLoss` | Image classification, next-token language modeling, audio phonemes |

### Loss Function Behavior Comparison

Each loss function creates different learning pressures on your model:

| Error Distance $\lvert e \rvert$ | MSE Loss ($\lvert e \rvert^2$) | BCE / CE Loss ($-\ln(1 - \lvert e \rvert)$) | Relative Learning Pressure |
| :---: | :---: | :---: | :--- |
| **Small Error ($0.1$)** | $0.010$ | $0.105$ | Moderate gradient push across both paradigms |
| **Medium Error ($0.5$)** | $0.250$ | $0.693$ | BCE/CE applies nearly $3\times$ higher penalty than MSE |
| **Large Error ($0.9$)** | $0.810$ | $2.303$ | BCE/CE penalty accelerates logarithmically toward asymptote |
| **Extreme Outlier ($1.0^-$)** | $1.000$ | $15.942$ (clamped, float32) | Cross-entropy generates an emergency restorative gradient |

*MSE scales quadratically (gentle on confident wrong guesses, sensitive to outliers). BCE/CE scales logarithmically, generating explosive gradient updates when the model is confidently wrong.*

### Pitfall: CrossEntropyLoss Takes Raw Logits

`CrossEntropyLoss` applies log-softmax itself, so it must be handed **raw logits**. Passing
it probabilities (the output of a softmax you already applied) is the most common mistake in
this API, and it inherits from PyTorch, where `torch.nn.CrossEntropyLoss` has the same
contract. There is no shape error and no exception. The loss is simply wrong.

Take Case 1 from the table above, a model that is correct and confident. Its logits
$[5.0, 1.0, 0.1]$ give a loss of $0.025$. Hand the same model's probabilities
$[0.975, 0.018, 0.007]$ to the loss instead and it returns $0.568$, more than twenty times
larger, because the loss softmaxes them a second time into $[0.567, 0.218, 0.215]$:

```python
ce = CrossEntropyLoss()
ce(Tensor([[5.0, 1.0, 0.1]]),    Tensor([0]))  # 0.0254  correct: raw logits
ce(Tensor([[0.975, 0.018, 0.007]]), Tensor([0]))  # 0.5675  wrong: already softmaxed
```

Probabilities sum to $1$, so their spread can never exceed $1$ and no logit gap can exceed
$e \approx 2.72$ after the second exponentiation. The model's confidence is flattened away.
Training still runs, the loss still falls, and it plateaus early with no obvious cause. Two
checks catch it. Feed the loss a deliberately confident correct example and confirm the loss
is near zero rather than near $\ln C$, and grep your forward pass for a `softmax` before the
loss. `BinaryCrossEntropyLoss` is the opposite contract, taking probabilities in $[0, 1]$,
which is why it validates its input range and `CrossEntropyLoss` does not.
"""

# %% nbgrader={"grade": false, "grade_id": "loss-comparison", "solution": false}
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


# %% nbgrader={"grade": false, "grade_id": "loss-sensitivity", "solution": false}
def analyze_loss_sensitivity():
    """
    📊 Analyze how sensitive each loss function is to prediction errors.

    This demonstrates the different error landscapes created by each loss.
    """
    print("\n📊 Analysis: Loss Function Sensitivity to Errors...")

    # Create a range of prediction errors for analysis
    true_value = 1.0
    predictions = np.linspace(0.1, 1.9, 181)  # From 0.1 to 1.9

    # Initialize loss functions
    mse_loss = MSELoss()
    bce_loss = BinaryCrossEntropyLoss()

    mse_losses = []
    bce_losses = []
    probabilities = np.linspace(0.01, 1.0, 100)

    for pred in predictions:
        # MSE analysis
        pred_tensor = Tensor([pred])
        target_tensor = Tensor([true_value])
        mse = mse_loss.forward(pred_tensor, target_tensor)
        mse_losses.append(mse.data)

    # BCE accepts probabilities, so analyze its own valid input range.
    for probability in probabilities:
        bce_pred_tensor = Tensor([probability])
        bce_target_tensor = Tensor([1.0])  # Target is "positive class"
        bce = bce_loss.forward(bce_pred_tensor, bce_target_tensor)
        bce_losses.append(bce.data)

    # Find minimum losses
    min_mse_idx = np.argmin(mse_losses)
    min_bce_idx = np.argmin(bce_losses)

    idx_05 = np.argmin(np.abs(predictions - 0.5))
    bce_idx_05 = np.argmin(np.abs(probabilities - 0.5))
    bce_idx_01 = np.argmin(np.abs(probabilities - 0.1))

    print(f"MSE Loss:")
    print(f"  Minimum at prediction = {predictions[min_mse_idx]:.2f}, loss = {mse_losses[min_mse_idx]:.4f}")
    print(f"  At prediction = 0.5: loss = {mse_losses[idx_05]:.4f}")
    print(f"  At prediction = 0.1: loss = {mse_losses[0]:.4f}")

    print(f"\nBinary Cross-Entropy Loss:")
    print(f"  Minimum at prediction = {probabilities[min_bce_idx]:.2f}, loss = {bce_losses[min_bce_idx]:.4f}")
    print(f"  At prediction = 0.5: loss = {bce_losses[bce_idx_05]:.4f}")
    print(f"  At prediction = 0.1: loss = {bce_losses[bce_idx_01]:.4f}")

    print(f"\n💡 Sensitivity Insights:")
    print("   - MSE grows quadratically with error distance")
    print("   - BCE grows logarithmically, heavily penalizing wrong confident predictions")
    print("   - Both encourage correct predictions but with different curvatures")

if __name__ == "__main__":
    analyze_loss_behaviors()
    analyze_loss_sensitivity()

# %% [markdown]
r"""
## 📊 Systems Analysis: Understanding Loss Function Performance

Loss functions seem simple, but they have critical computational and memory bandwidth implications that govern large-scale distributed training throughput.

### Computational Complexity Analysis

| Loss Function | Mathematical Core | Algorithmic Operations | Time Complexity | Memory Complexity |
| :--- | :--- | :--- | :---: | :---: |
| **`MSELoss`** | $\frac{1}{B} \sum_{b=1}^B (\hat{y}_b - y_b)^2$ | Subtraction $\to$ Square $\to$ Mean | $\mathcal{O}(B)$ | $\mathcal{O}(B)$ |
| **`CrossEntropyLoss`** | $-\frac{1}{B} \sum_{b=1}^B \log\left(\frac{e^{z_{b, y_b^*}}}{\sum_c e^{z_{b, c}}}\right)$ | Max-Reduction $\to$ Subtraction $\to \operatorname{Exp} \to \text{Sum} \to \operatorname{Log} \to \text{Gather}$ | $\mathcal{O}(B \cdot C)$ | $\mathcal{O}(B \cdot C)$ |
| **`BinaryCrossEntropy`** | $-\frac{1}{B} \sum_{b=1}^B \big[y_b \ln(\hat{y}_b) + (1-y_b)\ln(1-\hat{y}_b)\big]$ | Clip $\to \operatorname{Log} \to$ Linear Comb $\to$ Mean | $\mathcal{O}(B)$ | $\mathcal{O}(B)$ |

*For single-output regression, MSE processes $B$ scalars. In language models with vocabulary $C = 32{,}000$, CrossEntropy processes $B \cdot C$ floats, over $30{,}000\times$ more data elements per batch.*

### Memory Layout and Production Memory Footprint

A common systems pitfall is treating multi-class cross-entropy as matrix multiplication against one-hot targets. In reality, modern ML systems perform direct index gathering:

<div align="center">
  <img src="index_gather_memory.svg" width="680" alt="Direct Index Gather vs One-Hot Memory">
</div>

### Forward Pass Buffer Lifecycle

| Pipeline Step | `MSELoss` Buffer Allocated | `CrossEntropyLoss` Buffer Allocated |
| :--- | :--- | :--- |
| **Input Buffer** | $\mathbf{\hat{y}} \in \mathbb{R}^B$ ($4B$ bytes) | $\mathbf{z} \in \mathbb{R}^{B \times C}$ ($4BC$ bytes) |
| **Intermediate 1** | Residual $(\mathbf{\hat{y}} - \mathbf{y}) \in \mathbb{R}^B$ ($4B$ bytes) | Row Maximums $\mathbf{m} \in \mathbb{R}^{B \times 1}$ ($4B$ bytes) |
| **Intermediate 2** | Squared Residuals $(\mathbf{\hat{y}} - \mathbf{y})^2 \in \mathbb{R}^B$ ($4B$ bytes) | Exponentials $e^{\mathbf{z} - \mathbf{m}} \in \mathbb{R}^{B \times C}$ ($4BC$ bytes) |
| **Intermediate 3** | *(None)* | Normalizer $\sum_c e^{z_{b,c}-m_b} \in \mathbb{R}^{B \times 1}$ ($4B$ bytes) |
| **Intermediate 4** | *(None)* | Gathered Log-Probs $\mathbf{z}_{y^*} \in \mathbb{R}^B$ ($4B$ bytes) |
| **Total Transient Footprint** | $\approx 3B \times 4 \text{ bytes} = 12B \text{ bytes}$ | $\approx (2BC + 3B) \times 4 \text{ bytes}$ |
"""

# %% nbgrader={"grade": false, "grade_id": "analyze-numerical-stability", "solution": false}
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
    print("   With it: These finite float32 logits avoid exponential overflow")


# %% nbgrader={"grade": false, "grade_id": "analyze-loss-memory", "solution": false}
def analyze_loss_memory():
    """
    📊 Analyze memory usage patterns of different loss functions.

    Understanding memory helps with batch size decisions.
    """
    print("\n📊 Analysis: Selected Loss Buffers (Estimate)...")

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
    print("   - The estimate includes one CE intermediate buffer")
    print("   - Temporary arrays and allocator overhead are omitted; this is not measured peak memory")
    print(f"   - For batch=1024, the selected CE buffers total {ce_memory:.1f}MB")

if __name__ == "__main__":
    analyze_numerical_stability()
    analyze_loss_memory()

# %% [markdown]
r"""
### Production Context: How Loss Functions Scale

Understanding how loss functions behave in production helps make informed engineering decisions about model architecture and training strategies.

**Loss Function Scaling Challenges**

As models grow larger, loss function memory and compute bottlenecks become critical:

| Scaling Dimension | Small Model (MNIST) | Large Model (ImageNet) | Production Scale (LLM / GPT-4) |
| :--- | :---: | :---: | :---: |
| **Output Classes ($C$)** | $10$ | $1{,}000$ | $32{,}000$ to $128{,}000$ |
| **Batch Size ($B$)** | $64$ | $256$ | $2{,}048$ to $8{,}192$ |
| **Logit Memory ($B \cdot C \cdot 4\text{B}$)** | $2.5\text{ KB}$ | $1.0\text{ MB}$ | $262\text{ MB}$ to $4.19\text{ GB}$ |
| **Log-Softmax Temp Buffers** | $\approx 7.5\text{ KB}$ | $\approx 3.0\text{ MB}$ | $\approx 786\text{ MB}$ to $12.6\text{ GB}$ |
| **Primary System Bottleneck** | Compute bound (negligible) | Softmax GPU core reduction | HBM Memory Capacity & Bandwidth |

*Memory scales as $\mathcal{O}(B \cdot C)$ for standard cross-entropy. In modern LLMs, vocabulary size $C$ dominates loss computation.*

The production column reports the two corners of the ranges above it, not two independent
scenarios. The low end is the small batch against the small vocabulary,
$2{,}048 \times 32{,}000 \times 4\text{ B} = 262\text{ MB}$; the high end is the large batch
against the large vocabulary, $8{,}192 \times 128{,}000 \times 4\text{ B} = 4.19\text{ GB}$.
Both middle combinations land at $1.05\text{ GB}$, which is why the corners matter: the
logits alone move by $16\times$ across a range that each dimension only widens $4\times$.

### Engineering Optimizations in Production

| Optimization Technique | Mathematical Mechanism | Production Impact | Typical Application |
| :--- | :--- | :--- | :--- |
| **Hierarchical Softmax** | Decompose $C$ into balanced binary tree: $\mathcal{O}(\log_2 C)$ decisions | Reduces compute from $50{,}000$ ops to $\approx 16$ ops per sample | Word2Vec, extreme multi-label text |
| **Sampled Softmax** | Compute normalization over positive target + $K \ll C$ negative samples | $50\times$ speedup during pre-training | Recommendation candidate retrieval |
| **Label Smoothing** | Target $\mathbf{y}_{\text{smooth}} = (1 - \alpha)\mathbf{y} + \frac{\alpha}{C}$ | Prevents overconfident logit explosion ($z \to \infty$) | ImageNet training, Transformer translation |
| **Kernel Fusion / Triton** | Fuse logit write + softmax + cross-entropy into a single streaming GPU kernel | Eliminates $3\times B \cdot C$ intermediate DRAM roundtrips | FlashCrossEntropy, PyTorch Inductor |
"""

# %% nbgrader={"grade": false, "grade_id": "analyze-production-patterns", "solution": false}
def analyze_production_patterns():
    """
    📊 Analyze loss function patterns in production ML systems.

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

if __name__ == "__main__":
    analyze_production_patterns()

# %% [markdown]
"""
## 🧪 Module Integration Test

Final validation that everything works together correctly.
"""


# %% nbgrader={"grade": true, "grade_id": "module-integration", "locked": true, "points": 20}
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
    print("🧪 Integration Test: Realistic training scenario...")

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


# %% [markdown]
r"""
## 🤔 ML Systems Reflection Questions

Answer these to deepen your understanding of loss functions and their systems implications:

### Question 1: Memory and Performance

**Question**: Loss Function Selection for Large Vocabulary

You're building a language model with a 50,000 word vocabulary. Your GPU has 16GB of memory, and you want to use batch size 128.

Calculate:
- How much memory does CrossEntropyLoss need for one forward pass? (Hint: B=128, C=50,000, float32)
- What fraction of the 16GB budget is that, and is the loss the thing to optimize here?
- Now hold the 16GB budget and rerun the same arithmetic at production scale (C=128,000, B=8,192). At what point do the loss buffers alone crowd out the model, and what three strategies buy the space back?

<details>
<summary>💡 Hint</summary>

Memory for logits = Batch_Size × Num_Classes × 4 bytes (float32) = 128 × 50,000 × 4 = 25.6 MB

For full forward pass with intermediate tensors (softmax, log_softmax), multiply by ~3 = 76.8 MB

That is 0.5% of 16GB, so at this scale the loss is not the constraint. The weights, their optimizer state, and the activations of every earlier layer are. Optimizing the loss here would win back nothing measurable, and recognizing that is the point of the first calculation.

At production scale the same arithmetic reads differently: 8,192 × 128,000 × 4 = 4.19 GB of logits, and ~3× that for the forward pass is 12.6 GB, for the loss layer alone. The same 16GB budget still has to hold the weights, their optimizer state, and every activation, so the loss no longer fits alongside the model it is scoring. That is where these strategies stop being optional:
1. **Sampled softmax**: Only compute softmax over subset of vocabulary (1000 samples)
2. **Hierarchical softmax**: Use tree structure, O(log V) instead of O(V)
3. **Mixed precision**: Use FP16 for forward pass (2 bytes instead of 4)
4. **Recomputation**: Recompute intermediate results instead of storing them (trades compute for memory)
</details>

---

### Question 2: Loss Function Performance Bottleneck
**Question**: Performance Analysis

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

### Question 3: Numerical Stability
**Question**: Debugging Exploding Loss

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

### Question 4: Production Considerations
**Question**: Real-Time Inference Latency

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

**Performance gain**: 8ms saved, taking the request from 11ms to 3ms (73% faster, and back inside the 10ms budget), just by removing unnecessary loss computation.

**Key insight**: Loss functions measure "wrongness" during training. At inference, you already have the model's output - use it directly!
</details>

---

### Question 5: Class Imbalance in Medical Diagnosis
**Question**: Reweighting the BCE Formula

You're building a cancer detection system:
- 95% of samples are negative (healthy)
- 5% are positive (cancer)

Using vanilla BinaryCrossEntropyLoss, your model achieves 95% accuracy by always predicting "healthy."

Work from the formula you implemented, $\mathcal{L} = -[y \ln(\hat{y}) + (1 - y)\ln(1 - \hat{y})]$, averaged over a batch of 100 samples:

1. The always-healthy model predicts the base rate, $\hat{y} = 0.05$, for every sample. Compute its mean loss, and split that mean into the part the 95 negatives contribute and the part the 5 positives contribute.
2. Insert one coefficient $w$ in front of the $y \ln(\hat{y})$ term. Choose $w$ so the 5 positive samples carry the same total weight as the 95 negatives. What is $w$, and what quantity is it the ratio of?
3. With that $w$ in place, solve for the constant prediction that now minimizes the loss. Has the degenerate solution survived?

<details>
<summary>💡 Hint</summary>

**Part 1**: $0.95 \times (-\ln 0.95) + 0.05 \times (-\ln 0.05) = 0.0487 + 0.1498 = 0.1985$. The 5 positives already contribute 75% of the loss, so the gradient is not blind to them. The problem is that no constant prediction does better: $\hat{y} = 0.05$ is exactly the minimizer, so a model with no useful features parks there and stops.

**Part 2**: $w \times 5 = 95$, so $w = 19$, the ratio of negative count to positive count. It is class frequency inverted, nothing more.

**Part 3**: Minimize $-[19 \times 0.05 \ln p + 0.95 \ln(1 - p)]$ over the constant $p$. Setting the derivative to zero gives $0.95(1 - p) = 0.95p$, so $p = 0.5$. The degenerate solution is gone: weighting by $19$ moves the best do-nothing prediction from $0.05$ to $0.5$, where it is no longer 95% accurate and no longer looks like success. This is the whole mechanism behind PyTorch's `pos_weight` argument to `BCEWithLogitsLoss`.

**In code**:
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

**Beyond this module**: two other tools attack the same imbalance from outside the formula above, so you cannot derive them from this page, but they are worth knowing by name.

*Focal loss* multiplies each term by $(1 - p_{\text{correct}})^\gamma$, which downweights the easy majority samples automatically instead of by a fixed ratio you supply. *Resampling* changes the batch rather than the loss, by oversampling the minority class, undersampling the majority class, or synthesizing minority samples (SMOTE). You will build the batching machinery that makes resampling possible in Module 05 (DataLoader).

**Medical Reality**: Weighted loss is most common. False negatives (missed cancer) are MUCH worse than false positives (unnecessary tests).

**Critical Insight**: 95% accuracy is meaningless! Track precision, recall, F1, and AUC instead.
</details>

---

### Question 6: Batch Size and Loss Computation
**Question**: Systems Thinking

You're training on a GPU with 24GB memory. With batch size 32, memory usage is 8GB. You increase batch size to 128.

Will memory usage be 32GB (4× increase)? Why or why not?

What happens to:
- Loss computation time?
- Loss value (the actual number)?
- Quality of the error signal?

<details>
<summary>💡 Hint</summary>

**Memory Usage**: Almost. About 29 GB (1 GB fixed + 4 × 7 GB of activations) - **EXCEEDS GPU MEMORY! Training will crash.**

**Why linear scaling?**

$$\text{Memory} = \underbrace{\text{Model Params}}_{\text{Fixed } (1\text{ GB})} + \underbrace{B \times (\text{Intermediate Buffers})}_{\text{Scales linearly } (7\text{ GB} \to 28\text{ GB})}$$

**Loss computation time**: ~4× slower (linear with batch size)
- 32 samples: 0.5ms
- 128 samples: 2.0ms

**Loss value**: Averaging keeps the loss on a comparable scale as batch size
changes. Different samples can produce different means. For independent samples
from the same distribution, both batch means estimate the same expected loss;
the larger batch has lower sampling variance.
```python
losses = np.concatenate([np.zeros(32), np.ones(96)])
batch_32_loss = np.mean(losses[:32])   # 0.0
batch_128_loss = np.mean(losses[:128]) # 0.75: the added samples have higher loss
```

**Error signal quality**: **BETTER** - larger batch = more stable estimate of the true loss
- Batch 32: High variance, noisy estimate
- Batch 128: Lower variance, smoother estimate

**The Trade-off**:
- Larger batch = more accurate loss estimate but more memory
- Smaller batch = less memory but noisier estimate
- Sweet spot: Usually 64-256 depending on GPU memory

**Production Solution**: Process smaller batches sequentially and combine their results before updating the model. This technique is called gradient accumulation.
</details>

---

**Key insight**: These questions test your systems understanding of loss functions - not just "how do they work" but "how do they behave in production at scale." Keep these considerations in mind as you build real ML systems!
"""

# %% [markdown]
"""
## ⭐ Aha Moment: Loss Guides Learning

**What you built:** Loss functions that measure how wrong predictions are.

**Why it matters:** Without loss, there's no learning. The loss function is the "coach"
that tells the network whether its predictions are good or bad. Lower loss = better
predictions. Every training step aims to reduce this number.

Autograd computes gradients of this loss, giving the direction to adjust weights
to make predictions better!
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
- **Built 3 essential loss functions**: MSE, CrossEntropy, and BinaryCrossEntropy
- **Implemented numerical stability** with log-sum-exp trick for handling large logits
- **Discovered memory scaling patterns** with batch size and vocabulary size
- **Analyzed production trade-offs** between different loss function choices
- **All tests pass** (validated by `test_module()`)

### Systems Insights Discovered
- **Memory scaling**: CrossEntropy memory grows as B x C (batch x classes)
- **Numerical stability**: Log-sum-exp trick prevents overflow with large logits
- **Computational cost**: CE processes B x C logits; MSE scales with the number of predicted values
- **Production patterns**: Hierarchical softmax and sampled softmax for large vocabularies

### Ready for Next Steps
Your loss functions turn predictions into a single number to minimize. That
number is the only signal the optimizer ever sees, so getting it numerically
right matters more than getting it fast.

Export with: `tito module complete 04`

**Next**: Module 05 will add DataLoader for efficient data pipelines!
"""
