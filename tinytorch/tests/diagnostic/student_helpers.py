#!/usr/bin/env python
"""
Student Diagnostic Helpers for TinyTorch
=========================================
Helpful diagnostic tools that guide students when things go wrong.
Provides clear error messages and suggestions for fixes.

Usage:
    python tests/diagnostic/student_helpers.py --check-all
    python tests/diagnostic/student_helpers.py --debug-training
"""

import sys
import os
import numpy as np
import argparse
from typing import Optional, List, Tuple, Any

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU, Sigmoid
from tinytorch.core.training import MeanSquaredError
from tinytorch.core.optimizers import SGD, Adam
from tinytorch.nn import Sequential


class DiagnosticHelper:
    """Helps students diagnose common issues in their implementations."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.issues_found = []
        self.suggestions = []

    def print_header(self, title: str):
        """Print a formatted section header."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🔍 {title}")
            print(f"{'='*60}")

    def print_success(self, message: str):
        """Print success message."""
        if self.verbose:
            print(f"✅ {message}")

    def print_warning(self, message: str):
        """Print warning message."""
        if self.verbose:
            print(f"⚠️  {message}")
        self.issues_found.append(("warning", message))

    def print_error(self, message: str):
        """Print error message."""
        if self.verbose:
            print(f"❌ {message}")
        self.issues_found.append(("error", message))

    def suggest(self, suggestion: str):
        """Add a suggestion for fixing issues."""
        if self.verbose:
            print(f"💡 Suggestion: {suggestion}")
        self.suggestions.append(suggestion)

    def summary(self):
        """Print diagnostic summary."""
        if not self.verbose:
            return

        print(f"\n{'='*60}")
        print("📊 DIAGNOSTIC SUMMARY")
        print(f"{'='*60}")

        if not self.issues_found:
            print("🎉 No issues found! Your implementation looks good.")
        else:
            print(f"Found {len(self.issues_found)} issue(s):")
            for issue_type, message in self.issues_found:
                icon = "❌" if issue_type == "error" else "⚠️"
                print(f"  {icon} {message}")

        if self.suggestions:
            print("\n💡 Suggestions to try:")
            for i, suggestion in enumerate(self.suggestions, 1):
                print(f"  {i}. {suggestion}")


def check_tensor_operations(helper: DiagnosticHelper):
    """Check basic tensor operations are working."""
    helper.print_header("Checking Tensor Operations")

    try:
        # Create tensors
        a = Tensor(np.array([[1, 2], [3, 4]]))
        b = Tensor(np.array([[5, 6], [7, 8]]))

        # Test shape
        if a.shape == (2, 2):
            helper.print_success("Tensor shape property works")
        else:
            helper.print_error(f"Tensor shape incorrect: expected (2, 2), got {a.shape}")
            helper.suggest("Check your Tensor.__init__ and shape property")

        # Test basic operations
        try:
            c = a + b  # If addition is implemented
            helper.print_success("Tensor addition works")
        except:
            helper.print_warning("Tensor addition not implemented (optional)")

        # Test reshaping
        d = a.reshape(4)
        if d.shape == (4,):
            helper.print_success("Tensor reshape works")
        else:
            helper.print_error(f"Reshape failed: expected (4,), got {d.shape}")
            helper.suggest("Check your reshape implementation")

    except Exception as e:
        helper.print_error(f"Tensor operations failed: {e}")
        helper.suggest("Review your Tensor class implementation")


def check_layer_initialization(helper: DiagnosticHelper):
    """Check layers initialize correctly."""
    helper.print_header("Checking Layer Initialization")

    try:
        # Linear layer
        linear = Linear(10, 5)

        if hasattr(linear, 'weight'):
            if linear.weight.shape == (10, 5):
                helper.print_success("Linear layer weights initialized correctly")
            else:
                helper.print_error(f"Linear weights wrong shape: {linear.weight.shape}")
                helper.suggest("Weights should be (input_size, output_size)")
        else:
            helper.print_error("Linear layer has no 'weights' attribute")
            helper.suggest("Add self.weights = Parameter(...) in Linear.__init__")

        if hasattr(linear, 'bias'):
            if linear.bias is not None and linear.bias.shape == (5,):
                helper.print_success("Linear layer bias initialized correctly")
            elif linear.bias is None:
                helper.print_warning("Linear layer has no bias (might be intentional)")
        else:
            helper.print_warning("Linear layer has no 'bias' attribute")

        # Check parameter collection
        params = linear.parameters()
        if len(params) > 0:
            helper.print_success(f"Parameter collection works ({len(params)} parameters)")
        else:
            helper.print_error("No parameters collected from Linear layer")
            helper.suggest("Check Module.parameters() and Parameter usage")

    except Exception as e:
        helper.print_error(f"Layer initialization failed: {e}")
        helper.suggest("Review your Linear and Module class implementations")


def check_forward_pass(helper: DiagnosticHelper):
    """Check forward passes work correctly."""
    helper.print_header("Checking Forward Pass")

    try:
        # Simple model
        model = Sequential([
            Linear(10, 20),
            ReLU(),
            Linear(20, 5)
        ])

        x = Tensor(np.random.randn(3, 10))

        try:
            y = model(x)
            if y.shape == (3, 5):
                helper.print_success("Sequential forward pass works")
            else:
                helper.print_error(f"Output shape wrong: expected (3, 5), got {y.shape}")
                helper.suggest("Check dimension calculations in forward pass")
        except Exception as e:
            helper.print_error(f"Forward pass failed: {e}")
            helper.suggest("Check your Sequential.forward() implementation")

        # Test individual components
        linear = Linear(10, 5)
        x = Tensor(np.random.randn(2, 10))
        y = linear(x)

        if y.shape == (2, 5):
            helper.print_success("Linear forward pass works")
        else:
            helper.print_error(f"Linear output wrong: expected (2, 5), got {y.shape}")

    except Exception as e:
        helper.print_error(f"Forward pass setup failed: {e}")


def check_loss_functions(helper: DiagnosticHelper):
    """Check loss functions compute correctly."""
    helper.print_header("Checking Loss Functions")

    try:
        # MSE Loss
        y_pred = Tensor(np.array([[1, 2], [3, 4]]))
        y_true = Tensor(np.array([[1, 2], [3, 4]]))

        criterion = MeanSquaredError()
        loss = criterion(y_pred, y_true)

        loss_val = float(loss.data) if hasattr(loss, 'data') else float(loss)

        if abs(loss_val - 0.0) < 1e-6:
            helper.print_success("MSE loss correct for identical inputs")
        else:
            helper.print_warning(f"MSE loss unexpected: {loss_val} (should be ~0)")

        # Non-zero loss
        y_pred = Tensor(np.array([[1, 2], [3, 4]]))
        y_true = Tensor(np.array([[0, 0], [0, 0]]))
        loss = criterion(y_pred, y_true)
        loss_val = float(loss.data) if hasattr(loss, 'data') else float(loss)

        expected = np.mean((y_pred.data - y_true.data) ** 2)
        if abs(loss_val - expected) < 1e-6:
            helper.print_success("MSE loss computation correct")
        else:
            helper.print_error(f"MSE loss wrong: got {loss_val}, expected {expected}")
            helper.suggest("Check your MSE calculation: mean((pred - true)^2)")

    except Exception as e:
        helper.print_error(f"Loss function check failed: {e}")


def check_gradient_flow(helper: DiagnosticHelper):
    """Check if gradients flow through the network."""
    helper.print_header("Checking Gradient Flow")

    try:
        model = Linear(5, 3)
        x = Tensor(np.random.randn(2, 5))
        y_true = Tensor(np.random.randn(2, 3))

        y_pred = model(x)
        loss = MeanSquaredError()(y_pred, y_true)

        try:
            loss.backward()

            if hasattr(model.weights, 'grad') and model.weight.grad is not None:
                helper.print_success("Gradients computed for weights")
                grad_mag = np.abs(model.weight.grad.data).mean()
                if grad_mag > 1e-8:
                    helper.print_success(f"Gradient magnitude reasonable: {grad_mag:.6f}")
                else:
                    helper.print_warning(f"Gradients very small: {grad_mag}")
                    helper.suggest("Check for vanishing gradient issues")
            else:
                helper.print_warning("No gradients computed (autograd might not be implemented)")
                helper.suggest("This is okay if you haven't implemented autograd yet")

        except AttributeError:
            helper.print_warning("Autograd not implemented (expected for early modules)")
        except Exception as e:
            helper.print_error(f"Backward pass failed: {e}")

    except Exception as e:
        helper.print_error(f"Gradient flow check failed: {e}")


def check_optimizer_updates(helper: DiagnosticHelper):
    """Check if optimizers update parameters correctly."""
    helper.print_header("Checking Optimizer Updates")

    try:
        model = Linear(5, 3)
        optimizer = SGD(model.parameters(), learning_rate=0.1)

        # Save initial weights
        initial_weights = model.weight.data.copy()

        x = Tensor(np.random.randn(2, 5))
        y_true = Tensor(np.random.randn(2, 3))

        # Training step
        y_pred = model(x)
        loss = MeanSquaredError()(y_pred, y_true)

        try:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Check if weights changed
            if not np.allclose(initial_weights, model.weight.data):
                helper.print_success("SGD updates weights")
                update_size = np.abs(model.weight.data - initial_weights).mean()
                helper.print_success(f"Average weight update: {update_size:.6f}")
            else:
                helper.print_error("Weights didn't change after optimizer.step()")
                helper.suggest("Check your SGD.step() implementation")

        except AttributeError:
            helper.print_warning("Optimizer operations not fully implemented")
        except Exception as e:
            helper.print_error(f"Optimizer update failed: {e}")

    except Exception as e:
        helper.print_error(f"Optimizer check failed: {e}")


def diagnose_training_loop(helper: DiagnosticHelper):
    """Diagnose issues in a complete training loop."""
    helper.print_header("Diagnosing Training Loop")

    try:
        # Simple dataset
        X = Tensor(np.random.randn(20, 5))
        y = Tensor(np.random.randn(20, 2))

        # Simple model
        model = Sequential([
            Linear(5, 10),
            ReLU(),
            Linear(10, 2)
        ])

        optimizer = Adam(model.parameters(), learning_rate=0.01)
        criterion = MeanSquaredError()

        losses = []
        for epoch in range(5):
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss_val = float(loss.data) if hasattr(loss, 'data') else float(loss)
            losses.append(loss_val)

            try:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except:
                pass

        # Analyze training
        if len(losses) == 5:
            helper.print_success("Training loop completed 5 epochs")

            # Check if loss is decreasing
            if losses[-1] < losses[0]:
                helper.print_success(f"Loss decreased: {losses[0]:.4f} → {losses[-1]:.4f}")
            elif losses[-1] > losses[0] * 1.5:
                helper.print_warning("Loss increased during training")
                helper.suggest("Try reducing learning rate")
                helper.suggest("Check for bugs in backward pass")
            else:
                helper.print_warning("Loss didn't decrease much")
                helper.suggest("Try increasing learning rate or training longer")

            # Check for NaN
            if any(np.isnan(loss) for loss in losses):
                helper.print_error("NaN detected in losses")
                helper.suggest("Learning rate might be too high")
                helper.suggest("Check for numerical instability")

        else:
            helper.print_error(f"Training incomplete: only {len(losses)} epochs")

    except Exception as e:
        helper.print_error(f"Training loop failed: {e}")
        helper.suggest("Check your training setup step by step")


def check_common_mistakes(helper: DiagnosticHelper):
    """Check for common student mistakes."""
    helper.print_header("Checking Common Mistakes")

    # Check 1: Forgetting to call zero_grad
    model = Linear(5, 3)
    optimizer = SGD(model.parameters(), learning_rate=0.01)

    x = Tensor(np.random.randn(2, 5))
    y_true = Tensor(np.random.randn(2, 3))

    try:
        # First forward/backward
        loss1 = MeanSquaredError()(model(x), y_true)
        loss1.backward()

        # Second forward/backward WITHOUT zero_grad
        loss2 = MeanSquaredError()(model(x), y_true)
        loss2.backward()

        # Gradients would accumulate if zero_grad not called
        helper.print_warning("Remember to call optimizer.zero_grad() before each backward()")
    except:
        pass

    # Check 2: Wrong tensor dimensions
    try:
        linear = Linear(10, 5)
        wrong_input = Tensor(np.random.randn(5, 20))  # Wrong shape!
        try:
            output = linear(wrong_input)
            helper.print_error("Linear layer accepted wrong input shape!")
        except:
            helper.print_success("Linear layer correctly rejects wrong input shape")
    except:
        pass

    # Check 3: Uninitialized parameters
    try:
        linear = Linear(10, 5)
        if hasattr(linear, 'weight'):
            if np.all(linear.weight.data == 0):
                helper.print_error("Weights initialized to all zeros")
                helper.suggest("Use random initialization to break symmetry")
            else:
                helper.print_success("Weights randomly initialized")
    except:
        pass

    # Check 4: Learning rate issues
    helper.print_success("Common mistake checks completed")
    helper.suggest("Common learning rates to try: 0.001, 0.01, 0.1")
    helper.suggest("Start with small learning rate and increase if loss decreases slowly")


def run_all_diagnostics(verbose: bool = True):
    """Run all diagnostic checks."""
    helper = DiagnosticHelper(verbose=verbose)

    print("\n" + "="*60)
    print("🏥 TINYTORCH DIAGNOSTIC TOOL")
    print("Helping you debug your implementation")
    print("="*60)

    # Run all checks
    check_tensor_operations(helper)
    check_layer_initialization(helper)
    check_forward_pass(helper)
    check_loss_functions(helper)
    check_gradient_flow(helper)
    check_optimizer_updates(helper)
    diagnose_training_loop(helper)
    check_common_mistakes(helper)

    # Summary
    helper.summary()

    return len(helper.issues_found) == 0


def main():
    """Main entry point for diagnostic tool."""
    parser = argparse.ArgumentParser(
        description="TinyTorch Student Diagnostic Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--check-all",
        action="store_true",
        help="Run all diagnostic checks"
    )
    parser.add_argument(
        "--debug-training",
        action="store_true",
        help="Debug training loop issues"
    )
    parser.add_argument(
        "--check-shapes",
        action="store_true",
        help="Check tensor shape operations"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less verbose output"
    )

    args = parser.parse_args()

    verbose = not args.quiet

    if args.check_all or (not any([args.debug_training, args.check_shapes])):
        success = run_all_diagnostics(verbose=verbose)
        sys.exit(0 if success else 1)

    helper = DiagnosticHelper(verbose=verbose)

    if args.debug_training:
        diagnose_training_loop(helper)
        check_gradient_flow(helper)
        check_optimizer_updates(helper)

    if args.check_shapes:
        check_tensor_operations(helper)
        check_forward_pass(helper)

    helper.summary()
    sys.exit(0 if not helper.issues_found else 1)


if __name__ == "__main__":
    main()
