"""
Integration test for Module 03: Activations

Validates that the activations module integrates correctly with the TinyTorch package.
This is a quick validation test, not a comprehensive capability test.
"""

import sys
import importlib
import warnings
import numpy as np


def test_activations_module_integration():
    """Test that activations module integrates correctly with package."""

    # Suppress warnings for cleaner test output
    warnings.filterwarnings("ignore")

    results = {
        "module_name": "03_activations",
        "integration_type": "activations_validation",
        "tests": [],
        "success": True,
        "errors": []
    }

    try:
        # Test 1: Activations module imports from package
        try:
            from tinytorch.core.activations import ReLU, Sigmoid, Tanh
            results["tests"].append({
                "name": "activations_import",
                "status": "✅ PASS",
                "description": "Activation classes import from package"
            })
        except ImportError as e:
            results["tests"].append({
                "name": "activations_import",
                "status": "❌ FAIL",
                "description": f"Activations import failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Activations import error: {e}")
            return results

        # Test 2: Basic activation instantiation
        try:
            relu = ReLU()
            sigmoid = Sigmoid()
            tanh = Tanh()
            results["tests"].append({
                "name": "activation_creation",
                "status": "✅ PASS",
                "description": "Activation functions can be instantiated"
            })
        except Exception as e:
            results["tests"].append({
                "name": "activation_creation",
                "status": "❌ FAIL",
                "description": f"Activation creation failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Activation creation error: {e}")
            return results

        # Test 3: Integration with Tensor (if available)
        try:
            from tinytorch.core.tensor import Tensor

            # Test basic forward pass
            data = np.array([[-1.0, 0.0, 1.0, 2.0]])
            tensor = Tensor(data)

            # Test ReLU forward
            relu_result = relu.forward(tensor)
            assert hasattr(relu_result, 'data'), "ReLU should return tensor-like object"

            results["tests"].append({
                "name": "tensor_integration",
                "status": "✅ PASS",
                "description": "Activations work with Tensor objects"
            })
        except ImportError:
            # Tensor not available yet, skip this test
            results["tests"].append({
                "name": "tensor_integration",
                "status": "⏸️ SKIP",
                "description": "Tensor not available, skipping integration test"
            })
        except Exception as e:
            results["tests"].append({
                "name": "tensor_integration",
                "status": "❌ FAIL",
                "description": f"Tensor integration failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Tensor integration error: {e}")

        # Test 4: Required methods exist
        try:
            required_methods = ['forward', 'backward']
            for activation in [relu, sigmoid, tanh]:
                missing_methods = []
                for method in required_methods:
                    if not hasattr(activation, method):
                        missing_methods.append(f"{activation.__class__.__name__}.{method}")

                if missing_methods:
                    results["tests"].append({
                        "name": "required_methods",
                        "status": "❌ FAIL",
                        "description": f"Missing methods: {missing_methods}"
                    })
                    results["success"] = False
                    results["errors"].append(f"Missing methods: {missing_methods}")
                    break
            else:
                results["tests"].append({
                    "name": "required_methods",
                    "status": "✅ PASS",
                    "description": "All activation functions have required methods"
                })

        except Exception as e:
            results["tests"].append({
                "name": "required_methods",
                "status": "❌ FAIL",
                "description": f"Method check failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Method check error: {e}")

        # Test 5: Package structure integration
        try:
            import tinytorch
            from tinytorch.core.activations import ReLU

            # Check no conflicts with other imports
            try:
                from tinytorch.core.tensor import Tensor
            except ImportError:
                pass  # Tensor might not be available

            results["tests"].append({
                "name": "package_integration",
                "status": "✅ PASS",
                "description": "Activations integrate with package structure"
            })
        except Exception as e:
            results["tests"].append({
                "name": "package_integration",
                "status": "❌ FAIL",
                "description": f"Package integration failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Package integration error: {e}")

    except Exception as e:
        results["success"] = False
        results["errors"].append(f"Unexpected error in activations integration test: {e}")
        results["tests"].append({
            "name": "unexpected_error",
            "status": "❌ FAIL",
            "description": f"Unexpected error: {e}"
        })

    return results


def run_integration_test():
    """Run the integration test and return results."""
    return test_activations_module_integration()


if __name__ == "__main__":
    # Run test when script is executed directly
    result = run_integration_test()

    print(f"=== Integration Test: {result['module_name']} ===")
    print(f"Type: {result['integration_type']}")
    print(f"Overall Success: {result['success']}")
    print("\nTest Results:")

    for test in result["tests"]:
        print(f"  {test['status']} {test['name']}: {test['description']}")

    if result["errors"]:
        print(f"\nErrors:")
        for error in result["errors"]:
            print(f"  - {error}")

    # Exit with appropriate code
    sys.exit(0 if result["success"] else 1)
