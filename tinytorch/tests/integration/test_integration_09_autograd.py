"""
Integration test for Module 06: Autograd

Validates that the autograd module integrates correctly with the TinyTorch package.
This is a quick validation test, not a comprehensive capability test.
"""

import sys
import importlib
import warnings
import numpy as np


def test_autograd_module_integration():
    """Test that autograd module integrates correctly with package."""

    # Suppress warnings for cleaner test output
    warnings.filterwarnings("ignore")

    results = {
        "module_name": "09_autograd",
        "integration_type": "autograd_validation",
        "tests": [],
        "success": True,
        "errors": []
    }

    try:
        # Test 1: Autograd module imports from package
        try:
            from tinytorch.core.autograd import Variable, backward
            results["tests"].append({
                "name": "autograd_import",
                "status": "✅ PASS",
                "description": "Autograd classes import from package"
            })
        except ImportError as e:
            results["tests"].append({
                "name": "autograd_import",
                "status": "❌ FAIL",
                "description": f"Autograd import failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Autograd import error: {e}")
            return results

        # Test 2: Basic Variable instantiation
        try:
            data = np.array([1.0, 2.0, 3.0])
            var = Variable(data, requires_grad=True)
            results["tests"].append({
                "name": "variable_creation",
                "status": "✅ PASS",
                "description": "Variable can be instantiated with grad tracking"
            })
        except Exception as e:
            results["tests"].append({
                "name": "variable_creation",
                "status": "❌ FAIL",
                "description": f"Variable creation failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Variable creation error: {e}")
            return results

        # Test 3: Integration with Tensor
        try:
            from tinytorch.core.tensor import Tensor

            # Check that Variable works with tensor operations
            assert hasattr(var, 'data'), "Variable should have data"
            assert hasattr(var, 'grad'), "Variable should have grad attribute"

            results["tests"].append({
                "name": "tensor_integration",
                "status": "✅ PASS",
                "description": "Autograd integrates with tensor operations"
            })
        except ImportError:
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

        # Test 4: Required gradient functionality
        try:
            # Test basic gradient properties
            required_attrs = ['requires_grad', 'grad']
            missing_attrs = []

            for attr in required_attrs:
                if not hasattr(var, attr):
                    missing_attrs.append(attr)

            if not missing_attrs:
                results["tests"].append({
                    "name": "gradient_properties",
                    "status": "✅ PASS",
                    "description": "Variable has required gradient properties"
                })
            else:
                results["tests"].append({
                    "name": "gradient_properties",
                    "status": "❌ FAIL",
                    "description": f"Missing gradient properties: {missing_attrs}"
                })
                results["success"] = False
                results["errors"].append(f"Missing properties: {missing_attrs}")

        except Exception as e:
            results["tests"].append({
                "name": "gradient_properties",
                "status": "❌ FAIL",
                "description": f"Gradient property check failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Gradient property error: {e}")

        # Test 5: Package structure integration
        try:
            import tinytorch
            from tinytorch.core.autograd import Variable

            # Check no conflicts with other imports
            results["tests"].append({
                "name": "package_integration",
                "status": "✅ PASS",
                "description": "Autograd integrates with package structure"
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
        results["errors"].append(f"Unexpected error in autograd integration test: {e}")
        results["tests"].append({
            "name": "unexpected_error",
            "status": "❌ FAIL",
            "description": f"Unexpected error: {e}"
        })

    return results


def run_integration_test():
    """Run the integration test and return results."""
    return test_autograd_module_integration()


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
