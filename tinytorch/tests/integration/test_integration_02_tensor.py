"""
Integration test for Module 02: Tensor

Validates that the tensor module integrates correctly with the TinyTorch package.
This is a quick validation test, not a comprehensive capability test.
"""

import sys
import importlib
import warnings
import numpy as np


def test_tensor_module_integration():
    """Test that tensor module integrates correctly with package."""

    # Suppress warnings for cleaner test output
    warnings.filterwarnings("ignore")

    results = {
        "module_name": "02_tensor",
        "integration_type": "tensor_validation",
        "tests": [],
        "success": True,
        "errors": []
    }

    try:
        # Test 1: Tensor module imports from package
        try:
            from tinytorch.core.tensor import Tensor
            results["tests"].append({
                "name": "tensor_import",
                "status": "✅ PASS",
                "description": "Tensor class imports from package"
            })
        except ImportError as e:
            results["tests"].append({
                "name": "tensor_import",
                "status": "❌ FAIL",
                "description": f"Tensor import failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Tensor import error: {e}")
            return results  # Can't continue without Tensor

        # Test 2: Basic Tensor instantiation
        try:
            data = np.array([1, 2, 3, 4])
            tensor = Tensor(data)
            results["tests"].append({
                "name": "tensor_creation",
                "status": "✅ PASS",
                "description": "Tensor can be instantiated"
            })
        except Exception as e:
            results["tests"].append({
                "name": "tensor_creation",
                "status": "❌ FAIL",
                "description": f"Tensor creation failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Tensor creation error: {e}")
            return results

        # Test 3: Basic properties work
        try:
            assert hasattr(tensor, 'data'), "Tensor should have data attribute"
            assert hasattr(tensor, 'shape'), "Tensor should have shape attribute"
            assert tensor.shape == (4,), f"Expected shape (4,), got {tensor.shape}"

            results["tests"].append({
                "name": "basic_properties",
                "status": "✅ PASS",
                "description": "Basic tensor properties work"
            })
        except Exception as e:
            results["tests"].append({
                "name": "basic_properties",
                "status": "❌ FAIL",
                "description": f"Properties test failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Properties error: {e}")

        # Test 4: Package integration (can import alongside other modules)
        try:
            import tinytorch
            from tinytorch.core.tensor import Tensor
            # Try importing other core modules to check for conflicts
            try:
                import tinytorch.core.setup
            except ImportError:
                pass  # Setup might not be exported yet

            results["tests"].append({
                "name": "package_integration",
                "status": "✅ PASS",
                "description": "Tensor integrates with package structure"
            })
        except Exception as e:
            results["tests"].append({
                "name": "package_integration",
                "status": "❌ FAIL",
                "description": f"Package integration failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Package integration error: {e}")

        # Test 5: Basic arithmetic operations exist
        try:
            # Just check methods exist, don't test full functionality
            required_methods = ['__add__', '__mul__', '__repr__']
            missing_methods = []

            for method in required_methods:
                if not hasattr(tensor, method):
                    missing_methods.append(method)

            if not missing_methods:
                results["tests"].append({
                    "name": "basic_methods",
                    "status": "✅ PASS",
                    "description": "Required tensor methods are present"
                })
            else:
                results["tests"].append({
                    "name": "basic_methods",
                    "status": "❌ FAIL",
                    "description": f"Missing methods: {missing_methods}"
                })
                results["success"] = False
                results["errors"].append(f"Missing methods: {missing_methods}")

        except Exception as e:
            results["tests"].append({
                "name": "basic_methods",
                "status": "❌ FAIL",
                "description": f"Method check failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Method check error: {e}")

    except Exception as e:
        results["success"] = False
        results["errors"].append(f"Unexpected error in tensor integration test: {e}")
        results["tests"].append({
            "name": "unexpected_error",
            "status": "❌ FAIL",
            "description": f"Unexpected error: {e}"
        })

    return results


def run_integration_test():
    """Run the integration test and return results."""
    return test_tensor_module_integration()


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
