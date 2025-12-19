"""
Integration test for Module 04: Linear

Validates that the dense module integrates correctly with the TinyTorch package.
This is a quick validation test, not a comprehensive capability test.
"""

import sys
import importlib
import warnings
import numpy as np


def test_dense_module_integration():
    """Test that dense module integrates correctly with package."""

    # Suppress warnings for cleaner test output
    warnings.filterwarnings("ignore")

    results = {
        "module_name": "05_dataloader",
        "integration_type": "dense_validation",
        "tests": [],
        "success": True,
        "errors": []
    }

    try:
        # Test 1: Linear networks import from package
        try:
            from tinytorch.core.dense import MLP, LinearNetwork
            results["tests"].append({
                "name": "dense_import",
                "status": "✅ PASS",
                "description": "Linear network classes import from package"
            })
        except ImportError as e:
            # Try alternative imports
            try:
                from tinytorch.core.networks import MLP
                results["tests"].append({
                    "name": "dense_import",
                    "status": "✅ PASS",
                    "description": "Linear networks import from alternative location"
                })
            except ImportError:
                results["tests"].append({
                    "name": "dense_import",
                    "status": "❌ FAIL",
                    "description": f"Linear import failed: {e}"
                })
                results["success"] = False
                results["errors"].append(f"Linear import error: {e}")
                return results

        # Test 2: Linear network instantiation
        try:
            mlp = MLP(input_size=4, hidden_sizes=[8, 4], output_size=2)
            results["tests"].append({
                "name": "dense_creation",
                "status": "✅ PASS",
                "description": "Linear networks can be instantiated"
            })
        except Exception as e:
            results["tests"].append({
                "name": "dense_creation",
                "status": "❌ FAIL",
                "description": f"Linear creation failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Linear creation error: {e}")
            return results

        # Test 3: Integration with previous modules
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear

            # Test forward pass
            data = np.random.randn(2, 4)  # batch_size=2, input_size=4
            tensor = Tensor(data)

            output = mlp.forward(tensor)
            assert hasattr(output, 'data'), "MLP should return tensor-like object"

            results["tests"].append({
                "name": "module_integration",
                "status": "✅ PASS",
                "description": "Linear networks work with previous modules"
            })
        except ImportError as e:
            results["tests"].append({
                "name": "module_integration",
                "status": "⏸️ SKIP",
                "description": f"Previous modules not available: {e}"
            })
        except Exception as e:
            results["tests"].append({
                "name": "module_integration",
                "status": "❌ FAIL",
                "description": f"Module integration failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Module integration error: {e}")

        # Test 4: Network structure
        try:
            layers = mlp.layers if hasattr(mlp, 'layers') else getattr(mlp, '_layers', [])
            assert len(layers) > 0, "MLP should have layers"

            results["tests"].append({
                "name": "network_structure",
                "status": "✅ PASS",
                "description": "Linear network has proper layer structure"
            })
        except Exception as e:
            results["tests"].append({
                "name": "network_structure",
                "status": "❌ FAIL",
                "description": f"Network structure test failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Network structure error: {e}")

        # Test 5: Required methods exist
        try:
            required_methods = ['forward']
            missing_methods = []

            for method in required_methods:
                if not hasattr(mlp, method):
                    missing_methods.append(method)

            if not missing_methods:
                results["tests"].append({
                    "name": "required_methods",
                    "status": "✅ PASS",
                    "description": "Linear network has all required methods"
                })
            else:
                results["tests"].append({
                    "name": "required_methods",
                    "status": "❌ FAIL",
                    "description": f"Missing methods: {missing_methods}"
                })
                results["success"] = False
                results["errors"].append(f"Missing methods: {missing_methods}")

        except Exception as e:
            results["tests"].append({
                "name": "required_methods",
                "status": "❌ FAIL",
                "description": f"Method check failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Method check error: {e}")

    except Exception as e:
        results["success"] = False
        results["errors"].append(f"Unexpected error in dense integration test: {e}")
        results["tests"].append({
            "name": "unexpected_error",
            "status": "❌ FAIL",
            "description": f"Unexpected error: {e}"
        })

    return results


def run_integration_test():
    """Run the integration test and return results."""
    return test_dense_module_integration()


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
