"""
Integration test for Module 04: Layers

Validates that the layers module integrates correctly with the TinyTorch package.
This is a quick validation test, not a comprehensive capability test.
"""

import sys
import importlib
import warnings
import numpy as np


def test_layers_module_integration():
    """Test that layers module integrates correctly with package."""

    # Suppress warnings for cleaner test output
    warnings.filterwarnings("ignore")

    results = {
        "module_name": "04_layers",
        "integration_type": "layers_validation",
        "tests": [],
        "success": True,
        "errors": []
    }

    try:
        # Test 1: Layers module imports from package
        try:
            from tinytorch.core.layers import Linear, Module
            results["tests"].append({
                "name": "layers_import",
                "status": "✅ PASS",
                "description": "Layer classes import from package"
            })
        except ImportError as e:
            results["tests"].append({
                "name": "layers_import",
                "status": "❌ FAIL",
                "description": f"Layers import failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Layers import error: {e}")
            return results

        # Test 2: Basic layer instantiation
        try:
            linear = Linear(4, 2)  # input_size=4, output_size=2
            results["tests"].append({
                "name": "layer_creation",
                "status": "✅ PASS",
                "description": "Linear layer can be instantiated"
            })
        except Exception as e:
            results["tests"].append({
                "name": "layer_creation",
                "status": "❌ FAIL",
                "description": f"Layer creation failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Layer creation error: {e}")
            return results

        # Test 3: Layer has required properties
        try:
            assert hasattr(linear, 'weight'), "Linear layer should have weight"
            assert hasattr(linear, 'bias'), "Linear layer should have bias"
            results["tests"].append({
                "name": "layer_properties",
                "status": "✅ PASS",
                "description": "Layer has required weight and bias properties"
            })
        except Exception as e:
            results["tests"].append({
                "name": "layer_properties",
                "status": "❌ FAIL",
                "description": f"Layer properties test failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Layer properties error: {e}")

        # Test 4: Integration with previous modules
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.activations import ReLU

            # Test forward pass
            data = np.random.randn(1, 4)
            tensor = Tensor(data)

            # Forward through linear layer
            output = linear.forward(tensor)
            assert hasattr(output, 'data'), "Layer should return tensor-like object"

            results["tests"].append({
                "name": "module_integration",
                "status": "✅ PASS",
                "description": "Layers work with Tensor and Activations"
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

        # Test 5: Required methods exist
        try:
            required_methods = ['forward', 'parameters']
            missing_methods = []

            for method in required_methods:
                if not hasattr(linear, method):
                    missing_methods.append(method)

            if not missing_methods:
                results["tests"].append({
                    "name": "required_methods",
                    "status": "✅ PASS",
                    "description": "Layer has all required methods"
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
        results["errors"].append(f"Unexpected error in layers integration test: {e}")
        results["tests"].append({
            "name": "unexpected_error",
            "status": "❌ FAIL",
            "description": f"Unexpected error: {e}"
        })

    return results


def run_integration_test():
    """Run the integration test and return results."""
    return test_layers_module_integration()


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
