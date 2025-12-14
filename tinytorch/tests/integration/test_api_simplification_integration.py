"""
Integration test for API Simplification

Validates that the new PyTorch-compatible API integrates correctly across all components:
- nn module with Module, Linear, Conv2d
- nn.functional with relu, flatten, max_pool2d
- optim module with Adam, SGD
- Complete workflow integration (model creation → optimizer → training setup)

This follows TinyTorch testing conventions:
1. Unit tests in test_api_simplification.py
2. Integration tests here (cross-component)
3. Examples as ultimate integration validation
"""

import sys
import warnings
import numpy as np


def test_modern_api_integration():
    """Test complete modern API integration across all components."""

    # Suppress warnings for cleaner test output
    warnings.filterwarnings("ignore")

    results = {
        "module_name": "api_simplification",
        "integration_type": "modern_api_validation",
        "tests": [],
        "success": True,
        "errors": []
    }

    try:
        # Test 1: Modern imports work
        try:
            import tinytorch.nn as nn
            import tinytorch.nn.functional as F
            import tinytorch.optim as optim
            from tinytorch.core.tensor import Tensor, Parameter

            results["tests"].append({
                "name": "modern_imports",
                "status": "✅ PASS",
                "description": "Modern PyTorch-like imports work"
            })
        except ImportError as e:
            results["tests"].append({
                "name": "modern_imports",
                "status": "❌ FAIL",
                "description": f"Modern imports failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Import error: {e}")
            return results

        # Test 2: Complete MLP workflow integration
        try:
            class SimpleMLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(4, 8)
                    self.fc2 = nn.Linear(8, 2)

                def forward(self, x):
                    x = F.relu(self.fc1(x))
                    return self.fc2(x)

            # Create model and optimizer
            model = SimpleMLP()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Test forward pass
            x = Tensor([[1.0, 2.0, 3.0, 4.0]])
            output = model(x)

            # Verify integration
            assert len(list(model.parameters())) == 4, "Should have 4 parameters"
            assert output.shape == (1, 2), f"Expected (1, 2), got {output.shape}"
            assert len(optimizer.parameters) == 4, "Optimizer should have 4 parameters"

            results["tests"].append({
                "name": "mlp_workflow_integration",
                "status": "✅ PASS",
                "description": "Complete MLP workflow integrates correctly"
            })
        except Exception as e:
            results["tests"].append({
                "name": "mlp_workflow_integration",
                "status": "❌ FAIL",
                "description": f"MLP workflow failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"MLP workflow error: {e}")

        # Test 3: Complete CNN workflow integration
        try:
            class SimpleCNN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(3, 8, (3, 3))
                    self.fc1 = nn.Linear(200, 10)  # Simplified calculation

                def forward(self, x):
                    x = F.relu(self.conv1(x))
                    x = F.flatten(x)
                    return self.fc1(x)

            # Create model and optimizer
            model = SimpleCNN()
            optimizer = optim.SGD(model.parameters(), lr=0.01)

            # Verify CNN integration
            params = list(model.parameters())
            assert len(params) == 4, f"Expected 4 parameters, got {len(params)}"
            assert len(optimizer.parameters) == 4, "Optimizer should have 4 parameters"

            # Test parameter shapes
            conv_weight = model.conv1.weight
            conv_bias = model.conv1.bias
            fc_weight = model.fc1.weights
            fc_bias = model.fc1.bias

            assert conv_weight.shape == (8, 3, 3, 3), f"Conv weight shape: {conv_weight.shape}"
            assert conv_bias.shape == (8,), f"Conv bias shape: {conv_bias.shape}"
            assert fc_weight.shape == (200, 10), f"FC weight shape: {fc_weight.shape}"
            assert fc_bias.shape == (10,), f"FC bias shape: {fc_bias.shape}"

            results["tests"].append({
                "name": "cnn_workflow_integration",
                "status": "✅ PASS",
                "description": "Complete CNN workflow integrates correctly"
            })
        except Exception as e:
            results["tests"].append({
                "name": "cnn_workflow_integration",
                "status": "❌ FAIL",
                "description": f"CNN workflow failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"CNN workflow error: {e}")

        # Test 4: Functional interface integration
        try:
            x = Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])

            # Test relu
            relu_out = F.relu(x)
            expected_relu = np.array([[0.0, 0.0, 0.0, 1.0, 2.0]])
            np.testing.assert_array_equal(relu_out.data, expected_relu)

            # Test flatten
            x2 = Tensor([[[[1, 2], [3, 4]]]])  # (1, 1, 2, 2)
            flat_out = F.flatten(x2)
            assert flat_out.shape == (1, 4), f"Flatten shape: {flat_out.shape}"

            results["tests"].append({
                "name": "functional_interface_integration",
                "status": "✅ PASS",
                "description": "Functional interface integrates correctly"
            })
        except Exception as e:
            results["tests"].append({
                "name": "functional_interface_integration",
                "status": "❌ FAIL",
                "description": f"Functional interface failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Functional interface error: {e}")

        # Test 5: Backward compatibility integration
        try:
            # Test old names still work
            from tinytorch.core.layers import Linear as Dense
            from tinytorch.core.spatial import MultiChannelConv2D

            dense = Linear(5, 3)
            conv = MultiChannelConv2D(3, 8, (3, 3))

            # Should be the same classes as new names
            assert type(dense).__name__ == 'Linear', f"Dense should be Linear, got {type(dense).__name__}"
            assert type(conv).__name__ == 'Conv2d', f"MultiChannelConv2D should be Conv2d, got {type(conv).__name__}"

            results["tests"].append({
                "name": "backward_compatibility_integration",
                "status": "✅ PASS",
                "description": "Backward compatibility maintained"
            })
        except Exception as e:
            results["tests"].append({
                "name": "backward_compatibility_integration",
                "status": "❌ FAIL",
                "description": f"Backward compatibility failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Backward compatibility error: {e}")

        # Test 6: Cross-module parameter integration
        try:
            # Test that parameters flow correctly across modules
            class ComplexModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv_block = nn.Module()
                    self.conv_block.conv1 = nn.Conv2d(3, 16, (3, 3))
                    self.conv_block.conv2 = nn.Conv2d(16, 32, (3, 3))

                    self.classifier = nn.Module()
                    self.classifier.fc1 = nn.Linear(512, 128)
                    self.classifier.fc2 = nn.Linear(128, 10)

                def forward(self, x):
                    return x  # Stub

            model = ComplexModel()
            params = list(model.parameters())

            # Should collect from nested modules
            assert len(params) == 8, f"Expected 8 parameters from nested modules, got {len(params)}"

            # Test optimizer works with nested parameters
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            assert len(optimizer.parameters) == 8, "Optimizer should get nested parameters"

            results["tests"].append({
                "name": "cross_module_parameter_integration",
                "status": "✅ PASS",
                "description": "Cross-module parameter collection works"
            })
        except Exception as e:
            results["tests"].append({
                "name": "cross_module_parameter_integration",
                "status": "❌ FAIL",
                "description": f"Cross-module parameters failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Cross-module error: {e}")

    except Exception as e:
        results["success"] = False
        results["errors"].append(f"Unexpected error: {e}")
        results["tests"].append({
            "name": "unexpected_error",
            "status": "❌ FAIL",
            "description": f"Unexpected error: {e}"
        })

    return results


def test_pytorch_api_compatibility():
    """Test that the API closely matches PyTorch patterns."""

    results = {
        "module_name": "api_simplification",
        "integration_type": "pytorch_compatibility",
        "tests": [],
        "success": True,
        "errors": []
    }

    try:
        # Test PyTorch-like import patterns
        import tinytorch.nn as nn
        import tinytorch.nn.functional as F
        import tinytorch.optim as optim

        # Test 1: PyTorch-like model definition
        try:
            class PyTorchLikeModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.features = nn.Module()
                    self.features.conv1 = nn.Conv2d(3, 64, (3, 3))
                    self.features.conv2 = nn.Conv2d(64, 128, (3, 3))

                    self.classifier = nn.Module()
                    self.classifier.fc1 = nn.Linear(2048, 512)
                    self.classifier.fc2 = nn.Linear(512, 10)

                def forward(self, x):
                    # Conv features
                    x = F.relu(self.features.conv1(x))
                    x = F.max_pool2d(x, (2, 2))
                    x = F.relu(self.features.conv2(x))
                    x = F.max_pool2d(x, (2, 2))

                    # Classifier
                    x = F.flatten(x)
                    x = F.relu(self.classifier.fc1(x))
                    x = self.classifier.fc2(x)
                    return x

            model = PyTorchLikeModel()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Should work exactly like PyTorch
            assert callable(model), "Model should be callable"
            assert len(list(model.parameters())) > 0, "Should have parameters"
            assert hasattr(optimizer, 'parameters'), "Optimizer should have parameters"

            results["tests"].append({
                "name": "pytorch_like_model_definition",
                "status": "✅ PASS",
                "description": "PyTorch-like model definition works"
            })
        except Exception as e:
            results["tests"].append({
                "name": "pytorch_like_model_definition",
                "status": "❌ FAIL",
                "description": f"PyTorch-like definition failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"PyTorch compatibility error: {e}")

        # Test 2: PyTorch-like training setup pattern
        try:
            # This should look exactly like PyTorch code
            model = nn.Linear(784, 10)
            optimizer = optim.SGD(model.parameters(), lr=0.01)

            # Test that syntax matches PyTorch
            params = model.parameters()
            assert hasattr(params, '__iter__'), "parameters() should be iterable"

            param_list = list(model.parameters())
            assert len(param_list) == 2, "Linear should have weight + bias"

            results["tests"].append({
                "name": "pytorch_training_setup",
                "status": "✅ PASS",
                "description": "PyTorch-like training setup works"
            })
        except Exception as e:
            results["tests"].append({
                "name": "pytorch_training_setup",
                "status": "❌ FAIL",
                "description": f"Training setup failed: {e}"
            })
            results["success"] = False
            results["errors"].append(f"Training setup error: {e}")

    except Exception as e:
        results["success"] = False
        results["errors"].append(f"PyTorch compatibility test error: {e}")

    return results


if __name__ == '__main__':
    print("🧪 TinyTorch API Simplification Integration Tests")
    print("=" * 60)

    # Run integration tests
    api_results = test_modern_api_integration()
    pytorch_results = test_pytorch_api_compatibility()

    # Print results
    all_results = [api_results, pytorch_results]
    total_tests = sum(len(r["tests"]) for r in all_results)
    total_passed = sum(len([t for t in r["tests"] if t["status"] == "✅ PASS"]) for r in all_results)
    total_failed = total_tests - total_passed

    print(f"\n📊 Integration Test Summary:")
    print(f"   Total tests: {total_tests}")
    print(f"   ✅ Passed: {total_passed}")
    print(f"   ❌ Failed: {total_failed}")

    # Detailed results
    for results in all_results:
        print(f"\n🔍 {results['integration_type']}:")
        for test in results["tests"]:
            print(f"   {test['status']} {test['name']}: {test['description']}")

        if results["errors"]:
            print(f"   Errors: {results['errors']}")

    # Overall success
    overall_success = all(r["success"] for r in all_results)
    if overall_success:
        print("\n✅ All integration tests passed!")
        print("🎉 API simplification integration is successful!")
    else:
        print("\n❌ Some integration tests failed!")
        print("🔧 Fix integration issues before deployment.")

    sys.exit(0 if overall_success else 1)
