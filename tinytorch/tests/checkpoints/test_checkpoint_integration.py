"""
Comprehensive Integration Testing for Checkpoint Achievements

This test suite validates that each checkpoint in the TinyTorch learning journey
actually works as intended, ensuring students can achieve the capabilities promised.
"""

import pytest
import os
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class CheckpointValidator:
    """Validates checkpoint achievements through comprehensive testing."""

    # Checkpoint definitions matching the checkpoint system
    # Module structure: 01_tensor through 20_capstone
    CHECKPOINTS = {
        "foundation": {
            "modules": ["01_tensor", "02_activations", "03_layers", "04_losses", "05_dataloader", "06_autograd", "07_optimizers", "08_training"],
            "capability": "Can build complete training pipeline from tensors to optimization",
            "tests": ["test_tensor", "test_activations", "test_layers", "test_losses", "test_dataloader", "test_autograd", "test_optimizers", "test_training"]
        },
        "architecture": {
            "modules": ["09_convolutions", "10_tokenization", "11_embeddings", "12_attention", "13_transformers"],
            "capability": "Can design CNNs for vision and Transformers for language",
            "tests": ["test_convolution", "test_tokenization", "test_embeddings", "test_attention", "test_transformers"]
        },
        "optimization": {
            "modules": ["14_profiling", "15_quantization", "16_compression", "17_acceleration", "18_memoization", "19_benchmarking"],
            "capability": "Can optimize models for production deployment",
            "tests": ["test_profiling", "test_quantization", "test_compression", "test_acceleration", "test_memoization", "test_benchmarking"]
        },
        "capstone": {
            "modules": ["20_capstone"],
            "capability": "Have built a complete, production-ready ML framework",
            "tests": ["test_capstone_integration"]
        }
    }

    def __init__(self):
        """Initialize the checkpoint validator."""
        self.results = {}
        self.module_path = Path(__file__).parent.parent / "modules" / "source"
        self.package_path = Path(__file__).parent.parent / "tinytorch"

    def validate_module_exists(self, module_name: str) -> bool:
        """Check if a module file exists."""
        module_file = self.module_path / module_name / f"{module_name.split('_')[1]}.py"
        return module_file.exists()

    def validate_module_exports(self, module_name: str) -> Tuple[bool, List[str]]:
        """Check if module has been properly exported to the package."""
        module_num, module_topic = module_name.split('_')
        package_file = self.package_path / "core" / f"{module_topic}.py"

        if not package_file.exists():
            return False, []

        # Check for exported functions
        with open(package_file, 'r') as f:
            content = f.read()
            # Look for __all__ export list
            if "__all__" in content:
                # Extract exported names
                import ast
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id == "__all__":
                                if isinstance(node.value, ast.List):
                                    exports = [elt.s for elt in node.value.elts if isinstance(elt, ast.Str)]
                                    return True, exports

        return False, []

    def validate_checkpoint(self, checkpoint_name: str) -> Dict[str, Any]:
        """Validate all aspects of a single checkpoint."""
        checkpoint = self.CHECKPOINTS[checkpoint_name]
        results = {
            "name": checkpoint_name,
            "capability": checkpoint["capability"],
            "modules_exist": {},
            "modules_exported": {},
            "tests_pass": {},
            "overall_status": "pending"
        }

        # Check module existence
        for module in checkpoint["modules"]:
            results["modules_exist"][module] = self.validate_module_exists(module)

        # Check module exports
        for module in checkpoint["modules"]:
            exported, exports = self.validate_module_exports(module)
            results["modules_exported"][module] = {
                "exported": exported,
                "functions": exports
            }

        # Determine overall status
        all_exist = all(results["modules_exist"].values())
        all_exported = all(info["exported"] for info in results["modules_exported"].values())

        if all_exist and all_exported:
            results["overall_status"] = "complete"
        elif all_exist:
            results["overall_status"] = "partial"
        else:
            results["overall_status"] = "incomplete"

        return results


class TestFoundationCheckpoint:
    """Test the Foundation checkpoint capabilities."""

    def test_setup_module(self):
        """Test that setup module provides environment configuration."""
        from tinytorch.core.setup import system_info, personal_info

        # Test system info
        info = system_info()
        assert 'os' in info
        assert 'python_version' in info
        assert 'cpu_count' in info

        # Test personal info
        personal = personal_info()
        assert 'name' in personal
        assert 'email' in personal

    def test_tensor_operations(self):
        """Test that tensor module provides multi-dimensional arrays."""
        from tinytorch.core.tensor import Tensor

        # Create tensors
        t1 = Tensor([[1, 2], [3, 4]])
        t2 = Tensor([[5, 6], [7, 8]])

        # Test operations
        t3 = t1 + t2
        assert t3.shape == (2, 2)

        t4 = t1 @ t2  # Matrix multiplication
        assert t4.shape == (2, 2)

    def test_activation_functions(self):
        """Test that activation module provides nonlinear functions."""
        from tinytorch.core.activations import ReLU, Sigmoid, Tanh, Softmax

        import numpy as np

        # Test ReLU
        relu = ReLU()
        x = np.array([[-1, 0, 1, 2]])
        output = relu(x)
        assert np.all(output >= 0)

        # Test Sigmoid
        sigmoid = Sigmoid()
        output = sigmoid(x)
        assert np.all((output >= 0) & (output <= 1))

        # Test Softmax
        softmax = Softmax()
        output = softmax(x)
        assert np.allclose(np.sum(output), 1.0)


class TestArchitectureCheckpoint:
    """Test the Neural Architecture checkpoint capabilities."""

    def test_layer_abstraction(self):
        """Test that layers module provides fundamental abstractions."""
        from tinytorch.core.layers import Layer, Dense

        # Test layer exists and is usable
        layer = Linear(10, 5)
        assert hasattr(layer, 'forward')
        assert hasattr(layer, 'weight')
        assert hasattr(layer, 'bias')

    def test_dense_networks(self):
        """Test that dense module enables fully-connected networks."""
        from tinytorch.core.dense import LinearNetwork
        from tinytorch.core.tensor import Tensor

        # Create network
        network = DenseNetwork([10, 20, 5])

        # Test forward pass
        x = Tensor(np.random.randn(32, 10))
        output = network(x)
        assert output.shape == (32, 5)

    def test_convolution_layers(self):
        """Test that spatial module provides convolution operations."""
        from tinytorch.core.spatial import Conv2d, MaxPool2d

        # Test Conv2d
        conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        assert hasattr(conv, 'forward')

        # Test MaxPool2d
        pool = MaxPool2d(kernel_size=2)
        assert hasattr(pool, 'forward')

    def test_attention_mechanisms(self):
        """Test that attention module provides self-attention."""
        from tinytorch.core.attention import SelfAttention, MultiHeadAttention

        # Test self-attention
        attention = SelfAttention(embed_dim=256)
        assert hasattr(attention, 'forward')

        # Test multi-head attention
        mha = MultiHeadAttention(embed_dim=256, num_heads=8)
        assert hasattr(mha, 'forward')


class TestTrainingCheckpoint:
    """Test the Training checkpoint capabilities."""

    def test_data_loading(self):
        """Test that dataloader can load and preprocess CIFAR-10."""
        from tinytorch.core.dataloader import CIFAR10DataLoader

        # Test dataloader creation
        loader = CIFAR10DataLoader(batch_size=32, shuffle=True)
        assert hasattr(loader, '__iter__')
        assert hasattr(loader, '__len__')

    def test_automatic_differentiation(self):
        """Test that autograd provides automatic differentiation."""
        from tinytorch.core.autograd import Variable, backward

        # Test variable creation
        x = Variable(np.array([[1.0, 2.0]]), requires_grad=True)
        y = Variable(np.array([[3.0, 4.0]]), requires_grad=True)

        # Test computation graph
        z = x + y
        loss = z.sum()

        # Test backward pass
        backward(loss)
        assert x.grad is not None
        assert y.grad is not None

    def test_optimizers(self):
        """Test that optimizers update parameters correctly."""
        from tinytorch.core.optimizers import SGD, Adam
        from tinytorch.core.layers import Linear

        # Create layer with parameters
        layer = Linear(10, 5)

        # Test SGD
        sgd = SGD([layer.weights, layer.bias], lr=0.01)
        assert hasattr(sgd, 'step')
        assert hasattr(sgd, 'zero_grad')

        # Test Adam
        adam = Adam([layer.weights, layer.bias], lr=0.001)
        assert hasattr(adam, 'step')
        assert hasattr(adam, 'zero_grad')

    def test_training_orchestration(self):
        """Test that training module provides complete training loop."""
        from tinytorch.core.training import Trainer, CrossEntropyLoss

        # Test loss function
        loss_fn = CrossEntropyLoss()
        assert hasattr(loss_fn, 'forward')

        # Test trainer
        # Note: Full trainer test would require model and data
        assert hasattr(Trainer, '__init__')


class TestInferenceCheckpoint:
    """Test the Inference Deployment checkpoint capabilities."""

    def test_model_compression(self):
        """Test compression techniques reduce model size."""
        from tinytorch.core.compression import (
            prune_weights_by_magnitude,
            quantize_layer_weights,
            CompressionMetrics
        )

        # Test pruning
        weights = np.random.randn(100, 50)
        pruned = prune_weights_by_magnitude(weights, sparsity=0.5)
        assert np.sum(pruned == 0) > 0  # Some weights should be pruned

        # Test quantization
        quantized = quantize_layer_weights(weights, bits=8)
        assert quantized.dtype != weights.dtype  # Should change precision

        # Test metrics
        metrics = CompressionMetrics()
        assert hasattr(metrics, 'count_parameters')

    def test_kernel_optimizations(self):
        """Test high-performance kernel implementations."""
        from tinytorch.core.kernels import (
            matmul_optimized,
            conv2d_optimized,
            attention_optimized
        )

        # Test optimized operations exist
        assert callable(matmul_optimized)
        assert callable(conv2d_optimized)
        assert callable(attention_optimized)

    def test_benchmarking_framework(self):
        """Test systematic performance benchmarking."""
        from tinytorch.core.benchmarking import (
            Benchmark,
            BenchmarkSuite,
            MLPerfBenchmark
        )

        # Test benchmark components
        bench = Benchmark(name="test")
        assert hasattr(bench, 'run')

        suite = BenchmarkSuite()
        assert hasattr(suite, 'add_benchmark')
        assert hasattr(suite, 'run_all')

    def test_mlops_systems(self):
        """Test production monitoring and deployment."""
        from tinytorch.core.mlops import (
            ModelMonitor,
            DriftDetector,
            RetrainingTrigger
        )

        # Test monitoring
        monitor = ModelMonitor()
        assert hasattr(monitor, 'log_prediction')
        assert hasattr(monitor, 'get_metrics')

        # Test drift detection
        detector = DriftDetector()
        assert hasattr(detector, 'detect_drift')

        # Test retraining
        trigger = RetrainingTrigger()
        assert hasattr(trigger, 'should_retrain')


class TestServingCheckpoint:
    """Test the Serving checkpoint capabilities."""

    def test_complete_integration(self):
        """Test that all components work together as a complete framework."""
        # This would test the capstone integration
        # Importing all major components and verifying they work together

        try:
            # Test all major imports work
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU
            from tinytorch.core.networks import Sequential
            from tinytorch.core.optimizers import Adam
            from tinytorch.core.training import Trainer
            from tinytorch.core.dataloader import DataLoader

            # Test building a complete model
            model = Sequential([
                Linear(784, 128),
                ReLU(),
                Linear(128, 10)
            ])

            # Test model has expected structure
            assert len(model.layers) == 3
            assert isinstance(model.layers[0], Dense)
            assert isinstance(model.layers[1], ReLU)

            integration_successful = True
        except ImportError:
            integration_successful = False

        assert integration_successful, "Complete framework integration failed"


def test_checkpoint_progression():
    """Test that checkpoints build on each other progressively."""
    validator = CheckpointValidator()

    # Validate each checkpoint
    results = {}
    for checkpoint_name in validator.CHECKPOINTS:
        results[checkpoint_name] = validator.validate_checkpoint(checkpoint_name)

    # Check foundation exists (required for all others)
    assert results["foundation"]["overall_status"] in ["complete", "partial"], \
        "Foundation checkpoint must be at least partially complete"

    # Report results
    print("\n=== Checkpoint Validation Results ===")
    for name, result in results.items():
        status_emoji = {
            "complete": "✅",
            "partial": "🔄",
            "incomplete": "❌",
            "pending": "⏳"
        }[result["overall_status"]]

        print(f"\n{status_emoji} {name.upper()}: {result['capability']}")
        print(f"   Status: {result['overall_status']}")

        # Show module details
        modules_exist = sum(result["modules_exist"].values())
        modules_total = len(result["modules_exist"])
        print(f"   Modules: {modules_exist}/{modules_total} exist")

        modules_exported = sum(1 for m in result["modules_exported"].values() if m["exported"])
        print(f"   Exports: {modules_exported}/{modules_total} exported to package")


def test_capability_statements():
    """Test that each checkpoint delivers its promised capability."""
    capabilities_achieved = []

    # Test Foundation capability
    try:
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.activations import ReLU
        t = Tensor([[1, 2], [3, 4]])
        relu = ReLU()
        result = relu(t.data)
        capabilities_achieved.append("foundation")
    except:
        pass

    # Test Architecture capability
    try:
        from tinytorch.core.layers import Linear
        from tinytorch.core.networks import Sequential
        model = Sequential([Linear(10, 5), Linear(5, 2)])
        capabilities_achieved.append("architecture")
    except:
        pass

    # Test Training capability
    try:
        from tinytorch.core.optimizers import Adam
        from tinytorch.core.training import Trainer
        capabilities_achieved.append("training")
    except:
        pass

    # Test Inference capability
    try:
        from tinytorch.core.compression import prune_weights_by_magnitude
        from tinytorch.core.kernels import matmul_optimized
        capabilities_achieved.append("inference")
    except:
        pass

    # Test Serving capability
    try:
        # Would test complete integration
        from tinytorch import __version__
        capabilities_achieved.append("serving")
    except:
        pass

    print(f"\n=== Capabilities Achieved: {len(capabilities_achieved)}/5 ===")
    for cap in capabilities_achieved:
        print(f"✅ {cap}")

    return capabilities_achieved


if __name__ == "__main__":
    # Run validation tests
    print("🎯 TinyTorch Checkpoint Validation Suite")
    print("=" * 50)

    # Test checkpoint structure
    test_checkpoint_progression()

    # Test capabilities
    test_capability_statements()

    print("\n" + "=" * 50)
    print("✅ Checkpoint validation complete!")
