"""
Module 14: Progressive Integration Tests
Tests that Module 14 (Profiling) works correctly AND that prior modules (01→13) still work.

DEPENDENCY CHAIN: 01_tensor → ... → 12_attention → 13_transformers → 14_profiling

⚠️ IMPORTANT: This test ONLY uses modules 01-14.
   Future modules (15_quantization, 16_compression, 19_benchmarking, etc.) are NOT tested here.

🎯 WHAT THIS TESTS:
- Module 14: Profiler, memory profiling, execution timing
- Integration: Profiling works with transformers (13) and prior modules
- Regression: All previous modules still work correctly
"""

import numpy as np
rng = np.random.default_rng(7)
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestProfilingCore:
    """
    🆕 NEW FUNCTIONALITY: Test Module 14 (Profiling) core implementation.
    """

    def test_profiler_exists(self):
        """
        ✅ TEST: Profiler class exists
        """
        from tinytorch.perf.profiling import Profiler

        assert Profiler is not None

    def test_profiler_context_manager(self):
        """
        ✅ TEST: Profiler works as context manager
        """
        from tinytorch.perf.profiling import Profiler
        from tinytorch.core.tensor import Tensor

        profiler = Profiler()

        with profiler:
            # Some computation
            x = Tensor(rng.standard_normal((100, 100)))
            y = x @ x.transpose()

        # Should have recorded timing
        assert hasattr(profiler, 'elapsed') or hasattr(profiler, 'duration'), \
            "Profiler missing timing"

    def test_memory_profiling(self):
        from tinytorch.perf.profiling import Profiler
        from tinytorch.core.layers import Linear
        report = Profiler().measure_memory(Linear(4, 2), (3, 4))
        assert report['parameter_memory_mb'] > 0
        assert report['peak_memory_mb'] >= report['parameter_memory_mb']

    def test_execution_timing(self):
        from tinytorch.perf.profiling import Profiler
        from tinytorch.core.layers import Linear
        from tinytorch.core.tensor import Tensor
        assert Profiler().measure_latency(Linear(4, 2), Tensor([[1., 2., 3., 4.]]),
                                          warmup=1, iterations=3) > 0


class TestProfilingWithModels:
    """
    🔗 INTEGRATION: Profiling + Models (Modules 03-13)
    """

    def test_profile_linear_layer(self):
        """
        ✅ TEST: Profile Linear layer execution
        """
        from tinytorch.perf.profiling import Profiler
        from tinytorch.core.layers import Linear
        from tinytorch.core.tensor import Tensor

        layer = Linear(100, 50)
        profiler = Profiler()

        x = Tensor(rng.standard_normal((32, 100)))

        with profiler:
            for _ in range(10):
                output = layer(x)

        # Profiler should capture timing
        assert hasattr(profiler, 'elapsed') or hasattr(profiler, 'stats'), \
            "Profiler should capture stats"

    def test_profile_conv_layer(self):
        """
        ✅ TEST: Profile Conv2d layer execution
        """
        from tinytorch.perf.profiling import Profiler
        from tinytorch.core.spatial import Conv2d
        from tinytorch.core.tensor import Tensor

        conv = Conv2d(3, 16, kernel_size=3, padding=1)
        profiler = Profiler()

        x = Tensor(rng.standard_normal((4, 3, 32, 32)))

        with profiler:
            output = conv(x)

        assert output.shape[1] == 16

    def test_profile_transformer_block(self):
        """
        ✅ TEST: Profile TransformerBlock execution
        """
        from tinytorch.perf.profiling import Profiler
        from tinytorch.core.transformers import TransformerBlock
        from tinytorch.core.tensor import Tensor

        block = TransformerBlock(64, 8, ff_dim=256)
        profiler = Profiler()

        x = Tensor(rng.standard_normal((2, 10, 64)))

        with profiler:
            output = block(x)

        assert output.shape == x.shape


class TestProfilingWithTraining:
    """
    🔗 INTEGRATION: Profiling + Training (Module 08)
    """

    def test_profile_training_step(self):
        """
        ✅ TEST: Profile training step
        """
        from tinytorch.perf.profiling import Profiler
        from tinytorch.core.layers import Linear
        from tinytorch.core.losses import MSELoss
        from tinytorch.core.optimizers import SGD
        from tinytorch.core.tensor import Tensor

        layer = Linear(10, 5)
        loss_fn = MSELoss()
        optimizer = SGD(layer.parameters(), lr=0.1)

        profiler = Profiler()

        x = Tensor(rng.standard_normal((4, 10)))
        target = Tensor(rng.standard_normal((4, 5)))

        with profiler:
            pred = layer(x)
            loss = loss_fn(pred, target)

            if hasattr(loss, 'backward'):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        assert loss.data.size == 1


class TestRegressionPrevention:
    """
    🔄 REGRESSION: Verify all previous modules (01-13) still work.
    """

    def test_tensor_still_works(self):
        """✅ Module 01"""
        from tinytorch.core.tensor import Tensor
        a = Tensor([1, 2, 3])
        assert a.shape == (3,)

    def test_activations_still_work(self):
        """✅ Module 02"""
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.activations import ReLU
        relu = ReLU()
        x = Tensor([-1, 0, 1])
        y = relu(x)
        assert y.data[0] == 0

    def test_layers_still_work(self):
        """✅ Module 03"""
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.layers import Linear
        layer = Linear(4, 2)
        x = Tensor(rng.standard_normal((2, 4)))
        y = layer(x)
        assert y.shape == (2, 2)

    def test_losses_still_work(self):
        """✅ Module 04"""
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.losses import MSELoss
        loss_fn = MSELoss()
        loss = loss_fn(Tensor([[1.0]]), Tensor([[2.0]]))
        assert loss.data.size == 1

    def test_dataloader_still_works(self):
        """✅ Module 05"""
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.dataloader import TensorDataset, DataLoader
        data = Tensor(rng.standard_normal((10, 3)))
        targets = Tensor(np.arange(10).astype(float))
        dataset = TensorDataset(data, targets)
        dataloader = DataLoader(dataset, batch_size=2)
        assert sum(1 for _ in dataloader) == 5

    def test_optimizers_still_work(self):
        """✅ Module 07"""
        from tinytorch.core.optimizers import SGD
        from tinytorch.core.layers import Linear
        layer = Linear(3, 2)
        opt = SGD(layer.parameters(), lr=0.01)
        assert hasattr(opt, 'step')

    def test_convolutions_still_work(self):
        """✅ Module 09"""
        from tinytorch.core.spatial import Conv2d
        from tinytorch.core.tensor import Tensor
        conv = Conv2d(3, 8, kernel_size=3, padding=1)
        x = Tensor(rng.standard_normal((2, 3, 8, 8)))
        y = conv(x)
        assert y.shape[0] == 2

    def test_attention_still_works(self):
        """✅ Module 12"""
        from tinytorch.core.attention import MultiHeadAttention
        from tinytorch.core.tensor import Tensor
        mha = MultiHeadAttention(32, 4)
        x = Tensor(rng.standard_normal((1, 5, 32)))
        out = mha(x)
        assert out.shape == x.shape

    def test_transformers_still_work(self):
        """✅ Module 13"""
        from tinytorch.core.transformers import TransformerBlock
        from tinytorch.core.tensor import Tensor
        block = TransformerBlock(32, 4, ff_dim=128)
        x = Tensor(rng.standard_normal((1, 5, 32)))
        out = block(x)
        assert out.shape == x.shape


class TestModule14Completion:
    """
    ✅ COMPLETION CHECK: Module 14 ready for next module.
    """

    def test_profiling_foundation_complete(self):
        """
        ✅ FINAL TEST: Profiling ready for quantization

        🎯 SUCCESS = Ready for Module 15: Quantization!
        """
        from tinytorch.perf.profiling import Profiler
        from tinytorch.core.layers import Linear
        from tinytorch.core.tensor import Tensor
        report = Profiler().profile_forward_pass(Linear(4, 2), Tensor([[1., 2., 3., 4.]]))
        assert report['parameters'] == 10
        assert report['flops'] == 16
        assert report['latency_ms'] > 0
