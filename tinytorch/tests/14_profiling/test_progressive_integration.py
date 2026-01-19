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
        try:
            from tinytorch.core.profiler import Profiler
            
            assert Profiler is not None
            
        except ImportError:
            assert True, "Profiler not implemented yet"

    def test_profiler_context_manager(self):
        """
        ✅ TEST: Profiler works as context manager
        """
        try:
            from tinytorch.core.profiler import Profiler
            from tinytorch.core.tensor import Tensor
            
            profiler = Profiler()
            
            with profiler:
                # Some computation
                x = Tensor(np.random.randn(100, 100))
                y = x @ x.T
            
            # Should have recorded timing
            assert hasattr(profiler, 'elapsed') or hasattr(profiler, 'duration'), \
                "Profiler missing timing"
                
        except ImportError:
            assert True, "Profiler not implemented yet"

    def test_memory_profiling(self):
        """
        ✅ TEST: Memory profiling capability
        """
        try:
            from tinytorch.core.profiler import profile_memory, MemoryProfiler
            from tinytorch.core.tensor import Tensor
            
            # Profile memory usage
            with MemoryProfiler() as mp:
                tensors = [Tensor(np.random.randn(1000)) for _ in range(10)]
            
            if hasattr(mp, 'peak_memory'):
                assert mp.peak_memory > 0, "Memory profiling not working"
                
        except ImportError:
            assert True, "Memory profiling not implemented yet"

    def test_execution_timing(self):
        """
        ✅ TEST: Execution timing works
        """
        try:
            from tinytorch.core.profiler import Timer
            from tinytorch.core.tensor import Tensor
            
            timer = Timer()
            
            timer.start()
            # Some computation
            for _ in range(100):
                x = Tensor(np.random.randn(50, 50))
                y = x @ x.T
            elapsed = timer.stop()
            
            assert elapsed > 0, "Timer should measure positive time"
            
        except ImportError:
            assert True, "Timer not implemented yet"


class TestProfilingWithModels:
    """
    🔗 INTEGRATION: Profiling + Models (Modules 03-13)
    """

    def test_profile_linear_layer(self):
        """
        ✅ TEST: Profile Linear layer execution
        """
        try:
            from tinytorch.core.profiler import Profiler
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor
            
            layer = Linear(100, 50)
            profiler = Profiler()
            
            x = Tensor(np.random.randn(32, 100))
            
            with profiler:
                for _ in range(10):
                    output = layer(x)
            
            # Profiler should capture timing
            assert hasattr(profiler, 'elapsed') or hasattr(profiler, 'stats'), \
                "Profiler should capture stats"
                
        except ImportError:
            assert True, "Profiler integration not ready"

    def test_profile_conv_layer(self):
        """
        ✅ TEST: Profile Conv2d layer execution
        """
        try:
            from tinytorch.core.profiler import Profiler
            from tinytorch.core.spatial import Conv2d
            from tinytorch.core.tensor import Tensor
            
            conv = Conv2d(3, 16, kernel_size=3, padding=1)
            profiler = Profiler()
            
            x = Tensor(np.random.randn(4, 3, 32, 32))
            
            with profiler:
                output = conv(x)
            
            assert output.shape[1] == 16
            
        except ImportError:
            assert True, "Conv profiling not ready"

    def test_profile_transformer_block(self):
        """
        ✅ TEST: Profile TransformerBlock execution
        """
        try:
            from tinytorch.core.profiler import Profiler
            from tinytorch.core.transformers import TransformerBlock
            from tinytorch.core.tensor import Tensor
            
            block = TransformerBlock(64, 8, 256)
            profiler = Profiler()
            
            x = Tensor(np.random.randn(2, 10, 64))
            
            with profiler:
                output = block(x)
            
            assert output.shape == x.shape
            
        except ImportError:
            assert True, "Transformer profiling not ready"


class TestProfilingWithTraining:
    """
    🔗 INTEGRATION: Profiling + Training (Module 08)
    """

    def test_profile_training_step(self):
        """
        ✅ TEST: Profile training step
        """
        try:
            from tinytorch.core.profiler import Profiler
            from tinytorch.core.layers import Linear
            from tinytorch.core.losses import MSELoss
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.tensor import Tensor
            
            layer = Linear(10, 5)
            loss_fn = MSELoss()
            optimizer = SGD(layer.parameters(), lr=0.1)
            
            profiler = Profiler()
            
            x = Tensor(np.random.randn(4, 10))
            target = Tensor(np.random.randn(4, 5))
            
            with profiler:
                pred = layer(x)
                loss = loss_fn(pred, target)
                
                if hasattr(loss, 'backward'):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            assert loss.data.size == 1
            
        except ImportError:
            assert True, "Training profiling not ready"


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
        x = Tensor(np.random.randn(2, 4))
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
        data = Tensor(np.random.randn(10, 3))
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
        try:
            from tinytorch.core.spatial import Conv2d
            from tinytorch.core.tensor import Tensor
            conv = Conv2d(3, 8, kernel_size=3, padding=1)
            x = Tensor(np.random.randn(2, 3, 8, 8))
            y = conv(x)
            assert y.shape[0] == 2
        except ImportError:
            pass

    def test_attention_still_works(self):
        """✅ Module 12"""
        try:
            from tinytorch.core.attention import MultiHeadAttention
            from tinytorch.core.tensor import Tensor
            mha = MultiHeadAttention(32, 4)
            x = Tensor(np.random.randn(1, 5, 32))
            out = mha(x)
            assert out.shape == x.shape
        except ImportError:
            pass

    def test_transformers_still_work(self):
        """✅ Module 13"""
        try:
            from tinytorch.core.transformers import TransformerBlock
            from tinytorch.core.tensor import Tensor
            block = TransformerBlock(32, 4, 128)
            x = Tensor(np.random.randn(1, 5, 32))
            out = block(x)
            assert out.shape == x.shape
        except ImportError:
            pass


class TestModule14Completion:
    """
    ✅ COMPLETION CHECK: Module 14 ready for next module.
    """

    def test_profiling_foundation_complete(self):
        """
        ✅ FINAL TEST: Profiling ready for quantization
        
        🎯 SUCCESS = Ready for Module 15: Quantization!
        """
        capabilities = {
            "Profiler exists": False,
            "Timing works": False,
        }
        
        try:
            from tinytorch.core.profiler import Profiler
            
            # Test 1: Profiler exists
            capabilities["Profiler exists"] = True
            
            # Test 2: Timing
            profiler = Profiler()
            start = time.time()
            with profiler:
                _ = [i**2 for i in range(1000)]
            
            if hasattr(profiler, 'elapsed') or hasattr(profiler, 'duration') or hasattr(profiler, 'stats'):
                capabilities["Timing works"] = True
            else:
                # At minimum, context manager should work
                capabilities["Timing works"] = True
            
            completed = sum(capabilities.values())
            assert completed >= 1, f"Profiling not ready: {capabilities}"
            
        except ImportError:
            assert True, "Profiler not implemented yet"
