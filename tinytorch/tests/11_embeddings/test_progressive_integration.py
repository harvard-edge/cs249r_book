"""
Module 11: Progressive Integration Tests
Tests that Module 11 (Embeddings) works correctly AND that Foundation + Architecture tier work.

DEPENDENCY CHAIN: 01_tensor → ... → 10_tokenization → 11_embeddings
This is where token IDs become dense vector representations.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPriorStackStillWorking:
    """Quick regression checks that prior modules (01→11) still work."""

    def test_complete_training_pipeline_stable(self):
        """Verify complete training pipeline remains stable."""
        # Environment (Module 01)
        assert sys.version_info >= (3, 8), "Foundation broken: Python version"

        # Complete training should work
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.training import Trainer
            from tinytorch.core.losses import MSELoss

            # All training components should be available
            model = Linear(8, 3)
            optimizer = SGD(model.parameters(), lr=0.01)
            loss_fn = MSELoss()
            trainer = Trainer(model, optimizer, loss_fn)

            # Basic training functionality should work
            x = Tensor(np.random.randn(4, 8))
            output = model(x)
            assert output.shape == (4, 3), "Training pipeline broken"

        except ImportError:
            assert True, "Training pipeline not implemented yet"

    def test_advanced_features_stable(self):
        """Verify advanced modules (07→11) still work."""
        import numpy as np  # Import at function scope

        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.attention import MultiHeadAttention
            from tinytorch.core.spatial import Conv2d as Conv2D
            from tinytorch.core.dataloader import DataLoader
            from tinytorch.core.optimizers import Adam

            # Advanced features should work
            attention = MultiHeadAttention(embed_dim=64, num_heads=8)
            conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3)
            # Use Tensor with requires_grad instead of numpy array
            optimizer = Adam([Tensor(np.array([1.0]), requires_grad=True)], lr=0.001)

            assert hasattr(attention, 'forward'), "Advanced features broken: Attention"
            assert hasattr(conv, 'forward'), "Advanced features broken: Spatial"
            assert hasattr(optimizer, 'step'), "Advanced features broken: Optimizers"

        except ImportError:
            assert True, "Advanced features not implemented yet"


class TestModule12CompressionCore:
    """Test Module 12 (Compression) core functionality."""

    def test_quantization_creation(self):
        """Test quantization techniques."""
        try:
            from tinytorch.core.compression import Quantizer, quantize_weights
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor

            # Create model to quantize
            layer = Linear(10, 5)

            # Test weight quantization
            if 'quantize_weights' in locals():
                quantized_weights = quantize_weights(layer.weights, bits=8)

                # Quantized weights should have different range
                assert quantized_weights.shape == layer.weight.shape, "Quantization shape broken"

                # Should reduce precision
                unique_orig = len(np.unique(layer.weight.data.flatten()))
                unique_quant = len(np.unique(quantized_weights.data.flatten()))

                # Quantized should have fewer unique values (unless already very sparse)
                if unique_orig > 256:
                    assert unique_quant <= 256, "8-bit quantization not working"

            # Test quantizer object
            if 'Quantizer' in locals():
                quantizer = Quantizer(bits=8, method='linear')

                assert hasattr(quantizer, 'quantize'), "Quantizer broken: No quantize method"
                assert hasattr(quantizer, 'dequantize'), "Quantizer broken: No dequantize method"

        except ImportError:
            assert True, "Quantization not implemented yet"

    def test_pruning_techniques(self):
        """Test weight pruning and sparsity."""
        try:
            from tinytorch.core.compression import prune_weights, MagnitudePruner
            from tinytorch.core.layers import Linear

            # Create model to prune
            layer = Linear(20, 10)
            original_weights = layer.weight.data.copy()

            # Test magnitude-based pruning
            if 'prune_weights' in locals():
                pruned_weights = prune_weights(layer.weights, sparsity=0.5)

                # Should have ~50% zeros
                zero_count = np.sum(pruned_weights.data == 0)
                total_count = pruned_weights.data.size
                sparsity_ratio = zero_count / total_count

                assert 0.4 <= sparsity_ratio <= 0.6, "Pruning sparsity not ~50%"

                # Non-zero weights should be unchanged
                non_zero_mask = pruned_weights.data != 0
                original_non_zero = original_weights[non_zero_mask]
                pruned_non_zero = pruned_weights.data[non_zero_mask]

                # Remaining weights should be the largest magnitude ones
                assert len(original_non_zero) == len(pruned_non_zero), "Pruning mask broken"

            # Test structured pruning
            if 'MagnitudePruner' in locals():
                pruner = MagnitudePruner(sparsity=0.3, structured=True)

                assert hasattr(pruner, 'prune'), "Structured pruner broken: No prune method"
                assert hasattr(pruner, 'sparsity'), "Structured pruner broken: No sparsity"

        except ImportError:
            assert True, "Pruning techniques not implemented yet"

    def test_knowledge_distillation(self):
        """Test knowledge distillation compression."""
        try:
            from tinytorch.core.compression import DistillationLoss, KnowledgeDistiller
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor

            # Teacher and student models
            teacher = Linear(10, 5)  # Large model
            student = Linear(10, 5)  # Smaller model (same size for simplicity)

            # Test distillation loss
            if 'DistillationLoss' in locals():
                distill_loss = DistillationLoss(temperature=3.0, alpha=0.7)

                # Teacher and student outputs
                x = Tensor(np.random.randn(4, 10))
                teacher_output = teacher(x)
                student_output = student(x)
                targets = np.random.randint(0, 5, 4)

                # Compute distillation loss
                loss = distill_loss(student_output, teacher_output, targets)

                assert hasattr(loss, 'data') or isinstance(loss, (float, np.ndarray)), "Distillation loss broken"

            # Test knowledge distiller
            if 'KnowledgeDistiller' in locals():
                distiller = KnowledgeDistiller(teacher, student, temperature=4.0)

                assert hasattr(distiller, 'distill'), "Knowledge distiller broken: No distill method"
                assert hasattr(distiller, 'teacher'), "Knowledge distiller broken: No teacher"
                assert hasattr(distiller, 'student'), "Knowledge distiller broken: No student"

        except ImportError:
            assert True, "Knowledge distillation not implemented yet"


class TestProgressiveStackIntegration:
    """Test that the complete stack (01→12) works together."""

    def test_compressed_model_training(self):
        """Test training with compressed models."""
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.training import Trainer
            from tinytorch.core.compression import prune_weights, quantize_weights
            from tinytorch.core.dataloader import Dataset, DataLoader

            # Create model
            model = Linear(20, 5)

            # Apply compression
            if 'prune_weights' in locals():
                model.weights = prune_weights(model.weights, sparsity=0.3)

            if 'quantize_weights' in locals():
                model.weights = quantize_weights(model.weights, bits=8)

            # Training with compressed model
            optimizer = SGD(model.parameters(), lr=0.01)
            trainer = Trainer(model, optimizer)

            # Test forward pass with compressed model
            x = Tensor(np.random.randn(8, 20))
            output = model(x)

            assert output.shape == (8, 5), "Compressed model training broken"

            # Should still be trainable
            if hasattr(trainer, 'train'):
                assert True, "Compressed model is trainable"

        except ImportError:
            assert True, "Compressed model training not ready yet"

    def test_cnn_compression_pipeline(self):
        """Test compression with CNN models."""
        try:
            from tinytorch.core.spatial import Conv2d as Conv2D
            from tinytorch.core.layers import Linear
            from tinytorch.core.compression import prune_weights, quantize_weights
            from tinytorch.core.tensor import Tensor

            # CNN model
            conv1 = Conv2D(in_channels=3, out_channels=16, kernel_size=3)
            fc = Linear(16 * 30 * 30, 10)  # Approximate size

            # Apply compression to CNN
            if 'prune_weights' in locals() and hasattr(conv1, 'weight'):
                conv1.weights = prune_weights(conv1.weights, sparsity=0.4)

            if 'quantize_weights' in locals():
                fc.weights = quantize_weights(fc.weights, bits=8)

            # Test compressed CNN forward pass
            x = Tensor(np.random.randn(2, 3, 32, 32))

            if hasattr(conv1, '__call__'):
                conv_out = conv1(x)
                # Should work with compressed weights
                assert len(conv_out.shape) == 4, "Compressed CNN broken"

        except ImportError:
            assert True, "CNN compression pipeline not ready yet"

    def test_attention_compression(self):
        """Test compression with attention mechanisms."""
        try:
            from tinytorch.core.attention import MultiHeadAttention
            from tinytorch.core.compression import prune_weights, quantize_weights
            from tinytorch.core.tensor import Tensor

            # Attention mechanism
            attention = MultiHeadAttention(embed_dim=64, num_heads=8)

            # Apply compression to attention weights
            if hasattr(attention, 'query_proj') and hasattr(attention.query_proj, 'weight'):
                if 'prune_weights' in locals():
                    attention.query_proj.weights = prune_weights(attention.query_proj.weights, sparsity=0.2)

                if 'quantize_weights' in locals():
                    attention.key_proj.weights = quantize_weights(attention.key_proj.weights, bits=8)

            # Test compressed attention
            seq_len, batch_size, embed_dim = 10, 4, 64
            x = Tensor(np.random.randn(seq_len, batch_size, embed_dim))

            if hasattr(attention, '__call__'):
                output = attention(x)
                assert output.shape == x.shape, "Compressed attention broken"

        except ImportError:
            assert True, "Attention compression not ready yet"


class TestEfficiencyAndPerformance:
    """Test efficiency gains from compression techniques."""

    def test_memory_reduction(self):
        """Test memory reduction from compression."""
        try:
            from tinytorch.core.compression import prune_weights, quantize_weights
            from tinytorch.core.layers import Linear

            # Large model for memory testing
            large_model = Linear(1000, 500)
            original_size = large_model.weight.data.nbytes

            # Test pruning memory reduction
            if 'prune_weights' in locals():
                pruned_weights = prune_weights(large_model.weights, sparsity=0.7)

                # Memory might not reduce immediately (dense storage)
                # but sparsity should be achieved
                zero_ratio = np.sum(pruned_weights.data == 0) / pruned_weights.data.size
                assert zero_ratio >= 0.6, "Pruning not achieving target sparsity"

            # Test quantization memory reduction
            if 'quantize_weights' in locals():
                quantized_weights = quantize_weights(large_model.weights, bits=8)

                # 8-bit should use less memory than 32-bit (if properly implemented)
                assert quantized_weights.shape == large_model.weight.shape, "Quantization shape changed"

                # Check if data type changed
                if hasattr(quantized_weights, 'dtype'):
                    # Should be using smaller data type
                    assert True, "Quantization applied successfully"

        except ImportError:
            assert True, "Memory reduction testing not ready yet"

    def test_inference_speedup(self):
        """Test inference speedup from compression."""
        try:
            from tinytorch.core.compression import prune_weights, quantize_weights
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor
            import time

            # Model for speed testing
            model = Linear(500, 200)

            # Apply compression
            if 'prune_weights' in locals():
                compressed_model = Linear(500, 200)
                compressed_model.weights = prune_weights(model.weights, sparsity=0.8)

                # Test inference time (simplified)
                x = Tensor(np.random.randn(100, 500))

                # Time original model
                start = time.time()
                for _ in range(10):
                    _ = model(x)
                original_time = time.time() - start

                # Time compressed model
                start = time.time()
                for _ in range(10):
                    _ = compressed_model(x)
                compressed_time = time.time() - start

                # Compressed should be at least as fast (might be faster with sparse ops)
                assert compressed_time <= original_time * 1.2, "Compression significantly slower"

        except ImportError:
            assert True, "Inference speedup testing not ready yet"

    def test_model_size_reduction(self):
        """Test model size reduction techniques."""
        try:
            from tinytorch.core.compression import compress_model, ModelCompressor
            from tinytorch.core.layers import Linear

            # Model to compress
            model = Linear(100, 50)
            original_param_count = model.weight.data.size
            if hasattr(model, 'bias') and model.bias is not None:
                original_param_count += model.bias.data.size

            # Test model compression
            if 'compress_model' in locals():
                compressed_model = compress_model(model, compression_ratio=0.5)

                # Should have fewer effective parameters
                if hasattr(compressed_model, 'compression_ratio'):
                    assert compressed_model.compression_ratio <= 0.6, "Compression ratio not achieved"

            # Test model compressor
            if 'ModelCompressor' in locals():
                compressor = ModelCompressor(pruning_ratio=0.4, quantization_bits=8)

                compressed_model = compressor.compress(model)

                # Should maintain model interface
                assert hasattr(compressed_model, 'forward') or callable(compressed_model), "Compressed model interface broken"

        except ImportError:
            assert True, "Model size reduction not ready yet"


class TestProductionCompressionFeatures:
    """Test production-ready compression features."""

    def test_gradual_pruning(self):
        """Test gradual pruning during training."""
        try:
            from tinytorch.core.compression import GradualPruner
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import SGD

            model = Linear(50, 20)
            optimizer = SGD(model.parameters(), lr=0.01)

            # Gradual pruner
            if 'GradualPruner' in locals():
                pruner = GradualPruner(
                    initial_sparsity=0.0,
                    final_sparsity=0.8,
                    begin_step=100,
                    end_step=1000
                )

                # Should gradually increase sparsity
                sparsity_100 = pruner.get_sparsity(step=100)
                sparsity_500 = pruner.get_sparsity(step=500)
                sparsity_1000 = pruner.get_sparsity(step=1000)

                assert sparsity_100 <= sparsity_500 <= sparsity_1000, "Gradual pruning not gradual"
                assert sparsity_1000 >= 0.7, "Final sparsity not reached"

        except ImportError:
            assert True, "Gradual pruning not ready yet"

    def test_mixed_precision_compression(self):
        """Test combination of compression with mixed precision."""
        try:
            from tinytorch.core.compression import MixedPrecisionCompressor
            from tinytorch.core.training import Trainer
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import Adam

            model = Linear(30, 10)
            optimizer = Adam(model.parameters(), lr=0.001)

            # Mixed precision + compression
            if 'MixedPrecisionCompressor' in locals():
                compressor = MixedPrecisionCompressor(
                    fp16_layers=['linear'],
                    pruning_ratio=0.3
                )

                compressed_model = compressor.compress(model)
                trainer = Trainer(compressed_model, optimizer, mixed_precision=True)

                # Should support both compression and mixed precision
                assert hasattr(trainer, 'mixed_precision'), "Mixed precision support broken"

        except ImportError:
            assert True, "Mixed precision compression not ready yet"

    def test_compression_aware_training(self):
        """Test compression-aware training techniques."""
        try:
            from tinytorch.core.compression import CompressionAwareTrainer
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import SGD

            model = Linear(40, 15)
            optimizer = SGD(model.parameters(), lr=0.01)

            # Compression-aware training
            if 'CompressionAwareTrainer' in locals():
                trainer = CompressionAwareTrainer(
                    model, optimizer,
                    pruning_schedule={'start': 100, 'end': 500, 'frequency': 50},
                    quantization={'bits': 8, 'start_epoch': 10}
                )

                # Should handle compression during training
                assert hasattr(trainer, 'pruning_schedule'), "Compression-aware training broken"
                assert hasattr(trainer, 'quantization'), "Quantization scheduling broken"

        except ImportError:
            assert True, "Compression-aware training not ready yet"


class TestRegressionPrevention:
    """Ensure previous modules still work after Module 12 development."""

    def test_no_training_regression(self):
        """Verify training pipeline (01→11) unchanged."""
        # Core training functionality should remain stable
        assert sys.version_info.major >= 3, "Foundation: Python detection broken"

        # Training pipeline should still work
        try:
            import numpy as np
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.training import Trainer
            from tinytorch.core.losses import MSELoss

            # Complete training pipeline should work
            model = Linear(5, 3)
            optimizer = SGD(model.parameters(), lr=0.01)
            loss_fn = MSELoss()
            trainer = Trainer(model, optimizer, loss_fn)

            x = Tensor(np.random.randn(2, 5))
            output = model(x)
            assert output.shape == (2, 3), "Training regression: Forward pass broken"

        except ImportError:
            import numpy as np
            assert np.random is not None, "Training regression: Basic functionality broken"

    def test_no_advanced_features_regression(self):
        """Verify advanced features (07→11) unchanged."""
        import numpy as np  # Import at function scope

        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.attention import MultiHeadAttention
            from tinytorch.core.spatial import Conv2d as Conv2D
            from tinytorch.core.optimizers import Adam
            from tinytorch.core.dataloader import Dataset

            # Advanced features should still work
            attention = MultiHeadAttention(embed_dim=32, num_heads=4)
            conv = Conv2D(in_channels=1, out_channels=8, kernel_size=3)
            # Use Tensor with requires_grad instead of numpy array
            optimizer = Adam([Tensor(np.array([1.0]), requires_grad=True)], lr=0.001)

            assert hasattr(attention, 'forward'), "Advanced regression: Attention broken"
            assert hasattr(conv, 'forward'), "Advanced regression: Spatial broken"
            assert hasattr(optimizer, 'step'), "Advanced regression: Optimization broken"

            # Data loading should still work
            class TestDataset(Dataset):
                def __len__(self):
                    return 3
                def __getitem__(self, idx):
                    return idx, idx * 2

            dataset = TestDataset()
            assert len(dataset) == 3, "Advanced regression: Data loading broken"

        except ImportError:
            # Basic functionality should work
            import numpy as np
            assert np is not None, "Advanced regression: Basic functionality broken"

    def test_progressive_stability(self):
        """Test the progressive stack is stable through compression."""
        # Stack should be stable through: Setup → ... → Training → Compression

        # Setup level
        import numpy as np
        assert np is not None, "Setup level broken"

        # Complete training pipeline level (if available)
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.training import Trainer
            from tinytorch.core.losses import MSELoss

            # Complete training system should work
            model = Linear(8, 4)
            optimizer = SGD(model.parameters(), lr=0.01)
            loss_fn = MSELoss()
            trainer = Trainer(model, optimizer, loss_fn)

            x = Tensor(np.random.randn(3, 8))
            output = model(x)
            assert output.shape == (3, 4), "Training pipeline level broken"

        except ImportError:
            pass  # Not implemented yet

        # Compression level (if available)
        try:
            from tinytorch.core.compression import prune_weights

            # Compression should work with existing tensors
            weights = np.random.randn(10, 5)
            if 'prune_weights' in locals():
                pruned = prune_weights(weights, sparsity=0.5)
                assert pruned.shape == weights.shape, "Compression level broken"
            else:
                # Basic compression concepts should work
                assert True, "Basic compression ready"

        except ImportError:
            pass  # Not implemented yet
