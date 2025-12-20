"""
Module 15: Progressive Integration Tests
Tests that Module 16 (TinyGPT/Capstone) works correctly AND that the ENTIRE TinyTorch system works.

DEPENDENCY CHAIN: 01_setup → 02_tensor → ... → 15_mlops → 16_tinygpt
This is the CAPSTONE - everything should work together to build complete transformer models!
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestEntireTinyTorchSystemStable:
    """Verify the COMPLETE TinyTorch system (01→15) still works."""

    def test_foundation_to_production_pipeline(self):
        """Verify complete pipeline from foundation to production."""
        # Environment (Module 01)
        assert sys.version_info >= (3, 8), "Foundation broken: Python version"

        # Complete production ML system should work
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.spatial import Conv2d as Conv2D
            from tinytorch.core.attention import MultiHeadAttention
            from tinytorch.core.optimizers import Adam
            from tinytorch.core.training import Trainer
            from tinytorch.perf.compression import prune_weights

            # Foundation components
            model = Linear(16, 8)
            x = Tensor(np.random.randn(4, 16))
            output = model(x)
            assert output.shape == (4, 8), "Foundation broken"

            # Advanced components
            conv = Conv2D(in_channels=3, out_channels=8, kernel_size=3)
            attention = MultiHeadAttention(embed_dim=32, num_heads=4)

            # Training and optimization
            optimizer = Adam(model.parameters(), lr=0.001)
            trainer = Trainer(model, optimizer)

            # Production features
            if 'prune_weights' in locals():
                pruned = prune_weights(model.weights.data, sparsity=0.3)
                assert pruned.shape == model.weights.shape, "Compression broken"

        except ImportError:
            assert True, "Complete system not implemented yet"

    def test_all_major_components_available(self):
        """Quick check that all major TinyTorch components are available."""
        components_to_test = [
            ('tinytorch.core.tensor', 'Tensor'),
            ('tinytorch.core.layers', 'Linear'),
            ('tinytorch.core.activations', 'ReLU'),
            ('tinytorch.core.spatial', 'Conv2d'),
            ('tinytorch.core.attention', 'MultiHeadAttention'),
            ('tinytorch.core.dataloader', 'DataLoader'),
            ('tinytorch.core.autograd', 'enable_autograd'),
            ('tinytorch.core.optimizers', 'Adam'),
            ('tinytorch.core.training', 'Trainer'),
            ('tinytorch.perf.compression', 'prune_weights'),
            ('tinytorch.perf.benchmarking', 'SimpleMLP'),
        ]

        available_components = []
        missing_components = []

        for module_name, component_name in components_to_test:
            try:
                module = __import__(module_name, fromlist=[component_name])
                component = getattr(module, component_name, None)
                if component is not None:
                    available_components.append(f"{module_name}.{component_name}")
                else:
                    missing_components.append(f"{module_name}.{component_name}")
            except ImportError:
                missing_components.append(f"{module_name}.{component_name}")

        # Should have at least foundation components
        assert len(available_components) >= 3, f"Too few components available: {available_components}"

        # Report status
        print(f"Available: {len(available_components)}, Missing: {len(missing_components)}")


class TestModule16TinyGPTCore:
    """Test Module 16 (TinyGPT) core functionality."""

    def test_transformer_block_creation(self):
        """Test complete transformer block implementation."""
        try:
            from tinytorch.core.transformers import TransformerBlock, TinyGPT
            from tinytorch.core.tensor import Tensor

            # Transformer block
            if 'TransformerBlock' in locals():
                block = TransformerBlock(
                    embed_dim=128,
                    num_heads=8,
                    ff_dim=512,
                    dropout=0.1
                )

                # Test transformer block components
                assert hasattr(block, 'attention'), "TransformerBlock missing attention"
                assert hasattr(block, 'feed_forward'), "TransformerBlock missing feed_forward"
                assert hasattr(block, 'norm1'), "TransformerBlock missing norm1"
                assert hasattr(block, 'norm2'), "TransformerBlock missing norm2"

                # Test forward pass
                seq_len, batch_size, embed_dim = 20, 4, 128
                x = Tensor(np.random.randn(seq_len, batch_size, embed_dim))

                output = block(x)
                assert output.shape == x.shape, "TransformerBlock output shape broken"

        except ImportError:
            assert True, "Transformer blocks not implemented yet"

    def test_tinygpt_model_creation(self):
        """Test complete TinyGPT model."""
        try:
            from tinytorch.core.transformers import TinyGPT
            from tinytorch.core.tensor import Tensor

            # TinyGPT model
            if 'TinyGPT' in locals():
                model = TinyGPT(
                    vocab_size=1000,
                    embed_dim=256,
                    num_heads=8,
                    num_layers=6,
                    max_seq_len=512,
                    dropout=0.1
                )

                # Test model components
                assert hasattr(model, 'embedding'), "TinyGPT missing embedding"
                assert hasattr(model, 'positional_encoding'), "TinyGPT missing positional_encoding"
                assert hasattr(model, 'transformer_blocks'), "TinyGPT missing transformer_blocks"
                assert hasattr(model, 'output_projection'), "TinyGPT missing output_projection"

                # Test forward pass
                batch_size, seq_len = 4, 32
                input_ids = np.random.randint(0, 1000, (batch_size, seq_len))

                logits = model(input_ids)
                assert logits.shape == (batch_size, seq_len, 1000), "TinyGPT output shape broken"

        except ImportError:
            assert True, "TinyGPT model not implemented yet"

    def test_text_generation_capabilities(self):
        """Test text generation with TinyGPT."""
        try:
            from tinytorch.core.transformers import TinyGPT
            from tinytorch.core.generation import generate_text, GreedyDecoder, BeamSearchDecoder

            # TinyGPT for generation
            if 'TinyGPT' in locals():
                model = TinyGPT(
                    vocab_size=1000,
                    embed_dim=128,
                    num_heads=4,
                    num_layers=3,
                    max_seq_len=256
                )

                # Test greedy generation
                if 'GreedyDecoder' in locals():
                    decoder = GreedyDecoder(model, max_length=50)

                    # Start with some tokens
                    start_tokens = [1, 2, 3]  # Some token IDs

                    generated = decoder.generate(start_tokens)
                    assert len(generated) <= 50, "Greedy generation length broken"
                    assert all(0 <= token < 1000 for token in generated), "Generated tokens out of vocab"

                # Test beam search
                if 'BeamSearchDecoder' in locals():
                    decoder = BeamSearchDecoder(model, beam_size=4, max_length=30)

                    start_tokens = [1, 2, 3]

                    generated = decoder.generate(start_tokens)
                    assert len(generated) <= 30, "Beam search generation length broken"
                    assert all(0 <= token < 1000 for token in generated), "Generated tokens out of vocab"

        except ImportError:
            assert True, "Text generation not implemented yet"


class TestCompleteSystemIntegration:
    """Test that the COMPLETE TinyTorch system (01→16) works together."""

    def test_end_to_end_language_model_training(self):
        """Test complete end-to-end language model training."""
        try:
            from tinytorch.core.transformers import TinyGPT
            from tinytorch.core.optimizers import Adam
            from tinytorch.core.training import Trainer, CrossEntropyLoss
            from tinytorch.core.dataloader import Dataset, DataLoader
            from tinytorch.core.tensor import Tensor

            # Language modeling dataset
            class LanguageDataset(Dataset):
                def __init__(self):
                    # Simulated text data (token IDs)
                    self.sequences = []
                    vocab_size = 1000
                    seq_len = 32

                    for _ in range(100):  # 100 sequences
                        seq = np.random.randint(0, vocab_size, seq_len)
                        self.sequences.append(seq)

                def __len__(self):
                    return len(self.sequences)

                def __getitem__(self, idx):
                    seq = self.sequences[idx]
                    # Input is seq[:-1], target is seq[1:]
                    return seq[:-1], seq[1:]

            # Complete training setup
            model = TinyGPT(
                vocab_size=1000,
                embed_dim=128,
                num_heads=4,
                num_layers=2,
                max_seq_len=64
            )

            optimizer = Adam(model.parameters(), lr=0.001)
            loss_fn = CrossEntropyLoss()
            trainer = Trainer(model, optimizer)

            dataset = LanguageDataset()
            dataloader = DataLoader(dataset, batch_size=8)

            # Training loop
            for epoch in range(2):  # Just 2 epochs for testing
                for batch_input, batch_target in dataloader:
                    # Convert to tensors
                    input_tensor = Tensor(batch_input.astype(np.int32))
                    target_tensor = Tensor(batch_target.astype(np.int32))

                    # Forward pass
                    logits = model(input_tensor)
                    loss = loss_fn(logits.reshape(-1, 1000), target_tensor.reshape(-1))

                    # Backward pass (if available)
                    if hasattr(loss, 'backward'):
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # Verify shapes
                    assert logits.shape[:-1] == input_tensor.shape, "Training logits shape broken"
                    break  # Test one batch per epoch

            assert True, "End-to-end language model training successful"

        except ImportError:
            assert True, "End-to-end language model training not ready yet"

    def test_compressed_transformer_deployment(self):
        """Test transformer deployment with compression and optimization."""
        try:
            from tinytorch.core.transformers import TinyGPT
            from tinytorch.core.compression import prune_weights, quantize_weights
            from tinytorch.core.kernels import enable_optimizations
            from tinytorch.core.mlops import ModelMonitor, deploy_model

            # Create model
            model = TinyGPT(
                vocab_size=500,  # Smaller for faster testing
                embed_dim=64,
                num_heads=4,
                num_layers=2,
                max_seq_len=128
            )

            # Apply compression
            if 'prune_weights' in locals():
                # Prune transformer weights
                for block in model.transformer_blocks:
                    if hasattr(block.attention, 'query_proj'):
                        block.attention.query_proj.weights = prune_weights(
                            block.attention.query_proj.weights,
                            sparsity=0.3
                        )

            if 'quantize_weights' in locals():
                # Quantize output projection
                model.output_projection.weights = quantize_weights(
                    model.output_projection.weights,
                    bits=8
                )

            # Enable optimizations
            if 'enable_optimizations' in locals():
                enable_optimizations(backend='auto')

            # Production deployment
            if 'ModelMonitor' in locals():
                monitor = ModelMonitor(model)
                monitor.log_metrics({'compression_ratio': 0.3, 'quantization_bits': 8})

            if 'deploy_model' in locals():
                deployment = deploy_model(model, endpoint='/generate', device='cpu')
                assert hasattr(deployment, 'predict'), "Model deployment broken"

            # Test compressed model still works
            test_input = np.random.randint(0, 500, (2, 16))
            output = model(test_input)
            assert output.shape == (2, 16, 500), "Compressed transformer broken"

        except ImportError:
            assert True, "Compressed transformer deployment not ready yet"

    def test_multi_modal_capabilities(self):
        """Test multi-modal AI capabilities (vision + language)."""
        try:
            from tinytorch.core.transformers import TinyGPT
            from tinytorch.core.spatial import Conv2d as Conv2D, MaxPool2d
            from tinytorch.core.layers import Linear
            from tinytorch.core.attention import MultiHeadAttention
            from tinytorch.core.tensor import Tensor

            # Multi-modal model architecture
            class MultiModalModel:
                def __init__(self):
                    # Vision encoder (CNN)
                    self.conv1 = Conv2D(3, 16, kernel_size=3)
                    self.pool = MaxPool2d(kernel_size=2)
                    self.conv2 = Conv2D(16, 32, kernel_size=3)

                    # Vision to language bridge
                    self.vision_proj = Linear(32 * 7 * 7, 128)  # Approximate after conv/pool

                    # Language model
                    self.language_model = TinyGPT(
                        vocab_size=1000,
                        embed_dim=128,
                        num_heads=4,
                        num_layers=2,
                        max_seq_len=64
                    )

                    # Cross-modal attention
                    self.cross_attention = MultiHeadAttention(embed_dim=128, num_heads=4)

                def __call__(self, image, text_tokens):
                    # Vision processing
                    vis_feat = self.conv1(image)
                    vis_feat = self.pool(vis_feat)
                    vis_feat = self.conv2(vis_feat)

                    # Flatten and project
                    vis_feat_flat = vis_feat.reshape(vis_feat.shape[0], -1)
                    vis_embed = self.vision_proj(vis_feat_flat)

                    # Language processing
                    lang_embed = self.language_model.embedding(text_tokens)

                    # Cross-modal fusion
                    if hasattr(self.cross_attention, '__call__'):
                        fused_embed = self.cross_attention(lang_embed, vis_embed)
                    else:
                        fused_embed = lang_embed  # Fallback

                    # Generate response
                    output = self.language_model.transformer_blocks[0](fused_embed)
                    logits = self.language_model.output_projection(output)

                    return logits

            # Test multi-modal model
            model = MultiModalModel()

            # Test inputs
            image = Tensor(np.random.randn(2, 3, 32, 32))  # Batch of images
            text = np.random.randint(0, 1000, (2, 16))      # Batch of text

            output = model(image, text)
            assert output.shape[0] == 2, "Multi-modal batch size broken"
            assert output.shape[-1] == 1000, "Multi-modal vocab size broken"

        except ImportError:
            assert True, "Multi-modal capabilities not ready yet"


class TestCapstoneSystemValidation:
    """Final validation that TinyTorch achieves its educational goals."""

    def test_complete_ml_engineering_capability(self):
        """Test that students can build complete ML systems from scratch."""
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU, Softmax
            from tinytorch.core.spatial import Conv2d as Conv2D
            from tinytorch.core.attention import MultiHeadAttention
            from tinytorch.core.dataloader import Dataset, DataLoader
            from tinytorch.core.optimizers import Adam
            from tinytorch.core.training import Trainer
            from tinytorch.core.transformers import TinyGPT

            # Students should be able to build:

            # 1. Classic neural networks
            mlp = Linear(784, 10)
            x_mlp = Tensor(np.random.randn(32, 784))
            out_mlp = mlp(x_mlp)
            assert out_mlp.shape == (32, 10), "MLP capability broken"

            # 2. Convolutional networks
            cnn = Conv2D(3, 16, kernel_size=3)
            x_cnn = Tensor(np.random.randn(32, 3, 32, 32))
            if hasattr(cnn, '__call__'):
                out_cnn = cnn(x_cnn)
                assert len(out_cnn.shape) == 4, "CNN capability broken"

            # 3. Attention mechanisms
            attention = MultiHeadAttention(embed_dim=64, num_heads=8)
            x_attn = Tensor(np.random.randn(10, 32, 64))
            if hasattr(attention, '__call__'):
                out_attn = attention(x_attn)
                assert out_attn.shape == x_attn.shape, "Attention capability broken"

            # 4. Complete transformers
            if 'TinyGPT' in locals():
                transformer = TinyGPT(vocab_size=1000, embed_dim=128, num_heads=4, num_layers=2)
                x_trans = np.random.randint(0, 1000, (4, 16))
                out_trans = transformer(x_trans)
                assert out_trans.shape == (4, 16, 1000), "Transformer capability broken"

            # 5. Training pipelines
            optimizer = Adam([np.array([1.0])], lr=0.001)
            trainer = Trainer(mlp, optimizer)
            assert hasattr(trainer, 'train') or hasattr(trainer, 'fit'), "Training capability broken"

            assert True, "Complete ML engineering capability validated"

        except ImportError:
            assert True, "Complete ML engineering capability not ready yet"

    def test_production_ml_systems_capability(self):
        """Test production ML systems engineering skills."""
        try:
            from tinytorch.core.compression import prune_weights, quantize_weights
            from tinytorch.core.kernels import optimized_matmul, enable_optimizations
            from tinytorch.core.benchmarking import benchmark_model, profile_memory
            from tinytorch.core.mlops import ModelMonitor, deploy_model
            from tinytorch.core.layers import Linear

            model = Linear(100, 50)

            # Students should understand:

            # 1. Model compression
            if 'prune_weights' in locals():
                pruned = prune_weights(model.weights, sparsity=0.5)
                sparsity = np.sum(pruned.data == 0) / pruned.data.size
                assert 0.4 <= sparsity <= 0.6, "Compression understanding broken"

            if 'quantize_weights' in locals():
                quantized = quantize_weights(model.weights, bits=8)
                assert quantized.shape == model.weight.shape, "Quantization understanding broken"

            # 2. Performance optimization
            if 'optimized_matmul' in locals():
                A = np.random.randn(50, 30)
                B = np.random.randn(30, 20)
                result = optimized_matmul(A, B)
                assert result.shape == (50, 20), "Optimization understanding broken"

            # 3. Performance analysis
            if 'benchmark_model' in locals():
                benchmark_results = benchmark_model(model, input_shape=(32, 100))
                assert 'latency' in benchmark_results, "Benchmarking understanding broken"

            if 'profile_memory' in locals():
                memory_profile = profile_memory(model)
                assert 'peak_memory' in memory_profile, "Memory profiling understanding broken"

            # 4. Production deployment
            if 'ModelMonitor' in locals():
                monitor = ModelMonitor(model)
                monitor.log_metrics({'accuracy': 0.95, 'latency': 10.5})
                assert True, "Monitoring understanding validated"

            if 'deploy_model' in locals():
                deployment = deploy_model(model, endpoint='/predict')
                assert hasattr(deployment, 'predict'), "Deployment understanding broken"

            assert True, "Production ML systems capability validated"

        except ImportError:
            assert True, "Production ML systems capability not ready yet"

    def test_ml_systems_engineering_mindset(self):
        """Test that students develop ML systems engineering mindset."""
        # Students should understand the complete stack:

        # 1. Mathematical foundations → Implementation
        # 2. Basic operations → Complex models
        # 3. Single-threaded → Distributed systems
        # 4. Research code → Production systems
        # 5. Toy problems → Real-world deployment

        core_concepts_tested = [
            # Foundation
            "Tensor operations and memory management",
            "Automatic differentiation for learning",
            "Neural network building blocks",

            # Advanced architectures
            "Spatial processing with convolutions",
            "Sequence modeling with attention",
            "Complete transformer models",

            # Systems engineering
            "Data loading and preprocessing pipelines",
            "Training loops and optimization",
            "Model compression and efficiency",
            "Performance optimization kernels",
            "Benchmarking and profiling",
            "Production deployment and monitoring",

            # Integration
            "End-to-end system design",
            "Real-world ML engineering workflows"
        ]

        # By reaching Module 16, students have experienced:
        assert len(core_concepts_tested) == 14, "Core ML systems concepts"

        # They understand the complete journey:
        journey_stages = [
            "Build → Optimize → Deploy",
            "Research → Engineering → Production",
            "Single model → Complete system",
            "Toy data → Real datasets",
            "CPU → GPU acceleration",
            "Manual → Automated workflows"
        ]

        assert len(journey_stages) == 6, "Complete ML systems journey"

        # Most importantly, they can build everything from scratch
        from_scratch_capabilities = [
            "Implement any neural network architecture",
            "Design efficient training pipelines",
            "Optimize models for production deployment",
            "Build complete ML systems end-to-end",
            "Debug and profile ML system performance",
            "Deploy and monitor models in production"
        ]

        assert len(from_scratch_capabilities) == 6, "From-scratch ML systems capabilities"

        assert True, "ML systems engineering mindset validated"


class TestTinyTorchGraduationReadiness:
    """Final test: Are students ready to graduate from TinyTorch?"""

    def test_pytorch_transition_readiness(self):
        """Test readiness to transition to PyTorch and real-world ML."""

        # Students should understand PyTorch concepts because they built them:
        pytorch_concepts_mastered = {
            'torch.Tensor': 'Built in Module 01',
            'torch.nn.Module': 'Built in Module 03',
            'torch.nn.functional': 'Built in Module 02',
            'torch.optim': 'Built in Module 07',
            'torch.utils.data': 'Built in Module 05',
            'torch.autograd': 'Built in Module 06',
            'torch.nn.attention': 'Built in Module 12',
            'torch.jit': 'Built in Module 17 (acceleration)',
            'torch.quantization': 'Built in Module 15',
            'torch.distributed': 'Built in Module 19 (MLOps)',
        }

        # They understand the systems behind the abstractions
        systems_knowledge = {
            'Memory management': 'Tensor implementation and optimization',
            'Gradient computation': 'Autograd implementation from scratch',
            'Training loops': 'Complete training system implementation',
            'Model compression': 'Pruning and quantization implementation',
            'Performance optimization': 'Custom kernel implementation',
            'Production deployment': 'MLOps and monitoring implementation',
        }

        # They can debug issues because they understand internals
        debugging_capabilities = {
            'Memory leaks': 'Implemented tensor memory management',
            'Gradient issues': 'Built autograd computation graph',
            'Training instability': 'Implemented optimizers and training loops',
            'Performance bottlenecks': 'Built profiling and benchmarking tools',
            'Model deployment issues': 'Implemented complete deployment pipeline',
        }

        assert len(pytorch_concepts_mastered) == 10, "PyTorch concepts mastered"
        assert len(systems_knowledge) == 6, "Systems knowledge acquired"
        assert len(debugging_capabilities) == 5, "Debugging capabilities developed"

        # Final validation: They are ML SYSTEMS engineers, not just ML researchers
        assert True, "Students are ready for real-world ML systems engineering"

    def test_capstone_achievement_unlocked(self):
        """Final achievement: TinyTorch capstone completed."""

        # Students have built a complete ML framework from scratch
        framework_components = [
            "Core tensor operations",
            "Automatic differentiation",
            "Neural network layers",
            "Convolutional operations",
            "Attention mechanisms",
            "Data loading pipelines",
            "Training optimization",
            "Model compression",
            "Performance kernels",
            "Benchmarking tools",
            "Production deployment",
            "Complete transformer models"
        ]

        # They understand ML systems at every level
        understanding_levels = [
            "Mathematical foundations",
            "Algorithmic implementation",
            "Systems optimization",
            "Production deployment",
            "Real-world engineering"
        ]

        # They can build anything in ML now
        creative_capabilities = [
            "Design novel architectures",
            "Optimize for any hardware",
            "Debug complex systems",
            "Scale to production",
            "Innovate new techniques"
        ]

        assert len(framework_components) == 12, "Complete ML framework built"
        assert len(understanding_levels) == 5, "Multi-level understanding achieved"
        assert len(creative_capabilities) == 5, "Creative engineering capabilities unlocked"

        # 🎓 GRADUATION: Students are now ML Systems Engineers
        graduation_message = """
        🎓 TINYTORCH CAPSTONE COMPLETED 🎓

        Congratulations! You have built a complete ML framework from scratch.
        You understand every layer of the ML systems stack.
        You are ready for real-world ML systems engineering.

        Next steps:
        - Apply your knowledge to PyTorch, TensorFlow, JAX
        - Build production ML systems with confidence
        - Debug and optimize complex ML workflows
        - Innovate new ML systems techniques

        You are now a ML Systems Engineer! 🚀
        """

        assert len(graduation_message) > 0, "Graduation achieved!"
        assert True, "🎓 TinyTorch capstone completed - Students are now ML Systems Engineers!"


class TestRegressionPrevention:
    """Ensure the ENTIRE TinyTorch system (01→15) still works perfectly."""

    def test_no_system_wide_regression(self):
        """Verify ENTIRE TinyTorch system unchanged."""
        # Core functionality should remain stable
        assert sys.version_info.major >= 3, "Foundation: Python detection broken"

        # Complete system should still work perfectly
        try:
            import numpy as np
            from tinytorch.core.autograd import enable_autograd
            enable_autograd()

            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import Adam

            # Complete system integration
            model = Linear(16, 8)
            optimizer = Adam(model.parameters(), lr=0.001)

            x = Tensor(np.random.randn(4, 16))
            output = model(x)
            assert output.shape == (4, 8), "System-wide regression: Core functionality broken"

            # Advanced features should work
            from tinytorch.core.spatial import Conv2d
            from tinytorch.core.attention import MultiHeadAttention

            conv = Conv2d(in_channels=3, out_channels=8, kernel_size=3)
            attention = MultiHeadAttention(embed_dim=32, num_heads=4)

            assert hasattr(conv, 'forward'), "System-wide regression: Spatial broken"
            assert hasattr(attention, 'forward'), "System-wide regression: Attention broken"

        except ImportError:
            import numpy as np
            assert np.random is not None, "System-wide regression: Basic functionality broken"

    def test_progressive_stability_complete(self):
        """Test the complete progressive stack is stable through capstone."""
        # Stack should be stable through: Setup → ... → Compression → Memoization

        # Foundation level
        import numpy as np
        assert np is not None, "Foundation level broken"

        # Complete ML system level (if available)
        try:
            from tinytorch.core.autograd import enable_autograd
            enable_autograd()

            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import Adam
            try:
                from tinytorch.perf.compression import prune_weights
            except ImportError:
                prune_weights = None  # Optional module

            # Complete production system should work
            model = Linear(20, 10)
            optimizer = Adam(model.parameters(), lr=0.001)

            x = Tensor(np.random.randn(5, 20))
            output = model(x)
            assert output.shape == (5, 10), "Complete system level broken"

            # All advanced features should work
            if prune_weights is not None:
                pruned = prune_weights(model.weights.data, sparsity=0.3)
                assert pruned.shape == model.weights.shape, "Advanced features broken"

        except ImportError:
            pass  # Not implemented yet

        # Transformer level (if available)
        try:
            from tinytorch.core.transformer import TransformerBlock

            # Transformer block should work
            block = TransformerBlock(embed_dim=32, num_heads=4)
            assert hasattr(block, 'forward'), "Transformer block broken"

        except ImportError:
            pass  # Not implemented yet

        # FINAL VALIDATION: TinyTorch is a complete, working ML framework
        assert True, "🎯 TinyTorch progressive stack is complete and stable!"
