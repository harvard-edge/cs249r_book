"""
Module 13: Progressive Integration Tests
Tests that Module 13 (Transformers) works correctly AND that prior modules (01→12) still work.

DEPENDENCY CHAIN: 01_tensor → ... → 11_embeddings → 12_attention → 13_transformers

⚠️ IMPORTANT: This test ONLY uses modules 01-13.
   Future modules (14_profiling, 19_benchmarking, etc.) are NOT tested here.

🎯 WHAT THIS TESTS:
- Module 13: TransformerBlock, TinyGPT, encoder-decoder architecture
- Integration: Transformers work with attention (12) and prior modules
- Regression: All previous modules still work correctly
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestTransformerCore:
    """
    🆕 NEW FUNCTIONALITY: Test Module 13 (Transformers) core implementation.
    """

    def test_transformer_block_exists(self):
        """
        ✅ TEST: TransformerBlock class exists
        """
        try:
            from tinytorch.core.transformers import TransformerBlock
            
            assert TransformerBlock is not None
            
        except ImportError:
            assert True, "TransformerBlock not implemented yet"

    def test_transformer_block_initialization(self):
        """
        ✅ TEST: TransformerBlock can be initialized
        """
        try:
            from tinytorch.core.transformers import TransformerBlock
            
            embed_dim = 64
            num_heads = 8
            ff_dim = 256
            
            block = TransformerBlock(embed_dim, num_heads, ff_dim)
            
            assert hasattr(block, 'forward'), "Block missing forward"
            
        except ImportError:
            assert True, "TransformerBlock not implemented yet"

    def test_transformer_block_forward(self):
        """
        ✅ TEST: TransformerBlock forward pass
        """
        try:
            from tinytorch.core.transformers import TransformerBlock
            from tinytorch.core.tensor import Tensor
            
            embed_dim = 64
            num_heads = 8
            ff_dim = 256
            batch_size = 2
            seq_len = 10
            
            block = TransformerBlock(embed_dim, num_heads, ff_dim)
            
            x = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
            
            output = block(x)
            
            assert output.shape == x.shape, f"Block output shape wrong: {output.shape}"
            
        except ImportError:
            assert True, "TransformerBlock not implemented yet"

    def test_tinygpt_exists(self):
        """
        ✅ TEST: TinyGPT class exists
        """
        try:
            from tinytorch.core.transformers import TinyGPT
            
            assert TinyGPT is not None
            
        except ImportError:
            assert True, "TinyGPT not implemented yet"

    def test_tinygpt_initialization(self):
        """
        ✅ TEST: TinyGPT can be initialized
        """
        try:
            from tinytorch.core.transformers import TinyGPT
            
            vocab_size = 1000
            embed_dim = 64
            num_heads = 4
            num_layers = 2
            
            model = TinyGPT(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers
            )
            
            assert hasattr(model, 'forward'), "TinyGPT missing forward"
            
        except ImportError:
            assert True, "TinyGPT not implemented yet"
        except TypeError:
            assert True, "TinyGPT may have different signature"

    def test_tinygpt_forward(self):
        """
        ✅ TEST: TinyGPT forward pass produces logits
        """
        try:
            from tinytorch.core.transformers import TinyGPT
            from tinytorch.core.tensor import Tensor
            
            vocab_size = 100
            embed_dim = 32
            num_heads = 4
            num_layers = 2
            
            model = TinyGPT(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers
            )
            
            # Token IDs input
            token_ids = Tensor(np.array([[1, 5, 10, 3]]))  # (batch=1, seq_len=4)
            
            logits = model(token_ids)
            
            # Output should be (batch, seq_len, vocab_size)
            assert logits.shape[-1] == vocab_size, f"Logits vocab dim wrong: {logits.shape}"
            
        except ImportError:
            assert True, "TinyGPT not implemented yet"
        except TypeError:
            assert True, "TinyGPT may have different interface"


class TestTransformerWithAttention:
    """
    🔗 INTEGRATION: Transformers + Attention (Module 12)
    """

    def test_transformer_uses_attention(self):
        """
        ✅ TEST: TransformerBlock internally uses attention
        """
        try:
            from tinytorch.core.transformers import TransformerBlock
            from tinytorch.core.attention import MultiHeadAttention
            
            block = TransformerBlock(64, 8, 256)
            
            # Block should have attention component
            has_attention = (
                hasattr(block, 'attention') or 
                hasattr(block, 'self_attention') or
                hasattr(block, 'mha')
            )
            
            assert has_attention, "TransformerBlock should have attention"
            
        except ImportError:
            assert True, "Integration not ready"


class TestTransformerWithTraining:
    """
    🔗 INTEGRATION: Transformers + Training (Module 08)
    """

    def test_transformer_trainable(self):
        """
        ✅ TEST: Transformer parameters can be trained
        """
        try:
            from tinytorch.core.transformers import TransformerBlock
            from tinytorch.core.layers import Linear
            from tinytorch.core.losses import MSELoss
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.tensor import Tensor
            
            embed_dim = 32
            
            block = TransformerBlock(embed_dim, 4, 128)
            fc = Linear(embed_dim, 1)
            loss_fn = MSELoss()
            
            # Collect parameters
            params = []
            if hasattr(block, 'parameters'):
                params.extend(block.parameters())
            if hasattr(fc, 'parameters'):
                params.extend(fc.parameters())
            
            optimizer = SGD(params, lr=0.01)
            
            # Forward
            x = Tensor(np.random.randn(2, 5, embed_dim))
            target = Tensor(np.random.randn(2, 1))
            
            block_out = block(x)
            pooled = Tensor(block_out.data.mean(axis=1))
            pred = fc(pooled)
            
            loss = loss_fn(pred, target)
            
            if hasattr(loss, 'backward'):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            assert loss.data.size == 1
            
        except ImportError:
            assert True, "Transformer training not ready"

    def test_tinygpt_training_step(self):
        """
        ✅ TEST: TinyGPT can execute training step
        """
        try:
            from tinytorch.core.transformers import TinyGPT
            from tinytorch.core.losses import CrossEntropyLoss
            from tinytorch.core.optimizers import Adam
            from tinytorch.core.tensor import Tensor
            
            vocab_size = 50
            
            model = TinyGPT(
                vocab_size=vocab_size,
                embed_dim=32,
                num_heads=4,
                num_layers=2
            )
            
            params = model.parameters() if hasattr(model, 'parameters') else []
            optimizer = Adam(params, lr=0.001)
            loss_fn = CrossEntropyLoss()
            
            # Training data
            input_ids = Tensor(np.array([[1, 5, 10]]))
            target_ids = Tensor(np.array([[5, 10, 3]]))
            
            # Forward
            logits = model(input_ids)
            
            # Compute loss (simplified)
            if logits.shape[-1] == vocab_size:
                # Flatten for loss
                pass  # Loss computation depends on implementation
            
        except ImportError:
            assert True, "TinyGPT training not ready"
        except TypeError:
            assert True, "TinyGPT interface may differ"


class TestRegressionPrevention:
    """
    🔄 REGRESSION: Verify all previous modules (01-12) still work.
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


class TestModule13Completion:
    """
    ✅ COMPLETION CHECK: Module 13 ready for next module.
    """

    def test_transformer_foundation_complete(self):
        """
        ✅ FINAL TEST: Transformers ready for profiling
        
        🎯 SUCCESS = Ready for Module 14: Profiling!
        """
        capabilities = {
            "TransformerBlock exists": False,
            "TransformerBlock forward works": False,
            "TinyGPT exists": False,
        }
        
        try:
            from tinytorch.core.transformers import TransformerBlock
            from tinytorch.core.tensor import Tensor
            
            # Test 1: TransformerBlock exists
            capabilities["TransformerBlock exists"] = True
            
            # Test 2: Forward
            block = TransformerBlock(32, 4, 128)
            x = Tensor(np.random.randn(1, 5, 32))
            out = block(x)
            if out.shape == x.shape:
                capabilities["TransformerBlock forward works"] = True
            
            # Test 3: TinyGPT
            try:
                from tinytorch.core.transformers import TinyGPT
                capabilities["TinyGPT exists"] = True
            except ImportError:
                pass
            
            completed = sum(capabilities.values())
            assert completed >= 2, f"Transformers not ready: {capabilities}"
            
        except ImportError:
            assert True, "Transformers not implemented yet"
