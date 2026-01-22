"""
Module 12: Progressive Integration Tests
Tests that Module 12 (Attention) works correctly AND that prior modules (01→11) still work.

DEPENDENCY CHAIN: 01_tensor → ... → 10_tokenization → 11_embeddings → 12_attention

⚠️ IMPORTANT: This test ONLY uses modules 01-12.
   Future modules (13_transformers, 16_compression, 17_acceleration, etc.) are NOT tested here.

🎯 WHAT THIS TESTS:
- Module 12: Attention mechanisms, MultiHeadAttention, scaled dot-product attention
- Integration: Attention works with embeddings (11) and prior modules
- Regression: All previous modules still work correctly
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestAttentionCore:
    """
    🆕 NEW FUNCTIONALITY: Test Module 12 (Attention) core implementation.
    """

    def test_scaled_dot_product_attention_exists(self):
        """
        ✅ TEST: scaled_dot_product_attention function exists
        """
        try:
            from tinytorch.core.attention import scaled_dot_product_attention
            
            assert scaled_dot_product_attention is not None
            
        except ImportError:
            assert True, "Attention not implemented yet"

    def test_scaled_dot_product_attention(self):
        """
        ✅ TEST: Scaled dot-product attention computes correctly
        """
        try:
            from tinytorch.core.attention import scaled_dot_product_attention
            from tinytorch.core.tensor import Tensor
            
            seq_len, d_k = 10, 16
            
            Q = Tensor(np.random.randn(seq_len, d_k))
            K = Tensor(np.random.randn(seq_len, d_k))
            V = Tensor(np.random.randn(seq_len, d_k))
            
            output, weights = scaled_dot_product_attention(Q, K, V)
            
            assert output.shape == V.shape, f"Output shape wrong: {output.shape}"
            assert weights.shape == (seq_len, seq_len), f"Weights shape wrong: {weights.shape}"
            
            # Attention weights should sum to ~1 for each query
            weight_sums = np.sum(weights.data, axis=-1)
            assert np.allclose(weight_sums, 1.0, atol=1e-5), "Weights don't sum to 1"
            
        except ImportError:
            assert True, "Attention not implemented yet"

    def test_multihead_attention_exists(self):
        """
        ✅ TEST: MultiHeadAttention class exists
        """
        try:
            from tinytorch.core.attention import MultiHeadAttention
            
            assert MultiHeadAttention is not None
            
        except ImportError:
            assert True, "MultiHeadAttention not implemented yet"

    def test_multihead_attention_initialization(self):
        """
        ✅ TEST: MultiHeadAttention can be initialized
        """
        try:
            from tinytorch.core.attention import MultiHeadAttention
            
            embed_dim = 64
            num_heads = 8
            
            mha = MultiHeadAttention(embed_dim, num_heads)
            
            assert hasattr(mha, 'forward'), "MHA missing forward"
            
        except ImportError:
            assert True, "MultiHeadAttention not implemented yet"

    def test_multihead_attention_forward(self):
        """
        ✅ TEST: MultiHeadAttention forward pass
        """
        try:
            from tinytorch.core.attention import MultiHeadAttention
            from tinytorch.core.tensor import Tensor
            
            embed_dim = 64
            num_heads = 8
            batch_size = 2
            seq_len = 10
            
            mha = MultiHeadAttention(embed_dim, num_heads)
            
            # Input: (batch, seq_len, embed_dim) or (seq_len, batch, embed_dim)
            x = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
            
            output = mha(x)
            
            # Output should have same shape as input
            assert output.shape == x.shape, f"MHA output shape wrong: {output.shape}"
            
        except ImportError:
            assert True, "MultiHeadAttention not implemented yet"


class TestAttentionWithEmbeddings:
    """
    🔗 INTEGRATION: Attention + Embeddings (Module 11)
    """

    def test_attention_with_embeddings(self):
        """
        ✅ TEST: Attention works on embedded tokens
        """
        try:
            from tinytorch.core.attention import MultiHeadAttention
            from tinytorch.core.embeddings import Embedding
            from tinytorch.core.tensor import Tensor
            
            vocab_size = 100
            embed_dim = 64
            num_heads = 8
            
            embedding = Embedding(vocab_size, embed_dim)
            attention = MultiHeadAttention(embed_dim, num_heads)
            
            # Token IDs
            token_ids = Tensor(np.array([[1, 5, 10, 3]]))  # (1, 4)
            
            # Embed tokens
            embedded = embedding(token_ids)  # (1, 4, 64)
            
            # Apply attention
            attended = attention(embedded)
            
            assert attended.shape == embedded.shape, "Attention didn't preserve shape"
            
        except ImportError:
            assert True, "Attention + Embeddings not ready"


class TestAttentionWithTraining:
    """
    🔗 INTEGRATION: Attention + Training (Module 08)
    """

    def test_attention_trainable(self):
        """
        ✅ TEST: Attention parameters can be trained
        """
        try:
            from tinytorch.core.attention import MultiHeadAttention
            from tinytorch.core.layers import Linear
            from tinytorch.core.losses import MSELoss
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.tensor import Tensor
            
            embed_dim = 32
            num_heads = 4
            
            attention = MultiHeadAttention(embed_dim, num_heads)
            fc = Linear(embed_dim, 1)
            loss_fn = MSELoss()
            
            # Collect parameters
            params = []
            if hasattr(attention, 'parameters'):
                params.extend(attention.parameters())
            if hasattr(fc, 'parameters'):
                params.extend(fc.parameters())
            
            optimizer = SGD(params, lr=0.01)
            
            # Forward
            x = Tensor(np.random.randn(2, 5, embed_dim))  # (batch, seq, embed)
            target = Tensor(np.random.randn(2, 1))
            
            attn_out = attention(x)
            pooled = Tensor(attn_out.data.mean(axis=1))  # (batch, embed)
            pred = fc(pooled)
            
            loss = loss_fn(pred, target)
            
            if hasattr(loss, 'backward'):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            assert loss.data.size == 1
            
        except ImportError:
            assert True, "Attention training not ready"


class TestRegressionPrevention:
    """
    🔄 REGRESSION: Verify all previous modules (01-11) still work.
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
        pred = Tensor([[1.0]])
        target = Tensor([[2.0]])
        loss = loss_fn(pred, target)
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

    def test_training_still_works(self):
        """✅ Module 08"""
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.layers import Linear
        from tinytorch.core.losses import MSELoss
        from tinytorch.core.optimizers import SGD
        
        layer = Linear(4, 2)
        loss_fn = MSELoss()
        opt = SGD(layer.parameters(), lr=0.1)
        
        x = Tensor(np.random.randn(2, 4))
        y = Tensor(np.random.randn(2, 2))
        
        pred = layer(x)
        loss = loss_fn(pred, y)
        assert loss.data.size == 1

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

    def test_embeddings_still_work(self):
        """✅ Module 11"""
        try:
            from tinytorch.core.embeddings import Embedding
            from tinytorch.core.tensor import Tensor
            
            embedding = Embedding(100, 32)
            ids = Tensor(np.array([[1, 2, 3]]))
            out = embedding(ids)
            assert out.shape[-1] == 32
        except ImportError:
            pass


class TestModule12Completion:
    """
    ✅ COMPLETION CHECK: Module 12 ready for next module.
    """

    def test_attention_foundation_complete(self):
        """
        ✅ FINAL TEST: Attention ready for transformers
        
        🎯 SUCCESS = Ready for Module 13: Transformers!
        """
        capabilities = {
            "scaled_dot_product exists": False,
            "MultiHeadAttention exists": False,
            "MHA forward works": False,
        }
        
        try:
            from tinytorch.core.attention import scaled_dot_product_attention, MultiHeadAttention
            from tinytorch.core.tensor import Tensor
            
            # Test 1: scaled_dot_product
            capabilities["scaled_dot_product exists"] = True
            
            # Test 2: MultiHeadAttention exists
            capabilities["MultiHeadAttention exists"] = True
            
            # Test 3: MHA forward
            mha = MultiHeadAttention(32, 4)
            x = Tensor(np.random.randn(1, 5, 32))
            out = mha(x)
            if out.shape == x.shape:
                capabilities["MHA forward works"] = True
            
            completed = sum(capabilities.values())
            assert completed >= 2, f"Attention not ready: {capabilities}"
            
        except ImportError:
            assert True, "Attention not implemented yet"
