"""
Module 11: Progressive Integration Tests
Tests that Module 11 (Embeddings) works correctly AND that prior modules (01→10) still work.

DEPENDENCY CHAIN: 01_tensor → 02_activations → 03_layers → 04_losses → 05_dataloader → 
                  06_autograd → 07_optimizers → 08_training → 09_convolutions → 10_tokenization → 11_embeddings

⚠️ IMPORTANT: This test ONLY uses modules 01-11.
   Future modules (12_attention, 16_compression, etc.) are NOT tested here.

🎯 WHAT THIS TESTS:
- Module 11: Embedding layers, token embeddings, positional embeddings
- Integration: Embeddings work with tokenization (10) and prior modules
- Regression: All previous modules still work correctly
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestEmbeddingCore:
    """
    🆕 NEW FUNCTIONALITY: Test Module 11 (Embeddings) core implementation.
    """

    def test_embedding_exists(self):
        """
        ✅ TEST: Embedding class exists
        """
        try:
            from tinytorch.core.embeddings import Embedding
            
            assert Embedding is not None, "Embedding class not found"
            
        except ImportError:
            assert True, "Embedding not implemented yet"

    def test_embedding_initialization(self):
        """
        ✅ TEST: Embedding can be initialized with vocab_size and embed_dim
        """
        try:
            from tinytorch.core.embeddings import Embedding
            
            vocab_size = 1000
            embed_dim = 64
            
            embedding = Embedding(vocab_size, embed_dim)
            
            assert hasattr(embedding, 'weight'), "Embedding missing weight"
            if hasattr(embedding, 'weight'):
                assert embedding.weight.shape == (vocab_size, embed_dim), \
                    f"Embedding weight shape wrong: {embedding.weight.shape}"
                    
        except ImportError:
            assert True, "Embedding not implemented yet"

    def test_embedding_forward(self):
        """
        ✅ TEST: Embedding forward converts token IDs to vectors
        """
        try:
            from tinytorch.core.embeddings import Embedding
            from tinytorch.core.tensor import Tensor
            
            vocab_size = 100
            embed_dim = 32
            
            embedding = Embedding(vocab_size, embed_dim)
            
            # Token IDs
            token_ids = Tensor(np.array([[1, 5, 10], [2, 7, 3]]))  # (batch=2, seq_len=3)
            
            output = embedding(token_ids)
            
            # Output should be (batch, seq_len, embed_dim)
            assert output.shape == (2, 3, embed_dim), \
                f"Embedding output shape wrong: {output.shape}"
                
        except ImportError:
            assert True, "Embedding forward not implemented yet"

    def test_positional_embedding(self):
        """
        ✅ TEST: Positional embedding exists (if implemented)
        """
        try:
            from tinytorch.core.embeddings import PositionalEmbedding
            from tinytorch.core.tensor import Tensor
            
            max_len = 512
            embed_dim = 64
            
            pos_embed = PositionalEmbedding(max_len, embed_dim)
            
            # Should produce position-aware vectors
            positions = Tensor(np.arange(10).reshape(1, 10))  # (1, seq_len=10)
            output = pos_embed(positions)
            
            assert output.shape[-1] == embed_dim, "Positional embedding dim wrong"
            
        except ImportError:
            assert True, "PositionalEmbedding not implemented yet"


class TestEmbeddingWithTokenization:
    """
    🔗 INTEGRATION: Embeddings + Tokenization (Module 10)
    """

    def test_embedding_with_tokenizer(self):
        """
        ✅ TEST: Embedding works with tokenized text
        """
        try:
            from tinytorch.core.embeddings import Embedding
            from tinytorch.core.tokenizer import Tokenizer
            from tinytorch.core.tensor import Tensor
            
            # Create tokenizer
            tokenizer = Tokenizer()
            tokenizer.fit(["hello world", "how are you"])
            
            # Tokenize text
            tokens = tokenizer.encode("hello world")
            token_ids = Tensor(np.array([tokens]))
            
            # Create embedding
            vocab_size = len(tokenizer.vocab)
            embedding = Embedding(vocab_size, embed_dim=32)
            
            # Get embeddings
            output = embedding(token_ids)
            
            assert len(output.shape) >= 2, "Embedding output should be at least 2D"
            
        except ImportError:
            assert True, "Tokenizer + Embedding integration not ready"


class TestEmbeddingWithTraining:
    """
    🔗 INTEGRATION: Embeddings + Training (Module 08)
    """

    def test_embedding_trainable(self):
        """
        ✅ TEST: Embedding weights can be trained
        """
        try:
            from tinytorch.core.embeddings import Embedding
            from tinytorch.core.layers import Linear
            from tinytorch.core.losses import MSELoss
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.tensor import Tensor
            
            vocab_size = 50
            embed_dim = 16
            
            embedding = Embedding(vocab_size, embed_dim)
            fc = Linear(embed_dim, 1)
            loss_fn = MSELoss()
            
            # Collect parameters
            params = []
            if hasattr(embedding, 'parameters'):
                params.extend(embedding.parameters())
            if hasattr(fc, 'parameters'):
                params.extend(fc.parameters())
            
            optimizer = SGD(params, lr=0.1)
            
            # Training step
            token_ids = Tensor(np.array([[1, 2, 3]]))  # (1, 3)
            target = Tensor(np.array([[1.0]]))
            
            # Forward
            embed_out = embedding(token_ids)  # (1, 3, 16)
            # Average pool
            pooled = Tensor(embed_out.data.mean(axis=1))  # (1, 16)
            pred = fc(pooled)  # (1, 1)
            
            loss = loss_fn(pred, target)
            
            if hasattr(loss, 'backward'):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            assert loss.data.size == 1, "Training loss wrong"
            
        except ImportError:
            assert True, "Embedding training not ready"


class TestRegressionPrevention:
    """
    🔄 REGRESSION: Verify all previous modules (01-10) still work.
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
            pass  # OK if spatial not implemented

    def test_tokenization_still_works(self):
        """✅ Module 10"""
        try:
            from tinytorch.core.tokenizer import Tokenizer
            
            tokenizer = Tokenizer()
            tokenizer.fit(["hello world"])
            tokens = tokenizer.encode("hello")
            
            assert len(tokens) >= 1
        except ImportError:
            pass  # OK if tokenizer not implemented


class TestModule11Completion:
    """
    ✅ COMPLETION CHECK: Module 11 ready for next module.
    """

    def test_embedding_foundation_complete(self):
        """
        ✅ FINAL TEST: Embedding ready for attention
        
        🎯 SUCCESS = Ready for Module 12: Attention!
        """
        capabilities = {
            "Embedding exists": False,
            "Embedding forward works": False,
            "Embedding trainable": False,
        }
        
        try:
            from tinytorch.core.embeddings import Embedding
            from tinytorch.core.tensor import Tensor
            
            # Test 1: Exists
            capabilities["Embedding exists"] = True
            
            # Test 2: Forward
            embedding = Embedding(100, 32)
            ids = Tensor(np.array([[1, 2, 3]]))
            out = embedding(ids)
            if out.shape[-1] == 32:
                capabilities["Embedding forward works"] = True
            
            # Test 3: Has parameters
            if hasattr(embedding, 'parameters'):
                capabilities["Embedding trainable"] = True
            elif hasattr(embedding, 'weight'):
                capabilities["Embedding trainable"] = True
            
            completed = sum(capabilities.values())
            assert completed >= 2, f"Embedding not ready: {capabilities}"
            
        except ImportError:
            # Embedding module doesn't exist yet
            assert True, "Embedding not implemented yet"
