"""Embedding training through the existing tokenizer, loader, loss, and optimizer."""
import numpy as np

from tinytorch.core.tensor import Tensor
import tinytorch.core.autograd
from tinytorch.core.tokenization import CharTokenizer, tokenize_dataset
from tinytorch.core.dataloader import TensorDataset, DataLoader
from tinytorch.core.embeddings import Embedding
from tinytorch.core.layers import Linear
from tinytorch.core.losses import MSELoss
from tinytorch.core.optimizers import SGD


def test_tokenizer_to_embedding_training_updates_only_used_rows():
    tokenizer = CharTokenizer(list('abcd'))
    ids = Tensor(tokenize_dataset(['ab', 'ba'], tokenizer))
    loader = DataLoader(TensorDataset(ids, Tensor([[1.0], [0.0]])), batch_size=2)
    embedding = Embedding(tokenizer.vocab_size, 3)
    classifier = Linear(3, 1)
    classifier.weight.data[:] = 0.25
    classifier.bias.data[:] = 0
    params = embedding.parameters() + classifier.parameters()
    optimizer = SGD(params, lr=0.05)
    before = embedding.weight.data.copy()
    for tokens, targets in loader:
        optimizer.zero_grad()
        loss = MSELoss()(classifier(embedding(tokens).mean(axis=1)), targets)
        loss.backward()
        assert all(p.grad is not None for p in params)
        optimizer.step()
    assert not np.array_equal(before[1:3], embedding.weight.data[1:3])
    np.testing.assert_array_equal(before[[0, 3, 4]], embedding.weight.data[[0, 3, 4]])
