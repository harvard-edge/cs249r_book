"""Module 10: text becomes token IDs that compose with Modules 01–09.

These tests deliberately stop before embeddings (Module 11). Missing imports
are failures: this suite validates the completed instructor implementation.
"""
import numpy as np

from tinytorch.core.tensor import Tensor
from tinytorch.core.dataloader import DataLoader, TensorDataset
from tinytorch.core.tokenization import CharTokenizer, BPETokenizer, tokenize_dataset


def test_character_tokens_round_trip_through_batches():
    tokenizer = CharTokenizer()
    texts = ["hello", "world", "there"]
    tokenizer.build_vocab(texts)
    token_ids = Tensor(tokenize_dataset(texts, tokenizer))
    dataset = TensorDataset(token_ids, Tensor([0, 1, 2]))
    batches = list(DataLoader(dataset, batch_size=2, shuffle=False))
    assert [x.shape[0] for x, _ in batches] == [2, 1]
    decoded = [tokenizer.decode(row.tolist()) for x, _ in batches for row in x.data]
    assert decoded == texts


def test_bpe_learns_merges_and_preserves_known_words():
    tokenizer = BPETokenizer(vocab_size=20)
    tokenizer.train(["low lower low lowest"])
    assert tokenizer.merges
    text = "low lower"
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert all(0 <= token < len(tokenizer.vocab) for token in ids)
    # This teaching tokenizer normalizes whitespace rather than preserving bytes.
    assert tokenizer.decode(tokenizer.encode("  low\t lower  ")) == text


def test_unknown_character_does_not_change_known_ids():
    tokenizer = CharTokenizer(['a', 'b'])
    before = tokenizer.encode('ab')
    assert tokenizer.encode('a?b') == [before[0], tokenizer.unk_id, before[1]]
    assert tokenizer.encode('ab') == before


def test_tokenization_truncates_each_sequence():
    tokenizer = CharTokenizer(list('abcd'))
    assert tokenize_dataset(['abcd', 'ab'], tokenizer, max_length=2) == [[1, 2], [1, 2]]


def test_zero_length_truncation_returns_empty_sequences():
    assert tokenize_dataset(['abc'], CharTokenizer(list('abc')), max_length=0) == [[]]
