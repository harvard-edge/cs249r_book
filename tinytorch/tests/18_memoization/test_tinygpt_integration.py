"""The current GPT API composes with cache enable/disable and generation."""
import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.transformers import GPT
from tinytorch.perf.memoization import enable_kv_cache, disable_kv_cache, _cached_generate


def test_gpt_cached_generation_and_restore():
    model = GPT(vocab_size=12, embed_dim=16, num_heads=2, num_layers=2, max_seq_len=8)
    prompt = Tensor([[1, 2, 3]])
    baseline = model.generate(prompt, max_new_tokens=3, temperature=0).data[0, -3:]
    cache = enable_kv_cache(model)
    generated = _cached_generate(model, [1, 2, 3], 3, 0, cache)
    np.testing.assert_array_equal(generated, baseline)
    # Every request starts afresh, including its position embeddings.
    np.testing.assert_array_equal(_cached_generate(model, [1, 2, 3], 3, 0, cache), baseline)
    disable_kv_cache(model)
    np.testing.assert_array_equal(model.generate(prompt, 3, temperature=0).data[0, -3:], baseline)
