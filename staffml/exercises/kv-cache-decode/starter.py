"""KV-cache decode: starter.

Implement the four TODOs, then run

    STAFFML_IMPL=starter pytest staffml/exercises/kv-cache-decode

See README.md for the task and the napkin math.
"""
import numpy as np


class KVCache:
    """Pre-allocated key/value cache for one attention head."""

    def __init__(self, max_len: int, d_head: int, dtype=np.float64):
        self.keys = np.zeros((max_len, d_head), dtype=dtype)
        self.values = np.zeros((max_len, d_head), dtype=dtype)
        self.length = 0

    def append(self, k: np.ndarray, v: np.ndarray) -> None:
        # TODO 1: write k and v into the next free row (no reallocation), raise
        # ValueError when the cache is full, and advance self.length.
        raise NotImplementedError

    def view(self):
        """Keys and values of the tokens cached so far, shape (length, d_head)."""
        # TODO 2: return the filled rows of self.keys and self.values.
        raise NotImplementedError


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    return e / e.sum(axis=-1, keepdims=True)


def decode_step(x_t, w_q, w_k, w_v, cache: KVCache, project=np.matmul):
    """Attend from token ``x_t`` (shape (d_model,)) to itself and every cached token.

    ``project(x, w)`` computes ``x @ w``; the test passes a counting version, so use
    it for every projection.  Returns the attention output, shape (d_head,).
    """
    # TODO 3: project only the new token, append its key and value to the cache,
    # then compute softmax(K q / sqrt(d_head)) V over the cached tokens.
    raise NotImplementedError


def full_causal_attention(x, w_q, w_k, w_v, project=np.matmul):
    """Recompute every output from scratch: the baseline decode_step must match."""
    q, k, v = project(x, w_q), project(x, w_k), project(x, w_v)
    scores = q @ k.T / np.sqrt(q.shape[-1])
    scores = np.where(np.tril(np.ones(scores.shape, dtype=bool)), scores, -np.inf)
    return softmax(scores) @ v


def kv_cache_bytes(n_layers: int, n_kv_heads: int, d_head: int, seq_len: int,
                   bytes_per_elem: int, batch: int = 1) -> int:
    """Bytes held by the KV cache: keys and values for every layer, head and token."""
    # TODO 4: the napkin-math formula.
    raise NotImplementedError
