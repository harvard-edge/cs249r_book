"""KV-cache decode: reference solution.

Single-head causal attention, decoded one token at a time.  Each step projects
only the new token and reads the cached keys and values of earlier tokens, so the
work per step is O(t * d) instead of recomputing all t projections and scores.
"""
import numpy as np


class KVCache:
    """Pre-allocated key/value cache for one attention head."""

    def __init__(self, max_len: int, d_head: int, dtype=np.float64):
        self.keys = np.zeros((max_len, d_head), dtype=dtype)
        self.values = np.zeros((max_len, d_head), dtype=dtype)
        self.length = 0

    def append(self, k: np.ndarray, v: np.ndarray) -> None:
        if self.length == self.keys.shape[0]:
            raise ValueError("KV cache is full")
        self.keys[self.length] = k
        self.values[self.length] = v
        self.length += 1

    def view(self):
        """Keys and values of the tokens cached so far, shape (length, d_head)."""
        return self.keys[: self.length], self.values[: self.length]


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    return e / e.sum(axis=-1, keepdims=True)


def decode_step(x_t, w_q, w_k, w_v, cache: KVCache, project=np.matmul):
    """Attend from token ``x_t`` (shape (d_model,)) to itself and every cached token.

    ``project(x, w)`` computes ``x @ w``; the test passes a counting version, so use
    it for every projection.  Returns the attention output, shape (d_head,).
    """
    q = project(x_t, w_q)
    cache.append(project(x_t, w_k), project(x_t, w_v))
    keys, values = cache.view()
    weights = softmax(keys @ q / np.sqrt(q.shape[-1]))
    return weights @ values


def full_causal_attention(x, w_q, w_k, w_v, project=np.matmul):
    """Recompute every output from scratch: the baseline decode_step must match."""
    q, k, v = project(x, w_q), project(x, w_k), project(x, w_v)
    scores = q @ k.T / np.sqrt(q.shape[-1])
    scores = np.where(np.tril(np.ones(scores.shape, dtype=bool)), scores, -np.inf)
    return softmax(scores) @ v


def kv_cache_bytes(n_layers: int, n_kv_heads: int, d_head: int, seq_len: int,
                   bytes_per_elem: int, batch: int = 1) -> int:
    """Bytes held by the KV cache: keys and values for every layer, head and token."""
    return 2 * n_layers * n_kv_heads * d_head * seq_len * bytes_per_elem * batch
