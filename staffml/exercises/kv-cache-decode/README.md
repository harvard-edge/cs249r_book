# KV-cache decode

**Task.** Generate attention outputs one token at a time for a single causal
attention head. Each step should project only the new token, append its key and
value to a pre-allocated cache, and attend over the cached rows. Implement the
`TODO`s in `starter.py`: `KVCache.append`, `KVCache.view`, `decode_step` and
`kv_cache_bytes`.

**What the tests check.**
- Outputs match `full_causal_attention` (recompute from scratch) to 1e-5.
- Over `T` steps, exactly `3T` token rows go through the Q/K/V projections.
  Recomputing the prefix each step would take `3T(T+1)/2` (6,240 instead of 192
  for `T = 64`).
- The cache is written in place and raises when full.
- `kv_cache_bytes` gives the right size for Llama-2-7B.

## Napkin math

Without a cache, step `t` recomputes `t` key and value projections and `t²` scores:
`O(t² · d)` work per step and `O(T³ · d)` for a `T`-token generation. With a
cache, step `t` does one projection and reads `t` cached rows: `O(t · d)`.

The price is memory:

```
KV bytes = 2 (K and V) · layers · KV heads · d_head · tokens · bytes per element
```

For Llama-2-7B (32 layers, 32 KV heads, d_head 128) in FP16 that is 0.5 MiB per
token, or 2 GiB at 4,096 tokens, about a sixth of the 13.5 GB of weights. At
batch 8 the cache (16 GiB) outgrows the weights. Each decode step also reads the
whole cache, so at long context the cache, not the weights, sets the bandwidth
cost. This is why grouped-query attention (8 KV heads cut it 4×) and KV
quantization matter.
