# StaffML coding exercises (prototype)

Hands-on companions to StaffML questions, from #2155 (option B): the vault schema
is unchanged, and each exercise lives beside it in its own folder.

| Exercise | Implement | Time | Tests |
|---|---|---|---|
| [`kv-cache-decode/`](kv-cache-decode/) | incremental attention with a pre-allocated KV cache | 20–30 min | matches full recompute to 1e-5; projects each token once (linear, not quadratic, work) |
| [`per-channel-int8/`](per-channel-int8/) | per-channel INT8 weights and an INT8 × INT8 → INT32 matmul | 25–35 min | error within the analytic quantization bound; one scale per channel; no INT32 overflow |

Each folder has:

- `README.md`: the task and the napkin math (what the optimization buys, in bytes or FLOPs);
- `starter.py`: the interface, with `TODO`s;
- `solution.py`: a reference solution;
- `test_<exercise>.py`: tests that grade either file.

Requirements: Python 3.10+, NumPy and pytest; CPU only.

```bash
pytest staffml/exercises                          # check the reference solutions
STAFFML_IMPL=starter pytest staffml/exercises/kv-cache-decode   # grade your starter
```
