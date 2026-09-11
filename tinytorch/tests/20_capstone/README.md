# Module 20: Capstone Tests

These tests check the classroom benchmark and submission pipeline. They do not
prove that every earlier module is correct or that a model is production-ready.

- `test_capstone_core.py` checks tensor/gradient/layer composition and a small
  training loop, plus the benchmark suite's construction.
- `test_capstone_integration.py` measures two differently sized models, checks
  JSON provenance and improvement arithmetic, rejects malformed evaluation
  inputs, validates baseline and optimized metrics, and runs the pruning plus
  simulated-quantization example. Its storage assertion distinguishes actual
  dense FP32 arrays from hypothetical packed INT8 storage.

Run from the TinyTorch directory after exporting the reference solutions:

```bash
python3 -m pytest tests/20_capstone/ -v
```

Milestone convergence and the rest of the framework require their own suites;
passing these submission tests does not replace those checks.
