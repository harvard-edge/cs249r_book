# Standardized Lab Template

Use the existing labs in `vol1/` and `vol2/` as the canonical implementation
examples. New labs should follow the same Marimo structure:

1. Setup cell with WASM bootstrap, `mlsysim` imports, shared styles, and ledger.
2. Opening/header section with learning objectives and scenario.
3. Structured prediction widgets before instruments are revealed.
4. Interactive analysis parts organized with `mo.ui.tabs`.
5. Synthesis section with takeaways and Design Ledger save.

After creating or editing a lab, run:

```bash
python3 -m pytest labs/tests/test_static.py -v
python3 -m pytest labs/tests/test_engine.py -v -k engine
```
