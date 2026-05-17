# MLSys·im 0.1.2 — CLI, Website, and Serving Model Polish

**Release date:** 2026-05-17

This patch release improves first-run CLI usability, release readiness, and
serving-model coverage. Existing serving predictions remain unchanged unless
the new optional chunked-prefill parameter is used.

## Install

```bash
pip install mlsysim==0.1.2
```

Verify the install:

```bash
python -c "import mlsysim; print(mlsysim.__version__)"
mlsysim eval Llama3_8B H100 --batch-size 32
```

## What's fixed

- `ServingModel.solve()` now accepts `prefill_chunk_tokens` to estimate a
  Sarathi-Serve-style chunked-prefill stall proxy:

  ```python
  from mlsysim import Hardware, Models, ServingModel

  result = ServingModel().solve(
      Models.Llama3_8B,
      Hardware.H100,
      seq_len=8192,
      prefill_chunk_tokens=512,
  )
  print(result.decode_stall_bound)
  ```

- `mlsysim serve` exposes the same analytical control from the CLI:

  ```bash
  mlsysim serve Llama3_8B H100 --seq-len 8192 --prefill-chunk-tokens 512 -o json
  ```

- `-o/--output` now works before or after subcommands:

  ```bash
  mlsysim -o json eval Llama3_8B H100
  mlsysim eval Llama3_8B H100 -o json
  ```

- `mlsysim audit -o json` now emits a single parseable JSON object instead of
  mixing human-readable audit banners into stdout.
- `mlsysim schema -o json` is accepted for command consistency. Schema output
  remains JSON by design.
- The website citation snippets, CLI reference, and instructor version-pinning
  guidance now match the current release state.
- MLSysBook browser wheel references now point to
  `mlsysim-0.1.2-py3-none-any.whl`, and the wheel is present under `wheels/`
  for hosted Pyodide execution.

## Validation

Before release, this tree was checked with:

```bash
python3 -m pytest tests -q
PYTHONPATH=mlsysim python3 -m pytest book/tests -q --no-cov
python3 -m ruff check .
PYTHONPATH=. quarto render docs
python3 -m build --sdist --wheel
```

## Project links

- Docs: [mlsysbook.ai/mlsysim/](https://mlsysbook.ai/mlsysim/)
- Source: [github.com/harvard-edge/cs249r_book/tree/mlsysim-v0.1.2/mlsysim](https://github.com/harvard-edge/cs249r_book/tree/mlsysim-v0.1.2/mlsysim)
- Issues: [github.com/harvard-edge/cs249r_book/issues](https://github.com/harvard-edge/cs249r_book/issues)
