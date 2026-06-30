# MLPerf EDU Install Guide

This is the install contract for the runnable preview. The public command is
`mlperf`; the distribution package is `mlperf-edu`; the Python compatibility
package is `mlperf_edu`.

## Recommended: uv From A Checkout

```bash
cd mlperf-edu
uv sync --extra dev
uv run mlperf doctor
uv run mlperf list profiles
uv run mlperf init --profile min
```

Use this path for development, classrooms, and artifact evaluation from a
checked-out repository. It installs the project in an isolated `.venv` and keeps
the `mlperf` command tied to the current source tree.

## Install The CLI As A Tool

```bash
cd mlperf-edu
uv tool install .
mlperf doctor
mlperf run --profile min --dry-run
```

Use this path when the goal is to make `mlperf` available as a normal command.
The command uses the packaged `workloads.yaml` registry when it is not running
inside a source checkout.

## Build A Wheel And Install It

```bash
cd mlperf-edu
uv build
uv tool install --force dist/mlperf_edu-0.1.0-py3-none-any.whl
mlperf doctor
mlperf audit
mlperf validate smoke --dry-run
```

The wheel contains `src/mlperf_edu/workloads.yaml`, a materialized copy of the
generated flat registry. Keep it synchronized with:

```bash
python3 tools/export_flat_registry.py --check
python3 tools/export_flat_registry.py
```

## pip Fallback

```bash
cd mlperf-edu
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
mlperf doctor
```

## Optional Extras

```bash
uv sync --extra audio
uv sync --extra dev --extra audio
```

`audio` installs `torchaudio` for full Speech Commands experiments. The default
suite can run without it because the local preview uses lightweight paths for
the tiny workloads.

## Packaging Checks Before Release

```bash
python3 tools/export_registry_layout.py --check
python3 tools/export_flat_registry.py --check
python3 tools/generate_review_packets.py --check
uv build
python3 -m pytest -q
mlperf audit
mlperf validate smoke --dry-run
```

Long validation remains separate:

```bash
mlperf validate release --keep-going --skip-doctor
```
