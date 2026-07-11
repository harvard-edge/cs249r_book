# MLPerf EDU Install Guide

The supported preview install comes from a source checkout. The distribution
name is `mlperf-edu`, the public command is `mlperf`, and the Python
compatibility package is `mlperf_edu`. No package-index release is claimed.

## Locked Checkout Install

```bash
cd mlperf-edu
uv sync --locked --extra dev
uv run mlperf doctor
uv run mlperf list profiles
uv run mlperf validate smoke --output-dir submissions/install-smoke
```

Use this path for development, classrooms, and artifact evaluation. It creates
an isolated `.venv` and runs the command from the current source tree. Python
3.10 or newer is required. A GPU is optional.

`doctor` checks the environment and registry. The actual smoke preset executes
and grades the fast benchmark collection. A successful `doctor` alone does not
prove that workload execution works.

## Local Tool Install

```bash
cd mlperf-edu
uv tool install .
mlperf doctor
mlperf run --profile min --dry-run
```

This makes `mlperf` available as a normal command while installing from the
checkout. It is not equivalent to a published package-index install.

## Build and Test the Wheel

```bash
cd mlperf-edu
uv run python tools/export_flat_registry.py --check
uv run python tools/build_wheel.py
```

The wheel must include the packaged registry and the bundled SLM quality
fixture. Inspect and install it in a fresh environment outside the checkout.

```bash
wheel=$(find dist -maxdepth 1 -name '*.whl' -print -quit)
test -n "$wheel"
unzip -l "$wheel" | grep -q 'mlperf_edu/workloads.yaml'
unzip -l "$wheel" | grep -q 'mlperf_edu/slm_quality_prompts.json'

uv venv /tmp/mlperf-edu-wheel-smoke --python 3.12
uv pip install --python /tmp/mlperf-edu-wheel-smoke/bin/python "$wheel"
(
  cd /tmp
  /tmp/mlperf-edu-wheel-smoke/bin/mlperf list --format json \
    > /tmp/mlperf-edu-workloads.json
  /tmp/mlperf-edu-wheel-smoke/bin/mlperf audit
)
python3 -c 'import json; assert json.load(open("/tmp/mlperf-edu-workloads.json"))["workloads"]'
```

The native registry under `registry/` is the authoring source. The root
`workloads.yaml` and `src/mlperf_edu/workloads.yaml` files are generated
compatibility mirrors. Keep them synchronized with these commands.

```bash
uv run python tools/export_registry_layout.py --check
uv run python tools/export_flat_registry.py --check
```

Run the generators without `--check` only when intentionally refreshing their
outputs.

## pip Fallback

```bash
cd mlperf-edu
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
mlperf doctor
mlperf validate smoke --output-dir submissions/pip-smoke
```

The lockfile-backed `uv` path is the release reference because it constrains
the environment more tightly. The editable `pip` path is a convenience path,
not independent release evidence.

## Optional Extras

```bash
uv sync --locked --extra audio
uv sync --locked --extra tutorial
uv sync --locked --extra dev --extra audio --extra tutorial
```

The `audio` extra installs `torchaudio` for Speech Commands experiments. The
DS-CNN row remains systems-only, and its fast paths do not require the full
dataset. The `tutorial` extra installs marimo for the implemented first
notebook.

## Local Notebook Expectations

The core install and all lab smoke paths run on CPU without a network after
dependencies are installed. Score-bearing `max` runs fetch datasets on first
use. The SLM `max` path fetches pinned model weights. Cache those assets before
a class, airplane run, or reproducibility session.

```bash
uv run mlperf fetch --profile max --dry-run
uv run mlperf cache list
uv run python examples/lab1_optimization.py --smoke
uv run python examples/lab2_inference_sut.py --smoke
uv run python examples/lab3_arch_comparison.py --smoke
uv run python tutorials/smoke_first_benchmark.py
```

The repository does not yet provide a complete offline bundle containing all
dependencies, datasets, and model weights.

## Release Checks

Use [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) as the executable ledger. The
minimum install and packaging subset follows.

```bash
set -euo pipefail
uv sync --locked --extra dev
uv run pytest
uv run python tools/export_registry_layout.py --check
uv run python tools/export_flat_registry.py --check
uv run python tools/check_taxonomy.py
uv run python tools/generate_review_packets.py --check
uv run python tools/generate_docs.py --check
uv run mlperf audit
uv run mlperf validate smoke --output-dir submissions/release-smoke
uv run python tools/build_wheel.py
```

Actual `max` and `release` validation remain separate evidence-bearing gates.
Selection-only dry runs do not satisfy them.

## Documentation Site

```bash
uv run python tools/generate_docs.py --check
quarto render site
python3 ../shared/scripts/check-internal-links.py site --quiet
```

The workflows can build a development preview and a manually confirmed live
preview. Their presence does not prove that the current revision has deployed.
The live workflow is documentation publication only. It does not publish the
Python package or imply MLCommons endorsement.
