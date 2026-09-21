#!/usr/bin/env python3
"""Check source solutions in a disposable package, without overwriting notebooks.

Shared by pre-commit and CI to check numerical and training behavior.
"""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
REGRESSIONS = (
    # Build these tests against the canonical source exports.
    "01_tensor/test_tensor_core.py",
    "04_losses/test_losses_core.py",
    "05_dataloader/test_batch_contract.py",
    "06_autograd/test_batched_matmul_backward.py",
    "07_optimizers/test_state_restore.py",
    "08_training/test_checkpoint_continuation.py",
    "09_convolutions/test_spatial_contracts.py",
    "11_embeddings/test_embedding_contracts.py",
    "12_attention/test_attention_contracts.py",
    "13_transformers/test_transformer_contracts.py",
    "14_profiling/test_profile_accounting.py",
    "16_compression/test_pruning_contracts.py",
    "18_memoization/test_cached_generation_equivalence.py",
    "20_capstone/test_event_policy.py",
    "06_autograd/test_graph_lifetime.py",
    "08_training/test_accumulation_windows.py",
    "15_quantization/test_constant_roundtrip.py",
    "16_compression/test_distillation_training.py",
    "02_activations/test_activations_core.py",
    "03_layers/test_layers_notebook_rng.py",
    "04_losses/test_losses_source.py",
    "05_dataloader/test_image_layout_source.py",
    "06_autograd/test_backward_boundaries.py",
    "07_optimizers/test_optimizer_validation.py",
    "08_training/test_training_edge_cases.py",
    "09_convolutions/test_pooling_source_regressions.py",
    "10_tokenization/test_tokenization_source_regressions.py",
    "13_transformers/test_transformer_gradient_flow.py",
    "15_quantization/test_quantization_composition.py",
    "16_compression/test_compression_source.py",
    "18_memoization/test_18_memoization_progressive.py",
    "19_benchmarking/test_benchmark_contracts.py",
    "20_capstone/test_source_validation.py",
)


def main() -> int:
    import jupytext
    from nbdev.export import nb_export

    sources = sorted((ROOT / "src").glob("[0-9][0-9]_*/*.py"))
    if len(sources) != 20:
        print(f"Expected 20 source modules, found {len(sources)}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(dir=ROOT, prefix="tinytorch-reference-") as tmp:
        sandbox = Path(tmp)
        package = sandbox / "tinytorch"
        for subdir in ("", "core", "perf"):
            target = package / subdir
            target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "tinytorch" / subdir / "__init__.py", target / "__init__.py")
        for source in sources:
            notebook = sandbox / f"{source.stem}.ipynb"
            jupytext.write(jupytext.read(source), notebook)
            nb_export(str(notebook), lib_path=str(package))

        tests = sandbox / "tests"
        tests.mkdir()
        # Keep the canonical layout for tests that load notebook source cells.
        shutil.copytree(ROOT / "src", sandbox / "src")
        for relative in REGRESSIONS:
            source = ROOT / "tests" / relative
            destination = tests / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        env = os.environ.copy()
        env.update(PYTHONPATH=str(sandbox), TINYTORCH_QUIET="1")
        return subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=short",
             "-c", str(ROOT / "pyproject.toml"), "--confcutdir", str(sandbox), str(tests)],
            cwd=sandbox, env=env,
        ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
