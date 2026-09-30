"""
Milestone 07 pass gate: it grades the student's Module 17 kernels.

2026-09-29: Milestone 07 used to pass by running the native kernels bundled in
tinytorch/extensions/ (hand-written and tracked in git, auto-compiled; the MPS
path is torch.matmul), so no student code was checked, and a machine without a
C++ compiler failed it. It now requires YOUR tiled_matmul, fused_gelu, im2col,
im2col_conv2d, and col2im to match NumPy on ragged shapes; the bundled kernels
are timed as reference points only and never decide the result.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

TINYTORCH_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(os.environ.get("MILESTONE_GATE_ROOT", TINYTORCH_ROOT))
SCRIPT = SCRIPT_ROOT / "milestones" / "07_2024_kernels" / "01_custom_kernels.py"

SABOTAGE_HOOK = textwrap.dedent('''
    import os
    _S = os.environ.get("TT_GATE_SABOTAGE", "")
    if _S:
        import numpy as np
        from tinytorch.core.tensor import Tensor
        if _S == "tiled_drops_ragged_tile":
            # The classic blocking bug: iterate only over full tiles.
            import tinytorch.perf.acceleration as A
            def tiled_matmul(a, b, tile_size=64):
                X, Y = a.data, b.data
                M, K = X.shape
                N = Y.shape[1]
                C = np.zeros((M, N), dtype=X.dtype)
                for i in range(0, M - tile_size + 1, tile_size):
                    for j in range(0, N - tile_size + 1, tile_size):
                        for k in range(0, K - tile_size + 1, tile_size):
                            C[i:i+tile_size, j:j+tile_size] += X[i:i+tile_size, k:k+tile_size] @ Y[k:k+tile_size, j:j+tile_size]
                return Tensor(C)
            A.tiled_matmul = tiled_matmul
        elif _S == "gelu_wrong_constant":
            import tinytorch.perf.acceleration as A
            A.fused_gelu = lambda x: Tensor(0.5 * x.data * (1.0 + np.tanh(0.8 * (x.data + 0.044715 * x.data ** 3))))
        elif _S == "no_native":
            # A machine with no C++ compiler and no GPU.
            import tinytorch.extensions.simd_ops as S
            S.simd_build_info = lambda: {"built": False, "openmp": False, "command": None,
                                         "error": "no C++ compiler found"}
            import tinytorch.extensions.mps_ops as MPS
            MPS.has_mps_support = lambda: False
            import tinytorch.extensions.triton_gelu as T
            T._HAS_TRITON = False
        else:
            raise SystemExit("unknown sabotage " + _S)
''')


@pytest.fixture(scope="module")
def hook_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("gate_hook_kernels")
    (d / "sitecustomize.py").write_text(SABOTAGE_HOOK)
    return d


def run_script(hook_dir, sabotage=""):
    env = dict(os.environ)
    env.update(TINYTORCH_NON_INTERACTIVE="1", CI="true",
               PYTHONPATH=os.pathsep.join([str(hook_dir), str(TINYTORCH_ROOT)]),
               TT_GATE_SABOTAGE=sabotage)
    proc = subprocess.run([sys.executable, str(SCRIPT)], cwd=TINYTORCH_ROOT, env=env,
                          capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=300)
    return proc.returncode, proc.stdout + proc.stderr


def test_correct_code_passes(hook_dir):
    rc, out = run_script(hook_dir)
    assert rc == 0, out[-3000:]
    assert "PART A" in out and "[PASS] tiled_matmul" in out


@pytest.mark.parametrize("sabotage, function", [
    ("tiled_drops_ragged_tile", "tiled_matmul"),
    ("gelu_wrong_constant", "fused_gelu"),
])
def test_wrong_module17_kernel_fails(hook_dir, sabotage, function):
    rc, out = run_script(hook_dir, sabotage)
    assert rc == 1, f"{sabotage} passed Milestone 07:\n{out[-3000:]}"
    assert f"[FAIL] {function}" in out, out[-3000:]
    assert "Traceback" not in out
    assert "SUCCESS" not in out


def test_no_compiler_still_grades_student_code(hook_dir):
    """Native kernels are optional context: no compiler must not fail correct code."""
    rc, out = run_script(hook_dir, "no_native")
    assert rc == 0, out[-3000:]
    assert "not built" in out
    assert "No bundled native kernel ran here" in out


def test_no_student_credit_for_bundled_kernels():
    text = SCRIPT.read_text()
    assert "YOUR C++" not in text and "YOUR compiled" not in text
