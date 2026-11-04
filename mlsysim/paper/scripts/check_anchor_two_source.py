#!/usr/bin/env python3
"""Re-derive the Benchmark 2 (A2) constants in ``paper_scenarios.py`` from the published raw data.

Source: LLM-Inference-Bench (Chitty-Venkata et al., 2024, arXiv:2411.00136), raw data in the
argonne-lcf/LLM-Inference-Bench GitHub repository. Clone it and pin the commit the paper cites::

    git clone https://github.com/argonne-lcf/LLM-Inference-Bench
    git -C LLM-Inference-Bench checkout 4211f5d4e9bc0145ed541b6d372d5d8223b44235
    python paper/scripts/check_anchor_two_source.py LLM-Inference-Bench

The script uses the standard library only and does not import mlsysim: it reads the A2 constants
from ``paper_scenarios.py`` with ``ast`` so it runs without the package's dependencies. It checks

1. the checked-out commit (when the directory is a git checkout) equals ``A2_SOURCE_COMMIT``;
2. every ``A2_REPORTED_E2E_S`` value equals the ``Latency`` column of
   ``Plots/All_results.csv`` for "Nvidia H100 GPU, 4, <engine>, meta-llama/Llama-2-70b-hf, 128, 1";
3. the same rows satisfy Throughput = (input + output tokens) / Latency, which is how we know the
   ``Latency`` column is end-to-end request latency for 128 input and 128 output tokens;
4. no row of ``Plots/Fig_22/all_results_TTFT_ITL.csv`` covers Llama-2-70B (so the comparison must
   be labelled end-to-end), and the ``A2_PREFILL_REF`` TTFT values equal its H100 Llama-2-7B rows;

then prints the per-token values and the prefill bound, plus SHA-256 digests of both CSVs.
It exits non-zero on any mismatch.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import math
import subprocess
import sys
from pathlib import Path

SCENARIOS = Path(__file__).resolve().parent / "paper_scenarios.py"
E2E_CSV = Path("Plots/All_results.csv")
TTFT_CSV = Path("Plots/Fig_22/all_results_TTFT_ITL.csv")
MODEL_70B = "meta-llama/Llama-2-70b-hf"
MODEL_7B = "meta-llama/Llama-2-7b-hf"
# Size-class parameter counts, as in mlsysim's registry (models/data/language.yaml).
PARAMS_70B, PARAMS_7B = 70e9, 7e9
REL_TOL = 1e-12


def a2_constants() -> dict:
    """Literal A2_* assignments from paper_scenarios.py, evaluated without importing it."""
    tree = ast.parse(SCENARIOS.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if not name.startswith("A2_"):
                continue
            value = node.value
            if isinstance(value, ast.Call) and getattr(value.func, "id", None) == "dict":
                out[name] = {kw.arg: ast.literal_eval(kw.value) for kw in value.keywords}
            else:
                try:
                    out[name] = ast.literal_eval(value)
                except ValueError:
                    pass
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    root = Path(argv[1])
    c = a2_constants()
    failures: list[str] = []

    def check(ok: bool, msg: str) -> None:
        print(("ok    " if ok else "FAIL  ") + msg)
        if not ok:
            failures.append(msg)

    # 1. Source version.
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        check(head == c["A2_SOURCE_COMMIT"], f"source commit {head} == A2_SOURCE_COMMIT")
    except (OSError, subprocess.CalledProcessError):
        print(f"note  {root} is not a git checkout; commit not checked (expected {c['A2_SOURCE_COMMIT']})")

    # 2-3. End-to-end rows.
    n_in, n_out = c["A2_REPORTED_INPUT_TOKENS"], c["A2_REPORTED_OUTPUT_TOKENS"]
    with open(root / E2E_CSV, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r["Hardware"] == "Nvidia H100 GPU" and r["Num of Hardware"] == str(c["A2_TP_DEGREE"])
                and r["Model"] == MODEL_70B and r["Input Output Length"] == str(n_in) and r["Batch Size"] == "1"]
    by_engine: dict[str, dict] = {}
    for r in rows:
        by_engine.setdefault(r["Framework"], r)   # the file repeats some rows verbatim
    for engine, e2e in c["A2_REPORTED_E2E_S"].items():
        r = by_engine.get(engine)
        check(r is not None, f"{E2E_CSV}: row for {engine}, {MODEL_70B}, H100 x{c['A2_TP_DEGREE']}, {n_in}/{n_out}, batch 1")
        if r is None:
            continue
        lat, thr = float(r["Latency"]), float(r["Throughput"])
        check(math.isclose(lat, e2e, rel_tol=REL_TOL), f"{engine}: Latency {lat!r} == A2_REPORTED_E2E_S {e2e!r}")
        check(math.isclose(thr, (n_in + n_out) / lat, rel_tol=1e-9),
              f"{engine}: Throughput {thr:.6f} == ({n_in}+{n_out}) / Latency, so Latency is end-to-end")

    # 4. TTFT coverage and the prefill reference.
    ref = c["A2_PREFILL_REF"]
    with open(root / TTFT_CSV, newline="", encoding="utf-8") as f:
        ttft_rows = [r for r in csv.DictReader(f) if r.get("Model")]
    check(not any("70b" in r["Model"].lower() for r in ttft_rows),
          f"{TTFT_CSV}: no 70B rows, so A2 has no matching TTFT and is labelled end-to-end")
    for engine, ttft in ref["ttft_s"].items():
        match = [r for r in ttft_rows if r["Hardware"] == "Nvidia H100 GPU" and r["Framework"] == engine
                 and r["Model"] == MODEL_7B and r["Num of Hardware"] == str(ref["gpus"])
                 and r["Input Length"] == str(ref["input_tokens"]) and r["Batch Size"] == str(ref["prompts"])]
        check(len(match) == 1 and math.isclose(float(match[0]["TTFT Latency"]), ttft, rel_tol=REL_TOL),
              f"{engine}: H100 Llama-2-7B TTFT ({ref['prompts']} x {ref['input_tokens']} tokens) == {ttft!r}")

    # Derived values (same formulas as paper_scenarios.anchor_two).
    print()
    per_token = {e: 1e3 * v / n_out for e, v in c["A2_REPORTED_E2E_S"].items()}
    for e, v in per_token.items():
        print(f"{e:13s} end-to-end latency / output token = {v:.4f} ms (includes prefill)")
    rate = 2 * PARAMS_7B * ref["prompts"] * ref["input_tokens"] / (max(ref["ttft_s"].values()) * ref["gpus"])
    prefill_s = 2 * PARAMS_70B * n_in / (c["A2_TP_DEGREE"] * rate)
    share = prefill_s / min(c["A2_REPORTED_E2E_S"].values())
    print(f"measured H100 prefill rate (slowest engine) = {rate / 1e12:.0f} TFLOP/s per GPU")
    print(f"70B prefill of {n_in} tokens on {c['A2_TP_DEGREE']} GPUs <= {prefill_s * 1e3:.2f} ms "
          f"= {share:.2%} of end-to-end latency ({c['A2_PREFILL_PESSIMISM']:g}x pessimistic: "
          f"{c['A2_PREFILL_PESSIMISM'] * share:.2%})")
    print(f"\nsha256 {E2E_CSV}  {sha256(root / E2E_CSV)}")
    print(f"sha256 {TTFT_CSV}  {sha256(root / TTFT_CSV)}")

    if failures:
        print(f"\n{len(failures)} check(s) failed", file=sys.stderr)
        return 1
    print("\nAll A2 constants match the source data.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
