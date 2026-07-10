"""MLPerf EDU Tutorial 01: Anatomy of a Benchmark Run.

Launch interactively with:

    uv run marimo edit tutorials/01_first_benchmark.py
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import subprocess
    import sys
    from pathlib import Path

    import marimo as mo

    return Path, json, mo, subprocess, sys


@app.cell
def _(mo):
    mo.md(
        """
        # Tutorial 01: Anatomy of a Benchmark Run

        In this session we run one real benchmark, then take its output
        apart. The point is not the score. The point is learning what a
        trustworthy ML systems measurement carries with it: the metrics,
        the environment fingerprint, and the provenance artifact.

        We use `micro-lstm-train`, a white-box recurrent workload small
        enough to train in seconds on a laptop CPU.
        """
    )
    return


@app.cell
def _(Path, mo, subprocess, sys):
    run_dir = Path("tutorials/_runs/01").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlperf_edu",
            "run",
            "--workload",
            "micro-lstm-train",
            "--profile",
            "min",
            "--output-dir",
            str(run_dir),
        ],
        capture_output=True,
        text=True,
    )
    mo.md(
        f"""
        ## Step 1: Run the Benchmark

        ```text
        {completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else completed.stderr.strip()[-400:]}
        ```

        Exit code: `{completed.returncode}`. The run wrote its artifacts to
        `{run_dir}`.
        """
    )
    return (run_dir,)


@app.cell
def _(json, mo, run_dir):
    report_path = next(run_dir.glob("micro-lstm-train_min_report.json"))
    report = json.loads(report_path.read_text())
    metrics = report.get("metrics", report)
    rows = [
        {"field": key, "value": str(value)}
        for key, value in sorted(metrics.items())
        if not isinstance(value, (dict, list))
    ]
    mo.vstack(
        [
            mo.md(
                """
                ## Step 2: The Report Is Data, Not Console Text

                Every run emits JSON, HTML, and CSV. Below are the scalar
                metrics from the JSON report. Notice that timing, quality,
                and workload identity travel together in one artifact.
                """
            ),
            mo.ui.table(rows, selection=None),
        ]
    )
    return (report,)


@app.cell
def _(mo, report):
    fingerprint = report.get("fingerprint") or report.get("environment") or {}
    fp_rows = [
        {"field": key, "value": str(value)}
        for key, value in sorted(fingerprint.items())
        if not isinstance(value, (dict, list))
    ]
    mo.vstack(
        [
            mo.md(
                """
                ## Step 3: The Fingerprint

                A number without its environment is not a measurement. The
                report embeds the hardware and software fingerprint of the
                machine that produced it. When two results disagree, this
                block is where the explanation starts.
                """
            ),
            mo.ui.table(fp_rows, selection=None) if fp_rows else mo.md("_No fingerprint block found; inspect the raw report._"),
        ]
    )
    return


@app.cell
def _(mo, run_dir):
    provd = sorted(run_dir.glob("*.provd.json"))
    mo.md(
        f"""
        ## Step 4: Provenance

        The run also produced `{provd[0].name if provd else "(missing)"}`,
        the provenance artifact. It carries SHA-256 hashes of the trained
        state and the assets used, which is what lets an instructor or a
        reviewer verify a result without re-running it:

        ```bash
        mlperf verify {provd[0] if provd else "<run>.provd.json"}
        ```

        ## Exercise

        Re-run Step 1 twice and compare the two JSON reports. Which fields
        changed, which stayed fixed, and which *should* stay fixed if the
        benchmark is deterministic? Where would you look to explain a
        timing difference between your laptop and your neighbor's?
        """
    )
    return


if __name__ == "__main__":
    app.run()
