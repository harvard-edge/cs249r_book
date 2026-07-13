"""MLPerf EDU Tutorial 01. Anatomy of a benchmark run.

Launch the interactive notebook with:

    uv run marimo edit tutorials/01_first_benchmark.py

Validate its benchmark and provenance path with:

    python tutorials/smoke_first_benchmark.py
"""

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from smoke_first_benchmark import PROJECT_ROOT, run_tutorial_benchmark

    return PROJECT_ROOT, mo, run_tutorial_benchmark


@app.cell
def _(mo):
    mo.md("""
    # Tutorial 01. Anatomy of a Benchmark Run

    This session executes one registered workload and examines the output
    contract. The `min` profile for `time-series-forecasting` is a deterministic
    functional run over a synthetic smoke fixture. It is intentionally not a
    quality baseline. The exercise focuses on the metrics, environment
    fingerprint, and provenance needed to interpret any benchmark result.
    """)
    return


@app.cell
def _(PROJECT_ROOT, mo, run_tutorial_benchmark):
    run_dir = (PROJECT_ROOT / "tutorials" / "_runs" / "01").resolve()
    artifacts = run_tutorial_benchmark(run_dir)
    last_line = artifacts["stdout"].strip().splitlines()[-1]
    mo.md(f"""
        ## Step 1. Run the Benchmark

        ```text
        {last_line}
        ```

        The command completed successfully and wrote verified artifacts to
        `{run_dir}`.
        """)
    return (artifacts,)


@app.cell
def _(artifacts, mo):
    report = artifacts["report"]
    metrics = report["metrics"]
    rows = [
        {"field": key, "value": str(value)}
        for key, value in sorted(metrics.items())
        if not isinstance(value, (dict, list))
    ]
    mo.vstack(
        [
            mo.md("""
                ## Step 2. Treat the Report as Data

                Each workload run emits JSON and derived HTML and CSV views.
                These are the scalar metrics in the workload JSON. The profile,
                status, dataset mode, and workload identity remain attached to
                the measurement rather than living only in console output.
                """),
            mo.ui.table(rows, selection=None),
        ]
    )
    return (report,)


@app.cell
def _(mo, report):
    fingerprint = report["run_fingerprint"]
    hardware = fingerprint.get("hardware", {})
    software = fingerprint.get("software", {})
    fp_rows = [
        {"field": f"hardware.{key}", "value": str(value)}
        for key, value in sorted(hardware.items())
        if not isinstance(value, (dict, list))
    ]
    fp_rows.extend(
        {"field": f"software.{key}", "value": str(value)}
        for key, value in sorted(software.items())
        if not isinstance(value, (dict, list))
    )
    mo.vstack(
        [
            mo.md("""
                ## Step 3. Read the Run Fingerprint

                A timing result needs its execution context. The run fingerprint
                records hardware and software fields that help explain differences
                between machines and environments.
                """),
            mo.ui.table(fp_rows, selection=None),
        ]
    )
    return


@app.cell
def _(artifacts, mo):
    mo.md(f"""
        ## Step 4. Verify Provenance

        The run produced `{artifacts["manifest_path"].name}` and this notebook
        successfully verified it. The manifest binds the report and recorded
        inputs with cryptographic hashes for tamper detection. Its local digest
        does not authenticate who created the result.

        ```bash
        mlperf verify {artifacts["manifest_path"]}
        ```

        ## Exercise

        Re-run Step 1 twice and compare the workload JSON files. Separate fields
        that should remain deterministic from measurements that can vary with
        system load or thermal state. Then compare your run fingerprint with a
        classmate's and identify the first environmental difference you would
        investigate.
        """)
    return


if __name__ == "__main__":
    app.run()
