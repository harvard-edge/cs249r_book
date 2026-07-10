# MLPerf EDU Tutorials

Hands-on tutorial materials for teaching ML systems benchmarking, designed to
run as a half-day conference tutorial (ISCA, MICRO, ASPLOS, HPCA) or as a
course module. Every exercise runs on an attendee laptop; no cluster, no GPU
required.

## Format

Notebooks are [marimo](https://marimo.io) files: plain Python, reactive,
version-controllable, and runnable either as interactive notebooks or as
scripts. Install the tutorial extra and launch:

```bash
uv sync --extra tutorial
uv run marimo edit tutorials/01_first_benchmark.py
```

Notebooks consume the run report JSON that every `mlperf run` produces. They
never import harness internals, so they keep working as the harness evolves.

## The Half-Day Program (3.5 hours)

| # | Session | Time | Notebook |
|:--|:---|:---|:---|
| 0 | **Setup and doctor** — install, `mlperf doctor`, offline fallback kit | 0:15 | — |
| 1 | **Anatomy of a benchmark run** — profiles, scenarios, reports, provenance | 0:45 | `01_first_benchmark.py` |
| 2 | **The systems lens** — compute-, memory-, and dispatch-bound regimes across three workloads; roofline telemetry from real runs | 1:00 | `02_regimes_roofline.py` (planned) |
| 3 | **Optimize like MLPerf** — SUT plugin lab: baseline decode, add KV cache and quantization, measure honestly | 1:00 | `03_optimize_sut.py` (planned) |
| 4 | **The research envelope** — `pro` profile variant sweeps (int8, batching, long context); citing results in a paper | 0:30 | `04_research_envelope.py` (planned) |

Session 0 exists because conference Wi-Fi fails. The offline kit is a wheel
plus pre-fetched dataset cache produced ahead of time:

```bash
uv build
mlperf fetch --profile min          # populate datasets/local_tensors
tar czf mlperf-edu-offline-kit.tgz dist/ datasets/local_tensors/
```

Attendees who cannot reach PyPI install from the kit and still finish every
exercise.

## Design Rules for Tutorial Material

1. **Laptop budget.** Every cell completes in under two minutes on a
   several-year-old laptop CPU; full sessions never depend on an accelerator.
2. **Reports are the interface.** Exercises read `*_report.json` artifacts,
   teaching students that benchmark output is data to analyze, not console
   text to glance at.
3. **Measurement discipline is the lesson.** Each session ends by asking what
   would make the number untrustworthy (warmup, seeds, thermal state,
   fingerprint disclosure) and shows how the harness addresses it.
4. **One benchmark fact source.** Notebooks link to the generated benchmark
   pages on the documentation site rather than restating registry facts.

## Course Use

Sessions 1 through 3 map directly onto a problem-set arc: run the baseline,
diagnose the bottleneck from the report, optimize through a SUT plugin, and
submit a packaged result that `mlperf grade` can score. The lab files under
`examples/` are the graded counterparts of these tutorial notebooks.
