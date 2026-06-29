# MLPerf EDU Release Checklist

This checklist separates packaging readiness from MLCommons endorsement
readiness. A local teaching release can ship before every public-result policy
question is closed, but the release notes must say that clearly.

## 0.1 Preview Release Bar

| Status | Item | Evidence |
|---|---|---|
| Done | Public CLI is `mlperf` | `pyproject.toml` exposes `mlperf = "mlperf.edu_cli:main"` |
| Done | uv install path exists | `INSTALL.md` documents `uv sync`, `uv tool install`, and `uv build` |
| Done | Wheel has registry metadata | `src/mlperf_edu/workloads.yaml` is a real packaged data file |
| Done | Native registry has generated flat mirror | `tools/export_flat_registry.py --check` checks root and packaged mirrors |
| Done | Local audit is clean | `mlperf audit` passes with 0 local warnings |
| Done | Local validation is clean | latest recorded `validate release` passed 60/60 with 0 local warnings |
| Done | HTML/JSON/CSV/provenance artifacts are default | `run`, `init`, `validate`, `report`, and `grade` paths emit them |
| Done | Review packets exist | `review_packets/` has nine current public-result candidate packets |
| Open | Publishable license statement for the component | Add or point to the authoritative MLPerf EDU component license before external package publication |
| Open | Fresh Linux package smoke in CI | GitHub Actions workflow exists; confirm it passes after pushing |

## 1.0 Public/Endorsement Release Bar

| Status | Item | Decision Needed |
|---|---|---|
| Open | `mlperf audit --policy public` passes | Resolve MovieLens-100K approval or replace the score-bearing recommender dataset |
| Open | Quality targets approved | Experts sign off on score-bearing and performance-bearing target rationale |
| Open | Public result wording approved | MLCommons decides whether MLPerf EDU can use endorsed/result language |
| Open | Dataset/model attribution checked | Every public asset has source, license, citation, fetch policy, and report fields |
| Open | Reproducibility packet reviewed | At least one external fresh-machine run checks install, fetch, run, report, grade, and verify |
| Open | Release notes written | Clearly distinguish MLPerf EDU teaching/research results from official competitive MLPerf submissions |

## Non-Blocking Follow-Ups

- Add a published package index target only after the license statement is final.
- Add optional backend extras for ONNX Runtime, MLX, and llama.cpp once the
  corresponding runners are stable enough to be user-facing.
- Add comparison tooling for repeated `pro` runs after the base install path is
  stable.
