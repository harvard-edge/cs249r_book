# MLPerf EDU

MLPerf EDU is an independent preview of a locally executable, quality-gated
benchmark specification for teaching and studying single-node ML systems. It
adapts the reproducibility, verification, disclosure, and comparability
discipline of mature benchmark suites to classroom-scale PyTorch workloads.
It is not an official MLCommons benchmark and is not endorsed by MLCommons.

The v0.1 portfolio contains fourteen workloads. The current quality-evidence
scope covers nine workloads and twelve evidence cases. Five additional
workloads are in the functional spiral, where the public CLI validates bounded
execution, reporting, and provenance without claiming the authoritative
quality result. MLPerf EDU contributes the thin PyTorch adapter, execution
harness, measurement controls, provenance, and reports needed to move each
definition through functional, quality-conformant, repeatability-verified, and
promotion-ready stages.

## Install From the Checkout

The review build uses Python 3.10 or newer and a locked `uv` environment.

```bash
git clone https://github.com/harvard-edge/cs249r_book
cd cs249r_book/mlperf-edu
uv sync --locked --extra dev
uv run mlperf doctor
```

Fetch assets before measurement so network transfer is outside the timed
region.

```bash
uv run mlperf fetch --workload image-classification --profile max
uv run mlperf run --workload image-classification --profile max \
  --output-dir submissions/image-review --open-report
```

## Workload Portfolio

| **Workload** | **Authoritative Definition** | **Canonical Mode or Phase** | **Quality Contract** |
|:---|:---|:---|:---|
| `image-classification` | MLPerf Tiny float ResNet8 and its 200-sample CIFAR-10 accuracy set | inference | top-1 accuracy at least 0.85 |
| `keyword-spotting` | MLPerf Tiny DS-CNN and the 1,000-example EEMBC accuracy set | inference | top-1 accuracy at least 0.90 |
| `anomaly-detection` | MLPerf Tiny ToyADMOS autoencoder and the 248-recording ToyCar accuracy set | inference | ROC AUC at least 0.85 |
| `visual-wake-words` | MLPerf Tiny MobileNetV1 0.25 and the 1,000-example EEMBC accuracy set | inference | top-1 accuracy at least 0.80 |
| `causal-language-modeling` | nanoGPT Shakespeare character configuration and Tiny Shakespeare | training; full, prefill, and decode inference | validation cross-entropy at most 1.4697; every inference run passes its functional gate |
| `text-classification` | Pinned DistilBERT SST-2 checkpoint and GLUE development split | inference | accuracy at least the exact verified model-index result of 0.9105504587155964 |
| `information-retrieval` | Sentence Transformers CrossEncoder NanoBEIR example | inference | exact documented mean nDCG@10 |
| `graph-node-classification` | Official OGB GCN recipe on `ogbn-arxiv` | training | test accuracy within the published GCN reference tolerance |
| `time-series-forecasting` | Official PatchTST ETTm1 recipe and split | training | test MSE at most 0.29292929292929293, the direction-aware 99%-of-reference gate |
| `code-generation` | Qwen2.5-Coder and HumanEval+ | inference | published 0.573 HumanEval+ pass@1; functional integration only today |
| `function-calling` | Qwen3-1.7B and BFCL V4 Non-Live AST | inference | published 0.8292 AST accuracy; functional integration only today |
| `recommendation` | Meta DLRM and Criteo Terabyte | inference | published 0.8025 ROC AUC; functional integration only today |
| `image-generation` | NVIDIA EDM and the CIFAR-10 50,000-image FID protocol | inference | published 1.79 FID; functional integration only today |
| `reinforcement-learning` | MLPerf Training v0.5 MiniGo | training | 0.40 professional-move prediction and the upstream playoff contract; functional integration only today |

The [selection ledger](registry/selection-ledger.yaml) records the authority,
rationale, laptop evidence, and quality-conformance blocker for every audited
task. The five functional-stage candidates preserve their authoritative task
identity and published target. Their bounded probes are integration evidence,
not substitutes or benchmark results.

For the current quality-readiness milestone, one complete authoritative run is
enough to accept or reject a quality target. The complete required evaluation
set and every declared quality gate still apply. Repeated runs and timing
variation remain part of the later promotion and stability phase.

## Workload Identity

A workload ID names the stable learning task. Training and inference are
modes. Full, prefill, and decode are inference phases. Batching, precision,
quantization, compilation, scheduling, and serving behavior are configurations
recorded in reports. They do not create new workload IDs.

```bash
uv run mlperf run --workload causal-language-modeling --profile max \
  --mode training --output-dir submissions/causal-training

uv run mlperf run --workload causal-language-modeling --profile max \
  --mode inference --phase decode \
  --output-dir submissions/causal-decode
```

## Profiles

| **Profile** | **Purpose** | **Result Boundary** |
|:---|:---|:---|
| `min` | Fast installation, teaching, and CI check | Functional only; never a public score or performance baseline |
| `max` | Canonical classroom and comparison scale | Uses the real-data quality contract after conformance; functional-stage candidates remain bounded, nonpromotable probes |
| `pro` | Extended single-node research envelope | Changes controlled configurations without changing workload identity |

The research envelope supports processors, memory systems, runtimes,
compilers, and model execution. Distributed and datacenter-scale claims are
outside v0.1.

## Evidence and Provenance

The current twelve evidence cases consist of one canonical `max` case for each
of the nine quality-evidence workloads plus full, prefill, and decode inference for
`causal-language-modeling`. The draft snapshot in
`provisional_results/index.json` contains six five-run verified records and six
explicitly provisional records. Provisional records establish execution and
gate passage only; they do not establish repeatability or qualify as promoted
baselines.

The five functional-stage workloads are outside this importer by construction.
Once each passes its authoritative quality contract, the full fourteen-workload
portfolio will contain seventeen evidence cases.

Promotion still requires five fresh processes at the canonical seed. Every run
must pass its quality or functional gate, and the primary timing coefficient
of variation must not exceed 5%. A future complete promoted index will be
written under `reference_results/`. Both index forms bind each result to its
SHA-256 digest, exact source revision, mode, phase, result role, metric
aggregate, repeatability decision, and optional training lineage. Raw
checkpoints and dataset-derived files remain in the local review handoff. They
are not committed to the repository.

Every run writes these review artifacts:

| **Artifact** | **Purpose** |
|:---|:---|
| `*_report.json` | Complete metrics, quality status, configuration, and environment |
| `*_report.csv` | Flat metrics for analysis |
| `*_report.html` | Human-readable run report |
| `*.provd.json` | Content-addressed provenance manifest |

```bash
uv run mlperf verify submissions/image-review/image-classification_max.provd.json
uv run mlperf package submissions/image-review/image-classification_max.provd.json
uv run mlperf report submissions/image-review --format html --open
```

Verification checks recorded bytes and provenance. It does not authenticate
the producer or imply MLCommons acceptance.

## Validation

```bash
uv run pytest
uv run mlperf validate smoke --output-dir submissions/smoke
uv run mlperf validate pro --dry-run --output-dir submissions/pro-plan
uv run python tools/export_flat_registry.py --check
uv run python tools/sync_verified_baselines.py --check
uv run python tools/check_taxonomy.py
uv run python tools/check_reference_claims.py --check
uv run python tools/generate_docs.py --check
uv run make -C paper clean all check
quarto render site
uv run python tools/check_site_layout.py --build-dir site/_build --report-dir site-layout-report
uv run python tools/build_wheel.py --out-dir /tmp/mlperf-edu-wheel
```

The release workflow also installs the wheel in a clean Python environment,
checks packaged registry and evidence resources, verifies portable archives,
and builds the site and paper.

## Documentation and Review

- [Specification](SPEC.md)
- [Proposal](PROPOSAL.md)
- [Public result rules](PUBLIC_RULES.md)
- [Quality target review](QUALITY_TARGET_REVIEW.md)
- [Dataset release review](DATASET_RELEASE_REVIEW.md)
- [Release checklist](RELEASE_CHECKLIST.md)
- [Generated benchmark site](site/benchmarks/index.qmd)
- [Companion paper](paper/paper.tex)

The component license, public package publication, dataset redistribution
decisions, and MLCommons naming and governance decisions remain external
release gates. The benchmark implementation and review artifacts fail closed
while those decisions are pending.
