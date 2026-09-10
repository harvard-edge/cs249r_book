# MLPerf EDU

A benchmark suite for ML systems that runs on a laptop without giving up the
quality gate.

Production benchmark suites teach a real discipline. They hold the task, the
data, the metric, and the quality target fixed, so that two results can be
compared at all. Their compute and submission requirements also put them out of
reach of a course. The usual substitute is a folder of small models, which
solves the resource problem and quietly throws away the gate.

MLPerf EDU keeps both. Fourteen workloads run locally, and **no quality target
is set by this project**. Every one is inherited from an upstream paper,
leaderboard, or official evaluator, so the suite has no way to move a threshold
it fails to reach. It reports its own misses, and it currently has six of them.

> This is an independent research project and a working name. It is not an
> official MLCommons benchmark and is not endorsed by MLCommons.

## Quickstart

Python 3.10 or newer. A GPU is optional. Roughly five minutes.

```bash
git clone https://github.com/harvard-edge/cs249r_book
cd cs249r_book/mlperf-edu

uv sync --locked                 # create the locked environment
uv run mlperf doctor             # check this machine
uv run mlperf init --profile min --output-dir submissions/first-run
```

`doctor` prints a table of your Python, PyTorch, device, caches, and registry.
`init` then fetches what it needs and runs the four-workload `min` path, which
is the fast functional check. It writes JSON, CSV, and HTML reports and
verifies their provenance. Nothing opens a browser unless you ask.

Look at what you produced:

```bash
uv run mlperf report submissions/first-run --format html --open
```

Then run something that carries a real quality gate:

```bash
uv run mlperf fetch --workload image-classification --profile max
uv run mlperf run --workload image-classification --profile max \
  --output-dir submissions/image-review
```

That one loads the official MLPerf Tiny ResNet8 and its 200-example CIFAR-10
accuracy set and grades the result against the inherited 85% top-1 target. It
finishes in seconds.

## What Just Happened

The `min` run proved your install works. The `max` run produced a result that
means something, because it was graded against a target this project did not
choose.

Three ideas explain most of the system.

**A workload is a task, not a configuration.** `causal-language-modeling` is
one workload whether you train it or run inference on it, and whether you batch
it, compile it, or quantize it. Modes, phases, and optimizations are recorded
in the report. They never create a new workload name, because if they did, no
two results would be comparable.

**A profile is why you ran it, not how hard it is.**

| Profile | Use it for | What a result can claim |
|:---|:---|:---|
| `min` | install checks, pre-labs, CI | functional only, never a score |
| `max` | the comparable reference path | the inherited quality gate |
| `pro` | ablations and research variants | only what its recorded configuration supports |

The workload, model, data, evaluator, and gate are identical across all three.

**Every run carries its own evidence.** Each produces a `*_report.json`, a CSV
and HTML view, and a `*.provd.json` provenance manifest binding the source,
assets, hardware, metrics, and any training checkpoint by SHA-256.

```bash
uv run mlperf verify submissions/image-review/image-classification_max.provd.json
```

Verification checks recorded bytes and lineage. It does not authenticate who
produced them, and the framework says so rather than implying that a checksum
prevents fraud.

## Read the Paper

[**mlperf-edu-paper.pdf**](paper/mlperf-edu-paper.pdf) explains the design, the
workload selection, and the results in about fifteen pages. It is committed
here so you can read it straight from a clone without installing TeX.

Its numbers come from the same registry the CLI runs, regenerated on every
build. If the paper and the suite ever disagreed, the build would fail rather
than print a stale figure.

## The Workloads

Fourteen workloads, all of which run locally. Eight reproduce their inherited
target and six are recorded misses.

| Workload | Inherited From | Mode | Gate |
|:---|:---|:---|:---|
| `image-classification` | MLPerf Tiny ResNet8, 200-example CIFAR-10 set | inference | top-1 ≥ 0.85 |
| `keyword-spotting` | MLPerf Tiny DS-CNN, 1,000-example EEMBC set | inference | top-1 ≥ 0.90 |
| `anomaly-detection` | MLPerf Tiny ToyADMOS ToyCar, 248 recordings | inference | ROC AUC ≥ 0.85 |
| `visual-wake-words` | MLPerf Tiny MobileNetV1 0.25, 1,000-example set | inference | top-1 ≥ 0.80 |
| `causal-language-modeling` | nanoGPT Shakespeare | training; full, prefill, decode | val loss ≤ 1.4697 |
| `text-classification` | DistilBERT SST-2 model card | inference | accuracy ≥ 0.91055 |
| `information-retrieval` | Sentence Transformers CrossEncoder, NanoBEIR | inference | mean nDCG@10 ≥ 0.60716 |
| `graph-node-classification` | Official OGB GCN on `ogbn-arxiv` | training | test acc ≥ 71.74% |
| `time-series-forecasting` | Official PatchTST ETTm1 recipe | training | test MSE ≤ 0.290 · **miss** |
| `recommendation` | MLPerf Training v0.5 NCF, MovieLens-20M | training | HR@10 ≥ 0.635 · **miss** |
| `code-generation` | Qwen2.5-Coder, HumanEval+ | inference | pass@1 ≥ 0.573 · **miss** |
| `function-calling` | Qwen3-1.7B, BFCL V4 non-live AST | inference | AST acc ≥ 82.92% · **miss** |
| `image-generation` | NVIDIA EDM, three-trial CIFAR-10 FID | inference | FID ≤ 1.79 · **miss** |
| `reinforcement-learning` | MLPerf Training v0.5 MiniGo | training | move prediction ≥ 0.40 · **miss** |

The misses are the interesting part. Each one is a place where a laptop
implementation of a real contract did not reach the published number, recorded
rather than rescued. The [selection ledger](registry/selection-ledger.yaml)
gives the authority, rationale, and evidence for every task, and
[MISS_DIAGNOSIS.md](docs/internal/MISS_DIAGNOSIS.md) investigates each shortfall.

Running locally is not the same as carrying committed reference evidence, and
the difference matters when you cite a number. The retained evidence snapshot
covers nine workloads across twelve evidence cases, counting training and the
full, prefill, and decode inference phases of `causal-language-modeling`
separately. The other five workloads execute their contract end to end but are
held at a functional stage, so they produce results you can inspect and
reproduce, not baselines you should quote. Every workload in the suite remains
experimental until external review.

## Where To Go Next

| If you want to | Read |
|:---|:---|
| A guided setup with more detail | [Getting Started](site/getting-started.qmd) |
| Understand suites, profiles, modes, and phases | [Running Benchmarks](site/guide/running.qmd) |
| Fix a failing run | [Troubleshooting](site/guide/troubleshooting.qmd) |
| Teach with it | [For Instructors](site/guide/instructors.qmd) |
| Run controlled experiments | [Research Guide](site/guide/research.qmd) |
| Work through a notebook | [Tutorial 01](tutorials/README.md) |
| See classroom labs | [Examples](examples/README.md) |
| Look up a command or variable | [CLI](site/reference/cli.qmd) · [Environment](site/reference/environment.qmd) |
| Install without `uv` | [INSTALL.md](INSTALL.md) |

The same pages render as a browsable site with `quarto render site`.

## Alternate Installs

The quickstart uses a locked `uv` environment from a source checkout, which is
the supported path for classrooms and artifact evaluation. [INSTALL.md](INSTALL.md)
covers the tool install, the wheel build, and the offline and air-gapped paths.
No package-index release is published yet.

## For Maintainers and Reviewers

```bash
uv run pytest
uv run mlperf validate smoke --output-dir submissions/smoke
uv run python tools/generate_docs.py --check
uv run python tools/export_flat_registry.py --check
uv run make -C paper clean all check
```

- [Specification](SPEC.md) · [Public result rules](PUBLIC_RULES.md) · [Security boundary](SECURITY_REVIEW.md)
- [Status and open blockers](docs/internal/STATUS.md) · [Quality target review](docs/internal/QUALITY_TARGET_REVIEW.md)
- [Independent audit](docs/internal/INDEPENDENT_AUDIT.md) · [Dataset release review](docs/internal/DATASET_RELEASE_REVIEW.md)
- [Miss diagnosis](docs/internal/MISS_DIAGNOSIS.md), which is where each recorded miss is investigated

The component license, package publication, dataset redistribution, and
MLCommons naming decisions remain external release gates. The implementation
fails closed while they are pending.
