"""Dataset series behind the book's trend figures (added 2026-10-12).

A series is a multi-row, multi-field snapshot of a public dataset (for example,
Epoch AI's hardware database or a paper's results table). The scalar-anchor YAML
schema cannot express that shape, so each series is a compact CSV snapshot under
``literature/data/series/`` paired here with its provenance. Rows are stored as
fetched or transcribed. Filtering and fitting happen in the chapter cell that
draws the figure, so the snapshot remains the auditable record.

Usage::

    from mlsysim import Literature
    for row in Literature.Series.TestSetLabelErrors.entries:
        ...
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..core.provenance import Provenance, ProvenanceKind

_DIR = Path(__file__).parent / "data" / "series"


class TrendSeries:
    """A provenance-carrying table loaded from ``data/series/<stem>.csv``.

    ``numeric`` names the columns parsed as floats; empty cells become ``None``.
    """

    def __init__(self, stem: str, provenance: Provenance, doc: str, numeric: tuple[str, ...] = ()):
        self.name = stem
        self.provenance = provenance
        self.__doc__ = doc
        rows = []
        with open(_DIR / f"{stem}.csv", newline="") as f:
            for raw in csv.DictReader(f):
                row = {}
                for k, v in raw.items():
                    if k in numeric:
                        row[k] = float(v) if v not in ("", None) else None
                    else:
                        row[k] = v
                rows.append(row)
        self.entries = tuple(rows)

    def column(self, key: str) -> list:
        """Return one column, in file order."""
        return [r[key] for r in self.entries]

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"TrendSeries({self.name!r}, rows={len(self.entries)})"


def _lit(ref: str, url: str, notes: str | None = None) -> Provenance:
    return Provenance(kind=ProvenanceKind.LITERATURE, ref=ref, url=url, verified="2026-10-12", notes=notes)


class Series:
    """Namespace of dataset series (see module docstring)."""

    MicroprocessorTrends = TrendSeries(
        "microprocessor_trends",
        _lit("Karl Rupp, 50 Years of Microprocessor Trend Data (CC BY 4.0), 50yrs/*.dat",
             "https://github.com/karlrupp/microprocessor-trend-data",
             "series: transistors (thousands), specint (x1e3), frequency (MHz), watts, cores; 1971-2021"),
        "Microprocessor transistor count, single-thread performance, frequency, power, and cores by year.",
        numeric=("year", "value"),
    )
    StoragePricePerTB = TrendSeries(
        "storage_price_per_tb",
        _lit("Our World in Data, Historical price of computer memory and storage (CC BY 4.0), "
             "from John C. McCallum (2023) deflated with the U.S. BLS CPI",
             "https://ourworldindata.org/grapher/historical-cost-of-computer-memory-and-storage",
             "constant 2020 US$ per TB; each value is the lowest price recorded up to that year"),
        "Price per terabyte of DRAM, flash, solid-state drives, and hard disk drives, 1956-2023.",
        numeric=("year", "dram", "flash", "ssd", "hdd"),
    )
    TestSetLabelErrors = TrendSeries(
        "test_set_label_errors",
        _lit("Northcutt, Athalye, and Mueller (2021), Pervasive Label Errors in Test Sets Destabilize "
             "Machine Learning Benchmarks, Table 1", "https://arxiv.org/abs/2103.14749",
             "errors are human-validated except QuickDraw and Amazon Reviews (estimated=1), which "
             "extrapolate from a validated sample; ImageNet uses the ILSVRC 2012 validation set"),
        "Share of benchmark test-set labels found to be wrong, by dataset.",
        numeric=("test_size", "errors", "pct_error", "estimated"),
    )
    NotableModels = TrendSeries(
        "notable_models",
        _lit("Epoch AI, Notable AI Models database (CC BY 4.0), snapshot 2026-10-12",
             "https://epoch.ai/data/notable-ai-models",
             "training_cost_2023usd follows Cottier et al. (2024); frontier = Epoch's frontier-model flag"),
        "Parameters, training cost, domain, and publication date of notable models.",
        numeric=("parameters", "training_cost_2023usd", "frontier"),
    )
    PretrainingTokens = TrendSeries(
        "pretraining_tokens",
        _lit("Pre-training token counts as reported in each model's paper or release (see source column)",
             "https://arxiv.org/abs/2211.04325",
             "url points to Villalobos et al. (2024), whose stock estimate (about 3e14 tokens, 90% CI "
             "1e14-1e15) the figure overlays; per-model sources are in the CSV"),
        "Reported pre-training tokens for landmark language models, 2020-2025.",
        numeric=("tokens",),
    )
    PublicTextStock = TrendSeries(
        "public_text_stock",
        _lit("Villalobos et al. (2024), Will We Run Out of Data? Limits of LLM Scaling Based on "
             "Human-Generated Data (arXiv v2)", "https://arxiv.org/abs/2211.04325",
             "effective stock of quality- and repetition-adjusted public human text: median about "
             "3e14 tokens, 90% confidence interval 1e14 to 1e15"),
        "Estimated stock of public human-generated text (tokens): low, median, and high bounds.",
        numeric=("tokens",),
    )
    MMLUDenseModels = TrendSeries(
        "mmlu_dense_models",
        _lit("Epoch AI Benchmarking Hub, MMLU (external results) (CC BY 4.0), snapshot 2026-10-12",
             "https://epoch.ai/benchmarks",
             "parameter counts parsed from dense model names (for example '7B'); mixture-of-experts "
             "names excluded; scores as reported by each row's source"),
        "MMLU accuracy and parameter count of dense language models by release date.",
        numeric=("parameters", "mmlu"),
    )
    AcceleratorBandwidthCompute = TrendSeries(
        "accelerator_bandwidth_compute",
        _lit("Epoch AI, Machine Learning Hardware database (CC BY 4.0), snapshot 2026-10-12",
             "https://epoch.ai/data/machine-learning-hardware",
             "devices with >= 16 GB off-chip memory and a dense FP16/BF16 tensor rate; MI308X excluded "
             "(listed at 10.3 TB/s although it uses MI300X's 5.3 TB/s HBM3 stack)"),
        "Memory bandwidth and dense FP16/BF16 tensor throughput of ML accelerators by release date.",
        numeric=("memory_bandwidth_bytes_per_s", "fp16_tensor_flops", "memory_bytes"),
    )
    TemporalShiftWildTime = TrendSeries(
        "temporal_shift_wildtime",
        _lit("Yao et al. (2022), Wild-Time: A Benchmark of in-the-Wild Distribution Shift over Time, "
             "Table 1 (ERM, Eval-Fix, mean and std over three seeds)", "https://arxiv.org/abs/2211.14238"),
        "Performance on test data from the training period versus a later period, six datasets.",
        numeric=("id_mean", "id_std", "ood_mean", "ood_std"),
    )
    PyTorchOperatorSchemas = TrendSeries(
        "pytorch_operator_schemas",
        _lit("PyTorch aten/src/ATen/native/native_functions.yaml at each release tag (mined 2026-10-12)",
             "https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml",
             "aten_schemas counts '- func:' entries including overloads; core_aten counts schemas tagged "
             "'core' (Core ATen IR), first tagged in v2.0; dates are GitHub release dates"),
        "PyTorch ATen operator schemas and Core ATen IR size by release.",
        numeric=("aten_schemas", "core_aten"),
    )
    TensorFlowOperators = TrendSeries(
        "tensorflow_operators",
        _lit("TensorFlow tensorflow/core/ops/ops.pbtxt at each release tag (mined 2026-10-12)",
             "https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/ops.pbtxt",
             "ops counts registered op definitions ('op {' entries); dates are GitHub release dates; "
             "v1.15 omitted because it shipped after v2.0"),
        "Registered TensorFlow operators by release.",
        numeric=("ops",),
    )
    XLAHLOOpcodes = TrendSeries(
        "xla_hlo_opcodes",
        _lit("XLA hlo_opcode.h (HLO_OPCODE_LIST) as vendored in TensorFlow at each release tag (mined 2026-10-12)",
             "https://github.com/openxla/xla/blob/main/xla/hlo/ir/hlo_opcode.h",
             "opcodes counts V(k...) entries; XLA is the compiler that both TensorFlow and JAX lower to"),
        "Number of XLA HLO opcodes by TensorFlow release.",
        numeric=("opcodes",),
    )
    JAXLaxPrimitives = TrendSeries(
        "jax_lax_primitives",
        _lit("JAX jax/_src/lax/**/*.py at each release tag (mined 2026-10-12)",
             "https://github.com/jax-ml/jax/tree/main/jax/_src/lax",
             "primitives counts module-level '<name>_p = ...(' primitive objects, JAX's naming convention"),
        "Number of jax.lax primitives by JAX release.",
        numeric=("primitives",),
    )
    ONNXOpsetOperators = TrendSeries(
        "onnx_opset_operators",
        _lit("ONNX operator schemas with history (onnx 1.23.2, ai.onnx domain) and docs/Versioning.md",
             "https://github.com/onnx/onnx/blob/main/docs/Versioning.md",
             "operators counts distinct ai.onnx operators whose first since_version <= opset; "
             "date is the GitHub release that introduced the opset"),
        "Number of standard ONNX operators defined through each opset release.",
        numeric=("opset", "operators"),
    )
    InferenceXDeepSeekR1 = TrendSeries(
        "inferencex_dsr1_fp8_1k1k",
        _lit("SemiAnalysis InferenceX (formerly InferenceMAX), open benchmark results (Apache-2.0), "
             "DeepSeek-R1-0528, FP8, 1,024 input / 1,024 output tokens, snapshot 2026-10-12",
             "https://github.com/SemiAnalysisAI/InferenceX",
             "one row per measured serving configuration (hardware, framework, parallelism, concurrency), "
             "latest published curve per configuration from inferencex.semianalysis.com/api/v1/benchmarks; "
             "interactivity is median output tokens/s per user (1/TPOT); throughputs are per GPU; "
             "run_url links the GitHub Actions run that produced each row"),
        "Measured per-GPU throughput versus per-user interactivity for LLM serving configurations.",
        numeric=("isl", "osl", "conc", "disagg", "num_prefill_gpu", "num_decode_gpu", "interactivity_tok_s_user",
                 "output_tok_s_gpu", "total_tok_s_gpu", "joules_per_output_token"),
    )
    AcceleratorPowerRatings = TrendSeries(
        "accelerator_power_ratings",
        _lit("Published accelerator power ratings from vendor white papers, datasheets, and architecture "
             "papers (per-row source_key and source_note); transcription verified 2026-09-29",
             "https://doi.org/10.1145/3079856.3080246",
             "ratings mix TDP, maximum board power, maximum TGP, and maximum power as each vendor states "
             "them (power_metric column); where a part ships in several configurations (MI250X: 500 W and "
             "560 W) the base configuration is listed; they are design envelopes, not measured workload power; parts "
             "without a citable published rating (MI100, Gaudi 1, TPU v4 onward, wafer-scale) are omitted; "
             "url points to the TPU v1 paper, the earliest row's architecture source"),
        "Published power rating per accelerator package by release year, 2012-2025.",
        numeric=("release_year", "power_w"),
    )
    AISupercomputers = TrendSeries(
        "ai_supercomputers",
        _lit("Epoch AI, Data on AI Supercomputers (CC BY 4.0), snapshot 2026-10-12; method of Pilz et al. "
             "(2025), Trends in AI Supercomputers", "https://epoch.ai/data/ai-supercomputers",
             "existing single-cluster systems in Epoch's standard-analysis set; h100_equivalents is Epoch's "
             "performance-normalized size; power_mw is facility power as Epoch records or estimates it; "
             "some Chinese systems are anonymized and rounded; two owner names corrected from "
             "'Univeristy' to 'University'"),
        "Size, chip count, and power of AI supercomputers by first operational date.",
        numeric=("h100_equivalents", "total_chips", "power_mw"),
    )
    NotableModelTrainingRuns = TrendSeries(
        "notable_model_training_runs",
        _lit("Epoch AI, Notable AI Models database (CC BY 4.0), snapshot 2026-10-12",
             "https://epoch.ai/data/notable-ai-models",
             "models published 2012 onward with a training-compute or hardware-quantity estimate; values "
             "as published by Epoch, many of them Epoch estimates; hardware_quantity is accelerator count, "
             "training_time_hours is wall-clock duration, mfu as reported; frontier = Epoch's frontier flag"),
        "Training compute, accelerator count, duration, and utilization of notable training runs.",
        numeric=("training_compute_flop", "hardware_quantity", "training_time_hours", "mfu", "frontier"),
    )
    MLPerfStorageV1Training = TrendSeries(
        "mlperf_storage_v1_training",
        _lit("MLCommons, MLPerf Storage v1.0 results (Apache-2.0), published 2024-09-25, snapshot 2026-10-12",
             "https://github.com/mlcommons/storage_results_v1.0",
             "one row per submitted system x workload x emulated accelerator, aggregated over its runs; "
             "accelerators are emulated by DLIO with a fixed per-sample compute time, not real GPUs; "
             "io_GBps is mean training read throughput; au_check is the accelerator-utilization pass "
             "(>= 90% for unet3d and resnet50, >= 70% for cosmoflow)"),
        "Storage throughput sustained per MLPerf Storage training submission by emulated accelerator count.",
        numeric=("num_accelerators", "num_hosts", "num_runs", "au_mean_pct", "io_GBps"),
    )
    MLPerfTrainingLLMScaling = TrendSeries(
        "mlperf_training_llm_scaling",
        _lit("MLCommons, MLPerf Training results v3.1, v4.1, v5.0, v5.1 (Apache-2.0), snapshot 2026-10-12",
             "https://mlcommons.org/benchmarks/training/",
             "one row per system x LLM benchmark (gpt3 = GPT-3 175B pretraining slice; llama31_405b); "
             "minutes recomputed from MLLOG run_start to run_stop of successful runs (olympic mean when "
             ">= 5 runs), so they can differ slightly from official RCP-normalized scores; global batch "
             "size varies with scale, so time-to-train mixes strong scaling with batch-dependent convergence"),
        "Time-to-train of MLPerf Training LLM benchmarks by accelerator count and system.",
        numeric=("accelerators", "runs", "min_minutes", "mean_minutes", "score_minutes", "global_batch_size", "trained_samples_median"),
    )
    Llama3Interruptions = TrendSeries(
        "llama3_interruptions",
        _lit("Llama Team, AI @ Meta (2024), The Llama 3 Herd of Models, arXiv:2407.21783 v3, Table 5 "
             "(root causes of unexpected interruptions over a 54-day Llama 3 405B pre-training period)",
             "https://arxiv.org/abs/2407.21783",
             "transcribed exactly; counts sum to 419, matching the text; the published percentage for "
             "Faulty GPU (30.1) disagrees with 148/419 (35.3) while every other row matches its count, so "
             "derive shares from interruption_count"),
        "Root-cause component and category of each unexpected Llama 3 405B training interruption.",
        numeric=("interruption_count", "published_pct"),
    )
    AcmeKalosJobSizes = TrendSeries(
        "acme_kalos_job_sizes",
        _lit("Hu et al. (2024), Characterization of Large Language Model Development in the Datacenter "
             "(NSDI 24); AcmeTrace data/job_trace/trace_kalos.csv (CC BY 4.0), snapshot 2026-10-12",
             "https://github.com/InternLM/AcmeTrace",
             "Kalos cluster (Shanghai AI Laboratory, A100, March to August 2023); GPU jobs aggregated "
             "by requested gpu_num; gpu_seconds as given in the trace, split by final job state"),
        "Job count and GPU time by requested GPUs per job in an LLM-development cluster.",
        numeric=("gpu_num", "jobs", "job_share", "gpu_seconds", "gpu_time_share", "gpu_seconds_completed",
                 "gpu_seconds_failed", "gpu_seconds_canceled"),
    )
    MLPerfInferenceDGXH100 = TrendSeries(
        "mlperf_inference_dgx_h100",
        _lit("MLCommons, MLPerf Inference datacenter results v3.1, v4.0, v4.1, v5.0 (Apache-2.0), NVIDIA "
             "closed-division DGX-H100_H100-SXM-80GBx8_TRT system, snapshot 2026-10-12",
             "https://mlcommons.org/benchmarks/inference-datacenter/",
             "same 8x H100-SXM-80GB hardware in every round; software version per the round's system "
             "description; values from each run's mlperf_log_summary.txt, all VALID; llama2-70b at the "
             "99.9 accuracy target; pub_month is the round's publication month"),
        "Throughput of one fixed eight-H100 system across MLPerf Inference rounds as its software changed.",
        numeric=("value",),
    )
    AcceleratorFlopsPerWatt = TrendSeries(
        "accelerator_flops_per_watt",
        _lit("Epoch AI, Machine Learning Hardware database (CC BY 4.0), snapshot 2026-10-12",
             "https://epoch.ai/data/machine-learning-hardware",
             "accelerators with both a dense FP16/BF16 tensor rate and a TDP, released 2017 onward; "
             "flops_per_w = fp16_tensor_flops / tdp_w, a peak-arithmetic ratio, not measured energy per "
             "workload; board variants of one chip appear as separate rows, as in Epoch's table"),
        "Dense FP16 tensor throughput per watt of TDP for ML accelerators by release date.",
        numeric=("fp16_tensor_flops", "tdp_w", "flops_per_w"),
    )
    ImageNetTestbedShift = TrendSeries(
        "imagenet_testbed_shift",
        _lit("Taori et al. (2020), Measuring Robustness to Natural Distribution Shifts in Image "
             "Classification, testbed results robustness_top1s.csv (MIT), snapshot 2026-10-12; ImageNet-V2 "
             "from Recht et al. (2019)", "https://github.com/modestyachts/imagenet-testbed",
             "top-1 accuracy (%) per model on the ImageNet validation set (val), ImageNetV2 matched-frequency, "
             "ObjectNet, ImageNet-R, and ImageNet-Sketch; the testbed includes partially trained and low-data "
             "models, so figures filter on val accuracy; empty cells mean not evaluated"),
        "Top-1 accuracy of ImageNet models on the original test set and on shifted test sets.",
        numeric=("val", "imagenet_v2", "objectnet", "imagenet_r", "imagenet_sketch"),
    )
    RobustBenchCIFAR10Linf = TrendSeries(
        "robustbench_cifar10_linf",
        _lit("Croce et al. (2021), RobustBench, model_info/cifar10/Linf/*.json (MIT), snapshot 2026-10-12",
             "https://github.com/RobustBench/robustbench",
             "CIFAR-10, L-infinity threat model, epsilon 8/255; clean_acc and autoattack_acc in percent as "
             "evaluated by RobustBench with AutoAttack; additional_data marks training beyond CIFAR-10; "
             "unreliable is RobustBench's flag (empty when not set); year parsed from the venue string"),
        "Clean and AutoAttack accuracy of published CIFAR-10 L-infinity defenses by publication year.",
        numeric=("clean_acc", "autoattack_acc"),
    )
    GenderShadesAudit = TrendSeries(
        "gender_shades_audit",
        _lit("Buolamwini and Gebru (2018), Gender Shades, PMLR 81, Table 4 (error rate, audit of "
             "April-May 2017); Raji and Buolamwini (2019), Actionable Auditing, AIES, Table 1 (re-audit "
             "of August 2018); Pilot Parliaments Benchmark", "https://proceedings.mlr.press/v81/buolamwini18a.html",
             "gender-classification error rate (%) by subgroup; target = 1 for the three companies named "
             "in the 2018 publication, 0 for companies measured only in the re-audit; the re-audit's "
             "Table 2 differences reproduce the original darker-female values for all three targets"),
        "Gender-classification error by skin-type and gender subgroup, before and after a public audit.",
        numeric=("target", "error_may2017_pct", "error_aug2018_pct"),
    )
    AIBenchmarkSoCs = TrendSeries(
        "ai_benchmark_socs",
        _lit("ETH Zurich AI Benchmark, mobile processor ranking (ai-benchmark.com), snapshot 2026-10-12; "
             "method of Ignatov et al. (2019)", "https://ai-benchmark.com/ranking_processors.html",
             "one row per mobile SoC; ai_score is the benchmark's aggregate inference score; subscores for "
             "INT8 and FP16 CNN and transformer workloads; lib is the software delegate used, so scores "
             "reflect each chip's software path as well as its silicon; year is the SoC release year; "
             "published scores cited as facts, no redistribution license stated by the source"),
        "Measured on-device AI inference scores of mobile systems-on-chip by release year.",
        numeric=("year", "int8_cnns", "int8_transformer", "int8_accuracy", "fp16_cnns", "fp16_transformer",
                 "fp16_accuracy", "int16_cnns", "int8_parallel", "fp16_parallel", "ai_score"),
    )
    WaymoRiderOnlyMiles = TrendSeries(
        "waymo_rider_only_miles",
        _lit("Waymo Safety Impact Data Hub, rider-only mileage releases June 2024 to June 2026 "
             "(published data cited as facts; no redistribution license stated)", "https://waymo.com/safety/impact/",
             "cumulative rider-only (no human driver) miles in millions; early releases summed per-location "
             "rows, later releases report an All Locations row; the 2025-09 value matches the release "
             "note's 127 million"),
        "Cumulative rider-only miles driven by Waymo's driverless fleet.",
        numeric=("cumulative_ro_miles_millions",),
    )
    WaymoVsHumanBenchmark = TrendSeries(
        "waymo_vs_human_benchmark",
        _lit("Waymo Safety Impact Data Hub, crash rates versus human benchmarks through June 2026, method of "
             "Kusano et al. (2024)", "https://waymo.com/safety/impact/",
             "incidents per million miles (ipmm) with 95% confidence intervals for Waymo rider-only driving "
             "and for Waymo's human-driver benchmark on the same roads; benchmark variants as published "
             "(reported, observed, dynamic); reduction is Waymo's published relative difference"),
        "Crash outcomes per million miles for Waymo rider-only driving and a human-driver benchmark.",
        numeric=("waymo_count", "waymo_ipmm", "waymo_lo", "waymo_hi", "bench_ipmm", "bench_lo", "bench_hi", "reduction"),
    )
