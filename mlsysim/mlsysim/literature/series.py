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
             "them (power_metric column); they are design envelopes, not measured workload power; parts "
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
