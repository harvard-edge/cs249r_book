"""Published measurement series (multi-row, multi-field) cited by the books.

The scalar-anchor YAML schema in ``literature/data/*.yaml`` cannot express a
table of results, so each series here is a CSV under ``literature/data/series/``
plus one ``Provenance`` record. The CSVs are curated extracts of the source's
own published file: no value is typed by hand, and the extraction scripts and
raw downloads are recorded in each record's ``notes``.

Each series exposes ``provenance``, ``columns``, and ``entries`` (a tuple of
dicts, numeric fields as ``float``, empty cells as ``None``), plus ``list`` for
parity with the ``Registry`` convention. ``where(**match)`` filters entries on
exact field values.
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..core.provenance import Provenance, ProvenanceKind

_SERIES = Path(__file__).parent / "data" / "series"

_EPOCH_HUB = "https://epoch.ai/benchmarks"
_EPOCH_LICENSE = "Epoch AI data, CC-BY 4.0 (credit Epoch AI)."


def _coerce(value: str):
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        return float(value)
    except ValueError:
        return value


class _Series:
    """A provenance-carrying table loaded from one CSV."""

    def __init__(self, stem: str, provenance: Provenance, doc: str):
        self.__doc__ = doc
        self.provenance = provenance
        with open(_SERIES / f"{stem}.csv", newline="") as fh:
            reader = csv.DictReader(fh)
            self.columns = tuple(reader.fieldnames or ())
            self.entries = tuple({k: _coerce(v) for k, v in row.items()} for row in reader)
        self.list = self.entries

    def where(self, **match):
        return tuple(e for e in self.entries if all(e.get(k) == v for k, v in match.items()))

    def __len__(self):
        return len(self.entries)


AgentTimeHorizon = _Series(
    "metr_time_horizon_v1_1",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="METR, Time Horizon v1.1 benchmark results (50% task-completion horizon, minutes of expert human time)",
        url="https://metr.org/assets/benchmark_results_1_1.yaml",
        verified="2026-10-12",
        notes=(
            "Copied from METR's published YAML (benchmark METR-Horizon-v1.1). "
            "METR's stated 2023-on doubling time is 128.744 days (CI 104.4-158.0), "
            "fit on frontier models with p50 <= 16 h; refitting these rows the same "
            "way reproduces it."
        ),
    ),
    "50%-success task horizon by model release (METR Time Horizon v1.1).",
)

LowestPriceAtCapability = _Series(
    "epoch_lowest_price_at_capability",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="Epoch AI, LLM inference price trends: cheapest model above each benchmark threshold",
        url="https://epoch.ai/data-insights/llm-inference-price-trends",
        verified="2026-10-12",
        notes=(
            "Chart data lowest_price_models_data.csv. Prices are a 3:1 weighted "
            "average of input and output token prices, first-party API where one "
            "exists, otherwise the median across providers (Epoch methodology). "
            "Series end February 2025. " + _EPOCH_LICENSE
        ),
    ),
    "Cheapest price (USD per million tokens) to reach a fixed benchmark level, by date.",
)

TerminalBenchHarness = _Series(
    "terminal_bench_2_harness",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="Terminal-Bench 2.0 leaderboard (model x agent harness), via the Epoch AI Benchmarking Hub",
        url=_EPOCH_HUB,
        verified="2026-10-12",
        notes=(
            "Rows whose source is the Terminal-Bench 2.0 leaderboard "
            "(tbench.ai/leaderboard/terminal-bench/2.0); '_unknown' suffix removed "
            "from model ids; duplicate (model, harness) rows resolved to the latest "
            "run. " + _EPOCH_LICENSE
        ),
    ),
    "Terminal-Bench 2.0 accuracy for the same model under different agent harnesses.",
)

LongContextComprehension = _Series(
    "fiction_livebench_by_length",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="Fiction.LiveBench long-context comprehension scores by context length, via the Epoch AI Benchmarking Hub",
        url=_EPOCH_HUB,
        verified="2026-10-12",
        notes="Scores copied per length column (0 to 192k tokens). " + _EPOCH_LICENSE,
    ),
    "Long-context comprehension accuracy by tokens of story placed in context.",
)

SweBenchVerifiedRuns = _Series(
    "epoch_swe_bench_verified_runs",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="Epoch AI evaluations of SWE-bench Verified under one harness (mean score and standard error)",
        url=_EPOCH_HUB,
        verified="2026-10-12",
        notes="One row per model, highest-scoring run kept. " + _EPOCH_LICENSE,
    ),
    "SWE-bench Verified resolve fraction with standard error, one harness.",
)

CybenchSolveRate = _Series(
    "cybench_unguided",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="Cybench unguided solve rate on 40 professional-level CTF tasks, via the Epoch AI Benchmarking Hub",
        url="https://cybench.github.io/",
        verified="2026-10-12",
        notes=(
            "Task count from cybench.github.io. The source column separates "
            "leaderboard entries from lab-reported model/system-card figures. "
            + _EPOCH_LICENSE
        ),
    ),
    "Fraction of Cybench CTF tasks solved without subtask hints, by model release.",
)

OSWorldStepBudget = _Series(
    "osworld_step_budget",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="OSWorld leaderboard results for the same model at 15, 50, and 100 step limits, via the Epoch AI Benchmarking Hub",
        url="https://os-world.github.io/",
        verified="2026-10-12",
        notes="Step limit parsed from the leaderboard agent label '(N steps)'. " + _EPOCH_LICENSE,
    ),
    "OSWorld task success (percent) by step ceiling, per model.",
)

FrontierSweTrials = _Series(
    "frontier_swe_v2_trials",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="FrontierSWE V2 leaderboard: mean, best, and worst of 5 trials, cost and duration per trial",
        url="https://www.frontierswe.com/",
        verified="2026-10-12",
        notes=(
            "34 tasks, 5 trials per task, 20-hour budget per trial; cost and time "
            "are per-trial averages (frontierswe.com). Values via the Epoch AI "
            "Benchmarking Hub. " + _EPOCH_LICENSE
        ),
    ),
    "Long-horizon software tasks: score spread over 5 trials, cost and hours per trial.",
)

SweBenchTrajectories = _Series(
    "swe_bench_verified_trajectories",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="SWE-bench Verified submission trajectories (SWE-agent, OpenHands, mini-SWE-agent), per-trajectory statistics",
        url="https://github.com/SWE-bench/experiments",
        verified="2026-10-12",
        notes=(
            "Five public submissions: 2,100 trajectories (500 each; the SWE-agent 1.1 "
            "run is a seed-0 random sample of 100). Trajectory files from the public "
            "swe-bench-submissions S3 bucket named in each submission's metadata; "
            "resolved flags from results/results.json at repo commit 40f164d5. "
            "Sizes are characters (no tokenizer), classified by message role: "
            "model output, environment observations, and prompt (system, "
            "demonstrations, task). Token and cost columns are copied as each "
            "agent recorded them; OpenHands records none."
        ),
    ),
    "Per-trajectory model calls, role-split characters, tokens, and cost for public repair-agent runs.",
)

# --- One mini-SWE-agent run, per task (added 2026-10-20) ---------------------
SweBenchMiniSonnet4Run = _Series(
    "swe_bench_verified_mini_v1_sonnet4_run",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref=(
            "SWE-bench Verified submission 20250726_mini-v1.0.0_claude-sonnet-4-20250514 "
            "(mini-swe-agent 1.0.0 with Claude 4 Sonnet, bash only, one attempt per task), per task"
        ),
        url=(
            "https://github.com/SWE-bench/experiments/tree/main/evaluation/verified/"
            "20250726_mini-v1.0.0_claude-sonnet-4-20250514"
        ),
        verified="2026-10-17",
        notes=(
            "500 tasks. Cost, model calls, and resolved flag from the submission's "
            "per_instance_details.json; exit status, observations, nonzero exit codes, "
            "truncation warnings, syntax errors seen, command classes (read, write, run, "
            "net, by command pattern), and files in the submitted diff from the public "
            "trajectories (swe-bench-submissions S3 bucket, bash-only/<submission>/trajs). "
            "Reduced by extract_run.py (slides tree, images/ of the deck that plots it); command "
            "classes are pattern matches and so approximate."
        ),
    ),
    "Per-task cost, model calls, acceptance, and command classes of one public repair-agent run.",
)

# Model calls and agent trajectories (added 2026-10-20)
LlmInferenceTraceCode2023 = _Series(
    "azure_llm_inference_code_2023",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="Azure LLM inference trace 2023, code (Patel et al., Splitwise, ISCA 2024)",
        url="https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md",
        verified="2026-10-18",
        notes=(
            "The released file AzureLLMInferenceTrace_code.csv, unmodified: one row per request "
            "(timestamp, context tokens, generated tokens), 8,819 requests on 2023-11-16 to a "
            "production code-completion service. Raw: raw.githubusercontent.com/Azure/"
            "AzurePublicDataset/master/data/AzureLLMInferenceTrace_code.csv. License CC BY 4.0."
        ),
    ),
    "Prompt (context) and output (generated) tokens of production code-completion requests.",
)

_MONKEYS_NOTES = (
    "Brown et al. (2024), Large Language Monkeys. Recomputed from the released "
    "per-sample correctness labels: Hugging Face ScalingIntelligence/monkey_business "
    "@ a9f8f73b (10,000 samples per problem) and github.com/ScalingIntelligence/"
    "swe-bench-lite-samples @ 17594ed2 (250 samples per problem; crashed samples "
    "count as incorrect, per its README). Coverage is the unbiased pass@k estimator "
    "averaged over problems."
)

RepeatedSamplingCoverage = _Series(
    "repeated_sampling_coverage",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="Large Language Monkeys repeated-sampling datasets: coverage (pass@k) vs. samples",
        url="https://arxiv.org/abs/2407.21787",
        verified="2026-10-12",
        notes=_MONKEYS_NOTES,
    ),
    "Fraction of problems with at least one correct sample among k, by model and dataset.",
)

RepeatedSamplingMajority = _Series(
    "repeated_sampling_majority",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="Large Language Monkeys repeated-sampling datasets: majority-vote accuracy vs. samples (GSM8K, MATH)",
        url="https://arxiv.org/abs/2407.21787",
        verified="2026-10-12",
        notes=(
            _MONKEYS_NOTES + " Majority vote pools final answers extracted with the "
            "authors' GSM8K regex and the lm-eval Minerva MATH normalizer; small k is "
            "a Monte Carlo estimate over 200 draws. Normalization can only split "
            "equivalent answers, so these accuracies are if anything low."
        ),
    ),
    "Majority-vote accuracy alongside coverage at the same k (GSM8K, MATH).",
)

InjectionDefenseResults = _Series(
    "agentdojo_results",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="AgentDojo published results: benign utility, utility under attack, and targeted attack success by model and defense",
        url="https://agentdojo.spylab.ai/results/",
        verified="2026-10-12",
        notes=(
            "Parsed from docs/results-table.html at github.com/ethz-spylab/agentdojo @ "
            "089ed468 (identical to the live results page). The authors state it is "
            "not a leaderboard: only gpt-4o-2024-05-13 was run with every defense and "
            "attack; other models use the important_instructions attack and no "
            "defense. Release dates are omitted because one source date is a typo."
        ),
    ),
    "Prompt-injection benchmark: utility and attack success per model and defense.",
)

# ---------------------------------------------------------------------------
# Physical AI series (added 2026-10-12)
# ---------------------------------------------------------------------------

ModelsByDomain = _Series(
    "epoch_models_by_domain",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="Epoch AI, Data on AI Models (all_ai_models.csv): training compute by domain (Robotics, Language, Vision)",
        url="https://epoch.ai/data/ai-models",
        verified="2026-10-12",
        notes=(
            "Rows with a training-compute estimate only. A model tagged with several "
            "domains appears once per matching domain_group. Compute values are Epoch's "
            "estimates (confidence column), not first-party measurements; Epoch tags NVIDIA "
            "Cosmos world models as Robotics. " + _EPOCH_LICENSE
        ),
    ),
    "Training compute (FLOP) and publication date of AI models, by domain group.",
)

OpenXEmbodimentComposition = _Series(
    "open_x_embodiment_composition",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="Open X-Embodiment dataset overview spreadsheet: episodes, robot, morphology, control frequency per constituent dataset",
        url="https://github.com/google-deepmind/open_x_embodiment",
        verified="2026-10-12",
        notes=(
            "Exported from the official Google Sheet linked in the repository README "
            "(repo @ 9eeb68b9). 72 datasets; episode total 2,419,193 matches the sheet's own "
            "total. The sheet is a live document and cannot be pinned to a revision."
        ),
    ),
    "Composition of the Open X-Embodiment robot-learning collection.",
)

LeRobotHubMonthly = _Series(
    "lerobot_hub_monthly",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="Hugging Face Hub API: datasets tagged LeRobot, new and cumulative count by creation month",
        url="https://huggingface.co/api/datasets?filter=LeRobot",
        verified="2026-10-12",
        notes=(
            "81 API pages (80,132 unique datasets). Counts include only datasets still public "
            "at fetch time, so earlier cumulative values are lower bounds."
        ),
    ),
    "Public LeRobot-format datasets on the Hugging Face Hub, by creation month.",
)

LeRobotEpisodeHistogram = _Series(
    "lerobot_hub_episode_histogram",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="Hugging Face Hub LeRobot datasets: episodes per dataset, power-of-two bins by creation year",
        url="https://huggingface.co/api/datasets?filter=LeRobot",
        verified="2026-10-12",
        notes=(
            "Episode counts parsed from each dataset card's embedded meta/info.json for the "
            "59,031 datasets that carry one; a 25-dataset spot check against info.json matched "
            "21 exactly (cards can lag the data). Bin b holds counts in [bin_lo, bin_hi)."
        ),
    ),
    "Distribution of episodes per public LeRobot dataset.",
)

EdgeInferenceOrin = _Series(
    "mlperf_inference_edge_orin",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="MLPerf Inference Edge, closed division, results for NVIDIA Orin-family systems (v3.1, v4.0, v4.1)",
        url="https://github.com/mlcommons/inference_results_v3.1",
        verified="2026-10-12",
        notes=(
            "Parsed from the mlcommons inference_results repositories (v3.1 @ f7b504d4, "
            "v4.0 @ 343c3d2c, v4.1 @ 198fc467), loadgen-valid results only. Single-stream "
            "latency metrics as reported (p50, p90 with early stopping)."
        ),
    ),
    "Measured inference latency and throughput on Orin-family edge systems.",
)

KittiOdometry = _Series(
    "kitti_odometry_leaderboard",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="KITTI odometry benchmark leaderboard: translation and rotation error per method",
        url="https://www.cvlibs.net/datasets/kitti/eval_odometry.php",
        verified="2026-10-12",
        notes=(
            "127 methods. sensor_group comes from the page's sensor icons: stereo, laser, or "
            "none (a mix of monocular, some lidar-inertial, and event-camera methods). "
            "Runtimes are self-reported on different hardware."
        ),
    ),
    "Odometry drift (percent of distance) by method and sensor type.",
)

CaliforniaDisengagements = _Series(
    "ca_dmv_disengagements",
    Provenance(
        kind=ProvenanceKind.INDUSTRY_REPORT,
        ref="California DMV autonomous vehicle disengagement and mileage reports, per company, report year, and permit type",
        url="https://web.archive.org/web/20260519/https://www.dmv.ca.gov/portal/vehicle-industry-services/autonomous-vehicles/disengagement-reports/",
        verified="2026-10-12",
        notes=(
            "Report year runs Dec 1 to Nov 30. 2020 onward from the DMV consolidated "
            "CSVs (per-VIN rows summed per permit); 2019 from the DMV spreadsheets; "
            "earlier years from each company's filed PDF (text layer or OCR, as "
            "source_method records). What counts as a reportable disengagement is "
            "self-defined by each company under 13 CCR 227.50, and the DMV states the "
            "reports are not designed for comparison across companies. Commercial "
            "deployment miles are excluded by regulation. The live DMV page returned 404 "
            "on the fetch date; files came from the DMV-hosted bundle the archived page links."
        ),
    ),
    "Autonomous testing miles and reported disengagements filed with the California DMV.",
)

WaymoSafetyImpact = _Series(
    "waymo_safety_impact_headline",
    Provenance(
        kind=ProvenanceKind.INDUSTRY_REPORT,
        ref="Waymo Safety Impact data hub: rider-only crash rates vs. a location-adjusted human benchmark, by outcome and data cut",
        url="https://waymo.com/safety/impact/",
        verified="2026-10-12",
        notes=(
            "Rows: All Locations (mileage blended), All Crashes, dynamic (location-adjusted) "
            "benchmark built from 2022 human crash data. pct_reduction is the rate ratio "
            "minus one; 95% confidence intervals by the rate-ratio method the data "
            "dictionary cites. The 202606 cut uses the corrected v2 file the release notes "
            "name as the replacement. ro_miles_millions is cumulative rider-only miles "
            "through the cut (before 202512, the sum of operating-depot rows)."
        ),
    ),
    "Waymo rider-only crash-rate reductions with confidence intervals, by outcome and data cut.",
)


# ---------------------------------------------------------------------------
# Trend series behind Volume I and II figures, exposed as Literature.Series.<Name>.
# Merged 2026-10-12 from a parallel branch; the module-level series above are
# exposed as Literature.<Name>. Both share data/series/.
# ---------------------------------------------------------------------------
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

    def largest(self, key: str, date_key: str | None = None, on_or_before: str | None = None) -> dict:
        """The row with the largest ``key``, optionally among rows dated on or before an ISO date."""
        rows = [r for r in self.entries if r[key] is not None
                and (on_or_before is None or r[date_key] <= on_or_before)]
        return max(rows, key=lambda r: r[key])

    def find(self, key: str, fragment: str) -> dict:
        """The first row whose ``key`` contains ``fragment``."""
        return next(r for r in self.entries if fragment in r[key])

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
    LlmApiInputPrices = TrendSeries(
        "llm_api_input_prices",
        Provenance(
            kind=ProvenanceKind.INDUSTRY_REPORT,
            ref="Vendor launch and pricing announcements (OpenAI, Anthropic, Google, DeepSeek); "
                "per-row bib key in source_key",
            url="https://www.anthropic.com/news/claude-3-family",
            verified="2026-10-13",
            notes=("Input (prompt) token list price in USD per million tokens at each model version's "
                   "announcement or effective date. The check column records, row by row, what was "
                   "re-checked on 2026-10-13: five rows against the cited page, two against the Epoch AI "
                   "lowest-price series, and two against OpenAI's API pricing page (openai.com refuses "
                   "automated fetches of the announcements). Prices are snapshots, not current rates."),
        ),
        "Public API input-token list prices for named model versions, 2022-2025.",
        numeric=("usd_per_mtok_input",),
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
    # Added 2026-10-15: the pre-2012 complement of NotableModelTrainingRuns, so a
    # long-run compute-trend figure (1950 onward, two-era fit after Sevilla et
    # al. 2022) can be drawn from Epoch's data instead of reproducing Epoch's chart.
    NotableModelComputePre2012 = TrendSeries(
        "notable_model_compute_pre2012",
        Provenance(
            kind=ProvenanceKind.LITERATURE,
            ref="Epoch AI, Notable AI Models database (CC BY 4.0), snapshot 2026-10-15",
            url="https://epoch.ai/data/notable-ai-models",
            verified="2026-10-15",
            notes=(
                "every model published before 2012-01-01 with a training-compute estimate (72 rows, "
                "1950-2011), extracted from https://epoch.ai/data/notable_ai_models.csv; values as "
                "published by Epoch (mostly Epoch estimates); frontier = Epoch's frontier flag. "
                "The 2026-10-15 download agrees with the 2026-10-12 snapshot behind "
                "NotableModelTrainingRuns on all 501 post-2012 rows but one compute value, so the two "
                "series join into one 1950-onward history"),
        ),
        "Training compute of notable models published before 2012 (complements NotableModelTrainingRuns).",
        numeric=("training_compute_flop", "frontier"),
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
    # Added 2026-10-20 (verified against the arXiv HTML on 2026-10-20).
    GoyalLargeMinibatchResNet50 = TrendSeries(
        "goyal_resnet50_large_minibatch",
        _lit("Goyal et al. (2017), Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour, "
             "arXiv:1706.02677, Table 1 and Figure 3",
             "https://arxiv.org/abs/1706.02677",
             "ResNet-50 ImageNet top-1 validation error, mean of 5 runs. source=table1: the 256 baseline "
             "and the three 8k warmup variants of Table 1 (text). source=figure3: the validation errors in "
             "the legends of Figure 3 (linear scaling rule with gradual warmup), read from the figure; "
             "256 to 16k match Table 3(a) in the text, while 128, 32k and 64k appear only in the figure"),
        "ResNet-50 ImageNet validation error by global minibatch and warmup schedule.",
        numeric=("global_batch", "top1_error_pct"),
    )
    MegatronLmWeakScaling = TrendSeries(
        "megatron_lm_weak_scaling",
        _lit("Shoeybi et al. (2019), Megatron-LM, arXiv:1909.08053, Figure 5 and Section 5.1.1",
             "https://arxiv.org/abs/1909.08053",
             "weak-scaling efficiency against the 1.2B-parameter one-GPU baseline on DGX-2H V100 servers "
             "(NVSwitch, 300 GB/s inside a server); mode=model is tensor parallelism inside one server, "
             "model+data adds data parallelism across servers. Bar values read from Figure 5; the 8-GPU "
             "77 percent and 512-GPU 74 percent are also stated in Section 5.1.1. The abstract's 76 percent "
             "(Literature.TrainingRuns.MegatronLm2019ScalingEfficiency) is sustained FLOP/s against the "
             "39 TFLOP/s single-GPU baseline, a different measure"),
        "Megatron-LM weak-scaling efficiency by GPU count, model and model-plus-data parallel.",
        numeric=("gpus", "efficiency_pct"),
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
    NCCLAllReduceH100Node = TrendSeries(
        "nccl_allreduce_h100_node",
        _lit("NVIDIA exemplar-performance, microbenchmarks/nccl README, all_reduce_perf sample output (MIT), "
             "file at v26.08.00, snapshot 2026-10-12",
             "https://github.com/NVIDIA/exemplar-performance/blob/main/microbenchmarks/nccl/README.md",
             "one node, 8x H100 80GB HBM3 over NVLink/NVSwitch; float sum, 8 B to 16 GB in factor-2 steps; "
             "default algorithm; whether NVLink SHARP (NVLS) was enabled for this excerpt is not stated; "
             "rows copied verbatim; times in microseconds, bandwidths in GB/s (busbw scales algbw by 2(N-1)/N)"),
        "Measured NCCL all-reduce time and bandwidth by message size on one eight-H100 node.",
        numeric=("size_bytes", "count", "oop_time_us", "oop_algbw_GBps", "oop_busbw_GBps", "ip_time_us", "ip_algbw_GBps", "ip_busbw_GBps"),
    )
    HorovodScaling128Gpu = TrendSeries(
        "horovod_scaling_128gpu",
        _lit("Sergeev and Del Balso (2018), Horovod, arXiv:1802.05799, scaling bar chart (Figure 7 in v3)",
             "https://arxiv.org/abs/1802.05799",
             "images per second at 128 NVIDIA Pascal GPUs over 25GbE TCP: ideal (one-GPU rate times 128), "
             "standard distributed TensorFlow, and Horovod; bar heights read from the figure to about "
             "+/-250 images/s, added 2026-10-20"),
        "Training throughput of two image models at 128 GPUs: ideal, distributed TensorFlow, and Horovod.",
        numeric=("ideal_images_per_s", "tensorflow_images_per_s", "horovod_images_per_s"),
    )
    HorovodTimelineTrace = TrendSeries(
        "horovod_timeline_trace",
        _lit("Sergeev and Del Balso (2018), Horovod, arXiv:1802.05799, Horovod Timeline screenshot (Figure 6 in v3)",
             "https://arxiv.org/abs/1802.05799",
             "phase edges of five gradient tensors' AllReduces, read against the screenshot's 100 ms ticks to "
             "about +/-2 ms, in ms after its first tick (Literature.Communication.HorovodTimelineOriginMs); "
             "tensors 0-1 and 2-4 share their rows' phases; added 2026-10-20"),
        "Phase edges (negotiate, wait, schedule, NCCL, copy) of AllReduces in a Horovod Timeline trace.",
        numeric=("start_ms", "end_ms"),
    )
    InterconnectGenerations = TrendSeries(
        "interconnect_generations",
        _lit("Interconnect generation rates and dates from IEEE 802.3, IBTA, PCI-SIG, and NVIDIA primary "
             "sources (per-row source_spec_url and source_product_url), compiled 2026-10-12",
             "https://standards.ieee.org/ieee/802.3/",
             "line_rate_gbps and per_direction_GBps are per direction: per port for Ethernet and InfiniBand "
             "(4x), per x16 link after encoding for PCIe, and the sum of a GPU's links for NVLink (half "
             "NVIDIA's bidirectional total); spec_year is the standard's publication (GPU launch for NVLink); "
             "first_product_year is the first commercial product; blanks mean unconfirmed (see notes)"),
        "Per-direction bandwidth of Ethernet, InfiniBand, NVLink, and PCIe generations with spec and product years.",
        numeric=("line_rate_gbps", "per_direction_GBps", "spec_year", "first_product_year"),
    )
    AppleNeuralEngine = TrendSeries(
        "apple_neural_engine",
        _lit("Apple Newsroom and apple.com product pages for each A-series chip (per-row quote and source_url), "
             "checked 2026-10-12", "https://www.apple.com/newsroom/",
             "tops is Apple's stated peak Neural Engine rating where basis = stated (A16 and A17 Pro are "
             "'nearly'); A13 basis = derived from two Apple speedup ratios (A12 x 1.2 and A14 / 1.8); A18 Pro "
             "basis = secondary (Apple states only 'faster than the previous generation'); A19 Pro has no "
             "figure; Apple never states the numeric precision, and the FP16-to-INT8 change at A17 Pro is a "
             "secondary-source inference"),
        "Apple Neural Engine peak ratings by A-series chip, with the basis of each value.",
        numeric=("year", "tops"),
    )
    GeminiPromptEnergy = TrendSeries(
        "gemini_prompt_energy",
        _lit("Elsworth et al. (2025), Measuring the environmental impact of delivering AI at Google Scale, "
             "Table 1 (median Gemini Apps text prompt, May 2025), arXiv:2508.15734", "https://arxiv.org/abs/2508.15734",
             "Wh per median text prompt by component; the Existing Approach counts only active accelerators on "
             "prompts in the top 10% most efficient data centers (other components shown but not counted, "
             "total 0.10 Wh); the Comprehensive Approach counts every component across the fleet (total 0.24 "
             "Wh); Table 2 gives 0.03 gCO2e and 0.26 mL water per prompt; transcribed exactly"),
        "Energy per median Gemini text prompt by serving-stack component under two measurement boundaries.",
        numeric=("existing_wh", "existing_counted", "comprehensive_wh", "comprehensive_counted"),
    )
    GoogleServingFactors = TrendSeries(
        "google_serving_factors",
        _lit("Elsworth et al. (2025), Measuring the environmental impact of delivering AI at Google Scale, "
             "Section 3.3, Section 4.3, and Table 2, arXiv:2508.15734", "https://arxiv.org/abs/2508.15734",
             "emission factors are Google data-center annual averages (location-based, carbon-free-energy "
             "credit, and market-based net); the May 2025 prompt metrics apply the 2024 factors; WUE is "
             "ISO Category 2 (consumptive) freshwater use; per-prompt values are for the median Gemini Apps "
             "text prompt under the Comprehensive Approach; transcribed exactly"),
        "Emission factors, WUE, PUE, and per-prompt water and emissions reported for Gemini serving.",
        numeric=("value",),
    )
