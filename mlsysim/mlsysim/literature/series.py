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

FirecrackerBootPreconfigured = _Series(
    "firecracker_boot_preconfigured",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref=("Agache et al. (2020), Firecracker: Lightweight Virtualization for Serverless "
             "Applications (NSDI '20), released boot-time CDFs, pre-configured microVMs on m5d.metal"),
        url="https://github.com/firecracker-microvm/nsdi2020-data",
        verified="2026-10-20",
        notes=(
            "Second data block (gnuplot index 1, plotted as 'FC-pre') of "
            "data/m5d.metal/boot-serial-fc-api-cdf.dat, boot-parallel-50-fc-api-cdf.dat, and "
            "boot-parallel-100-fc-api-cdf.dat, copied row for row: boot time in ms and the "
            "cumulative fraction booted. Concurrency 1 is the serial run. Apache-2.0."
        ),
    ),
    "Firecracker pre-configured microVM boot-time CDF (ms) by number booting at once.",
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

# Serving at scale: request traces (added 2026-10-20)
LlmInferenceTraceConversation2023 = _Series(
    "azure_llm_inference_conv_2023",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="Azure LLM inference trace 2023, conversation (Patel et al., Splitwise, ISCA 2024)",
        url="https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md",
        verified="2026-10-18",
        notes=(
            "The released file AzureLLMInferenceTrace_conv.csv, unmodified: one row per request "
            "(timestamp, context tokens, generated tokens), 19,366 requests in one hour on "
            "2023-11-16 to a production conversation service. Raw: raw.githubusercontent.com/"
            "Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv. License CC BY 4.0."
        ),
    ),
    "Arrival time, prompt (context) tokens, and output (generated) tokens of production chat requests.",
)

LlmInferenceTraceConversation2024TenMinute = _Series(
    "azure_llm_inference_conv_2024_10min",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="Azure LLM inference trace 2024, conversation, one week (Stojkovic et al., DynamoLLM, HPCA 2025)",
        url="https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md",
        verified="2026-10-18",
        notes=(
            "Requests and tokens per 10-minute bin, 2024-05-12 00:00 to 2024-05-18 23:50, counted "
            "from the released AzureLLMInferenceTrace_conv_1week.csv by aggregate_azure_2024.py "
            "(slides tree, images/data of the serving lecture): a bin is the timestamp truncated "
            "to its ten-minute digit. License CC BY 4.0."
        ),
    ),
    "Requests, prompt tokens, and output tokens per 10-minute bin over one week of a chat service.",
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

# Sim-to-real ablations of a dexterous hand (added 2026-10-20)
DactylPhysicalTrials = _Series(
    "dactyl_physical_trials",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="OpenAI: Andrychowicz et al. (2020), Learning Dexterous In-Hand Manipulation, IJRR 39(1), "
            "Tables 4 and 5 (block, state), arXiv:1808.00177v5",
        url="https://arxiv.org/abs/1808.00177",
        verified="2026-10-18",
        notes=(
            "Consecutive rotations in each of 10 physical trials per condition, sorted values as "
            "published; printed_median is the table's own median column. Table 4 holds randomization "
            "groups out; Table 5 compares LSTM and feed-forward policy and value networks under all "
            "randomizations (its LSTM row repeats Table 4's all-randomizations row)."
        ),
    ),
    "Dactyl's physical-hand trials by training condition.",
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
    # Added 2026-10-20: FedAvg rounds to target accuracy against local work.
    FedAvgMnistCnnRounds = TrendSeries(
        "fedavg_mnist_cnn_rounds",
        Provenance(kind=ProvenanceKind.LITERATURE,
                   ref="McMahan et al. (2017), Communication-Efficient Learning of Deep Networks from Decentralized "
                       "Data, AISTATS 2017, Table 2 (MNIST CNN, 99% accuracy)",
                   url="https://arxiv.org/abs/1602.05629", verified="2026-10-20",
                   notes="Rounds to reach 99% test accuracy with 100 clients (C = 0.1); first row is FedSGD; "
                         "batch_size inf means the full local dataset; updates_per_round u = E n / (K B); "
                         "non-IID is the pathological two-digits-per-client split."),
        "Communication rounds to 99% MNIST CNN accuracy for FedSGD and FedAvg, IID and non-IID.",
        numeric=("local_epochs", "updates_per_round", "iid_rounds", "non_iid_rounds"),
    )
    # Added 2026-10-20: MNIST accuracy against epsilon for @fig-privacy-utility-frontier.
    GaussianDpMnistNoisySgd = TrendSeries(
        "bu_gdp_mnist_noisysgd",
        Provenance(kind=ProvenanceKind.LITERATURE,
                   ref="Bu, Dong, Long, and Su (2020), Deep Learning with Gaussian Differential Privacy, Harvard "
                       "Data Science Review 2(3), Table 1 (Experimental results for NoisySGD and their privacy "
                       "analyses on MNIST)",
                   url="https://pmc.ncbi.nlm.nih.gov/articles/PMC7695347/", verified="2026-10-20",
                   notes="Columns transcribed exactly from Table 1: learning rate eta, clipping norm R, noise "
                         "multiplier sigma, epochs, test accuracy (%), CLT mu, CLT epsilon, and moments-accountant "
                         "epsilon; both epsilon columns are with respect to delta = 1e-5. The table has no "
                         "nonprivate row."),
        "MNIST NoisySGD test accuracy against privacy budget under the Gaussian DP central limit theorem.",
        numeric=("learning_rate", "clip_norm", "noise_multiplier", "epochs", "test_accuracy_pct", "clt_mu",
                 "clt_epsilon", "ma_epsilon"),
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

    # --- Silent degradation and memory-bound decode evidence (added 2026-10-20) ---
    GoogleFluTrendsVsCdc = TrendSeries(
        "google_flu_trends_cdc_ilinet",
        Provenance(
            kind=ProvenanceKind.LITERATURE,
            ref="Google Flu Trends US national estimate and CDC ILINet weighted ILI (US national), as archived "
                "by the Delphi Epidata API (Carnegie Mellon); failure analysis in Lazer et al. (2014), "
                "The Parable of Google Flu, Science 343:1203",
            url="https://api.delphi.cmu.edu/epidata/gft/?locations=nat",
            verified="2026-10-20",
            notes=("weekly rows (epiweek = year*100 + MMWR week) where both series report, 2003-40 to 2015-32; "
                   "gft_ili_pct is Google's estimate of influenza-like illness per 100,000 doctor visits "
                   "divided by 1,000 (percent of visits); cdc_wili_pct is ILINet weighted percent of visits "
                   "(https://api.delphi.cmu.edu/epidata/fluview/?regions=nat), retrieved 2026-10-16; the CDC "
                   "series is the current revised one, not the 2013 real-time release"),
        ),
        "Weekly Google Flu Trends estimate against the CDC count of doctor visits for flu-like illness.",
        numeric=("epiweek", "gft_ili_pct", "cdc_wili_pct"),
    )
    ImageNetV2MatchedFrequency = TrendSeries(
        "imagenetv2_matched_frequency_top1",
        _lit("Recht, Roelofs, Schmidt, and Shankar (2019), Do ImageNet Classifiers Generalize to ImageNet?, "
             "ICML, appendix tables imagenetv2-b-33_top1_model_results_table.{1,2}.tex (arXiv 1902.10811 source)",
             "https://arxiv.org/abs/1902.10811",
             "top-1 accuracy (%) of 67 models on the original ImageNet validation set and on ImageNetV2 "
             "MatchedFrequency, with the paper's rounded gap in points; models named fv_* are the Fisher-vector "
             "baselines, the rest neural networks; extracted by pattern from the LaTeX rows, not typed"),
        "Top-1 accuracy of 67 ImageNet models on the original test set and on a re-collected one.",
        numeric=("original_rank", "original_top1", "imagenet_v2_top1", "gap"),
    )
    MLPerfInferenceV4Llama2Offline = TrendSeries(
        "mlperf_inference_v4_llama2_70b_offline",
        _lit("MLCommons, MLPerf Inference datacenter v4.0 (Apache-2.0), closed division, NVIDIA, "
             "llama2-70b-99.9 Offline, closed/NVIDIA/results/<system>/llama2-70b-99.9/Offline/performance/"
             "run_1/mlperf_log_summary.txt", "https://github.com/mlcommons/inference_results_v4.0",
             "tokens per second for eight-GPU H100-SXM-80GB and H200-SXM-141GB systems; the CTS system is "
             "NVIDIA's H200 with a custom thermal solution and a higher power limit; accelerator count from "
             "each system name; all results VALID"),
        "Llama 2 70B offline throughput on eight H100 and eight H200 GPUs (MLPerf Inference v4.0).",
        numeric=("accelerators", "tokens_per_s"),
    )

    # --- Deployment-tier measurements (added 2026-10-20) --------------------
    # Measured field data behind the light-barrier, memory-wall, TPU roofline,
    # and TinyML energy comparisons; each row transcribed from the cited source.
    AwsInterRegionRtt = TrendSeries(
        "aws_interregion_rtt_p50",
        Provenance(
            kind=ProvenanceKind.INDUSTRY_REPORT,
            ref="cloudping.co AWS inter-region latency, /api/latencies?percentile=p_50&timeframe=1W, "
                "retrieved 2026-10-18",
            url="https://www.cloudping.co/",
            verified="2026-10-18",
            notes=("p50_rtt_ms is the one-week median round trip measured from us-east-1 (N. Virginia) "
                   "to each region, as the API returned it. Coordinates are those of the metro area AWS "
                   "names for each region (Wikipedia city coordinates); AWS does not publish building "
                   "locations, so great-circle distances from them are approximate."),
        ),
        "Measured median round trips from AWS us-east-1 to four regions, with metro coordinates.",
        numeric=("source_lat", "source_lon", "dest_lat", "dest_lon", "p50_rtt_ms"),
    )
    TpuV1ProductionApps = TrendSeries(
        "tpu_v1_production_apps",
        _lit("Jouppi et al. (2017), In-Datacenter Performance Analysis of a Tensor Processing Unit, "
             "ISCA 2017 (arXiv:1704.04760), Table 1 and Table 3",
             "https://arxiv.org/abs/1704.04760",
             "ops_per_weight_byte: Table 1, TPU Ops / Weight Byte; measured_tops: Table 3, row 9, "
             "TeraOps/sec measured with hardware counters (92 peak); six applications that were 95% "
             "of the TPU datacenter workload in July 2016"),
        "Arithmetic intensity and measured throughput of the six production applications on TPU v1.",
        numeric=("ops_per_weight_byte", "measured_tops"),
    )
    TpuV1DeploymentMix = TrendSeries(
        "tpu_v1_deployment_mix",
        _lit("Jouppi et al. (2017), In-Datacenter Performance Analysis of a Tensor Processing Unit, "
             "ISCA 2017 (arXiv:1704.04760), Table 1, percent of deployed TPUs in July 2016",
             "https://arxiv.org/abs/1704.04760"),
        "Share of deployed TPU v1s by application family, July 2016.",
        numeric=("deployed_share_pct",),
    )
    MLPerfTinyV12KwsEnergy = TrendSeries(
        "mlperf_tiny_v1_2_kws_energy",
        _lit("MLCommons, MLPerf Tiny v1.2 results (Apache-2.0), summary.csv, retrieved 2026-10-18",
             "https://github.com/mlcommons/tiny_results_v1.2",
             "keyword spotting (kws), closed division, rows with an energy measurement (HasPower); "
             "energy_uj is the reported Power column, microjoules per inference"),
        "Measured energy per keyword-spotting inference in MLPerf Tiny v1.2, closed division.",
        numeric=("energy_uj",),
    )
    MemoryWallHardwareScaling = TrendSeries(
        "memory_wall_hardware_scaling",
        _lit("Gholami et al. (2024), AI and Memory Wall, IEEE Micro 44(3) (arXiv:2403.14123), Fig. 1",
             "https://arxiv.org/abs/2403.14123",
             "markers digitized from the vector figure in the arXiv source (figs/hw_scaling.pdf): each "
             "filled marker's center read back to (year, value) through the figure's own axis ticks; "
             "series: flops (server chips' peak FLOP/s), dram (DRAM bandwidth), interconnect "
             "(interconnect bandwidth), each normalized to its first point"),
        "Peak compute, DRAM bandwidth, and interconnect bandwidth of server hardware, normalized, by year.",
        numeric=("year", "normalized"),
    )
    # --- Responsible AI (added 2026-10-20) ----------------------------------
    OfqualGradeMatrix2020 = TrendSeries(
        "ofqual_2020_grade_matrix",
        _lit("Ofqual (2021), Grading gaps in summer 2020, Table 1 (A level, England), Crown Copyright, OGL v3.0",
             "https://www.gov.uk/government/publications/grading-gaps-in-summer-2020",
             "one row per centre assessment grade (CAG): entries is N for that CAG grade; columns A* to U "
             "are the row percentages of entries by calculated grade as printed; cells printed as '~' "
             "(under 0.5 percent) are entered as 0, so a row may not sum to exactly 100"),
        "A level entries by teacher-assessed grade against the grade the 2020 standardization calculated.",
        numeric=("entries", "A*", "A", "B", "C", "D", "E", "U"),
    )
    OfqualCandidatesLowered2020 = TrendSeries(
        "ofqual_2020_candidates_lowered",
        _lit("Ofqual (2021), Grading gaps in summer 2020, Table 2 (A level, England), Crown Copyright, OGL v3.0",
             "https://www.gov.uk/government/publications/grading-gaps-in-summer-2020",
             "candidates by the number of their A levels whose CAG exceeded the calculated grade, as "
             "printed (rounded to the nearest 10)"),
        "A level candidates by how many of their grades the 2020 standardization lowered.",
        numeric=("grades_lowered", "candidates"),
    )
    CompasTwoYearConfusion = TrendSeries(
        "compas_two_year_confusion",
        _lit("ProPublica, compas-analysis, compas-scores-two-years.csv (Angwin et al. 2016), recomputed "
             "2026-10-18", "https://github.com/propublica/compas-analysis",
             "ProPublica's filters: |days_b_screening_arrest| <= 30, is_recid != -1, c_charge_degree != 'O', "
             "score_text != 'N/A'; flagged = score_text Medium or High; outcome two_year_recid; "
             "tp flagged and reoffended, fp flagged and did not, fn not flagged and reoffended, tn neither; "
             "race as coded in the file"),
        "COMPAS decile-score flags against two-year recidivism, by race, from ProPublica's released data.",
        numeric=("tp", "fp", "fn", "tn"),
    )
    AbadiDpSgdAccuracy = TrendSeries(
        "abadi_dpsgd_accuracy",
        _lit("Abadi et al. (2016), Deep Learning with Differential Privacy, ACM CCS 2016, Sec. 5.2 and Fig. 6",
             "https://arxiv.org/abs/1607.00133",
             "test accuracy (%) of DP-SGD at each reported (epsilon, delta): MNIST 90, 95, 97 percent at "
             "epsilon 0.5, 2, 8; CIFAR-10 67, 70, 73 percent at epsilon 2, 4, 8; nonprivate baselines in "
             "Literature.DifferentialPrivacy"),
        "Differentially private training accuracy by privacy budget on MNIST and CIFAR-10.",
        numeric=("epsilon", "delta", "accuracy_pct"),
    )

    # --- Training and fleet footprints (added 2026-10-20) ---------------------
    PattersonTrainingFootprints = TrendSeries(
        "patterson_training_footprints",
        Provenance(kind=ProvenanceKind.LITERATURE,
                   ref="Patterson et al. (2021), Carbon Emissions and Large Neural Network Training, Table 4 "
                       "(arXiv:2104.10350)",
                   url="https://arxiv.org/abs/2104.10350", verified="2026-10-20",
                   notes=("Final-run figures for the five NLP models (the Evolved Transformer NAS column is "
                          "omitted); kg_per_kwh is the data center's CO2e per kWh when the model ran, gross "
                          "and net of carbon-free purchases; measured_w is the measured system average power "
                          "per accelerator including memory, network interface, fans, and host CPU; "
                          "activated_pct is the percent of the model activated on every token; transcribed "
                          "exactly from the PDF table.")),
        "Final-run energy, grid factor, PUE, measured power, and emissions of five NLP models (2021).",
        numeric=("params_b", "activated_pct", "gross_kg_per_kwh", "net_kg_per_kwh", "pue", "tdp_w",
                 "measured_w", "chips", "days", "energy_mwh", "gross_tco2e", "net_tco2e"),
    )
    BloomProjectEmissions = TrendSeries(
        "bloom_project_emissions",
        Provenance(kind=ProvenanceKind.LITERATURE,
                   ref="Luccioni, Viguier, and Ligozat (2023), Estimating the Carbon Footprint of BLOOM, a 176B "
                       "Parameter Language Model, JMLR 24(253), Table 5 (arXiv:2211.02001)",
                   url="https://arxiv.org/abs/2211.02001", verified="2026-10-20",
                   notes=("Dynamic-power emissions of the BigScience project by process, t CO2eq, French grid "
                          "at 57 gCO2eq/kWh, GPU energy on a TDP basis; transcribed for the Sustainable AI "
                          "lecture and moved into MLSysIM 2026-10-20.")),
        "Dynamic-power emissions of the BLOOM project by model and process.",
        numeric=("tco2e",),
    )
    CsoIrelandDatacenterElectricity = TrendSeries(
        "cso_ireland_datacenter_electricity",
        Provenance(kind=ProvenanceKind.INDUSTRY_REPORT,
                   ref="Central Statistics Office (Ireland), Data Centres Metered Electricity Consumption 2023, "
                       "statistical release of 23 July 2024, Table 1",
                   url="https://www.cso.ie/en/releasesandpublications/ep/p-dcmec/datacentresmeteredelectricityconsumption2023/",
                   verified="2026-10-20",
                   notes=("share_pct is data centers' percent of total metered electricity consumption; gwh is "
                          "their consumption, transcribed for 2015 and 2023 only (other years left empty).")),
        "Irish data centers' share of metered electricity, 2015 to 2023.",
        numeric=("year", "share_pct", "gwh"),
    )
    EuropeGenerationWeek47 = TrendSeries(
        "europe_generation_2023w47",
        Provenance(kind=ProvenanceKind.DATASHEET,
                   ref="Energy-Charts (Fraunhofer ISE), public net electricity generation and load, EU, "
                       "20 to 26 November 2023, hourly (api.energy-charts.info/public_power, country=eu)",
                   url="https://api.energy-charts.info/public_power?country=eu",
                   verified="2026-10-20",
                   notes=("Hourly generation by source class in MW (time_cet is the hour start in CET); "
                          "coal_oil groups hard coal, lignite, and oil; copied from the Sustainable AI "
                          "chapter's data file, which this series replaces.")),
        "Hourly EU electricity generation by source and load, one week of November 2023 (MW).",
        numeric=("nuclear", "hydro", "biomass_other", "wind", "solar", "gas", "coal_oil", "load"),
    )
    # --- Fleet-scale measurements recalled by the Volume II conclusion (added 2026-10-20) ---
    MegaScaleStrongScaling = TrendSeries(
        "megascale_strong_scaling",
        _lit("Jiang et al. (2024), MegaScale: Scaling Large Language Model Training to More Than 10,000 "
             "GPUs, NSDI 2024, Table 2 (arXiv:2402.15627 v1)",
             "https://arxiv.org/abs/2402.15627",
             "Strong scaling of the 175B model at a fixed batch of 768 sequences (sequence 2,048) on "
             "NVIDIA Ampere GPUs: MegaScale training throughput in tokens per second and MFU, with "
             "Megatron-LM's MFU on the same configuration. Read from the paper's table."),
        "MegaScale strong scaling of a 175B model at batch 768: throughput and MFU by GPU count.",
        numeric=("gpus", "megascale_tokens_s", "megascale_mfu_pct", "megatron_mfu_pct"),
    )
    TailAtScaleLeafLatency = TrendSeries(
        "tail_at_scale_leaf_latency",
        _lit("Dean and Barroso (2013), The Tail at Scale, Communications of the ACM 56(2), Table 1",
             "https://doi.org/10.1145/2408776.2408794",
             "Latency measured at the root of a Google service whose requests fan out through "
             "intermediate servers to many leaf servers, by how many leaves the root waits for. Only "
             "the cells the books cite are transcribed (the 50th and 99th percentiles of one random "
             "leaf and of all leaves, and the 99th percentile of 95 percent of the leaves). The paper "
             "names neither the service nor its leaf count. Read from the paper."),
        "Root-measured latency of a fan-out service by the share of leaves awaited and percentile.",
        numeric=("percentile", "latency_ms"),
    )

    # --- Model compression (added 2026-10-20) ---
    MobileNetV3Table3 = TrendSeries(
        "mobilenetv3_table3_pixel1",
        Provenance(kind=ProvenanceKind.LITERATURE,
                   ref="Howard et al. (2019), Searching for MobileNetV3, ICCV 2019, Table 3",
                   url="https://arxiv.org/abs/1905.02244",
                   verified="2026-10-20",
                   notes=("ImageNet top-1 (percent), Pixel 1 latency in floating point (ms; TFLite "
                          "benchmark tool, one large core, batch 1), multiply-adds and parameters "
                          "(millions) of the classification models of Table 3.")),
        "Accuracy and measured phone latency of efficient ImageNet models (MobileNetV3 Table 3).",
        numeric=("top1", "pixel1_ms", "madds_m", "params_m"),
    )
    HanAlexNetPruning = TrendSeries(
        "han2015_alexnet_pruning",
        Provenance(kind=ProvenanceKind.LITERATURE,
                   ref="Han, Pool, Tran, and Dally (2015), Learning both Weights and Connections for "
                       "Efficient Neural Networks, NeurIPS 2015, figure 5",
                   url="https://arxiv.org/abs/1506.02626",
                   verified="2026-10-20",
                   notes=("AlexNet, ImageNet top-5 change (points) against the share of parameters "
                          "pruned (percent), L2-regularized runs: curve no_retrain, retrain (one prune "
                          "then retraining), iterative (pruning and retraining repeated). Points taken "
                          "from the vector paths of the published figure (arXiv v3, p. 7).")),
        "Pruning without retraining, with one retraining, and iteratively (AlexNet, Han et al. 2015).",
        numeric=("pruned_pct", "top5_change"),
    )
    DeepCompressionMethods = TrendSeries(
        "han2016_compression_methods",
        Provenance(kind=ProvenanceKind.LITERATURE,
                   ref="Han, Mao, and Dally (2016), Deep Compression, ICLR 2016, figure 6",
                   url="https://arxiv.org/abs/1510.00149",
                   verified="2026-10-20",
                   notes=("AlexNet on ImageNet: accuracy change (points) against model size kept "
                          "(percent of original) for four methods, as transcribed in the Model "
                          "Compression chapter's figure; extension=True marks a line's continuation "
                          "off the plotted axis, not a reported point.")),
        "Accuracy against model size for pruning, quantization, both, and SVD (Deep Compression).",
        numeric=("size_pct", "accuracy_change"),
    )
    MishraInt8SparseGemm = TrendSeries(
        "mishra2021_int8_sparse_gemm",
        Provenance(kind=ProvenanceKind.LITERATURE,
                   ref="Mishra et al. (2021), Accelerating Sparse Deep Neural Networks, figure 3",
                   url="https://arxiv.org/abs/2104.08378",
                   verified="2026-10-20",
                   notes=("Measured INT8 2:4 sparse-over-dense GEMM speedup on an A100 at M = N = 10240 "
                          "for three inner dimensions K; values read from the published plot "
                          "(approximate).")),
        "2:4 sparse INT8 GEMM speedup over dense by inner dimension (A100).",
        numeric=("k", "speedup"),
    )
    # --- Benchmarking (added 2026-10-20) ---------------------------------------
    MLPerfTrainingV40ResNetRuns = TrendSeries(
        "mlperf_training_v4_0_resnet_runs",
        _lit("MLCommons, MLPerf Training v4.0 results (Apache-2.0), closed division, resnet, result_N.txt "
             "MLLOG lines (run_start, run_stop, last eval_accuracy, status, seed), retrieved 2026-10-17",
             "https://github.com/mlcommons/training_results_v4.0",
             "One row per run of four submissions: NVIDIA eos_ngc24.04_mxnet (8x H100), Dell "
             "XE9680x8H100-SXM-80GB (8x H100), NVIDIA h200_ngc24.04_mxnet (8x H200), and NVIDIA "
             "eos_n8_ngc24.04_mxnet (64x H100). minutes is run_stop minus run_start; top1 is the last "
             "eval_accuracy in percent; status 'aborted' marks a run that missed the quality target. "
             "The published score is the mean of the middle three of five runs (a missed run counts as "
             "the slowest) times the scaling.json factor (Literature.Benchmarks."
             "MLPerfTrainingV40ResNetScalingFactor)."),
        "Per-run time to the ResNet-50 quality target in MLPerf Training v4.0 for four 8- and 64-accelerator systems.",
        numeric=("run", "minutes", "top1", "epochs", "seed"),
    )
    MLPerfTrainingV40EosResNetEvals = TrendSeries(
        "mlperf_training_v4_0_eos_resnet_result0_evals",
        _lit("MLCommons, MLPerf Training v4.0 results (Apache-2.0), closed/NVIDIA/results/"
             "eos_ngc24.04_mxnet/resnet/result_0.txt, eval_accuracy and run_start MLLOG lines, retrieved 2026-10-17",
             "https://github.com/mlcommons/training_results_v4.0",
             "One row per evaluation of run 0 on NVIDIA Eos (8x H100): epoch, wall-clock minutes since "
             "run_start, and ImageNet top-1 in percent."),
        "ImageNet top-1 by epoch of one MLPerf Training v4.0 ResNet-50 run on eight H100s (time to quality).",
        numeric=("epoch", "minutes", "top1"),
    )
    MLPerfInferenceV50ResNet50 = TrendSeries(
        "mlperf_inference_v5_0_resnet50",
        _lit("MLCommons, MLPerf Inference v5.0 results (Apache-2.0), closed division, resnet50, "
             "summary_results.json, results 5.0-0032, 5.0-0010, 5.0-0065",
             "https://github.com/mlcommons/inference_results_v5.0",
             "One RTX 4090 (SingleStream p90 latency and Offline throughput) and two 8x H100-SXM servers "
             "(Offline and Server throughput); system totals as reported, all VALID."),
        "ResNet-50 MLPerf Inference v5.0 results by scenario for one RTX 4090 and two eight-H100 servers.",
        numeric=("accelerators", "value"),
    )
    MLPerfTinyV12VwwEnergy = TrendSeries(
        "mlperf_tiny_v1_2_vww_energy",
        _lit("MLCommons, MLPerf Tiny v1.2 results (Apache-2.0), summary.csv, retrieved 2026-10-18",
             "https://github.com/mlcommons/tiny_results_v1.2",
             "visual wake words (vww), closed division, rows with an energy measurement; energy_uj is the "
             "reported Power column, microjoules per inference; throughput_ips is the reported "
             "inferences per second where the slides use it; clock_mhz and voltage_v are the operating point "
             "the submission's system description names (syntiant_9120_0v9_30mhz: 0.9 V, 30.7 MHz)"),
        "Measured energy per visual-wake-words inference in MLPerf Tiny v1.2, closed division.",
        numeric=("clock_mhz", "voltage_v", "throughput_ips", "energy_uj"),
    )
    LeeMobileGpuLatency = TrendSeries(
        "lee2019_mobile_gpu_latency",
        _lit("Lee et al. (2019), On-Device Neural Net Inference with Mobile GPUs, Tables 4 and 5",
             "https://arxiv.org/abs/1907.01989",
             "MobileNet v1 latency in ms, mean of 100 runs after 10 warm-up runs, by device and software path"),
        "MobileNet v1 latency on one iPhone XS and one Pixel 3 through different software paths.",
        numeric=("latency_ms",),
    )
    LeeSustainedLatency = TrendSeries(
        "lee2019_sustained_latency",
        _lit("Lee et al. (2019), On-Device Neural Net Inference with Mobile GPUs, Figure 8",
             "https://arxiv.org/abs/1907.01989",
             "MobileNet v1 latency in ms at the start and after 200 s of continuous inference; values "
             "read from the published plot (approximate)"),
        "MobileNet v1 latency at the start and after 200 s of sustained load, by device and processor.",
        numeric=("start_ms", "end_ms"),
    )
    TummalapalliIphoneThermal = TrendSeries(
        "tummalapalli2026_iphone_thermal",
        _lit("Tummalapalli et al. (2026), arXiv 2603.23640 (preprint)",
             "https://arxiv.org/abs/2603.23640",
             "Qwen 2.5 1.5B decode rate on an iPhone 16 Pro at peak, once warm, and once hot, tokens per second"),
        "Small-language-model decode rate on an iPhone 16 Pro by thermal state.",
        numeric=("tokens_per_s",),
    )

# --- Context engineering (added 2026-10-20) ----------------------------------
RulerEffectiveLength = _Series(
    "ruler_effective_length",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="Hsieh et al. (2024), RULER: What's the Real Context Size of Your Long-Context Language Models?, COLM 2024, Table 3",
        url="https://arxiv.org/abs/2404.06654",
        verified="2026-10-20",
        notes=(
            "Rows with a claimed length up to 128K, transcribed from Table 3 of arXiv 2404.06654v3: "
            "claimed and effective length in thousands of tokens, model names as printed. LongChat "
            "(7B) prints an effective length of <4K (empty effective_k). Effective length is the "
            "longest tested length scoring above 85.6 (Literature.ContextUse.RulerEffectiveLengthThreshold)."
        ),
    ),
    "Claimed and effective context length (thousand tokens) of long-context models on RULER.",
)

AnswerPositionAccuracy = _Series(
    "lost_in_the_middle_gpt35_turbo_20docs",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="Liu et al. (2024), Lost in the Middle: How Language Models Use Long Contexts, TACL 12, Table 6",
        url="https://arxiv.org/abs/2307.03172",
        verified="2026-10-20",
        notes=(
            "GPT-3.5-Turbo (gpt-3.5-turbo-0613) multi-document QA accuracy in percent with the answer "
            "document at each reported position among 20 total retrieved documents (about 4K tokens), "
            "transcribed from Table 6 of arXiv 2307.03172v3. Closed-book accuracy: "
            "Literature.ContextUse.Gpt35TurboClosedBook."
        ),
    ),
    "Multi-document QA accuracy by position of the answer document among 20 (GPT-3.5-Turbo).",
)


# --- Retrieval from long-term memory (added 2026-10-20) ---
SweBenchBm25Context = _Series(
    "swe_bench_bm25_context",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref=("Jimenez et al. (2024), SWE-bench: Can Language Models Resolve Real-World GitHub "
             "Issues?, ICLR 2024, Table 2 (Claude 2) and Table 3 (Avg. row)"),
        url="https://arxiv.org/abs/2310.06770",
        verified="2026-10-20",
        notes=(
            "By maximum retrieved context in thousand tokens (BM25 over the repository, the issue "
            "text as the query): recall of the files the reference fix edits, in percent, and "
            "Claude 2's percent of issues resolved. Transcribed from arXiv 2310.06770v3."
        ),
    ),
    "SWE-bench BM25 retrieval by context limit: oracle-file recall and Claude 2 resolve rate (percent).",
)

PoisonedRagBlackBox = _Series(
    "poisonedrag_black_box_palm2",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref=("Zou et al. (2025), PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented "
             "Generation of Large Language Models, USENIX Security 2025, Section 5.1 and Table 1"),
        url="https://arxiv.org/abs/2402.07867",
        verified="2026-10-20",
        notes=(
            "Clean knowledge-base size in texts (Section 5.1) and black-box attack success rate "
            "with PaLM 2 as the generator (Table 1), five malicious texts per target question "
            "(Literature.AgentRetrieval.PoisonedRagTextsPerQuestion). Transcribed from arXiv "
            "2402.07867v3."
        ),
    ),
    "PoisonedRAG black-box attack: knowledge-base size and attack success rate per dataset (PaLM 2).",
)

# --- KV cache management (added 2026-10-20) ----------------------------------
PagedAttentionMemoryBreakdown = _Series(
    "pagedattention_kv_memory_breakdown",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="Kwon et al. (2023), Efficient Memory Management for Large Language Model Serving with "
            "PagedAttention, SOSP 2023, Figure 2",
        url="https://arxiv.org/abs/2309.06180",
        verified="2026-10-18",
        notes=(
            "Average percentage of KV cache memory by use in three Orca variants (reservation for the "
            "maximum length, the next power of two, or the exact final length) and vLLM, OPT-13B on one "
            "A100 (Section 6.2 setup), read from the bar labels of Figure 2. vLLM reports only its token "
            "states (96.3); the remaining columns are empty."
        ),
    ),
    "KV memory holding token states, reservations, internal and external fragmentation (percent).",
)

# --- Network architectures (added 2026-10-20) ---------------------------------
FlashAttentionBenchmark = _Series(
    "flashattention_a100_benchmark",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="Dao et al. (2022), FlashAttention: Fast and Memory-Efficient Exact Attention with "
            "IO-Awareness, NeurIPS 2022, Tables 20 and 21",
        url="https://arxiv.org/abs/2205.14135",
        verified="2026-10-20",
        notes=(
            "arXiv 2205.14135v2: forward plus backward runtime (ms, no dropout or masking, Table 20) and "
            "memory (MB, Table 21) of standard PyTorch attention and FlashAttention on one A100 40 GB, "
            "batch 16, 8 heads of 64, FP16. PyTorch runs out of memory beyond 4,096 tokens (empty cells)."
        ),
    ),
    "Attention runtime and memory against sequence length, standard PyTorch and FlashAttention (A100).",
)

# --- Fitting a training step on one accelerator (added 2026-10-20) -------------
SsdGradientHistogram = _Series(
    "micikevicius_ssd_gradient_histogram",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="Micikevicius et al. (2018), Mixed Precision Training, ICLR 2018, Figure 3",
        url="https://arxiv.org/abs/1710.03740",
        verified="2026-10-18",
        notes=(
            "Activation-gradient magnitudes of the SSD detector in FP32 training, binned by log2 magnitude "
            "[log2_low, log2_high). bar_top_px is the pixel row of each bar top, read from the figure "
            "image (arXiv 1710.03740v3, page 4, 898 x 575 px; log-scale y axis, gridline rows 35 at 64 "
            "percent and 504 at 1/512 percent). An empty cell marks an empty bin; the 67 percent of "
            "exact zeros (caption) is not a bar."
        ),
    ),
    "SSD activation-gradient histogram by log2 magnitude, read from Micikevicius et al. (2018), Figure 3.",
)

ToolInterceptionTimes = _Series(
    "infercept_interception_times",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="Abhyankar et al. (2024), InferCept: Efficient Intercept Support for Augmented Large "
            "Language Model Inference, ICML 2024, Table 1",
        url="https://arxiv.org/abs/2402.01869",
        verified="2026-10-18",
        notes=(
            "Mean interception (tool or human pause) time per augmentation, in seconds, read from "
            "Table 1; the chatbot and image-generation times are partly estimated by the authors. "
            "Recomputation share of forwarding time: Literature.ToolInterception."
        ),
    ),
    "Mean pause per augmentation type while an LLM waits on a tool or a person (seconds).",
)

AttentionSinkPerplexity = _Series(
    "streamingllm_attention_sink_perplexity",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="Xiao et al. (2024), Efficient Streaming Language Models with Attention Sinks, ICLR 2024, "
            "Tables 1 and 2",
        url="https://arxiv.org/abs/2309.17453",
        verified="2026-10-18",
        notes=(
            "PG19 perplexity with a KV cache of cache_tokens holding only the most recent tokens "
            "(window_perplexity) against the same cache with the first sink_tokens kept "
            "(sink_perplexity, 4 + cache_tokens - 4); Llama-2-13B from Table 1, Llama-2-7B from Table 2."
        ),
    ),
    "Sliding-window perplexity with and without the first four tokens kept (PG19).",
)

MetastableFailures = _Series(
    "metastable_failures_in_the_wild",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="Huang et al. (2022), Metastable Failures in the Wild, OSDI 2022, Table 1",
        url="https://www.usenix.org/conference/osdi22/presentation/huang-lexiang",
        verified="2026-10-20",
        notes=(
            "One row per incident of Table 1 (21 rows): date and duration in hours as printed "
            "(empty where the table prints NA), and whether 'retry' is listed among its sustaining "
            "effects (12 rows). The text names 11 of them (GGL2, GGL3, AWS1, AWS2, AWS3, AZR2, AZR4, "
            "IBM1, SPF1, SPF2, CAS2) and omits AWS4, whose row lists retry; it describes the share as "
            "'more than 50% of the studied incidents'. Read from the paper PDF."
        ),
    ),
    "Metastable failures from public sources: duration and whether retries sustained them.",
)


# --- Agent evaluation results (added 2026-10-20) -------------------------------
JudgePositionBias = _Series(
    "judge_position_bias",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref=("Zheng et al. (2023), Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena, "
             "Table 2 (position bias, default prompt)"),
        url="https://arxiv.org/abs/2306.05685",
        verified="2026-10-20",
        notes=("Three rows transcribed from Table 2 of the HTML version (arxiv.org/html/2306.05685) and "
               "checked against it: percent of answer pairs, each judged twice with the order swapped, "
               "that kept the verdict, favored the first or second position, or produced an error."),
    ),
    "Pairwise model judges: verdict consistency and position bias when two answers swap places.",
)

HumanEvalAgentCost = _Series(
    "humaneval_agent_cost_accuracy",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="Kapoor et al. (2024), AI Agents That Matter, Table A1 (HumanEval accuracy and total cost)",
        url="https://arxiv.org/abs/2407.01502",
        verified="2026-10-20",
        notes=("Nine rows transcribed from Table A1 (arxiv.org/html/2407.01502): mean accuracy (percent) "
               "and mean total API cost (US dollars, 2024 prices) over five runs on the 164 problems. "
               "role: published agent, author baseline (Retry, Warming, Escalation), or single call."),
    ),
    "Code agents on HumanEval: accuracy against total cost, mean of five runs.",
)

DapperSamplingOverhead = _Series(
    "dapper_sampling_overhead",
    Provenance(
        kind=ProvenanceKind.LITERATURE,
        ref="Sigelman et al. (2010), Dapper, a Large-Scale Distributed Systems Tracing Infrastructure, "
            "Google Technical Report dapper-2010-1, Table 2",
        url="https://research.google/pubs/dapper-a-large-scale-distributed-systems-tracing-infrastructure/",
        verified="2026-10-20",
        notes=(
            "One row per non-adaptive sampling frequency of Table 2: the change in average latency and "
            "in average throughput of a Web search cluster, percent as printed. The caption gives "
            "experimental errors of 2.5 and 0.15 percent (Literature.DistributedTracing). Read from "
            "the report PDF with pdftotext."
        ),
    ),
    "Dapper tracing overhead on a Web search cluster at six sampling frequencies (added 2026-10-20).",
)
