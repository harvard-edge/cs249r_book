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
