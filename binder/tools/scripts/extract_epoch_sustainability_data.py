#!/usr/bin/env python3
"""Extract reproducible Epoch AI compute and energy series and provenance manifest.

Reads an Epoch AI ``notable_ai_models.csv`` snapshot from ``--input`` (or
fetches ``https://epoch.ai/data/notable_ai_models.csv`` when ``--download`` is
passed), writes or verifies:

  1. ``books/vol2/01_introduction/data/notable_model_training_runs.csv``
     (post-2012 notable models with training compute or hardware quantity), and
  2. ``books/vol2/15_sustainable_ai/data/notable_model_training_energy.csv``
     (post-2016 standalone pretraining runs with positive power draw and duration),

each carrying an eight-line structured provenance and computation header
(``Source``, ``URL``, ``Verified``, ``Epistemic``, ``Generator``,
``Upstream-SHA256``, ``Computation``, ``Notes``), and emits a JSON provenance
manifest with SHA-256 digests, OLS + White HC3 95% confidence intervals, and
builder-anchor boundary ratios.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import platform
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

EPOCH_CSV_URL = "https://epoch.ai/data/notable_ai_models.csv"
EPOCH_PAGE_URL = "https://epoch.ai/data/notable-ai-models"
COMPUTE_START_DATE = "2012-01-01"
ENERGY_START_DATE = "2016-01-01"
PEAK_PRODUCT_WARNING = "cannot multiply the hardware quantity by training time"
GENERATOR_CMD = "python3 binder/tools/scripts/extract_epoch_sustainability_data.py --download"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENERGY_OUT = REPO_ROOT / "books/vol2/15_sustainable_ai/data/notable_model_training_energy.csv"
DEFAULT_COMPUTE_CSV = REPO_ROOT / "books/vol2/01_introduction/data/notable_model_training_runs.csv"
DEFAULT_MANIFEST_OUT = REPO_ROOT / "binder/tools/audits/epoch_sustainability_manifest.json"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _decimal_year(iso_date: str) -> float:
    d = dt.date.fromisoformat(iso_date)
    return d.year + (d.timetuple().tm_yday - 1) / 365.25


def _t_crit_975(df: int) -> float:
    """Two-sided 95% Student-t critical value (scipy if available, Cornish-Fisher fallback)."""
    try:
        from scipy.stats import t as tdist

        return float(tdist.ppf(0.975, df))
    except Exception:
        z = 1.959963984540054
        g1 = (z**3 + z) / 4.0
        g2 = (5 * z**5 + 16 * z**3 + 3 * z) / 96.0
        g3 = (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / 384.0
        g4 = (79 * z**9 + 776 * z**7 + 1482 * z**5 - 1920 * z**3 - 945 * z) / 92160.0
        return z + g1 / df + g2 / (df**2) + g3 / (df**3) + g4 / (df**4)


def _ols_with_hc3(xs: list[float], ys: list[float]) -> dict[str, float]:
    n = len(xs)
    if n < 3:
        raise ValueError(f"Need at least 3 points for OLS + HC3 fit, got {n}")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    resids = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sse = sum(e**2 for e in resids)
    r2 = 1.0 - sse / syy
    leverages = [1.0 / n + ((x - mx) ** 2) / sxx for x in xs]
    hc3_var_slope = sum(((x - mx) ** 2) * ((e / (1.0 - h)) ** 2) for x, e, h in zip(xs, resids, leverages)) / (
        sxx**2
    )
    se_hc3 = math.sqrt(hc3_var_slope)
    tcrit = _t_crit_975(n - 2)
    return {
        "slope": slope,
        "intercept": intercept,
        "se_hc3": se_hc3,
        "slope_ci95_low": slope - tcrit * se_hc3,
        "slope_ci95_high": slope + tcrit * se_hc3,
        "r2": r2,
    }


def _git_metadata(cwd: Path) -> dict[str, object]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        return {"commit_sha": sha, "dirty": bool(status)}
    except Exception:
        return {"commit_sha": None, "dirty": None}


def _infer_snapshot_date(input_path: Path | None, explicit_date: str | None) -> str:
    if explicit_date:
        dt.date.fromisoformat(explicit_date)
        return explicit_date
    if input_path is not None:
        match = re.search(r"(\d{4}-\d{2}-\d{2})", input_path.name)
        if match:
            return match.group(1)
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


def _format_optional_float(raw: str) -> str:
    val = (raw or "").strip()
    if not val:
        return ""
    return str(float(val))


def load_raw_epoch_bytes(
    input_path: Path | None, download: bool, save_raw: Path | None
) -> tuple[bytes, str]:
    if download:
        req = urllib.request.Request(EPOCH_CSV_URL, headers={"User-Agent": "MLSysBook-DataExtractor/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw_bytes = resp.read()
        if save_raw is not None:
            save_raw.parent.mkdir(parents=True, exist_ok=True)
            save_raw.write_bytes(raw_bytes)
        return raw_bytes, EPOCH_CSV_URL
    if input_path is None:
        raise ValueError("Provide either --input <snapshot.csv> or --download.")
    return input_path.read_bytes(), str(input_path.name)


def extract_compute_csv(
    raw_rows: list[dict[str, str]], snapshot_date: str, raw_sha256: str
) -> tuple[str, list[dict[str, object]]]:
    candidates = [
        r
        for r in raw_rows
        if (r.get("Publication date") or "").strip() >= COMPUTE_START_DATE
        and ((r.get("Training compute (FLOP)") or "").strip() or (r.get("Hardware quantity") or "").strip())
    ]
    ordered = sorted(candidates, key=lambda r: (r.get("Publication date") or "").strip())
    selected: list[dict[str, object]] = []
    for row in ordered:
        selected.append(
            {
                "model": row["Model"].strip(),
                "date": (row.get("Publication date") or "").strip(),
                "domain": (row.get("Domain") or "").strip(),
                "training_compute_flop": _format_optional_float(row.get("Training compute (FLOP)") or ""),
                "hardware_quantity": _format_optional_float(row.get("Hardware quantity") or ""),
                "training_hardware": (row.get("Training hardware") or "").strip(),
                "training_time_hours": _format_optional_float(row.get("Training time (hours)") or ""),
                "mfu": _format_optional_float(row.get("Hardware utilization (MFU)") or ""),
                "frontier": 1 if (row.get("Frontier model") or "").strip() == "True" else 0,
            }
        )

    with_flop = sum(1 for r in selected if r["training_compute_flop"] != "")
    n_frontier = sum(1 for r in selected if r["training_compute_flop"] != "" and r["frontier"] == 1)

    buf = io.StringIO(newline="")
    buf.write(
        f"# Source: Epoch AI, Notable AI Models database (CC BY 4.0), snapshot {snapshot_date} (@epochai2026notable; @sevilla2022)\n"
        f"# URL: {EPOCH_PAGE_URL}\n"
        f"# Verified: {snapshot_date}\n"
        "# Epistemic: modeled (Epoch AI training-compute estimates and builder-disclosed FLOP counts)\n"
        f"# Generator: {GENERATOR_CMD} --write-compute-csv\n"
        f"# Upstream-SHA256: {raw_sha256}\n"
        "# Computation: filter Publication date >= 2012-01-01 with non-empty Training compute (FLOP) or "
        "Hardware quantity, ordered by publication date; @fig-ai-compute-growth filters rows with positive "
        f"training_compute_flop (N={with_flop}, frontier n={n_frontier} where Frontier model == True) and fits "
        "OLS + White HC3 95% confidence bounds on log2(training_compute_flop) vs. decimal publication year\n"
        "# Notes: training_compute_flop follows Epoch AI's cumulative convention (summing parent pretraining and "
        "post-training FLOPs where applicable); hardware_quantity is accelerator count, training_time_hours is "
        "wall-clock duration, mfu is reported model FLOPs utilization, and frontier is Epoch's release-date top-10 flag\n"
    )
    fieldnames = [
        "model",
        "date",
        "domain",
        "training_compute_flop",
        "hardware_quantity",
        "training_hardware",
        "training_time_hours",
        "mfu",
        "frontier",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for item in selected:
        writer.writerow(item)
    return buf.getvalue(), selected


def extract_energy_csv(
    raw_rows: list[dict[str, str]], snapshot_date: str, raw_sha256: str
) -> tuple[str, list[dict[str, object]], dict[str, object]]:
    selected: list[dict[str, object]] = []
    excluded_base_models: list[str] = []
    excluded_peak_warnings: list[str] = []

    for row in raw_rows:
        pub_date = (row.get("Publication date") or "").strip()
        power_raw = (row.get("Training power draw (W)") or "").strip()
        hours_raw = (row.get("Training time (hours)") or "").strip()
        if not (pub_date and power_raw and hours_raw) or pub_date < ENERGY_START_DATE:
            continue
        try:
            dt.date.fromisoformat(pub_date)
            power_w = float(power_raw)
            hours = float(hours_raw)
        except ValueError:
            continue
        if power_w <= 0 or hours <= 0:
            continue

        model_name = row["Model"].strip()
        base_model = (row.get("Base model") or "").strip()
        if base_model:
            excluded_base_models.append(model_name)
            continue

        notes_blob = " ".join(
            (row.get(col) or "")
            for col in ("Training time notes", "Training compute notes", "Compute cost notes")
        ).lower()
        if PEAK_PRODUCT_WARNING in notes_blob:
            excluded_peak_warnings.append(model_name)
            continue

        selected.append(
            {
                "model": model_name,
                "date": pub_date,
                "domain": (row.get("Domain") or "").strip(),
                "training_power_draw_w": power_w,
                "training_time_hours": hours,
                "hardware_quantity": _format_optional_float(row.get("Hardware quantity") or ""),
                "training_hardware": (row.get("Training hardware") or "").strip(),
                "frontier": 1 if (row.get("Frontier model") or "").strip() == "True" else 0,
            }
        )

    selected.sort(key=lambda r: (str(r["date"]), str(r["model"])))
    n_frontier = sum(1 for r in selected if r["frontier"] == 1)

    buf = io.StringIO(newline="")
    buf.write(
        f"# Source: Epoch AI, Notable AI Models database (CC BY 4.0), snapshot {snapshot_date} (@epochai2026notable; @cottier2024rising)\n"
        f"# URL: {EPOCH_PAGE_URL}\n"
        f"# Verified: {snapshot_date}\n"
        "# Epistemic: modeled (peak cluster power capacity integrated over wall-clock pretraining duration; Cottier et al. 2024 App. C)\n"
        f"# Generator: {GENERATOR_CMD}\n"
        f"# Upstream-SHA256: {raw_sha256}\n"
        "# Computation: filter Publication date >= 2016-01-01, Training power draw (W) > 0, Training time (hours) > 0, "
        f"Base model empty (excludes {len(excluded_base_models)} fine-tune/checkpoint rows), and no hardware-quantity * runtime warning "
        "(excludes OpenAI Five); @fig-frontier-training-energy computes E_MWh = training_power_draw_w * training_time_hours / 1e6 "
        "where training_power_draw_w = N_chips * TDP_W * 1.82 * PUE(t) for N_chips > 1 and TDP_W * PUE(t) for N_chips = 1, "
        f"fits OLS + White HC3 95% confidence bounds on log10(E_MWh) across standalone frontier runs (n={n_frontier}), "
        "and rounds modeled upper-bound callouts to 2 significant figures\n"
        f"# Notes: standalone pretraining runs (N={len(selected)}, frontier n={n_frontier}); upper bounds omit Cottier et al. (2024) "
        "App. A.5 average-utilization derating and are contrasted against builder-based anchors at their stated accounting boundaries\n"
    )
    fieldnames = [
        "model",
        "date",
        "domain",
        "training_power_draw_w",
        "training_time_hours",
        "hardware_quantity",
        "training_hardware",
        "frontier",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for item in selected:
        writer.writerow(item)

    exclusions = {
        "excluded_base_model_count": len(excluded_base_models),
        "excluded_invalid_peak_product_count": len(excluded_peak_warnings),
        "excluded_invalid_peak_product_models": excluded_peak_warnings,
    }
    return buf.getvalue(), selected, exclusions


def summarize_and_verify(
    raw_rows: list[dict[str, str]],
    compute_csv_text: str,
    energy_rows: list[dict[str, object]],
    exclusions: dict[str, object],
    compute_csv_path: Path | None,
) -> dict[str, object]:
    compute_rows = []
    for row in raw_rows:
        pub_date = (row.get("Publication date") or "").strip()
        flop_raw = (row.get("Training compute (FLOP)") or "").strip()
        if not (pub_date and flop_raw) or pub_date < COMPUTE_START_DATE:
            continue
        try:
            dt.date.fromisoformat(pub_date)
            flops = float(flop_raw)
        except ValueError:
            continue
        if flops > 0:
            compute_rows.append(
                {
                    "model": row["Model"].strip(),
                    "date": pub_date,
                    "flops": flops,
                    "base_model": (row.get("Base model") or "").strip(),
                    "frontier": 1 if (row.get("Frontier model") or "").strip() == "True" else 0,
                }
            )

    frontier_compute = [r for r in compute_rows if r["frontier"] == 1]
    frontier_compute_standalone = [r for r in frontier_compute if not r["base_model"]]
    fit_c = _ols_with_hc3(
        [_decimal_year(str(r["date"])) for r in frontier_compute],
        [math.log2(float(r["flops"])) for r in frontier_compute],
    )
    fit_c_standalone = _ols_with_hc3(
        [_decimal_year(str(r["date"])) for r in frontier_compute_standalone],
        [math.log2(float(r["flops"])) for r in frontier_compute_standalone],
    )

    if compute_csv_path is not None and compute_csv_path.exists():
        expected_body = "".join(
            line for line in compute_csv_text.splitlines(keepends=True) if not line.startswith("#")
        )
        repo_body = "".join(
            line for line in compute_csv_path.read_text(encoding="utf-8").splitlines(keepends=True) if not line.startswith("#")
        )
        if repo_body != expected_body:
            raise AssertionError(
                f"Compute CSV row body mismatch between raw snapshot and {compute_csv_path.name}"
            )

    frontier_energy = [r for r in energy_rows if r["frontier"] == 1]
    fit_e = _ols_with_hc3(
        [_decimal_year(str(r["date"])) for r in frontier_energy],
        [
            math.log10(float(r["training_power_draw_w"]) * float(r["training_time_hours"]) / 1e6)
            for r in frontier_energy
        ],
    )

    by_model_energy = {
        str(r["model"]): float(r["training_power_draw_w"]) * float(r["training_time_hours"]) / 1e6
        for r in energy_rows
    }
    gpt3_epoch_mwh = by_model_energy["GPT-3 175B (davinci)"]
    llama31_epoch_mwh = by_model_energy["Llama 3.1-405B"]
    grok3_epoch_mwh = by_model_energy["Grok 3"]

    gpt3_reported_facility_mwh = 1287.0  # Patterson et al. (2021), Table 4 (330 W/GPU metered system x PUE 1.10)
    llama31_implied_accel_mwh = 30.84e6 * 700.0 / 1e6  # Meta Llama 3.1 model card (30.84M GPU-h x 700 W TDP)

    return {
        "compute_post_2012": {
            "epistemic_status": "modeled (Epoch AI Notable AI Models cumulative training-compute estimates and builder reports)",
            "n_notable": len(compute_rows),
            "n_frontier": len(frontier_compute),
            "n_frontier_excluding_base_model": len(frontier_compute_standalone),
            "frontier_growth_factor_per_year": 2.0 ** fit_c["slope"],
            "frontier_growth_hc3_ci95": [2.0 ** fit_c["slope_ci95_low"], 2.0 ** fit_c["slope_ci95_high"]],
            "frontier_doubling_months": 12.0 / fit_c["slope"],
            "frontier_doubling_hc3_ci95": [12.0 / fit_c["slope_ci95_high"], 12.0 / fit_c["slope_ci95_low"]],
            "r2": fit_c["r2"],
            "ols_log2_slope_per_year": fit_c["slope"],
            "ols_log2_intercept": fit_c["intercept"],
            "standalone_only_sensitivity_growth_per_year": 2.0 ** fit_c_standalone["slope"],
            "standalone_only_sensitivity_doubling_months": 12.0 / fit_c_standalone["slope"],
        },
        "energy_post_2016": {
            "epistemic_status": "modeled peak-capacity facility upper bound (N_chips * TDP * server_overhead * PUE * wall_clock_hours; Cottier et al. 2024 App. C)",
            "n_standalone_notable": len(energy_rows),
            "n_standalone_frontier": len(frontier_energy),
            "exclusions": exclusions,
            "standalone_frontier_growth_factor_per_year": 10.0 ** fit_e["slope"],
            "standalone_frontier_growth_hc3_ci95": [10.0 ** fit_e["slope_ci95_low"], 10.0 ** fit_e["slope_ci95_high"]],
            "standalone_frontier_doubling_months": 12.0 / (fit_e["slope"] * math.log2(10.0)),
            "r2": fit_e["r2"],
            "ols_log10_slope_per_year": fit_e["slope"],
            "ols_log10_intercept": fit_e["intercept"],
            "gpt3_epoch_upper_mwh": gpt3_epoch_mwh,
            "llama31_epoch_upper_mwh": llama31_epoch_mwh,
            "grok3_epoch_upper_mwh": grok3_epoch_mwh,
        },
        "builder_based_anchors": {
            "gpt3_175b": {
                "epistemic_status": "measured facility boundary (Patterson et al. 2021, arXiv:2104.10350 Table 4: 330 W/GPU metered system draw x PUE 1.10)",
                "reported_mwh": gpt3_reported_facility_mwh,
                "epoch_upper_to_reported_ratio": gpt3_epoch_mwh / gpt3_reported_facility_mwh,
            },
            "llama31_405b": {
                "epistemic_status": "constructed accelerator-TDP boundary from builder-reported inputs (Meta Llama 3.1 Model Card: 30.84M H100 GPU-hours x 700 W TDP, pre-PUE)",
                "implied_accelerator_tdp_mwh": llama31_implied_accel_mwh,
                "epoch_upper_to_implied_ratio": llama31_epoch_mwh / llama31_implied_accel_mwh,
            },
        },
    }


def _repo_rel_or_name(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return str(resolved.relative_to(REPO_ROOT))
    return str(resolved.name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--input", type=Path, help="Local Epoch AI notable_ai_models.csv snapshot")
    src_group.add_argument(
        "--download", action="store_true", help=f"Download live CSV from {EPOCH_CSV_URL}"
    )
    parser.add_argument("--save-raw", type=Path, help="Where to save downloaded raw CSV when --download is set")
    parser.add_argument("--snapshot-date", help="Snapshot YYYY-MM-DD for provenance comments")
    parser.add_argument("--energy-out", type=Path, default=DEFAULT_ENERGY_OUT, help="Output Tier-2 energy CSV path")
    parser.add_argument(
        "--verify-compute-csv",
        type=Path,
        default=DEFAULT_COMPUTE_CSV,
        help="Existing post-2012 compute CSV to cross-verify (or overwrite with --write-compute-csv)",
    )
    parser.add_argument(
        "--write-compute-csv",
        action="store_true",
        help="Also write the post-2012 compute CSV with its structured provenance header",
    )
    parser.add_argument("--manifest", type=Path, help="Optional JSON provenance manifest output path")
    args = parser.parse_args(argv)

    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    raw_bytes, source_desc = load_raw_epoch_bytes(args.input, args.download, args.save_raw)
    raw_sha256 = _sha256_bytes(raw_bytes)
    snapshot_date = _infer_snapshot_date(args.input, args.snapshot_date)

    raw_text = raw_bytes.decode("utf-8")
    raw_rows = list(csv.DictReader(io.StringIO(raw_text)))

    compute_csv_text, compute_all_rows = extract_compute_csv(raw_rows, snapshot_date, raw_sha256)
    if args.write_compute_csv and args.verify_compute_csv is not None:
        args.verify_compute_csv.parent.mkdir(parents=True, exist_ok=True)
        args.verify_compute_csv.write_bytes(compute_csv_text.encode("utf-8"))

    energy_csv_text, energy_rows, exclusions = extract_energy_csv(raw_rows, snapshot_date, raw_sha256)
    energy_bytes = energy_csv_text.encode("utf-8")
    args.energy_out.parent.mkdir(parents=True, exist_ok=True)
    args.energy_out.write_bytes(energy_bytes)
    energy_sha256 = _sha256_bytes(energy_bytes)

    metrics = summarize_and_verify(raw_rows, compute_csv_text, energy_rows, exclusions, args.verify_compute_csv)
    compute_sha256 = (
        _sha256_bytes(args.verify_compute_csv.read_bytes())
        if args.verify_compute_csv is not None and args.verify_compute_csv.exists()
        else _sha256_bytes(compute_csv_text.encode("utf-8"))
    )
    finished_utc = dt.datetime.now(dt.timezone.utc).isoformat()

    cmd_args = [
        "python3",
        "binder/tools/scripts/extract_epoch_sustainability_data.py",
        "--download" if args.download else f"--input {args.input.name if args.input else ''}".strip(),
    ]
    if args.write_compute_csv:
        cmd_args.append("--write-compute-csv")

    manifest = {
        "schema": "epoch-sustainability-extraction-v3",
        "command": " ".join(cmd_args),
        "git": _git_metadata(REPO_ROOT),
        "environment": {
            "python_version": sys.version.split()[0],
            "python_executable": Path(sys.executable).name,
            "platform": platform.platform(),
        },
        "timestamps_utc": {"started": started_utc, "finished": finished_utc},
        "inputs": {
            "epoch_raw_csv": {
                "source": source_desc,
                "url": EPOCH_CSV_URL,
                "snapshot_date": snapshot_date,
                "sha256": raw_sha256,
                "total_rows": len(raw_rows),
            }
        },
        "outputs": {
            "notable_model_training_runs_csv": {
                "repo_relative_path": _repo_rel_or_name(args.verify_compute_csv)
                if args.verify_compute_csv is not None
                else "books/vol2/01_introduction/data/notable_model_training_runs.csv",
                "sha256": compute_sha256,
                "rows": len(compute_all_rows),
            },
            "notable_model_training_energy_csv": {
                "repo_relative_path": _repo_rel_or_name(args.energy_out),
                "sha256": energy_sha256,
                "rows": len(energy_rows),
            },
        },
        "summary_metrics": metrics,
    }

    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    c_stats = metrics["compute_post_2012"]
    e_stats = metrics["energy_post_2016"]
    print(f"Input snapshot ({snapshot_date}) SHA-256: {raw_sha256}")
    if args.write_compute_csv and args.verify_compute_csv is not None:
        print(f"Wrote {args.verify_compute_csv} ({len(compute_all_rows)} rows, SHA-256: {compute_sha256})")
    print(f"Wrote {args.energy_out} ({len(energy_rows)} rows, SHA-256: {energy_sha256})")
    print(
        f"Compute >=2012: N={c_stats['n_notable']}, frontier n={c_stats['n_frontier']}, "
        f"growth={c_stats['frontier_growth_factor_per_year']:.3f}x/yr "
        f"(HC3 95% CI {c_stats['frontier_growth_hc3_ci95'][0]:.2f}..{c_stats['frontier_growth_hc3_ci95'][1]:.2f}, R2={c_stats['r2']:.3f}), "
        f"doubling={c_stats['frontier_doubling_months']:.2f} months"
    )
    print(
        f"Energy  >=2016: N={e_stats['n_standalone_notable']}, standalone frontier n={e_stats['n_standalone_frontier']}, "
        f"growth={e_stats['standalone_frontier_growth_factor_per_year']:.3f}x/yr "
        f"(HC3 95% CI {e_stats['standalone_frontier_growth_hc3_ci95'][0]:.2f}..{e_stats['standalone_frontier_growth_hc3_ci95'][1]:.2f}, R2={e_stats['r2']:.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
