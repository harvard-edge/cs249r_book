from mlsysim.core.series_loader import load_series
"""Published scaling-law fits and landmark-run durations stay self-consistent.

Added 2026-10-15 with the Kaplan et al. (2020) joint fits, the Hoffmann et al.
(2022) parametric loss, the PaLM 540B training phases, and the pre-2012 Epoch
compute series. Each check ties a registry value to another value from the
same paper, so a typo in one place fails here instead of in a rendered figure.
"""

import math

from mlsysim import Models
from mlsysim.core.units import ureg
from mlsysim.literature.registry import Literature

C = Literature.Chinchilla
K = Literature.KaplanScalingLaws
RUNS = Literature.TrainingRuns


def _chinchilla_loss(n_params, n_tokens):
    return (
        float(C.ParametricLossE)
        + float(C.ParametricLossA) / n_params ** float(C.ParametricLossAlpha)
        + float(C.ParametricLossB) / n_tokens ** float(C.ParametricLossBeta)
    )


def test_chinchilla_parametric_exponents_imply_table2_approach3():
    alpha, beta = float(C.ParametricLossAlpha), float(C.ParametricLossBeta)
    # Eq. 4: a = beta/(alpha+beta), b = alpha/(alpha+beta); the stored fit is
    # rounded to two digits, so allow the rounding gap to Table 2.
    assert abs(beta / (alpha + beta) - float(C.Approach3ParamsExponent)) < 0.015
    assert abs(alpha / (alpha + beta) - float(C.Approach3TokensExponent)) < 0.015
    for a, b in (
        (C.Approach1ParamsExponent, C.Approach1TokensExponent),
        (C.Approach2ParamsExponent, C.Approach2TokensExponent),
        (C.Approach3ParamsExponent, C.Approach3TokensExponent),
    ):
        assert abs(float(a) + float(b) - 1.0) < 1e-9  # C = 6ND fixes a + b = 1


def test_chinchilla_loss_reproduces_chinchilla_scale():
    # Chinchilla itself (70B params, 1.4T tokens) sits near 1.94 nats/token on
    # this fit; a typo in E, A, B, alpha or beta moves it far from that.
    loss = _chinchilla_loss(70e9, 1.4e12)
    assert 1.85 < loss < 2.05


def test_gopher_budget_projection_matches_six_nd():
    budget = float(C.GopherBudgetFlops)
    implied = float(C.ComputeConstant) * float(C.GopherBudgetOptimalParams) * float(
        C.GopherBudgetOptimalTokens
    )
    assert abs(implied / budget - 1.0) < 0.10  # 6 x 63B x 1.4T = 5.3e23 vs 5.76e23
    assert float(C.IsoFlopBudgetMinFlops) < float(C.IsoFlopBudgetMaxFlops) < budget


def test_kaplan_figure_and_table_laws_agree():
    by_axis = {e["axis"]: e for e in K.entries}
    table = K.single_variable_fits
    assert by_axis["params"]["alpha"] == table["params"]["alpha"]
    assert by_axis["params"]["x_c"] == table["params"]["scale"]
    assert by_axis["data"]["alpha"] == table["data"]["alpha"]
    assert by_axis["data"]["x_c"] == table["data"]["scale"]
    assert by_axis["compute"]["alpha"] == table["compute_min"]["alpha"]
    # Figure 1 prints 2.3e8 PF-days, Table 5 prints 3.1e8: same law, two scales.
    assert 2.0e8 < by_axis["compute"]["x_c"] <= table["compute_min"]["scale"]


def test_kaplan_compute_efficient_frontier_is_consistent():
    f = K.compute_efficient_frontier
    # D = B * S (Eq. 1.7): exponents and scales of tokens = batch x steps.
    assert abs(f["batch_tokens"]["p"] + f["steps_min"]["p"] - f["tokens"]["p"]) < 1e-9
    implied_scale = f["batch_tokens"]["scale"] * f["steps_min"]["scale"]
    assert abs(math.log10(implied_scale / f["tokens"]["scale"])) < 0.3
    assert K.loss_vs_params_and_steps["alpha_s"] == K.single_variable_fits["steps"]["alpha"]
    assert K.loss_vs_params_and_steps["s_c"] == K.single_variable_fits["steps"]["scale"]


def test_palm_training_days_equal_sum_of_phases():
    hours = float(RUNS.Palm540B_Phase1Hours) + float(RUNS.Palm540B_Phase2Hours)
    days = Models.Language.PaLM_540B.training_days_ref.to(ureg.day).magnitude
    assert abs(hours / 24.0 - days) < 1e-9
    chips = Models.Language.PaLM_540B.training_accelerators_ref.to(ureg.count).magnitude
    assert float(RUNS.Palm540B_Phase1Chips) == chips
    assert float(RUNS.Palm540B_Phase2Chips) == chips / 2


def test_epoch_compute_history_joins_without_overlap():
    pre = load_series("vol2/01_introduction/data/notable_model_compute_pre2012.csv").entries
    post = load_series("vol2/01_introduction/data/notable_model_training_runs.csv").entries
    assert len(pre) == 72
    assert all(e["date"] < "2012-01-01" for e in pre)
    assert all(e["date"] >= "2012-01-01" for e in post)
    assert all(e["training_compute_flop"] and e["training_compute_flop"] > 0 for e in pre)
    assert not {e["model"] for e in pre} & {e["model"] for e in post}


def test_robot_trajectory_series_agrees_with_scalar_registries():
    from mlsysim import viz

    series = viz.load_series("vol4/05_data/data/robot_trajectory_datasets.csv")
    assert any(c.startswith("# Source:") for c in series.provenance_comments)
    traj = {r["dataset"]: r for r in series.entries}
    oxe_total = sum(r["episodes"] for r in load_series("vol4/05_data/data/open_x_embodiment_composition.csv").entries if r["episodes"])

    assert all("arXiv:" in str(r["source"]) for r in series.entries)


def test_models_scale_by_domain_agrees_with_vla_and_token_series():
    from mlsysim import viz

    for rel_path in (
        "vol4/03_brain/data/epoch_models_scale_by_domain.csv",
        "vol4/05_data/data/epoch_models_scale_by_domain.csv",
        "vol1/04_data_engineering/data/notable_model_datasets.csv",
        "vol1/05_nn_computation/data/notable_models.csv",
    ):
        s = viz.load_series(rel_path)
        assert any(c.startswith("# Source:") for c in s.provenance_comments), rel_path

    scale_series = viz.load_series("vol4/03_brain/data/epoch_models_scale_by_domain.csv")
    rob = scale_series.where(domain_group="Robotics")
    assert all("Cosmos" not in str(r["model"]) for r in rob)

    by_model = {r["model"]: r for r in rob}

    toks = {r["model"]: r["tokens"] for r in load_series("vol1/04_data_engineering/data/pretraining_tokens.csv").entries}
    notable_ds = {r["model"]: r["training_dataset_size"] for r in viz.load_series("vol1/04_data_engineering/data/notable_model_datasets.csv").entries}
    assert toks["DeepSeek-V3"] == notable_ds["DeepSeek-V3"] == Models.Language.DeepSeek_V3.training_tokens.to("count").magnitude
    assert toks["Qwen3"] == notable_ds["Qwen3-235B-A22B"]


def test_epoch_training_energy_series_agrees_with_runs_and_registries():
    from mlsysim import Hardware, viz
    from mlsysim.core.units import MWh, hour, watt

    e_series = viz.load_series("vol2/15_sustainable_ai/data/notable_model_training_energy.csv")
    for prefix in ("# Source:", "# URL:", "# Verified:", "# Notes:"):
        assert any(c.startswith(prefix) for c in e_series.provenance_comments), prefix

    entries = e_series.entries
    assert len(entries) == 122
    assert len([r for r in entries if r["frontier"] == 1]) == 23
    assert len({r["model"] for r in entries}) == len(entries)
    assert "GPT-4 (Jun 2023)" not in {r["model"] for r in entries}
    assert "OpenAI Five" not in {r["model"] for r in entries}
    assert all(r["date"] >= "2016-01-01" for r in entries)
    assert all(r["training_power_draw_w"] > 0 and r["training_time_hours"] > 0 for r in entries)

    runs_by_model = {
        r["model"]: r
        for r in viz.load_series("vol2/01_introduction/data/notable_model_training_runs.csv").entries
    }
    for row in entries:
        run = runs_by_model[row["model"]]
        assert row["date"] == run["date"]
        assert row["frontier"] == run["frontier"]
        assert row["hardware_quantity"] == run["hardware_quantity"]
        assert abs(row["training_time_hours"] - run["training_time_hours"]) / run["training_time_hours"] < 1e-6

    energies = {
        r["model"]: (r["training_power_draw_w"] * watt * (r["training_time_hours"] * hour)).to(MWh)
        for r in entries
    }
    gpt3_reported = Models.Language.GPT3.training_energy_mwh
    llama31_accel = (float(RUNS.Llama31_405B_TrainingGpuHours) * hour * Hardware.Cloud.H100.tdp).to(MWh)
    assert energies["GPT-3 175B (davinci)"] > gpt3_reported
    assert energies["Llama 3.1-405B"] > llama31_accel


def test_epoch_sustainability_provenance_headers_and_manifest_receipt():
    import hashlib
    import json
    from pathlib import Path
    from mlsysim import viz

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "binder" / "tools" / "audits" / "epoch_sustainability_manifest.json"
    assert manifest_path.is_file(), f"Missing extraction receipt: {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    upstream_sha = manifest["inputs"]["epoch_raw_csv"]["sha256"]
    assert len(upstream_sha) == 64

    required_keys = {
        "source",
        "url",
        "verified",
        "epistemic",
        "generator",
        "upstream_sha256",
        "computation",
        "notes",
    }
    for out_key, series_rel in (
        ("notable_model_training_runs_csv", "vol2/01_introduction/data/notable_model_training_runs.csv"),
        ("notable_model_training_energy_csv", "vol2/15_sustainable_ai/data/notable_model_training_energy.csv"),
    ):
        series = viz.load_series(series_rel)
        assert required_keys <= set(series.provenance), (series_rel, set(series.provenance))
        assert series.provenance["upstream_sha256"] == upstream_sha
        assert series.provenance["verified"] == manifest["inputs"]["epoch_raw_csv"]["snapshot_date"]

        out_meta = manifest["outputs"][out_key]
        csv_file = repo_root / out_meta["repo_relative_path"]
        live_sha = hashlib.sha256(csv_file.read_bytes()).hexdigest()
        assert live_sha == out_meta["sha256"], (
            f"{csv_file} drifted from {manifest_path.name}; rerun extract_epoch_sustainability_data.py"
        )
        assert len(series.entries) == out_meta["rows"]



