from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mlperf.runners import reinforcement


def test_container_command_requires_gpu_and_immutable_image(tmp_path: Path):
    image = "registry.example/minigo@sha256:" + "a" * 64
    command = reinforcement.build_container_command(
        runtime="docker",
        image=image,
        execution_root=tmp_path / "execution",
        results_mount=tmp_path / "results",
        seed=1,
    )

    assert command[:5] == ["docker", "run", "--rm", "--gpus", "all"]
    assert image in command
    assert command[-1] == "1"
    assert "/research/reinforcement/mlperf_edu_resume.sh" in command


def test_quality_event_parser_reports_latest_and_maximum(tmp_path: Path):
    stats = tmp_path / "stats"
    stats.mkdir()
    events = [
        {
            "name": "puzzle_summary",
            "wall_time": 1,
            "value": {"total_pct": 0.35, "model": "000001-first"},
        },
        {
            "name": "puzzle_summary",
            "wall_time": 2,
            "value": {"total_pct": 0.40, "model": "000002-target"},
        },
        {
            "name": "eval_summary",
            "wall_time": 3,
            "value": {"win_pct": 0.56, "model": "000002-target", "keep": True},
        },
    ]
    (stats / "events.json").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )

    result = reinforcement.parse_quality_events(stats)

    assert result["professional_move_prediction"] == pytest.approx(0.40)
    assert result["max_professional_move_prediction"] == pytest.approx(0.40)
    assert result["evaluated_models"] == 2
    assert result["evaluated_model"] == "000002-target"
    assert result["playoff_win_rate"] == pytest.approx(0.56)
    assert result["model_promoted"] is True


def test_checkpoint_manifest_binds_tensorflow_checkpoint_set(tmp_path: Path):
    results = tmp_path / "results"
    models = results / "models"
    models.mkdir(parents=True)
    model = "000002-target"
    expected_files = []
    for suffix in (".data-00000-of-00001", ".index", ".meta"):
        path = models / f"{model}{suffix}"
        path.write_bytes(suffix.encode())
        expected_files.append(path)
    destination = tmp_path / "checkpoint.json"

    payload, files = reinforcement.build_checkpoint_manifest(
        results_root=results,
        model=model,
        destination=destination,
    )

    assert files == tuple(expected_files)
    assert payload["model"] == model
    assert payload["merkle_root"].startswith("sha256:")
    assert [item["sha256"] for item in payload["files"]] == [
        f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
        for path in expected_files
    ]


def test_preflight_requires_professional_game_review(monkeypatch):
    monkeypatch.delenv(reinforcement.PRO_GAMES_REVIEW_ENV, raising=False)

    with pytest.raises(RuntimeError, match=reinforcement.PRO_GAMES_REVIEW_ENV):
        reinforcement.preflight_environment()


def test_resumable_wrapper_preserves_full_historical_contract():
    wrapper = reinforcement.RESUMABLE_WRAPPER

    assert "loop_init.py" in wrapper
    assert "loop_selfplay.py" in wrapper
    assert "loop_train_eval.py" in wrapper
    assert "TERMINATE_FLAG" in wrapper
    assert "iteration <= 1000" in wrapper


def test_official_contract_constants_are_not_reduced_for_classroom_runs():
    assert reinforcement.TARGET_PROFESSIONAL_MOVE_ACCURACY == pytest.approx(0.40)
    assert reinforcement.SELF_PLAY_GAMES_PER_GENERATION == 2_000
    assert reinforcement.SELF_PLAY_WORKERS == 16
    assert reinforcement.SEARCH_READOUTS == 200
    assert reinforcement.PLAYOFF_GAMES == 100
    assert reinforcement.PLAYOFF_WIN_RATE == pytest.approx(0.55)


def test_environment_handoff_requires_immutable_nvidia_runtime():
    handoff = reinforcement.environment_handoff_contract()

    assert handoff["execution_status"] == "environment-gated-quality-conformance"
    assert handoff["required_hardware"]["accelerator"] == "NVIDIA GPU"
    assert handoff["external_assets"]["professional_games"]["count"] == 4
    assert handoff["external_assets"]["container_image"][
        "immutable_digest_required"
    ]
    assert reinforcement.IMAGE_ENV in handoff["environment"]
    assert handoff["resumable"] is True
    assert handoff["production_ready"] is False
