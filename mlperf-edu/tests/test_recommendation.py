from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlperf.runners import recommendation


def test_official_command_preserves_complete_accuracy_contract(tmp_path: Path):
    command = recommendation.build_official_command(
        python="/runtime/python",
        inference_root=tmp_path / "inference",
        checkpoint=tmp_path / "tb00_40M.pt",
        data_dir=tmp_path / "criteo",
        official_output=tmp_path / "official",
        trace_path=tmp_path / "trace.txt",
        seed=42,
        device="cpu",
    )

    assert "--accuracy" in command
    assert command[command.index("--max-ind-range") + 1] == "40000000"
    assert command[command.index("--samples-per-query-offline") + 1] == "204800"
    assert command[command.index("--max-batchsize") + 1] == "2048"
    assert "--count-samples" not in command
    assert "--count-queries" not in command
    assert "--use-gpu" not in command


def test_gpu_command_only_adds_official_device_flag(tmp_path: Path):
    command = recommendation.build_official_command(
        python="python",
        inference_root=tmp_path / "inference",
        checkpoint=tmp_path / "checkpoint.pt",
        data_dir=tmp_path / "data",
        official_output=tmp_path / "official",
        trace_path=tmp_path / "trace.txt",
        seed=1,
        device="gpu",
    )

    assert command[-1] == "--use-gpu"


def test_results_parser_converts_official_percent_auc(tmp_path: Path):
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "runtime": "pytorch-native-dlrm",
                "version": "1.9.0",
                "TestScenario.Offline": {
                    "roc_auc": 80.25,
                    "accuracy": 78.1,
                    "took": 12.5,
                    "qps": 100.0,
                    "count": 42,
                    "total_items": 89_137_319,
                },
            }
        )
    )

    result = recommendation.parse_official_results(path)

    assert result["roc_auc"] == pytest.approx(0.8025)
    assert result["official_roc_auc_percent"] == pytest.approx(80.25)
    assert result["evaluated_pairs"] == 89_137_319


def test_preflight_requires_explicit_criteo_terms_acceptance(monkeypatch):
    monkeypatch.delenv(recommendation.TERMS_ENV, raising=False)

    with pytest.raises(RuntimeError, match=recommendation.TERMS_ENV):
        recommendation.preflight_environment()


def test_preprocessed_file_contract_covers_all_24_days(tmp_path: Path):
    files = recommendation.required_preprocessed_files(tmp_path)

    assert len(files) == 26
    assert files[0].name == "day_day_count.npz"
    assert files[1].name == "day_fea_count.npz"
    assert files[2].name == "day_0_reordered.npz"
    assert files[-1].name == "day_23_reordered.npz"
