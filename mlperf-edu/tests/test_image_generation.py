from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mlperf.runners import image_generation


class _ZeroDenoiser:
    sigma_min = 0.002
    sigma_max = 80.0

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, x, _sigma, _labels):
        self.calls += 1
        return torch.zeros_like(x)


def test_mps_adapter_preserves_official_35_evaluation_schedule():
    model = _ZeroDenoiser()
    latents = torch.ones([1, 1, 2, 2])

    result = image_generation.edm_mps_sampler(model, latents, None)

    assert result.shape == latents.shape
    assert torch.isfinite(result).all()
    assert model.calls == image_generation.NETWORK_EVALUATIONS_PER_IMAGE == 35


def test_per_seed_randomness_is_independent_of_batch_partitioning():
    together = image_generation.StackedRandomGenerator(
        torch.device("cpu"), [7, 11]
    ).randn([2, 3], device=torch.device("cpu"))
    separate = torch.cat(
        [
            image_generation.StackedRandomGenerator(torch.device("cpu"), [seed]).randn(
                [1, 3], device=torch.device("cpu")
            )
            for seed in [7, 11]
        ]
    )

    assert torch.equal(together, separate)


def test_image_manifest_binds_every_seed_addressed_png(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(image_generation, "EXPECTED_IMAGES", 2)
    images_root = tmp_path / "images"
    for seed, value in enumerate((0, 255)):
        path = image_generation._seed_image_path(images_root, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((32, 32, 3), value, dtype=np.uint8), "RGB").save(path)
    manifest_path = tmp_path / "images.jsonl"

    result = image_generation.build_image_manifest(images_root, manifest_path)
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]

    assert result["images"] == 2
    assert result["merkle_root"].startswith("sha256:")
    assert result["manifest_sha256"].startswith("sha256:")
    assert [record["seed"] for record in records] == [0, 1]
    assert all(record["sha256"].startswith("sha256:") for record in records)


def test_image_manifest_preserves_nonzero_trial_seed_range(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(image_generation, "EXPECTED_IMAGES", 2)
    images_root = tmp_path / "images"
    for seed in (50_000, 50_001):
        path = image_generation._seed_image_path(images_root, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), "RGB").save(path)
    manifest_path = tmp_path / "images.jsonl"

    result = image_generation.build_image_manifest(
        images_root, manifest_path, seed_start=50_000
    )
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]

    assert result["seed_start"] == 50_000
    assert result["seed_end"] == 50_001
    assert [record["seed"] for record in records] == [50_000, 50_001]


def test_quality_score_is_minimum_of_exactly_three_trials():
    assert image_generation.quality_score_from_trials([1.9, 1.8, 1.85]) == 1.8
    with pytest.raises(ValueError, match="exactly three"):
        image_generation.quality_score_from_trials([1.8])


def test_official_contract_constants_are_not_reduced_for_classroom_runs():
    assert image_generation.EXPECTED_IMAGES == 50_000
    assert image_generation.QUALITY_TRIAL_SEED_STARTS == (0, 50_000, 100_000)
    assert image_generation.NUM_STEPS == 18
    assert image_generation.NETWORK_EVALUATIONS_PER_IMAGE == 35
    assert image_generation.TARGET_FID == pytest.approx(1.79)
