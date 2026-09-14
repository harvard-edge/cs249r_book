from __future__ import annotations

import hashlib
import json
import pickle
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
        self.round_calls = 0

    def round_sigma(self, values):
        self.round_calls += 1
        return values

    def __call__(self, x, _sigma, _labels):
        self.calls += 1
        return torch.zeros_like(x)


def test_pinned_pickle_is_hashed_before_deserialization(monkeypatch, tmp_path: Path):
    path = tmp_path / "checkpoint.pkl"
    path.write_bytes(b"not the pinned checkpoint")
    deserialized = False

    def unexpected_load(_handle):
        nonlocal deserialized
        deserialized = True
        raise AssertionError("unverified pickle bytes must not be deserialized")

    monkeypatch.setattr(image_generation.pickle, "load", unexpected_load)

    with pytest.raises(ValueError, match="refusing to deserialize"):
        image_generation._load_pinned_pickle(
            path,
            torch.device("cpu"),
            expected_sha256="0" * 64,
        )

    assert not deserialized


def test_pinned_pickle_loads_the_verified_module(tmp_path: Path):
    path = tmp_path / "checkpoint.pkl"
    with path.open("wb") as handle:
        pickle.dump({"ema": torch.nn.Linear(2, 1)}, handle)

    module = image_generation._load_pinned_pickle(
        path,
        torch.device("cpu"),
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        key="ema",
    )

    assert isinstance(module, torch.nn.Linear)
    assert not module.training


def test_mps_adapter_preserves_official_35_evaluation_schedule():
    model = _ZeroDenoiser()
    latents = torch.ones([1, 1, 2, 2])

    result = image_generation.edm_mps_sampler(model, latents, None)

    assert result.shape == latents.shape
    assert result.dtype == torch.float64
    assert torch.isfinite(result).all()
    assert model.calls == image_generation.NETWORK_EVALUATIONS_PER_IMAGE == 35
    assert model.round_calls == 1


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


def test_fid_statistics_use_the_scipy_api_without_removed_disp_argument():
    mu = np.array([0.0, 1.0], dtype=np.float64)
    sigma = np.eye(2, dtype=np.float64)

    score = image_generation.fid_from_statistics(mu, sigma, mu, sigma)

    assert score == pytest.approx(0.0, abs=1e-12)


def test_official_contract_constants_are_not_reduced_for_classroom_runs():
    assert image_generation.EXPECTED_IMAGES == 50_000
    assert image_generation.QUALITY_TRIAL_SEED_STARTS == (0, 50_000, 100_000)
    assert image_generation.NUM_STEPS == 18
    assert image_generation.NETWORK_EVALUATIONS_PER_IMAGE == 35
    assert image_generation.TARGET_FID == pytest.approx(1.79)


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _write_edm_import_packet(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(image_generation, "EXPECTED_IMAGES", 2)
    monkeypatch.setattr(image_generation, "QUALITY_TRIAL_SEED_STARTS", (0, 2, 4))
    generator = tmp_path / "generator.py"
    evaluator = tmp_path / "evaluator.py"
    generator.write_text("# generator\n")
    evaluator.write_text("# evaluator\n")
    trials = []
    for trial, seed_start in enumerate((0, 2, 4), start=1):
        images_root = tmp_path / f"images-{trial}"
        image_records = []
        for seed in range(seed_start, seed_start + 2):
            path = image_generation._seed_image_path(images_root, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(
                np.full((32, 32, 3), seed, dtype=np.uint8), "RGB"
            ).save(path)
            image_records.append(
                {
                    "path": path.relative_to(images_root).as_posix(),
                    "seed": seed,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            )
        manifest = tmp_path / f"source-manifest-{trial}.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": "mlperf-edu-edm-image-set/0.1",
                    "generator_status": {
                        "adapter_sha256": _sha256(generator).removeprefix("sha256:"),
                        "state": "complete",
                        "completed": 2,
                        "requested_seed_range": [seed_start, seed_start + 2],
                        "network_sha256": image_generation.EDM_CIFAR10_CHECKPOINT_SHA256,
                        "upstream_revision": image_generation.EDM_COMMIT,
                        "device": "mps",
                        "elapsed_seconds": 10.0 + trial,
                        "sampler": {
                            "name": "edm",
                            "num_steps": image_generation.NUM_STEPS,
                            "nfe": image_generation.NETWORK_EVALUATIONS_PER_IMAGE,
                            "stochasticity": 0,
                        },
                    },
                    "image_contract": {
                        "count": 2,
                        "mode": "RGB",
                        "resolution": [32, 32],
                    },
                    "images": image_records,
                }
            )
            + "\n"
        )
        evaluation = tmp_path / f"source-evaluation-{trial}.json"
        evaluation.write_text(
            json.dumps(
                {
                    "adapter_sha256": _sha256(evaluator).removeprefix("sha256:"),
                    "state": "complete",
                    "num_expected": 2,
                    "upstream_revision": image_generation.EDM_COMMIT,
                    "detector_sha256": image_generation.EDM_INCEPTION_SHA256,
                    "reference_sha256": image_generation.EDM_CIFAR10_FID_REFERENCE_SHA256,
                    "fid": 1.8 + trial / 100,
                }
            )
            + "\n"
        )
        trials.append(
            {
                "seed_start": seed_start,
                "seed_end": seed_start + 1,
                "images_root": str(images_root),
                "source_manifest": {
                    "path": manifest.name,
                    "sha256": _sha256(manifest),
                },
                "source_evaluation": {
                    "path": evaluation.name,
                    "sha256": _sha256(evaluation),
                },
            }
        )
    packet = tmp_path / "packet.json"
    packet.write_text(
        json.dumps(
            {
                "schema": image_generation.EDM_IMPORT_PACKET_SCHEMA,
                "model": {
                    "checkpoint_sha256": (
                        f"sha256:{image_generation.EDM_CIFAR10_CHECKPOINT_SHA256}"
                    ),
                    "upstream_revision": image_generation.EDM_COMMIT,
                },
                "sampler": {
                    "name": "edm-algorithm-2",
                    "sampler_steps": image_generation.NUM_STEPS,
                    "network_evaluations_per_image": (
                        image_generation.NETWORK_EVALUATIONS_PER_IMAGE
                    ),
                    "stochasticity": 0,
                    "state_precision": "float64-cpu-state-float32-mps-network",
                    "random_device": "cpu",
                },
                "adapters": [
                    {
                        "path": generator.name,
                        "sha256": _sha256(generator),
                        "role": "generator",
                    },
                    {
                        "path": evaluator.name,
                        "sha256": _sha256(evaluator),
                        "role": "evaluator",
                    },
                ],
                "trials": trials,
            }
        )
        + "\n"
    )
    return packet


def test_edm_import_binds_source_code_images_and_prior_evaluation(
    monkeypatch, tmp_path: Path
):
    packet = _write_edm_import_packet(monkeypatch, tmp_path)

    evidence = image_generation.load_edm_generation_evidence(packet)
    first = evidence["trials"][0]
    current_manifest = tmp_path / "current.jsonl"
    image_set = image_generation.build_image_manifest(
        first["images_root"],
        current_manifest,
        seed_start=0,
        expected_manifest=first["source_manifest"],
    )

    assert evidence["mode"] == "imported-hash-bound-generation"
    assert len(evidence["trials"]) == 3
    assert first["prior_fid"] == pytest.approx(1.81)
    assert image_set["images"] == 2


def test_edm_import_rejects_changed_image_bytes(monkeypatch, tmp_path: Path):
    packet = _write_edm_import_packet(monkeypatch, tmp_path)
    evidence = image_generation.load_edm_generation_evidence(packet)
    first = evidence["trials"][0]
    image_path = image_generation._seed_image_path(first["images_root"], 0)
    Image.fromarray(np.ones((32, 32, 3), dtype=np.uint8), "RGB").save(image_path)

    with pytest.raises(ValueError, match="changed at seed 0"):
        image_generation.build_image_manifest(
            first["images_root"],
            tmp_path / "current.jsonl",
            seed_start=0,
            expected_manifest=first["source_manifest"],
        )
