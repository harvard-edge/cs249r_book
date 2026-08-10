from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from mlperf.assets import (
    EDM_CIFAR10_CHECKPOINT_SHA256,
    EDM_CIFAR10_FID_REFERENCE_SHA256,
    EDM_COMMIT,
    EDM_INCEPTION_SHA256,
    EDM_SOURCE_FILES,
    edm_cifar10_paths,
    ensure_edm_cifar10,
    sha256_file,
)
from mlperf.fingerprint import detect_hardware
from mlperf.manifest import build_provd
from mlperf.registry import Workload, find_project_root
from mlperf.runners.common import (
    configured_seed,
    select_torch_device,
    synchronize_device,
)


EXPECTED_IMAGES = 50_000
QUALITY_TRIAL_SEED_STARTS = (0, 50_000, 100_000)
NUM_STEPS = 18
NETWORK_EVALUATIONS_PER_IMAGE = 35
TARGET_FID = 1.79
FEATURE_DIMENSION = 2048
EDM_IMPORT_PACKET_ENV = "MLPERF_EDU_EDM_IMPORT_PACKET"
EDM_IMPORT_PACKET_SCHEMA = "mlperf-edu-edm-generation-evidence/0.1"
EDM_IMPORT_FID_TOLERANCE = 1e-6


class StackedRandomGenerator:
    """NVIDIA EDM's independent per-seed random generator."""

    def __init__(self, device: torch.device, seeds: list[int]) -> None:
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size: list[int], **kwargs: Any) -> torch.Tensor:
        if size[0] != len(self.generators):
            raise ValueError("EDM batch size and seed count do not match")
        return torch.stack(
            [
                torch.randn(size[1:], generator=generator, **kwargs)
                for generator in self.generators
            ]
        )

    def randint(self, *args: Any, size: list[int], **kwargs: Any) -> torch.Tensor:
        if size[0] != len(self.generators):
            raise ValueError("EDM batch size and seed count do not match")
        return torch.stack(
            [
                torch.randint(*args, size=size[1:], generator=generator, **kwargs)
                for generator in self.generators
            ]
        )


def edm_mps_sampler(
    net: Any,
    latents: torch.Tensor,
    class_labels: torch.Tensor | None,
    *,
    num_steps: int = NUM_STEPS,
    sigma_min: float = 0.002,
    sigma_max: float = 80,
    rho: float = 7,
    network_device: torch.device | None = None,
) -> torch.Tensor:
    """Run Algorithm 2 with float64 CPU state and float32 MPS evaluations."""
    sigma_min = max(sigma_min, float(net.sigma_min))
    sigma_max = min(sigma_max, float(net.sigma_max))
    step_indices = torch.arange(num_steps, dtype=torch.float64, device="cpu")
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    if hasattr(net, "round_sigma"):
        t_steps = net.round_sigma(t_steps)
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    device = network_device or latents.device
    device_labels = None if class_labels is None else class_labels.to(device)

    def denoise(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        result = net(
            x.to(torch.float32).to(device),
            sigma.to(torch.float32).to(device),
            device_labels,
        )
        return result.to("cpu").to(torch.float64)

    x_next = latents.to("cpu").to(torch.float64) * t_steps[0]
    for index, (t_cur, t_next) in enumerate(
        zip(t_steps[:-1], t_steps[1:], strict=True)
    ):
        x_hat = x_next
        denoised = denoise(x_hat, t_cur)
        d_cur = (x_hat - denoised) / t_cur
        x_next = x_hat + (t_next - t_cur) * d_cur
        if index < num_steps - 1:
            denoised = denoise(x_next, t_next)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next


def _import_edm_source(source_root: Path) -> Any:
    source_text = str(source_root.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from generate import edm_sampler

    return edm_sampler


def _load_pinned_pickle(
    path: Path,
    device: torch.device,
    *,
    expected_sha256: str,
    key: str | None = None,
) -> Any:
    """Deserialize one trusted upstream pickle only after checking its bytes."""
    expected = expected_sha256.removeprefix("sha256:")
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
        observed = digest.hexdigest()
        if observed != expected:
            raise ValueError(
                f"refusing to deserialize {path.name}: expected SHA-256 "
                f"{expected}, found {observed}"
            )
        handle.seek(0)
        payload = pickle.load(handle)

    if key is not None:
        if not isinstance(payload, dict) or key not in payload:
            raise ValueError(f"pinned pickle does not contain the required {key!r} key")
        value = payload[key]
    else:
        value = payload
    if not callable(getattr(value, "to", None)) or not callable(
        getattr(value, "eval", None)
    ):
        raise TypeError("pinned pickle did not contain a loadable PyTorch module")
    return value.to(device).eval()


def _seed_image_path(images_root: Path, seed: int) -> Path:
    return images_root / f"{seed - seed % 1000:06d}" / f"{seed:06d}.png"


def _load_progress(path: Path, *, batch_size: int) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid EDM progress JSONL at line {line_number}"
            ) from exc
        start_seed = record.get("start_seed")
        end_seed = record.get("end_seed")
        if (
            not isinstance(start_seed, int)
            or not isinstance(end_seed, int)
            or start_seed < 0
            or end_seed <= start_seed
            or record.get("batch_size") != batch_size
            or record.get("source_revision") != EDM_COMMIT
        ):
            raise ValueError("EDM progress file does not match this run contract")
        if start_seed in records:
            raise ValueError(f"duplicate EDM progress batch at seed {start_seed}")
        records[start_seed] = record
    return records


def _validate_completed_batch(
    images_root: Path, *, start_seed: int, end_seed: int
) -> None:
    for seed in range(start_seed, end_seed):
        path = _seed_image_path(images_root, seed)
        if not path.is_file():
            raise FileNotFoundError(f"EDM completed batch is missing {path}")
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (32, 32):
                raise ValueError(
                    f"EDM generated image has invalid shape or mode: {path}"
                )


def _generate_images(
    *,
    net: Any,
    device: torch.device,
    official_sampler: Any,
    images_root: Path,
    progress_path: Path,
    batch_size: int,
    seed_start: int = 0,
) -> tuple[float, int]:
    progress = _load_progress(progress_path, batch_size=batch_size)
    for start_seed, record in sorted(progress.items()):
        _validate_completed_batch(
            images_root, start_seed=start_seed, end_seed=int(record["end_seed"])
        )
    mode = "a" if progress else "w"
    with progress_path.open(mode, encoding="utf-8") as progress_handle:
        seed_end = seed_start + EXPECTED_IMAGES
        for start_seed in range(seed_start, seed_end, batch_size):
            end_seed = min(start_seed + batch_size, seed_end)
            if start_seed in progress:
                if int(progress[start_seed]["end_seed"]) != end_seed:
                    raise ValueError("EDM resumed batch boundary changed")
                continue
            seeds = list(range(start_seed, end_seed))
            random_device = torch.device("cpu") if device.type == "mps" else device
            random = StackedRandomGenerator(random_device, seeds)
            latents = random.randn(
                [len(seeds), net.img_channels, net.img_resolution, net.img_resolution],
                device=random_device,
            )
            class_labels = None
            if net.label_dim:
                labels = random.randint(
                    net.label_dim, size=[len(seeds)], device=random_device
                )
                class_labels = torch.eye(net.label_dim, device=random_device)[labels]
            synchronize_device(device)
            start = time.perf_counter()
            with torch.inference_mode():
                if device.type == "mps":
                    images = edm_mps_sampler(
                        net, latents, class_labels, network_device=device
                    )
                else:
                    images = official_sampler(net, latents, class_labels)
            synchronize_device(device)
            duration = time.perf_counter() - start
            images_uint8 = (
                (images * 127.5 + 128)
                .clip(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
            for seed, image in zip(seeds, images_uint8, strict=True):
                destination = _seed_image_path(images_root, seed)
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image, "RGB").save(destination)
            record = {
                "start_seed": start_seed,
                "end_seed": end_seed,
                "batch_size": batch_size,
                "generation_seconds": duration,
                "source_revision": EDM_COMMIT,
            }
            progress_handle.write(json.dumps(record, sort_keys=True) + "\n")
            progress_handle.flush()
            progress[start_seed] = record
    return (
        sum(float(record["generation_seconds"]) for record in progress.values()),
        sum(int(record["end_seed"]) - start for start, record in progress.items()),
    )


def build_image_manifest(
    images_root: Path,
    manifest_path: Path,
    *,
    seed_start: int = 0,
    expected_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Hash all 50,000 seed-addressed PNGs into a portable image-set root."""
    expected_images = None
    if expected_manifest is not None:
        expected_images = expected_manifest.get("images")
        if not isinstance(expected_images, list) or len(expected_images) != EXPECTED_IMAGES:
            raise ValueError("EDM source image manifest has the wrong image count")
    root_hash = hashlib.sha256()
    total_bytes = 0
    with manifest_path.open("w", encoding="utf-8") as handle:
        for index, seed in enumerate(range(seed_start, seed_start + EXPECTED_IMAGES)):
            path = _seed_image_path(images_root, seed)
            if not path.is_file():
                raise FileNotFoundError(f"EDM image set is incomplete: {path}")
            digest = sha256_file(path)
            size = path.stat().st_size
            relative = path.relative_to(images_root).as_posix()
            if expected_images is not None and expected_images[index] != {
                "path": relative,
                "seed": seed,
                "sha256": digest,
                "size_bytes": size,
            }:
                raise ValueError(f"EDM imported image evidence changed at seed {seed}")
            record = {
                "seed": seed,
                "path": relative,
                "sha256": f"sha256:{digest}",
                "n_bytes": size,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            root_hash.update(
                json.dumps(record, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            )
            root_hash.update(b"\0")
            total_bytes += size
    return {
        "images": EXPECTED_IMAGES,
        "seed_start": seed_start,
        "seed_end": seed_start + EXPECTED_IMAGES - 1,
        "n_bytes": total_bytes,
        "merkle_root": f"sha256:{root_hash.hexdigest()}",
        "manifest_sha256": f"sha256:{sha256_file(manifest_path)}",
    }


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label} JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _resolve_import_file(packet_path: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"EDM import packet {label} must be a nonempty path")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = packet_path.parent / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"EDM import packet {label} is missing: {candidate}")
    return candidate


def _resolve_import_directory(packet_path: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"EDM import packet {label} must be a nonempty path")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = packet_path.parent / candidate
    candidate = candidate.resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"EDM import packet {label} is missing: {candidate}")
    return candidate


def _verified_import_file(
    packet_path: Path, entry: Any, *, label: str
) -> tuple[Path, str]:
    if not isinstance(entry, dict):
        raise ValueError(f"EDM import packet {label} entry is invalid")
    path = _resolve_import_file(packet_path, entry.get("path"), label=label)
    expected = entry.get("sha256")
    actual = f"sha256:{sha256_file(path)}"
    if expected != actual:
        raise ValueError(f"EDM import SHA-256 mismatch for {label}")
    return path, actual


def load_edm_generation_evidence(packet_path: Path) -> dict[str, Any]:
    """Validate retained three-trial EDM generation evidence before rescoring."""
    packet_path = packet_path.expanduser().resolve()
    if not packet_path.is_file():
        raise FileNotFoundError(f"EDM import packet is missing: {packet_path}")
    packet = _load_json_object(packet_path, label="EDM import packet")
    if packet.get("schema") != EDM_IMPORT_PACKET_SCHEMA:
        raise ValueError(
            f"EDM import packet schema must be {EDM_IMPORT_PACKET_SCHEMA}"
        )
    if packet.get("model") != {
        "checkpoint_sha256": f"sha256:{EDM_CIFAR10_CHECKPOINT_SHA256}",
        "upstream_revision": EDM_COMMIT,
    }:
        raise ValueError("EDM import packet model pin does not match the runner")
    expected_sampler = {
        "name": "edm-algorithm-2",
        "sampler_steps": NUM_STEPS,
        "network_evaluations_per_image": NETWORK_EVALUATIONS_PER_IMAGE,
        "stochasticity": 0,
        "state_precision": "float64-cpu-state-float32-mps-network",
        "random_device": "cpu",
    }
    if packet.get("sampler") != expected_sampler:
        raise ValueError("EDM import packet sampler does not match the MPS adapter")

    adapters = packet.get("adapters")
    if not isinstance(adapters, list) or not adapters:
        raise ValueError("EDM import packet must bind its generator and evaluator code")
    adapter_by_digest: dict[str, dict[str, Any]] = {}
    source_files: list[Path] = [packet_path]
    adapter_evidence: list[dict[str, Any]] = []
    for index, entry in enumerate(adapters):
        path, digest = _verified_import_file(
            packet_path, entry, label=f"adapter {index + 1}"
        )
        role = entry.get("role")
        if role not in {"generator", "evaluator"}:
            raise ValueError("EDM import adapter role must be generator or evaluator")
        if digest in adapter_by_digest:
            raise ValueError("EDM import adapter SHA-256 values must be unique")
        adapter_by_digest[digest] = {"path": path, "role": role}
        adapter_evidence.append(
            {"path": str(path), "sha256": digest, "role": role}
        )
        source_files.append(path)

    trials = packet.get("trials")
    if not isinstance(trials, list) or len(trials) != len(QUALITY_TRIAL_SEED_STARTS):
        raise ValueError("EDM import packet must contain exactly three trials")
    trial_evidence: list[dict[str, Any]] = []
    for trial_index, (entry, seed_start) in enumerate(
        zip(trials, QUALITY_TRIAL_SEED_STARTS, strict=True), start=1
    ):
        if not isinstance(entry, dict):
            raise ValueError(f"EDM import trial {trial_index} is invalid")
        seed_end = seed_start + EXPECTED_IMAGES
        if entry.get("seed_start") != seed_start or entry.get("seed_end") != seed_end - 1:
            raise ValueError(f"EDM import seed range changed for trial {trial_index}")
        images_root = _resolve_import_directory(
            packet_path, entry.get("images_root"), label=f"trial {trial_index} images"
        )
        manifest_path, manifest_digest = _verified_import_file(
            packet_path,
            entry.get("source_manifest"),
            label=f"trial {trial_index} source manifest",
        )
        evaluation_path, evaluation_digest = _verified_import_file(
            packet_path,
            entry.get("source_evaluation"),
            label=f"trial {trial_index} source evaluation",
        )
        source_manifest = _load_json_object(
            manifest_path, label=f"EDM trial {trial_index} source manifest"
        )
        status = source_manifest.get("generator_status")
        if (
            source_manifest.get("schema") != "mlperf-edu-edm-image-set/0.1"
            or not isinstance(status, dict)
            or status.get("state") != "complete"
            or status.get("completed") != EXPECTED_IMAGES
            or status.get("requested_seed_range") != [seed_start, seed_end]
            or status.get("network_sha256") != EDM_CIFAR10_CHECKPOINT_SHA256
            or status.get("upstream_revision") != EDM_COMMIT
            or status.get("device") != "mps"
            or status.get("sampler")
            != {
                "name": "edm",
                "num_steps": NUM_STEPS,
                "nfe": NETWORK_EVALUATIONS_PER_IMAGE,
                "stochasticity": 0,
            }
        ):
            raise ValueError(f"EDM source generation contract changed for trial {trial_index}")
        adapter_digest = f"sha256:{status.get('adapter_sha256')}"
        adapter = adapter_by_digest.get(adapter_digest)
        if adapter is None or adapter["role"] != "generator":
            raise ValueError(f"EDM generator adapter is unbound for trial {trial_index}")
        image_contract = source_manifest.get("image_contract")
        source_images = source_manifest.get("images")
        if (
            not isinstance(image_contract, dict)
            or image_contract.get("count") != EXPECTED_IMAGES
            or image_contract.get("mode") != "RGB"
            or image_contract.get("resolution") != [32, 32]
            or not isinstance(source_images, list)
            or len(source_images) != EXPECTED_IMAGES
        ):
            raise ValueError(f"EDM source image contract changed for trial {trial_index}")
        for offset, record in enumerate(source_images):
            seed = seed_start + offset
            if (
                not isinstance(record, dict)
                or record.get("seed") != seed
                or record.get("path")
                != _seed_image_path(Path(), seed).as_posix()
                or not isinstance(record.get("sha256"), str)
                or len(record["sha256"]) != 64
                or not isinstance(record.get("size_bytes"), int)
                or record["size_bytes"] <= 0
            ):
                raise ValueError(f"EDM source manifest changed at seed {seed}")

        source_evaluation = _load_json_object(
            evaluation_path, label=f"EDM trial {trial_index} source evaluation"
        )
        evaluator_digest = f"sha256:{source_evaluation.get('adapter_sha256')}"
        evaluator_adapter = adapter_by_digest.get(evaluator_digest)
        prior_fid = source_evaluation.get("fid")
        if (
            source_evaluation.get("state") != "complete"
            or source_evaluation.get("num_expected") != EXPECTED_IMAGES
            or source_evaluation.get("upstream_revision") != EDM_COMMIT
            or source_evaluation.get("detector_sha256") != EDM_INCEPTION_SHA256
            or source_evaluation.get("reference_sha256")
            != EDM_CIFAR10_FID_REFERENCE_SHA256
            or evaluator_adapter is None
            or evaluator_adapter["role"] != "evaluator"
            or not isinstance(prior_fid, (int, float))
            or not np.isfinite(float(prior_fid))
            or float(prior_fid) <= 0
        ):
            raise ValueError(f"EDM source evaluation changed for trial {trial_index}")
        generation_seconds = status.get("elapsed_seconds")
        if (
            not isinstance(generation_seconds, (int, float))
            or not np.isfinite(float(generation_seconds))
            or float(generation_seconds) <= 0
        ):
            raise ValueError(f"EDM generation duration is invalid for trial {trial_index}")
        source_files.extend([manifest_path, evaluation_path])
        trial_evidence.append(
            {
                "trial": trial_index,
                "seed_start": seed_start,
                "seed_end": seed_end - 1,
                "images_root": images_root,
                "source_manifest": source_manifest,
                "source_manifest_path": manifest_path,
                "source_manifest_sha256": manifest_digest,
                "source_evaluation_path": evaluation_path,
                "source_evaluation_sha256": evaluation_digest,
                "generation_seconds": float(generation_seconds),
                "prior_fid": float(prior_fid),
                "generator_adapter_sha256": adapter_digest,
                "evaluator_adapter_sha256": evaluator_digest,
            }
        )

    return {
        "mode": "imported-hash-bound-generation",
        "schema": EDM_IMPORT_PACKET_SCHEMA,
        "packet": str(packet_path),
        "packet_sha256": f"sha256:{sha256_file(packet_path)}",
        "sampler": expected_sampler,
        "adapters": adapter_evidence,
        "trials": trial_evidence,
        "source_files": source_files,
    }


def _image_batch(paths: list[Path]) -> torch.Tensor:
    arrays: list[np.ndarray] = []
    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        arrays.append(array)
    batch = np.stack(arrays).transpose(0, 3, 1, 2)
    return torch.from_numpy(batch)


def fid_from_statistics(
    mu: np.ndarray,
    sigma: np.ndarray,
    reference_mu: np.ndarray,
    reference_sigma: np.ndarray,
) -> float:
    """Calculate FID from float64 statistics across supported SciPy releases."""
    from scipy import linalg

    mean_distance = np.square(mu - reference_mu).sum()
    covariance_root = linalg.sqrtm(np.dot(sigma, reference_sigma))
    fid = mean_distance + np.trace(sigma + reference_sigma - covariance_root * 2)
    return float(np.real(fid))


def calculate_fid(
    *,
    detector: Any,
    images_root: Path,
    reference_path: Path,
    device: torch.device,
    batch_size: int,
    seed_start: int = 0,
) -> tuple[float, float]:
    """Apply NVIDIA EDM's Inception features and float64 FID statistics."""
    feature_sum = torch.zeros(FEATURE_DIMENSION, dtype=torch.float64)
    feature_outer_sum = torch.zeros(
        [FEATURE_DIMENSION, FEATURE_DIMENSION], dtype=torch.float64
    )
    synchronize_device(device)
    start = time.perf_counter()
    with torch.inference_mode():
        seed_end = seed_start + EXPECTED_IMAGES
        for batch_start in range(seed_start, seed_end, batch_size):
            paths = [
                _seed_image_path(images_root, seed)
                for seed in range(batch_start, min(batch_start + batch_size, seed_end))
            ]
            images = _image_batch(paths).to(device)
            features = detector(images, return_features=True).cpu().to(torch.float64)
            feature_sum += features.sum(0)
            feature_outer_sum += features.T @ features
    synchronize_device(device)
    evaluation_seconds = time.perf_counter() - start
    mu = feature_sum / EXPECTED_IMAGES
    sigma = feature_outer_sum - mu.ger(mu) * EXPECTED_IMAGES
    sigma /= EXPECTED_IMAGES - 1
    reference = dict(np.load(reference_path))
    mu_numpy = mu.numpy()
    sigma_numpy = sigma.numpy()
    fid = fid_from_statistics(
        mu_numpy,
        sigma_numpy,
        reference["mu"],
        reference["sigma"],
    )
    return fid, evaluation_seconds


def quality_score_from_trials(fids: list[float]) -> float:
    """Return NVIDIA EDM's reported minimum across exactly three trials."""
    if len(fids) != len(QUALITY_TRIAL_SEED_STARTS):
        raise ValueError("EDM quality requires exactly three FID trials")
    if not all(np.isfinite(fid) and fid > 0 for fid in fids):
        raise ValueError("EDM FID trials must be finite positive values")
    return min(fids)


def run_image_generation_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Generate the official three-trial EDM packet and calculate CIFAR-10 FID."""
    contract = workload.raw.get("canonical_max_contract") or {}
    config = contract.get("config") or {}
    expected_seed_ranges = [
        f"{start}-{start + EXPECTED_IMAGES - 1}" for start in QUALITY_TRIAL_SEED_STARTS
    ]
    if (
        int(config.get("quality_trials", 0)) != len(QUALITY_TRIAL_SEED_STARTS)
        or int(config.get("generated_images_per_trial", 0)) != EXPECTED_IMAGES
        or int(config.get("total_generated_images", 0))
        != EXPECTED_IMAGES * len(QUALITY_TRIAL_SEED_STARTS)
        or list(config.get("trial_seeds") or []) != expected_seed_ranges
        or int(config.get("sampler_steps", 0)) != NUM_STEPS
        or int(config.get("network_evaluations_per_image", 0))
        != NETWORK_EVALUATIONS_PER_IMAGE
    ):
        raise ValueError("registry EDM quality contract does not match the runner")

    root = find_project_root()
    torch.manual_seed(configured_seed())
    device = select_torch_device()
    asset = ensure_edm_cifar10(download=True)
    paths = edm_cifar10_paths()
    official_sampler = _import_edm_source(paths["source"])
    net = _load_pinned_pickle(
        paths["checkpoint"],
        device,
        expected_sha256=EDM_CIFAR10_CHECKPOINT_SHA256,
        key="ema",
    )
    n_params = sum(parameter.numel() for parameter in net.parameters())
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = (output_dir / "image-generation_max_evaluation.json").resolve()
    report_path = (output_dir / "image-generation_max_report.json").resolve()
    provenance_path = (output_dir / "image-generation_max.provd.json").resolve()
    batch_size = int(os.environ.get("MLPERF_EDU_EDM_BATCH_SIZE", "64"))
    if batch_size < 1 or batch_size > 256:
        raise ValueError("MLPERF_EDU_EDM_BATCH_SIZE must be between 1 and 256")

    imported_generation: dict[str, Any] | None = None
    import_packet = os.environ.get(EDM_IMPORT_PACKET_ENV)
    if import_packet:
        imported_generation = load_edm_generation_evidence(Path(import_packet))
    imported_trials = imported_generation["trials"] if imported_generation else []
    generation_source_files = (
        list(imported_generation["source_files"]) if imported_generation else []
    )

    detector = _load_pinned_pickle(
        paths["inception"],
        device,
        expected_sha256=EDM_INCEPTION_SHA256,
    )
    trial_records: list[dict[str, Any]] = []
    trial_artifacts: list[Path] = []
    for trial_index, seed_start in enumerate(QUALITY_TRIAL_SEED_STARTS, start=1):
        trial_root = (output_dir / f"edm-cifar10-trial-{trial_index}").resolve()
        trial_root.mkdir(parents=True, exist_ok=True)
        progress_path = trial_root / "progress.jsonl"
        image_manifest_path = trial_root / "images.jsonl"
        trial_results_path = trial_root / "evaluation.json"
        imported_trial = imported_trials[trial_index - 1] if imported_trials else None
        if imported_trial:
            images_root = imported_trial["images_root"]
            generation_seconds = float(imported_trial["generation_seconds"])
            generated_images = EXPECTED_IMAGES
        else:
            images_root = trial_root / "images"
            images_root.mkdir(parents=True, exist_ok=True)
            generation_seconds, generated_images = _generate_images(
                net=net,
                device=device,
                official_sampler=official_sampler,
                images_root=images_root,
                progress_path=progress_path,
                batch_size=batch_size,
                seed_start=seed_start,
            )
        if generated_images != EXPECTED_IMAGES:
            raise RuntimeError(
                f"EDM trial {trial_index} did not cover exactly 50,000 seeds"
            )
        image_set = build_image_manifest(
            images_root,
            image_manifest_path,
            seed_start=seed_start,
            expected_manifest=(
                imported_trial["source_manifest"] if imported_trial else None
            ),
        )
        trial_fid, evaluation_seconds = calculate_fid(
            detector=detector,
            images_root=images_root,
            reference_path=paths["fid_reference"],
            device=device,
            batch_size=batch_size,
            seed_start=seed_start,
        )
        if imported_trial and not np.isclose(
            trial_fid,
            float(imported_trial["prior_fid"]),
            rtol=0.0,
            atol=EDM_IMPORT_FID_TOLERANCE,
        ):
            raise RuntimeError(
                f"EDM trial {trial_index} rescore changed from "
                f"{imported_trial['prior_fid']} to {trial_fid}"
            )
        trial_result = {
            "trial": trial_index,
            "seed_start": seed_start,
            "seed_end": seed_start + EXPECTED_IMAGES - 1,
            "fid": trial_fid,
            "generation_seconds": generation_seconds,
            "evaluation_seconds": evaluation_seconds,
            "image_set": image_set,
            "reference_statistics_sha256": f"sha256:{EDM_CIFAR10_FID_REFERENCE_SHA256}",
            "generation_evidence": (
                {
                    "mode": "imported-hash-bound-generation",
                    "source_manifest": str(imported_trial["source_manifest_path"]),
                    "source_manifest_sha256": imported_trial[
                        "source_manifest_sha256"
                    ],
                    "source_evaluation": str(
                        imported_trial["source_evaluation_path"]
                    ),
                    "source_evaluation_sha256": imported_trial[
                        "source_evaluation_sha256"
                    ],
                    "prior_fid": imported_trial["prior_fid"],
                    "rescore_delta": trial_fid - imported_trial["prior_fid"],
                    "generator_adapter_sha256": imported_trial[
                        "generator_adapter_sha256"
                    ],
                    "evaluator_adapter_sha256": imported_trial[
                        "evaluator_adapter_sha256"
                    ],
                }
                if imported_trial
                else {"mode": "current-run-or-exact-resume"}
            ),
        }
        trial_results_path.write_text(
            json.dumps(trial_result, indent=2, sort_keys=True) + "\n"
        )
        trial_result["evaluation_sha256"] = f"sha256:{sha256_file(trial_results_path)}"
        trial_result["artifacts"] = {
            "images": str(images_root),
            "progress": str(progress_path) if not imported_trial else None,
            "image_manifest": str(image_manifest_path),
            "evaluation": str(trial_results_path),
        }
        trial_records.append(trial_result)
        if not imported_trial:
            trial_artifacts.append(progress_path)
        trial_artifacts.extend([image_manifest_path, trial_results_path])

    fids = [float(record["fid"]) for record in trial_records]
    fid = quality_score_from_trials(fids)
    generation_seconds = sum(
        float(record["generation_seconds"]) for record in trial_records
    )
    evaluation_seconds = sum(
        float(record["evaluation_seconds"]) for record in trial_records
    )
    results = {
        "fid": fid,
        "score_statistic": "minimum_of_three_trials",
        "trial_fids": fids,
        "trials": [
            {key: value for key, value in record.items() if key != "artifacts"}
            for record in trial_records
        ],
        "reference_statistics_sha256": f"sha256:{EDM_CIFAR10_FID_REFERENCE_SHA256}",
    }
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    target = float(workload.quality_value or TARGET_FID)
    tolerance = float(workload.quality_tolerance or 0.0)
    target_met = fid <= target + tolerance
    generation_report_evidence = (
        {
            "mode": imported_generation["mode"],
            "schema": imported_generation["schema"],
            "packet": imported_generation["packet"],
            "packet_sha256": imported_generation["packet_sha256"],
            "sampler": imported_generation["sampler"],
            "adapters": imported_generation["adapters"],
            "trials": [record["generation_evidence"] for record in trial_records],
        }
        if imported_generation
        else {
            "mode": "current-run-or-exact-resume",
            "sampler": "edm-algorithm-2",
            "trials": len(QUALITY_TRIAL_SEED_STARTS),
        }
    )
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "max",
        "status": "passed" if target_met else "quality_failed",
        "backend": (
            f"pytorch-mps-imported-generation/pytorch-{device.type}-evaluation"
            if imported_generation
            else f"pytorch-{device.type}"
        ),
        "data_mode": (
            "imported-generated-images-plus-official-reference-statistics"
            if imported_generation
            else "generated-images-plus-official-reference-statistics"
        ),
        "model": {
            "id": "NVIDIA-EDM-CIFAR10-conditional",
            "revision": EDM_COMMIT,
            "n_params": n_params,
        },
        "model_source": {
            "repository": "https://github.com/NVlabs/edm",
            "revision": EDM_COMMIT,
            "checkpoint": str(paths["checkpoint"]),
            "checkpoint_sha256": f"sha256:{EDM_CIFAR10_CHECKPOINT_SHA256}",
            "training_entrypoint": str(paths["source"] / "train.py"),
        },
        "dataset": {
            "name": asset.name,
            "source": asset.source,
            "root": str(asset.root),
            "sha256": asset.sha256,
            "n_bytes": asset.n_bytes,
            "quality_trials": len(QUALITY_TRIAL_SEED_STARTS),
            "generated_images_per_trial": EXPECTED_IMAGES,
            "total_generated_images": EXPECTED_IMAGES * len(QUALITY_TRIAL_SEED_STARTS),
        },
        "evaluator": {
            "repository": "https://github.com/NVlabs/edm",
            "revision": EDM_COMMIT,
            "source_files": {
                name: f"sha256:{digest}" for name, digest in EDM_SOURCE_FILES.items()
            },
            "inception_sha256": f"sha256:{EDM_INCEPTION_SHA256}",
            "reference_statistics_sha256": f"sha256:{EDM_CIFAR10_FID_REFERENCE_SHA256}",
            "results_sha256": f"sha256:{sha256_file(results_path)}",
        },
        "seed": configured_seed(),
        "generation_evidence": generation_report_evidence,
        "measurement_protocol": workload.raw.get("measurement_protocol", {}),
        "config": {
            "quality_trials": len(QUALITY_TRIAL_SEED_STARTS),
            "trial_seeds": expected_seed_ranges,
            "class_conditional": True,
            "sampler": "edm-algorithm-2",
            "sampler_steps": NUM_STEPS,
            "network_evaluations_per_image": NETWORK_EVALUATIONS_PER_IMAGE,
            "state_precision": "float64-cpu-state-float32-mps-network"
            if device.type == "mps"
            else "official-float64-state",
            "batch_size": batch_size,
            "generation_source": (
                "imported-hash-bound-generation"
                if imported_generation
                else "current-run-or-exact-resume"
            ),
        },
        "metrics": {
            "fid": fid,
            "fid_trials": fids,
            "score_statistic": "minimum_of_three_trials",
            "quality_trials": len(QUALITY_TRIAL_SEED_STARTS),
            "generated_images_per_trial": EXPECTED_IMAGES,
            "total_generated_images": EXPECTED_IMAGES * len(QUALITY_TRIAL_SEED_STARTS),
            "generation_seconds": generation_seconds,
            "evaluation_seconds": evaluation_seconds,
            "duration_seconds": generation_seconds + evaluation_seconds,
            "images_per_second": (EXPECTED_IMAGES * len(QUALITY_TRIAL_SEED_STARTS))
            / generation_seconds,
            "n_params": n_params,
        },
        "quality": {
            "metric": workload.quality_metric,
            "metric_key": "fid",
            "target": target,
            "tolerance": tolerance,
            "direction": "lower",
            "target_met": target_met,
            "quality_required": True,
            "override": False,
        },
        "functional_readiness": {
            "schema": "mlperf-edu-functional-readiness/0.1",
            "stage": "quality-conformance",
            "end_to_end_execution": True,
            "authoritative_quality_contract_executed": True,
            "current_invocation_generated_images": not bool(imported_generation),
            "repeatability_verified": False,
            "promotion_eligible": False,
            "next_stage": "stability" if target_met else "quality-target-review",
        },
        "artifacts": {
            "report": str(report_path),
            "provenance": str(provenance_path),
            "weights": str(paths["checkpoint"]),
            "trials": [record["artifacts"] for record in trial_records],
            "evaluation_results": str(results_path),
            "generation_packet": (
                imported_generation["packet"] if imported_generation else None
            ),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = build_provd(
        workload=workload.id,
        scenario=workload.scenario or "offline",
        division="open",
        hardware_fingerprint=detect_hardware(),
        report=report,
        report_path=report_path,
        weights_path=paths["checkpoint"],
        weights_n_params=n_params,
        weights_dtype="float32",
        dataset_name="edm-cifar10-generated-images-and-evaluator",
        dataset_files=[
            *asset.files,
            *generation_source_files,
            *trial_artifacts,
            results_path,
        ],
        rng_seed=configured_seed(),
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    provenance_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report
