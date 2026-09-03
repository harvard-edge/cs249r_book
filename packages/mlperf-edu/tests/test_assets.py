import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from platformdirs import user_cache_path

from mlperf import assets, registry


def test_source_checkout_preserves_existing_asset_layout(monkeypatch, tmp_path):
    source = tmp_path / "mlperf-edu"
    source.mkdir()
    (source / "workloads.yaml").write_text("workloads: []\n")
    monkeypatch.delenv("MLPERF_EDU_DATA_DIR", raising=False)
    monkeypatch.setattr(registry, "find_project_root", lambda: source)

    assert assets.asset_cache_root() == source / "data"
    assert assets.data_root() == source / "datasets" / "local_tensors"
    assert assets.cifar10_paths()["root"] == source / "data" / "cifar10"


def test_installed_artifact_uses_stable_user_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("MLPERF_EDU_DATA_DIR", raising=False)
    monkeypatch.setattr(registry, "find_project_root", lambda: tmp_path)
    expected = user_cache_path("mlperf-edu").resolve()

    assert assets.asset_cache_root() == expected
    assert assets.data_root() == expected / "tinyshakespeare"
    assert assets.cifar10_paths()["root"] == expected / "cifar10"
    assert assets.sst2_paths()["root"] == expected / "sst2"


def test_data_directory_override_remains_authoritative(monkeypatch, tmp_path):
    override = tmp_path / "course-data"
    monkeypatch.setenv("MLPERF_EDU_DATA_DIR", str(override))

    assert assets.asset_cache_root() == override.resolve()
    assert assets.data_root() == override.resolve()
    assert assets.cifar10_paths()["root"] == override.resolve() / "cifar10"
    assert assets.ettm1_paths()["root"] == override.resolve() / "ettm1"


def test_explicit_asset_root_takes_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("MLPERF_EDU_DATA_DIR", str(tmp_path / "ignored"))
    explicit = Path(tmp_path / "explicit")

    assert assets.cifar10_paths(explicit)["root"] == explicit.resolve()


def test_humaneval_plus_fetch_validates_complete_release(monkeypatch, tmp_path):
    records = [
        {"task_id": f"HumanEval/{index}", "prompt": "pass"} for index in range(164)
    ]
    payload = "".join(json.dumps(record) + "\n" for record in records).encode()
    source = tmp_path / "source.jsonl.gz"
    with gzip.open(source, "wb") as handle:
        handle.write(payload)

    monkeypatch.setattr(assets, "HUMANEVAL_PLUS_BYTES", len(payload))
    monkeypatch.setattr(
        assets, "HUMANEVAL_PLUS_SHA256", hashlib.sha256(payload).hexdigest()
    )
    monkeypatch.setattr(
        assets,
        "HUMANEVAL_PLUS_ARCHIVE_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        assets,
        "_download",
        lambda _url, destination: shutil.copyfile(source, destination),
    )

    result = assets.ensure_humaneval_plus(root=tmp_path / "cache")

    assert result.name == "humaneval-plus"
    assert result.sha256 == f"sha256:{hashlib.sha256(payload).hexdigest()}"
    assert len(result.files) == 2
    assert len(result.files[1].read_text().splitlines()) == 164


def test_evalplus_fetch_extracts_pinned_evaluator_source(monkeypatch, tmp_path):
    source_root = tmp_path / "source"
    (source_root / "evalplus").mkdir(parents=True)
    (source_root / "Dockerfile").write_text("FROM python:3.10-slim\n")
    (source_root / "evalplus" / "evaluate.py").write_text("print('evaluate')\n")
    archive = tmp_path / "evalplus.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(
            source_root,
            arcname=f"evalplus-{assets.EVALPLUS_COMMIT}",
        )
    monkeypatch.setattr(
        assets,
        "EVALPLUS_ARCHIVE_SHA256",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        assets,
        "_download",
        lambda _url, destination: shutil.copyfile(archive, destination),
    )

    result = assets.ensure_evalplus_evaluator(root=tmp_path / "cache")

    assert result.name == "evalplus-evaluator"
    assert (result.root / "Dockerfile").is_file()
    assert (result.root / "evalplus" / "evaluate.py").is_file()


def test_mlperf_tiny_image_fetch_pins_model_index_and_evaluator(monkeypatch, tmp_path):
    sources = {
        "trained_models/pretrainedResnet.tflite": b"model",
        "perf_samples_idxs.npy": b"indices",
        "eval_functions_eembc.py": b"def calculate_accuracy(): pass\n",
    }
    monkeypatch.setattr(
        assets,
        "MLPERF_TINY_IMAGE_FLOAT_MODEL_SHA256",
        hashlib.sha256(sources["trained_models/pretrainedResnet.tflite"]).hexdigest(),
    )
    monkeypatch.setattr(
        assets,
        "MLPERF_TINY_IMAGE_PERF_INDICES_SHA256",
        hashlib.sha256(sources["perf_samples_idxs.npy"]).hexdigest(),
    )
    monkeypatch.setattr(
        assets,
        "MLPERF_TINY_IMAGE_EVALUATOR_SHA256",
        hashlib.sha256(sources["eval_functions_eembc.py"]).hexdigest(),
    )

    def fake_download(url, destination):
        relative = url.removeprefix(assets.MLPERF_TINY_IMAGE_BASE_URL + "/")
        destination.write_bytes(sources[relative])

    monkeypatch.setattr(assets, "_download", fake_download)

    result = assets.ensure_mlperf_tiny_image(root=tmp_path / "cache")

    assert result.name == "mlperf-tiny-image-evaluation"
    assert [path.name for path in result.files] == [
        "pretrainedResnet.tflite",
        "perf_samples_idxs.npy",
        "eval_functions_eembc.py",
    ]
    assert all(path.is_file() for path in result.files)


def test_bfcl_fetch_validates_all_non_live_ast_examples(monkeypatch, tmp_path):
    source_root = tmp_path / "source"
    data_root = source_root / "bfcl_eval" / "data"
    counts = {
        "BFCL_v4_multiple.json": 200,
        "BFCL_v4_parallel.json": 200,
        "BFCL_v4_parallel_multiple.json": 200,
        "BFCL_v4_simple_java.json": 100,
        "BFCL_v4_simple_javascript.json": 50,
        "BFCL_v4_simple_python.json": 400,
    }
    expected_hashes = {}
    for relative, count in counts.items():
        for path in (data_root / relative, data_root / "possible_answer" / relative):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "".join(json.dumps({"id": index}) + "\n" for index in range(count))
            )
            expected_hashes[str(path.relative_to(data_root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

    archive = tmp_path / "bfcl.tar.gz"
    prefix = f"gorilla-{assets.BFCL_COMMIT}/berkeley-function-call-leaderboard"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(source_root, arcname=prefix)
    monkeypatch.setattr(assets, "BFCL_DATA_FILES", expected_hashes)
    monkeypatch.setattr(assets, "BFCL_EVALUATOR_FILES", {})
    monkeypatch.setattr(
        assets, "BFCL_ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        assets,
        "_download",
        lambda _url, destination: shutil.copyfile(archive, destination),
    )

    result = assets.ensure_bfcl_non_live_ast(root=tmp_path / "cache")

    assert result.name == "bfcl-v4-non-live-ast"
    assert len(result.files) == 12
    assert all(path.is_file() for path in result.files)


def test_edm_fetch_validates_checkpoint_and_fid_reference(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint.pkl"
    reference = tmp_path / "reference.npz"
    inception = tmp_path / "inception.pkl"
    archive = tmp_path / "edm.tar.gz"
    checkpoint.write_bytes(b"checkpoint")
    reference.write_bytes(b"fid-reference")
    inception.write_bytes(b"inception")
    archive.write_bytes(b"source-archive")
    sources = {
        assets.EDM_CIFAR10_CHECKPOINT_URL: checkpoint,
        assets.EDM_CIFAR10_FID_REFERENCE_URL: reference,
        assets.EDM_INCEPTION_URL: inception,
        assets.EDM_ARCHIVE_URL: archive,
    }
    monkeypatch.setattr(
        assets,
        "EDM_CIFAR10_CHECKPOINT_SHA256",
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        assets,
        "EDM_CIFAR10_FID_REFERENCE_SHA256",
        hashlib.sha256(reference.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        assets, "EDM_CIFAR10_CHECKPOINT_BYTES", checkpoint.stat().st_size
    )
    monkeypatch.setattr(
        assets,
        "EDM_INCEPTION_SHA256",
        hashlib.sha256(inception.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(assets, "EDM_INCEPTION_BYTES", inception.stat().st_size)
    monkeypatch.setattr(
        assets,
        "EDM_ARCHIVE_SHA256",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(assets, "EDM_SOURCE_FILES", {})
    monkeypatch.setattr(
        assets,
        "_download",
        lambda url, destination: shutil.copyfile(sources[url], destination),
    )

    result = assets.ensure_edm_cifar10(root=tmp_path / "cache")

    assert result.name == "edm-cifar10-quality-assets"
    assert result.files == (
        tmp_path / "cache" / "edm-cifar10-32x32-cond-vp.pkl",
        tmp_path / "cache" / "cifar10-32x32.npz",
        tmp_path / "cache" / "inception-2015-12-05.pkl",
        tmp_path / "cache" / f"edm-{assets.EDM_COMMIT}.tar.gz",
    )


def test_dlrm_fetch_pins_inference_and_implementation_sources(monkeypatch, tmp_path):
    inference_tree = tmp_path / "inference-tree"
    implementation_tree = tmp_path / "implementation-tree"
    (inference_tree / "recommendation").mkdir(parents=True)
    implementation_tree.mkdir()
    inference_file = inference_tree / "recommendation" / "main.py"
    implementation_file = implementation_tree / "dlrm_s_pytorch.py"
    inference_file.write_text("print('inference')\n")
    implementation_file.write_text("print('dlrm')\n")

    inference_archive = tmp_path / "inference.tar.gz"
    implementation_archive = tmp_path / "implementation.tar.gz"
    with tarfile.open(inference_archive, "w:gz") as bundle:
        bundle.add(
            inference_tree,
            arcname=f"inference-{assets.DLRM_INFERENCE_COMMIT}",
        )
    with tarfile.open(implementation_archive, "w:gz") as bundle:
        bundle.add(
            implementation_tree,
            arcname=f"dlrm-{assets.DLRM_IMPLEMENTATION_COMMIT}",
        )

    sources = {
        assets.DLRM_INFERENCE_ARCHIVE_URL: inference_archive,
        assets.DLRM_IMPLEMENTATION_ARCHIVE_URL: implementation_archive,
    }
    monkeypatch.setattr(
        assets,
        "DLRM_INFERENCE_ARCHIVE_SHA256",
        hashlib.sha256(inference_archive.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        assets,
        "DLRM_IMPLEMENTATION_ARCHIVE_SHA256",
        hashlib.sha256(implementation_archive.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        assets,
        "DLRM_INFERENCE_FILES",
        {
            "recommendation/main.py": hashlib.sha256(
                inference_file.read_bytes()
            ).hexdigest()
        },
    )
    monkeypatch.setattr(
        assets,
        "DLRM_IMPLEMENTATION_FILES",
        {
            "dlrm_s_pytorch.py": hashlib.sha256(
                implementation_file.read_bytes()
            ).hexdigest()
        },
    )
    monkeypatch.setattr(
        assets,
        "_download",
        lambda url, destination: shutil.copyfile(sources[url], destination),
    )

    result = assets.ensure_dlrm_reference(root=tmp_path / "cache")

    assert result.name == "mlperf-inference-v1.0.1-dlrm-reference"
    assert len(result.files) == 4
    assert all(path.is_file() for path in result.files)


def test_minigo_fetch_pins_professional_games_and_quality_source(monkeypatch, tmp_path):
    source_root = tmp_path / "training-tree"
    games_root = (
        source_root / "reinforcement" / "tensorflow" / "minigo" / "benchmark_sgf"
    )
    games_root.mkdir(parents=True)
    source_files = {}
    for index in range(4):
        path = games_root / f"game-{index}.sgf"
        path.write_text(f"(;GM[1]SZ[9]C[game-{index}])\n")
        source_files[str(path.relative_to(source_root))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    archive = tmp_path / "training.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(source_root, arcname=f"training-{assets.MINIGO_COMMIT}")

    monkeypatch.setattr(
        assets,
        "MINIGO_ARCHIVE_SHA256",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(assets, "MINIGO_ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(assets, "MINIGO_SOURCE_FILES", source_files)
    monkeypatch.setattr(
        assets,
        "_download",
        lambda _url, destination: shutil.copyfile(archive, destination),
    )

    result = assets.ensure_minigo_reference(root=tmp_path / "cache")

    assert result.name == "mlperf-training-v0.5-minigo-reference"
    assert len([path for path in result.files if path.suffix == ".sgf"]) == 4
