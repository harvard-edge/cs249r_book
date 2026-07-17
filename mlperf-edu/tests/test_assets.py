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
        {"task_id": f"HumanEval/{index}", "prompt": "pass"}
        for index in range(164)
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
            path.write_text("".join(json.dumps({"id": index}) + "\n" for index in range(count)))
            expected_hashes[str(path.relative_to(data_root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

    archive = tmp_path / "bfcl.tar.gz"
    prefix = f"gorilla-{assets.BFCL_COMMIT}/berkeley-function-call-leaderboard"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(source_root, arcname=prefix)
    monkeypatch.setattr(assets, "BFCL_DATA_FILES", expected_hashes)
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
    checkpoint.write_bytes(b"checkpoint")
    reference.write_bytes(b"fid-reference")
    sources = {
        assets.EDM_CIFAR10_CHECKPOINT_URL: checkpoint,
        assets.EDM_CIFAR10_FID_REFERENCE_URL: reference,
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
    monkeypatch.setattr(assets, "EDM_CIFAR10_CHECKPOINT_BYTES", checkpoint.stat().st_size)
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
    )
