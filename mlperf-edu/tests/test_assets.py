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
