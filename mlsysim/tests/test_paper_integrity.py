"""Corruption checks for paper publication artifacts using constructed test data."""

from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import statistics
import shutil

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "paper/scripts"
SPEC = importlib.util.spec_from_file_location("paper_integrity", SCRIPTS / "paper_integrity.py")
integrity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(integrity)


@pytest.fixture
def artifact_bundle(tmp_path):
    """Construct evidence only to test the verifier, never as a benchmark result."""
    for directory in ("mlsysim", "tests", "paper/scripts", "paper/data", "paper/generated", "paper/figures"):
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("# constructed fixture\n")
    (tmp_path / "mlsysim/model.py").write_text("# constructed fixture\n")
    (tmp_path / "paper/scripts/paper_integrity.py").write_text("# constructed protocol fixture\n")
    records = {}

    def add(name, value):
        records[name] = dict(raw=value, printed=str(value), tex=str(value), section="constructed fixture",
                             source="constructed test input, not measured", volatile=True)

    for name, value in (("MachineCpu", "test CPU"), ("MachineCores", 1),
                        ("MachineOs", "test OS"), ("MachinePython", "3.12")):
        add(name, value)
    samples = []
    for name, (runs, warmup, macro, scale) in integrity.TIMING_PROTOCOL.items():
        seconds = [float(i + 1) for i in range(runs)]
        med = statistics.median(seconds)
        samples.append(dict(function=name, warmup_runs=warmup, seconds=seconds, median_seconds=med))
        add(macro, med * scale)
    add("SweepTimingConfigs", 1000)
    add("SweepTimingPerConfigUs", records["SweepTimingMedianMs"]["raw"] * 1e3 / records["SweepTimingConfigs"]["raw"])
    for path in integrity.output_paths(tmp_path):
        path.write_text("constructed fixture\n")
    generated = tmp_path / "paper/generated"
    (generated / "values.tex").write_text(integrity.render_values_tex(records))
    (generated / "values.json").write_text(json.dumps({"values": records}))
    source = integrity.file_hashes(tmp_path, integrity.source_paths(tmp_path))
    receipt = {
        "schema_version": 2, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "path_base": "mlsysim project root", "command_kind": "normalized equivalent invocation",
        "working_directory": ".", "project_directory": ".",
        "hash_path_base": "project_directory", "command": ["python", "paper/scripts/generate_paper_values.py"],
        "replay": {"cwd": "mlsysim project directory", "argv": ["python", "paper/scripts/generate_paper_values.py"],
                   "environment": {"PYTHONPATH": "."}},
        "code_sha": "a" * 40, "dirty": True, "python": "3.12", "seed": None,
        "seed_status": "constructed test input", "dependencies": {n: "1" for n in integrity.REQUIRED_DEPENDENCIES},
        "source_sha256": source, "output_sha256": integrity.file_hashes(tmp_path, integrity.output_paths(tmp_path)),
        "machine": {n: records[n]["raw"] for n in ("MachineCpu", "MachineCores", "MachineOs", "MachinePython")},
        "protocol": {name: {"runs": spec[0], "warmup_runs": spec[1]} for name, spec in integrity.TIMING_PROTOCOL.items()},
        "preregistration": {"status": "not independently preregistered", "reference": None,
                            "protocol_source": "paper/scripts/paper_integrity.py",
                            "protocol_source_sha256": source["paper/scripts/paper_integrity.py"],
                            "interpretation": "constructed verifier test"},
        "samples": samples,
    }
    (generated / "timing_samples.json").write_text(json.dumps(receipt))
    return tmp_path, records, receipt


def test_consistent_constructed_bundle_passes(artifact_bundle):
    root, records, _ = artifact_bundle
    assert integrity.check_timing_receipt(root, records) == []


@pytest.mark.parametrize("mutation", ["missing_samples", "altered_duration", "fake_median", "negative_duration",
                                     "nan_duration", "duplicate_scenario", "wrong_warmup", "missing_dependency",
                                     "missing_cwd", "missing_output_hashes", "missing_source_hash",
                                     "missing_preregistration", "wrong_machine", "missing_seed_status"])
def test_timing_receipt_corruption_is_rejected(artifact_bundle, mutation):
    root, records, receipt = artifact_bundle
    if mutation == "missing_samples":
        receipt["samples"] = []
    elif mutation == "altered_duration":
        receipt["samples"][0]["seconds"] = [s * 2 for s in receipt["samples"][0]["seconds"]]
    elif mutation == "fake_median":
        receipt["samples"][0]["median_seconds"] *= 2
    elif mutation == "negative_duration":
        receipt["samples"][0]["seconds"][0] = -1
    elif mutation == "nan_duration":
        receipt["samples"][0]["seconds"][0] = float("nan")
    elif mutation == "duplicate_scenario":
        receipt["samples"][-1] = deepcopy(receipt["samples"][0])
    elif mutation == "wrong_warmup":
        receipt["samples"][0]["warmup_runs"] = 0
    elif mutation == "missing_dependency":
        receipt["dependencies"] = {}
    elif mutation == "missing_cwd":
        del receipt["working_directory"]
    elif mutation == "missing_output_hashes":
        receipt["output_sha256"] = {}
    elif mutation == "missing_source_hash":
        receipt["source_sha256"].pop("mlsysim/model.py")
    elif mutation == "missing_preregistration":
        del receipt["preregistration"]
    elif mutation == "wrong_machine":
        receipt["machine"]["MachineCpu"] = "different CPU"
    elif mutation == "missing_seed_status":
        del receipt["seed_status"]
    (root / "paper/generated/timing_samples.json").write_text(json.dumps(receipt))
    assert integrity.check_timing_receipt(root, records), mutation


@pytest.mark.parametrize("changed_path", ["mlsysim/model.py", "paper/generated/values.tex",
                                          "paper/generated/values.json", "paper/generated/hw_catalog.tex"])
def test_stale_source_or_output_is_rejected(artifact_bundle, changed_path):
    root, records, _ = artifact_bundle
    (root / changed_path).write_text("altered after measurement\n")
    assert any("stale" in p for p in integrity.check_timing_receipt(root, records))


def test_new_source_cannot_be_omitted_from_receipt(artifact_bundle):
    root, records, _ = artifact_bundle
    (root / "mlsysim/new_model.py").write_text("# new input\n")
    assert any("inventory" in p for p in integrity.check_timing_receipt(root, records))


def test_freshness_gate_rejects_display_only_tampering(tmp_path, monkeypatch):
    """Exercise the public generator gate, not only the hash helper."""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import generate_paper_values as generator

    v = generator.Values()
    v.num("Example", 2, "constructed verifier test")
    monkeypatch.setattr(generator, "GENERATED_DIR", tmp_path)
    monkeypatch.setattr(generator, "sync_diagrams", lambda v, write: [])
    monkeypatch.setattr(generator, "check_timing_receipt", lambda *args: [])
    monkeypatch.setattr(generator, "check_volatile_display", lambda *args: [])
    generator.write_outputs(v, {})
    assert generator.check_drift(v, {}) == []
    (tmp_path / "values.tex").write_text(r"\newcommand{\Example}{999999}")
    assert any("values.tex" in p for p in generator.check_drift(v, {}))
    # Altering JSON's displayed field and regenerating TeX must still fail even
    # when the raw arithmetic value is unchanged.
    payload = json.loads((tmp_path / "values.json").read_text())
    payload["values"]["Example"]["tex"] = "999999"
    (tmp_path / "values.json").write_text(json.dumps(payload))
    (tmp_path / "values.tex").write_text(integrity.render_values_tex(payload["values"]))
    assert any("tex differs" in p for p in generator.check_drift(v, {}))


def test_freshness_gate_calls_receipt_verifier(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import generate_paper_values as generator

    v = generator.Values()
    v.num("Example", 2, "constructed verifier test")
    monkeypatch.setattr(generator, "GENERATED_DIR", tmp_path)
    monkeypatch.setattr(generator, "sync_diagrams", lambda v, write: [])
    generator.write_outputs(v, {})
    assert any("timing receipt missing" in p for p in generator.check_drift(v, {}))


def test_source_edit_during_measurement_is_rejected(artifact_bundle):
    root, records, receipt = artifact_bundle
    before = integrity.file_hashes(root, integrity.source_paths(root))
    (root / "mlsysim/model.py").write_text("# edited while measurement ran\n")
    with pytest.raises(ValueError, match="changed during generation"):
        integrity.write_timing_receipt(root, records, receipt["samples"], before)


@pytest.mark.parametrize("field", ["source_sha256", "samples", "dependencies", "preregistration", "machine"])
def test_malformed_receipt_returns_failure(artifact_bundle, field):
    root, records, receipt = artifact_bundle
    receipt[field] = ["malformed"]
    (root / "paper/generated/timing_samples.json").write_text(json.dumps(receipt))
    assert integrity.check_timing_receipt(root, records)


def test_source_archive_needs_neither_git_checkout_nor_built_pdfs(artifact_bundle):
    root, records, _ = artifact_bundle
    assert not (root / ".git").exists()
    assert not list((root / "paper").rglob("*.pdf"))
    assert integrity.check_timing_receipt(root, records) == []


def test_recorded_runtime_display_cannot_be_changed_independently(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import generate_paper_values as generator

    records = json.loads((SCRIPTS.parent / "generated/values.json").read_text())["values"]
    records["SweepTimingMedianMs"]["tex"] = "999999"
    assert any("SweepTimingMedianMs: displayed value differs" in p
               for p in generator.check_volatile_display(records))


def test_written_receipt_uses_portable_paths(artifact_bundle, monkeypatch):
    root, records, receipt = artifact_bundle
    monkeypatch.setattr(integrity.subprocess, "check_output", lambda argv, **kwargs:
                        "a" * 40 if argv[1] == "rev-parse" else " M changed.py\n")
    monkeypatch.chdir(root.parent)
    integrity.write_timing_receipt(root, records, receipt["samples"], receipt["source_sha256"])
    text = (root / "paper/generated/timing_samples.json").read_text()
    written = json.loads(text)
    assert str(root) not in text
    assert written["working_directory"] == written["project_directory"] == "."
    assert written["command"] == ["python", "paper/scripts/generate_paper_values.py"]
    assert written["command_kind"] == "normalized equivalent invocation"
    assert integrity.check_timing_receipt(root, records) == []


def test_receipt_remains_valid_after_checkout_relocation(artifact_bundle, tmp_path_factory):
    root, records, _ = artifact_bundle
    destination = tmp_path_factory.mktemp("relocated") / "project"
    shutil.copytree(root, destination)
    assert integrity.check_timing_receipt(destination, records) == []


@pytest.mark.parametrize("field,value", [
    ("working_directory", "/tmp/recording-checkout"),
    ("project_directory", "/tmp/recording-checkout"),
    ("working_directory", ".."),
    ("path_base", "unspecified"),
    ("command_kind", "actual host invocation"),
    ("command", ["/tmp/venv/bin/python", "paper/scripts/generate_paper_values.py"]),
    ("schema_version", 1),
])
def test_receipt_rejects_nonportable_or_ambiguous_invocation(artifact_bundle, field, value):
    root, records, receipt = artifact_bundle
    receipt[field] = value
    (root / "paper/generated/timing_samples.json").write_text(json.dumps(receipt))
    assert integrity.check_timing_receipt(root, records)
