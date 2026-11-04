"""Regression contracts for adaptation state and fixed-subset delivery."""

import json

import pytest

from mlsysim.labs.experiments import v1_08_experiments as training
from mlsysim.labs.experiments import v1_09_experiments as selection


@pytest.mark.parametrize("track", training.TRACKS)
@pytest.mark.parametrize("checkpointing", ("none", "full"))
def test_freezing_preserves_weights_and_backward_activations(track, checkpointing):
    full = training.adaptation_outcome(track, checkpointing=checkpointing)
    restricted = training.adaptation_outcome(track, adaptation_scope="restricted", checkpointing=checkpointing)
    assert restricted["weights_gb"] == full["weights_gb"]
    assert restricted["activations_gb"] == full["activations_gb"] > 0
    assert restricted["gradients_gb"] == pytest.approx(full["gradients_gb"] * restricted["trainable_fraction"], abs=1e-6)
    assert restricted["optimizer_state_gb"] == pytest.approx(full["optimizer_state_gb"] * restricted["trainable_fraction"], abs=1e-6)
    assert restricted["total_memory_gb"] < full["total_memory_gb"]
    assert restricted["total_memory_gb"] == pytest.approx(sum(restricted[f"{field}_gb"] for field in (
        "weights", "gradients", "optimizer_state", "activations", "communication_buffers")), abs=3e-6)
    assert restricted["quality_evidence"] is None
    assert training.replay(restricted["model_key"], json.loads(json.dumps(restricted["inputs"]))) == restricted


@pytest.mark.parametrize("track", selection.TRACK_IDS)
def test_input_paths_preserve_selection_and_compute_but_charge_different_work(track):
    stream, scatter, material = [selection.input_path_outcome(track, input_path=path) for path in selection.INPUT_PATHS]
    assert stream["selected_ids"] == scatter["selected_ids"] == material["selected_ids"]
    assert stream["compute_time_per_epoch"] == scatter["compute_time_per_epoch"] == material["compute_time_per_epoch"]
    assert stream["read_bytes_per_epoch"] > scatter["read_bytes_per_epoch"] == material["read_bytes_per_epoch"]
    assert stream["transform_time_per_epoch"] > scatter["transform_time_per_epoch"] > material["transform_time_per_epoch"]
    assert scatter["access_time_per_epoch"].magnitude > 0
    assert stream["preparation_time"].magnitude == scatter["preparation_time"].magnitude == 0
    assert material["preparation_time"].magnitude > 0
    assert material["peak_storage"] == material["source_storage"] + material["additional_storage"]
    assert material["preparation_time"] == material["preparation_read_time"] + material["preparation_transform_time"] + material["preparation_write_time"]
    assert material["preparation_write_bytes"] == material["read_bytes_per_epoch"]
    for result in (stream, scatter, material):
        assert result["epoch_time"] == max(result["input_time_per_epoch"], result["compute_time_per_epoch"])
        restored = json.loads(json.dumps(selection.serialize_value(result["inputs"])))
        assert selection.replay(result["model_key"], restored) == result


@pytest.mark.parametrize("track", selection.TRACK_IDS)
def test_materialization_pays_only_after_prep_is_recovered(track):
    first = selection.input_path_outcome(track, input_path="materialized")
    boundary = first["first_profitable_materialized_epochs"]
    assert boundary is not None and boundary > 1
    before = selection.input_path_outcome(track, input_path="materialized", epochs=boundary - 1)
    before_stream = selection.input_path_outcome(track, epochs=boundary - 1)
    after = selection.input_path_outcome(track, input_path="materialized", epochs=boundary)
    after_stream = selection.input_path_outcome(track, epochs=boundary)
    assert before["total_time"] >= before_stream["total_time"]
    assert after["total_time"] < after_stream["total_time"]
    assert after["amortized_time_per_epoch"] < first["amortized_time_per_epoch"]
    assert after["additional_storage"] == first["additional_storage"]


def test_new_instruments_reject_unsupported_recipes():
    with pytest.raises(ValueError, match="adaptation_scope"):
        training.adaptation_outcome("tinyml", adaptation_scope="erase_activations")
    with pytest.raises(ValueError, match="input_path"):
        selection.input_path_outcome("tinyml", input_path="magic")
    for epochs in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="epochs"):
            selection.input_path_outcome("tinyml", epochs=epochs)
