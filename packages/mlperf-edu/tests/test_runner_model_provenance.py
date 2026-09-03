from pathlib import Path

import pytest

from mlperf.runners import code_generation, retrieval, text


@pytest.mark.parametrize(
    ("builder", "expected_files"),
    (
        (code_generation._model_file_records, set(code_generation.MODEL_FILES)),
        (retrieval._model_file_records, set(retrieval.MODEL_FILES)),
        (text._model_file_records, set(text.DISTILBERT_HASHES)),
    ),
)
def test_model_provenance_binds_weights_tokenizer_and_config(
    tmp_path: Path, builder, expected_files
):
    records = builder(tmp_path)

    assert {record["logical_path"] for record in records} == expected_files
    assert {Path(record["path"]).name for record in records} == expected_files
    assert any(record["role"] == "weights" for record in records)
    assert any(record["role"] == "model-config" for record in records)
