from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mlperf.runners import function_calling


def test_qwen_fc_prompt_matches_pinned_bfcl_shape():
    messages = [{"role": "user", "content": "What is the weather in Zurich?"}]
    functions = [
        {
            "name": "weather.lookup",
            "description": "Look up weather.",
            "parameters": {
                "type": "dict",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]

    prompt = function_calling.official_qwen_fc_prompt(messages, functions)

    assert prompt.startswith("<|im_start|>system\n# Tools\n\n")
    assert '<tools>\n{"name": "weather.lookup"' in prompt
    assert '{"name": <function-name>, "arguments": <args-json-object>}' in prompt
    assert "What is the weather in Zurich?<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    assert function_calling.MAX_NEW_TOKENS == 4096


def test_qwen_tool_call_parser_matches_bfcl_json_contract():
    response = """<tool_call>
{"name": "weather.lookup", "arguments": {"city": "Zurich"}}
</tool_call>
<tool_call>
not-json
</tool_call>"""

    parsed = function_calling.extract_qwen_tool_calls(response)

    assert parsed == [{"weather.lookup": {"city": "Zurich"}}]
    assert (
        function_calling.clean_qwen_response(
            "<think>private reasoning</think>\n\n" + response
        )
        == response
    )


def test_official_non_live_ast_summary_is_not_raw_example_accuracy():
    local_category_accuracy = {
        "simple_python": 0.9125,
        "simple_java": 0.36,
        "simple_javascript": 0.32,
        "multiple": 0.925,
        "parallel": 0.845,
        "parallel_multiple": 0.84,
    }

    score = function_calling.official_non_live_ast_summary(local_category_accuracy)

    assert score == pytest.approx(0.7852083333333333)
    assert score != pytest.approx(
        (400 * 0.9125 + 100 * 0.36 + 50 * 0.32 + 200 * 0.925 + 200 * 0.845 + 200 * 0.84)
        / 1150
    )


def test_reference_self_check_pins_known_java_representation_exceptions():
    assert function_calling.BFCL_REFERENCE_FAILING_TASKS == (
        "simple_java_36",
        "simple_java_64",
        "simple_java_65",
    )
    assert function_calling.MODEL_PARAMETER_COUNT == 1_720_574_976
    assert len(function_calling.MODEL_FILES) == 9


def test_bfcl_progress_message_reports_durable_prefix_and_eta():
    samples = [
        {"latency_seconds": 6.0},
        {"latency_seconds": 10.0},
    ]

    message = function_calling.bfcl_progress_message(2, 1_150, samples)

    assert message == (
        "BFCL progress: 2/1150 | mean 8.0s/example | "
        "estimated remaining 2h 33m"
    )


def _write_import_packet(monkeypatch, tmp_path: Path) -> tuple[Path, list[dict]]:
    counts = {category: 1 for category in function_calling.CATEGORY_COUNTS}
    monkeypatch.setattr(function_calling, "CATEGORY_COUNTS", counts)
    monkeypatch.setattr(function_calling, "EXPECTED_EXAMPLES", len(counts))
    tasks = [
        {"id": f"{category}_0", "category": category, "fingerprint": f"fp-{category}"}
        for category in counts
    ]
    results = {}
    for task in tasks:
        path = tmp_path / f"{task['category']}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "id": task["id"],
                    "result": "<think>hidden</think>\n<tool_call>\n{}\n</tool_call>",
                    "input_token_count": 10,
                    "output_token_count": 5,
                    "latency": 1.25,
                    "audit_execution": "pytorch-bfloat16-mps-greedy",
                }
            )
            + "\n"
        )
        results[task["category"]] = {
            "path": path.name,
            "sha256": f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}",
            "examples": 1,
        }
    packet = tmp_path / "packet.json"
    packet.write_text(
        json.dumps(
            {
                "schema": function_calling.BFCL_IMPORT_PACKET_SCHEMA,
                "model": {
                    "id": function_calling.MODEL_ID,
                    "revision": function_calling.MODEL_REVISION,
                },
                "dataset_revision": function_calling.BFCL_COMMIT,
                "generation": {
                    "backend": "pytorch-bfloat16-mps-greedy",
                    "execution_dtype": "bfloat16",
                    "decoding": "greedy",
                },
                "results": results,
            }
        )
        + "\n"
    )
    return packet, tasks


def test_bfcl_import_binds_complete_raw_generation_evidence(
    monkeypatch, tmp_path: Path
):
    packet, tasks = _write_import_packet(monkeypatch, tmp_path)
    samples_path = tmp_path / "samples.jsonl"

    samples, evidence, source_files = (
        function_calling.import_bfcl_generation_evidence(
            packet_path=packet,
            tasks=tasks,
            samples_path=samples_path,
        )
    )

    assert len(samples) == len(tasks) == 6
    assert samples[0]["response"].startswith("<tool_call>")
    assert evidence["mode"] == "imported-hash-bound-generation"
    assert evidence["examples"] == 6
    assert len(source_files) == 7
    assert samples_path.is_file()


def test_bfcl_import_rejects_changed_source_file(monkeypatch, tmp_path: Path):
    packet, tasks = _write_import_packet(monkeypatch, tmp_path)
    source = tmp_path / f"{tasks[0]['category']}.jsonl"
    source.write_text(source.read_text() + "{}\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        function_calling.import_bfcl_generation_evidence(
            packet_path=packet,
            tasks=tasks,
            samples_path=tmp_path / "samples.jsonl",
        )
