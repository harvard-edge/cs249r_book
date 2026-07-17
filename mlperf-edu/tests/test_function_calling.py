from __future__ import annotations

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
