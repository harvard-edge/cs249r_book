from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlperf.runners import code_generation


def test_official_qwen_prompt_and_stop_rules_match_published_recipe():
    prompt = "def add(a, b):\n    pass"

    rendered = code_generation.official_qwen_chatml_prompt(prompt)

    assert rendered.startswith("<|im_start|>system\n")
    assert "You are an intelligent programming assistant" in rendered
    assert f"```python\n{prompt}\n```" in rendered
    assert rendered.endswith("<|im_start|>assistant\n```python\n")
    assert code_generation.MAX_NEW_TOKENS == 2048
    assert code_generation.EVALPLUS_REFERENCE_FAILING_TASKS == ("HumanEval/32",)
    assert "\n#" in code_generation.STOP_STRINGS
    assert "\n```" in code_generation.STOP_STRINGS


@pytest.mark.parametrize(
    ("generation", "expected"),
    [
        ("def add(a, b):\n    return a + b\n```\ntext", "def add(a, b):\n    return a + b"),
        ("def add(a, b):\n\treturn a + b\n# note", "def add(a, b):\n    return a + b"),
        ("def add(a, b):\n    return a + b", "def add(a, b):\n    return a + b"),
    ],
)
def test_truncate_generation_uses_qwen_stop_boundaries(generation, expected):
    assert code_generation.truncate_generation(generation) == expected


def test_parse_evalplus_results_requires_one_result_for_all_164_tasks(tmp_path: Path):
    task_ids = {f"HumanEval/{index}" for index in range(164)}
    payload = {
        "hash": "dataset-md5",
        "eval": {
            task_id: [
                {
                    "base_status": "pass",
                    "plus_status": "pass" if index < 94 else "fail",
                }
            ]
            for index, task_id in enumerate(sorted(task_ids))
        },
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload))

    parsed = code_generation.parse_evalplus_results(
        path, expected_task_ids=task_ids
    )

    assert parsed["evaluation_tasks"] == 164
    assert parsed["passing_tasks"] == 94
    assert parsed["pass_at_1"] == pytest.approx(94 / 164)
    assert len(parsed["failing_task_ids"]) == 70


def test_evalplus_container_has_no_network_or_host_execution_surface(tmp_path: Path):
    workspace = tmp_path / "workspace"
    dataset = tmp_path / "HumanEvalPlus.jsonl.gz"
    command = code_generation.evalplus_docker_command(
        image="evalplus:test",
        workspace=workspace,
        dataset_archive=dataset,
        workers=3,
    )
    rendered = " ".join(command)

    assert "--network none" in rendered
    assert "--read-only" in command
    assert "--init" in command
    assert "--cap-drop ALL" in rendered
    assert "--security-opt no-new-privileges" in rendered
    assert "--pids-limit 512" in rendered
    assert "--ulimit core=0:0" in rendered
    assert "--ulimit nofile=1024:1024" in rendered
    assert "--user" in command
    assert "PYTHONDONTWRITEBYTECODE=1" in command
    assert "PYTHONNOUSERSITE=1" in command
    assert "HUMANEVAL_OVERRIDE_PATH=/input/HumanEvalPlus.jsonl.gz" in command
    assert "/workspace/samples.jsonl" in command
    assert "@sha256:" in code_generation.EVALPLUS_BASE_IMAGE
    assert "COPY . /evalplus" in code_generation.EVALPLUS_DOCKERFILE
    assert "git" not in code_generation.EVALPLUS_RUNTIME_PACKAGES
