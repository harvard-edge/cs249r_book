from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch

from mlperf.assets import (
    BFCL_ARCHIVE_SHA256,
    BFCL_COMMIT,
    BFCL_EVALUATOR_COMMIT,
    BFCL_EVALUATOR_FILES,
    bfcl_non_live_ast_paths,
    ensure_bfcl_non_live_ast,
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


MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
# The pinned checkpoint ties its input embedding and LM head. Count unique
# parameter tensors, matching ``model.parameters()``, rather than double-counting
# the shared 151,936 x 2,048 embedding matrix.
MODEL_PARAMETER_COUNT = 1_720_574_976
MODEL_FILES = {
    "config.json": "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197",
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model-00001-of-00002.safetensors": "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5",
    "model-00002-of-00002.safetensors": "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9",
    "model.safetensors.index.json": "0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
MODEL_WEIGHT_FILES = {
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
}
MODEL_REGISTRY_NAME = f"{MODEL_ID}-FC"
MAX_NEW_TOKENS = 4096
PROGRESS_INTERVAL = 10
TARGET_ACCURACY = 0.8292
EXPECTED_EXAMPLES = 1150
BFCL_IMPORT_PACKET_ENV = "MLPERF_EDU_BFCL_IMPORT_PACKET"
BFCL_IMPORT_PACKET_SCHEMA = "mlperf-edu-bfcl-generation-evidence/0.1"
BFCL_REFERENCE_FAILING_TASKS = (
    "simple_java_36",
    "simple_java_64",
    "simple_java_65",
)
CATEGORY_COUNTS = {
    "simple_python": 400,
    "simple_java": 100,
    "simple_javascript": 50,
    "multiple": 200,
    "parallel": 200,
    "parallel_multiple": 200,
}
CATEGORY_FILES = {category: f"BFCL_v4_{category}.json" for category in CATEGORY_COUNTS}
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\n(.*?)\n</tool_call>", re.DOTALL)


def official_qwen_fc_prompt(
    messages: list[dict[str, Any]], functions: list[dict[str, Any]]
) -> str:
    """Render the exact single-turn prompt used by BFCL's pinned Qwen FC handler."""
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("BFCL Non-Live AST requires exactly one user message")
    if not functions:
        raise ValueError("BFCL Non-Live AST entries must declare at least one function")
    formatted = "<|im_start|>system\n"
    formatted += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    formatted += (
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>"
    )
    for function in functions:
        formatted += f"\n{json.dumps(function)}"
    formatted += (
        "\n</tools>\n\nFor each function call, return a json object with function name "
        "and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call><|im_end|>\n"
    )
    formatted += f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"
    return formatted


def clean_qwen_response(text: str) -> str:
    """Match BFCL's Qwen response cleanup while retaining tool-call tags."""
    if "</think>" in text:
        return text.split("</think>")[-1].lstrip("\n")
    return text


def extract_qwen_tool_calls(text: str) -> list[dict[str, dict[str, Any]]]:
    """Match the pinned BFCL Qwen FC JSON parser and normalized AST shape."""
    calls: list[dict[str, dict[str, Any]]] = []
    for match in TOOL_CALL_PATTERN.findall(text):
        try:
            call = json.loads(match)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            continue
        calls.append({name: dict(arguments)})
    return calls


def task_fingerprint(task: dict[str, Any]) -> str:
    payload = json.dumps(
        task, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"JSONL record is not an object at {path}:{line_number}")
        records.append(record)
    return records


def load_bfcl_tasks(data_root: Path) -> list[dict[str, Any]]:
    """Load and strictly align all six official Non-Live AST categories."""
    tasks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for category, expected_count in CATEGORY_COUNTS.items():
        prompts = _read_jsonl(data_root / CATEGORY_FILES[category])
        answers = _read_jsonl(data_root / "possible_answer" / CATEGORY_FILES[category])
        if len(prompts) != expected_count or len(answers) != expected_count:
            raise ValueError(
                f"BFCL {category} expected {expected_count} prompt/answer pairs"
            )
        for prompt, answer in zip(prompts, answers, strict=True):
            task_id = prompt.get("id")
            if task_id != answer.get("id") or not isinstance(task_id, str):
                raise ValueError(f"BFCL {category} prompt/answer IDs do not align")
            if task_id in seen_ids:
                raise ValueError(f"duplicate BFCL task ID: {task_id}")
            question = prompt.get("question")
            if (
                not isinstance(question, list)
                or len(question) != 1
                or not isinstance(question[0], list)
            ):
                raise ValueError(f"BFCL task is not single-turn: {task_id}")
            task = {
                "id": task_id,
                "category": category,
                "messages": question[0],
                "functions": prompt.get("function"),
                "ground_truth": answer.get("ground_truth"),
            }
            if not isinstance(task["functions"], list) or not isinstance(
                task["ground_truth"], list
            ):
                raise ValueError(
                    f"BFCL task is missing functions or answers: {task_id}"
                )
            task["fingerprint"] = task_fingerprint(task)
            seen_ids.add(task_id)
            tasks.append(task)
    if len(tasks) != EXPECTED_EXAMPLES:
        raise ValueError(f"BFCL expected {EXPECTED_EXAMPLES} tasks, found {len(tasks)}")
    return tasks


def _snapshot_model(workload: Workload) -> Path:
    from huggingface_hub import snapshot_download

    source = workload.raw.get("model_source") or {}
    if source.get("repo_id") != MODEL_ID or source.get("revision") != MODEL_REVISION:
        raise ValueError(
            "registry function-calling model pin does not match the runner"
        )
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=list(MODEL_FILES),
            local_files_only=os.environ.get("MLPERF_EDU_HF_LOCAL_ONLY", "0") == "1",
        )
    ).resolve()
    for filename, expected in MODEL_FILES.items():
        path = snapshot / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"pinned Qwen3 model file failed SHA-256: {filename}")
    return snapshot


def _official_ast_checker(
    source_root: Path,
) -> tuple[Callable[..., dict[str, Any]], type[Any]]:
    """Load the byte-pinned BFCL AST checker without BFCL's API dependencies."""
    source_text = str(source_root.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    stub_name = "bfcl_eval.constants.model_config"
    stub = types.ModuleType(stub_name)
    stub.MODEL_CONFIG_MAPPING = {
        MODEL_REGISTRY_NAME: SimpleNamespace(underscore_to_dot=False)
    }
    sys.modules[stub_name] = stub
    from bfcl_eval.constants.enums import Language
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

    return ast_checker, Language


def _reference_calls(task: dict[str, Any]) -> list[dict[str, dict[str, Any]]]:
    descriptions = {
        description["name"]: description for description in task["functions"]
    }
    calls: list[dict[str, dict[str, Any]]] = []
    for ground_truth_call in task["ground_truth"]:
        if not isinstance(ground_truth_call, dict) or len(ground_truth_call) != 1:
            raise ValueError(f"invalid BFCL ground truth for {task['id']}")
        function_name, parameters = next(iter(ground_truth_call.items()))
        description = descriptions.get(function_name)
        if description is None:
            raise ValueError(f"BFCL ground truth function is unavailable: {task['id']}")
        required = set(description["parameters"].get("required") or [])
        selected: dict[str, Any] = {}
        for name, candidates in parameters.items():
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(
                    f"BFCL ground truth candidates are invalid: {task['id']}"
                )
            if name not in required and "" in candidates:
                continue
            schema = description["parameters"]["properties"].get(name, {})
            selected[name] = _serialize_reference_value(
                _reference_value(
                    candidates,
                    schema,
                ),
                schema,
                task["category"],
            )
        calls.append({function_name: selected})
    return calls


def _reference_value(candidates: list[Any], schema: dict[str, Any]) -> Any:
    """Select one schema-valid value from BFCL's nested acceptable-value lists."""
    value = next(
        (candidate for candidate in candidates if candidate != ""), candidates[0]
    )
    value_type = schema.get("type")
    if value_type in {"dict", "object", "HashMap"} and isinstance(value, dict):
        properties = schema.get("properties") or {}
        normalized: dict[str, Any] = {}
        for key, nested_candidates in value.items():
            if isinstance(nested_candidates, list) and nested_candidates:
                normalized[key] = _reference_value(
                    nested_candidates, properties.get(key, {})
                )
            else:
                normalized[key] = nested_candidates
        return normalized
    if value_type in {"array", "tuple", "Array", "ArrayList"} and isinstance(
        value, list
    ):
        item_schema = schema.get("items") or {}
        if item_schema.get("type") in {"dict", "object", "HashMap"}:
            normalized_items: list[Any] = []
            for item in value:
                if isinstance(item, dict):
                    normalized_items.append(
                        _reference_value([item], {"type": "dict", **item_schema})
                    )
                else:
                    normalized_items.append(item)
            return normalized_items
    return value


def _java_literal(value: Any, value_type: str | None = None) -> str:
    if value_type == "long":
        return f"{value}L"
    if value_type == "float":
        return f"{value}f"
    if value_type in {"String", "char"}:
        return json.dumps(value)
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _serialize_reference_value(
    value: Any, schema: dict[str, Any], category: str
) -> Any:
    """Express a canonical value in the representation BFCL expects per language."""
    value_type = schema.get("type")
    if category == "simple_java":
        if value_type == "Array" and isinstance(value, list):
            item_type = (schema.get("items") or {}).get("type")
            serialized_type = None if item_type == "String" else item_type
            items = ", ".join(_java_literal(item, serialized_type) for item in value)
            return f"new {item_type or 'Object'}[]{{{items}}}"
        if value_type == "ArrayList" and isinstance(value, list):
            item_type = (schema.get("items") or {}).get("type")
            items = ", ".join(_java_literal(item, item_type) for item in value)
            return f"new ArrayList<>(Arrays.asList({items}))"
        if value_type == "HashMap" and isinstance(value, dict):
            puts = " ".join(
                f"put({json.dumps(str(key))}, {_java_literal(item)});"
                for key, item in value.items()
            )
            return f"new HashMap<String, Object>() {{{{ {puts} }}}}"
        if value_type == "boolean":
            return str(value).lower()
        if value_type == "long":
            return f"{value}L"
        if value_type == "float":
            return f"{value}f"
        return str(value)
    if category == "simple_javascript":
        if value_type == "Boolean":
            return str(value).lower()
        if value_type == "Bigint":
            return f"{value}n"
        if value_type in {"array", "dict"} and isinstance(value, (list, dict)):
            return json.dumps(value, separators=(",", ":"))
        return str(value)
    return value


def _language_for(category: str, language_enum: type[Any]) -> Any:
    if category == "simple_java":
        return language_enum.JAVA
    if category == "simple_javascript":
        return language_enum.JAVASCRIPT
    return language_enum.PYTHON


def official_non_live_ast_summary(category_accuracies: dict[str, float]) -> float:
    """Apply BFCL's equal-weight language-simple then four-way AST summary."""
    if set(category_accuracies) != set(CATEGORY_COUNTS):
        raise ValueError("BFCL Non-Live AST aggregation requires all six categories")
    simple_accuracy = (
        sum(
            category_accuracies[name]
            for name in ("simple_python", "simple_java", "simple_javascript")
        )
        / 3
    )
    return (
        sum(
            (
                simple_accuracy,
                category_accuracies["multiple"],
                category_accuracies["parallel"],
                category_accuracies["parallel_multiple"],
            )
        )
        / 4
    )


def evaluate_bfcl_samples(
    *,
    tasks: list[dict[str, Any]],
    samples: list[dict[str, Any]] | None,
    source_root: Path,
) -> dict[str, Any]:
    """Score exact task coverage with the official BFCL AST checker."""
    ast_checker, language_enum = _official_ast_checker(source_root)
    if samples is not None:
        if len(samples) != len(tasks):
            raise ValueError("BFCL evaluation requires all 1,150 generated samples")
        sample_by_id = {sample.get("id"): sample for sample in samples}
        if set(sample_by_id) != {task["id"] for task in tasks}:
            raise ValueError("BFCL generated sample IDs do not match the task set")
    else:
        sample_by_id = {}

    category_results: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for category, expected_count in CATEGORY_COUNTS.items():
        category_tasks = [task for task in tasks if task["category"] == category]
        correct = 0
        for task in category_tasks:
            if samples is None:
                decoded = _reference_calls(task)
                raw_response = None
            else:
                raw_response = sample_by_id[task["id"]].get("response")
                if not isinstance(raw_response, str):
                    raise ValueError(f"BFCL sample has no response: {task['id']}")
                decoded = extract_qwen_tool_calls(raw_response)
            try:
                result = ast_checker(
                    task["functions"],
                    decoded,
                    task["ground_truth"],
                    _language_for(category, language_enum),
                    category,
                    MODEL_REGISTRY_NAME,
                )
            except Exception as exc:
                result = {
                    "valid": False,
                    "error": [
                        f"official AST evaluator raised {type(exc).__name__}: {exc}"
                    ],
                    "error_type": "ast_evaluator:exception",
                }
            if result.get("valid") is True:
                correct += 1
            else:
                failures.append(
                    {
                        "id": task["id"],
                        "category": category,
                        "error": result.get("error"),
                        "error_type": result.get("error_type"),
                        "response": raw_response,
                        "decoded": decoded,
                    }
                )
        if len(category_tasks) != expected_count:
            raise ValueError(f"BFCL category coverage changed: {category}")
        category_results[category] = {
            "accuracy": correct / expected_count,
            "correct": correct,
            "total": expected_count,
        }

    category_accuracies = {
        name: float(result["accuracy"]) for name, result in category_results.items()
    }
    simple_accuracy = (
        sum(
            category_accuracies[name]
            for name in ("simple_python", "simple_java", "simple_javascript")
        )
        / 3
    )
    score = official_non_live_ast_summary(category_accuracies)
    return {
        "non_live_ast_accuracy": score,
        "simple_ast_accuracy": simple_accuracy,
        "categories": category_results,
        "correct": sum(item["correct"] for item in category_results.values()),
        "total": sum(item["total"] for item in category_results.values()),
        "failures": failures,
    }


def _load_resumable_samples(
    samples_path: Path, tasks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not samples_path.is_file():
        return []
    samples = _read_jsonl(samples_path)
    if len(samples) > len(tasks):
        raise ValueError("BFCL samples file contains more records than the task set")
    for index, sample in enumerate(samples):
        task = tasks[index]
        if (
            sample.get("id") != task["id"]
            or sample.get("task_fingerprint") != task["fingerprint"]
            or sample.get("model_revision") != MODEL_REVISION
        ):
            raise ValueError(
                "BFCL resumable samples are not an exact prefix of this run contract"
            )
    return samples


def _resolve_packet_file(packet_path: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"BFCL import packet {label} must be a nonempty path")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = packet_path.parent / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"BFCL import packet {label} is missing: {candidate}")
    return candidate


def import_bfcl_generation_evidence(
    *, packet_path: Path, tasks: list[dict[str, Any]], samples_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], list[Path]]:
    """Import a complete, hash-bound BFCL generation campaign for rescoring."""
    packet_path = packet_path.expanduser().resolve()
    if not packet_path.is_file():
        raise FileNotFoundError(f"BFCL import packet is missing: {packet_path}")
    try:
        packet = json.loads(packet_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid BFCL import packet JSON: {packet_path}") from exc
    if not isinstance(packet, dict) or packet.get("schema") != BFCL_IMPORT_PACKET_SCHEMA:
        raise ValueError(
            f"BFCL import packet schema must be {BFCL_IMPORT_PACKET_SCHEMA}"
        )
    if packet.get("model") != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise ValueError("BFCL import packet model pin does not match the runner")
    if packet.get("dataset_revision") != BFCL_COMMIT:
        raise ValueError("BFCL import packet dataset revision does not match the runner")
    generation = packet.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("BFCL import packet must describe the generation runtime")
    backend = generation.get("backend")
    execution_dtype = generation.get("execution_dtype")
    if not isinstance(backend, str) or not backend:
        raise ValueError("BFCL import packet generation backend must be nonempty")
    if execution_dtype not in {"float32", "bfloat16"}:
        raise ValueError("BFCL import packet execution dtype is unsupported")
    if generation.get("decoding") != "greedy":
        raise ValueError("BFCL import packet must bind greedy decoding")

    results = packet.get("results")
    if not isinstance(results, dict) or set(results) != set(CATEGORY_COUNTS):
        raise ValueError("BFCL import packet must bind all six result categories")
    task_by_category = {
        category: [task for task in tasks if task["category"] == category]
        for category in CATEGORY_COUNTS
    }
    imported_by_id: dict[str, dict[str, Any]] = {}
    source_files: list[Path] = [packet_path]
    file_evidence: dict[str, Any] = {}
    for category, expected_count in CATEGORY_COUNTS.items():
        entry = results[category]
        if not isinstance(entry, dict):
            raise ValueError(f"BFCL import entry is invalid for {category}")
        path = _resolve_packet_file(packet_path, entry.get("path"), label=category)
        expected_digest = entry.get("sha256")
        if expected_digest != f"sha256:{sha256_file(path)}":
            raise ValueError(f"BFCL import SHA-256 mismatch for {category}")
        records = _read_jsonl(path)
        category_tasks = task_by_category[category]
        if len(records) != expected_count or entry.get("examples") != expected_count:
            raise ValueError(f"BFCL import count mismatch for {category}")
        for task, record in zip(category_tasks, records, strict=True):
            if record.get("id") != task["id"]:
                raise ValueError(f"BFCL import task order changed for {category}")
            if record.get("audit_execution") != backend:
                raise ValueError(f"BFCL import backend changed for {task['id']}")
            response = record.get("result")
            input_tokens = record.get("input_token_count")
            output_tokens = record.get("output_token_count")
            latency = record.get("latency")
            if (
                not isinstance(response, str)
                or not isinstance(input_tokens, int)
                or input_tokens <= 0
                or not isinstance(output_tokens, int)
                or output_tokens <= 0
                or not isinstance(latency, (int, float))
                or not math.isfinite(float(latency))
                or float(latency) <= 0
            ):
                raise ValueError(f"BFCL import measurement is invalid for {task['id']}")
            imported_by_id[task["id"]] = {
                "id": task["id"],
                "category": category,
                "task_fingerprint": task["fingerprint"],
                "model_revision": MODEL_REVISION,
                "response": clean_qwen_response(response),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_seconds": float(latency),
            }
        source_files.append(path)
        file_evidence[category] = {
            "path": str(path),
            "sha256": expected_digest,
            "examples": expected_count,
        }

    samples = [imported_by_id[task["id"]] for task in tasks]
    if len(samples) != EXPECTED_EXAMPLES:
        raise ValueError("BFCL import did not cover all 1,150 tasks")
    if samples_path.is_file():
        if _read_jsonl(samples_path) != samples:
            raise ValueError(
                "BFCL import conflicts with the existing canonical samples file"
            )
    else:
        temporary = samples_path.with_suffix(samples_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        os.replace(temporary, samples_path)

    evidence = {
        "mode": "imported-hash-bound-generation",
        "schema": BFCL_IMPORT_PACKET_SCHEMA,
        "packet": str(packet_path),
        "packet_sha256": f"sha256:{sha256_file(packet_path)}",
        "backend": backend,
        "execution_dtype": execution_dtype,
        "examples": len(samples),
        "files": file_evidence,
    }
    return samples, evidence, source_files


def _generate_samples(
    *,
    model: Any,
    tokenizer: Any,
    tasks: list[dict[str, Any]],
    existing: list[dict[str, Any]],
    device: torch.device,
    samples_path: Path,
) -> list[dict[str, Any]]:
    mode = "a" if existing else "w"
    with samples_path.open(mode, encoding="utf-8") as handle, torch.inference_mode():
        for task in tasks[len(existing) :]:
            prompt = official_qwen_fc_prompt(task["messages"], task["functions"])
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            synchronize_device(device)
            start = time.perf_counter()
            output = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            synchronize_device(device)
            latency = time.perf_counter() - start
            output_ids = output[0, input_ids.shape[-1] :]
            response = clean_qwen_response(
                tokenizer.decode(output_ids, skip_special_tokens=True)
            )
            sample = {
                "id": task["id"],
                "category": task["category"],
                "task_fingerprint": task["fingerprint"],
                "model_revision": MODEL_REVISION,
                "response": response,
                "input_tokens": int(input_ids.shape[-1]),
                "output_tokens": int(output_ids.numel()),
                "latency_seconds": latency,
            }
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            handle.flush()
            existing.append(sample)
            completed = len(existing)
            if (
                completed == 1
                or completed % PROGRESS_INTERVAL == 0
                or completed == len(tasks)
            ):
                print(
                    bfcl_progress_message(completed, len(tasks), existing),
                    flush=True,
                )
    return existing


def bfcl_progress_message(
    completed: int, total: int, samples: list[dict[str, Any]]
) -> str:
    generation_seconds = sum(
        float(sample.get("latency_seconds") or 0.0) for sample in samples
    )
    mean_seconds = generation_seconds / completed if completed else 0.0
    remaining_seconds = mean_seconds * max(total - completed, 0)
    remaining_minutes = int(round(remaining_seconds / 60.0))
    hours, minutes = divmod(remaining_minutes, 60)
    eta = f"{hours}h {minutes:02d}m" if hours else f"{minutes}m"
    return (
        f"BFCL progress: {completed}/{total} | "
        f"mean {mean_seconds:.1f}s/example | estimated remaining {eta}"
    )


def _model_file_records(snapshot: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for filename in MODEL_FILES:
        records.append(
            {
                "path": snapshot / filename,
                "logical_path": filename,
                "role": "weights" if filename in MODEL_WEIGHT_FILES else "model-config",
            }
        )
    return records


def run_function_calling_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Generate and officially score all 1,150 BFCL V4 Non-Live AST cases."""
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    contract = workload.raw.get("canonical_max_contract") or {}
    config = contract.get("config") or {}
    if (
        int(config.get("evaluation_examples", 0)) != EXPECTED_EXAMPLES
        or contract.get("dataset_revision") != BFCL_COMMIT
        or contract.get("evaluator_revision") != BFCL_EVALUATOR_COMMIT
    ):
        raise ValueError("registry BFCL quality contract does not match the runner")

    root = find_project_root()
    seed = configured_seed()
    torch.manual_seed(seed)
    device = select_torch_device()
    dataset_asset = ensure_bfcl_non_live_ast(download=True)
    paths = bfcl_non_live_ast_paths()
    source_root = paths["source"]
    tasks = load_bfcl_tasks(paths["data"])
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = (output_dir / f"{workload.id}_max_samples.jsonl").resolve()
    results_path = (output_dir / f"{workload.id}_max_evaluation.json").resolve()
    report_path = (output_dir / f"{workload.id}_max_report.json").resolve()
    manifest_path = (output_dir / f"{workload.id}_max.provd.json").resolve()

    reference_self_check = evaluate_bfcl_samples(
        tasks=tasks, samples=None, source_root=source_root
    )
    reference_failing_ids = tuple(
        failure["id"] for failure in reference_self_check["failures"]
    )
    if reference_failing_ids != BFCL_REFERENCE_FAILING_TASKS:
        raise RuntimeError(
            "BFCL evaluator reference self-check changed: expected the three "
            "pinned Java representation exceptions"
        )

    snapshot = _snapshot_model(workload)
    generation_evidence: dict[str, Any] | None = None
    generation_source_files: list[Path] = []
    import_packet = os.environ.get(BFCL_IMPORT_PACKET_ENV)
    if import_packet:
        _, generation_evidence, generation_source_files = (
            import_bfcl_generation_evidence(
                packet_path=Path(import_packet),
                tasks=tasks,
                samples_path=samples_path,
            )
        )
    samples = _load_resumable_samples(samples_path, tasks)
    resumed_examples = len(samples)
    execution_dtype = torch.float32
    if device.type in {"cuda", "mps"}:
        execution_dtype = torch.bfloat16
    if len(samples) < len(tasks):
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        model = (
            AutoModelForCausalLM.from_pretrained(
                snapshot,
                local_files_only=True,
                dtype=execution_dtype,
                attn_implementation="eager",
            )
            .to(device)
            .eval()
        )
        n_params = sum(parameter.numel() for parameter in model.parameters())
        if n_params != MODEL_PARAMETER_COUNT:
            raise ValueError(
                f"Qwen3 parameter count changed: expected {MODEL_PARAMETER_COUNT}, found {n_params}"
            )
        samples = _generate_samples(
            model=model,
            tokenizer=tokenizer,
            tasks=tasks,
            existing=samples,
            device=device,
            samples_path=samples_path,
        )

    evaluation_start = time.perf_counter()
    evaluation = evaluate_bfcl_samples(
        tasks=tasks, samples=samples, source_root=source_root
    )
    evaluation_seconds = time.perf_counter() - evaluation_start
    results_path.write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    generation_seconds = sum(float(sample["latency_seconds"]) for sample in samples)
    generated_tokens = sum(int(sample["output_tokens"]) for sample in samples)
    if not math.isfinite(generation_seconds) or generation_seconds <= 0:
        raise RuntimeError("function-calling generation duration must be positive")
    if not math.isfinite(evaluation_seconds) or evaluation_seconds <= 0:
        raise RuntimeError("function-calling evaluation duration must be positive")
    score = float(evaluation["non_live_ast_accuracy"])
    target = float(workload.quality_value or TARGET_ACCURACY)
    tolerance = float(workload.quality_tolerance or 0.0)
    target_met = score + tolerance >= target
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "max",
        "status": "passed" if target_met else "quality_failed",
        "backend": (
            f"{generation_evidence['backend']}-generation/"
            f"pytorch-{device.type}-evaluation"
            if generation_evidence
            else f"pytorch-{device.type}"
        ),
        "data_mode": (
            "real-imported-generation-evidence"
            if generation_evidence
            else "real"
        ),
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "n_params": MODEL_PARAMETER_COUNT,
        },
        "model_source": {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "snapshot": str(snapshot),
            "files": {
                filename: f"sha256:{digest}" for filename, digest in MODEL_FILES.items()
            },
        },
        "dataset": {
            "name": dataset_asset.name,
            "version": f"bfcl-{BFCL_COMMIT}",
            "source": dataset_asset.source,
            "root": str(dataset_asset.root),
            "sha256": dataset_asset.sha256,
            "source_archive_sha256": f"sha256:{BFCL_ARCHIVE_SHA256}",
            "n_bytes": dataset_asset.n_bytes,
            "examples": len(tasks),
            "categories": dict(CATEGORY_COUNTS),
        },
        "evaluator": {
            "repository": "https://github.com/ShishirPatil/gorilla",
            "leaderboard_revision": BFCL_EVALUATOR_COMMIT,
            "execution_source_revision": BFCL_COMMIT,
            "execution_source_note": (
                "The pinned evaluator and Qwen handler files are byte-identical "
                "at the leaderboard and execution-source revisions."
            ),
            "files": {
                filename: f"sha256:{digest}"
                for filename, digest in BFCL_EVALUATOR_FILES.items()
            },
            "reference_self_check": {
                "correct": reference_self_check["correct"],
                "total": reference_self_check["total"],
                "expected_java_representation_exceptions": list(
                    BFCL_REFERENCE_FAILING_TASKS
                ),
                "non_live_ast_accuracy": reference_self_check["non_live_ast_accuracy"],
            },
            "results_sha256": f"sha256:{sha256_file(results_path)}",
            "transformers_version": transformers.__version__,
        },
        "seed": seed,
        "generation_evidence": generation_evidence
        or {
            "mode": "current-run-or-exact-resume",
            "backend": f"pytorch-{device.type}",
            "execution_dtype": str(execution_dtype).removeprefix("torch."),
            "examples": len(samples),
        },
        "measurement_protocol": workload.raw.get("measurement_protocol", {}),
        "config": {
            "evaluation_examples": len(tasks),
            "categories": list(CATEGORY_COUNTS),
            "decoding": "greedy",
            "upstream_temperature_equivalent": 0.001,
            "max_new_tokens": MAX_NEW_TOKENS,
            "prompt_format": "bfcl-qwen3-fc-chatml",
            "execution_dtype": str(execution_dtype).removeprefix("torch."),
            "attention_implementation": "eager",
            "aggregation": "official-non-live-ast-summary",
            "resumable_prefix_examples": resumed_examples,
            "generation_source": (
                "imported-hash-bound-generation"
                if generation_evidence
                else "current-run-or-exact-resume"
            ),
        },
        "metrics": {
            "non_live_ast_accuracy": score,
            "simple_ast_accuracy": evaluation["simple_ast_accuracy"],
            "category_accuracy": {
                name: result["accuracy"]
                for name, result in evaluation["categories"].items()
            },
            "correct_examples": evaluation["correct"],
            "evaluation_examples": evaluation["total"],
            "generated_tokens": generated_tokens,
            "generation_seconds": generation_seconds,
            "evaluation_seconds": evaluation_seconds,
            "duration_seconds": generation_seconds + evaluation_seconds,
            "tokens_per_second": generated_tokens / generation_seconds,
            "n_params": MODEL_PARAMETER_COUNT,
        },
        "quality": {
            "metric": workload.quality_metric,
            "metric_key": "non_live_ast_accuracy",
            "target": target,
            "tolerance": tolerance,
            "direction": "higher",
            "target_met": target_met,
            "quality_required": True,
            "override": False,
        },
        "functional_readiness": {
            "schema": "mlperf-edu-functional-readiness/0.1",
            "stage": "quality-conformance",
            "end_to_end_execution": True,
            "authoritative_quality_contract_executed": True,
            "current_invocation_generated_model_outputs": not bool(
                generation_evidence
            ),
            "repeatability_verified": False,
            "promotion_eligible": False,
            "next_stage": "stability" if target_met else "quality-target-review",
        },
        "artifacts": {
            "report": str(report_path),
            "provenance": str(manifest_path),
            "weights": str(snapshot),
            "samples": str(samples_path),
            "evaluation_results": str(results_path),
            "generation_packet": (
                generation_evidence["packet"] if generation_evidence else None
            ),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = build_provd(
        workload=workload.id,
        scenario=workload.scenario or "offline",
        division="open",
        hardware_fingerprint=detect_hardware(),
        report=report,
        report_path=report_path,
        weights_files=_model_file_records(snapshot),
        weights_name=MODEL_ID,
        weights_revision=MODEL_REVISION,
        weights_n_params=MODEL_PARAMETER_COUNT,
        weights_dtype=(
            f"bfloat16-source/{str(execution_dtype).removeprefix('torch.')}-execution"
        ),
        dataset_name="bfcl-v4-non-live-ast-and-pinned-evaluator",
        dataset_files=[
            paths["archive"],
            *dataset_asset.files,
            *generation_source_files,
            samples_path,
            results_path,
        ],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report
