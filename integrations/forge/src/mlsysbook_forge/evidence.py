"""JSON evidence with explicit units and conservative field filtering."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Mapping
from enum import Enum
from typing import Any

# Interpretations remain in the course's local reports; publishing them is not
# needed to compare experiments. Do not collect identity from the environment.
PRIVATE_FIELDS = frozenset({
    "student_id", "student_name", "email", "username", "hostname", "user_id",
    "reflection", "reflections", "interpretation", "interpretations", "discussion",
    "decision", "decisions", "recommendation", "rationale", "markdown", "text",
    "residual_risk", "password", "secret", "token", "api_key", "authorization",
    "credentials", "cookie", "cookies",
})
SECRET = re.compile(r"(?:\bsk-[A-Za-z0-9_-]{16,}|\bBearer\s+\S+)", re.I)


def private_field(key: str) -> bool:
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    return normalized in PRIVATE_FIELDS or any(
        normalized.endswith("_" + suffix)
        for suffix in ("api_key", "password", "secret", "token", "credentials")
    )


def plain(value: Any) -> Any:
    """Convert supported result types without losing quantity units."""
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "magnitude") and hasattr(value, "units"):
        return {"value": plain(value.magnitude), "unit": str(value.units)}
    if hasattr(value, "model_dump"):
        return plain(value.model_dump(mode="python"))
    if hasattr(value, "to_dict"):
        return plain(value.to_dict())
    # EvidenceCapture is a dataclass containing encoded JSON. Its public export
    # must be decoded before recursive privacy filtering can inspect the fields.
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return plain(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"Unsupported or non-finite evidence value: {type(value).__name__}")


def sanitize(value: Any) -> Any:
    value = plain(value)
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items() if not private_field(key)}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        return SECRET.sub("[redacted]", value)
    return value


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def numeric_metrics(value: Any, prefix: str = "") -> dict[str, float | int]:
    """Flatten numeric results; leave strings and units in the evidence artifact."""
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(numeric_metrics(item, f"{prefix}/{key}" if prefix else key))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result[prefix] = value
    return result
