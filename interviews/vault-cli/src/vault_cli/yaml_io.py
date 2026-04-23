"""Hardened YAML I/O.

Wraps ``yaml.safe_load`` with limits that defeat billion-laughs, unbounded
string allocation, and deeply nested payloads. Every vault YAML load goes
through this module — never call ``yaml.load`` directly.

Implements REVIEWS.md H-7 (YAML DoS defenses).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import yaml
from yaml.resolver import BaseResolver

MAX_BYTES = 256 * 1024  # 256 KB per question
MAX_DEPTH = 10


class VaultYamlError(ValueError):
    """Raised when a YAML file violates a hardening rule."""


class _NoAliasSafeLoader(yaml.SafeLoader):
    """SafeLoader that refuses aliases and anchors (no legitimate use in vault YAML)."""


def _reject_alias(_loader: Any, node: Any) -> None:
    raise VaultYamlError(
        f"YAML aliases/anchors are not permitted in vault files (at {node.start_mark})"
    )


_NoAliasSafeLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, yaml.SafeLoader.construct_mapping)
_NoAliasSafeLoader.add_constructor(BaseResolver.DEFAULT_SEQUENCE_TAG, yaml.SafeLoader.construct_sequence)
_NoAliasSafeLoader.add_constructor(BaseResolver.DEFAULT_SCALAR_TAG, yaml.SafeLoader.construct_scalar)

# Override the "represent alias" path so any `&anchor` / `*alias` in the input is rejected.
def _compose_node(self: Any, parent: Any, index: Any) -> Any:  # noqa: ANN001
    event = self.peek_event()
    if isinstance(event, yaml.AliasEvent):
        raise VaultYamlError(
            f"YAML aliases are not permitted in vault files (at {event.start_mark})"
        )
    return yaml.SafeLoader.compose_node(self, parent, index)


_NoAliasSafeLoader.compose_node = _compose_node  # type: ignore[method-assign]


def _check_depth(obj: Any, max_depth: int = MAX_DEPTH, depth: int = 0) -> None:
    if depth > max_depth:
        raise VaultYamlError(f"YAML nesting depth exceeds {max_depth}")
    if isinstance(obj, dict):
        for v in obj.values():
            _check_depth(v, max_depth, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            _check_depth(v, max_depth, depth + 1)


def load_bytes(data: bytes, source: str = "<bytes>") -> Any:
    """Load a YAML document from raw bytes with hardening enforced."""
    if len(data) > MAX_BYTES:
        raise VaultYamlError(f"{source}: {len(data)} bytes exceeds max {MAX_BYTES}")
    obj = yaml.load(io.BytesIO(data), Loader=_NoAliasSafeLoader)  # noqa: S506 — hardened loader
    _check_depth(obj)
    return obj


def load_file(path: Path) -> Any:
    """Load a YAML file with all hardening rules enforced."""
    raw = path.read_bytes()
    return load_bytes(raw, source=str(path))


def dump_str(data: Any) -> str:
    """Serialize for writing vault YAML with stable formatting."""
    return yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=100,
    )


__all__ = ["VaultYamlError", "MAX_BYTES", "MAX_DEPTH", "load_bytes", "load_file", "dump_str"]
