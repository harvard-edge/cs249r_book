"""Helpers for resolving MLSysIM registry reference strings."""

from __future__ import annotations

from typing import Any


_ALLOWED_ROOTS = frozenset({"Hardware", "Models", "Systems", "Infrastructure"})


def resolve_mlsysim_ref(ref: str) -> Any:
    """Resolve a canonical MLSysIM registry string such as `Hardware.Cloud.H100`."""
    if not ref or "." not in ref:
        raise ValueError(f"Expected a dotted MLSysIM registry reference, got {ref!r}")

    root_name, *parts = ref.split(".")
    if root_name not in _ALLOWED_ROOTS:
        allowed = ", ".join(sorted(_ALLOWED_ROOTS))
        raise ValueError(f"Unsupported MLSysIM registry root {root_name!r}; expected one of: {allowed}")

    import mlsysim

    current = getattr(mlsysim, root_name)
    for part in parts:
        try:
            current = getattr(current, part)
        except AttributeError as exc:
            raise KeyError(f"Could not resolve MLSysIM registry reference {ref!r}") from exc
    return current


__all__ = ["resolve_mlsysim_ref"]
