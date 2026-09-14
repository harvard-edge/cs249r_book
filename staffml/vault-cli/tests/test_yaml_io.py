"""Tests for the hardened YAML loader (REVIEWS.md H-7)."""

from __future__ import annotations

import pytest

from vault_cli.yaml_io import MAX_BYTES, VaultYamlError, load_bytes


def test_simple_mapping_loads() -> None:
    assert load_bytes(b"a: 1\nb: 2\n") == {"a": 1, "b": 2}


def test_size_cap() -> None:
    big = b"x: " + b"a" * (MAX_BYTES + 1)
    with pytest.raises(VaultYamlError, match="exceeds max"):
        load_bytes(big)


def test_depth_cap() -> None:
    # 12 nested mappings (> MAX_DEPTH=10)
    doc = "".join(f"{' ' * i}a:\n" for i in range(12)) + " " * 24 + "b: 1\n"
    with pytest.raises(VaultYamlError, match="nesting depth"):
        load_bytes(doc.encode())


def test_rejects_aliases() -> None:
    # Classic billion-laughs-lite pattern
    doc = b"a: &A\n  x: 1\nb: *A\n"
    with pytest.raises(VaultYamlError, match="aliases"):
        load_bytes(doc)
