"""Binder LEGO/fmt scopes respect --path and expose formatter prose contract."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ONE_QMD = "book/quarto/contents/vol1/index.qmd"


def _binder_json(*args: str) -> dict:
    proc = subprocess.run(
        ["./book/binder", *args, "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(proc.stdout)


def _single_run(summary: dict) -> dict:
    assert summary["status"] == "passed"
    assert len(summary["runs"]) == 1
    return summary["runs"][0]


def test_lego_prose_units_respects_path_scope():
    run = _single_run(
        _binder_json("check", "code", "--scope", "lego-prose-units", "--path", ONE_QMD)
    )
    assert run["name"] == "lego-prose-units"
    assert run["files_checked"] == 1


def test_lego_units_respects_path_scope():
    run = _single_run(
        _binder_json("check", "code", "--scope", "lego-units", "--path", ONE_QMD)
    )
    assert run["name"] == "lego-units"
    assert run["files_checked"] == 1


def test_fmt_prose_contract_is_binder_native_and_path_scoped():
    run = _single_run(
        _binder_json("check", "math", "--scope", "prose-contract", "--path", ONE_QMD)
    )
    assert run["name"] == "fmt-prose-contract"
    assert run["files_checked"] == 1
