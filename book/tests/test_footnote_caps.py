from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "book" / "tools" / "scripts" / "mit_press" / "check_footnote_caps.py"

spec = importlib.util.spec_from_file_location("check_footnote_caps_for_test", SCRIPT)
footnote_caps = importlib.util.module_from_spec(spec)
sys.modules["check_footnote_caps_for_test"] = footnote_caps
assert spec.loader is not None
spec.loader.exec_module(footnote_caps)


def _write_qmd(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "chapter.qmd"
    path.write_text(body, encoding="utf-8")
    return path


def test_lowercase_significant_word_in_bold_term_head_fails(tmp_path):
    qmd = _write_qmd(
        tmp_path,
        "[^fn-arith-intensity]: **Arithmetic intensity**: Body.\n",
    )

    violations = footnote_caps.scan_file(qmd, set())

    assert len(violations) == 1
    assert violations[0].kind == "term_head_case"
    assert violations[0].detail == "intensity"
    assert (
        footnote_caps.apply_fix(violations[0])
        == "[^fn-arith-intensity]: **Arithmetic Intensity**: Body."
    )


def test_term_head_check_allows_protected_tokens(tmp_path):
    qmd = _write_qmd(
        tmp_path,
        "\n".join(
            [
                "[^fn-grpc-inference]: **gRPC Inference**: Body.",
                "[^fn-vllm-paging]: **vLLM Paging**: Body.",
                "[^fn-nn-module-composition]: **nn.Module Composition**: Body.",
                "[^fn-k-center-coverage]: **k-Center Algorithm**: Body.",
                "[^fn-pj-mac-hierarchy]: **pJ/MAC Hierarchy**: Body.",
                "[^fn-flop-rate]: **FLOP/s Throughput**: Body.",
                "[^fn-math]: **$k$-Anonymity**: Body.",
                "[^fn-code]: **`torch.compile` Guard**: Body.",
            ]
        ),
    )

    allowlist = {
        "fn-grpc-inference",
        "fn-vllm-paging",
        "fn-nn-module-composition",
        "fn-k-center-coverage",
        "fn-pj-mac-hierarchy",
    }

    assert footnote_caps.scan_file(qmd, allowlist) == []


def test_allowlist_does_not_hide_other_lowercase_term_words(tmp_path):
    qmd = _write_qmd(
        tmp_path,
        "[^fn-grpc-inference]: **gRPC service**: Body.\n",
    )

    violations = footnote_caps.scan_file(qmd, {"fn-grpc-inference"})

    assert len(violations) == 1
    assert violations[0].kind == "term_head_case"
    assert violations[0].detail == "service"


def test_existing_lowercase_first_letter_check_still_fails(tmp_path):
    qmd = _write_qmd(tmp_path, "[^fn-lower]: lowercase body.\n")

    violations = footnote_caps.scan_file(qmd, set())

    assert len(violations) == 1
    assert violations[0].kind == "lowercase_first_letter"
    assert (
        footnote_caps.apply_fix(violations[0])
        == "[^fn-lower]: Lowercase body."
    )
