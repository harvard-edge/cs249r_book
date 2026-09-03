import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
cli_path = str(ROOT / "book")
if cli_path not in sys.path:
    sys.path.insert(0, cli_path)

from cli.checks import footnote_caps


def _write_qmd(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "chapter.qmd"
    path.write_text(body, encoding="utf-8")
    return path


def test_sentence_case_bold_term_head_passes(tmp_path):
    qmd = _write_qmd(
        tmp_path,
        "[^fn-arith-intensity]: **Arithmetic intensity**: Body.\n",
    )

    violations = footnote_caps.scan_file(qmd, set())

    assert violations == []


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


def test_allowlist_accepts_lowercase_brand_and_sentence_case_head(tmp_path):
    qmd = _write_qmd(
        tmp_path,
        "[^fn-grpc-inference]: **gRPC service**: Body.\n",
    )

    violations = footnote_caps.scan_file(qmd, {"fn-grpc-inference"})

    assert violations == []


def test_existing_lowercase_first_letter_check_still_fails(tmp_path):
    qmd = _write_qmd(tmp_path, "[^fn-lower]: lowercase body.\n")

    violations = footnote_caps.scan_file(qmd, set())

    assert len(violations) == 1
    assert violations[0].kind == "lowercase_first_letter"
    assert (
        footnote_caps.apply_fix(violations[0])
        == "[^fn-lower]: Lowercase body."
    )
