import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
cli_path = str(ROOT / "book")
if cli_path not in sys.path:
    sys.path.insert(0, cli_path)

from cli.commands.headings import (
    is_exempt_callout_title,
    is_exempt_heading,
    transform_sentence_case,
)


def test_sentence_case_basic_heading():
    assert transform_sentence_case("Overview of Modern Hardware Accelerators") == "Overview of modern hardware accelerators"


def test_sentence_case_preserves_acronyms_and_proper_nouns():
    heading = "Deploying PyTorch Models on GPU Clusters Using Bayesian Optimization"
    expected = "Deploying PyTorch models on GPU clusters using Bayesian optimization"
    assert transform_sentence_case(heading) == expected
    assert transform_sentence_case(heading, is_callout=True) == expected


def test_sentence_case_preserves_compound_names():
    heading = "Architecting the Model Context Protocol With Tensor Cores"
    expected = "Architecting the Model Context Protocol with Tensor Cores"
    assert transform_sentence_case(heading) == expected
    assert transform_sentence_case(heading, is_callout=True) == expected


def test_sentence_case_preserves_math_and_greek_in_callouts():
    callout = "The C³ taxonomy and the α-β model"
    expected = "The C³ taxonomy and the α-β model"
    assert transform_sentence_case(callout, is_callout=True) == expected

    callout_c2 = "The C² continuity invariant"
    expected_c2 = "The C² continuity invariant"
    assert transform_sentence_case(callout_c2, is_callout=True) == expected_c2


def test_sentence_case_preserves_parenthetical_axis():
    heading = "Partitioning the Memory Hierarchy (Compute)"
    expected = "Partitioning the memory hierarchy (Compute)"
    assert transform_sentence_case(heading) == expected
    assert transform_sentence_case(heading, is_callout=True) == expected


def test_heading_and_callout_exemptions():
    exempt_h = "Duplex transport framing: stdio vs. SSE"
    assert is_exempt_heading(exempt_h) is True
    assert transform_sentence_case(exempt_h, is_callout=False) == exempt_h

    assert is_exempt_heading("Purpose") is True
    assert transform_sentence_case("Purpose", is_callout=False) == "Purpose"

    assert is_exempt_callout_title(exempt_h) is True
    assert transform_sentence_case(exempt_h, is_callout=True) == exempt_h
