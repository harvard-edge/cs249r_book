"""Unit tests for standalone prose detector.

Verifies that series-dependent volume references ('this volume', 'Volume I',
'companion volume', 'four-volume arc', etc.) are flagged, while legitimate
systems/physics usages ('data volume', 'storage volume', 'traffic volume')
and code/math/HTML-comment contexts are permitted.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from cli.checks.prose_integrity import find_standalone_prose
from cli.commands.validate import ValidateCommand


class _StubConfig:
    def __init__(self, book_dir: Path):
        self.book_dir = book_dir


def _make_validator(root: Path) -> ValidateCommand:
    cmd = ValidateCommand.__new__(ValidateCommand)
    cmd.config_manager = _StubConfig(root)
    return cmd


# ── Direct detector unit tests ──────────────────────────────────────────────


def test_standalone_prose_flags_series_references():
    positives = [
        "In this volume, we examine neural network architectures.",
        "This Volume covers distributed training at scale.",
        "Refer to the companion volume for mathematical proofs.",
        "The companion volumes provide additional context.",
        "This four-volume series covers ML systems.",
        "A three-volume curriculum was designed for practitioners.",
        "As discussed in the previous volume, kernels require careful tuning.",
        "In the first volume, single-node execution was covered.",
        "In the subsequent volume, we move to physical robotics.",
        "Volume II focuses on distributed systems.",
        "Volume 1 covers single-machine acceleration.",
        "See Vol. 1 for details on Roofline modeling.",
        "See Vol. II for network fabric topologies.",
        "Across all four volumes, we maintain invariant closure.",
        "The four volumes together form a coherent curriculum.",
        "This invariant is preserved across volumes.",
        "Consult the other volumes for prerequisites.",
        "In each volume, we follow the same pedagogical structure.",
    ]
    for text in positives:
        hits = find_standalone_prose(text)
        assert hits, f"Expected violation for: {text!r}"


def test_standalone_prose_permits_legitimate_volume_terms():
    negatives = [
        "The data volume increases exponentially with dataset size.",
        "We provision a persistent storage volume on AWS EBS.",
        "High traffic volume can cause network congestion on spine links.",
        "Sound volume is measured in decibels.",
        "Docker volume mounting is supported in containerized deployments.",
        "The instantaneous volume of liquid in the reservoir is $V(t)$.",
        "A large volume of requests entered the inference queue.",
        "The volumetric flow rate is 5.0 L/s across the manifold.",
        "The accumulator has a volume of 0.04 cubic meters.",
        "Measure the control volume boundary according to thermodynamic laws.",
    ]
    for text in negatives:
        hits = find_standalone_prose(text)
        assert not hits, f"Unexpected false positive for: {text!r} (got {hits})"


def test_standalone_prose_masks_code_and_math():
    code_and_math = """
Here is an inline code reference `Volume II` that should be ignored.
Also `$V_{\\text{volume}}$` and `$$\\int \\text{volume} dV$$` are math.
And a cross-reference @fig-volume-arc or {#fig-volume-arc} shouldn't trigger.
<!-- Volume I is mentioned in this HTML comment -->

```python
# In this volume, we write python code
volume_id = "Volume-IV"
```
"""
    hits = find_standalone_prose(code_and_math)
    assert not hits, f"Unexpected hits in code/math/comments: {hits}"


def test_standalone_prose_flags_headings_and_callouts():
    text = """
## Takeaways from the Previous Volume

> Note: In Volume II we will explore distributed fleet execution.

- Volume I introduced the Iron Law of Processor Performance.
"""
    hits = find_standalone_prose(text)
    assert len(hits) == 3
    matches = [h.match for h in hits]
    assert any("previous volume" in m.lower() for m in matches)
    assert "Volume II" in matches
    assert "Volume I" in matches


# ── Validator integration tests ─────────────────────────────────────────────


def test_validator_standalone_prose_clean_file():
    with tempfile.NamedTemporaryFile("w", suffix=".qmd", delete=False) as f:
        f.write("# Introduction\n\nThis book explores machine learning systems.\n")
        f_path = Path(f.name)

    validator = _make_validator(f_path.parent)
    try:
        res = validator._run_standalone_prose(f_path)
        assert res.passed
        assert len(res.issues) == 0
    finally:
        f_path.unlink(missing_ok=True)


def test_validator_standalone_prose_detects_violation():
    with tempfile.NamedTemporaryFile("w", suffix=".qmd", delete=False) as f:
        f.write("# Introduction\n\nIn this volume, we examine single-machine systems.\n")
        f_path = Path(f.name)

    validator = _make_validator(f_path.parent)
    try:
        res = validator._run_standalone_prose(f_path)
        assert not res.passed
        assert len(res.issues) == 1
        assert res.issues[0].code == "standalone_prose"
        assert "this volume" in res.issues[0].message
        assert res.issues[0].line == 3
    finally:
        f_path.unlink(missing_ok=True)
