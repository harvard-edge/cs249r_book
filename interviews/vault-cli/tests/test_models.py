"""Tests for the v0.1.2 Pydantic validators.

Three validators added in the 2026-04-25 release-readiness push:

1. Visual class hardening — kind enum, path regex, alt/caption min lengths.
2. Question._zone_bloom_compatible — zone × bloom_level matrix.
3. Question._visual_path_resolves — visual.path must point at a real SVG
   file under interviews/vault/visuals/<track>/. Skipped when the working
   tree is absent (production deploys).

Each test below is a small fixture that constructs a Question (or Visual)
and asserts validation passes or fails as expected. Runnable as
``python3 tests/test_models.py`` if pytest isn't available.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make vault_cli importable.
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))

from vault_cli.models import Question, Visual  # noqa: E402

# ─── Visual.kind ─────────────────────────────────────────────────────────────


def test_visual_kind_svg_accepted():
    v = Visual(
        kind="svg",
        path="cloud-1.svg",
        alt="A diagram of a CPU cache hierarchy.",
        caption="Cache Hierarchy",
    )
    assert v.kind == "svg"


def test_visual_kind_mermaid_rejected():
    """v0.1.2: mermaid was reserved but never shipped — retired."""
    try:
        Visual(kind="mermaid", path="x.svg", alt="long enough alt", caption="cap1")
    except Exception as e:
        assert "mermaid was reserved but never implemented" in str(e)
        return
    raise AssertionError("Visual.kind=mermaid should have been rejected")


# ─── Visual.path regex ───────────────────────────────────────────────────────


def test_visual_path_uppercase_rejected():
    try:
        Visual(kind="svg", path="Cloud-1.svg", alt="long enough alt", caption="cap1")
    except Exception as e:
        assert "lowercase + dash + dot only" in str(e)
        return
    raise AssertionError("uppercase path should have been rejected")


def test_visual_path_underscore_rejected():
    try:
        Visual(kind="svg", path="cloud_1.svg", alt="long enough alt", caption="cap1")
    except Exception as e:
        assert "must match" in str(e)
        return
    raise AssertionError("underscore path should have been rejected")


def test_visual_path_traversal_rejected():
    try:
        Visual(
            kind="svg", path="../etc/passwd.svg",
            alt="long enough alt", caption="cap1",
        )
    except Exception as e:
        assert "safe relative filename" in str(e)
        return
    raise AssertionError("path traversal should have been rejected")


def test_visual_path_no_extension_rejected():
    try:
        Visual(kind="svg", path="cloud-1", alt="long enough alt", caption="cap1")
    except Exception as e:
        assert "must match" in str(e)
        return
    raise AssertionError("no-extension path should have been rejected")


# ─── Visual.alt and caption min lengths ──────────────────────────────────────


def test_visual_alt_too_short_rejected():
    try:
        Visual(kind="svg", path="cloud-1.svg", alt="short", caption="cap1")
    except Exception as e:
        assert "≥10 chars" in str(e)
        return
    raise AssertionError("short alt should have been rejected")


def test_visual_caption_too_short_rejected():
    try:
        Visual(
            kind="svg", path="cloud-1.svg",
            alt="A long enough alt text.", caption="x",
        )
    except Exception as e:
        assert "≥5 chars" in str(e)
        return
    raise AssertionError("short caption should have been rejected")


def test_visual_caption_required():
    try:
        Visual(kind="svg", path="cloud-1.svg", alt="A long enough alt.")
    except Exception:
        return  # caption missing → pydantic field-required error is fine
    raise AssertionError("missing caption should have been rejected")


# ─── Question._zone_bloom_compatible ─────────────────────────────────────────


def _question(zone: str, bloom: str) -> dict:
    return {
        "schema_version": "1.0",
        "id": "cloud-9999",
        "track": "cloud",
        "level": "L4",
        "zone": zone,
        "topic": "memory-hierarchy-design",
        "competency_area": "memory",
        "bloom_level": bloom,
        "title": "Test",
        "scenario": "A scenario long enough to be realistic.",
        "details": {"realistic_solution": "x"},
    }


def test_zone_bloom_recall_remember_accepted():
    Question(**_question("recall", "remember"))


def test_zone_bloom_recall_evaluate_rejected():
    try:
        Question(**_question("recall", "evaluate"))
    except Exception as e:
        assert "incompatible" in str(e) and "recall" in str(e)
        return
    raise AssertionError("recall+evaluate should have been rejected")


# ─── Phase 6: Details extra="forbid" + format-marker structural check ────────


def test_details_extra_forbidden():
    """Phase 6: Details rejects unknown keys."""
    q = _question("recall", "remember")
    q["details"] = {"realistic_solution": "x", "made_up_field": "boom"}
    try:
        Question(**q)
    except Exception as e:
        assert "made_up_field" in str(e) or "extra" in str(e).lower()
        return
    raise AssertionError("Details should reject unknown keys")


def test_format_markers_invariant_passes_on_compliant_cm_nm():
    """The structural check accepts marker-compliant cm/nm on published."""
    from vault_cli.loader import LoadedQuestion
    from vault_cli.validator import _check_format_markers
    base = {**_question("analyze", "analyze"), "status": "published"}
    q = Question(**{
        **base,
        "details": {
            "realistic_solution": "x",
            "common_mistake": (
                "**The Pitfall:** wrong intuition.\n"
                "**The Rationale:** because reasons.\n"
                "**The Consequence:** bad outcome."
            ),
            "napkin_math": (
                "**Assumptions:** ok.\n\n"
                "**Calculations:**\n- step 1\n\n"
                "**Conclusion:** done."
            ),
        },
    })
    lq = LoadedQuestion(question=q, path=Path("/tmp/x.yaml"))
    assert _check_format_markers([lq]) == []


def test_format_markers_invariant_flags_malformed_cm_on_published():
    """A published YAML with malformed common_mistake fails the structural check."""
    from vault_cli.loader import LoadedQuestion
    from vault_cli.validator import _check_format_markers
    base = {**_question("analyze", "analyze"), "status": "published"}
    q = Question(**{
        **base,
        "details": {
            "realistic_solution": "x",
            "common_mistake": "Just a sentence with no markers at all.",
        },
    })
    lq = LoadedQuestion(question=q, path=Path("/tmp/x.yaml"))
    failures = _check_format_markers([lq])
    assert len(failures) == 1
    assert failures[0].check == "format-markers"
    assert "common_mistake" in failures[0].message


def test_format_markers_invariant_skips_drafts():
    """Drafts are exempt — author-in-progress malformed markers are fine."""
    from vault_cli.loader import LoadedQuestion
    from vault_cli.validator import _check_format_markers
    base = _question("analyze", "analyze")
    base["status"] = "draft"
    q = Question(**{
        **base,
        "details": {
            "realistic_solution": "x",
            "common_mistake": "no markers here",
            "napkin_math": "no markers here either",
        },
    })
    lq = LoadedQuestion(question=q, path=Path("/tmp/x.yaml"))
    assert _check_format_markers([lq]) == []


def test_zone_bloom_mastery_remember_rejected():
    try:
        Question(**_question("mastery", "remember"))
    except Exception as e:
        assert "incompatible" in str(e) and "mastery" in str(e)
        return
    raise AssertionError("mastery+remember should have been rejected")


def test_zone_bloom_evaluation_evaluate_accepted():
    q = _question("evaluation", "evaluate")
    q["competency_area"] = "cross-cutting"
    Question(**q)


def test_zone_bloom_design_create_accepted():
    q = _question("design", "create")
    q["competency_area"] = "cross-cutting"
    Question(**q)


# ─── Question._visual_path_resolves ──────────────────────────────────────────


def test_visual_path_must_resolve():
    """A visual whose path doesn't resolve to a real SVG should fail."""
    q = _question("evaluation", "evaluate")
    q["competency_area"] = "memory"
    q["visual"] = {
        "kind": "svg",
        "path": "cloud-doesnotexist-9999.svg",
        "alt": "A fake visual that does not exist on disk.",
        "caption": "Will fail",
    }
    try:
        Question(**q)
    except Exception as e:
        assert "does not resolve to a real file" in str(e)
        return
    raise AssertionError("missing visual.path should have been rejected")


# ─── Test runner (no pytest dependency) ──────────────────────────────────────


def _run() -> int:
    tests = [
        (n, fn) for n, fn in globals().items()
        if n.startswith("test_") and callable(fn)
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{len(tests)} passed; {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run())
