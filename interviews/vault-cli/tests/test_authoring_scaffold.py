"""Tests for the `vault new` scaffold templates.

Guards against accidental removal or rewording of the markup-convention
markers in `vault new`'s scaffolded YAML. The format-compliance gate
(currently in validate_drafts.py; CORPUS_HARDENING_PLAN.md Phase 6
lifts it into vault check --strict) requires these exact bold markers
to be present in every published common_mistake / napkin_math block.

If a contributor removes one of the markers from the scaffold (e.g.,
dropping `**The Rationale:**` to "save space"), every new question
authored via `vault new` would fail the format gate. These tests
catch that at the contract level instead of after a CI red.
"""

from __future__ import annotations

from vault_cli.commands.authoring import (
    COMMON_MISTAKE_TEMPLATE,
    NAPKIN_MATH_TEMPLATE,
)

# Markers required by the format-compliance gate. Mirrored from
# interviews/vault-cli/scripts/validate_drafts.py so a marker rename
# in either place breaks the test loudly.
COMMON_MISTAKE_REQUIRED = (
    "**The Pitfall:**",
    "**The Rationale:**",
    "**The Consequence:**",
)
NAPKIN_MATH_REQUIRED = (
    "**Assumptions",     # accepts "Assumptions & Constraints" or "Assumptions:"
    "**Calculations:**",
    "**Conclusion",      # accepts "Conclusion:" or "Conclusion & Interpretation:"
)


def test_common_mistake_template_has_all_required_markers() -> None:
    for marker in COMMON_MISTAKE_REQUIRED:
        assert marker in COMMON_MISTAKE_TEMPLATE, (
            f"vault new common_mistake scaffold is missing {marker!r}; "
            f"new questions would fail the format-compliance gate."
        )


def test_napkin_math_template_has_all_required_markers() -> None:
    for marker in NAPKIN_MATH_REQUIRED:
        assert marker in NAPKIN_MATH_TEMPLATE, (
            f"vault new napkin_math scaffold is missing {marker!r}; "
            f"new questions would fail the format-compliance gate."
        )


def test_templates_are_strings_with_todo_markers() -> None:
    """Sanity: scaffolds are plain strings and contain <TODO so authors
    see what to fill in."""
    assert isinstance(COMMON_MISTAKE_TEMPLATE, str)
    assert isinstance(NAPKIN_MATH_TEMPLATE, str)
    assert "<TODO" in COMMON_MISTAKE_TEMPLATE
    assert "<TODO" in NAPKIN_MATH_TEMPLATE
