"""Principle callout boxes must carry their book-wide global number.

The `principle` group declares `scope: global` in
config/shared/base/custom-numbered-blocks.yml, so the reader should see
"Principle 3: The Iron Law of ML Systems" wherever that box appears. The Lua
filter honours the scope by seeding its counter from `book.render`, which only
exists for a `project.type: book` render. The HTML build is a website, so the
seed no-ops and every parts file restarts at 1 -- leaving the conclusion's
resolved `\\ref{pri-iron-law}` pointing at 3 while the box itself said 1.

renumber_principle_boxes repairs the target side during the HTML post-pass.
These tests pin the behaviour that made it correct.
"""

from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "quarto" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import resolve_cross_references as rcr  # noqa: E402


def _box(pri_id, number, title):
    """Render a principle callout the way Quarto's HTML build emits one."""
    return (
        f'<div id="{pri_id}" class="callout callout-principle" title="{title}">\n'
        f'<p></p><details class="callout-principle fbx-simplebox fbx-default" open="">'
        f"<summary><strong>Principle {number}: {title}</strong></summary>"
        f"<div><strong>Invariant</strong>: ...</div></details></div>"
    )


def _numbers(monkeypatch, mapping):
    monkeypatch.setattr(rcr, "get_principle_numbers", lambda: mapping)


def test_renumbers_box_to_global_number(monkeypatch):
    _numbers(monkeypatch, {"pri-iron-law": "3"})
    html = _box("pri-iron-law", "1", "The Iron Law of ML Systems")

    out, fixed, unmapped = rcr.renumber_principle_boxes(html)

    assert fixed == 1
    assert unmapped == []
    assert "<strong>Principle 3: The Iron Law of ML Systems</strong>" in out
    # The title attribute and the anchor id are untouched.
    assert 'id="pri-iron-law"' in out
    assert 'title="The Iron Law of ML Systems"' in out


def test_leaves_already_correct_box_untouched(monkeypatch):
    _numbers(monkeypatch, {"pri-data-as-code": "1"})
    html = _box("pri-data-as-code", "1", "The Data-as-Code Principle")

    out, fixed, _ = rcr.renumber_principle_boxes(html)

    assert fixed == 0
    assert out == html


def test_is_idempotent(monkeypatch):
    _numbers(monkeypatch, {"pri-amdahl": "8"})
    html = _box("pri-amdahl", "1", "Amdahl's Law")

    once, first, _ = rcr.renumber_principle_boxes(html)
    twice, second, _ = rcr.renumber_principle_boxes(once)

    assert first == 1
    assert second == 0
    assert twice == once


def test_renumbers_each_box_independently(monkeypatch):
    _numbers(
        monkeypatch,
        {
            "pri-verification-gap": "9",
            "pri-statistical-drift": "10",
            "pri-bias-feedback": "13",
        },
    )
    html = "\n".join(
        [
            _box("pri-verification-gap", "1", "The Verification Gap"),
            _box("pri-statistical-drift", "2", "The Statistical Drift Diagnostic"),
            _box("pri-bias-feedback", "5", "The Bias Feedback Model"),
        ]
    )

    out, fixed, unmapped = rcr.renumber_principle_boxes(html)

    assert fixed == 3
    assert unmapped == []
    assert "<strong>Principle 9: The Verification Gap</strong>" in out
    assert "<strong>Principle 10: The Statistical Drift Diagnostic</strong>" in out
    assert "<strong>Principle 13: The Bias Feedback Model</strong>" in out


def test_skips_unnumbered_principle_box(monkeypatch):
    """A chapter-local `.unnumbered` principle has no "Principle N:" title.

    responsible_engr's compliance principle is the book's only one. It must not
    be rewritten, and must not be reported as unmapped -- it is deliberately
    outside the thirteen-principle ledger.
    """
    _numbers(monkeypatch, {"pri-iron-law": "3"})
    html = (
        '<div id="pri-responsible-engr-compliance-engineering" '
        'class="callout callout-principle" title="Compliance as Engineering Constraint">\n'
        '<p></p><details open=""><summary><strong>Principle</strong></summary>'
        "<div><strong>Invariant</strong>: ...</div></details></div>"
    )

    out, fixed, unmapped = rcr.renumber_principle_boxes(html)

    assert fixed == 0
    assert unmapped == []
    assert out == html


def test_reports_unknown_principle_id_as_unmapped(monkeypatch):
    _numbers(monkeypatch, {"pri-iron-law": "3"})
    html = _box("pri-not-in-ledger", "1", "Some Principle")

    out, fixed, unmapped = rcr.renumber_principle_boxes(html)

    assert fixed == 0
    assert unmapped == ["pri-not-in-ledger"]
    assert out == html


def test_does_not_reach_across_into_a_neighbouring_box(monkeypatch):
    """A div with no title of its own must not consume the next box's title."""
    _numbers(monkeypatch, {"pri-empty": "4", "pri-real": "7"})
    filler = "x" * (rcr._PRINCIPLE_TITLE_WINDOW + 50)
    html = (
        f'<div id="pri-empty" class="callout callout-principle">{filler}</div>\n'
        + _box("pri-real", "2", "A Real Principle")
    )

    out, fixed, unmapped = rcr.renumber_principle_boxes(html)

    assert fixed == 1
    assert unmapped == []
    assert "<strong>Principle 7: A Real Principle</strong>" in out


def test_no_mapping_available_is_a_no_op(monkeypatch):
    _numbers(monkeypatch, {})
    html = _box("pri-iron-law", "1", "The Iron Law of ML Systems")

    out, fixed, unmapped = rcr.renumber_principle_boxes(html)

    assert (out, fixed, unmapped) == (html, 0, [])
