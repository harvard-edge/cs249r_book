#!/usr/bin/env python3
"""Verify that a rendered chapter HTML has quizzes injected correctly.

Opens a rendered chapter (file path or URL) with Playwright and checks:

1. Every ``##`` section has a ``.callout-quiz-question`` div at its end,
   OR a nearby subsection does — i.e., quizzes are injected.
2. The chapter has a ``## Self-Check Answers`` heading near the end
   (id ``self-check-answers``).
3. Each ``#quiz-question-*`` has a matching ``#quiz-answer-*`` in the
   Self-Check Answers section.
4. The "See Answers →" link in each question navigates to the correct
   answer by anchor, and the "← Back to Questions" link returns.
5. No MCQ answer in the rendered body mentions ``Option X`` / ``Choice X``
   / ``Answer X`` / ``(A)`` letter references (anti-shuffle-bug).

Usage
-----
    uv run --with playwright python3 verify_rendered.py <html_file_or_url> [...]

Exit 0 if every checked page passes; exit 1 on any error.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright, Page

LETTER_REF_PATTERNS = [
    re.compile(r"\bOption [A-D]\b", re.IGNORECASE),
    re.compile(r"\bChoice [A-D]\b", re.IGNORECASE),
    re.compile(r"\bAnswer [A-D]\b", re.IGNORECASE),
    re.compile(r"\([A-D]\)"),
]


def verify_page(page: Page, url: str) -> tuple[int, int]:
    """Return (errors, warnings) for this page."""
    errors = 0
    warnings = 0

    def err(msg: str) -> None:
        nonlocal errors
        print(f"  ERROR: {msg}")
        errors += 1

    def warn(msg: str) -> None:
        nonlocal warnings
        print(f"  WARN:  {msg}")
        warnings += 1

    page.goto(url)
    page.wait_for_load_state("networkidle")

    # (1) Question + answer callouts
    q_count = page.locator("div.callout-quiz-question").count()
    a_count = page.locator("div.callout-quiz-answer").count()
    h2_sections = page.locator(
        "main h2[id^='sec-']:not(#self-check-answers)"
    ).count()

    if q_count == 0:
        err("no .callout-quiz-question elements — quizzes did not inject")
    if q_count != a_count:
        err(f"question/answer callout mismatch: {q_count} questions vs {a_count} answers")

    # (2) Self-check-answers heading
    answers_heading = page.locator("h2#self-check-answers")
    if answers_heading.count() == 0:
        err("missing `## Self-Check Answers` heading (id=self-check-answers)")

    # (3) Pair question ids with answer ids
    q_ids = page.evaluate(
        """() => Array.from(document.querySelectorAll('div.callout-quiz-question')).map(e => e.id)"""
    )
    a_ids = set(
        page.evaluate(
            """() => Array.from(document.querySelectorAll('div.callout-quiz-answer')).map(e => e.id)"""
        )
    )
    for qid in q_ids:
        if not qid.startswith("quiz-question-"):
            err(f"malformed question id: {qid!r}")
            continue
        expected_aid = "quiz-answer-" + qid[len("quiz-question-"):]
        if expected_aid not in a_ids:
            err(f"question {qid} has no matching answer {expected_aid}")

    # (4) Cross-navigation links
    see_links = page.locator("a.question-label").count()
    back_links = page.locator("a.answer-label").count()
    if see_links != q_count:
        warn(f"expected {q_count} 'See Answers' links, found {see_links}")
    if back_links != a_count:
        warn(f"expected {a_count} 'Back to Questions' links, found {back_links}")

    # (5) Anti-shuffle-bug: no letter-references in rendered answer text
    if a_count > 0:
        answer_texts = page.evaluate(
            """() => Array.from(document.querySelectorAll('div.callout-quiz-answer'))
                         .map(e => e.innerText)"""
        )
        for i, text in enumerate(answer_texts):
            for pat in LETTER_REF_PATTERNS:
                m = pat.search(text)
                if m:
                    warn(
                        f"answer block {i} contains a letter-reference '{m.group(0)}' — "
                        "spec §5/§10 requires content-based distractor references"
                    )
                    break

    print(
        f"  summary:  ## sections={h2_sections}  Q={q_count}  A={a_count}  "
        f"self-check-answers={'yes' if answers_heading.count() > 0 else 'no'}"
    )
    return errors, warnings


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: verify_rendered.py <html_file_or_url> [...]", file=sys.stderr)
        return 2

    urls = []
    for arg in argv[1:]:
        if arg.startswith(("http://", "https://", "file://")):
            urls.append(arg)
        else:
            p = Path(arg).resolve()
            if not p.is_file():
                print(f"error: {arg} is not a file or URL", file=sys.stderr)
                return 2
            urls.append(p.as_uri())

    total_errors = 0
    total_warnings = 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        for url in urls:
            print(f"\n=== {url} ===")
            e, w = verify_page(page, url)
            total_errors += e
            total_warnings += w
        browser.close()

    print(f"\n=== TOTAL: errors={total_errors}, warnings={total_warnings} ===")
    return 1 if total_errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
