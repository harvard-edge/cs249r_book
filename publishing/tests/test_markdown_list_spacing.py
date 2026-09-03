"""Tests for Markdown list spacing around bold lead-in paragraphs."""

from book.cli.checks.markdown_list_spacing import check_text


def test_flags_bold_leadin_before_bullet_without_blank():
    issues = check_text(
        "**Step 2**: Add IT support power and apply PUE.\n"
        "- Facility PUE: 1.12\n"
    )

    assert len(issues) == 1
    assert issues[0].line_number == 2
    assert issues[0].previous_line.startswith("**Step 2**")


def test_flags_bold_leadin_before_numbered_list_without_blank():
    issues = check_text(
        "**Inputs**: Use the following values.\n"
        "1. Batch size\n"
    )

    assert len(issues) == 1


def test_allows_blank_line_between_bold_leadin_and_list():
    assert check_text(
        "**Step 2**: Add IT support power and apply PUE.\n"
        "\n"
        "- Facility PUE: 1.12\n"
    ) == []


def test_ignores_nested_list_after_list_item():
    assert check_text(
        "- **Step 2**: Add support power.\n"
        "  - Facility PUE: 1.12\n"
    ) == []


def test_ignores_outdented_parent_list_after_nested_list_item():
    assert check_text(
        "* Batch Pipeline (Daily): Events are aggregated at midnight.\n"
        "    *   **Impact**: User leaves before recommendations update.\n"
        "* Streaming Pipeline (Real-time): Events flow through Kafka.\n"
    ) == []


def test_ignores_code_fences():
    assert check_text(
        "```markdown\n"
        "**Step 2**: Add support power.\n"
        "- Facility PUE: 1.12\n"
        "```\n"
    ) == []
