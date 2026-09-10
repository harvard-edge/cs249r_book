#!/usr/bin/env python3
"""
Test cases for table formatter.

Tests various edge cases including:
- Standard tables with multiple rows
- Tables with empty cells
- Tables with multi-row cells
- Tables with Unicode characters
- Tables with already bolded content
"""

import sys
from pathlib import Path
from format_tables import (
    display_width,
    parse_table,
    parse_row,
    bold_text,
    is_bolded,
    calculate_column_widths,
    extract_alignment,
    build_border,
    build_separator,
    format_cell,
    format_row,
    format_table,
    check_table_format
)


def test_display_width():
    """Test display width calculation."""
    print("Testing display_width...")

    # Basic ASCII
    assert display_width("hello") == 5

    # Bold markers should not count
    assert display_width("**hello**") == 5

    # Unicode characters
    assert display_width("↑↑ High") == 7

    # Mixed
    assert display_width("**↑↑ High**") == 7

    print("  ✓ display_width tests passed")


def test_bold_text():
    """Test bolding text."""
    print("Testing bold_text...")

    # Basic text
    assert bold_text("hello") == "**hello**"

    # Already bolded
    assert bold_text("**hello**") == "**hello**"

    # Empty text
    assert bold_text("") == ""
    assert bold_text("   ") == ""

    # Text with spaces
    assert bold_text("  hello  ") == "**hello**"

    print("  ✓ bold_text tests passed")


def test_is_bolded():
    """Test checking if text is bolded."""
    print("Testing is_bolded...")

    assert is_bolded("**hello**") == True
    assert is_bolded("hello") == False
    assert is_bolded("**hello") == False
    assert is_bolded("hello**") == False
    assert is_bolded("") == False

    print("  ✓ is_bolded tests passed")


def test_parse_row():
    """Test parsing table rows."""
    print("Testing parse_row...")

    row = "| Header 1 | Header 2 | Header 3 |"
    cells = parse_row(row)
    assert cells == ["Header 1", "Header 2", "Header 3"]

    # Empty cells
    row = "| Value 1 |  | Value 3 |"
    cells = parse_row(row)
    assert cells == ["Value 1", "", "Value 3"]

    print("  ✓ parse_row tests passed")


def test_extract_alignment():
    """Test extracting alignment from separator."""
    print("Testing extract_alignment...")

    # Left aligned
    sep = "+:===+:===+:===+"
    alignments = extract_alignment(sep)
    assert alignments == ["left", "left", "left"]

    # Center aligned
    sep = "+:===:+:===:+:===:+"
    alignments = extract_alignment(sep)
    assert alignments == ["center", "center", "center"]

    # Mixed
    sep = "+:===+:===:+===:+"
    alignments = extract_alignment(sep)
    assert alignments == ["left", "center", "right"]

    print("  ✓ extract_alignment tests passed")


def test_build_border():
    """Test building border lines."""
    print("Testing build_border...")

    widths = [10, 15, 20]
    border = build_border(widths)
    expected = "+------------+-----------------+----------------------+"
    assert border == expected

    print("  ✓ build_border tests passed")


def test_build_separator():
    """Test building separator lines."""
    print("Testing build_separator...")

    widths = [10, 15, 20]
    alignments = ["left", "center", "right"]
    sep = build_separator(widths, alignments)
    expected = "+:==========+:===============:+====================:+"
    assert sep == expected

    print("  ✓ build_separator tests passed")


def test_format_cell():
    """Test formatting cell content."""
    print("Testing format_cell...")

    # Left aligned
    cell = format_cell("Hello", 10, "left")
    assert cell == "Hello     "
    assert len(cell) == 10

    # Center aligned
    cell = format_cell("Hi", 10, "center")
    assert cell == "    Hi    "
    assert len(cell) == 10

    # Right aligned
    cell = format_cell("Bye", 10, "right")
    assert cell == "       Bye"
    assert len(cell) == 10

    # With Unicode
    cell = format_cell("↑ High", 10, "left")
    assert len(cell) == 10

    print("  ✓ format_cell tests passed")


def test_simple_table():
    """Test formatting a simple table."""
    print("Testing simple table formatting...")

    table_lines = [
        "+----------+----------+",
        "| Header 1 | Header 2 |",
        "+:=========+:=========+",
        "| Value 1  | Value 2  |",
        "+----------+----------+",
        "| Value 3  | Value 4  |",
        "+----------+----------+",
        "",
        ": Test table caption {#tbl-test}"
    ]

    table_data = parse_table(table_lines)
    assert table_data is not None

    # Check for issues (should find unbolded headers)
    issues = check_table_format(table_data)
    assert len(issues) > 0

    # Format the table
    formatted = format_table(table_data)

    # Check that headers are bolded
    assert "**Header 1**" in formatted[1]
    assert "**Header 2**" in formatted[1]

    # Check that first column is bolded
    assert "**Value 1**" in formatted[3]
    assert "**Value 3**" in formatted[5]

    print("  ✓ Simple table formatting passed")


def test_table_with_empty_cells():
    """Test table with empty cells in first column."""
    print("Testing table with empty cells...")

    table_lines = [
        "+-----------+---------+",
        "| Technique | Goal    |",
        "+:==========+:=======:+",
        "| Pruning   | Reduce  |",
        "+-----------+---------+",
        "|           | Size    |",
        "+-----------+---------+"
    ]

    table_data = parse_table(table_lines)
    assert table_data is not None

    # Format the table
    formatted = format_table(table_data)

    # Check that headers are bolded
    assert "**Technique**" in formatted[1]

    # Check that first column with content is bolded
    assert "**Pruning**" in formatted[3]

    # Empty cell should remain empty (no bold markers)
    # Should not have "****" for empty cells
    assert "****" not in formatted[5]

    print("  ✓ Table with empty cells passed")


def test_table_with_unicode():
    """Test table with Unicode characters."""
    print("Testing table with Unicode characters...")

    table_lines = [
        "+----------+----------+",
        "| Type     | Status   |",
        "+:=========+:========:+",
        "| Memory   | ↑↑ High  |",
        "+----------+----------+",
        "| Speed    | → Neutral|",
        "+----------+----------+"
    ]

    table_data = parse_table(table_lines)
    assert table_data is not None

    formatted = format_table(table_data)

    # Check formatting preserved Unicode
    assert "↑↑ High" in " ".join(formatted)
    assert "→ Neutral" in " ".join(formatted)

    print("  ✓ Table with Unicode passed")


def test_already_formatted_table():
    """Test table that's already properly formatted."""
    print("Testing already formatted table...")

    table_lines = [
        "+--------------+--------------+",
        "| **Header 1** | **Header 2** |",
        "+:============:+:============:+",
        "| **Row 1**    | Value        |",
        "+--------------+--------------+",
        "| **Row 2**    | Value        |",
        "+--------------+--------------+"
    ]

    table_data = parse_table(table_lines)
    assert table_data is not None

    # Should have no issues
    issues = check_table_format(table_data)
    # Note: May still have border width issues to check

    formatted = format_table(table_data)

    # Headers should stay bolded (not double-bolded)
    assert "**Header 1**" in formatted[1]
    assert "****Header 1****" not in formatted[1]

    print("  ✓ Already formatted table passed")


def run_all_tests():
    """Run all test cases."""
    print("=" * 60)
    print("Running Table Formatter Tests")
    print("=" * 60)
    print()

    try:
        test_display_width()
        test_bold_text()
        test_is_bolded()
        test_parse_row()
        test_extract_alignment()
        test_build_border()
        test_build_separator()
        test_format_cell()
        test_simple_table()
        test_table_with_empty_cells()
        test_table_with_unicode()
        test_already_formatted_table()

        print()
        print("=" * 60)
        print("All tests passed! ✅")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"Test failed: {e}")
        print("=" * 60)
        return 1
    except Exception as e:
        print()
        print("=" * 60)
        print(f"Error running tests: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
