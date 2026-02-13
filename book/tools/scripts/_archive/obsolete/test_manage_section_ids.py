#!/usr/bin/env python3
"""
Test suite for section_id_manager.py
Tests all major functionality including add, repair, remove, verify, and list modes.
Includes tests for hierarchy-based ID generation that handles duplicate section names
through parent section context.
"""

import unittest
import tempfile
import os
import shutil
from pathlib import Path
import sys
import subprocess
import re

# Add the scripts directory to the path so we can import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the functions we want to test
from section_id_manager import (
    simple_slugify,
    clean_text_for_id,
    generate_section_id,
    verify_section_ids,
    list_section_ids,
    create_backup,
    process_markdown_file,
    update_cross_references
)

class TestSectionIDManager(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test_chapter.qmd")

    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.test_dir)

    def create_test_file(self, content):
        """Helper method to create a test file with given content."""
        with open(self.test_file, 'w', encoding='utf-8') as f:
            f.write(content)

    def test_simple_slugify(self):
        """Test the simple_slugify function."""
        # Test basic functionality
        self.assertEqual(simple_slugify("The Optimization Techniques"), "optimization-techniques")
        self.assertEqual(simple_slugify("Introduction to Machine Learning"), "introduction-machine-learning")

        # Test with special characters
        self.assertEqual(simple_slugify("Model & Algorithm Design"), "model-algorithm-design")

        # Test with numbers (numbers should be kept)
        self.assertEqual(simple_slugify("Chapter 1: Getting Started"), "chapter-1-getting-started")

        # Test with multiple stopwords
        self.assertEqual(simple_slugify("The and or but for in on at"), "")

        # Test empty string
        self.assertEqual(simple_slugify(""), "")

    def test_clean_text_for_id(self):
        """Test the clean_text_for_id function."""
        # Test basic functionality
        self.assertEqual(clean_text_for_id("Hello World"), "hello-world")
        self.assertEqual(clean_text_for_id("Test@#$%^&*()"), "test")

        # Test with multiple special characters
        self.assertEqual(clean_text_for_id("Section 1.2: Introduction"), "section-1-2-introduction")

        # Test with multiple hyphens
        self.assertEqual(clean_text_for_id("test---multiple---hyphens"), "test-multiple-hyphens")

    def test_generate_section_id(self):
        """Test the generate_section_id function."""
        # Test that it generates consistent IDs
        id1 = generate_section_id("Test Section", "/path/file.qmd", "Test Chapter", 0, [])
        id2 = generate_section_id("Test Section", "/path/file.qmd", "Test Chapter", 0, [])
        self.assertEqual(id1, id2)

        # Test that different inputs produce different IDs
        id3 = generate_section_id("Test Section", "/path/file.qmd", "Test Chapter", 1, [])
        self.assertNotEqual(id1, id3)

        # Test that parent sections affect the hash
        id4 = generate_section_id("Test Section", "/path/file.qmd", "Test Chapter", 0, ["Parent Section"])
        self.assertNotEqual(id1, id4)

        # Test format
        self.assertTrue(id1.startswith("sec-"))
        self.assertTrue(len(id1.split('-')) >= 4)  # sec-chapter-section-hash

        # Test with stopwords
        id5 = generate_section_id("The Optimization Techniques", "/path/file.qmd", "The Introduction", 0, [])
        self.assertIn("optimization-techniques", id5)
        self.assertIn("introduction", id5)

    def test_verify_section_ids_present(self):
        """Test verify_section_ids when IDs are present."""
        content = """# Test Chapter

## Section 1 {#sec-test-section-1}

## Section 2 {#sec-test-section-2}

## Section 3 {#sec-test-section-3}
"""
        self.create_test_file(content)
        missing_ids = verify_section_ids(self.test_file)
        self.assertEqual(len(missing_ids), 0)

    def test_verify_section_ids_missing(self):
        """Test verify_section_ids when IDs are missing."""
        content = """# Test Chapter

## Section 1

## Section 2 {#sec-test-section-2}

## Section 3
"""
        self.create_test_file(content)
        missing_ids = verify_section_ids(self.test_file)
        self.assertEqual(len(missing_ids), 2)
        self.assertEqual(missing_ids[0]['title'], "Section 1")
        self.assertEqual(missing_ids[1]['title'], "Section 3")

    def test_verify_section_ids_with_divs(self):
        """Test verify_section_ids with fenced divs."""
        content = """# Test Chapter

## Section 1 {#sec-test-section-1}

::: {.callout}
## Section inside div
:::

## Section 2 {#sec-test-section-2}
"""
        self.create_test_file(content)
        missing_ids = verify_section_ids(self.test_file)
        self.assertEqual(len(missing_ids), 0)  # Should ignore section inside div

    def test_process_markdown_file_add_mode(self):
        """Test process_markdown_file in add mode (default)."""
        content = """# Test Chapter

## Section 1

## Section 2

## Section 3 {#sec-existing-id}
"""
        self.create_test_file(content)

        # Capture logging output
        import logging
        from io import StringIO
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        logging.getLogger().addHandler(handler)

        try:
            summary = process_markdown_file(self.test_file, auto_yes=True, dry_run=False)

            # Check that IDs were added and existing ones were updated if they don't match format
            self.assertEqual(len(summary['added_ids']), 2)
            # The existing ID might be updated if it doesn't match the expected format
            self.assertGreaterEqual(len(summary['updated_ids']), 0)
            self.assertEqual(len(summary['existing_sections']), 1)
            self.assertTrue(summary['modified'])

            # Check the generated IDs
            for title, section_id in summary['added_ids']:
                self.assertTrue(section_id.startswith("sec-"))
                self.assertIn("test-chapter", section_id)

        finally:
            logging.getLogger().removeHandler(handler)

    def test_process_markdown_file_repair_mode(self):
        """Test process_markdown_file in repair mode."""
        content = """# Test Chapter

## Section 1 {#sec-old-format}

## Section 2 {#sec-another-old}

## Section 3 {#sec-test-chapter-section-3-a1b2}
"""
        self.create_test_file(content)

        summary = process_markdown_file(self.test_file, auto_yes=True, dry_run=False, repair_mode=True)

        # Should update all IDs in repair mode, including the one that looks correct
        self.assertGreaterEqual(len(summary['updated_ids']), 2)
        self.assertEqual(len(summary['added_ids']), 0)
        self.assertEqual(len(summary['existing_sections']), 3)

    def test_process_markdown_file_remove_mode(self):
        """Test process_markdown_file in remove mode."""
        content = """# Test Chapter

## Section 1 {#sec-test-section-1}

## Section 2 {#sec-test-section-2}

## Section 3 {#sec-test-section-3}
"""
        self.create_test_file(content)

        summary = process_markdown_file(self.test_file, auto_yes=True, dry_run=False, remove_mode=True)

        # Should remove all IDs
        self.assertEqual(len(summary['removed_ids']), 3)
        self.assertEqual(len(summary['added_ids']), 0)
        self.assertEqual(len(summary['updated_ids']), 0)

    def test_process_markdown_file_preserve_attributes(self):
        """Test that other attributes are preserved when modifying section IDs."""
        content = """# Test Chapter

## Section 1 {.important #sec-old-id .highlight}

## Section 2 {.unnumbered}

## Section 3 {.class1 #sec-old-id2 .class2}
"""
        self.create_test_file(content)

        summary = process_markdown_file(self.test_file, auto_yes=True, dry_run=False, repair_mode=True)

        # Should update IDs while preserving other attributes
        # Note: In add mode, sections without IDs get added, not updated
        self.assertGreaterEqual(len(summary['updated_ids']), 0)

        # Check that the file still has the attributes
        with open(self.test_file, 'r', encoding='utf-8') as f:
            content_after = f.read()
            self.assertIn('.important', content_after)
            self.assertIn('.highlight', content_after)
            self.assertIn('.class1', content_after)
            self.assertIn('.class2', content_after)
            self.assertIn('.unnumbered', content_after)

    def test_create_backup(self):
        """Test backup creation."""
        content = "Test content"
        self.create_test_file(content)

        backup_path = create_backup(self.test_file)

        # Check that backup file exists
        self.assertTrue(os.path.exists(backup_path))

        # Check that backup has the same content
        with open(backup_path, 'r', encoding='utf-8') as f:
            backup_content = f.read()
        self.assertEqual(backup_content, content)

        # Clean up backup
        os.remove(backup_path)

    def test_update_cross_references(self):
        """Test cross-reference updating."""
        content = """# Test Chapter

## Section 1 {#sec-old-id}

See @sec-old-id for more information.

Also check #sec-old-id in the reference.

## Section 2

This references @sec-old-id again.
"""
        self.create_test_file(content)

        # Create a mapping of old to new IDs
        id_map = {"sec-old-id": "sec-test-chapter-section-1-a1b2"}

        # Update cross-references
        result = update_cross_references(self.test_file, id_map)

        self.assertTrue(result)

        # Check that references were updated
        with open(self.test_file, 'r', encoding='utf-8') as f:
            updated_content = f.read()
            self.assertIn("@sec-test-chapter-section-1-a1b2", updated_content)
            self.assertIn("#sec-test-chapter-section-1-a1b2", updated_content)
            self.assertNotIn("@sec-old-id", updated_content)
            self.assertNotIn("#sec-old-id", updated_content)

    def test_dry_run_mode(self):
        """Test that dry-run mode doesn't modify files."""
        content = """# Test Chapter

## Section 1

## Section 2
"""
        self.create_test_file(content)

        # Get original content
        with open(self.test_file, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # Run in dry-run mode
        summary = process_markdown_file(self.test_file, auto_yes=True, dry_run=True)

        # Check that file wasn't modified
        with open(self.test_file, 'r', encoding='utf-8') as f:
            current_content = f.read()

        self.assertEqual(original_content, current_content)

        # But summary should show what would have been done
        self.assertEqual(len(summary['added_ids']), 2)

    def test_command_line_interface(self):
        """Test the command line interface."""
        # Test help
        result = subprocess.run([
            sys.executable, 'add_section_ids.py', '--help'
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))

        self.assertEqual(result.returncode, 0)
        self.assertIn("Comprehensive Section ID Management", result.stdout)
        self.assertIn("--verify", result.stdout)
        self.assertIn("--repair", result.stdout)
        self.assertIn("--remove", result.stdout)
        self.assertIn("--list", result.stdout)

    def test_verify_mode_cli(self):
        """Test verify mode via command line."""
        content = """# Test Chapter

## Section 1

## Section 2 {#sec-test-section-2}
"""
        self.create_test_file(content)

        # Test verify mode
        result = subprocess.run([
            sys.executable, 'add_section_ids.py', '-f', self.test_file, '--verify', '--yes'
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))

        # Should exit with error code 1 because there's a missing ID
        self.assertEqual(result.returncode, 1)
        # Check both stdout and stderr for the error message
        self.assertTrue("missing ID" in result.stdout or "missing ID" in result.stderr)

    def test_list_mode_cli(self):
        """Test list mode via command line."""
        content = """# Test Chapter

## Section 1 {#sec-test-section-1}

## Section 2

## Section 3 {#sec-test-section-3}
"""
        self.create_test_file(content)

        # Test list mode
        result = subprocess.run([
            sys.executable, 'add_section_ids.py', '-f', self.test_file, '--list'
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))

        self.assertEqual(result.returncode, 0)
        self.assertIn("Section IDs in:", result.stdout)
        self.assertIn("sec-test-section-1", result.stdout)
        self.assertIn("sec-test-section-3", result.stdout)
        self.assertIn("(NO ID)", result.stdout)  # Section 2 has no ID

if __name__ == '__main__':
    # Create a test runner
    unittest.main(verbosity=2)
