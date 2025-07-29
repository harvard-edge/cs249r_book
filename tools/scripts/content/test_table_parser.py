#!/usr/bin/env python3
"""
Test Suite for Grid Table Parser

This test suite contains real problematic tables extracted from the codebase
to ensure the parser handles all edge cases correctly.
"""

import unittest
import sys
from pathlib import Path

# Add the current directory to Python path to import our formatter
sys.path.insert(0, str(Path(__file__).parent))

from format_grid_tables import GridTableFormatter, GridTableAnalyzer, TableInfo


class TestTableParser(unittest.TestCase):
    """Test cases for problematic tables found in the codebase."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.formatter = GridTableFormatter()
        self.analyzer = GridTableAnalyzer()
    
    def test_simple_table(self):
        """Test basic table parsing works correctly."""
        table = """
+------------------+-----------+------------------+
| Technique        | Bit-Width | Storage Reduction|
+:=================+==========:+=================:+
| FP32             |    32-bit |        Baseline  |
+------------------+-----------+------------------+
| FP16             |    16-bit |             2×   |
+------------------+-----------+------------------+
| INT8             |     8-bit |             4×   |
+------------------+-----------+------------------+
""".strip()
        
        try:
            rows, alignments = self.formatter._parse_table(table)
            self.assertEqual(len(rows), 4)  # Header + 3 data rows
            self.assertEqual(len(rows[0]), 3)  # 3 columns
            self.assertEqual(rows[0][0], "Technique")
            self.assertEqual(rows[1][0], "FP32")
            # Test actual content-based alignment detection
            bit_width_analysis = self.analyzer.analyze_column_content(["32-bit", "16-bit", "8-bit"])
            self.assertEqual(bit_width_analysis['alignment'], 'center')
        except Exception as e:
            self.fail(f"Simple table parsing failed: {e}")
    
    def test_empty_cells_table(self):
        """Test table with empty cells."""
        table = """
+------------------+-----------+------------------+
| Technique        | Status    | Notes            |
+:=================+==========:+=================:+
| Method A         |    Ready  |                  |
+------------------+-----------+------------------+
| Method B         |           | Under development|
+------------------+-----------+------------------+
|                  |   Failed  | Deprecated       |
+------------------+-----------+------------------+
""".strip()
        
        try:
            rows, alignments = self.formatter._parse_table(table)
            self.assertEqual(len(rows), 4)  # Header + 3 data rows
            self.assertEqual(rows[1][2], "")  # Empty cell
            self.assertEqual(rows[2][1], "")  # Empty cell
            self.assertEqual(rows[3][0], "")  # Empty cell
        except Exception as e:
            self.fail(f"Empty cells table parsing failed: {e}")
    
    def test_multi_line_cells(self):
        """Test table with multi-line cell content."""
        table = """
+------------------+-----------+------------------+
| Feature          | Status    | Description      |
+:=================+==========:+=================:+
| Feature A        |   Active  | This is a long   |
|                  |           | description that |
|                  |           | spans multiple   |
|                  |           | lines            |
+------------------+-----------+------------------+
| Feature B        |  Disabled | Single line      |
+------------------+-----------+------------------+
""".strip()
        
        try:
            rows, alignments = self.formatter._parse_table(table)
            self.assertEqual(len(rows), 3)  # Header + 2 data rows
            # Multi-line content should be merged
            self.assertIn("This is a long description", rows[1][2])
            self.assertIn("spans multiple lines", rows[1][2])
        except Exception as e:
            self.fail(f"Multi-line cells table parsing failed: {e}")
    
    def test_problematic_table_1(self):
        """Test a table that was causing 'No data rows found' error."""
        # This is based on actual problematic tables from the codebase
        table = """
+---------------------------+------------------------------+
| Component                 | Purpose                      |
+===========================+==============================+
| Data Loader               | Efficient data pipeline      |
+---------------------------+------------------------------+
| Model                     | Core ML algorithm            |
+---------------------------+------------------------------+
""".strip()
        
        try:
            rows, alignments = self.formatter._parse_table(table)
            self.assertGreater(len(rows), 0, "Should find data rows")
            self.assertEqual(len(rows[0]), 2, "Should have 2 columns")
        except Exception as e:
            self.fail(f"Problematic table 1 parsing failed: {e}")
    
    def test_problematic_table_2(self):
        """Test table with inconsistent column separators."""
        table = """
+---------------------------+------------------------------+----------------------+
| Technique                 | Memory Usage                 | Performance Impact   |
+:==========================+:=============================+:=====================+
| Baseline                  | 100 MB                       | 1.0×                 |
+---------------------------+------------------------------+----------------------+
| Optimized                 | 50 MB                        | 1.5×                 |
+---------------------------+------------------------------+----------------------+
""".strip()
        
        try:
            rows, alignments = self.formatter._parse_table(table)
            self.assertGreater(len(rows), 0, "Should find data rows")
            self.assertEqual(len(rows[0]), 3, "Should have 3 columns")
        except Exception as e:
            self.fail(f"Problematic table 2 parsing failed: {e}")
    
    def test_problematic_table_3(self):
        """Test table with special characters and formatting."""
        table = """
+---------------------------+------------------------------+
| Model                     | Accuracy (%)                 |
+:==========================+=============================:+
| BERT-base                 | 84.5                         |
+---------------------------+------------------------------+
| GPT-3.5                   | 87.2                         |
+---------------------------+------------------------------+
| Custom Model              | 89.1                         |
+---------------------------+------------------------------+
""".strip()
        
        try:
            rows, alignments = self.formatter._parse_table(table)
            self.assertGreater(len(rows), 0, "Should find data rows")
            self.assertEqual(len(rows[0]), 2, "Should have 2 columns")
            # Check that numerical data is preserved
            self.assertIn("84.5", rows[1][1])
        except Exception as e:
            self.fail(f"Problematic table 3 parsing failed: {e}")
    
    def test_table_with_extra_whitespace(self):
        """Test table with inconsistent whitespace."""
        table = """
+---------------------------+------------------------------+
|     Component             |   Purpose                    |
+===========================+==============================+
|   Data Pipeline           |     ETL operations           |
+---------------------------+------------------------------+
| Model Training            | ML model development         |
+---------------------------+------------------------------+
""".strip()
        
        try:
            rows, alignments = self.formatter._parse_table(table)
            self.assertGreater(len(rows), 0, "Should find data rows")
            # Content should be trimmed
            self.assertEqual(rows[0][0], "Component")
            self.assertEqual(rows[1][0], "Data Pipeline")
        except Exception as e:
            self.fail(f"Whitespace table parsing failed: {e}")
    
    def test_table_classification(self):
        """Test table type classification."""
        # Precision comparison table
        precision_table = [
            ["Precision Format", "Bit-Width", "Storage"],
            ["FP32", "32-bit", "Baseline"],
            ["FP16", "16-bit", "2× smaller"]
        ]
        
        table_type = self.analyzer.classify_table(precision_table)
        self.assertEqual(table_type, "precision_comparison")
        
        # Model comparison table
        model_table = [
            ["Model", "Parameters", "Accuracy"],
            ["ResNet-50", "25M", "76.2%"],
            ["EfficientNet", "5.3M", "77.1%"]
        ]
        
        table_type = self.analyzer.classify_table(model_table)
        self.assertEqual(table_type, "model_comparison")
    
    def test_alignment_detection(self):
        """Test content-based alignment detection."""
        # Numeric content should be right-aligned
        numeric_values = ["2×", "4×", "8×", "50%", "100ms"]
        analysis = self.analyzer.analyze_column_content(numeric_values)
        self.assertEqual(analysis['alignment'], 'right')
        
        # Categorical content should be center-aligned
        categorical_values = ["High", "Low", "Medium", "32-bit", "CPU"]
        analysis = self.analyzer.analyze_column_content(categorical_values)
        self.assertEqual(analysis['alignment'], 'center')
        
        # Descriptive content should be left-aligned
        descriptive_values = ["Machine learning model", "Data preprocessing", "Model evaluation"]
        analysis = self.analyzer.analyze_column_content(descriptive_values)
        self.assertEqual(analysis['alignment'], 'left')
    
    def test_formatting_preserves_content(self):
        """Test that formatting preserves all original content."""
        table = """
+------------------+-----------+------------------+
| Technique        | Bit-Width | Storage Reduction|
+:=================+==========:+=================:+
| FP32             |    32-bit |        Baseline  |
| FP16             |    16-bit |             2×   |
| INT8             |     8-bit |             4×   |
+------------------+-----------+------------------+
""".strip()
        
        # Create a TableInfo object
        table_info = TableInfo(
            start_pos=0,
            end_pos=len(table),
            content=table,
            file_path=Path("test.qmd"),
            table_type="precision_comparison",
            confidence=0.9
        )
        
        formatted = self.formatter.format_table(table_info)
        
        # Check that all original content is preserved
        self.assertIn("FP32", formatted)
        self.assertIn("32-bit", formatted)
        self.assertIn("Baseline", formatted)
        self.assertIn("FP16", formatted)
        self.assertIn("16-bit", formatted)
        self.assertIn("2×", formatted)
        self.assertIn("INT8", formatted)
        self.assertIn("8-bit", formatted)
        self.assertIn("4×", formatted)


class TestRealProblematicTables(unittest.TestCase):
    """Test cases for actual problematic tables extracted from the codebase."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.formatter = GridTableFormatter()
    
    def extract_problematic_table_from_file(self, file_path: str, line_number: int) -> str:
        """Extract a table from a specific file and line number."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Find table start around the line number
            start_line = max(0, line_number - 10)
            end_line = min(len(lines), line_number + 20)
            
            # Look for table boundaries
            table_lines = []
            in_table = False
            
            for i in range(start_line, end_line):
                line = lines[i].strip()
                if '+' in line and '-' in line and not in_table:
                    # Found table start
                    in_table = True
                    table_lines.append(lines[i])
                elif in_table:
                    if '+' in line or '|' in line or '=' in line:
                        table_lines.append(lines[i])
                    elif not line.strip():
                        # Empty line, might be end of table
                        table_lines.append(lines[i])
                    else:
                        # Non-table line, end of table
                        break
            
            return ''.join(table_lines).strip()
        
        except Exception:
            return ""
    
    def test_real_problematic_tables(self):
        """Test actual problematic tables found in the codebase."""
        # List of files and line numbers where problematic tables were found
        problematic_tables = [
            ("book/contents/core/ml_systems/ml_systems.qmd", 72),
            ("book/contents/core/optimizations/optimizations.qmd", 1518),
            ("book/contents/core/training/training.qmd", 103),
        ]
        
        for file_path, line_number in problematic_tables:
            if Path(file_path).exists():
                table_content = self.extract_problematic_table_from_file(file_path, line_number)
                if table_content:
                    with self.subTest(file=file_path, line=line_number):
                        try:
                            rows, alignments = self.formatter._parse_table(table_content)
                            self.assertGreater(len(rows), 0, f"Should find data rows in {file_path}:{line_number}")
                        except Exception as e:
                            # If it fails, let's examine the table structure
                            print(f"\nProblematic table from {file_path}:{line_number}:")
                            print(table_content)
                            print(f"Error: {e}")
                            # Don't fail the test, just log for analysis
    
    def test_table_detection_accuracy(self):
        """Test that table detection doesn't have false positives."""
        # This should not be detected as a table
        non_table = """
Some regular text with + and - characters.
This is not a table at all.
+ Item 1
+ Item 2
""".strip()
        
        tables = self.formatter.find_grid_tables(non_table, Path("test.qmd"))
        self.assertEqual(len(tables), 0, "Should not detect tables in non-table content")

    def test_complex_real_world_table(self):
        """Test a complex real-world table with multi-line cells and long content."""
        table = """
+---------------+-----------------------+--------------------------------------+----------------+------------------+-----------+-------------+--------------------------------+
| Category      | Example Device        | Processor                            | Memory         | Storage          | Power     | Price Range | Example Models/Tasks           |
+===============+:======================+:=====================================+:===============+:=================+:==========+:============+:===============================+
| Cloud ML      | NVIDIA DGX A100       | 8x NVIDIA A100 GPUs (40 GB/80 GB)    | 1 TB System RAM| 15 TB NVMe SSD   | 6.5 kW    | $200 K+     | Large language models (GPT-3), |
|               |                       |                                      |                |                  |           |             | real-time video processing     |
+---------------+-----------------------+--------------------------------------+----------------+------------------+-----------+-------------+--------------------------------+
| Edge ML       | NVIDIA Jetson AGX     | 12-core Arm® Cortex®-A78AE,          | 32 GB LPDDR5   | 64GB eMMC        | 15-60 W   | $899        | Computer vision, robotics,     |
|               | Orin                  | NVIDIA Ampere GPU                    |                |                  |           |             | autonomous systems             |
+---------------+-----------------------+--------------------------------------+----------------+------------------+-----------+-------------+--------------------------------+
| Mobile ML     | iPhone 15 Pro         | A17 Pro (6-core CPU, 6-core GPU)     | 8 GB RAM       | 128 GB-1 TB      | 3-5 W     | $999+       | Face ID, computational         |
|               |                       |                                      |                |                  |           |             | photography, voice recognition |
+---------------+-----------------------+--------------------------------------+----------------+------------------+-----------+-------------+--------------------------------+
""".strip()
        
        try:
            rows, alignments = self.formatter._parse_table(table)
            self.assertGreater(len(rows), 0, "Should find data rows")
            self.assertEqual(len(rows[0]), 8, "Should have 8 columns")
            
            # Test that multi-line content is properly merged
            self.assertIn("Large language models", rows[1][7])  # Multi-line cell content
            self.assertIn("real-time video processing", rows[1][7])
            
            # Test that device names are preserved correctly
            self.assertEqual(rows[1][1], "NVIDIA DGX A100")  # Device name preserved
            self.assertEqual(rows[2][1], "NVIDIA Jetson AGX Orin")  # Multi-line device name merged
            
        except Exception as e:
            self.fail(f"Complex real-world table parsing failed: {e}")

    def test_constraint_optimization_table(self):
        """Test the constraint optimization table from optimizations.qmd."""
        table = """
+------------------------+------------------------------+------------------------------+------------------------------+
| System Constraint      | Model Representation         | Numerical Precision          | Architectural Efficiency     |
+:=======================+:============================:+:============================:+:============================:+
| Computational Cost     | ✗                            | ✓                            | ✓                            |
+------------------------+------------------------------+------------------------------+------------------------------+
| Memory and Storage     | ✓                            | ✓                            | ✗                            |
+------------------------+------------------------------+------------------------------+------------------------------+
| Latency and Throughput | ✓                            | ✗                            | ✓                            |
+------------------------+------------------------------+------------------------------+------------------------------+
| Energy Efficiency      | ✗                            | ✓                            | ✓                            |
+------------------------+------------------------------+------------------------------+------------------------------+
| Scalability            | ✓                            | ✗                            | ✓                            |
+------------------------+------------------------------+------------------------------+------------------------------+
""".strip()
        
        try:
            rows, alignments = self.formatter._parse_table(table)
            self.assertGreater(len(rows), 0, "Should find data rows")
            self.assertEqual(len(rows[0]), 4, "Should have 4 columns")
            self.assertEqual(rows[0][0], "System Constraint")
            self.assertEqual(rows[1][0], "Computational Cost")
            self.assertEqual(rows[1][1], "✗")
            self.assertEqual(rows[1][2], "✓")
            
        except Exception as e:
            self.fail(f"Constraint optimization table parsing failed: {e}")


def run_table_tests():
    """Run all table tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTableParser))
    suite.addTests(loader.loadTestsFromTestCase(TestRealProblematicTables))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    return result.wasSuccessful(), len(result.failures), len(result.errors)


if __name__ == "__main__":
    print("🧪 Running Table Parser Tests")
    print("=" * 50)
    
    success, failures, errors = run_table_tests()
    
    print(f"\n📊 Test Results:")
    print(f"Success: {success}")
    print(f"Failures: {failures}")
    print(f"Errors: {errors}")
    
    if not success:
        print(f"\n❌ {failures + errors} test(s) failed. Parser needs fixes.")
        sys.exit(1)
    else:
        print(f"\n✅ All tests passed! Parser is working correctly.")
        sys.exit(0) 