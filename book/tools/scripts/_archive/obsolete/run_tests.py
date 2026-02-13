#!/usr/bin/env python3
"""
Test runner for section_id_manager.py
Run this script to execute all tests.
"""

import sys
import os
import subprocess

def run_tests():
    """Run the test suite."""
    print("🧪 Running Section ID Manager Tests...")
    print("=" * 50)

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Run the tests
    result = subprocess.run([
        sys.executable, 'test_section_id_manager.py'
    ], cwd=script_dir)

    print("\n" + "=" * 50)
    if result.returncode == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        print("Check the output above for details.")

    return result.returncode

if __name__ == '__main__':
    sys.exit(run_tests())
