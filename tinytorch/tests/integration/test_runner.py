#!/usr/bin/env python3
"""
Simplified Integration Test Runner for TinyTorch
Runs tests without complex dependencies
"""

import sys
import os
from pathlib import Path
from typing import Dict, List
import time
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class SimpleTestRunner:
    """Run tests and generate reports."""

    def __init__(self):
        self.results = []

    def run_test_file(self, module_num: str) -> Dict:
        """Run tests for a module and return results."""

        # Mock problematic imports for testing
        import unittest.mock as mock
        with mock.patch.dict('sys.modules', {
            'matplotlib': mock.MagicMock(),
            'matplotlib.pyplot': mock.MagicMock(),
            'zstandard': mock.MagicMock(),
            'pytest': mock.MagicMock(skip=lambda msg: None)
        }):
            try:
                # Import our custom test module
                test_module = __import__(f'tests.integration.test_module_{module_num}',
                                        fromlist=[''])

                # Find test classes
                test_classes = []
                for name in dir(test_module):
                    obj = getattr(test_module, name)
                    if (isinstance(obj, type) and
                        name.startswith('Test') and
                        hasattr(obj, '__module__')):
                        test_classes.append((name, obj))

                # Run tests
                total_tests = 0
                passed_tests = 0
                failed_tests = 0
                test_details = []

                for class_name, test_class in test_classes:
                    try:
                        instance = test_class()
                    except Exception as e:
                        failed_tests += 1
                        test_details.append({
                            'class': class_name,
                            'test': '__init__',
                            'status': 'FAILED',
                            'error': str(e)
                        })
                        continue

                    # Get test methods
                    for method_name in dir(instance):
                        if method_name.startswith('test_'):
                            total_tests += 1
                            test_method = getattr(instance, method_name)

                            # Run test
                            start_time = time.time()
                            try:
                                test_method()
                                passed_tests += 1
                                test_details.append({
                                    'class': class_name,
                                    'test': method_name,
                                    'status': 'PASSED',
                                    'duration': time.time() - start_time
                                })
                            except AssertionError as e:
                                failed_tests += 1
                                test_details.append({
                                    'class': class_name,
                                    'test': method_name,
                                    'status': 'FAILED',
                                    'error': f"Assertion: {e}",
                                    'duration': time.time() - start_time
                                })
                            except Exception as e:
                                # Handle skip exceptions
                                if 'skip' in str(e).lower() or 'Skipped' in str(type(e)):
                                    test_details.append({
                                        'class': class_name,
                                        'test': method_name,
                                        'status': 'SKIPPED',
                                        'reason': str(e)
                                    })
                                else:
                                    failed_tests += 1
                                    test_details.append({
                                        'class': class_name,
                                        'test': method_name,
                                        'status': 'ERROR',
                                        'error': str(e),
                                        'duration': time.time() - start_time
                                    })

                return {
                    'module': module_num,
                    'total': total_tests,
                    'passed': passed_tests,
                    'failed': failed_tests,
                    'skipped': sum(1 for d in test_details if d.get('status') == 'SKIPPED'),
                    'status': 'PASSED' if failed_tests == 0 else 'FAILED',
                    'details': test_details
                }

            except ImportError as e:
                return {
                    'module': module_num,
                    'status': 'ERROR',
                    'error': f"Could not import test module: {e}",
                    'total': 0,
                    'passed': 0,
                    'failed': 0
                }

    def print_report(self, results: Dict):
        """Print a formatted test report."""
        print("\n" + "="*60)
        print(f"📊 Integration Test Report: Module {results['module']}")
        print("="*60)

        # Summary
        status_icon = "✅" if results['status'] == 'PASSED' else "❌"
        print(f"\n{status_icon} Status: {results['status']}")
        print(f"📋 Total Tests: {results['total']}")
        print(f"✅ Passed: {results['passed']}")
        print(f"❌ Failed: {results['failed']}")
        if results.get('skipped'):
            print(f"⏭️  Skipped: {results['skipped']}")

        # Detailed results
        if results.get('details'):
            print("\n📝 Test Details:")
            print("-" * 60)

            # Group by class
            by_class = {}
            for detail in results['details']:
                class_name = detail['class']
                if class_name not in by_class:
                    by_class[class_name] = []
                by_class[class_name].append(detail)

            for class_name, tests in by_class.items():
                print(f"\n  {class_name}:")
                for test in tests:
                    status_icon = {
                        'PASSED': '✅',
                        'FAILED': '❌',
                        'ERROR': '💥',
                        'SKIPPED': '⏭️'
                    }.get(test['status'], '❓')

                    print(f"    {status_icon} {test['test']}")
                    if test['status'] in ['FAILED', 'ERROR']:
                        error_msg = test.get('error', 'Unknown error')
                        # Truncate long errors
                        if len(error_msg) > 100:
                            error_msg = error_msg[:100] + "..."
                        print(f"       → {error_msg}")
                    elif test['status'] == 'SKIPPED':
                        print(f"       → {test.get('reason', 'Skipped')}")

        # Final message
        print("\n" + "="*60)
        if results['status'] == 'PASSED':
            print("🎉 All tests passed! Ready for capability demonstration.")
        else:
            print("⚠️  Some tests failed. Please fix issues before proceeding.")
        print("="*60 + "\n")

        return results['status'] == 'PASSED'

    def save_report(self, results: Dict, output_file: str = None):
        """Save test report to JSON file."""
        if output_file is None:
            output_file = f"test_report_{results['module']}_{int(time.time())}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"📄 Report saved to: {output_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run TinyTorch integration tests")
    parser.add_argument("module", nargs='?', default="05_dataloader",
                      help="Module number (e.g., 05_dataloader)")
    parser.add_argument("--save", action="store_true",
                      help="Save report to JSON file")
    parser.add_argument("--quiet", action="store_true",
                      help="Suppress output")

    args = parser.parse_args()

    runner = SimpleTestRunner()
    results = runner.run_test_file(args.module)

    if not args.quiet:
        success = runner.print_report(results)

    if args.save:
        runner.save_report(results)

    # Return appropriate exit code
    return 0 if results['status'] == 'PASSED' else 1


if __name__ == "__main__":
    sys.exit(main())
