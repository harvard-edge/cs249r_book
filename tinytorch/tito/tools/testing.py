"""
Shared testing infrastructure for TinyTorch modules.

This module provides a standardized testing framework that ensures consistent
output format and behavior across all TinyTorch modules.
"""

import sys
import traceback
import inspect
from typing import List, Callable, Tuple, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

class ModuleTestRunner:
    """
    Standardized test runner for TinyTorch modules.

    Provides consistent output formatting, error handling, and reporting
    across all modules.
    """

    def __init__(self, module_name: str):
        """Initialize the test runner for a specific module."""
        self.module_name = module_name
        self.tests: List[Tuple[str, Callable]] = []
        self.console = Console()
        self.results: List[Tuple[str, bool, str]] = []

    def register_test(self, test_name: str, test_function: Callable) -> None:
        """Register a test function with a descriptive name."""
        self.tests.append((test_name, test_function))

    def auto_discover_tests(self, calling_module=None) -> None:
        """
        Automatically discover and register test functions from the calling module.

        Looks for functions that match specific patterns:
        - Start with 'test_'
        - End with '_comprehensive', '_integration', or specific activation names
        - Are callable functions

        Args:
            calling_module: The module to search for tests (defaults to caller's module)
        """
        if calling_module is None:
            # Get the calling module from the stack
            frame = inspect.currentframe()
            try:
                # Go up the stack to find the caller
                if frame is not None:
                    # Go up multiple frames to find the actual calling module
                    # frame.f_back is run_module_tests_auto
                    # frame.f_back.f_back is the actual module
                    caller_frame = frame.f_back
                    if caller_frame is not None:
                        caller_frame = caller_frame.f_back
                        calling_module = inspect.getmodule(caller_frame)
            finally:
                del frame

        if calling_module is None:
            print("⚠️  Could not auto-discover tests - no calling module found")
            return

        # Get all members of the calling module
        discovered_tests = []

        for name, obj in inspect.getmembers(calling_module):
            if self._is_valid_test_function(name, obj):
                # Convert function name to readable test name
                test_name = self._function_name_to_test_name(name)
                discovered_tests.append((test_name, obj))

        # Sort tests for logical sequence: unit tests first, then integration tests
        def test_priority(test_tuple):
            test_name, test_function = test_tuple
            function_name = test_function.__name__

            # Unit tests (test_unit_*) have highest priority
            if function_name.startswith('test_unit_'):
                return 0
            # Integration tests (test_module_*) have medium priority
            elif function_name.startswith('test_module_'):
                return 1
            # All other tests have lowest priority
            else:
                return 2

        discovered_tests.sort(key=test_priority)

        # Register discovered tests
        for test_name, test_function in discovered_tests:
            self.register_test(test_name, test_function)

        print(f"🔍 Auto-discovered {len(discovered_tests)} test functions")

    def _is_valid_test_function(self, name: str, obj: object) -> bool:
        """
        Check if an object is a valid test function.

        Args:
            name: Name of the object
            obj: The object to check

        Returns:
            bool: True if this is a valid test function
        """
        # Must be callable
        if not callable(obj):
            return False

        # Must be a function (not a class or other callable)
        if not inspect.isfunction(obj):
            return False

        # Must start with 'test_'
        if not name.startswith('test_'):
            return False

        # That's it! Any function starting with 'test_' is a valid test
        return True

    def _function_name_to_test_name(self, function_name: str) -> str:
        """
        Convert a function name to a readable test name.

        Args:
            function_name: The function name (e.g., 'test_tensor_creation_comprehensive')

        Returns:
            str: Human-readable test name (e.g., 'Tensor Creation')
        """
        # Remove 'test_' prefix
        name = function_name.replace('test_', '')

        # Handle specific cases
        name_mappings = {
            'tensor_creation_comprehensive': 'Tensor Creation',
            'tensor_properties_comprehensive': 'Tensor Properties',
            'tensor_arithmetic_comprehensive': 'Tensor Arithmetic',
            'tensor_integration': 'Tensor Integration',
            'relu_activation': 'ReLU Activation',
            'sigmoid_activation': 'Sigmoid Activation',
            'tanh_activation': 'Tanh Activation',
            'softmax_activation': 'Softmax Activation',
            'activations_integration': 'Activations Integration'
        }

        if name in name_mappings:
            return name_mappings[name]

        # Generic conversion: replace underscores with spaces and title case
        return name.replace('_', ' ').title()

    def run_all_tests(self) -> bool:
        """
        Run all registered tests and return overall success.

        Returns:
            bool: True if all tests passed, False otherwise
        """
        if not self.tests:
            print(f"⚠️  No tests registered for {self.module_name}")
            return False

        print(f"\n🧪 Running {self.module_name} Module Tests...")
        print("=" * 50)

        all_passed = True

        for test_name, test_function in self.tests:
            success, output = self._run_single_test(test_name, test_function)
            self.results.append((test_name, success, output))

            # Get the actual function name
            function_name = test_function.__name__

            if success:
                print(f"✅ {test_name} ({function_name}): PASSED")
            else:
                print(f"❌ {test_name} ({function_name}): FAILED")
                if output:
                    print(f"   Error: {output}")
                all_passed = False

        print("=" * 50)

        # Final summary
        total_tests = len(self.tests)
        passed_tests = sum(1 for _, success, _ in self.results if success)

        if all_passed:
            print(f"🎉 All tests passed! ({passed_tests}/{total_tests})")
            print(f"✅ {self.module_name} module is working correctly!")
        else:
            print(f"❌ {passed_tests}/{total_tests} tests passed")
            print(f"🔧 {self.module_name} module needs fixes")

        return all_passed

    def _run_single_test(self, test_name: str, test_function: Callable) -> Tuple[bool, str]:
        """
        Run a single test function and capture its result.

        Args:
            test_name: Name of the test for reporting
            test_function: The test function to execute

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Capture any output from the test function
            import io
            import contextlib

            with contextlib.redirect_stdout(io.StringIO()) as captured_stdout:
                with contextlib.redirect_stderr(io.StringIO()) as captured_stderr:
                    # Execute the test function
                    test_function()

                    # If we get here, the test passed
                    return True, ""

        except AssertionError as e:
            # Test failed with assertion
            return False, str(e)
        except Exception as e:
            # Test failed with other exception
            error_msg = f"{type(e).__name__}: {str(e)}"
            return False, error_msg

def create_test_runner(module_name: str) -> ModuleTestRunner:
    """
    Factory function to create a test runner for a module.

    Args:
        module_name: Name of the module being tested

    Returns:
        ModuleTestRunner: Configured test runner instance
    """
    return ModuleTestRunner(module_name)

def run_module_tests_auto(module_name: str) -> bool:
    """
    Automatically discover and run all tests in the calling module.

    Args:
        module_name: Name of the module being tested

    Returns:
        bool: True if all tests passed, False otherwise
    """
    test_runner = create_test_runner(module_name)
    test_runner.auto_discover_tests()
    return test_runner.run_all_tests()

# Legacy compatibility function
def run_module_tests(module_name: str, test_functions: List[Tuple[str, Callable]]) -> bool:
    """
    Legacy function for backward compatibility.

    Args:
        module_name: Name of the module being tested
        test_functions: List of (test_name, test_function) tuples

    Returns:
        bool: True if all tests passed, False otherwise
    """
    test_runner = create_test_runner(module_name)

    for test_name, test_function in test_functions:
        test_runner.register_test(test_name, test_function)

    return test_runner.run_all_tests()
