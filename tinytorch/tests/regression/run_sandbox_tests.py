#!/usr/bin/env python
"""
TinyTorch Sandbox Integrity Tests
==================================
Run this to ensure the student learning sandbox is robust.
All core infrastructure must work perfectly so students can
focus on learning ML systems, not debugging framework issues.
"""

import sys
import os
import importlib

# Test modules to run
TEST_MODULES = [
    'test_conv_linear_dimensions',
    'test_transformer_reshaping',
]

def run_sandbox_tests():
    """Run all sandbox integrity tests."""
    print("="*60)
    print("🧪 TINYTORCH SANDBOX INTEGRITY CHECK")
    print("="*60)
    print("\nEnsuring the learning environment is robust...\n")

    all_passed = True
    results = []

    for test_module in TEST_MODULES:
        try:
            # Import and run the test module
            print(f"Running {test_module}...")
            module = importlib.import_module(test_module)

            # Look for a main function or run tests directly
            if hasattr(module, 'main'):
                result = module.main()
            elif '__main__' in dir(module):
                # Module runs tests when imported
                result = True
            else:
                # Try to run all test functions
                test_funcs = [f for f in dir(module) if f.startswith('test_')]
                for func_name in test_funcs:
                    func = getattr(module, func_name)
                    func()
                result = True

            results.append((test_module, True, "PASSED"))
            print(f"  ✅ {test_module}: PASSED\n")

        except Exception as e:
            results.append((test_module, False, str(e)))
            print(f"  ❌ {test_module}: FAILED")
            print(f"     Error: {e}\n")
            all_passed = False

    # Summary
    print("="*60)
    print("📊 SANDBOX TEST SUMMARY")
    print("="*60)

    for module, passed, status in results:
        icon = "✅" if passed else "❌"
        print(f"{icon} {module}: {status}")

    if all_passed:
        print("\n🎉 SANDBOX IS ROBUST!")
        print("Students can focus on learning ML systems.")
        return 0
    else:
        print("\n⚠️  SANDBOX NEEDS ATTENTION")
        print("Some infrastructure tests failed.")
        print("Students might encounter framework issues.")
        return 1

if __name__ == "__main__":
    # Add the test directory to path
    test_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, test_dir)

    # Run tests
    exit_code = run_sandbox_tests()
    sys.exit(exit_code)
