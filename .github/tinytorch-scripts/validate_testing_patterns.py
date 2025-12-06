#!/usr/bin/env python3
"""
Validate testing patterns in module development files.
Ensures:
- Unit tests use test_unit_* naming
- Module integration test is named test_module()
- Tests are protected with if __name__ == "__main__"
"""

import re
import sys
from pathlib import Path


def check_module_tests(module_file):
    """Check testing patterns in a module file"""
    content = module_file.read_text()
    issues = []

    # Check for test_unit_* pattern
    unit_tests = re.findall(r'def\s+(test_unit_\w+)\s*\(', content)

    # Check for test_module() function
    has_test_module = bool(re.search(r'def\s+test_module\s*\(', content))

    # Check for if __name__ == "__main__" blocks
    has_main_guard = bool(re.search(r'if\s+__name__\s*==\s*["\']__main__["\']', content))

    # Check for improper test names (test_* but not test_unit_*)
    improper_tests = [
        name for name in re.findall(r'def\s+(test_\w+)\s*\(', content)
        if not name.startswith('test_unit_') and name != 'test_module'
    ]

    # Validate patterns
    if not unit_tests and not has_test_module:
        issues.append("No tests found (missing test_unit_* or test_module)")

    if not has_test_module:
        issues.append("Missing test_module() integration test")

    if not has_main_guard:
        issues.append("Missing if __name__ == '__main__' guard")

    if improper_tests:
        issues.append(f"Improper test names (should be test_unit_*): {', '.join(improper_tests)}")

    return {
        'unit_tests': len(unit_tests),
        'has_test_module': has_test_module,
        'has_main_guard': has_main_guard,
        'issues': issues
    }


def main():
    """Validate testing patterns across all modules"""
    modules_dir = Path("modules")
    errors = []
    warnings = []

    print("🧪 Validating Testing Patterns")
    print("=" * 60)

    # Find all module development files
    module_files = sorted(modules_dir.glob("*/*_dev.py"))

    for module_file in module_files:
        module_name = module_file.parent.name

        result = check_module_tests(module_file)

        if result['issues']:
            errors.append(f"❌ {module_name}:")
            for issue in result['issues']:
                errors.append(f"   - {issue}")
        else:
            print(f"✅ {module_name}: {result['unit_tests']} unit tests + test_module()")

    print("\n" + "=" * 60)

    # Print errors
    if errors:
        print("\n❌ Testing Pattern Issues:")
        for error in errors:
            print(f"  {error}")
        print(f"\n{len([e for e in errors if '❌' in e])} modules with testing issues!")
        sys.exit(1)
    else:
        print("\n✅ All modules follow correct testing patterns!")
        sys.exit(0)


if __name__ == "__main__":
    main()
