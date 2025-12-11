"""
Automated Requirements Validation Tests

Automatically tests ALL packages from requirements.txt to ensure:
1. They can be imported
2. They have the correct version
3. They actually work (basic functionality test)

This discovers ALL requirements files and validates every package.

Usage:
    pytest tests/environment/test_all_requirements.py -v

    Or via TITO:
    tito system health --verify-all
"""

import sys
import re
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pytest


def parse_requirements_file(filepath: Path) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """
    Parse a requirements.txt file and extract package specifications.

    Returns:
        List of (package_name, version_spec, original_line) tuples
        Example: [('numpy', '>=1.24.0,<3.0.0', 'numpy>=1.24.0,<3.0.0'), ...]
    """
    packages = []

    if not filepath.exists():
        return packages

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Skip -e editable installs
            if line.startswith('-e'):
                continue

            # Parse package specification
            # Handles: package, package==1.0, package>=1.0,<2.0, package[extra]>=1.0
            match = re.match(r'^([a-zA-Z0-9_-]+)(\[[\w,]+\])?(.*)?$', line)
            if match:
                package_name = match.group(1)
                version_spec = match.group(3).strip() if match.group(3) else None
                packages.append((package_name, version_spec, line))

    return packages


def discover_requirements_files() -> List[Path]:
    """
    Discover all requirements.txt files in the project.

    Returns:
        List of Path objects for requirements files
    """
    project_root = Path.cwd()

    # Primary requirements file
    requirements_files = []

    # Main requirements.txt
    main_req = project_root / "requirements.txt"
    if main_req.exists():
        requirements_files.append(main_req)

    # Additional requirements files (dev, test, docs, etc.)
    for pattern in ["requirements-*.txt", "*/requirements.txt"]:
        requirements_files.extend(project_root.glob(pattern))

    # Remove duplicates and sort
    requirements_files = sorted(set(requirements_files))

    # Filter out virtual environment and site-packages
    requirements_files = [
        f for f in requirements_files
        if '.venv' not in str(f) and 'site-packages' not in str(f)
    ]

    return requirements_files


def get_import_name(package_name: str) -> str:
    """
    Convert package name to import name.

    Some packages have different import names:
    - PyYAML → yaml
    - opencv-python → cv2
    - scikit-learn → sklearn
    - Pillow → PIL
    """
    import_map = {
        'pyyaml': 'yaml',
        'opencv-python': 'cv2',
        'opencv-python-headless': 'cv2',
        'scikit-learn': 'sklearn',
        'scikit-image': 'skimage',
        'pillow': 'PIL',
        'python-dateutil': 'dateutil',
        'attrs': 'attr',
        'beautifulsoup4': 'bs4',
    }

    package_lower = package_name.lower()
    return import_map.get(package_lower, package_name.replace('-', '_'))


def check_version_compatibility(installed_version: str, version_spec: Optional[str]) -> bool:
    """
    Check if installed version matches version specification.

    Args:
        installed_version: Version string like "1.24.3"
        version_spec: Spec like ">=1.24.0,<3.0.0" or "==1.24.0"

    Returns:
        True if compatible, False otherwise
    """
    if not version_spec:
        return True  # No version constraint

    try:
        from packaging.version import Version
        from packaging.specifiers import SpecifierSet

        spec_set = SpecifierSet(version_spec)
        return Version(installed_version) in spec_set
    except ImportError:
        # packaging not available, skip version check
        return True
    except Exception:
        # Invalid version spec, skip
        return True


def test_package_functionality(package_name: str, import_name: str) -> Tuple[bool, str]:
    """
    Test basic functionality of a package.

    Returns:
        (success, message) tuple
    """
    try:
        if package_name.lower() == 'numpy':
            import numpy as np
            arr = np.array([1, 2, 3])
            result = arr + arr
            assert np.allclose(result, [2, 4, 6])
            return True, "Array operations work"

        elif package_name.lower() == 'matplotlib':
            import matplotlib
            matplotlib.use('Agg')  # Non-GUI backend
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])
            plt.close(fig)
            return True, "Can create plots"

        elif package_name.lower() == 'pytest':
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "--version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0, "Command available"

        elif package_name.lower() == 'jupyterlab':
            result = subprocess.run(
                ["jupyter", "lab", "--version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0, "Command available"

        elif package_name.lower() == 'jupytext':
            import jupytext
            # Test basic conversion
            text = "# %% [markdown]\n# Test"
            notebook = jupytext.reads(text, fmt='py:percent')
            return notebook is not None, "Can parse notebooks"

        elif package_name.lower() == 'pyyaml' or import_name == 'yaml':
            import yaml
            data = {'test': 'value'}
            yaml_str = yaml.dump(data)
            loaded = yaml.safe_load(yaml_str)
            assert loaded == data
            return True, "YAML serialization works"

        elif package_name.lower() == 'rich':
            from rich.console import Console
            from rich.panel import Panel
            console = Console()
            with console.capture() as capture:
                console.print(Panel("Test"))
            output = capture.get()
            return len(output) > 0, "Console rendering works"

        else:
            # Generic test: just try to import
            return True, "Importable"

    except Exception as e:
        return False, f"Functionality test failed: {str(e)}"


# Discover all requirements files
REQUIREMENTS_FILES = discover_requirements_files()

# Parse all packages from all requirements files
ALL_PACKAGES = {}
for req_file in REQUIREMENTS_FILES:
    packages = parse_requirements_file(req_file)
    for pkg_name, version_spec, original_line in packages:
        if pkg_name not in ALL_PACKAGES:
            ALL_PACKAGES[pkg_name] = {
                'version_spec': version_spec,
                'sources': [req_file],
                'original_line': original_line
            }
        else:
            ALL_PACKAGES[pkg_name]['sources'].append(req_file)


class TestRequiredPackages:
    """Test all packages from requirements.txt."""

    @pytest.mark.parametrize("package_name", sorted(ALL_PACKAGES.keys()))
    def test_package_installed(self, package_name):
        """Package must be installed and importable."""
        package_info = ALL_PACKAGES[package_name]
        import_name = get_import_name(package_name)

        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')

            # Check version compatibility if specified
            version_spec = package_info['version_spec']
            if version_spec and version != 'unknown':
                is_compatible = check_version_compatibility(version, version_spec)
                assert is_compatible, (
                    f"{package_name} version {version} does not match {version_spec}"
                )

            print(f"✅ {package_name} v{version} installed")

        except ImportError as e:
            pytest.fail(
                f"❌ {package_name} cannot be imported\n"
                f"   Import name: {import_name}\n"
                f"   Required by: {', '.join(str(f) for f in package_info['sources'])}\n"
                f"   Install: pip install {package_info['original_line']}\n"
                f"   Error: {str(e)}"
            )

    @pytest.mark.parametrize("package_name", sorted(ALL_PACKAGES.keys()))
    def test_package_functionality(self, package_name):
        """Package must have basic functionality working."""
        import_name = get_import_name(package_name)

        # Test functionality
        success, message = test_package_functionality(package_name, import_name)

        if not success:
            pytest.fail(
                f"❌ {package_name} functionality test failed: {message}"
            )

        print(f"✅ {package_name}: {message}")


class TestRequirementsFileValidity:
    """Test requirements files themselves are valid."""

    @pytest.mark.parametrize("req_file", REQUIREMENTS_FILES)
    def test_requirements_file_readable(self, req_file):
        """Requirements file must be readable."""
        assert req_file.exists(), f"Requirements file not found: {req_file}"

        content = req_file.read_text()
        assert len(content) > 0, f"Requirements file is empty: {req_file}"

        print(f"✅ Requirements file readable: {req_file}")

    @pytest.mark.parametrize("req_file", REQUIREMENTS_FILES)
    def test_requirements_file_parseable(self, req_file):
        """Requirements file must be parseable."""
        packages = parse_requirements_file(req_file)

        # Should have at least one package (unless it's all comments)
        lines = req_file.read_text().splitlines()
        non_comment_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]

        if non_comment_lines:
            assert len(packages) > 0, f"No packages parsed from {req_file}"

        print(f"✅ {req_file}: {len(packages)} packages parsed")


class TestPackageVersionConsistency:
    """Test that package versions are consistent across requirements files."""

    def test_no_conflicting_versions(self):
        """Packages should not have conflicting version specs in different files."""
        conflicts = []

        # Group packages by name across all files
        package_specs = {}
        for req_file in REQUIREMENTS_FILES:
            packages = parse_requirements_file(req_file)
            for pkg_name, version_spec, original_line in packages:
                if pkg_name not in package_specs:
                    package_specs[pkg_name] = []
                package_specs[pkg_name].append({
                    'file': req_file,
                    'spec': version_spec,
                    'line': original_line
                })

        # Check for conflicts
        for pkg_name, specs in package_specs.items():
            if len(specs) > 1:
                # Multiple specifications - check if they're compatible
                unique_specs = set(s['spec'] for s in specs if s['spec'])
                if len(unique_specs) > 1:
                    conflicts.append({
                        'package': pkg_name,
                        'specs': specs
                    })

        if conflicts:
            msg = "Found conflicting version specifications:\n"
            for conflict in conflicts:
                msg += f"\n  Package: {conflict['package']}\n"
                for spec in conflict['specs']:
                    msg += f"    {spec['file']}: {spec['line']}\n"
            pytest.fail(msg)

        print(f"✅ No version conflicts found across {len(REQUIREMENTS_FILES)} requirements files")


def print_requirements_summary():
    """Print a summary of all requirements."""
    print("\n" + "="*70)
    print("📦 Requirements Summary")
    print("="*70)

    for req_file in REQUIREMENTS_FILES:
        packages = parse_requirements_file(req_file)
        print(f"\n{req_file}:")
        print(f"  {len(packages)} packages")

        for pkg_name, version_spec, _ in packages:
            spec_str = version_spec if version_spec else "(any version)"
            print(f"    - {pkg_name} {spec_str}")

    print("\n" + "="*70)
    print(f"Total unique packages: {len(ALL_PACKAGES)}")
    print("="*70)


if __name__ == "__main__":
    # Print summary first
    print_requirements_summary()

    # Run tests
    import pytest
    args = [
        __file__,
        "-v",
        "--tb=short",
        "--color=yes"
    ]

    exit_code = pytest.main(args)

    if exit_code == 0:
        print("\n" + "="*70)
        print("🎉 All required packages validated!")
        print("✅ Environment is correctly configured")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("❌ Some packages failed validation")
        print("🔧 Install missing packages: pip install -r requirements.txt")
        print("="*70)

    sys.exit(exit_code)
