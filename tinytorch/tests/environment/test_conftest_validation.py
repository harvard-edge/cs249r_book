"""
Conftest Export Validation Unit Tests

Tests the _MODULE_REGISTRY table and _check_module_exported helper that were
added to conftest.py to extend export validation from 4 hardcoded filenames to
all 20 TinyTorch modules.

The validation logic in conftest.py runs before every test session and is the
last line of defence against the silent-pass bug, where missing exports cause
tinytorch/__init__.py to set symbols to None and tests to pass vacuously.
Like any critical gatekeeper, the gatekeeper itself must be tested.

Usage:
    pytest tests/environment/test_conftest_validation.py -v

    Or via TITO:
    tito system health --verify

Categories:
    -k registry        # _MODULE_REGISTRY completeness and schema
    -k paths           # export-file path invariants
    -k flags           # required=True/False split
    -k check           # _check_module_exported helper behaviour
    -k soft_vs_hard    # two-tier warning vs hard-failure routing

What we test:
    1. Registry completeness  - all 20 modules have an entry
    2. Registry schema        - every row has correct field types (int, str, str, str, str|None, bool)
    3. Export paths           - forward slashes, .py extension, correct sub-package prefix
    4. Required flags         - exactly modules 01-04 are marked required=True
    5. Key-symbol naming      - symbols are valid Python identifiers
    6. _check_module_exported - file-missing, symbol-None, no-symbol-needed cases
    7. Soft warning routing   - optional module failures warn to stderr, not hard-fail
"""

import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Pull symbols directly out of conftest.py without triggering pytest_configure
# (which would run the full export check and potentially abort the session).
# ---------------------------------------------------------------------------
_TESTS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_TESTS_DIR))

# Import only the validation helpers — not the full module (avoids hook side-effects)
import importlib.util as _ilu

_conftest_path = _TESTS_DIR / "conftest.py"
_spec = _ilu.spec_from_file_location("_conftest_under_test", _conftest_path)
_conftest = _ilu.module_from_spec(_spec)
# Patch pytest so pytest_configure doesn't run during import
import unittest.mock as _mock
with _mock.patch.dict("sys.modules", {"pytest": pytest}):
    _spec.loader.exec_module(_conftest)

_MODULE_REGISTRY      = _conftest._MODULE_REGISTRY
_check_module_exported = _conftest._check_module_exported
_validate_package_exported = _conftest._validate_package_exported


# ===========================================================================
# 1. Registry completeness
# ===========================================================================

class TestRegistryCompleteness:
    """_MODULE_REGISTRY must cover exactly all 20 modules."""

    def test_has_20_entries(self):
        assert len(_MODULE_REGISTRY) == 20, (
            f"Expected 20 module entries, got {len(_MODULE_REGISTRY)}.\n"
            "Add a row for every new module or remove orphan rows."
        )

    def test_module_numbers_are_1_to_20(self):
        """Module numbers must be exactly 1..20 with no gaps or duplicates."""
        nums = [entry[0] for entry in _MODULE_REGISTRY]
        assert sorted(nums) == list(range(1, 21)), (
            f"Module numbers are not contiguous 1-20: {sorted(nums)}"
        )

    def test_module_numbers_are_unique(self):
        nums = [entry[0] for entry in _MODULE_REGISTRY]
        assert len(nums) == len(set(nums)), (
            f"Duplicate module numbers found: "
            f"{[n for n in nums if nums.count(n) > 1]}"
        )


# ===========================================================================
# 2. Registry schema
# ===========================================================================

class TestRegistrySchema:
    """Every row must have (int, str, str, str, str|None, bool)."""

    @pytest.mark.parametrize("entry", _MODULE_REGISTRY,
                             ids=[f"module_{e[0]:02d}_{e[1]}" for e in _MODULE_REGISTRY])
    def test_entry_types(self, entry):
        num, title, export_file, import_path, key_symbol, required = entry
        assert isinstance(num, int),          f"num must be int, got {type(num)}"
        assert isinstance(title, str),        f"title must be str, got {type(title)}"
        assert isinstance(export_file, str),  f"export_file must be str, got {type(export_file)}"
        assert isinstance(import_path, str),  f"import_path must be str, got {type(import_path)}"
        assert key_symbol is None or isinstance(key_symbol, str), (
            f"key_symbol must be str or None, got {type(key_symbol)}"
        )
        assert isinstance(required, bool),    f"required must be bool, got {type(required)}"

    @pytest.mark.parametrize("entry", _MODULE_REGISTRY,
                             ids=[f"module_{e[0]:02d}_{e[1]}" for e in _MODULE_REGISTRY])
    def test_non_empty_strings(self, entry):
        num, title, export_file, import_path, key_symbol, required = entry
        assert title.strip(),       f"Module {num:02d}: title must not be empty"
        assert export_file.strip(), f"Module {num:02d}: export_file must not be empty"
        assert import_path.strip(), f"Module {num:02d}: import_path must not be empty"
        if key_symbol is not None:
            assert key_symbol.strip(), f"Module {num:02d}: key_symbol must not be empty"


# ===========================================================================
# 3. Export paths
# ===========================================================================

class TestExportPaths:
    """export_file values must follow the known sub-package layout."""

    VALID_PREFIXES = ("core/", "perf/", "olympics.")

    @pytest.mark.parametrize("entry", _MODULE_REGISTRY,
                             ids=[f"module_{e[0]:02d}_{e[1]}" for e in _MODULE_REGISTRY])
    def test_export_file_prefix(self, entry):
        num, title, export_file, *_ = entry
        assert any(export_file.startswith(p) for p in self.VALID_PREFIXES), (
            f"Module {num:02d} ({title}): export_file '{export_file}' must start "
            f"with one of {self.VALID_PREFIXES}"
        )

    @pytest.mark.parametrize("entry", _MODULE_REGISTRY,
                             ids=[f"module_{e[0]:02d}_{e[1]}" for e in _MODULE_REGISTRY])
    def test_export_file_uses_forward_slashes(self, entry):
        num, title, export_file, *_ = entry
        assert "\\" not in export_file, (
            f"Module {num:02d}: export_file must use forward slashes, got '{export_file}'"
        )

    @pytest.mark.parametrize("entry", _MODULE_REGISTRY,
                             ids=[f"module_{e[0]:02d}_{e[1]}" for e in _MODULE_REGISTRY])
    def test_export_file_ends_with_py(self, entry):
        num, title, export_file, *_ = entry
        assert export_file.endswith(".py"), (
            f"Module {num:02d}: export_file '{export_file}' must end with .py"
        )

    def test_module_09_exports_to_spatial_not_convolutions(self):
        """Module 09 exports to core.spatial (non-obvious — must not drift)."""
        entry = next(e for e in _MODULE_REGISTRY if e[0] == 9)
        assert "spatial" in entry[2], (
            f"Module 09 should export to core/spatial.py, got '{entry[2]}'"
        )

    def test_module_20_exports_to_olympics(self):
        """Module 20 exports to olympics.py, a module and not a package."""
        entry = next(e for e in _MODULE_REGISTRY if e[0] == 20)
        assert entry[2] == "olympics.py", (
            f"Module 20 should export to olympics.py, got '{entry[2]}'"
        )

    def test_modules_14_to_19_export_to_perf(self):
        """Performance modules (14-19) must export under perf/."""
        for entry in _MODULE_REGISTRY:
            num = entry[0]
            export_file = entry[2]
            if 14 <= num <= 19:
                assert export_file.startswith("perf/"), (
                    f"Module {num:02d} should export to perf/, got '{export_file}'"
                )


# ===========================================================================
# 4. Required flags
# ===========================================================================

class TestRegistryMatchesRealPackage:
    """The registry is a hand-maintained copy of each module's `#| default_exp`
    directive, so nothing stops it drifting from the tree it describes. Every
    other test in this file compares the registry against itself and would pass
    just as happily if every path in it were wrong; these two compare it against
    what the export actually produces.

    Skips rather than fails when the package is not exported, so a fresh clone
    or a progressive student build does not see a spurious red."""

    PKG = Path(__file__).parent.parent.parent / "tinytorch"

    @pytest.mark.parametrize("entry", _MODULE_REGISTRY,
                             ids=[f"module_{e[0]:02d}_{e[1]}" for e in _MODULE_REGISTRY])
    def test_registry_path_exists_in_exported_package(self, entry):
        num, title, export_file, *_ = entry
        if not (self.PKG / "core" / "tensor.py").exists():
            pytest.skip("package not exported; run `tito dev export --all`")
        assert (self.PKG / export_file).exists(), (
            f"Module {num:02d} ({title}): registry points at "
            f"tinytorch/{export_file}, which does not exist. The registry has "
            f"drifted from the module's `#| default_exp` target."
        )

    def test_registry_path_matches_default_exp_directive(self):
        """Read each module's own `#| default_exp` and compare, so the registry
        cannot drift even when the package has not been exported."""
        src_root = Path(__file__).parent.parent.parent / "src"
        if not src_root.exists():
            pytest.skip("src/ not present")
        mismatches = []
        for num, title, export_file, *_ in _MODULE_REGISTRY:
            module_dir = next((d for d in sorted(src_root.iterdir())
                               if d.is_dir() and d.name.startswith(f"{num:02d}_")), None)
            if module_dir is None:
                continue
            py = next((f for f in sorted(module_dir.glob("*.py"))), None)
            if py is None:
                continue
            declared = None
            for line in py.read_text(encoding="utf-8").splitlines():
                if line.startswith("#| default_exp"):
                    declared = line.split("default_exp", 1)[1].strip()
                    break
            if declared is None:
                continue
            expected = declared.replace(".", "/") + ".py"
            if expected != export_file:
                mismatches.append(
                    f"Module {num:02d} ({title}): registry says '{export_file}', "
                    f"but `#| default_exp {declared}` means '{expected}'"
                )
        assert not mismatches, "\n".join(mismatches)


class TestRequiredFlags:
    """Exactly modules 01-04 must be required=True; 05-20 must be False."""

    def test_foundational_modules_are_required(self):
        """Modules 01-04 (Tensor, Activations, Layers, Losses) must be required."""
        for entry in _MODULE_REGISTRY:
            num, title, _, _, _, required = entry
            if 1 <= num <= 4:
                assert required is True, (
                    f"Module {num:02d} ({title}) is foundational and must be "
                    f"required=True, but required={required}"
                )

    def test_progressive_modules_are_not_required(self):
        """Modules 05-20 are optional (student may not have reached them yet)."""
        for entry in _MODULE_REGISTRY:
            num, title, _, _, _, required = entry
            if num >= 5:
                assert required is False, (
                    f"Module {num:02d} ({title}) is progressive and must be "
                    f"required=False, but required={required}"
                )

    def test_exactly_four_required_modules(self):
        required_count = sum(1 for e in _MODULE_REGISTRY if e[5] is True)
        assert required_count == 4, (
            f"Expected exactly 4 required modules (01-04), got {required_count}"
        )


# ===========================================================================
# 5. Key-symbol naming (must be valid Python identifiers)
# ===========================================================================

class TestKeySymbols:

    @pytest.mark.parametrize("entry", _MODULE_REGISTRY,
                             ids=[f"module_{e[0]:02d}_{e[1]}" for e in _MODULE_REGISTRY])
    def test_key_symbol_is_valid_identifier(self, entry):
        num, title, _, _, key_symbol, _ = entry
        if key_symbol is None:
            return  # Module 20 has no specific symbol — that's fine
        assert key_symbol.isidentifier(), (
            f"Module {num:02d} ({title}): key_symbol '{key_symbol}' is not a "
            f"valid Python identifier"
        )


# ===========================================================================
# 6. _check_module_exported behaviour
# ===========================================================================

class TestCheckModuleExported:
    """Verify _check_module_exported detects missing files and bad symbols."""

    def test_returns_error_when_file_missing(self, tmp_path):
        """A non-existent export file must produce an error entry."""
        # Temporarily redirect project_root to tmp_path
        original_root = _conftest.project_root
        _conftest.project_root = tmp_path

        try:
            errors = _check_module_exported(
                99, "Ghost", "core/ghost.py",
                "tinytorch.core.ghost", "Ghost"
            )
        finally:
            _conftest.project_root = original_root

        assert len(errors) == 1
        assert "Ghost" in errors[0]
        assert "missing exported file" in errors[0]

    def test_returns_empty_when_file_exists_and_symbol_found(self, tmp_path):
        """A file that exists with the expected symbol must return no errors."""
        # Create a fake module file
        pkg_root = tmp_path / "tinytorch"
        core_dir = pkg_root / "core"
        core_dir.mkdir(parents=True)
        mod_file = core_dir / "fake.py"
        mod_file.write_text("class FakeThing: pass\n", encoding="utf-8")

        # Register the fake module so importlib can find it
        fake_mod = types.ModuleType("tinytorch.core.fake")
        fake_mod.FakeThing = type("FakeThing", (), {})
        sys.modules["tinytorch.core.fake"] = fake_mod

        original_root = _conftest.project_root
        _conftest.project_root = tmp_path

        try:
            errors = _check_module_exported(
                99, "Fake", "core/fake.py",
                "tinytorch.core.fake", "FakeThing"
            )
        finally:
            _conftest.project_root = original_root
            sys.modules.pop("tinytorch.core.fake", None)

        assert errors == [], f"Expected no errors, got: {errors}"

    def test_detects_symbol_set_to_none(self, tmp_path):
        """A symbol that exists but is None must be reported."""
        pkg_root = tmp_path / "tinytorch"
        core_dir = pkg_root / "core"
        core_dir.mkdir(parents=True)
        mod_file = core_dir / "nullmod.py"
        mod_file.write_text("NullThing = None\n", encoding="utf-8")

        null_mod = types.ModuleType("tinytorch.core.nullmod")
        null_mod.NullThing = None
        sys.modules["tinytorch.core.nullmod"] = null_mod

        original_root = _conftest.project_root
        _conftest.project_root = tmp_path

        try:
            errors = _check_module_exported(
                99, "NullMod", "core/nullmod.py",
                "tinytorch.core.nullmod", "NullThing"
            )
        finally:
            _conftest.project_root = original_root
            sys.modules.pop("tinytorch.core.nullmod", None)

        assert len(errors) == 1
        assert "is None" in errors[0]

    def test_no_error_when_key_symbol_is_none(self, tmp_path):
        """When key_symbol=None (Module 20 style), only file existence is checked."""
        pkg_root = tmp_path / "tinytorch"
        olympics_dir = pkg_root / "olympics"
        olympics_dir.mkdir(parents=True)
        (olympics_dir / "__init__.py").write_text("", encoding="utf-8")

        original_root = _conftest.project_root
        _conftest.project_root = tmp_path

        try:
            errors = _check_module_exported(
                20, "Capstone", "olympics/__init__.py",
                "tinytorch.olympics", None
            )
        finally:
            _conftest.project_root = original_root

        assert errors == [], f"No symbol check should mean no errors: {errors}"


# ===========================================================================
# 7. _validate_package_exported: soft warning vs hard failure split
# ===========================================================================

class TestValidatePackageSoftVsHard:
    """
    The two-tier strategy must route errors correctly.
    We patch _check_module_exported to inject controlled failures.
    """

    def test_required_module_failure_causes_hard_error(self, monkeypatch):
        """A failure in a required (01-04) module must appear in hard_errors."""
        def fake_check(num, title, export_file, import_path, key_symbol):
            if num == 1:  # Tensor — required
                return [f"Module 01 (Tensor): missing exported file tinytorch/core/tensor.py"]
            return []

        monkeypatch.setattr(_conftest, "_check_module_exported", fake_check)
        # Also patch Tensor import to avoid unrelated errors
        import unittest.mock as mock
        with mock.patch("builtins.__import__", wraps=__builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__):
            is_valid, hard_errors = _conftest._validate_package_exported()

        assert not is_valid, "A required module failure should make is_valid=False"
        assert any("Module 01" in e for e in hard_errors)

    def test_optional_module_failure_does_not_hard_fail(self, monkeypatch, capsys):
        """A failure in a non-required module (05-20) must NOT produce hard errors."""
        def fake_check(num, title, export_file, import_path, key_symbol):
            if num == 15:  # Quantization — optional
                return [f"Module 15 (Quantization): missing exported file tinytorch/perf/quantization.py"]
            return []

        monkeypatch.setattr(_conftest, "_check_module_exported", fake_check)

        # Patch Tensor import so the instantiation check passes
        import unittest.mock as mock
        mock_tensor = mock.MagicMock()
        mock_tensor_instance = mock.MagicMock(spec=["data", "shape", "size", "dtype"])
        mock_tensor.return_value = mock_tensor_instance

        with mock.patch.dict("sys.modules", {"tinytorch": mock.MagicMock(Tensor=mock_tensor)}):
            is_valid, hard_errors = _conftest._validate_package_exported()

        # Module 15 is optional — no hard failure
        assert not any("Module 15" in e for e in hard_errors), (
            "Optional module failures must not appear in hard_errors"
        )

        # Warning should have been printed to stderr
        captured = capsys.readouterr()
        assert "Module 15" in captured.err, (
            "Optional module failures must produce a stderr warning"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
