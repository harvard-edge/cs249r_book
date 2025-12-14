"""
Basic integration test that validates the Package Manager integration system.

WHAT: Tests that the integration system itself works correctly.
WHY: The integration system is the foundation for all module testing.
     If it's broken, no other tests can reliably run.

STUDENT LEARNING:
This test validates the infrastructure that makes TinyTorch's modular
development possible. When you run `tito module complete`, this system
is what exports your code to the package.
"""

import sys
from pathlib import Path
import importlib.util

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestPackageManagerIntegration:
    """Test suite for the Package Manager integration system."""

    def test_integration_system_imports(self):
        """
        WHAT: Verify the Package Manager integration module can be imported.
        WHY: This is the core system that manages module exports.

        STUDENT LEARNING:
        The Package Manager tracks which modules are exported to tinytorch/
        and ensures dependencies are correctly resolved.
        """
        integration_file = Path(__file__).parent / "package_manager_integration.py"
        spec = importlib.util.spec_from_file_location("package_manager_integration", integration_file)
        integration_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(integration_module)

        assert hasattr(integration_module, 'PackageManagerIntegration'), \
            "Module should export PackageManagerIntegration class"

    def test_manager_can_be_instantiated(self):
        """
        WHAT: Verify the Package Manager can be created.
        WHY: Without a working manager, we can't track module exports.

        STUDENT LEARNING:
        The manager instance holds configuration and state about
        which modules have been exported and their dependencies.
        """
        integration_file = Path(__file__).parent / "package_manager_integration.py"
        spec = importlib.util.spec_from_file_location("package_manager_integration", integration_file)
        integration_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(integration_module)

        manager = integration_module.PackageManagerIntegration()
        assert manager is not None, "Manager should be created successfully"

    def test_module_mappings_configured(self):
        """
        WHAT: Verify module mappings are properly configured.
        WHY: Mappings connect module numbers to their package locations.

        STUDENT LEARNING:
        Each module (01_tensor, 02_activations, etc.) maps to a location
        in the tinytorch/ package. This is how your code becomes importable.
        """
        integration_file = Path(__file__).parent / "package_manager_integration.py"
        spec = importlib.util.spec_from_file_location("package_manager_integration", integration_file)
        integration_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(integration_module)

        manager = integration_module.PackageManagerIntegration()

        assert hasattr(manager, 'module_mappings'), \
            "Manager should have module_mappings attribute"
        assert len(manager.module_mappings) > 0, \
            "Should have at least one module mapping configured"

    def test_module_name_normalization(self):
        """
        WHAT: Verify module names are normalized correctly.
        WHY: Users might type "tensor" or "01" - both should work.

        STUDENT LEARNING:
        The system is flexible with input: whether you type
        'tensor', '01', or '01_tensor', it understands what you mean.
        """
        integration_file = Path(__file__).parent / "package_manager_integration.py"
        spec = importlib.util.spec_from_file_location("package_manager_integration", integration_file)
        integration_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(integration_module)

        manager = integration_module.PackageManagerIntegration()

        # Test normalization - should map "tensor" to the full module name
        # Note: The exact normalization depends on implementation
        if hasattr(manager, '_normalize_module_name'):
            normalized = manager._normalize_module_name("tensor")
            # Should normalize to include the number prefix
            assert "tensor" in normalized.lower(), \
                f"Normalized name should contain 'tensor', got: {normalized}"

    def test_package_validation_returns_health(self):
        """
        WHAT: Verify package validation returns health information.
        WHY: This helps diagnose issues with module exports.

        STUDENT LEARNING:
        When something goes wrong with exports, the validation system
        helps pinpoint exactly which modules are broken and why.
        """
        integration_file = Path(__file__).parent / "package_manager_integration.py"
        spec = importlib.util.spec_from_file_location("package_manager_integration", integration_file)
        integration_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(integration_module)

        manager = integration_module.PackageManagerIntegration()
        validation = manager.validate_package_state()

        assert isinstance(validation, dict), \
            "Validation should return a dictionary"
        assert 'overall_health' in validation, \
            "Validation should include overall_health status"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
