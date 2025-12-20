"""
Package Manager Integration Test Runner

This module coordinates module-level integration testing for TinyTorch.
It provides immediate validation after each module completion, separate from
the larger checkpoint milestones.

Integration tests verify:
- Module exports correctly to the package
- Can be imported without errors
- Basic functionality works
- No conflicts with other modules

This is different from checkpoint tests which validate complete capabilities.
"""

import sys
import importlib
import importlib.util
import warnings
from pathlib import Path
from typing import Dict, List, Optional
import time


class PackageManagerIntegration:
    """Manages module-level integration testing for TinyTorch package."""

    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.module_mappings = {
            "01_tensor": "test_integration_01_tensor",
            "02_activations": "test_integration_02_activations",
            "03_layers": "test_integration_03_layers",
            "04_losses": "test_integration_04_losses",
            "05_dataloader": "test_integration_05_dataloader",
            "06_autograd": "test_integration_06_autograd",
            "07_optimizers": "test_integration_07_optimizers",
            "08_training": "test_integration_08_training",
            "09_convolutions": "test_integration_09_convolutions",
            "10_tokenization": "test_integration_10_tokenization",
            "11_embeddings": "test_integration_11_embeddings",
            "12_attention": "test_integration_12_attention",
            "13_transformers": "test_integration_13_transformers",
            "14_profiling": "test_integration_14_profiling",
            "15_quantization": "test_integration_15_quantization",
            "16_compression": "test_integration_16_compression",
            "17_acceleration": "test_integration_17_acceleration",
            "18_memoization": "test_integration_18_memoization",
            "19_benchmarking": "test_integration_19_benchmarking",
            "20_capstone": "test_integration_20_capstone",
        }

    def run_module_integration_test(self, module_name: str) -> Dict:
        """
        Run integration test for a specific module.

        Args:
            module_name: Module name (e.g., "02_tensor", "tensor")

        Returns:
            Dict with test results and status
        """
        # Normalize module name
        normalized_name = self._normalize_module_name(module_name)
        if not normalized_name:
            return {
                "success": False,
                "error": f"Unknown module: {module_name}",
                "module_name": module_name,
                "test_type": "integration"
            }

        # Get test file name
        test_file = self.module_mappings.get(normalized_name)
        if not test_file:
            return {
                "success": False,
                "error": f"No integration test available for module: {normalized_name}",
                "module_name": normalized_name,
                "test_type": "integration"
            }

        # Run the integration test
        try:
            # Suppress warnings during testing
            warnings.filterwarnings("ignore")

            # Import and run the test
            test_file_path = self.test_dir / f"{test_file}.py"

            if not test_file_path.exists():
                return {
                    "success": False,
                    "error": f"Integration test file not found: {test_file_path}",
                    "module_name": normalized_name,
                    "test_type": "integration"
                }

            # Load the test module dynamically
            spec = importlib.util.spec_from_file_location(test_file, test_file_path)
            test_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test_module)

            start_time = time.time()
            result = test_module.run_integration_test()
            end_time = time.time()

            # Add timing and metadata
            result["duration"] = round(end_time - start_time, 2)
            result["test_type"] = "integration"
            result["normalized_module_name"] = normalized_name

            return result

        except ImportError as e:
            return {
                "success": False,
                "error": f"Failed to import integration test for {normalized_name}: {e}",
                "module_name": normalized_name,
                "test_type": "integration"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Integration test failed for {normalized_name}: {e}",
                "module_name": normalized_name,
                "test_type": "integration"
            }

    def run_all_available_tests(self) -> Dict:
        """
        Run integration tests for all modules that have been implemented.

        Returns:
            Dict with overall results and individual test results
        """
        results = {
            "overall_success": True,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "module_results": {},
            "test_type": "integration_suite"
        }

        for module_name in self.module_mappings.keys():
            test_result = self.run_module_integration_test(module_name)
            results["module_results"][module_name] = test_result
            results["total_tests"] += 1

            if test_result["success"]:
                results["passed_tests"] += 1
            else:
                results["failed_tests"] += 1
                results["overall_success"] = False

        return results

    def validate_package_state(self) -> Dict:
        """
        Validate the overall state of the TinyTorch package.

        Returns:
            Dict with package validation results
        """
        validation_results = {
            "package_importable": False,
            "core_structure_exists": False,
            "no_import_conflicts": False,
            "essential_modules_present": [],
            "missing_modules": [],
            "overall_health": "unknown"
        }

        try:
            # Test basic package import
            import tinytorch
            validation_results["package_importable"] = True

            # Test core structure
            import tinytorch.core
            validation_results["core_structure_exists"] = True

            # Test for import conflicts
            try:
                import tinytorch.core.tensor
                import tinytorch.core.activations
                import tinytorch.core.layers
                validation_results["no_import_conflicts"] = True
            except ImportError:
                pass  # Some modules might not be implemented yet

            # Check which modules are present
            core_modules = [
                "tensor", "activations", "layers", "dense", "spatial",
                "attention", "dataloader", "autograd", "optimizers",
                "training", "compression", "kernels", "benchmarking", "mlops"
            ]

            for module in core_modules:
                try:
                    importlib.import_module(f"tinytorch.core.{module}")
                    validation_results["essential_modules_present"].append(module)
                except ImportError:
                    validation_results["missing_modules"].append(module)

            # Determine overall health
            if validation_results["package_importable"] and validation_results["core_structure_exists"]:
                if len(validation_results["essential_modules_present"]) >= 3:
                    validation_results["overall_health"] = "good"
                elif len(validation_results["essential_modules_present"]) >= 1:
                    validation_results["overall_health"] = "fair"
                else:
                    validation_results["overall_health"] = "poor"
            else:
                validation_results["overall_health"] = "critical"

        except Exception as e:
            validation_results["error"] = str(e)
            validation_results["overall_health"] = "critical"

        return validation_results

    def get_integration_status_summary(self, module_name: str) -> str:
        """
        Get a human-readable summary of integration status.

        Args:
            module_name: Module to check

        Returns:
            String with status summary
        """
        result = self.run_module_integration_test(module_name)

        if result["success"]:
            return f"✅ Module {result['module_name']} integrated successfully into package"
        else:
            return f"❌ Module {module_name} integration failed: {result.get('error', 'Unknown error')}"

    def _normalize_module_name(self, module_name: str) -> Optional[str]:
        """Normalize module name to standard format."""
        # If already in full format (e.g., "02_tensor")
        if module_name in self.module_mappings:
            return module_name

        # Try to find by short name (e.g., "tensor" -> "02_tensor")
        for full_name in self.module_mappings.keys():
            if full_name.endswith(f"_{module_name}"):
                return full_name

        return None


def run_integration_test_for_module(module_name: str) -> Dict:
    """
    Convenience function to run integration test for a single module.

    Args:
        module_name: Module name to test

    Returns:
        Dict with test results
    """
    manager = PackageManagerIntegration()
    return manager.run_module_integration_test(module_name)


def run_all_integration_tests() -> Dict:
    """
    Convenience function to run all available integration tests.

    Returns:
        Dict with all test results
    """
    manager = PackageManagerIntegration()
    return manager.run_all_available_tests()


def validate_package() -> Dict:
    """
    Convenience function to validate overall package state.

    Returns:
        Dict with package validation results
    """
    manager = PackageManagerIntegration()
    return manager.validate_package_state()


if __name__ == "__main__":
    """Run integration tests from command line."""

    if len(sys.argv) > 1:
        module_name = sys.argv[1]
        print(f"Running integration test for module: {module_name}")
        result = run_integration_test_for_module(module_name)

        print(f"\n=== Integration Test Results: {result.get('module_name', module_name)} ===")
        print(f"Success: {result['success']}")

        if 'tests' in result:
            print("\nTest Details:")
            for test in result['tests']:
                print(f"  {test['status']} {test['name']}: {test['description']}")

        if 'errors' in result and result['errors']:
            print(f"\nErrors:")
            for error in result['errors']:
                print(f"  - {error}")

        sys.exit(0 if result['success'] else 1)
    else:
        print("Running all available integration tests...")
        results = run_all_integration_tests()

        print(f"\n=== Integration Test Suite Results ===")
        print(f"Overall Success: {results['overall_success']}")
        print(f"Tests: {results['passed_tests']}/{results['total_tests']} passed")

        if results['failed_tests'] > 0:
            print(f"\nFailed modules:")
            for module, result in results['module_results'].items():
                if not result['success']:
                    print(f"  ❌ {module}: {result.get('error', 'Unknown error')}")

        sys.exit(0 if results['overall_success'] else 1)
