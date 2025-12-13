"""
Comprehensive Integration Tests for Module 01: Setup

This test suite provides comprehensive pytest testing for the enhanced Module 1 Setup
functionality including:

1. Individual function testing (setup_environment, verify_environment, configure_and_display)
2. Enhanced package verification with command execution
3. Integration workflow testing
4. Error handling and edge cases
5. Timeout handling and performance validation

Tests validate both success and failure scenarios with proper assertions.
"""

import pytest
import sys
import subprocess
import platform
import psutil
import time
import os
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, MagicMock

# Add the modules path to sys.path for testing
MODULES_PATH = Path(__file__).parent.parent.parent / "modules" / "01_setup"
sys.path.insert(0, str(MODULES_PATH))

try:
    from setup_dev import setup_environment, verify_environment, configure_and_display
except ImportError as e:
    pytest.skip(f"Cannot import setup module: {e}", allow_module_level=True)


class TestSetupEnvironmentIndividual:
    """Test setup_environment function individually."""

    def test_setup_environment_return_structure(self):
        """Test that setup_environment returns the correct structure."""
        result = setup_environment()

        assert isinstance(result, dict), "Should return a dictionary"

        required_keys = [
            'current_environment', 'python_version_ok', 'packages_status',
            'action_needed', 'next_steps', 'auto_install_attempted'
        ]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

    def test_setup_environment_data_types(self):
        """Test that setup_environment returns correct data types."""
        result = setup_environment()

        assert isinstance(result['current_environment'], str), "current_environment should be string"
        assert isinstance(result['python_version_ok'], bool), "python_version_ok should be boolean"
        assert isinstance(result['packages_status'], dict), "packages_status should be dict"
        assert isinstance(result['action_needed'], bool), "action_needed should be boolean"
        assert isinstance(result['next_steps'], list), "next_steps should be list"
        assert isinstance(result['auto_install_attempted'], bool), "auto_install_attempted should be boolean"

    def test_setup_environment_detects_environment_type(self):
        """Test that setup correctly detects environment type."""
        result = setup_environment()

        valid_environments = ['virtual_environment', 'system_python', 'google_colab']
        assert result['current_environment'] in valid_environments, \
            f"Unknown environment type: {result['current_environment']}"

    def test_setup_environment_checks_python_version(self):
        """Test that setup correctly checks Python version."""
        result = setup_environment()

        # Should correctly identify if Python version is OK
        expected_version_ok = sys.version_info >= (3, 8)
        assert result['python_version_ok'] == expected_version_ok, \
            f"Python version check mismatch: expected {expected_version_ok}, got {result['python_version_ok']}"

    def test_setup_environment_checks_required_packages(self):
        """Test that setup checks all required packages."""
        result = setup_environment()

        required_packages = ['numpy', 'psutil', 'jupytext']
        for package in required_packages:
            assert package in result['packages_status'], f"Missing package status for {package}"
            assert result['packages_status'][package] in [
                'installed', 'missing', 'auto_installed'
            ], f"Invalid status for {package}: {result['packages_status'][package]}"

    def test_setup_environment_provides_guidance(self):
        """Test that setup provides helpful guidance."""
        result = setup_environment()

        # Should provide next steps as strings
        for step in result['next_steps']:
            assert isinstance(step, str), "Each next step should be a string"
            assert step.strip(), "Next steps should not be empty strings"


class TestVerifyEnvironmentIndividual:
    """Test verify_environment function individually with enhanced verification."""

    def test_verify_environment_return_structure(self):
        """Test that verify_environment returns the correct structure."""
        result = verify_environment()

        assert isinstance(result, dict), "Should return a dictionary"

        required_keys = [
            'tests_run', 'tests_passed', 'tests_failed', 'problems',
            'detailed_results', 'package_versions', 'system_info',
            'all_systems_go', 'execution_summary'
        ]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

    def test_verify_environment_data_types(self):
        """Test that verify_environment returns correct data types."""
        result = verify_environment()

        assert isinstance(result['tests_run'], list), "tests_run should be list"
        assert isinstance(result['tests_passed'], list), "tests_passed should be list"
        assert isinstance(result['tests_failed'], list), "tests_failed should be list"
        assert isinstance(result['problems'], list), "problems should be list"
        assert isinstance(result['detailed_results'], list), "detailed_results should be list"
        assert isinstance(result['package_versions'], dict), "package_versions should be dict"
        assert isinstance(result['system_info'], dict), "system_info should be dict"
        assert isinstance(result['all_systems_go'], bool), "all_systems_go should be boolean"
        assert isinstance(result['execution_summary'], dict), "execution_summary should be dict"

    def test_verify_environment_runs_comprehensive_tests(self):
        """Test that verify_environment runs all expected tests."""
        result = verify_environment()

        # Enhanced version should run at least 6 comprehensive tests
        assert len(result['tests_run']) >= 6, \
            f"Expected at least 6 tests, got {len(result['tests_run'])}"

        expected_tests = [
            'python_version_command',
            'numpy_functionality_comprehensive',
            'system_resources_comprehensive',
            'development_tools',
            'package_installation_comprehensive',
            'memory_performance_stress'
        ]

        for test in expected_tests:
            assert test in result['tests_run'], f"Missing expected test: {test}"

    def test_verify_environment_command_execution(self):
        """Test that command execution actually works."""
        result = verify_environment()

        # Should have detailed results with command execution info
        assert len(result['detailed_results']) > 0, "Should have detailed test results"

        for detail in result['detailed_results']:
            assert 'test' in detail, "Each detail should have test name"
            assert 'status' in detail, "Each detail should have status"
            assert 'details' in detail, "Each detail should have details"
            assert detail['status'] in ['PASS', 'FAIL', 'TIMEOUT', 'ERROR'], \
                f"Invalid status: {detail['status']}"

    def test_verify_environment_package_versions(self):
        """Test that package versions are captured correctly."""
        result = verify_environment()

        # Should capture versions for available packages
        if 'numpy' in result['package_versions']:
            numpy_version = result['package_versions']['numpy']
            assert isinstance(numpy_version, str), "NumPy version should be string"
            assert '.' in numpy_version, "Version should contain dots"

        if 'psutil' in result['package_versions']:
            psutil_version = result['package_versions']['psutil']
            assert isinstance(psutil_version, str), "psutil version should be string"
            assert '.' in psutil_version, "Version should contain dots"

    def test_verify_environment_system_info(self):
        """Test that system information is captured correctly."""
        result = verify_environment()

        # Should capture system information
        if 'cpu_physical' in result['system_info']:
            assert result['system_info']['cpu_physical'] > 0, "CPU count should be positive"

        if 'memory_total_gb' in result['system_info']:
            assert result['system_info']['memory_total_gb'] > 0, "Memory should be positive"

        if 'python_version' in result['system_info']:
            assert isinstance(result['system_info']['python_version'], str), \
                "Python version should be string"

    def test_verify_environment_execution_summary(self):
        """Test that execution summary is correct."""
        result = verify_environment()

        summary = result['execution_summary']
        assert 'total_tests' in summary, "Should have total_tests count"
        assert 'tests_passed' in summary, "Should have tests_passed count"
        assert 'tests_failed' in summary, "Should have tests_failed count"
        assert 'success_rate' in summary, "Should have success_rate"

        # Validate counts
        assert summary['total_tests'] == len(result['tests_run']), \
            "Total tests should match tests_run length"
        assert summary['tests_passed'] == len(result['tests_passed']), \
            "Passed tests should match tests_passed length"
        assert summary['tests_failed'] == len(result['tests_failed']), \
            "Failed tests should match tests_failed length"

        # Validate success rate
        if summary['total_tests'] > 0:
            expected_rate = summary['tests_passed'] / summary['total_tests']
            assert abs(summary['success_rate'] - expected_rate) < 0.001, \
                "Success rate calculation incorrect"

    @pytest.mark.slow
    def test_verify_environment_performance(self):
        """Test that verify_environment completes in reasonable time."""
        start_time = time.time()
        result = verify_environment()
        end_time = time.time()

        execution_time = end_time - start_time

        # Should complete within 60 seconds (generous for CI environments)
        assert execution_time < 60, \
            f"verify_environment took too long: {execution_time:.2f}s"

        # Should still return valid results
        assert isinstance(result, dict), "Should return valid results even with timeouts"
        assert 'all_systems_go' in result, "Should have overall status"


class TestConfigureAndDisplayIndividual:
    """Test configure_and_display function individually."""

    def test_configure_and_display_return_structure(self):
        """Test that configure_and_display returns the correct structure."""
        result = configure_and_display()

        assert isinstance(result, dict), "Should return a dictionary"

        required_sections = ['developer', 'hardware', 'software', 'capabilities']
        for section in required_sections:
            assert section in result, f"Missing required section: {section}"

    def test_configure_and_display_developer_section(self):
        """Test that developer information is properly configured."""
        result = configure_and_display()

        developer = result['developer']
        assert isinstance(developer, dict), "Developer section should be dict"

        required_fields = ['name', 'email', 'institution', 'system_name', 'system_id', 'version']
        for field in required_fields:
            assert field in developer, f"Missing developer field: {field}"
            assert isinstance(developer[field], str) and developer[field], \
                f"Developer {field} should be non-empty string"

        # Validate specific fields
        assert '@' in developer['email'], "Email should contain @ symbol"
        assert developer['version'] == '1.0.0', "Version should be '1.0.0'"
        assert len(developer['system_id']) == 8, "System ID should be 8 characters"

    def test_configure_and_display_hardware_section(self):
        """Test that hardware information is correctly gathered."""
        result = configure_and_display()

        hardware = result['hardware']
        assert isinstance(hardware, dict), "Hardware section should be dict"

        required_fields = [
            'cpu_count', 'memory_total_gb', 'performance_class',
            'description', 'architecture'
        ]
        for field in required_fields:
            assert field in hardware, f"Missing hardware field: {field}"

        # Validate reasonable values
        assert hardware['cpu_count'] > 0, "CPU count should be positive"
        assert hardware['memory_total_gb'] > 0, "Memory should be positive"
        assert hardware['performance_class'] in [
            'basic', 'standard', 'development_ready', 'high_performance'
        ], f"Invalid performance class: {hardware['performance_class']}"

        # Description should be meaningful
        assert len(hardware['description']) > 10, "Description should be meaningful"
        assert 'cores' in hardware['description'] or 'CPU' in hardware['description'], \
            "Description should mention CPU/cores"

    def test_configure_and_display_software_section(self):
        """Test that software information is correctly gathered."""
        result = configure_and_display()

        software = result['software']
        assert isinstance(software, dict), "Software section should be dict"

        # Should have python and platform subsections
        assert 'python' in software, "Should have python information"
        assert 'platform' in software, "Should have platform information"
        assert 'environment_type' in software, "Should have environment type"

        # Validate Python info
        python_info = software['python']
        assert 'version' in python_info, "Should have Python version"
        assert 'implementation' in python_info, "Should have Python implementation"

        # Validate platform info
        platform_info = software['platform']
        assert 'system' in platform_info, "Should have system information"
        assert 'node' in platform_info, "Should have node information"

    def test_configure_and_display_capabilities_section(self):
        """Test that capabilities are properly assessed."""
        result = configure_and_display()

        capabilities = result['capabilities']
        assert isinstance(capabilities, dict), "Capabilities section should be dict"

        required_fields = [
            'neural_network_ready', 'parallel_processing',
            'development_optimized', 'performance_tier'
        ]
        for field in required_fields:
            assert field in capabilities, f"Missing capability field: {field}"

        # Validate capability logic
        assert isinstance(capabilities['neural_network_ready'], bool), \
            "neural_network_ready should be boolean"
        assert isinstance(capabilities['parallel_processing'], bool), \
            "parallel_processing should be boolean"
        assert isinstance(capabilities['development_optimized'], bool), \
            "development_optimized should be boolean"

        # Performance tier should match hardware performance class
        hardware_class = result['hardware']['performance_class']
        capability_tier = capabilities['performance_tier']
        assert hardware_class == capability_tier, \
            f"Performance tier mismatch: {hardware_class} vs {capability_tier}"


class TestIntegrationWorkflow:
    """Test the complete integration workflow."""

    def test_complete_setup_to_configure_workflow(self):
        """Test complete setup → verify → configure flow."""
        # Execute complete workflow
        setup_result = setup_environment()
        verify_result = verify_environment()
        config_result = configure_and_display()

        # All should return valid dictionaries
        assert isinstance(setup_result, dict), "setup_environment should return dict"
        assert isinstance(verify_result, dict), "verify_environment should return dict"
        assert isinstance(config_result, dict), "configure_and_display should return dict"

        # Workflow should provide comprehensive information
        assert len(setup_result) >= 6, "Setup should provide comprehensive info"
        assert len(verify_result) >= 8, "Verify should provide comprehensive info"
        assert len(config_result) >= 4, "Config should provide comprehensive info"

    def test_module_imports_correctly(self):
        """Test that module imports work correctly."""
        # Should be able to import all functions
        from setup_dev import setup_environment, verify_environment, configure_and_display

        # All should be callable
        assert callable(setup_environment), "setup_environment should be callable"
        assert callable(verify_environment), "verify_environment should be callable"
        assert callable(configure_and_display), "configure_and_display should be callable"

    def test_function_return_types_consistency(self):
        """Test that all functions return consistent data structures."""
        setup_result = setup_environment()
        verify_result = verify_environment()
        config_result = configure_and_display()

        # All should return dictionaries with string keys
        for result_name, result in [
            ('setup', setup_result),
            ('verify', verify_result),
            ('config', config_result)
        ]:
            assert isinstance(result, dict), f"{result_name} should return dict"
            for key in result.keys():
                assert isinstance(key, str), f"{result_name} keys should be strings"

    def test_data_flow_integration(self):
        """Test that data flows correctly between functions."""
        setup_result = setup_environment()
        verify_result = verify_environment()
        config_result = configure_and_display()

        # Environment detection should be consistent
        setup_env = setup_result.get('current_environment')
        config_env = config_result.get('software', {}).get('environment_type')

        if setup_env and config_env:
            assert setup_env == config_env, \
                f"Environment detection mismatch: {setup_env} vs {config_env}"

        # Package information should be consistent
        setup_packages = setup_result.get('packages_status', {})
        verify_packages = verify_result.get('package_versions', {})

        # If a package is installed in setup, it should be versioned in verify
        for package, status in setup_packages.items():
            if status in ['installed', 'auto_installed']:
                # verify_environment should also detect it (may not always get version)
                if package in verify_packages:
                    assert verify_packages[package], f"Package {package} should have version"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_timeout_handling(self):
        """Test that functions handle timeouts gracefully."""
        # Run verify_environment which has subprocess calls with timeouts
        result = verify_environment()

        # Should complete and return results even if some commands timeout
        assert isinstance(result, dict), "Should return dict even with potential timeouts"
        assert 'all_systems_go' in result, "Should have status even with timeouts"

        # Should report timeout status in detailed results if any occurred
        detailed_results = result.get('detailed_results', [])
        timeout_tests = [d for d in detailed_results if d.get('status') == 'TIMEOUT']

        # If timeouts occurred, they should be properly reported
        for timeout_test in timeout_tests:
            assert 'timeout' in timeout_test['details'].lower(), \
                "Timeout tests should mention timeout in details"

    def test_missing_packages_handling(self):
        """Test behavior when packages are missing."""
        # This is harder to test in a real environment, but we can check
        # that the functions handle missing packages gracefully

        setup_result = setup_environment()
        verify_result = verify_environment()

        # Functions should complete even if packages are missing
        assert isinstance(setup_result, dict), "Should handle missing packages"
        assert isinstance(verify_result, dict), "Should handle missing packages"

        # If any packages are missing, should be reported
        missing_packages = [
            pkg for pkg, status in setup_result.get('packages_status', {}).items()
            if status == 'missing'
        ]

        if missing_packages:
            # Should provide guidance
            assert setup_result.get('action_needed', False), \
                "Should indicate action needed when packages missing"
            assert len(setup_result.get('next_steps', [])) > 0, \
                "Should provide next steps when packages missing"

    def test_error_conditions(self):
        """Test that functions handle error conditions gracefully."""
        # All functions should be defensive and not crash
        try:
            setup_result = setup_environment()
            verify_result = verify_environment()
            config_result = configure_and_display()

            # Should all return valid results
            assert setup_result is not None, "setup_environment should not return None"
            assert verify_result is not None, "verify_environment should not return None"
            assert config_result is not None, "configure_and_display should not return None"

        except Exception as e:
            pytest.fail(f"Functions should handle errors gracefully, but got: {e}")


class TestEnhancedPackageVerification:
    """Test the enhanced package verification specifically."""

    def test_command_execution_works(self):
        """Test that commands are actually executed."""
        result = verify_environment()

        # Should include command-based tests
        command_tests = [
            'python_version_command',
            'numpy_functionality_comprehensive',
            'system_resources_comprehensive',
            'package_installation_comprehensive'
        ]

        for test in command_tests:
            assert test in result['tests_run'], f"Missing command test: {test}"

    def test_detailed_results_provided(self):
        """Test that detailed results are returned."""
        result = verify_environment()

        detailed_results = result.get('detailed_results', [])
        assert len(detailed_results) > 0, "Should provide detailed results"

        # Each detailed result should have proper structure
        for detail in detailed_results:
            assert 'test' in detail, "Should have test name"
            assert 'status' in detail, "Should have status"
            assert 'details' in detail, "Should have details"

            # Status should be valid
            assert detail['status'] in ['PASS', 'FAIL', 'TIMEOUT', 'ERROR'], \
                f"Invalid status: {detail['status']}"

            # Details should be informative
            assert len(detail['details']) > 0, "Details should not be empty"

    def test_error_handling_for_failed_commands(self):
        """Test error handling for failed commands."""
        result = verify_environment()

        # Should handle command failures gracefully
        failed_tests = result.get('tests_failed', [])
        problems = result.get('problems', [])

        # If there are failed tests, should have problem descriptions
        if failed_tests:
            assert len(problems) > 0, "Failed tests should generate problem reports"

            # Each problem should be descriptive
            for problem in problems:
                assert isinstance(problem, str), "Problems should be strings"
                assert len(problem) > 0, "Problems should not be empty"


# Performance and stress testing
@pytest.mark.slow
class TestPerformanceAndStress:
    """Test performance characteristics and stress scenarios."""

    def test_memory_performance_stress_execution(self):
        """Test that memory/performance stress test actually runs."""
        result = verify_environment()

        # Should include memory stress test
        assert 'memory_performance_stress' in result['tests_run'], \
            "Should include memory/performance stress test"

        # If it passed, should have performance metrics
        if 'memory_performance_stress' in result['tests_passed']:
            system_info = result.get('system_info', {})
            assert 'performance_metrics' in system_info, \
                "Should have performance metrics when stress test passes"

            metrics = system_info['performance_metrics']
            assert 'stress_test_time_seconds' in metrics, "Should have execution time"
            assert 'peak_memory_usage_mb' in metrics, "Should have memory usage"

    def test_concurrent_execution_safety(self):
        """Test that functions can be called multiple times safely."""
        # Run functions multiple times to test for side effects
        results = []
        for i in range(3):
            setup_result = setup_environment()
            verify_result = verify_environment()
            config_result = configure_and_display()

            results.append((setup_result, verify_result, config_result))

        # Results should be consistent across runs
        for i in range(1, 3):
            prev_setup, prev_verify, prev_config = results[i-1]
            curr_setup, curr_verify, curr_config = results[i]

            # Basic structure should be consistent
            assert set(prev_setup.keys()) == set(curr_setup.keys()), \
                "setup_environment structure should be consistent"
            assert set(prev_config.keys()) == set(curr_config.keys()), \
                "configure_and_display structure should be consistent"


# Test fixtures and utilities
@pytest.fixture
def clean_test_environment():
    """Fixture to ensure clean test environment."""
    # Setup - nothing needed for these tests
    yield
    # Teardown - nothing needed for these tests
    pass


@pytest.fixture
def mock_subprocess_timeout():
    """Fixture to mock subprocess timeouts for testing."""
    with patch('subprocess.run') as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired('test', 10)
        yield mock_run


@pytest.fixture
def mock_subprocess_failure():
    """Fixture to mock subprocess failures for testing."""
    with patch('subprocess.run') as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Mocked error"
        mock_run.return_value = mock_result
        yield mock_run


# Main comprehensive test
def test_module_01_setup_comprehensive():
    """Main comprehensive test that validates all Module 1 Setup functionality."""
    print("🔬 Running comprehensive Module 1 Setup integration tests...")

    # Test all three main functions
    setup_result = setup_environment()
    verify_result = verify_environment()
    config_result = configure_and_display()

    # Basic validation that functions work
    assert setup_result is not None, "setup_environment should return results"
    assert verify_result is not None, "verify_environment should return results"
    assert config_result is not None, "configure_and_display should return results"

    # Validate enhanced functionality
    assert len(verify_result.get('tests_run', [])) >= 6, \
        "Should run comprehensive test suite"
    assert 'execution_summary' in verify_result, \
        "Should provide execution summary"
    assert 'detailed_results' in verify_result, \
        "Should provide detailed test results"

    # Report summary
    summary = verify_result.get('execution_summary', {})
    tests_passed = summary.get('tests_passed', 0)
    tests_total = summary.get('total_tests', 0)
    success_rate = summary.get('success_rate', 0)

    performance_class = config_result.get('hardware', {}).get('performance_class', 'unknown')

    print(f"✅ All functions executed successfully")
    print(f"✅ Environment verification: {tests_passed}/{tests_total} tests passing ({success_rate*100:.1f}%)")
    print(f"✅ System performance class: {performance_class}")
    print(f"✅ Enhanced command execution verification: ACTIVE")
    print(f"✅ Comprehensive integration testing: PASSED")
    print(f"🚀 Module 1 Setup comprehensive integration test completed!")

    return {
        'setup_result': setup_result,
        'verify_result': verify_result,
        'config_result': config_result,
        'comprehensive_test_passed': True,
        'enhanced_verification_active': True,
        'tests_summary': summary
    }


if __name__ == "__main__":
    # Run comprehensive test when executed directly
    pytest.main([__file__, "-v", "--tb=short", "--durations=10"])
