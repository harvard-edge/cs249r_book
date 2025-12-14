"""
E2E Test Configuration

Registers pytest markers for categorizing tests by speed and purpose.
"""

import pytest


def pytest_configure(config):
    """Register custom markers for E2E tests."""
    config.addinivalue_line("markers", "quick: Quick verification tests (~30s total)")
    config.addinivalue_line("markers", "module_flow: Module workflow tests (~2min)")
    config.addinivalue_line("markers", "milestone_flow: Milestone workflow tests")
    config.addinivalue_line("markers", "full_journey: Complete journey tests (~10min)")
    config.addinivalue_line("markers", "slow: Slow tests that train models")
    config.addinivalue_line("markers", "release: Release validation tests")
