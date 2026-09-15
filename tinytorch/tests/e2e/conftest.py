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


@pytest.fixture(autouse=True)
def isolated_journey_project(request, tmp_path, monkeypatch):
    """Never run student workflows against the checkout or personal credentials."""
    import shutil

    source = getattr(request.module, "PROJECT_ROOT", None)
    if source is None:
        return
    project = tmp_path / "project"
    project.mkdir()
    for name in ("bin", "tito", "src", "tinytorch", "modules", "tests",
                 "milestones", "tools", "datasets", "etc"):
        if (source / name).exists():
            shutil.copytree(source / name, project / name,
                            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    for name in ("pyproject.toml", "requirements.txt", "settings.ini", "settings.json", ".tinyrc"):
        if (source / name).exists():
            shutil.copy2(source / name, project / name)
    monkeypatch.setattr(request.module, "PROJECT_ROOT", project)
    monkeypatch.setenv("TINYTORCH_CREDENTIALS_DIR", str(tmp_path / "credentials"))
    monkeypatch.setenv("CI", "1")
    monkeypatch.setenv("PYTHONPATH", str(project))
