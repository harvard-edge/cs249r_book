import subprocess
import sys
import os

def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "mlperf_edu.cli", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "."}
    )
    assert result.returncode == 0
    assert "MLPerf EDU" in result.stdout

def test_cli_run_cloud_help():
    result = subprocess.run(
        [sys.executable, "-m", "mlperf_edu.cli", "run", "cloud", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "."}
    )
    assert result.returncode == 0
    assert "--task" in result.stdout
    assert "llm-inference" in result.stdout

def test_cli_run_cloud_inference_offline():
    # Run a very small test to keep it fast
    # We can't easily pass arguments to change total_samples from CLI yet without more changes
    # but we can verify it runs.
    result = subprocess.run(
        [sys.executable, "-m", "mlperf_edu.cli", "run", "cloud", "--task", "llm-inference", "--scenario", "Offline"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "."}
    )
    assert result.returncode == 0
    assert "MLPerf EDU Cloud Benchmark Report" in result.stdout
    assert "Scenario: Offline" in result.stdout
