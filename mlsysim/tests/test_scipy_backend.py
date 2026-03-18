import pytest

pytest.importorskip("scipy", reason="scipy not installed (optional dependency)")
from mlsysim.core.optimization.scipy_backend import ScipyBackend

def test_scipy_continuous_optimization():
    """
    Test that the SciPy backend can analytically find the optimal batch size 
    to maximize throughput (a parabolic function) without resorting to python for-loops.
    """
    # A simple mock analytical Roofline equation:
    # Throughput increases with batch size but degrades after batch=64 due to memory bounds.
    # We want to MINIMIZE the negative throughput (which maximizes throughput)
    def mock_analytical_roofline(batch_size: float) -> float:
        # e.g., T = -( - (b - 64)^2 + 4096 )
        return (batch_size - 64.0)**2 - 4096.0

    backend = ScipyBackend()
    backend.compile(
        objective_fn=mock_analytical_roofline,
        bounds=(1.0, 128.0)
    )
    
    result = backend.solve()
    
    assert result.feasible is True
    # The minimum of (x-64)^2 is exactly at x=64
    assert result.best_configuration["optimal_variable"] == pytest.approx(64.0, abs=1e-3)
    assert result.optimal_value == pytest.approx(-4096.0, abs=1e-3)
    assert "scipy.optimize" in result.solver_name

def test_scipy_with_constraints():
    """
    Test that the backend correctly routes to SLSQP when strict SLA latency walls are applied.
    """
    def objective_throughput(b: float) -> float:
        return -b  # We want to maximize batch size

    # Latency grows linearly: L = 10 * b. We have a strict SLA of 500ms.
    # SciPy requires constraint functions to be >= 0.
    # So: 500 - (10 * b) >= 0
    constraint_dict = {
        'type': 'ineq',
        'fun': lambda b: 500.0 - (10.0 * b)
    }

    backend = ScipyBackend()
    backend.compile(
        objective_fn=objective_throughput,
        bounds=(1.0, 100.0),
        constraints=[constraint_dict]
    )
    
    result = backend.solve()
    
    assert result.feasible is True
    assert "SLSQP" in result.solver_name
    # The max batch size before hitting the 500ms wall is 50.
    assert result.best_configuration["optimal_variable"] == pytest.approx(50.0, abs=1e-3)
