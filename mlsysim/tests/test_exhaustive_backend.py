import pytest
from mlsysim.core.optimization.registry import OptimizationRegistry

def test_exhaustive_backend():
    backend = OptimizationRegistry.get_backend("exhaustive")
    
    # A function that hits a "wall" at x=42 and goes to infinity
    def objective(x_array):
        x = x_array[0]
        if x < 42:
            return 1e12 # Infinite queue
        return x # We want to minimize x, but it must be >= 42
        
    backend.compile(objective_fn=objective, ranges=[(1, 100)], grid_size=100)
    res = backend.solve()
    
    assert res.feasible is True
    assert res.best_configuration["optimal_variable"] == 42.0
    assert "exhaustive_grid" in res.solver_name


def test_exhaustive_backend_multidimensional_configuration():
    backend = OptimizationRegistry.get_backend("exhaustive")

    def objective(x_array):
        x, y = x_array
        return (x - 0.0) ** 2 + (y - 1.0) ** 2

    backend.compile(objective_fn=objective, ranges=[(0, 1), (0, 1)], grid_size=2)
    res = backend.solve()

    assert res.feasible is True
    assert res.best_configuration["optimal_variables"] == [0.0, 1.0]
    assert res.best_configuration["optimal_variable"] == 0.0
