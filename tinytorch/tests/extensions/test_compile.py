from tinytorch.extensions.compile import compile_graph

def test_graph_compilation_trace():
    def simple_net(x, w, b):
        return (x * w) + b

    code = compile_graph(simple_net, "x", "w", "b")

    assert "def fused_kernel(x, w, b):" in code
    assert "(x * w) + b" in code or "((x * w) + b)" in code


def test_graph_compilation_with_scalars():
    """Verify tracing handles scalar literals without crashing on missing .name attribute."""
    def scaled_offset_net(x):
        # Forward and reverse scalar operations
        return (2.0 * x + 1.0) * 3

    code = compile_graph(scaled_offset_net, "x")

    assert "def fused_kernel(x):" in code
    assert "2.0" in code
    assert "1.0" in code
    assert "3" in code

