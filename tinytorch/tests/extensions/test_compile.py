from tinytorch.extensions.compile import compile_graph

def test_graph_compilation_trace():
    def simple_net(x, w, b):
        return (x * w) + b
        
    code = compile_graph(simple_net, "x", "w", "b")
    
    assert "def fused_kernel(x, w, b):" in code
    assert "(x * w) + b" in code or "((x * w) + b)" in code
