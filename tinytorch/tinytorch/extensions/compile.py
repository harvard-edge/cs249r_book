class TracedNode:
    def __init__(self, name):
        self.name = name

    def __add__(self, other):
        other_name = other.name if isinstance(other, TracedNode) else str(other)
        return TracedNode(f"({self.name} + {other_name})")

    def __radd__(self, other):
        other_name = other.name if isinstance(other, TracedNode) else str(other)
        return TracedNode(f"({other_name} + {self.name})")

    def __mul__(self, other):
        other_name = other.name if isinstance(other, TracedNode) else str(other)
        return TracedNode(f"({self.name} * {other_name})")

    def __rmul__(self, other):
        other_name = other.name if isinstance(other, TracedNode) else str(other)
        return TracedNode(f"({other_name} * {self.name})")

def compile_graph(func, *input_names):
    """
    Compiles a PyTorch-like define-by-run function into a static fused graph.
    """
    # 1. Tracing phase: pass dummy nodes through the function
    traced_inputs = [TracedNode(name) for name in input_names]
    output = func(*traced_inputs)
    
    # 2. Code generation phase: fuse into a single fast loop
    code = f"def fused_kernel({', '.join(input_names)}):\n"
    code += f"    # Fused by TinyTorch Compiler\n"
    code += f"    return {output.name}\n"
    return code
