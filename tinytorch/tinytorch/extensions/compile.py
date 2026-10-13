# Hand-maintained extension, NOT generated. This file is not produced by
# `tito dev export` or `tito module complete`; it is tracked in git and
# edited directly here. It is the graph-capture example of the compiler
# extension point; no src/ module or student notebook
# exports to it.
"""Graph Capture for TinyTorch: an example extension.

What it is: the first half of a compiler. It captures a define-by-run
function as a single expression, and stops before the half that makes it
faster. Hand-written and tracked in git as-is, not generated from src/.

Why it exists: in eager execution every operation is a separate call that
reads its inputs from memory and writes a full intermediate back. A compiler
first needs the whole computation in one place to see that, for example,
(x + bias) * scale can be one loop that never stores x + bias. Tracing is how
it gets there: run the function once on placeholder objects that record what
is done to them instead of computing anything.

How it is triggered: you call compile_graph once, on a function, and get
source code back. Tracing happens during that call; your data is never
involved.

    from tinytorch.extensions import compile_graph
    def f(x, bias, scale):
        return (x + bias) * scale
    src = compile_graph(f, "x", "bias", "scale")
    print(src)               # def fused_kernel(x, bias, scale): ...
    namespace = {}
    exec(src, namespace)     # turn the source into a callable
    y = namespace["fused_kernel"](x, bias, scale)

    On the compile_graph call: f runs once, on TracedNode placeholders.
    On every call of the result: ordinary Python evaluates the expression.

How it works:

1. compile_graph passes one TracedNode per input name through the function.
2. TracedNode overloads + and * (including the reflected forms, so 2.0 * x
   works), and each operation returns a new TracedNode whose name is the
   expression so far. The final node's name is the whole computation.
3. It returns Python source for a function that evaluates that expression.

What it leaves out: fusion itself. It only understands + and *, it returns
source text rather than a callable, and running that source with NumPy still
evaluates one operation at a time, so x + bias is still materialized.

Where to go next: write the second half, a code generator that emits one loop
over the elements. The hand-fused tinytorch_cpp_fused_bias_gelu kernel in
cpp_simd_gemm.cpp is what such a loop looks like.
"""

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
    Traces a define-by-run function into one expression and returns it as
    Python source for a function of the named inputs.
    """
    # 1. Tracing phase: pass placeholder nodes through the function
    traced_inputs = [TracedNode(name) for name in input_names]
    output = func(*traced_inputs)

    # 2. Code generation phase: emit the traced expression as one function
    # (it still runs one operation at a time; a fusing backend would emit a loop)
    code = f"def fused_kernel({', '.join(input_names)}):\n"
    code += f"    # Traced by TinyTorch compile_graph\n"
    code += f"    return {output.name}\n"
    return code
