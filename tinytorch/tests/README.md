# TinyTorch Test Suite

Tests are organized by scope: what one module does on its own, what modules do
together, and what the whole system does end to end.

## Layout

| Directory | Scope | What belongs here |
|---|---|---|
| `NN_modulename/` | One module | Behavior of a single module's exported code |
| `integration/` | Several modules | The seams between modules, especially gradient flow |
| `regression/` | One past bug | A test named for the defect it prevents returning |
| `e2e/` | The student's path | The journey a learner takes through the course |
| `milestones/` | Historical results | The six milestone scripts still run and still learn |
| `cli/` | The `tito` tool | Command registry, help text, and student workflows |
| `environment/` | The machine | Required interpreter, packages, and project layout |

Every module also carries its own unit tests inside its source file. Those run
when a student executes the notebook. The suite here tests the exported package.

## Running

```bash
pytest tests/                      # everything
pytest tests/06_autograd/          # one module
pytest tests/integration/          # the cross-module seams
pytest tests/ --tinytorch          # educational output, see below
```

Educational mode prints each test with a pass or fail marker, and on failure
shows the WHAT and WHY from the test's docstring so the message teaches rather
than just reports. Nineteen `*_core.py` files carry that docstring format:

```python
def test_tensor_addition(self):
    """
    WHAT: Element-wise tensor addition.

    WHY: Addition is used everywhere in neural networks:
    - Adding bias to layer output: y = Wx + b
    - Residual connections: output = layer(x) + x

    STUDENT LEARNING: Operations return new Tensors (functional style).
    """
```

## Where a new test goes

- Does it exercise one module's exported code? Put it in `NN_modulename/`.
- Does it cross a module boundary? Put it in `integration/`.
- Does it pin a bug you just fixed? Put it in `regression/`, named for the bug.
- Does it check the tool or the machine rather than the framework? Put it in
  `cli/` or `environment/`.

## What a test has to do

A test earns its place by being able to fail. Three rules follow from that.

**Assert the value, not the shape.** A gradient with the right shape and the
wrong number trains a model that quietly learns the wrong thing. Shape checks
pass for a transposed matmul gradient and for a subtraction that lost its sign.

**Never swallow the failure.** No bare `except`, no handler whose body is a
`pass` or a `print`, and no test that reports trouble by printing a warning. If
a condition is not required, it is not a test. Two release gates enforce this.

**Prove it can fail.** Break the implementation, watch the test go red, restore
it, watch it go green. A test that has never been seen to fail is a guess. This
suite has caught real defects that way, including a slice gradient that was
discarded silently while 379 tests stayed green.

## The tests that matter most

`integration/test_integration_gradient_flow.py` is the one to watch. If
gradients stop flowing through the stack, training is broken no matter what the
unit tests say.

`milestones/` is the other. Those scripts reproduce historical results with the
student's own code, so they fail when the framework is subtly wrong in a way a
unit test cannot see.
