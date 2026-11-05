# Named scenarios

This package supplies the named input configurations exposed by `Scenarios`.
It is organized by modeling subject, independently of textbook volume. Topic
modules keep related examples together: for example, `serving/kv_cache.py`
contains cache-management cases, while `robotics/stopping.py` contains physical
stopping cases. The corresponding input classes live in `scenario_definitions`;
solvers and physics functions own their calculations.

Each configuration is constructed once. A topic module explicitly imports any
previously defined configuration it builds on. These imports must form an
acyclic graph; the compatibility test checks this. There is no lazy registry,
dynamic loader, or factory layer. A cross-topic comparison imports the existing
object and uses its documented `variant` method.

`engine/scenarios.py` is the stable public facade. Its explicit class bindings
preserve the historical registry order and object identity. Add a named case in
the appropriate topic module, then bind it in that facade. `Scenarios.list()`
continues to use the shared registry's alphabetical traversal and identity
based alias deduplication. Existing imports and textbook references remain
valid.

A few shared reference cases remain under `base/applications.py`: these compose
a workload, a system, and evaluation constraints rather than one narrow
calculation. `foundations.py` modules contain initial reference cases for their
own domain; later topics import those cases rather than repeating inputs.
