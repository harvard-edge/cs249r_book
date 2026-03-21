# The mlsysim Optimization Engine

`mlsysim` bridges the gap between pedagogical simplicity and production-grade Operations Research (OR).

While many academic tools rely on hardcoded Python `for` loops to find optimal configurations, `mlsysim` implements a strict **Solver Protocol**. This allows the engine to route mathematical problems to the correct industrial backend (Google OR-Tools, SciPy) without exposing the complexity to the textbook reader.

## Why this Architecture?

In production ML infrastructure, finding the optimal deployment strategy is rarely a simple grid search.
* Finding the optimal batch size against a latency SLA requires **continuous gradient descent** down a queueing theory curve.
* Routing 10,000 heterogeneous workloads across global datacenters based on variable carbon intensity requires **linear programming**.
* Designing a custom ASICs memory hierarchy requires **mixed-integer constraint satisfaction**.

By decoupling the *Physics* (the Models) from the *Search* (the Optimizers), `mlsysim` serves as both a pristine educational tool for understanding the bounds of computation, and a highly extensible framework for systems researchers building the next generation of ML infrastructure.
