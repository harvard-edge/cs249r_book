# MLSys·im Output & Presentation Best Practices

When designing the data output for a professional CLI/Framework, the goal is to make the results as easy to consume, parse, and reason about as possible, without scattering business logic across the codebase.

### 1. The "Data Envelope" Pattern (Unified Payloads)
**Problem:** A user runs an optimization and gets a single `objective_value`. They have no context about the system state that generated that value.
**Practice:** Every core operation must return a comprehensive "Envelope" object (e.g., `SystemEvaluation` or `OptimizerResult`) that contains the final answer *plus* the contextual metrics (latency, throughput, footprint, bottlenecks). This means the user gets everything in one place, predictably.

### 2. Decouple Orchestration from Presentation (The Dumb CLI)
**Problem:** The CLI file (`commands/eval.py`) manually runs `SingleNodeModel`, then runs `EconomicsModel`, and stitches them together into a `SystemEvaluation` dictionary. This means the CLI is doing business orchestration, making it hard for Python API users to get the same unified scorecard without rewriting the code.
**Practice:** The CLI should be a "dumb pipe." It parses the YAML, passes the raw configuration objects to an `Orchestrator` inside the `core/` package, receives the `SystemEvaluation` object, and simply passes it to `render_scorecard`.

### 3. Metric Flattening & Formatting is strictly for Renderers
**Problem:** Storing pre-formatted strings (`"4.8 ms"`) inside the core data objects prevents downstream scripts from doing math on the outputs.
**Practice:** The core engine outputs raw, typed data (`float` or `Quantity`). Only the presentation layer (`renderers.py`) formats these numbers with SI units, colors, and commas. If a script requests `--output json`, it receives the unformatted floats.

### 4. Deterministic SLA Assertions
**Problem:** A user has to parse JSON to find out if latency is under 50ms.
**Practice:** SLA assertions are evaluated mathematically by the core or the CLI logic, triggering a semantic `Exit Code 3` (SLA Failure). The output object explicitly lists which assertions passed and which failed.

---
*These practices ensure that whether a user consumes the data via Python API, bash JSON parsing, or human terminal UI, the data is complete, trustworthy, and consistently formatted.*