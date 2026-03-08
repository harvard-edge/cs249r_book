# mlsysim: Engineering & Modeling Best Practices

To ensure `mlsysim` remains a reliable pedagogical and research tool, all contributions must adhere to these four core pillars.

---

## 1. The "Units-First" Mandate
**Rule:** No naked floats in physics or economic equations.
*   **Requirement:** All physical quantities (Latency, Throughput, Memory, Power) must be wrapped in a `pint.Quantity`.
*   **Reasoning:** This enables automatic dimensional analysis. If a formula for "Time" results in "Bytes", the simulator will raise a `DimensionalityError` rather than providing a hallucinated result.

## 2. Citable Constants
**Rule:** Every constant must be traceable to a primary source.
*   **Requirement:** Every entry in `constants.py` or the registries must include a comment citing a datasheet, peer-reviewed paper (e.g., Jouppi et al. 2023), or industry log (e.g., Meta Llama-3 logs).
*   **Reasoning:** Prevents "magic numbers" and ensures students can verify the "Ground Truth" themselves.

## 3. Strict Type Safety (Pydantic)
**Rule:** All layers must use Pydantic `BaseModel` for data integrity.
*   **Requirement:** Use `mlsysim.core.types.Quantity` for all validated fields.
*   **Reasoning:** This allows for robust configuration validation. If a student tries to set `fleet_size: "lots"`, the simulator will provide a clear, actionable validation error before execution.

## 4. Analytical Determinism
**Rule:** Prefer closed-form physics over stochastic simulation.
*   **Requirement:** The core engine should use established systems laws (Iron Law, Amdahl's Law, Roofline, Young-Daly). Avoid random noise unless modeling specific failure distributions (e.g., MTBF).
*   **Reasoning:** Autograders and textbook worked examples require exact, reproducible results across different machines (Mac vs. Linux vs. Browser).

## 5. Progressive Lowering
**Rule:** Maintain separation between Workload (What) and Hardware (Where).
*   **Requirement:** High-level models (Layer A) should not know about specific hardware quirks. The **Solver (Layer E)** is the only place where Workloads are "lowered" onto Hardware.
*   **Reasoning:** This allows researchers to test the *same* workload across *unbuilt* hardware architectures just by swapping the Hardware Node.

## 6. Source Transparency (The Provenance Anchor)
**Rule:** Every constant must have a "clickable" primary source.
*   **Requirement:** Registry entries for Hardware, Grid Intensity, and Pricing must include a `metadata` field with a URL to the official source (e.g., IEA Report, NVIDIA Whitepaper, AWS Price List).
*   **Reasoning:** This ensures the simulator acts as an "Audit Trail" for students, allowing them to verify the physics and economics against publicly available information.
