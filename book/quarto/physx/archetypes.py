"""
archetypes.py
Workload archetypes (constraint regimes) used across volumes.
"""

ARCHETYPES = {
    "compute_beast": {
        "label": "Compute Beast",
        "constraint": "Compute throughput",
        "notes": "High arithmetic intensity; dense compute-bound.",
    },
    "bandwidth_hog": {
        "label": "Bandwidth Hog",
        "constraint": "Memory bandwidth",
        "notes": "Autoregressive or low reuse; memory-bound.",
    },
    "sparse_scatter": {
        "label": "Sparse Scatter",
        "constraint": "Memory capacity/latency",
        "notes": "Irregular access patterns, large embeddings.",
    },
    "tiny_constraint": {
        "label": "Tiny Constraint",
        "constraint": "Energy/power",
        "notes": "Always-on, power-limited edge devices.",
    },
}
