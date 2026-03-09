"""The 17 ML Systems Walls — canonical taxonomy.

This module is the single source of truth for the wall classification
used throughout mlsysim, the paper, and the textbook.  Every wall
represents a physical or logical constraint that bounds system
performance; each is resolved by a dedicated solver.

The walls are organized into five domains that progress from local
hardware resources to global societal impact:

    Domain 1 — Node      : What a single accelerator can achieve in isolation.
    Domain 2 — Data      : How data moves to and through the accelerator.
    Domain 3 — Algorithm : How much computation the model requires.
    Domain 4 — Fleet     : Consequences of distributed, multi-node operation.
    Domain 5 — Analysis  : Cross-cutting diagnostic and synthesis tools.

Reference
---------
Janapa Reddi et al. (2025), "MLSYSIM: A Composable Analytical Framework
for Machine Learning Systems."  Table 1 and Section 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Type


# ── Domain Enum ───────────────────────────────────────────────────

class Domain(Enum):
    """The five domains of the ML Systems Wall taxonomy.

    Each domain groups walls by the *scope* of the constraint:
    single node → data movement → algorithm → fleet → cross-cutting.
    """
    NODE      = "node"        # Domain 1: Single-accelerator resource ceilings
    DATA      = "data"        # Domain 2: Data movement and pipelines
    ALGORITHM = "algorithm"   # Domain 3: Algorithmic and scaling laws
    FLEET     = "fleet"       # Domain 4: Multi-node, fleet-scale, societal
    ANALYSIS  = "analysis"    # Domain 5: Cross-cutting diagnostic tools


# ── Wall Dataclass ────────────────────────────────────────────────

@dataclass(frozen=True)
class Wall:
    """A single ML Systems Wall.

    Attributes
    ----------
    number : int
        Canonical wall number (1–17).
    name : str
        Short human-readable name (e.g., "Compute", "Memory").
    domain : Domain
        Which domain this wall belongs to.
    solver_name : str
        The solver class that resolves this wall.
    constraint : str
        One-line description of the physical/logical constraint.
    equation : str
        Core equation in plain text (for display / docs).
    sources : list[str]
        Key literature references.
    """
    number: int
    name: str
    domain: Domain
    solver_name: str
    constraint: str
    equation: str
    sources: List[str] = field(default_factory=list)


# ── The 17 Walls ─────────────────────────────────────────────────
# Numbering and domain assignment follow the paper (Table 1).

# Domain 1: Node — single-accelerator resource ceilings
COMPUTE = Wall(
    number=1,
    name="Compute",
    domain=Domain.NODE,
    solver_name="SingleNodeSolver",
    constraint="Peak FLOPS ceiling of a single accelerator.",
    equation="T = OPs / (Peak × η)",
    sources=["Williams et al. (2009), Roofline"],
)

MEMORY = Wall(
    number=2,
    name="Memory",
    domain=Domain.NODE,
    solver_name="SingleNodeSolver",
    constraint="HBM capacity and bandwidth ceilings.",
    equation="T = |W| / BW_HBM",
    sources=["Williams et al. (2009), Roofline"],
)

SOFTWARE = Wall(
    number=3,
    name="Software",
    domain=Domain.NODE,
    solver_name="EfficiencySolver",
    constraint="Gap between peak and achieved FLOPS (kernel fusion, occupancy).",
    equation="η (MFU adjustment factor)",
    sources=[
        "Chowdhery et al. (2022), PaLM",
        "Dao et al. (2022), FlashAttention",
    ],
)

# Domain 2: Data — data movement and pipelines
INGESTION = Wall(
    number=4,
    name="Ingestion",
    domain=Domain.DATA,
    solver_name="DataSolver",
    constraint="Storage I/O must supply data at the rate the accelerator consumes it.",
    equation="ρ = BW_demand / BW_supply",
    sources=["Mohan et al. (2022), Data Bottlenecks"],
)

TRANSFORMATION = Wall(
    number=5,
    name="Transformation",
    domain=Domain.DATA,
    solver_name="TransformationSolver",
    constraint="CPU preprocessing (decode, tokenize, augment) cannot keep pace.",
    equation="T = B · S / C_throughput",
    sources=["Murray et al. (2021), tf.data"],
)

LOCALITY = Wall(
    number=6,
    name="Locality",
    domain=Domain.DATA,
    solver_name="TopologySolver",
    constraint="Network topology limits bisection bandwidth between nodes.",
    equation="BW_eff = BW_link · β / oversubscription",
    sources=[
        "Leiserson (1985), Fat-Trees",
        "Dally & Towles (2003), Interconnection Networks",
    ],
)

# Domain 3: Algorithm — scaling laws and compression
COMPLEXITY = Wall(
    number=7,
    name="Complexity",
    domain=Domain.ALGORITHM,
    solver_name="ScalingSolver",
    constraint="Chinchilla scaling laws govern compute-optimal training.",
    equation="C = 6PD; P* = √(C/120)",
    sources=["Hoffmann et al. (2022), Chinchilla"],
)

REASONING = Wall(
    number=8,
    name="Reasoning",
    domain=Domain.ALGORITHM,
    solver_name="InferenceScalingSolver",
    constraint="Inference-time compute scales with reasoning steps K.",
    equation="T = K × T_step",
    sources=[
        "Wei et al. (2022), Chain-of-Thought",
        "Snell et al. (2024), Test-Time Compute",
    ],
)

FIDELITY = Wall(
    number=9,
    name="Fidelity",
    domain=Domain.ALGORITHM,
    solver_name="CompressionSolver",
    constraint="Compression trades model fidelity for efficiency.",
    equation="r = 32/b (quantization); r = 1/(1-s) (pruning)",
    sources=[
        "Han et al. (2015), Deep Compression",
        "Gholami et al. (2021), Quantization Survey",
    ],
)

# Domain 4: Fleet — multi-node and societal
COMMUNICATION = Wall(
    number=10,
    name="Communication",
    domain=Domain.FLEET,
    solver_name="DistributedSolver",
    constraint="Distributed training requires synchronization across N nodes.",
    equation="T = 2(N-1)/N · M/β + 2(N-1)α",
    sources=[
        "Shoeybi et al. (2019), Megatron-LM",
        "Patarasuk & Mueller (2009), Ring AllReduce",
    ],
)

FRAGILITY = Wall(
    number=11,
    name="Fragility",
    domain=Domain.FLEET,
    solver_name="ReliabilitySolver",
    constraint="Component failures are inevitable at scale.",
    equation="MTBF_cluster = MTBF_node / N",
    sources=[
        "Young (1974), Checkpoint Interval",
        "Daly (2006), Higher-Order Estimate",
    ],
)

MULTI_TENANT = Wall(
    number=12,
    name="Multi-tenant",
    domain=Domain.FLEET,
    solver_name="OrchestrationSolver",
    constraint="Shared clusters introduce queueing delays.",
    equation="T_wait = ρ / [2μ(1-ρ)]",
    sources=["Little (1961), L = λW"],
)

CAPITAL = Wall(
    number=13,
    name="Capital",
    domain=Domain.FLEET,
    solver_name="EconomicsSolver",
    constraint="Total cost of ownership bounds what is economically feasible.",
    equation="TCO = CapEx + OpEx",
    sources=["Barroso et al. (2018), Datacenter as a Computer"],
)

SUSTAINABILITY = Wall(
    number=14,
    name="Sustainability",
    domain=Domain.FLEET,
    solver_name="SustainabilitySolver",
    constraint="Energy consumption converts to carbon and water footprint.",
    equation="CO₂ = E × PUE × CI",
    sources=["Patterson et al. (2022), Carbon Emissions"],
)

ETHICS = Wall(
    number=15,
    name="Ethics",
    domain=Domain.FLEET,
    solver_name="ResponsibleEngineeringSolver",
    constraint="Privacy and fairness guarantees impose computational overhead.",
    equation="σ ∝ 1/ε (DP-SGD slowdown)",
    sources=["Abadi et al. (2016), DP-SGD"],
)

# Domain 5: Analysis — cross-cutting diagnostics
SENSITIVITY = Wall(
    number=16,
    name="Sensitivity",
    domain=Domain.ANALYSIS,
    solver_name="SensitivitySolver",
    constraint="Identifies the binding constraint via numerical partial derivatives.",
    equation="∂T/∂xᵢ (binding constraint)",
    sources=["Williams et al. (2009), Roofline"],
)

SYNTHESIS = Wall(
    number=17,
    name="Synthesis",
    domain=Domain.ANALYSIS,
    solver_name="SynthesisSolver",
    constraint="Inverse Roofline: derive hardware specs from an SLA target.",
    equation="BW_req = |W| / T_target",
    sources=["Williams et al. (2009), Roofline"],
)

# Also expose ServingSolver's wall — it resolves the two-phase
# inference lifecycle (prefill vs decode), which is a sub-wall of
# the Memory Wall (#2) specific to autoregressive models.
SERVING = Wall(
    number=2,  # Sub-wall of Memory
    name="Serving",
    domain=Domain.NODE,
    solver_name="ServingSolver",
    constraint="LLM inference has two distinct regimes: compute-bound prefill and memory-bound decode.",
    equation="TTFT = OPs_prefill / Peak; ITL = |W| / BW_HBM",
    sources=[
        "Pope et al. (2023), LLM Inference",
        "Yu et al. (2022), ORCA",
    ],
)


# ── Wall Registry ────────────────────────────────────────────────

ALL_WALLS = [
    COMPUTE, MEMORY, SOFTWARE,
    INGESTION, TRANSFORMATION, LOCALITY,
    COMPLEXITY, REASONING, FIDELITY,
    COMMUNICATION, FRAGILITY, MULTI_TENANT,
    CAPITAL, SUSTAINABILITY, ETHICS,
    SENSITIVITY, SYNTHESIS,
]

# Lookup helpers
_BY_NUMBER = {w.number: w for w in ALL_WALLS}
_BY_NAME = {w.name.lower(): w for w in ALL_WALLS}
_BY_SOLVER = {}
for w in ALL_WALLS + [SERVING]:
    _BY_SOLVER.setdefault(w.solver_name, []).append(w)


def wall(number: int) -> Wall:
    """Look up a wall by its canonical number."""
    if number not in _BY_NUMBER:
        raise KeyError(f"No wall with number {number}. Valid: 1–17.")
    return _BY_NUMBER[number]


def walls_for_solver(solver_name: str) -> List[Wall]:
    """Return all walls resolved by a given solver class."""
    return _BY_SOLVER.get(solver_name, [])


def walls_in_domain(domain: Domain) -> List[Wall]:
    """Return all walls in a given domain, ordered by number."""
    return sorted([w for w in ALL_WALLS if w.domain == domain],
                  key=lambda w: w.number)


def taxonomy() -> str:
    """Return a human-readable summary of the full wall taxonomy.

    This is the pedagogical entry point — print this in a notebook
    and students see the entire analytical framework at a glance.
    """
    lines = ["═══ The 17 ML Systems Walls ═══", ""]
    for domain in Domain:
        domain_walls = walls_in_domain(domain)
        label = {
            Domain.NODE:      "Domain 1 — Node (Single-Accelerator Resources)",
            Domain.DATA:      "Domain 2 — Data (Movement & Pipelines)",
            Domain.ALGORITHM: "Domain 3 — Algorithm (Scaling & Compression)",
            Domain.FLEET:     "Domain 4 — Fleet (Multi-Node & Society)",
            Domain.ANALYSIS:  "Domain 5 — Analysis (Cross-Cutting Diagnostics)",
        }[domain]
        lines.append(f"  {label}")
        for w in domain_walls:
            lines.append(f"    Wall {w.number:2d}: {w.name:<16s} → {w.solver_name}")
            lines.append(f"             {w.constraint}")
        lines.append("")
    return "\n".join(lines)
