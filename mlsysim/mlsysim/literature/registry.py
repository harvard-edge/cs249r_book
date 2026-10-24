"""Published literature anchors used by MLSysIM (MFU, Chinchilla, benchmarks, ...).

Each anchor is a provenance-carrying scalar; the values live as YAML under
``literature/data/<category>.yaml`` and are loaded via ``load_sourced_registry``
(provenance referenced by catalog key). See the project MLSysIM rules →
*Storage format*. Literature holds ONLY genuinely-cited field figures.
"""
from pathlib import Path

from ..core.registry import Registry
from ..core.loader import load_sourced_registry

_DATA = Path(__file__).parent / "data"


def _load(stem: str, name: str, doc: str = "") -> type:
    return load_sourced_registry(_DATA / f"{stem}.yaml", name=name, doc=doc)


Training = _load("training", "Training")
Benchmarks = _load("benchmarks", "Benchmarks", "Published benchmark anchors used by empirical validation tests.")
Chinchilla = _load("chinchilla", "Chinchilla")
Communication = _load("communication", "Communication")
BatchSize = _load("batchsize", "BatchSize", "McCandlish et al. (2018) critical batch size estimates.")
TrainingRuns = _load("training_runs", "TrainingRuns", "Cited training-run field figures (GPU counts, durations, MFU, interruptions).")


from ..core.provenance import Provenance, ProvenanceKind

# ---------------------------------------------------------------------------
# Kaplan et al. (2020) test-loss power laws for @fig-model-scaling. Each law is
# a published closed-form fit L = (X / x_c)^(-alpha) over the studied range, so
# the figure regenerates from these constants instead of a traced raster. The
# constants are the ones printed on the source figure (Kaplan et al. 2020).
# ---------------------------------------------------------------------------
_KAPLAN_SCALING_PROVENANCE = Provenance(
    kind=ProvenanceKind.LITERATURE,
    ref="Kaplan et al. (2020), Scaling Laws for Neural Language Models (Figure 1; Tables 2, 3, 5, 6)",
    url="https://arxiv.org/abs/2001.08361",
    verified="2026-10-15",
)

# (axis, x_c, alpha, x_min, x_max)  -- L = (X / x_c)^(-alpha) on [x_min, x_max]
_KAPLAN_SCALING_ROWS = (
    ("compute", 2.3e8, 0.050, 1e-9, 1e1),   # C_min, PF-days (non-embedding)
    ("data", 5.4e13, 0.095, 1e8, 1e10),     # D, tokens
    ("params", 8.8e13, 0.076, 1e3, 1e9),    # N, non-embedding parameters
)


class KaplanScalingLaws:
    """Kaplan et al. (2020) test-loss power laws vs compute, data, parameters.

    Source: see ``provenance``. Consumed by @fig-model-scaling. Each entry is a
    dict ``{"axis", "x_c", "alpha", "x_min", "x_max"}`` giving the published fit
    ``L = (X / x_c)^(-alpha)`` over the studied range ``[x_min, x_max]``.
    """

    provenance = _KAPLAN_SCALING_PROVENANCE
    entries = tuple(
        {"axis": k, "x_c": xc, "alpha": a, "x_min": lo, "x_max": hi}
        for (k, xc, a, lo, hi) in _KAPLAN_SCALING_ROWS
    )
    list = entries  # parity with the Registry `.list` convention

    # -- Added 2026-10-15: the joint fits and the compute-efficient frontier, so
    # learning-curve and allocation figures regenerate from the published laws.
    # Values verified against arXiv:2001.08361v1. ``entries`` above keep the
    # constants printed on Figure 1 (C_min scale 2.3e8 PF-days); Table 5 states
    # the same law's scale as 3.1e8 PF-days (``single_variable_fits`` below).

    # Table 5 (Appendix A): single-variable power laws L = (X_c / X)^alpha.
    # Units: params (non-embedding), tokens, PF-days, tokens, steps.
    single_variable_fits = {
        "params": {"alpha": 0.076, "scale": 8.8e13},
        "data": {"alpha": 0.095, "scale": 5.4e13},
        "compute_naive": {"alpha": 0.057, "scale": 1.6e7},
        "compute_min": {"alpha": 0.050, "scale": 3.1e8},
        "critical_batch": {"alpha": 0.21, "scale": 2.1e8},
        "steps": {"alpha": 0.76, "scale": 2.1e3},
    }

    # Table 2 (Section 4.2): L(N, D) = [(N_c/N)^(alpha_N/alpha_D) + D_c/D]^alpha_D,
    # Eq. 1.5, fit with early stopping. N in non-embedding params, D in tokens.
    loss_vs_params_and_data = {"alpha_n": 0.076, "alpha_d": 0.103, "n_c": 6.4e13, "d_c": 1.8e13}

    # Table 3 (Section 5.2): L(N, S_min) = (N_c/N)^alpha_N + (S_c/S_min)^alpha_S,
    # Eq. 1.6 / 5.6, infinite-data learning curves. S_min = S / (1 + B_crit(L)/B)
    # (Eq. 5.4), with B_crit(L) = B_star / L^(1/alpha_B) (Eq. 5.3, Table 5 values).
    loss_vs_params_and_steps = {"alpha_n": 0.077, "alpha_s": 0.76, "n_c": 6.5e13, "s_c": 2.1e3}

    # Table 6 (Appendix A): compute-efficient frontier X = X_e * C_min^p,
    # C_min in PF-days. Units: params, tokens, steps, tokens.
    compute_efficient_frontier = {
        "params": {"p": 0.73, "scale": 1.3e9},
        "batch_tokens": {"p": 0.24, "scale": 2.0e6},
        "steps_min": {"p": 0.03, "scale": 5.4e3},
        "tokens": {"p": 0.27, "scale": 2e10},
    }

    # Section 2.2: default run, 2.5e5 steps of 512 sequences x 1024 tokens
    # (B = 2^19 tokens). Compute is C = 6NBS (Eq. 5.5; the 6 is
    # Literature.Chinchilla.ComputeConstant); a PF-day is 1e15 FLOP/s for one
    # day (Section 1.2), so build it with pint units rather than a constant here.
    default_steps = 2.5e5
    default_batch_sequences = 512
    default_context_tokens = 1024




class Literature(Registry):
    """Registry namespace for Literature."""
    Training = Training
    Benchmarks = Benchmarks
    Chinchilla = Chinchilla
    Communication = Communication
    BatchSize = BatchSize
    TrainingRuns = TrainingRuns
    KaplanScalingLaws = KaplanScalingLaws
