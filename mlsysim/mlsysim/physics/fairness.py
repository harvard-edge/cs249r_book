"""Fairness arithmetic: error-rate identities, the bias feedback model, and gap sample sizes.

Added 2026-10-20. The closed forms a fairness release gate rests on:

- Chouldechova's identity, which fixes a group's false positive rate from its
  base rate, positive predictive value, and false negative rate;
- the bias feedback model, a disparity multiplied by a feedback factor at each
  retraining cycle, and the cycle at which it crosses a release gate;
- the per-group sample size that pins a gap between two groups' rates.
"""

from __future__ import annotations

import math


def calc_fpr_from_ppv(base_rate, ppv, fnr):
    """False positive rate implied by a group's base rate, precision, and false negative rate.

    Equation (Chouldechova 2017, "Fair Prediction with Disparate Impact"):
        FPR = p / (1 - p) * (1 - PPV) / PPV * (1 - FNR)

    With equal PPV and unequal base rates ``p``, the false positive and false
    negative rates cannot both be equal across groups.
    """
    p = float(base_rate)
    if not 0.0 < p < 1.0 or not 0.0 < ppv <= 1.0:
        raise ValueError("base_rate must be in (0, 1) and ppv in (0, 1]")
    return p / (1 - p) * (1 - ppv) / ppv * (1 - fnr)


def calc_feedback_disparity(launch_gap, factor, cycles):
    """Disparity after ``cycles`` retraining cycles: ``gap(k) = gap(0) * factor^k``.

    A factor above one grows a gap passed at launch; below one it decays.
    """
    return launch_gap * factor ** cycles


def calc_cycles_to_gate(launch_gap, factor, gate):
    """Unrounded cycle count at which ``launch_gap * factor^k`` reaches ``gate``.

    ``k = ln(gate / gap(0)) / ln(factor)``; the gate first fails at the next
    whole cycle above it. Requires a growing gap (``factor > 1``) that starts
    inside the gate.
    """
    if factor <= 1.0 or not 0.0 < launch_gap < gate:
        raise ValueError("need factor > 1 and 0 < launch_gap < gate")
    return math.log(gate / launch_gap) / math.log(factor)


def calc_gap_samples_per_group(rate, half_width, z_critical):
    """Per-group draws, equal groups, for a gap between two rates to have half-width ``half_width``.

    The gap's variance is the sum of both groups' variances:
    ``h = z sqrt(p (1 - p) / n + p (1 - p) / n)``, so
    ``n = ceil(2 z^2 p (1 - p) / h^2)``, twice the single-rate count.
    """
    return math.ceil(2 * z_critical ** 2 * rate * (1 - rate) / half_width ** 2)
