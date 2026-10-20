"""Agentic ML systems physics and memory accounting formulas (Volume III)."""

from __future__ import annotations

import math
from mlsysim.core.units import ureg
from mlsysim.core._validation import validate_at_least


def calc_kv_cache_bytes_per_token(
    model=None,
    *,
    n_layers: int | None = None,
    n_kv_heads: int | None = None,
    head_dim: int | None = None,
    bytes_per_elem=2,
):
    """
    Calculate the Key-Value (KV) cache memory footprint per token.

    For a transformer architecture with L layers, H_kv Key/Value attention
    heads, and head dimension d_head:

        m_token = 2 * L * H_kv * d_head * bytes_per_elem

    The leading factor of 2 accounts for storing separate Key and Value tensors.

    Parameters
    ----------
    model : TransformerWorkload or AgentArchitecture, optional
        Model from which layers, heads, and dimensions are extracted.
    n_layers : int, optional
        Number of transformer layers (if model is omitted).
    n_kv_heads : int, optional
        Number of Key/Value attention heads (if model is omitted).
    head_dim : int, optional
        Dimension per attention head (if model is omitted).
    bytes_per_elem : Quantity or int, optional
        Numerical precision in bytes (default: 2 for FP16/BF16).

    Returns
    -------
    Quantity
        Memory size per token (in bytes).
    """
    if model is not None:
        if hasattr(model, "reference_model"):
            model = model.reference_model
        layers = model.layers if n_layers is None else n_layers
        kv_heads = getattr(model, "kv_heads", model.heads) if n_kv_heads is None else n_kv_heads
        if head_dim is not None:
            d_head = head_dim
        elif hasattr(model, "hidden_dim") and hasattr(model, "heads") and model.heads > 0:
            d_head = model.hidden_dim // model.heads
        else:
            d_head = getattr(model, "head_dim", 128)
    else:
        layers = n_layers
        kv_heads = n_kv_heads
        d_head = head_dim

    if layers is None or kv_heads is None or d_head is None:
        raise ValueError("Must provide model or all of (n_layers, n_kv_heads, head_dim).")

    validate_at_least(layers, 1, "n_layers")
    validate_at_least(kv_heads, 1, "n_kv_heads")
    validate_at_least(d_head, 1, "head_dim")

    bpe = bytes_per_elem if hasattr(bytes_per_elem, "units") else bytes_per_elem * ureg.byte
    total = 2 * layers * kv_heads * d_head * bpe
    return total.to(ureg.byte) if hasattr(total, "to") else total * ureg.byte



# --- Trajectory reliability and cost per accepted task (added 2026-10-20) ----
# Horizon compounding, cost per accepted task, independent retries, and the
# input an append-only trajectory resends. Each is the closed form of a
# recurring agent calculation; the solvers in engine/solvers/agents.py use them.


def calc_unchecked_success(step_success: float, horizon: float) -> float:
    """
    Success of a task whose ``horizon`` steps must all be right, unchecked.

        Pr(success) = (1 - eps)^H = p_step^H

    Assumes independent steps, no check between them, and no recovery.
    """
    return step_success ** horizon


def calc_step_error_for_target(target_success: float, horizon: float) -> float:
    """
    Per-step error rate that holds ``target_success`` over ``horizon`` unchecked steps.

        eps = 1 - target^(1/H)

    For small eps, eps * H ~ -ln(target) at any horizon.
    """
    return 1 - target_success ** (1 / horizon)


def calc_first_horizon_below(step_success: float, threshold: float) -> int:
    """
    First whole horizon ``n`` at which ``step_success ** n`` falls below ``threshold``.

        n = floor(ln(threshold) / ln(step_success)) + 1
    """
    return math.floor(math.log(threshold) / math.log(step_success)) + 1


def calc_cost_per_accepted(attempt_cost, acceptance_probability: float):
    """
    Expected cost per accepted task when attempts repeat until one is accepted.

        C_eff = E[C_attempt] / p_acc

    Attempts are geometric with mean 1 / p_acc, so every rejected attempt is
    charged to the accepted one.
    """
    return attempt_cost / acceptance_probability


def calc_break_even_acceptance(acceptance_probability: float, cut: float, share: float) -> float:
    """
    Acceptance that matches a price cut on one component of the attempt cost.

    Removing a fraction ``cut`` of a component that is ``share`` of the attempt
    cost gives the same cost per accepted task as raising acceptance to

        p' = p_acc / (1 - cut * share)
    """
    return acceptance_probability / (1 - cut * share)


def calc_independent_success(acceptance_probability: float, attempts: int) -> float:
    """
    Chance that at least one of ``attempts`` independent attempts is accepted.

        1 - (1 - p_acc)^k
    """
    return 1 - (1 - acceptance_probability) ** attempts


def calc_append_only_context(initial_tokens, tokens_per_turn, turns: int):
    """
    Context of the last call and total input sent by an append-only trajectory.

    Turn t (from 1) reads s0 + delta (t - 1) tokens, so the last of H turns reads
    s0 + delta (H - 1), and the H calls together send

        H s0 + delta H (H - 1) / 2

    tokens: input grows quadratically with the horizon when every turn resends
    the history. Returns (last_call, cumulative).
    """
    last_call = initial_tokens + tokens_per_turn * (turns - 1)
    cumulative = turns * initial_tokens + tokens_per_turn * (turns * (turns - 1) // 2)
    return last_call, cumulative
# --- Model calls and agent trajectories (added 2026-10-20) -------------------
# Equations behind one model call (admission, sampling, a fallible check) and
# behind a trajectory of calls (horizon compounding, cost per accepted task, and
# resent history live above). Pure arithmetic on stated inputs; the solvers in
# engine/solvers/model_call.py compose them.

import math as _math




def calc_tempered_softmax(logits, temperature: float = 1.0) -> list:
    """Softmax of ``logits / T``: the sampling distribution at temperature ``T``."""
    exps = [_math.exp(z / temperature) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def calc_nucleus(probabilities, top_p: float):
    """Top-p truncation: the shortest highest-probability prefix whose mass reaches ``top_p``.

    Returns ``(kept, mass, renormalized)``: how many tokens stay, their total
    probability, and their probabilities rescaled to sum to one.
    """
    ordered = sorted(probabilities, reverse=True)
    mass, kept = 0.0, 0
    for p in ordered:
        mass += p
        kept += 1
        if mass >= top_p:
            break
    return kept, mass, [p / mass for p in ordered[:kept]]


def calc_pass_posterior(defect_rate: float, false_accept_rate: float, valid_accept_rate: float) -> float:
    """Bayes' rule: share of passing candidates that are defective, ``P(Def | PASS)``."""
    defective_pass = false_accept_rate * defect_rate
    return defective_pass / (defective_pass + valid_accept_rate * (1 - defect_rate))




def calc_reused_prefix_prompt_tokens(turns: int, first_prompt: int, appended: int) -> int:
    """Fresh prefill over ``turns`` calls when each earlier prompt prefix is reused: ``S0 + (H - 1) D``."""
    return first_prompt + (turns - 1) * appended
