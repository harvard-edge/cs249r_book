"""
Shared pass/fail checks for Milestone 05 (Parts 1, 3 and 4).
=============================================================

A low training loss does not prove that a transformer works. Two broken
implementations reach one anyway (measured 2026-09-29, release audit):

* **Attention that sees the future.** Without the causal mask, position i
  reads token i+1 and copies it. Training loss collapses (0.05 on
  Shakespeare), yet the model is useless for generation, because at
  generation time the future token does not exist yet.
* **No attention at all.** With the attention sublayer returning zeros, the
  model still learns which character tends to follow which (a bigram model),
  and the old gates accepted that as convergence.

The causality probe below catches the first case directly. The second is
caught by each part's loss threshold, calibrated against a no-attention run.

The probe is the test a production team would run on a new attention kernel:
change the tokens AFTER position k and check that nothing at positions <= k
moves. A causal model must give bit-for-bit the same logits there, because
those positions never read the changed tokens.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from tinytorch.core.tensor import Tensor

# Logits for positions <= k must match to this absolute tolerance. A correct
# causal model matches exactly (masked keys get weight exp(-inf) = 0), so
# any real leak shows up many orders of magnitude above it.
CAUSALITY_ATOL = 1e-4


@dataclass
class CausalityResult:
    """Outcome of the causality probe."""

    passed: bool
    max_leak: float        # largest |logit change| at a position that must not move
    cut: int               # the cut k where the largest leak appeared
    leaking_position: int  # first position <= k whose logits moved (-1 if none)
    seq_len: int


def causality_probe(model, tokens: Sequence[int], vocab_size: int,
                    atol: float = CAUSALITY_ATOL) -> CausalityResult:
    """
    Check that changing future tokens never changes past predictions.

    For several cut points k, every token after k is replaced by a different
    token, and the logits at positions 0..k are compared with the original.

    Args:
        model: Trained model with forward(Tensor[B, S]) -> Tensor[B, S, V].
        tokens: A real token sequence (for example a training window).
        vocab_size: Size of the vocabulary, used to pick replacement tokens.
        atol: Largest allowed change in any logit at positions <= k.

    Returns:
        CausalityResult describing the worst leak found.
    """
    base_ids = np.asarray(tokens, dtype=np.int32).reshape(-1)
    seq_len = int(base_ids.shape[0])
    if seq_len < 3:
        raise ValueError("causality probe needs a sequence of at least 3 tokens")

    base_logits = np.asarray(model.forward(Tensor(base_ids[np.newaxis, :])).data)[0]

    worst = CausalityResult(True, 0.0, -1, -1, seq_len)
    for cut in sorted({0, seq_len // 4, seq_len // 2, seq_len - 2}):
        changed_ids = base_ids.copy()
        # Shift every future token to a different vocabulary entry.
        changed_ids[cut + 1:] = (changed_ids[cut + 1:] + 1) % vocab_size
        changed_logits = np.asarray(
            model.forward(Tensor(changed_ids[np.newaxis, :])).data
        )[0]

        diff = np.abs(changed_logits[:cut + 1] - base_logits[:cut + 1]).max(axis=-1)
        leak = float(diff.max())
        if leak > worst.max_leak or not np.isfinite(leak):
            moved = np.nonzero(~(diff <= atol))[0]
            worst = CausalityResult(
                passed=False,
                max_leak=leak,
                cut=cut,
                leaking_position=int(moved[0]) if moved.size else -1,
                seq_len=seq_len,
            )

    worst.passed = bool(np.isfinite(worst.max_leak) and worst.max_leak <= atol)
    return worst


def causality_report(result: CausalityResult) -> str:
    """One line for the results panel."""
    if result.passed:
        return (f"Causality probe  : passed (changing future tokens moved past "
                f"logits by at most {result.max_leak:.1e})")
    return (f"Causality probe  : FAILED (changing tokens after position {result.cut} "
            f"moved the logits at position {result.leaking_position} by "
            f"{result.max_leak:.3g})")


def causality_failure_message(result: CausalityResult) -> str:
    """Teaching message printed when the probe fails."""
    return (
        "[bold red]❌ YOUR ATTENTION LETS A POSITION SEE FUTURE TOKENS[/bold red]\n\n"
        f"  • Changed every token after position {result.cut} of a "
        f"{result.seq_len}-token sequence.\n"
        f"  • The prediction at position {result.leaking_position} changed by "
        f"{result.max_leak:.3g}, but position {result.leaking_position} must only "
        f"read tokens 0..{result.leaking_position}.\n\n"
        "  A language model predicts token i+1 from tokens 0..i. If position i can\n"
        "  attend to i+1, training just copies the answer: the loss looks excellent\n"
        "  while generation (where the future does not exist yet) falls apart.\n\n"
        "  Check the causal mask:\n"
        "    • create_causal_mask(seq_len) is lower-triangular (1 = may attend)\n"
        "    • _apply_mask sets blocked scores to -inf BEFORE the softmax\n"
        "    • MultiHeadAttention.forward passes the mask through to every head"
    )


# ---------------------------------------------------------------------------
# Loss measured outside the student's code
# ---------------------------------------------------------------------------
# 2026-09-29 (release audit): every loss gate used to read the value from the
# student's CrossEntropyLoss forward (Module 04) or from Trainer.train_epoch
# (Module 08). Both can be wrong while the model still trains, because the
# gradients come from Module 06: a cross-entropy forward that returns 0 read
# "loss 0.0000" and passed Parts 1 and 3, and a Trainer.train_epoch that does
# nothing passed Part 4 on an untrained model. Gate losses are now computed
# here, in NumPy, from the model's logits.

# The student's cross-entropy must agree with the NumPy value to this
# tolerance (float32 log-softmax on a ~100-way vocabulary differs by ~1e-6).
CE_ATOL = 1e-3
CE_RTOL = 1e-3

# Logits at different positions of a one-token sequence must differ by at
# least this much. Without positional information every position sees the
# same bag of identical tokens and matches to float rounding (~1e-7); a
# learned position table moves them by ~0.1 or more.
POSITION_MIN_SPREAD = 1e-3


def numpy_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Mean negative log-likelihood of the targets under softmax(logits).

    Numerically stable: subtract each row's max before exponentiating, so
    log-softmax = (z - max) - log(sum(exp(z - max))) never overflows.

    Args:
        logits: Array of shape (..., V) of unnormalized scores.
        targets: Integer array of shape (...) with the correct class ids.

    Returns:
        Mean cross-entropy in nats per target.
    """
    z = np.asarray(logits, dtype=np.float64)
    z = z.reshape(-1, z.shape[-1])
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    if z.shape[0] != y.shape[0]:
        raise ValueError(f"{z.shape[0]} rows of logits but {y.shape[0]} targets")
    shifted = z - z.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(y.shape[0]), y].mean())


def measured_loss(model, inputs: np.ndarray, targets: np.ndarray,
                  batch_size: int = 64) -> float:
    """
    Token-averaged cross-entropy of the model on (inputs, targets), no updates.

    Only model.forward runs student code; the loss itself is computed in
    NumPy (numpy_cross_entropy), so a broken loss function or training loop
    cannot change the number the gates read.
    """
    inputs = np.asarray(inputs, dtype=np.int32)
    targets = np.asarray(targets, dtype=np.int32)
    total, count = 0.0, 0
    for start in range(0, len(inputs), batch_size):
        x = inputs[start:start + batch_size]
        y = targets[start:start + batch_size]
        logits = np.asarray(model.forward(Tensor(x)).data)
        total += numpy_cross_entropy(logits, y) * y.size
        count += y.size
    return total / max(count, 1)


# Kept for callers written before 2026-09-29; the criterion is ignored.
def untrained_loss(model, inputs, targets, criterion=None, vocab_size=None,
                   batch_size: int = 64) -> float:
    """Loss before training; same as measured_loss (criterion ignored)."""
    return measured_loss(model, inputs, targets, batch_size=batch_size)


@dataclass
class LossCheckResult:
    """Your cross-entropy forward compared with the NumPy value on one batch."""

    passed: bool
    student_loss: float
    reference_loss: float


def cross_entropy_check(model, criterion, inputs: np.ndarray, targets: np.ndarray,
                        vocab_size: int, atol: float = CE_ATOL,
                        rtol: float = CE_RTOL) -> LossCheckResult:
    """
    Run YOUR CrossEntropyLoss forward on one batch and compare it with NumPy.

    Training can still work when the forward value is wrong (the gradient is
    Module 06's backward), so without this check a broken forward only shows
    up as a loss curve that means nothing.
    """
    x = np.asarray(inputs, dtype=np.int32)
    y = np.asarray(targets, dtype=np.int32)
    logits = model.forward(Tensor(x))
    reference = numpy_cross_entropy(np.asarray(logits.data), y)
    try:
        student = float(np.asarray(
            criterion(logits.reshape(-1, vocab_size), Tensor(y.reshape(-1))).data
        ).reshape(-1)[0])
    except Exception:  # a crash is a failed check, reported with the message
        student = float("nan")
    ok = bool(np.isfinite(student) and abs(student - reference) <= atol + rtol * abs(reference))
    return LossCheckResult(ok, student, reference)


def cross_entropy_failure_message(result: LossCheckResult) -> str:
    """Teaching message printed when the cross-entropy check fails."""
    return (
        "[bold red]❌ YOUR CROSS-ENTROPY FORWARD RETURNS THE WRONG LOSS[/bold red]\n\n"
        f"  • Your CrossEntropyLoss forward returns {result.student_loss:.4f},\n"
        f"    but the loss of these logits is {result.reference_loss:.4f}\n"
        "    (mean of -log softmax(logits)[target], computed in NumPy).\n\n"
        "  Training can still move the weights, because the gradient comes from\n"
        "  the backward pass, but every loss you print is then meaningless and\n"
        "  no gate could tell a trained model from an untrained one.\n\n"
        "  Check CrossEntropyLoss / CrossEntropyFunction.forward (Module 04):\n"
        "    • log-softmax subtracts the row max, then log(sum(exp(...)))\n"
        "    • pick the log-probability of each row's target class\n"
        "    • return the MEAN of the negated values over all rows"
    )


@dataclass
class PositionResult:
    """Outcome of the positional-information probe."""

    passed: bool
    spread: float   # largest |logit difference| between any position and position 0
    seq_len: int


def position_probe(model, token_id: int, seq_len: int,
                   min_spread: float = POSITION_MIN_SPREAD) -> PositionResult:
    """
    Check that the model knows WHERE each token sits.

    Feed one token repeated seq_len times. Without positional information
    every position holds the same vector and attends to a set of identical
    vectors, so every position produces the same logits. With a working
    positional encoding the positions differ.
    """
    ids = np.full((1, seq_len), int(token_id), dtype=np.int32)
    logits = np.asarray(model.forward(Tensor(ids)).data)[0]
    spread = float(np.abs(logits - logits[0:1]).max())
    return PositionResult(bool(np.isfinite(spread) and spread > min_spread), spread, seq_len)


def position_failure_message(result: PositionResult) -> str:
    """Teaching message printed when the position probe fails."""
    return (
        "[bold red]❌ YOUR MODEL CANNOT TELL POSITIONS APART[/bold red]\n\n"
        f"  • Fed one token repeated {result.seq_len} times. The logits at every\n"
        f"    position matched to within {result.spread:.1e}.\n\n"
        "  Attention is a weighted sum over a set: without positional encoding,\n"
        "  'the cat sat' and 'sat the cat' look the same to it. The position\n"
        "  table has to be ADDED to the token embeddings.\n\n"
        "  Check PositionalEncoding.forward (Module 11): it must return\n"
        "  x + position_embeddings[start_pos:start_pos + seq_len], not x."
    )


# ---------------------------------------------------------------------------
# Attention that ignores content
# ---------------------------------------------------------------------------
# Measured 2026-09-29 (mean entropy / entropy of uniform, over rows i >= 1,
# heads and layers, after each part's default training):
#   correct code : Part 1 0.555, 0.556; Part 3 0.617, 0.616; Part 4 0.708, 0.694
#   uniform attention (scores x 0): exactly 1.000 in every part
# Uniform attention passed the optional Parts 3 and 4 before this probe (it
# still memorizes and parses some prompts). 0.90 sits 0.19 above the
# highest correct value and also catches near-uniform scores.
ATTENTION_ENTROPY_MAX = 0.90


@dataclass
class AttentionContentResult:
    """Outcome of the attention-content probe."""

    passed: bool
    entropy_ratio: float   # mean entropy / entropy of uniform attention (1.0 = uniform)
    most_uniform: float    # the largest ratio of any single layer and head
    uniform_gap: float     # |attention output - causal average of V| / |attention output|
    seq_len: int


def attention_content_probe(model, tokens: Sequence[int],
                            max_ratio: float = ATTENTION_ENTROPY_MAX) -> AttentionContentResult:
    """
    Check that trained attention picks WHICH earlier tokens to read.

    For every block, the probe feeds the block's real input through YOUR
    Q/K/V projections, _split_heads and scaled_dot_product_attention, and
    reads the attention weights that function returns. Row i may spread its
    weight over positions 0..i, so its entropy is at most log(i + 1), reached
    only when every earlier token gets the same weight. The probe reports
    the mean of entropy / log(i + 1) over rows i >= 1, heads and layers:
    exactly 1.0 for attention that ignores content, clearly lower once
    training has taught Q and K what to match.

    It also compares the attention sublayer's real output with the causal
    average of its values (what uniform attention would produce), a check
    that does not depend on how scaled_dot_product_attention is called.

    Nothing in your code is replaced: the probe calls your model's own
    layers, in the order TinyGPT.forward calls them.
    """
    from tinytorch.core.attention import scaled_dot_product_attention
    from tinytorch.core.transformers import create_causal_mask

    ids = np.asarray(tokens, dtype=np.int32).reshape(1, -1)
    seq_len = int(ids.shape[1])
    if seq_len < 3:
        raise ValueError("attention probe needs a sequence of at least 3 tokens")

    x = model.embedding_layer.forward(Tensor(ids))
    mask = create_causal_mask(seq_len)
    mask4 = mask.reshape(1, 1, seq_len, seq_len)
    rows = np.arange(1, seq_len)
    uniform_entropy = np.log(rows + 1.0)                       # log(i + 1) for rows i >= 1
    causal_counts = np.arange(1, seq_len + 1, dtype=np.float64)[:, None]

    head_ratios, gaps = [], []
    for block in model.blocks:
        attn = block.attention
        normed = block.ln1.forward(x)

        # Attention weights from YOUR scaled_dot_product_attention.
        q, k, v = (attn._split_heads(proj.forward(normed), 1, seq_len)
                   for proj in (attn.q_proj, attn.k_proj, attn.v_proj))
        _, weights = scaled_dot_product_attention(q, k, v, mask=mask4)
        w = np.asarray(weights.data, dtype=np.float64).reshape(-1, seq_len, seq_len)
        w = w[:, 1:, :]                                        # rows i >= 1 (row 0 has one choice)
        p = np.clip(w, 1e-12, 1.0)
        entropy = -(w * np.log(p)).sum(axis=-1)                # (heads, S-1)
        head_ratios.extend((entropy / uniform_entropy).mean(axis=-1).tolist())

        # The sublayer's real output vs. the output of uniform causal attention.
        actual = np.asarray(attn.forward(normed, mask).data, dtype=np.float64)[0]
        values = np.asarray(attn.v_proj.forward(normed).data, dtype=np.float64)[0]
        averaged = np.cumsum(values, axis=0) / causal_counts
        uniform = np.asarray(attn.out_proj.forward(
            Tensor(averaged[np.newaxis].astype(np.float32))).data, dtype=np.float64)[0]
        gaps.append(float(np.linalg.norm(actual - uniform) /
                          max(np.linalg.norm(actual), 1e-12)))

        x = block.forward(x, mask)

    ratio = float(np.mean(head_ratios))
    return AttentionContentResult(
        passed=bool(np.isfinite(ratio) and ratio <= max_ratio),
        entropy_ratio=ratio,
        most_uniform=float(np.max(head_ratios)),
        uniform_gap=float(np.min(gaps)),
        seq_len=seq_len,
    )


def attention_content_report(result: AttentionContentResult) -> str:
    """One line for the results panel."""
    state = "passed" if result.passed else "FAILED"
    return (f"Attention probe  : {state} (weight entropy {result.entropy_ratio:.2f} "
            f"of uniform; 1.00 = every earlier token weighted equally)")


def attention_content_failure_message(result: AttentionContentResult) -> str:
    """Teaching message printed when the attention-content probe fails."""
    return (
        "[bold red]❌ YOUR ATTENTION WEIGHTS EVERY EARLIER TOKEN EQUALLY[/bold red]\n\n"
        f"  • Read the attention weights of every layer and head on a "
        f"{result.seq_len}-token window.\n"
        f"  • Their entropy is {result.entropy_ratio:.3f} of the uniform maximum "
        f"(needed at most {ATTENTION_ENTROPY_MAX:.2f}).\n"
        f"    Trained attention concentrates on the tokens that matter; yours\n"
        f"    spreads its weight evenly, so Q and K never influence the result.\n\n"
        "  The model can still lower its loss this way (an average of the past is\n"
        "  some context), which is why the loss alone did not flag it.\n\n"
        "  Check that scores = QKᵀ/√d_k are used before softmax:\n"
        "    • _compute_attention_scores returns Q @ K^T (not zeros, not a constant)\n"
        "    • _scale_scores multiplies the scores by 1/sqrt(d_k), not by 0\n"
        "    • the softmax is taken over those scores, along the key axis (dim=-1)"
    )


def bigram_cross_entropy(train_ids: Sequence[int], eval_ids: Sequence[int],
                         vocab_size: int, smoothing: float = 1.0) -> float:
    """
    Cross-entropy (nats per token) of a counted bigram model.

    The bigram table P(next | current) is counted on train_ids with add-k
    smoothing and scored on eval_ids. A transformer that uses its context
    should beat it; a model with broken attention cannot see past the
    current character, so it lands at or above this number.
    """
    train_ids = np.asarray(train_ids, dtype=np.int64)
    eval_ids = np.asarray(eval_ids, dtype=np.int64)
    counts = np.full((vocab_size, vocab_size), smoothing, dtype=np.float64)
    np.add.at(counts, (train_ids[:-1], train_ids[1:]), 1.0)
    log_probs = np.log(counts / counts.sum(axis=1, keepdims=True))
    return float(-log_probs[eval_ids[:-1], eval_ids[1:]].mean())


def main() -> int:
    """
    Run the causality probe on a tiny untrained TinyGPT from Part 1.

    `python transformer_gates.py` checks YOUR attention in about a second,
    without training anything: exit 0 if future tokens stay invisible.
    """
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parent / "01_tinygpt_shakespeare.py"
    spec = importlib.util.spec_from_file_location("tinygpt_part1", script)
    part1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(part1)

    rng = np.random.default_rng(0)
    model, _ = part1.build_model(vocab_size=20, embed_dim=16, num_layers=1,
                                 num_heads=2, max_seq_len=16)
    result = causality_probe(model, rng.integers(0, 20, size=12), vocab_size=20)
    print(causality_report(result))
    return 0 if result.passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
