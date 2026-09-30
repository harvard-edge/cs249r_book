"""
Milestone 05 (Transformer) gates must fail when the student's attention is broken.
=================================================================================

History (2026-09-29, release audit):
  * Part 1 (Shakespeare) passed on ``loss < 2.50 or drop > 1.20``. A model
    with attention zeroed stalls at the bigram entropy (2.37) and passed; a
    model without the causal mask reads the next token (loss 0.05) and passed.
  * Part 4 (chat) passed a no-attention model through its ``drop > 1.20``
    clause, and passed no-mask and uniform-attention models too.
  * Part 3 (TinyCopilot) scored SAMPLED completions against a 40% target, so
    correct code failed about one run in four.
  * Every loss gate read the student's own numbers. A cross-entropy forward
    returning 0 (training still works through the backward) passed Parts 1
    and 3; a no-op Trainer.train_epoch passed Part 4 on an untrained model.
    Gate losses are now computed in NumPy from the logits, and the student's
    cross-entropy is checked against that value before training.
  * An identity positional encoding passed Parts 3 and 4 (Part 4 also added
    a second position table). A repeated-token probe now catches it.

The fast tests check the causality probe in-process on a tiny untrained
model. The slow tests run the real scripts as subprocesses with one piece of
attention broken through a ``sitecustomize`` hook (see test_gates_early.py).

Set TT_GATES_MILESTONES_DIR to point the tests at another copy of the
milestones directory (used to show the tests fail on older versions).
"""

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

TINYTORCH_ROOT = Path(__file__).resolve().parents[2]
MILESTONES_DIR = Path(os.environ.get("TT_GATES_MILESTONES_DIR", TINYTORCH_ROOT / "milestones"))
M05 = MILESTONES_DIR / "05_2017_transformer"

SHAKESPEARE = "05_2017_transformer/01_tinygpt_shakespeare.py"
COPILOT = "05_2017_transformer/03_tinycopilot.py"
CHAT = "05_2017_transformer/04_tinygpt_chat.py"

SITECUSTOMIZE = textwrap.dedent('''
    """Test-only sabotage hook: TT_SABOTAGE=<name> breaks one piece of the student's code."""
    import os
    import sys

    _name = os.environ.get("TT_SABOTAGE", "")

    if _name:
        import numpy as np
        import tinytorch.core.attention as A
        from tinytorch.core.tensor import Tensor

        if _name == "no_mask":
            # The causal mask is ignored: position i can read token i+1.
            A._apply_mask = lambda scores, mask: scores
        elif _name == "no_attn":
            # The attention sublayer contributes nothing (a bigram model).
            A.MultiHeadAttention.forward = (
                lambda self, x, mask=None: Tensor(np.zeros_like(x.data)))
        elif _name == "uniform_attn":
            # Scores all zero: causal attention averages every earlier token equally.
            A._scale_scores = lambda scores, d_k: scores * 0.0
        elif _name == "ce_zero":
            # Cross-entropy forward reports 0; its backward (and training) still works.
            import tinytorch.core.losses as LS
            _fwd = LS.CrossEntropyFunction.forward
            LS.CrossEntropyFunction.forward = (
                lambda self, *a, **k: np.zeros_like(_fwd(self, *a, **k)))
        elif _name == "no_train":
            # Trainer.train_epoch does nothing and reports a perfect loss.
            import tinytorch.core.training as TR
            TR.Trainer.train_epoch = lambda self, *a, **k: 0.0
        elif _name == "pe_identity":
            # The positional encoding returns its input: positions are invisible.
            import tinytorch.core.embeddings as E
            E.PositionalEncoding.forward = lambda self, x, start_pos=0: x
        else:
            raise SystemExit(f"unknown sabotage {_name}")

        sys.stderr.write(f"[SABOTAGE] {_name}\\n")
''')


# ---------------------------------------------------------------------------
# Fast: the causality probe itself, in-process
# ---------------------------------------------------------------------------

def _load(path, name):
    folder = str(path.parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gates():
    return _load(M05 / "transformer_gates.py", "transformer_gates_under_test")


@pytest.fixture(scope="module")
def tiny_model():
    part1 = _load(M05 / "01_tinygpt_shakespeare.py", "tinygpt_part1_under_test")
    model, _ = part1.build_model(vocab_size=20, embed_dim=16, num_layers=1,
                                 num_heads=2, max_seq_len=16)
    return model


TOKENS = np.random.default_rng(0).integers(0, 20, size=12)


def test_probe_passes_correct_causal_attention(gates, tiny_model):
    result = gates.causality_probe(tiny_model, TOKENS, vocab_size=20)
    assert result.passed, result
    assert result.max_leak == 0.0, "a causal model must match exactly at past positions"


def test_probe_fails_attention_without_the_mask(gates, tiny_model, monkeypatch):
    import tinytorch.core.attention as A
    monkeypatch.setattr(A, "_apply_mask", lambda scores, mask: scores)
    result = gates.causality_probe(tiny_model, TOKENS, vocab_size=20)
    assert not result.passed
    assert result.max_leak > 1e-3
    assert 0 <= result.leaking_position <= result.cut
    message = gates.causality_failure_message(result)
    assert "see future tokens" in message.lower() and "causal mask" in message


def test_probe_fails_a_mask_shifted_by_one(gates, tiny_model, monkeypatch):
    # An off-by-one mask (np.tril(..., k=1)) lets position i read token i+1.
    import tinytorch.core.attention as A
    from tinytorch.core.tensor import Tensor
    original = A._apply_mask

    def shifted(scores, mask):
        s = mask.data.shape[-1]
        return original(scores, Tensor(np.tril(np.ones((1, s, s), dtype=np.float32), k=1)))

    monkeypatch.setattr(A, "_apply_mask", shifted)
    assert not gates.causality_probe(tiny_model, TOKENS, vocab_size=20).passed


def test_bigram_baseline_matches_hand_count(gates):
    # "abab..." is perfectly predictable after one character, so the smoothed
    # bigram loss is small; shuffled text is not.
    ids = np.array([0, 1] * 200)
    assert gates.bigram_cross_entropy(ids, ids, vocab_size=2, smoothing=1.0) < 0.02
    noise = np.random.default_rng(1).integers(0, 8, size=4000)
    assert gates.bigram_cross_entropy(noise, noise, vocab_size=8) > 2.0


def test_numpy_cross_entropy_matches_hand_computation(gates):
    # Two classes with equal logits: -log(1/2) = ln 2 for either target.
    assert gates.numpy_cross_entropy(np.zeros((3, 2)), np.array([0, 1, 1])) == pytest.approx(np.log(2))
    # logits [ln 1, ln 3] -> p = [0.25, 0.75]; targets 1 and 0 -> mean(-ln .75, -ln .25)
    logits = np.log(np.array([[1.0, 3.0], [1.0, 3.0]]))
    expected = -(np.log(0.75) + np.log(0.25)) / 2
    assert gates.numpy_cross_entropy(logits, np.array([1, 0])) == pytest.approx(expected)


def test_numpy_cross_entropy_is_stable_and_shape_agnostic(gates):
    # Huge logits would overflow a naive exp; the max-shift keeps it finite.
    logits = np.array([[[1e4, 0.0, -1e4], [0.0, 1e4, 0.0]]])   # (B=1, S=2, V=3)
    value = gates.numpy_cross_entropy(logits, np.array([[0, 1]]))
    assert np.isfinite(value) and value == pytest.approx(0.0, abs=1e-6)
    assert gates.numpy_cross_entropy(logits, np.array([[2, 0]])) > 1e3
    with pytest.raises(ValueError):
        gates.numpy_cross_entropy(np.zeros((4, 3)), np.zeros(5, dtype=int))


def test_measured_loss_uses_numpy_not_the_criterion(gates, tiny_model, monkeypatch):
    rng = np.random.default_rng(2)
    x = rng.integers(0, 20, size=(5, 12))
    y = rng.integers(0, 20, size=(5, 12))
    from tinytorch.core.tensor import Tensor
    expected = gates.numpy_cross_entropy(tiny_model.forward(Tensor(x.astype(np.int32))).data, y)
    assert gates.measured_loss(tiny_model, x, y, batch_size=2) == pytest.approx(expected, rel=1e-6)
    # A broken cross-entropy forward cannot change the measured value.
    import tinytorch.core.losses as LS
    monkeypatch.setattr(LS.CrossEntropyFunction, "forward",
                        lambda self, *a, **k: np.zeros(()))
    assert gates.measured_loss(tiny_model, x, y) == pytest.approx(expected, rel=1e-6)


def test_cross_entropy_check_passes_correct_loss(gates, tiny_model):
    from tinytorch.core.losses import CrossEntropyLoss
    rng = np.random.default_rng(3)
    x, y = rng.integers(0, 20, size=(4, 12)), rng.integers(0, 20, size=(4, 12))
    result = gates.cross_entropy_check(tiny_model, CrossEntropyLoss(), x, y, vocab_size=20)
    assert result.passed, result
    assert result.student_loss == pytest.approx(result.reference_loss, abs=1e-4)


@pytest.mark.parametrize("broken", ["zero", "sum_not_mean", "crash"])
def test_cross_entropy_check_fails_broken_forward(gates, tiny_model, monkeypatch, broken):
    import tinytorch.core.losses as LS
    from tinytorch.core.losses import CrossEntropyLoss
    original = LS.CrossEntropyFunction.forward

    def forward(self, *a, **k):
        value = original(self, *a, **k)
        if broken == "zero":
            return np.zeros_like(value)
        if broken == "sum_not_mean":
            return value * 48            # 4 x 12 targets summed instead of averaged
        raise RuntimeError("boom")

    monkeypatch.setattr(LS.CrossEntropyFunction, "forward", forward)
    rng = np.random.default_rng(3)
    x, y = rng.integers(0, 20, size=(4, 12)), rng.integers(0, 20, size=(4, 12))
    result = gates.cross_entropy_check(tiny_model, CrossEntropyLoss(), x, y, vocab_size=20)
    assert not result.passed
    message = gates.cross_entropy_failure_message(result)
    assert f"{result.reference_loss:.4f}" in message and "cross-entropy forward returns" in message.lower()


def test_position_probe_passes_correct_positional_encoding(gates, tiny_model):
    result = gates.position_probe(tiny_model, token_id=3, seq_len=12)
    assert result.passed, result
    assert result.spread > 100 * gates.POSITION_MIN_SPREAD


def test_position_probe_fails_identity_positional_encoding(gates, tiny_model, monkeypatch):
    import tinytorch.core.embeddings as E
    monkeypatch.setattr(E.PositionalEncoding, "forward", lambda self, x, start_pos=0: x)
    result = gates.position_probe(tiny_model, token_id=3, seq_len=12)
    assert not result.passed, result
    assert "POSITIONS APART" in gates.position_failure_message(result)


# ---------------------------------------------------------------------------
# Slow: the real scripts as subprocesses, with one module broken
# ---------------------------------------------------------------------------

def _sharp_attention_model():
    """A tiny model whose Q/K weights are scaled up, so attention is peaked
    the way trained attention is (a fresh model's attention is near uniform)."""
    part1 = _load(M05 / "01_tinygpt_shakespeare.py", "tinygpt_part1_sharp")
    model, _ = part1.build_model(vocab_size=20, embed_dim=16, num_layers=1,
                                 num_heads=2, max_seq_len=16)
    for block in model.blocks:
        for proj in (block.attention.q_proj, block.attention.k_proj):
            proj.weight.data = proj.weight.data * 12.0
    return model


def test_attention_probe_passes_content_based_attention(gates):
    result = gates.attention_content_probe(_sharp_attention_model(), TOKENS)
    assert result.passed, result
    assert result.entropy_ratio < gates.ATTENTION_ENTROPY_MAX


def test_attention_probe_fails_uniform_attention(gates, monkeypatch):
    # 2026-09-29: uniform attention (scores x 0) passed the optional Parts 3-4.
    import tinytorch.core.attention as A
    monkeypatch.setattr(A, "_scale_scores", lambda scores, d_k: scores * 0.0)
    result = gates.attention_content_probe(_sharp_attention_model(), TOKENS)
    assert not result.passed, result
    assert abs(result.entropy_ratio - 1.0) < 1e-4
    assert "EVERY EARLIER TOKEN EQUALLY" in gates.attention_content_failure_message(result)


@pytest.fixture(scope="module")
def sabotage_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("sabotage_m05")
    (path / "sitecustomize.py").write_text(SITECUSTOMIZE)
    return path


def run_script(script, sabotage_dir, sabotage="", args=(), timeout=900):
    env = dict(os.environ)
    env.update(
        TINYTORCH_NON_INTERACTIVE="1",
        CI="true",
        PYTHONPATH=f"{sabotage_dir}{os.pathsep}{TINYTORCH_ROOT}",
        TT_SABOTAGE=sabotage,
        PYTHONUNBUFFERED="1",
        COLUMNS="160",
    )
    proc = subprocess.run(
        [sys.executable, str(MILESTONES_DIR / script), *args],
        cwd=TINYTORCH_ROOT, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, timeout=timeout,
    )
    if sabotage:
        assert f"[SABOTAGE] {sabotage}" in proc.stderr, "sabotage hook did not load"
    return proc


def describe(proc):
    return f"rc={proc.returncode}\n--- stdout (tail) ---\n{proc.stdout[-3000:]}\n--- stderr ---\n{proc.stderr[-2000:]}"


@pytest.mark.slow
def test_shakespeare_correct_code_passes(sabotage_dir):
    proc = run_script(SHAKESPEARE, sabotage_dir)
    assert proc.returncode == 0, describe(proc)


@pytest.mark.slow
def test_shakespeare_without_causal_mask_fails_on_causality(sabotage_dir):
    proc = run_script(SHAKESPEARE, sabotage_dir, "no_mask")
    assert proc.returncode == 1, describe(proc)
    assert "SEE FUTURE TOKENS" in proc.stdout, describe(proc)


@pytest.mark.slow
def test_shakespeare_with_attention_zeroed_fails(sabotage_dir):
    proc = run_script(SHAKESPEARE, sabotage_dir, "no_attn")
    assert proc.returncode == 1, describe(proc)


@pytest.mark.slow
def test_chat_with_attention_zeroed_fails(sabotage_dir):
    proc = run_script(CHAT, sabotage_dir, "no_attn")
    assert proc.returncode == 1, describe(proc)


@pytest.mark.slow
def test_tinycopilot_correct_code_passes(sabotage_dir):
    proc = run_script(COPILOT, sabotage_dir)
    assert proc.returncode == 0, describe(proc)


@pytest.mark.slow
def test_tinycopilot_with_attention_zeroed_fails(sabotage_dir):
    proc = run_script(COPILOT, sabotage_dir, "no_attn")
    assert proc.returncode == 1, describe(proc)


@pytest.mark.slow
def test_chat_correct_code_passes(sabotage_dir):
    proc = run_script(CHAT, sabotage_dir)
    assert proc.returncode == 0, describe(proc)


@pytest.mark.slow
def test_chat_without_causal_mask_fails_on_causality(sabotage_dir):
    proc = run_script(CHAT, sabotage_dir, "no_mask")
    assert proc.returncode == 1, describe(proc)
    assert "SEE FUTURE TOKENS" in proc.stdout, describe(proc)


@pytest.mark.slow
def test_chat_with_noop_train_epoch_fails(sabotage_dir):
    # The Trainer reports 0.0 but the weights never move: the NumPy-measured
    # train loss stays at the untrained value and held-out loss never drops.
    proc = run_script(CHAT, sabotage_dir, "no_train")
    assert proc.returncode == 1, describe(proc)
    assert "MILESTONE ACHIEVED" not in proc.stdout, describe(proc)


@pytest.mark.slow
@pytest.mark.parametrize("script", [SHAKESPEARE, COPILOT, CHAT])
def test_cross_entropy_returning_zero_fails(sabotage_dir, script):
    proc = run_script(script, sabotage_dir, "ce_zero")
    assert proc.returncode == 1, describe(proc)
    assert "CROSS-ENTROPY FORWARD RETURNS THE WRONG LOSS" in proc.stdout, describe(proc)


@pytest.mark.slow
@pytest.mark.parametrize("script", [SHAKESPEARE, COPILOT, CHAT])
def test_identity_positional_encoding_fails(sabotage_dir, script):
    proc = run_script(script, sabotage_dir, "pe_identity")
    assert proc.returncode == 1, describe(proc)
    assert "CANNOT TELL POSITIONS APART" in proc.stdout, describe(proc)


@pytest.mark.slow
def test_attention_part2_catches_zero_loss_forward(sabotage_dir):
    # Part 2 (required) trains through Module 06's backward only, so a
    # CrossEntropyLoss forward that returns 0 passed until it checked the value.
    ok = run_script("05_2017_transformer/02_vaswani_attention.py", sabotage_dir)
    assert ok.returncode == 0, describe(ok)
    proc = run_script("05_2017_transformer/02_vaswani_attention.py", sabotage_dir, sabotage="ce_zero")
    assert proc.returncode == 1, describe(proc)
    assert "WRONG LOSS" in proc.stdout, describe(proc)


@pytest.mark.parametrize("script", [
    "05_2017_transformer/01_tinygpt_shakespeare.py",
    "05_2017_transformer/03_tinycopilot.py",
    "05_2017_transformer/04_tinygpt_chat.py",
])
@pytest.mark.slow
def test_uniform_attention_fails_every_trained_part(sabotage_dir, script):
    # Measured entropy ratio: correct 0.56-0.71, uniform exactly 1.00.
    proc = run_script(script, sabotage_dir, sabotage="uniform_attn")
    assert proc.returncode == 1, describe(proc)
    assert "EVERY EARLIER TOKEN EQUALLY" in proc.stdout, describe(proc)
