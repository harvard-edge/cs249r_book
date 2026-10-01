"""
"Try it" prompts after a milestone passes (milestones/try_it.py).

2026-09-29: TinyCopilot (Milestone 05 Part 3) and the KV-cache part of
Milestone 06 end with a prompt where the student types their own input. The
prompt is play, not grading, so these tests pin the contract: it never runs
without a person at a terminal (CI, pipes, --non-interactive, the recorded
demos), it cannot change a milestone's exit code, and it ends on an empty
line, an exit word, Ctrl-D, or Ctrl-C.
"""

import importlib.util
import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from rich.console import Console

TINYTORCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TINYTORCH_ROOT))
from milestones.try_it import try_it, try_it_enabled  # noqa: E402

M05 = TINYTORCH_ROOT / "milestones" / "05_2017_transformer"
M06 = TINYTORCH_ROOT / "milestones" / "06_2018_mlperf"


def _reader(*lines, end=EOFError):
    """Return a read() that yields ``lines`` and then raises ``end``."""
    queue = list(lines)

    def read():
        if queue:
            return queue.pop(0)
        raise end()
    return read


def _console():
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def _load(path, name):
    folder = str(path.parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# The loop contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stop", ["", "   ", "exit", "QUIT", "q"])
def test_loop_stops_on_empty_line_or_exit_word(stop):
    seen = []
    handled = try_it(_console(), "intro", "> ", seen.append,
                     read=_reader("def f(x):", stop, "never read"))
    assert seen == ["def f(x):"]
    assert handled == 1


@pytest.mark.parametrize("end", [EOFError, KeyboardInterrupt])
def test_loop_stops_on_ctrl_d_and_ctrl_c(end):
    seen = []
    assert try_it(_console(), "intro", "> ", seen.append,
                  read=_reader("a", "b", end=end)) == 2
    assert seen == ["a", "b"]


def test_error_in_response_is_reported_and_the_loop_continues():
    console = _console()
    seen = []

    def respond(text):
        if text == "bad":
            raise ValueError("shape mismatch")
        seen.append(text)

    assert try_it(console, "intro", "> ", respond, read=_reader("bad", "good", "")) == 2
    assert seen == ["good"]
    assert "shape mismatch" in console.file.getvalue()


def test_ctrl_c_during_a_response_ends_the_loop_quietly():
    def respond(text):
        raise KeyboardInterrupt

    assert try_it(_console(), "intro", "> ", respond, read=_reader("a", "b")) == 0


@pytest.mark.parametrize("var,value", [
    ("CI", "true"),
    ("TINYTORCH_NON_INTERACTIVE", "1"),
])
def test_opt_outs_disable_the_prompt(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    assert not try_it_enabled()


def test_without_a_terminal_the_prompt_never_reads(monkeypatch):
    # pytest's stdin is not a TTY; reaching console.input would raise here.
    for var in ("CI", "TINYTORCH_NON_INTERACTIVE"):
        monkeypatch.delenv(var, raising=False)
    console = _console()
    console.input = lambda *a, **k: pytest.fail("try_it read input without a terminal")
    assert try_it(console, "intro", "> ", lambda t: None) == 0
    assert console.file.getvalue() == ""


def test_demo_recordings_opt_out():
    recorder = (TINYTORCH_ROOT / "guide" / "tools" / "record_tapes.py").read_text()
    assert 'env["TINYTORCH_NON_INTERACTIVE"] = "1"' in recorder


# ---------------------------------------------------------------------------
# Milestone 06 Part 2: YOUR cache timed on the student's own sentence
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def m06_part2():
    return _load(M06 / "02_generation_speedup.py", "m06_part2_try_it")


def test_sentence_maps_to_the_27_token_alphabet(m06_part2):
    tokens = m06_part2.text_to_tokens("Hi, Bob!", max_len=64)
    assert tokens.tolist() == [[8, 9, 0, 2, 15, 2]]
    assert m06_part2.text_to_tokens("x" * 100, max_len=64).shape == (1, 64)
    assert m06_part2.text_to_tokens("123 !?", max_len=64).shape == (1, 1)


def test_m06_try_it_times_the_students_cache_on_their_sentence(m06_part2):
    model = m06_part2.GPT(vocab_size=27, embed_dim=32, num_layers=2, num_heads=2, max_seq_len=64)
    console = _console()
    handled = m06_part2.try_cache_lengths(
        model, console, read=_reader("hello", "the cache grows with the sentence", "!!!", ""))
    out = console.file.getvalue()
    assert handled == 3
    assert " 5 tokens:" in out and "33 tokens:" in out
    assert "outputs identical" in out
    assert "did not run" not in out
    assert "Type some letters" in out
    # Leaves the model uncached, as the milestone's own comparison does.
    assert not getattr(model, "_cache_enabled", False)


# ---------------------------------------------------------------------------
# Milestone 05 Part 3: YOUR TinyCopilot completes the student's prompt
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def m05_part3():
    return _load(M05 / "03_tinycopilot.py", "m05_part3_try_it")


def test_m05_try_it_completes_and_checks_the_students_prompt(m05_part3):
    from tinytorch.core.tokenization import CharTokenizer

    text = "def add(a, b):\n    return a + b\n"
    tokenizer = CharTokenizer()
    tokenizer.build_vocab([text])
    model, _ = m05_part3.build_model(vocab_size=tokenizer.vocab_size)
    console = _console()
    m05_part3.console = console
    handled = m05_part3.try_completions(
        model, tokenizer, should_stream=False,
        read=_reader("def add(a, b):", "def ~~~", "d" * 81, ""))
    out = console.file.getvalue()
    assert handled == 3
    assert "did not run" not in out
    assert "YOUR TinyCopilot |" in out  # the result panel, not the intro
    assert ("Valid AST" in out) or ("SyntaxError" in out) or ("No completion" in out)
    assert "never saw" in out
    assert "under 80 characters" in out


# ---------------------------------------------------------------------------
# End to end: a piped run of the milestone never waits for input
# ---------------------------------------------------------------------------

def test_piped_milestone_run_exits_without_prompting():
    env = {k: v for k, v in os.environ.items()
           if k not in ("CI", "TINYTORCH_NON_INTERACTIVE")}
    result = subprocess.run(
        [sys.executable, str(M06 / "02_generation_speedup.py")],
        cwd=TINYTORCH_ROOT, env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "MILESTONE 06.2 COMPLETE" in result.stdout
    assert "Try it" not in result.stdout
