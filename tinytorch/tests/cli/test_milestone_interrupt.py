"""
Ctrl-C during `tito milestone run` belongs to the milestone script.

2026-09-29: tito caught the Ctrl-C a student pressed to leave the post-pass
try-it prompt and printed "Part not recorded", dropping a pass the script had
already earned. Now the script decides: it exits 0 after a pass (recorded) or
dies of SIGINT mid-run (not recorded). These tests send real signals, so they
run on POSIX only.
"""

import os
import signal
import sys
import threading

import pytest

from tito.commands.milestone import INTERRUPTED_EXIT_CODES, run_milestone_script

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")


def _signal_self_soon(delay=0.5):
    timer = threading.Timer(delay, os.kill, args=(os.getpid(), signal.SIGINT))
    timer.start()
    return timer


def test_ctrl_c_reaching_tito_does_not_stop_the_script(tmp_path):
    script = tmp_path / "passes.py"
    script.write_text("import time\ntime.sleep(1.5)\nraise SystemExit(0)\n")
    timer = _signal_self_soon()
    try:
        # Before the fix this raised KeyboardInterrupt in tito.
        assert run_milestone_script(script) == 0
    finally:
        timer.cancel()


def test_script_killed_by_ctrl_c_counts_as_interrupted(tmp_path):
    script = tmp_path / "interrupted.py"
    script.write_text("import os, signal\nos.kill(os.getpid(), signal.SIGINT)\n"
                      "import time\ntime.sleep(5)\n")
    assert run_milestone_script(script) in INTERRUPTED_EXIT_CODES


def test_handler_is_restored_after_the_run(tmp_path):
    script = tmp_path / "quick.py"
    script.write_text("raise SystemExit(3)\n")
    before = signal.getsignal(signal.SIGINT)
    assert run_milestone_script(script) == 3
    assert signal.getsignal(signal.SIGINT) is before
