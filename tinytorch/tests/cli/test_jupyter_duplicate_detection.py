"""
Tests for _find_running_jupyter_lab_pids(), the duplicate-process guard
for `tito module start`/`resume`.

Repeated `tito module resume` calls previously spawned a brand new
Jupyter Lab server process every time, with nothing tracking or
cleaning up the previous one, an unbounded resource leak. These tests
mock psutil directly rather than spawning real Jupyter processes, which
would be slow and environment-fragile in CI.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tito.commands.module.workflow import _find_running_jupyter_lab_pids


def _make_proc(pid, cmdline):
    proc = MagicMock()
    proc.info = {"pid": pid, "cmdline": cmdline}
    return proc


class TestFindRunningJupyterLabPids:
    def test_returns_none_when_psutil_not_installed(self):
        with patch.dict(sys.modules, {"psutil": None}):
            result = _find_running_jupyter_lab_pids()

        assert result is None

    def test_returns_empty_list_when_no_jupyter_running(self):
        fake_procs = [
            _make_proc(100, ["python", "some_script.py"]),
            _make_proc(101, ["node", "server.js"]),
        ]
        fake_psutil = MagicMock()
        fake_psutil.process_iter.return_value = fake_procs
        fake_psutil.NoSuchProcess = Exception
        fake_psutil.AccessDenied = Exception

        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            result = _find_running_jupyter_lab_pids()

        assert result == []

    def test_finds_running_jupyter_lab_process(self):
        fake_procs = [
            _make_proc(200, ["python", "-m", "jupyter", "lab", "--notebook-dir=."]),
        ]
        fake_psutil = MagicMock()
        fake_psutil.process_iter.return_value = fake_procs
        fake_psutil.NoSuchProcess = Exception
        fake_psutil.AccessDenied = Exception

        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            result = _find_running_jupyter_lab_pids()

        assert result == [200]

    def test_ignores_processes_matching_jupyter_but_not_lab(self):
        """'jupyter kernelspec' or 'jupyter notebook' aren't the target
        server process this guard is meant to deduplicate against."""
        fake_procs = [
            _make_proc(300, ["jupyter", "kernelspec", "list"]),
        ]
        fake_psutil = MagicMock()
        fake_psutil.process_iter.return_value = fake_procs
        fake_psutil.NoSuchProcess = Exception
        fake_psutil.AccessDenied = Exception

        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            result = _find_running_jupyter_lab_pids()

        assert result == []

    def test_skips_processes_that_raise_during_inspection(self):
        """A process that exits between process_iter() listing it and
        reading its cmdline (psutil.NoSuchProcess) must not crash the
        whole scan, just be skipped."""
        vanished_proc = MagicMock()
        vanished_proc.info = {"pid": 400}

        class FakeNoSuchProcess(Exception):
            pass

        def raise_on_get(key, default=None):
            raise FakeNoSuchProcess()

        vanished_proc.info = MagicMock()
        vanished_proc.info.get.side_effect = raise_on_get

        fake_psutil = MagicMock()
        fake_psutil.process_iter.return_value = [vanished_proc]
        fake_psutil.NoSuchProcess = FakeNoSuchProcess
        fake_psutil.AccessDenied = Exception

        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            result = _find_running_jupyter_lab_pids()

        assert result == []

    def test_returns_none_on_unexpected_scan_failure(self):
        """Any other unexpected failure scanning processes (e.g. a
        platform-specific psutil error) must degrade to 'unknown' rather
        than crash the module start/resume flow."""
        fake_psutil = MagicMock()
        fake_psutil.process_iter.side_effect = RuntimeError("platform error")
        fake_psutil.NoSuchProcess = Exception
        fake_psutil.AccessDenied = Exception

        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            result = _find_running_jupyter_lab_pids()

        assert result is None
