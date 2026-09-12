"""Worktree-isolated parallel builds and the debugger built on them (2026-09-12).

The runner tests create a real throwaway git repository whose ``binder/binder``
is a stand-in script that writes the artifact a real build would, so git
worktrees, snapshots, process handling, and output collection are exercised
without Quarto.
"""

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from binder.cli.commands import debug as debug_module
from binder.cli.core import workspace
from binder.cli.core.discovery import ChapterDiscovery
from binder.cli.core.parallel import (
    DEFAULT_WORKERS,
    BuildJob,
    BuildSession,
    JobResult,
    plan_build_jobs,
    run_jobs,
)
from binder.cli.core.process import (
    interrupts_as_keyboard_interrupt,
    start_process_group,
    stop_process_group,
)
from binder.cli.main import MLSysBookCLI

FAKE_BINDER = textwrap.dedent('''\
    #!/usr/bin/env python3
    """Stand-in for binder: writes the artifact a real build would."""
    import os
    import sys
    import time
    from pathlib import Path

    args = [arg for arg in sys.argv[1:] if arg != "-v"]
    assert args[0] == "build", args
    fmt = args[1]
    volume = next(arg[2:] for arg in args if arg.startswith("--vol"))
    positional = [arg for arg in args[2:] if not arg.startswith("--")]
    chapter = positional[0] if positional else None
    if chapter == "broken":
        print("renderer failed")
        sys.exit(3)
    if chapter == "slow":
        time.sleep(60)
    books = Path("books")
    output = books / "_build" / f"{fmt}-{volume}"
    if chapter:
        output = output / "chapters" / chapter
    output.mkdir(parents=True, exist_ok=True)
    name = "index.html" if fmt == "html" else f"{chapter or volume}.{fmt}"
    untracked = books / "untracked.txt"
    (output / name).write_text(
        (books / "marker.txt").read_text()
        + (untracked.read_text() if untracked.exists() else "")
        + os.environ.get("XDG_CACHE_HOME", "shared-cache")
    )
    print("built", fmt, volume, chapter)
''')


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "core.hooksPath=/dev/null", *args],
        check=True, capture_output=True, text=True,
    ).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "binder").mkdir(parents=True)
    (root / "binder" / "binder").write_text(FAKE_BINDER)
    books = root / "books"
    (books / "config").mkdir(parents=True)
    for volume in ("vol1", "vol2"):
        for fmt in ("html", "pdf"):
            (books / "config" / f"_quarto-{fmt}-{volume}.yml").write_text(
                yaml.safe_dump({"project": {"output-dir": f"_build/{fmt}-{volume}"}}))
    (books / "marker.txt").write_text("committed\n")
    (root / ".gitignore").write_text("books/_build/\n")
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-q", "-m", "fixture")
    monkeypatch.setattr(workspace, "WORKSPACE_ROOT", tmp_path / "workspaces")
    return root


def registered_worktrees(repo: Path) -> int:
    return git(repo, "worktree", "list", "--porcelain").count("worktree ")


def test_snapshot_carries_uncommitted_and_untracked_files(repo):
    (repo / "books" / "marker.txt").write_text("uncommitted\n")
    (repo / "books" / "untracked.txt").write_text("new file\n")
    snapshot = workspace.take_snapshot(repo)
    created = workspace.create_workspace(snapshot, workspace.workspace_path("run", "w1"))
    try:
        assert (created.path / "books" / "marker.txt").read_text() == "uncommitted\n"
        assert (created.path / "books" / "untracked.txt").read_text() == "new file\n"
        assert registered_worktrees(repo) == 2
    finally:
        workspace.remove_workspace(created)
    assert not created.path.exists()
    assert registered_worktrees(repo) == 1
    assert git(repo, "stash", "list") == ""
    assert (repo / "books" / "marker.txt").read_text() == "uncommitted\n"


def test_workspaces_cannot_point_outside_the_workspace_root(repo, tmp_path):
    snapshot = workspace.take_snapshot(repo)
    with pytest.raises(ValueError):
        workspace.create_workspace(snapshot, tmp_path / "elsewhere")
    with pytest.raises(ValueError):
        workspace.remove_workspace(workspace.Workspace(repo, snapshot))
    assert (repo / "books" / "marker.txt").exists()


def test_run_jobs_collects_output_and_logs_then_removes_worktrees(repo, tmp_path):
    (repo / "books" / "marker.txt").write_text("uncommitted\n")
    jobs = [BuildJob("pdf", "vol1"), BuildJob("html", "vol2"), BuildJob("pdf", "vol1", "broken")]
    started, finished = [], []
    results = run_jobs(repo, jobs, tmp_path / "run", workers=2,
                       on_start=started.append, on_finish=finished.append)

    assert [result.ok for result in results] == [True, True, False]
    assert sorted(job.name for job in started) == sorted(job.name for job in jobs)
    assert len(finished) == 3
    pdf = results[0]
    assert pdf.artifact == tmp_path / "run" / "vol1-pdf" / "output" / "vol1.pdf"
    assert pdf.artifact.read_text().startswith("uncommitted\n")
    assert ".binder-cache" in pdf.artifact.read_text()  # private diagram cache per worker
    assert results[1].artifact.name == "index.html"
    broken = results[2]
    assert broken.returncode == 3 and broken.note == "exit code 3"
    assert "renderer failed" in broken.log_text()
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert [job["job"] for job in summary["jobs"]] == ["vol1-pdf", "vol2-html", "vol1-pdf-broken"]
    assert registered_worktrees(repo) == 1
    assert (repo / "books" / "marker.txt").read_text() == "uncommitted\n"


def test_run_jobs_rejects_duplicate_job_names(repo, tmp_path):
    with pytest.raises(ValueError, match="unique"):
        run_jobs(repo, [BuildJob("pdf", "vol1"), BuildJob("pdf", "vol1")], tmp_path / "run")


def test_session_reuses_one_workspace_and_prepare_never_touches_the_checkout(repo, tmp_path):
    snapshot = workspace.take_snapshot(repo)
    with BuildSession(snapshot, tmp_path / "steps", name="bisect", run_id="run") as session:
        edited = session.run(
            BuildJob("pdf", "vol1", "chapter", label="edited"),
            prepare=lambda root: (root / "books" / "marker.txt").write_text("edited in workspace\n"))
        first_path = session.path
        plain = session.run(BuildJob("pdf", "vol1", "chapter", label="plain"))
        assert session.path == first_path
    assert edited.ok and edited.artifact.read_text().startswith("edited in workspace\n")
    assert plain.ok
    assert (repo / "books" / "marker.txt").read_text() == "committed\n"
    assert not first_path.exists()
    assert registered_worktrees(repo) == 1


def test_a_build_that_exceeds_its_timeout_is_stopped(repo, tmp_path):
    snapshot = workspace.take_snapshot(repo)
    with BuildSession(snapshot, tmp_path / "steps", name="slow", run_id="run", timeout_seconds=1) as session:
        result = session.run(BuildJob("pdf", "vol1", "slow"))
    assert not result.ok
    assert result.returncode is None
    assert result.note == "timed out after 1s"


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
def test_stop_process_group_kills_a_child_that_ignores_sigterm():
    process = start_process_group([
        sys.executable, "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(60)",
    ], stdout=subprocess.PIPE, text=True)
    assert process.stdout.readline().strip() == "ready"
    started = time.monotonic()
    stop_process_group(process, grace_seconds=0.5)
    assert process.returncode == -signal.SIGKILL
    assert time.monotonic() - started < 10


def test_interrupt_context_raises_keyboard_interrupt_and_restores_handlers():
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    with interrupts_as_keyboard_interrupt():
        with pytest.raises(KeyboardInterrupt):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    assert {sig: signal.getsignal(sig) for sig in previous} == previous


@pytest.fixture
def discovery(tmp_path):
    books = tmp_path / "books"
    for volume, stems in (("vol1", ("01_intro", "02_training")), ("vol2", ("01_fleet",))):
        for stem in stems:
            chapter = books / volume / stem / f"{stem}.qmd"
            chapter.parent.mkdir(parents=True)
            chapter.write_text(f"# {stem.split('_', 1)[1].title()}\n")
        config = books / "config" / f"_quarto-pdf-{volume}.yml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(yaml.safe_dump({"book": {"chapters": [
            "index.qmd", *(f"{volume}/{stem}/{stem}.qmd" for stem in stems)]}}))
    return ChapterDiscovery(books)


def test_plan_builds_every_format_and_volume_with_pdf_only_flags(discovery):
    jobs = plan_build_jobs(discovery, ["html", "pdf"], ["vol1", "vol2"],
                           extra_args=["--skip-validate"], volume_pdf_args=["--no-cover"])
    assert [job.name for job in jobs] == ["vol1-html", "vol1-pdf", "vol2-html", "vol2-pdf"]
    assert jobs[0].extra_args == ("--skip-validate",)
    assert jobs[1].extra_args == ("--skip-validate", "--no-cover")


def test_plan_resolves_chapter_patterns_and_each_chapter(discovery):
    jobs = plan_build_jobs(discovery, ["pdf"], ["vol1"], ["0*"])
    assert [job.chapter for job in jobs] == ["01_intro", "02_training"]
    jobs = plan_build_jobs(discovery, ["pdf"], ["vol1", "vol2"], each_chapter=True)
    assert [(job.volume, job.chapter) for job in jobs] == [
        ("vol1", "01_intro"), ("vol1", "02_training"), ("vol2", "01_fleet")]
    assert jobs[0].binder_args() == ["build", "pdf", "01_intro", "--vol1"]


@pytest.mark.parametrize("kwargs, message", [
    (dict(formats=["docx"], volumes=["vol1"]), "Unknown format"),
    (dict(formats=["pdf"], volumes=[]), "Select volumes"),
    (dict(formats=["pdf"], volumes=["vol1", "vol2"], chapters=["01_intro"]), "exactly one volume"),
    (dict(formats=["pdf"], volumes=["vol1"], chapters=["01_intro"], each_chapter=True), "not both"),
    (dict(formats=["pdf"], volumes=["tinytorch"]), "Parallel builds support"),
])
def test_plan_rejects_invalid_requests(discovery, kwargs, message):
    with pytest.raises(ValueError, match=message):
        plan_build_jobs(discovery, **kwargs)


@pytest.mark.parametrize("args, expected", [
    (["pdf", "--vol1"], (None, False, False, ["pdf", "--vol1"])),
    (["pdf", "--vol1", "--parallel", "3"], (3, False, False, ["pdf", "--vol1"])),
    (["--parallel=2", "html"], (2, False, False, ["html"])),
    (["html", "--parallel"], (DEFAULT_WORKERS, False, False, ["html"])),
    (["pdf", "--each-chapter"], (1, False, True, ["pdf"])),
    (["pdf", "--parallel", "--keep-workspaces"], (DEFAULT_WORKERS, True, False, ["pdf"])),
])
def test_parallel_flags_are_separated_from_build_arguments(args, expected):
    assert MLSysBookCLI._extract_parallel_options(args) == expected


@pytest.mark.parametrize("args", (["--parallel", "0"], ["--parallel=many"]))
def test_parallel_worker_count_must_be_positive(args):
    with pytest.raises(ValueError):
        MLSysBookCLI._extract_parallel_options(args)


def test_cli_parallel_build_plans_jobs_and_reports_failure(discovery, monkeypatch, tmp_path):
    import cli.core.parallel as cli_parallel

    captured = {}

    def fake_run_jobs(repo_root, jobs, run_dir, *, workers, keep_workspaces, on_start, on_finish):
        captured.update(jobs=jobs, workers=workers, run_dir=run_dir)
        results = []
        for job in jobs:
            log = tmp_path / f"{job.name}.log"
            log.write_text("")
            results.append(cli_parallel.JobResult(job, job.format_type == "html", 0, 1.0, log))
        return results

    monkeypatch.setattr(cli_parallel, "run_jobs", fake_run_jobs)
    cli = object.__new__(MLSysBookCLI)
    cli.config_manager = SimpleNamespace(book_dir=tmp_path / "books")
    cli.chapter_discovery = discovery
    assert not cli.handle_build_command(
        ["html,pdf", "--vol1", "--vol2", "--parallel", "2", "--skip-validate", "--print-marks"])
    assert captured["workers"] == 2
    assert [job.name for job in captured["jobs"]] == ["vol1-html", "vol1-pdf", "vol2-html", "vol2-pdf"]
    assert captured["jobs"][1].extra_args == ("--skip-validate", "--print-marks")
    assert captured["run_dir"].parent == tmp_path / "books" / "_build" / "parallel"


def test_cli_parallel_build_rejects_layout(discovery, tmp_path):
    cli = object.__new__(MLSysBookCLI)
    cli.config_manager = SimpleNamespace(book_dir=tmp_path / "books")
    cli.chapter_discovery = discovery
    assert not cli.handle_build_command(["pdf", "--vol1", "--layout", "--parallel"])


def _debugger(tmp_path, chapters):
    config_manager = SimpleNamespace(book_dir=tmp_path / "books")
    chapter_discovery = Mock()
    chapter_discovery.get_chapters_from_config.return_value = chapters
    return debug_module.DebugCommand(config_manager, chapter_discovery)


def test_debug_scan_builds_every_chapter_and_bisects_only_failures(monkeypatch, tmp_path):
    debugger = _debugger(tmp_path, ["intro", "training"])
    calls = {}

    def fake_run_jobs(repo_root, jobs, run_dir, *, workers, keep_workspaces, on_finish):
        calls.update(jobs=jobs, workers=workers, repo_root=repo_root)
        results = []
        for job in jobs:
            log = tmp_path / f"{job.name}.log"
            log.write_text("Duplicate note reference 'fn-x'\n")
            result = JobResult(job, job.chapter == "intro", 0, 2.0, log)
            on_finish(result)
            results.append(result)
        return results

    monkeypatch.setattr(debug_module, "run_jobs", fake_run_jobs)
    debugger._bisect_chapter = Mock(return_value=True)
    assert debugger.debug_build("pdf", "vol1", workers=3)
    assert calls["workers"] == 3
    assert calls["repo_root"] == tmp_path
    assert [job.label for job in calls["jobs"]] == ["01_intro", "02_training"]
    assert all(job.extra_args == ("--skip-validate",) for job in calls["jobs"])
    debugger._bisect_chapter.assert_called_once()
    assert debugger._bisect_chapter.call_args.args[0] == "training"


def test_debug_bisection_writes_truncated_chapters_only_into_the_workspace(tmp_path):
    debugger = _debugger(tmp_path, [])
    workspace_root = tmp_path / "workspace"
    relative = Path("books/vol1/training/training.qmd")
    (workspace_root / relative).parent.mkdir(parents=True)
    sections = [SimpleNamespace(index=i, title=f"Section {i}", content=f"## Section {i}\n")
                for i in range(6)]
    sections[3].content += "BROKEN\n"
    chapter = SimpleNamespace(frontmatter="---\ntitle: T\n---", pre_content="# Training", sections=sections)

    class FakeSession:
        labels = []

        def run(self, job, prepare=None):
            prepare(workspace_root)
            text = (workspace_root / relative).read_text()
            self.labels.append(job.label)
            log = tmp_path / f"{job.label}.log"
            log.write_text("")
            return JobResult(job, "BROKEN" not in text, 0, 0.1, log)

    session = FakeSession()
    failing = debugger._binary_search_sections(session, chapter, "training", "vol1", "pdf", relative)
    assert failing == 3
    assert session.labels[:2] == ["preamble", "full"]
