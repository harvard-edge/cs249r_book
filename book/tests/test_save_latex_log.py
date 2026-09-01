from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "quarto" / "scripts" / "save_latex_log.py"
SPEC = importlib.util.spec_from_file_location("save_latex_log", SCRIPT)
assert SPEC and SPEC.loader
save_latex_log = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(save_latex_log)


def test_active_volume_follows_binder_config_link(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    volume_config = config_dir / "_quarto-pdf-vol2.yml"
    volume_config.write_text("project: {}\n", encoding="utf-8")
    (tmp_path / "_quarto.yml").symlink_to(volume_config.relative_to(tmp_path))

    assert save_latex_log._active_volume(tmp_path) == "vol2"


def test_find_intermediate_accepts_quarto_index_stem(tmp_path: Path) -> None:
    aux = tmp_path / "index.aux"
    aux.write_text("\\newlabel{sec-test}{{1}{1}}\n", encoding="utf-8")

    assert save_latex_log._find_intermediate(tmp_path, "Book-Vol2", ".aux") == aux


def test_regenerate_auxiliary_files_uses_draft_mode(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "Book-Vol2.tex").write_text("document", encoding="utf-8")
    observed = []

    def fake_run(command, **kwargs):
        observed.append((command, kwargs))
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(save_latex_log.subprocess, "run", fake_run)

    assert save_latex_log._regenerate_auxiliary_files(tmp_path, "Book-Vol2")
    assert len(observed) == 3
    assert all("-draftmode" in command for command, _ in observed)
    assert all(command[-1] == "Book-Vol2.tex" for command, _ in observed)
    assert all(kwargs["cwd"] == tmp_path for _, kwargs in observed)


def test_regenerate_auxiliary_files_stops_after_failed_pass(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "Book-Vol2.tex").write_text("document", encoding="utf-8")
    returncodes = iter((0, 1, 0))
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return type("Completed", (), {"returncode": next(returncodes)})()

    monkeypatch.setattr(save_latex_log.subprocess, "run", fake_run)

    assert not save_latex_log._regenerate_auxiliary_files(tmp_path, "Book-Vol2")
    assert len(calls) == 2
