"""Keep volume-owned sources and generated entry points aligned (2026-09-11)."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from binder.cli.core.config import ConfigManager
from binder.cli.core.volume_index import volume_index_source, write_volume_index

ROOT = Path(__file__).resolve().parents[2]
FORMATS = ("html", "pdf", "epub")


@pytest.mark.parametrize("format_type", FORMATS)
def test_volume_four_activation_uses_one_canonical_source(tmp_path, format_type):
    books = tmp_path / "books"
    (books / "config").mkdir(parents=True)
    source = volume_index_source(books, "vol4", format_type)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"# Volume four\r\nCanonical content.\r\n")
    (books / "config" / f"_quarto-{format_type}-vol4.yml").write_text("project:\n  type: book\n")
    manager = ConfigManager(tmp_path)
    manager.activate_config(format_type, "vol4")
    assert manager.active_index.read_bytes() == source.read_bytes()
    assert not manager.active_index.is_symlink()
    assert manager.active_index_source() == source.relative_to(books).as_posix()


@pytest.mark.parametrize("volume", ("vol1", "vol2", "vol3"))
@pytest.mark.parametrize("format_type", FORMATS)
def test_other_volumes_retain_existing_entrypoints(tmp_path, volume, format_type):
    source = tmp_path / f"index-{volume}.qmd"
    source.write_text(f"# {volume}\n")
    assert write_volume_index(tmp_path, volume, format_type) == source
    assert (tmp_path / "index.qmd").read_bytes() == source.read_bytes()


def test_switching_formats_and_old_symlinks_cannot_modify_sources(tmp_path):
    homepage = tmp_path / "vol4/index.qmd"
    preface = tmp_path / "vol4/frontmatter/about.qmd"
    preface.parent.mkdir(parents=True)
    homepage.write_text("# Homepage\n")
    preface.write_text("# Preface\n")
    (tmp_path / "index.qmd").symlink_to(homepage)
    for format_type, expected in (("pdf", preface), ("html", homepage), ("epub", preface)):
        write_volume_index(tmp_path, "vol4", format_type)
        assert (tmp_path / "index.qmd").read_bytes() == expected.read_bytes()
        assert homepage.read_text() == "# Homepage\n"
        assert preface.read_text() == "# Preface\n"


def test_missing_source_fails_instead_of_building_stale_content(tmp_path):
    (tmp_path / "index.qmd").write_text("Stale volume\n")
    with pytest.raises(FileNotFoundError, match="Volume entry point not found"):
        write_volume_index(tmp_path, "vol4", "pdf")


@pytest.mark.parametrize("format_type", FORMATS)
def test_ci_entrypoint_uses_same_selection(tmp_path, format_type):
    source = volume_index_source(tmp_path, "vol4", format_type)
    source.parent.mkdir(parents=True)
    source.write_text("# Canonical content\n")
    result = subprocess.run(
        [sys.executable, str(ROOT / "binder/cli/core/volume_index.py"),
         "--volume", "vol4", "--format", format_type.upper()],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "index.qmd").read_bytes() == source.read_bytes()


@pytest.mark.parametrize("format_type", FORMATS)
def test_real_configs_render_preface_once_and_resolve_sources(format_type):
    books = ROOT / "books"
    config = yaml.safe_load((books / "config" / f"_quarto-{format_type}-vol4.yml").read_text())
    entries = config["project"]["render"] if format_type == "html" else config["book"]["chapters"]

    def flatten(items):
        for entry in items:
            if isinstance(entry, str):
                yield entry
            else:
                yield from flatten(entry.get("chapters", []))

    source_paths = [volume_index_source(books, "vol4", format_type) if entry == "index.qmd"
                    else books / entry for entry in flatten(entries)]
    assert all(path.is_file() for path in source_paths)
    assert source_paths.count(books / "vol4/frontmatter/about.qmd") == 1
    assert len(source_paths) == len(set(source_paths))
    if format_type == "html":
        assert source_paths.count(books / "vol4/index.qmd") == 1
        assert config["website"]["sidebar"][0]["contents"][0]["href"] == "index.qmd"


def test_both_ci_platforms_use_shared_generator():
    workflow = yaml.safe_load((ROOT / ".github/workflows/book-build-container.yml").read_text())
    scripts = [step["run"] for job in workflow["jobs"].values() for step in job.get("steps", [])
               if "run" in step and "volume_index.py" in step["run"]]
    assert len(scripts) == 2
    assert all('--volume' in script and '--format' in script for script in scripts)


@pytest.mark.parametrize("format_type", FORMATS)
@pytest.mark.parametrize("quoted", (False, True))
def test_relocated_preface_includes_still_reach_canonical_files(tmp_path, quoted, format_type):
    import re

    books = tmp_path / "books"
    source = books / "vol4/frontmatter/about.qmd"
    included = books / "shared/_includes/living textbook.qmd"
    (books / "config").mkdir(parents=True)
    source.parent.mkdir(parents=True)
    included.parent.mkdir(parents=True)
    included.write_text("Canonical included content.\n")
    relative = "../../shared/_includes/living textbook.qmd" if quoted else "../../shared/_includes/living.qmd"
    if not quoted:
        included = included.with_name("living.qmd")
        included.write_text("Canonical included content.\n")
    argument = f'"{relative}"' if quoted else relative
    original = f"# Preface\r\n\r\n{{{{< include {argument} >}}}}\r\n".encode()
    source.write_bytes(original)
    write_volume_index(books, "vol4", format_type)
    copied = (books / "index.qmd").read_text()
    target = re.search(r'\{\{< include (.*?) >\}\}', copied).group(1).strip('"')
    assert (books / target).resolve() == included.resolve()
    assert (books / target).is_file()
    assert source.read_bytes() == original
    assert ConfigManager(tmp_path).active_index_source() == "vol4/frontmatter/about.qmd"
    assert b"\r\n" in (books / "index.qmd").read_bytes()
