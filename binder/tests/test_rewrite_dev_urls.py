from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "rewrite-dev-urls.sh"


def test_volume_release_manifest_becomes_page_relative(tmp_path: Path) -> None:
    root_page = tmp_path / "index.html"
    chapter_dir = tmp_path / "introduction"
    chapter_dir.mkdir()
    chapter_page = chapter_dir / "introduction.html"
    metadata = '<meta name="release-manifest" content="/vol1/release-manifest.json">'
    root_page.write_text(metadata, encoding="utf-8")
    chapter_page.write_text(metadata, encoding="utf-8")

    subprocess.run(
        ["bash", str(SCRIPT), "vol1", str(tmp_path), "--live-external"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert 'content="./release-manifest.json"' in root_page.read_text(encoding="utf-8")
    assert 'content="../release-manifest.json"' in chapter_page.read_text(encoding="utf-8")

