"""Keep the SocratiQ reader wired into every HTML volume build."""

import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
VOLUMES = ("vol1", "vol2", "vol3", "vol4")
READER_SRC = "/shared/assets/scripts/socratiq/reader.js"
READER_RESOURCE = "/shared/assets/scripts/socratiq/**"
READER_ENDPOINT = "https://proxy-worker-streaming.mlsysbook.workers.dev/ai/stream"


class ScriptTagCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.scripts.append(dict(attrs))


@pytest.mark.parametrize("volume", VOLUMES)
def test_html_config_loads_and_stages_socratiq_reader(volume):
    books = ROOT / "books"
    config_path = books / "config" / f"_quarto-html-{volume}.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    resources = config["project"].get("resources", [])
    assert READER_RESOURCE in resources

    # Quarto resolves leading-slash resource entries from the project root and
    # preserves their relative path under output-dir. Ensure the reader source
    # exists and maps to the URL requested by the HTML include.
    source = books / READER_SRC.lstrip("/")
    assert source.is_file(), f"missing Quarto resource source: {source}"
    staged_path = Path(READER_SRC.lstrip("/"))
    assert staged_path.as_posix() == "shared/assets/scripts/socratiq/reader.js"

    include_text = "\n".join(
        item.get("text", "")
        for item in config["format"]["html"].get("include-in-header", [])
        if isinstance(item, dict)
    )
    parser = ScriptTagCollector()
    parser.feed(include_text)

    reader_tags = [tag for tag in parser.scripts if tag.get("src") == READER_SRC]
    assert reader_tags == [
        {
            "src": READER_SRC,
            "data-volume": volume,
            "data-endpoint": READER_ENDPOINT,
            "defer": None,
        }
    ]

    # The old, inactive bundle URL must not remain as a competing reader include.
    assert "/tools/scripts/socratiQ/bundle.js" not in include_text


@pytest.mark.skipif(shutil.which("quarto") is None, reason="Quarto required for resource staging")
def test_quarto_stages_reader_under_site_root(tmp_path):
    books = ROOT / "books"
    source = books / READER_SRC.lstrip("/")
    assert source.is_file(), f"missing Quarto resource source: {source}"

    # Render a minimal website with the same project-relative resource pattern
    # and deployment prefix used by the volume configs.
    project = tmp_path / "project"
    staged_source = project / READER_SRC.lstrip("/")
    staged_source.parent.mkdir(parents=True)
    shutil.copy2(source, staged_source)
    (project / "index.qmd").write_text("# Resource staging check\n", encoding="utf-8")
    chapter = project / "vol1" / "01_chapter" / "01_chapter.qmd"
    chapter.parent.mkdir(parents=True)
    chapter.write_text("# Nested chapter\n", encoding="utf-8")
    fixture_config = {
        "project": {
            "type": "website",
            "output-dir": "_build/html",
            "resources": [READER_RESOURCE],
            "render": ["index.qmd", "vol1/01_chapter/01_chapter.qmd"],
        },
        "website": {
            "site-url": "https://mlsysbook.ai/vol1/",
        },
        "format": {
            "html": {
                "include-in-header": [
                    {
                        "text": (
                            f'<script src="{READER_SRC}" data-volume="vol1" '
                            f'data-endpoint="{READER_ENDPOINT}" defer></script>'
                        )
                    }
                ]
            }
        },
    }
    (project / "_quarto.yml").write_text(
        yaml.safe_dump(fixture_config, sort_keys=False), encoding="utf-8"
    )

    subprocess.run(
        ["quarto", "render", "--quiet"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )

    output = project / "_build/html"
    flatten = ROOT / ".github" / "scripts" / "flatten-vol-urls.sh"
    flattened = subprocess.run(
        ["bash", str(flatten), str(output), "vol1"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "vol1 flattened" in flattened.stdout
    staged_reader = output / READER_SRC.lstrip("/")
    assert staged_reader.is_file()
    assert staged_reader.read_bytes() == source.read_bytes()

    root_html = (output / "index.html").read_text(encoding="utf-8")
    chapter_html = (output / "01_chapter" / "01_chapter.html").read_text(encoding="utf-8")
    root_parser = ScriptTagCollector()
    root_parser.feed(root_html)
    chapter_parser = ScriptTagCollector()
    chapter_parser.feed(chapter_html)
    root_tags = [tag for tag in root_parser.scripts if tag.get("data-volume") == "vol1"]
    chapter_tags = [tag for tag in chapter_parser.scripts if tag.get("data-volume") == "vol1"]
    assert len(root_tags) == len(chapter_tags) == 1
    assert root_tags[0].get("data-endpoint") == chapter_tags[0].get("data-endpoint") == READER_ENDPOINT

    # The production site is mounted at /vol1/. Quarto makes the root include
    # relative, and the flatten script adjusts the nested page after moving it
    # up one directory. Both references must resolve to the deployed asset.
    root_asset_url = urljoin("https://mlsysbook.ai/vol1/", root_tags[0]["src"])
    chapter_url = urljoin("https://mlsysbook.ai/vol1/", "01_chapter/01_chapter.html")
    chapter_asset_url = urljoin(chapter_url, chapter_tags[0]["src"])
    expected_asset_url = "https://mlsysbook.ai/vol1/shared/assets/scripts/socratiq/reader.js"
    assert root_asset_url == expected_asset_url
    assert chapter_asset_url == expected_asset_url
    deployed_asset_path = f"vol1/{READER_SRC.lstrip('/')}"
    assert urlparse(root_asset_url).path.lstrip("/") == deployed_asset_path
