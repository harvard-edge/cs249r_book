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
STUDY_SRC = "/shared/assets/scripts/socratiq/study.js"
READER_RESOURCE = "/shared/assets/scripts/socratiq/**"
READER_ENDPOINT = "https://mlsysbook.ai/api/socratiq/ai/stream"
GUIDE_SRC = "shared/frontmatter/socratiq/socratiq.qmd"
GUIDE_LABEL = "AI Companion (SocratiQ)"


class ScriptTagCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.scripts.append(dict(attrs))


def sidebar_items(items):
    """Collect sidebar link entries regardless of the volume's sidebar grouping."""
    if isinstance(items, dict):
        if "href" in items:
            yield items
        for value in items.values():
            yield from sidebar_items(value)
    elif isinstance(items, list):
        for item in items:
            yield from sidebar_items(item)


def sidebar_hrefs(items):
    """Collect guide links regardless of the volume's sidebar grouping."""
    for item in sidebar_items(items):
        yield item["href"]


@pytest.mark.parametrize("volume", VOLUMES)
def test_html_config_loads_and_stages_socratiq_reader(volume):
    books = ROOT / "books"
    config_path = books / "config" / f"_quarto-html-{volume}.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    resources = config["project"].get("resources", [])
    assert READER_RESOURCE in resources
    guide_path = books / GUIDE_SRC
    assert guide_path.is_file()
    assert f"# {GUIDE_LABEL} {{.unnumbered}}" in guide_path.read_text(encoding="utf-8")
    assert config["project"]["render"].count(GUIDE_SRC) == 1
    guide_entries = [
        item for item in sidebar_items(config["website"]["sidebar"]) if item.get("href") == GUIDE_SRC
    ]
    assert guide_entries == [{"text": GUIDE_LABEL, "href": GUIDE_SRC}]

    # Quarto resolves leading-slash resource entries from the project root and
    # preserves their relative path under output-dir. Ensure the reader source
    # exists and maps to the URL requested by the HTML include.
    source = books / READER_SRC.lstrip("/")
    assert source.is_file(), f"missing Quarto resource source: {source}"
    study_source = books / STUDY_SRC.lstrip("/")
    assert study_source.is_file(), f"missing lazy study resource: {study_source}"
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
@pytest.mark.parametrize("volume", VOLUMES)
def test_quarto_stages_reader_under_site_root(tmp_path, volume):
    books = ROOT / "books"
    source = books / READER_SRC.lstrip("/")
    assert source.is_file(), f"missing Quarto resource source: {source}"

    # Render a minimal website with the same project-relative resource pattern
    # and deployment prefix used by the volume configs.
    project = tmp_path / "project"
    staged_source = project / READER_SRC.lstrip("/")
    staged_source.parent.mkdir(parents=True)
    shutil.copy2(source, staged_source)
    study_source = books / STUDY_SRC.lstrip("/")
    assert study_source.is_file(), f"missing lazy study resource: {study_source}"
    shutil.copy2(study_source, project / STUDY_SRC.lstrip("/"))
    (project / "index.qmd").write_text("# Resource staging check\n", encoding="utf-8")
    chapter = project / volume / "01_chapter" / "01_chapter.qmd"
    chapter.parent.mkdir(parents=True)
    chapter.write_text("# Nested chapter\n", encoding="utf-8")
    fixture_config = {
        "project": {
            "type": "website",
            "output-dir": "_build/html",
            "resources": [READER_RESOURCE],
            "render": ["index.qmd", f"{volume}/01_chapter/01_chapter.qmd"],
        },
        "website": {
            "site-url": f"https://mlsysbook.ai/{volume}/",
        },
        "format": {
            "html": {
                "include-in-header": [
                    {
                        "text": (
                            f'<script src="{READER_SRC}" data-volume="{volume}" '
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
        ["bash", str(flatten), str(output), volume],
        check=True,
        capture_output=True,
        text=True,
    )
    assert f"{volume} flattened" in flattened.stdout
    staged_reader = output / READER_SRC.lstrip("/")
    assert staged_reader.is_file()
    assert staged_reader.read_bytes() == source.read_bytes()
    staged_study = output / STUDY_SRC.lstrip("/")
    assert staged_study.read_bytes() == study_source.read_bytes()

    root_html = (output / "index.html").read_text(encoding="utf-8")
    chapter_html = (output / "01_chapter" / "01_chapter.html").read_text(encoding="utf-8")
    root_parser = ScriptTagCollector()
    root_parser.feed(root_html)
    chapter_parser = ScriptTagCollector()
    chapter_parser.feed(chapter_html)
    root_tags = [tag for tag in root_parser.scripts if tag.get("data-volume") == volume]
    chapter_tags = [tag for tag in chapter_parser.scripts if tag.get("data-volume") == volume]
    assert len(root_tags) == len(chapter_tags) == 1
    assert root_tags[0].get("data-endpoint") == chapter_tags[0].get("data-endpoint") == READER_ENDPOINT

    # Reuse the rendered and flattened HTML for both deployment mounts. Quarto
    # makes the include relative, so neither the volume nor the dev repository
    # prefix may be lost when loading the reader or its lazy study sibling.
    for deployment_root in (
        "https://mlsysbook.ai/",
        "https://harvard-edge.github.io/cs249r_book_dev/",
    ):
        volume_url = urljoin(deployment_root, f"{volume}/")
        root_asset_url = urljoin(volume_url, root_tags[0]["src"])
        chapter_url = urljoin(volume_url, "01_chapter/01_chapter.html")
        chapter_asset_url = urljoin(chapter_url, chapter_tags[0]["src"])
        expected_asset_url = urljoin(volume_url, READER_SRC.lstrip("/"))
        assert root_asset_url == expected_asset_url
        assert chapter_asset_url == expected_asset_url
        for reader_url in (root_asset_url, chapter_asset_url):
            assert urljoin(reader_url, "study.js") == urljoin(volume_url, STUDY_SRC.lstrip("/"))
        deployed_asset_path = urlparse(volume_url).path.lstrip("/") + READER_SRC.lstrip("/")
        assert urlparse(root_asset_url).path.lstrip("/") == deployed_asset_path
