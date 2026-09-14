from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "flatten-vol-urls.sh"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run(site: Path, vol: str = "vol1") -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), str(site), vol], capture_output=True, text=True)


def build_site(site: Path) -> None:
    """A small render in the layout Quarto produces from books/vol1/."""
    write(
        site / "index.html",
        """
        <link href="./site_libs/quarto.css" rel="stylesheet">
        <a href="./vol1/index.qmd">Homepage</a>
        <a href="./vol1/frontmatter/about.html">Preface</a>
        <a href="./vol1/data_engineering/data_engineering.html#sec-data">Data Engineering</a>
        <a href="./shared/frontmatter/socratiq/socratiq.html">SocratiQ</a>
        <a href="https://mlsysbook.ai/vol1/vol1/data_engineering/data_engineering.html">abs</a>
        """,
    )
    write(
        site / "vol1" / "data_engineering" / "data_engineering.html",
        """
        <link href="../../site_libs/quarto.css" rel="stylesheet">
        <img src="../../shared/assets/images/cover.png">
        <a href="../../vol1/index.qmd">Homepage</a>
        <a href="../frontmatter/about.html">Preface</a>
        <a href="../../vol1/backmatter/glossary/glossary.html?x=1#g">Glossary</a>
        <a href="../../shared/frontmatter/socratiq/socratiq.html">SocratiQ</a>
        <a href="#local">Local</a>
        <script src="/tools/scripts/socratiQ/bundle.js"></script>
        <a href="https://mlsysbook.ai/vol1/contents/vol1/training/training.html">legacy</a>
        """,
    )
    write(
        site / "vol1" / "backmatter" / "glossary" / "glossary.html",
        """
        <link href="../../../site_libs/quarto.css" rel="stylesheet">
        <a href="../../data_engineering/data_engineering.html">Data</a>
        """,
    )
    write(site / "vol1" / "frontmatter" / "about.html", '<a href="../data_engineering/data_engineering.html">Data</a>')
    write(site / "vol1" / "index.html", "<p>sidebar home</p>")
    write(site / "shared" / "frontmatter" / "socratiq" / "socratiq.html",
          '<a href="../../../vol1/data_engineering/data_engineering.html">Data</a>')
    write(site / "site_libs" / "quarto.css", "")
    write(site / "search.json", '[{"href": "vol1/data_engineering/data_engineering.html#sec-data"},{"href":"shared/x.html"}]')
    write(site / "sitemap.xml", "<loc>https://mlsysbook.ai/vol1/vol1/data_engineering/data_engineering.html</loc>")


def test_moves_pages_and_rewrites_links_at_every_depth(tmp_path: Path) -> None:
    site = tmp_path / "vol1-site"
    build_site(site)
    result = run(site)
    assert result.returncode == 0, result.stderr

    assert (site / "data_engineering" / "data_engineering.html").exists()
    assert (site / "frontmatter" / "about.html").exists()
    assert (site / "backmatter" / "glossary" / "glossary.html").exists()

    root_html = (site / "index.html").read_text(encoding="utf-8")
    assert 'href="./"' in root_html
    assert 'href="./frontmatter/about.html"' in root_html
    assert 'href="./data_engineering/data_engineering.html#sec-data"' in root_html
    assert 'href="./shared/frontmatter/socratiq/socratiq.html"' in root_html
    assert 'href="https://mlsysbook.ai/vol1/data_engineering/data_engineering.html"' in root_html
    assert "./vol1/" not in root_html

    chapter = (site / "data_engineering" / "data_engineering.html").read_text(encoding="utf-8")
    assert 'href="../site_libs/quarto.css"' in chapter
    assert 'src="../shared/assets/images/cover.png"' in chapter
    assert 'href="../"' in chapter
    assert 'href="../frontmatter/about.html"' in chapter
    assert 'href="../backmatter/glossary/glossary.html?x=1#g"' in chapter
    assert 'href="../shared/frontmatter/socratiq/socratiq.html"' in chapter
    assert 'href="#local"' in chapter
    assert 'src="/tools/scripts/socratiQ/bundle.js"' in chapter
    assert 'href="https://mlsysbook.ai/vol1/training/training.html"' in chapter
    assert "../../vol1/" not in chapter

    glossary = (site / "backmatter" / "glossary" / "glossary.html").read_text(encoding="utf-8")
    assert 'href="../../site_libs/quarto.css"' in glossary
    assert 'href="../../data_engineering/data_engineering.html"' in glossary

    shared = (site / "shared" / "frontmatter" / "socratiq" / "socratiq.html").read_text(encoding="utf-8")
    assert 'href="../../../data_engineering/data_engineering.html"' in shared

    assert '"href": "data_engineering/data_engineering.html#sec-data"' in (site / "search.json").read_text()
    assert "/vol1/vol1/" not in (site / "sitemap.xml").read_text()


def test_legacy_urls_redirect_to_clean_pages(tmp_path: Path) -> None:
    site = tmp_path / "vol1-site"
    build_site(site)
    assert run(site).returncode == 0

    for tree, up in (("vol1", "../../"), ("contents/vol1", "../../../")):
        stub = (site / tree / "data_engineering" / "data_engineering.html").read_text(encoding="utf-8")
        assert f'http-equiv="refresh" content="0; url={up}data_engineering/data_engineering.html"' in stub
    home = (site / "contents" / "vol1" / "index.html").read_text(encoding="utf-8")
    assert 'url=../../"' in home


def test_unprefixed_chapter_urls_redirect_to_numbered_pages(tmp_path: Path) -> None:
    """Chapter folders gained a two-digit order prefix; the old URLs must still work."""
    site = tmp_path / "vol1-site"
    write(site / "index.html", '<a href="./vol1/08_training/08_training.html#sec-x">Training</a>')
    write(site / "vol1" / "08_training" / "08_training.html", '<a href="../../vol1/index.qmd">Home</a>')
    write(site / "vol1" / "index.html", "<p>sidebar home</p>")
    result = run(site)
    assert result.returncode == 0, result.stderr

    assert (site / "08_training" / "08_training.html").exists()
    for alias, up in (
        ("training/training.html", "../"),
        ("vol1/training/training.html", "../../"),
        ("contents/vol1/training/training.html", "../../../"),
    ):
        stub = (site / alias).read_text(encoding="utf-8")
        assert f'http-equiv="refresh" content="0; url={up}08_training/08_training.html"' in stub

    assert run(site).returncode == 0


def test_link_into_unprefixed_alias_fails(tmp_path: Path) -> None:
    """A generated page must link to the numbered chapter, not its redirect alias."""
    site = tmp_path / "vol1-site"
    write(site / "vol1" / "08_training" / "08_training.html", "<p>chapter</p>")
    write(site / "vol1" / "index.html", '<a href="../training/training.html">stale</a>')
    result = run(site)
    assert result.returncode != 0
    assert "redirect trees" in result.stderr


def test_rerun_is_a_no_op(tmp_path: Path) -> None:
    site = tmp_path / "vol1-site"
    build_site(site)
    assert run(site).returncode == 0
    before = {p: p.read_bytes() for p in site.rglob("*") if p.is_file()}
    second = run(site)
    assert second.returncode == 0, second.stderr
    assert {p: p.read_bytes() for p in site.rglob("*") if p.is_file()} == before


def test_missing_volume_tree_fails_loudly(tmp_path: Path) -> None:
    site = tmp_path / "vol1-site"
    write(site / "index.html", "<p>no volume tree</p>")
    result = run(site)
    assert result.returncode != 0
    assert "does not exist" in (result.stderr + result.stdout)
