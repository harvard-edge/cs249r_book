from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "flatten-vol-urls.sh"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_flatten_vol_urls_rewrites_generated_volume_links(tmp_path: Path) -> None:
    site = tmp_path / "vol1-site"

    write(
        site / "index.html",
        """
        <a href="./contents/vol1/index.qmd">Homepage</a>
        <a href="./contents/vol1/frontmatter/about.html">Preface</a>
        <a href="./contents/vol1/data_engineering/data_engineering.html">Data Engineering</a>
        <a href="./contents/frontmatter/socratiq/socratiq.html">SocratiQ</a>
        """,
    )
    write(
        site / "contents" / "vol1" / "data_engineering" / "data_engineering.html",
        """
        <link href="../../../site_libs/quarto.css" rel="stylesheet">
        <a href="../contents/vol1/frontmatter/about.html">Preface</a>
        <a href="../contents/vol1/backmatter/glossary/glossary.html">Glossary</a>
        <a href="../contents/frontmatter/socratiq/socratiq.html">SocratiQ</a>
        """,
    )
    write(
        site / "contents" / "vol1" / "backmatter" / "glossary" / "glossary.html",
        '<a href="../../contents/vol1/data_engineering/data_engineering.html">Data</a>',
    )
    write(
        site / "contents" / "vol1" / "frontmatter" / "about.html",
        '<a href="../contents/vol1/data_engineering/data_engineering.html">Data</a>',
    )
    write(site / "contents" / "frontmatter" / "socratiq" / "socratiq.html", "<p>SocratiQ</p>")
    write(site / "search.json", '{"href":"contents/vol1/data_engineering/data_engineering.html"}')
    write(
        site / "sitemap.xml",
        "<loc>https://mlsysbook.ai/vol1/contents/vol1/data_engineering/data_engineering.html</loc>",
    )

    subprocess.run(["bash", str(SCRIPT), str(site), "vol1"], check=True)

    assert not (site / "contents" / "vol1").exists()
    assert (site / "data_engineering" / "data_engineering.html").exists()
    assert (site / "frontmatter" / "about.html").exists()

    root_html = (site / "index.html").read_text(encoding="utf-8")
    assert 'href="./"' in root_html
    assert 'href="./frontmatter/about.html"' in root_html
    assert 'href="./data_engineering/data_engineering.html"' in root_html
    assert 'href="./contents/frontmatter/socratiq/socratiq.html"' in root_html
    assert "contents/vol1" not in root_html

    chapter_html = (site / "data_engineering" / "data_engineering.html").read_text(encoding="utf-8")
    assert 'href="../frontmatter/about.html"' in chapter_html
    assert 'href="../backmatter/glossary/glossary.html"' in chapter_html
    assert 'href="../contents/frontmatter/socratiq/socratiq.html"' in chapter_html
    assert 'href="../contents/vol1/' not in chapter_html
    assert 'href="../site_libs/quarto.css"' in chapter_html

    glossary_html = (site / "backmatter" / "glossary" / "glossary.html").read_text(encoding="utf-8")
    assert 'href="../../data_engineering/data_engineering.html"' in glossary_html

    assert '"data_engineering/data_engineering.html"' in (site / "search.json").read_text(encoding="utf-8")
    assert "/vol1/data_engineering/data_engineering.html" in (site / "sitemap.xml").read_text(
        encoding="utf-8"
    )
