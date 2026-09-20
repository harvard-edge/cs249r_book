from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[2] / "books" / "shared" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from fix_figure_alts import repair_figure_alts, repair_navbar_logo_links  # noqa: E402


# 2026-09-20: Quarto float wrappers hid Volume IV figure descriptions from img.
def test_moves_float_alt_to_sole_static_and_generated_images():
    source = (
        '<div id="fig-static" class="quarto-float quarto-figure" '
        'alt="A &amp; B"><figure><div><a href="figure.svg">'
        '<img src="figure.svg" class="figure-img"></a></div>'
        '<figcaption>Static image</figcaption></figure></div>\n'
        '<div class="quarto-float quarto-figure" alt="Chart slope">'
        '<figure><div><img src="figure.png" alt=""></div>'
        '<figcaption>Generated chart</figcaption></figure></div>'
    )

    output, counts = repair_figure_alts(source)

    assert counts == {
        "transferred": 2, "wrappers_cleaned": 2, "fallback_labeled": 0,
    }
    assert '<img src="figure.svg" class="figure-img" alt="A &amp; B">' in output
    assert '<img src="figure.png" alt="Chart slope">' in output
    assert 'class="quarto-float quarto-figure" alt=' not in output
    assert repair_figure_alts(output)[0] == output


def test_keeps_existing_image_alt_and_unrelated_markup():
    source = (
        '<div class="quarto-float quarto-figure" alt="Duplicate">'
        '<figure><img src="x.svg" alt="Correct description"></figure></div>'
        '<div class="quarto-float quarto-figure"><img src="y.svg"></div>'
        '<div class="other" alt="Unrelated"><img src="z.svg"></div>'
    )

    output, counts = repair_figure_alts(source)

    assert counts == {
        "transferred": 0, "wrappers_cleaned": 1, "fallback_labeled": 0,
    }
    assert '<img src="x.svg" alt="Correct description">' in output
    assert '<div class="quarto-float quarto-figure"><img src="y.svg"></div>' in output
    assert '<div class="other" alt="Unrelated"><img src="z.svg"></div>' in output


def test_preserves_description_for_multi_image_and_image_free_floats():
    source = (
        "<div class='quarto-figure' alt='Two panels'>"
        "<img src='a.png'/><img src='b.png'/></div>"
        '<div class="quarto-figure" alt="Interactive plot"><canvas></canvas></div>'
    )

    output, counts = repair_figure_alts(source)

    assert counts == {
        "transferred": 0, "wrappers_cleaned": 2, "fallback_labeled": 2,
    }
    assert "aria-label='Two panels'" in output
    assert 'aria-label="Interactive plot"' in output
    assert "<img src='a.png'/>" in output
    assert "<img src='b.png'/>" in output
    assert repair_figure_alts(output)[0] == output


# 2026-09-20: Quarto emits a second, logo-only home link without a name.
def test_names_only_unnamed_logo_home_link_and_keeps_logo_images_decorative():
    source = (
        '<a href="../../index.html" class="navbar-brand navbar-brand-logo">\n'
        '<img src="logo.png" alt="" class="navbar-logo light-content">\n'
        '<img src="logo.png" alt="" class="navbar-logo dark-content">\n'
        '</a>'
        '<a class="navbar-brand" href="../../index.html">'
        '<span>Machine Learning Systems</span></a>'
        '<a href="index.html" class="navbar-brand navbar-brand-logo" '
        'aria-label="Existing name">'
        '<img src="other.png" alt="" class="navbar-logo"></a>'
        '<a class="navbar-brand navbar-brand-logo">Visible text</a>'
        '<a href="about.html" class="navbar-brand navbar-brand-logo">'
        '<img src="other.png" alt="" class="navbar-logo"></a>'
    )

    output, count = repair_navbar_logo_links(source)

    assert count == 1
    assert ('<a href="../../index.html" class="navbar-brand navbar-brand-logo" '
            'aria-label="Machine Learning Systems home">') in output
    assert output.count('alt="" class="navbar-logo') == 4
    assert 'aria-label="Existing name"' in output
    assert '<a class="navbar-brand navbar-brand-logo">Visible text</a>' in output
    assert ('<a href="about.html" class="navbar-brand navbar-brand-logo">'
            '<img src="other.png" alt="" class="navbar-logo"></a>') in output
    assert repair_navbar_logo_links(output) == (output, 0)
