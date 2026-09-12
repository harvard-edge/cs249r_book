"""Math spans must not be scanned for citation keys by bib-integrity.

Pandoc's `@` citation syntax has no meaning inside math mode, so notation such
as `$\\text{pass@k}$` is not a citation. The refs citations scope already masks
math; these tests keep the bib-integrity scanner consistent with it.
"""

from types import SimpleNamespace

from binder.cli.commands.validate import ValidateCommand


def _citation_keys(tmp_path, body):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(body, encoding="utf-8")
    command = ValidateCommand(
        config_manager=SimpleNamespace(book_dir=tmp_path),
        chapter_discovery=None,
    )
    command._book_bib_scope_for_qmd = lambda path: SimpleNamespace(
        name="vol3", bib_paths=[]
    )
    return [occ.key for occ in command._citation_occurrences_for_qmd(qmd)]


def test_inline_math_at_sign_is_not_a_citation(tmp_path):
    body = "Pipelines compute the unbiased $\\text{pass@k}$ estimator [@chen2021codex].\n"

    assert _citation_keys(tmp_path, body) == ["chen2021codex"]


def test_multiline_display_math_is_not_scanned(tmp_path):
    body = "\n".join(
        [
            "Before the estimator [@smith2020].",
            "",
            "$$",
            "\\text{pass@k} = 1 - \\binom{n-c}{k} / \\binom{n}{k}",
            "$$ {#eq-pass-at-k}",
            "",
            "After the estimator [@jones2021].",
        ]
    )

    assert _citation_keys(tmp_path, body) == ["smith2020", "jones2021"]


def test_escaped_currency_does_not_hide_citations(tmp_path):
    body = "A run costs \\$5 [@doe2022], not \\$10.\n"

    assert _citation_keys(tmp_path, body) == ["doe2022"]
