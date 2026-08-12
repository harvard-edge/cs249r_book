r"""Currency style checks for book currency notation.

Policy:
  * Reader-facing prices use ``$``.
  * Source prose and LEGO currency prefixes escape the dollar sign as ``\$``.
  * ``USD`` appears only once, in the shared notation definition that says
    dollar-denominated costs are U.S. dollars unless noted otherwise.

The checker scans source files that can render visible text in the book:
QMD prose/cells, quiz JSON, and SVG labels. It can also scan rendered HTML
artifacts for currency/LaTeX collisions that source checks cannot see.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable


TARGET_SUFFIXES = {".qmd", ".json", ".svg"}
HTML_SUFFIXES = {".html", ".htm"}
USD_PATTERN = re.compile(r"\bUSD\b")
DOUBLE_DOLLAR_PATTERN = re.compile(r"\$\$")
# Currency must be rendered exclusively through ``fmt_usd()`` (the currency
# member of the fmt_* family). Decorating a formatter call with a dollar sign
# via ``prefix=``/``suffix=`` is the pre-fmt_usd pattern and is forbidden: it
# is the only place a raw or escaped ``$`` should never appear, because the
# escaping then lives at the call site instead of inside ``fmt_usd``. This
# also guards against regressions back to the unescaped ``prefix="$"`` form.
CURRENCY_FMT_DECORATION_PATTERN = re.compile(
    r"(prefix|suffix)\s*=\s*(['\"])[^'\"]*\$[^'\"]*\2"
)
RAW_LATEX_PATTERN = re.compile(
    r"\\(?:frac|text|times|left|right|begin|end)\b|\\\(|\\\)"
)
CURRENCY_MATH_SPAN_PATTERN = re.compile(
    r"\\\([^)]*\b\d[\d,]*(?:\.\d+)?[KMBT]\),\s+[A-Za-z][\w-]*"
)
NOTATION_REL_PATH = Path("book/quarto/contents/vol1/frontmatter/_notation_body.qmd")
NOTATION_DEFINITION = (
    "*   Currency: Dollar amounts use the dollar sign (`$`); unless otherwise "
    "noted, dollar-denominated costs are U.S. dollars (USD)."
)
NOTATION_RENDERED_CONTEXT = "U.S. dollars (USD)"


@dataclass(frozen=True)
class Violation:
    file: str
    line: int
    code: str
    message: str
    context: str = ""
    suggestion: str = ""


def iter_target_files(paths: Iterable[Path]) -> list[Path]:
    """Return content source files that can carry visible currency text."""
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(
                p
                for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in TARGET_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() in TARGET_SUFFIXES:
            files.append(path)
    return sorted(dict.fromkeys(files))


def iter_html_files(paths: Iterable[Path]) -> list[Path]:
    """Return rendered HTML files under *paths*."""
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(
                p
                for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in HTML_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() in HTML_SUFFIXES:
            files.append(path)
    return sorted(dict.fromkeys(files))


def _is_allowed_notation_definition(path: Path, line: str) -> bool:
    normalized = Path(*path.parts[-len(NOTATION_REL_PATH.parts) :])
    return normalized == NOTATION_REL_PATH and line.strip() == NOTATION_DEFINITION


def _audit_file(path: Path) -> list[Violation]:
    violations: list[Violation] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    in_python_cell = False

    for lineno, line in enumerate(lines, 1):
        if line.startswith("```{python}"):
            in_python_cell = True
            continue
        if in_python_cell and line.strip() == "```":
            in_python_cell = False
            continue

        # Pint uses USD as a currency unit in LEGO cells; prose policy is separate.
        if in_python_cell:
            pass
        elif "USD" in line and not _is_allowed_notation_definition(path, line):
            for _match in USD_PATTERN.finditer(line):
                violations.append(
                    Violation(
                        file=str(path),
                        line=lineno,
                        code="currency_usd_literal",
                        message=(
                            "Use `$` for dollar amounts; `USD` is defined once "
                            "in the notation file."
                        ),
                        context=line.strip()[:180],
                        suggestion=(
                            "Render dollar amounts with `fmt_usd(value, ...)` "
                            "(escapes `$` automatically); for plain prose use "
                            "`$`. Keep the currency-code definition only in "
                            "vol1/frontmatter/_notation_body.qmd."
                        ),
                    )
                )

        if path.suffix.lower() == ".qmd":
            for match in CURRENCY_FMT_DECORATION_PATTERN.finditer(line):
                kind = match.group(1)
                violations.append(
                    Violation(
                        file=str(path),
                        line=lineno,
                        code="currency_fmt_prefix_suffix",
                        message=(
                            f"Currency in a `{kind}=` formatter argument is no "
                            "longer allowed; route all dollar amounts through "
                            "`fmt_usd()`, which owns the escaped `$`."
                        ),
                        context=line.strip()[:180],
                        suggestion=(
                            "Replace `fmt(value, ..., prefix=\"$\")` /"
                            " `prefix=\"\\\\$\"` with `fmt_usd(value, ...)`. Use "
                            "`approx=True` for `~$`, and `suffix=\"M\"`/`\"/GB\"` "
                            "for scale or rate labels (no `$` in the suffix)."
                        ),
                    )
                )

    return violations


class _VisibleTextParser(HTMLParser):
    """Collect visible prose text while skipping code, scripts, styles, and math."""

    _SKIP_TAGS = {"script", "style", "code", "pre"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._math_depth = 0
        self.chunks: list[tuple[int, str]] = []
        self.math_chunks: list[tuple[int, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = {
            token
            for key, value in attrs
            if key == "class" and value
            for token in value.split()
        }
        starts_math = "math" in classes
        if self._math_depth or starts_math:
            self._math_depth += 1
        if self._skip_depth or tag in self._SKIP_TAGS or starts_math:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._math_depth:
            self._math_depth -= 1
        if self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._math_depth and data.strip():
            self.math_chunks.append((self.getpos()[0], data))
        elif not self._skip_depth and data.strip():
            self.chunks.append((self.getpos()[0], data))


def _context(text: str, start: int, end: int, *, width: int = 80) -> str:
    context = text[max(0, start - width) : min(len(text), end + width)]
    return re.sub(r"\s+", " ", context).strip()[:220]


def _is_allowed_rendered_notation_usd(path: Path, context: str) -> bool:
    return path.name == "notation.html" and NOTATION_RENDERED_CONTEXT in context


def audit_rendered_file(path: Path) -> list[Violation]:
    """Scan one rendered HTML file for visible currency/math artifacts."""
    parser = _VisibleTextParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))

    violations: list[Violation] = []
    checks = [
        (
            USD_PATTERN,
            "rendered_usd_literal",
            "Rendered HTML contains visible `USD`; only notation.html may define it.",
            "Use `$` in reader-facing content; keep `USD` only in the notation definition.",
        ),
        (
            DOUBLE_DOLLAR_PATTERN,
            "rendered_double_dollar",
            "Rendered HTML contains visible `$$`, usually from combining a literal dollar sign with a formatted currency value.",
            "Remove the prose-side `$` when the LEGO value already uses `prefix=\"\\\\$\"`.",
        ),
        (
            RAW_LATEX_PATTERN,
            "rendered_raw_latex",
            "Rendered HTML contains visible raw LaTeX outside a math span.",
            "Move the expression into normal Markdown math or export a math-safe value from the LEGO cell.",
        ),
    ]

    for line, chunk in parser.chunks:
        for pattern, code, message, suggestion in checks:
            for match in pattern.finditer(chunk):
                context = _context(chunk, match.start(), match.end())
                if code == "rendered_usd_literal" and _is_allowed_rendered_notation_usd(path, context):
                    continue
                violations.append(
                    Violation(
                        file=str(path),
                        line=line,
                        code=code,
                        message=message,
                        context=context,
                        suggestion=suggestion,
                    )
                )

    for line, chunk in parser.math_chunks:
        for match in CURRENCY_MATH_SPAN_PATTERN.finditer(chunk):
            violations.append(
                Violation(
                    file=str(path),
                    line=line,
                    code="rendered_currency_math_span",
                    message=(
                        "Rendered HTML has currency/prose-looking text inside a "
                        "math span, usually from an unescaped currency dollar."
                    ),
                    context=_context(chunk, match.start(), match.end()),
                    suggestion=(
                        "Escape prose currency dollars as `\\$`, or use "
                        "`fmt(..., prefix=\"\\\\$\")` for LEGO currency values."
                    ),
                )
            )

    return violations


def audit_rendered_html(paths: Iterable[Path]) -> list[Violation]:
    """Scan rendered HTML files for visible currency/math artifacts."""
    violations: list[Violation] = []
    for path in iter_html_files(paths):
        try:
            violations.extend(audit_rendered_file(path))
        except OSError:
            continue
    return violations


def audit(paths: Iterable[Path]) -> list[Violation]:
    """Scan target content files for disallowed literal ``USD``."""
    violations: list[Violation] = []
    for path in iter_target_files(paths):
        try:
            violations.extend(_audit_file(path))
        except (OSError, UnicodeDecodeError):
            continue
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("book/quarto/contents")])
    parser.add_argument(
        "--rendered-html",
        action="store_true",
        help="Scan rendered HTML instead of source files",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args(argv)

    if args.rendered_html:
        violations = audit_rendered_html(args.paths)
    else:
        violations = audit(args.paths or [Path("book/quarto/contents")])
    if args.json:
        print(json.dumps([v.__dict__ for v in violations], indent=2, ensure_ascii=False))
        return 1 if violations else 0

    for violation in violations:
        print(f"{violation.file}:{violation.line} [{violation.code}] {violation.message}")
        if violation.context:
            print(f"  source: {violation.context}")
        if violation.suggestion:
            print(f"  fix: {violation.suggestion}")
    print(f"Total violations: {len(violations)}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
