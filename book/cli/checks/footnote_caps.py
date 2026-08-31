"""Check footnote definition capitalization."""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

FOOTNOTE_DEF = re.compile(r"^\[\^([A-Za-z0-9_:.\-]+)\]:\s+(.*)$")
FOOTNOTE_DEF_PREFIX = re.compile(r"^\[\^[A-Za-z0-9_:.\-]+\]:\s+")

BOLD_OPENER = re.compile(r"^\*\*\s*")
TERM_HEAD = re.compile(r"^(\s*\*\*)(.+?)(\*\*:)")

MINOR_WORDS = {
    "a", "an", "the", "and", "but", "or", "nor", "yet", "so", "of", "in",
    "into", "on", "onto", "to", "for", "with", "vs", "vs.", "at", "by",
    "as", "from", "via",
}

CLI_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_ALLOWLIST = CLI_ROOT / "data" / "footnote_caps_allowlist.txt"


def load_allowlist(path: pathlib.Path) -> set[str]:
    """Parse an allowlist file; return ids with intentional lowercase starts."""
    if not path.exists():
        return set()
    ids: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            ids.add(line)
    return ids


@dataclass
class Violation:
    path: pathlib.Path
    line_no: int
    raw_line: str
    prefix: str
    body: str
    first_char: str
    kind: str = "lowercase_first_letter"
    detail: str = ""


def first_meaningful_char(body: str) -> tuple[str, int]:
    stripped = body
    offset = 0
    m = BOLD_OPENER.match(stripped)
    if m:
        stripped = stripped[m.end():]
        offset += m.end()

    for i, ch in enumerate(stripped):
        if ch.isalpha():
            return ch, offset + i
        if not ch.isspace():
            return "", -1
    return "", -1


def _is_protected_term_token(term: str, start: int, end: int, token: str) -> bool:
    if len(token) == 1:
        return True
    if any(ch.isupper() for ch in token[1:]):
        return True

    before = term[start - 1] if start > 0 else ""
    after = term[end] if end < len(term) else ""
    if (before and before in ".`_") or (after and after in ".`_"):
        return True
    if before.isdigit() or after.isdigit():
        return True
    if before == "-":
        prefix = term[: start - 1].rsplit("-", 1)[-1]
        if len(prefix) == 1 and prefix.isupper():
            return True
    return False


def term_head_case_issues(term: str) -> list[tuple[int, str]]:
    issues: list[tuple[int, str]] = []
    i = 0
    is_first_word = True
    while i < len(term):
        ch = term[i]
        if ch == "`":
            end = term.find("`", i + 1)
            i = len(term) if end == -1 else end + 1
            continue
        if ch == "$":
            end = term.find("$", i + 1)
            i = len(term) if end == -1 else end + 1
            continue
        if not ch.isalpha():
            i += 1
            continue

        start = i
        while i < len(term) and term[i].isalpha():
            i += 1
        token = term[start:i]

        if token.lower() in MINOR_WORDS:
            if is_first_word and token[0].islower():
                issues.append((start, token))
            elif not is_first_word and token.istitle() and not _is_protected_term_token(term, start, i, token):
                issues.append((start, token))
        else:
            if token[0].islower() and not _is_protected_term_token(term, start, i, token):
                issues.append((start, token))

        is_first_word = False
    return issues


def leading_term_head(body: str) -> tuple[str, int] | None:
    m = TERM_HEAD.match(body)
    if not m:
        return None
    return m.group(2), m.start(2)


def scan_file(path: pathlib.Path, allowlist: set[str]) -> list[Violation]:
    violations: list[Violation] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return violations

    for line_no, line in enumerate(text.splitlines(), start=1):
        m = FOOTNOTE_DEF.match(line)
        if not m:
            continue
        fn_id, body = m.group(1), m.group(2)
        prefix_match = FOOTNOTE_DEF_PREFIX.match(line)
        assert prefix_match is not None
        prefix = prefix_match.group(0)
        ch, _ = first_meaningful_char(body)
        if fn_id not in allowlist and ch and ch.islower():
            violations.append(
                Violation(
                    path=path,
                    line_no=line_no,
                    raw_line=line,
                    prefix=prefix,
                    body=body,
                    first_char=ch,
                )
            )
        term_match = leading_term_head(body)
        if term_match is not None:
            term, _ = term_match
            issues = term_head_case_issues(term)
            if issues:
                words = ", ".join(word for _, word in issues)
                violations.append(
                    Violation(
                        path=path,
                        line_no=line_no,
                        raw_line=line,
                        prefix=prefix,
                        body=body,
                        first_char=issues[0][1][0],
                        kind="term_head_case",
                        detail=words,
                    )
                )
    return violations


def _fix_term_head_case(body: str) -> str:
    m = TERM_HEAD.match(body)
    if not m:
        return body
    term = m.group(2)
    chars = list(term)
    for offset, _word in term_head_case_issues(term):
        chars[offset] = chars[offset].upper()
    return body[: m.start(2)] + "".join(chars) + body[m.end(2):]


def apply_fix(v: Violation) -> str:
    """Return the corrected line for a single violation."""
    if v.kind == "term_head_case":
        return v.prefix + _fix_term_head_case(v.body)

    body = v.body
    m = BOLD_OPENER.match(body)
    head = ""
    rest = body
    if m:
        head = body[: m.end()]
        rest = body[m.end():]
    for i, ch in enumerate(rest):
        if ch.isalpha():
            rest = rest[:i] + ch.upper() + rest[i + 1:]
            break
    return v.prefix + head + rest


def fix_files(violations: list[Violation]) -> None:
    """Rewrite each affected file, applying all fixes within it."""
    by_file: dict[pathlib.Path, list[Violation]] = {}
    for v in violations:
        by_file.setdefault(v.path, []).append(v)

    for path, vs in by_file.items():
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        for v in vs:
            original = lines[v.line_no - 1]
            newline = ""
            stripped = original
            if stripped.endswith("\r\n"):
                newline = "\r\n"
                stripped = stripped[:-2]
            elif stripped.endswith("\n"):
                newline = "\n"
                stripped = stripped[:-1]
            assert stripped == v.raw_line, f"File changed under us at {path}:{v.line_no}"
            lines[v.line_no - 1] = apply_fix(v) + newline
        path.write_text("".join(lines), encoding="utf-8")
