"""Mapped single-chapter PDF builds for layout reconstruction.

The full-volume LaTeX ``.aux`` file is the numbering authority.  External
cross-references in a generated copy of the chapter are replaced with their
resolved display text, while the chapter and folio counters are restored to
their full-book values.  The source QMD is never modified.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml


XREF_RE = re.compile(
    r"@(?P<kind>Sec|sec|Fig|fig|Tbl|tbl|Eq|eq|Lst|lst|Alg|alg)-"
    r"(?P<rest>[A-Za-z0-9_-]+)"
)
ID_RE = re.compile(r"#([A-Za-z][A-Za-z0-9_-]+)")
CELL_LABEL_RE = re.compile(
    r"^\s*#\|\s*label:\s*([A-Za-z][A-Za-z0-9_-]+)\s*$", re.MULTILINE
)
H1_RE = re.compile(r"^#\s+.+?\{#([A-Za-z][A-Za-z0-9_-]+)\}\s*$", re.MULTILINE)
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
CUSTOM_HEADING_RE = re.compile(
    r"(\\begin\{fbxSimple\}\{callout-[^}]+\}\{[^{}\n]*? )\d+(\.\d+:\})"
)


def _braced_fields(payload: str) -> list[str]:
    """Return top-level braced fields from a LaTeX aux payload."""
    fields: list[str] = []
    i = 0
    while i < len(payload):
        if payload[i].isspace():
            i += 1
            continue
        if payload[i] != "{":
            break
        depth = 0
        start = i + 1
        for j in range(i, len(payload)):
            char = payload[j]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    fields.append(payload[start:j])
                    i = j + 1
                    break
        else:
            break
    return fields


def parse_aux(aux_path: Path) -> dict[str, dict[str, str]]:
    """Parse Quarto/Pandoc label number, page, and anchor data from ``.aux``."""
    labels: dict[str, dict[str, str]] = {}
    prefix = r"\newlabel{"
    for line in aux_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(prefix):
            continue
        label_end = line.find("}", len(prefix))
        if label_end < 0:
            continue
        label = line[len(prefix) : label_end]
        payload = line[label_end + 1 :].strip()
        if not (payload.startswith("{") and payload.endswith("}")):
            continue
        fields = _braced_fields(payload[1:-1])
        if len(fields) >= 4:
            number, page, _, anchor = fields[:4]
            labels[label] = {"number": number, "page": page, "anchor": anchor}
    return labels


def _display_text(kind_token: str, record: dict[str, str]) -> str:
    kind = kind_token.lower()
    if kind == "sec":
        noun = "chapter" if record["anchor"].startswith("chapter.") else "section"
    else:
        noun = {
            "fig": "figure",
            "tbl": "table",
            "eq": "equation",
            "lst": "listing",
            "alg": "algorithm",
        }[kind]
    if kind_token[0].isupper():
        noun = noun.capitalize()
    return f"{noun}\u00a0{record['number']}"


def _replace_prose_segment(
    segment: str,
    internal: set[str],
    labels: dict[str, dict[str, str]],
    missing: set[str],
) -> tuple[str, int]:
    """Map xrefs in prose while preserving inline-code spans."""
    parts = re.split(r"(`+[^`]*`+)", segment)
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        kind_token = match.group("kind")
        label = f"{kind_token.lower()}-{match.group('rest')}"
        if label in internal:
            return match.group(0)
        record = labels.get(label)
        if record is None:
            missing.add(label)
            return match.group(0)
        count += 1
        return _display_text(kind_token, record)

    for i in range(0, len(parts), 2):
        parts[i] = XREF_RE.sub(replace, parts[i])
    return "".join(parts), count


def mapped_source(
    source: str,
    labels: dict[str, dict[str, str]],
) -> tuple[str, int, list[str]]:
    """Map external xrefs without touching fenced code, inline code, or comments."""
    internal = set(ID_RE.findall(source)) | set(CELL_LABEL_RE.findall(source))
    missing: set[str] = set()
    replaced = 0
    output: list[str] = []
    fence: str | None = None
    in_comment = False

    for line in source.splitlines(keepends=True):
        fence_match = FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker[0]
            elif marker[0] == fence:
                fence = None
            output.append(line)
            continue
        if fence is not None:
            output.append(line)
            continue

        pieces: list[str] = []
        cursor = 0
        while cursor < len(line):
            if in_comment:
                end = line.find("-->", cursor)
                if end < 0:
                    pieces.append(line[cursor:])
                    cursor = len(line)
                else:
                    pieces.append(line[cursor : end + 3])
                    cursor = end + 3
                    in_comment = False
                continue
            start = line.find("<!--", cursor)
            prose_end = len(line) if start < 0 else start
            mapped, count = _replace_prose_segment(
                line[cursor:prose_end], internal, labels, missing
            )
            pieces.append(mapped)
            replaced += count
            cursor = prose_end
            if start >= 0:
                in_comment = True
        output.append("".join(pieces))
    return "".join(output), replaced, sorted(missing)


def _counter_hook(chapter_number: int, start_page: int) -> dict[str, str]:
    return {
        "text": (
            "\\makeatletter\n"
            "\\AtBeginDocument{%\n"
            "  \\let\\LayoutMapMainMatter\\mainmatter\n"
            "  \\renewcommand{\\mainmatter}{%\n"
            "    \\LayoutMapMainMatter\n"
            f"    \\setcounter{{chapter}}{{{chapter_number - 1}}}%\n"
            f"    \\setcounter{{page}}{{{start_page}}}%\n"
            "    \\@firstnumberedfalse% preserve the mapped folio\n"
            "  }%\n"
            "}\n"
            "\\makeatother"
        )
    }


@contextmanager
def _worktree_lock(lock_path: Path) -> Iterator[None]:
    """Serialize renders that temporarily redirect the active Quarto config."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another mapped chapter build is active in this worktree. "
                "Use a separate worktree for parallel renders."
            ) from exc
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _resolve_chapter(quarto_dir: Path, volume: str, spec: str) -> Path:
    supplied = Path(spec)
    direct = supplied if supplied.is_absolute() else quarto_dir / supplied
    if direct.is_file():
        path = direct.resolve()
    else:
        stem = supplied.stem
        matches = sorted((quarto_dir / "contents" / volume).rglob(f"{stem}.qmd"))
        if len(matches) != 1:
            detail = "none" if not matches else ", ".join(str(p) for p in matches)
            raise ValueError(f"Chapter {spec!r} did not resolve uniquely: {detail}")
        path = matches[0].resolve()
    volume_root = (quarto_dir / "contents" / volume).resolve()
    if volume_root not in path.parents:
        raise ValueError(f"Chapter must be under contents/{volume}: {path}")
    return path


def _correct_callout_numbers(
    quarto_dir: Path,
    output_dir: Path,
    output_stem: str,
    chapter_number: int,
) -> int:
    """Correct filter-generated callout prefixes and rebuild when necessary."""
    tex_path = quarto_dir / f"{output_stem}.tex"
    if not tex_path.is_file():
        raise RuntimeError(f"Quarto did not retain expected TeX: {tex_path}")
    tex = tex_path.read_text(encoding="utf-8")
    corrected, count = CUSTOM_HEADING_RE.subn(
        lambda match: f"{match.group(1)}{chapter_number}{match.group(2)}", tex
    )
    if count:
        tex_path.write_text(corrected, encoding="utf-8")
        latex = [
            "lualatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            tex_path.name,
        ]
        subprocess.run(latex, cwd=quarto_dir, check=True, stdout=subprocess.DEVNULL)
        idx_path = quarto_dir / f"{output_stem}.idx"
        if idx_path.exists():
            subprocess.run(
                ["makeindex", idx_path.name],
                cwd=quarto_dir,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        subprocess.run(latex, cwd=quarto_dir, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(latex, cwd=quarto_dir, check=True, stdout=subprocess.DEVNULL)
        shutil.copy2(
            quarto_dir / f"{output_stem}.pdf", output_dir / f"{output_stem}.pdf"
        )
    shutil.copy2(tex_path, output_dir.parent / f"{output_stem}.tex")
    for suffix in (
        ".aux",
        ".idx",
        ".ilg",
        ".ind",
        ".log",
        ".out",
        ".pdf",
        ".toc",
        ".tex",
    ):
        (quarto_dir / f"{output_stem}{suffix}").unlink(missing_ok=True)
    return count


def render_mapped_chapter(
    config_manager: Any,
    *,
    volume: str,
    chapter: str,
    aux_path: Path,
) -> dict[str, Any]:
    """Render an isolated chapter with full-volume numbering and references."""
    quarto_dir = Path(config_manager.book_dir).resolve()
    source_path = _resolve_chapter(quarto_dir, volume, chapter)
    aux_path = aux_path.expanduser().resolve()
    if not aux_path.is_file():
        raise FileNotFoundError(f"Full-build aux file not found: {aux_path}")

    source = source_path.read_text(encoding="utf-8")
    labels = parse_aux(aux_path)
    h1 = H1_RE.search(source)
    if h1 is None:
        raise ValueError(f"Could not find a labeled chapter H1 in {source_path}")
    chapter_label = h1.group(1)
    chapter_record = labels.get(chapter_label)
    if chapter_record is None or not chapter_record["anchor"].startswith("chapter."):
        raise ValueError(f"No full-build chapter record for {chapter_label}")
    chapter_number = int(chapter_record["number"])
    start_page = int(chapter_record["page"])

    transformed, replacement_count, missing = mapped_source(source, labels)
    if missing:
        raise ValueError(
            "Full-build aux is missing external labels: " + ", ".join(missing)
        )

    mapped_path = source_path.with_name(f"{source_path.stem}_layoutmapped.qmd")
    if mapped_path.exists():
        raise FileExistsError(f"Refusing to overwrite generated file: {mapped_path}")
    mapped_rel = mapped_path.relative_to(quarto_dir).as_posix()
    harness_dir = quarto_dir / "tmp" / "layout-harness" / source_path.stem
    output_dir = harness_dir / "output"
    harness_config = harness_dir / "_quarto.yml"
    manifest_path = harness_dir / "manifest.json"
    harness_dir.mkdir(parents=True, exist_ok=True)

    pdf_config = Path(config_manager.get_config_file("pdf", volume)).resolve()
    config = yaml.safe_load(pdf_config.read_text(encoding="utf-8"))
    config["project"]["output-dir"] = output_dir.relative_to(quarto_dir).as_posix()
    config["project"]["post-render"] = []
    config["project"]["render"] = ["index.qmd", mapped_rel]
    output_stem = f"Mapped-{source_path.stem}"
    config["book"]["output-file"] = output_stem
    config["book"]["chapters"] = ["index.qmd", mapped_rel]
    config["book"]["appendices"] = []
    config["format"]["titlepage-pdf"].setdefault("include-in-header", []).append(
        _counter_hook(chapter_number, start_page)
    )
    harness_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    active_config = quarto_dir / "_quarto.yml"
    active_index = quarto_dir / "index.qmd"
    if not active_config.is_symlink() or not active_index.is_symlink():
        raise RuntimeError("Expected _quarto.yml and index.qmd to be symlinks")
    original_config = os.readlink(active_config)
    original_index = os.readlink(active_index)
    target_index = f"index-{volume}.qmd"

    with _worktree_lock(quarto_dir / "tmp" / "layout-harness" / ".render.lock"):
        mapped_path.write_text(transformed, encoding="utf-8")
        try:
            active_config.unlink()
            active_config.symlink_to(harness_config.relative_to(quarto_dir))
            active_index.unlink()
            active_index.symlink_to(target_index)
            subprocess.run(
                ["quarto", "render", "--to=titlepage-pdf"],
                cwd=quarto_dir,
                check=True,
            )
        finally:
            if active_config.exists() or active_config.is_symlink():
                active_config.unlink()
            active_config.symlink_to(original_config)
            if active_index.exists() or active_index.is_symlink():
                active_index.unlink()
            active_index.symlink_to(original_index)
            mapped_path.unlink(missing_ok=True)

    corrected_count = _correct_callout_numbers(
        quarto_dir, output_dir, output_stem, chapter_number
    )
    pdf_path = output_dir / f"{output_stem}.pdf"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "volume": volume,
        "chapter_source": source_path.relative_to(quarto_dir).as_posix(),
        "chapter_label": chapter_label,
        "chapter_number": chapter_number,
        "start_page": start_page,
        "external_references_mapped": replacement_count,
        "custom_callout_prefixes_corrected": corrected_count,
        "numbering_source": str(aux_path),
        "pdf": str(pdf_path),
        "source_modified": False,
        "active_config_restored": os.readlink(active_config) == original_config,
        "active_index_restored": os.readlink(active_index) == original_index,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest
