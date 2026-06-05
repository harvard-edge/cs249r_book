#!/usr/bin/env python3
"""
Post-render step: resolve cross-chapter cross-references in HTML output.

WHY THIS SCRIPT EXISTS:
-----------------------
vol1 and vol2 build as `project.type: website` (each chapter is its own Pandoc
invocation), so Quarto cannot resolve `@sec-`, `@fig-`, `@tbl-`, `@eq-`, `@lst-`,
`@pri-`, or `@nb-` references that point into a sibling chapter. They ship as
literal `?@xxx-yyy` text in the rendered HTML. Selective builds (only rendering
a few chapters during development) hit this for every cross-chapter ref.

This script is the **resolution step** of the render pipeline — not a "fix" for
a bug. The website-mode build pipeline relies on it to wire cross-chapter
references end-to-end.

WHAT THIS SCRIPT DOES:
----------------------
1. Scans QMD source files to dynamically build a mapping of section/figure/table
   IDs → HTML paths and titles.
2. Scans ALL HTML files and the generated search index in the build directory
   after Quarto finishes.
3. Finds unresolved references that appear as: <strong>?@sec-xxx</strong>
   (and the parallel forms for `<a href="@xxx-yyy">` and quarto-unresolved spans).
4. Converts them to proper HTML links: <strong><a href="../path/to/chapter.html#sec-xxx">Title</a></strong>

The dynamic approach means you never need to update this script when adding
chapters or renaming sections — it reads the source QMDs directly.

WHEN IT RUNS:
-------------
This script runs as a post-render hook in the Quarto configuration:
  post-render:
    - scripts/clean_svgs.py
    - scripts/resolve_cross_references.py  # <-- Runs after all HTML is generated

HOW TO USE:
-----------
1. Automatic: Runs automatically during `quarto render` as a post-render hook
2. Manual: python3 scripts/resolve_cross_references.py [specific-file.html]
3. Test all: python3 scripts/resolve_cross_references.py  # processes all HTML files
"""

import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Crossref prefixes that we resolve at post-render time.
#
# Quarto book projects resolve cross-chapter refs natively, but vol1/vol2 build
# as `project.type: website` (see config/_quarto-html-vol*.yml), so any ref
# pointing into a sibling chapter ships as a literal `?@xxx-yyy` in the HTML.
# This script patches those refs by maintaining its own label registry.
#
# Prefixes covered: Quarto's built-ins (sec/fig/tbl/eq/lst/thm-family) plus the
# book's custom prefixes (pri for callout-principle, nb for notebook callouts).
# When adding a new prefix elsewhere in the codebase, add it here too.
# ---------------------------------------------------------------------------

CROSSREF_PREFIXES_LIST = [
    "sec", "pri", "fig", "tbl", "eq", "lst", "nb",
    "thm", "lem", "cor", "prp", "cnj", "def", "exm", "exr", "rem", "sol",
]
CROSSREF_PREFIXES = "(?:" + "|".join(CROSSREF_PREFIXES_LIST) + ")"


# ---------------------------------------------------------------------------
# Dynamic mapping: built by scanning QMD sources at runtime
# ---------------------------------------------------------------------------

def _extract_heading_text(line: str) -> str:
    """Strip Markdown heading markers, {#...} anchors, and attributes from a line."""
    # Remove leading # chars and space
    text = re.sub(r'^#+\s*', '', line)
    # Remove {#id ...} block (Quarto heading attribute)
    text = re.sub(r'\{[^}]*\}', '', text)
    return _to_visible_title_text(text.strip())


# Pull the **Bold Title** slice out of a caption line, fig-cap, or tbl-cap.
# Captions in this book follow the convention `**Bold Title**: Explanation.`
# (book-prose.md §6 Figure Captions & Alt-Text). The bold title is the
# reader-facing name of the figure/table — exactly the right link text for a
# cross-chapter reference.
_BOLD_TITLE_RE = re.compile(r'\*\*([^*]+?)\*\*')
_TITLE_TEXT_WRAP_RE = re.compile(r"\\(?:text|mathrm|mathbf|mathit|mathsf|operatorname)\{([^{}]*)\}")
_TITLE_SPACING_RE = re.compile(r"\\[,;:! ]|\\quad|\\qquad")
_TITLE_GLYPHS = (
    (r"\times", "×"),
    (r"\cdot", "·"),
    (r"\approx", "≈"),
    (r"\leq", "≤"),
    (r"\geq", "≥"),
    (r"\neq", "≠"),
    (r"\le", "≤"),
    (r"\ge", "≥"),
    (r"\pm", "±"),
    (r"\sim", "~"),
    (r"\rightarrow", "→"),
    (r"\to", "→"),
    (r"\leftarrow", "←"),
    (r"\infty", "∞"),
    (r"\mu", "μ"),
    (r"\alpha", "α"),
    (r"\beta", "β"),
    (r"\rho", "ρ"),
)
_TITLE_CURRENCY_SENTINEL = "\x00USD\x00"


def _to_visible_title_text(text: str) -> str:
    """Convert simple inline math in source-extracted titles to visible text.

    Cross-chapter xrefs are patched after rendering, so titles pulled from QMD
    source do not pass through Pandoc's math parser. This handles short title
    fragments such as ``$\\times$`` without treating captions themselves as
    invalid math contexts.
    """
    text = text.replace(r"\$", _TITLE_CURRENCY_SENTINEL)
    for _ in range(3):
        text, changed = _TITLE_TEXT_WRAP_RE.subn(r"\1", text)
        if not changed:
            break
    text = text.replace(r"\%", "%")
    for command, glyph in _TITLE_GLYPHS:
        text = re.sub(re.escape(command) + r"(?![A-Za-z])", glyph, text)
    text = _TITLE_SPACING_RE.sub("", text)
    text = text.replace("$$", "").replace("$", "")
    text = text.replace(_TITLE_CURRENCY_SENTINEL, "$")
    return re.sub(r"\s+", " ", text).strip()


def _extract_bold_title(text: str) -> str | None:
    m = _BOLD_TITLE_RE.search(text)
    if not m:
        return None
    title = m.group(1)
    # Defensive: strip any \index{...} that might have slipped inside the bold span.
    title = re.sub(r'\\index\{[^}]*\}', '', title).strip()
    title = _to_visible_title_text(title)
    return title or None


def _fallback_label(ref_id: str) -> str:
    """Human-readable fallback when a label has no caption or title attribute."""
    prefix = ref_id.split("-", 1)[0]
    return {
        "eq": "equation",
        "lst": "listing",
        "fig": "figure",
        "tbl": "table",
        "nb": "Notebook",
        "thm": "Theorem",
        "lem": "Lemma",
        "cor": "Corollary",
        "prp": "Proposition",
        "cnj": "Conjecture",
        "def": "Definition",
        "exm": "Example",
        "exr": "Exercise",
        "rem": "Remark",
        "sol": "Solution",
    }.get(prefix, ref_id)


def build_qmd_mapping(qmd_roots: list[Path]) -> tuple[dict, dict]:
    """
    Scan QMD source trees to build section ID → (html_path, title) mappings.

    For each QMD file we:
      - Find heading-level IDs:  ## Heading Text {#sec-foo}
      - Find div-level IDs:      :::: {#pri-foo .callout-principle title="The Title"}
    The HTML output path is derived by replacing .qmd → .html, keeping the same
    relative path from the project root (which mirrors the build output layout).

    Each qmd_root is a (scan_dir, path_prefix) pair — scan_dir is where we recurse,
    path_prefix is prepended to every output path so it matches the Quarto build tree.

    Args:
        qmd_roots: list of (scan_dir, path_prefix) tuples, or plain Path objects
                   (plain Path → scan_dir == path_prefix parent == scan_dir)

    Returns:
        (chapter_mapping, chapter_titles) — same shape as the old hardcoded dicts
    """
    chapter_mapping: dict[str, str] = {}
    chapter_titles: dict[str, str] = {}

    # Regex patterns to find IDs in QMD source
    # Heading: # Title {#sec-foo} or # Title {#sec-foo .class}
    heading_id_re = re.compile(r'^(#+)\s+(.+?)\s*\{#([\w-]+)[^}]*\}')
    # Div / callout — match any attribute block containing #id, regardless of position
    # Handles both:
    #   :::: {#pri-foo .callout-principle title="..."}
    #   :::: {.callout-principle #pri-foo title="..."}
    div_id_re = re.compile(r'^\s*:{2,}\s*\{([^}]*)\}')
    id_attr_re = re.compile(r'#([\w-]+)')
    title_attr_re = re.compile(r'title="([^"]*)"')
    # fig-cap / tbl-cap / lst-cap on figure/table/listing divs.
    cap_attr_re = re.compile(r'(?:fig|tbl|lst)-cap="([^"]*)"')

    # Caption-attribute label syntax (Quarto table & listing captions).
    # Example:  : **Bold Title**: explanation. {#tbl-foo tbl-colwidths="..."}
    # Captures (caption_text, label_id) where label_id starts with tbl-/lst-/fig-.
    # We also accept any other prefix in case a future caption-attribute form
    # appears for another type; the prefix filter in CROSSREF_PREFIXES gates the
    # patching pass anyway.
    caption_id_re = re.compile(r'^:\s+(.+?)\s*\{[^}]*#([\w-]+)[^}]*\}')

    # Equation label syntax:  $$ ... $$ {#eq-foo}
    # Single-line equations always have the {#eq-id} on the same line as the
    # closing $$. For multi-line equations Pandoc accepts the label on a line
    # immediately after the closing $$ as well — both cases land here because
    # we walk every line and match the {#eq-id} fragment anywhere.
    equation_id_re = re.compile(r'\{[^}]*#(eq-[\w-]+)[^}]*\}')

    # Normalise qmd_roots: accept plain Path (scan from project root)
    # We need paths relative to the *project* root, not the contents/ subdir
    normalised: list[tuple[Path, Path]] = []
    for entry in qmd_roots:
        if isinstance(entry, tuple):
            normalised.append(entry)
        else:
            # entry is a Path like Path("contents"); its parent is the project root
            normalised.append((entry, entry.parent))

    for scan_dir, path_root in normalised:
        if not scan_dir.exists():
            continue

        for qmd_file in sorted(scan_dir.rglob("*.qmd")):
            # Derive the relative HTML path from the project root
            rel_qmd = qmd_file.relative_to(path_root)
            rel_html = rel_qmd.with_suffix(".html")
            html_path_str = str(rel_html).replace("\\", "/")

            try:
                lines = qmd_file.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue

            for line in lines:
                # Try heading pattern first
                m = heading_id_re.match(line)
                if m:
                    hashes, raw_title, sec_id = m.group(1), m.group(2), m.group(3)
                    title = _extract_heading_text(raw_title)
                    depth = len(hashes)
                    # Only add if not already registered (first occurrence wins)
                    if sec_id not in chapter_mapping:
                        chapter_mapping[sec_id] = f"{html_path_str}#{sec_id}"
                        chapter_titles[sec_id] = title
                    continue

                # Try div/callout pattern (handles any attribute order inside { })
                m = div_id_re.match(line)
                if m:
                    attrs = m.group(1)
                    im = id_attr_re.search(attrs)
                    if not im:
                        continue
                    sec_id = im.group(1)
                    # Prefer explicit title=, then bold-title from fig-cap/tbl-cap/lst-cap,
                    # then a human fallback ("Figure", "Equation", ...), then the bare ID.
                    tm = title_attr_re.search(attrs)
                    if tm:
                        title = tm.group(1)
                    else:
                        title = None
                        cm = cap_attr_re.search(attrs)
                        if cm:
                            title = _extract_bold_title(cm.group(1))
                        if not title:
                            title = _fallback_label(sec_id)
                    if sec_id not in chapter_mapping:
                        chapter_mapping[sec_id] = f"{html_path_str}#{sec_id}"
                        chapter_titles[sec_id] = title
                    continue

                # Caption-attribute label syntax (table & listing captions).
                # Form: `: **Bold Title**: explanation. {#tbl-foo ...}`
                m = caption_id_re.match(line)
                if m:
                    caption_text, sec_id = m.group(1), m.group(2)
                    if sec_id not in chapter_mapping:
                        title = _extract_bold_title(caption_text) or _fallback_label(sec_id)
                        chapter_mapping[sec_id] = f"{html_path_str}#{sec_id}"
                        chapter_titles[sec_id] = title
                    continue

                # Equation label syntax: $$ ... $$ {#eq-foo}.
                # We match the {#eq-id} fragment anywhere on the line so both the
                # single-line and trailing-attribute multi-line forms register.
                # Skip lines starting with `:` (already handled above as caption)
                # or `#` (already handled as heading) to avoid double-registration.
                if not line.lstrip().startswith((":", "#")):
                    m = equation_id_re.search(line)
                    if m:
                        sec_id = m.group(1)
                        if sec_id not in chapter_mapping:
                            chapter_mapping[sec_id] = f"{html_path_str}#{sec_id}"
                            chapter_titles[sec_id] = _fallback_label(sec_id)

    return chapter_mapping, chapter_titles


def _find_qmd_roots() -> list[tuple[Path, Path]]:
    """
    Locate the Quarto project source tree from common run locations.

    Quarto post-render hooks run from the project root (where _quarto.yml lives).
    Manual runs may happen from the scripts/ directory or the repo root.

    Returns list of (scan_dir, path_root) tuples for build_qmd_mapping.
    """
    candidates = [
        Path("."),           # run from project root (most common)
        Path(".."),          # run from scripts/
        Path("book/quarto"), # run from repo root
    ]
    for c in candidates:
        if (c / "contents").exists():
            return [(c / "contents", c)]
    return []


# Build the mapping once at import time (lazy cache via module-level variable)
_CHAPTER_MAPPING: dict | None = None
_CHAPTER_TITLES: dict | None = None
_PRINCIPLE_NUMBERS: dict | None = None


def get_mappings() -> tuple[dict, dict]:
    global _CHAPTER_MAPPING, _CHAPTER_TITLES
    if _CHAPTER_MAPPING is None:
        roots = _find_qmd_roots()
        if roots:
            _CHAPTER_MAPPING, _CHAPTER_TITLES = build_qmd_mapping(roots)
        else:
            _CHAPTER_MAPPING, _CHAPTER_TITLES = {}, {}
    return _CHAPTER_MAPPING, _CHAPTER_TITLES


# ---------------------------------------------------------------------------
# Principle numbering
#
# Principle callouts use a per-volume global counter that increments across the
# four parts/*_principles.qmd files in declared order. The Lua filter does this
# at render time, but the count is only available within the Pandoc invocation
# that's currently rendering a parts file — sibling chapters that reference
# principles via `Principle \ref{pri-X}` never see the number.
#
# We replay the count here so cross-chapter references in HTML can substitute
# the resolved `Principle N` text. The parts file render order must match what
# Quarto + Lua see during a full build (the order in config/_quarto-html-vol*.yml).
# ---------------------------------------------------------------------------

# Stable parts-file order per volume. Matches the declared order in
# config/_quarto-html-vol*.yml and config/_quarto-pdf-vol*-copyedit.yml.
PRINCIPLE_PARTS_ORDER: dict[str, list[str]] = {
    "vol1": [
        "contents/vol1/parts/foundations_principles.qmd",
        "contents/vol1/parts/build_principles.qmd",
        "contents/vol1/parts/optimize_principles.qmd",
        "contents/vol1/parts/deploy_principles.qmd",
    ],
    "vol2": [
        "contents/vol2/parts/fleet_principles.qmd",
        "contents/vol2/parts/distributed_ml_principles.qmd",
        "contents/vol2/parts/deployment_principles.qmd",
        "contents/vol2/parts/responsible_fleet_principles.qmd",
    ],
}


def build_principle_numbers(roots: list[tuple[Path, Path]]) -> dict[str, str]:
    """
    Walk the parts/*_principles.qmd files in declared order and count
    `.callout-principle` divs to compute each principle's assigned number.

    Returns {pri_id: number_str}. Per-volume independent numbering.
    """
    # Match a div opener that carries BOTH `.callout-principle` and `#pri-X`.
    # Attributes can appear in any order inside the {...} block.
    div_opener_re = re.compile(r'^\s*:{2,}\s*\{([^}]*)\}')
    pri_id_re = re.compile(r'#(pri-[\w-]+)')
    has_callout_principle = re.compile(r'\.callout-principle\b')

    result: dict[str, str] = {}
    for _, path_root in roots:
        if not isinstance(path_root, Path):
            continue
        for vol, parts_paths in PRINCIPLE_PARTS_ORDER.items():
            counter = 0
            for rel in parts_paths:
                qmd = path_root / rel
                if not qmd.exists():
                    continue
                try:
                    lines = qmd.read_text(encoding="utf-8").splitlines()
                except Exception:
                    continue
                for line in lines:
                    m_open = div_opener_re.match(line)
                    if not m_open:
                        continue
                    attrs = m_open.group(1)
                    if not has_callout_principle.search(attrs):
                        continue
                    m_id = pri_id_re.search(attrs)
                    if not m_id:
                        continue
                    counter += 1
                    pri_id = m_id.group(1)
                    if pri_id not in result:
                        result[pri_id] = str(counter)
        break  # roots[0] is the project root we want
    return result


def get_principle_numbers() -> dict:
    global _PRINCIPLE_NUMBERS
    if _PRINCIPLE_NUMBERS is None:
        roots = _find_qmd_roots()
        if roots:
            _PRINCIPLE_NUMBERS = build_principle_numbers(roots)
        else:
            _PRINCIPLE_NUMBERS = {}
    return _PRINCIPLE_NUMBERS


# ---------------------------------------------------------------------------
# EPUB support
# ---------------------------------------------------------------------------

def build_epub_section_mapping(epub_dir: Path) -> dict:
    """
    Build mapping from section IDs to EPUB chapter files by scanning actual chapters.

    Args:
        epub_dir: Path to EPUB build directory (_build/epub or extracted EPUB root)

    Returns:
        Dictionary mapping section IDs to chapter filenames (e.g., {"sec-xxx": "ch004.xhtml"})
    """
    mapping: dict[str, str] = {}

    possible_text_dirs = [
        epub_dir / "text",
        epub_dir / "EPUB" / "text",
    ]

    text_dir = None
    for dir_path in possible_text_dirs:
        if dir_path.exists():
            text_dir = dir_path
            break

    if not text_dir:
        return mapping

    for xhtml_file in sorted(text_dir.glob("ch*.xhtml")):
        try:
            content = xhtml_file.read_text(encoding="utf-8")
            for sec_id in re.findall(r'id="(sec-[^"]+)"', content):
                mapping[sec_id] = xhtml_file.name
        except Exception:
            continue

    return mapping


# ---------------------------------------------------------------------------
# Path calculation
# ---------------------------------------------------------------------------

def calculate_relative_path(
    from_file: Path,
    to_path: str,
    build_dir: Path,
    epub_mapping: dict | None = None,
) -> str:
    """
    Calculate relative path from one file to another.

    Args:
        from_file: Path object of the source file
        to_path: String path from build root (e.g., "contents/vol1/chapter/file.html#anchor")
        build_dir: Path object of the build directory root
        epub_mapping: Optional dict mapping section IDs to EPUB chapter files

    Returns:
        Relative path string from from_file to to_path
    """
    if epub_mapping is not None:
        if "#" in to_path:
            _, sec_id = to_path.split("#", 1)
            target_chapter = epub_mapping.get(sec_id)
            if target_chapter:
                return f"{target_chapter}#{sec_id}"
        return to_path

    if "#" in to_path:
        target_path_str, anchor = to_path.split("#", 1)
        anchor = f"#{anchor}"
    else:
        target_path_str = to_path
        anchor = ""

    target_abs = build_dir / target_path_str
    source_abs = from_file

    try:
        rel_path = Path(target_abs).relative_to(source_abs.parent)
        result = str(rel_path).replace("\\", "/")
    except ValueError:
        source_parts = source_abs.parent.parts
        target_parts = target_abs.parts

        common_length = 0
        for s, t in zip(source_parts, target_parts):
            if s == t:
                common_length += 1
            else:
                break

        up_levels = len(source_parts) - common_length
        down_parts = target_parts[common_length:]
        rel_parts = [".."] * up_levels + list(down_parts)
        result = "/".join(rel_parts)

    return result + anchor


# ---------------------------------------------------------------------------
# Cross-reference fixing
# ---------------------------------------------------------------------------

def fix_cross_reference_link(match, from_file, build_dir, epub_mapping=None):
    """Replace a single cross-reference link with proper HTML link."""
    full_match = match.group(0)
    sec_ref = match.group(1)

    chapter_mapping, chapter_titles = get_mappings()
    abs_path = chapter_mapping.get(sec_ref)
    title = chapter_titles.get(sec_ref)

    if abs_path and title:
        rel_path = calculate_relative_path(from_file, abs_path, build_dir, epub_mapping)
        return f'<a href="{rel_path}">{title}</a>'
    else:
        print(f"   ⚠️  No mapping found for: {sec_ref}")
        return full_match


def resolve_cross_references(
    html_content: str,
    from_file: Path,
    build_dir: Path,
    epub_mapping: dict | None = None,
) -> tuple[str, int, list]:
    """
    Resolve all cross-reference links in HTML/XHTML content.

    Quarto generates three types of unresolved references when chapters aren't built:
    1. Full unresolved links: <a href="#sec-xxx" class="quarto-xref"><span class="quarto-unresolved-ref">...</span></a>
    2. Simple unresolved refs: <strong>?@sec-xxx</strong> (common in selective builds)
    3. EPUB unresolved refs: <a href="@sec-xxx">Link Text</a>
    """
    chapter_mapping, chapter_titles = get_mappings()

    # All three patterns cover the full set of crossref prefixes (CROSSREF_PREFIXES)
    # — sec, pri, fig, tbl, eq, lst, nb, plus Quarto's theorem family. Quarto emits
    # unresolved refs in three shapes:
    #   1. Full xref link with a quarto-unresolved-ref span (rare in our build).
    #   2. Bare `?@xxx-yyy` wrapped in <strong> (common — every cross-chapter ref
    #      ships as this when the project is `type: website`).
    #   3. EPUB-specific `<a href="@xxx-yyy">…</a>`.
    pattern1 = rf'<a href="#({CROSSREF_PREFIXES}-[a-zA-Z0-9-]+)" class="quarto-xref"><span class="quarto-unresolved-ref">[^<]*</span></a>'
    pattern2 = rf'<strong>\?\@({CROSSREF_PREFIXES}-[a-zA-Z0-9-]+)</strong>'
    pattern3 = rf'<a href="@({CROSSREF_PREFIXES}-[a-zA-Z0-9-]+)"([^>]*)>([^<]*)</a>'
    # Pattern 4 — `Principle \ref{pri-X}` leaks as an inline-math span.
    # Source prose says: `Principle \ref{pri-data-as-code}`. Pandoc parses the
    # `\ref{...}` as inline math, so it ships to HTML as:
    #     <span class="math inline">\(\ref{pri-data-as-code}\)</span>
    # MathJax then renders the undefined `\ref` as `???`. We replace the whole
    # math span with a link to the principles page using the resolved number.
    pattern4 = r'<span class="math inline">\\\(\\ref\{(pri-[a-zA-Z0-9-]+)\}\\\)</span>'

    matches1 = re.findall(pattern1, html_content)
    matches2 = re.findall(pattern2, html_content)
    matches3 = re.findall(pattern3, html_content)
    matches4 = re.findall(pattern4, html_content)
    total_matches = len(matches1) + len(matches2) + len(matches3) + len(matches4)

    # Fix Pattern 1
    fixed_content = re.sub(
        pattern1,
        lambda m: fix_cross_reference_link(m, from_file, build_dir, epub_mapping),
        html_content,
    )

    # Fix Pattern 2
    unmapped_refs = []

    def fix_simple_reference(match):
        sec_ref = match.group(1)
        abs_path = chapter_mapping.get(sec_ref)
        title = chapter_titles.get(sec_ref)
        if abs_path and title:
            rel_path = calculate_relative_path(from_file, abs_path, build_dir, epub_mapping)
            return f'<strong><a href="{rel_path}">{title}</a></strong>'
        else:
            unmapped_refs.append(sec_ref)
            return match.group(0)

    fixed_content = re.sub(pattern2, fix_simple_reference, fixed_content)

    # Fix Pattern 3 (EPUB)
    def fix_epub_reference(match):
        sec_ref = match.group(1)
        attrs = match.group(2)
        link_text = match.group(3)

        if epub_mapping:
            target_chapter = epub_mapping.get(sec_ref)
            if target_chapter:
                return f'<a href="{target_chapter}#{sec_ref}"{attrs}>{link_text}</a>'
            else:
                unmapped_refs.append(sec_ref)
                return match.group(0)
        else:
            abs_path = chapter_mapping.get(sec_ref)
            if abs_path:
                rel_path = calculate_relative_path(from_file, abs_path, build_dir, None)
                return f'<a href="{rel_path}"{attrs}>{link_text}</a>'
            else:
                unmapped_refs.append(sec_ref)
                return match.group(0)

    fixed_content = re.sub(pattern3, fix_epub_reference, fixed_content)

    # Fix Pattern 4 — Principle \ref{pri-X} math-span leak
    principle_numbers = get_principle_numbers()

    def fix_principle_reference(match):
        pri_ref = match.group(1)
        number = principle_numbers.get(pri_ref)
        abs_path = chapter_mapping.get(pri_ref)
        if number and abs_path:
            rel_path = calculate_relative_path(from_file, abs_path, build_dir, epub_mapping)
            return f'<a href="{rel_path}">{number}</a>'
        elif number:
            # Have the number but no link target — emit number as plain text.
            return number
        else:
            unmapped_refs.append(pri_ref)
            return match.group(0)

    fixed_content = re.sub(pattern4, fix_principle_reference, fixed_content)

    remaining1 = re.findall(pattern1, fixed_content)
    remaining2 = re.findall(pattern2, fixed_content)
    remaining3 = re.findall(pattern3, fixed_content)
    remaining4 = re.findall(pattern4, fixed_content)
    fixed_count = total_matches - len(remaining1) - len(remaining2) - len(remaining3) - len(remaining4)

    return fixed_content, fixed_count, unmapped_refs


def resolve_search_text(text: str) -> tuple[str, int, list[str]]:
    """Resolve bare `?@label` tokens in Quarto's generated search index text."""
    chapter_mapping, chapter_titles = get_mappings()
    unmapped_refs: list[str] = []

    pattern = rf'\?\@({CROSSREF_PREFIXES}-[a-zA-Z0-9-]+)'

    def fix_search_reference(match):
        ref = match.group(1)
        title = chapter_titles.get(ref)
        if ref in chapter_mapping and title:
            return title
        unmapped_refs.append(ref)
        return match.group(0)

    fixed_text, fixed_count = re.subn(pattern, fix_search_reference, text)
    return fixed_text, fixed_count, unmapped_refs


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def process_html_file(html_file: Path, base_dir: Path, epub_mapping: dict | None = None):
    """Process a single HTML/XHTML file to fix cross-references."""
    try:
        html_content = html_file.read_text(encoding="utf-8")
    except Exception:
        return None, 0, []

    fixed_content, fixed_count, unmapped = resolve_cross_references(
        html_content, html_file, base_dir, epub_mapping
    )

    if fixed_count > 0:
        try:
            html_file.write_text(fixed_content, encoding="utf-8")
            return html_file.relative_to(base_dir), fixed_count, unmapped
        except Exception:
            return None, 0, []

    return None, 0, []


def _map_json_strings(value, mapper):
    """Return JSON value with mapper applied to every nested string."""
    if isinstance(value, str):
        return mapper(value)
    if isinstance(value, list):
        return [_map_json_strings(item, mapper) for item in value]
    if isinstance(value, dict):
        return {key: _map_json_strings(item, mapper) for key, item in value.items()}
    return value


def process_search_json_file(search_file: Path, base_dir: Path):
    """Process Quarto's search.json to remove stale `?@label` text."""
    try:
        data = json.loads(search_file.read_text(encoding="utf-8"))
    except Exception:
        return None, 0, []

    total_fixed = 0
    all_unmapped: list[str] = []

    def mapper(text: str) -> str:
        nonlocal total_fixed, all_unmapped
        fixed_text, fixed_count, unmapped = resolve_search_text(text)
        total_fixed += fixed_count
        all_unmapped.extend(unmapped)
        return fixed_text

    fixed_data = _map_json_strings(data, mapper)

    if total_fixed > 0:
        try:
            search_file.write_text(
                json.dumps(fixed_data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return search_file.relative_to(base_dir), total_fixed, all_unmapped
        except Exception:
            return None, 0, []

    return None, 0, []


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Main entry point. Runs in three modes:
    1. Post-render hook (no args): Processes HTML or EPUB builds from _build/
    2. Directory mode (dir arg): Processes extracted EPUB directory
    3. Manual mode (file arg): Processes a specific file
    """
    # Pre-load mappings and report count
    chapter_mapping, chapter_titles = get_mappings()
    if chapter_mapping:
        print(f"📖 Loaded {len(chapter_mapping)} section IDs from QMD sources")
    else:
        print("⚠️  No QMD sources found — cross-reference mapping will be empty")

    skip_patterns = [
        "search.html", "404.html", "site_libs",
        "nav.xhtml", "cover.xhtml", "title_page.xhtml",
    ]

    if len(sys.argv) == 1:
        # MODE 1: Running as Quarto post-render hook
        build_root = Path("_build")
        html_candidates = sorted(
            [p for p in build_root.glob("html*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        epub_candidates = sorted(
            [p for p in build_root.glob("epub*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        epub_mapping = None
        if html_candidates:
            build_dir = html_candidates[0]
            file_pattern = "*.html"
            file_type = "HTML"
        elif epub_candidates and list(epub_candidates[0].glob("*.xhtml")):
            build_dir = epub_candidates[0]
            file_pattern = "*.xhtml"
            file_type = "XHTML (EPUB)"
            print("📚 Building EPUB section mapping...")
            epub_mapping = build_epub_section_mapping(build_dir)
            print(f"   Found {len(epub_mapping)} section IDs across chapters")
        elif Path("EPUB").exists() and list(Path("EPUB").rglob("*.xhtml")):
            build_dir = Path(".")
            file_pattern = "*.xhtml"
            file_type = "XHTML (EPUB - extracted)"
            print("📚 Building EPUB section mapping...")
            epub_mapping = build_epub_section_mapping(Path("."))
            print(f"   Found {len(epub_mapping)} section IDs across chapters")
        else:
            print("⚠️  No HTML or EPUB build directory found — skipping")
            sys.exit(0)

        files = list(build_dir.rglob(file_pattern))
        search_files = [build_dir / "search.json"] if (build_dir / "search.json").exists() else []
        print(
            f"🔗 [Cross-Reference Fix] Scanning {len(files)} {file_type} files"
            f" and {len(search_files)} search index file(s)..."
        )

        files_fixed = []
        total_refs_fixed = 0
        all_unmapped: set[str] = set()

        for file in files:
            if any(skip in str(file) for skip in skip_patterns):
                continue

            rel_path, fixed_count, unmapped = process_html_file(file, build_dir, epub_mapping)
            if fixed_count > 0:
                files_fixed.append((rel_path, fixed_count))
                total_refs_fixed += fixed_count
            all_unmapped.update(unmapped)

        for file in search_files:
            rel_path, fixed_count, unmapped = process_search_json_file(file, build_dir)
            if fixed_count > 0:
                files_fixed.append((rel_path, fixed_count))
                total_refs_fixed += fixed_count
            all_unmapped.update(unmapped)

        if files_fixed:
            print(f"✅ Fixed {total_refs_fixed} cross-references in {len(files_fixed)} files:")
            for path, count in files_fixed:
                print(f"   📄 {path}: {count} refs")
        else:
            print("✅ No unresolved cross-references found")

        if all_unmapped:
            print(f"⚠️  Unmapped references: {', '.join(sorted(all_unmapped))}")

    elif len(sys.argv) == 2:
        # MODE 2: Running with explicit file argument
        html_file = Path(sys.argv[1])
        if not html_file.exists():
            print(f"❌ File not found: {html_file}")
            sys.exit(1)

        epub_mapping = None
        if "text" in html_file.parts and html_file.suffix == ".xhtml":
            epub_base = html_file.parent.parent
            print("📚 Building EPUB section mapping...")
            epub_mapping = build_epub_section_mapping(epub_base)
            print(f"   Found {len(epub_mapping)} section IDs across chapters")

        print(f"🔗 Fixing cross-reference links in: {html_file}")
        if html_file.name == "search.json":
            rel_path, fixed_count, unmapped = process_search_json_file(
                html_file, html_file.parent
            )
        else:
            rel_path, fixed_count, unmapped = process_html_file(
                html_file, html_file.parent, epub_mapping
            )
        if fixed_count > 0:
            print(f"✅ Fixed {fixed_count} cross-references")
            if unmapped:
                print(f"⚠️  Unmapped references: {', '.join(sorted(unmapped))}")
        else:
            print("✅ No cross-reference fixes needed")

    else:
        print("Usage: python3 resolve_cross_references.py [<html-or-xhtml-file>]")
        sys.exit(1)


if __name__ == "__main__":
    main()
