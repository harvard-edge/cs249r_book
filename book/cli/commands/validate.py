"""
Native validation commands for MLSysBook Binder CLI.

Validation logic is implemented in Binder where possible (e.g. references,
citations, labels, figures, rendering). Some checks still delegate to scripts
under book/tools/scripts/ (tables, spelling, epub, sources, grid-tables,
images).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import reference_check

console = Console()


@dataclass
class ValidationIssue:
    file: str
    line: int
    code: str
    message: str
    severity: str = "error"
    context: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "context": self.context,
        }


@dataclass
class ValidationRunResult:
    name: str
    description: str
    files_checked: int
    issues: List[ValidationIssue]
    elapsed_ms: int

    @property
    def passed(self) -> bool:
        return len(self.issues) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "files_checked": self.files_checked,
            "passed": self.passed,
            "issue_count": len(self.issues),
            "elapsed_ms": self.elapsed_ms,
            "issues": [issue.to_dict() for issue in self.issues],
        }


INLINE_REF_PATTERN = re.compile(r"`\{python\}\s+(\w+(?:\.\w+)?)`")
CELL_START_PATTERN = re.compile(r"^```\{python\}|^```python")
CELL_END_PATTERN = re.compile(r"^```\s*$")
ASSIGN_PATTERN = re.compile(r"^([A-Za-z_]\w*)\s*=")
# Tuple unpacking: "a, b = ..." — captures all names on the left side
TUPLE_ASSIGN_PATTERN = re.compile(r"^((?:[A-Za-z_]\w*\s*,\s*)+[A-Za-z_]\w*)\s*=")
CLASS_DEF_PATTERN = re.compile(r"^class\s+(\w+)\s*[:(]")
GRID_TABLE_SEP_PATTERN = re.compile(r"^\+[-:=+]+\+$")
LATEX_INLINE_PATTERN = re.compile(r"(?<!\\)\$\s*`\{python\}\s+(?!\w+(?:\.\w+)?_str)[^`]+`|`\{python\}\s+(?!\w+(?:\.\w+)?_str)[^`]+`\s*(?<!\\)\$")
LATEX_ADJACENT_PATTERN = re.compile(r"`\{python\}\s+(?!\w+(?:\.\w+)?_str)[^`]+`\s*\$\\(times|approx|ll|gg|mu)\$")

CITATION_REF_PATTERN = re.compile(r"@([A-Za-z0-9_:\-.]+)")
CITATION_BRACKET_PATTERN = re.compile(r"\[-?@[A-Za-z0-9_:\-.]+(?:;\s*-?@[A-Za-z0-9_:\-.]+)*\]")

LABEL_DEF_PATTERNS = {
    "Figure": [
        re.compile(r"\{#(fig-[\w-]+)"),              # {#fig-xyz ...}
        re.compile(r"#\|\s*label:\s*(fig-[\w-]+)"),  # #| label: fig-xyz
        re.compile(r"%%\|\s*label:\s*(fig-[\w-]+)"), # %%| label: fig-xyz (Jupyter)
    ],
    "Table": [
        re.compile(r"\{#(tbl-[\w-]+)"),              # {#tbl-xyz}
        re.compile(r"#\|\s*label:\s*(tbl-[\w-]+)"),  # #| label: tbl-xyz
    ],
    "Section": [
        re.compile(r"\{#(sec-[\w-]+)"),              # {#sec-xyz}
        re.compile(r"^id:\s*(sec-[\w-]+)"),          # YAML id: sec-xyz
    ],
    "Equation": [re.compile(r"\{#(eq-[\w-]+)")],     # {#eq-xyz}
    "Listing": [
        re.compile(r"\{#(lst-[\w-]+)"),              # {#lst-xyz ...}
        re.compile(r"#\|\s*label:\s*(lst-[\w-]+)"),  # #| label: lst-xyz
    ],
}
LABEL_REF_PATTERN = re.compile(r"@((?:fig|tbl|sec|eq|lst)-[\w-]+)")

EXCLUDED_CITATION_PREFIXES = ("fig-", "tbl-", "sec-", "eq-", "lst-", "ch-", "nb-")


class ValidateCommand:
    """Native `binder check` command group (also available as `binder validate`).

    Groups:
        refs        — inline-python, cross-refs, citations, inline patterns
        labels      — duplicate labels, orphaned/unreferenced labels
        headers     — section header IDs
        footnotes   — placement rules, reference integrity, capitalization
        figures     — captions/alt-text, float flow, image files
        rendering   — render patterns, indexes, dropcaps, parts
        all         — run every check
    """

    # Maps group name → list of (scope_name, runner_method_name) pairs.
    # This is the single source of truth for the hierarchy.
    GROUPS: Dict[str, List[tuple]] = {
        "refs": [
            ("python-syntax", "_run_python_syntax"),
            ("inline-python", "_run_inline_python"),
            ("cross-refs", "_run_refs"),
            ("citations", "_run_citations"),
            ("inline", "_run_inline_refs"),
            ("self-ref", "_run_self_referential"),
            ("capitalized", "_run_mitpress_capitalized_refs"),  # "chapter 12" lowercase in prose
        ],
        "labels": [
            ("duplicates", "_run_duplicate_labels"),
            ("orphans", "_run_unreferenced_labels"),
            ("fig-labels", "_run_fig_label_underscores"),
        ],
        "headers": [
            ("ids", "_run_headers"),
            ("case", "_run_heading_case"),
        ],
        "bib": [
            ("hygiene", "_run_bib_hygiene"),
        ],
        "footnotes": [
            ("placement", "_run_footnote_placement"),
            ("integrity", "_run_footnote_refs"),
            ("cross-chapter", "_run_footnote_cross_chapter"),
            ("capitalization", "_run_footnote_capitalization"),
        ],
        "figures": [
            ("captions", "_run_figures"),
            ("div-syntax", "_run_figure_div_syntax"),
            ("flow", "_run_float_flow"),
            ("files", "_run_images"),
        ],
        # ------------------------------------------------------------------
        # Semantic check groups: classify checks by WHAT is validated
        # (markup patterns, prose style, punctuation, numbers, math, etc.),
        # not by where-the-rule-came-from. A rule's provenance belongs in
        # a comment, not a command name — hence no `mitpress-` prefix on
        # scopes. See .claude/rules/book-prose.md for style provenance.
        # ------------------------------------------------------------------
        "markup": [
            ("patterns", "_run_rendering"),       # low-level markup patterns (backticks, dollar signs, asterisks)
            ("div-fences", "_run_div_fences"),    # ::: / :::: balance + form
            ("dropcaps", "_run_dropcaps"),        # drop-cap compatibility
        ],
        "prose": [
            ("contractions", "_run_contractions"),           # no "can't", "it's" in body prose
            ("duplicate-words", "_run_duplicate_words"),     # consecutive dup words
            ("unblended-prose", "_run_unblended_prose"),     # space after period
            ("above-below", "_run_mitpress_above_below"),    # no "above"/"below" spatial refs
            ("ascii", "_run_ascii"),                         # non-ASCII chars
            ("acknowledgements", "_run_mitpress_acknowledgements"),  # American spelling
        ],
        "punctuation": [
            ("emdash", "_run_mitpress_spaced_emdash"),        # word—word, no spaces
            ("slash", "_run_mitpress_spaced_slash"),          # training/inference, no spaces
            ("vs-period", "_run_mitpress_vs_period"),         # vs. not vs
            ("eg-ie-comma", "_run_mitpress_eg_ie_comma"),     # comma after e.g./i.e.
            ("hyphen-range", "_run_mitpress_hyphen_range"),   # en-dash for number ranges
        ],
        "numbers": [
            ("unit-spacing", "_run_unit_spacing"),                          # 100 ms, not 100ms
            ("binary-units", "_run_binary_units"),                          # GB/TB not GiB/TiB
            ("percent-spacing", "_run_percent_spacing"),                    # no space before %
            ("percent-in-captions", "_run_mitpress_percent_in_captions"),   # spell out in captions
        ],
        "math": [
            ("times-spacing", "_run_times_spacing"),                 # space after $\times$
            ("times-product-spacing", "_run_times_product_spacing"), # space before $\times$ after inline code
            ("attr-leaks", "_run_attr_latex_leaks"),                 # LaTeX in title=/fig-cap/tbl-cap/fig-alt/tbl-alt — won't render or leaks to lightbox
            ("render-audit", "_run_math_render_audit"),              # full HTML build + leak scan (slow; manual stage)
        ],
        "structure": [
            ("heading-levels", "_run_heading_levels"),    # H1→H2→H3 hierarchy
            ("parts", "_run_parts"),                       # part keys valid
            ("purpose-unnumbered", "_run_purpose_unnumbered"),  # Purpose sections unnumbered
        ],
        "code": [
            ("python-echo", "_run_python_echo"),           # echo: false on python blocks
            ("str-latex-leak", "_run_str_latex_leak"),     # *_str exports must not contain raw LaTeX (use md()/md_math())
        ],
        "tables": [
            ("grid-tables", "_run_grid_tables"),           # prefer pipe tables
            ("content", "_run_table_content"),             # bare pipes, fracs, HTML entities
        ],
        "index": [
            ("placement", "_run_indexes"),                 # \index{} not inline with headings/callouts
        ],

        "images": [
            ("formats", "_run_image_formats"),
            ("external", "_run_external_images"),
            ("svg-xml", "_run_svg_wellformedness"),
        ],
        "json": [
            ("syntax", "_run_json_syntax"),
        ],
        "units": [
            ("physics", "_run_unit_tests"),
        ],
        "spelling": [
            ("prose", "_run_spelling_prose"),
            ("tikz", "_run_spelling_tikz"),
        ],
        "epub": [
            # Fast source-level invariants. No EPUB build required. Suitable
            # for pre-commit — runs in <1s across all SVGs + .bib files.
            ("hygiene", "_run_epub_hygiene"),
            # Reader-compatibility smoke checks against the built EPUB.
            # No Java required; catches patterns epubcheck does not, e.g.
            # CSS custom properties (ClearView / Tolino compat) and
            # external resource references.
            ("smoke", "_run_epub_smoke"),
            # Full W3C epubcheck validation of built EPUB artifacts under
            # _build/epub-vol*/. Requires epubcheck + JRE. Slow (~30s per
            # volume) — appropriate for CI, not pre-commit.
            ("epubcheck", "_run_epubcheck"),
        ],
        "sources": [
            ("citations", "_run_sources"),
        ],
        "references": [
            ("hallucinator", "_run_check_references"),
        ],
        "content": [
            ("tree", "_run_content_tree"),
        ],
    }

    def __init__(self, config_manager, chapter_discovery):
        self.config_manager = config_manager
        self.chapter_discovery = chapter_discovery

    def run(self, args: List[str]) -> bool:
        # Per-group help: `./binder check <group> help` prints a
        # dedicated help panel for that group, including concrete
        # error codes. This lives above argparse because `help` is
        # not a valid scope in the GROUPS dict; intercepting it here
        # keeps the argparse surface narrow while still giving each
        # group room for bespoke guidance.
        if len(args) == 2 and args[1] == "help":
            group = args[0]
            if group in self.GROUPS:
                self._print_group_help(group)
                return True

        all_group_names = list(self.GROUPS.keys()) + ["all"]
        parser = argparse.ArgumentParser(
            prog="binder check",
            description="Run quality checks on book content",
            add_help=True,
        )
        parser.add_argument(
            "subcommand",
            nargs="?",
            choices=all_group_names,
            help="Check group to run (refs, labels, headers, footnotes, figures, rendering, references, content, all)",
        )
        parser.add_argument("--scope", default=None, help="Narrow to a specific check within a group")
        parser.add_argument("--path", default=None, help="File or directory path to check")
        parser.add_argument("--vol1", action="store_true", help="Scope to Volume I")
        parser.add_argument("--vol2", action="store_true", help="Scope to Volume II")
        parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
        parser.add_argument("--verbose", "-v", action="store_true", default=True, help="Show context for each issue (default)")
        parser.add_argument("--quiet", "-q", action="store_true", dest="quiet", help="Suppress verbose output")
        parser.add_argument("--citations-in-code", action="store_true", help="refs: check citations in code fences")
        parser.add_argument("--citations-in-raw", action="store_true", help="refs: check citations in raw blocks")
        parser.add_argument("--check-patterns", action="store_true", default=True, help="refs --scope inline: include pattern hazard checks (default: on)")
        parser.add_argument("--no-check-patterns", action="store_false", dest="check_patterns", help="refs --scope inline: skip pattern hazard checks")
        parser.add_argument("--check-scope", action="store_true", default=False, help="refs --scope inline: detect bare variable refs in class bodies that need ClassName.attr")
        parser.add_argument("--no-check-scope", action="store_false", dest="check_scope", help="refs --scope inline: skip scope analysis")
        parser.add_argument("--include-lightbox", action="store_true", default=False, help="math --scope attr-leaks: also surface fig-cap/tbl-cap math that leaks into HTML lightbox tooltips (warning, opt-in; ~70 pre-existing instances on dev)")
        parser.add_argument("--figures", action="store_true", help="labels: filter to figures")
        parser.add_argument("--tables", action="store_true", help="labels: filter to tables")
        parser.add_argument("--sections", action="store_true", help="labels: filter to sections")
        parser.add_argument("--equations", action="store_true", help="labels: filter to equations")
        parser.add_argument("--listings", action="store_true", help="labels: filter to listings")
        parser.add_argument("--all-types", action="store_true", help="labels: all label types")
        parser.add_argument("-f", "--file", dest="refs_file", action="append", metavar="BIB", help="references: .bib file(s) to check")
        parser.add_argument("-o", "--output", dest="refs_output", metavar="FILE", help="references: write report to FILE")
        parser.add_argument("--limit", type=int, dest="refs_limit", metavar="N", help="references: check only first N refs (quick test)")
        parser.add_argument("--skip-verified", dest="refs_skip_verified", action="store_true", help="references: skip refs already verified in cache")
        parser.add_argument("--thorough", dest="refs_thorough", action="store_true", help="references: revalidate all refs (ignore cache)")
        parser.add_argument("--refs-cache", dest="refs_cache", metavar="FILE", help="references: cache file (default: .references_verified.json in repo root)")
        parser.add_argument("--only-from-report", dest="refs_only_from_report", metavar="FILE", help="references: validate only keys that had issues in this report file")
        parser.add_argument("--only-keys", dest="refs_only_keys_file", metavar="FILE", help="references: validate only keys listed in FILE (one key per line)")
        # epub --scope hygiene: auto-repair source files in-place.
        parser.add_argument(
            "--fix", action="store_true",
            help="epub --scope hygiene: auto-repair SVG / BibTeX source invariants in place",
        )
        # epub --scope epubcheck: thresholds for FATAL and ERROR counts.
        # Defaults: MAX_FATAL=0 (any FATAL fails the check — Kindle /
        # ClearView reject), MAX_ERRORS=unlimited (grandfathered while the
        # RSC-005/RSC-012 baselines stabilize). Tighten with --max-errors 0
        # once the baseline is clean.
        parser.add_argument(
            "--max-fatal", type=int, default=0,
            help="epub --scope epubcheck: max FATAL issues allowed before failure (default: 0)",
        )
        parser.add_argument(
            "--max-errors", type=int, default=None,
            help="epub --scope epubcheck: max ERROR issues allowed before failure (default: unlimited)",
        )
        # Ratchet: fail only when counts *increase* over a recorded baseline.
        # Lets CI block regression without cliff-failing during incremental
        # cleanup. Pairs with --update-baseline to re-record after a fix.
        parser.add_argument(
            "--baseline",
            metavar="PATH",
            default=None,
            help="epub --scope epubcheck: JSON baseline of per-volume counts; fail only if current counts exceed baseline",
        )
        parser.add_argument(
            "--update-baseline",
            action="store_true",
            help="epub --scope epubcheck: rewrite --baseline file with current counts (requires --baseline)",
        )

        try:
            ns = parser.parse_args(args)
        except SystemExit:
            # argparse uses SystemExit(0) for --help and non-zero for parse errors.
            return ("-h" in args) or ("--help" in args)

        if not ns.subcommand:
            self._print_check_help()
            return False

        root_path = self._resolve_path(ns.path, ns.vol1, ns.vol2)
        if not root_path.exists():
            self._emit(ns.json, {"status": "error", "message": f"Path not found: {root_path}"}, failed=True)
            return False

        runs: List[ValidationRunResult] = []

        if ns.subcommand == "all":
            for group_name in self.GROUPS:
                runs.extend(self._run_group(group_name, None, root_path, ns))
        else:
            group_name = ns.subcommand
            scope = ns.scope
            if scope and not any(s == scope for s, _ in self.GROUPS.get(group_name, [])):
                valid = [s for s, _ in self.GROUPS[group_name]]
                console.print(f"[red]Unknown scope '{scope}' for group '{group_name}'.[/red]")
                console.print(f"[yellow]Valid scopes: {', '.join(valid)}[/yellow]")
                return False
            runs.extend(self._run_group(group_name, scope, root_path, ns))

        any_failed = any(not run.passed for run in runs)
        summary = {
            "status": "failed" if any_failed else "passed",
            "command": ns.subcommand,
            "path": str(root_path),
            "runs": [run.to_dict() for run in runs],
            "total_issues": sum(len(run.issues) for run in runs),
        }

        if ns.json:
            print(json.dumps(summary, indent=2))
        else:
            verbose = not getattr(ns, "quiet", False)
            self._print_human_summary(summary, verbose=verbose)

        return not any_failed

    # ------------------------------------------------------------------
    # Group dispatch
    # ------------------------------------------------------------------

    def _run_group(
        self,
        group: str,
        scope: Optional[str],
        root: Path,
        ns: argparse.Namespace,
    ) -> List[ValidationRunResult]:
        """Run all checks in *group*, or just the one matching *scope*."""
        results: List[ValidationRunResult] = []
        for scope_name, method_name in self.GROUPS[group]:
            if scope and scope != scope_name:
                continue
            method = getattr(self, method_name)
            # Some runners need extra kwargs
            if method_name == "_run_refs":
                checks_code = ns.citations_in_code or (not ns.citations_in_code and not ns.citations_in_raw)
                checks_raw = ns.citations_in_raw or (not ns.citations_in_code and not ns.citations_in_raw)
                results.append(method(root, citations_in_code=checks_code, citations_in_raw=checks_raw))
            elif method_name == "_run_inline_refs":
                results.append(method(root, check_patterns=ns.check_patterns,
                                      check_scope=getattr(ns, 'check_scope', False)))
            elif method_name == "_run_attr_latex_leaks":
                results.append(method(root, include_lightbox=getattr(ns, 'include_lightbox', False)))
            elif method_name in ("_run_duplicate_labels", "_run_unreferenced_labels"):
                results.append(method(root, self._selected_label_types(ns)))
            elif method_name == "_run_check_references":
                results.append(method(root, ns))
            elif method_name == "_run_epubcheck":
                # Thresholds come from --max-fatal / --max-errors; when not
                # supplied on the CLI we keep max_errors=None (unlimited).
                # The ratchet takes over when --baseline is supplied.
                results.append(method(
                    root,
                    max_fatal=ns.max_fatal,
                    max_errors=ns.max_errors,
                    baseline_path=ns.baseline,
                    update_baseline=ns.update_baseline,
                ))
            elif method_name == "_run_epub_hygiene":
                results.append(method(root, fix=getattr(ns, 'fix', False)))
            else:
                results.append(method(root))
        return results

    def _print_check_help(self) -> None:
        """Print a nicely formatted help for the check command."""
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Group", style="cyan", width=14)
        table.add_column("Scopes", style="yellow", width=38)
        table.add_column("Description", style="white", width=32)

        descriptions = {
            "refs": "References, citations, inline Python, self-ref",
            "labels": "Duplicate labels, orphans, fig-label underscores",
            "headers": "Section header IDs ({#sec-...}), H1-H5 case policy (MIT Press §10.3.1)",
            "bib": "Bibliography hygiene — schema + canonical forms (§5)",
            "footnotes": "Placement, integrity, cross-chapter duplicates, sentence-case first letter",
            "figures": "Captions, float flow, image files",
            "markup": "Low-level markup (patterns, div-fences, dropcaps)",
            "prose": "Prose style (contractions, dup words, ASCII, above/below, Acknowledgments)",
            "punctuation": "Em-dash, slash, vs. period, e.g./i.e. comma, en-dash ranges",
            "numbers": "Units + percent spacing, binary units, percent-in-captions",
            "math": "\\times spacing, attribute-string LaTeX leaks, optional render audit",
            "structure": "Heading levels, parts, Purpose-unnumbered",
            "code": "Python code blocks (echo: false, _str/_math export hygiene)",
            "tables": "Grid tables → pipe, table content hygiene",
            "index": "Index placement (\\index{} outside headings/callouts)",
            "images": "Image file formats, external URLs",
            "json": "JSON file syntax validation",
            "units": "Physics engine unit conversion tests",
            "spelling": "Prose and TikZ spell checking (requires aspell)",
            "epub": "EPUB hygiene (source), epubcheck (built), structure (legacy)",
            "sources": "Source citation analysis and validation",
            "references": "Bibliography vs academic DBs (hallucinator)",
            "content": "Content tree (shared/, frontmatter/ required)",
        }
        for group_name, checks in self.GROUPS.items():
            scopes = ", ".join(s for s, _ in checks)
            desc = descriptions.get(group_name, "")
            table.add_row(group_name, scopes, desc)
        table.add_row("all", "(everything)", "Run all checks")

        console.print(Panel(table, title="binder check <group> [--scope <name>]", border_style="cyan"))
        console.print("[dim]Examples:[/dim]")
        console.print("  [cyan]./binder check refs[/cyan]              [dim]# all reference checks[/dim]")
        console.print("  [cyan]./binder check refs --scope citations[/cyan]  [dim]# only citation check[/dim]")
        console.print("  [cyan]./binder check figures --vol1[/cyan]    [dim]# all figure checks, Vol I[/dim]")
        console.print("  [cyan]./binder check all[/cyan]               [dim]# everything[/dim]")
        console.print("  [cyan]./binder check <group> help[/cyan]      [dim]# per-group error codes + guidance[/dim]")
        console.print()

    # ------------------------------------------------------------------

    def _print_group_help(self, group: str) -> None:
        """Dispatch to per-group help. Falls back to a generic listing."""
        if group == "epub":
            self._print_epub_help()
            return
        # Generic fallback: list scopes and a one-line description.
        scopes = self.GROUPS.get(group, [])
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Scope", style="yellow")
        table.add_column("Method", style="dim")
        for s, m in scopes:
            table.add_row(s, m)
        console.print(Panel(table, title=f"binder check {group}", border_style="cyan"))
        console.print(f"[dim]Run `./binder check {group}` to run every scope, or "
                      f"`./binder check {group} --scope <name>` for one.[/dim]")
        console.print()

    def _print_epub_help(self) -> None:
        """Dedicated help for the `epub` group: the three scopes, the
        error codes each can surface, and how to fix them at source.

        This panel exists because the `epub` group is the most common
        source of unexpected failures for new contributors — the error
        codes (RSC-005, RSC-016, RSC-020, OPF-014, smoke-css-*) are
        cryptic in isolation, but map cleanly to a few source-level
        patterns. Printing the mapping here saves cross-referencing
        the README when a CI failure is in hand.
        """
        # --- Scopes panel -----------------------------------------------
        scope_table = Table(show_header=True, header_style="bold cyan", box=None)
        scope_table.add_column("Scope", style="yellow", width=12)
        scope_table.add_column("When to run", style="white", width=30)
        scope_table.add_column("Cost", style="dim", width=18)
        scope_table.add_column("Needs", style="dim", width=12)
        scope_table.add_row("hygiene", "Every commit (pre-commit)", "<1s, scans source", "—")
        scope_table.add_row("smoke", "After build, for reader compat", "~200ms, scans built EPUB", "—")
        scope_table.add_row("epubcheck", "CI, before release", "~7s per volume", "Java + epubcheck")
        console.print(Panel(scope_table, title="./binder check epub — scopes", border_style="cyan"))

        # --- Error code table ------------------------------------------
        # Scope is derivable from the code prefix: svg-/bib-* = hygiene,
        # smoke-* = smoke, RSC-*/OPF-* = epubcheck. Dropping the column
        # leaves room for readable Trigger / Fix copy.
        code_table = Table(show_header=True, header_style="bold cyan", box=None)
        code_table.add_column("Code", style="red", width=32)
        code_table.add_column("Trigger", style="white", width=30)
        code_table.add_column("Fix at source", style="green", width=32)

        rows = [
            ("svg-c0",
             "C0 control char in SVG aria-label",
             "Strip in Python plot title; or --fix"),
            ("svg-dupe-marker",
             "Duplicate <marker id=…/> in one SVG",
             "Delete duplicate; or --fix"),
            ("bib-url-escape-underscore",
             r"\_ in bib url= or http doi=",
             r"\_ → _ in the .bib; or --fix"),
            ("bib-url-escape-percent",
             r"\% in bib URL field",
             r"\% → % in the .bib; or --fix"),
            ("bib-url-raw-angle",
             "raw < or > in bib URL field",
             "%3C / %3E encode; or --fix"),
            ("smoke-css-custom-property-decl",
             "--var-name: value; in packaged CSS",
             "Inline the literal value"),
            ("smoke-css-custom-property-use",
             "var(--x) in packaged CSS",
             "Inline the literal value"),
            ("smoke-external-resource",
             "src=/<link href= to http(s)://",
             "Package the asset or drop"),
            ("RSC-016 (FATAL)",
             "XML malformed (--, <br>, C0)",
             "epub_postprocess sanitizes; fix at source if it recurs"),
            ("RSC-005",
             "Malformed markup (alt on wrong elem)",
             "Fix the emitter in the Quarto filter"),
            ("RSC-020",
             "Invalid URL syntax in href",
             "Same as bib-url-* above"),
            ("RSC-012",
             "Broken fragment ID",
             "fix_cross_references.py handles this"),
            ("OPF-014",
             "Missing mathml / other OPF property",
             "epub_postprocess declares mathml on nav"),
        ]
        for code, trigger, fix in rows:
            code_table.add_row(code, trigger, fix)
        console.print(Panel(code_table, title="Error codes", border_style="red"))

        # --- Command quick reference -----------------------------------
        console.print("[bold]Common invocations:[/bold]")
        console.print("  [cyan]./binder check epub[/cyan]                        [dim]# hygiene + smoke + epubcheck[/dim]")
        console.print("  [cyan]./binder check epub --scope hygiene[/cyan]        [dim]# source-only, fast[/dim]")
        console.print("  [cyan]./binder check epub --scope hygiene --fix[/cyan]  [dim]# auto-repair source issues[/dim]")
        console.print("  [cyan]./binder check epub --scope epubcheck --max-fatal 0 --max-errors 0[/cyan]")
        console.print("  [dim]                                                    # CI-style strict gating[/dim]")
        console.print("  [cyan]./binder check epub --json[/cyan]                 [dim]# structured output[/dim]")
        console.print("  [cyan]./binder build epub --vol1 --skip-hygiene[/cyan]  [dim]# emergency build bypass[/dim]")
        console.print()
        console.print("[dim]Deep dive: book/cli/README.md → \"EPUB Checks — Two Layers, One CLI Surface\"[/dim]")
        console.print()

    # ------------------------------------------------------------------

    def _resolve_path(self, path_arg: Optional[str], vol1: bool, vol2: bool) -> Path:
        if path_arg:
            path = Path(path_arg)
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            return path
        base = self.config_manager.book_dir / "contents"
        if vol1 and not vol2:
            return base / "vol1"
        if vol2 and not vol1:
            return base / "vol2"
        return base

    def _selected_label_types(self, ns: argparse.Namespace) -> Dict[str, List[re.Pattern[str]]]:
        explicit = ns.figures or ns.tables or ns.sections or ns.equations or ns.listings
        if ns.all_types:
            return LABEL_DEF_PATTERNS
        if explicit:
            selected: Dict[str, List[re.Pattern[str]]] = {}
            if ns.figures:
                selected["Figure"] = LABEL_DEF_PATTERNS["Figure"]
            if ns.tables:
                selected["Table"] = LABEL_DEF_PATTERNS["Table"]
            if ns.sections:
                selected["Section"] = LABEL_DEF_PATTERNS["Section"]
            if ns.equations:
                selected["Equation"] = LABEL_DEF_PATTERNS["Equation"]
            if ns.listings:
                selected["Listing"] = LABEL_DEF_PATTERNS["Listing"]
            return selected
        # default: all label types
        return LABEL_DEF_PATTERNS

    def _qmd_files(self, root: Path) -> List[Path]:
        if root.is_file():
            return [root] if root.suffix == ".qmd" else []
        return sorted(root.rglob("*.qmd"))

    def _read_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="utf-8", errors="ignore")

    def _relative_file(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.config_manager.book_dir))
        except ValueError:
            return str(path)

    def _run_python_syntax(self, root: Path) -> ValidationRunResult:
        """Compile every ```{python} code block to catch syntax errors."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        block_start_re = re.compile(r"^```\{python\}")
        block_end_re = re.compile(r"^```\s*$")

        for file in files:
            content = self._read_text(file)
            lines = content.split("\n")
            rel = str(file.relative_to(root)) if file.is_relative_to(root) else str(file)

            in_block = False
            block_lines: List[str] = []
            block_start_line = 0

            for i, line in enumerate(lines, start=1):
                if block_start_re.match(line):
                    in_block = True
                    block_lines = []
                    block_start_line = i
                    continue
                if in_block and block_end_re.match(line):
                    in_block = False
                    # Skip YAML-style #| directives before compiling
                    source_lines = [
                        ln for ln in block_lines
                        if not ln.strip().startswith("#|")
                    ]
                    source = "\n".join(source_lines)
                    if not source.strip():
                        continue
                    try:
                        compile(source, f"{rel}:{block_start_line}", "exec")
                    except SyntaxError as exc:
                        err_line = block_start_line + (exc.lineno or 1)
                        issues.append(ValidationIssue(
                            file=rel,
                            line=err_line,
                            code="python_syntax",
                            message=f"Python syntax error: {exc.msg}",
                            severity="error",
                            context=(exc.text or "").strip()[:120],
                        ))
                    continue
                if in_block:
                    block_lines.append(line)

        return ValidationRunResult(
            name="python-syntax",
            description="Validate Python code block syntax (compile check)",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def _run_inline_python(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        regex_checks = [
            ("missing_backtick", re.compile(r"(?<!`)(\{python\}\s+\w+`)"), "Missing opening backtick before {python}", "error"),
            ("dollar_as_backtick", re.compile(r"\$\{python\}\s+\w+`"), "Dollar sign used instead of backtick before {python}", "error"),
            ("display_math", re.compile(r"\$\$[^$]*`?\{python\}"), "Inline Python inside $$...$$ display math", "error"),
            # NOTE: $\times$ adjacent to inline Python is the PREFERRED convention.
            # Only flag non-_str variables inside $...$ math (decimal stripping risk).
            ("latex_adjacent_raw", re.compile(r"`\{python\}\s+(?!\w+_str)[^`]+`\s*\$\\(times|approx|ll|gg|mu|le|ge|neq|pm|cdot|div)"), "Non-_str inline Python adjacent to LaTeX operator (decimal stripping risk)", "warning"),
        ]

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code_block = False
            in_grid = False

            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    continue

                for code, pattern, message, severity in regex_checks:
                    for match in pattern.finditer(line):
                        issues.append(ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code=code,
                            message=message,
                            severity=severity,
                            context=match.group(0)[:160],
                        ))

                if LATEX_INLINE_PATTERN.search(line):
                    issues.append(ValidationIssue(
                        file=self._relative_file(file),
                        line=idx,
                        code="python_in_math",
                        message="Inline Python inside $...$ math can render incorrectly",
                        severity="error",
                        context=line.strip()[:160],
                    ))

                if GRID_TABLE_SEP_PATTERN.match(stripped):
                    in_grid = True
                elif in_grid and not stripped.startswith("|") and stripped:
                    in_grid = False

                if in_grid and "`{python}" in line:
                    issues.append(ValidationIssue(
                        file=self._relative_file(file),
                        line=idx,
                        code="grid_table_python",
                        message="Inline Python in grid table; convert to pipe table",
                        severity="error",
                        context=line.strip()[:160],
                    ))

                # Unwrapped {python} — missing backticks entirely
                # Match {python} NOT preceded by ` and NOT at start of #| label line
                if "{python}" in line and not stripped.startswith("#|"):
                    for um in re.finditer(r"(?<!`)\{python\}\s+\w+", line):
                        # Make sure it's not inside a backtick span
                        before = line[:um.start()]
                        if before.count("`") % 2 == 0:  # even backticks = not inside span
                            issues.append(ValidationIssue(
                                file=self._relative_file(file),
                                line=idx,
                                code="unwrapped_python",
                                message="Inline Python missing backtick wrapping — will render as literal text",
                                severity="error",
                                context=um.group(0)[:120],
                            ))

                # Inline Python in headings — fragile for TOC/bookmarks/PDF
                if stripped.startswith("#") and not stripped.startswith("#|") and "`{python}" in line:
                    issues.append(ValidationIssue(
                        file=self._relative_file(file),
                        line=idx,
                        code="python_in_heading",
                        message="Inline Python in heading — fragile for TOC, bookmarks, and PDF",
                        severity="warning",
                        context=stripped[:120],
                    ))

        return ValidationRunResult(
            name="inline-python",
            description="Validate inline Python syntax and placement",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def _run_refs(self, root: Path, citations_in_code: bool, citations_in_raw: bool) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        fenced_code_pattern = re.compile(r"```\{([^}]+)\}(.*?)```", re.DOTALL)
        raw_block_pattern = re.compile(r"```\{=(html|latex|tex)\}(.*?)```", re.DOTALL | re.IGNORECASE)
        problematic_classes = {"tikz", "latex", "tex"}

        for file in files:
            content = self._read_text(file)
            if citations_in_code:
                for match in fenced_code_pattern.finditer(content):
                    attrs = match.group(1)
                    code_content = match.group(2)
                    class_match = re.search(r"\.([A-Za-z][A-Za-z0-9_-]*)", attrs)
                    cls = class_match.group(1).lower() if class_match else "unknown"
                    if cls not in problematic_classes:
                        continue
                    for cite_match in CITATION_BRACKET_PATTERN.finditer(code_content):
                        offset = match.start() + len(f"```{{{attrs}}}") + cite_match.start()
                        line_no = content[:offset].count("\n") + 1
                        line = content.splitlines()[line_no - 1] if line_no - 1 < len(content.splitlines()) else ""
                        issues.append(ValidationIssue(
                            file=self._relative_file(file),
                            line=line_no,
                            code="citation_in_code",
                            message=f"Citation in .{cls} code block will not be processed",
                            severity="error",
                            context=line.strip()[:160],
                        ))

            if citations_in_raw:
                for match in raw_block_pattern.finditer(content):
                    raw_type = match.group(1).lower()
                    block = match.group(2)
                    for cite_match in CITATION_BRACKET_PATTERN.finditer(block):
                        offset = match.start() + cite_match.start()
                        line_no = content[:offset].count("\n") + 1
                        line = content.splitlines()[line_no - 1] if line_no - 1 < len(content.splitlines()) else ""
                        issues.append(ValidationIssue(
                            file=self._relative_file(file),
                            line=line_no,
                            code="citation_in_raw",
                            message=f"Citation in raw {raw_type} block will not be processed",
                            severity="error",
                            context=line.strip()[:160],
                        ))

        return ValidationRunResult(
            name="refs",
            description="Validate citation/reference placement in raw/code blocks",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def _bibliography_for_qmd(self, file: Path) -> Optional[Path]:
        """Resolve the volume backmatter references.bib for a .qmd from its path."""
        try:
            rel = file.relative_to(self.config_manager.book_dir)
        except ValueError:
            return None
        parts = rel.parts
        if "vol1" in parts:
            bib_file = self.config_manager.book_dir / "contents" / "vol1" / "backmatter" / "references.bib"
        elif "vol2" in parts:
            bib_file = self.config_manager.book_dir / "contents" / "vol2" / "backmatter" / "references.bib"
        else:
            return None
        return bib_file if bib_file.exists() else None

    def _run_citations(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        bib_key_pattern = re.compile(r"@\w+\{([^,\s]+)")

        for file in files:
            bib_file = self._bibliography_for_qmd(file)
            if bib_file is None:
                continue

            content = self._read_text(file)
            bib_content = self._read_text(bib_file)
            bib_keys = set(bib_key_pattern.findall(bib_content))
            # Strip YAML frontmatter (--- ... --- at file top) to avoid email false positives
            qmd_content_no_code = re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL)
            # Strip HTML style/script blocks to avoid CSS @media false positives
            qmd_content_no_code = re.sub(r"<style\b[^>]*>.*?</style>", "", qmd_content_no_code, flags=re.DOTALL)
            qmd_content_no_code = re.sub(r"```.*?```", "", qmd_content_no_code, flags=re.DOTALL)
            qmd_content_no_code = re.sub(r"`[^`]+`", "", qmd_content_no_code)
            refs = set(CITATION_REF_PATTERN.findall(qmd_content_no_code))
            refs = {r.rstrip(".,;:") for r in refs if not r.startswith(EXCLUDED_CITATION_PREFIXES)}
            refs = {r for r in refs if not re.match(r"^\d+\.\d+", r)}
            missing = sorted(refs - bib_keys)
            for key in missing:
                line_no = self._line_for_token(content, f"@{key}")
                issues.append(ValidationIssue(
                    file=self._relative_file(file),
                    line=line_no,
                    code="missing_citation",
                    message=f"Citation key @{key} missing in bibliography",
                    severity="error",
                    context=f"@{key}",
                ))

        return ValidationRunResult(
            name="citations",
            description="Validate citation keys against bibliography files",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def _run_duplicate_labels(self, root: Path, label_types: Dict[str, List[re.Pattern[str]]]) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []
        definitions: Dict[str, List[Tuple[Path, int, str]]] = {}

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                for label_type, patterns in label_types.items():
                    for pattern in patterns:
                        for match in pattern.finditer(line):
                            label = match.group(1)
                            definitions.setdefault(label, []).append((file, idx, label_type))

        for label, locations in definitions.items():
            if len(locations) <= 1:
                continue
            for file, line_no, label_type in locations:
                issues.append(ValidationIssue(
                    file=self._relative_file(file),
                    line=line_no,
                    code="duplicate_label",
                    message=f"Duplicate {label_type.lower()} label: {label}",
                    severity="error",
                    context=label,
                ))

        return ValidationRunResult(
            name="duplicate-labels",
            description="Detect duplicate label definitions",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def _run_unreferenced_labels(self, root: Path, label_types: Dict[str, List[re.Pattern[str]]]) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        defined: Dict[str, Tuple[Path, int, str]] = {}
        references: Dict[str, List[Tuple[Path, int]]] = {}

        for file in files:
            lines = self._read_text(file).splitlines()
            for idx, line in enumerate(lines, 1):
                for label_type, patterns in label_types.items():
                    for pattern in patterns:
                        for match in pattern.finditer(line):
                            defined.setdefault(match.group(1), (file, idx, label_type))

                for match in LABEL_REF_PATTERN.finditer(line):
                    label = match.group(1)
                    references.setdefault(label, []).append((file, idx))

        # unreferenced definitions (skip section defaults, consistent with legacy behavior)
        for label, (file, line_no, label_type) in defined.items():
            if label_type == "Section":
                continue
            if label not in references:
                issues.append(ValidationIssue(
                    file=self._relative_file(file),
                    line=line_no,
                    code="unreferenced_label",
                    message=f"{label_type} label {label} is never referenced",
                    severity="warning",
                    context=label,
                ))

        # unresolved references
        defined_labels = set(defined.keys())
        for label, locations in references.items():
            if label in defined_labels:
                continue
            for file, line_no in locations:
                issues.append(ValidationIssue(
                    file=self._relative_file(file),
                    line=line_no,
                    code="unresolved_reference",
                    message=f"Reference @{label} has no matching label definition",
                    severity="error",
                    context=f"@{label}",
                ))

        return ValidationRunResult(
            name="unreferenced-labels",
            description="Detect unreferenced labels and unresolved references",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def _run_inline_refs(self, root: Path, check_patterns: bool,
                         check_scope: bool = False) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        yaml_option_inline = re.compile(r"^#\|\s*(fig-cap|tbl-cap|lst-cap|fig-alt):\s*.*`\{python\}")
        caption_syntax_inline = re.compile(r"^:\s+.*`\{python\}.*\{#(tbl|fig|lst)-")
        inline_fstring = re.compile(r"`\{python\}\s*f\"[^`]+`")
        inline_func_call = re.compile(r"`\{python\}\s*\w+\([^`]+\)`")

        for file in files:
            lines = self._read_text(file).splitlines()
            refs: List[Tuple[int, str]] = []
            compute_vars: Set[str] = set()
            compute_classes: Set[str] = set()
            in_cell = False

            for idx, line in enumerate(lines, 1):
                if CELL_START_PATTERN.match(line.strip()):
                    in_cell = True
                    continue
                if in_cell and CELL_END_PATTERN.match(line.strip()):
                    in_cell = False
                    continue
                if in_cell:
                    cls_match = CLASS_DEF_PATTERN.match(line.strip())
                    if cls_match:
                        compute_classes.add(cls_match.group(1))
                    assign = ASSIGN_PATTERN.match(line.strip())
                    if assign:
                        compute_vars.add(assign.group(1))
                    tuple_assign = TUPLE_ASSIGN_PATTERN.match(line.strip())
                    if tuple_assign:
                        for name in re.split(r'\s*,\s*', tuple_assign.group(1)):
                            compute_vars.add(name.strip())

                for match in INLINE_REF_PATTERN.finditer(line):
                    refs.append((idx, match.group(1)))

            for line_no, ref in refs:
                if "." in ref:
                    cls_name = ref.split(".", 1)[0]
                    resolved = cls_name in compute_classes or cls_name in compute_vars
                else:
                    resolved = ref in compute_vars
                if not resolved:
                    issues.append(ValidationIssue(
                        file=self._relative_file(file),
                        line=line_no,
                        code="undefined_inline_ref",
                        message=f"Inline reference `{ref}` is not defined in python cells",
                        severity="error",
                        context=f"`{{python}} {ref}`",
                    ))

            if check_patterns:
                in_grid = False
                for idx, line in enumerate(lines, 1):
                    stripped = line.strip()
                    if LATEX_INLINE_PATTERN.search(line):
                        issues.append(ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="latex_math_inline_python",
                            message="Inline Python inside LaTeX math can strip decimals",
                            severity="warning",
                            context=stripped[:160],
                        ))
                    if LATEX_ADJACENT_PATTERN.search(line):
                        issues.append(ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="latex_adjacent_inline_python",
                            message="Inline Python adjacent to LaTeX operator is fragile",
                            severity="warning",
                            context=stripped[:160],
                        ))
                    if GRID_TABLE_SEP_PATTERN.match(stripped):
                        in_grid = True
                    elif in_grid and stripped and not stripped.startswith("|"):
                        in_grid = False
                    if in_grid and "`{python}" in line:
                        issues.append(ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="grid_table_inline_python",
                            message="Inline Python in grid tables is unsupported",
                            severity="error",
                            context=stripped[:160],
                        ))
                    if inline_fstring.search(line):
                        issues.append(ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="inline_fstring",
                            message="Inline f-string should be precomputed in Python cell",
                            severity="warning",
                            context=stripped[:160],
                        ))
                    if inline_func_call.search(line):
                        issues.append(ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="inline_function_call",
                            message="Inline function call should be precomputed in Python cell",
                            severity="warning",
                            context=stripped[:160],
                        ))
                    if yaml_option_inline.search(line):
                        issues.append(ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="yaml_option_inline_python",
                            message="Inline Python in YAML fig/tbl/lst metadata will not render",
                            severity="error",
                            context=stripped[:160],
                        ))
                    if caption_syntax_inline.search(line):
                        issues.append(ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="caption_inline_python",
                            message="Inline Python in caption syntax will not render",
                            severity="error",
                            context=stripped[:160],
                        ))

            if check_scope:
                from book.tools.scripts.maintenance.validate_inline_refs import check_scope as _check_scope, BOOK_ROOT
                try:
                    scope_warnings = _check_scope(file, verbose=False)
                    for filepath, lineno, check_type, msg in scope_warnings:
                        issues.append(ValidationIssue(
                            file=self._relative_file(file),
                            line=lineno,
                            code=check_type.lower(),
                            message=msg,
                            severity="warning",
                            context="",
                        ))
                except Exception:
                    pass

        return ValidationRunResult(
            name="inline-refs",
            description="Validate inline Python refs and rendering hazard patterns",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Headers  (ported from manage_section_ids.py --verify)
    # ------------------------------------------------------------------

    def _run_headers(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        header_pat = re.compile(r"^(#{1,6})\s+(.+?)(?:\s*\{[^}]*\})?$")
        div_start_pat = re.compile(r"^:::\s*\{\.")
        div_end_pat = re.compile(r"^:::\s*$")
        code_block_pat = re.compile(r"^```[^`]*$")
        sec_id_pat = re.compile(r"\{#sec-[^}]+\}")

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            in_div = False

            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if code_block_pat.match(stripped):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                if div_start_pat.match(stripped):
                    in_div = True
                    continue
                if div_end_pat.match(stripped):
                    in_div = False
                    continue
                if in_div:
                    continue

                match = header_pat.match(line)
                if not match:
                    continue

                # Extract existing attributes
                existing_attrs = ""
                if "{" in line:
                    attrs_start = line.find("{")
                    attrs_end = line.rfind("}")
                    if attrs_end > attrs_start:
                        existing_attrs = line[attrs_start : attrs_end + 1]

                if ".unnumbered" in existing_attrs:
                    continue

                if not sec_id_pat.search(line):
                    title = match.group(2).strip()
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="missing_section_id",
                            message=f"Header missing section ID: {title}",
                            severity="error",
                            context=line.strip()[:160],
                        )
                    )

        return ValidationRunResult(
            name="headers",
            description="Verify section headers have {#sec-...} IDs",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Footnote Placement  (ported from check_forbidden_footnotes.py)
    # ------------------------------------------------------------------

    def _run_footnote_placement(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        fn_pat = re.compile(r"\[\^fn-[\w-]+\]")
        inline_fn_pat = re.compile(r"\^\[[^\]]+\]")
        table_sep_pat = re.compile(r"^\|[\s\-:+]+\|")

        for file in files:
            lines = self._read_text(file).splitlines()
            div_depth = 0
            div_start_line = 0

            for idx, line in enumerate(lines, 1):
                stripped = line.strip()

                # Track div nesting
                if re.match(r"^:{3,4}\s*\{", stripped) or re.match(r"^:{3,4}\s+\w", stripped):
                    div_depth += 1
                    if div_depth == 1:
                        div_start_line = idx
                elif re.match(r"^:{3,4}\s*$", stripped):
                    if div_depth > 0:
                        div_depth -= 1
                        if div_depth == 0:
                            div_start_line = 0

                # Check inline footnotes (always forbidden)
                for m in inline_fn_pat.finditer(line):
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="inline_footnote",
                            message=f"Inline footnote syntax; use [^fn-name] reference format",
                            severity="error",
                            context=m.group(0)[:80],
                        )
                    )

                footnotes = fn_pat.findall(line)
                if not footnotes:
                    continue

                # Table cell check
                if stripped.startswith("|") and stripped.count("|") >= 2 and not table_sep_pat.match(stripped):
                    for fn in footnotes:
                        issues.append(
                            ValidationIssue(
                                file=self._relative_file(file),
                                line=idx,
                                code="footnote_in_table",
                                message=f"Footnote {fn} in table cell",
                                severity="error",
                                context=stripped[:80],
                            )
                        )

                # YAML caption check
                if re.match(r"^\s*(fig-cap|tbl-cap):", line):
                    cap_type = "figure" if "fig-cap" in line else "table"
                    for fn in footnotes:
                        issues.append(
                            ValidationIssue(
                                file=self._relative_file(file),
                                line=idx,
                                code=f"footnote_in_{cap_type}_caption",
                                message=f"Footnote {fn} in {cap_type} caption",
                                severity="error",
                                context=stripped[:80],
                            )
                        )

                # Markdown caption check
                if re.match(r"^:\s*\*\*[^*]+\*\*:", line):
                    for fn in footnotes:
                        issues.append(
                            ValidationIssue(
                                file=self._relative_file(file),
                                line=idx,
                                code="footnote_in_markdown_caption",
                                message=f"Footnote {fn} in markdown caption",
                                severity="error",
                                context=stripped[:80],
                            )
                        )

                # Callout title check
                if re.match(r"^:{3,4}\s*\{.*title=", stripped):
                    title_match = re.search(r'title="([^"]*)"', line)
                    if title_match and fn_pat.search(title_match.group(1)):
                        for fn in fn_pat.findall(title_match.group(1)):
                            issues.append(
                                ValidationIssue(
                                    file=self._relative_file(file),
                                    line=idx,
                                    code="footnote_in_callout_title",
                                    message=f"Footnote {fn} in callout title (breaks LaTeX)",
                                    severity="error",
                                    context=stripped[:80],
                                )
                            )

                # Div block check
                if div_depth > 0 and div_start_line != idx:
                    for fn in footnotes:
                        issues.append(
                            ValidationIssue(
                                file=self._relative_file(file),
                                line=idx,
                                code="footnote_in_div",
                                message=f"Footnote {fn} inside div block (started line {div_start_line})",
                                severity="error",
                                context=stripped[:80],
                            )
                        )

        return ValidationRunResult(
            name="footnote-placement",
            description="Check footnotes in forbidden locations",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Footnote Refs  (ported from footnote_cleanup.py --validate)
    # ------------------------------------------------------------------

    def _run_footnote_refs(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        ref_pat = re.compile(r"\[\^([^]]+)\]")
        def_pat = re.compile(r"^\[\^([^]]+)\]:\s*(.+)$", re.MULTILINE)

        for file in files:
            content = self._read_text(file)
            lines = content.split("\n")

            # Collect definitions
            fn_defs: Dict[str, str] = {}
            for m in def_pat.finditer(content):
                fn_defs[m.group(1)] = m.group(2)

            # Collect references (excluding definition lines themselves)
            fn_refs: Dict[str, List[int]] = defaultdict(list)
            for line_num, line in enumerate(lines, 1):
                for m in ref_pat.finditer(line):
                    fn_id = m.group(1)
                    dm = def_pat.match(line)
                    if dm and dm.group(1) == fn_id:
                        continue  # definition line, not a reference
                    fn_refs[fn_id].append(line_num)

            # Undefined references
            for fn_id in sorted(set(fn_refs.keys()) - set(fn_defs.keys())):
                first_line = fn_refs[fn_id][0]
                issues.append(
                    ValidationIssue(
                        file=self._relative_file(file),
                        line=first_line,
                        code="undefined_footnote_ref",
                        message=f"Undefined footnote reference: [^{fn_id}]",
                        severity="error",
                        context=f"[^{fn_id}]",
                    )
                )

            # Unused definitions
            for fn_id in sorted(set(fn_defs.keys()) - set(fn_refs.keys())):
                def_line = self._line_for_token(content, f"[^{fn_id}]:")
                issues.append(
                    ValidationIssue(
                        file=self._relative_file(file),
                        line=def_line,
                        code="unused_footnote_def",
                        message=f"Unused footnote definition: [^{fn_id}]",
                        severity="warning",
                        context=f"[^{fn_id}]:",
                    )
                )

            # Duplicate definitions
            def_counts: Dict[str, int] = defaultdict(int)
            for line in lines:
                dm = re.match(r"^\[\^([^]]+)\]:", line)
                if dm:
                    def_counts[dm.group(1)] += 1
            for fn_id, count in def_counts.items():
                if count > 1:
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=self._line_for_token(content, f"[^{fn_id}]:"),
                            code="duplicate_footnote_def",
                            message=f"Duplicate footnote definition ({count}x): [^{fn_id}]",
                            severity="error",
                            context=f"[^{fn_id}]:",
                        )
                    )

            # Missing blank line before footnote definition
            # Pandoc requires footnote definitions to start a new block.
            # Without a preceding blank line, Pandoc treats the definition
            # as continuation text and renders [^fn-name] as literal text.
            fn_def_line_pat = re.compile(r"^\[\^[^\]]+\]:")
            for idx, line in enumerate(lines):
                if fn_def_line_pat.match(line) and idx > 0:
                    prev = lines[idx - 1]
                    if prev.strip():  # previous line is not blank
                        fn_match = re.match(r"^\[\^([^\]]+)\]:", line)
                        fn_id_str = fn_match.group(1) if fn_match else "?"
                        issues.append(
                            ValidationIssue(
                                file=self._relative_file(file),
                                line=idx + 1,
                                code="footnote_missing_blank_line",
                                message=(
                                    f"Footnote definition [^{fn_id_str}] has no blank line before it — "
                                    f"Pandoc will not parse it as a footnote"
                                ),
                                severity="error",
                                context=f"prev: {prev.strip()[:60]}",
                            )
                        )

        return ValidationRunResult(
            name="footnote-refs",
            description="Validate footnote references and definitions",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Figures  (ported from check_figure_completeness.py)
    # ------------------------------------------------------------------

    def _run_figures(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        fig_id_pat = re.compile(r"\{#(fig-[a-zA-Z0-9_-]+)[\s}]")
        md_cap_pat = re.compile(r"!\[(.+?)\]\(")

        for file in files:
            lines = self._read_text(file).splitlines()
            seen_ids: Set[str] = set()

            # Pass 1: attribute-based figures
            for idx, line in enumerate(lines, 1):
                m = fig_id_pat.search(line)
                if not m:
                    continue
                fig_id = m.group(1)
                has_cap = bool(re.search(r'fig-cap="[^"]+', line))
                has_alt = bool(re.search(r'fig-alt="[^"]+', line))

                if "![" in line:
                    md_m = md_cap_pat.search(line)
                    if md_m and md_m.group(1).strip():
                        has_cap = True

                seen_ids.add(fig_id)
                missing = []
                if not has_cap:
                    missing.append("caption")
                if not has_alt:
                    missing.append("alt-text")
                if missing:
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="incomplete_figure",
                            message=f"Figure {fig_id} missing: {', '.join(missing)}",
                            severity="error",
                            context=line.strip()[:120],
                        )
                    )

            # Pass 2: code-cell figures
            in_code = False
            code_start = 0
            cell_opts: Dict[str, str] = {}
            for idx, line in enumerate(lines, 1):
                stripped = line.rstrip()
                if not in_code and re.match(r"^```\{(?:python|r|julia|ojs)", stripped):
                    in_code = True
                    code_start = idx
                    cell_opts = {}
                    continue
                if in_code and stripped == "```":
                    label = cell_opts.get("label", "")
                    if label.startswith("fig-") and label not in seen_ids:
                        cap_val = cell_opts.get("fig-cap", "")
                        alt_val = cell_opts.get("fig-alt", "")
                        missing = []
                        if not cap_val:
                            missing.append("caption")
                        if not alt_val:
                            missing.append("alt-text")
                        if missing:
                            issues.append(
                                ValidationIssue(
                                    file=self._relative_file(file),
                                    line=code_start,
                                    code="incomplete_figure",
                                    message=f"Figure {label} missing: {', '.join(missing)}",
                                    severity="error",
                                    context=f"code-cell figure {label}",
                                )
                            )
                        seen_ids.add(label)
                    in_code = False
                    cell_opts = {}
                    continue
                if in_code:
                    opt_m = re.match(r"^#\|\s*([\w-]+):\s*(.+)$", stripped)
                    if opt_m:
                        val = opt_m.group(2).strip().strip("\"'")
                        cell_opts[opt_m.group(1)] = val

        return ValidationRunResult(
            name="figures",
            description="Check figures have captions and alt-text",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Figure div syntax  (ported from check_figure_div_syntax.py)
    # ------------------------------------------------------------------

    _MARKDOWN_IMAGE_FIG = re.compile(r"!\[.*\]\s*\([^)]+\)\s*\{#fig-")
    _CHUNK_FIG_OPTION = re.compile(r"^#\|\s*(fig-cap|fig-alt)\s*[:=]")

    def _run_figure_div_syntax(self, root: Path) -> ValidationRunResult:
        """Enforce div syntax for figures (no markdown-image figs or chunk options)."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        for file in files:
            lines = self._read_text(file).splitlines()
            for idx, line in enumerate(lines, 1):
                if self._MARKDOWN_IMAGE_FIG.search(line):
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="markdown_image_fig",
                            message="Use div syntax for figures, not ![Caption](path){#fig-...}",
                            severity="error",
                            context=line.strip()[:80],
                        )
                    )
                if self._CHUNK_FIG_OPTION.match(line.strip()):
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="chunk_fig_option",
                            message="Use div syntax for fig-cap/fig-alt, not #| chunk options",
                            severity="error",
                            context=line.strip()[:80],
                        )
                    )

        return ValidationRunResult(
            name="div-syntax",
            description="Enforce figure div syntax (no markdown-image or chunk fig-cap/fig-alt)",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Float Flow  (ported from figure_table_flow_audit.py)
    # ------------------------------------------------------------------

    def _run_float_flow(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        div_def_pat = re.compile(r":::\s*\{[^}]*#((?:fig|tbl)-[\w-]+)")
        img_def_pat = re.compile(r"!\[.*?\]\(.*?\)\s*\{[^}]*#((?:fig|tbl)-[\w-]+)")
        tbl_cap_pat = re.compile(r"^:\s+.*\{[^}]*#((?:fig|tbl)-[\w-]+)")
        ref_pat = re.compile(r"@((?:fig|tbl)-[\w-]+)")

        for file in files:
            lines = self._read_text(file).splitlines()
            defs: Dict[str, int] = {}
            refs: Dict[str, List[int]] = defaultdict(list)
            in_code = False
            in_float = False
            float_label: Optional[str] = None
            code_spans: List[Tuple[int, int]] = []
            code_start = 0
            cell_opts: Dict[str, str] = {}

            for idx, line in enumerate(lines, 1):
                stripped = line.rstrip()

                # Code block tracking
                if not in_code and re.match(r"^```\{", stripped):
                    in_code = True
                    code_start = idx
                    cell_opts = {}
                    continue
                if in_code and stripped == "```":
                    code_spans.append((code_start, idx))
                    label = cell_opts.get("label", "")
                    if label.startswith(("fig-", "tbl-")) and label not in defs:
                        defs[label] = code_start
                    in_code = False
                    cell_opts = {}
                    continue
                if in_code:
                    opt_m = re.match(r"^#\|\s*([\w-]+):\s*(.+)$", stripped)
                    if opt_m:
                        cell_opts[opt_m.group(1)] = opt_m.group(2).strip().strip("\"'")
                    continue

                # Attribute-based definitions
                for pat in [div_def_pat, img_def_pat, tbl_cap_pat]:
                    m = pat.search(line)
                    if m:
                        label = m.group(1)
                        if label not in defs:
                            defs[label] = idx
                        if pat == div_def_pat:
                            in_float = True
                            float_label = label

                # Track float block end
                if in_float:
                    ls = line.strip()
                    if ls.startswith(":::") and not ls.startswith("::: {"):
                        in_float = False
                        float_label = None

                # References
                if "fig-cap=" in line or "fig-alt=" in line:
                    continue
                for m in ref_pat.finditer(line):
                    label = m.group(1)
                    if in_float and label == float_label:
                        continue
                    refs[label].append(idx)

            # Evaluate status
            all_labels = set(defs.keys()) | set(refs.keys())
            for label in sorted(all_labels):
                def_line = defs.get(label)
                ref_lines = refs.get(label, [])
                first_ref = min(ref_lines) if ref_lines else None

                if not def_line:
                    continue  # XREF — informational, skip
                if not first_ref:
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=def_line,
                            code="orphan_float",
                            message=f"{'Figure' if label.startswith('fig-') else 'Table'} {label} defined but never referenced",
                            severity="warning",
                            context=label,
                        )
                    )
                    continue

                # Compute prose gap
                gap = def_line - first_ref
                code_lines = 0
                if gap > 0:
                    for cs, ce in code_spans:
                        os_ = max(first_ref, cs)
                        oe_ = min(def_line, ce)
                        if os_ <= oe_:
                            code_lines += oe_ - os_ + 1
                prose_gap = gap - code_lines

                if prose_gap > 30:
                    # Check closest reference
                    closest = min(ref_lines, key=lambda r: abs(def_line - r))
                    closest_gap = def_line - closest
                    if -10 <= closest_gap <= 30:
                        continue  # OK
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=def_line,
                            code="late_float",
                            message=f"{label} defined at L{def_line}, first referenced at L{first_ref} (too far after mention)",
                            severity="warning",
                            context=label,
                        )
                    )
                elif prose_gap < -10:
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=def_line,
                            code="early_float",
                            message=f"{label} defined at L{def_line}, first referenced at L{first_ref} (appears before mention)",
                            severity="warning",
                            context=label,
                        )
                    )

        return ValidationRunResult(
            name="float-flow",
            description="Audit figure/table placement relative to first reference",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Indexes  (ported from check_index_placement.py)
    # ------------------------------------------------------------------

    def _run_indexes(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        checks = [
            ("index_on_heading", re.compile(r"^#{1,6}\s+.*\\index\{"), "\\index{} on same line as heading"),
            ("index_before_div", re.compile(r"\\index\{[^}]*\}:::"), "\\index{} directly before ::: (div/callout)"),
            ("index_after_div", re.compile(r"^::+\s+\{[^}]*\}\s*\\index\{"), "\\index{} on same line as div/callout"),
            ("index_before_footnote", re.compile(r"^\\index\{[^}]*\}.*\[\^[^\]]+\]:"), "\\index{} before footnote definition"),
        ]

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            for idx, line in enumerate(lines, 1):
                if line.strip().startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue

                for code, pattern, message in checks:
                    # Skip fig-cap lines for index_after_div
                    if code == "index_after_div" and "fig-cap=" in line:
                        continue
                    if pattern.search(line):
                        issues.append(
                            ValidationIssue(
                                file=self._relative_file(file),
                                line=idx,
                                code=code,
                                message=message,
                                severity="error",
                                context=line.strip()[:120],
                            )
                        )

        return ValidationRunResult(
            name="indexes",
            description="Check LaTeX \\index{} placement",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Rendering  (ported from check_render_patterns.py)
    # ------------------------------------------------------------------

    def _run_rendering(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        regex_checks = [
            ("missing_opening_backtick", re.compile(r"(?<!`)(\{python\}\s+\w+`)"), "Missing opening backtick on inline Python", "error"),
            ("dollar_before_python", re.compile(r"\$\{python\}\s+\w+`"), "Dollar sign instead of backtick before {python}", "error"),
            ("quad_asterisks", re.compile(r"\*{4,}"), "Quad asterisks — likely malformed bold/italic", "warning"),
            ("footnote_in_table", re.compile(r"^\|.*\[\^fn-[^\]]+\].*\|"), "Footnote in table cell — may break PDF", "warning"),
            ("double_dollar_python", re.compile(r"\$\$[^$]*`\{python\}"), "Inline Python in display math", "error"),
            # Currency: unescaped $ before number can be parsed as math. Use \$ for currency (see book-prose.md).
            # Match: $1,000 (comma), $4.00 (decimal), $50 million/billion/etc.
            # Exclude: $1.5 \times (math), $0.5$ (inline math), $4.6 / (division).
            ("unescaped_currency", re.compile(
                r"(?<!\\)\$[0-9]{1,3}(?:,[0-9]{3})+(?=\s(?!\s*\\times)|,[0-9]|\)|$)"  # $1,000, exclude $25,000 \times
                r"|(?<!\\)\$[0-9]+\.[0-9]+(?=\s(?!\s*\\times)(?!\s*/)(?!\s*-)(?!\s*\+)(?!\s*\\ll)|,[0-9]|\)|$|/)(?!\\$)"  # $4.00, exclude math
                r"|(?<!\\)\$[0-9]+(?=\s+(?:million|billion|thousand|M|B|K|per|each|/))"  # $50 million
            ), "Unescaped dollar before number — use \\$ for currency", "warning"),
        ]

        grid_sep_pat = re.compile(r"^\+[-:=+]+\+$")
        math_span_pat = re.compile(r"(?<!\\)\$(?!\$)(?!`)(.+?)(?<!\\)\$")

        # Lowercase 'x' used as multiplication in prose (should be $\times$).
        # Matches: `...`x word, NUMx word — but NOT hex (0x61), code, fig-alt, or \index.
        # The pattern requires a lowercase letter after x+space, which naturally
        # excludes hardware counts like "8x A100" (uppercase after x).
        lowercase_x_mult_pat = re.compile(
            r"""`x\s+[a-z]"""    # `...`x word  (after inline python)
            r"""|"""
            r"""\dx\s+[a-z]"""   # Nx word  (digit then x then lowercase)
        )
        # Hex literal pattern to exclude matches like 0x61, 0xff
        hex_literal_pat = re.compile(r"0x[0-9a-fA-F]")
        # Cross-reference ID pattern: @tbl-foo300x, @fig-bar2x, @sec-baz1x — these are labels not multiplication
        xref_id_pat = re.compile(r"@(?:tbl|fig|sec|eq|lst)-[a-z0-9_-]+x\b")
        # fig-alt lines to skip
        fig_alt_pat = re.compile(r'fig-alt\s*=\s*"')

        for file in files:
            lines = self._read_text(file).splitlines()
            in_grid = False
            in_code = False

            for idx, line in enumerate(lines, 1):
                stripped = line.strip()

                # Code block tracking
                if stripped.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue

                # Grid table tracking
                if grid_sep_pat.match(stripped):
                    in_grid = True
                elif in_grid and not stripped.startswith("|") and not grid_sep_pat.match(stripped) and stripped:
                    in_grid = False

                if in_grid and "`{python}" in line:
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="grid_table_python",
                            message="Grid table with inline Python — convert to pipe table",
                            severity="error",
                            context=stripped[:120],
                        )
                    )

                # Python inside $...$ math
                for m in math_span_pat.finditer(line):
                    inner = m.group(1)
                    if "{python}" not in inner:
                        continue
                    inner_clean = re.sub(r"\^\{[^}]*`\{python\}[^`]*`[^}]*\}", "", inner)
                    if "{python}" in inner_clean:
                        issues.append(
                            ValidationIssue(
                                file=self._relative_file(file),
                                line=idx,
                                code="python_in_dollar_math",
                                message="Inline Python inside $...$ math block",
                                severity="error",
                                context=m.group(0)[:120],
                            )
                        )

                # Lowercase 'x' used as multiplication in prose
                # Skip fig-alt lines and index entries
                if not fig_alt_pat.search(line) and not stripped.startswith("\\index"):
                    # Strip cross-reference IDs before checking (e.g. @tbl-mi300x, @fig-foo2x)
                    line_no_xrefs = xref_id_pat.sub("", line)
                    for rm in lowercase_x_mult_pat.finditer(line_no_xrefs):
                        # Exclude hex literals like 0x61, 0xff
                        ctx_start = max(0, rm.start() - 1)
                        if hex_literal_pat.match(line_no_xrefs[ctx_start : rm.end()]):
                            continue
                        issues.append(
                            ValidationIssue(
                                file=self._relative_file(file),
                                line=idx,
                                code="lowercase_x_multiplication",
                                message="Lowercase 'x' used as multiplication — use $\\times$ instead",
                                severity="warning",
                                context=rm.group(0)[:120],
                            )
                        )

                # Standard regex checks
                # For currency check, strip inline math spans first to avoid
                # false positives like $312 \times 10^{12} being flagged as currency.
                line_no_math = math_span_pat.sub("MATHSPAN", line)
                for code, pattern, message, severity in regex_checks:
                    check_line = line_no_math if code == "unescaped_currency" else line
                    for rm in pattern.finditer(check_line):
                        issues.append(
                            ValidationIssue(
                                file=self._relative_file(file),
                                line=idx,
                                code=code,
                                message=message,
                                severity=severity,
                                context=rm.group(0)[:120],
                            )
                        )

        return ValidationRunResult(
            name="rendering",
            description="Check for problematic rendering patterns",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def _run_python_echo(self, root: Path) -> ValidationRunResult:
        """Ensure every ```{python} block has #| echo: false (code must not appear in output)."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        block_start_re = re.compile(r"^```\{python\}")
        block_end_re = re.compile(r"^```\s*$")
        # Quarto chunk option: #| echo: false (with optional whitespace)
        echo_false_re = re.compile(r"#\|\s*echo\s*:\s*false", re.IGNORECASE)

        for file in files:
            lines = self._read_text(file).splitlines()
            i = 0
            while i < len(lines):
                line = lines[i]
                if not block_start_re.match(line):
                    i += 1
                    continue
                start_line = i + 1
                found_echo_false = False
                j = i + 1
                # Scan option lines: #| key: value, or blank, until we hit code or closing ```
                while j < len(lines):
                    next_line = lines[j]
                    if block_end_re.match(next_line):
                        break
                    stripped = next_line.strip()
                    if echo_false_re.search(stripped):
                        found_echo_false = True
                        break
                    # Option line or blank — keep scanning
                    if stripped.startswith("#|") or not stripped:
                        j += 1
                        continue
                    # Non-option line (actual code or comment) — options are done
                    break
                if not found_echo_false:
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=start_line,
                            code="python_missing_echo_false",
                            message="Python block must include #| echo: false — code must not appear in rendered output",
                            severity="error",
                            context="Add #| echo: false as first line after ```{python}",
                        )
                    )
                # Advance past this block to the line after closing ```
                k = j
                while k < len(lines) and not block_end_re.match(lines[k]):
                    k += 1
                i = k + 1

        return ValidationRunResult(
            name="python-echo",
            description="Check Python blocks have echo: false",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Dropcaps  (ported from validate_dropcap_compat.py)
    # ------------------------------------------------------------------

    def _run_dropcaps(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        chapter_hdr = re.compile(r"^#\s+[^#].*\{#sec-")
        numbered_h2 = re.compile(r"^##\s+[^#]")
        unnumbered_h2 = re.compile(r"^##\s+.*\{.*\.unnumbered.*\}")
        starts_xref = re.compile(r"^\s*@(sec|fig|tbl|lst|eq)-")
        starts_link = re.compile(r"^\s*\[")
        starts_inline = re.compile(r"^\s*`")
        yaml_fence = re.compile(r"^---\s*$")
        code_fence = re.compile(r"^```")
        div_fence = re.compile(r"^:::")
        blank = re.compile(r"^\s*$")
        html_comment = re.compile(r"^\s*<!--")
        raw_latex = re.compile(r"^\s*\\")
        list_item = re.compile(r"^\s*[-*+]|\s*\d+\.")

        for file in files:
            lines = self._read_text(file).splitlines()
            in_fm = False
            in_code = False
            in_div = 0
            found_chapter = False
            found_h2 = False

            for idx, line in enumerate(lines, 1):
                if idx == 1 and yaml_fence.match(line):
                    in_fm = True
                    continue
                if in_fm:
                    if yaml_fence.match(line):
                        in_fm = False
                    continue
                if code_fence.match(line):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                if div_fence.match(line):
                    stripped = line.strip()
                    if stripped == ":::":
                        in_div = max(0, in_div - 1)
                    elif stripped.startswith(":::"):
                        in_div += 1
                    continue
                if in_div > 0:
                    continue

                if chapter_hdr.match(line):
                    found_chapter = True
                    found_h2 = False
                    continue
                if not found_chapter:
                    continue
                if numbered_h2.match(line) and not unnumbered_h2.match(line):
                    if not found_h2:
                        found_h2 = True
                    continue
                if not found_h2:
                    continue
                if blank.match(line) or html_comment.match(line) or raw_latex.match(line) or list_item.match(line):
                    continue
                if line.strip().startswith("#"):
                    continue

                # First paragraph line
                if starts_xref.match(line):
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="dropcap_crossref",
                            message="Drop cap paragraph starts with cross-reference",
                            severity="error",
                            context=line.strip()[:120],
                        )
                    )
                elif starts_link.match(line):
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="dropcap_link",
                            message="Drop cap paragraph starts with markdown link",
                            severity="error",
                            context=line.strip()[:120],
                        )
                    )
                elif starts_inline.match(line):
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="dropcap_inline",
                            message="Drop cap paragraph starts with inline code",
                            severity="error",
                            context=line.strip()[:120],
                        )
                    )
                # Only check first paragraph per file
                break

        return ValidationRunResult(
            name="dropcaps",
            description="Validate drop cap compatibility",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Parts  (ported from validate_part_keys.py)
    # ------------------------------------------------------------------

    def _run_parts(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        part_key_pat = re.compile(r"\\part\{key:([^}]+)\}")

        # Load summaries
        summaries_keys: Set[str] = set()
        possible_paths = [
            self.config_manager.book_dir / "contents" / "parts" / "summaries.yml",
            self.config_manager.book_dir / "contents" / "vol1" / "parts" / "summaries.yml",
            self.config_manager.book_dir / "contents" / "vol2" / "parts" / "summaries.yml",
        ]

        try:
            import yaml
        except ImportError:
            return ValidationRunResult(
                name="parts",
                description="Validate part keys (skipped — pyyaml not installed)",
                files_checked=0,
                issues=[],
                elapsed_ms=int((time.time() - start) * 1000),
            )

        for yml_path in possible_paths:
            if yml_path.exists():
                try:
                    data = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
                    for part in data.get("parts", []):
                        if "key" in part:
                            summaries_keys.add(part["key"].lower().replace("_", "").replace("-", ""))
                except Exception:
                    pass

        if not summaries_keys:
            # No summaries found — skip gracefully
            return ValidationRunResult(
                name="parts",
                description="Validate part keys (skipped — no summaries.yml found)",
                files_checked=0,
                issues=[],
                elapsed_ms=int((time.time() - start) * 1000),
            )

        for file in files:
            content = self._read_text(file)
            for m in part_key_pat.finditer(content):
                key = m.group(1)
                norm = key.lower().replace("_", "").replace("-", "")
                if norm not in summaries_keys:
                    line_no = content[: m.start()].count("\n") + 1
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=line_no,
                            code="invalid_part_key",
                            message=f"Part key '{key}' not found in summaries.yml",
                            severity="error",
                            context=m.group(0),
                        )
                    )

        return ValidationRunResult(
            name="parts",
            description="Validate \\part{{key:...}} against summaries.yml",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Heading levels  (detect skipped heading levels)
    # ------------------------------------------------------------------

    def _run_heading_levels(self, root: Path) -> ValidationRunResult:
        """Detect heading level skips outside of div contexts.

        Headings inside Quarto divs (callouts, panels, columns, etc.) are
        in a separate nesting context and are excluded from the hierarchy
        check.  Only headings at the top-level (div depth 0) are compared
        against each other.
        """
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        heading_pat = re.compile(r"^(#{1,6})\s+")
        code_fence = re.compile(r"^```")
        yaml_fence = re.compile(r"^---\s*$")
        # Div open: ::: or :::: (with optional class/id)
        div_open_pat = re.compile(r"^(:{3,})\s*\{")
        # Div close: bare ::: or :::: on its own line
        div_close_pat = re.compile(r"^(:{3,})\s*$")

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            in_yaml = False
            prev_level = 0
            div_depth = 0

            for idx, line in enumerate(lines, 1):
                stripped = line.strip()

                # Track YAML front matter
                if idx == 1 and yaml_fence.match(line):
                    in_yaml = True
                    continue
                if in_yaml:
                    if yaml_fence.match(line):
                        in_yaml = False
                    continue

                # Track code blocks
                if code_fence.match(stripped):
                    in_code = not in_code
                    continue
                if in_code:
                    continue

                # Track div nesting depth
                if div_open_pat.match(stripped):
                    div_depth += 1
                    continue
                if div_close_pat.match(stripped) and div_depth > 0:
                    div_depth -= 1
                    continue

                # Skip headings inside divs — they're in a nested context
                if div_depth > 0:
                    continue

                m = heading_pat.match(line)
                if not m:
                    continue

                level = len(m.group(1))

                # Only flag if we skip a level going deeper
                # (e.g., ## -> #### skips ###)
                if prev_level > 0 and level > prev_level + 1:
                    skipped = ", ".join(
                        f"H{i}" for i in range(prev_level + 1, level)
                    )
                    heading_text = line.lstrip("#").strip()
                    # Truncate at { to remove attributes
                    if "{" in heading_text:
                        heading_text = heading_text[: heading_text.index("{")].strip()
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="heading_level_skip",
                            message=f"Heading jumps from H{prev_level} to H{level} (skips {skipped})",
                            severity="warning",
                            context=heading_text[:80],
                        )
                    )

                prev_level = level

        return ValidationRunResult(
            name="heading-levels",
            description="Detect skipped heading levels (e.g., ## to ####)",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Duplicate consecutive words  (detect "the the", "is is", etc.)
    # ------------------------------------------------------------------

    _DUPE_WORD_PAT = re.compile(
        r"\b(\w{2,})\s+\1\b",
        re.IGNORECASE,
    )
    # Known false positives: intentional repetitions
    _DUPE_WORD_ALLOW = frozenset({
        "had", "that", "do", "bye", "bla", "cha", "go",
        "log",  # "log log n" is valid math
    })

    def _run_duplicate_words(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        code_fence = re.compile(r"^```")
        yaml_fence = re.compile(r"^---\s*$")

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            in_yaml = False

            for idx, line in enumerate(lines, 1):
                # Track YAML front matter
                if idx == 1 and yaml_fence.match(line):
                    in_yaml = True
                    continue
                if in_yaml:
                    if yaml_fence.match(line):
                        in_yaml = False
                    continue

                # Skip code blocks
                if code_fence.match(line.strip()):
                    in_code = not in_code
                    continue
                if in_code:
                    continue

                # Skip HTML comments, raw LaTeX, div fences, HTML tags
                stripped = line.strip()
                if stripped.startswith("<!--") or stripped.startswith("\\") or stripped.startswith(":::"):
                    continue
                if stripped.startswith("<") and not stripped.startswith("<http"):
                    continue
                # Skip lines that are mostly attributes/metadata
                if stripped.startswith("#|") or stripped.startswith("%%|"):
                    continue

                for m in self._DUPE_WORD_PAT.finditer(line):
                    word = m.group(1).lower()
                    if word in self._DUPE_WORD_ALLOW:
                        continue
                    # Skip if inside a LaTeX command or attribute
                    before = line[: m.start()]
                    if before.rstrip().endswith("\\") or "{" in line[m.start() : m.end() + 5]:
                        continue
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="duplicate_word",
                            message=f'Duplicate word: "{m.group(1)} {m.group(1)}"',
                            severity="warning",
                            context=line.strip()[:120],
                        )
                    )

        return ValidationRunResult(
            name="duplicate-words",
            description="Detect duplicate consecutive words (typos)",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Images  (ported from validate_image_references.py)
    # ------------------------------------------------------------------

    def _run_images(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        img_pat = re.compile(r"!\[[^\]]{0,1000}\]\(([^)]+)\)(?:\{[^}]*\})?")
        valid_exts = {".png", ".jpg", ".jpeg", ".gif", ".svg"}

        for file in files:
            content = self._read_text(file)
            for m in img_pat.finditer(content):
                img_path = m.group(1).strip()
                if img_path.startswith(("http://", "https://")):
                    continue
                ext = Path(img_path).suffix.lower()
                if ext not in valid_exts:
                    continue

                resolved = (file.parent / img_path).resolve()
                line_no = content[: m.start()].count("\n") + 1

                if not resolved.exists():
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=line_no,
                            code="missing_image",
                            message=f"Image not found: {img_path}",
                            severity="error",
                            context=img_path,
                        )
                    )
                else:
                    # Case check
                    try:
                        actual = self._realcase(str(resolved))
                        if str(resolved) != actual:
                            issues.append(
                                ValidationIssue(
                                    file=self._relative_file(file),
                                    line=line_no,
                                    code="image_case_mismatch",
                                    message=f"Image case mismatch: ref='{Path(str(resolved)).name}' disk='{Path(actual).name}'",
                                    severity="error",
                                    context=img_path,
                                )
                            )
                    except (FileNotFoundError, OSError):
                        pass

        return ValidationRunResult(
            name="images",
            description="Validate image references exist on disk",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    @staticmethod
    def _realcase(path: str) -> str:
        """Resolve actual case of a path on disk."""
        dirname, basename = os.path.split(path)
        if dirname == path:
            return dirname
        dirname = ValidateCommand._realcase(dirname)
        norm_base = os.path.normcase(basename)
        try:
            for child in os.listdir(dirname):
                if os.path.normcase(child) == norm_base:
                    return os.path.join(dirname, child)
        except OSError:
            pass
        return path

    # ------------------------------------------------------------------
    # Self-referential sections  (ported from check_self_referential_sections.py)
    # ------------------------------------------------------------------

    def _run_self_referential(self, root: Path) -> ValidationRunResult:
        """Detect sections that reference themselves, their parent, or child."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        heading_pat = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+\{#([^}]+)\})?$")
        ref_pat = re.compile(r"@(sec-[a-zA-Z0-9-]+)")

        for file in files:
            lines = self._read_text(file).splitlines()

            # Build heading hierarchy
            headings: List[Dict] = []
            parent_stack: Dict[int, Dict] = {}

            for idx, line in enumerate(lines, 1):
                m = heading_pat.match(line)
                if not m:
                    continue
                level = len(m.group(1))
                title = m.group(2).strip()
                sec_id = m.group(3)
                parent_id = None
                for plevel in range(level - 1, 0, -1):
                    if plevel in parent_stack:
                        parent_id = parent_stack[plevel].get("id")
                        break
                hd = {"level": level, "title": title, "id": sec_id,
                      "line": idx, "parent_id": parent_id}
                headings.append(hd)
                parent_stack[level] = hd
                parent_stack = {k: v for k, v in parent_stack.items() if k <= level}

            # Build section map and children map
            section_map: Dict[str, Dict] = {}
            children_map: Dict[str, List[str]] = defaultdict(list)
            for hd in headings:
                if hd["id"]:
                    section_map[hd["id"]] = hd
                    if hd["parent_id"]:
                        children_map[hd["parent_id"]].append(hd["id"])

            # Check references
            for idx, line in enumerate(lines, 1):
                for m in ref_pat.finditer(line):
                    ref_id = m.group(1)
                    # Find which section this line belongs to
                    current = None
                    for hd in headings:
                        if hd["line"] <= idx:
                            current = hd
                        else:
                            break
                    if not current or not current["id"]:
                        continue

                    cur_id = current["id"]
                    if ref_id == cur_id:
                        issues.append(ValidationIssue(
                            file=self._relative_file(file), line=idx,
                            code="self_reference",
                            message=f"Section '{current['title']}' references itself (@{ref_id})",
                            severity="warning",
                            context=line.strip()[:120],
                        ))
                    elif current["parent_id"] == ref_id:
                        parent = section_map.get(ref_id)
                        ptitle = parent["title"] if parent else ref_id
                        issues.append(ValidationIssue(
                            file=self._relative_file(file), line=idx,
                            code="parent_reference",
                            message=f"Section '{current['title']}' references its parent '{ptitle}' (@{ref_id})",
                            severity="warning",
                            context=line.strip()[:120],
                        ))
                    elif ref_id in children_map.get(cur_id, []):
                        child = section_map.get(ref_id)
                        ctitle = child["title"] if child else ref_id
                        issues.append(ValidationIssue(
                            file=self._relative_file(file), line=idx,
                            code="child_reference",
                            message=f"Section '{current['title']}' references its child '{ctitle}' (@{ref_id})",
                            severity="warning",
                            context=line.strip()[:120],
                        ))

        return ValidationRunResult(
            name="self-referential",
            description="Detect self-referential section references",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Figure label underscores  (ported from check_fig_references.py)
    # ------------------------------------------------------------------

    def _run_fig_label_underscores(self, root: Path) -> ValidationRunResult:
        """Find figure references containing underscores (invalid in Quarto)."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        fig_ref_pat = re.compile(r"(?:\{#|@)fig-([^}\s]+)")

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            for idx, line in enumerate(lines, 1):
                if line.strip().startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                for m in fig_ref_pat.finditer(line):
                    label_suffix = m.group(1)
                    if "_" in label_suffix:
                        issues.append(ValidationIssue(
                            file=self._relative_file(file), line=idx,
                            code="fig_label_underscore",
                            message=f"Figure label contains underscore: fig-{label_suffix} (use hyphens)",
                            severity="error",
                            context=line.strip()[:120],
                        ))

        return ValidationRunResult(
            name="fig-labels",
            description="Detect underscores in figure labels",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # ASCII check  (ported from check_ascii.py)
    # ------------------------------------------------------------------

    def _run_ascii(self, root: Path) -> ValidationRunResult:
        """Find non-ASCII Unicode characters in QMD files."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        non_ascii_pat = re.compile(r"[^\x00-\x7F]")

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                # Skip LaTeX raw blocks and HTML comments
                if stripped.startswith("\\") or stripped.startswith("<!--"):
                    continue
                for m in non_ascii_pat.finditer(line):
                    char = m.group(0)
                    col = m.start()
                    context = line[max(0, col - 10):min(len(line), col + 10)]
                    issues.append(ValidationIssue(
                        file=self._relative_file(file), line=idx,
                        code="non_ascii",
                        message=f"Non-ASCII character '{char}' (U+{ord(char):04X})",
                        severity="warning",
                        context=context.strip(),
                    ))

        return ValidationRunResult(
            name="ascii",
            description="Detect non-ASCII characters in QMD files",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Percent spacing  (no space between number/str and %)
    # ------------------------------------------------------------------

    PERCENT_SPACING_PATTERN = re.compile(r"`[^`]*`\s+%")

    def _run_percent_spacing(self, root: Path) -> ValidationRunResult:
        """Flag space between inline expression and % (e.g. `{python} x` % → use `{python} x`%)."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                for m in self.PERCENT_SPACING_PATTERN.finditer(line):
                    context = line[max(0, m.start() - 5) : min(len(line), m.end() + 10)].strip()
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="percent_spacing",
                            message="Remove space between value and % (use e.g. `{python} x`% not `{python} x` %)",
                            severity="error",
                            context=context,
                        )
                    )

        return ValidationRunResult(
            name="percent-spacing",
            description="No space between inline value and % in QMD prose",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Unit spacing  (style: "100 ms", "4 GB" — never "100ms" or "4GB")
    # ------------------------------------------------------------------

    # Number (optional decimal) immediately followed by unit with no space (invalid per book-prose.md).
    UNIT_SPACING_PATTERN = re.compile(
        r"\d+(?:\.\d+)?"
        r"(?:ms|GB|TB|MB|KB|Gbps|Mbps|Tbps|TFLOPS|GFLOPS|W)\b"
    )

    def _run_unit_spacing(self, root: Path) -> ValidationRunResult:
        """Flag number+unit with no space (e.g. 100ms → 100 ms, 4GB → 4 GB)."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                for m in self.UNIT_SPACING_PATTERN.finditer(line):
                    context = line[max(0, m.start() - 2) : min(len(line), m.end() + 5)].strip()
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="unit_spacing",
                            message="Insert space between number and unit (e.g. 100 ms not 100ms, 4 GB not 4GB)",
                            severity="warning",
                            context=context,
                        )
                    )

        return ValidationRunResult(
            name="unit-spacing",
            description="Require space between number and unit (book-prose.md)",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Binary units  (style: "GB" and "TB", not "GiB" or "TiB" in prose)
    # ------------------------------------------------------------------

    BINARY_UNITS_PATTERN = re.compile(r"\b(GiB|TiB|MiB|KiB)\b")

    def _run_binary_units(self, root: Path) -> ValidationRunResult:
        """Flag GiB/TiB in prose — use GB/TB per book-prose.md."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                for m in self.BINARY_UNITS_PATTERN.finditer(line):
                    context = line[max(0, m.start() - 3) : min(len(line), m.end() + 3)].strip()
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="binary_units",
                            message="Use GB/TB not GiB/TiB in prose (book-prose.md)",
                            severity="warning",
                            context=context,
                        )
                    )

        return ValidationRunResult(
            name="binary-units",
            description="No GiB/TiB in prose — use GB/TB",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Contractions  (forbidden in body prose per book-prose.md)
    # ------------------------------------------------------------------

    CONTRACTIONS_PATTERN = re.compile(
        r"\b(can't|don't|it's|we'll|won't|hasn't|haven't|isn't|aren't|wasn't|weren't|"
        r"doesn't|didn't|wouldn't|couldn't|shouldn't|that's|there's|here's|what's|"
        r"you're|we're|they're|they've|let's|who's)\b",
        re.IGNORECASE,
    )

    def _run_contractions(self, root: Path) -> ValidationRunResult:
        """Flag contractions in prose — use full forms (cannot, do not, etc.)."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                if stripped.startswith("|") or stripped.startswith("<!--"):
                    continue
                for m in self.CONTRACTIONS_PATTERN.finditer(line):
                    context = line[max(0, m.start() - 2) : min(len(line), m.end() + 2)].strip()
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="contractions",
                            message="Contractions forbidden in body prose — use full form (e.g. cannot, do not)",
                            severity="warning",
                            context=context,
                        )
                    )

        return ValidationRunResult(
            name="contractions",
            description="No contractions in body prose (book-prose.md)",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Unblended prose  (paragraph split with leading space after period)
    # ------------------------------------------------------------------

    def _run_unblended_prose(self, root: Path) -> ValidationRunResult:
        """Flag line starting with single space after previous line ended with period."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            for i in range(1, len(lines)):
                if lines[i - 1].strip().startswith("```"):
                    in_code = not in_code
                if in_code:
                    continue
                prev = lines[i - 1].strip()
                curr = lines[i]
                if not prev.endswith("."):
                    continue
                if not (len(curr) > 1 and curr[0] == " " and curr[1].isupper()):
                    continue
                context = (curr[:60] + "…") if len(curr) > 60 else curr
                issues.append(
                    ValidationIssue(
                        file=self._relative_file(file),
                        line=i + 1,
                        code="unblended_prose",
                        message="Paragraph likely split: line starts with space after period — merge into one paragraph",
                        severity="warning",
                        context=context.strip(),
                    )
                )

        return ValidationRunResult(
            name="unblended-prose",
            description="Detect wrongly split paragraphs (leading space after period)",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Times spacing  (space after $\\times$ before word/unit per book-prose.md)
    # ------------------------------------------------------------------

    # $\times$ followed immediately by letter or ( with no separating space.
    TIMES_SPACING_PATTERN = re.compile(r"\$\\times\s*\$(?=[a-zA-Z\(])")

    def _run_times_spacing(self, root: Path) -> ValidationRunResult:
        """Flag $\\times$ immediately followed by word/paren with no space."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                for m in self.TIMES_SPACING_PATTERN.finditer(line):
                    context = line[max(0, m.start() - 2) : min(len(line), m.end() + 10)].strip()
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="times_spacing",
                            message="Add space after $\\times$ before word or unit (e.g. $\\times$ speedup)",
                            severity="warning",
                            context=context,
                        )
                    )

        return ValidationRunResult(
            name="times-spacing",
            description="Space after $\\times$ before word/unit (book-prose.md)",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Times product spacing (space around $\times$ in product expressions)
    # ------------------------------------------------------------------

    # Catch tight $\times$ in product expressions involving inline code or
    # numbers.  Multiplier patterns like `val`$\times$ speedup (space after,
    # followed by a word) are intentional and NOT flagged.
    #
    # Patterns flagged (all need spaces around $\times$):
    #   `val`$\times$`val`   — inline × inline
    #   `val`$\times$3       — inline × number
    #   3$\times$`val`       — number × inline
    #   `val`$\times$$math$  — inline × math span
    TIMES_PRODUCT_PATTERNS = [
        # inline × inline:  `..`$\times$`..`
        re.compile(r"`\$\\times\$`"),
        # inline × number:  `..`$\times$<digit>
        re.compile(r"`\$\\times\$\d"),
        # number × inline:  <digit>$\times$`..`
        re.compile(r"\d\$\\times\$`"),
        # inline × math:    `..`$\times$$..$ (tight dollar after times)
        re.compile(r"`\$\\times\$\$"),
    ]

    def _run_times_product_spacing(self, root: Path) -> ValidationRunResult:
        """Flag tight $\\times$ in product expressions (should be spaced)."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                for pat in self.TIMES_PRODUCT_PATTERNS:
                    for m in pat.finditer(line):
                        context = line[max(0, m.start() - 10) : min(len(line), m.end() + 10)].strip()
                        issues.append(
                            ValidationIssue(
                                file=self._relative_file(file),
                                line=idx,
                                code="times_product_spacing",
                                message="Add space around $\\times$ in product (e.g. ` $\\times$ ` not `$\\times$`)",
                                severity="warning",
                                context=context,
                            )
                        )

        return ValidationRunResult(
            name="times-product-spacing",
            description="Space around $\\times$ in product expressions",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Math: LaTeX leaks inside attribute strings (titles + captions + alt text)
    # ------------------------------------------------------------------
    #
    # WHY THIS EXISTS
    #
    # Quarto runs Pandoc's math parser only on document body prose. Attribute
    # values — callout `title="..."`, `fig-cap`, `tbl-cap`, `lst-cap`,
    # `fig-alt`, `tbl-alt` — are extracted as plain strings and reused by
    # downstream consumers (HTML lightbox `<img title="...">` tooltip, EPUB
    # bookmarks, screen readers, PDF outline). Any LaTeX inside them either
    # renders as literal `\command` / `^{N}` / `$..$` or, in the lightbox
    # case, leaks raw LaTeX into the hover tooltip. See the audit retrospective
    # in `.claude/rules/book-prose.md` ("Anti-pattern: LaTeX inside attribute
    # strings"). Fix by switching to Unicode (×, ³, α, β, ρ, ≤, ≥, →, ∞, …).
    #
    # FAILURE MODES CAUGHT
    #   1. Callout `title="..."` containing $...$, \command, or ^{...}/_{N}
    #      → ERROR (never renders correctly, in any output format)
    #   2. fig-cap/tbl-cap/etc. with bare `^{N}` or `_{N}` outside `$...$`
    #      → ERROR (LaTeX-only syntax; renders literally without delimiters)
    #   3. fig-cap/tbl-cap with inline `$\command$` math fragments
    #      → WARNING (caption renders fine, but lightbox tooltip leaks raw
    #      LaTeX). Reserve $...$ for genuinely complex math; prefer Unicode
    #      for short fragments like ρ, ≤, ×.

    # Backslash-LaTeX commands commonly seen leaking. Kept conservative —
    # adding new commands here is safe (more catches), but any addition that
    # introduces false positives in legitimate prose should be reviewed.
    _ATTR_LATEX_COMMANDS = (
        r"times|frac|approx|alpha|beta|gamma|delta|epsilon|zeta|eta|theta|"
        r"iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|"
        r"omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Phi|Psi|Omega|"
        r"sqrt|sum|int|cdot|leq|geq|neq|partial|nabla|infty|"
        r"log|ln|exp|min|max|pm|mp|forall|exists|in|notin|to|gets|"  # codespell:ignore notin
        r"mapsto|rightarrow|Rightarrow|prod|equiv|sim|propto|"
        r"land|lor|neg|text|mathbf|mathit|mathcal|mathbb"
    )
    _ATTR_LATEX_CMD_RE = re.compile(rf"\\(?:{_ATTR_LATEX_COMMANDS})\b")
    # Candidate `$...$` span. We post-filter the captured inner content with
    # _looks_like_math() to reject currency false positives like
    # `$500 outweighs ... ($100)` where two unescaped `$` digit-prefixes
    # bracket a sentence that is not math at all.
    _ATTR_DOLLAR_MATH_RE = re.compile(r"(?<!\\)\$([^\s$][^$\n]*?)(?<!\\)\$")
    _ATTR_CARET_BRACE_RE = re.compile(r"(?<![A-Za-z0-9_])\^\{[^}\n]{1,40}\}")
    _ATTR_UNDER_BRACE_RE = re.compile(r"(?<![A-Za-z0-9_])_\{[^}\n]{1,40}\}")

    @classmethod
    def _looks_like_math(cls, content: str) -> bool:
        """True iff a `$...$` span's inner content is plausibly math.

        Distinguishes real math from unescaped-currency false positives like
        `$500 outweighs ... ($100)`. Heuristic:

          - Anything with a backslash command, `^{...}` superscript, or
            `_{...}` subscript is math.
          - A span whose first character is a digit AND that contains no
            LaTeX-y signals (backslash / ^ / _) is treated as currency
            (false positive: `$500 outweighs ... ($100)` matches the
            outer regex but is two adjacent dollar-prefixed amounts).
          - Otherwise (single symbols like `$B$`, equations like `$T=0.5$`,
            short mixes like `$y < 0$`) — math.
        """
        if not content:
            return False
        if "\\" in content or "^{" in content or "_{" in content:
            return True
        if content[0].isdigit():
            return False
        return True

    @classmethod
    def _has_attr_math(cls, value: str) -> bool:
        """True iff *value* contains a `$...$` math span (after currency
        false-positive filtering)."""
        return any(cls._looks_like_math(m.group(1))
                   for m in cls._ATTR_DOLLAR_MATH_RE.finditer(value))

    # Callout open fence with title= attribute (single line; multi-line attr
    # values are rare in this codebase and intentionally out of scope here).
    _CALLOUT_TITLE_RE = re.compile(
        r"""^:::+\s*\{[^}]*\btitle\s*=\s*"([^"]*)"[^}]*\}\s*$"""
    )
    # YAML caption-like keys at top of a chunk options block or block YAML.
    # Match e.g. `fig-cap: "..."`, `fig-cap: '...'`, `fig-cap: bare text`.
    _YAML_CAP_KEYS = ("fig-cap", "tbl-cap", "lst-cap", "fig-alt", "tbl-alt")
    _YAML_CAP_RE = re.compile(
        r"""^(?P<indent>\s*)(?:\#\|\s*)?(?P<key>fig-cap|tbl-cap|lst-cap|fig-alt|tbl-alt)\s*:\s*(?P<value>.*?)\s*$"""
    )
    # Inline div attribute: `{#fig-x ... fig-cap="..." fig-alt="..."}` or
    # similar single-line attribute blocks. Match any of the cap/alt keys.
    _INLINE_CAP_RE = re.compile(
        r"""\b(?P<key>fig-cap|tbl-cap|lst-cap|fig-alt|tbl-alt)\s*=\s*"(?P<value>[^"]*)\""""
    )

    @classmethod
    def _strip_quotes(cls, raw: str) -> str:
        s = raw.strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            return s[1:-1]
        return s

    @classmethod
    def _strip_inline_math(cls, value: str) -> str:
        """Remove `$...$` math spans so we can scan the *non-math* remainder
        for bare `^{}`/`_{}` that should have been wrapped in dollars.

        Only strips spans that pass the math heuristic — currency false
        positives (`$500 ... $100`) are left in place because they're
        not math zones we want to ignore for the bare-brace scan.
        """
        def _sub(m: re.Match) -> str:
            return " " if cls._looks_like_math(m.group(1)) else m.group(0)
        return cls._ATTR_DOLLAR_MATH_RE.sub(_sub, value)

    def _scan_title_value(self, value: str) -> List[tuple]:
        """Return list of (code, severity, message_suffix) for issues in a
        callout `title=` value. Math NEVER renders inside title; flag any."""
        out: list[tuple] = []
        if self._has_attr_math(value):
            out.append(("attr_title_math", "error",
                        "callout title= contains $...$ — math is not parsed in attribute values; use Unicode (×, ³, α, β, ρ, …)"))
        elif self._ATTR_LATEX_CMD_RE.search(value):
            out.append(("attr_title_latex", "error",
                        "callout title= contains a LaTeX command — markdown is not parsed in attribute values; use Unicode"))
        if self._ATTR_CARET_BRACE_RE.search(value) or self._ATTR_UNDER_BRACE_RE.search(value):
            out.append(("attr_title_brace", "error",
                        "callout title= contains ^{...} or _{...} — use Unicode superscript/subscript (³, ₂, …)"))
        return out

    # Lightbox tooltip leak applies to fig-cap only — Quarto's lightbox is
    # image-specific. Listing captions (lst-cap), table captions (tbl-cap),
    # and alt text (fig-alt/tbl-alt) do not get a lightbox tooltip and so
    # cannot leak through that channel. Math in those caps still renders
    # correctly; warning about a non-existent leak there is just noise.
    _LIGHTBOX_AFFECTED_KEYS = frozenset({"fig-cap"})

    def _scan_caption_value(self, key: str, value: str, include_lightbox: bool = False) -> List[tuple]:
        """Return list of (code, severity, message_suffix) for issues in a
        fig-cap/tbl-cap/lst-cap/fig-alt/tbl-alt value.

        Math IS parsed inside the body of these captions, so `$...$` is
        valid markup. Two failure modes:

          - Bare `^{}/_{}` outside `$...$` is an error (LaTeX-only syntax;
            won't render without delimiters). Applies to all caption-like
            attribute keys.
          - Inline `$...$` math in fig-cap renders fine in the caption
            itself but Quarto's lightbox plugin extracts the caption string
            for the `<img title>` hover tooltip with markdown stripped,
            leaking raw LaTeX. Hover-only HTML cosmetic artifact; surfaces
            only when *include_lightbox=True* (opt-in) and only for
            fig-cap (other keys don't drive a lightbox).
        """
        out: list[tuple] = []
        non_math = self._strip_inline_math(value)
        if self._ATTR_CARET_BRACE_RE.search(non_math) or self._ATTR_UNDER_BRACE_RE.search(non_math):
            out.append((f"{key.replace('-', '_')}_bare_brace", "error",
                        f"{key} contains bare `^{{...}}` / `_{{...}}` outside `$...$` — wrap in `$...$` or use Unicode (³, ₂, …)"))
        if (include_lightbox and key in self._LIGHTBOX_AFFECTED_KEYS
                and self._has_attr_math(value)):
            out.append((f"{key.replace('-', '_')}_lightbox_leak", "warning",
                        f"{key} contains `$...$` math — the caption renders fine, but Quarto's lightbox copies this string into the `<img title>` tooltip with markdown stripped, leaking raw LaTeX. Prefer Unicode for short fragments."))
        return out

    def _run_attr_latex_leaks(self, root: Path, include_lightbox: bool = False) -> ValidationRunResult:
        """LaTeX leakage inside Quarto attribute strings.

        ``include_lightbox`` (default False) opts into surfacing the
        warning class for fig-cap/tbl-cap math fragments that leak into
        the HTML lightbox tooltip. Off by default so pre-commit only
        blocks on true rendering errors.
        """
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        for file in files:
            text = self._read_text(file)
            lines = text.splitlines()
            in_code = False
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                # Track fenced code blocks so we don't scan code samples
                # that legitimately contain `title="..."` or `fig-cap=...`.
                if stripped.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue

                # 1. Callout title= attribute
                m = self._CALLOUT_TITLE_RE.match(line)
                if m:
                    value = m.group(1)
                    for code, severity, message in self._scan_title_value(value):
                        issues.append(ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code=code,
                            message=message,
                            severity=severity,
                            context=stripped[:160],
                        ))

                # 2. YAML-style caption / alt key (chunk options or block YAML)
                ym = self._YAML_CAP_RE.match(line)
                if ym and not stripped.startswith(":::") and not stripped.startswith("{"):
                    key = ym.group("key")
                    value = self._strip_quotes(ym.group("value"))
                    if value:
                        for code, severity, message in self._scan_caption_value(key, value, include_lightbox):
                            issues.append(ValidationIssue(
                                file=self._relative_file(file),
                                line=idx,
                                code=code,
                                message=message,
                                severity=severity,
                                context=stripped[:160],
                            ))

                # 3. Inline div attribute: {... fig-cap="..." ...}
                for im in self._INLINE_CAP_RE.finditer(line):
                    key = im.group("key")
                    value = im.group("value")
                    for code, severity, message in self._scan_caption_value(key, value, include_lightbox):
                        issues.append(ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code=code,
                            message=message,
                            severity=severity,
                            context=stripped[:160],
                        ))

        return ValidationRunResult(
            name="attr-leaks",
            description="LaTeX inside attribute strings (callout title=, fig-cap, tbl-cap, fig-alt, tbl-alt)",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Code: LaTeX inside *_str Python exports (the original bug)
    # ------------------------------------------------------------------
    #
    # WHY THIS EXISTS
    #
    # The `_str` suffix is the project's convention for "this Python value is
    # plain text destined for an inline `{python}` substitution". Inline
    # `{python}` substitutions happen AFTER Pandoc's math parser runs, so any
    # raw LaTeX (`$...$`, `\\command`, `^{...}`) embedded in a `_str` export
    # survives into the rendered output as literal text. The fix is either:
    #   - Wrap the export in md_math() / md() — and rename it `_math` so the
    #     suffix tells the next reader "this is a Markdown-wrapped LaTeX
    #     object, not a plain string."
    #   - Or compute and emit the value as plain text / Unicode.
    # See `.claude/rules/book-prose.md` ("Anti-pattern: bare LaTeX inside a
    # `_str` export").

    # `_str` assignments inside Python code blocks. Capture the variable name
    # so we can include it in the error message and skip `_math`/other names.
    _STR_ASSIGN_RE = re.compile(
        r"""^\s*(?P<lhs>[A-Za-z_][A-Za-z0-9_.]*_str)\s*=\s*(?P<rhs>[fFrR]{0,2}["'].*)$"""
    )
    # Common LaTeX commands likely to appear inside an f-string. Pre-compiled
    # with the leading backslash already escaped for the Python source.
    _STR_LATEX_CMD_RE = re.compile(rf"\\\\(?:{_ATTR_LATEX_COMMANDS})\b")
    # `$...$` math span inside a Python string. Conservative on two axes:
    #   1. Requires a non-space char immediately after the opening `$` to
    #      avoid matching legitimate uses of `$` for currency in plain prose.
    #   2. Excludes `$` immediately followed by `{` — that is f-string
    #      substitution syntax for currency formatting (`f"${rate:.0f}"`),
    #      which writes a literal `$` followed by the substituted value.
    #      Math spans never have `${`-form openings.
    _STR_DOLLAR_MATH_RE = re.compile(r"(?<!\\)\$(?!\{)[^\s$][^$\n]{0,80}?(?<!\\)\$")
    # `^{...}` inside a Python string — LaTeX-only syntax, never legitimate
    # in a plain `_str` value.
    _STR_CARET_BRACE_RE = re.compile(r"\^\{[^}\n]{1,40}\}")
    # Comment / docstring discriminator: skip lines that are obviously
    # docstrings or comments (these don't affect rendering).
    _PY_COMMENT_RE = re.compile(r"^\s*#")

    def _run_str_latex_leak(self, root: Path) -> ValidationRunResult:
        """`*_str` exports must be plain text — no embedded LaTeX."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        block_start = re.compile(r"^```\{python\}")
        block_end = re.compile(r"^```\s*$")

        for file in files:
            lines = self._read_text(file).splitlines()
            in_python = False
            for idx, line in enumerate(lines, 1):
                if block_start.match(line):
                    in_python = True
                    continue
                if block_end.match(line):
                    in_python = False
                    continue
                if not in_python:
                    continue
                if self._PY_COMMENT_RE.match(line):
                    continue

                m = self._STR_ASSIGN_RE.match(line)
                if not m:
                    continue
                lhs = m.group("lhs")
                rhs = m.group("rhs")

                hits: list[tuple] = []
                if self._STR_DOLLAR_MATH_RE.search(rhs):
                    hits.append(("str_export_dollar_math",
                                 f"`{lhs}` contains `$...$` math — `_str` exports must be plain text. "
                                 f"Rename to `{lhs[:-4]}_math` and wrap in `md_math(...)` (or `md(...)` for mixed prose+math)."))
                if self._STR_LATEX_CMD_RE.search(rhs):
                    hits.append(("str_export_latex_cmd",
                                 f"`{lhs}` contains a LaTeX command (`\\\\times`, `\\\\frac`, `\\\\alpha`, …) — "
                                 f"`_str` exports must be plain text. Rename to `{lhs[:-4]}_math` and wrap in `md_math(...)`."))
                if self._STR_CARET_BRACE_RE.search(rhs):
                    hits.append(("str_export_caret_brace",
                                 f"`{lhs}` contains `^{{...}}` / `_{{...}}` — `_str` exports must be plain text. "
                                 f"Rename to `{lhs[:-4]}_math` and wrap in `md_math(...)`, or emit Unicode (³, ₂, …)."))

                for code, message in hits:
                    issues.append(ValidationIssue(
                        file=self._relative_file(file),
                        line=idx,
                        code=code,
                        message=message,
                        severity="error",
                        context=line.strip()[:160],
                    ))

        return ValidationRunResult(
            name="str-latex-leak",
            description="*_str Python exports must not contain raw LaTeX (use md()/md_math())",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Math: full HTML render audit (delegates to tools/audit/audit_math_rendering.py)
    # ------------------------------------------------------------------
    #
    # WHY THIS EXISTS
    #
    # The static `attr-leaks` and `str-latex-leak` checks catch the source-
    # level patterns we know about. The render audit is the empirical fallback:
    # it actually builds each chapter's HTML and scans the *rendered* output
    # for any LaTeX that escaped MathJax — including failure modes we haven't
    # written a static check for yet. It is also the source of truth for the
    # lightbox-tooltip leak class.
    #
    # SLOW. Wired through `binder check math --scope render-audit` so it can
    # be run interactively or in CI, and registered as a `manual`-stage hook
    # in `.pre-commit-config.yaml` (do not let it block routine commits).

    def _run_math_render_audit(self, root: Path) -> ValidationRunResult:
        """Build each chapter's HTML and scan for unrendered LaTeX leakage.

        Delegates to `tools/audit/audit_math_rendering.py`. This is slow
        (~10 minutes for the full book) and is registered as a manual-stage
        pre-commit hook; it is NOT run on every commit.
        """
        start = time.time()
        script = root / "tools" / "audit" / "audit_math_rendering.py"
        issues: List[ValidationIssue] = []
        if not script.exists():
            issues.append(ValidationIssue(
                file=str(script.relative_to(root)) if script.is_absolute() else str(script),
                line=0,
                code="render_audit_missing",
                message="Render-audit script not found — expected tools/audit/audit_math_rendering.py",
                severity="error",
                context="",
            ))
            return ValidationRunResult(
                name="render-audit",
                description="Full HTML build + leak scan (delegates to tools/audit/)",
                files_checked=0,
                issues=issues,
                elapsed_ms=int((time.time() - start) * 1000),
            )

        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=root,
            capture_output=True,
            text=True,
        )
        # The audit script writes audit-math-report.json; surface the summary
        # line + nonzero exit as a single issue if there was leakage.
        if proc.returncode != 0:
            tail = (proc.stdout or "").strip().splitlines()
            summary = next((line for line in reversed(tail)
                            if "leaky" in line.lower() or "fail" in line.lower()),
                           tail[-1] if tail else "render audit reported leaks")
            issues.append(ValidationIssue(
                file="audit-math-report.json",
                line=0,
                code="render_audit_leaks",
                message=f"Render audit found leaks (exit {proc.returncode}): {summary}. "
                        f"See audit-math-report.md for details.",
                severity="error",
                context="",
            ))

        return ValidationRunResult(
            name="render-audit",
            description="Full HTML build + leak scan (delegates to tools/audit/)",
            files_checked=0,
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Purpose sections must be unnumbered
    # ------------------------------------------------------------------

    _PURPOSE_HEADING = re.compile(r"^## Purpose\b")

    def _run_purpose_unnumbered(self, root: Path) -> ValidationRunResult:
        """Ensure all '## Purpose' headings have {.unnumbered}."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        for file in files:
            lines = self._read_text(file).splitlines()
            for idx, line in enumerate(lines, 1):
                if self._PURPOSE_HEADING.match(line) and ".unnumbered" not in line:
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="purpose_not_unnumbered",
                            message="Purpose section must include {.unnumbered}",
                            severity="error",
                            context=line.strip()[:80],
                        )
                    )

        return ValidationRunResult(
            name="purpose-unnumbered",
            description="Ensure Purpose sections are unnumbered",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Div fence validation  (malformed ::: / :::: fences)
    # ------------------------------------------------------------------

    # ::: or :::: followed by space then non-{ character — malformed fence
    _DIV_FENCE_3 = re.compile(r"^:::[ ]+[^{ ]")
    _DIV_FENCE_4 = re.compile(r"^::::[ ]+[^{ ]")

    def _run_div_fences(self, root: Path) -> ValidationRunResult:
        """Flag ::: or :::: lines with trailing text instead of {.class} or closure."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                if self._DIV_FENCE_4.match(stripped):
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="malformed_div_fence_4",
                            message="Nested div fence (::::) has trailing text; must be bare :::: or :::: {.class}",
                            severity="error",
                            context=stripped[:80],
                        )
                    )
                elif self._DIV_FENCE_3.match(stripped):
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="malformed_div_fence_3",
                            message="Div fence (:::) has trailing text; must be bare ::: or ::: {.class}",
                            severity="error",
                            context=stripped[:80],
                        )
                    )

        return ValidationRunResult(
            name="div-fences",
            description="Malformed div fences (::: / ::::) with trailing text",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ==================================================================
    # MIT PRESS CHECKS (§10 of book-prose-merged.md)
    # ==================================================================

    def _run_mitpress_percent_in_captions(self, root: Path) -> ValidationRunResult:
        """Flag % in fig-cap/tbl-cap prose — should be 'percent' (§10.2)."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []
        pct_in_cap = re.compile(r'(fig-cap|tbl-cap)="([^"]*)"')
        pct_pat = re.compile(r"\d\s*%")

        for file in files:
            for idx, line in enumerate(self._read_text(file).splitlines(), 1):
                for cap_m in pct_in_cap.finditer(line):
                    cap_text = cap_m.group(2)
                    for m in pct_pat.finditer(cap_text):
                        context = cap_text[max(0, m.start() - 10) : m.end() + 10].strip()
                        issues.append(
                            ValidationIssue(
                                file=self._relative_file(file),
                                line=idx,
                                code="mitpress_percent_in_caption",
                                message='Use "percent" not "%" in fig-cap/tbl-cap prose (§10.2)',
                                severity="warning",
                                context=context,
                            )
                        )

        return ValidationRunResult(
            name="mitpress-percent-in-captions",
            description="No % in figure/table captions — spell out 'percent' (MIT Press §10.2)",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def _run_mitpress_spaced_emdash(self, root: Path) -> ValidationRunResult:
        """Flag spaced em dashes (word — word) in prose — should be closed (word—word)."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []
        spaced_em = re.compile(r"\S — \S")

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                if stripped.startswith("#|") or stripped.startswith("# "):
                    continue
                # Table placeholder dashes are fine
                if "| —" in line or "— |" in line:
                    continue
                if "fig-cap=" in line or "fig-alt=" in line or "title=" in line:
                    continue
                for m in spaced_em.finditer(line):
                    context = line[max(0, m.start() - 5) : min(len(line), m.end() + 15)].strip()
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="mitpress_spaced_emdash",
                            message="Close up em dash: word—word not word — word (§1)",
                            severity="warning",
                            context=context,
                        )
                    )

        return ValidationRunResult(
            name="mitpress-spaced-emdash",
            description="No spaced em dashes in prose — use word—word (MIT Press §1)",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def _run_mitpress_spaced_slash(self, root: Path) -> ValidationRunResult:
        """Flag spaced slashes (word / word) in prose — should be closed (word/word)."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []
        spaced_slash = re.compile(r"[a-zA-Z0-9]\s+/\s+[a-zA-Z0-9]")

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                if stripped.startswith("#|") or stripped.startswith("# "):
                    continue
                for m in spaced_slash.finditer(line):
                    context = line[max(0, m.start() - 5) : min(len(line), m.end() + 15)].strip()
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="mitpress_spaced_slash",
                            message="Close up slash: word/word not word / word (§1)",
                            severity="warning",
                            context=context,
                        )
                    )

        return ValidationRunResult(
            name="mitpress-spaced-slash",
            description="No spaced slashes in prose — use word/word (MIT Press §1)",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def _run_mitpress_vs_period(self, root: Path) -> ValidationRunResult:
        """Flag bare 'vs' without period — should be 'vs.' (§10.5)."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []
        # Match ' vs ' in prose (space-bounded) NOT followed by '.'
        bare_vs = re.compile(r"(?<= )vs(?=[ ,;:)\]])")

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                if stripped.startswith("#|") or stripped.startswith("#"):
                    continue
                # Skip lines with IDs, labels, URLs, fig-attributes, index, tables
                if any(skip in line for skip in ["{#", "@sec-", "@fig-", "@tbl-",
                       "fig-cap=", "fig-alt=", "title=", "http", "-vs-",
                       "\\index{", "`{python}", "| "]):
                    continue
                for m in bare_vs.finditer(line):
                    context = line[max(0, m.start() - 10) : min(len(line), m.end() + 10)].strip()
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="mitpress_vs_period",
                            message='Use "vs." with period, not bare "vs" (§10.5)',
                            severity="warning",
                            context=context,
                        )
                    )

        return ValidationRunResult(
            name="mitpress-vs-period",
            description='Use "vs." not bare "vs" (MIT Press §10.5)',
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def _run_mitpress_eg_ie_comma(self, root: Path) -> ValidationRunResult:
        """Flag missing comma after e.g. and i.e. (§10.10)."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []
        # e.g. or i.e. followed by a space and a word (no comma)
        missing_comma = re.compile(r"\b(e\.g|i\.e)\.\s+(?!,)(\w)")

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code or stripped.startswith("#|"):
                    continue
                for m in missing_comma.finditer(line):
                    context = line[max(0, m.start() - 5) : min(len(line), m.end() + 15)].strip()
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="mitpress_eg_ie_comma",
                            message='Missing comma after e.g./i.e. — write "e.g.," or "i.e.," (§10.10)',
                            severity="warning",
                            context=context,
                        )
                    )

        return ValidationRunResult(
            name="mitpress-eg-ie-comma",
            description='Comma after e.g. and i.e. (MIT Press §10.10)',
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def _run_mitpress_acknowledgements(self, root: Path) -> ValidationRunResult:
        """Flag British spelling 'Acknowledgements' — should be 'Acknowledgments' (American)."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        for file in files:
            lines = self._read_text(file).splitlines()
            for idx, line in enumerate(lines, 1):
                if "Acknowledgements" in line:
                    context = line.strip()[:80]
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="mitpress_acknowledgements",
                            message='Use American spelling "Acknowledgments" not "Acknowledgements"',
                            severity="warning",
                            context=context,
                        )
                    )

        return ValidationRunResult(
            name="mitpress-acknowledgements",
            description='American spelling: "Acknowledgments" not "Acknowledgements"',
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def _run_mitpress_capitalized_refs(self, root: Path) -> ValidationRunResult:
        """Flag capitalized prose references: 'Chapter 12' → 'chapter 12' (§10.4)."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []
        # Match "Chapter N", "Section N", "Figure N", "Table N" in prose
        # but NOT at sentence start, in headings, or in protected contexts
        cap_ref = re.compile(r"(?<=[a-z,;:] )(Chapter|Section|Figure|Table) \d")

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code or stripped.startswith("#"):
                    continue
                if "fig-cap=" in line or "fig-alt=" in line or "title=" in line:
                    continue
                for m in cap_ref.finditer(line):
                    context = line[max(0, m.start() - 10) : min(len(line), m.end() + 10)].strip()
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="mitpress_capitalized_refs",
                            message=f'Lowercase "{m.group(1).lower()}" in prose refs (§10.4)',
                            severity="warning",
                            context=context,
                        )
                    )

        return ValidationRunResult(
            name="mitpress-capitalized-refs",
            description='Lowercase "chapter/section/figure/table" in prose references (MIT Press §10.4)',
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def _run_mitpress_above_below(self, root: Path) -> ValidationRunResult:
        """Flag 'above'/'below' spatial references — use @sec- cross-refs or 'earlier'/'later'."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []
        spatial = re.compile(
            r"\b(as shown above|as shown below|see above|see below|discussed above|"
            r"discussed below|mentioned above|mentioned below|described above|"
            r"described below|noted above|noted below|"
            r"outlined above|outlined below|listed above|listed below)\b",
            re.IGNORECASE,
        )

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code or stripped.startswith("#|") or stripped.startswith("#"):
                    continue
                for m in spatial.finditer(line):
                    context = line[max(0, m.start() - 10) : min(len(line), m.end() + 10)].strip()
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="mitpress_above_below",
                            message='Use @sec-/@fig- cross-refs or "earlier"/"later", not "above"/"below"',
                            severity="warning",
                            context=context,
                        )
                    )

        return ValidationRunResult(
            name="mitpress-above-below",
            description='No "above"/"below" spatial refs — use cross-refs or "earlier"/"later"',
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def _run_mitpress_hyphen_range(self, root: Path) -> ValidationRunResult:
        """Flag hyphen in number ranges — should use en dash (100–200 not 100-200)."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []
        # Two+ digit numbers separated by a single hyphen (not en dash)
        hyphen_range = re.compile(r"(\d{2,})-(\d{2,})")

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code or stripped.startswith("#|"):
                    continue
                # Skip math, URLs, labels, code refs
                if "$" in line or "http" in line or "{#" in line or "@" in line:
                    continue
                if stripped.startswith("|"):
                    continue
                for m in hyphen_range.finditer(line):
                    a, b = int(m.group(1)), int(m.group(2))
                    if b > a and a >= 10:  # likely a range
                        context = line[max(0, m.start() - 10) : min(len(line), m.end() + 10)].strip()
                        issues.append(
                            ValidationIssue(
                                file=self._relative_file(file),
                                line=idx,
                                code="mitpress_hyphen_range",
                                message=f'Use en dash for ranges: {m.group(1)}–{m.group(2)} not {m.group()} (§2)',
                                severity="warning",
                                context=context,
                            )
                        )

        return ValidationRunResult(
            name="mitpress-hyphen-range",
            description="En dash for number ranges (100–200 not 100-200, MIT Press §2)",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Cross-chapter footnote duplicates  (ported from audit_footnotes_cross_chapter.py)
    # ------------------------------------------------------------------

    def _run_footnote_cross_chapter(self, root: Path) -> ValidationRunResult:
        """Find duplicate footnote IDs across chapters."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        fn_def_pat = re.compile(r"\[\^(fn-[^\]]+)\]:\s*(.+?)(?=\n\n|\n\[\^|\Z)", re.DOTALL)

        # Collect all footnotes by ID
        footnotes_by_id: Dict[str, List[Tuple[Path, str]]] = defaultdict(list)

        for file in files:
            content = self._read_text(file)
            for m in fn_def_pat.finditer(content):
                fn_id = m.group(1)
                fn_content = " ".join(m.group(2).split())[:200]
                footnotes_by_id[fn_id].append((file, fn_content))

        # Report duplicates
        for fn_id, occurrences in footnotes_by_id.items():
            if len(occurrences) <= 1:
                continue
            for file, content in occurrences:
                line_no = self._line_for_token(self._read_text(file), f"[^{fn_id}]:")
                issues.append(ValidationIssue(
                    file=self._relative_file(file), line=line_no,
                    code="cross_chapter_footnote",
                    message=f"Footnote [^{fn_id}] also defined in {len(occurrences) - 1} other file(s)",
                    severity="warning",
                    context=content[:80],
                ))

        return ValidationRunResult(
            name="cross-chapter-footnotes",
            description="Detect duplicate footnote IDs across chapters",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Footnote Capitalization  (delegates to check_footnote_caps.py)
    # ------------------------------------------------------------------
    # MIT Press body prose lowercases concept terms ("memory wall",
    # "scaling laws") per §10.3, but the first letter of a footnote
    # definition must still be a sentence-case capital. A global
    # lowercasing sweep can accidentally strip that capital; this
    # check catches regressions. Intentional lowercase prefixes
    # (brand names like cuDNN/gRPC/vLLM, math variables like k-Center,
    # SI units like pJ/MAC) are declared in
    # book/tools/scripts/mit_press/footnote_caps_allowlist.txt.
    # The standalone script (with a --fix flag) remains the single
    # source of truth; this method imports and reuses its core
    # logic so the check and the fixer cannot drift apart.

    def _run_footnote_capitalization(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        import importlib.util
        import sys as _sys
        script_path = (
            Path(__file__).resolve().parents[2]
            / "tools" / "scripts" / "mit_press" / "check_footnote_caps.py"
        )
        mod_name = "mlsys_check_footnote_caps"
        if mod_name in _sys.modules:
            mod = _sys.modules[mod_name]
        else:
            spec = importlib.util.spec_from_file_location(mod_name, script_path)
            mod = importlib.util.module_from_spec(spec)
            # Register before exec_module so @dataclass can resolve cls.__module__.
            _sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)

        allowlist = mod.load_allowlist(mod.DEFAULT_ALLOWLIST)
        for f in files:
            for v in mod.scan_file(f, allowlist):
                snippet = v.raw_line if len(v.raw_line) <= 160 else v.raw_line[:157] + "..."
                issues.append(
                    ValidationIssue(
                        file=self._relative_file(f),
                        line=v.line_no,
                        code="footnote_lowercase_first_letter",
                        message=(
                            f"Footnote opens with lowercase {v.first_char!r}; "
                            f"capitalize, or add the id to footnote_caps_allowlist.txt "
                            f"if the lowercase is canonical (brand/math/SI)"
                        ),
                        severity="error",
                        context=snippet,
                    )
                )

        return ValidationRunResult(
            name="footnote_capitalization",
            description="Footnote definitions must begin with a capital letter",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Table content validation  (delegated to validate_tables.py)
    # ------------------------------------------------------------------

    def _run_table_content(self, root: Path) -> ValidationRunResult:
        """Validate grid table content (bare pipes, fracs, HTML entities, etc.)."""
        script = (
            Path(__file__).resolve().parent.parent.parent
            / "tools" / "scripts" / "content" / "validate_tables.py"
        )
        args = ["-d", str(root)]
        return self._delegate_script(script, args, "table-content")

    # ------------------------------------------------------------------
    # Spelling checks  (delegated to check_prose_spelling.py / check_tikz_spelling.py)
    # ------------------------------------------------------------------

    def _run_spelling_prose(self, root: Path) -> ValidationRunResult:
        """Spell check prose text (requires aspell)."""
        script = (
            Path(__file__).resolve().parent.parent.parent
            / "tools" / "scripts" / "content" / "check_prose_spelling.py"
        )
        return self._delegate_script(script, [str(root)], "spelling-prose")

    def _run_spelling_tikz(self, root: Path) -> ValidationRunResult:
        """Spell check TikZ diagram text (requires aspell)."""
        script = (
            Path(__file__).resolve().parent.parent.parent
            / "tools" / "scripts" / "content" / "check_tikz_spelling.py"
        )
        # check_tikz_spelling.py auto-scans from repo root, so pass no args
        return self._delegate_script(script, [], "spelling-tikz")

    # ------------------------------------------------------------------
    # EPUB validation
    #
    # Three scopes share this section, each targeting a different layer of
    # the EPUB build pipeline. See book/cli/commands/_epub_checks.py for
    # the per-check logic and the rationale for the layering.
    #
    #   hygiene   — source-level invariants (pre-build, fast, pre-commit)
    #   epubcheck — W3C validator on built artifacts (post-build, CI)
    #   structure — legacy custom checks on built EPUB (no Java required)
    # ------------------------------------------------------------------

    def _run_epub_hygiene(self, root: Path, *, fix: bool = False) -> ValidationRunResult:
        """Run the pre-commit-grade EPUB source hygiene checks.

        Walks `book/quarto/contents/**/*.svg` and `book/quarto/**/*.bib`
        looking for the four source-level patterns that produced FATAL
        or ERROR-level epubcheck failures in April 2026:

          1. C0 control chars inside SVG aria-label attributes
          2. Duplicate `<marker id="X"/>` defs inside one SVG
          3. BibTeX `\\_` or `\\%` escapes inside url= / http doi= fields
          4. Raw `<` or `>` inside those same URL fields

        Runs in <1s and is safe to invoke from pre-commit on every push.

        When `fix=True` (from `--fix`), the detected issues are rewritten
        in place before reporting. The returned issue list then reflects
        what the check found BEFORE the fix — so a first `--fix` run
        reports "N issues found, N auto-fixed" and a second run reports 0.
        """
        from cli.commands._epub_checks import (
            find_hygiene_issues,
            fix_hygiene_issues,
        )

        t0 = time.time()
        # Walk from repo root so the check sees both volumes.
        repo_root = Path(__file__).resolve().parents[3]

        fixes: Dict[str, int] = {}
        if fix:
            fixes = fix_hygiene_issues(repo_root)

        epub_issues, files_checked = find_hygiene_issues(repo_root)

        issues = [
            ValidationIssue(
                file=e.file,
                line=e.line,
                code=e.code,
                message=e.message,
                severity="error",  # hygiene issues are always blocking
            )
            for e in epub_issues
        ]

        description = "EPUB source hygiene (SVG + BibTeX invariants)"
        if fix:
            touched = sum(fixes.values())
            description += (
                f" — --fix applied: "
                f"{fixes.get('svg_c0_chars_removed', 0)} C0 chars stripped, "
                f"{fixes.get('svg_duplicate_markers', 0)} duplicate markers removed, "
                f"{fixes.get('bib_url_rewrites', 0)} URL-field rewrites "
                f"(touched {touched} total)"
            )

        return ValidationRunResult(
            name="epub-hygiene",
            description=description,
            files_checked=files_checked,
            issues=issues,
            elapsed_ms=int((time.time() - t0) * 1000),
        )

    def _check_baseline(
        self,
        *,
        baseline_path: Optional[str],
        update_baseline: bool,
        per_volume_counts: Dict[str, Dict[str, int]],
    ) -> List[ValidationIssue]:
        """Ratchet: compare per-volume counts against a recorded baseline.

        If *baseline_path* is None, do nothing (the flat --max-fatal /
        --max-errors thresholds take over in the caller).

        If *update_baseline* is True, rewrite the baseline file with the
        current counts and return an empty issue list. This is the way
        to lower the ceiling after a cleanup lands.

        Otherwise: load the baseline file, compare to current counts,
        and return a synthetic ValidationIssue per volume + severity
        that regressed.
        """
        if baseline_path is None:
            if update_baseline:
                # No path to update into — emit a clear error.
                return [ValidationIssue(
                    file="(baseline)", line=0,
                    code="EPUBCHECK-BASELINE-NO-PATH",
                    message="--update-baseline supplied without --baseline PATH",
                    severity="error",
                )]
            return []

        import datetime as _dt
        import json as _json

        baseline_file = Path(baseline_path)

        # --- Update mode ------------------------------------------------
        if update_baseline:
            payload = {
                "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "generated_by": "binder check epub --scope epubcheck --update-baseline",
                "description": (
                    "Max allowed per-volume epubcheck counts. "
                    "`./binder check epub --scope epubcheck --baseline <this>` "
                    "fails only if any current count exceeds the recorded "
                    "value. Run `--update-baseline` again after a cleanup "
                    "to lower the ceiling."
                ),
                "volumes": per_volume_counts,
            }
            baseline_file.parent.mkdir(parents=True, exist_ok=True)
            baseline_file.write_text(_json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            console.print(
                f"[green]✓ Baseline updated:[/green] {baseline_file} "
                f"[dim]({len(per_volume_counts)} volumes recorded)[/dim]"
            )
            return []

        # --- Compare mode -----------------------------------------------
        if not baseline_file.exists():
            return [ValidationIssue(
                file=str(baseline_file), line=0,
                code="EPUBCHECK-BASELINE-MISSING",
                message=(
                    f"Baseline file not found: {baseline_file}. Initialize "
                    "with `./binder check epub --scope epubcheck "
                    f"--baseline {baseline_file} --update-baseline`."
                ),
                severity="error",
            )]

        try:
            payload = _json.loads(baseline_file.read_text(encoding="utf-8"))
        except _json.JSONDecodeError as e:
            return [ValidationIssue(
                file=str(baseline_file), line=0,
                code="EPUBCHECK-BASELINE-INVALID",
                message=f"Baseline file is not valid JSON: {e}",
                severity="error",
            )]

        recorded = payload.get("volumes", {}) or {}
        issues: List[ValidationIssue] = []
        any_under = False

        for vol, counts in per_volume_counts.items():
            base = recorded.get(vol, {})
            for severity in ("FATAL", "ERROR", "WARNING"):
                current = counts.get(severity, 0)
                allowed = int(base.get(severity, 0))
                if current > allowed:
                    issues.append(ValidationIssue(
                        file="(epubcheck baseline)",
                        line=0,
                        code=f"EPUBCHECK-BASELINE-{severity}",
                        message=(
                            f"{vol}: {severity} count regressed — current "
                            f"{current} > baseline {allowed} (delta "
                            f"+{current - allowed}). Fix the underlying "
                            f"issues or run `--update-baseline` after "
                            f"verifying this is an accepted increase."
                        ),
                        severity="error",
                    ))
                elif current < allowed:
                    any_under = True

        if any_under and not issues:
            console.print(
                "[dim]ⓘ epubcheck counts are under baseline in one or more "
                "volumes; run `--update-baseline` to lower the ceiling.[/dim]"
            )

        return issues

    def _run_epub_smoke(self, root: Path) -> ValidationRunResult:
        """Reader-compatibility smoke checks against the built EPUB(s).

        Epubcheck validates EPUB 3 spec conformance. This scope catches
        patterns that pass epubcheck but break specific readers:

          * CSS custom properties (`--var`, `var(--x)`) that older
            ClearView / Tolino firmware cannot resolve.
          * External resource references (`src="https://..."`,
            `<link href="https://...">`) that EPUB readers do not fetch.

        Runs against every EPUB discovered under `_build/epub-vol*/`.
        Does not require Java — safe to run on any dev machine.
        """
        from cli.commands._epub_checks import (
            _discover_built_epubs,
            run_smoke_checks_on,
        )

        t0 = time.time()
        repo_root = Path(__file__).resolve().parents[3]
        epubs = _discover_built_epubs(repo_root)

        if not epubs:
            return ValidationRunResult(
                name="epub-smoke",
                description="EPUB reader-compatibility smoke (no built EPUBs found)",
                files_checked=0,
                issues=[],
                elapsed_ms=int((time.time() - t0) * 1000),
            )

        all_issues: List[ValidationIssue] = []
        for epub in epubs:
            for e in run_smoke_checks_on(epub, repo_root=repo_root):
                all_issues.append(ValidationIssue(
                    file=e.file,
                    line=e.line,
                    code=e.code,
                    message=e.message,
                    # Smoke issues are advisory (warnings), not blockers,
                    # because they flag reader-subset behavior rather than
                    # spec violations. Promote to error by fixing at source.
                    severity=e.severity,
                ))

        return ValidationRunResult(
            name="epub-smoke",
            description=(
                f"EPUB reader-compatibility smoke "
                f"({len(epubs)} EPUB(s) scanned)"
            ),
            files_checked=len(epubs),
            issues=all_issues,
            elapsed_ms=int((time.time() - t0) * 1000),
        )

    def _run_epubcheck(
        self,
        root: Path,
        *,
        max_fatal: int = 0,
        max_errors: Optional[int] = None,
        baseline_path: Optional[str] = None,
        update_baseline: bool = False,
    ) -> ValidationRunResult:
        """Run the W3C `epubcheck` validator against the built EPUBs.

        Discovers every `book/quarto/_build/epub-vol*/*.epub` (most recent
        per volume), invokes `epubcheck --json -` on each, parses the
        JSON message array, and converts each message to a
        `ValidationIssue`. Also emits GitHub Actions `::error` annotations
        when running under CI so the findings appear inline on PR diffs.

        Threshold semantics
        -------------------
        Any FATAL above *max_fatal* marks the run as failed. Any ERROR
        above *max_errors* (when supplied) marks the run as failed.
        ERRORs below the threshold are still reported in the issue list
        so reviewers see them; they just do not fail the build. The
        default `max_errors=None` treats ERRORs as report-only, which
        matches the current grandfathered CI policy.
        """
        from cli.commands._epub_checks import (
            _discover_built_epubs,
            emit_github_annotations,
            run_epubcheck_on,
        )

        t0 = time.time()
        repo_root = Path(__file__).resolve().parents[3]
        epubs = _discover_built_epubs(repo_root)

        if not epubs:
            # Not a failure by itself; mirror the existing _run_epub
            # behaviour when no EPUB has been built yet.
            return ValidationRunResult(
                name="epubcheck",
                description="epubcheck (no built EPUBs found under _build/epub-vol*/)",
                files_checked=0,
                issues=[],
                elapsed_ms=int((time.time() - t0) * 1000),
            )

        all_issues: List[ValidationIssue] = []
        per_volume_counts: Dict[str, Dict[str, int]] = {}
        total_fatal = 0
        total_errors = 0

        for epub in epubs:
            # Derive a stable volume key from the parent directory name.
            # `_build/epub-vol1/foo.epub` → "vol1"; falls back to the
            # EPUB's stem if the path layout is unexpected.
            vol_key = epub.parent.name.replace("epub-", "") or epub.stem

            epub_issues, counts = run_epubcheck_on(epub, repo_root=repo_root)
            emit_github_annotations(epub_issues)

            per_volume_counts[vol_key] = {
                "FATAL": counts.get("FATAL", 0),
                "ERROR": counts.get("ERROR", 0),
                "WARNING": counts.get("WARNING", 0),
            }
            total_fatal += counts.get("FATAL", 0)
            total_errors += counts.get("ERROR", 0)

            for e in epub_issues:
                all_issues.append(ValidationIssue(
                    file=e.file,
                    line=e.line,
                    code=e.code,
                    message=e.message,
                    # Map FATAL to "error" for ValidationIssue (which only
                    # has error/warning/info). The code field and message
                    # preserve the FATAL distinction for the summary line
                    # below so the user sees FATAL vs ERROR counts.
                    severity="error" if e.severity in ("fatal", "error") else e.severity,
                ))

        # --- Ratchet / baseline handling --------------------------------
        # The baseline is per-volume, per-severity. Semantics:
        #   current == baseline  → pass silently
        #   current <  baseline  → pass; print a hint that the baseline
        #                          can be tightened (don't auto-update
        #                          without --update-baseline to avoid
        #                          silently lowering the bar).
        #   current >  baseline  → fail; emit a synthetic issue with the
        #                          per-volume delta so reviewers see what
        #                          regressed.
        baseline_issues = self._check_baseline(
            baseline_path=baseline_path,
            update_baseline=update_baseline,
            per_volume_counts=per_volume_counts,
        )
        all_issues.extend(baseline_issues)

        # Threshold enforcement (applied only when the ratchet is not in
        # use — supplying --baseline tells the tool to use the ratchet
        # instead of flat thresholds).
        if baseline_path is None:
            if total_fatal > max_fatal:
                all_issues.append(ValidationIssue(
                    file="(epubcheck)",
                    line=0,
                    code="EPUBCHECK-FATAL-THRESHOLD",
                    message=(
                        f"FATAL count {total_fatal} exceeds threshold "
                        f"{max_fatal} — Kindle / ClearView will reject this EPUB."
                    ),
                    severity="error",
                ))
            if max_errors is not None and total_errors > max_errors:
                all_issues.append(ValidationIssue(
                    file="(epubcheck)",
                    line=0,
                    code="EPUBCHECK-ERROR-THRESHOLD",
                    message=(
                        f"ERROR count {total_errors} exceeds threshold "
                        f"{max_errors}. Tighten or fix the underlying RSC-* issues."
                    ),
                    severity="error",
                ))

        # Build the description so the user sees the severity summary even
        # when all issues are warnings/info.
        desc = (
            f"epubcheck ({len(epubs)} EPUB(s); "
            f"{total_fatal} FATAL, {total_errors} ERROR)"
        )

        return ValidationRunResult(
            name="epubcheck",
            description=desc,
            files_checked=len(epubs),
            issues=all_issues,
            elapsed_ms=int((time.time() - t0) * 1000),
        )

    # ------------------------------------------------------------------
    # EPUB structure (legacy custom check — no Java required)
    # ------------------------------------------------------------------

    def _run_epub(self, root: Path) -> ValidationRunResult:
        """Validate EPUB file structure and content."""
        script = (
            Path(__file__).resolve().parent.parent.parent
            / "tools" / "scripts" / "utilities" / "validate_epub.py"
        )
        # Find EPUB files in build output directories
        book_dir = Path(__file__).resolve().parent.parent.parent
        epub_files = list(book_dir.rglob("*.epub"))
        if not epub_files:
            return ValidationRunResult(
                name="epub", description="EPUB validation (no .epub files found)",
                files_checked=0, issues=[], elapsed_ms=0,
            )
        # Validate the most recent EPUB
        epub_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return self._delegate_script(script, ["--quick", str(epub_files[0])], "epub")

    # ------------------------------------------------------------------
    # Content tree: require shared/ and frontmatter/ (not only vol1/vol2)
    # ------------------------------------------------------------------

    # Required paths under contents/ so that scripts don't assume only vol1/vol2 exist.
    CONTENT_TREE_REQUIRED: List[tuple] = [
        ("shared", True),           # (path relative to contents, is_dir)
        ("shared/notation.qmd", False),
        ("frontmatter", True),
    ]

    def _run_content_tree(self, root: Path) -> ValidationRunResult:
        """Ensure contents/ has shared/ and frontmatter/; fail if they are missing."""
        t0 = time.time()
        # Resolve to contents dir: root may be contents, or contents/vol1, or contents/vol2
        if root.name in ("vol1", "vol2") and root.parent.name == "contents":
            contents_dir = root.parent
        else:
            contents_dir = root
        if not (contents_dir / "vol1").is_dir() or not (contents_dir / "vol2").is_dir():
            # Not the book contents root; skip (e.g. user passed a chapter path)
            return ValidationRunResult(
                name="content-tree",
                description="Content tree (shared/frontmatter required)",
                files_checked=0,
                issues=[],
                elapsed_ms=int((time.time() - t0) * 1000),
            )
        issues: List[ValidationIssue] = []
        for rel, is_dir in self.CONTENT_TREE_REQUIRED:
            path = contents_dir / rel
            if is_dir:
                if not path.is_dir():
                    issues.append(
                        ValidationIssue(
                            file=str(path),
                            line=0,
                            code="content-tree",
                            message=f"Required directory missing: contents/{rel} (shared content used by both volumes)",
                            severity="error",
                        )
                    )
            else:
                if not path.is_file():
                    issues.append(
                        ValidationIssue(
                            file=str(path),
                            line=0,
                            code="content-tree",
                            message=f"Required file missing: contents/{rel}",
                            severity="error",
                        )
                    )
        elapsed = int((time.time() - t0) * 1000)
        return ValidationRunResult(
            name="content-tree",
            description="Content tree (shared/frontmatter required)",
            files_checked=len(self.CONTENT_TREE_REQUIRED),
            issues=issues,
            elapsed_ms=elapsed,
        )

    # ------------------------------------------------------------------
    # Source citation validation  (delegated to manage_sources.py)
    # ------------------------------------------------------------------

    def _run_sources(self, root: Path) -> ValidationRunResult:
        """Validate source citations (asterisk sources, formatting, etc.)."""
        import subprocess as _sp
        script = (
            Path(__file__).resolve().parent.parent.parent
            / "tools" / "scripts" / "utilities" / "manage_sources.py"
        )
        # manage_sources.py expects to be run from the quarto root (where contents/ lives)
        quarto_dir = Path(__file__).resolve().parent.parent.parent / "quarto"
        t0 = time.time()
        cmd = ["python3", str(script), "--problems"]
        try:
            result = _sp.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(quarto_dir))
            elapsed = int((time.time() - t0) * 1000)
            if result.returncode == 0:
                return ValidationRunResult(
                    name="sources", description="Source citation validation",
                    files_checked=0, issues=[], elapsed_ms=elapsed,
                )
            output = (result.stdout + result.stderr).strip()
            return ValidationRunResult(
                name="sources", description="Source citation validation",
                files_checked=0, elapsed_ms=elapsed,
                issues=[ValidationIssue(
                    file="(script output)", line=0, code="sources",
                    message=output[:500] if output else f"Script exited with code {result.returncode}",
                    severity="error",
                )],
            )
        except FileNotFoundError:
            elapsed = int((time.time() - t0) * 1000)
            return ValidationRunResult(
                name="sources", description="Source citation validation",
                files_checked=0, elapsed_ms=elapsed,
                issues=[ValidationIssue(
                    file=str(script), line=0, code="sources",
                    message=f"Script not found: {script}", severity="error",
                )],
            )

    def _run_check_references(self, root: Path, ns: Optional[argparse.Namespace] = None) -> ValidationRunResult:
        """Validate .bib references against academic DBs (native implementation)."""
        repo_root = self.config_manager.book_dir.parent.parent
        if getattr(ns, "refs_file", None):
            bib_paths = [Path(f) if Path(f).is_absolute() else repo_root / f for f in ns.refs_file]
        else:
            bib_paths = [repo_root / p for p in reference_check.DEFAULT_BIB_REL_PATHS]
        output_path = Path(ns.refs_output) if getattr(ns, "refs_output", None) else None
        limit = getattr(ns, "refs_limit", None)
        skip_verified = getattr(ns, "refs_skip_verified", False)
        thorough = getattr(ns, "refs_thorough", False)
        cache_path = getattr(ns, "refs_cache", None)
        if cache_path is not None:
            cache_path = Path(cache_path) if Path(cache_path).is_absolute() else repo_root / cache_path
        else:
            cache_path = repo_root / ".references_verified.json"

        only_keys: Optional[List[str]] = None
        only_from_report = getattr(ns, "refs_only_from_report", None)
        only_keys_file = getattr(ns, "refs_only_keys_file", None)
        if only_from_report:
            report_path = Path(only_from_report) if Path(only_from_report).is_absolute() else repo_root / only_from_report
            if report_path.exists():
                only_keys = reference_check.parse_report_keys(report_path)
            else:
                console.print(f"[red]Report not found: {report_path}[/red]")
                return ValidationRunResult(name="references", description="Bibliography vs academic DBs (hallucinator)", files_checked=0, issues=[ValidationIssue(file=str(report_path), line=0, code="references", message=f"Report not found: {report_path}", severity="error")], elapsed_ms=0)
        elif only_keys_file:
            keys_path = Path(only_keys_file) if Path(only_keys_file).is_absolute() else repo_root / only_keys_file
            if keys_path.exists():
                only_keys = [line.strip() for line in keys_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            else:
                console.print(f"[red]Keys file not found: {keys_path}[/red]")
                return ValidationRunResult(name="references", description="Bibliography vs academic DBs (hallucinator)", files_checked=0, issues=[ValidationIssue(file=str(keys_path), line=0, code="references", message=f"Keys file not found: {keys_path}", severity="error")], elapsed_ms=0)

        passed, elapsed_ms, issue_dicts, files_checked = reference_check.run(
            bib_paths,
            output_path=output_path,
            limit=limit,
            dedupe=True,
            resilient=True,
            console=console,
            cache_path=cache_path,
            skip_verified=skip_verified,
            thorough=thorough,
            only_keys=only_keys,
        )
        issues = [
            ValidationIssue(
                file=d["file"],
                line=d["line"],
                code=d["code"],
                message=d["message"],
                severity=d.get("severity", "error"),
            )
            for d in issue_dicts
        ]
        return ValidationRunResult(
            name="references",
            description="Bibliography vs academic DBs (hallucinator)",
            files_checked=files_checked,
            issues=issues,
            elapsed_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _line_for_token(self, content: str, token: str) -> int:
        index = content.find(token)
        if index < 0:
            return 1
        return content[:index].count("\n") + 1

    def _print_human_summary(self, summary: Dict[str, Any], verbose: bool = True) -> None:
        runs = summary["runs"]
        total = summary["total_issues"]
        status = summary["status"]

        # On success, stay silent — pre-commit shows "Passed" and direct
        # callers see exit code 0.
        if total == 0:
            return

        # Show the summary table when there are issues
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Check", style="cyan")
        table.add_column("Files", style="dim")
        table.add_column("Issues", style="yellow")
        table.add_column("Elapsed", style="dim")
        table.add_column("Status", style="white")
        for run in runs:
            table.add_row(
                run["name"],
                str(run["files_checked"]),
                str(run["issue_count"]),
                f'{run["elapsed_ms"]}ms',
                "[green]PASS[/green]" if run["passed"] else "[red]FAIL[/red]",
            )
        console.print(Panel(table, title="Binder Check Summary", border_style="cyan"))

        # Count errors vs warnings across all runs
        total_errors = 0
        total_warnings = 0
        for run in runs:
            for issue in run["issues"]:
                if issue["severity"] == "error":
                    total_errors += 1
                else:
                    total_warnings += 1

        for run in runs:
            if run["issue_count"] == 0:
                continue
            run_errors = sum(1 for i in run["issues"] if i["severity"] == "error")
            run_warnings = run["issue_count"] - run_errors
            parts = []
            if run_errors:
                parts.append(f"{run_errors} error(s)")
            if run_warnings:
                parts.append(f"{run_warnings} warning(s)")
            label = ", ".join(parts)
            color = "bold red" if run_errors else "bold yellow"
            console.print(f"[{color}]{run['name']}[/{color}] ({label})")
            for issue in run["issues"][:30]:
                line = issue["line"]
                file = issue["file"]
                msg = issue["message"]
                sev = issue["severity"]
                sev_icon = "❌" if sev == "error" else "⚠️"
                console.print(f"  {sev_icon} {file}:{line} {msg}")
                if verbose and issue.get("context"):
                    console.print(f"     [dim]{issue['context']}[/dim]")
            if run["issue_count"] > 30:
                console.print(f"  [dim]... {run['issue_count'] - 30} more[/dim]")
            console.print()

        parts = []
        if total_errors:
            parts.append(f"{total_errors} error(s)")
        if total_warnings:
            parts.append(f"{total_warnings} warning(s)")
        label = " and ".join(parts)
        console.print(f"[red]❌ Validation failed with {label}.[/red]")

    def _emit(self, as_json: bool, payload: Dict[str, Any], failed: bool) -> None:
        if as_json:
            print(json.dumps(payload, indent=2))
            return
        if failed:
            console.print(f"[red]{payload.get('message', 'Operation failed')}[/red]")
        else:
            console.print(f"[green]{payload.get('message', 'Operation succeeded')}[/green]")

    # ------------------------------------------------------------------
    # Delegated checks (call existing scripts via subprocess)
    # ------------------------------------------------------------------

    @staticmethod
    def _delegate_script(script_path: Path, args: List[str], run_name: str) -> ValidationRunResult:
        """Run an external script and convert its exit code to a ValidationRunResult."""
        import subprocess
        t0 = time.time()
        cmd = ["python3", str(script_path)] + args
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            elapsed = int((time.time() - t0) * 1000)
            if result.returncode == 0:
                return ValidationRunResult(
                    name=run_name, description=run_name,
                    files_checked=0, issues=[], elapsed_ms=elapsed,
                )
            # Script failed — report its output as a single error
            output = (result.stdout + result.stderr).strip()
            return ValidationRunResult(
                name=run_name, description=run_name,
                files_checked=0, elapsed_ms=elapsed,
                issues=[ValidationIssue(
                    file="(script output)", line=0, code=run_name,
                    message=output[:500] if output else f"Script exited with code {result.returncode}",
                    severity="error",
                )],
            )
        except FileNotFoundError:
            elapsed = int((time.time() - t0) * 1000)
            return ValidationRunResult(
                name=run_name, description=run_name,
                files_checked=0, elapsed_ms=elapsed,
                issues=[ValidationIssue(
                    file=str(script_path), line=0, code=run_name,
                    message=f"Script not found: {script_path}", severity="error",
                )],
            )
        except subprocess.TimeoutExpired:
            elapsed = int((time.time() - t0) * 1000)
            return ValidationRunResult(
                name=run_name, description=run_name,
                files_checked=0, elapsed_ms=elapsed,
                issues=[ValidationIssue(
                    file=str(script_path), line=0, code=run_name,
                    message="Script timed out after 120s", severity="error",
                )],
            )

    def _run_grid_tables(self, root: Path) -> ValidationRunResult:
        """Check for grid tables (should be converted to pipe tables)."""
        script = (
            Path(__file__).resolve().parent.parent.parent
            / "tools" / "scripts" / "utilities" / "convert_grid_to_pipe_tables.py"
        )
        qmd_files = [str(f) for f in sorted(root.rglob("*.qmd"))]
        if not qmd_files:
            return ValidationRunResult(name="grid-tables", issues=[])
        return self._delegate_script(script, ["--check"] + qmd_files, "grid-tables")

    def _run_heading_case(self, root: Path) -> ValidationRunResult:
        """Enforce H1/H2 headline case + H3+ sentence case (MIT Press §10.3.1).

        Native implementation: imports from cli.commands.headings directly.
        Emits one ValidationIssue per violation with the offending current
        form and the expected sentence-case form.
        """
        from cli.commands.headings import find_violations
        t0 = time.time()
        qmd_files = [str(f) for f in sorted(root.rglob("*.qmd"))]
        violations = find_violations(qmd_files)
        issues: List[ValidationIssue] = []
        for v in violations:
            rel = v.path.replace(str(root) + "/", "")
            issues.append(ValidationIssue(
                file=rel, line=v.line, code="heading-case",
                message=f"expected: {v.expected.strip()}",
                severity="error", context=v.current.strip(),
            ))
        return ValidationRunResult(
            name="heading-case",
            description="H1/H2 headline + H3+ sentence case (§10.3.1)",
            files_checked=len(qmd_files), issues=issues,
            elapsed_ms=int((time.time() - t0) * 1000),
        )

    def _run_bib_hygiene(self, root: Path) -> ValidationRunResult:
        """Validate .bib files against §5 Bibliography Hygiene schema.

        Delegates to `book/tools/bib_lint.py --check`, which enforces the
        canonical schema: required fields per entry type, canonical field
        order, quoting style, author-list rules, journal spell-out,
        publisher canonical forms. Violations against the pre-existing
        baseline (`book/tools/bib_lint_baseline.json`) are grandfathered;
        only NEW violations block.
        """
        script = (
            Path(__file__).resolve().parent.parent.parent
            / "tools" / "bib_lint.py"
        )
        bib_files = [str(f) for f in sorted(root.rglob("*.bib"))]
        if not bib_files:
            return ValidationRunResult(
                name="bib-hygiene", description="bib-hygiene",
                files_checked=0, issues=[], elapsed_ms=0,
            )
        return self._delegate_script(script, ["--check"] + bib_files, "bib-hygiene")

    def _run_image_formats(self, root: Path) -> ValidationRunResult:
        """Validate image file formats using Pillow."""
        script = (
            Path(__file__).resolve().parent.parent.parent
            / "tools" / "scripts" / "images" / "manage_images.py"
        )
        image_files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.gif"):
            image_files.extend(str(f) for f in sorted(root.rglob(ext)))
        if not image_files:
            return ValidationRunResult(name="image-formats", issues=[])
        return self._delegate_script(script, image_files, "image-formats")

    def _run_external_images(self, root: Path) -> ValidationRunResult:
        """Check for external image URLs in QMD files."""
        script = (
            Path(__file__).resolve().parent.parent.parent
            / "tools" / "scripts" / "images" / "manage_external_images.py"
        )
        return self._delegate_script(
            script, ["--validate", str(root)], "external-images"
        )

    def _run_svg_wellformedness(self, root: Path) -> ValidationRunResult:
        """Validate SVG files are well-formed XML."""
        start = time.time()
        svg_files = [
            f for f in sorted(root.rglob("*.svg"))
            if "_files/mediabag/" not in str(f)
        ]
        issues: List[ValidationIssue] = []

        try:
            from lxml import etree
        except ImportError:
            return ValidationRunResult(
                name="svg-xml",
                description="Validate SVG XML well-formedness",
                files_checked=0,
                issues=[ValidationIssue(
                    file="(system)", line=0, code="svg_xml",
                    message="lxml not installed — skipping SVG validation",
                    severity="warning",
                )],
                elapsed_ms=int((time.time() - start) * 1000),
            )

        for svg_file in svg_files:
            try:
                etree.parse(str(svg_file))
            except etree.XMLSyntaxError as e:
                issues.append(
                    ValidationIssue(
                        file=self._relative_file(svg_file),
                        line=getattr(e, "lineno", 0) or 0,
                        code="svg_xml_error",
                        message=f"Malformed SVG XML: {e}",
                        severity="error",
                        context=str(e)[:120],
                    )
                )

        return ValidationRunResult(
            name="svg-xml",
            description="Validate SVG XML well-formedness",
            files_checked=len(svg_files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def _run_json_syntax(self, root: Path) -> ValidationRunResult:
        """Validate JSON file syntax."""
        t0 = time.time()
        json_files = sorted(root.rglob("*.json"))
        if not json_files:
            return ValidationRunResult(
                name="json-syntax", description="Validate JSON file syntax",
                files_checked=0, issues=[], elapsed_ms=0,
            )
        issues: List[ValidationIssue] = []
        for fpath in json_files:
            try:
                with open(fpath, "r") as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                issues.append(ValidationIssue(
                    file=str(fpath), line=e.lineno or 0, code="json-syntax",
                    message=f"Invalid JSON: {e.msg}", severity="error",
                ))
            except Exception as e:
                issues.append(ValidationIssue(
                    file=str(fpath), line=0, code="json-syntax",
                    message=f"Cannot read: {e}", severity="error",
                ))
        elapsed = int((time.time() - t0) * 1000)
        return ValidationRunResult(
            name="json-syntax", description="Validate JSON file syntax",
            files_checked=len(json_files), issues=issues, elapsed_ms=elapsed,
        )

    def _run_unit_tests(self, root: Path) -> ValidationRunResult:
        """Run physics engine unit conversion tests."""
        # validate.py is at book/cli/commands/validate.py
        # test_units.py is at book/tests/test_units.py
        book_dir = Path(__file__).resolve().parent.parent.parent  # book/
        script = book_dir / "tests" / "test_units.py"
        return self._delegate_script(script, [], "unit-tests")
