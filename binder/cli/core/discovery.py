"""
File and chapter discovery for MLSysBook CLI.

Handles finding chapter files, validating paths, and managing file operations.
Supports volume-aware discovery for vol1 through vol4.

Single source of truth for chapter ordering: `get_chapters_from_config()` reads
the PDF YAML config for a volume and returns the ordered list of testable chapter
stems. All commands (debug, build, validate, etc.) should call this method rather
than maintaining their own exclusion lists or filesystem scans.
"""

import re
import fnmatch
from pathlib import Path
from typing import List, Optional, Dict, Any
from rich.console import Console

console = Console()

def discover_volumes(book_dir: Path) -> List[str]:
    """Dynamically discover all volume directories under book_dir.

    Scans the given directory for subdirectories matching the pattern ``vol\\d+``
    as well as standalone named volumes like ``tinytorch``. Sorts volume names
    numerically (e.g., vol1, vol2, vol3, vol4), followed by non-numeric volumes.
    If the directory does not exist or yields no volume folders, returns the
    canonical default list (``["vol1", "vol2", "vol3", "vol4"]``).

    Args:
        book_dir: Path to the books root directory containing volume folders.

    Returns:
        Sorted list of volume directory names.
    """
    if not book_dir.exists():
        return ["vol1", "vol2", "vol3", "vol4"]
    vols = [
        d.name for d in book_dir.iterdir()
        if d.is_dir() and re.match(r"^vol\d+$", d.name)
    ]
    if (book_dir / "tinytorch").is_dir():
        vols.append("tinytorch")
    if not vols:
        return ["vol1", "vol2", "vol3", "vol4"]
    return sorted(
        vols,
        key=lambda v: (0, int(v[3:])) if v.startswith("vol") and v[3:].isdigit() else (1, v)
    )


def format_volume_display_name(volume: str) -> str:
    """Format volume identifier into human-friendly display name.

    Converts identifiers like ``"vol1"`` to ``"Volume I"``, ``"vol4"`` to
    ``"Volume IV"``, and ``"tinytorch"`` to ``"TinyTorch"``. Any other volume
    identifier is capitalized.

    Args:
        volume: Volume directory identifier (e.g., ``"vol1"``, ``"tinytorch"``).

    Returns:
        Formatted human-readable display string.
    """
    roman_map = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X"}
    if volume.startswith("vol") and volume[3:].isdigit():
        num = int(volume[3:])
        return f"Volume {roman_map.get(num, str(num))}"
    if volume == "tinytorch":
        return "TinyTorch"
    return volume.capitalize()


# Default volume directories; dynamic discovery should be preferred via ChapterDiscovery or discover_volumes()
VOLUME_DIRS = ["vol1", "vol2", "vol3", "vol4"]

# Shared content directory (sibling to vol1/, vol2/ under contents/)
SHARED_DIR = "shared"

# Chapter stems that cannot be rendered standalone and are always excluded from
# per-chapter build/debug operations.
SKIP_STEMS = frozenset({"index", "references"})



def _chapters_from_html_sidebar(book_dir: Path, volume: str) -> List[str]:
    """Extract buildable chapter stems from the HTML config sidebar (href entries).

    Parses ``binder/config/_quarto-html-{volume}.yml`` for sidebar hrefs matching
    the volume, excluding stems in ``SKIP_STEMS`` (e.g. index, references).

    Args:
        book_dir: Path to the books root directory.
        volume: Target volume identifier (e.g., ``"vol1"``, ``"vol4"``).

    Returns:
        Ordered list of buildable chapter file stems from the sidebar.
    """
    html_config = book_dir / "config" / f"_quarto-html-{volume}.yml"
    if not html_config.is_file():
        return []
    content = html_config.read_text(encoding="utf-8")
    chapters: List[str] = []
    seen: set = set()
    for m in re.finditer(r'href:\s*(?:contents/)?([^\s#]+\.qmd)', content):
        path_str = m.group(1)
        if f"/{volume}/" not in path_str and not path_str.startswith(f"{volume}/"):
            continue
        stem = Path(path_str).stem
        if stem in SKIP_STEMS or stem in seen:
            continue
        seen.add(stem)
        chapters.append(stem)
    return chapters

def get_chapters_from_config(book_dir: Path, volume: str) -> List[str]:
    """Return the ordered list of buildable file stems from the PDF config.

    Reads ``binder/config/_quarto-pdf-{volume}.yml`` and extracts every entry
    under ``book.chapters`` — including frontmatter, parts pages, and shared
    files.  Appendices are excluded.  Only ``index.qmd`` and ``references.qmd``
    are skipped, as they cannot be rendered standalone.

    Args:
        book_dir: Path to the ``books/`` directory.
        volume: Volume identifier (e.g., ``"vol1"``, ``"vol2"``, ``"vol4"``).

    Returns:
        Ordered list of file stems in YAML order (e.g. ``["dedication",
        "introduction", "distributed_training", ...]``).  Empty list if the
        config is missing or cannot be parsed.
    """
    config_file = book_dir / "config" / f"_quarto-pdf-{volume}.yml"
    if not config_file.exists():
        sidebar = _chapters_from_html_sidebar(book_dir, volume)
        if sidebar:
            return sidebar
        vol_dir = book_dir / volume
        if vol_dir.is_dir():
            return sorted([
                p.stem for p in vol_dir.rglob("*.qmd")
                if p.stem not in SKIP_STEMS and not p.name.startswith("_")
            ])
        return []

    def _is_testable(path_str: str) -> bool:
        return Path(path_str).stem not in SKIP_STEMS

    # --- YAML-aware path (preferred) ---
    try:
        import yaml  # type: ignore

        raw = yaml.safe_load(config_file.read_text())
        chapter_entries = raw.get("book", {}).get("chapters", [])

        chapters: List[str] = []
        seen: set = set()
        for entry in chapter_entries:
            if isinstance(entry, str):
                path = entry
            elif isinstance(entry, dict):
                path = entry.get("file", "")
            else:
                continue
            if not path or not _is_testable(path):
                continue
            stem = Path(path).stem
            if stem and stem not in seen:
                seen.add(stem)
                chapters.append(stem)
    except Exception:
        pass

    if len(chapters) < 5:
        content = config_file.read_text()
        chapters_block_match = re.search(
            r'^\s{2}chapters:\s*\n(.*?)(?=^\s{2}\w|\Z)',
            content,
            re.MULTILINE | re.DOTALL,
        )
        block = chapters_block_match.group(1) if chapters_block_match else content
        for line in block.splitlines():
            if line.lstrip().startswith("#"):
                continue
            m = re.search(r'-\s*(?:contents/)?([^\s#]+\.qmd)', line)
            if not m:
                continue
            path_str = m.group(1)
            if not _is_testable(path_str):
                continue
            stem = Path(path_str).stem
            if stem not in seen:
                seen.add(stem)
                chapters.append(stem)

    if len(chapters) < 5:
        sidebar = _chapters_from_html_sidebar(book_dir, volume)
        if len(sidebar) > len(chapters):
            return sidebar

    return chapters


class AmbiguousChapterError(Exception):
    """Raised when a query identifies more than one chapter."""

    def __init__(self, chapter_name: str, locations: List[str]):
        self.chapter_name = chapter_name
        self.locations = locations
        super().__init__(
            f"'{chapter_name}' matches multiple chapters: {', '.join(locations)}"
        )


class ChapterDiscovery:
    """Discovers and manages chapter files in the MLSysBook project."""

    def __init__(self, book_dir: Path):
        """Initialize chapter discovery.

        Args:
            book_dir: Path to the book directory (usually 'quarto')
        """
        self.book_dir = Path(book_dir)
        self.contents_dir = self.book_dir

    def get_available_volumes(self) -> List[str]:
        """Return list of dynamically discovered volume names."""
        return discover_volumes(self.contents_dir)

    def get_chapters_from_config(self, volume: str) -> List[str]:
        """Return the ordered list of testable chapter stems for a volume.

        Delegates to the module-level ``get_chapters_from_config`` function so
        that all CLI commands share a single implementation.  Call this instead
        of ``get_volume_chapters`` whenever the canonical build order matters.

        Args:
            volume: Volume name (e.g. ``"vol1"``, ``"vol2"``, etc.).

        Returns:
            Ordered list of chapter stems from the PDF config (e.g.
            ``["introduction", "distributed_training", ...]``).
        """
        return get_chapters_from_config(self.book_dir, volume)

    def _get_volume_from_path(self, path: Path) -> Optional[str]:
        """Extract volume from a file path.

        Args:
            path: Path to check

        Returns:
            Volume string (e.g. 'vol1'), or None if not in a volume directory
        """
        try:
            rel_path = path.relative_to(self.contents_dir)
            parts = rel_path.parts
            if parts and (parts[0] in self.get_available_volumes() or re.match(r"^vol\d+$", parts[0])):
                return parts[0]
        except ValueError:
            pass
        return None

    def _parse_chapter_spec(self, chapter_spec: str) -> tuple[Optional[str], str]:
        """Parse a chapter specification that may include volume prefix.

        Args:
            chapter_spec: Chapter name, optionally with volume prefix (e.g., 'vol1/intro')

        Returns:
            Tuple of (volume, chapter_name) where volume may be None
        """
        if "/" in chapter_spec:
            parts = chapter_spec.split("/", 1)
            if parts[0] in self.get_available_volumes() or re.match(r"^vol\d+$", parts[0]) or parts[0] == "tinytorch":
                return parts[0], parts[1]
        return None, chapter_spec

    @staticmethod
    def _normalized_name(value: str) -> str:
        """Normalize chapter name or title for case- and punctuation-insensitive matching.

        Args:
            value: Raw string to normalize.

        Returns:
            Space-delimited string of lowercased alphanumeric tokens.
        """
        return " ".join(re.findall(r"[a-z0-9]+", value.lower()))

    @staticmethod
    def _chapter_title(path: Path) -> str:
        """Read the first H1 or YAML title without scanning the full chapter body.

        Args:
            path: Path to the target QMD file.

        Returns:
            Extracted title string, or an empty string if neither H1 nor YAML title was found.
        """
        with path.open(encoding="utf-8") as source:
            opening = source.read(16384)
        heading = re.search(r"^# +(.+?)\s*(?:\{[^}]*\})?\s*$", opening, re.MULTILINE)
        if heading:
            return heading.group(1)
        if opening.startswith("---\n"):
            import yaml
            try:
                metadata = yaml.safe_load(opening.split("---", 2)[1]) or {}
                return str(metadata.get("title", ""))
            except (ValueError, yaml.YAMLError):
                pass
        return ""

    def find_chapter_file(self, chapter_spec: str, allow_fuzzy: bool = False) -> Optional[Path]:
        """Resolve a filename, relative path, or readable chapter title.

        Explicit volume prefixes constrain every matching stage. Exact paths
        and stems win, then normalized titles, unique substrings, and finally
        conservative typo matching. Ambiguous queries always list candidates.

        Args:
            chapter_spec: Chapter filename, path, or title (e.g., 'intro', 'vol1/intro', '01_intro').
            allow_fuzzy: If True, enable substring and typo-tolerant fuzzy matching.

        Returns:
            Resolved Path to the chapter QMD file, or None if not found.

        Raises:
            AmbiguousChapterError: If the specification matches multiple chapters.
        """
        from difflib import SequenceMatcher

        volume, name = self._parse_chapter_spec(chapter_spec.strip())
        name = name.removesuffix(".qmd")
        search_dir = self.contents_dir / volume if volume else self.contents_dir
        if not search_dir.is_dir():
            return None
        candidates = [p for p in sorted(search_dir.rglob("*.qmd"))
                      if self._get_volume_from_path(p)]
        shared_dir = self.contents_dir / SHARED_DIR
        if shared_dir.is_dir():
            candidates.extend(sorted(shared_dir.rglob("*.qmd")))

        def choose(matches):
            matches = list(dict.fromkeys(matches))
            if len(matches) > 1:
                raise AmbiguousChapterError(chapter_spec, [
                    p.relative_to(self.contents_dir).as_posix().removesuffix(".qmd")
                    for p in matches
                ])
            return matches[0] if matches else None

        # Match relative paths literally; never interpret path traversal/globs.
        if "/" in name:
            matches = [p for p in candidates if
                       p.relative_to(search_dir if volume and self._get_volume_from_path(p)
                                     else self.contents_dir).as_posix().removesuffix(".qmd") == name]
            return choose(matches)
        exact = [p for p in candidates if p.stem.lower() == name.lower()]
        if exact:
            return choose(exact)
        if not allow_fuzzy:
            return None

        query = self._normalized_name(name)
        if not query:
            return None
        names = {p: (self._normalized_name(p.stem),
                     self._normalized_name(self._chapter_title(p))) for p in candidates}
        exact = [p for p, aliases in names.items() if query in aliases]
        if exact:
            return choose(exact)
        partial = [p for p, aliases in names.items() if any(query in alias for alias in aliases)]
        if partial:
            return choose(partial)
        # A typo must resemble most of the name; a two-letter coincidence is
        # insufficient. Close runners-up are shown instead of chosen by order.
        scored = sorted(((max(SequenceMatcher(None, query, alias).ratio()
                              for alias in aliases), p) for p, aliases in names.items()),
                        key=lambda item: (-item[0], str(item[1])))
        if not scored or scored[0][0] < 0.82:
            return None
        return choose([p for score, p in scored if score >= scored[0][0] - 0.08])

    def get_all_chapters(self, volume: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all chapter files with metadata.

        Args:
            volume: Optional volume filter (e.g., 'vol1', 'vol2', 'vol4', or None for all)

        Returns:
            List of dictionaries containing chapter information
        """
        chapters = []

        if not self.contents_dir.exists():
            return chapters

        # Determine search directory
        if volume:
            search_dir = self.contents_dir / volume
            if not search_dir.is_dir():
                console.print(f"[red]Volume directory not found: {volume}[/red]")
                return chapters
        else:
            search_dir = self.contents_dir

        for qmd_file in search_dir.rglob("*.qmd"):
            # Skip certain files
            if qmd_file.name in ["index.qmd", "404.qmd"]:
                continue

            # Get relative path from contents directory
            rel_path = qmd_file.relative_to(self.contents_dir)

            # Determine volume
            vol = self._get_volume_from_path(qmd_file)

            # Skip non-volume files (frontmatter, backmatter) unless searching all
            if not volume and not vol:
                continue

            # Extract chapter info
            chapter_info = {
                "name": qmd_file.stem,
                "path": qmd_file,
                "relative_path": rel_path,
                "directory": qmd_file.parent.name,
                "volume": vol,
                "size": qmd_file.stat().st_size if qmd_file.exists() else 0
            }

            chapters.append(chapter_info)

        # Sort by path for consistent ordering
        chapters.sort(key=lambda x: str(x["relative_path"]))
        return chapters

    def get_volume_chapters(self, volume: str) -> List[Path]:
        """Get all chapter file paths for a specific volume.

        Args:
            volume: Volume to get chapters for (e.g., 'vol1', 'vol2', 'vol4')

        Returns:
            List of chapter file paths
        """
        chapters = self.get_all_chapters(volume=volume)
        return [ch["path"] for ch in chapters]

    def show_chapters(self, volume: Optional[str] = None) -> None:
        """Display available chapters in a formatted table.

        Args:
            volume: Optional volume filter (e.g., 'vol1', 'vol2', 'vol4', or None for all)
        """
        from rich.table import Table

        chapters = self.get_all_chapters(volume=volume)

        if not chapters:
            console.print("[yellow]No chapters found[/yellow]")
            return

        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Chapter", style="green", width=25)
        table.add_column("Volume", style="magenta", width=8)
        table.add_column("Directory", style="cyan", width=20)
        table.add_column("Size", style="dim", width=10)

        for chapter in chapters:
            size_kb = chapter["size"] / 1024 if chapter["size"] > 0 else 0
            size_str = f"{size_kb:.1f} KB" if size_kb > 0 else "0 KB"

            table.add_row(
                chapter["name"],
                chapter["volume"] or "-",
                chapter["directory"],
                size_str
            )

        console.print(table)

        # Show volume summary
        counts = ", ".join(f"{vol}: {sum(ch['volume'] == vol for ch in chapters)}"
                           for vol in self.get_available_volumes())

        if volume:
            console.print(f"\n[dim]Found {len(chapters)} chapters in {volume}[/dim]")
        else:
            console.print(f"\n[dim]Found {len(chapters)} chapters ({counts})[/dim]")

    def validate_chapters(self, chapter_names: List[str]) -> List[Path]:
        """Validate a list of chapter names and return their paths.

        Args:
            chapter_names: List of chapter names to validate

        Returns:
            List of valid chapter file paths

        Raises:
            FileNotFoundError: If any chapter is not found
            AmbiguousChapterError: If chapter exists in multiple volumes
        """
        chapter_files = []

        for chapter_name in chapter_names:
            try:
                chapter_file = self.find_chapter_file(chapter_name, allow_fuzzy=True)
            except AmbiguousChapterError as e:
                console.print(f"[red]Ambiguous chapter: {e.chapter_name}[/red]")
                console.print("[yellow]Use a more specific title or one of these paths:[/yellow]")
                for loc in e.locations:
                    console.print(f"  - {loc}")
                raise

            if not chapter_file:
                available_chapters = [ch["name"] for ch in self.get_all_chapters()]
                console.print(f"[red]Chapter not found: {chapter_name}[/red]")
                console.print("[yellow]Available chapters:[/yellow]")
                for ch in available_chapters[:10]:  # Show first 10
                    console.print(f"  - {ch}")
                if len(available_chapters) > 10:
                    console.print(f"  ... and {len(available_chapters) - 10} more")
                raise FileNotFoundError(f"Chapter not found: {chapter_name}")

            chapter_files.append(chapter_file)

        return chapter_files

    def expand_chapter_patterns(
        self,
        chapter_specs: List[str],
        *,
        volume: Optional[str] = None,
    ) -> List[str]:
        """
        Expand glob/regex chapter patterns into concrete chapter specs.

        Supported pattern forms:
        - **Glob** (default): `appendix*`, `*principles`, `vol1/appendix_*`
        - **Regex**: prefix with `re:` (matched with `re.search`), e.g. `re:^appendix_`

        Notes:
        - If a token has no wildcard/meta and doesn't start with `re:`, it is returned unchanged.
        - If a pattern matches nothing, it is returned unchanged (so existing fuzzy matching
          behavior remains available); callers may still fail later during validation.
        - Order is preserved; duplicates are removed.
        """
        # Candidate names come from discovery; includes front/backmatter too (useful for appendix*).
        all_candidates = [ch["name"] for ch in self.get_all_chapters(volume=volume)]

        expanded: List[str] = []
        seen = set()

        def _append(spec: str) -> None:
            if spec not in seen:
                expanded.append(spec)
                seen.add(spec)

        for spec in chapter_specs:
            spec = spec.strip()
            if not spec:
                continue

            spec_volume, name_or_pat = self._parse_chapter_spec(spec)
            local_volume = spec_volume or volume

            candidates = (
                [ch["name"] for ch in self.get_all_chapters(volume=local_volume)]
                if local_volume
                else all_candidates
            )

            is_regex = name_or_pat.startswith("re:")
            is_glob = any(ch in name_or_pat for ch in ["*", "?", "["])

            matches: List[str] = []
            if is_regex:
                pat = name_or_pat[len("re:") :]
                try:
                    rx = re.compile(pat)
                    matches = [c for c in candidates if rx.search(c)]
                except re.error:
                    matches = []
            elif is_glob:
                matches = [c for c in candidates if fnmatch.fnmatchcase(c, name_or_pat)]

            if matches:
                for m in matches:
                    _append(f"{local_volume}/{m}" if spec_volume else m)
            else:
                # Not a pattern, or didn't match: keep original token for existing behavior.
                _append(spec)

        return expanded

    def get_chapter_dependencies(self, chapter_file: Path) -> List[Path]:
        """Get dependencies for a chapter (images, includes, etc.).

        Args:
            chapter_file: Path to the chapter file

        Returns:
            List of dependency file paths
        """
        dependencies = []

        if not chapter_file.exists():
            return dependencies

        try:
            content = chapter_file.read_text(encoding='utf-8')

            # Find image references
            image_pattern = r'!\[.*?\]\((.*?)\)'
            for match in re.finditer(image_pattern, content):
                image_path = match.group(1)
                if not image_path.startswith('http'):
                    # Resolve relative to chapter file
                    full_path = (chapter_file.parent / image_path).resolve()
                    if full_path.exists():
                        dependencies.append(full_path)

            # Find include references
            include_pattern = r'{{< include (.*?) >}}'
            for match in re.finditer(include_pattern, content):
                include_path = match.group(1)
                full_path = (chapter_file.parent / include_path).resolve()
                if full_path.exists():
                    dependencies.append(full_path)

        except Exception as e:
            console.print(f"[yellow]⚠️  Error reading chapter dependencies: {e}[/yellow]")

        return dependencies
