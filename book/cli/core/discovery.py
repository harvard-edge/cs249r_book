"""
File and chapter discovery for MLSysBook CLI.

Handles finding chapter files, validating paths, and managing file operations.
Supports volume-aware discovery for vol1 and vol2.
"""

import re
import fnmatch
from pathlib import Path
from typing import List, Optional, Dict, Any
from rich.console import Console

console = Console()

# Volume directories
VOLUME_DIRS = ["vol1", "vol2"]


class AmbiguousChapterError(Exception):
    """Raised when a chapter name exists in multiple volumes."""

    def __init__(self, chapter_name: str, locations: List[str]):
        self.chapter_name = chapter_name
        self.locations = locations
        super().__init__(
            f"'{chapter_name}' exists in multiple volumes: {', '.join(locations)}"
        )


class ChapterDiscovery:
    """Discovers and manages chapter files in the MLSysBook project."""

    def __init__(self, book_dir: Path):
        """Initialize chapter discovery.

        Args:
            book_dir: Path to the book directory (usually 'quarto')
        """
        self.book_dir = Path(book_dir)
        self.contents_dir = self.book_dir / "contents"

    def _get_volume_from_path(self, path: Path) -> Optional[str]:
        """Extract volume (vol1/vol2) from a file path.

        Args:
            path: Path to check

        Returns:
            'vol1', 'vol2', or None if not in a volume directory
        """
        try:
            rel_path = path.relative_to(self.contents_dir)
            parts = rel_path.parts
            if parts and parts[0] in VOLUME_DIRS:
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
            if parts[0] in VOLUME_DIRS:
                return parts[0], parts[1]
        return None, chapter_spec

    @staticmethod
    def _match_score(query: str, candidate: str) -> int:
        """Score how well a query matches a candidate chapter name.

        Higher score = better match. Uses longest common substring length
        as primary metric, with shorter candidate names preferred as tiebreaker.

        Args:
            query: The search term (e.g., 'dnn_')
            candidate: The chapter file stem (e.g., 'nn_architectures')

        Returns:
            Match score (higher is better), 0 if no match
        """
        q = query.lower()
        c = candidate.lower()

        # Exact match gets highest score
        if q == c:
            return 10000

        # Starts-with match gets high score, weighted by coverage
        if c.startswith(q):
            return 5000 + int(1000 * len(q) / len(c))

        # Contains match gets medium score, weighted by coverage
        if q in c:
            return 2000 + int(1000 * len(q) / len(c))

        # Partial overlap: find longest common substring
        best = 0
        for i in range(len(q)):
            for j in range(i + 1, len(q) + 1):
                sub = q[i:j]
                if sub in c and len(sub) > best:
                    best = len(sub)
        if best >= 2:
            return 500 + int(1000 * best / len(c))

        return 0

    def find_chapter_file(self, chapter_spec: str, allow_fuzzy: bool = False) -> Optional[Path]:
        """Find a chapter file by name, using best-match scoring.

        Supports volume-prefixed names (e.g., 'vol1/intro') for disambiguation.
        Raises AmbiguousChapterError if chapter exists in multiple volumes
        without a volume prefix.

        Matching strategy (in order of priority):
        1. Exact stem match (e.g., 'nn_computation' → nn_computation.qmd)
        2. Best fuzzy match scored by: starts-with > contains > partial overlap,
           with higher coverage (query length / candidate length) preferred.

        Args:
            chapter_spec: Chapter name to search for, optionally with volume prefix
            allow_fuzzy: If True, allow fuzzy fallback for non-exact matches.

        Returns:
            Path to the chapter file if found, None otherwise

        Raises:
            AmbiguousChapterError: If chapter exists in multiple volumes without prefix
        """
        if not self.contents_dir.exists():
            console.print(f"[red]Contents directory not found: {self.contents_dir}[/red]")
            return None

        # Parse volume prefix if present
        volume_filter, chapter_name = self._parse_chapter_spec(chapter_spec)

        # Determine search directory
        if volume_filter:
            search_dir = self.contents_dir / volume_filter
            if not search_dir.exists():
                console.print(f"[red]Volume directory not found: {search_dir}[/red]")
                return None
        else:
            search_dir = self.contents_dir

        # Try exact match first
        exact_matches = list(search_dir.rglob(f"{chapter_name}.qmd"))

        # Filter to actual chapter files (in volume directories, not frontmatter/backmatter)
        chapter_matches = []
        for match in exact_matches:
            vol = self._get_volume_from_path(match)
            if vol or volume_filter:
                chapter_matches.append(match)

        if not chapter_matches and allow_fuzzy:
            # No exact match — score all .qmd files and pick the best
            all_qmd_files = list(search_dir.rglob("*.qmd"))

            scored = []
            for match in all_qmd_files:
                vol = self._get_volume_from_path(match)
                if not (vol or volume_filter):
                    continue
                score = self._match_score(chapter_name, match.stem)
                if score > 0:
                    scored.append((score, match))

            if scored:
                # Sort by score descending
                scored.sort(key=lambda x: x[0], reverse=True)
                best_score = scored[0][0]
                # Reject weak fuzzy matches to avoid incorrect chapter resolution.
                if best_score < 2000:
                    return None
                # Collect all matches with the same best score
                chapter_matches = [m for s, m in scored if s == best_score]

        if not chapter_matches:
            return None

        if len(chapter_matches) == 1:
            return chapter_matches[0]

        # Multiple matches - check if they're in different volumes (ambiguous)
        if not volume_filter:
            volumes_found = {}
            for match in chapter_matches:
                vol = self._get_volume_from_path(match)
                if vol:
                    if vol not in volumes_found:
                        volumes_found[vol] = match

            if len(volumes_found) > 1:
                # Ambiguous only when the matched chapter stem is actually the same
                # in multiple volumes (e.g., vol1/introduction and vol2/introduction).
                stems = {m.stem for m in volumes_found.values()}
                if len(stems) == 1:
                    stem = next(iter(stems))
                    locations = [f"{vol}/{stem}" for vol in sorted(volumes_found.keys())]
                    raise AmbiguousChapterError(stem, locations)
                # Tied fuzzy matches with different stems are not reliable.
                return None

        # Return the first match
        return chapter_matches[0]

    def get_all_chapters(self, volume: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all chapter files with metadata.

        Args:
            volume: Optional volume filter ('vol1', 'vol2', or None for all)

        Returns:
            List of dictionaries containing chapter information
        """
        chapters = []

        if not self.contents_dir.exists():
            return chapters

        # Determine search directory
        if volume:
            if volume not in VOLUME_DIRS:
                console.print(f"[red]Invalid volume: {volume}. Use 'vol1' or 'vol2'[/red]")
                return chapters
            search_dir = self.contents_dir / volume
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
            volume: Volume to get chapters for ('vol1' or 'vol2')

        Returns:
            List of chapter file paths
        """
        chapters = self.get_all_chapters(volume=volume)
        return [ch["path"] for ch in chapters]

    def show_chapters(self, volume: Optional[str] = None) -> None:
        """Display available chapters in a formatted table.

        Args:
            volume: Optional volume filter ('vol1', 'vol2', or None for all)
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
        vol1_count = sum(1 for ch in chapters if ch["volume"] == "vol1")
        vol2_count = sum(1 for ch in chapters if ch["volume"] == "vol2")

        if volume:
            console.print(f"\n[dim]Found {len(chapters)} chapters in {volume}[/dim]")
        else:
            console.print(f"\n[dim]Found {len(chapters)} chapters (vol1: {vol1_count}, vol2: {vol2_count})[/dim]")

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
                chapter_file = self.find_chapter_file(chapter_name)
            except AmbiguousChapterError as e:
                console.print(f"[red]Ambiguous chapter: '{e.chapter_name}' exists in multiple volumes[/red]")
                console.print("[yellow]Please specify the volume:[/yellow]")
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
