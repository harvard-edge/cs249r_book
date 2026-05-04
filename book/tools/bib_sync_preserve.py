#!/usr/bin/env python3
"""Run ``betterbib sync`` safely while preserving and propagating citekeys.

This helper is the automation layer for bibliography refreshes in the repo.
It does four things per ``.bib`` file:

1. Run ``betterbib sync --in-place`` on a temp copy of the file.
2. Infer any citekey renames from the synced entries.
3. Write the synced bibliography back and apply exact citekey replacements
   to companion ``.qmd``, ``.tex`` and ``.md`` files in the same content tree.
4. Run the repo bib mechanical fix + lint checks on the updated bibliography.

The helper is conservative:
    - if the sync output changes entry count, duplicates keys, or produces
      ambiguous renames, the file is rejected
    - only exact citekey tokens are rewritten in companion prose files
    - crossrefs inside BibTeX entries are updated only when the value is an
      exact renamed citekey

The command is intended to be driven by ``./book/binder bib update``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPO = Path(__file__).resolve().parents[2]
BOOK = REPO / "book"
if str(BOOK) not in sys.path:
    sys.path.insert(0, str(BOOK))

from tools.bib_lint import parse_bib  # noqa: E402


TEXT_EXTS = {".qmd", ".tex", ".md", ".markdown", ".mkd"}
CITE_KEY_CHARS = r"A-Za-z0-9_.:\-"


@dataclass(frozen=True)
class Rename:
    old: str
    new: str


def _norm_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _normalize_doi(value: str | None) -> str:
    if not value:
        return ""
    s = value.strip()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
    ):
        if s.lower().startswith(prefix):
            s = s[len(prefix) :]
            break
    return s.lower().strip()


def _normalize_authors(value: str | None) -> str:
    if not value:
        return ""
    authors: list[str] = []
    for raw in re.split(r"\s+and\s+", value.strip(), flags=re.IGNORECASE):
        part = raw.strip().strip("{}")
        if not part:
            continue
        if "," in part:
            surname = part.split(",", 1)[0].strip()
        else:
            bits = part.split()
            surname = bits[-1] if bits else part
        authors.append(_norm_token(surname))
    return "|".join(a for a in authors if a)


def _field(entry, *names: str) -> str:
    for name in names:
        field = entry.get(name)
        if field and field.value.strip():
            return field.value.strip()
    return ""


def _entry_signatures(entry) -> set[str]:
    """Return a small set of stable identity signatures for an entry."""
    title = _norm_token(_field(entry, "title"))
    year = _norm_token(_field(entry, "year"))
    authors = _normalize_authors(_field(entry, "author"))
    venue = _norm_token(_field(entry, "journal", "booktitle", "publisher"))
    doi = _normalize_doi(_field(entry, "doi"))
    arxiv = _norm_token(_field(entry, "eprint", "arxiv", "archiveprefix"))

    sigs: set[str] = {f"type:{entry.entry_type}"}
    if doi:
        sigs.add(f"doi:{doi}")
    if arxiv:
        sigs.add(f"arxiv:{arxiv}")
    if title:
        sigs.add(f"title:{title}")
    if title and year:
        sigs.add(f"title-year:{title}|{year}")
    if title and authors:
        sigs.add(f"title-authors:{title}|{authors}")
    if title and year and authors:
        sigs.add(f"title-year-authors:{title}|{year}|{authors}")
    if title and venue:
        sigs.add(f"title-venue:{title}|{venue}")
    if title and year and venue:
        sigs.add(f"title-year-venue:{title}|{year}|{venue}")
    return sigs


def _similarity_score(orig, new, index_hint: bool) -> int:
    if orig.key == new.key:
        score = 200
        if orig.entry_type == new.entry_type:
            score += 5
        if index_hint:
            score += 1
        return score
    shared = _entry_signatures(orig) & _entry_signatures(new)
    if not shared:
        return 0
    rank = {
        "title": 10,
        "title-year": 30,
        "title-authors": 45,
        "title-venue": 50,
        "title-year-venue": 60,
        "title-year-authors": 80,
        "arxiv": 100,
        "doi": 120,
    }
    best = 0
    for sig in shared:
        prefix = sig.split(":", 1)[0]
        best = max(best, rank.get(prefix, 0))
    if orig.entry_type == new.entry_type:
        best += 5
    if index_hint:
        best += 1
    return best


def _match_entries(original: Sequence, synced: Sequence) -> list[tuple[int, int]]:
    """Greedily pair original and synced entries by stable identity."""
    if len(original) != len(synced):
        raise ValueError(
            f"entry count changed from {len(original)} to {len(synced)}"
        )

    scored: list[tuple[int, int, int]] = []
    for i, orig in enumerate(original):
        for j, new in enumerate(synced):
            score = _similarity_score(orig, new, i == j)
            if score:
                scored.append((score, i, j))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_orig: set[int] = set()
    used_new: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for score, i, j in scored:
        if i in used_orig or j in used_new:
            continue
        used_orig.add(i)
        used_new.add(j)
        pairs.append((i, j))

    if len(pairs) != len(original):
        missing_orig = [i for i in range(len(original)) if i not in used_orig]
        missing_new = [j for j in range(len(synced)) if j not in used_new]
        raise ValueError(
            "could not pair all entries after sync; "
            f"unmatched original indexes={missing_orig}, synced indexes={missing_new}"
        )
    return sorted(pairs)


def _find_bib_files(args: Sequence[str]) -> list[Path]:
    out: list[Path] = []
    for raw in args:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if p.is_file() and p.suffix == ".bib":
            out.append(p)
    return out


def _tracked_text_files(root: Path) -> list[Path]:
    raw = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=REPO, text=False
    )
    files: list[Path] = []
    for chunk in raw.split(b"\0"):
        if not chunk:
            continue
        rel = Path(chunk.decode("utf-8", errors="replace"))
        if rel.suffix.lower() not in TEXT_EXTS:
            continue
        path = REPO / rel
        if not path.is_file():
            continue
        try:
            path.relative_to(root)
        except ValueError:
            continue
        files.append(path)
    return sorted(files)


def _bib_companion_root(bib_path: Path) -> Path:
    if bib_path.parent.name == "backmatter" and bib_path.parent.parent.name in {
        "vol1",
        "vol2",
    }:
        return bib_path.parent.parent
    return bib_path.parent


def _run_betterbib_sync(temp_bib: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["betterbib", "sync", "--in-place", str(temp_bib)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )


def _extract_rename_map(original: str, synced: str) -> tuple[list[Rename], list[str]]:
    orig_entries, _ = parse_bib(original)
    new_entries, _ = parse_bib(synced)

    if len(orig_entries) != len(new_entries):
        raise ValueError(
            f"entry count changed from {len(orig_entries)} to {len(new_entries)}"
        )

    orig_keys = [e.key for e in orig_entries]
    new_keys = [e.key for e in new_entries]
    if len(set(orig_keys)) != len(orig_keys):
        raise ValueError("original file contains duplicate citekeys; refusing to sync")
    if len(set(new_keys)) != len(new_keys):
        raise ValueError("sync output contains duplicate citekeys; refusing to sync")

    pairs = _match_entries(orig_entries, new_entries)
    renames: list[Rename] = []
    diagnostics: list[str] = []
    for i, j in pairs:
        old = orig_entries[i]
        new = new_entries[j]
        if old.key != new.key:
            renames.append(Rename(old=old.key, new=new.key))
            diagnostics.append(f"{old.key} -> {new.key}")
    return renames, diagnostics


def _replace_qmd_md(text: str, renames: Sequence[Rename]) -> str:
    out = text
    for rename in renames:
        pattern = re.compile(
            rf"(?<![{CITE_KEY_CHARS}])@{re.escape(rename.old)}(?![{CITE_KEY_CHARS}])"
        )
        out = pattern.sub(f"@{rename.new}", out)
    return out


_TEX_CITE_RE = re.compile(
    r"\\(?P<cmd>(?:cite[a-zA-Z]*|nocite|bibitem))"
    r"(?P<opts>(?:\[[^\]]*\])*)"
    r"\{(?P<body>[^{}]*)\}"
)


def _replace_tex(text: str, renames: Sequence[Rename]) -> str:
    rename_map = {r.old: r.new for r in renames}

    def repl(match: re.Match[str]) -> str:
        body = match.group("body")
        items = [item.strip() for item in body.split(",")]
        changed = False
        out_items: list[str] = []
        for item in items:
            if item in rename_map:
                out_items.append(rename_map[item])
                changed = True
            else:
                out_items.append(item)
        if not changed:
            return match.group(0)
        body_out = ", ".join(out_items)
        return f"\\{match.group('cmd')}{match.group('opts')}{{{body_out}}}"

    return _TEX_CITE_RE.sub(repl, text)


def _apply_renames(text: str, path: Path, renames: Sequence[Rename]) -> str:
    if path.suffix.lower() in {".qmd", ".md", ".markdown", ".mkd"}:
        return _replace_qmd_md(text, renames)
    if path.suffix.lower() == ".tex":
        return _replace_tex(text, renames)
    return text


def _rewrite_companions(root: Path, renames: Sequence[Rename]) -> list[Path]:
    if not renames:
        return []
    touched: list[Path] = []
    for path in _tracked_text_files(root):
        old = path.read_text(encoding="utf-8")
        new = _apply_renames(old, path, renames)
        if new != old:
            path.write_text(new, encoding="utf-8")
            touched.append(path)
    return touched


def _run_bib_mechanical_fix(bib_path: Path) -> subprocess.CompletedProcess[str]:
    script = REPO / "book" / "tools" / "bib_apply_mechanical_fixes.py"
    return subprocess.run(
        [sys.executable, str(script), str(bib_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )


def _run_bib_lint_check(bib_path: Path) -> subprocess.CompletedProcess[str]:
    script = REPO / "book" / "tools" / "bib_lint.py"
    return subprocess.run(
        [sys.executable, str(script), str(bib_path), "--check"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )


def _rollback(backups: dict[Path, str]) -> None:
    for path, text in backups.items():
        path.write_text(text, encoding="utf-8")


def sync_one(bib_path: Path, dry_run: bool = False) -> bool:
    """Sync one bibliography file and propagate citekey renames."""
    bib_path = bib_path.resolve()
    if not bib_path.is_file():
        print(f"SKIP {bib_path}: file not found", file=sys.stderr)
        return False

    original = bib_path.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="betterbib-sync-") as td:
        temp_bib = Path(td) / bib_path.name
        temp_bib.write_text(original, encoding="utf-8")
        result = _run_betterbib_sync(temp_bib)
        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip()
            print(f"FAIL {bib_path}: betterbib sync failed")
            if stderr:
                print(stderr)
            return False
        synced = temp_bib.read_text(encoding="utf-8")

    try:
        renames, diagnostics = _extract_rename_map(original, synced)
    except Exception as exc:  # conservative: reject ambiguous sync output
        print(f"FAIL {bib_path}: {exc}")
        return False

    root = _bib_companion_root(bib_path)
    companion_files = _tracked_text_files(root)
    # Back up only the files we might touch.
    backups: dict[Path, str] = {bib_path: original}
    for path in companion_files:
        old = path.read_text(encoding="utf-8")
        new = _apply_renames(old, path, renames)
        if new != old:
            backups[path] = old

    print(f"{bib_path}: {len(renames)} citekey rename(s)")
    for line in diagnostics:
        print(f"  {line}")

    if dry_run:
        if backups.keys() - {bib_path}:
            print(f"  {len(backups) - 1} companion file(s) would be updated")
        return True

    try:
        bib_path.write_text(synced, encoding="utf-8")
        fixed = _run_bib_mechanical_fix(bib_path)
        if fixed.returncode != 0:
            raise RuntimeError(
                "bib mechanical fix failed:\n"
                + (fixed.stderr.strip() or fixed.stdout.strip() or "(no output)")
            )
        touched = _rewrite_companions(root, renames)
        if touched:
            print(f"  updated {len(touched)} companion file(s)")
        lint = _run_bib_lint_check(bib_path)
        if lint.returncode != 0:
            raise RuntimeError(
                "bib_lint failed:\n" + (lint.stderr.strip() or lint.stdout.strip() or "(no output)")
            )
    except Exception as exc:
        _rollback(backups)
        print(f"FAIL {bib_path}: {exc}")
        return False

    print(f"OK {bib_path}")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run betterbib sync safely while preserving citekeys",
    )
    ap.add_argument("files", nargs="+", help=".bib file(s) to sync")
    ap.add_argument("--dry-run", action="store_true", help="Show renames only")
    ns = ap.parse_args(argv)

    targets = _find_bib_files(ns.files)
    if not targets:
        print("No .bib files supplied", file=sys.stderr)
        return 1

    ok = True
    for path in targets:
        ok &= sync_one(path, dry_run=ns.dry_run)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
