#!/usr/bin/env python3
"""WARNING: reviewed bibliography migration helper only.

Run ``betterbib sync`` with a reviewed per-entry merge.

This helper is the automation layer for bibliography refreshes in the repo.
It does four things per ``.bib`` file:

1. Run ``betterbib sync --in-place`` on a temp copy of the file.
2. Compare the original and synced entries, keeping only same-work updates.
3. Infer any accepted citekey renames and apply exact citekey replacements
   to companion ``.qmd``, ``.tex`` and ``.md`` files in the same content tree.
4. Run the repo bib mechanical fix + lint checks on the updated bibliography.

The helper is deliberately reviewed:
    - same-work sync updates are accepted
    - radical record swaps are rejected and the original entry is kept
    - only exact citekey tokens are rewritten in companion prose files

The command is intended to be driven by ``./book/binder bib update`` and
should not be run unattended.
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO = Path(__file__).resolve().parents[2]
BOOK = REPO / "book"
if str(BOOK) not in sys.path:
    sys.path.insert(0, str(BOOK))

from tools.bib_lint import parse_bib  # noqa: E402
from tools.bib_lint import format_entry  # noqa: E402
from tools.bib_lint import validate_entry  # noqa: E402


TEXT_EXTS = {".qmd", ".tex", ".md", ".markdown", ".mkd"}
CITE_KEY_CHARS = r"A-Za-z0-9_.:\-"
VOLUME_CHUNK_SIZE = 40
CHUNKED_BIBS = {
    "book/quarto/contents/vol1/backmatter/references.bib",
    "book/quarto/contents/vol2/backmatter/references.bib",
}


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


def _title_similarity(orig, new) -> float:
    a = _norm_token(_field(orig, "title"))
    b = _norm_token(_field(new, "title"))
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _title_tokens(entry) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _field(entry, "title").lower()))


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
    title_sim = _title_similarity(orig, new)
    overlap = len(_title_tokens(orig) & _title_tokens(new))
    if title_sim >= 0.25:
        best = max(best, int(title_sim * 50))
    if overlap >= 1:
        best += overlap * 4
    if overlap >= 2:
        best += 6
    if overlap >= 3:
        best += 6
    if _norm_token(_field(orig, "year")) and _norm_token(_field(orig, "year")) == _norm_token(_field(new, "year")):
        best += 14
    if _normalize_authors(_field(orig, "author")) and _normalize_authors(_field(orig, "author")) == _normalize_authors(_field(new, "author")):
        best += 16
    if _norm_token(_field(orig, "journal", "booktitle", "publisher")) and _norm_token(_field(orig, "journal", "booktitle", "publisher")) == _norm_token(_field(new, "journal", "booktitle", "publisher")):
        best += 8
    if orig.entry_type == new.entry_type:
        best += 5
    if index_hint:
        best += 1
    return best


def _match_entries(original: Sequence, synced: Sequence) -> list[tuple[int, int, int]]:
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
    pairs: list[tuple[int, int, int]] = []
    for score, i, j in scored:
        if i in used_orig or j in used_new:
            continue
        used_orig.add(i)
        used_new.add(j)
        pairs.append((score, i, j))

    if len(pairs) != len(original):
        missing_orig = [i for i in range(len(original)) if i not in used_orig]
        missing_new = [j for j in range(len(synced)) if j not in used_new]
        raise ValueError(
            "could not pair all entries after sync; "
            f"unmatched original indexes={missing_orig}, synced indexes={missing_new}"
        )
    return sorted(pairs, key=lambda item: item[1])


def _entry_field_map(entry) -> dict[str, str]:
    out: dict[str, str] = {}
    for field in entry.fields:
        out[field.name.lower()] = field.value.strip()
    return out


def _entry_diff_summary(orig, new) -> list[str]:
    old_fields = _entry_field_map(orig)
    new_fields = _entry_field_map(new)
    names = sorted(set(old_fields) | set(new_fields))
    changed: list[str] = []
    for name in names:
        if old_fields.get(name, "") != new_fields.get(name, ""):
            changed.append(name)
    return changed


def _merge_same_work_entry(orig, new):
    """Merge a synced entry with its original fallback fields."""
    merged_fields: list = []
    seen: set[str] = set()
    for field in new.fields:
        merged_fields.append(field)
        seen.add(field.name.lower())
    for field in orig.fields:
        name = field.name.lower()
        if name in seen:
            continue
        if not field.value.strip():
            continue
        merged_fields.append(field)
        seen.add(name)
    return type(orig)(
        entry_type=new.entry_type,
        key=new.key,
        fields=merged_fields,
        raw=new.raw,
        start_line=new.start_line,
    )


def _entry_error_messages(entry) -> list[str]:
    return [
        v.message
        for v in validate_entry(entry)
        if v.severity == "error"
    ]


def _entry_batches(entries: Sequence, chunk_size: int) -> list[list]:
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    return [list(entries[i : i + chunk_size]) for i in range(0, len(entries), chunk_size)]


def _render_entries(entries: Sequence) -> str:
    return "\n\n".join(format_entry(entry) for entry in entries).rstrip() + "\n"


def _review_entry_pairs(
    original_entries: Sequence,
    synced_entries: Sequence,
    *,
    reserved_keys: set[str] | None = None,
    global_original_keys: set[str] | None = None,
) -> tuple[list, list[Rename], list[str]]:
    if len(original_entries) != len(synced_entries):
        raise RuntimeError(
            f"entry count changed from {len(original_entries)} to {len(synced_entries)}"
        )
    merged_entries: list = []
    renames: list[Rename] = []
    diagnostics: list[str] = []
    original_keys = {e.key for e in original_entries}
    if len(original_keys) != len(original_entries):
        raise RuntimeError("original file contains duplicate citekeys")

    merged_key_set: set[str] = set()
    reserved_keys = reserved_keys or set()
    global_original_keys = global_original_keys or set()
    for i, (old, new) in enumerate(zip(original_entries, synced_entries)):
        if _same_work(old, new):
            merged_new = _merge_same_work_entry(old, new)
            new_errors = _entry_error_messages(merged_new)
            fallback = None
            fallback_errors: list[str] = []
            if new_errors and old.entry_type != new.entry_type:
                fallback = type(old)(
                    entry_type=old.entry_type,
                    key=new.key,
                    fields=merged_new.fields,
                    raw=new.raw,
                    start_line=new.start_line,
                )
                fallback_errors = _entry_error_messages(fallback)
            if new_errors and fallback_errors:
                if old.key != new.key:
                    if new.key in reserved_keys or (
                        new.key in global_original_keys and new.key != old.key
                    ):
                        chosen = old
                        diagnostics.append(
                            f"REJECT {old.key} -> {new.key} "
                            f"(changed fields: {', '.join(_entry_diff_summary(old, new)) or 'none'}, "
                            f"key-conflict; lint={'; '.join(new_errors)})"
                        )
                    else:
                        chosen = old
                        diagnostics.append(
                            f"REJECT {old.key} -> {new.key} "
                            f"(changed fields: {', '.join(_entry_diff_summary(old, new)) or 'none'}, "
                            f"lint={'; '.join(new_errors)})"
                        )
                else:
                    chosen = old
                    diagnostics.append(
                        f"REJECT {old.key} "
                        f"(changed fields: {', '.join(_entry_diff_summary(old, new)) or 'none'}, "
                        f"lint={'; '.join(new_errors)})"
                    )
            elif new_errors and fallback is not None:
                merged_new = fallback
                new_errors = fallback_errors
                if old.key != new.key:
                    if new.key in reserved_keys or (
                        new.key in global_original_keys and new.key != old.key
                    ):
                        chosen = old
                        diagnostics.append(
                            f"REJECT {old.key} -> {new.key} "
                            f"(changed fields: {', '.join(_entry_diff_summary(old, new)) or 'none'}, "
                            f"key-conflict; lint={'; '.join(fallback_errors)})"
                        )
                    else:
                        chosen = merged_new
                        renames.append(Rename(old=old.key, new=new.key))
                        diagnostics.append(
                            f"ACCEPT {old.key} -> {new.key} "
                            f"(changed fields: {', '.join(_entry_diff_summary(old, new)) or 'none'}, "
                            f"type-fallback={old.entry_type})"
                        )
                else:
                    chosen = merged_new
                    diagnostics.append(
                        f"ACCEPT {old.key} "
                        f"(changed fields: {', '.join(_entry_diff_summary(old, new)) or 'none'}, "
                        f"type-fallback={old.entry_type})"
                    )
            elif old.key != new.key:
                if new.key in reserved_keys or (
                    new.key in global_original_keys and new.key != old.key
                ):
                    chosen = old
                    diagnostics.append(
                        f"REJECT {old.key} -> {new.key} "
                        f"(changed fields: {', '.join(_entry_diff_summary(old, new)) or 'none'}, "
                        f"key-conflict)"
                    )
                else:
                    chosen = merged_new
                    renames.append(Rename(old=old.key, new=new.key))
                    diagnostics.append(
                        f"ACCEPT {old.key} -> {new.key} "
                        f"(changed fields: {', '.join(_entry_diff_summary(old, new)) or 'none'})"
                    )
            else:
                chosen = merged_new
                changed = _entry_diff_summary(old, new)
                diagnostics.append(
                    f"ACCEPT {old.key} (changed fields: {', '.join(changed) or 'none'})"
                )
        else:
            chosen = old
            diagnostics.append(
                f"REJECT {old.key} -> {new.key} "
                f"(changed fields: {', '.join(_entry_diff_summary(old, new)) or 'none'}, "
                f"index={i})"
            )
        if chosen.key in merged_key_set:
            raise RuntimeError(f"merged output would duplicate citekey `{chosen.key}`")
        merged_key_set.add(chosen.key)
        reserved_keys.add(chosen.key)
        merged_entries.append(chosen)
    return merged_entries, renames, diagnostics


def _same_work(orig, new) -> bool:
    if _normalize_doi(_field(orig, "doi")) and _normalize_doi(_field(orig, "doi")) == _normalize_doi(_field(new, "doi")):
        return True
    if _norm_token(_field(orig, "eprint", "arxiv", "archiveprefix")) and _norm_token(_field(orig, "eprint", "arxiv", "archiveprefix")) == _norm_token(_field(new, "eprint", "arxiv", "archiveprefix")):
        return True
    shared = _entry_signatures(orig) & _entry_signatures(new)
    if any(
        sig.split(":", 1)[0]
        in {"doi", "arxiv", "title-year", "title-authors", "title-venue",
            "title-year-authors", "title-year-venue"}
        for sig in shared
    ):
        return True

    title_sim = _title_similarity(orig, new)
    overlap = len(_title_tokens(orig) & _title_tokens(new))
    year_same = _norm_token(_field(orig, "year")) == _norm_token(_field(new, "year"))
    authors_same = _normalize_authors(_field(orig, "author")) == _normalize_authors(_field(new, "author"))
    venue_same = _norm_token(_field(orig, "journal", "booktitle", "publisher")) == _norm_token(_field(new, "journal", "booktitle", "publisher"))
    if title_sim >= 0.9 and (year_same or authors_same or venue_same):
        return True
    if title_sim >= 0.4 and overlap >= 2:
        return True
    if title_sim >= 0.25 and overlap >= 3:
        return True
    if overlap >= 4:
        return True
    return False


def _is_chunked_bib(bib_path: Path) -> bool:
    try:
        rel = bib_path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return False
    return rel in CHUNKED_BIBS


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


def _sync_entries(
    entries: Sequence,
    *,
    label: str,
    reserved_keys: set[str] | None = None,
    global_original_keys: set[str] | None = None,
) -> tuple[list, list[Rename], list[str]]:
    temp_text = _render_entries(entries)
    with tempfile.TemporaryDirectory(prefix="betterbib-sync-chunk-") as td:
        temp_bib = Path(td) / "chunk.bib"
        temp_bib.write_text(temp_text, encoding="utf-8")
        result = _run_betterbib_sync(temp_bib)
        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                f"betterbib sync failed for {label}" + (f": {stderr}" if stderr else "")
            )
        synced_text = temp_bib.read_text(encoding="utf-8")
    synced_entries, _ = parse_bib(synced_text)
    return _review_entry_pairs(
        entries,
        synced_entries,
        reserved_keys=reserved_keys,
        global_original_keys=global_original_keys,
    )


def sync_one(bib_path: Path, dry_run: bool = False) -> bool:
    """Sync one bibliography file and propagate citekey renames."""
    bib_path = bib_path.resolve()
    if not bib_path.is_file():
        print(f"SKIP {bib_path}: file not found", file=sys.stderr)
        return False

    original = bib_path.read_text(encoding="utf-8")
    original_entries, original_preamble = parse_bib(original)
    global_original_keys = {e.key for e in original_entries}
    chunked = _is_chunked_bib(bib_path)
    batches = _entry_batches(original_entries, VOLUME_CHUNK_SIZE) if chunked else [list(original_entries)]

    merged_entries: list = []
    renames: list[Rename] = []
    diagnostics: list[str] = []
    reserved_keys: set[str] = set()
    try:
        for batch_index, batch in enumerate(batches, start=1):
            label = f"{bib_path} chunk {batch_index}/{len(batches)}"
            merged_batch, batch_renames, batch_diagnostics = _sync_entries(
                batch,
                label=label,
                reserved_keys=reserved_keys,
                global_original_keys=global_original_keys,
            )
            merged_entries.extend(merged_batch)
            renames.extend(batch_renames)
            diagnostics.extend(batch_diagnostics)
    except Exception as exc:
        print(f"FAIL {bib_path}: {exc}")
        return False

    root = _bib_companion_root(bib_path)
    companion_files = _tracked_text_files(root)
    backups: dict[Path, str] = {bib_path: original}
    for path in companion_files:
        old = path.read_text(encoding="utf-8")
        new = _apply_renames(old, path, renames)
        if new != old:
            backups[path] = old

    accepted = sum(1 for line in diagnostics if line.startswith("ACCEPT"))
    rejected = sum(1 for line in diagnostics if line.startswith("REJECT"))
    print(
        f"{bib_path}: {accepted} accepted citekey rename(s), "
        f"{rejected} rejected entr{'y' if rejected == 1 else 'ies'}"
    )
    for line in diagnostics:
        print(f"  {line}")

    if dry_run:
        if backups.keys() - {bib_path}:
            print(f"  {len(backups) - 1} companion file(s) would be updated")
        return True

    try:
        merged = ""
        if original_preamble and original_preamble[0].strip():
            merged = original_preamble[0].rstrip() + "\n\n"
        merged += _render_entries(merged_entries).rstrip() + "\n"
        if merged != original:
            bib_path.write_text(merged, encoding="utf-8")
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
