"""Per-concept warrant counts for ANSI/NISO Z39.19-2005 section 5.3.5, computed from recorded evidence.

Literary warrant: covered topics in the paper, course, and standards checklists that cite the concept, and
`corpus_chapters`, the book chapters whose text contains its label or an alias (the content the vocabulary indexes).
Organizational warrant: covered MLSysBook chapter topics that cite it plus its section-level book links.
User warrant: the standard means user requests and searches, which this project has not recorded, so
`user_proxy` counts StaffML questions tagged with the concept (term-assignment frequency, section 11.1.3.4).
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import yaml

BOOK_CHECKLIST = re.compile(r"^mlsysbook-")
TAG = re.compile(r"^taxonomy_concepts:\n((?:[ \t]+- \S+\n)+)", re.M)


WORD = re.compile(r"[^\W_]+")
_CORPUS_CACHE: dict[tuple, Counter] = {}


def _corpus_chapters(root: Path, concepts: list[dict]) -> Counter:
    """Count chapters using a concept's label or an alias as a whole word sequence.

    Short and all-capital names stay case-sensitive. Chapter word n-grams are looked up in one pass, and the result is
    cached per process because one build asks for it several times.
    """
    key = (str(root), tuple(sorted((c["id"], c["label"], *c.get("aliases", [])) for c in concepts)))
    if key in _CORPUS_CACHE:
        return _CORPUS_CACHE[key]
    folded: dict[tuple, set] = {}
    exact: dict[tuple, set] = {}
    for concept in concepts:
        for name in [concept["label"], *concept.get("aliases", [])]:
            words = tuple(WORD.findall(name))
            if words and (name.isupper() or len(name) <= 3):
                exact.setdefault(words, set()).add(concept["id"])
            elif words:
                folded.setdefault(tuple(w.lower() for w in words), set()).add(concept["id"])
    sizes = {len(k) for k in [*folded, *exact]}
    counts: Counter = Counter()
    chapters = sorted((root / "books").glob("vol[1-4]/*/*.qmd")) if (root / "books").is_dir() else []
    for path in chapters:
        words = WORD.findall(path.read_text(errors="ignore"))
        lowered = [w.lower() for w in words]
        seen: set[str] = set()
        for size in sizes:
            seen.update(c for g in {tuple(lowered[i:i + size]) for i in range(len(lowered) - size + 1)} & folded.keys() for c in folded[g])
            seen.update(c for g in {tuple(words[i:i + size]) for i in range(len(words) - size + 1)} & exact.keys() for c in exact[g])
        counts.update(seen)
    _CORPUS_CACHE[key] = counts
    return counts


@lru_cache(maxsize=1)
def _question_tags(root: Path) -> Counter:
    proxy: Counter = Counter()
    questions = root / "staffml/vault/questions"
    if questions.is_dir():
        for path in questions.rglob("*.yaml"):
            found = TAG.search(path.read_text())
            if found:
                proxy.update(set(re.findall(r"- (\S+)", found.group(1))))
    return proxy


def compute_warrant(root: Path, concepts: list[dict], book_links: list[dict]) -> dict[str, dict[str, int]]:
    literary, organizational = Counter(), Counter()
    corpus = _corpus_chapters(root, concepts)
    for path in sorted((root / "taxonomy/coverage").glob("*.yml")):
        record = yaml.safe_load(path.read_text())
        bucket = organizational if BOOK_CHECKLIST.match(record.get("id", "")) else literary
        for topic in record.get("topics", []):
            if topic.get("status") == "covered":
                bucket.update(set(topic.get("concepts") or []))
    organizational.update(link["concept"] for link in book_links)
    proxy = _question_tags(root)
    return {c["id"]: {"literary": literary[c["id"]], "corpus_chapters": corpus[c["id"]],
                      "organizational": organizational[c["id"]], "user_proxy": proxy[c["id"]]} for c in concepts}
