#!/usr/bin/env python3
"""Find course repositories that use the book, for a maintainer to review.

Run weekly by .github/workflows/site-adopters-discover.yml. Searches GitHub
for repositories that link or fork the book, keeps the ones that look like a
course (a course code, course words, an organization owner, an academic
profile), drops anything already in community/adopters.yml or reported in an
earlier discovery issue, and writes a markdown checklist of what is left.

Nothing here changes the site. A maintainer reads the list and, for the real
courses, opens the "Add your course" form or edits adopters.yml.

Needs a token in GITHUB_TOKEN (or `gh auth`). Code search across public
repositories needs a personal or fine-grained token; with the Actions token
alone, code search may be refused and the run falls back to repository search
and forks, which it says in the report.

Usage:
  python3 site/scripts/adopters_discover.py --out report.md [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import build_adopters as ba  # noqa: E402

REPO = os.environ.get("GITHUB_REPOSITORY", "harvard-edge/cs249r_book")
ISSUE_PREFIX = "[Course discovery]"
CODE_QUERIES = ['"mlsysbook.ai"', '"harvard-edge.github.io/cs249r_book"', '"cs249r_book"']
REPO_QUERIES = ["mlsysbook in:readme,description", "cs249r in:readme,description",
                "\"machine learning systems\" janapa in:readme"]
COURSE_CODE = re.compile(r"\b[A-Za-z]{2,5}[\s_-]?\d{3,4}[A-Za-z]?\b")
COURSE_WORDS = re.compile(
    r"course|syllabus|lecture|semester|spring|fall|autumn|winter|homework|assignment|"
    r"class|university|universidad|universidade|universit|institut|college|"
    r"curso|disciplina|课程|教学|강의|講義", re.I)
ACADEMIC_HOST = re.compile(r"(\.edu|\.ac\.[a-z]{2}|\.edu\.[a-z]{2})(/|$)", re.I)
# The book's own identifiers look like a course code ("cs249r"); strip them
# before looking for one, or every fork of the book reads as a course.
BOOK_IDS = re.compile(r"cs249r(_book)?|mlsysbook|machine learning systems", re.I)
# Curated link lists link everything; they are not courses.
LIST_REPO = re.compile(r"awesome|curated|paper[- ]?list|reading[- ]?list|trackawesome", re.I)


def token() -> str:
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not tok:
        tok = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True).stdout.strip()
    if not tok:
        sys.exit("no GitHub token")
    return tok


def api(tok: str, path: str) -> tuple[int, dict | list | None]:
    url = path if path.startswith("http") else f"https://api.github.com/{path}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json",
        "User-Agent": "mlsysbook-adopters"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:  # noqa: BLE001
        return 0, None


def listed_repos() -> set[str]:
    """owner/repo slugs already in adopters.yml course links."""
    text = ba.DATA.read_text(encoding="utf-8")
    return {m.lower() for m in re.findall(r"github\.com/([\w.-]+/[\w.-]+)", text)} | \
           {m.lower() for m in re.findall(r"https?://([\w-]+)\.github\.io", text)}


def reported_repos(tok: str) -> set[str]:
    """Slugs listed in earlier discovery issues, open or closed."""
    seen: set[str] = set()
    q = urllib.parse.quote(f'repo:{REPO} is:issue in:title "{ISSUE_PREFIX}"')
    _, res = api(tok, f"search/issues?q={q}&per_page=50")
    for it in (res or {}).get("items", []):
        seen |= {m.lower() for m in re.findall(r"github\.com/([\w.-]+/[\w.-]+)", it.get("body") or "")}
    return seen


def gather(tok: str) -> tuple[dict[str, dict], list[str]]:
    found: dict[str, dict] = {}
    notes: list[str] = []

    def add(repo: dict, via: str) -> None:
        slug = repo["full_name"]
        if slug.split("/")[0].lower() == REPO.split("/")[0].lower():
            return
        r = found.setdefault(slug, {"repo": repo, "via": set()})
        r["via"].add(via)

    code_ok = True
    for q in CODE_QUERIES:
        for page in (1, 2, 3):
            status, res = api(tok, f"search/code?q={urllib.parse.quote(q)}&per_page=100&page={page}")
            time.sleep(7)  # code search allows about ten requests a minute
            if status != 200:
                code_ok = False
                break
            for it in res.get("items", []):
                add(it["repository"], f"links the book ({q.strip(chr(34))})")
            if len(res.get("items", [])) < 100:
                break
    if not code_ok:
        notes.append("Code search was refused for this token, so repos that only mention the book in a "
                     "file were not searched. Set a `DISCOVERY_TOKEN` secret (a fine-grained token with "
                     "public read access) to include them.")
    for q in REPO_QUERIES:
        status, res = api(tok, f"search/repositories?q={urllib.parse.quote(q)}&per_page=100")
        time.sleep(2)
        for it in (res or {}).get("items", []):
            add(it, "names the book in its README or description")
    page = 1
    while True:
        status, res = api(tok, f"repos/{REPO}/forks?per_page=100&page={page}")
        if status != 200 or not res:
            break
        for it in res:
            if it["owner"]["type"] == "Organization":
                add(it, "an organization forked the book")
        page += 1
        if len(res) < 100:
            break
    return found, notes


def score(tok: str, found: dict[str, dict], owners: dict[str, dict]) -> list[dict]:
    out = []
    for slug, f in found.items():
        repo = f["repo"]
        owner = repo["owner"]["login"]
        text = " ".join(filter(None, [repo.get("name"), repo.get("description"), repo.get("homepage")]))
        if LIST_REPO.search(text):
            continue
        text = BOOK_IDS.sub(" ", text)
        signals = []
        if COURSE_CODE.search(text.replace("_", " ")):
            signals.append("course code in name or description")
        if COURSE_WORDS.search(text):
            signals.append("course words")
        if repo["owner"]["type"] == "Organization":
            signals.append("owned by an organization")
        if owner not in owners and len(owners) < 120:
            owners[owner] = api(tok, f"users/{owner}")[1] or {}
        prof = owners.get(owner, {})
        prof_text = " ".join(filter(None, [prof.get("blog"), prof.get("company"), prof.get("email"),
                                           prof.get("name"), prof.get("bio")]))
        if ACADEMIC_HOST.search(prof_text) or COURSE_WORDS.search(prof.get("company") or ""):
            signals.append("academic owner profile")
        # Keep only real course evidence: a course code of its own, or an
        # academic owner. A fork needs the academic owner; any repo also
        # needs to link or name the book, which every source here implies.
        strong = {"course code in name or description", "academic owner profile"} & set(signals)
        only_fork = f["via"] == {"an organization forked the book"}
        if not strong or (only_fork and "academic owner profile" not in signals):
            continue
        weight = len(signals) + (1 if "links the book" in " ".join(f["via"]) else 0)
        if weight >= 2:
            out.append({"slug": slug, "url": repo["html_url"], "desc": (repo.get("description") or "")[:140],
                        "owner": prof.get("name") or owner, "where": prof.get("company") or prof.get("location") or "",
                        "via": sorted(f["via"]), "signals": signals, "weight": weight,
                        "pushed": (repo.get("pushed_at") or "")[:10]})
    return sorted(out, key=lambda c: (-c["weight"], c["slug"].lower()))


def report(cands: list[dict], notes: list[str]) -> str:
    esc = lambda s: s.replace("|", "\\|").replace("\n", " ")  # noqa: E731
    lines = [
        f"Weekly search for courses that use the book. {len(cands)} new candidate(s), not already listed "
        "in `site/community/adopters.yml` or in an earlier discovery issue.",
        "",
        "For each real course: check it assigns or links the book, then add it with the "
        f"[Add your course](https://github.com/{REPO}/issues/new?template=add_course.yml) form or by editing `adopters.yml`. "
        "Close this issue when done; anything left unticked is not reported again.",
        "",
    ]
    for c in cands:
        where = f" · {esc(c['where'])}" if c["where"] else ""
        desc = f": {esc(c['desc'])}" if c["desc"] else ""
        lines.append(f"- [ ] **[{c['slug']}](https://github.com/{c['slug']})**{desc}  \n"
                     f"  {esc(c['owner'])}{where}{' · last push ' + c['pushed'] if c['pushed'] else ''} · "
                     f"{', '.join(c['via'])}; {', '.join(c['signals']) or 'no course signals'}")
    if notes:
        lines += ["", "**Coverage note:** " + " ".join(notes)]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", help="skip the earlier-issue filter's API call")
    args = ap.parse_args()
    tok = token()
    found, notes = gather(tok)
    skip = listed_repos() | (set() if args.dry_run else reported_repos(tok))
    found = {k: v for k, v in found.items()
             if k.lower() not in skip and k.split("/")[0].lower() not in skip}
    cands = score(tok, found, {})
    args.out.write_text(report(cands, notes), encoding="utf-8")
    print(f"{len(found)} repos searched after filtering, {len(cands)} candidates -> {args.out}")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"count={len(cands)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
