#!/usr/bin/env python3
"""Course repositories that use the book, found on GitHub.

build_adopters.py gather calls discovered() on the scheduled refresh. It
searches GitHub for repositories that link, name, or fork the book and lists
a repository as a course only on strong evidence, all three of:
  - it links the book in a file, or names it in its README or description;
  - it carries a course code of its own (not the book's "cs249r");
  - its owner's profile names a school in the university list.
Weaker leads are not published. Nothing is committed: the courses go into
adopters.json with the rest of the refresh. A domain or link listed under
`hidden:` in community/adopters.yml keeps a course off the site.

Code search across public repositories needs a personal or fine-grained
token (DISCOVERY_TOKEN in the workflow); with the Actions token alone, code
search may be refused and only repository search and forks are used.

Review the full candidate list, strong and weak, by hand:
  python3 site/scripts/adopters_discover.py --out report.md
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


def resolve_school(c: dict, prof: dict, matcher, known: dict) -> str | None:
    """The school behind a candidate repository, or None.

    Tried from most to least specific: the owner's profile (company, blog
    host), then the school named in the repository itself or the owner's
    bio ("Lecture notes for COE 379L at UT Austin", "Ursinus-CS477"), by
    alias, by a listed school's domain stem, or by a full university name.
    """
    import affiliations

    blog_host = (urllib.parse.urlsplit(prof.get("blog") or "").hostname or "").removeprefix("www.")
    for piece in filter(None, [prof.get("company"), blog_host]):
        dom = matcher.match(piece) or (piece if piece in matcher.by_domain else None)
        if dom:
            return dom
    text = " ".join(filter(None, [c["slug"].split("/")[1], c["desc"], prof.get("bio"),
                                  prof.get("company"), prof.get("location")]))
    # The book's home is named by every fan repo ("labs from Harvard's
    # CS249r"); it says nothing about who runs the repo, and Harvard's own
    # courses are curated. Strip it before matching.
    words = affiliations.norm(re.sub(r"[_\-/]+", " ", text))
    words = re.sub(r"\bharvard\b[\w' ]{0,40}|\bcs\s?249r?\b|\bseas\b", " ", words)
    for alias, dom in sorted(affiliations.ALIAS.items(), key=lambda kv: -len(kv[0])):
        if len(alias) >= 3 and re.search(rf"\b{re.escape(alias)}\b", words):
            return dom
    for dom in known:
        if dom == "harvard.edu":
            continue
        stem = dom.split(".")[0]
        if len(stem) >= 5 and re.search(rf"\b{re.escape(stem)}\b", words):
            return dom
    best = max((n for n in matcher.long_names if n in words), key=len, default=None)
    return matcher.by_name[best]["domains"][0].lower() if best else None


def discovered(tok: str, *, matcher, known: dict, geo: dict, geocode) -> tuple[list[dict], list[str]]:
    """Course entries for candidates with strong evidence (see module doc)."""
    found, notes = gather(tok)
    skip = listed_repos()
    found = {k: v for k, v in found.items() if k.lower() not in skip and k.split("/")[0].lower() not in skip}
    owners: dict[str, dict] = {}
    entries = []
    for c in score(tok, found, owners):
        linked = any(v.startswith(("links the book", "names the book")) for v in c["via"])
        if not linked or "course code in name or description" not in c["signals"]:
            continue
        prof = owners.get(c["slug"].split("/")[0], {})
        domain = resolve_school(c, prof, matcher, known)
        if not domain:
            notes.append(f"{c['slug']}: strong course signals but no identifiable school")
            continue
        school = known.get(domain) or matcher.school(domain)
        latlon = school.get("latlon") or geocode(f"{school['name']}, {school['country']}", geo)
        if not latlon:
            continue
        title = c["desc"] or c["slug"].split("/")[1].replace("-", " ").replace("_", " ")
        entries.append({"name": school["name"], "domain": domain, "city": school.get("city", ""),
                        "country": school["country"], "latlon": latlon, "kind": "course",
                        "source": "discovered",
                        "courses": [{"title": title[:120], "term": None, "url": c["url"]}]})
    return entries, notes


def report(cands: list[dict], notes: list[str]) -> str:
    esc = lambda s: s.replace("|", "\\|").replace("\n", " ")  # noqa: E731
    lines = [
        f"Candidate courses that use the book: {len(cands)}, not already in `site/community/adopters.yml`. "
        "Only those with a course code, a link to the book, and an identifiable school are published "
        "automatically; the rest are listed here for review.",
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
    args = ap.parse_args()
    tok = token()
    found, notes = gather(tok)
    skip = listed_repos()
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
