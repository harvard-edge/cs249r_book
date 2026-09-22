#!/usr/bin/env python3
"""Self-reported courses, read from the "Add your course" issues.

build_adopters.py gather calls submissions() on the scheduled refresh: every
issue titled "[Course] ..." whose listing box is ticked becomes a course in
adopters.json, with no pull request and no commit. Closing an issue as "not
planned" takes its course off the site; so does listing its domain or link
under `hidden:` in community/adopters.yml.

The issue body is untrusted input. It is parsed as the issue form's markdown,
every field is length-capped and stripped of newlines, and only the parsed
fields are published, as JSON, by the site's own renderer (which escapes
them). No submitted text reaches a shell.

Check a submission by hand:
  python3 site/scripts/adopters_intake.py --body-file issue.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

TITLE_PREFIX = "[Course]"
MAX = 200  # longest accepted field, in characters

FIELDS = {
    "institution": "Institution",
    "location": "City and country",
    "course": "Course code and title",
    "term": "Term(s) taught",
    "url": "Course page or syllabus",
    "usage": "How do you use the book?",
    "listing": "Listing",
}


def parse_form(body: str) -> dict:
    """Issue-form markdown: '### Label' headings, each followed by its value."""
    out, current, buf = {}, None, []
    labels = {v: k for k, v in FIELDS.items()}
    for line in (body or "").replace("\r\n", "\n").split("\n"):
        m = re.match(r"^###\s+(.+?)\s*$", line)
        if m:
            if current:
                out[current] = "\n".join(buf).strip()
            current, buf = labels.get(m.group(1)), []
        elif current:
            buf.append(line)
    if current:
        out[current] = "\n".join(buf).strip()
    for k, v in list(out.items()):
        if k != "listing":
            v = "" if v in ("_No response_", "None") else v
            out[k] = re.sub(r"\s+", " ", v).strip()[:MAX]
    return out


def ticked(listing: str, phrase: str) -> bool:
    return any(re.match(r"^- \[[xX]\]", ln.strip()) and phrase in ln for ln in (listing or "").split("\n"))


def school_domain(name: str, url: str, hipo: dict, matcher) -> str | None:
    """The school's web domain: from the course link when it is on the
    school's own site, otherwise from the university list by name."""
    host = (urllib.parse.urlsplit(url).hostname or "").lower().removeprefix("www.")
    labels = host.split(".")
    for i in range(len(labels) - 1):
        if ".".join(labels[i:]) in hipo:
            return ".".join(labels[i:])
    return matcher.match(name)


def entry_from_form(body: str, *, known: dict, hipo: dict, matcher, geo: dict, geocode) -> tuple[dict | None, str]:
    """One adopters entry from an issue body, or (None, reason)."""
    f = parse_form(body)
    missing = [FIELDS[k] for k in ("institution", "location", "course") if not f.get(k)]
    if missing:
        return None, "missing " + ", ".join(missing)
    if not ticked(f.get("listing", ""), "List this course"):
        return None, "listing box not ticked"
    url = f.get("url", "")
    if url and not re.match(r"^https?://[^\s<>\"]+$", url):
        return None, "course link is not a web address"

    city, _, country = f["location"].rpartition(",")
    city, country = (city.strip() or f["location"]), country.strip()
    domain = school_domain(f["institution"], url, hipo, matcher)
    if not domain:
        return None, "school not found in the university list"
    k = known.get(domain)
    latlon = (k or {}).get("latlon") or geocode(f"{f['institution']}, {f['location']}", geo) \
        or geocode(f["location"], geo)
    if not latlon:
        return None, "could not place the school"
    course = {"title": f["course"], "term": f.get("term") or None, "url": url or None}
    return {"name": (k or {}).get("name") or f["institution"], "domain": domain,
            "city": (k or {}).get("city") or city, "country": (k or {}).get("country") or country,
            "latlon": latlon, "kind": "course", "source": "self-reported",
            "courses": [course]}, "ok"


def fetch_issues(api, repo: str) -> list[dict]:
    """All "[Course]" issues, open and closed, newest first."""
    out, page = [], 1
    while page <= 10:
        status, res = api(f"repos/{repo}/issues?state=all&per_page=100&page={page}&sort=created")
        if status != 200 or not res:
            break
        out += [i for i in res if "pull_request" not in i and (i.get("title") or "").startswith(TITLE_PREFIX)]
        if len(res) < 100:
            break
        page += 1
    return out


def submissions(api, repo: str, **ctx) -> tuple[list[dict], list[str]]:
    """Entries for every listable submission, plus one log line per issue."""
    entries, log = [], []
    for issue in fetch_issues(api, repo):
        if issue.get("state") == "closed" and issue.get("state_reason") == "not_planned":
            log.append(f"#{issue['number']}: closed as not planned, skipped")
            continue
        entry, why = entry_from_form(issue.get("body") or "", **ctx)
        log.append(f"#{issue['number']}: {why}")
        if entry:
            entries.append(entry)
    return entries, log


def main() -> int:
    ap = argparse.ArgumentParser(description="Check how a submission would be listed.")
    ap.add_argument("--body-file", type=Path, required=True)
    args = ap.parse_args()
    import affiliations
    import build_adopters as ba
    import build_institutions as bi

    bi.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hipo = bi.load_hipo()
    matcher = affiliations.Matcher(list({id(u): u for u in hipo.values()}.values()))
    entry, why = entry_from_form(args.body_file.read_text(encoding="utf-8"),
                                 known={s["domain"]: s for s in ba.teaching()},
                                 hipo=hipo, matcher=matcher, geo={}, geocode=ba.geocode)
    print(json.dumps(entry, indent=1, ensure_ascii=False) if entry else f"not listed: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
