#!/usr/bin/env python3
"""Turn an "Add your course" issue into an adopters.yml entry.

Run by .github/workflows/site-adopters-intake.yml when an issue titled
"[Course] ..." is opened or edited. The issue body is untrusted input: it is
parsed as the issue form's markdown, every field is length-capped and
stripped of newlines, and the entry is written with yaml.safe_dump, so no
submitted text can reach a shell or break the YAML. A maintainer still
reviews and merges the resulting pull request; nothing here publishes.

Reads the body from $ISSUE_BODY (or --body-file). Writes:
  - a new block appended to community/adopters.yml
  - the school's site icon, only when the submitter ticked the logo box
  - $GITHUB_OUTPUT: status (ok | skip | needs-info), message, title
  - --summary FILE: markdown for the pull-request body

Usage:
  ISSUE_BODY="..." python3 site/scripts/adopters_intake.py --issue 123 --summary pr.md
  python3 site/scripts/adopters_intake.py --body-file issue.md --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import build_adopters as ba  # noqa: E402  (teaching(), png_width, paths)

UA = "mlsysbook-adopters/1.0 (https://mlsysbook.ai)"
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
    for line in body.replace("\r\n", "\n").split("\n"):
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
    return any(re.match(r"^- \[[xX]\]", ln.strip()) and phrase in ln for ln in listing.split("\n"))


def get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=timeout).read()


def geocode(query: str) -> list[float] | None:
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "limit": 1})
    try:
        hits = json.loads(get(url))
    except Exception:  # noqa: BLE001 - a miss falls through to the next query
        return None
    finally:
        time.sleep(1.1)  # Nominatim policy: one request per second
    return [round(float(hits[0]["lat"]), 4), round(float(hits[0]["lon"]), 4)] if hits else None


def school_domain(name: str, url: str) -> str | None:
    """The school's web domain: from the course URL when it is on the
    school's own site, otherwise from the university list by name."""
    import affiliations
    import build_institutions as bi

    bi.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hipo = bi.load_hipo()
    host = (urllib.parse.urlsplit(url).hostname or "").lower().removeprefix("www.")
    labels = host.split(".")
    for i in range(len(labels) - 1):
        if ".".join(labels[i:]) in hipo:
            return ".".join(labels[i:])
    matcher = affiliations.Matcher(list({id(u): u for u in hipo.values()}.values()))
    return matcher.match(name)


def write_output(**kv: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    lines = []
    for k, v in kv.items():
        v = str(v).replace("\n", " ")
        lines.append(f"{k}={v}")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body-file", type=Path)
    ap.add_argument("--issue", default="")
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="print the entry, change nothing")
    args = ap.parse_args()

    body = args.body_file.read_text(encoding="utf-8") if args.body_file else os.environ.get("ISSUE_BODY", "")
    f = parse_form(body)
    missing = [FIELDS[k] for k in ("institution", "location", "course") if not f.get(k)]
    if missing:
        write_output(status="needs-info", message="Missing: " + ", ".join(missing) + ".")
        return 0
    if not ticked(f.get("listing", ""), "List this course"):
        write_output(status="needs-info",
                     message="The listing box is not ticked, so the course was not added.")
        return 0
    url = f.get("url", "")
    if url and not re.match(r"^https?://[^\s<>\"]+$", url):
        write_output(status="needs-info", message="The course link is not a web address.")
        return 0

    current = ba.teaching()
    if url and any(c.get("url") == url for s in current for c in s["courses"]):
        write_output(status="skip", message="That course link is already listed.")
        return 0

    city, _, country = f["location"].rpartition(",")
    city, country = (city.strip() or f["location"]), country.strip()
    domain = school_domain(f["institution"], url)
    known = next((s for s in current if domain and s["domain"] == domain), None)
    latlon = (known or {}).get("latlon") or geocode(f"{f['institution']}, {f['location']}") \
        or geocode(f["location"])
    if not domain or not latlon:
        write_output(status="needs-info",
                     message="Could not place the school automatically; a maintainer will add it by hand.")
        return 0

    course = {"title": f["course"]}
    if f.get("term"):
        course["term"] = f["term"]
    if url:
        course["url"] = url
    entry = {"name": (known or {}).get("name") or f["institution"], "domain": domain,
             "city": (known or {}).get("city") or city, "country": (known or {}).get("country") or country,
             "latlon": latlon, "kind": "course", "courses": [course]}

    block = yaml.safe_dump([entry], allow_unicode=True, sort_keys=False, width=100)
    block = "\n".join("  " + ln if ln else ln for ln in block.rstrip("\n").split("\n"))
    logo_note = "no logo (monogram)"
    if args.dry_run:
        print(block)
    else:
        with ba.DATA.open("a", encoding="utf-8") as fh:
            fh.write(f"\n  # via issue #{args.issue}\n{block}\n" if args.issue else f"\n{block}\n")
        logo = ba.LOGO_DIR / f"{domain}.png"
        if ticked(f.get("listing", ""), "logo") and not logo.exists():
            # The icon service answers for www.<domain> more often than the bare
            # domain; a miss on both leaves the monogram.
            for site in (f"www.{domain}", domain):
                try:
                    data = get("https://t1.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON"
                               f"&fallback_opts=TYPE,SIZE,URL&url=https://{site}&size=256", 20)
                except Exception:  # noqa: BLE001
                    continue
                if data[:8] == b"\x89PNG\r\n\x1a\n":
                    logo.write_bytes(data)
                    logo_note = f"site icon, {ba.png_width(logo)} px" + \
                        ("" if ba.png_width(logo) >= ba.MIN_LOGO_PX else " (too small, monogram shown)")
                    break
        elif logo.exists():
            logo_note = "existing logo"

    title = f"Add course: {f['course']} ({entry['name']})"
    if args.summary:
        args.summary.write_text(
            f"Adds a course submitted in #{args.issue}.\n\n"
            f"| | |\n|---|---|\n"
            f"| School | {entry['name']} (`{domain}`){' (already listed)' if known else ''} |\n"
            f"| Where | {entry['city']}, {entry['country']} ({latlon[0]}, {latlon[1]}) |\n"
            f"| Course | {f['course']}{' · ' + f['term'] if f.get('term') else ''} |\n"
            f"| Link | {url or '(none given)'} |\n"
            f"| Uses the book as | {f.get('usage') or '(not given)'} |\n"
            f"| Logo | {logo_note} |\n\n"
            f"**Before merging:** open the link and check that the course assigns or links the book. "
            f"Merging publishes it on the next site deploy.\n\nCloses #{args.issue}\n",
            encoding="utf-8")
    write_output(status="ok", message=f"Prepared {title}.", title=title[:120])
    return 0


if __name__ == "__main__":
    sys.exit(main())
