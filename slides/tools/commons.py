#!/usr/bin/env python3
"""Find and download Wikimedia Commons images for slides, with credits.

    commons.py search "NVIDIA H100"                 list candidate files
    commons.py get "File:Name.jpg" outdir name      download a 1600 px copy
                                                    and append its credit to
                                                    outdir/SOURCES.yml

The credit line comes from the file's own metadata (author, license), so
every downloaded image can carry \\mlsyscredit{...} on its slide.
"""

import html
import json
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "MLSysBook-slides/1.0 (lecture slides; https://mlsysbook.ai)"}


def call(params):
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return json.load(r)


def strip(s):
    s = html.unescape(re.sub(r"<[^>]+>", "", s or ""))
    return " ".join(s.split())


def short_author(a):
    """Long author lists become 'First et al.'; non-Latin text is dropped,
    since slides compile with pdflatex."""
    a = re.sub(r"[^\x00-ɏ]+", "", a).strip()
    if a.count(",") >= 3:
        a = a.split(",")[0].strip() + " et al."
    return a or "Unknown author"


def search(q, n=12):
    d = call({"action": "query", "list": "search", "srsearch": q,
              "srnamespace": 6, "srlimit": n})
    for hit in d["query"]["search"]:
        print(hit["title"])


def info(title, width=1600):
    d = call({"action": "query", "titles": title, "prop": "imageinfo",
              "iiprop": "url|size|extmetadata|mime", "iiurlwidth": width})
    page = next(iter(d["query"]["pages"].values()))
    return page["imageinfo"][0]


def get(title, outdir, name):
    ii = info(title)
    meta = ii.get("extmetadata", {})
    author = short_author(strip(meta.get("Artist", {}).get("value")))
    lic = strip(meta.get("LicenseShortName", {}).get("value")) or "see source"
    url = ii.get("thumburl") or ii["url"]
    ext = {"image/png": ".png", "image/jpeg": ".jpg"}.get(ii.get("mime"), ".jpg")
    if url.lower().endswith(".png"):
        ext = ".png"
    # A name given with an image extension keeps one extension, not two.
    name = re.sub(r"\.(jpe?g|png|gif|svg|webp)$", "", name, flags=re.I)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{name}{ext}"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        dest.write_bytes(r.read())
    kind = subprocess.run(["file", "-b", str(dest)], capture_output=True).stdout.decode("utf-8", "replace")
    page = ii.get("descriptionurl", "")
    credit = f"{author}, {lic}, Wikimedia Commons"
    entry = (f"- file: {dest.name}\n  title: \"{title}\"\n  source: {page}\n"
             f"  author: \"{author}\"\n  license: \"{lic}\"\n"
             f"  credit: \"{credit}\"\n")
    with open(out / "SOURCES.yml", "a") as f:
        f.write(entry)
    print(f"{dest}  [{kind.strip()[:40]}]\n  credit: {credit}")


if __name__ == "__main__":
    if sys.argv[1] == "search":
        search(" ".join(sys.argv[2:]))
    elif sys.argv[1] == "get":
        get(sys.argv[2], sys.argv[3], sys.argv[4])
