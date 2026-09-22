#!/usr/bin/env python3
"""Estimate which universities read the book, from GA4 referrer data.

GA4 never exposes visitor IP addresses or network owners, so the institution
has to be inferred from something GA4 does keep. The strongest signal is the
referrer. Courses link to the book from their LMS or course pages, and those
hostnames name the school (canvas.harvard.edu, moodle.tum.de,
bcourses.berkeley.edu, <slug>.instructure.com). This script:

  1. fetch     - pulls (referrer host, city, country, users) rows from the GA4
                 Data API, or reads a CSV exported from the GA4 UI;
  2. classify  - matches each host against the Hipo world university domain
                 list, with fallbacks for hosted LMS slugs and bare academic
                 TLDs;
  3. locate    - places each school with OpenStreetMap Nominatim (cached,
                 1 req/s), falling back to the city GA4 saw most often;
  4. emit      - writes institutions.json plus a self-contained preview page
                 (world map + icon grid).

The result is a rough proxy, and it errs low. It counts only schools whose
readers arrived by a link, so a class that hands out the URL on a slide is
invisible.

Credentials follow build_stats.py: GOOGLE_APPLICATION_CREDENTIALS pointing at a
key file, or GA4_SERVICE_ACCOUNT_JSON holding the key itself.

Usage:
  python3 site/scripts/build_institutions.py                  # live GA4
  python3 site/scripts/build_institutions.py --csv export.csv # GA4 UI export
  python3 site/scripts/build_institutions.py --offline        # reuse cached rows
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

SITE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = SITE_DIR / ".cache" / "institutions"
GA4_PROPERTY_ID = os.environ.get("GA4_PROPERTY_ID", "419684623")

HIPO_URL = ("https://raw.githubusercontent.com/Hipo/university-domains-list/"
            "master/world_universities_and_domains.json")
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
FAVICON_URL = "https://www.google.com/s2/favicons?sz=64&domain={}"
USER_AGENT = "mlsysbook-institutions/0.1 (https://mlsysbook.ai)"

# Hosts that carry no school identity even when they are academic in flavor.
IGNORED_HOSTS = {"mlsysbook.ai", "localhost", "(direct)", "(not set)", ""}

# Shared LMS / course platforms where the school lives in the first label.
HOSTED_LMS = {
    "instructure.com": "Canvas",
    "brightspace.com": "Brightspace",
    "blackboard.com": "Blackboard",
    "moodlecloud.com": "Moodle",
    "desire2learn.com": "Brightspace",
}

# First labels that say "this is a course system", used to tag the channel.
LMS_LABELS = {
    "canvas": "Canvas", "bcourses": "Canvas", "moodle": "Moodle",
    "blackboard": "Blackboard", "bb": "Blackboard", "learn": "LMS",
    "brightspace": "Brightspace", "d2l": "Brightspace", "elearning": "LMS",
    "lms": "LMS", "courses": "Course site", "courseworks": "Canvas",
    "sakai": "Sakai", "classes": "Course site", "sites": "Course site",
}

# Registrable suffixes that are academic on their own.
ACADEMIC_SUFFIX = re.compile(r"(^|\.)(edu|ac\.[a-z]{2}|edu\.[a-z]{2})$")


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def http_get(url: str, *, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# --------------------------------------------------------------------------
# 1. fetch
# --------------------------------------------------------------------------

def fetch_ga4_rows() -> list[dict]:
    """(host, city, country, users) rows from two GA4 referrer dimensions.

    pageReferrer is the full referring URL on each event; sessionSource is the
    referrer host of the session. They overlap heavily, so the two are merged
    per (host, city, country) by taking the larger user count rather than the
    sum, which would double count.
    """
    if os.environ.get("GA4_SERVICE_ACCOUNT_JSON") and not os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS"
    ):
        key_path = CACHE_DIR / ".ga4-key.json"
        key_path.write_text(os.environ["GA4_SERVICE_ACCOUNT_JSON"], encoding="utf-8")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)

    from google.analytics.data_v1beta import BetaAnalyticsDataClient
    from google.analytics.data_v1beta.types import (
        DateRange, Dimension, Metric, RunReportRequest,
    )

    client = BetaAnalyticsDataClient()
    merged: dict[tuple, int] = {}
    for dim in ("pageReferrer", "sessionSource"):
        offset, page = 0, 100_000
        while True:
            resp = client.run_report(RunReportRequest(
                property=f"properties/{GA4_PROPERTY_ID}",
                date_ranges=[DateRange(start_date="2015-08-14", end_date="today")],
                dimensions=[Dimension(name=dim), Dimension(name="city"),
                            Dimension(name="country")],
                metrics=[Metric(name="totalUsers")],
                limit=page, offset=offset,
            ))
            for r in resp.rows:
                host = to_host(r.dimension_values[0].value)
                key = (host, r.dimension_values[1].value, r.dimension_values[2].value)
                users = int(r.metric_values[0].value)
                merged[key] = max(merged.get(key, 0), users)
            log(f"  GA4 {dim}: {offset + len(resp.rows):,} / {resp.row_count:,} rows")
            offset += len(resp.rows)
            if not resp.rows or offset >= resp.row_count:
                break
    return [{"host": h, "city": c, "country": k, "users": u}
            for (h, c, k), u in merged.items()]


def read_csv_rows(path: Path) -> list[dict]:
    """Rows from a GA4 Exploration export: referrer, city, country, users.

    Column names are matched loosely so either "Page referrer" or "Session
    source" exports work, and GA4's comment header lines are skipped.
    """
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if ln and not ln.startswith("#")]
    rows = []
    for rec in csv.DictReader(lines):
        norm = {k.strip().lower(): v for k, v in rec.items() if k}
        ref = next((v for k, v in norm.items() if "referrer" in k or "source" in k), "")
        users = next((v for k, v in norm.items() if "user" in k), "0")
        rows.append({
            "host": to_host(ref),
            "city": norm.get("city", "(not set)"),
            "country": norm.get("country", "(not set)"),
            "users": int(float(users.replace(",", "") or 0)),
        })
    return rows


def to_host(value: str) -> str:
    value = value.strip().lower()
    if "://" in value:
        value = urllib.parse.urlsplit(value).hostname or ""
    value = value.split("/")[0].split(":")[0]
    return value.removeprefix("www.")


# --------------------------------------------------------------------------
# 2. classify
# --------------------------------------------------------------------------

def load_hipo() -> dict[str, dict]:
    path = CACHE_DIR / "hipo.json"
    if not path.exists():
        log("  downloading Hipo university domain list")
        path.write_bytes(http_get(HIPO_URL, timeout=60))
    by_domain: dict[str, dict] = {}
    for uni in json.loads(path.read_text(encoding="utf-8")):
        for dom in uni.get("domains", []):
            by_domain.setdefault(dom.lower().removeprefix("www."), uni)
    return by_domain


def classify(host: str, hipo: dict[str, dict],
             slug_index: dict[str, list[str]]) -> dict | None:
    """Map a referrer host to {domain, name, country, channel}, or None."""
    if host in IGNORED_HOSTS or host.endswith("mlsysbook.ai"):
        return None
    labels = host.split(".")
    channel = LMS_LABELS.get(labels[0], "Link")

    # Hosted LMS: harvard.instructure.com -> the school whose domain starts "harvard."
    for platform, name in HOSTED_LMS.items():
        if host.endswith("." + platform):
            slug = host[: -len(platform) - 1].split(".")[-1]
            candidates = slug_index.get(slug, [])
            if len(candidates) == 1:
                uni = hipo[candidates[0]]
                return {"domain": candidates[0], "name": uni["name"],
                        "country": uni.get("country", ""), "channel": name}
            return None

    # Longest matching suffix in the university list.
    for i in range(len(labels) - 1):
        dom = ".".join(labels[i:])
        if dom in hipo:
            uni = hipo[dom]
            return {"domain": dom, "name": uni["name"],
                    "country": uni.get("country", ""), "channel": channel}

    # Academic TLD with no list entry: keep it, named by its domain.
    m = ACADEMIC_SUFFIX.search(host)
    if m:
        keep = 3 if m.group(2).count(".") else 2
        dom = ".".join(labels[-keep:])
        return {"domain": dom, "name": dom, "country": "", "channel": channel}
    return None


# --------------------------------------------------------------------------
# 3. locate + icons
# --------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def geocode(query: str, cache: dict) -> list[float] | None:
    if query in cache:
        return cache[query]
    url = NOMINATIM_URL + "?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "limit": 1})
    try:
        hits = json.loads(http_get(url))
        cache[query] = [float(hits[0]["lat"]), float(hits[0]["lon"])] if hits else None
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        log(f"  ! geocode failed for {query!r}: {exc}")
        return None
    time.sleep(1.1)  # Nominatim usage policy: at most one request per second.
    return cache[query]


def favicon(domain: str, cache: dict) -> str:
    if domain not in cache:
        try:
            data = http_get(FAVICON_URL.format(domain), timeout=15)
            cache[domain] = "data:image/png;base64," + base64.b64encode(data).decode()
        except (urllib.error.URLError, TimeoutError):
            cache[domain] = ""
    return cache[domain]


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def build(rows: list[dict], min_users: int) -> list[dict]:
    hipo = load_hipo()
    slug_index: dict[str, list[str]] = defaultdict(list)
    for dom in hipo:
        slug_index[dom.split(".")[0]].append(dom)

    schools: dict[str, dict] = {}
    for row in rows:
        hit = classify(row["host"], hipo, slug_index)
        if not hit:
            continue
        s = schools.setdefault(hit["domain"], {
            **{k: hit[k] for k in ("domain", "name", "country")},
            "users": 0, "hosts": Counter(), "channels": Counter(), "cities": Counter(),
        })
        s["users"] += row["users"]
        s["hosts"][row["host"]] += row["users"]
        s["channels"][hit["channel"]] += row["users"]
        if row["city"] not in ("(not set)", ""):
            s["cities"][(row["city"], row["country"])] += row["users"]
        if not s["country"] and row["country"] != "(not set)":
            s["country"] = row["country"]

    geo_path, icon_path = CACHE_DIR / "geocode.json", CACHE_DIR / "favicons.json"
    geo_cache, icon_cache = load_json(geo_path), load_json(icon_path)
    out = []
    kept = [s for s in schools.values() if s["users"] >= min_users]
    log(f"  {len(schools)} institutions matched, {len(kept)} with >= {min_users} users")
    for s in sorted(kept, key=lambda s: -s["users"]):
        top_city = s["cities"].most_common(1)[0][0] if s["cities"] else None
        latlon = geocode(f"{s['name']}, {s['country']}", geo_cache)
        if not latlon and top_city:
            latlon = geocode(f"{top_city[0]}, {top_city[1]}", geo_cache)
        out.append({
            "name": s["name"], "domain": s["domain"], "country": s["country"],
            "users": s["users"], "latlon": latlon,
            "city": top_city[0] if top_city else "",
            "channel": s["channels"].most_common(1)[0][0],
            "hosts": [h for h, _ in s["hosts"].most_common(5)],
            "icon": favicon(s["domain"], icon_cache),
        })
    geo_path.write_text(json.dumps(geo_cache, indent=1), encoding="utf-8")
    icon_path.write_text(json.dumps(icon_cache), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--csv", type=Path, help="GA4 Exploration export instead of the API")
    src.add_argument("--offline", action="store_true", help="reuse cached GA4 rows")
    ap.add_argument("--min-users", type=int, default=3,
                    help="drop institutions below this many users (default 3)")
    ap.add_argument("--out", type=Path, default=CACHE_DIR / "institutions.json")
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = CACHE_DIR / "ga4-rows.json"
    if args.csv:
        rows = read_csv_rows(args.csv)
    elif args.offline:
        rows = json.loads(rows_path.read_text(encoding="utf-8"))
    else:
        rows = fetch_ga4_rows()
    rows_path.write_text(json.dumps(rows), encoding="utf-8")
    log(f"  {len(rows):,} referrer rows, {sum(r['users'] for r in rows):,} users")

    schools = build(rows, args.min_users)
    args.out.write_text(json.dumps(schools, indent=1), encoding="utf-8")
    log(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
