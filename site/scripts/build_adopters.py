#!/usr/bin/env python3
"""Where the book is taught and read, published as adopters.json.

Two sources, two lists:

  teaching  community/adopters.yml, curated. A school is listed when a public
            course page, syllabus or course repository assigns the book, or
            its instructor is named in the Vol. I acknowledgments.
  readers   inferred, refreshed on a schedule. Schools named in GitHub
            stargazers' profile affiliations (scripts/affiliations.py) and,
            when GA4 credentials are present, in referrer hostnames such as
            canvas.<school>.edu (scripts/build_institutions.py). A school is
            listed at MIN_READERS or more.

Usage:
  build_adopters.py gather    refresh the reader list into config/adopters-cache.json
                              (network; run by the refresh workflow, not by renders)
  build_adopters.py json OUT  write the published JSON alone (the refresh workflow)
  build_adopters.py           post-render: write _build/adopters.json and embed the
                              same data in the landing ticker, community/courses.html
                              (the course directory) and community/index.html (readers)

The pages render from that JSON in the browser: first from the copy embedded
at build time, then from /adopters.json fetched on load. Refreshing the one
file on gh-pages therefore updates every visitor without a rebuild, the same
arrangement stats.json uses.

The map's land is drawn here once, as a static SVG path in an Equal Earth
projection (world-atlas 110m, Natural Earth, public domain). The projection
constants ride along on the SVG so the browser can place pins with the same
formula.
"""

from __future__ import annotations

import json
import math
import struct
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

SITE_DIR = Path(__file__).resolve().parent.parent
DATA = SITE_DIR / "community" / "adopters.yml"
CACHE = SITE_DIR / "config" / "adopters-cache.json"
GEO_CACHE = SITE_DIR / "config" / "adopters-geo.json"
WORLD = SITE_DIR / "scripts" / "data" / "countries-110m.json"
LOGO_DIR = SITE_DIR / "community" / "assets" / "images" / "adopters"
BUILD = SITE_DIR / "_build"

MIN_LOGO_PX = 40   # smaller site icons read as blur at 44px; use a monogram
MIN_READERS = 3    # public threshold for an inferred school
MAP_W, MAP_H = 1000, 470
UA = "mlsysbook-adopters/1.0 (https://mlsysbook.ai)"


def log(msg: str) -> None:
    print(f"  {msg}", file=sys.stderr)


# --------------------------------------------------------------------------
# Teaching list (curated)
# --------------------------------------------------------------------------

def png_width(path: Path) -> int:
    with path.open("rb") as fh:
        head = fh.read(24)
    return struct.unpack(">I", head[16:20])[0] if head[:8] == b"\x89PNG\r\n\x1a\n" else 0


def monogram(name: str, short: str | None) -> str:
    if short:
        return short[:4]
    skip = {"of", "the", "and", "at", "de", "la", "di", "e", "in"}
    words = [w for w in name.replace(",", "").split() if w.lower() not in skip]
    return "".join(w[0] for w in words[:3]).upper()


def teaching() -> list[dict]:
    out = []
    for s in yaml.safe_load(DATA.read_text(encoding="utf-8"))["schools"]:
        logo = LOGO_DIR / f"{s['domain']}.png"
        out.append({
            "name": s["name"], "short": s.get("short"), "domain": s["domain"],
            "city": s["city"], "country": s["country"], "latlon": s["latlon"],
            "kind": s["kind"], "acknowledged": bool(s.get("acknowledged")) or s["kind"] == "acknowledged",
            "logo": (f"assets/images/adopters/{s['domain']}.png"
                     if logo.exists() and png_width(logo) >= MIN_LOGO_PX else None),
            "mono": monogram(s["name"], s.get("short")),
            "courses": [{"title": c["title"], "term": str(c["term"]) if c.get("term") else None,
                         "url": c.get("url")} for c in s.get("courses", [])],
        })
    return out


# --------------------------------------------------------------------------
# Reader list (inferred, gathered on a schedule)
# --------------------------------------------------------------------------

def geocode(query: str, cache: dict) -> list[float] | None:
    if query in cache:
        return cache[query]
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "limit": 1})
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        hits = json.loads(urllib.request.urlopen(req, timeout=30).read())
    except Exception as exc:  # noqa: BLE001 - a miss is retried on the next run
        log(f"! geocode failed for {query!r}: {exc}")
        return None
    time.sleep(1.1)  # Nominatim policy: one request per second
    cache[query] = [round(float(hits[0]["lat"]), 4), round(float(hits[0]["lon"]), 4)] if hits else None
    return cache[query]


def gather() -> int:
    sys.path.insert(0, str(Path(__file__).parent))
    import affiliations
    import build_institutions as bi

    bi.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hipo_by_domain = bi.load_hipo()
    matcher = affiliations.Matcher(list({id(u): u for u in hipo_by_domain.values()}.values()))
    signal: dict[str, int] = {}

    try:
        companies = affiliations.fetch_companies()
        for dom, n in affiliations.count_schools(companies, matcher).items():
            signal[dom] = signal.get(dom, 0) + n
        log(f"stargazers: {len(companies):,} affiliations, {len(signal)} schools")
    except Exception as exc:  # noqa: BLE001
        log(f"! stargazer affiliations unavailable ({exc}); keeping cached readers")
        return 0

    try:
        rows = bi.fetch_ga4_rows()
        slugs: dict[str, list[str]] = {}
        for d in hipo_by_domain:
            slugs.setdefault(d.split(".")[0], []).append(d)
        for row in rows:
            hit = bi.classify(row["host"], hipo_by_domain, slugs)
            if hit and hit["domain"] in hipo_by_domain:
                signal[hit["domain"]] = signal.get(hit["domain"], 0) + row["users"]
        log(f"GA4 referrers: {len(rows):,} rows merged")
    except Exception as exc:  # noqa: BLE001 - credentials are optional
        log(f"GA4 referrers skipped ({type(exc).__name__})")

    # The Hipo list can file a school under a different domain than
    # adopters.yml does (iisc.ernet.in vs iisc.ac.in), so match names too.
    taught = {s["domain"] for s in teaching()}
    taught_names = {affiliations.norm(s[k]) for s in teaching() for k in ("name", "short") if s.get(k)}
    geo = json.loads(GEO_CACHE.read_text(encoding="utf-8")) if GEO_CACHE.exists() else {}
    readers, used = [], {}
    for dom, n in sorted(signal.items(), key=lambda kv: -kv[1]):
        if n < MIN_READERS or dom in taught:
            continue
        s = matcher.school(dom)
        if affiliations.norm(s["name"]) in taught_names:
            continue
        key = f"{s['name']}, {s['country']}"
        latlon = geocode(key, geo)
        used[key] = latlon
        if not latlon:
            continue
        readers.append({**s, "latlon": latlon})
    # Keep only lookups still in use, so the committed file tracks the list.
    GEO_CACHE.write_text(json.dumps(used, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
                         encoding="utf-8")
    CACHE.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "readers": readers,
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"adopters cache: {len(readers)} reader schools at >= {MIN_READERS}")
    return 0


# --------------------------------------------------------------------------
# Publish (post-render)
# --------------------------------------------------------------------------

def payload() -> dict:
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    t = teaching()
    readers = cache.get("readers", [])
    return {
        "generated": cache.get("generated", ""),
        "counts": {
            "teaching": len(t),
            "countries": len({s["country"] for s in t}),
            "courses": sum(len(s["courses"]) for s in t if s["kind"] != "acknowledged"),
            "readers": len(readers),
            "reader_countries": len({s["country"] for s in t + readers}),
        },
        "teaching": t,
        "readers": readers,
    }


def equal_earth(lon: float, lat: float) -> tuple[float, float]:
    a1, a2, a3, a4 = 1.340264, -0.081106, 0.000893, 0.003796
    m = math.sqrt(3) / 2
    t = math.asin(m * math.sin(math.radians(lat)))
    t2, t6 = t * t, t ** 6
    x = math.radians(lon) * math.cos(t) / (m * (a1 + 3 * a2 * t2 + t6 * (7 * a3 + 9 * a4 * t2)))
    return x, t * (a1 + a2 * t2 + t6 * (a3 + a4 * t2))


def land_svg() -> str:
    topo = json.loads(WORLD.read_text(encoding="utf-8"))
    (sx, sy), (tx, ty) = topo["transform"]["scale"], topo["transform"]["translate"]
    arcs = []
    for arc in topo["arcs"]:
        x = y = 0
        pts = []
        for dx, dy in arc:
            x, y = x + dx, y + dy
            pts.append((x * sx + tx, y * sy + ty))
        arcs.append(pts)
    rings = []
    for geom in topo["objects"]["countries"]["geometries"]:
        if geom.get("id") == "010":  # Antarctica: no schools, and it eats height
            continue
        for poly in (geom["arcs"] if geom["type"] == "MultiPolygon" else [geom["arcs"]]):
            for idx in poly:
                pts: list[tuple[float, float]] = []
                for i in idx:
                    seg = arcs[i] if i >= 0 else arcs[~i][::-1]
                    pts.extend(seg[1:] if pts else seg)
                rings.append(pts)

    xs, ys = zip(*(equal_earth(*p) for r in rings for p in r))
    pad = 12
    k = min((MAP_W - 2 * pad) / (max(xs) - min(xs)), (MAP_H - 2 * pad) / (max(ys) - min(ys)))
    ox = (MAP_W - (max(xs) - min(xs)) * k) / 2 - min(xs) * k
    oy = (MAP_H - (max(ys) - min(ys)) * k) / 2 + max(ys) * k

    d = []
    for r in rings:
        prev = None
        for lon, lat in r:
            x, y = equal_earth(lon, lat)
            px, py = ox + x * k, oy - y * k
            # Lift the pen where a ring wraps the antimeridian.
            d.append(("M" if prev is None or abs(px - prev) > MAP_W / 2 else "L") + f"{px:.1f},{py:.1f}")
            prev = px
        d.append("Z")
    return (f'<svg class="adp-map" viewBox="0 0 {MAP_W} {MAP_H}" role="img" '
            f'aria-label="World map of the schools whose courses are listed" '
            f'data-k="{k:.4f}" data-ox="{ox:.4f}" data-oy="{oy:.4f}">'
            f'<path class="adp-land" d="{"".join(d)}"/><g class="adp-pins"></g></svg>')


def inject(path: Path, marker: str, fragment: str) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if marker not in text:
        if 'id="adp-data"' not in text:
            log(f"! {marker} not found in {path.relative_to(SITE_DIR)}")
        return False
    path.write_text(text.replace(marker, fragment), encoding="utf-8")
    return True


def publish() -> int:
    data = payload()
    BUILD.mkdir(exist_ok=True)
    (BUILD / "adopters.json").write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
    # "</" cannot appear inside an inline script; escape it the JSON-safe way.
    embedded = ('<script type="application/json" id="adp-data">'
                + json.dumps(data, ensure_ascii=False).replace("</", "<\\/") + "</script>")
    pages = [(BUILD / "index.html", "<!-- adopters:strip -->", embedded),
             (BUILD / "community" / "courses.html", "<!-- adopters:courses -->",
              embedded + "\n" + land_svg()),
             (BUILD / "community" / "index.html", "<!-- adopters:readers -->", embedded)]
    done = sum(inject(p, m, f) for p, m, f in pages)
    c = data["counts"]
    log(f"adopters: {c['teaching']} teaching in {c['countries']} countries, "
        f"{c['readers']} reader schools -> adopters.json + {done} page(s)")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["gather"]:
        sys.exit(gather())
    if len(args) == 2 and args[0] == "json":
        Path(args[1]).write_text(json.dumps(payload(), ensure_ascii=False) + "\n", encoding="utf-8")
        sys.exit(0)
    sys.exit(publish())
