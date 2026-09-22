"""University affiliations of the repository's GitHub stargazers.

A stargazer's profile "company" field is free text ("Harvard University",
"@stanford", "PhD student at IIT Madras"). This module fetches every
stargazer's company and maps it to a university in the Hipo world university
list, so build_adopters.py can report where readers study or work.

Only per-school counts leave this module. Logins are used to fetch profiles
and are never written out.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import unicodedata
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

REPO = os.environ.get("GITHUB_REPOSITORY", "harvard-edge/cs249r_book")

# Abbreviations and short forms people write, mapped to a Hipo domain.
ALIAS = {
    "mit": "mit.edu", "cmu": "cmu.edu", "carnegie mellon": "cmu.edu", "stanford": "stanford.edu",
    "harvard": "harvard.edu", "berkeley": "berkeley.edu", "uc berkeley": "berkeley.edu",
    "ucb": "berkeley.edu", "ucla": "ucla.edu", "ucsd": "ucsd.edu", "uc san diego": "ucsd.edu",
    "university of california san diego": "ucsd.edu", "uiuc": "illinois.edu",
    "illinois": "illinois.edu", "georgia tech": "gatech.edu", "gatech": "gatech.edu",
    "umich": "umich.edu", "university of michigan": "umich.edu", "uw": "washington.edu",
    "university of washington": "washington.edu", "cornell": "cornell.edu",
    "princeton": "princeton.edu", "columbia": "columbia.edu", "nyu": "nyu.edu", "usc": "usc.edu",
    "ut austin": "utexas.edu", "utexas": "utexas.edu", "purdue": "purdue.edu", "yale": "yale.edu",
    "upenn": "upenn.edu", "penn": "upenn.edu", "duke": "duke.edu", "jhu": "jhu.edu",
    "johns hopkins": "jhu.edu", "northeastern": "northeastern.edu", "asu": "asu.edu",
    "umass amherst": "umass.edu", "virginia tech": "vt.edu", "uw madison": "wisc.edu",
    "uw-madison": "wisc.edu", "university of wisconsin-madison": "wisc.edu",
    "university of maryland": "umd.edu", "umd": "umd.edu", "stony brook": "stonybrook.edu",
    "rice": "rice.edu", "rice university": "rice.edu", "brown university": "brown.edu",
    "ut dallas": "utdallas.edu", "ohio state university": "osu.edu", "university at buffalo": "buffalo.edu",
    "nus": "nus.edu.sg", "ntu": "ntu.edu.sg", "ntu singapore": "ntu.edu.sg",
    "kaist": "kaist.ac.kr", "snu": "snu.ac.kr", "seoul national university": "snu.ac.kr",
    "postech": "postech.ac.kr", "tsinghua": "tsinghua.edu.cn", "thu": "tsinghua.edu.cn",
    "peking university": "pku.edu.cn", "pku": "pku.edu.cn", "sjtu": "sjtu.edu.cn",
    "shanghai jiao tong university": "sjtu.edu.cn", "zju": "zju.edu.cn",
    "zhejiang university": "zju.edu.cn", "ustc": "ustc.edu.cn", "hust": "hust.edu.cn",
    "fudan": "fudan.edu.cn", "fudan university": "fudan.edu.cn", "nju": "nju.edu.cn",
    "nanjing university": "nju.edu.cn", "sysu": "sysu.edu.cn", "buaa": "buaa.edu.cn",
    "beihang university": "buaa.edu.cn", "westlake university": "westlake.edu.cn",
    "university of chinese academy of sciences": "ucas.ac.cn",
    "hkust": "ust.hk", "hku": "hku.hk", "cuhk": "cuhk.edu.hk",
    "national taiwan university": "ntu.edu.tw", "national tsing hua university": "nthu.edu.tw",
    "eth": "ethz.ch", "eth zurich": "ethz.ch", "eth zürich": "ethz.ch", "epfl": "epfl.ch",
    "tum": "tum.de", "technical university of munich": "tum.de", "kth": "kth.se",
    "oxford": "ox.ac.uk", "university of oxford": "ox.ac.uk", "cambridge": "cam.ac.uk",
    "university of cambridge": "cam.ac.uk", "imperial college london": "imperial.ac.uk",
    "ucl": "ucl.ac.uk", "university college london": "ucl.ac.uk",
    "trinity college dublin": "tcd.ie", "university of luxembourg": "uni.lu",
    "university of hamburg": "uni-hamburg.de", "university of bonn": "uni-bonn.de",
    "politecnico di milano": "polimi.it", "polimi": "polimi.it", "tu delft": "tudelft.nl",
    "ku leuven": "kuleuven.be", "university of toronto": "utoronto.ca", "uoft": "utoronto.ca",
    "waterloo": "uwaterloo.ca", "university of waterloo": "uwaterloo.ca", "ubc": "ubc.ca",
    "mcgill": "mcgill.ca", "iisc": "iisc.ac.in", "iisc bangalore": "iisc.ac.in",
    "iiit hyderabad": "iiit.ac.in", "the university of tokyo": "u-tokyo.ac.jp",
    "university of tokyo": "u-tokyo.ac.jp", "unsw": "unsw.edu.au", "monash": "monash.edu",
    "monash university": "monash.edu", "hcmut": "hcmut.edu.vn",
    "hcmc university of technology": "hcmut.edu.vn",
    "ho chi minh university of technology": "hcmut.edu.vn", "hust vietnam": "hust.edu.vn",
    "vnu": "vnu.edu.vn", "ptit": "ptit.edu.vn", "fpt university": "fpt.edu.vn",
    "vinuniversity": "vinuni.edu.vn", "korea university": "korea.ac.kr", "cairo university": "cu.edu.eg",
    "university of indonesia": "ui.ac.id", "sliit": "sliit.lk",
}
IIT = {"bombay": "iitb.ac.in", "delhi": "iitd.ac.in", "madras": "iitm.ac.in", "kanpur": "iitk.ac.in",
       "kharagpur": "iitkgp.ac.in", "roorkee": "iitr.ac.in", "guwahati": "iitg.ac.in",
       "hyderabad": "iith.ac.in", "bhu": "iitbhu.ac.in", "gandhinagar": "iitgn.ac.in",
       "indore": "iiti.ac.in", "jodhpur": "iitj.ac.in", "patna": "iitp.ac.in", "ropar": "iitrpr.ac.in",
       "mandi": "iitmandi.ac.in", "dhanbad": "iitism.ac.in", "tirupati": "iittp.ac.in",
       "palakkad": "iitpkd.ac.in", "bhubaneswar": "iitbbs.ac.in", "jammu": "iitjammu.ac.in",
       "goa": "iitgoa.ac.in", "bhilai": "iitbhilai.ac.in", "dharwad": "iitdh.ac.in"}
IIT_SHORT = {"iitb": "bombay", "iitd": "delhi", "iitm": "madras", "iitk": "kanpur",
             "iitkgp": "kharagpur", "iitr": "roorkee", "iitg": "guwahati", "iith": "hyderabad"}
# Display fixes for the Hipo list: schools it lacks, and official names or
# country forms that read badly on a public page.
DISPLAY = {
    "ucas.ac.cn": ("University of Chinese Academy of Sciences", "China"),
    "vinuni.edu.vn": ("VinUniversity", "Vietnam"),
    "ethz.ch": ("ETH Zurich", None),
    "polimi.it": ("Politecnico di Milano", None),
    "ucl.ac.uk": ("University College London", None),
    "sjtu.edu.cn": ("Shanghai Jiao Tong University", None),
    "kuleuven.be": ("KU Leuven", None),
    "korea.ac.kr": ("Korea University", "South Korea"),
    "kaist.ac.kr": ("KAIST", None),
    "nthu.edu.tw": ("National Tsing Hua University", None),
    "uni.lu": ("University of Luxembourg", None),
    "umich.edu": ("University of Michigan", None),
    "wisc.edu": ("University of Wisconsin-Madison", None),
    "umass.edu": ("University of Massachusetts Amherst", None),
}
COUNTRY = {"Korea, Republic of": "South Korea", "Taiwan, Province of China": "Taiwan",
           "Viet Nam": "Vietnam", "Iran, Islamic Republic of": "Iran",
           "Russian Federation": "Russia", "Turkiye": "Türkiye"}

# Names too generic to identify one school ("National University" exists in a
# dozen countries). They never match on their own.
DENY = {"university of technology", "national university", "university of science",
        "university of engineering", "university of information technology", "city university",
        "international university", "global university", "university of engineering and technology",
        "institute of technology", "state university", "open university"}
ACADEMIC = re.compile(r"universi|institut|college|polytechn|school of|iit|iiit|academy|hochschule|"
                      r"technion|\.edu|\.ac\.", re.I)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).strip().lower().lstrip("@").strip()
    s = re.sub(r"^the\s+", "", s)
    s = re.sub(r"[\s,;|/·\-–—]+$", "", s)
    return re.sub(r"\s+", " ", s)


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------

def _token() -> str | None:
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        return tok
    try:
        return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _api(token: str, url: str, body: dict | None = None) -> dict | list | None:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode() if body else None,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                 "User-Agent": "mlsysbook-adopters"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except Exception:  # noqa: BLE001 - retried, then treated as a missing page
            continue
    return None


def fetch_companies(workers: int = 8) -> list[str]:
    """Company strings of every stargazer that has one."""
    token = _token()
    if not token:
        raise RuntimeError("no GitHub token (GITHUB_TOKEN or gh auth)")
    meta = _api(token, f"https://api.github.com/repos/{REPO}") or {}
    pages = (int(meta.get("stargazers_count", 0)) + 99) // 100
    if not pages:
        raise RuntimeError("could not read the stargazer count")

    def page(n: int) -> list[str]:
        rows = _api(token, f"https://api.github.com/repos/{REPO}/stargazers?per_page=100&page={n}")
        return [r["login"] for r in rows or []]

    with ThreadPoolExecutor(workers) as ex:
        logins = list(dict.fromkeys(l for ls in ex.map(page, range(1, pages + 1)) for l in ls))

    def batch(chunk: list[str]) -> list[str]:
        q = "query{" + " ".join(f'u{i}: user(login:{json.dumps(l)}){{company}}'
                                for i, l in enumerate(chunk)) + "}"
        data = (_api(token, "https://api.github.com/graphql", {"query": q}) or {}).get("data") or {}
        return [v["company"] for v in data.values() if v and v.get("company")]

    chunks = [logins[i:i + 100] for i in range(0, len(logins), 100)]
    with ThreadPoolExecutor(workers) as ex:
        companies = [c for cs in ex.map(batch, chunks) for c in cs]
    if len(logins) < 0.9 * pages * 100 - 100:
        raise RuntimeError(f"only {len(logins)} stargazers fetched of ~{pages * 100}")
    return companies


# --------------------------------------------------------------------------
# Classify
# --------------------------------------------------------------------------

class Matcher:
    def __init__(self, hipo: list[dict]):
        self.by_domain: dict[str, dict] = {}
        self.by_name: dict[str, dict] = {}
        for u in hipo:
            for d in u.get("domains", []):
                self.by_domain.setdefault(d.lower(), u)
            self.by_name.setdefault(norm(u["name"]), u)
        for d in DENY:
            self.by_name.pop(d, None)
        names = sorted(self.by_name)
        joined = " | ".join(names)
        # A name contained in several longer names ("Kyoto University" inside
        # "Kyoto University of Education") is too ambiguous for substring use.
        self.long_names = [n for n in names if len(n) >= 14 and joined.count(n) < 4]

    def _domain_of(self, c: str) -> str | None:
        if c in ALIAS:
            return ALIAS[c]
        if c in self.by_name:
            return self.by_name[c]["domains"][0].lower()
        if c in self.by_domain:
            return c
        m = re.match(r"^(?:indian institute of technology|iit)[ ,-]*(.+)$", c)
        if m and m.group(1).strip() in IIT:
            return IIT[m.group(1).strip()]
        if c in IIT_SHORT:
            return IIT[IIT_SHORT[c]]
        c2 = re.sub(r"\s*\(.*?\)", "", c)
        c2 = re.sub(r"^(phd|ms|msc|student|researcher|professor|prof\.?)\s+(at|@)\s+", "", c2)
        if c2 in ALIAS:
            return ALIAS[c2]
        if c2 in self.by_name:
            return self.by_name[c2]["domains"][0].lower()
        return None

    def match(self, company: str) -> str | None:
        """Hipo domain of the school a company string names, or None."""
        parts = [company] + re.split(r"[|/&;]| @", company) + re.split(r"[,|/&;]| @", company)
        for part in parts:
            c = norm(part)
            if len(c) < 2:
                continue
            dom = self._domain_of(c)
            if dom:
                return dom
            if ACADEMIC.search(c) and len(c) >= 12:
                best = max((n for n in self.long_names if n in c), key=len, default=None)
                if best:
                    return self.by_name[best]["domains"][0].lower()
        return None

    def school(self, domain: str) -> dict:
        u = self.by_domain.get(domain, {})
        name, country = DISPLAY.get(domain, (None, None))
        country = country or u.get("country", "")
        return {"domain": domain, "name": name or u.get("name", domain),
                "country": COUNTRY.get(country, country)}


def count_schools(companies: list[str], matcher: Matcher) -> Counter:
    return Counter(d for d in (matcher.match(c) for c in companies) if d)
