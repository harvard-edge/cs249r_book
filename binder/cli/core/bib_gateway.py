"""Native Reference Gateway: Authenticated citation resolution, formatting, and ledger sealing.

Provides programmatic and CLI facilities to safely ingest, verify, and seal
new bibliography entries without direct, unvalidated file edits to bibs/references.bib.
"""
from __future__ import annotations

import hashlib
import json
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import certifi

from cli.checks import bib_lint

# SSL Contexts: Primary with certifi roots, fallback with permissive TLS for corporate proxies
SSL_CTX = ssl.create_default_context(cafile=certifi.where())
FALLBACK_SSL_CTX = ssl.create_default_context()
FALLBACK_SSL_CTX.check_hostname = False
FALLBACK_SSL_CTX.verify_mode = ssl.CERT_NONE

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
# 2026-10-13: some record sites (OSTI) reset connections that present a browser
# user agent from a script; fetch_url retries with this plain tool identifier.
TOOL_UA = "mlsysbook-binder/1.0 (+https://mlsysbook.ai)"
ACADEMIC_UA = "MLSysBook-Academic-Gateway/1.0 (mailto:vj@eecs.harvard.edu)"

DOI_REGEX = re.compile(r"\b(10\.[0-9]{4,9}/[^\s\"<>{}]+)", re.I)
ARXIV_REGEX = re.compile(r"(?:arxiv[:\s/]|abs/|pdf/)?([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)", re.I)

TRUSTED_DOMAINS = {
    "github.com", "huggingface.co", "arxiv.org", "openai.com", "anthropic.com",
    "deepmind.google", "google.com", "microsoft.com", "apple.com", "meta.com",
    "iso.org", "iec.ch", "sae.org", "nist.gov", "europa.eu", "un.org",
    "acm.org", "ieee.org", "usenix.org", "openreview.net", "mlsys.org",
    "neurips.cc", "icml.cc", "iclr.cc", "cvpr.thecvf.com", "roboticsproceedings.org",
    "proceedings.mlr.press", "ec.europa.eu", "dmv.ca.gov", "exponent.com",
    "assets.ctfassets.net", "york.ac.uk", "mit.edu", "stanford.edu", "berkeley.edu",
    "cmu.edu", "harvard.edu", "nature.com", "science.org", "theiet.org",
    "arm.com", "developer.arm.com", "nhtsa.gov", "esa.int", "itu.int",
    "broadcom.com", "nvidia.com", "whitehouse.gov", "sec.gov", "theguardian.com",
    "nytimes.com", "reuters.com", "press.roberthalf.com", "oreilly.com", "shop.oreilly.com",
    "schneider-electric.com", "agilityrobotics.com", "bosch-sensortec.com", "zltech.com",
    "pewresearch.org", "thegreengrid.org", "trustedcomputinggroup.org", "states-language.net",
    "fairmlbook.org", "consumerfinance.gov", "cigionline.org", "uptimeinstitute.com",
    "canada.ca", "pennelynn.com", "uber.com", "hhs.gov", "ilga.gov", "ibm.com",
    "intel.com", "karpathy.medium.com", "nostarch.com", "jfrog.com", "vkrakovna.wordpress.com",
    "toronto.edu", "mckinsey.com", "erights.org", "pragprog.com", "nrel.gov",
    "citeseerx.ist.psu.edu", "kernel.org", "se.com", "thron.com", "planning.wiki",
    "safetyresearch.net", "edn.com", "bloomberg.com", "dell.com", "deepseek.com",
    "discord.com", "deepspeed.ai", "readthedocs.io", "theatlantic.com", "amazon.com",
    "aws.amazon.com", "ti.com", "sciencedirect.com", "springer.com", "wiley.com",
    "mitpress.mit.edu", "kernelnewbies.org", "ai-benchmark.com", "dabeaz.com",
    "elie.net", "berkeleyvision.org", "legislature.ca.gov", "cftc.gov",
    "uni-kassel.de", "dlsyscourse.org", "gov.cn", "computerhistory.org",
    "circleci.com", "cloudflare.com", "cnbc.com", "tensorflow.org", "google.dev"
}


def compute_entry_hash(raw_text: str) -> str:
    """Compute canonical SHA-256 hash of normalized entry text."""
    normalized = re.sub(r"\s+", " ", raw_text.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def clean_str(s: str) -> str:
    """Remove LaTeX braces, quotes, and whitespace for fuzzy matching."""
    s = re.sub(r"[{}\\\'\"\`]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def string_similarity(a: str, b: str) -> float:
    """Normalized string similarity between 0.0 and 1.0."""
    return SequenceMatcher(None, clean_str(a).lower(), clean_str(b).lower()).ratio()


def derive_citekey(author: str, year: str | int, title: str) -> str:
    """Derive standard textbook citekey: <first_author_surname><year><first_distinctive_title_word>."""
    surname = ""
    if author:
        first_author = author.split(" and ")[0].strip()
        if "," in first_author:
            surname = first_author.split(",")[0].strip()
        else:
            surname = first_author.split()[-1].strip()
    surname = re.sub(r"[^a-zA-Z]", "", surname).lower()
    if not surname:
        surname = "ref"

    yr = str(year).strip() if year else "2026"
    m_yr = re.search(r"\b(19\d\d|20\d\d)\b", yr)
    yr_str = m_yr.group(1) if m_yr else "2026"

    stopwords = {
        "a", "an", "the", "on", "in", "for", "and", "of", "to", "with", "is", "at", "by",
        "from", "as", "via", "into", "towards", "using", "how", "why", "what", "which"
    }
    words = re.findall(r"[a-zA-Z]{3,}", title)
    kw = ""
    for w in words:
        w_low = w.lower()
        if w_low not in stopwords:
            kw = w_low
            break
    if not kw:
        kw = "work"
    return f"{surname}{yr_str}{kw}"


def format_bibtex_entry(entry_type: str, key: str, fields: dict[str, Any]) -> str:
    """Format BibTeX entry according to textbook conventions (2-space indent, braces)."""
    # Canonical field ordering
    field_order = [
        "title", "author", "year", "journal", "booktitle", "publisher",
        "institution", "volume", "number", "pages", "doi", "url",
        "isbn", "howpublished", "note"
    ]
    ordered_keys = [k for k in field_order if k in fields and fields[k]]
    # Add any remaining fields not in standard order
    for k in sorted(fields.keys()):
        if k not in ordered_keys and fields[k]:
            ordered_keys.append(k)

    lines = [f"@{entry_type}{{{key},"]
    for k in ordered_keys:
        val = str(fields[k]).strip()
        if k == "pages":
            val = re.sub(r"(\d+)-(\d+)", r"\1--\2", val)
        lines.append(f"  {k} = {{{val}}},")
    lines.append("}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# External Registry Resolution
# ----------------------------------------------------------------------

def fetch_doi(doi: str, custom_key: str | None = None) -> dict[str, Any] | None:
    """Authenticate and fetch metadata for a DOI via Crossref (fallback DataCite)."""
    clean_doi = doi.strip().replace("https://doi.org/", "").replace("http://doi.org/", "").rstrip("/.,")
    url = f"https://api.crossref.org/works/{urllib.parse.quote(clean_doi)}"
    req = urllib.request.Request(url, headers={"User-Agent": ACADEMIC_UA})

    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=10, context=SSL_CTX) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8")).get("message", {})
                    titles = data.get("title", [""])
                    title = titles[0] if titles else ""
                    if not title:
                        return None

                    # Authors
                    author_list = []
                    for a in data.get("author", []):
                        fam = a.get("family", "").strip()
                        giv = a.get("given", "").strip()
                        if fam and giv:
                            author_list.append(f"{fam}, {giv}")
                        elif fam:
                            author_list.append(fam)
                        elif a.get("name"):
                            author_list.append(f"{{{a.get('name')}}}")
                    author_str = " and ".join(author_list)

                    # Year
                    published = data.get("published-print") or data.get("published-online") or data.get("created", {})
                    parts = published.get("date-parts", [[None]])[0]
                    year = str(parts[0]) if parts and parts[0] else ""

                    # Venue & Entry Type
                    cr_type = data.get("type", "")
                    container = (data.get("container-title") or [""])[0]
                    publisher = data.get("publisher", "")

                    if cr_type == "proceedings-article":
                        entry_type = "inproceedings"
                        venue_field = "booktitle"
                    elif cr_type in ("book", "monograph"):
                        entry_type = "book"
                        venue_field = "publisher"
                    else:
                        entry_type = "article"
                        venue_field = "journal"

                    fields: dict[str, Any] = {
                        "title": title,
                        "author": author_str,
                        "year": year,
                        venue_field: container or publisher,
                        "doi": clean_doi,
                        "url": f"https://doi.org/{clean_doi}",
                    }
                    if data.get("volume"):
                        fields["volume"] = str(data["volume"])
                    if data.get("issue"):
                        fields["number"] = str(data["issue"])
                    if data.get("page"):
                        fields["pages"] = str(data["page"])
                    if publisher and entry_type != "book":
                        fields["publisher"] = publisher

                    key = custom_key or derive_citekey(author_str, year, title)
                    return {
                        "key": key,
                        "entry_type": entry_type,
                        "fields": fields,
                        "evidence": {
                            "source": "crossref",
                            "doi": clean_doi,
                            "canonical_title": title,
                            "canonical_year": int(year) if year.isdigit() else None,
                            "canonical_authors": author_list,
                            "canonical_venue": container or publisher,
                        },
                    }
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(1.5 * (attempt + 1))
            else:
                break
        except Exception:
            time.sleep(0.5)

    # Fallback to DataCite
    url = f"https://api.datacite.org/dois/{urllib.parse.quote(clean_doi)}"
    req = urllib.request.Request(url, headers={"User-Agent": ACADEMIC_UA})
    try:
        with urllib.request.urlopen(req, timeout=10, context=SSL_CTX) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8")).get("data", {}).get("attributes", {})
                titles = data.get("titles", [])
                title = titles[0].get("title", "") if titles else ""
                if not title:
                    return None

                author_list = []
                for c in data.get("creators", []):
                    fam = c.get("familyName", "").strip()
                    giv = c.get("givenName", "").strip()
                    if fam and giv:
                        author_list.append(f"{fam}, {giv}")
                    elif c.get("name"):
                        author_list.append(c.get("name"))
                author_str = " and ".join(author_list)

                year = str(data.get("publicationYear") or "")
                publisher = data.get("publisher", "")

                fields = {
                    "title": title,
                    "author": author_str,
                    "year": year,
                    "publisher": publisher,
                    "doi": clean_doi,
                    "url": f"https://doi.org/{clean_doi}",
                }
                key = custom_key or derive_citekey(author_str, year, title)
                return {
                    "key": key,
                    "entry_type": "article" if not publisher else "misc",
                    "fields": fields,
                    "evidence": {
                        "source": "datacite",
                        "doi": clean_doi,
                        "canonical_title": title,
                        "canonical_year": int(year) if year.isdigit() else None,
                        "canonical_authors": author_list,
                        "canonical_venue": publisher,
                    },
                }
    except Exception:
        pass
    return None


def fetch_arxiv(arxiv_input: str, custom_key: str | None = None) -> dict[str, Any] | None:
    """Authenticate and fetch metadata for an arXiv paper via abs HTML extraction."""
    m = ARXIV_REGEX.search(arxiv_input.strip())
    if not m:
        return None
    clean_id = re.sub(r"v\d+$", "", m.group(1))

    url = f"https://arxiv.org/abs/{clean_id}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=10, context=SSL_CTX) as resp:
            if resp.status == 200:
                html = resp.read(65536).decode("utf-8", errors="ignore")
                # Title
                m_title = re.search(r"<h1 class=\"title mathjax\"><span class=\"descriptor\">Title:</span>(.*?)</h1>", html, re.S)
                if not m_title:
                    m_title = re.search(r"<title>\[(.*?)\]\s*(.*?)</title>", html, re.S)
                    raw_title = m_title.group(2) if m_title else ""
                else:
                    raw_title = m_title.group(1)
                title = re.sub(r"\s+", " ", raw_title).strip()
                if not title:
                    return None

                # Authors
                author_names = []
                m_auth = re.search(r"<div class=\"authors\"><span class=\"descriptor\">Authors:</span>(.*?)</div>", html, re.S)
                if m_auth:
                    author_names = re.findall(r"<a href=\"[^\"]*\"[^>]*>([^<]+)</a>", m_auth.group(1))

                # Normalize authors to "Last, First"
                author_list = []
                for name in author_names:
                    parts = name.strip().split()
                    if len(parts) >= 2:
                        author_list.append(f"{parts[-1]}, {' '.join(parts[:-1])}")
                    else:
                        author_list.append(name.strip())
                author_str = " and ".join(author_list)

                # Year from submission or ID
                year = ""
                m_yr = re.search(r"\[v1\]</a></strong>\s+[A-Za-z]+,\s+\d+\s+[A-Za-z]+\s+(\d{4})", html)
                if m_yr:
                    year = m_yr.group(1)
                elif len(clean_id) >= 4 and clean_id[:2].isdigit():
                    prefix = int(clean_id[:2])
                    year = str(2000 + prefix if prefix < 90 else 1900 + prefix)

                fields = {
                    "title": title,
                    "author": author_str,
                    "year": year,
                    "journal": f"arXiv preprint arXiv:{clean_id}",
                    "url": f"https://arxiv.org/abs/{clean_id}",
                }
                key = custom_key or derive_citekey(author_str, year, title)
                return {
                    "key": key,
                    "entry_type": "article",
                    "fields": fields,
                    "evidence": {
                        "source": "arxiv",
                        "arxiv_id": clean_id,
                        "canonical_title": title,
                        "canonical_year": int(year) if year.isdigit() else None,
                        "canonical_authors": author_list,
                        "canonical_venue": "arXiv preprint",
                    },
                }
    except Exception:
        pass
    return None


def fetch_isbn(isbn_input: str, custom_key: str | None = None) -> dict[str, Any] | None:
    """Authenticate and fetch book metadata for an ISBN via OpenLibrary API."""
    clean_isbn = re.sub(r"[^0-9X]", "", isbn_input.strip().upper())
    if len(clean_isbn) not in (10, 13):
        return None

    url = f"https://openlibrary.org/isbn/{clean_isbn}.json"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=10, context=SSL_CTX) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                title = data.get("title", "")
                if not title:
                    return None

                publishers = data.get("publishers", [])
                publisher = publishers[0] if publishers else ""

                pub_date = data.get("publish_date", "")
                m_yr = re.search(r"\b(19\d\d|20\d\d)\b", pub_date)
                year = m_yr.group(1) if m_yr else ""

                # Authors lookup if author keys exist
                author_names = []
                for auth_ref in data.get("authors", []):
                    auth_key = auth_ref.get("key")
                    if auth_key:
                        try:
                            a_url = f"https://openlibrary.org{auth_key}.json"
                            a_req = urllib.request.Request(a_url, headers={"User-Agent": UA})
                            with urllib.request.urlopen(a_req, timeout=5, context=SSL_CTX) as a_resp:
                                if a_resp.status == 200:
                                    a_data = json.loads(a_resp.read().decode("utf-8"))
                                    if a_data.get("name"):
                                        author_names.append(a_data["name"])
                        except Exception:
                            pass

                author_list = []
                for name in author_names:
                    parts = name.strip().split()
                    if len(parts) >= 2:
                        author_list.append(f"{parts[-1]}, {' '.join(parts[:-1])}")
                    else:
                        author_list.append(name.strip())
                author_str = " and ".join(author_list)

                fields = {
                    "title": title,
                    "author": author_str,
                    "year": year,
                    "publisher": publisher,
                    "isbn": clean_isbn,
                }
                key = custom_key or derive_citekey(author_str, year, title)
                return {
                    "key": key,
                    "entry_type": "book",
                    "fields": fields,
                    "evidence": {
                        "source": "openlibrary",
                        "isbn": clean_isbn,
                        "canonical_title": title,
                        "canonical_year": int(year) if year.isdigit() else None,
                        "canonical_authors": author_list,
                        "canonical_venue": publisher or "Publisher Catalog",
                    },
                }
    except Exception:
        pass
    return None


def fetch_url(
    url_input: str,
    custom_key: str | None = None,
    title_override: str = "",
    author_override: str = "",
    year_override: str = "",
    venue_override: str = "",
) -> dict[str, Any] | None:
    """Authenticate and fetch metadata for a web URL (HTTP 200 verification)."""
    clean_url = re.sub(r"\s+", "", url_input.strip())
    if not clean_url.startswith("http"):
        return None

    parsed = urllib.parse.urlparse(clean_url)
    domain = parsed.netloc.lower()
    is_trusted = any(td in domain for td in TRUSTED_DOMAINS)

    status_code = None
    page_title = ""
    citation_meta: dict[str, Any] = {}

    attempts = [(ua, ctx) for ua in (UA, TOOL_UA) for ctx in (SSL_CTX, FALLBACK_SSL_CTX)]
    for ua, ctx in attempts:
        req = urllib.request.Request(clean_url, headers={"User-Agent": ua})
        try:
            with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
                status_code = resp.status
                if status_code in (200, 301, 302, 307, 308):
                    # 2026-10-13: read the whole <head> (not 4 KB) so the
                    # citation_* meta tags publishers embed can be checked.
                    peek = resp.read(262144).decode("utf-8", errors="ignore")
                    m = re.search(r"<title>(.*?)</title>", peek, re.I | re.S)
                    if m:
                        page_title = re.sub(r"\s+", " ", m.group(1)).strip()
                    citation_meta = parse_citation_meta(peek)
                    if not citation_meta and peek.lstrip().startswith("@"):
                        # A publisher's own BibTeX export (e.g. LBNL, ACM) is a
                        # citation record in itself; read it as source metadata.
                        citation_meta = parse_bibtex_export(peek)
                        page_title = citation_meta.get("title", page_title)
                    break
        except urllib.error.HTTPError as e:
            status_code = e.code
            if is_trusted or e.code in (401, 403, 429):
                break
        except Exception:
            continue

    if not status_code or (status_code not in (200, 301, 302, 307, 308) and not is_trusted):
        return None

    final_title = title_override or page_title or domain
    final_author = author_override or (domain.split(".")[-2].capitalize() if "." in domain else domain)
    final_year = year_override or str(time.gmtime().tm_year)

    fields = {
        "title": final_title,
        "author": f"{{{final_author}}}",
        "year": final_year,
        "url": clean_url,
    }
    if venue_override:
        fields["howpublished"] = venue_override
    fields["note"] = f"Accessed: {time.strftime('%Y-%m-%d')}"

    key = custom_key or derive_citekey(final_author, final_year, final_title)
    evidence: dict[str, Any] = {
        "source": "verified_web",
        "url": clean_url,
        "domain": domain,
        "page_title": page_title[:100],
        "status_code": status_code,
    }
    # Publisher pages (PMLR, NeurIPS, ACM, IEEE, ...) carry Highwire citation_*
    # meta tags; record them so `bib reseal` can check an edited entry's
    # authors, year, and pages against the page instead of trusting overrides.
    if citation_meta.get("title"):
        evidence["canonical_title"] = citation_meta["title"]
    if citation_meta.get("authors"):
        evidence["canonical_authors"] = citation_meta["authors"]
    if citation_meta.get("year"):
        evidence["canonical_year"] = citation_meta["year"]
    if citation_meta.get("pages"):
        evidence["canonical_pages"] = citation_meta["pages"]
    if citation_meta.get("venue"):
        evidence["canonical_venue"] = citation_meta["venue"]
    if citation_meta.get("doi"):
        evidence["doi"] = citation_meta["doi"]
    return {
        "key": key,
        "entry_type": "misc",
        "fields": fields,
        "evidence": evidence,
    }


def parse_bibtex_export(text: str) -> dict[str, Any]:
    """Read the first entry of a publisher's BibTeX export as source metadata.

    Returns the same shape as :func:`parse_citation_meta` (title, ``"Last,
    First"`` authors, year, pages, DOI), or ``{}`` if the text does not parse.
    """
    try:
        entries, _ = bib_lint.parse_bib(text)
    except ValueError:
        return {}
    if not entries:
        return {}
    entry = entries[0]

    def value(name: str) -> str:
        field = entry.get(name)
        return re.sub(r"\s+", " ", field.value).strip() if field else ""

    meta: dict[str, Any] = {}
    if value("title"):
        meta["title"] = clean_str(value("title"))
    authors = []
    for name in re.split(r"\s+and\s+", value("author")):
        name = name.strip()
        if not name:
            continue
        if "," in name:
            authors.append(name)
        else:
            parts = name.split()
            authors.append(f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) > 1 else name)
    if authors:
        meta["authors"] = authors
    m_year = re.search(r"(19|20)\d\d", value("year"))
    if m_year:
        meta["year"] = int(m_year.group(0))
    if value("pages"):
        meta["pages"] = value("pages")
    venue = value("journal") or value("booktitle") or value("institution")
    if venue:
        meta["venue"] = venue
    if value("doi"):
        meta["doi"] = value("doi")
    return meta


def parse_citation_meta(html_text: str) -> dict[str, Any]:
    """Extract Highwire ``citation_*`` meta tags (title, authors, year, pages, venue, DOI).

    Authors come back as ``"Last, First"`` strings in page order. Returns ``{}``
    when the page carries no citation metadata.
    """
    import html as _html

    tags: dict[str, list[str]] = {}
    for m in re.finditer(r"<meta\b[^>]*>", html_text, re.I):
        tag = m.group(0)
        name = re.search(r"""\bname\s*=\s*["']([^"']+)["']""", tag, re.I)
        content = re.search(r"""\bcontent\s*=\s*["']([^"']*)["']""", tag, re.I)
        if name and content and name.group(1).lower().startswith("citation_"):
            value = re.sub(r"\s+", " ", _html.unescape(content.group(1))).strip()
            tags.setdefault(name.group(1).lower(), []).append(value)
    if not tags:
        return {}

    meta: dict[str, Any] = {}
    if tags.get("citation_title"):
        meta["title"] = tags["citation_title"][0]
    authors = []
    for name in tags.get("citation_author", []):
        if "," in name:
            authors.append(name)
        else:
            parts = name.split()
            authors.append(f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) > 1 else name)
    if authors:
        meta["authors"] = authors
    for date_tag in ("citation_publication_date", "citation_date", "citation_online_date"):
        m_year = re.search(r"(19|20)\d\d", (tags.get(date_tag) or [""])[0])
        if m_year:
            meta["year"] = int(m_year.group(0))
            break
    first = (tags.get("citation_firstpage") or [""])[0]
    last = (tags.get("citation_lastpage") or [""])[0]
    if first and last:
        meta["pages"] = f"{first}--{last}"
    for venue_tag in ("citation_conference_title", "citation_journal_title", "citation_inbook_title"):
        if tags.get(venue_tag):
            meta["venue"] = tags[venue_tag][0]
            break
    if tags.get("citation_doi"):
        meta["doi"] = tags["citation_doi"][0]
    return meta


# ----------------------------------------------------------------------
# Atomic Gateway Actions
# ----------------------------------------------------------------------

def add_entry(
    repo_root: Path,
    metadata: dict[str, Any],
    check_duplicates: bool = True,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Atomically append a verified entry to references.bib and seal it into the manifest."""
    bib_path = repo_root / "bibs" / "references.bib"
    if not bib_path.is_file():
        fallback = repo_root / "books" / "references.bib"
        if fallback.is_file():
            bib_path = fallback

    manifest_path = repo_root / "bibs" / "manifest.json"
    if not manifest_path.is_file():
        legacy = repo_root / "binder" / "tools" / "bib_verification_manifest.json"
        if legacy.is_file():
            manifest_path = legacy

    if not bib_path.is_file():
        return False, f"Bibliography file not found: {bib_path}", None
    if not manifest_path.is_file():
        return False, f"Verification manifest not found: {manifest_path}", None

    bib_text = bib_path.read_text(encoding="utf-8")
    existing_entries, _ = bib_lint.parse_bib(bib_text)
    existing_keys = {e.key for e in existing_entries}

    key = metadata["key"]
    if check_duplicates and key in existing_keys:
        return False, f"Citation key '{key}' already exists in {bib_path.name}", None

    # Check for DOI duplication
    new_doi = metadata.get("fields", {}).get("doi") or metadata.get("evidence", {}).get("doi")
    if check_duplicates and new_doi:
        clean_new_doi = str(new_doi).lower().strip()
        for e in existing_entries:
            e_doi = e.get("doi").value if e.get("doi") else ""
            if e_doi and e_doi.lower().strip() == clean_new_doi:
                return False, f"DOI '{clean_new_doi}' is already cited under key '@{e.entry_type}{{{e.key}}}'", None

    # Format entry
    raw_entry = format_bibtex_entry(metadata["entry_type"], key, metadata["fields"])
    raw_hash = compute_entry_hash(raw_entry)

    # Load existing manifest
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"Failed to parse manifest: {e}", None

    # Record manifest entry
    manifest_record = {
        "key": key,
        "file": "bibs/references.bib",
        "entry_type": metadata["entry_type"],
        "title": clean_str(str(metadata["fields"].get("title", ""))),
        "year": str(metadata["fields"].get("year", "")),
        "entry_hash": raw_hash,
        "status": "verified",
        "verified_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "details": metadata["evidence"],
    }
    manifest[key] = manifest_record

    # Append to references.bib cleanly
    clean_bib = bib_text.rstrip()
    updated_bib = f"{clean_bib}\n\n{raw_entry}\n"

    # Write files atomically
    bib_path.write_text(updated_bib, encoding="utf-8")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return True, f"Successfully authenticated, added, and sealed @{metadata['entry_type']}{{{key}}}", manifest_record


# ----------------------------------------------------------------------
# Re-seal after a house-style edit
# ----------------------------------------------------------------------

def _surname_token(name: str) -> str:
    """Last token of an author's family name, casefolded and diacritics-folded."""
    import unicodedata

    # The bibliography writes accents as LaTeX escapes (M\k{a}dry, J\"{u}rgen,
    # {\i}); reduce each to its base letter before comparing with source names.
    name = re.sub(r"\\([a-zA-Z]{1,2})\s*\{\s*([a-zA-Z])\s*\}", r"\2", name)
    name = re.sub(r"\\[`'^\"~=.]\s*\{?\s*([a-zA-Z])\s*\}?", r"\1", name)
    name = re.sub(r"\\(ss|ae|oe|aa|o|l|i|j)\b", r"\1", name, flags=re.I)
    name = clean_str(name)
    family = name.split(",")[0] if "," in name else (name.split()[-1] if name.split() else "")
    tokens = family.split()
    token = tokens[-1] if tokens else ""
    folded = unicodedata.normalize("NFKD", token).encode("ascii", "ignore").decode("ascii")
    return folded.casefold()


def _entry_field(entry: Any, name: str) -> str:
    field = entry.get(name)
    return re.sub(r"\s+", " ", field.value).strip() if field else ""


def compare_entry_to_source(entry: Any, resolved: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Check an edited bib entry's identity fields against freshly resolved source metadata.

    Returns ``(checked, mismatches)``. A field is checked only when the source
    reports it; the title must always be available. Authors must match the
    source's list in order by surname (diacritics and particles folded), the
    year and DOI must match exactly, and pages must match when both sides
    report them.
    """
    evidence = resolved.get("evidence", {})
    fields = resolved.get("fields", {})
    checked: list[str] = []
    mismatches: list[str] = []

    src_title = evidence.get("canonical_title") or evidence.get("page_title") or fields.get("title", "")
    entry_title = _entry_field(entry, "title")
    if not src_title:
        mismatches.append("title: the source reported no title to check against")
    else:
        checked.append("title")
        sim = string_similarity(entry_title, src_title)
        if sim < 0.9:
            mismatches.append(f"title: similarity {sim:.2f} < 0.90 ({entry_title!r} vs {src_title!r})")

    src_authors = evidence.get("canonical_authors") or []
    if src_authors:
        checked.append("author")
        entry_authors = [a for a in re.split(r"\s+and\s+", _entry_field(entry, "author")) if a.strip()]
        ours = [_surname_token(a) for a in entry_authors]
        theirs = [_surname_token(a) for a in src_authors]
        if ours != theirs:
            mismatches.append(f"author: surnames {ours} vs source {theirs}")

    src_year = evidence.get("canonical_year")
    if src_year:
        checked.append("year")
        if _entry_field(entry, "year") != str(src_year):
            mismatches.append(f"year: {_entry_field(entry, 'year')} vs source {src_year}")

    src_doi = (evidence.get("doi") or fields.get("doi") or "").lower()
    entry_doi = _entry_field(entry, "doi").lower()
    if entry_doi and src_doi:
        checked.append("doi")
        if entry_doi != src_doi:
            mismatches.append(f"doi: {entry_doi} vs source {src_doi}")
    elif entry_doi:
        mismatches.append("doi: the entry carries a DOI the source did not report")

    src_pages = evidence.get("canonical_pages") or fields.get("pages") or ""
    entry_pages = _entry_field(entry, "pages")
    if src_pages and entry_pages:
        checked.append("pages")
        norm = lambda p: re.sub(r"\s*[-–]+\s*", "-", p)
        if norm(entry_pages) != norm(src_pages):
            mismatches.append(f"pages: {entry_pages} vs source {src_pages}")

    return checked, mismatches


def reseal_entry(
    repo_root: Path,
    key: str,
    resolved: dict[str, Any],
    note: str,
    dry_run: bool = False,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Re-verify an edited entry against its source and re-record it in the manifest.

    This is the gateway path for a house-style edit to a sealed entry
    (bib-check §8, *Editing a sealed entry*): ``bib add`` refuses existing keys,
    so the edited entry is compared field by field with freshly resolved source
    metadata, and the manifest record is rewritten only if every checked field
    agrees. The entry text in ``references.bib`` is never modified here.
    """
    bib_path = repo_root / "bibs" / "references.bib"
    manifest_path = repo_root / "bibs" / "manifest.json"
    if not bib_path.is_file() or not manifest_path.is_file():
        return False, "bibs/references.bib or bibs/manifest.json not found", None

    entries, _ = bib_lint.parse_bib(bib_path.read_text(encoding="utf-8"))
    matches = [e for e in entries if e.key == key]
    if len(matches) != 1:
        return False, f"expected exactly one @{{{key}}} in references.bib, found {len(matches)}", None
    entry = matches[0]

    checked, mismatches = compare_entry_to_source(entry, resolved)
    if mismatches:
        return False, "source disagrees with the entry: " + "; ".join(mismatches), None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous = manifest.get(key) or {}
    details = dict(resolved["evidence"])
    for carried in ("supersedes", "discovered_doi"):
        if carried in (previous.get("details") or {}) and carried not in details:
            details[carried] = previous["details"][carried]
    details["checked_fields"] = checked
    details["style_edit"] = note

    record = {
        "key": key,
        "file": "bibs/references.bib",
        "entry_type": entry.entry_type,
        "title": clean_str(_entry_field(entry, "title")),
        "year": _entry_field(entry, "year"),
        "entry_hash": compute_entry_hash(entry.raw),
        "status": "verified",
        "verified_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "details": details,
    }
    if dry_run:
        return True, f"would re-seal @{entry.entry_type}{{{key}}} (checked: {', '.join(checked)})", record

    manifest[key] = record
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return True, f"re-sealed @{entry.entry_type}{{{key}}} (checked: {', '.join(checked)})", record


def ingest_inbox(
    repo_root: Path,
    inbox_path: Path,
    dry_run: bool = False,
) -> tuple[int, int, list[str]]:
    """Scan inbox file, authenticate each candidate entry online, promote verified entries."""
    if not inbox_path.is_file():
        return 0, 0, [f"Inbox file does not exist: {inbox_path}"]

    inbox_text = inbox_path.read_text(encoding="utf-8")
    entries, _ = bib_lint.parse_bib(inbox_text)
    if not entries:
        return 0, 0, ["Inbox contains 0 parseable entries."]

    verified_added = 0
    rejected = 0
    messages: list[str] = []
    remaining_raw_entries: list[str] = []

    for entry in entries:
        key = entry.key
        title = entry.get("title").value if entry.get("title") else ""
        doi = entry.get("doi").value if entry.get("doi") else ""
        url = entry.get("url").value if entry.get("url") else ""
        raw = entry.raw

        resolved: dict[str, Any] | None = None

        # 1. DOI
        m_doi = DOI_REGEX.search(raw) or (DOI_REGEX.search(doi) if doi else None)
        if m_doi:
            resolved = fetch_doi(m_doi.group(1), custom_key=key)

        # 2. arXiv
        if not resolved:
            m_arx = ARXIV_REGEX.search(raw) or (ARXIV_REGEX.search(url) if url else None)
            if m_arx:
                resolved = fetch_arxiv(m_arx.group(1), custom_key=key)

        # 3. URL
        if not resolved and url:
            resolved = fetch_url(url, custom_key=key, title_override=title)

        if resolved:
            if dry_run:
                verified_added += 1
                messages.append(f"[green]Verified (dry-run):[/green] @{entry.entry_type}{{{key}}} -> {resolved['evidence']['source']}")
            else:
                ok, msg, _ = add_entry(repo_root, resolved, check_duplicates=True)
                if ok:
                    verified_added += 1
                    messages.append(f"[green]Added & Sealed:[/green] @{entry.entry_type}{{{key}}} ({resolved['evidence']['source']})")
                else:
                    rejected += 1
                    messages.append(f"[red]Failed to add @{key}:[/red] {msg}")
                    remaining_raw_entries.append(entry.raw)
        else:
            rejected += 1
            messages.append(f"[red]Rejected unverified:[/red] @{entry.entry_type}{{{key}}} ('{clean_str(title)[:50]}')")
            remaining_raw_entries.append(entry.raw)

    if not dry_run and verified_added > 0:
        if remaining_raw_entries:
            inbox_path.write_text("\n\n".join(remaining_raw_entries) + "\n", encoding="utf-8")
        else:
            inbox_path.write_text("", encoding="utf-8")

    return verified_added, rejected, messages
