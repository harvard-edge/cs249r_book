"""Native Reference Gateway: Authenticated citation resolution, formatting, and ledger sealing.

Provides programmatic and CLI facilities to safely ingest, verify, and seal
new bibliography entries without direct, unvalidated file edits to books/references.bib.
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

    req = urllib.request.Request(clean_url, headers={"User-Agent": UA})
    status_code = None
    page_title = ""

    for ctx in (SSL_CTX, FALLBACK_SSL_CTX):
        try:
            with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
                status_code = resp.status
                if status_code in (200, 301, 302, 307, 308):
                    peek = resp.read(4096).decode("utf-8", errors="ignore")
                    m = re.search(r"<title>(.*?)</title>", peek, re.I | re.S)
                    if m:
                        page_title = re.sub(r"\s+", " ", m.group(1)).strip()
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
    return {
        "key": key,
        "entry_type": "misc",
        "fields": fields,
        "evidence": {
            "source": "verified_web",
            "url": clean_url,
            "domain": domain,
            "page_title": page_title[:100],
            "status_code": status_code,
        },
    }


# ----------------------------------------------------------------------
# Atomic Gateway Actions
# ----------------------------------------------------------------------

def add_entry(
    repo_root: Path,
    metadata: dict[str, Any],
    check_duplicates: bool = True,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Atomically append a verified entry to references.bib and seal it into the manifest."""
    bib_path = repo_root / "books" / "references.bib"
    manifest_path = repo_root / "binder" / "tools" / "bib_verification_manifest.json"

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
        "file": "books/references.bib",
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
