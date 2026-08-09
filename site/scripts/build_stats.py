#!/usr/bin/env python3
"""Gather every number the site displays into one cache file.

Runs as the site's Quarto pre-render step. Three tiers of source:

  1. Repo-derived  - counted from the filesystem and git. Always available.
  2. Public API    - GitHub stars and merged PRs. No credentials required,
                     but an available token raises the rate limit.
  3. Credentialed  - GA4 readership and Buttondown subscribers. Skipped
                     unless the corresponding secret is in the environment.

Any source that fails leaves the previous value untouched, so a rate limit or
an expired credential degrades to the last known good number rather than to a
blank page. Failures are reported on stderr and, in CI, annotated as warnings.

Output: site/config/stats-cache.json, consumed by inject_stats.py.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SITE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SITE_DIR.parent
CACHE_PATH = SITE_DIR / "config" / "stats-cache.json"

GITHUB_REPO = "harvard-edge/cs249r_book"
BUTTONDOWN_NEWSLETTER = "mlsysbook"
# The GA4 property behind the Looker Studio readership report. A property ID is
# not a credential -- it identifies the property, it does not grant access --
# so it lives here rather than in a secret. Access is controlled by the service
# account's Viewer grant in GA4 Property Access Management.
GA4_PROPERTY_ID = os.environ.get("GA4_PROPERTY_ID", "419684623")

# Stats rendered as an exact figure vs. floored to two significant digits with
# a trailing "+". Exact for anything countable from the repo (a deck count is
# precisely knowable, and "35" reads better than "35+"); floored for community
# numbers that are approximate at the source anyway.
#
# "stars" is deliberately exact. A floored "27,000+" would not visibly change
# until the count crossed 28,000, which defeats the point of a live counter
# next to a "Star on GitHub" call to action. The rounded form is still
# available as {{stats.stars_rounded}} for milestone-style copy.
# Reach figures render exact. In a card that presents them as measured
# readings, "312,523" carries more authority than "310,000+", which reads
# as marketing. Newsletter mockup copy keeps the rounded style.
PLUS_STYLE = {"merged_prs", "subscribers"}

# Figures that only ever grow in reality, whatever the measurement of them does.
MONOTONIC_KEYS = ("readers", "countries")

# When set, the figures currently on the live page are read back and used as the
# floor for MONOTONIC_KEYS. The scheduled refresh publishes stats.json without
# committing its cache, so the committed file can be days behind what the site
# actually shows; without this the high-water check in main() would measure
# against a stale figure and could still publish a number lower than the one
# already on the page.
PUBLISHED_STATS_URL = os.environ.get("PUBLISHED_STATS_URL", "").strip()

# Consecutive clamped runs before the build is failed rather than warned. The
# schedule is every six hours, so eight is about two days.
CLAMP_ALARM_RUNS = 8

warnings: list[str] = []


def warn(message: str) -> None:
    warnings.append(message)
    print(f"  ! {message}", file=sys.stderr)


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------

def floor_2sf(value: int) -> int:
    """Floor to two significant digits. 27708 -> 27000, 1172 -> 1100, 180 -> 180.

    Never rounds up: a displayed "27,000+" is always a claim the real number
    supports.
    """
    if value < 100:
        return value
    magnitude = 10 ** (len(str(value)) - 2)
    return (value // magnitude) * magnitude


def fetch_published() -> dict:
    """The display block currently served at PUBLISHED_STATS_URL, or {}.

    The site sits behind Cloudflare, which answers urllib's default agent with a
    403, so identify the same way the GitHub calls do.
    """
    if not PUBLISHED_STATS_URL:
        return {}
    request = urllib.request.Request(PUBLISHED_STATS_URL, headers={
        "User-Agent": "mlsysbook-site-stats",
        "Accept": "application/json",
        "Cache-Control": "no-cache",
    })
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8")).get("display", {})
    except Exception as exc:  # noqa: BLE001 - degrade to the committed cache
        warn(f"could not read published stats ({exc}); "
             "using the committed cache as the baseline")
        return {}


def seed_from_published(values: dict, published: dict) -> None:
    """Raise the baseline to whatever the live page is currently showing.

    Only ever raises. A published figure lower than the cached one is ignored,
    so this corrects a stale baseline but never lowers one.
    """
    for key in MONOTONIC_KEYS:
        raw = published.get(key)
        if not isinstance(raw, str):
            continue
        try:
            live = int(raw.replace(",", ""))
        except ValueError:
            continue        # a "1,100+" style figure is not a hard baseline
        current = values.get(key)
        if not isinstance(current, int) or live > current:
            values[key] = live


def render(key: str, value: int) -> str:
    if key in PLUS_STYLE:
        return f"{floor_2sf(value):,}+"
    return f"{value:,}"


# --------------------------------------------------------------------------
# tier 1: repo-derived
# --------------------------------------------------------------------------

def tracked_files(*pathspecs: str) -> list[str]:
    """Files git tracks under the given paths.

    Counting tracked files rather than walking the filesystem is deliberate.
    Several generated trees (tinytorch/modules/, every _build/) exist in a
    working checkout but are gitignored, so a filesystem walk gives one answer
    locally and a different one in CI's fresh clone. git ls-files is the only
    view both environments share.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "--", *pathspecs],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=30,
        )
        return [line for line in result.stdout.splitlines() if line]
    except subprocess.SubprocessError as exc:
        warn(f"git ls-files failed for {pathspecs} ({exc})")
        return []


def count_slide_decks() -> int:
    """One Beamer .tex per chapter deck across both volumes."""
    return sum(1 for f in tracked_files("slides/vol1", "slides/vol2")
               if f.endswith(".tex"))


def count_slide_svgs() -> int:
    return sum(1 for f in tracked_files("slides/vol1", "slides/vol2")
               if f.endswith(".svg"))


def count_tinytorch_modules() -> int:
    """Counted from tinytorch/quarto/modules/NN_*.qmd.

    Not from tinytorch/modules/, which .gitignore excludes because nbdev
    regenerates it from src/. That directory is populated locally and empty in
    CI, which would silently publish "0 progressive modules".
    """
    return sum(1 for f in tracked_files("tinytorch/quarto/modules")
               if re.search(r"/\d{2}_[^/]+\.qmd$", f))


def read_vol2_chapters() -> dict[str, int]:
    """Map a Vol II chapter slug to its printed chapter number.

    Parsed from the PDF config, which is the canonical chapter order. Part
    openers (contents/vol2/parts/*) and frontmatter do not take chapter
    numbers, so they are skipped rather than counted.
    """
    config = REPO_ROOT / "book" / "quarto" / "config" / "_quarto-pdf-vol2.yml"
    if not config.is_file():
        return {}

    chapters: dict[str, int] = {}
    number = 0
    in_chapters = False
    for line in config.read_text(encoding="utf-8").splitlines():
        if re.match(r"^\s*chapters:\s*$", line):
            in_chapters = True
            continue
        if in_chapters and re.match(r"^\s*appendices:\s*$", line):
            break
        if not in_chapters:
            continue

        match = re.search(r"contents/vol2/([^/]+)/([^/\s]+)\.qmd", line)
        if not match:
            continue
        directory, _ = match.groups()
        if directory in {"frontmatter", "parts", "backmatter"}:
            continue
        number += 1
        chapters[directory] = number
    return chapters


def read_latest_lab() -> tuple[int, str]:
    """Total lab count and a display label for the newest Volume II lab.

    Read from labs/mlsysbook_labs/catalog.py, the versioned catalog that is the
    labs' own source of truth. The lab directories on disk are build output.
    """
    catalog = REPO_ROOT / "labs" / "mlsysbook_labs" / "catalog.py"
    if not catalog.is_file():
        warn("labs catalog absent; keeping cached lab values")
        return 0, ""

    text = catalog.read_text(encoding="utf-8")
    entries = re.findall(
        r'"(vol[12])/lab_(\d+)_[^"]*\.py":\s*LabMetadata\((?:[^)]*?)title="([^"]*)"',
        text, re.DOTALL,
    )
    if not entries:
        warn("labs catalog matched no entries; keeping cached lab values")
        return 0, ""

    vol2 = [(int(num), title) for volume, num, title in entries if volume == "vol2"]
    if not vol2:
        return len(entries), ""
    number, title = max(vol2, key=lambda item: item[0])
    return len(entries), f"Lab {number} · {title}"


def read_last_updated() -> str:
    """Month and year of the newest commit touching book content."""
    try:
        stamp = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", "book/quarto/contents"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        return datetime.fromisoformat(stamp).strftime("%B %Y")
    except (subprocess.SubprocessError, ValueError) as exc:
        warn(f"last-updated date unavailable ({exc}); keeping cached value")
        return ""


# --------------------------------------------------------------------------
# tier 2: public GitHub API
# --------------------------------------------------------------------------

def github_request(url: str) -> dict | None:
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "mlsysbook-site-stats",
    })
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        warn(f"GitHub API failed for {url} ({exc}); keeping cached value")
        return None


def fetch_stars() -> int | None:
    data = github_request(f"https://api.github.com/repos/{GITHUB_REPO}")
    return data.get("stargazers_count") if data else None


def fetch_merged_prs() -> int | None:
    # The search API is rate-limited to 10 req/min unauthenticated, which is
    # why CI should supply GITHUB_TOKEN.
    query = f"repo:{GITHUB_REPO}+is:pr+is:merged"
    data = github_request(f"https://api.github.com/search/issues?q={query}&per_page=1")
    return data.get("total_count") if data else None


# --------------------------------------------------------------------------
# tier 3: credentialed
# --------------------------------------------------------------------------

def fetch_ga4() -> dict:
    """Readership figures from the GA4 Data API.

    Returns a dict with any of: readers (all-time users), countries (distinct
    countries), readers_monthly (mean users across three 30-day windows),
    top_countries (a
    display string of the largest few). Missing keys mean that query did not
    run, and the caller leaves the cached value in place.

    Requires GA4_PROPERTY_ID plus service-account credentials discoverable by
    google.auth (GOOGLE_APPLICATION_CREDENTIALS pointing at the key file).
    """
    if not GA4_PROPERTY_ID:
        return {}
    if not (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            or os.environ.get("GA4_SERVICE_ACCOUNT_JSON")):
        warn("GA4 credentials absent; keeping cached readership numbers")
        return {}

    try:
        from google.analytics.data_v1beta import BetaAnalyticsDataClient
        from google.analytics.data_v1beta.types import (
            DateRange, Dimension, Metric, RunReportRequest,
        )
    except ImportError:
        warn("google-analytics-data not installed; keeping cached readership numbers")
        return {}

    # A JSON blob in the environment is written out so google.auth can find it.
    if os.environ.get("GA4_SERVICE_ACCOUNT_JSON") and not os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS"
    ):
        key_path = SITE_DIR / ".ga4-key.json"
        key_path.write_text(os.environ["GA4_SERVICE_ACCOUNT_JSON"], encoding="utf-8")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)

    # 2015-08-14 is the earliest date the Data API accepts, so it stands in for
    # "all time" regardless of when the property started collecting.
    all_time = [DateRange(start_date="2015-08-14", end_date="today")]
    # Three consecutive 30-day windows. The monthly figure is their mean, which
    # is a genuine average of monthly readers and can honestly be labelled one.
    #
    # It cannot be done by dividing a 90-day count by three. These are distinct
    # user counts, so somebody who reads in all three months counts once across
    # 90 days but three times across three months; the shortcut would understate
    # the average badly. Averaging three separate windows is the correct form,
    # and it also damps both the weekly traffic swings and the HLL++ noise.
    month_windows = [
        [DateRange(start_date="30daysAgo", end_date="today")],
        [DateRange(start_date="60daysAgo", end_date="31daysAgo")],
        [DateRange(start_date="90daysAgo", end_date="61daysAgo")],
    ]
    prop = f"properties/{GA4_PROPERTY_ID}"
    out: dict = {}

    try:
        client = BetaAnalyticsDataClient()

        def run(ranges, dims=(), mets=("totalUsers",), limit=1):
            return client.run_report(RunReportRequest(
                property=prop, date_ranges=ranges,
                dimensions=[Dimension(name=d) for d in dims],
                metrics=[Metric(name=m) for m in mets],
                limit=limit,
            ))

        # The headline is totalUsers, and that choice is now evidence-based.
        #
        # Google estimates distinct-user metrics with HyperLogLog++ rather than
        # counting them, at a documented precision of 14 for Total Users, so
        # this figure carries a residual error and can come back slightly lower
        # than last time. That produced a 0.27% drop on 2026-08-09.
        #   https://developers.google.com/analytics/blog/2022/hll
        #
        # newUsers was tried as a replacement, on the reasoning that counting
        # first_visit events avoids the approximation and that every user is new
        # exactly once, so the two should agree. They do not: on 2026-08-09 the
        # same property reported 313,674 total users and 309,653 new users, a
        # gap of about 4,000. Readers who arrived before the property began
        # recording first_visit are absent from the newer metric. Switching
        # would have dropped the headline by that gap and then frozen it under
        # the high-water rule for months while it climbed back.
        #
        # So the approximation is kept and its downside is handled instead: the
        # high-water rule in main() absorbs the wobble, which is small enough
        # (0.27% against a documented 1.6% bound) to recover within a run or two.
        totals = run(all_time)
        if totals.rows:
            out["readers"] = int(totals.rows[0].metric_values[0].value)

        # The rejected metric is still recorded, unpublished, so the gap between
        # the two stays visible and this decision can be revisited if it closes.
        alt = run(all_time, mets=("newUsers",))
        if alt.rows:
            out["diag_new_users"] = int(alt.rows[0].metric_values[0].value)

        months, monthly = [], None
        for window in month_windows:
            response = run(window)
            if monthly is None:
                monthly = response          # the most recent, for the quality check
            if response.rows:
                months.append(int(response.rows[0].metric_values[0].value))
        if months:
            # Average whatever came back, so a single failed window degrades to
            # a two-month mean rather than losing the figure entirely.
            out["readers_monthly"] = round(sum(months) / len(months))
            out["diag_month_windows"] = " / ".join(f"{m:,}" for m in months)

        # The country breakdown supplies a row count and the leading names.
        # It is never summed.
        by_country = run(all_time, dims=("country",), limit=1000)
        rows = [
            (r.dimension_values[0].value, int(r.metric_values[0].value))
            for r in by_country.rows
            if r.dimension_values[0].value not in ("", "(not set)")
            and int(r.metric_values[0].value) > 0
        ]
        if rows:
            rows.sort(key=lambda r: r[1], reverse=True)
            out["countries"] = len(rows)
            out["top_countries"] = " · ".join(name for name, _ in rows[:5])

        # The API says when it sampled a response or folded high-cardinality
        # rows into "(other)". Recording it means a degraded run is visible in
        # the cache afterwards instead of silently moving a published figure.
        # Read defensively: this is diagnostic only and must never be the thing
        # that breaks a build.
        flags = []
        for label, response in (("readers", totals), ("new_users", alt),
                                ("readers_monthly", monthly), ("countries", by_country)):
            try:
                meta = getattr(response, "metadata", None)
                if getattr(meta, "sampling_metadatas", None):
                    flags.append(f"{label}:sampled")
                if getattr(meta, "data_loss_from_other_row", False):
                    flags.append(f"{label}:other-row")
            except Exception:  # noqa: BLE001 - diagnostics never block a build
                pass
        out["diag_ga4_quality"] = ", ".join(flags) if flags else "clean"
        if flags:
            warn("GA4 returned degraded data (" + ", ".join(flags)
                 + "); figures may be less reliable this run")

        print(f"  GA4: {out.get('readers', 0):,} readers all-time, "
              f"{out.get('readers_monthly', 0):,} a month on average, "
              f"{out.get('countries', 0)} countries",
              file=sys.stderr)
        return out
    except Exception as exc:  # noqa: BLE001 - any API failure degrades to cache
        warn(f"GA4 query failed ({exc}); keeping cached readership numbers")
        return {}


def fetch_subscribers() -> int | None:
    api_key = os.environ.get("BUTTONDOWN_API_KEY")
    if not api_key:
        warn("BUTTONDOWN_API_KEY absent; keeping cached subscriber count")
        return None

    url = "https://api.buttondown.email/v1/subscribers?type=regular&page=1"
    request = urllib.request.Request(url, headers={
        "Authorization": f"Token {api_key}",
        "User-Agent": "mlsysbook-site-stats",
    })
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read()).get("count")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        warn(f"Buttondown API failed ({exc}); keeping cached subscriber count")
        return None


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def main() -> int:
    cache = {}
    if CACHE_PATH.is_file():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    values: dict[str, int | str] = dict(cache.get("values", {}))
    source: dict[str, str] = dict(cache.get("source", {}))
    # What the page is currently publishing, captured before this run starts
    # overwriting it, so a figure that only ever grows can be held to that.
    # Seeded from the live page first, because the committed cache is not
    # necessarily what the site is showing.
    published = fetch_published()
    seed_from_published(values, published)
    previous: dict[str, int | str] = dict(values)

    def record(key: str, value, origin: str) -> None:
        """Store a value only when its source actually produced one.

        A failed fetch leaves both the value and its recorded origin alone, so
        the cache always says where the number currently on the page came from.
        """
        if value is None or value == "":
            return
        values[key] = value
        source[key] = origin

    print("Gathering site stats...", file=sys.stderr)

    # Tier 1: the repo is always readable, so these overwrite unconditionally.
    record("slide_decks", count_slide_decks(), "repo")
    record("slide_svgs", count_slide_svgs(), "repo")
    record("tinytorch_modules", count_tinytorch_modules(), "repo")
    labs_total, latest_lab = read_latest_lab()
    record("labs_total", labs_total, "repo")
    record("latest_lab", latest_lab, "repo")

    chapters = read_vol2_chapters()
    if chapters:
        record("vol2_chapters", len(chapters), "repo")
        for slug, number in chapters.items():
            record(f"vol2_ch.{slug}", number, "repo")

    record("last_updated", read_last_updated(), "git")

    # Tiers 2 and 3: only overwrite on success.
    record("stars", fetch_stars(), "github")
    record("merged_prs", fetch_merged_prs(), "github")

    for key, value in fetch_ga4().items():
        record(key, value, "ga4")

    # Cumulative readership and country reach only ever grow in reality. The
    # measurement of them does not: a sampled response, or the residual error
    # in an approximated metric, can return less than was published last time.
    # Showing that would tell a visitor the book had lost readers, and would
    # run the counter on the card backwards. Hold the high-water mark instead,
    # and say in the cache that it was held.
    #
    # This is a safety net, not the fix. If it fires often, the metric feeding
    # it is the thing to change; diag_total_users and diag_ga4_quality are
    # there to show which.
    clamped = False
    for key in MONOTONIC_KEYS:
        prior, fresh = previous.get(key), values.get(key)
        if isinstance(prior, int) and isinstance(fresh, int) and fresh < prior:
            warn(f"GA4 returned {key}={fresh:,}, below the published "
                 f"{prior:,}; holding the higher figure")
            values[key] = prior
            source[key] = "ga4+highwater"
            clamped = True

    # One clamp is noise. A clamp on every run for two days is a broken
    # pipeline, and the dangerous part is that the page would look perfectly
    # healthy while showing a frozen number. Carry the streak in the published
    # file, because this job never commits its cache, and make enough noise to
    # be seen when it stops looking like noise.
    try:
        streak = int(str(published.get("diag_clamp_streak", "0")).replace(",", ""))
    except ValueError:
        streak = 0
    streak = streak + 1 if clamped else 0
    values["diag_clamp_streak"] = streak
    source["diag_clamp_streak"] = "derived"
    alarm = streak >= CLAMP_ALARM_RUNS

    record("subscribers", fetch_subscribers(), "buttondown")

    # Anything still carrying a "seed" origin is a figure nobody has verified
    # against a live source. It renders so the page is not broken, but it is
    # called out on every build so it cannot quietly become permanent.
    stale = sorted(k for k, v in source.items() if v == "seed")
    if stale:
        warn("UNVERIFIED figures still on the page (no live source connected): "
             + ", ".join(f"{k}={values[k]}" for k in stale))

    missing = sorted(set(values) - set(source))
    for key in missing:
        source[key] = "unknown"

    # Render every numeric value into its display string once, here, so the
    # injector stays a dumb substitution and the "+" policy lives in one place.
    display = {
        key: render(key, value) if isinstance(value, int) else str(value)
        for key, value in values.items()
    }

    # Milestone copy wants the round number even though the headline counter is
    # exact, so both forms of the star count are published.
    if isinstance(values.get("stars"), int):
        display["stars_rounded"] = f"{floor_2sf(values['stars']):,}+"

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "warnings": warnings,
        "source": source,
        "values": values,
        "display": display,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"  wrote {CACHE_PATH.relative_to(REPO_ROOT)} "
          f"({len(display)} values, {len(warnings)} warning(s))", file=sys.stderr)

    if alarm:
        # Annotated as an error and returned non-zero so the run goes red. The
        # publish step is skipped, which is the point: stop quietly restating a
        # figure the source no longer supports.
        print(f"::error::Readership has been held at the high-water mark for "
              f"{streak} consecutive runs. GA4 is reporting materially less "
              f"than the published figure, which is no longer explainable as "
              f"estimator noise. Check that tracking is still installed.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
