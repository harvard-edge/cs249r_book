# MLSysBook — Unified site (landing)

This directory is the **Quarto source** for the public **landing and community** experience at [mlsysbook.ai](https://mlsysbook.ai): home, about, community, newsletter sync, and related shared pages (together with assets under `shared/`).

---

<!-- EARLY-RELEASE-CALLOUT:START -->
> [!NOTE]
> **📌 Early release (2026)**
>
> The unified site shipped with the **2026** MLSysBook refresh. Information architecture, copy, newsletter wiring, and styling are **actively iterated** as the book and sibling projects evolve.
>
> **Feedback:** [GitHub issues](https://github.com/harvard-edge/cs249r_book/issues) or pull requests.
>
> [![dev branch](https://img.shields.io/badge/branch-dev-orange?logo=git&logoColor=white)](https://github.com/harvard-edge/cs249r_book/tree/dev) [![live site](https://img.shields.io/badge/live_site-mlsysbook.ai-blue?logo=safari&logoColor=white)](https://mlsysbook.ai)
<!-- EARLY-RELEASE-CALLOUT:END -->

For build and deploy, see `.github/workflows/site-validate-dev.yml` and `site-publish-live.yml` in this repository.

Site stats use the current GA4 report, including downward corrections to reader
and country counts. Older, higher estimates are never substituted for fresh
readings. Live publishing and scheduled refreshes set `REQUIRE_GA4=1`: missing
credentials or missing/malformed `readers`, `countries`, or `readers_monthly`
metrics block publication. Offline development and validation may retain cached
fields, with `diag_ga4_status` marked stale; `diag_ga4_last_success` records the
last complete report rather than the cache file's regeneration time.
When `PUBLISHED_STATS_URL` is configured in required mode, that endpoint must
also supply a valid baseline, including any existing reader-health counters.
An unavailable or malformed baseline blocks before changing the cache, so an
older checkout cannot erase published evidence. A first deployment without a
configured baseline URL can initialize from a fresh report.

Reader health is separate from displayed counts. A reader estimate more than
5% below `diag_readers_reference` increments `diag_readers_drop_streak`; eight
fresh observations block publication for investigation. These are operational
review thresholds, not a statistical significance test or a claim about elapsed
time. Recovery within tolerance clears the streak; missing data preserves it.
Country corrections do not affect this reader-only counter. The reference is
comparison evidence, never a display floor; an investigated lasting report
change may require an explicit, evidence-backed reference update. This guard
flags suspicious reports but cannot prove whether tracking is installed.

Run `python3 -m pytest site/scripts/test_build_stats.py -q` for the offline
regressions also run by Site validation. They use constructed reports and block
network access; no analytics credentials are needed.
