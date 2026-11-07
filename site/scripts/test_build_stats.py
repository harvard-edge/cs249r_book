"""Constructed GA4 report regressions; these tests never contact analytics."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def harness(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("stats_under_test", Path(__file__).with_name("build_stats.py"))
    stats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats)
    real_cache = stats.CACHE_PATH
    before = real_cache.read_bytes()
    monkeypatch.setattr(stats, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(stats, "CACHE_PATH", tmp_path / "nested/site/config/stats-cache.json")
    monkeypatch.setattr(stats, "PUBLISHED_STATS_URL", "")
    state = SimpleNamespace(
        stats=stats,
        report={"readers": 1000, "countries": 211, "readers_monthly": 100},
        published={"readers": "1,000", "countries": "212", "readers_monthly": "100"},
    )
    for key in ("GOOGLE_APPLICATION_CREDENTIALS", "GA4_SERVICE_ACCOUNT_JSON", "REQUIRE_GA4"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "constructed-test-path-not-opened")
    monkeypatch.setattr(stats.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("network forbidden"))
    monkeypatch.setattr(stats, "fetch_published", lambda: copy.deepcopy(state.published))
    monkeypatch.setattr(stats, "fetch_ga4", lambda: copy.deepcopy(state.report))
    for name in ("fetch_stars", "fetch_merged_prs", "fetch_subscribers"):
        monkeypatch.setattr(stats, name, lambda: None)
    for name in ("count_slide_decks", "count_slide_svgs", "count_tinytorch_modules"):
        monkeypatch.setattr(stats, name, lambda: 1)
    monkeypatch.setattr(stats, "read_latest_lab", lambda: (1, "Constructed lab"))
    monkeypatch.setattr(stats, "read_vol2_chapters", lambda: {})
    monkeypatch.setattr(stats, "read_last_updated", lambda: "Constructed date")

    def run():
        result = stats.main()
        cache = json.loads(stats.CACHE_PATH.read_text())
        if result == 0:
            state.published = cache["display"]
        return result, cache

    state.run = run
    yield state
    assert real_cache.read_bytes() == before


def test_country_correction_does_not_inherit_legacy_reader_alarm(harness):
    harness.published["diag_clamp_streak"] = "7"
    result, cache = harness.run()
    assert result == 0
    assert cache["values"]["countries"] == 211
    assert cache["display"]["countries"] == "211"
    assert cache["source"]["countries"] == "ga4"
    assert cache["values"]["diag_readers_drop_streak"] == 0
    assert "diag_clamp_streak" not in cache["display"]


@pytest.mark.parametrize("readers", [999, 950, 1000, 1100])
def test_ordinary_reader_changes_publish_current_report(harness, readers):
    harness.report["readers"] = readers
    result, cache = harness.run()
    assert result == 0
    assert cache["values"]["readers"] == readers
    assert cache["source"]["readers"] == "ga4"
    assert cache["values"]["diag_readers_drop_streak"] == 0
    assert cache["values"]["diag_ga4_status"] == "fresh"


def test_material_reader_drop_blocks_at_eighth_fresh_observation_without_inflating_display(harness):
    harness.report["readers"] = 900
    for observation in range(1, harness.stats.READER_DROP_ALARM_OBSERVATIONS + 1):
        result, cache = harness.run()
        assert result == int(observation >= harness.stats.READER_DROP_ALARM_OBSERVATIONS)
        assert cache["values"]["readers"] == 900
        assert cache["values"]["diag_readers_reference"] == 1000
        assert cache["values"]["diag_readers_drop_streak"] == observation


def test_recovery_within_operational_tolerance_clears_reader_streak(harness):
    harness.published.update(diag_readers_reference="1,000", diag_readers_drop_streak="7")
    harness.report["readers"] = 950
    result, cache = harness.run()
    assert result == 0
    assert cache["values"]["diag_readers_drop_streak"] == 0
    assert cache["values"]["readers"] == 950


def test_missing_offline_data_preserves_reader_evidence_and_last_success(harness, monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS")
    harness.published.update(diag_readers_reference="1,000", diag_readers_drop_streak="7",
                             diag_ga4_last_success="2026-11-01T00:00:00+00:00")
    harness.report = {}
    result, cache = harness.run()
    assert result == 0
    assert cache["values"]["diag_ga4_status"] == "stale:offline"
    assert cache["values"]["diag_readers_drop_streak"] == 7
    assert cache["values"]["diag_readers_reference"] == 1000
    assert cache["values"]["diag_ga4_last_success"] == "2026-11-01T00:00:00+00:00"


def test_required_missing_data_cannot_clear_streak_or_count_as_recovery(harness, monkeypatch):
    monkeypatch.setenv("REQUIRE_GA4", "1")
    harness.published.update(diag_readers_reference="1,000", diag_readers_drop_streak="7")
    harness.report = {}
    result, cache = harness.run()
    assert result == 1
    assert cache["values"]["diag_ga4_status"] == "stale:unavailable"
    assert cache["values"]["diag_readers_drop_streak"] == 7
    assert cache["values"]["diag_readers_reference"] == 1000
    harness.report = {"readers": 900, "countries": 211, "readers_monthly": 100}
    result, cache = harness.run()
    assert result == 1
    assert cache["values"]["diag_readers_drop_streak"] == 8


@pytest.mark.parametrize("key", ["readers", "countries", "readers_monthly"])
@pytest.mark.parametrize("bad", [None, True, -1, 1.5, "211"])
def test_required_malformed_metric_is_not_published_as_a_fresh_count(harness, monkeypatch, key, bad):
    monkeypatch.setenv("REQUIRE_GA4", "1")
    old = harness.published[key]
    harness.report[key] = bad
    result, cache = harness.run()
    assert result == 1
    assert cache["display"][key] == old
    assert cache["source"][key] == "published"
    assert cache["values"]["diag_ga4_status"] == "stale:unavailable"


@pytest.mark.parametrize("missing", ["readers", "countries", "readers_monthly"])
def test_required_missing_metric_blocks_publication(harness, monkeypatch, missing):
    monkeypatch.setenv("REQUIRE_GA4", "1")
    del harness.report[missing]
    assert harness.run()[0] == 1


def test_required_missing_credentials_blocks_even_if_cached_numbers_exist(harness, monkeypatch):
    monkeypatch.setenv("REQUIRE_GA4", "1")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS")
    harness.report = {}
    assert harness.run()[0] == 1


def test_required_valid_report_can_publish(harness, monkeypatch):
    monkeypatch.setenv("REQUIRE_GA4", "1")
    result, cache = harness.run()
    assert result == 0
    assert cache["display"]["countries"] == "211"


@pytest.mark.parametrize("payload", [None, [], "invalid report"])
def test_required_nonobject_report_keeps_reader_evidence(harness, monkeypatch, payload):
    monkeypatch.setenv("REQUIRE_GA4", "1")
    harness.published.update(diag_readers_reference="1,000", diag_readers_drop_streak="7")
    harness.report = payload
    result, cache = harness.run()
    assert result == 1
    assert cache["values"]["diag_readers_drop_streak"] == 7


def test_reader_drop_just_beyond_tolerance_is_an_observation(harness):
    harness.report["readers"] = 949
    result, cache = harness.run()
    assert result == 0
    assert cache["values"]["readers"] == 949
    assert cache["values"]["diag_readers_drop_streak"] == 1


def test_fresh_zero_is_a_count_not_missing_data(harness, monkeypatch):
    monkeypatch.setenv("REQUIRE_GA4", "1")
    harness.report = {key: 0 for key in harness.stats.GA4_METRICS}
    result, cache = harness.run()
    assert result == 0
    assert cache["values"]["diag_ga4_status"] == "fresh"
    assert all(cache["values"][key] == 0 for key in harness.stats.GA4_METRICS)
    assert cache["values"]["diag_readers_drop_streak"] == 1


def test_latest_published_fallback_can_be_lower_than_committed_cache(harness, monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS")
    harness.stats.CACHE_PATH.parent.mkdir(parents=True)
    harness.stats.CACHE_PATH.write_text(json.dumps({"values": {"countries": 220},
                                                 "source": {"countries": "ga4"}}))
    harness.report = {}
    result, cache = harness.run()
    assert result == 0
    assert cache["values"]["countries"] == 212
    assert cache["source"]["countries"] == "published"


@pytest.mark.parametrize("baseline", [None, [], {}, {"readers": "1,000"},
    {"readers": "1,000", "countries": "212", "readers_monthly": "100", "diag_readers_drop_streak": "invalid"}])
def test_required_bad_published_baseline_preserves_cache_before_other_fetches(harness, monkeypatch, baseline):
    monkeypatch.setenv("REQUIRE_GA4", "1")
    monkeypatch.setattr(harness.stats, "PUBLISHED_STATS_URL", "https://stats.invalid/stats.json")
    harness.published = baseline
    harness.stats.CACHE_PATH.parent.mkdir(parents=True)
    before = '{"values":{"diag_readers_reference":1000,"diag_readers_drop_streak":7}}\n'
    harness.stats.CACHE_PATH.write_text(before)
    monkeypatch.setattr(harness.stats, "fetch_ga4", lambda: pytest.fail("baseline must block first"))
    result, _ = harness.run()
    assert result == 1
    assert harness.stats.CACHE_PATH.read_text() == before


def test_required_valid_baseline_merges_reader_evidence_and_current_measurement(harness, monkeypatch):
    monkeypatch.setenv("REQUIRE_GA4", "1")
    monkeypatch.setattr(harness.stats, "PUBLISHED_STATS_URL", "https://stats.invalid/stats.json")
    harness.published.update(diag_readers_reference="1,000", diag_readers_drop_streak="2")
    harness.report["readers"] = 900
    result, cache = harness.run()
    assert result == 0
    assert cache["values"]["diag_readers_drop_streak"] == 3
    assert cache["values"]["diag_readers_reference"] == 1000
    assert cache["values"]["readers"] == 900


def test_offline_missing_baseline_retains_cached_evidence_as_stale(harness, monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS")
    monkeypatch.setattr(harness.stats, "PUBLISHED_STATS_URL", "https://stats.invalid/stats.json")
    harness.published = {}
    harness.report = {}
    harness.stats.CACHE_PATH.parent.mkdir(parents=True)
    harness.stats.CACHE_PATH.write_text(json.dumps({"values": {
        "readers": 1000, "countries": 212, "readers_monthly": 100,
        "diag_readers_reference": 1000, "diag_readers_drop_streak": 7}}))
    result, cache = harness.run()
    assert result == 0
    assert cache["values"]["diag_readers_drop_streak"] == 7
    assert cache["values"]["diag_ga4_status"] == "stale:offline"


def test_required_first_deployment_without_baseline_url_uses_fresh_data(harness, monkeypatch):
    monkeypatch.setenv("REQUIRE_GA4", "1")
    harness.published = {}
    result, cache = harness.run()
    assert result == 0
    assert cache["values"]["diag_readers_reference"] == harness.report["readers"]


def test_ga4_query_failure_is_reported_without_network_or_secret_access(harness, monkeypatch):
    spec = importlib.util.spec_from_file_location("ga4_fetch_under_test", Path(__file__).with_name("build_stats.py"))
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)

    class FailingClient:
        def run_report(self, request):
            raise RuntimeError("constructed query failure")

    monkeypatch.setitem(sys.modules, "google.analytics.data_v1beta", SimpleNamespace(BetaAnalyticsDataClient=FailingClient))
    monkeypatch.setitem(sys.modules, "google.analytics.data_v1beta.types", SimpleNamespace(
        DateRange=lambda **kw: kw, Dimension=lambda **kw: kw,
        Metric=lambda **kw: kw, RunReportRequest=lambda **kw: kw))
    assert original.fetch_ga4() == {}
    assert any("GA4 query failed" in message for message in original.warnings)


def test_live_workflow_modes_and_offline_regressions_are_wired():
    import yaml

    root = Path(__file__).resolve().parents[2]
    for workflow, command in (("site-publish-live.yml", "quarto render"),
                              ("site-refresh-stats.yml", "python3 site/scripts/refresh_site_data.py")):
        document = yaml.safe_load((root / ".github/workflows" / workflow).read_text())
        steps = [step for job in document["jobs"].values() for step in job.get("steps", [])
                 if command in step.get("run", "")]
        assert steps
        assert all(step.get("env", {}).get("REQUIRE_GA4") == "1" for step in steps)
    validate = yaml.safe_load((root / ".github/workflows/site-validate-dev.yml").read_text())
    steps = [step for job in validate["jobs"].values() for step in job.get("steps", [])]
    assert any("site/scripts/test_build_stats.py" in step.get("run", "") for step in steps)
    assert all("REQUIRE_GA4" not in step.get("env", {}) for step in steps)
