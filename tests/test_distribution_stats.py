"""Keep distribution counts distinct from usage and handle counter discontinuities."""
import importlib.util
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

path = Path(__file__).resolve().parents[1] / ".github/scripts/distribution_stats.py"
spec = importlib.util.spec_from_file_location("distribution_stats", path)
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)


def snapshot(count=10, identity=1, when="2026-09-15T00:00:00+00:00"):
    return {"repository": "owner/repo", "observed_at": when,
            "assets": [{"asset_id": identity, "name": "p.whl", "kind": "package", "download_count": count}]}


def test_first_observation_is_a_baseline():
    change = stats.compare(snapshot(100), None)["assets"][0]
    assert change["status"] == "baseline" and change["download_increase"] is None


def test_counter_difference_is_between_observations():
    change = stats.compare(snapshot(12), snapshot(10))["assets"][0]
    assert change["status"] == "observed" and change["download_increase"] == 2


def test_replaced_asset_is_not_a_negative_download():
    changes = stats.compare(snapshot(1, identity=2), snapshot(100, identity=1))
    assert changes["removed_asset_ids"] == [1]
    assert changes["assets"][0]["status"] == "baseline"
    decreased = stats.compare(snapshot(1), snapshot(100))["assets"][0]
    assert decreased["status"] == "counter_decreased" and decreased["download_increase"] is None


def test_collection_excludes_drafts_and_separates_checksum_downloads():
    def fetch(path, **kwargs):
        if "releases" not in path:
            return {"stargazers_count": 5}
        release = {"id": 2, "tag_name": "v1", "prerelease": True, "draft": False,
                   "assets": [{"id": 1, "state": "uploaded", "name": "p.whl", "download_count": 3},
                              {"id": 2, "state": "uploaded", "name": "SHA256SUMS.txt", "download_count": 7}]}
        return [release, {**release, "draft": True}]
    result = stats.collect("owner/repo", fetch=fetch)
    assert len(result["assets"]) == 2
    assert [item["kind"] for item in result["assets"]] == ["package", "other"]
    assert result["local_cli_usage"]["status"] == "not_observed"


def test_saved_history_retains_prior_observation(tmp_path):
    stats.save(snapshot(10), tmp_path)
    stats.save(snapshot(12, when="2026-09-16T00:00:00+00:00"), tmp_path)
    assert len(list((tmp_path / "observations").glob("*.json"))) == 2
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert latest["changes"]["assets"][0]["download_increase"] == 2
    with pytest.raises(ValueError, match="mix repositories"):
        stats.save({**snapshot(), "repository": "another/repo"}, tmp_path)


NOW = datetime(2026, 9, 16, 7, 23, tzinfo=timezone.utc)


def pypi_payload(rows=None):
    rows = [("2026-09-15", 5), ("2026-09-14", 3), ("2026-09-08", 20)] if rows is None else rows
    return {"package": "optinpy", "type": "overall_downloads",
            "data": [{"category": "without_mirrors", "date": day, "downloads": count}
                     for day, count in rows]}


def http_failure(code):
    def fetch(url):
        raise HTTPError(url, code, "unavailable", {}, None)
    return fetch


def test_pypi_windows_use_source_dates_and_expose_missing_days():
    result = stats.pypi_downloads("Optinpy", now=NOW, fetch=lambda _: pypi_payload())
    assert result["status"] == "available"
    assert result["mirrors"] == "excluded"
    assert result["data_through"] == "2026-09-15"
    assert result["windows"]["1"]["reported_downloads"] == 5
    assert result["windows"]["7"] == {
        "start_date": "2026-09-09", "end_date": "2026-09-15",
        "reported_downloads": 8, "reported_days": 2}
    assert result["windows"]["30"]["reported_downloads"] == 28
    assert [row["date"] for row in result["daily"]] == ["2026-09-08", "2026-09-14", "2026-09-15"]


@pytest.mark.parametrize("code,status", [(404, "no_data"), (429, "unavailable"), (503, "unavailable")])
def test_missing_pypi_data_is_never_zero(code, status):
    result = stats.pypi_downloads("optinpy", now=NOW, fetch=http_failure(code))
    assert result["status"] == status
    assert result["windows"] is None and result["data_through"] is None


def test_pypi_failure_keeps_prior_data_with_original_timestamp():
    before = stats.pypi_downloads("optinpy", now=NOW, fetch=lambda _: pypi_payload())
    tomorrow = NOW.replace(day=17)
    result = stats.pypi_downloads("optinpy", before, now=tomorrow, fetch=http_failure(429))
    assert result["status"] == "stale" and result["error"] == "rate_limited"
    assert result["fetched_at"] == before["fetched_at"]
    assert result["daily"] == before["daily"] and result["windows"] == before["windows"]
    assert result["last_attempt_at"] == tomorrow.isoformat()


@pytest.mark.parametrize("failure", [False, True])
def test_pypi_fetches_at_most_once_per_utc_day(failure):
    calls = []
    def fetch(url):
        calls.append(url)
        if failure:
            return http_failure(404)(url)
        return pypi_payload()
    before = stats.pypi_downloads("optinpy", now=NOW, fetch=fetch)
    result = stats.pypi_downloads("optinpy", before, now=NOW.replace(hour=23), fetch=fetch)
    assert result == before and len(calls) == 1
    stats.pypi_downloads("optinpy", result, now=NOW.replace(day=17), fetch=fetch)
    assert len(calls) == 2


@pytest.mark.parametrize("payload", [
    {"package": "another-package", "type": "overall_downloads", "data": []},
    {"package": "optinpy", "type": "overall_downloads", "data": [
        {"category": "with_mirrors", "date": "2026-09-15", "downloads": 10}]},
    pypi_payload([("2026-09-15", 1), ("2026-09-15", 2)]),
    pypi_payload([("2026-09-15", -1)]),
    pypi_payload([("2026-09-15", True)]),
    pypi_payload([("2026-09-18", 1)]),
])
def test_invalid_pypi_counts_cannot_inflate_metrics(payload):
    result = stats.pypi_downloads("optinpy", now=NOW, fetch=lambda _: payload)
    assert result["status"] == "unavailable" and result["error"] == "invalid_response"
    assert result["windows"] is None


def test_empty_pypi_response_reports_no_data():
    result = stats.pypi_downloads("optinpy", now=NOW, fetch=lambda _: pypi_payload([]))
    assert result["status"] == "no_data" and result["windows"] is None


def test_pypi_outage_does_not_drop_github_snapshot(tmp_path):
    def github(path, **kwargs):
        return [] if "releases" in path else {"stargazers_count": 49}
    result = stats.collect("owner/repo", fetch=github, observed_at=NOW.isoformat(),
                           pypi_package="optinpy", fetch_pypi=http_failure(503))
    stats.save(result, tmp_path)
    saved = json.loads((tmp_path / "latest.json").read_text())
    assert saved["stars"] == 49 and saved["pypi_downloads"]["status"] == "unavailable"
