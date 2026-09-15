"""Central collection must not duplicate requests or turn missing data into zero."""
from datetime import datetime, timezone
import importlib.util
import json
import subprocess
from pathlib import Path
from urllib.error import HTTPError

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("package_stats", ROOT / ".github/scripts/package_stats.py")
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)
REGISTRY = json.loads((ROOT / ".github/package-registry.json").read_text())
NOW = datetime(2026, 9, 16, 7, 23, tzinfo=timezone.utc)


def payload(package, count=5):
    return {"package": package, "type": "overall_downloads", "data": [
        {"category": "without_mirrors", "date": "2026-09-15", "downloads": count}]}


def fetch(url):
    package = url.split("/packages/", 1)[1].split("/", 1)[0]
    assert url.endswith("/overall?mirrors=false")
    return payload(package)


def github(path, **kwargs):
    return [] if "releases" in path else {"stargazers_count": 7}


def test_confirmed_registry_and_no_local_usage_claim():
    report = stats.collect_packages(REGISTRY, now=NOW, fetch=fetch)
    assert [row["id"] for row in report["packages"]] == ["optinpy", "mkin4py", "xl2py"]
    assert report["local_cli_usage"] == {"status": "not_observed"}


def test_each_package_is_requested_once_and_cached_for_same_utc_day():
    calls = []
    def counted(url):
        calls.append(url)
        return fetch(url)
    first = stats.collect_packages(REGISTRY, now=NOW, fetch=counted)
    second = stats.collect_packages(REGISTRY, first, now=NOW.replace(hour=23), fetch=counted)
    assert len(calls) == 3
    assert [row["pypi_downloads"] for row in second["packages"]] == [row["pypi_downloads"] for row in first["packages"]]
    stats.collect_packages(REGISTRY, second, now=NOW.replace(day=17), fetch=counted)
    assert len(calls) == 6


def test_package_failures_do_not_discard_other_packages_or_become_zero():
    def mixed(url):
        if "/optinpy/" in url:
            raise HTTPError(url, 404, "No data", {}, None)
        if "/xl2py/" in url:
            raise HTTPError(url, 429, "Rate limit", {}, None)
        return payload("mkin4py", 0)
    rows = {row["id"]: row["pypi_downloads"] for row in stats.collect_packages(REGISTRY, now=NOW, fetch=mixed)["packages"]}
    assert rows["optinpy"]["status"] == "no_data" and rows["optinpy"]["windows"] is None
    assert rows["xl2py"]["status"] == "unavailable" and rows["xl2py"]["windows"] is None
    assert rows["mkin4py"]["windows"]["1"]["reported_downloads"] == 0


def test_cutover_reuses_legacy_optinpy_attempt_even_when_404():
    def missing(url):
        raise HTTPError(url, 404, "No data", {}, None)
    legacy = {"repository": stats.REPOSITORY, "pypi_downloads": stats.distribution.pypi_downloads("optinpy", now=NOW, fetch=missing)}
    calls = []
    def counted(url):
        calls.append(url)
        return fetch(url)
    report = stats.collect_packages(REGISTRY, bootstrap=legacy, now=NOW.replace(hour=9), fetch=counted)
    assert len(calls) == 2 and not any("/optinpy/" in url for url in calls)
    assert report["packages"][0]["pypi_downloads"]["status"] == "no_data"


def test_failed_refresh_retains_last_good_data_and_date():
    before = stats.collect_packages(REGISTRY, now=NOW, fetch=fetch)
    def failed(url):
        raise HTTPError(url, 503, "Unavailable", {}, None)
    after = stats.collect_packages(REGISTRY, before, now=NOW.replace(day=17), fetch=failed)
    for old, new in zip(before["packages"], after["packages"]):
        assert new["pypi_downloads"]["status"] == "stale"
        assert new["pypi_downloads"]["daily"] == old["pypi_downloads"]["daily"]
        assert new["pypi_downloads"]["fetched_at"] == old["pypi_downloads"]["fetched_at"]


def test_corrupted_or_wrong_identity_cache_cannot_supply_counts():
    previous = stats.collect_packages(REGISTRY, now=NOW, fetch=fetch)
    previous["packages"][0]["pypi_downloads"]["daily"][0]["downloads"] = -99
    previous["packages"][1]["repository"] = "another/owner"
    calls = []
    def counted(url):
        calls.append(url)
        return fetch(url)
    report = stats.collect_packages(REGISTRY, previous, now=NOW, fetch=counted)
    assert len(calls) == 2
    assert all(row["pypi_downloads"]["daily"][0]["downloads"] == 5 for row in report["packages"])


@pytest.mark.parametrize("field,value", [("id", "../../data"), ("pypi_package", "https://example.com"), ("repository", "unreviewed/repo")])
def test_registry_rejects_unreviewed_inputs_before_requests(field, value):
    registry = json.loads(json.dumps(REGISTRY))
    registry["packages"][0][field] = value
    with pytest.raises(ValueError):
        stats.collect_packages(registry, now=NOW, fetch=lambda _: pytest.fail("network request"))


def test_duplicate_package_registry_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        stats.registry_entries({"schema_version": 1, "packages": REGISTRY["packages"] * 2})


def test_combined_run_preserves_legacy_github_metrics_and_uses_one_pypi_observation(tmp_path):
    calls = []
    def counted(url):
        calls.append(url)
        return fetch(url)
    report = stats.run(REGISTRY, tmp_path, now=NOW, fetch_pypi=counted, fetch_github=github)
    legacy = json.loads((tmp_path / "latest.json").read_text())
    aggregate = json.loads((tmp_path / "packages/latest.json").read_text())
    assert len(calls) == 3
    assert legacy["stars"] == 7 and legacy["assets"] == []
    assert legacy["pypi_downloads"] == report["packages"][0]["pypi_downloads"]
    assert aggregate == report
    assert len(list((tmp_path / "packages/observations").glob("*.json"))) == 1


def test_github_outage_does_not_discard_pypi_observations(tmp_path):
    def failed(*args, **kwargs):
        raise RuntimeError("GitHub API collection failed")
    stats.run(REGISTRY, tmp_path, now=NOW, fetch_pypi=fetch, fetch_github=github)
    legacy = (tmp_path / "latest.json").read_bytes()
    report = stats.run(REGISTRY, tmp_path, now=NOW.replace(day=17), fetch_pypi=fetch, fetch_github=failed)
    assert report["github_metrics_status"] == "unavailable"
    assert (tmp_path / "latest.json").read_bytes() == legacy
    assert all(row["pypi_downloads"]["status"] == "available" for row in report["packages"])
    assert len(list((tmp_path / "packages/observations").glob("*.json"))) == 2


def test_history_is_not_overwritten(tmp_path):
    snapshot = stats.collect_packages(REGISTRY, now=NOW, fetch=fetch)
    stats.save_packages(snapshot, tmp_path)
    with pytest.raises(FileExistsError):
        stats.save_packages(snapshot, tmp_path)


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired("gh", 60), KeyError("assets"), TypeError("bad shape")])
def test_github_timeout_or_schema_error_keeps_pypi_data(tmp_path, failure):
    def failed(*args, **kwargs):
        raise failure
    report = stats.run(REGISTRY, tmp_path, now=NOW, fetch_pypi=fetch, fetch_github=failed)
    assert report["github_metrics_status"] == "unavailable"
    assert len(json.loads((tmp_path / "packages/latest.json").read_text())["packages"]) == 3


@pytest.mark.parametrize("broken", ["latest.json", "packages/latest.json"])
def test_corrupt_cache_recovers_without_discarding_other_cache(tmp_path, broken):
    stats.run(REGISTRY, tmp_path, now=NOW, fetch_pypi=fetch, fetch_github=github)
    (tmp_path / broken).write_text("{broken")
    report = stats.run(REGISTRY, tmp_path, now=NOW.replace(day=17), fetch_pypi=fetch, fetch_github=github)
    assert all(row["pypi_downloads"]["status"] == "available" for row in report["packages"])
    assert json.loads((tmp_path / "latest.json").read_text())["stars"] == 7
    assert len(list((tmp_path / "packages/observations").glob("*.json"))) == 2


def test_empty_success_then_outage_stays_unknown_and_cached_for_that_day():
    def empty(url):
        package = url.split("/packages/")[1].split("/")[0]
        return {"package": package, "type": "overall_downloads", "data": []}
    before = stats.collect_packages(REGISTRY, now=NOW, fetch=empty)
    def failed(url):
        raise HTTPError(url, 503, "Unavailable", {}, None)
    after = stats.collect_packages(REGISTRY, before, now=NOW.replace(day=17), fetch=failed)
    assert all(row["pypi_downloads"]["status"] == "unavailable" and row["pypi_downloads"]["windows"] is None for row in after["packages"])
    stats.collect_packages(REGISTRY, after, now=NOW.replace(day=17, hour=22), fetch=lambda _: pytest.fail("duplicate query"))


def test_valid_json_with_broken_legacy_shape_recovers(tmp_path):
    (tmp_path / "latest.json").write_text(json.dumps({"repository": stats.REPOSITORY, "assets": [{}]}))
    stats.run(REGISTRY, tmp_path, now=NOW, fetch_pypi=fetch, fetch_github=github)
    assert json.loads((tmp_path / "latest.json").read_text())["stars"] == 7


def test_output_for_another_repository_is_not_overwritten(tmp_path):
    previous = json.dumps({"repository": "another/project", "assets": []})
    (tmp_path / "latest.json").write_text(previous)
    with pytest.raises(ValueError, match="another repository"):
        stats.run(REGISTRY, tmp_path, now=NOW, fetch_pypi=lambda _: pytest.fail("network request"), fetch_github=github)
    assert (tmp_path / "latest.json").read_text() == previous
