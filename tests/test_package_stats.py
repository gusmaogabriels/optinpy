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
    assert [row["id"] for row in report["packages"]] == ["optinpy", "mkin4py", "xl2py", "kinn"]
    assert report["local_cli_usage"] == {"status": "not_observed"}


def test_each_package_is_requested_once_and_cached_for_same_utc_day():
    calls = []
    def counted(url):
        calls.append(url)
        return fetch(url)
    first = stats.collect_packages(REGISTRY, now=NOW, fetch=counted)
    second = stats.collect_packages(REGISTRY, first, now=NOW.replace(hour=23), fetch=counted)
    assert len(calls) == len(REGISTRY["packages"])
    assert [row["pypi_downloads"] for row in second["packages"]] == [row["pypi_downloads"] for row in first["packages"]]
    stats.collect_packages(REGISTRY, second, now=NOW.replace(day=17), fetch=counted)
    assert len(calls) == 2 * len(REGISTRY["packages"])


def test_package_failures_do_not_discard_other_packages_or_become_zero():
    def mixed(url):
        if "/optinpy/" in url:
            raise HTTPError(url, 404, "No data", {}, None)
        if "/xl2py/" in url:
            raise HTTPError(url, 429, "Rate limit", {}, None)
        return payload(url.split("/packages/")[1].split("/")[0], 0)
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
    assert len(calls) == len(REGISTRY["packages"]) - 1
    assert not any("/optinpy/" in url for url in calls)
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
    assert len(calls) == len(REGISTRY["packages"])
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
    assert len(json.loads((tmp_path / "packages/latest.json").read_text())["packages"]) == len(REGISTRY["packages"])


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


def asset(identity=1, count=10, name="example.whl", **overrides):
    return {"id": identity, "state": "uploaded", "name": name, "download_count": count, **overrides}


def repository_fetch(count=10, *, overrides=None, failures=(), calls=None):
    """All responses are fixtures: no public API or GitHub token is used in tests."""
    def request(path, **kwargs):
        if calls is not None:
            calls.append(path)
        repository = "/".join(path.split("/")[1:3])
        if repository in failures:
            raise subprocess.TimeoutExpired("gh", 60)
        if "releases" not in path:
            return {"stargazers_count": 7}
        uploaded = (overrides or {}).get(repository, [asset(count=count)])
        return [{"id": 100, "tag_name": "v2.0.0a1", "prerelease": True,
                 "draft": False, "assets": uploaded}]
    return request


def github_rows(report):
    return {row["id"]: row["github_release_downloads"] for row in report["packages"]}


def test_github_each_repo_collected_once_and_optinpy_result_reused_in_legacy(tmp_path):
    calls, pypi_calls = [], []
    def pypi(url):
        pypi_calls.append(url)
        return fetch(url)
    report = stats.run(REGISTRY, tmp_path, now=NOW, fetch_pypi=pypi,
                       fetch_github=repository_fetch(calls=calls))
    for entry in REGISTRY["packages"]:
        assert calls.count(f"repos/{entry['repository']}") == 1
        assert calls.count(f"repos/{entry['repository']}/releases?per_page=100") == 1
    assert len(calls) == 2 * len(REGISTRY["packages"])
    assert len(pypi_calls) == len(REGISTRY["packages"])
    legacy = json.loads((tmp_path / "latest.json").read_text())
    assert legacy["assets"] == github_rows(report)["optinpy"]["assets"]
    assert legacy["stars"] == 7
    stats.run(REGISTRY, tmp_path, now=NOW.replace(hour=23), fetch_pypi=pypi,
              fetch_github=repository_fetch(calls=calls))
    assert len(pypi_calls) == len(REGISTRY["packages"])


def test_github_first_asset_observation_is_baseline_not_a_download_increase(tmp_path):
    report = stats.run(REGISTRY, tmp_path, now=NOW, fetch_pypi=fetch, fetch_github=repository_fetch(count=90))
    for repository in github_rows(report).values():
        assert repository["status"] == "available"
        assert repository["assets"][0]["download_count"] == 90
        assert repository["changes"]["previous_observed_at"] is None
        assert repository["changes"]["assets"][0]["status"] == "baseline"
        assert repository["changes"]["assets"][0]["download_increase"] is None
        assert repository["observed_at"] == repository["last_attempt_at"] == NOW.isoformat()


def test_github_cutover_compares_optinpy_against_existing_legacy_observation(tmp_path):
    old = stats.distribution.collect(stats.REPOSITORY, fetch=repository_fetch(count=10),
                                     observed_at=NOW.replace(day=15).isoformat())
    stats.distribution.save(old, tmp_path)
    report = stats.run(REGISTRY, tmp_path, now=NOW, fetch_pypi=fetch, fetch_github=repository_fetch(count=13))
    rows = github_rows(report)
    assert rows["optinpy"]["changes"]["assets"][0]["download_increase"] == 3
    assert rows["mkin4py"]["changes"]["assets"][0]["status"] == "baseline"
    assert rows["xl2py"]["changes"]["previous_observed_at"] is None


def test_github_repo_failure_isolated_and_recovery_compares_last_success(tmp_path):
    first = stats.run(REGISTRY, tmp_path, now=NOW, fetch_pypi=fetch, fetch_github=repository_fetch(count=10))
    before = github_rows(first)["mkin4py"]
    second = stats.run(REGISTRY, tmp_path, now=NOW.replace(day=17), fetch_pypi=fetch,
                       fetch_github=repository_fetch(count=15, failures=("gusmaogabriels/mkin4py",)))
    rows = github_rows(second)
    assert rows["optinpy"]["status"] == rows["xl2py"]["status"] == "available"
    assert rows["mkin4py"] == {**before, "status": "stale", "last_attempt_at": NOW.replace(day=17).isoformat(), "error": "collection_failed"}
    assert all(row["pypi_downloads"]["status"] == "available" for row in second["packages"])
    recovered = stats.run(REGISTRY, tmp_path, now=NOW.replace(day=18), fetch_pypi=fetch, fetch_github=repository_fetch(count=25))
    changes = github_rows(recovered)["mkin4py"]["changes"]
    assert changes["previous_observed_at"] == NOW.isoformat()
    assert changes["assets"][0]["download_increase"] == 15


def test_github_first_failure_is_unavailable_with_unknown_assets(tmp_path):
    report = stats.run(REGISTRY, tmp_path, now=NOW, fetch_pypi=fetch,
                       fetch_github=repository_fetch(failures=("gusmaogabriels/xl2py",)))
    value = github_rows(report)["xl2py"]
    assert value == {"status": "unavailable", "repository": "gusmaogabriels/xl2py",
                     "source": "https://api.github.com/repos/gusmaogabriels/xl2py/releases",
                     "observed_at": None, "last_attempt_at": NOW.isoformat(),
                     "assets": None, "changes": None, "error": "collection_failed"}


def test_github_decreases_replacements_removals_and_checksums_stay_separate(tmp_path):
    repository = "gusmaogabriels/mkin4py"
    first = [asset(1, 10, "a.whl"), asset(2, 20, "b.tar.gz"), asset(3, 7, "SHA256SUMS.txt")]
    stats.run(REGISTRY, tmp_path, now=NOW, fetch_pypi=fetch, fetch_github=repository_fetch(overrides={repository: first}))
    second = [asset(1, 8, "a.whl"), asset(4, 3, "b.tar.gz"), asset(3, 11, "SHA256SUMS.txt")]
    report = stats.run(REGISTRY, tmp_path, now=NOW.replace(day=17), fetch_pypi=fetch,
                       fetch_github=repository_fetch(overrides={repository: second}))
    value = github_rows(report)["mkin4py"]
    changes = {row["asset_id"]: row for row in value["changes"]["assets"]}
    assert changes[1]["status"] == "counter_decreased" and changes[1]["download_increase"] is None
    assert changes[4]["status"] == "baseline" and changes[4]["download_increase"] is None
    assert changes[3]["kind"] == "other" and changes[3]["download_increase"] == 4
    assert value["changes"]["removed_asset_ids"] == [2]
    assert sum(row["download_count"] for row in value["assets"] if row["kind"] == "package") == 11


def test_github_no_release_or_no_uploaded_assets_is_successful_empty_observation(tmp_path):
    def empty(path, **kwargs):
        if "releases" not in path:
            return {"stargazers_count": 0}
        if "mkin4py" in path:
            return [{"id": 2, "draft": True, "assets": [asset()]},
                    {"id": 3, "draft": False, "assets": [asset(state="starter")]}]
        return []
    report = stats.run(REGISTRY, tmp_path, now=NOW, fetch_pypi=fetch, fetch_github=empty)
    for value in github_rows(report).values():
        assert value["status"] == "available" and value["assets"] == []
        assert value["changes"]["assets"] == [] and value["changes"]["removed_asset_ids"] == []


@pytest.mark.parametrize("field,value", [
    ("id", False), ("id", 0), ("id", 2**53), ("download_count", True),
    ("download_count", -1), ("download_count", 2**53), ("name", None),
    ("name", "unsafe\n.whl"), ("name", "x" * 256), ("name", "☃" * 171),
])
def test_invalid_github_asset_values_are_isolated_per_repository(tmp_path, field, value):
    bad = asset(**{field: value})
    report = stats.run(REGISTRY, tmp_path, now=NOW, fetch_pypi=fetch,
                       fetch_github=repository_fetch(overrides={"gusmaogabriels/mkin4py": [bad]}))
    rows = github_rows(report)
    assert rows["mkin4py"]["status"] == "unavailable"
    assert rows["optinpy"]["status"] == rows["xl2py"]["status"] == "available"


@pytest.mark.parametrize("bad_assets", [
    [asset(), asset()],
    [asset(index + 1) for index in range(101)],
    [asset(1, 2**53 - 1), asset(2, 1)],
])
def test_duplicate_oversized_or_unsafe_total_assets_are_rejected_without_truncation(tmp_path, bad_assets):
    report = stats.run(REGISTRY, tmp_path, now=NOW, fetch_pypi=fetch,
                       fetch_github=repository_fetch(overrides={"gusmaogabriels/xl2py": bad_assets}))
    value = github_rows(report)["xl2py"]
    assert value["status"] == "unavailable" and value["assets"] is None


@pytest.mark.parametrize("mutate", [
    lambda value: value.update(source="https://example.invalid/releases"),
    lambda value: value.update(observed_at="2099-01-01T00:00:00+00:00"),
    lambda value: value["assets"][0].update(kind="other"),
    lambda value: value["changes"]["assets"][0].update(status="observed", download_increase=-1),
    lambda value: value["changes"].update(removed_asset_ids=[1]),
])
def test_malformed_github_cache_is_not_reused_as_stale_or_a_delta_baseline(tmp_path, mutate):
    stats.run(REGISTRY, tmp_path, now=NOW, fetch_pypi=fetch, fetch_github=repository_fetch())
    latest = tmp_path / "packages/latest.json"
    previous = json.loads(latest.read_text())
    mutate(previous["packages"][1]["github_release_downloads"])
    latest.write_text(json.dumps(previous))
    report = stats.run(REGISTRY, tmp_path, now=NOW.replace(day=17), fetch_pypi=fetch,
                       fetch_github=repository_fetch(failures=("gusmaogabriels/mkin4py",)))
    assert github_rows(report)["mkin4py"]["status"] == "unavailable"
    assert github_rows(report)["optinpy"]["status"] == "available"


def test_aggregate_size_limit_preserves_both_previous_latest_files(tmp_path, monkeypatch):
    stats.run(REGISTRY, tmp_path, now=NOW, fetch_pypi=fetch, fetch_github=repository_fetch())
    before = {name: (tmp_path / name).read_bytes() for name in ("latest.json", "packages/latest.json")}
    monkeypatch.setattr(stats, "MAX_BYTES", 1024)
    with pytest.raises(ValueError, match="aggregate exceeds"):
        stats.run(REGISTRY, tmp_path, now=NOW.replace(day=17), fetch_pypi=fetch, fetch_github=repository_fetch())
    assert all((tmp_path / name).read_bytes() == body for name, body in before.items())


@pytest.mark.parametrize("draft,state", [("false", "uploaded"), (0, "uploaded"), (False, None)])
def test_malformed_release_filter_flags_do_not_look_like_empty_success(tmp_path, draft, state):
    healthy = repository_fetch()
    def malformed(path, **kwargs):
        response = healthy(path, **kwargs)
        if "mkin4py/releases" in path:
            response[0]["draft"] = draft
            response[0]["assets"][0]["state"] = state
        return response
    report = stats.run(REGISTRY, tmp_path, now=NOW, fetch_pypi=fetch, fetch_github=malformed)
    assert github_rows(report)["mkin4py"]["status"] == "unavailable"
    assert github_rows(report)["optinpy"]["status"] == "available"


def kinnlib_pypi_fixture(url):
    """The KINN project uses kinnlib; PyPI's unrelated kinn must never be queried."""
    package = url.split("/packages/", 1)[1].split("/", 1)[0]
    assert package != "kinn"
    assert package in {row["pypi_package"] for row in REGISTRY["packages"]}
    if url.endswith("/recent"):
        return {"package": package, "type": "recent_downloads",
                "data": {"last_day": 1, "last_week": 5, "last_month": 12}}
    return fetch(url)


def test_kinn_project_uses_kinnlib_sources_and_stable_badge_identity(tmp_path):
    calls, github_calls = [], []
    def counted(url):
        calls.append(url)
        return kinnlib_pypi_fixture(url)
    report = stats.run(REGISTRY, tmp_path, now=NOW, fetch_pypi=counted,
                       fetch_recent=counted, fetch_github=repository_fetch(count=17, calls=github_calls))
    row = next(row for row in report["packages"] if row["id"] == "kinn")
    assert {key: row[key] for key in ("id", "pypi_package", "repository")} == {
        "id": "kinn", "pypi_package": "kinnlib", "repository": "gusmaogabriels/kinn"}
    assert row["pypi_downloads"]["package"] == row["pypi_recent"]["package"] == "kinnlib"
    assert row["pypi_downloads"]["source"].endswith("/kinnlib/overall?mirrors=false")
    assert row["pypi_recent"]["source"].endswith("/kinnlib/recent")
    assert github_calls.count("repos/gusmaogabriels/kinn/releases?per_page=100") == 1
    badges = tmp_path / "packages/badges"
    assert json.loads((badges / "kinn-pypi.json").read_text())["message"] == "12"
    assert json.loads((badges / "kinn-github.json").read_text())["message"] == "17"
    assert not (badges / "kinnlib-pypi.json").exists()
    assert report["schema_version"] == 1
    stats.run(REGISTRY, tmp_path, now=NOW.replace(hour=23), fetch_pypi=counted,
              fetch_recent=counted, fetch_github=repository_fetch())
    assert len(calls) == 2 * len(REGISTRY["packages"])
    assert len(set(calls)) == len(calls)


@pytest.mark.parametrize("code,status", [(404, "no_data"), (429, "unavailable"), (503, "unavailable")])
def test_unobserved_kinnlib_downloads_remain_unknown_and_cached(tmp_path, code, status):
    calls = []
    def missing(url):
        calls.append(url)
        kinnlib_pypi_fixture(url)  # Reject accidental requests for the unrelated project.
        if "/kinnlib/" in url:
            raise HTTPError(url, code, "No observation", {}, None)
        return kinnlib_pypi_fixture(url)
    report = stats.run(REGISTRY, tmp_path, now=NOW, fetch_pypi=missing,
                       fetch_recent=missing, fetch_github=repository_fetch())
    row = next(row for row in report["packages"] if row["id"] == "kinn")
    assert row["pypi_downloads"]["status"] == row["pypi_recent"]["status"] == status
    assert row["pypi_downloads"]["windows"] is None and row["pypi_downloads"]["daily"] == []
    assert row["pypi_recent"]["counts"] is None
    assert json.loads((tmp_path / "packages/badges/kinn-pypi.json").read_text())["message"] == "unavailable"
    stats.run(REGISTRY, tmp_path, now=NOW.replace(hour=23), fetch_pypi=missing,
              fetch_recent=missing, fetch_github=repository_fetch())
    assert len(calls) == 2 * len(REGISTRY["packages"])


@pytest.mark.parametrize("rename_registry_field", [False, True])
def test_kinnlib_does_not_reuse_unrelated_kinn_counts(rename_registry_field):
    previous = stats.collect_packages(REGISTRY, now=NOW, fetch=kinnlib_pypi_fixture,
                                     fetch_recent=kinnlib_pypi_fixture)
    row = next(row for row in previous["packages"] if row["id"] == "kinn")
    if rename_registry_field:
        row["pypi_package"] = "kinn"
    for key in ("pypi_downloads", "pypi_recent"):
        row[key]["package"] = "kinn"
        row[key]["source"] = row[key]["source"].replace("/kinnlib/", "/kinn/")
    calls = []
    def wrong_response(url):
        calls.append(url)
        result = kinnlib_pypi_fixture(url)
        result["package"] = "kinn"
        return result
    report = stats.collect_packages(REGISTRY, previous, now=NOW,
                                    fetch=wrong_response, fetch_recent=wrong_response)
    row = next(row for row in report["packages"] if row["id"] == "kinn")
    assert len(calls) == 2 and all("/kinnlib/" in url for url in calls)
    assert row["pypi_downloads"]["status"] == row["pypi_recent"]["status"] == "unavailable"
    assert row["pypi_downloads"]["windows"] is None and row["pypi_recent"]["counts"] is None
