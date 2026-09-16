"""README counters must preserve the collector's count and missing-data semantics."""
from copy import deepcopy
from datetime import date, timedelta
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

SPEC = importlib.util.spec_from_file_location(
    "package_badges", Path(__file__).parents[1] / ".github/scripts/package_badges.py")
badges = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(badges)


@pytest.fixture
def snapshot():
    return {"observed_at": "2026-09-15T10:00:00+00:00", "packages": [{
        "id": "example",
        "github_release_downloads": {
            "status": "available", "observed_at": "2026-09-15T09:00:00+00:00",
            "assets": [{"kind": "package", "download_count": 2},
                       {"kind": "package", "download_count": 3},
                       {"kind": "other", "download_count": 99}]},
        "pypi_downloads": {
            "status": "available", "data_through": "2026-09-14",
            "daily": [{"date": "2026-08-15", "downloads": 1000},
                      {"date": "2026-09-14", "downloads": 4}],
            "windows": {"30": {"reported_downloads": 99999}}}}]}


def fields(svg):
    root = ET.fromstring(svg)
    namespace = {"s": "http://www.w3.org/2000/svg"}
    return [node.text for node in root.findall("s:g/s:text", namespace)], root.find("s:title", namespace).text


def test_github_counts_only_package_assets_and_preserves_observation_date(snapshot):
    text, detail = fields(badges.render(snapshot)["example-github.svg"])
    assert text == ["GitHub downloads", "5"]
    assert "2026-09-15T09:00:00+00:00" in detail


def test_pypi_recomputes_window_and_labels_missing_days(snapshot):
    text, detail = fields(badges.render(snapshot)["example-pypi.svg"])
    assert text == ["PyPI downloads (30d)", "4 (partial)"]
    assert "Reported days: 1" in detail
    assert "2026-08-16 through 2026-09-14" in detail


@pytest.mark.parametrize("status,expected", [("no_data", "no data"), ("unavailable", "unavailable")])
def test_missing_data_never_becomes_zero(snapshot, status, expected):
    row = snapshot["packages"][0]
    row["pypi_downloads"] = {"status": status, "daily": []}
    row["github_release_downloads"] = {"status": "unavailable", "assets": None}
    result = badges.render(snapshot)
    assert fields(result["example-pypi.svg"])[0][-1] == expected
    assert fields(result["example-github.svg"])[0][-1] == "unavailable"


def test_explicit_observed_zero_stays_zero(snapshot):
    row = snapshot["packages"][0]
    row["github_release_downloads"]["assets"] = []
    end = date(2026, 9, 14)
    row["pypi_downloads"]["daily"] = [
        {"date": str(end - timedelta(days=i)), "downloads": 0} for i in range(30)]
    result = badges.render(snapshot)
    assert fields(result["example-github.svg"])[0][-1] == "0"
    assert fields(result["example-pypi.svg"])[0][-1] == "0"


@pytest.mark.parametrize("source", ["github_release_downloads", "pypi_downloads"])
def test_retained_failure_preserves_count_and_marks_stale(snapshot, source):
    snapshot["packages"][0][source]["status"] = "stale"
    suffix = "github" if source.startswith("github") else "pypi"
    message = fields(badges.render(snapshot)[f"example-{suffix}.svg"])[0][-1]
    assert message.startswith("5" if suffix == "github" else "4")
    assert "stale" in message


def test_old_source_dates_are_stale_even_after_successful_fetch(snapshot):
    snapshot["observed_at"] = "2026-09-20T10:00:00+00:00"
    assert all("stale" in fields(value)[0][-1] for value in badges.render(snapshot).values())


def test_svg_text_is_escaped_and_cannot_create_markup():
    malicious = '<script>alert("x")</script>&'
    root = ET.fromstring(badges.svg(malicious, malicious, malicious))
    assert not root.findall(".//{http://www.w3.org/2000/svg}script")
    assert malicious in root.attrib["aria-label"]


def test_save_keeps_each_package_separate_and_updates_existing_files(snapshot, tmp_path):
    second = deepcopy(snapshot["packages"][0])
    second["id"] = "another"
    second["github_release_downloads"]["assets"] = []
    snapshot["packages"].append(second)
    badges.save(snapshot, tmp_path)
    assert len(list(tmp_path.glob("*.svg"))) == 4
    assert len(list(tmp_path.glob("*.json"))) == 4
    assert fields((tmp_path / "another-github.svg").read_text())[0][-1] == "0"
    snapshot["packages"][0]["github_release_downloads"]["assets"][0]["download_count"] = 7
    badges.save(snapshot, tmp_path)
    assert fields((tmp_path / "example-github.svg").read_text())[0][-1] == "10"
    assert not list(tmp_path.glob("*.tmp"))


def test_shields_endpoints_withhold_partial_totals_without_analytics_links(snapshot):
    payloads = {name: json.loads(body) for name, body in badges.endpoints(snapshot).items()}
    github = payloads["example-github.json"]
    pypi = payloads["example-pypi.json"]
    assert github["message"] == "5" and github["namedLogo"] == "github"
    assert pypi["message"] == "unavailable" and pypi["namedLogo"] == "pypi"
    assert pypi["label"] == "PyPI downloads" and pypi["color"] == "#777"
    for payload in payloads.values():
        assert payload["schemaVersion"] == 1 and payload["style"] == "flat"
        assert "analytics" not in json.dumps(payload) and "link" not in payload


def test_shields_unknown_is_unavailable_not_zero(snapshot):
    snapshot["packages"][0]["pypi_downloads"] = {"status": "no_data", "daily": []}
    assert json.loads(badges.endpoints(snapshot)["example-pypi.json"])["message"] == "unavailable"


@pytest.mark.parametrize("source", ["daily", "recent"])
@pytest.mark.parametrize("state", ["fresh", "failed", "aged"])
@pytest.mark.parametrize("count", [0, 1234])
def test_shields_monthly_totals_require_complete_current_data(snapshot, source, state, count):
    row = snapshot["packages"][0]
    if source == "recent":
        row["pypi_recent"] = {
            "status": "stale" if state == "failed" else "available",
            "fetched_at": "2026-09-15T09:00:00+00:00",
            "counts": {"last_month": count},
        }
    else:
        end = date(2026, 9, 14)
        row["pypi_downloads"]["daily"] = [
            {"date": str(end - timedelta(days=i)), "downloads": count if i == 0 else 0}
            for i in range(30)
        ]
        row["pypi_downloads"]["status"] = "stale" if state == "failed" else "available"
    if state == "aged":
        snapshot["observed_at"] = "2026-09-20T10:00:00+00:00"
    original = deepcopy(snapshot)
    payload = json.loads(badges.endpoints(snapshot)["example-pypi.json"])
    assert payload["message"] == (f"{count:,}" if state == "fresh" else "unavailable")
    if state != "fresh":
        assert payload["label"] == "PyPI downloads" and payload["color"] == "#777"
        assert "stale" in fields(badges.render(snapshot)["example-pypi.svg"])[0][-1]
    assert snapshot == original


def test_shields_partial_stale_badge_is_compact_and_preserves_recorded_count(snapshot):
    snapshot["observed_at"] = "2026-09-20T10:00:00+00:00"
    payload = json.loads(badges.endpoints(snapshot)["example-pypi.json"])
    assert payload["label"] == "PyPI downloads" and payload["message"] == "unavailable"
    assert fields(badges.render(snapshot)["example-pypi.svg"])[0][-1] == "4 (partial, stale)"


@pytest.fixture
def sparse_pypistats(snapshot):
    snapshot["observed_at"] = "2026-09-16T16:46:05+00:00"
    row = snapshot["packages"][0]
    row["pypi_package"] = "example"
    row["pypi_recent"] = {"status": "unavailable", "counts": None, "error": "rate_limited"}
    row["pypi_downloads"].update(
        package="example", source="https://pypistats.org/api/packages/example/overall?mirrors=false",
        mirrors="excluded", fetched_at="2026-09-16T12:38:00+00:00",
        data_through="2026-09-09", daily=[{"date": "2026-09-09", "downloads": 1}])
    return snapshot


def test_sparse_success_recovers_monthly_badge_after_recent_rate_limit(sparse_pypistats):
    original = deepcopy(sparse_pypistats)
    payload = json.loads(badges.endpoints(sparse_pypistats)["example-pypi.json"])
    assert payload["label"] == "PyPI downloads/month" and payload["message"] == "1"
    text, detail = fields(badges.render(sparse_pypistats)["example-pypi.svg"])
    assert text[-1] == "1"
    assert "2026-08-17 through 2026-09-15" in detail
    assert "inferred" in detail and "Zero-download dates are omitted" in detail
    assert sparse_pypistats == original


def test_sparse_month_uses_fetch_window_not_last_download(sparse_pypistats):
    daily = sparse_pypistats["packages"][0]["pypi_downloads"]
    daily["daily"] = [{"date": "2026-08-16", "downloads": 1000},
                      {"date": "2026-08-17", "downloads": 2},
                      {"date": "2026-09-15", "downloads": 3},
                      {"date": "2026-09-16", "downloads": 1000}]
    assert badges.observations(sparse_pypistats)["example-pypi"]["message"] == "5"
    # No reported events in the window is a reported zero, given a successful series.
    daily["daily"] = daily["daily"][:1]
    assert badges.observations(sparse_pypistats)["example-pypi"]["message"] == "0"


@pytest.mark.parametrize("change", [
    {"status": "stale"}, {"status": "unavailable"}, {"status": "no_data"}, {"daily": []},
    {"fetched_at": "2026-09-12T12:38:00+00:00"}, {"fetched_at": None},
    {"fetched_at": "invalid"}, {"fetched_at": "2026-09-16T12:38:00"},
    {"fetched_at": "2026-09-17T12:38:00+00:00"}, {"source": "other"},
    {"package": "another"}, {"mirrors": "included"},
])
def test_sparse_fallback_requires_fresh_success_with_correct_provenance(sparse_pypistats, change):
    sparse_pypistats["packages"][0]["pypi_downloads"].update(change)
    assert json.loads(badges.endpoints(sparse_pypistats)["example-pypi.json"])["message"] == "unavailable"


@pytest.mark.parametrize("state,expected", [("fresh", "9"), ("failed", "1"), ("aged", "1")])
def test_prefer_fresh_recent_then_fresh_daily(sparse_pypistats, state, expected):
    sparse_pypistats["packages"][0]["pypi_recent"] = {
        "status": "stale" if state == "failed" else "available", "counts": {"last_month": 9},
        "fetched_at": "2026-09-12T09:00:00+00:00" if state == "aged" else "2026-09-16T09:00:00+00:00"}
    assert json.loads(badges.endpoints(sparse_pypistats)["example-pypi.json"])["message"] == expected


def test_cached_daily_window_does_not_slide_or_refresh_its_timestamp(sparse_pypistats):
    sparse_pypistats["observed_at"] = "2026-09-18T10:00:00+00:00"
    observation = badges.observations(sparse_pypistats)["example-pypi"]
    assert observation["message"] == "1"
    assert "2026-08-17 through 2026-09-15" in observation["detail"]
    sparse_pypistats["observed_at"] = "2026-09-19T10:00:00+00:00"
    assert json.loads(badges.endpoints(sparse_pypistats)["example-pypi.json"])["message"] == "unavailable"
