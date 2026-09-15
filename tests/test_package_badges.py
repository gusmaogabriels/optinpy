"""README counters must preserve the collector's count and missing-data semantics."""
from copy import deepcopy
from datetime import date, timedelta
import importlib.util
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
    assert text == ["GitHub download snapshot", "5"]
    assert "2026-09-15T09:00:00+00:00" in detail


def test_pypi_recomputes_window_and_labels_missing_days(snapshot):
    text, detail = fields(badges.render(snapshot)["example-pypi.svg"])
    assert text == ["PyPI 30d reported", "4 (partial)"]
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
    assert fields((tmp_path / "another-github.svg").read_text())[0][-1] == "0"
    snapshot["packages"][0]["github_release_downloads"]["assets"][0]["download_count"] = 7
    badges.save(snapshot, tmp_path)
    assert fields((tmp_path / "example-github.svg").read_text())[0][-1] == "10"
    assert not list(tmp_path.glob("*.tmp"))
