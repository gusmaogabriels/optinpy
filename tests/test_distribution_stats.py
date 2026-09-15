"""Keep distribution counts distinct from usage and handle counter discontinuities."""
import importlib.util
import json
from pathlib import Path

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
