"""Record public distribution counters; this does not measure local CLI usage."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess


def github(path, *, paginate=False):
    arguments = ["gh", "api", path]
    if paginate:
        arguments += ["--paginate", "--slurp"]
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=60)
    if result.returncode:
        # Never include credentials, headers or arbitrary remote diagnostics.
        raise RuntimeError("GitHub API collection failed; previous snapshots are retained")
    data = json.loads(result.stdout)
    return [item for page in data for item in page] if paginate else data


def collect(repository, fetch=github, observed_at=None):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("repository must be owner/name")
    info = fetch(f"repos/{repository}")
    releases = fetch(f"repos/{repository}/releases?per_page=100", paginate=True)
    assets = []
    for release in releases:
        if release["draft"]:
            continue
        for asset in release["assets"]:
            if asset["state"] != "uploaded":
                continue
            count = asset["download_count"]
            if type(count) is not int or count < 0:
                raise ValueError("invalid GitHub download counter")
            assets.append({"asset_id": asset["id"], "release_id": release["id"],
                           "tag": release["tag_name"], "prerelease": release["prerelease"],
                           "name": asset["name"], "download_count": count,
                           "kind": "package" if asset["name"].endswith((".whl", ".tar.gz")) else "other"})
    stars = info["stargazers_count"]
    if type(stars) is not int or stars < 0:
        raise ValueError("invalid GitHub star counter")
    return {"schema_version": 1, "repository": repository,
            "observed_at": observed_at or datetime.now(timezone.utc).isoformat(),
            "stars": stars, "assets": sorted(assets, key=lambda item: item["asset_id"]),
            "scope": "Public GitHub release-asset downloads and expressed interest; includes automation and test downloads",
            "local_cli_usage": {"status": "not_observed"},
            "pypi_downloads": {"status": "not_collected"},
            "source_clones": {"status": "not_collected"}}


def compare(snapshot, previous):
    """Compare stable asset IDs, without treating first observations as new use."""
    prior = {item["asset_id"]: item for item in previous["assets"]} if previous else {}
    changes = []
    current_ids = set()
    for asset in snapshot["assets"]:
        identity = asset["asset_id"]
        current_ids.add(identity)
        before = prior.get(identity)
        difference = None if before is None else asset["download_count"] - before["download_count"]
        status = "baseline" if before is None else "counter_decreased" if difference < 0 else "observed"
        changes.append({"asset_id": identity, "name": asset["name"], "kind": asset["kind"],
                        "status": status, "download_increase": difference if status == "observed" else None})
    return {"previous_observed_at": previous["observed_at"] if previous else None,
            "current_observed_at": snapshot["observed_at"], "assets": changes,
            "removed_asset_ids": sorted(set(prior) - current_ids)}


def save(snapshot, destination):
    destination = Path(destination)
    history = destination / "observations"
    latest = destination / "latest.json"
    previous = json.loads(latest.read_text()) if latest.exists() else None
    if previous and previous["repository"] != snapshot["repository"]:
        raise ValueError("refusing to mix repositories in one history")
    snapshot["changes"] = compare(snapshot, previous)
    history.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(snapshot["observed_at"]).strftime("%Y%m%dT%H%M%S%fZ")
    text = json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n"
    observation = history / f"{stamp}.json"
    if observation.exists():
        raise ValueError("observation already exists")
    observation.write_text(text)
    temporary = latest.with_suffix(".tmp")
    temporary.write_text(text)
    temporary.replace(latest)
    (destination / "README.md").write_text(
        "# Optinpy distribution statistics\n\n"
        "Daily public GitHub counters, collected by the repository's statistics workflow.\n"
        "`latest.json` contains the latest observation and changes since the prior observation; "
        "`observations/` retains the history. Timestamps are UTC.\n\n"
        "Download counters include repeated, automated and test downloads. They do not count "
        "installations, people or local solver runs. Checksum downloads are marked `other`, "
        "separately from package assets. Stars measure expressed interest.\n\n"
        "A first observation is a baseline. Replaced assets have new IDs; missing assets and "
        "decreased counters are flagged rather than reported as negative downloads. Check "
        "`observed_at` for freshness; failed collections retain the last successful snapshot. "
        "This feed does not currently collect PyPI downloads, private clone traffic, or CLI telemetry.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="gusmaogabriels/optinpy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = collect(args.repository)
    save(snapshot, args.output)
    print(f"Recorded {len(snapshot['assets'])} release assets at {snapshot['observed_at']}; local CLI use is not observed")
