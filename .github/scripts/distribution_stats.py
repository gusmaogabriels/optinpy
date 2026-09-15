"""Record public distribution counters; this does not measure local CLI usage."""
import argparse
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import re
import subprocess
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def pypi_json(url):
    request = Request(url, headers={
        "User-Agent": "optinpy-distribution-statistics/1 (+https://github.com/gusmaogabriels/optinpy)"})
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def pypi_downloads(package, previous=None, *, now=None, fetch=pypi_json):
    """Fetch one daily series, excluding known mirrors, with a UTC-day cache."""
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", package):
        raise ValueError("invalid PyPI package name")
    package = re.sub(r"[-_.]+", "-", package).lower()
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    source = f"https://pypistats.org/api/packages/{package}/overall?mirrors=false"
    if previous and (previous.get("package") != package or previous.get("source") != source):
        previous = None
    if previous and datetime.fromisoformat(previous["last_attempt_at"]).astimezone(timezone.utc).date() == now.date():
        return dict(previous)
    result = {"status": "no_data", "package": package, "source": source,
              "mirrors": "excluded", "last_attempt_at": now.isoformat(),
              "fetched_at": None, "data_through": None, "daily": [], "windows": None}
    try:
        payload = fetch(source)
        if (payload["package"] != package or payload["type"] != "overall_downloads"
                or not isinstance(payload["data"], list)):
            raise ValueError("unexpected PyPI Stats response")
        daily = {}
        for row in payload["data"]:
            day = date.fromisoformat(row["date"])
            count = row["downloads"]
            if (row["category"] != "without_mirrors" or day.isoformat() != row["date"]
                    or day > now.date() or day in daily or type(count) is not int or count < 0):
                raise ValueError("invalid PyPI Stats daily counter")
            daily[day] = count
        result["fetched_at"] = now.isoformat()
        result["daily"] = [{"date": day.isoformat(), "downloads": daily[day]} for day in sorted(daily)]
        if daily:
            end = max(daily)
            result.update(status="available", data_through=end.isoformat(), windows={})
            for days in (1, 7, 30):
                start = end - timedelta(days=days - 1)
                rows = [count for day, count in daily.items() if start <= day <= end]
                result["windows"][str(days)] = {
                    "start_date": start.isoformat(), "end_date": end.isoformat(),
                    "reported_downloads": sum(rows), "reported_days": len(rows)}
        return result
    except HTTPError as error:
        reason = "not_found" if error.code == 404 else "rate_limited" if error.code == 429 else "http_error"
    except (OSError, URLError):
        reason = "network_error"
    except (ValueError, KeyError, TypeError):
        reason = "invalid_response"
    # Keep prior observations visibly stale; an outage or a new package is not zero downloads.
    if previous and previous.get("fetched_at") and previous.get("daily") and previous.get("data_through"):
        result = {**previous, "status": "stale", "last_attempt_at": now.isoformat()}
    else:
        result["status"] = "no_data" if reason == "not_found" else "unavailable"
    result["error"] = reason
    return result


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


def collect(repository, fetch=github, observed_at=None, *, pypi_package=None, previous=None,
            fetch_pypi=pypi_json):
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
    observed_at = observed_at or datetime.now(timezone.utc).isoformat()
    downloads = {"status": "not_collected"}
    if pypi_package:
        downloads = pypi_downloads(pypi_package, (previous or {}).get("pypi_downloads"),
                                  now=datetime.fromisoformat(observed_at), fetch=fetch_pypi)
    return {"schema_version": 1, "repository": repository,
            "observed_at": observed_at,
            "stars": stars, "assets": sorted(assets, key=lambda item: item["asset_id"]),
            "scope": "Public distribution downloads and expressed interest; includes automation and test downloads",
            "local_cli_usage": {"status": "not_observed"},
            "pypi_downloads": downloads,
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


_READ_PREVIOUS = object()


def save(snapshot, destination, *, previous=_READ_PREVIOUS):
    destination = Path(destination)
    history = destination / "observations"
    latest = destination / "latest.json"
    if previous is _READ_PREVIOUS:
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
        "Daily public GitHub counters and PyPI download data, collected by the repository's statistics workflow.\n"
        "`latest.json` contains the latest observation and changes since the prior observation; "
        "`observations/` retains the history. Timestamps are UTC.\n\n"
        "Download counters include repeated, automated and test downloads. They do not count "
        "installations, people or local solver runs. Checksum downloads are marked `other`, "
        "separately from package assets. Stars measure expressed interest.\n\n"
        "A first observation is a baseline. Replaced assets have new IDs; missing assets and "
        "decreased counters are flagged rather than reported as negative downloads. Check "
        "`observed_at` for freshness; failed collections retain the last successful snapshot. "
        "This feed does not collect private clone traffic or CLI telemetry.\n\n"
        "## PyPI downloads\n\n"
        "[PyPI Stats](https://pypistats.org/packages/optinpy) supplies daily downloads excluding known mirrors. "
        "CI, repeat downloads and partial downloads can still be included; caches can hide installations. "
        "The [API](https://pypistats.org/api/) updates daily and retains 180 days; these snapshots archive its observations. "
        "Repeated collections reuse the PyPI result within the same UTC day.\n\n"
        "`pypi_downloads.daily` contains dated counts. The 1-, 7- and 30-day `windows` end at `data_through`, "
        "the latest date returned by the source, not necessarily yesterday. Totals sum reported rows; "
        "`reported_days` shows coverage and missing dates are not fabricated. Never sum rolling totals or "
        "repeat snapshots: use the newest observation for each date.\n\n"
        "Check `status`, `data_through`, `fetched_at` and `last_attempt_at` for freshness. New packages may "
        "have `no_data` until aggregation catches up. Unavailable or malformed responses retain prior data "
        "as `stale`, or `unavailable` when no prior data exists; they never become zero downloads. "
        "GitHub collection continues when the PyPI source is unavailable.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="gusmaogabriels/optinpy")
    parser.add_argument("--pypi-package", help="also collect public PyPI Stats downloads for this package")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    latest = args.output / "latest.json"
    previous = json.loads(latest.read_text()) if latest.exists() else None
    snapshot = collect(args.repository, pypi_package=args.pypi_package, previous=previous)
    save(snapshot, args.output)
    print(f"Recorded {len(snapshot['assets'])} release assets at {snapshot['observed_at']}; "
          f"PyPI: {snapshot['pypi_downloads']['status']}; local CLI use is not observed")
