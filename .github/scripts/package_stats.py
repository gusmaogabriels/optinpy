"""One daily PyPI collector for the reviewed registry; never measures local runs."""
import argparse
from datetime import date, datetime, timezone
import importlib.util
import json
from pathlib import Path
import re

_spec = importlib.util.spec_from_file_location(
    "distribution_stats", Path(__file__).with_name("distribution_stats.py"))
distribution = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(distribution)

_github_spec = importlib.util.spec_from_file_location(
    "github_package_stats", Path(__file__).with_name("github_package_stats.py"))
github_packages = importlib.util.module_from_spec(_github_spec)
_github_spec.loader.exec_module(github_packages)

_badge_spec = importlib.util.spec_from_file_location(
    "package_badges", Path(__file__).with_name("package_badges.py"))
package_badges = importlib.util.module_from_spec(_badge_spec)
_badge_spec.loader.exec_module(package_badges)

_recent_spec = importlib.util.spec_from_file_location(
    'pypi_recent', Path(__file__).with_name('pypi_recent.py'))
pypi_recent = importlib.util.module_from_spec(_recent_spec)
_recent_spec.loader.exec_module(pypi_recent)

MAX_BYTES = 1024 * 1024
REPOSITORY = "gusmaogabriels/optinpy"


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("snapshot exceeds size limit")
    return json.loads(path.read_text(encoding="utf-8"))


def read_cache(path):
    try:
        value = read_json(path)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        # Invalid cached observations do not establish zero downloads. Refetch the
        # affected source, while retaining independently valid caches/history.
        return None


def previous_metrics(value):
    if not value:
        return None
    if value.get("repository") not in (None, REPOSITORY):
        raise ValueError("refusing to overwrite another repository's statistics")
    try:
        if value["repository"] != REPOSITORY or not isinstance(value["assets"], list):
            return None
        if datetime.fromisoformat(value["observed_at"]).tzinfo is None:
            return None
        ids = set()
        for asset in value["assets"]:
            if (type(asset["asset_id"]) is not int or asset["asset_id"] in ids
                    or type(asset["download_count"]) is not int or asset["download_count"] < 0):
                return None
            ids.add(asset["asset_id"])
        return value
    except (KeyError, TypeError, ValueError):
        return None


def registry_entries(registry):
    if not isinstance(registry, dict) or registry.get("schema_version") != 1:
        raise ValueError("unsupported package registry")
    entries = registry.get("packages")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 30:
        raise ValueError("invalid registry size")
    ids, names = set(), set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"id", "pypi_package", "repository"}:
            raise ValueError("invalid registry fields")
        identity, package, repository = (entry[key] for key in ("id", "pypi_package", "repository"))
        if not all(isinstance(value, str) for value in (identity, package, repository)):
            raise ValueError("invalid registry values")
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", identity):
            raise ValueError("invalid package identity")
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", package):
            raise ValueError("use canonical PyPI names")
        if not re.fullmatch(r"gusmaogabriels/[A-Za-z0-9_.-]+", repository):
            raise ValueError("unreviewed repository owner")
        if identity in ids or package in names:
            raise ValueError("duplicate package identity")
        ids.add(identity)
        names.add(package)
    if not any(entry == {"id": "optinpy", "pypi_package": "optinpy", "repository": REPOSITORY} for entry in entries):
        raise ValueError("Optinpy compatibility entry is required")
    return entries


def prior_observation(value, entry, now):
    """Ignore malformed/mismatched cache rows instead of poisoning a package fetch."""
    try:
        if not isinstance(value, dict) or value.get("package") != entry["pypi_package"]:
            return None
        if value.get("source") != f"https://pypistats.org/api/packages/{entry['pypi_package']}/overall?mirrors=false":
            return None
        if value.get("mirrors") != "excluded" or value.get("status") not in {"available", "stale", "no_data", "unavailable"}:
            return None
        attempted = datetime.fromisoformat(value["last_attempt_at"])
        if attempted.tzinfo is None or attempted > now:
            return None
        rows = value["daily"]
        if not isinstance(rows, list) or len(rows) > 400:
            return None
        days = set()
        for row in rows:
            day = date.fromisoformat(row["date"])
            if (day.isoformat() != row["date"] or day in days or day > now.date()
                    or type(row["downloads"]) is not int or not 0 <= row["downloads"] <= 2**53 - 1):
                return None
            days.add(day)
        if value["status"] in {"available", "stale"}:
            fetched = datetime.fromisoformat(value["fetched_at"])
            if fetched.tzinfo is None or fetched > attempted or not days or max(days).isoformat() != value["data_through"]:
                return None
        return value
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def collect_packages(registry, previous=None, bootstrap=None, *, now=None,
                     fetch=distribution.pypi_json, fetch_recent=None):
    entries = registry_entries(registry)
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    previous_rows = previous.get("packages", []) if isinstance(previous, dict) and previous.get("schema_version") == 1 else []
    if not isinstance(previous_rows, list):
        previous_rows = []
    packages = []
    for entry in entries:
        candidates = [row.get("pypi_downloads") for row in previous_rows if isinstance(row, dict)
                      and all(row.get(key) == value for key, value in entry.items())]
        # On cutover, reuse the previous Optinpy job's attempt (including a 404).
        # Do not fetch twice in the same UTC day just because the owner changed.
        if entry["id"] == "optinpy" and isinstance(bootstrap, dict) and bootstrap.get("repository") == REPOSITORY:
            candidates.append(bootstrap.get("pypi_downloads"))
        candidates = [checked for value in candidates if (checked := prior_observation(value, entry, now)) is not None]
        cached = max(candidates, key=lambda value: datetime.fromisoformat(value["last_attempt_at"])) if candidates else None
        observed = distribution.pypi_downloads(entry["pypi_package"], cached, now=now, fetch=fetch)
        row = {**entry, "pypi_downloads": observed}
        if fetch_recent is not None:
            recent_candidates = [pypi_recent.prior(value.get('pypi_recent'), entry['pypi_package'], now)
                for value in previous_rows if isinstance(value, dict)
                and all(value.get(key) == item for key, item in entry.items())]
            recent_candidates = [value for value in recent_candidates if value is not None]
            cached_recent = max(recent_candidates, key=lambda value: datetime.fromisoformat(value['last_attempt_at'])) if recent_candidates else None
            row['pypi_recent'] = pypi_recent.collect(entry['pypi_package'], cached_recent,
                                                   now=now, fetch=fetch_recent)
        packages.append(row)
    return {"schema_version": 1, "observed_at": now.isoformat(), "packages": packages,
            "local_cli_usage": {"status": "not_observed"},
            "scope": "Daily PyPI downloads excluding known mirrors; includes automation and repeat downloads. Not users, installs or local runs."}


def save_packages(snapshot, destination):
    destination = Path(destination)
    history = destination / "observations"
    history.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(snapshot["observed_at"]).strftime("%Y%m%dT%H%M%S%fZ")
    body = json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n"
    if len(body.encode("utf-8")) > MAX_BYTES:
        raise ValueError("package aggregate exceeds size limit")
    # Never overwrite history, including a repeated explicit observation timestamp.
    with (history / f"{stamp}.json").open("x", encoding="utf-8") as stream:
        stream.write(body)
    temporary = destination / "latest.tmp"
    temporary.write_text(body, encoding="utf-8")
    temporary.replace(destination / "latest.json")
    (destination / "README.md").write_text(
        "# Per-package PyPI download observations\n\n"
        "One GitHub Actions job owns collection for the reviewed `.github/package-registry.json`. "
        "`latest.json` is a public aggregate; `observations/` retains dated snapshots. "
        "Optinpy's legacy root feed reuses the same observation, with no second PyPI request.\n\n"
        "Each package also has a separate `github_release_downloads` observation. "
        "These are cumulative counters for uploaded assets, with .whl/.tar.gz packages "
        "separated from checksum/other files. They must not be added to PyPI windows. "
        "First asset observations are baselines; later changes compare stable asset IDs "
        "and explicitly flag decreases/removals. Failed repository requests retain valid "
        "prior observations as stale, or remain unavailable without a prior observation.\n\n"
        "PyPI Stats updates once daily and retains 180 days. Known mirrors are excluded; "
        "CI and repeated downloads remain included. These are downloads, not users, "
        "successful installations, or CLI executions. No local telemetry is collected.\n\n"
        "Check each package's `status`, `data_through`, `fetched_at` and `last_attempt_at`. "
        "Missing dates are unknown; explicit dated zeros are zero. A 404 is `no_data`, "
        "not zero downloads. Errors retain earlier data as `stale` when possible. "
        "Compute windows from dated rows; never add overlapping windows or snapshots. "
        "Use the newest observation per package/date and preserve gaps and revisions.\n\n"
        "`pypi_recent` separately retains the source's explicit last-day/week/month totals. "
        "The badges prefer these totals when available, so sparse daily rows are not "
        "treated as an incomplete monthly total. The endpoint does not supply exact "
        "period dates: none are inferred. Successful results are cached for the UTC day. "
        "Temporary failures of the recent-totals endpoint can retry on a later collection "
        "after at least one hour, honoring longer Retry-After delays. "
        "Collection is scheduled at 07:23, 13:23 and 19:23 UTC. "
        "A failed request retains the prior aggregate as stale; a 404 is not zero.\n\n"
        "Sources: https://pypistats.org/api/ and https://pypistats.org/faqs .\n",
        encoding="utf-8")


def run(registry, destination, *, now=None, fetch_pypi=distribution.pypi_json,
        fetch_github=distribution.github, fetch_recent=None):
    destination = Path(destination)
    previous = read_cache(destination / "packages/latest.json")
    legacy = read_cache(destination / "latest.json")
    legacy_metrics = previous_metrics(legacy)
    aggregate = collect_packages(registry, previous, legacy, now=now,
                                  fetch=fetch_pypi, fetch_recent=fetch_recent)
    observed = datetime.fromisoformat(aggregate["observed_at"])
    previous_rows = previous.get("packages", []) if previous and previous.get("schema_version") == 1 else []
    if not isinstance(previous_rows, list):
        previous_rows = []
    legacy_snapshot = None
    for entry in aggregate["packages"]:
        candidates = [github_packages.prior(row.get("github_release_downloads"), entry["repository"], observed)
                      for row in previous_rows if isinstance(row, dict)
                      and all(row.get(key) == entry[key] for key in ("id", "pypi_package", "repository"))]
        if entry["id"] == "optinpy":
            candidates.append(github_packages.legacy_prior(legacy_metrics, REPOSITORY, observed, distribution))
        candidates = [candidate for candidate in candidates if candidate is not None]
        cached = max(candidates, key=lambda value: datetime.fromisoformat(value["observed_at"])) if candidates else None
        entry["github_release_downloads"], snapshot = github_packages.collect(
            entry, cached, now=observed, distribution=distribution, fetch=fetch_github)
        if entry["id"] == "optinpy":
            legacy_snapshot = snapshot
    # This compatibility status continues to describe the legacy Optinpy feed only.
    aggregate["github_metrics_status"] = "available" if legacy_snapshot is not None else "unavailable"
    aggregate["scope"] = ("Separate PyPI daily downloads excluding known mirrors and cumulative GitHub release-asset counters. "
                          "Includes automation and repeats; do not combine sources or infer users, installs or local runs.")
    # Serialize/size-check the aggregate before replacing either latest file.
    if len((json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n").encode("utf-8")) > MAX_BYTES:
        raise ValueError("package aggregate exceeds size limit")
    if legacy_snapshot is not None:
        legacy_snapshot["pypi_downloads"] = next(row["pypi_downloads"] for row in aggregate["packages"] if row["id"] == "optinpy")
        distribution.save(legacy_snapshot, destination, previous=legacy_metrics)
    save_packages(aggregate, destination / "packages")
    package_badges.save(aggregate, destination / "packages/badges")
    return aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path(__file__).parents[1] / "package-registry.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(read_json(args.registry), args.output, fetch_recent=distribution.pypi_json)
    print("Recorded package download states: " + ", ".join(
        row["id"] + "=" + row["pypi_downloads"]["status"] for row in report["packages"]))
