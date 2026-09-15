"""Bounded public GitHub asset observations, independent of PyPI downloads."""
from datetime import datetime, timezone
import subprocess

MAX_ASSETS = 100
MAX_INTEGER = 2**53 - 1
FAILURES = (RuntimeError, ValueError, KeyError, TypeError, AttributeError, OSError, subprocess.TimeoutExpired)


def integer(value, *, positive=False):
    if type(value) is not int or not (1 if positive else 0) <= value <= MAX_INTEGER:
        raise ValueError("invalid GitHub integer")
    return value


def text(value):
    if (not isinstance(value, str) or not value or len(value) > 255
            or len(value.encode("utf-8")) > 512 or not value.isprintable()):
        raise ValueError("invalid GitHub text")
    return value


def timestamp(value, now):
    if not isinstance(value, str) or len(value) > 40:
        raise ValueError("invalid GitHub observation time")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed > now:
        raise ValueError("invalid GitHub observation time")
    return parsed.astimezone(timezone.utc).isoformat()


def source(repository):
    return f"https://api.github.com/repos/{repository}/releases"


def assets(value):
    if not isinstance(value, list) or len(value) > MAX_ASSETS:
        raise ValueError("invalid GitHub asset list")
    result, ids, total = [], set(), 0
    for item in value:
        identity = integer(item["asset_id"], positive=True)
        name = text(item["name"])
        kind = "package" if name.endswith((".whl", ".tar.gz")) else "other"
        count = integer(item["download_count"])
        if identity in ids or type(item["prerelease"]) is not bool or item["kind"] != kind:
            raise ValueError("inconsistent GitHub asset")
        ids.add(identity)
        total += count
        integer(total)
        result.append({"asset_id": identity, "release_id": integer(item["release_id"], positive=True),
                       "tag": text(item["tag"]), "prerelease": item["prerelease"],
                       "name": name, "download_count": count, "kind": kind})
    return sorted(result, key=lambda item: item["asset_id"])


def changes(value, current, observed_at, now):
    """Validate retained interval metadata; never trust arbitrary cached strings."""
    if timestamp(value["current_observed_at"], now) != observed_at:
        raise ValueError("inconsistent GitHub change time")
    previous = value["previous_observed_at"]
    if previous is not None:
        previous = timestamp(previous, datetime.fromisoformat(observed_at))
    rows = value["assets"]
    removed = value["removed_asset_ids"]
    if not isinstance(rows, list) or len(rows) != len(current) or not isinstance(removed, list) or len(removed) > MAX_ASSETS:
        raise ValueError("invalid GitHub change list")
    by_id = {item["asset_id"]: item for item in current}
    normalized, seen = [], set()
    for row in rows:
        identity = integer(row["asset_id"], positive=True)
        item = by_id[identity]
        status, delta = row["status"], row["download_increase"]
        if identity in seen or row["name"] != item["name"] or row["kind"] != item["kind"]:
            raise ValueError("inconsistent GitHub change")
        if status == "observed":
            if previous is None or integer(delta) > item["download_count"]:
                raise ValueError("invalid GitHub download increase")
        elif status in ("baseline", "counter_decreased"):
            if delta is not None or (status == "counter_decreased" and previous is None):
                raise ValueError("invalid GitHub baseline or decrease")
        else:
            raise ValueError("invalid GitHub change status")
        seen.add(identity)
        normalized.append({"asset_id": identity, "name": item["name"], "kind": item["kind"],
                           "status": status, "download_increase": delta})
    removed = [integer(identity, positive=True) for identity in removed]
    if len(set(removed)) != len(removed) or set(removed) & set(by_id) or (removed and previous is None):
        raise ValueError("invalid removed GitHub assets")
    return {"previous_observed_at": previous, "current_observed_at": observed_at,
            "assets": sorted(normalized, key=lambda item: item["asset_id"]), "removed_asset_ids": sorted(removed)}


def prior(value, repository, now):
    try:
        if (value["repository"] != repository or value["source"] != source(repository)
                or value["status"] not in ("available", "stale")):
            return None
        attempted = timestamp(value["last_attempt_at"], now)
        observed = timestamp(value["observed_at"], datetime.fromisoformat(attempted))
        current = assets(value["assets"])
        return {"status": value["status"], "repository": repository, "source": source(repository),
                "observed_at": observed, "last_attempt_at": attempted, "assets": current,
                "changes": changes(value["changes"], current, observed, now)}
    except (KeyError, TypeError, ValueError, OverflowError, UnicodeError):
        return None


def legacy_prior(value, repository, now, distribution):
    """Existing Optinpy observations provide a valid baseline during cutover."""
    try:
        if value["repository"] != repository:
            return None
        observed = timestamp(value["observed_at"], now)
        current = assets(value["assets"])
        comparison = value.get("changes") or distribution.compare({"assets": current, "observed_at": observed}, None)
        return prior({"status": "available", "repository": repository, "source": source(repository),
                      "observed_at": observed, "last_attempt_at": observed, "assets": current,
                      "changes": comparison}, repository, now)
    except (KeyError, TypeError, ValueError, OverflowError, UnicodeError):
        return None


def collect(entry, previous, *, now, distribution, fetch):
    repository = entry["repository"]
    attempted = now.isoformat()
    def public_read(path, **kwargs):
        value = fetch(path, **kwargs)
        if path == f"repos/{repository}/releases?per_page=100":
            if not isinstance(value, list):
                raise ValueError("invalid GitHub release list")
            for release in value:
                if type(release["draft"]) is not bool:
                    raise ValueError("invalid GitHub draft flag")
                if release["draft"]:
                    continue
                if not isinstance(release["assets"], list) or any(
                        not isinstance(item["state"], str) for item in release["assets"]):
                    raise ValueError("invalid GitHub asset states")
        return value
    try:
        snapshot = distribution.collect(repository, fetch=public_read, observed_at=attempted)
        snapshot["assets"] = assets(snapshot["assets"])
        integer(snapshot["stars"])
        observation = {"status": "available", "repository": repository, "source": source(repository),
                       "observed_at": attempted, "last_attempt_at": attempted, "assets": snapshot["assets"],
                       "changes": distribution.compare(snapshot, previous)}
        # Validate generated and retained observations with the same contract.
        observation = prior(observation, repository, now)
        if observation is None:
            raise ValueError("invalid generated GitHub observation")
        return observation, snapshot
    except FAILURES:
        if previous is not None:
            return {**previous, "status": "stale", "last_attempt_at": attempted, "error": "collection_failed"}, None
        return {"status": "unavailable", "repository": repository, "source": source(repository),
                "observed_at": None, "last_attempt_at": attempted, "assets": None,
                "changes": None, "error": "collection_failed"}, None
