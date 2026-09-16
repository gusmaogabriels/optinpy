"""Render README download snapshots from the existing collector; no network calls."""
from datetime import date, datetime, timedelta, timezone
from html import escape
import json
from pathlib import Path


def svg(label, message, detail, *, color="#007ec6"):
    """Small self-contained SVG with escaped text and an accessible description."""
    left = 12 + 7 * len(label)
    right = 12 + 7 * len(message)
    description = escape(f"{label}: {message}. {detail}", quote=True)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{left + right}" height="20" '
        f'role="img" aria-label="{description}">\n<title>{description}</title>\n'
        f'<rect width="{left + right}" height="20" rx="3" fill="{color}"/>\n'
        f'<path d="M3 0H{left}V20H3Q0 20 0 17V3Q0 0 3 0" fill="#555"/>\n'
        '<g fill="#fff" text-anchor="middle" font-family="DejaVu Sans Mono,monospace" font-size="11">\n'
        f'<text x="{left / 2:g}" y="14">{escape(label)}</text>\n'
        f'<text x="{left + right / 2:g}" y="14">{escape(message)}</text>\n'
        '</g>\n</svg>\n'
    )


def daily_month(pypi, package, today):
    """Sum a fresh PyPIStats event series; zero-download days have no rows.

    Upstream update_recent_stats sums this same table over 30 days ending
    yesterday: https://github.com/psf/pypistats.org/blob/main/pypistats/tasks/pypi.py
    The cutoff follows its normal daily schedule, not the last nonzero event.
    """
    if (not package or pypi.get("package") != package
            or pypi.get("source") != f"https://pypistats.org/api/packages/{package}/overall?mirrors=false"
            or pypi.get("mirrors") != "excluded"
            or pypi.get("status") != "available" or not pypi.get("daily")):
        return None
    try:
        fetched = datetime.fromisoformat(pypi["fetched_at"])
    except (KeyError, TypeError, ValueError):
        return None
    if fetched.tzinfo is None:
        return None
    fetched_day = fetched.astimezone(timezone.utc).date()
    if not 0 <= (today - fetched_day).days <= 2:
        return None
    end = fetched_day - timedelta(days=1)
    start = end - timedelta(days=29)
    count = sum(row["downloads"] for row in pypi["daily"]
                if start <= date.fromisoformat(row["date"]) <= end)
    return dict(
        label="PyPI downloads/month", message=f"{count:,}", color="#007ec6",
        detail=(f"Sum of PyPIStats daily records fetched {pypi['fetched_at']}, {start} through {end}. "
                "Cutoff inferred from the source's normal previous-UTC-day update schedule; "
                "upstream reporting delays are not detectable from sparse rows. "
                "Zero-download dates are omitted by this endpoint. "
                "Known mirrors excluded; repeats and automation included."))


def observations(snapshot):
    """Keep unavailable counts unknown and respect each source's reporting format."""
    today = datetime.fromisoformat(snapshot["observed_at"]).date()
    result = {}
    for entry in snapshot["packages"]:
        identity = entry["id"]
        github = entry["github_release_downloads"]
        if github["status"] in ("available", "stale"):
            count = sum(asset["download_count"] for asset in github["assets"] if asset["kind"] == "package")
            observed = datetime.fromisoformat(github["observed_at"]).date()
            stale = github["status"] == "stale" or (today - observed).days > 2
            message = f"{count:,}" + (" (stale)" if stale else "")
            detail = (f"Observed {github['observed_at']}. Uploaded wheels and source packages only; "
                      "checksums and generated source archives excluded. Includes repeats and automation.")
            color = "#9f6000" if stale else "#007ec6"
        else:
            message, color = "unavailable", "#777"
            detail = "No successful GitHub download observation; this is not a zero count."
        result[f"{identity}-github"] = dict(label="GitHub downloads", message=message, detail=detail, color=color)

        pypi = entry["pypi_downloads"]
        recent = entry.get('pypi_recent') or {}
        daily = daily_month(pypi, entry.get('pypi_package'), today)
        has_recent = recent.get('status') in ('available', 'stale') and recent.get('counts') is not None
        recent_stale = has_recent and (recent['status'] == 'stale' or
                                      (today - datetime.fromisoformat(recent['fetched_at']).date()).days > 2)
        label = 'PyPI downloads (30d)'
        if has_recent and (not recent_stale or daily is None):
            label = 'PyPI downloads/month'
            fetched = datetime.fromisoformat(recent['fetched_at']).date()
            stale = recent['status'] == 'stale' or (today-fetched).days > 2
            message = f"{recent['counts']['last_month']:,}" + (' (stale)' if stale else '')
            detail = (f"Source-reported last-month total fetched {recent['fetched_at']}. "
                      'Exact period dates are not provided. Known mirrors excluded; repeats and automation included.')
            color = '#9f6000' if stale else '#007ec6'
        elif daily is not None:
            result[f"{identity}-pypi"] = daily
            continue
        elif pypi["status"] in ("available", "stale") and pypi["daily"]:
            end = date.fromisoformat(pypi["data_through"])
            start = end - timedelta(days=29)
            rows = [row for row in pypi["daily"] if start <= date.fromisoformat(row["date"]) <= end]
            count = sum(row["downloads"] for row in rows)
            flags = []
            if len(rows) < 30:
                flags.append("partial")
            if pypi["status"] == "stale" or (today - end).days > 2:
                flags.append("stale")
            message = f"{count:,}" + (f" ({', '.join(flags)})" if flags else "")
            detail = (f"Reported days: {len(rows)} in the source window {start} through {end}. "
                      "Missing days are unknown. Known mirrors excluded; repeats and automation included.")
            color = "#9f6000" if flags else "#007ec6"
        else:
            message = "no data" if pypi["status"] == "no_data" else "unavailable"
            detail, color = "No usable PyPI download observation; this is not a zero count.", "#777"
        result[f"{identity}-pypi"] = dict(label=label, message=message, detail=detail, color=color)
    return result


def render(snapshot):
    """Preserve existing SVG URLs for links from older README versions."""
    return {name + ".svg": svg(**value) for name, value in observations(snapshot).items()}


def endpoints(snapshot):
    """Show current totals compactly; retain historical detail in the feed and SVGs."""
    result = {}
    for name, value in observations(snapshot).items():
        badge = {"schemaVersion": 1, "label": value["label"], "message": value["message"],
                 "color": value["color"], "namedLogo": "github" if name.endswith("-github") else "pypi",
                 "logoColor": "white", "style": "flat"}
        if badge["message"] == "no data":
            badge["message"] = "unavailable"
        if name.endswith("-pypi"):
            if any(flag in badge["message"] for flag in ("partial", "stale")):
                badge["message"] = "unavailable"
            if badge["message"] == "unavailable":
                badge["label"] = "PyPI downloads"
                badge["color"] = "#777"
        result[name + ".json"] = json.dumps(badge, indent=2) + "\n"
    return result


def save(snapshot, destination):
    output = render(snapshot)
    output.update(endpoints(snapshot))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, body in output.items():
        temporary = destination / (name + ".tmp")
        temporary.write_text(body, encoding="utf-8")
        temporary.replace(destination / name)
