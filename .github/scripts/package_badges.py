"""Render README download snapshots from the existing collector; no network calls."""
from datetime import date, datetime, timedelta
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


def observations(snapshot):
    """Keep missing counts unknown, package assets separate, and partial PyPI explicit."""
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
        if pypi["status"] in ("available", "stale") and pypi["daily"]:
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
        result[f"{identity}-pypi"] = dict(label="PyPI downloads (30d)", message=message, detail=detail, color=color)
    return result


def render(snapshot):
    """Preserve existing SVG URLs for links from older README versions."""
    return {name + ".svg": svg(**value) for name, value in observations(snapshot).items()}


def endpoints(snapshot):
    """Use the standard Shields renderer with only already-public download data."""
    result = {}
    for name, value in observations(snapshot).items():
        badge = {"schemaVersion": 1, "label": value["label"], "message": value["message"],
                 "color": value["color"], "namedLogo": "github" if name.endswith("-github") else "pypi",
                 "logoColor": "white", "style": "flat"}
        if badge["message"] == "no data":
            badge["message"] = "unavailable"
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
