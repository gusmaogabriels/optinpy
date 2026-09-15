# Central package statistics

`package_stats.py` is the sole scheduled PyPI collector for the reviewed
`.github/package-registry.json`: Optinpy, mkin4py and xl2py. It runs in the existing
GitHub Actions statistics job, writing public data to `distribution-statistics`:

- `packages/latest.json`: the multi-package aggregate consumed by the website.
- `packages/observations/`: append-only snapshots, retaining data beyond the source's
  180-day window and making later revisions inspectable.
- `packages/badges/`: public Shields JSON endpoints and compatible SVG snapshots.
- Root `latest.json` / `observations/`: the existing Optinpy release-download/star
  feed, with Optinpy's PyPI result reused from the central observation.

No second PyPI query is made for the legacy feed. On the first central run, the
collector imports the latest Optinpy attempt from that feed, including a same-day
404, and reuses it until the next UTC day. It preserves a valid newer per-package
cache on later runs. There is one collection owner and one existing workflow
concurrency group, so migration does not leave two scheduled collectors.

This workflow adds no package runtime dependency, CLI telemetry, website backend,
VM workload or credentials. It uses the existing repository-scoped Actions token
to read GitHub counters and publish only its own statistics branch. Browsers read
the public aggregate without credentials; they never query PyPI Stats directly.

## Data interpretation and failures

The API is `overall?mirrors=false` from PyPI Stats. Known mirrors are excluded;
CI/CD and repeated downloads remain included. These counts are not unique users,
installations, optimizer invocations or agent attribution. No Optinpy-vs-other-CLI
usage is inferred from downloads. `local_cli_usage.status` remains `not_observed`.

Each package retains `data_through`, `fetched_at`, `last_attempt_at`, dated `daily`
rows and source-window coverage. Empty/new-package results remain `no_data`;
outages are `unavailable`, or `stale` with last-good values retained. A failing
package does not discard other packages. GitHub counter failures preserve the old
legacy file and still publish independent PyPI observations, with
`github_metrics_status: unavailable`. The frontend independently evaluates age;
a stopped workflow cannot make an old count fresh.

Dates without rows are unknown. An explicit dated zero remains zero. Never sum
overlapping windows or repeated snapshots; select the newest observation per
package/date. Package ownership must be verified before adding registry entries:
the PyPI name `kinn`, for example, belongs to an unrelated project and is excluded.

## Checks

The badge renderer uses only the collected snapshot and makes no requests. GitHub
badges sum uploaded wheels and source packages, excluding checksums. PyPI badges
recompute the reported 30-day source window from dated rows and label partial
coverage explicitly. Unknown data stays `no data` or `unavailable`, never zero;
retained failures and source dates over two days old are labelled stale. SVG
accessible descriptions contain observation dates or source windows. The same
observations also produce JSON endpoints for standard Shields badges. Shields
reads only this public JSON; badge rendering adds no collector network calls or
credentials. README badges link to public GitHub releases and PyPI Stats pages.
They never link to the owner's authenticated analytics dashboard. Original SVG
URLs remain available for older README links. The same job refreshes both formats.

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_package_stats.py tests/test_distribution_stats.py
```

Tests cover cache reuse, cutover with initial 404, failure isolation, retained stale
data, malformed/incorrect cache identity, registry validation, legacy metric
compatibility, immutable history, and GitHub outages. All source calls are mocked.
The existing package CI includes these tests; no package release is required for
this workflow-only change.

Sources: [PyPI Stats API](https://pypistats.org/api/),
[mirror and CI limitations](https://pypistats.org/faqs).

## GitHub release downloads

```json
{
  "status": "available",
  "repository": "gusmaogabriels/optinpy",
  "source": "https://api.github.com/repos/gusmaogabriels/optinpy/releases",
  "observed_at": "2026-09-16T07:23:00+00:00",
  "last_attempt_at": "2026-09-16T07:23:00+00:00",
  "assets": [
    {
      "asset_id": 123,
      "release_id": 456,
      "tag": "v2.0.0a1",
      "prerelease": true,
      "name": "example.whl",
      "download_count": 7,
      "kind": "package"
    }
  ],
  "changes": {
    "previous_observed_at": null,
    "current_observed_at": "2026-09-16T07:23:00+00:00",
    "assets": [
      {
        "asset_id": 123,
        "name": "example.whl",
        "kind": "package",
        "status": "baseline",
        "download_increase": null
      }
    ],
    "removed_asset_ids": []
  }
}
```

The example above is illustrative, not a real download observation. Counts are
cumulative counters for **currently uploaded assets on public, non-draft releases**.
Pre-releases are included and labelled. Only `.whl` and `.tar.gz` assets have
`kind: "package"`; checksum and other assets have `kind: "other"`. GitHub's
automatically generated source archives are not uploaded release assets and are
not included. These counters cannot establish users, successful installations,
agent adoption or local executions. Repeat, CI and test downloads may be included.

Changes compare **stable asset IDs** between successful observations, independently
of each package's PyPI source dates. A new asset is a `baseline` with a null
increase; its entire cumulative counter is not reported as new usage. For an
existing asset, `observed` gives a nonnegative increase over the stated observation
interval. A decreased counter is `counter_decreased` with a null increase.
Replacements get new IDs; the previous IDs appear in `removed_asset_ids`. Do not
sum cumulative snapshots, baseline counters, removed assets, or GitHub counts with
PyPI windows. A current asset subtotal excludes deleted assets and can decrease.

`available` with `assets: []` means the repository was successfully observed and
had no qualifying uploaded assets. On a repository collection failure, valid
prior data becomes `stale`: `observed_at`, assets and their original change interval
remain unchanged while `last_attempt_at` advances. Without valid prior data,
`unavailable` uses null `observed_at`, `assets` and `changes`. Both failure states
include only the bounded category `error: "collection_failed"`, never remote
diagnostics. Missing data is not an observed zero. Consumers should also mark an
old successful observation stale if the workflow stops updating.

Every repository is isolated: a timeout or malformed response for one does not
erase another repository's data or PyPI observations. The collector calls the
existing GitHub distribution reader once per repository (repository information
and paginated release assets). The Optinpy result is reused in its legacy root
`latest.json` feed, including existing stars and asset history, with no second
Optinpy request. The outer compatibility field `github_metrics_status` still
describes that legacy Optinpy feed only. Per-package status is authoritative for
the new view; no per-package star series is added.

## Bounds and retained data

Each repository is limited to 100 current assets, 100 current change records and
100 removed IDs. IDs must be positive safe integers; counters, increases and the
sum of current counters must be nonnegative safe integers (`<= 2**53 - 1`). Asset
names and tags must be nonempty printable text of at most 255 characters and 512
UTF-8 bytes. Kinds are checked against the filename. Unknown fields are dropped
from published objects. Malformed retained data is not reused as a baseline.

The public aggregate is capped at 1 MiB before either latest file is replaced.
Inputs or totals exceeding bounds fail visibly rather than being truncated into
misleading counts. Consumers should validate this schema and impose the same
response-size limit. Larger future registries require a reviewed format or limit
change before their aggregate exceeds this cap.
# Explicit rolling PyPI totals

The CLI collector also records `pypi_recent` from the official
[`/recent` endpoint](https://pypistats.org/api/). It is independent of
`pypi_downloads`: the original daily arrays, windows and historical observations
remain intact for existing consumers. Each endpoint is fetched at most once per
UTC day, including unsuccessful attempts; the first run after adding this source
can populate `/recent` while reusing an already-cached `/overall` observation.

The README badges prefer the source's `last_month` count and use the label
“PyPI downloads/month”. The endpoint does not provide exact period dates, so no
date range or daily coverage is invented. Explicit zero remains zero; missing or
invalid responses never become zero. Retained totals after a failed request are
marked stale. When no recent total exists, the badge falls back to the original
dated-row window with its existing partial/stale labels. Public badge links stay
on GitHub releases and PyPI Stats.
