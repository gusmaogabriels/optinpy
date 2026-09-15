# Central package statistics

`package_stats.py` is the sole scheduled PyPI collector for the reviewed
`.github/package-registry.json`: Optinpy, mkin4py and xl2py. It runs in the existing
GitHub Actions statistics job, writing public data to `distribution-statistics`:

- `packages/latest.json`: the multi-package aggregate consumed by the website.
- `packages/observations/`: append-only snapshots, retaining data beyond the source's
  180-day window and making later revisions inspectable.
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
