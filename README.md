# Optinpy distribution statistics

Daily public GitHub counters and PyPI download data, collected by the repository's statistics workflow.
`latest.json` contains the latest observation and changes since the prior observation; `observations/` retains the history. Timestamps are UTC.

Download counters include repeated, automated and test downloads. They do not count installations, people or local solver runs. Checksum downloads are marked `other`, separately from package assets. Stars measure expressed interest.

A first observation is a baseline. Replaced assets have new IDs; missing assets and decreased counters are flagged rather than reported as negative downloads. Check `observed_at` for freshness; failed collections retain the last successful snapshot. This feed does not collect private clone traffic or CLI telemetry.

## PyPI downloads

[PyPI Stats](https://pypistats.org/packages/optinpy) supplies daily downloads excluding known mirrors. CI, repeat downloads and partial downloads can still be included; caches can hide installations. The [API](https://pypistats.org/api/) updates daily and retains 180 days; these snapshots archive its observations. Repeated collections reuse the PyPI result within the same UTC day.

`pypi_downloads.daily` contains dated counts. The 1-, 7- and 30-day `windows` end at `data_through`, the latest date returned by the source, not necessarily yesterday. Totals sum reported rows; `reported_days` shows coverage and missing dates are not fabricated. Never sum rolling totals or repeat snapshots: use the newest observation for each date.

Check `status`, `data_through`, `fetched_at` and `last_attempt_at` for freshness. New packages may have `no_data` until aggregation catches up. Unavailable or malformed responses retain prior data as `stale`, or `unavailable` when no prior data exists; they never become zero downloads. GitHub collection continues when the PyPI source is unavailable.
