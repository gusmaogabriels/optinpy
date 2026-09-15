# Per-package PyPI download observations

One GitHub Actions job owns collection for the reviewed `.github/package-registry.json`. `latest.json` is a public aggregate; `observations/` retains dated snapshots. Optinpy's legacy root feed reuses the same observation, with no second PyPI request.

Each package also has a separate `github_release_downloads` observation. These are cumulative counters for uploaded assets, with .whl/.tar.gz packages separated from checksum/other files. They must not be added to PyPI windows. First asset observations are baselines; later changes compare stable asset IDs and explicitly flag decreases/removals. Failed repository requests retain valid prior observations as stale, or remain unavailable without a prior observation.

PyPI Stats updates once daily and retains 180 days. Known mirrors are excluded; CI and repeated downloads remain included. These are downloads, not users, successful installations, or CLI executions. No local telemetry is collected.

Check each package's `status`, `data_through`, `fetched_at` and `last_attempt_at`. Missing dates are unknown; explicit dated zeros are zero. A 404 is `no_data`, not zero downloads. Errors retain earlier data as `stale` when possible. Compute windows from dated rows; never add overlapping windows or snapshots. Use the newest observation per package/date and preserve gaps and revisions.

`pypi_recent` separately retains the source's explicit last-day/week/month totals. The badges prefer these totals when available, so sparse daily rows are not treated as an incomplete monthly total. The endpoint does not supply exact period dates: none are inferred. Each endpoint is cached once per UTC day. A failed request retains the prior aggregate as stale; a 404 is not zero.

Sources: https://pypistats.org/api/ and https://pypistats.org/faqs .
