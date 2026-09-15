# Per-package PyPI download observations

One GitHub Actions job owns collection for the reviewed `.github/package-registry.json`. `latest.json` is a public aggregate; `observations/` retains dated snapshots. Optinpy's legacy root feed reuses the same observation, with no second PyPI request.

PyPI Stats updates once daily and retains 180 days. Known mirrors are excluded; CI and repeated downloads remain included. These are downloads, not users, successful installations, or CLI executions. No local telemetry is collected.

Check each package's `status`, `data_through`, `fetched_at` and `last_attempt_at`. Missing dates are unknown; explicit dated zeros are zero. A 404 is `no_data`, not zero downloads. Errors retain earlier data as `stale` when possible. Compute windows from dated rows; never add overlapping windows or snapshots. Use the newest observation per package/date and preserve gaps and revisions.

Sources: https://pypistats.org/api/ and https://pypistats.org/faqs .
