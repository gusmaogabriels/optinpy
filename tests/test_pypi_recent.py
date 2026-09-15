from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
from urllib.error import HTTPError

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location('stats_recent_test', ROOT/'.github/scripts/package_stats.py')
stats = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stats)
recent = stats.pypi_recent
NOW = datetime(2026, 9, 15, 12, tzinfo=timezone.utc)


def payload(package='optinpy', **counts):
    return {'package': package, 'type': 'recent_downloads',
            'data': {'last_day': 0, 'last_week': 1, 'last_month': 3, **counts}}


def test_daily_cache_preserves_explicit_zero_and_is_per_package():
    calls = []
    def fetch(url):
        calls.append(url)
        return payload()
    value = recent.collect('optinpy', now=NOW, fetch=fetch)
    assert value['counts']['last_day'] == 0
    assert recent.collect('optinpy', value, now=NOW+timedelta(hours=1), fetch=fetch) == value
    assert len(calls) == 1
    recent.collect('optinpy', value, now=NOW+timedelta(days=1), fetch=fetch)
    assert len(calls) == 2
    result = recent.collect('mkin4py', value, now=NOW, fetch=lambda _: payload('mkin4py'))
    assert result['package'] == 'mkin4py'


@pytest.mark.parametrize('bad', [True, -1, 2**53, 1.5, '3', None])
def test_malformed_counts_are_unknown(bad):
    result = recent.collect('optinpy', now=NOW, fetch=lambda _: payload(last_month=bad))
    assert result['status'] == 'unavailable' and result['counts'] is None


def test_identity_and_monotonic_windows_are_validated():
    for data in (payload('another'), payload(last_day=4)):
        assert recent.collect('optinpy', now=NOW, fetch=lambda _: data)['counts'] is None


def test_outage_retains_valid_aggregate_without_rewriting_its_age():
    previous = recent.collect('optinpy', now=NOW, fetch=lambda _: payload())
    def fail(_): raise OSError('remote detail must not appear')
    stale = recent.collect('optinpy', previous, now=NOW+timedelta(days=1), fetch=fail)
    assert stale['status'] == 'stale'
    assert stale['fetched_at'] == previous['fetched_at'] and stale['counts'] == previous['counts']
    assert 'remote detail' not in str(stale)
    corrupt = {**previous, 'counts': {'last_day': False, 'last_week': 1, 'last_month': 3}}
    assert recent.collect('optinpy', corrupt, now=NOW, fetch=fail)['counts'] is None


def test_missing_package_is_not_zero():
    def missing(url): raise HTTPError(url, 404, 'missing', {}, None)
    value = recent.collect('optinpy', now=NOW, fetch=missing)
    assert value['status'] == 'no_data' and value['counts'] is None


def test_aggregate_collection_and_badge_use_recent_without_modifying_daily_history():
    registry = {'schema_version': 1, 'packages': [{'id': 'optinpy', 'pypi_package': 'optinpy', 'repository': 'gusmaogabriels/optinpy'}]}
    calls = []
    def daily(url):
        calls.append(url)
        return {'package': 'optinpy', 'type': 'overall_downloads',
                'data': [{'date': '2026-09-09', 'category': 'without_mirrors', 'downloads': 1}]}
    def total(url):
        calls.append(url)
        return payload(last_month=7)
    snapshot = stats.collect_packages(registry, now=NOW, fetch=daily, fetch_recent=total)
    again = stats.collect_packages(registry, snapshot, now=NOW, fetch=daily, fetch_recent=total)
    assert len(calls) == 2
    row = again['packages'][0]
    assert row['pypi_downloads']['daily'] == snapshot['packages'][0]['pypi_downloads']['daily']
    assert row['pypi_downloads']['windows']['30']['reported_downloads'] == 1
    row['github_release_downloads'] = {'status': 'unavailable', 'assets': None}
    badge = stats.package_badges.observations(again)['optinpy-pypi']
    assert badge['message'] == '7' and badge['label'] == 'PyPI downloads/month'
    row['pypi_recent']['status'] = 'stale'
    assert stats.package_badges.observations(again)['optinpy-pypi']['message'] == '7 (stale)'
    row['pypi_recent'] = {'status': 'unavailable', 'counts': None}
    assert stats.package_badges.observations(again)['optinpy-pypi']['message'] == '1 (partial, stale)'
