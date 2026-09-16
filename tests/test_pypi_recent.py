from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import importlib.util
import json
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


@pytest.mark.parametrize('failure', [429, 503, 408, 'network'])
def test_temporary_failure_retries_after_cooldown_then_caches_success(failure):
    calls = []
    def fetch(url):
        calls.append(url)
        if len(calls) == 1:
            if failure == 'network':
                raise OSError('untrusted remote detail')
            raise HTTPError(url, failure, 'untrusted remote detail', {}, None)
        return payload(last_month=1)
    failed = recent.collect('optinpy', now=NOW, fetch=fetch)
    assert failed['counts'] is None and failed['status'] == 'unavailable'
    assert 'untrusted remote detail' not in str(failed)
    assert recent.collect('optinpy', failed, now=NOW+timedelta(minutes=59), fetch=fetch) == failed
    assert len(calls) == 1
    recovered = recent.collect('optinpy', failed, now=NOW+timedelta(hours=1), fetch=fetch)
    assert recovered['status'] == 'available' and recovered['counts']['last_month'] == 1
    assert 'retry_at' not in recovered and 'error' not in recovered
    assert recent.collect('optinpy', recovered, now=NOW+timedelta(hours=8), fetch=fetch) == recovered
    assert len(calls) == 2


@pytest.mark.parametrize('header', ['7200', format_datetime(NOW+timedelta(hours=2), usegmt=True)])
def test_server_retry_after_is_respected_across_utc_midnight(header):
    # Move this failure close to midnight; the UTC-day cache must not shorten the delay.
    now = NOW.replace(hour=23)
    if not header.isdigit():
        header = format_datetime(now+timedelta(hours=2), usegmt=True)
    calls = []
    def fail(url):
        calls.append(url)
        raise HTTPError(url, 429, 'rate limit', {'Retry-After': header}, None)
    failed = recent.collect('optinpy', now=now, fetch=fail)
    assert failed['retry_at'] == (now+timedelta(hours=2)).isoformat()
    assert recent.collect('optinpy', failed, now=now+timedelta(hours=1), fetch=fail) == failed
    assert len(calls) == 1
    def recover(url):
        calls.append(url)
        return payload()
    assert recent.collect('optinpy', failed, now=now+timedelta(hours=2), fetch=recover)['status'] == 'available'
    assert len(calls) == 2


@pytest.mark.parametrize('header', ['', 'nonsense', '-1', '60', '9'*100,
                                    format_datetime(NOW-timedelta(days=1), usegmt=True)])
def test_invalid_or_short_retry_after_keeps_minimum_cooldown(header):
    def fail(url):
        raise HTTPError(url, 429, 'rate limit', {'Retry-After': header}, None)
    failed = recent.collect('optinpy', now=NOW, fetch=fail)
    assert failed['retry_at'] == (NOW+timedelta(hours=1)).isoformat()


@pytest.mark.parametrize('code', [403, 404])
def test_non_transient_http_failures_remain_cached_for_utc_day(code):
    calls = []
    def fail(url):
        calls.append(url)
        raise HTTPError(url, code, 'unavailable', {}, None)
    failed = recent.collect('optinpy', now=NOW, fetch=fail)
    assert 'retry_at' not in failed
    assert recent.collect('optinpy', failed, now=NOW+timedelta(hours=8), fetch=fail) == failed
    assert len(calls) == 1


def test_retry_failure_preserves_old_counts_and_starts_a_new_cooldown():
    original = recent.collect('optinpy', now=NOW-timedelta(days=1), fetch=lambda _: payload())
    def fail(url):
        raise HTTPError(url, 503, 'unavailable', {}, None)
    first = recent.collect('optinpy', original, now=NOW, fetch=fail)
    second = recent.collect('optinpy', first, now=NOW+timedelta(hours=1), fetch=fail)
    assert first['status'] == second['status'] == 'stale'
    assert second['counts'] == original['counts'] and second['fetched_at'] == original['fetched_at']
    assert second['retry_at'] == (NOW+timedelta(hours=2)).isoformat()


def test_legacy_rate_limit_cache_recovers_monthly_badge_without_refetching_successes():
    registry = {'schema_version': 1, 'packages': [
        {'id': name, 'pypi_package': name, 'repository': f'gusmaogabriels/{name}'}
        for name in ('optinpy', 'mkin4py')]}
    daily_calls, recent_calls = [], []
    def daily(url):
        daily_calls.append(url)
        name = url.split('/')[-2]
        return {'package': name, 'type': 'overall_downloads', 'data': [
            {'date': '2026-09-09', 'category': 'without_mirrors', 'downloads': 1}]}
    def initial(url):
        recent_calls.append(url)
        name = url.split('/')[-2]
        if name == 'mkin4py':
            raise HTTPError(url, 429, 'rate limit', {}, None)
        return payload(name)
    first = stats.collect_packages(registry, now=NOW, fetch=daily, fetch_recent=initial)
    first['packages'][1]['pypi_recent'].pop('retry_at')  # Snapshot from the old collector.
    def recover(url):
        recent_calls.append(url)
        return payload(url.split('/')[-2], last_month=1)
    again = stats.collect_packages(registry, first, now=NOW+timedelta(hours=2),
                                   fetch=daily, fetch_recent=recover)
    assert len(daily_calls) == 2 and len(recent_calls) == 3
    assert again['packages'][0]['pypi_recent'] == first['packages'][0]['pypi_recent']
    for before, after in zip(first['packages'], again['packages']):
        assert after['pypi_downloads'] == before['pypi_downloads']
        after['github_release_downloads'] = {'status': 'unavailable', 'assets': None}
    badge = json.loads(stats.package_badges.endpoints(again)['mkin4py-pypi.json'])
    assert badge['label'] == 'PyPI downloads/month' and badge['message'] == '1'
