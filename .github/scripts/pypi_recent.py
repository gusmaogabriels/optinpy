"""Cache PyPI Stats' explicit rolling aggregates separately from daily rows."""
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError

FIELDS = ('last_day', 'last_week', 'last_month')


def _counts(value):
    return (isinstance(value, dict) and all(type(value.get(key)) is int
            and 0 <= value[key] <= 2**53-1 for key in FIELDS)
            and value['last_day'] <= value['last_week'] <= value['last_month'])


def prior(value, package, now):
    """An invalid cached aggregate never becomes an observed count."""
    try:
        if (not isinstance(value, dict) or value['package'] != package
                or value['source'] != f'https://pypistats.org/api/packages/{package}/recent'
                or value['mirrors'] != 'excluded'
                or value['status'] not in ('available', 'stale', 'no_data', 'unavailable')):
            return None
        attempted = datetime.fromisoformat(value['last_attempt_at'])
        if attempted.tzinfo is None or attempted > now:
            return None
        if 'retry_at' in value:
            retry = datetime.fromisoformat(value['retry_at'])
            if retry.tzinfo is None or retry < attempted:
                return None
        if value['status'] in ('available', 'stale'):
            fetched = datetime.fromisoformat(value['fetched_at'])
            if fetched.tzinfo is None or fetched > attempted or not _counts(value['counts']):
                return None
        elif value.get('counts') is not None:
            return None
        return value
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _retry_at(now, header=None):
    """Wait at least an hour, and longer when the server requests it."""
    retry = now + timedelta(hours=1)
    if header:
        try:
            requested = (now + timedelta(seconds=int(header)) if header.strip().isdigit()
                         else parsedate_to_datetime(header))
            if requested.tzinfo is not None:
                retry = max(retry, requested)
        except (TypeError, ValueError, OverflowError):
            pass
    return retry.isoformat()


def _cached(previous, now):
    attempted = datetime.fromisoformat(previous['last_attempt_at'])
    if previous['status'] in ('stale', 'unavailable'):
        retry_at = previous.get('retry_at')
        if retry_at is None and previous.get('error') in ('rate_limited', 'network_error'):
            # Older snapshots did not retain the server's retry deadline.
            retry_at = _retry_at(attempted)
        if retry_at is not None:
            return now < max(attempted + timedelta(hours=1), datetime.fromisoformat(retry_at))
    return attempted.astimezone(timezone.utc).date() == now.date()


def collect(package, previous=None, *, now, fetch):
    """Cache daily totals; retry temporary failures only after their cooldown."""
    now = now.astimezone(timezone.utc)
    previous = prior(previous, package, now)
    if previous and _cached(previous, now):
        return dict(previous)
    source = f'https://pypistats.org/api/packages/{package}/recent'
    result = {'status': 'no_data', 'package': package, 'source': source,
              'mirrors': 'excluded', 'last_attempt_at': now.isoformat(),
              'fetched_at': None, 'counts': None,
              'scope': 'Source-reported last day/week/month aggregates; exact period dates are not supplied by this endpoint.'}
    retry_at = None
    try:
        payload = fetch(source)
        if (not isinstance(payload, dict) or payload.get('package') != package
                or payload.get('type') != 'recent_downloads' or not _counts(payload.get('data'))):
            raise ValueError('invalid recent aggregate')
        return {**result, 'status': 'available', 'fetched_at': now.isoformat(),
                'counts': {key: payload['data'][key] for key in FIELDS}}
    except HTTPError as error:
        reason = 'not_found' if error.code == 404 else 'rate_limited' if error.code == 429 else 'http_error'
        if error.code in (408, 429) or 500 <= error.code < 600:
            retry_at = _retry_at(now, error.headers.get('Retry-After') if error.headers else None)
    except (OSError, URLError):
        reason = 'network_error'
        retry_at = _retry_at(now)
    except (ValueError, KeyError, TypeError):
        reason = 'invalid_response'
    if previous and previous.get('counts') is not None:
        result = {**previous, 'status': 'stale', 'last_attempt_at': now.isoformat()}
    else:
        result['status'] = 'no_data' if reason == 'not_found' else 'unavailable'
    result.pop('retry_at', None)
    if retry_at is not None:
        result['retry_at'] = retry_at
    return {**result, 'error': reason}
