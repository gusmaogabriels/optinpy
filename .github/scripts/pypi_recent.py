"""Cache PyPI Stats' explicit rolling aggregates separately from daily rows."""
from datetime import datetime, timezone
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
        if value['status'] in ('available', 'stale'):
            fetched = datetime.fromisoformat(value['fetched_at'])
            if fetched.tzinfo is None or fetched > attempted or not _counts(value['counts']):
                return None
        elif value.get('counts') is not None:
            return None
        return value
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def collect(package, previous=None, *, now, fetch):
    now = now.astimezone(timezone.utc)
    previous = prior(previous, package, now)
    if previous and datetime.fromisoformat(previous['last_attempt_at']).astimezone(timezone.utc).date() == now.date():
        return dict(previous)
    source = f'https://pypistats.org/api/packages/{package}/recent'
    result = {'status': 'no_data', 'package': package, 'source': source,
              'mirrors': 'excluded', 'last_attempt_at': now.isoformat(),
              'fetched_at': None, 'counts': None,
              'scope': 'Source-reported last day/week/month aggregates; exact period dates are not supplied by this endpoint.'}
    try:
        payload = fetch(source)
        if (not isinstance(payload, dict) or payload.get('package') != package
                or payload.get('type') != 'recent_downloads' or not _counts(payload.get('data'))):
            raise ValueError('invalid recent aggregate')
        return {**result, 'status': 'available', 'fetched_at': now.isoformat(),
                'counts': {key: payload['data'][key] for key in FIELDS}}
    except HTTPError as error:
        reason = 'not_found' if error.code == 404 else 'rate_limited' if error.code == 429 else 'http_error'
    except (OSError, URLError):
        reason = 'network_error'
    except (ValueError, KeyError, TypeError):
        reason = 'invalid_response'
    if previous and previous.get('counts') is not None:
        result = {**previous, 'status': 'stale', 'last_attempt_at': now.isoformat()}
    else:
        result['status'] = 'no_data' if reason == 'not_found' else 'unavailable'
    return {**result, 'error': reason}
