"""Opt-in local CLI counts. No network, arguments, paths or model data stored."""
from contextlib import closing
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
from time import perf_counter


_SCHEMA = """CREATE TABLE IF NOT EXISTS cli_usage (
    package TEXT NOT NULL, version TEXT NOT NULL, day TEXT NOT NULL,
    command TEXT NOT NULL, exit_code INTEGER NOT NULL, runs INTEGER NOT NULL,
    command_seconds REAL NOT NULL,
    PRIMARY KEY (package, version, day, command, exit_code))"""


def _path(package):
    value = os.environ.get(package.upper() + '_USAGE_DB')
    return Path(value).expanduser() if value else None


def _record(package, version, command, exit_code, seconds):
    path = _path(package)
    if path is None:
        return
    # New databases are private. The parent directory is chosen by the caller.
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        pass
    else:
        os.close(fd)
    if not path.is_file() or path.is_symlink():
        raise OSError('usage database must be a regular file')
    # Serialize writers before reading/changing the schema. The sqlite context
    # manager commits transactions but does not close the connection itself.
    with closing(sqlite3.connect(path, timeout=5.)) as connection:
        with connection:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute(_SCHEMA)
            connection.execute("""INSERT INTO cli_usage VALUES (?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(package, version, day, command, exit_code) DO UPDATE SET
                runs = runs + 1, command_seconds = command_seconds + excluded.command_seconds""",
                (package, version, datetime.now(timezone.utc).date().isoformat(), command,
                 exit_code, seconds if math.isfinite(seconds) else 0.))



def report(package):
    """Read only; reporting never creates a database or counts itself."""
    path = _path(package)
    result = {'schema_version': 1, 'package': package, 'enabled': path is not None,
              'scope': 'Local CLI invocations in the selected database; not unique users or downloads.',
              'transmission': 'none', 'runs': 0, 'counts': []}
    if path is None or not path.exists():
        return result
    if not path.is_file() or path.is_symlink():
        raise OSError('usage database must be a regular file')
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=5.)) as connection:
        if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='cli_usage'").fetchone() is None:
            return result
        rows = connection.execute("""SELECT version, day, command, exit_code, runs, command_seconds
            FROM cli_usage WHERE package=? ORDER BY day, version, command, exit_code""", (package,)).fetchall()
    result['counts'] = [dict(zip(('version', 'day', 'command', 'exit_code', 'runs', 'command_seconds'), row))
                        for row in rows]
    result['runs'] = sum(row['runs'] for row in result['counts'])
    return result


def run(main, argv, *, package, version, commands):
    """Keep CLI output/exit behavior; count only when explicitly configured."""
    args = list(sys.argv[1:] if argv is None else argv)
    first = args[0] if args else ''
    if first == 'usage':
        if args[1:] in (['--help'], ['-h']):
            return main(args)
        if args[1:] not in ([], ['--json']):
            print(f'{package}: usage accepts only --json', file=sys.stderr)
            return 2
        try:
            print(json.dumps(report(package), allow_nan=False))
            return 0
        except (OSError, sqlite3.Error, ValueError, RuntimeError):
            print(f'{package}: could not read local usage counts', file=sys.stderr)
            return 2
    try:
        path = _path(package)
    except (OSError, ValueError, RuntimeError):
        print(f'{package}: could not update local usage counts', file=sys.stderr)
        return main(args)
    if path is None:
        return main(args)
    command = first if first in commands else {'--help': 'help', '-h': 'help', '--version': 'version'}.get(first, 'invalid')
    started, code = perf_counter(), 1
    try:
        code = main(args)
        return code
    except SystemExit as error:
        code = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
        raise
    finally:
        try:
            _record(package, version, command, code, perf_counter()-started)
        except (OSError, sqlite3.Error, ValueError, RuntimeError):
            print(f'{package}: could not update local usage counts', file=sys.stderr)
