"""Local accounting must not change solving, emit telemetry or store inputs."""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest
from optinpy import _usage


def test_disabled_means_no_writes(tmp_path, monkeypatch):
    monkeypatch.delenv('OPTINPY_USAGE_DB', raising=False)
    monkeypatch.chdir(tmp_path)
    assert _usage.run(lambda _: 0, ['solve', 'private-model.json'], package='optinpy', version='test', commands={'solve'}) == 0
    assert list(tmp_path.iterdir()) == []
    assert _usage.report('optinpy')['enabled'] is False


def test_counts_exit_codes_without_inputs(tmp_path, monkeypatch, capsys):
    path = tmp_path/'counts.sqlite3'
    monkeypatch.setenv('OPTINPY_USAGE_DB', str(path))
    for code in (0, 1, 2):
        assert _usage.run(lambda _, c=code: c, ['solve', '/sensitive/workbook.xlsx'],
            package='optinpy', version='test', commands={'solve'}) == code
    assert _usage.run(lambda _: 0, ['sensitive-formula'], package='optinpy', version='test', commands={'solve'}) == 0
    result = _usage.report('optinpy')
    assert result['runs'] == 4
    assert {row['command'] for row in result['counts']} == {'solve', 'invalid'}
    assert b'sensitive' not in path.read_bytes()
    assert _usage.run(None, ['usage', '--json'], package='optinpy', version='test', commands=set()) == 0
    assert json.loads(capsys.readouterr().out)['runs'] == 4
    assert _usage.report('optinpy')['runs'] == 4


def test_argparse_exit_and_storage_failure_preserve_exit_code(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('OPTINPY_USAGE_DB', str(tmp_path/'counts.sqlite3'))
    def invalid(_): raise SystemExit(2)
    with pytest.raises(SystemExit) as error:
        _usage.run(invalid, ['solve'], package='optinpy', version='test', commands={'solve'})
    assert error.value.code == 2
    assert _usage.report('optinpy')['counts'][0]['exit_code'] == 2
    monkeypatch.setenv('OPTINPY_USAGE_DB', str(tmp_path/'missing'/'counts.sqlite3'))
    assert _usage.run(lambda _: 0, ['solve'], package='optinpy', version='test', commands={'solve'}) == 0
    assert 'could not update' in capsys.readouterr().err


def test_unresolvable_path_keeps_command_result(monkeypatch, capsys):
    monkeypatch.setenv('OPTINPY_USAGE_DB', '~no-such-usage-account/counts.sqlite3')
    assert _usage.run(lambda _: 0, ['solve'], package='optinpy', version='test', commands={'solve'}) == 0
    assert 'could not update' in capsys.readouterr().err


def test_cli_usage_report_does_not_count_itself(tmp_path, monkeypatch, capsys):
    from optinpy.cli import main
    monkeypatch.setenv('OPTINPY_USAGE_DB', str(tmp_path/'counts.sqlite3'))
    assert main(['methods', '--json']) == 0
    assert json.loads(capsys.readouterr().out)
    for _ in range(2):
        assert main(['usage', '--json']) == 0
        data = json.loads(capsys.readouterr().out)
        assert data['runs'] == 1 and data['counts'][0]['command'] == 'methods'


def test_database_handles_are_released_for_removal(tmp_path, monkeypatch):
    path = tmp_path/'counts.sqlite3'
    monkeypatch.setenv('OPTINPY_USAGE_DB', str(path))
    _usage._record('optinpy', 'test', 'solve', 0, .1)
    assert _usage.report('optinpy')['runs'] == 1
    # Windows refuses removal while a SQLite file handle remains open.
    path.unlink()
    assert not path.exists()


def test_concurrent_cli_processes_and_readonly_reporting(tmp_path, monkeypatch):
    path = tmp_path/'counts.sqlite3'
    monkeypatch.setenv('OPTINPY_USAGE_DB', str(path))
    assert _usage.report('optinpy')['runs'] == 0
    assert not path.exists()
    source = "from optinpy._usage import _record; [_record('optinpy','test','solve',0,.01) for _ in range(20)]"
    processes = [subprocess.Popen([sys.executable, '-c', source], stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(3)]
    for process in processes:
        out, err = process.communicate(timeout=60)
        assert process.returncode == 0, err.decode()
    result = _usage.report('optinpy')
    assert result['runs'] == 60
    assert result['counts'][0]['command_seconds'] == pytest.approx(.6)


@pytest.mark.parametrize('lock', ['writer', 'reader'])
@pytest.mark.parametrize('expires', [False, True])
def test_contention_commits_once_or_rolls_back(tmp_path, monkeypatch, lock, expires):
    """Exercise real busy BEGIN and busy COMMIT, including budget exhaustion."""
    from contextlib import closing
    path = tmp_path/'counts.sqlite3'
    monkeypatch.setenv('OPTINPY_USAGE_DB', str(path))
    _usage._record('optinpy', 'test', 'solve', 0, .1)
    clock = [0.]
    waits = []
    monkeypatch.setattr(_usage, 'perf_counter', lambda: clock[0])
    with closing(sqlite3.connect(path, isolation_level=None, timeout=0.)) as blocker:
        blocker.execute('BEGIN IMMEDIATE' if lock == 'writer' else 'BEGIN')
        blocker.execute('SELECT * FROM cli_usage').fetchall()
        def release_or_advance(seconds):
            waits.append(seconds)
            clock[0] += seconds
            if not expires:
                blocker.rollback()
        monkeypatch.setattr(_usage, 'sleep', release_or_advance)
        if expires:
            with pytest.raises(sqlite3.OperationalError) as error:
                _usage._record('optinpy', 'test', 'solve', 0, .2)
            assert error.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
            assert clock[0] == pytest.approx(5.)
        else:
            _usage._record('optinpy', 'test', 'solve', 0, .2)
        assert waits  # The real SQLite lock forced the retry path.
        blocker.rollback()
    result = _usage.report('optinpy')
    assert result['runs'] == (1 if expires else 2)
    assert result['counts'][0]['command_seconds'] == pytest.approx(.1 if expires else .3)
    # The failed writer must release all handles/locks for the next invocation.
    _usage._record('optinpy', 'test', 'solve', 0, .1)
    assert _usage.report('optinpy')['runs'] == (2 if expires else 3)


def test_sql_errors_are_not_retried(tmp_path, monkeypatch):
    from contextlib import closing
    def unexpected_wait(_):
        pytest.fail('non-lock errors must not be retried')
    monkeypatch.setattr(_usage, 'sleep', unexpected_wait)
    with closing(sqlite3.connect(tmp_path/'bad.sqlite3')) as connection:
        with pytest.raises(sqlite3.OperationalError, match='syntax'):
            _usage._transaction_step(connection, 'NOT SQL', _usage.perf_counter() + 5.)
