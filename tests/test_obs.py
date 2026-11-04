"""Unit tests for obs.py's pure, offline logic: the SQLite usability probe and the
find-or-start loop. No servers are started and nothing touches the network."""

import socket
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import obs  # noqa: E402


def test_usable_sqlite_accepts_a_fresh_or_missing_db(tmp_path):
    db = tmp_path / "mlflow.db"
    assert obs._usable_sqlite(db)          # created on first connect
    assert db.exists()
    assert obs._usable_sqlite(db)          # and still usable once it exists


def test_usable_sqlite_rejects_a_non_sqlite_file(tmp_path):
    junk = tmp_path / "mlflow.db"
    junk.write_text("this is not a database")
    assert not obs._usable_sqlite(junk)


def test_usable_sqlite_rejects_an_unopenable_path(tmp_path):
    assert not obs._usable_sqlite(tmp_path / "no" / "such" / "dir" / "mlflow.db")


def test_store_path_falls_back_to_dated_file_when_default_is_unusable(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(obs, "REPO", tmp_path)
    monkeypatch.setattr(obs, "MLFLOW_DB", None)
    (tmp_path / "mlflow.db").write_text("corrupt")
    chosen = obs._store_path()
    assert chosen == tmp_path / f"mlflow-{obs.TODAY}.db"
    assert "not usable" in capsys.readouterr().out


def test_store_path_prefers_the_shared_db(tmp_path, monkeypatch):
    monkeypatch.setattr(obs, "REPO", tmp_path)
    monkeypatch.setattr(obs, "MLFLOW_DB", None)
    assert obs._store_path() == tmp_path / "mlflow.db"


def test_find_or_start_reuses_a_healthy_server(monkeypatch):
    started = []
    monkeypatch.setattr(obs, "_port_open", lambda port: port == 5000)
    port = obs._find_or_start("x", started.append, lambda p: True, 5000)
    assert port == 5000
    assert started == []


def test_find_or_start_skips_a_strangers_port_and_starts_on_the_next(monkeypatch):
    started = []
    open_ports = {5000}

    def start(p):
        started.append(p)
        open_ports.add(p)

    monkeypatch.setattr(obs, "_port_open", lambda port: port in open_ports)
    monkeypatch.setattr(obs.time, "sleep", lambda s: None)
    port = obs._find_or_start("x", start, lambda p: p != 5000, 5000)
    assert port == 5001
    assert started == [5001]


def test_find_or_start_gives_up_after_tries(monkeypatch):
    monkeypatch.setattr(obs, "_port_open", lambda port: False)
    monkeypatch.setattr(obs.time, "sleep", lambda s: None)
    port = obs._find_or_start("x", lambda p: None, lambda p: False, 5000, tries=2, wait_s=0)
    assert port is None


def test_port_open_reflects_a_real_listening_socket():
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        assert obs._port_open(port)
    assert not obs._port_open(port)
