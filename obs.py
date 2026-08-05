"""Optional observability helpers for the dspy-lab notebooks (taught in notebook 11).

Usage, from any notebook:

    import obs
    obs.enable_mlflow()     # trace this notebook's LM calls into the course store
    obs.mlflow_ui()         # find-or-start the MLflow UI; prints the clickable link
    obs.enable_phoenix()    # find-or-start Phoenix and route DSPy traces to it

Design notes (the robustness contract):
- Sessions are dated: everything logs under `dspy-lab-YYYY-MM-DD`, so work from
  different days never collides in the UIs.
- The MLflow store is pinned to `mlflow.db` next to this file (not the cwd). If a
  remnant file is unusable — unwritable, corrupt, or not SQLite at all — a fresh
  dated store is used instead: leftover files can never break a session.
- Servers are reused when they are genuinely ours (health-checked), skipped when a
  stranger holds the port (the next port is tried), and started otherwise. Started
  servers are plain background processes: they survive kernel restarts, and
  rerunning a cell just finds them again.
"""

import datetime
import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

TODAY = datetime.date.today().isoformat()
EXPERIMENT = f"dspy-lab-{TODAY}"
REPO = Path(__file__).resolve().parent
_STORE_TAG = "dspy_lab.store"     # experiments are stamped with their store's absolute path

MLFLOW_DB: Path | None = None


def _usable_sqlite(path: Path) -> bool:
    """True if MLflow could use the database at `path`: writable, and really SQLite."""
    try:
        conn = sqlite3.connect(path, timeout=2)
        try:
            conn.execute("BEGIN IMMEDIATE")   # wants write access + a valid header; changes nothing
            conn.rollback()
        finally:
            conn.close()
        return True
    except sqlite3.Error as exc:              # readonly, unopenable, corrupt, not-a-database, ...
        return "database is locked" in str(exc)   # ...but merely busy is a healthy store


def _store_path() -> Path:
    """Prefer the shared ./mlflow.db; fall back to a dated file if a remnant is unusable."""
    global MLFLOW_DB
    if MLFLOW_DB is None:
        default = REPO / "mlflow.db"
        if _usable_sqlite(default):
            MLFLOW_DB = default
        else:
            MLFLOW_DB = REPO / f"mlflow-{TODAY}.db"
            print(f"obs: {default.name} is not usable here; using {MLFLOW_DB.name} instead")
    return MLFLOW_DB


def _ensure_experiment():
    import mlflow

    store = str(_store_path())
    mlflow.set_tracking_uri(f"sqlite:///{store}")
    exp = mlflow.set_experiment(EXPERIMENT)
    if exp.tags.get(_STORE_TAG) != store:               # stamp the experiment with its store,
        mlflow.MlflowClient().set_experiment_tag(       # so a UI can prove which store it serves
            exp.experiment_id, _STORE_TAG, store)


def enable_mlflow():
    """Turn on MLflow tracing: dated experiment, repo-pinned store, autolog defaults."""
    import mlflow

    _ensure_experiment()
    mlflow.dspy.autolog()
    print(f"obs: MLflow tracing on — experiment {EXPERIMENT!r}, store {_store_path().name}")


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _http_get(url: str, timeout: float = 3.0) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read(65536) if response.status == 200 else None
    except Exception:
        return None


def _find_or_start(name, start_on_port, is_ours, first_port, tries=4, wait_s=90):
    """Reuse a server that passes `is_ours`; skip ports held by strangers; else start one."""
    for port in range(first_port, first_port + tries):
        if _port_open(port):
            if is_ours(port):
                print(f"obs: {name} already running — http://127.0.0.1:{port}")
                return port
            continue                                    # someone else's server; try next port
        start_on_port(port)
        deadline = time.time() + wait_s
        while time.time() < deadline:
            if _port_open(port) and is_ours(port):
                print(f"obs: {name} — http://127.0.0.1:{port}")
                return port
            time.sleep(0.5)
        print(f"obs: {name} did not come up on port {port}; trying {port + 1}")
    print(f"obs: could not find or start {name} on ports "
          f"{first_port}-{first_port + tries - 1}")
    return None


def _spawn(cmd, env=None):
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     env=env, cwd=REPO)


def mlflow_ui(port: int = 5000):
    """Find or start the MLflow UI on this course's store; prints the link."""
    _ensure_experiment()                                # the health check below needs it
    db = _store_path()

    def is_ours(p):                                     # ours = a UI serving OUR store: today's
        name = urllib.parse.quote(EXPERIMENT)           # experiment must exist there AND carry
        body = _http_get(f"http://127.0.0.1:{p}/api/2.0/mlflow/experiments/get-by-name"
                         f"?experiment_name={name}")    # this store's path stamp — the name
        if body is None:                                # alone could be another checkout's
            return False                                # same-day experiment
        try:
            tags = json.loads(body)["experiment"].get("tags", [])
            return any(t.get("key") == _STORE_TAG and t.get("value") == str(db)
                       for t in tags)
        except Exception:
            return False

    def start(p):
        _spawn([sys.executable, "-m", "mlflow", "ui",
                "--backend-store-uri", f"sqlite:///{db}", "--port", str(p)])

    return _find_or_start("MLflow UI", start, is_ours, port)


def enable_phoenix(port: int = 6006):
    """Find or start Phoenix, then route DSPy traces to it under the dated project."""

    def is_ours(p):                                     # any healthy Phoenix will do:
        body = _http_get(f"http://127.0.0.1:{p}")       # instances share ~/.phoenix data
        return body is not None and b"phoenix" in body.lower()

    def start(p):
        _spawn([sys.executable, "-m", "phoenix.server.main", "serve"],
               env={**os.environ, "PHOENIX_PORT": str(p)})

    chosen = _find_or_start("Phoenix", start, is_ours, port)
    if chosen is not None:
        from phoenix.otel import register

        register(project_name=EXPERIMENT, auto_instrument=True,
                 endpoint=f"http://127.0.0.1:{chosen}/v1/traces")
        print(f"obs: Phoenix tracing on — project {EXPERIMENT!r}")
    return chosen
