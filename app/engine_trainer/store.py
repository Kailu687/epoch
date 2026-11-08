"""SQLite experiment/registry store (metadata only; large artifacts live on disk)."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
  id TEXT PRIMARY KEY, name TEXT, source_path TEXT, stored_path TEXT, sha256 TEXT, domain TEXT,
  n_rows INTEGER, n_cols INTEGER, profile_json TEXT, imported_at TEXT);
CREATE TABLE IF NOT EXISTS experiments (
  id TEXT PRIMARY KEY, dataset_id TEXT REFERENCES datasets(id), task_json TEXT, split_json TEXT, seed INTEGER,
  status TEXT, env_json TEXT, run_dir TEXT, created_at TEXT, finished_at TEXT, error TEXT, findings_json TEXT);
CREATE TABLE IF NOT EXISTS candidates (
  id TEXT PRIMARY KEY, experiment_id TEXT REFERENCES experiments(id), cand_key TEXT, provider TEXT, estimator TEXT,
  params_json TEXT, status TEXT, fit_seconds REAL, val_metrics_json TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS models (
  id TEXT PRIMARY KEY, candidate_id TEXT REFERENCES candidates(id), experiment_id TEXT, artifact_path TEXT, artifact_sha256 TEXT,
  format TEXT, selected INTEGER DEFAULT 0, test_metrics_json TEXT, test_eval_count INTEGER DEFAULT 0, test_consumed_at TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS exports (
  id TEXT PRIMARY KEY, model_id TEXT REFERENCES models(id), path TEXT, manifest_json TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS log_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT, ts TEXT, level TEXT, event TEXT, payload_json TEXT);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)

    def close(self):
        self.db.close()

    def insert(self, table: str, **row):
        cols = ",".join(row)
        self.db.execute(f"INSERT INTO {table} ({cols}) VALUES ({','.join('?' * len(row))})", [self._v(v) for v in row.values()])
        self.db.commit()

    def update(self, table: str, id_: str, **row):
        sets = ",".join(f"{k}=?" for k in row)
        self.db.execute(f"UPDATE {table} SET {sets} WHERE id=?", [*[self._v(v) for v in row.values()], id_])
        self.db.commit()

    @staticmethod
    def _v(v):
        return json.dumps(v, default=str) if isinstance(v, (dict, list)) else v

    def one(self, sql, *args):
        r = self.db.execute(sql, args).fetchone()
        return dict(r) if r else None

    def all(self, sql, *args):
        return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    def log(self, experiment_id, level, event, payload=None):
        self.insert("log_events", experiment_id=experiment_id, ts=now(), level=level, event=event, payload_json=json.dumps(payload, default=str))
