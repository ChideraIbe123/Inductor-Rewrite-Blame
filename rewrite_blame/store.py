"""SQLite store of finished measurements so attribution never re-measures a seen configuration."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

PROTOCOL_VERSION = 3  # bump when Measurement semantics change so stale rows are ignored


def measurement_key(model: str, env_fingerprint: str, state_hash: str, protocol: str = "") -> str:
    raw = f"{PROTOCOL_VERSION}|{model}|{env_fingerprint}|{state_hash}|{protocol}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS measurements (
                   key TEXT PRIMARY KEY,
                   model TEXT NOT NULL,
                   env TEXT NOT NULL,
                   state_hash TEXT NOT NULL,
                   protocol TEXT NOT NULL,
                   created REAL NOT NULL,
                   payload TEXT NOT NULL)"""
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_model_env ON measurements(model, env)")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def get(self, model: str, env: str, state_hash: str, protocol: str = "") -> dict[str, Any] | None:
        key = measurement_key(model, env, state_hash, protocol)
        row = self._conn.execute("SELECT payload FROM measurements WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, model: str, env: str, state_hash: str, payload: dict[str, Any], protocol: str = "") -> str:
        key = measurement_key(model, env, state_hash, protocol)
        self._conn.execute(
            "INSERT OR REPLACE INTO measurements VALUES (?,?,?,?,?,?,?)",
            (key, model, env, state_hash, protocol, time.time(), json.dumps(payload)),
        )
        self._conn.commit()
        return key

    def all(self, model: str | None = None, env: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT payload FROM measurements"
        cond, args = [], []
        if model is not None:
            cond.append("model=?")
            args.append(model)
        if env is not None:
            cond.append("env=?")
            args.append(env)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY created"
        return [json.loads(r[0]) for r in self._conn.execute(q, args)]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM measurements").fetchone()[0]

    def models(self) -> list[str]:
        return [r[0] for r in self._conn.execute("SELECT DISTINCT model FROM measurements ORDER BY model")]

    def export_json(self, path: str | Path, models: Iterable[str] | None = None) -> int:
        rows = self.all()
        if models is not None:
            ms = set(models)
            rows = [r for r in rows if r.get("model") in ms]
        Path(path).write_text(json.dumps(rows, indent=1))
        return len(rows)
