from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "runtime_data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "aura_gateway.db"


def _load_env_file() -> None:
    env_path = APP_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file()
DATABASE_URL = os.getenv("AURA_DATABASE_URL") or os.getenv("DATABASE_URL")


def _using_postgres() -> bool:
    return bool(DATABASE_URL and DATABASE_URL.startswith(("postgresql://", "postgres://")))


def storage_backend() -> dict[str, Any]:
    if _using_postgres():
        return {
            "backend": "postgres",
            "database_url_configured": True,
            "sqlite_path": str(DB_PATH),
        }
    return {
        "backend": "sqlite",
        "database_url_configured": False,
        "sqlite_path": str(DB_PATH),
    }


def _connect() -> Any:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if _using_postgres():
        try:
            import psycopg  # type: ignore
            from psycopg.rows import dict_row  # type: ignore

            return psycopg.connect(DATABASE_URL, row_factory=dict_row)
        except ImportError:
            try:
                import psycopg2  # type: ignore
                from psycopg2.extras import RealDictCursor  # type: ignore

                return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
            except ImportError as exc:
                raise RuntimeError(
                    "PostgreSQL storage is configured but no Python driver is installed. "
                    "Install psycopg or psycopg2-binary."
                ) from exc
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _execute(conn: Any, sql: str, params: tuple[Any, ...] = ()) -> Any:
    if _using_postgres():
        cursor = conn.cursor()
        cursor.execute(sql.replace("?", "%s"), params)
        return cursor
    return conn.execute(sql, params)


def init_storage() -> None:
    with _connect() as conn:
        if _using_postgres():
            _execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    columns_json TEXT NOT NULL,
                    numeric_columns_json TEXT NOT NULL,
                    selected_columns_json TEXT NOT NULL,
                    uploaded_at DOUBLE PRECISION NOT NULL
                )
                """,
            )
            _execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    id SERIAL PRIMARY KEY,
                    started_at DOUBLE PRECISION NOT NULL,
                    finished_at DOUBLE PRECISION,
                    phase TEXT NOT NULL,
                    backend_mode TEXT,
                    dataset_id INTEGER,
                    selected_columns_json TEXT,
                    overrides_json TEXT NOT NULL,
                    metrics_json TEXT,
                    policy_metrics_json TEXT,
                    error TEXT
                )
                """,
            )
        else:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    columns_json TEXT NOT NULL,
                    numeric_columns_json TEXT NOT NULL,
                    selected_columns_json TEXT NOT NULL,
                    uploaded_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at REAL NOT NULL,
                    finished_at REAL,
                    phase TEXT NOT NULL,
                    backend_mode TEXT,
                    dataset_id INTEGER,
                    selected_columns_json TEXT,
                    overrides_json TEXT NOT NULL,
                    metrics_json TEXT,
                    policy_metrics_json TEXT,
                    error TEXT
                )
                """
            )


def _decode_row(row: Any) -> dict[str, Any]:
    out = dict(row)
    for key in ("columns_json", "numeric_columns_json", "selected_columns_json", "overrides_json", "metrics_json", "policy_metrics_json"):
        if key in out:
            value = out.pop(key)
            out[key.removesuffix("_json")] = json.loads(value) if value else None
    if "uploaded_at" in out:
        out["uploaded_at_iso"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(out["uploaded_at"]))
    if "started_at" in out:
        out["started_at_iso"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(out["started_at"]))
    if out.get("finished_at"):
        out["finished_at_iso"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(out["finished_at"]))
    return out


def save_dataset(filename: str, stored_path: Path, row_count: int, columns: list[str], numeric_columns: list[str], selected_columns: list[str]) -> dict[str, Any]:
    init_storage()
    with _connect() as conn:
        params = (
            filename,
            str(stored_path),
            row_count,
            json.dumps(columns),
            json.dumps(numeric_columns),
            json.dumps(selected_columns),
            time.time(),
        )
        if _using_postgres():
            cursor = _execute(
                conn,
                """
                INSERT INTO datasets(filename, stored_path, row_count, columns_json, numeric_columns_json, selected_columns_json, uploaded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                RETURNING id
                """,
                params,
            )
            dataset_id = int(cursor.fetchone()["id"])
        else:
            cursor = conn.execute(
                """
                INSERT INTO datasets(filename, stored_path, row_count, columns_json, numeric_columns_json, selected_columns_json, uploaded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                params,
            )
            dataset_id = int(cursor.lastrowid)
        row = _execute(conn, "SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
    return _decode_row(row)


def list_datasets() -> list[dict[str, Any]]:
    init_storage()
    with _connect() as conn:
        rows = _execute(conn, "SELECT * FROM datasets ORDER BY uploaded_at DESC").fetchall()
    return [_decode_row(row) for row in rows]


def get_dataset(dataset_id: int) -> dict[str, Any] | None:
    init_storage()
    with _connect() as conn:
        row = _execute(conn, "SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
    return _decode_row(row) if row else None


def update_dataset_selection(dataset_id: int, selected_columns: list[str]) -> dict[str, Any] | None:
    init_storage()
    with _connect() as conn:
        _execute(
            conn,
            "UPDATE datasets SET selected_columns_json = ? WHERE id = ?",
            (json.dumps(selected_columns), dataset_id),
        )
        row = _execute(conn, "SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
    return _decode_row(row) if row else None


def create_run(overrides: dict[str, Any], dataset_id: int | None, selected_columns: list[str] | None) -> int:
    init_storage()
    with _connect() as conn:
        params = (
            time.time(),
            "running",
            dataset_id,
            json.dumps(selected_columns or []),
            json.dumps(overrides),
        )
        if _using_postgres():
            cursor = _execute(
                conn,
                """
                INSERT INTO benchmark_runs(started_at, phase, dataset_id, selected_columns_json, overrides_json)
                VALUES (?, ?, ?, ?, ?)
                RETURNING id
                """,
                params,
            )
            return int(cursor.fetchone()["id"])
        cursor = conn.execute(
            """
            INSERT INTO benchmark_runs(started_at, phase, dataset_id, selected_columns_json, overrides_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            params,
        )
        return int(cursor.lastrowid)


def finish_run(run_id: int | None, phase: str, backend_mode: str, metrics: dict[str, Any] | None, policy_metrics: dict[str, Any] | None, error: str | None = None) -> None:
    if run_id is None:
        return
    init_storage()
    with _connect() as conn:
        _execute(
            conn,
            """
            UPDATE benchmark_runs
            SET finished_at = ?, phase = ?, backend_mode = ?, metrics_json = ?, policy_metrics_json = ?, error = ?
            WHERE id = ?
            """,
            (
                time.time(),
                phase,
                backend_mode,
                json.dumps(metrics or {}),
                json.dumps(policy_metrics or {}),
                error,
                run_id,
            ),
        )


def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    init_storage()
    with _connect() as conn:
        rows = _execute(
            conn,
            "SELECT * FROM benchmark_runs ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_decode_row(row) for row in rows]
