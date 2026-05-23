from __future__ import annotations

import csv
import io
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import storage


MAX_DEFAULT_SELECTED_COLUMNS = 64


def _safe_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", filename.strip()) or "dataset.csv"
    if not cleaned.lower().endswith(".csv"):
        cleaned += ".csv"
    return cleaned


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def inspect_csv_bytes(content: bytes) -> tuple[list[str], list[str], int]:
    text = content.decode("utf-8-sig", errors="replace")
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.excel

    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    columns = [str(column).strip() for column in (reader.fieldnames or []) if str(column).strip()]
    numeric_hits = {column: 0 for column in columns}
    observed_hits = {column: 0 for column in columns}
    row_count = 0

    for row in reader:
        row_count += 1
        if row_count <= 500:
            for column in columns:
                value = row.get(column)
                if value not in (None, ""):
                    observed_hits[column] += 1
                    if _is_float(str(value).strip()):
                        numeric_hits[column] += 1

    numeric_columns = [
        column
        for column in columns
        if observed_hits[column] > 0 and numeric_hits[column] / max(1, observed_hits[column]) >= 0.9
    ]
    return columns, numeric_columns, row_count


def save_uploaded_dataset(filename: str, content: bytes) -> dict[str, Any]:
    storage.init_storage()
    safe_name = _safe_filename(filename)
    stored_path = storage.UPLOAD_DIR / f"{int(time.time() * 1000)}_{safe_name}"
    stored_path.write_bytes(content)
    columns, numeric_columns, row_count = inspect_csv_bytes(content)
    selected_columns = numeric_columns[:MAX_DEFAULT_SELECTED_COLUMNS]
    return storage.save_dataset(safe_name, stored_path, row_count, columns, numeric_columns, selected_columns)


def dataset_start_overrides(dataset_id: int | None, selected_columns: list[str] | None) -> dict[str, Any]:
    if dataset_id is None:
        return {}
    dataset = storage.get_dataset(dataset_id)
    if not dataset:
        return {}
    allowed = set(dataset.get("numeric_columns") or [])
    selected = [column for column in (selected_columns or dataset.get("selected_columns") or []) if column in allowed]
    if not selected:
        selected = list(dataset.get("selected_columns") or [])[:MAX_DEFAULT_SELECTED_COLUMNS]
    row_count = int(dataset.get("row_count") or 0)
    return {
        "BENCH_SENSORS": max(4, min(500, len(selected))),
        "BENCH_STEPS": max(24, min(240, row_count or 120)),
    }


def load_dataset_arrays(dataset_id: int | None, selected_columns: list[str] | None) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    if dataset_id is None:
        return None
    dataset = storage.get_dataset(dataset_id)
    if not dataset:
        return None

    stored_path = Path(str(dataset.get("stored_path") or ""))
    if not stored_path.exists():
        return None

    allowed = set(dataset.get("numeric_columns") or [])
    selected = [column for column in (selected_columns or dataset.get("selected_columns") or []) if column in allowed]
    if not selected:
        selected = list(dataset.get("selected_columns") or [])[:MAX_DEFAULT_SELECTED_COLUMNS]
    selected = selected[:500]
    if len(selected) < 2:
        return None

    text = stored_path.read_text(encoding="utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:4096])
    except csv.Error:
        dialect = csv.excel

    rows: list[list[float]] = []
    for row in csv.DictReader(io.StringIO(text), dialect=dialect):
        values: list[float] = []
        for column in selected:
            raw = row.get(column)
            try:
                values.append(float(str(raw).strip()))
            except (TypeError, ValueError):
                values.append(float("nan"))
        rows.append(values)

    if len(rows) < 4:
        return None

    data = np.asarray(rows, dtype=np.float32)
    finite_columns = np.isfinite(data).any(axis=0)
    data = data[:, finite_columns]
    selected = [column for column, keep in zip(selected, finite_columns) if keep]
    if data.shape[1] < 2:
        return None

    medians = np.nanmedian(data, axis=0)
    data = np.where(np.isfinite(data), data, medians)
    q1 = np.percentile(data, 25, axis=0)
    q3 = np.percentile(data, 75, axis=0)
    iqr = np.maximum(q3 - q1, 1e-6)
    anomaly_mask = (data < (q1 - 3.0 * iqr)) | (data > (q3 + 3.0 * iqr))

    low = np.percentile(data, 1, axis=0)
    high = np.percentile(data, 99, axis=0)
    scale = np.maximum(high - low, 1e-6)
    normalized = np.clip((data - low) / scale, 0.0, 1.0).astype(np.float32)
    return normalized, anomaly_mask.astype(bool), selected
