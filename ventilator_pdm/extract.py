"""Fleet-filtered data extraction from QuestDB.

Extracts fleet-only, curated-variable data from QuestDB via HTTP API,
one day at a time to avoid memory/timeout issues with the 420M+ row
``pdm_medical_device`` table.

The primary entry point is :func:`extract_fleet_data`, which iterates over
calendar days, issues one SQL query per day via :func:`query_questdb`, and
writes the result as a daily Parquet file.  A ``manifest.json`` summary is
written alongside the Parquet files.

Typical DataFrame columns returned from QuestDB
(subject to the variable filter applied at extraction time)::

    timestamp        datetime64[ns, UTC]  — nanosecond-precision UTC
    device_serial    object               — 15-digit serial string
    variable_id      Int64                — numeric variable identifier
    value            float64              — measured value in SI / clinical units
    bitfield_*       Int8                 — decoded boolean sensor-status flags
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from ventilator_pdm.registry import FLEET_SERIALS

logger = logging.getLogger(__name__)

QUESTDB_URL = "http://localhost:9000/exec"
TABLE_NAME = "pdm_medical_device"


def query_questdb(
    sql: str,
    url: str = QUESTDB_URL,
    timeout: int = 600,
) -> pd.DataFrame:
    """Execute a SQL statement against the QuestDB HTTP query endpoint.

    Sends a GET request to the QuestDB ``/exec`` endpoint, parses the JSON
    response, and returns the result as a DataFrame.  Column names are taken
    directly from the ``columns`` array in the response payload.

    Args:
        sql: SQL statement to execute. Must be a single statement; QuestDB
            does not support multi-statement batches via the HTTP API.
        url: Full URL of the QuestDB ``/exec`` endpoint.  Defaults to
            ``http://localhost:9000/exec``.
        timeout: HTTP request timeout in seconds.  Defaults to 600 (10 min)
            to accommodate large date-range scans.

    Returns:
        DataFrame containing the query result.  Column dtypes are inferred
        by pandas from the raw JSON values; callers should cast as needed.
        Returns an empty DataFrame (zero rows) when the query matches no
        rows.

    Raises:
        requests.HTTPError: If the HTTP response status indicates an error
            (4xx or 5xx).
        RuntimeError: If the QuestDB JSON response contains an ``"error"``
            key, indicating a SQL-level error.
    """
    resp = requests.get(url, params={"query": sql}, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    if "error" in payload:
        raise RuntimeError(f"QuestDB error: {payload['error']}")

    columns = [c["name"] for c in payload["columns"]]
    rows = payload.get("dataset", [])
    return pd.DataFrame(rows, columns=columns)


def _build_day_query(
    day: str,
    variable_ids: set[int] | None = None,
) -> str:
    """Build SQL for one day of fleet-filtered data."""
    serials_csv = ", ".join(f"'{s}'" for s in sorted(FLEET_SERIALS))
    next_day = (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    clauses = [
        f"device_serial IN ({serials_csv})",
        f"timestamp >= '{day}T00:00:00.000000Z'",
        f"timestamp < '{next_day}T00:00:00.000000Z'",
    ]

    if variable_ids:
        # variable_id is SYMBOL type in QuestDB — quote the values
        ids_csv = ", ".join(f"'{v}'" for v in sorted(variable_ids))
        clauses.append(f"variable_id IN ({ids_csv})")

    where = " AND ".join(clauses)
    return f"SELECT * FROM {TABLE_NAME} WHERE {where} ORDER BY timestamp"


def _process_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw QuestDB result chunk."""
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    if "variable_id" in df.columns:
        df["variable_id"] = pd.to_numeric(df["variable_id"], errors="coerce").astype("Int64")

    # Cast bitfields to int8
    for col in df.columns:
        if col.startswith("bitfield_"):
            df[col] = df[col].astype("Int8")

    # Verify fleet-only
    if "device_serial" in df.columns:
        non_fleet = set(df["device_serial"].unique()) - FLEET_SERIALS
        if non_fleet:
            logger.warning("Dropping %d non-fleet serials: %s", len(non_fleet), non_fleet)
            df = df[df["device_serial"].isin(FLEET_SERIALS)]

    return df


def extract_fleet_data(
    output_dir: Path,
    exclude_before: str | None = "2025-11-17",
    variable_ids: set[int] | None = None,
    url: str = QUESTDB_URL,
    end_date: str | None = None,
) -> Path:
    """Extract fleet-only telemetry from QuestDB and write daily Parquet files.

    Queries the ``pdm_medical_device`` table one calendar day at a time to
    avoid HTTP timeouts and excessive memory use.  Only rows belonging to
    serials in ``FLEET_SERIALS`` are retained.  Each day is written to
    ``output_dir/fleet_YYYY-MM-DD.parquet``.  A ``manifest.json`` is written
    after all days complete.

    Days that produce zero rows are skipped silently.  Days where the query
    fails are logged as errors and skipped so that a single bad day does not
    abort a multi-week extraction.

    Args:
        output_dir: Directory to write Parquet files into.  Created
            (including parents) if it does not already exist.
        exclude_before: ISO date string (``"YYYY-MM-DD"``) acting as an
            inclusive lower bound for the extraction window.  Rows with
            timestamps before this date are excluded.  Defaults to
            ``"2025-11-17"`` (the first reliable fleet-deployment date).
            Pass ``None`` to start from the earliest row in the table.
        variable_ids: Restrict extraction to this set of ``variable_id``
            values.  Pass ``None`` to extract all variables present in the
            table for the fleet serials.
        url: QuestDB HTTP endpoint URL.  Defaults to
            ``http://localhost:9000/exec``.
        end_date: ISO date string (``"YYYY-MM-DD"``) acting as an inclusive
            upper bound for the extraction window.  Defaults to the latest
            timestamp found in the table.

    Returns:
        Path to ``output_dir`` (the directory containing the written files).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine date range
    range_sql = f"SELECT min(timestamp), max(timestamp) FROM {TABLE_NAME}"
    range_df = query_questdb(range_sql, url=url)
    db_min = pd.Timestamp(range_df.iloc[0, 0])
    db_max = pd.Timestamp(range_df.iloc[0, 1])

    start = pd.Timestamp(exclude_before, tz="UTC") if exclude_before else db_min
    end = pd.Timestamp(end_date, tz="UTC") if end_date else db_max
    if db_min.tzinfo is not None:
        start = start.tz_localize("UTC") if start.tzinfo is None else start
        end = end.tz_localize("UTC") if end.tzinfo is None else end
    else:
        start = start.tz_localize(None) if start.tzinfo is not None else start
        end = end.tz_localize(None) if end.tzinfo is not None else end
    start = max(start, db_min).normalize()
    end = end.normalize() + timedelta(days=1)

    logger.info("Extracting fleet data from %s to %s", start.date(), end.date())

    total_rows = 0
    n_chunks = 0
    day = start

    while day < end:
        day_str = day.strftime("%Y-%m-%d")
        sql = _build_day_query(day_str, variable_ids)

        logger.info("Querying %s...", day_str)
        try:
            df = query_questdb(sql, url=url)
        except Exception as e:
            logger.error("Failed on %s: %s", day_str, e)
            day += timedelta(days=1)
            continue

        if df.empty:
            logger.info("  %s: 0 rows (skipping)", day_str)
            day += timedelta(days=1)
            continue

        df = _process_chunk(df)

        chunk_path = output_dir / f"fleet_{day_str}.parquet"
        df.to_parquet(chunk_path, index=False)
        logger.info("  %s: %d rows → %s", day_str, len(df), chunk_path.name)

        total_rows += len(df)
        n_chunks += 1
        day += timedelta(days=1)

    _write_manifest(output_dir, total_rows, n_chunks, exclude_before, variable_ids)
    logger.info("Extraction complete: %d rows in %d daily files", total_rows, n_chunks)
    return output_dir


def _write_manifest(
    output_dir: Path,
    total_rows: int,
    n_chunks: int,
    exclude_before: str | None,
    variable_ids: set[int] | None = None,
) -> None:
    manifest = {
        "extracted_at": datetime.now().isoformat(),
        "total_rows": total_rows,
        "n_chunks": n_chunks,
        "exclude_before": exclude_before,
        "fleet_size": len(FLEET_SERIALS),
        "n_variable_ids": len(variable_ids) if variable_ids else "all",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Manifest written to %s", manifest_path)


def load_fleet_parquet(data_dir: Path) -> pd.DataFrame:
    """Load all daily fleet Parquet files from a directory into a single DataFrame.

    Reads every file matching the glob ``fleet_*.parquet`` inside
    ``data_dir``, concatenates them in chronological order (files are sorted
    by name, which is date-ordered because filenames follow
    ``fleet_YYYY-MM-DD.parquet``), and returns the result.

    Args:
        data_dir: Directory previously produced by :func:`extract_fleet_data`.
            Must contain at least one ``fleet_*.parquet`` file.

    Returns:
        DataFrame with columns as written by :func:`extract_fleet_data`
        (``timestamp``, ``device_serial``, ``variable_id``, ``value``,
        ``bitfield_*``).  The row index is reset to a contiguous integer
        range; the original per-file indices are discarded.

    Raises:
        FileNotFoundError: If no ``fleet_*.parquet`` files are found in
            ``data_dir``.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("fleet_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No fleet_*.parquet files in {data_dir}")

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d rows from %d files", len(df), len(files))
    return df
