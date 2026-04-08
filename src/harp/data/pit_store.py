"""Point-in-time (PIT) store for SMIM signal and intensity data.

Stores observations with dual timestamps — ``event_date`` (when the
economic event occurred) and ``pub_date`` (when the data became publicly
available). Querying with ``as_of`` enforces A1 compliance by returning
only rows where ``pub_date <= as_of``.

Storage format: Parquet files on disk, partitioned by ``source``.

Schema
------
Required columns in every stored frame:

    actor_id   : str   — actor identifier (CIK, ticker, country code, …)
    signal_id  : str   — signal / series identifier (FRED code, XBRL tag, …)
    event_date : datetime64[ns] (tz-naive)
    pub_date   : datetime64[ns] (tz-naive)
    value      : float64
    source     : str   — adapter source name ("fred", "edgar", …)
    vintage_id : str | None — optional revision identifier
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_SCHEMA_COLS = [
    "actor_id",
    "signal_id",
    "event_date",
    "pub_date",
    "value",
    "source",
    "vintage_id",
]
_REQUIRED_COLS = {"actor_id", "signal_id", "event_date", "pub_date", "value", "source"}


def _validate_schema(df: pd.DataFrame) -> None:
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required PIT columns: {missing}")


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct dtypes and add optional columns with defaults."""
    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.tz_localize(None)
    df["pub_date"] = pd.to_datetime(df["pub_date"]).dt.tz_localize(None)
    df["value"] = df["value"].astype("float64")
    df["actor_id"] = df["actor_id"].astype(str)
    df["signal_id"] = df["signal_id"].astype(str)
    df["source"] = df["source"].astype(str)
    if "vintage_id" not in df.columns:
        df["vintage_id"] = None
    # Reorder to canonical schema (extra columns preserved at end)
    extra = [c for c in df.columns if c not in _SCHEMA_COLS]
    return df[_SCHEMA_COLS + extra]


@dataclass
class PointInTimeStore:
    """Parquet-backed dual-timestamp store for SMIM observations.

    Each ``source`` is stored as a separate Parquet file under ``root_dir``:
    ``{root_dir}/{source}.parquet``

    Args:
        root_dir: Directory where Parquet files are written. Created on
            first write if it does not exist.
    """

    root_dir: Path | str = field(default="smim_pit_store")

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)

    # ── write ─────────────────────────────────────────────────────────────────

    def ingest(self, df: pd.DataFrame) -> None:
        """Write (append) a DataFrame to the store.

        Args:
            df: DataFrame conforming to the PIT schema. Must contain at
                minimum: ``actor_id``, ``signal_id``, ``event_date``,
                ``pub_date``, ``value``, ``source``.
        """
        _validate_schema(df)
        df = _normalise(df)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        for source, group in df.groupby("source"):
            path = self.root_dir / f"{source}.parquet"
            if path.exists():
                existing = pd.read_parquet(path)
                merged = pd.concat([existing, group], ignore_index=True)
                # De-duplicate on natural key: keep latest pub_date per
                # (actor_id, signal_id, event_date, source)
                merged = merged.sort_values("pub_date").drop_duplicates(
                    subset=["actor_id", "signal_id", "event_date", "source"],
                    keep="last",
                )
                merged.to_parquet(path, index=False)
            else:
                group.to_parquet(path, index=False)

    def bulk_ingest(self, frames: list[pd.DataFrame]) -> None:
        """Ingest multiple DataFrames in one pass.

        Args:
            frames: List of DataFrames, each conforming to the PIT schema.
        """
        if not frames:
            return
        combined = pd.concat(frames, ignore_index=True)
        self.ingest(combined)

    # ── read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        as_of: pd.Timestamp,
        *,
        sources: list[str] | None = None,
        actor_ids: list[str] | None = None,
        signal_ids: list[str] | None = None,
        date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    ) -> pd.DataFrame:
        """Return all observations available as of the given date.

        A1 filter: only rows with ``pub_date <= as_of`` are returned.

        Args:
            as_of: Point-in-time cutoff (A1 compliance).
            sources: If given, only load these source Parquet files.
            actor_ids: If given, filter to these actor IDs.
            signal_ids: If given, filter to these signal IDs.
            date_range: Optional ``(start, end)`` tuple to filter ``event_date``.

        Returns:
            DataFrame conforming to the PIT schema, filtered by A1 rule.
        """
        as_of_naive = as_of.tz_localize(None) if as_of.tzinfo else as_of
        frames: list[pd.DataFrame] = []

        for path in self._parquet_paths(sources):
            df = pd.read_parquet(path)
            df["pub_date"] = pd.to_datetime(df["pub_date"]).dt.tz_localize(None)
            df["event_date"] = pd.to_datetime(df["event_date"]).dt.tz_localize(None)
            df = df[df["pub_date"] <= as_of_naive]
            if actor_ids is not None:
                df = df[df["actor_id"].isin(actor_ids)]
            if signal_ids is not None:
                df = df[df["signal_id"].isin(signal_ids)]
            if date_range is not None:
                start, end = date_range
                df = df[(df["event_date"] >= start) & (df["event_date"] <= end)]
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame(columns=_SCHEMA_COLS)
        return pd.concat(frames, ignore_index=True)

    def query_signals(
        self,
        signal_ids: list[str],
        as_of: pd.Timestamp,
        date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    ) -> pd.DataFrame:
        """Convenience wrapper: query by signal_ids."""
        return self.query(
            as_of=as_of,
            signal_ids=signal_ids,
            date_range=date_range,
        )

    def query_intensity(
        self,
        actor_ids: list[str],
        as_of: pd.Timestamp,
        date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    ) -> pd.DataFrame:
        """Convenience wrapper: query by actor_ids (for intensity data)."""
        return self.query(
            as_of=as_of,
            actor_ids=actor_ids,
            date_range=date_range,
        )

    # ── catalogue ─────────────────────────────────────────────────────────────

    def list_actors(self, source: str | None = None) -> list[str]:
        """Return sorted list of unique actor_ids in the store."""
        frames = [
            pd.read_parquet(p)[["actor_id"]]
            for p in self._parquet_paths([source] if source else None)
        ]
        if not frames:
            return []
        return sorted(pd.concat(frames)["actor_id"].unique().tolist())

    def list_signals(self, source: str | None = None) -> list[str]:
        """Return sorted list of unique signal_ids in the store."""
        frames = [
            pd.read_parquet(p)[["signal_id"]]
            for p in self._parquet_paths([source] if source else None)
        ]
        if not frames:
            return []
        return sorted(pd.concat(frames)["signal_id"].unique().tolist())

    def list_sources(self) -> list[str]:
        """Return names of all sources with data in the store."""
        return [p.stem for p in sorted(self.root_dir.glob("*.parquet"))]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _parquet_paths(self, sources: list[str] | None) -> list[Path]:
        if not self.root_dir.exists():
            return []
        if sources is not None:
            paths = [self.root_dir / f"{s}.parquet" for s in sources]
            return [p for p in paths if p.exists()]
        return list(self.root_dir.glob("*.parquet"))
