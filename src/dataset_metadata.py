from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from .validation import ValidationError


DATASET_TYPES = {"synthetic", "downloaded", "manual"}
DEMO_SYMBOLS = {
    "ALPHA.NS",
    "BETA.NS",
    "GAMMA.NS",
    "DELTA.NS",
    "EPSILON.NS",
    "ZETA.NS",
    "ETA.NS",
    "THETA.NS",
    "IOTA.NS",
    "KAPPA.NS",
    "LAMBDA.NS",
    "MU.NS",
    "NU.NS",
    "XI.NS",
    "OMICRON.NS",
    "PI.NS",
    "RHO.NS",
    "SIGMA.NS",
}


@dataclass(frozen=True)
class DatasetMetadata:
    dataset_type: str
    created_at_utc: str
    price_source: str
    universe_source: str
    fundamentals_source: str
    universe_mode: str = "unknown"
    universe_snapshot_count: int = 0
    universe_snapshot_dates: List[str] = field(default_factory=list)
    universe_snapshot_min_date: Optional[str] = None
    universe_snapshot_max_date: Optional[str] = None
    universe_snapshot_frequency: str = "unknown"
    universe_source_quality: str = "unknown"
    universe_source_types: List[str] = field(default_factory=list)
    survivorship_bias_status: str = "unknown"
    universe_start_date: Optional[str] = None
    universe_end_date: Optional[str] = None
    notes: List[str] = field(default_factory=list)


def create_dataset_metadata(
    dataset_type: str,
    price_source: str,
    universe_source: str,
    fundamentals_source: str,
    universe_mode: str = "unknown",
    universe_snapshot_count: int = 0,
    universe_snapshot_dates: Optional[List[str]] = None,
    universe_snapshot_min_date: Optional[str] = None,
    universe_snapshot_max_date: Optional[str] = None,
    universe_snapshot_frequency: str = "unknown",
    universe_source_quality: str = "unknown",
    universe_source_types: Optional[List[str]] = None,
    survivorship_bias_status: str = "unknown",
    universe_start_date: Optional[str] = None,
    universe_end_date: Optional[str] = None,
    notes: Optional[List[str]] = None,
) -> DatasetMetadata:
    if dataset_type not in DATASET_TYPES:
        raise ValidationError(f"Unsupported dataset_type '{dataset_type}'. Expected one of {sorted(DATASET_TYPES)}.")
    return DatasetMetadata(
        dataset_type=dataset_type,
        created_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        price_source=price_source,
        universe_source=universe_source,
        fundamentals_source=fundamentals_source,
        universe_mode=universe_mode,
        universe_snapshot_count=universe_snapshot_count,
        universe_snapshot_dates=universe_snapshot_dates or [],
        universe_snapshot_min_date=universe_snapshot_min_date,
        universe_snapshot_max_date=universe_snapshot_max_date,
        universe_snapshot_frequency=universe_snapshot_frequency,
        universe_source_quality=universe_source_quality,
        universe_source_types=universe_source_types or [],
        survivorship_bias_status=survivorship_bias_status,
        universe_start_date=universe_start_date,
        universe_end_date=universe_end_date,
        notes=notes or [],
    )


def infer_dataset_type_from_symbols(symbols: List[str]) -> str:
    normalized = {str(symbol).strip().upper() for symbol in symbols}
    return "synthetic" if normalized and normalized.issubset(DEMO_SYMBOLS) else "manual"


def load_dataset_metadata(path: Path, fallback_symbols: Optional[List[str]] = None) -> DatasetMetadata:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = DatasetMetadata(
            dataset_type=payload["dataset_type"],
            created_at_utc=payload["created_at_utc"],
            price_source=payload["price_source"],
            universe_source=payload["universe_source"],
            fundamentals_source=payload["fundamentals_source"],
            universe_mode=payload.get("universe_mode", "unknown"),
            universe_snapshot_count=int(payload.get("universe_snapshot_count", 0)),
            universe_snapshot_dates=list(payload.get("universe_snapshot_dates", [])),
            universe_snapshot_min_date=payload.get("universe_snapshot_min_date"),
            universe_snapshot_max_date=payload.get("universe_snapshot_max_date"),
            universe_snapshot_frequency=payload.get("universe_snapshot_frequency", "unknown"),
            universe_source_quality=payload.get("universe_source_quality", "unknown"),
            universe_source_types=list(payload.get("universe_source_types", [])),
            survivorship_bias_status=payload.get("survivorship_bias_status", "unknown"),
            universe_start_date=payload.get("universe_start_date"),
            universe_end_date=payload.get("universe_end_date"),
            notes=list(payload.get("notes", [])),
        )
        if metadata.dataset_type not in DATASET_TYPES:
            raise ValidationError(
                f"Dataset metadata file {path} has unsupported dataset_type '{metadata.dataset_type}'."
            )
        return metadata

    inferred_type = infer_dataset_type_from_symbols(fallback_symbols or [])
    return create_dataset_metadata(
        dataset_type=inferred_type,
        price_source="inferred_without_metadata_file",
        universe_source="inferred_without_metadata_file",
        fundamentals_source="inferred_without_metadata_file",
        universe_mode="unknown",
        universe_snapshot_count=0,
        universe_snapshot_dates=[],
        universe_snapshot_min_date=None,
        universe_snapshot_max_date=None,
        universe_snapshot_frequency="unknown",
        universe_source_quality="unknown",
        universe_source_types=[],
        survivorship_bias_status="unknown",
        notes=[
            "dataset_metadata.json was missing, so dataset_type was inferred from the current universe symbols.",
        ],
    )


def write_dataset_metadata(path: Path, metadata: DatasetMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")


def ensure_research_dataset(metadata: DatasetMetadata, report_as_research: bool) -> None:
    if report_as_research and metadata.dataset_type == "synthetic":
        raise ValidationError(
            "Synthetic datasets cannot be labeled as research. Re-run without --as-research or switch to downloaded/manual data."
        )


def infer_survivorship_bias_status(universe_mode: str, universe_snapshot_frequency: str) -> str:
    if universe_mode == "current_snapshot_only":
        return "high"
    if universe_snapshot_frequency == "annual_snapshots":
        return "medium"
    if universe_snapshot_frequency == "quarterly_snapshots":
        return "medium_low"
    if universe_snapshot_frequency in {"monthly_snapshots", "exact_point_in_time"}:
        return "low"
    return "unknown"


def infer_universe_source_quality(source_types: Sequence[str]) -> str:
    normalized = {str(value).strip() for value in source_types if str(value).strip()}
    if not normalized:
        return "unknown"
    if normalized.issubset({"official_nse", "official_niftyindices"}):
        return "high"
    if normalized.issubset({"official_nse", "official_niftyindices", "broker_archive", "data_vendor"}):
        return "medium"
    if "manual" in normalized or "unknown" in normalized:
        return "low"
    return "unknown"
