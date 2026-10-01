from __future__ import annotations

import argparse
import csv
import shutil
import warnings
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import PathConfig
from .universe_history import _normalize_ticker
from .validation import ValidationError


MANIFEST_COLUMNS = [
    "snapshot_date",
    "filename",
    "source_url",
    "source_name",
    "downloaded_date",
    "source_type",
    "notes",
]
ALLOWED_SOURCE_TYPES = {"official_nse", "official_niftyindices", "broker_archive", "data_vendor", "manual", "unknown"}
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".zip"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import manually downloaded NIFTY 500 snapshot files into normalized dated CSV snapshots.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--source-type", required=True, choices=sorted(ALLOWED_SOURCE_TYPES))
    parser.add_argument("--source-name", default="")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--downloaded-date", default=None)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def ensure_manifest(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()


def load_manifest(path: Path) -> pd.DataFrame:
    ensure_manifest(path)
    manifest = pd.read_csv(path, keep_default_na=False)
    missing = [column for column in MANIFEST_COLUMNS if column not in manifest.columns]
    if missing:
        raise ValidationError(f"Snapshot sources manifest is missing required columns: {missing}")
    return manifest[MANIFEST_COLUMNS].copy()


def parse_snapshot_date_from_filename(path: Path) -> str:
    stem = path.stem
    for token in stem.replace(".", "_").replace("-", "_").split("_"):
        token = token.strip()
        if len(token) == 8 and token.isdigit():
            return pd.Timestamp(datetime.strptime(token, "%Y%m%d")).date().isoformat()
        try:
            return pd.Timestamp(token).date().isoformat()
        except Exception:
            continue
    raise ValidationError(f"Could not parse snapshot date from filename '{path.name}'. Include a date like 2016-01-01 or 20160101.")


def read_snapshot_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, keep_default_na=False)
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path, keep_default_na=False)
        except ImportError as exc:
            raise ImportError(f"Excel import for {path.name} requires an installed Excel engine such as openpyxl or xlrd.") from exc
    if suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if Path(name).suffix.lower() in {".csv", ".xlsx", ".xls"} and not name.endswith("/")]
            if not members:
                raise ValidationError(f"ZIP file {path.name} does not contain a supported snapshot file.")
            if len(members) > 1:
                warnings.warn(f"ZIP file {path.name} contains multiple supported files. Using {members[0]}.", stacklevel=2)
            member = members[0]
            extracted_path = path.parent / f"__tmp_{Path(member).name}"
            with archive.open(member) as src, extracted_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            try:
                return read_snapshot_table(extracted_path)
            finally:
                extracted_path.unlink(missing_ok=True)
    raise ValidationError(f"Unsupported snapshot file type '{path.suffix}' for {path.name}.")


def normalize_snapshot_frame(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    rename_map = {}
    for column in frame.columns:
        normalized = str(column).strip().lower().replace(" ", "_")
        if normalized in {"ticker", "symbol", "stock_code"}:
            rename_map[column] = "ticker"
        elif normalized in {"company_name", "company", "security_name", "name"}:
            rename_map[column] = "company_name"
    frame = frame.rename(columns=rename_map).copy()
    missing = [column for column in ["ticker", "company_name"] if column not in frame.columns]
    if missing:
        raise ValidationError(f"Snapshot data is missing required columns: {missing}")

    frame["ticker"] = frame["ticker"].astype(str).str.strip().map(_normalize_ticker)
    frame["company_name"] = frame["company_name"].astype(str).str.strip()
    if (frame["ticker"] == "").any():
        raise ValidationError("Snapshot contains blank tickers.")
    if frame["ticker"].duplicated().any():
        duplicates = sorted(frame.loc[frame["ticker"].duplicated(), "ticker"].unique().tolist())
        raise ValidationError(f"Snapshot contains duplicate tickers: {duplicates[:10]}")
    frame = frame.loc[:, ["ticker", "company_name"]].drop_duplicates(subset=["ticker"]).sort_values("ticker").reset_index(drop=True)

    count = len(frame)
    if count < 450 or count > 550:
        warnings.warn(
            f"Snapshot from {source_name or 'unknown source'} has {count} constituents, outside the expected 450-550 range.",
            stacklevel=2,
        )
    return frame


def append_manifest_row(manifest_path: Path, row: Dict[str, str]) -> None:
    manifest = load_manifest(manifest_path)
    manifest = manifest.loc[~((manifest["snapshot_date"] == row["snapshot_date"]) & (manifest["filename"] == row["filename"]))].copy()
    manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
    manifest = manifest.sort_values(["snapshot_date", "filename"]).reset_index(drop=True)
    manifest.to_csv(manifest_path, index=False)


def import_snapshot_files(
    input_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    source_type: str,
    source_name: str = "",
    source_url: str = "",
    downloaded_date: Optional[str] = None,
    notes: str = "",
) -> List[Path]:
    if source_type not in ALLOWED_SOURCE_TYPES:
        raise ValidationError(f"Unsupported source_type '{source_type}'. Expected one of {sorted(ALLOWED_SOURCE_TYPES)}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_manifest(manifest_path)
    imported_paths: List[Path] = []
    download_date_value = downloaded_date or pd.Timestamp.utcnow().date().isoformat()
    files = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No supported snapshot files found in {input_dir}.")

    for path in files:
        snapshot_date = parse_snapshot_date_from_filename(path)
        normalized = normalize_snapshot_frame(read_snapshot_table(path), source_name=source_name or path.name)
        output_path = output_dir / f"nifty500_{snapshot_date}.csv"
        normalized.to_csv(output_path, index=False)
        append_manifest_row(
            manifest_path,
            {
                "snapshot_date": snapshot_date,
                "filename": output_path.name,
                "source_url": source_url,
                "source_name": source_name or path.name,
                "downloaded_date": download_date_value,
                "source_type": source_type,
                "notes": notes,
            },
        )
        imported_paths.append(output_path)
    return imported_paths


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    paths = PathConfig.from_project_root(project_root)
    imported = import_snapshot_files(
        input_dir=Path(args.input_dir),
        output_dir=paths.universe_snapshots_dir,
        manifest_path=paths.universe_snapshots_dir / "sources_manifest.csv",
        source_type=args.source_type,
        source_name=args.source_name,
        source_url=args.source_url,
        downloaded_date=args.downloaded_date,
        notes=args.notes,
    )
    print(f"Imported {len(imported)} snapshot file(s) into {paths.universe_snapshots_dir}")
    for path in imported:
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
