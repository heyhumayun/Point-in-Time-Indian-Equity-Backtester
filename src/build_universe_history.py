from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

from .config import PathConfig
from .dataset_metadata import (
    infer_survivorship_bias_status,
    infer_universe_source_quality,
    load_dataset_metadata,
    write_dataset_metadata,
)
from .import_universe_snapshots import load_manifest
from .universe_history import load_universe_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build point-in-time universe intervals from dated NIFTY 500 snapshot CSV files.")
    parser.add_argument("--snapshots-dir", default=None)
    parser.add_argument("--output", "--output-file", dest="output_file", default=None)
    parser.add_argument("--metadata-file", default=None)
    return parser.parse_args()


def infer_snapshot_frequency(snapshot_dates: List[pd.Timestamp]) -> str:
    if len(snapshot_dates) < 2:
        return "current_snapshot_only"
    day_deltas = sorted((snapshot_dates[idx] - snapshot_dates[idx - 1]).days for idx in range(1, len(snapshot_dates)))
    median_delta = day_deltas[len(day_deltas) // 2]
    if median_delta <= 35:
        return "monthly_snapshots"
    if median_delta <= 110:
        return "quarterly_snapshots"
    if median_delta <= 400:
        return "annual_snapshots"
    return "historical_snapshots"


def build_universe_history_from_snapshots(snapshots_dir: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    manifest_path = snapshots_dir / "sources_manifest.csv"
    manifest = load_manifest(manifest_path)
    snapshot_files = sorted(snapshots_dir.glob("nifty500_*.csv"))
    if not snapshot_files:
        raise FileNotFoundError(
            f"No snapshot files found in {snapshots_dir}. Add dated files like nifty500_2016-01-01.csv before building historical universe intervals."
        )

    snapshots: List[Tuple[pd.Timestamp, pd.DataFrame, Path]] = []
    snapshot_stats: List[Dict[str, object]] = []
    for snapshot_file in snapshot_files:
        date_text = snapshot_file.stem.replace("nifty500_", "")
        snapshot_date = pd.Timestamp(date_text)
        frame = pd.read_csv(snapshot_file, keep_default_na=False).copy()
        missing = [column for column in ["ticker", "company_name"] if column not in frame.columns]
        if missing:
            raise ValueError(f"Snapshot file {snapshot_file} is missing required columns: {missing}")
        frame["ticker"] = frame["ticker"].astype(str).str.strip()
        frame["company_name"] = frame["company_name"].astype(str).str.strip()
        frame = frame.loc[frame["ticker"] != ""].drop_duplicates(subset=["ticker"]).reset_index(drop=True)
        snapshots.append((snapshot_date, frame, snapshot_file))
        snapshot_stats.append(
            {
                "snapshot_date": snapshot_date.date().isoformat(),
                "filename": snapshot_file.name,
                "constituent_count": int(len(frame)),
            }
        )
        if len(frame) < 450 or len(frame) > 550:
            warnings.warn(
                f"Snapshot {snapshot_file.name} has {len(frame)} constituents, outside the expected 450-550 range.",
                stacklevel=2,
            )

    interval_map: Dict[str, Dict[str, object]] = {}
    rows: List[Dict[str, object]] = []
    snapshot_dates = [item[0] for item in snapshots]
    next_dates = snapshot_dates[1:] + [None]
    manifest_window = manifest.loc[manifest["filename"].isin([path.name for path in snapshot_files])].copy()
    manifest_by_filename = {row["filename"]: row for row in manifest_window.to_dict("records")}
    transition_records: List[Dict[str, object]] = []
    previous_tickers: Set[str] = set()

    for (snapshot_date, frame, snapshot_file), next_snapshot_date in zip(snapshots, next_dates):
        current_tickers = set(frame["ticker"].tolist())
        entered = sorted(current_tickers - previous_tickers) if previous_tickers else sorted(current_tickers)
        exited = sorted(previous_tickers - current_tickers) if previous_tickers else []
        transition_records.append(
            {
                "snapshot_date": snapshot_date.date().isoformat(),
                "filename": snapshot_file.name,
                "constituent_count": len(current_tickers),
                "entered_count": len(entered),
                "exited_count": len(exited),
            }
        )
        for _, record in frame.iterrows():
            ticker = record["ticker"]
            if ticker not in interval_map:
                manifest_row = manifest_by_filename.get(snapshot_file.name, {})
                interval_map[ticker] = {
                    "company_name": record["company_name"],
                    "start_date": snapshot_date,
                    "last_seen": snapshot_date,
                    "source": manifest_row.get("source_name") or manifest_row.get("source_type") or snapshot_file.name,
                    "notes": manifest_row.get("notes", ""),
                }
            else:
                interval_map[ticker]["last_seen"] = snapshot_date

        closed_tickers = []
        for ticker, interval in interval_map.items():
            if ticker in current_tickers:
                continue
            end_date = snapshot_date - pd.Timedelta(days=1)
            rows.append(
                {
                    "ticker": ticker,
                    "company_name": interval["company_name"],
                    "start_date": interval["start_date"].date().isoformat(),
                    "end_date": end_date.date().isoformat(),
                    "source": interval["source"],
                    "notes": f"{interval.get('notes', '')} Inferred from dated snapshots through {interval['last_seen'].date().isoformat()}".strip(),
                }
            )
            closed_tickers.append(ticker)
        for ticker in closed_tickers:
            interval_map.pop(ticker, None)

        if next_snapshot_date is None:
            for ticker, interval in interval_map.items():
                rows.append(
                    {
                        "ticker": ticker,
                        "company_name": interval["company_name"],
                        "start_date": interval["start_date"].date().isoformat(),
                        "end_date": "",
                        "source": interval["source"],
                        "notes": f"{interval.get('notes', '')} Open interval inferred from final snapshot dated {snapshot_date.date().isoformat()}".strip(),
                    }
                )
        previous_tickers = current_tickers

    universe = pd.DataFrame(rows).sort_values(["ticker", "start_date", "end_date"]).reset_index(drop=True)
    mode = infer_snapshot_frequency(snapshot_dates)
    snapshot_date_strings = [date.date().isoformat() for date in snapshot_dates]
    source_types = sorted(manifest_window["source_type"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist())
    source_quality = infer_universe_source_quality(source_types)
    survivorship_bias_status = infer_survivorship_bias_status(mode, mode)
    warnings_list: List[str] = []
    if mode == "annual_snapshots":
        warnings_list.append(
            "Annual snapshots reduce but do not eliminate survivorship bias because index changes between snapshot dates may be missed."
        )
    metadata_fields = {
        "universe_mode": mode,
        "universe_snapshot_count": len(snapshot_files),
        "universe_snapshot_dates": snapshot_date_strings,
        "universe_snapshot_min_date": snapshot_date_strings[0],
        "universe_snapshot_max_date": snapshot_date_strings[-1],
        "universe_snapshot_frequency": mode,
        "universe_source_quality": source_quality,
        "universe_source_types": source_types,
        "survivorship_bias_status": survivorship_bias_status,
        "universe_start_date": snapshot_dates[0].date().isoformat(),
        "universe_end_date": snapshot_dates[-1].date().isoformat(),
        "snapshot_transition_summary": pd.DataFrame(transition_records),
        "snapshot_constituent_summary": pd.DataFrame(snapshot_stats),
        "build_warnings": warnings_list,
    }
    return universe, metadata_fields


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    paths = PathConfig.from_project_root(project_root)
    snapshots_dir = Path(args.snapshots_dir) if args.snapshots_dir else paths.universe_snapshots_dir
    output_file = Path(args.output_file) if args.output_file else paths.universe_file
    metadata_file = Path(args.metadata_file) if args.metadata_file else paths.dataset_metadata_file

    universe, metadata_fields = build_universe_history_from_snapshots(snapshots_dir)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(output_file, index=False)
    load_universe_history(output_file)
    coverage_summary_path = output_file.parent / "universe_coverage_summary.csv"
    metadata_fields["snapshot_constituent_summary"].to_csv(coverage_summary_path, index=False)
    transition_summary_path = output_file.parent / "universe_snapshot_transitions.csv"
    metadata_fields["snapshot_transition_summary"].to_csv(transition_summary_path, index=False)

    metadata = load_dataset_metadata(metadata_file, fallback_symbols=universe["ticker"].tolist())
    warnings_list = list(metadata.notes)
    warnings_list.extend(metadata_fields.pop("build_warnings"))
    snapshot_transition_summary = metadata_fields.pop("snapshot_transition_summary")
    snapshot_constituent_summary = metadata_fields.pop("snapshot_constituent_summary")
    if paths.prices_file.exists():
        prices = pd.read_csv(paths.prices_file, usecols=["date"])
        first_price_date = pd.to_datetime(prices["date"]).min()
        first_snapshot_date = pd.Timestamp(metadata_fields["universe_snapshot_min_date"]) if metadata_fields.get("universe_snapshot_min_date") else None
        if first_snapshot_date is not None and first_price_date < first_snapshot_date:
            warnings_list.append("Backtest before first historical universe snapshot is not point-in-time reliable.")
    updated_metadata = type(metadata)(**{**metadata.__dict__, **metadata_fields, "notes": warnings_list})
    write_dataset_metadata(metadata_file, updated_metadata)
    print(f"Saved historical universe to {output_file}")
    print(f"Saved universe coverage summary to {coverage_summary_path}")
    print(f"Saved snapshot transition summary to {transition_summary_path}")
    print(f"Updated dataset metadata at {metadata_file}")
    for row in snapshot_transition_summary.to_dict("records"):
        print(
            f"{row['snapshot_date']}: constituents={row['constituent_count']}, entered={row['entered_count']}, exited={row['exited_count']}"
        )
    for note in updated_metadata.notes:
        if "survivorship bias" in note.lower() or "point-in-time" in note.lower():
            print(f"Warning: {note}")


if __name__ == "__main__":
    main()
