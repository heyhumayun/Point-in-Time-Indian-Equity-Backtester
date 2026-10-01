from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import re
import urllib.request

import pandas as pd

from .data_loader import normalize_symbol
from .dataset_metadata import create_dataset_metadata, infer_survivorship_bias_status, infer_universe_source_quality, write_dataset_metadata


NIFTY_500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/NIFTY_500"
NIFTY_500_WIKIPEDIA_RAW_URL = "https://en.wikipedia.org/w/index.php?title=NIFTY_500&action=raw"


def download_nifty_500_constituents() -> pd.DataFrame:
    request = urllib.request.Request(NIFTY_500_WIKIPEDIA_RAW_URL, headers={"User-Agent": "Mozilla/5.0"})
    raw_text = urllib.request.urlopen(request, timeout=30).read().decode("utf-8", errors="ignore")
    lines = raw_text.splitlines()
    constituents_started = False
    current_row: List[str] = []
    rows: List[List[str]] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|+Nifty 500 List"):
            constituents_started = True
            continue
        if not constituents_started:
            continue
        if stripped == "|}":
            break
        if stripped == "|-":
            if len(current_row) >= 6:
                rows.append(current_row[:6])
            current_row = []
            continue
        if stripped.startswith("|'''"):
            continue
        if stripped.startswith("|"):
            current_row.append(_clean_wiki_cell(stripped[1:]))
    if len(current_row) >= 6:
        rows.append(current_row[:6])

    universe = pd.DataFrame(rows, columns=["sl_no", "company_name", "sector", "symbol", "series", "isin_code"])
    universe["ticker"] = universe["symbol"].map(normalize_symbol)
    universe["sector"] = universe["sector"].replace("", "Unknown").fillna("Unknown")
    universe["company_name"] = universe["company_name"].replace("", pd.NA).fillna(universe["ticker"])
    universe = universe[["ticker", "company_name", "sector"]].drop_duplicates(subset=["ticker"]).sort_values("ticker").reset_index(drop=True)
    if len(universe) < 450:
        raise ValueError(f"Expected a broad NIFTY 500 universe, but only found {len(universe)} symbols.")
    return universe


def build_dynamic_universe_from_prices(universe: pd.DataFrame, prices: pd.DataFrame, end_date: str) -> pd.DataFrame:
    first_dates = (
        prices.loc[~prices["symbol"].str.startswith("^")]
        .groupby("symbol", as_index=False)["date"]
        .min()
        .rename(columns={"symbol": "ticker", "date": "start_date"})
    )
    merged = universe.merge(first_dates, on="ticker", how="inner")
    merged["end_date"] = ""
    merged["start_date"] = pd.to_datetime(merged["start_date"]).dt.date.astype(str)
    merged["source"] = "wikipedia_current_nifty500_snapshot"
    merged["notes"] = "Current constituent snapshot projected backward from the earliest available price date."
    return merged[["ticker", "company_name", "start_date", "end_date", "source", "notes", "sector"]].sort_values("ticker").reset_index(drop=True)


def write_downloaded_dataset_metadata(
    metadata_path: Path,
    universe_source: str,
    price_source: str,
    notes: List[str],
    universe_start_date: Optional[str] = None,
    universe_end_date: Optional[str] = None,
) -> None:
    write_dataset_metadata(
        metadata_path,
        create_dataset_metadata(
            dataset_type="downloaded",
            universe_source=universe_source,
            price_source=price_source,
            fundamentals_source="manual_or_optional_for_price_only_strategy",
            universe_mode="current_snapshot_only",
            universe_snapshot_count=1,
            universe_snapshot_dates=[universe_start_date] if universe_start_date else [],
            universe_snapshot_min_date=universe_start_date,
            universe_snapshot_max_date=universe_start_date,
            universe_snapshot_frequency="current_snapshot_only",
            universe_source_quality=infer_universe_source_quality(["manual"]),
            universe_source_types=["manual"],
            survivorship_bias_status=infer_survivorship_bias_status("current_snapshot_only", "current_snapshot_only"),
            universe_start_date=universe_start_date,
            universe_end_date=universe_end_date,
            notes=notes,
        ),
    )


def _clean_wiki_cell(value: str) -> str:
    cleaned = value.strip()
    if "|" in cleaned and cleaned.startswith("[["):
        cleaned = cleaned.split("|")[-1]
    cleaned = cleaned.replace("[[", "").replace("]]", "")
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = cleaned.replace("'''", "").replace("''", "")
    return cleaned.strip()
