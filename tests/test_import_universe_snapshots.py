from pathlib import Path

import pandas as pd
import pytest

from src.dataset_metadata import infer_survivorship_bias_status
from src.import_universe_snapshots import import_snapshot_files, load_manifest, normalize_snapshot_frame
from src.validation import ValidationError


def test_import_csv_snapshot_normalizes_and_updates_manifest(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    manifest_path = output_dir / "sources_manifest.csv"
    input_dir.mkdir()
    pd.DataFrame({"Symbol": ["abc", "XYZ.NS"], "Company Name": ["ABC Ltd", "XYZ Ltd"]}).to_csv(
        input_dir / "nifty500_2020-01-01.csv", index=False
    )

    imported = import_snapshot_files(
        input_dir=input_dir,
        output_dir=output_dir,
        manifest_path=manifest_path,
        source_type="official_niftyindices",
        source_name="Nifty Indices",
        source_url="https://example.com",
        downloaded_date="2026-06-22",
        notes="manual import",
    )

    assert len(imported) == 1
    normalized = pd.read_csv(imported[0])
    assert normalized["ticker"].tolist() == ["ABC.NS", "XYZ.NS"]
    manifest = load_manifest(manifest_path)
    assert manifest.iloc[0]["source_type"] == "official_niftyindices"


def test_import_excel_snapshot_if_engine_available(tmp_path: Path):
    pytest.importorskip("openpyxl")
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    manifest_path = output_dir / "sources_manifest.csv"
    input_dir.mkdir()
    pd.DataFrame({"ticker": ["AAA", "BBB"], "company_name": ["AAA Ltd", "BBB Ltd"]}).to_excel(
        input_dir / "nifty500_2021-01-01.xlsx", index=False
    )
    imported = import_snapshot_files(
        input_dir=input_dir,
        output_dir=output_dir,
        manifest_path=manifest_path,
        source_type="manual",
    )
    assert imported[0].name == "nifty500_2021-01-01.csv"


def test_reject_duplicate_tickers_in_snapshot():
    frame = pd.DataFrame({"ticker": ["AAA", "AAA"], "company_name": ["AAA Ltd", "AAA Ltd"]})
    with pytest.raises(ValidationError):
        normalize_snapshot_frame(frame, source_name="dup")


def test_warn_when_snapshot_count_far_from_500():
    frame = pd.DataFrame({"ticker": [f"T{i}" for i in range(10)], "company_name": [f"Name {i}" for i in range(10)]})
    with pytest.warns(UserWarning):
        normalize_snapshot_frame(frame, source_name="small")


def test_load_manifest_creates_and_reads_headers(tmp_path: Path):
    manifest_path = tmp_path / "sources_manifest.csv"
    manifest = load_manifest(manifest_path)
    assert manifest.columns.tolist() == [
        "snapshot_date",
        "filename",
        "source_url",
        "source_name",
        "downloaded_date",
        "source_type",
        "notes",
    ]


def test_survivorship_bias_status_assignment():
    assert infer_survivorship_bias_status("current_snapshot_only", "current_snapshot_only") == "high"
    assert infer_survivorship_bias_status("annual_snapshots", "annual_snapshots") == "medium"
    assert infer_survivorship_bias_status("quarterly_snapshots", "quarterly_snapshots") == "medium_low"
    assert infer_survivorship_bias_status("monthly_snapshots", "monthly_snapshots") == "low"
