from pathlib import Path

import pandas as pd
import pytest

from src.build_universe_history import build_universe_history_from_snapshots
from src.universe_history import get_universe_on_date, summarize_universe_coverage, validate_universe_intervals
from src.validation import ValidationError


def test_get_universe_on_date_respects_interval_bounds_and_blank_end_dates():
    universe = pd.DataFrame(
        {
            "ticker": ["AAA.NS", "BBB.NS"],
            "company_name": ["AAA", "BBB"],
            "start_date": ["2020-01-01", "2020-06-01"],
            "end_date": ["2020-12-31", ""],
            "source": ["snap", "snap"],
            "notes": ["", ""],
        }
    )
    june = get_universe_on_date(universe, pd.Timestamp("2020-06-15"))
    assert sorted(june["ticker"].tolist()) == ["AAA.NS", "BBB.NS"]
    jan_2021 = get_universe_on_date(universe, pd.Timestamp("2021-01-15"))
    assert jan_2021["ticker"].tolist() == ["BBB.NS"]


def test_validate_universe_intervals_rejects_duplicate_and_invalid_ranges():
    duplicate = pd.DataFrame(
        {
            "ticker": ["AAA.NS", "AAA.NS"],
            "company_name": ["AAA", "AAA"],
            "start_date": ["2020-01-01", "2020-01-01"],
            "end_date": ["2020-12-31", "2020-12-31"],
            "source": ["snap", "snap"],
            "notes": ["", ""],
        }
    )
    with pytest.raises(ValidationError):
        validate_universe_intervals(duplicate)

    invalid = pd.DataFrame(
        {
            "ticker": ["AAA.NS"],
            "company_name": ["AAA"],
            "start_date": ["2020-02-01"],
            "end_date": ["2020-01-01"],
            "source": ["snap"],
            "notes": [""],
        }
    )
    with pytest.raises(ValidationError):
        validate_universe_intervals(invalid)


def test_summarize_universe_coverage_reports_monthly_counts():
    universe = pd.DataFrame(
        {
            "ticker": ["AAA.NS", "BBB.NS"],
            "company_name": ["AAA", "BBB"],
            "start_date": ["2020-01-01", "2020-03-01"],
            "end_date": ["", ""],
            "source": ["snap", "snap"],
            "notes": ["", ""],
        }
    )
    summary = summarize_universe_coverage(universe, "2020-01-01", "2020-03-31")
    assert summary["eligible_ticker_count"].tolist() == [1, 1, 2]


def test_build_universe_history_from_snapshots_infers_intervals(tmp_path: Path):
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()
    pd.DataFrame({"ticker": ["AAA.NS", "BBB.NS"], "company_name": ["AAA", "BBB"]}).to_csv(
        snapshots_dir / "nifty500_2020-01-01.csv", index=False
    )
    pd.DataFrame({"ticker": ["AAA.NS", "CCC.NS"], "company_name": ["AAA", "CCC"]}).to_csv(
        snapshots_dir / "nifty500_2021-01-01.csv", index=False
    )

    universe, metadata_fields = build_universe_history_from_snapshots(snapshots_dir)
    aaa = universe.loc[universe["ticker"] == "AAA.NS"].iloc[0]
    bbb = universe.loc[universe["ticker"] == "BBB.NS"].iloc[0]
    ccc = universe.loc[universe["ticker"] == "CCC.NS"].iloc[0]

    assert aaa["start_date"] == "2020-01-01"
    assert aaa["end_date"] == ""
    assert bbb["end_date"] == "2020-12-31"
    assert ccc["start_date"] == "2021-01-01"
    assert metadata_fields["universe_mode"] == "annual_snapshots"


def test_build_universe_history_handles_ticker_reappearing(tmp_path: Path):
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()
    (snapshots_dir / "sources_manifest.csv").write_text(
        "snapshot_date,filename,source_url,source_name,downloaded_date,source_type,notes\n"
        "2020-01-01,nifty500_2020-01-01.csv,,,2026-06-22,manual,\n"
        "2021-01-01,nifty500_2021-01-01.csv,,,2026-06-22,manual,\n"
        "2022-01-01,nifty500_2022-01-01.csv,,,2026-06-22,manual,\n",
        encoding="utf-8",
    )
    pd.DataFrame({"ticker": ["AAA.NS"], "company_name": ["AAA"]}).to_csv(snapshots_dir / "nifty500_2020-01-01.csv", index=False)
    pd.DataFrame({"ticker": [], "company_name": []}).to_csv(snapshots_dir / "nifty500_2021-01-01.csv", index=False)
    pd.DataFrame({"ticker": ["AAA.NS"], "company_name": ["AAA"]}).to_csv(snapshots_dir / "nifty500_2022-01-01.csv", index=False)

    universe, _ = build_universe_history_from_snapshots(snapshots_dir)
    aaa_rows = universe.loc[universe["ticker"] == "AAA.NS"].reset_index(drop=True)
    assert len(aaa_rows) == 2
    assert aaa_rows.loc[0, "end_date"] == "2020-12-31"
    assert aaa_rows.loc[1, "start_date"] == "2022-01-01"
